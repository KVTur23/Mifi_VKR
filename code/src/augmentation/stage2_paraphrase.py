"""
stage2_paraphrase.py — Этап 2: ruT5-парафраз с чанкованием.

Берём классы с 15–34 примерами и доводим до 35 через
fyaronskiy/ruT5-large-paraphraser. Длинные письма режутся на
tokenizer-aware чанки и собираются обратно перед валидацией.

Вход:  Data/data_after_stage1.csv  (или data_after_stage2.csv если чекпоинт есть)
Выход: Data/data_after_stage2.csv

Запуск:
    python src/augmentation/stage2_paraphrase.py --config configs/model_vllm.json
"""

import sys
import random
import argparse
import gc
from pathlib import Path

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
)
from src.augmentation.llm_utils import load_llm, select_top_paraphrases
from src.augmentation.rut5_paraphraser import RuT5Paraphraser
from src.augmentation.validation import validate_generated_texts


STAGE = 2
TARGET_COUNT = 35
MAX_RETRIES = 5
OVERSAMPLE_FACTOR = 5
RUT5_JUDGE_MIN_SCORE = 4.5


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    paraphraser: RuT5Paraphraser,
    n_original: int | None = None,
    paraphrase_sources: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Builds a validated candidate pool for one class.

    Returns pairs: (paraphrase, source_original). LLM-judge selection happens
    after ruT5 is unloaded, so that vLLM has GPU memory available.
    """
    candidate_pairs: list[tuple[str, str]] = []
    current_existing = list(existing_texts)

    if paraphrase_sources is None:
        paraphrase_sources = existing_texts

    for attempt in range(1, MAX_RETRIES + 1):
        pool_target = max(n_needed * 2, n_needed)
        if len(candidate_pairs) >= pool_target:
            break

        pool_gap = pool_target - len(candidate_pairs)
        batch_size = max(pool_gap, 1) * OVERSAMPLE_FACTOR + 1
        sources = _select_sources(paraphrase_sources, batch_size)

        print(f"  [Раунд {attempt}/{MAX_RETRIES}] Нужно в пул ещё {pool_gap}, "
              f"генерируем {len(sources)} ruT5-парафразов")

        raw_outputs = paraphraser.paraphrase_texts(sources)
        pairs = [
            (text, source)
            for text, source in zip(raw_outputs, sources)
            if text and text.strip()
        ]

        print(f"  [Раунд {attempt}] Получено {len(pairs)} парафразов")
        if not pairs:
            continue

        paraphrased = [p[0] for p in pairs]
        originals = [p[1] for p in pairs]

        sim_threshold = min(0.95 + max(0, attempt - 2) * 0.01, 0.98)
        valid = validate_generated_texts(
            paraphrased, current_existing, class_name,
            similarity_threshold=sim_threshold, n_original=n_original,
        )
        n_after_validation = len(valid)

        valid_set = set(valid)
        valid_pairs = [
            (para, orig)
            for para, orig in zip(paraphrased, originals)
            if para in valid_set
        ]

        candidate_pairs.extend(valid_pairs)
        current_existing.extend([p[0] for p in valid_pairs])

        print(f"  [Раунд {attempt}] После фильтров {n_after_validation}, "
              f"в пул +{len(valid_pairs)}, всего в пуле {len(candidate_pairs)}/{pool_target}")

    if len(candidate_pairs) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: после ruT5 и фильтров только "
              f"{len(candidate_pairs)}/{n_needed} кандидатов")

    return candidate_pairs


def _select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """Выбирает источники равномерно по кругу."""
    if not existing_texts:
        return []

    sources = []
    shuffled = list(existing_texts)
    random.shuffle(shuffled)

    full_rounds = n_needed // len(shuffled)
    remainder = n_needed % len(shuffled)

    for _ in range(full_rounds):
        round_copy = list(shuffled)
        random.shuffle(round_copy)
        sources.extend(round_copy)

    if remainder > 0:
        extra = list(shuffled)
        random.shuffle(extra)
        sources.extend(extra[:remainder])

    return sources


def run(config_path: str, pipeline_cfg=None) -> None:
    """Основная функция этапа 2."""
    global TARGET_COUNT, MAX_RETRIES, OVERSAMPLE_FACTOR, RUT5_JUDGE_MIN_SCORE

    originals_only_sources = True

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage2
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        originals_only_sources = bool(s.get("originals_only_sources", True))
        RUT5_JUDGE_MIN_SCORE = float(s.get("judge_min_score", RUT5_JUDGE_MIN_SCORE))

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ЭТАП 2: ruT5-парафраз с чанкованием (< {TARGET_COUNT} → {TARGET_COUNT})")
    print(f"       источник парафраза: "
          f"{'ТОЛЬКО ОРИГИНАЛЫ (stage 0)' if originals_only_sources else 'ВСЁ (legacy, каскад)'}")
    print(f"       LLM-judge threshold: {RUT5_JUDGE_MIN_SCORE}")
    print("=" * 60)

    df = load_dataset(stage=STAGE)
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 2] Все классы уже имеют >= 35 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 2] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    df_original = load_dataset(stage=0)
    original_counts = df_original[LABEL_COL].value_counts().to_dict()

    class_pools: dict[str, dict] = {}
    paraphraser = RuT5Paraphraser.from_pipeline_config(pipeline_cfg)

    try:
        for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
            n_needed = TARGET_COUNT - current_count
            existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()
            real_original_count = original_counts.get(class_name, current_count)

            if originals_only_sources:
                paraphrase_sources = df_original[
                    df_original[LABEL_COL] == class_name
                ][TEXT_COL].tolist()
                if not paraphrase_sources:
                    print(f"  [Внимание] Нет оригиналов для «{class_name}», "
                          f"fallback на существующие тексты (legacy режим)")
                    paraphrase_sources = existing_texts
            else:
                paraphrase_sources = existing_texts

            print(f"\n[Этап 2] Класс «{class_name}»: есть {current_count} "
                  f"(из них {real_original_count} оригиналов), "
                  f"источников: {len(paraphrase_sources)}, нужно ещё {n_needed}")

            try:
                pairs = augment_class(
                    class_name=class_name,
                    existing_texts=existing_texts,
                    n_needed=n_needed,
                    paraphraser=paraphraser,
                    n_original=real_original_count,
                    paraphrase_sources=paraphrase_sources,
                )
            except Exception as e:
                print(f"[Этап 2] Ошибка при ruT5-парафразе класса «{class_name}»: {e}")
                print("[Этап 2] Пропускаю класс, продолжаю с остальными")
                continue

            class_pools[class_name] = {
                "pairs": pairs,
                "n_needed": n_needed,
                "class_idx": class_idx,
            }
    finally:
        paraphraser.unload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[Этап 2] Грузим LLM-судью для отбора ruT5-парафразов...")
    llm, _, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

    new_rows = []

    try:
        for class_name, state in class_pools.items():
            pairs = state["pairs"]
            n_needed = state["n_needed"]

            if not pairs:
                print(f"[Этап 2] Класс «{class_name}»: нет кандидатов для судьи")
                continue

            paras = [p[0] for p in pairs]
            origs = [p[1] for p in pairs]
            print(f"\n[Этап 2] Класс «{class_name}»: "
                  f"{len(pairs)} кандидатов после ruT5, нужно {n_needed}")

            selected = select_top_paraphrases(
                paras, origs, class_name, llm,
                n_needed=n_needed, min_score=RUT5_JUDGE_MIN_SCORE,
            )

            for text in selected:
                new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

            print(f"[Этап 2] Класс «{class_name}»: добавлено {len(selected)} парафразов")

            if new_rows:
                df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                save_checkpoint(df_tmp, stage=STAGE)
                print(f"[Этап 2] Промежуточное сохранение: {len(df_tmp)} записей")
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"\n[Этап 2] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 2] Новых текстов не сгенерировано")

    save_checkpoint(df, stage=STAGE)

    print(f"\n[Этап 2] Итоговое распределение:")
    dist = get_class_distribution(df)
    for name, count in dist.items():
        marker = " ✓" if count >= TARGET_COUNT else " ✗"
        print(f"  «{name}»: {count}{marker}")

    print(f"\n[Этап 2] Завершён. Всего записей: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Этап 2: ruT5-парафраз для классов с < 35 примерами"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь до JSON-конфига LLM-судьи (например, configs/model_vllm.json)",
    )
    args = parser.parse_args()

    run(args.config)
