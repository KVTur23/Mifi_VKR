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
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
    DATA_DIR,
)
from src.augmentation.llm_utils import load_llm, select_top_paraphrases
from src.augmentation.rut5_paraphraser import RuT5Paraphraser
from src.augmentation.validation import validate_generated_texts


STAGE = 2
TARGET_COUNT = 35
MAX_RETRIES = 5
OVERSAMPLE_FACTOR = 5
RUT5_JUDGE_MIN_SCORE = 4.5
RUT5_MIN_TEXT_LENGTH = 250
RUT5_MIN_LENGTH_RATIO = 0.35
RUT5_APPLY_PROMPT_LEAK_FILTER = False
POST_JUDGE_RETRIES = 3
FINAL_FILL_BELOW_THRESHOLD = True
SMALL_CLASS_SOURCE_THRESHOLD = 6
SMALL_CLASS_TEMPERATURE = 0.95
TEMPERATURE_INCREASE_AFTER_ATTEMPT = 3
TEMPERATURE_STEP = 0.05
MAX_TEMPERATURE = 1.10

_PAIRS_CSV = DATA_DIR / "_stage2_pairs_cache.csv"


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    paraphraser: RuT5Paraphraser,
    n_original: int | None = None,
    paraphrase_sources: list[str] | None = None,
    generation_attempt_offset: int = 0,
    initial_pairs: list[tuple[str, str]] | None = None,
    cache_callback: Callable[[list[tuple[str, str]]], None] | None = None,
) -> list[tuple[str, str]]:
    """
    Builds a validated candidate pool for one class.

    Returns pairs: (paraphrase, source_original). LLM-judge selection happens
    after ruT5 is unloaded, so that vLLM has GPU memory available.
    """
    candidate_pairs: list[tuple[str, str]] = list(initial_pairs or [])
    current_existing = list(existing_texts) + [p[0] for p in candidate_pairs]

    if paraphrase_sources is None:
        paraphrase_sources = existing_texts

    for attempt in range(1, MAX_RETRIES + 1):
        global_attempt = generation_attempt_offset + attempt
        pool_target = max(n_needed * 2, n_needed)
        if len(candidate_pairs) >= pool_target:
            break

        pool_gap = pool_target - len(candidate_pairs)
        batch_size = max(pool_gap, 1) * OVERSAMPLE_FACTOR + 1
        sources = _select_sources(paraphrase_sources, batch_size)
        temperature = _temperature_for_attempt(
            base_temperature=paraphraser.cfg.temperature,
            source_count=len(paraphrase_sources),
            attempt=global_attempt,
        )

        print(f"  [Раунд {attempt}/{MAX_RETRIES}] Нужно в пул ещё {pool_gap}, "
              f"генерируем {len(sources)} ruT5-парафразов, "
              f"global_attempt={global_attempt}, temperature={temperature:.2f}")

        sim_threshold = min(0.95 + max(0, attempt - 2) * 0.01, 0.98)
        received_total = 0
        valid_total = 0
        added_total = 0
        source_batch_size = max(1, paraphraser.cfg.batch_size)

        for batch_start in range(0, len(sources), source_batch_size):
            if len(candidate_pairs) >= pool_target:
                break

            source_batch = sources[batch_start:batch_start + source_batch_size]
            raw_outputs = paraphraser.paraphrase_texts(source_batch, temperature=temperature)
            pairs = [
                (text, source)
                for text, source in zip(raw_outputs, source_batch)
                if text and text.strip()
            ]
            received_total += len(pairs)

            if not pairs:
                continue

            paraphrased = [p[0] for p in pairs]
            originals = [p[1] for p in pairs]

            valid = validate_generated_texts(
                paraphrased, current_existing, class_name,
                similarity_threshold=sim_threshold, n_original=n_original,
                min_length=RUT5_MIN_TEXT_LENGTH,
                source_texts=originals,
                min_length_ratio=RUT5_MIN_LENGTH_RATIO,
                apply_prompt_leak_filter=RUT5_APPLY_PROMPT_LEAK_FILTER,
            )
            n_after_validation = len(valid)
            valid_total += n_after_validation

            valid_set = set(valid)
            valid_pairs = [
                (para, orig)
                for para, orig in zip(paraphrased, originals)
                if para in valid_set
            ]

            candidate_pairs.extend(valid_pairs)
            current_existing.extend([p[0] for p in valid_pairs])
            added_total += len(valid_pairs)

            if cache_callback is not None and valid_pairs:
                cache_callback(candidate_pairs)

        print(f"  [Раунд {attempt}] Получено {received_total} парафразов")
        print(f"  [Раунд {attempt}] После фильтров {valid_total}, "
              f"в пул +{added_total}, всего в пуле {len(candidate_pairs)}/{pool_target}")

    if len(candidate_pairs) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: после ruT5 и фильтров только "
              f"{len(candidate_pairs)}/{n_needed} кандидатов")

    return candidate_pairs


def _temperature_for_attempt(
    base_temperature: float,
    source_count: int,
    attempt: int,
) -> float:
    temperature = base_temperature
    if source_count <= SMALL_CLASS_SOURCE_THRESHOLD:
        temperature = max(temperature, SMALL_CLASS_TEMPERATURE)

    if attempt >= TEMPERATURE_INCREASE_AFTER_ATTEMPT:
        extra_steps = attempt - TEMPERATURE_INCREASE_AFTER_ATTEMPT + 1
        temperature += extra_steps * TEMPERATURE_STEP

    return min(temperature, MAX_TEMPERATURE)


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


def _load_pairs_cache(classes_to_augment: dict[str, int]) -> dict[str, list[tuple[str, str]]]:
    """Loads validated ruT5 candidate pairs from the previous interrupted run."""
    if not _PAIRS_CSV.exists():
        return {}

    try:
        df_cache = pd.read_csv(_PAIRS_CSV)
    except Exception as e:
        print(f"[Этап 2][Кэш] Не удалось прочитать {_PAIRS_CSV.name}: {e}")
        return {}

    required = {LABEL_COL, "paraphrase", "original"}
    if not required.issubset(df_cache.columns):
        print(f"[Этап 2][Кэш] {_PAIRS_CSV.name} имеет старый формат, игнорирую")
        return {}

    pending_labels = set(classes_to_augment)
    loaded: dict[str, list[tuple[str, str]]] = {}
    for _, row in df_cache.iterrows():
        class_name = row[LABEL_COL]
        if class_name not in pending_labels:
            continue
        paraphrase = str(row["paraphrase"]).strip()
        original = str(row["original"]).strip()
        if paraphrase and original:
            loaded.setdefault(class_name, []).append((paraphrase, original))

    total = sum(len(v) for v in loaded.values())
    if total:
        print(f"[Этап 2][Кэш] Загружено {total} валидированных кандидатов из {_PAIRS_CSV.name}")
    return loaded


def _save_pairs_cache(class_pools: dict[str, dict]) -> None:
    """Persists current ruT5 candidate pools so a runtime disconnect does not lose them."""
    rows = []
    for class_name, state in class_pools.items():
        for paraphrase, original in state.get("pairs", []):
            rows.append({
                LABEL_COL: class_name,
                "paraphrase": paraphrase,
                "original": original,
            })

    if not rows:
        return

    pd.DataFrame(rows).to_csv(_PAIRS_CSV, index=False)
    print(f"[Этап 2][Кэш] Сохранено {len(rows)} кандидатов в {_PAIRS_CSV.name}")


def _archive_pairs_cache(cycle: int) -> None:
    """Moves consumed pool cache aside after judge/checkpoint processing."""
    if not _PAIRS_CSV.exists():
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = _PAIRS_CSV.with_name(f"_stage2_pairs_cache_cycle{cycle}_{stamp}.bak.csv")
    _PAIRS_CSV.rename(backup)
    print(f"[Этап 2][Кэш] Пул кандидатов обработан → {backup.name}")


def run(config_path: str, pipeline_cfg=None) -> None:
    """Основная функция этапа 2."""
    global TARGET_COUNT, MAX_RETRIES, OVERSAMPLE_FACTOR, RUT5_JUDGE_MIN_SCORE
    global RUT5_MIN_TEXT_LENGTH, RUT5_MIN_LENGTH_RATIO, RUT5_APPLY_PROMPT_LEAK_FILTER
    global POST_JUDGE_RETRIES, FINAL_FILL_BELOW_THRESHOLD
    global SMALL_CLASS_SOURCE_THRESHOLD, SMALL_CLASS_TEMPERATURE
    global TEMPERATURE_INCREASE_AFTER_ATTEMPT, TEMPERATURE_STEP, MAX_TEMPERATURE

    originals_only_sources = True

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage2
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        originals_only_sources = bool(s.get("originals_only_sources", True))
        RUT5_JUDGE_MIN_SCORE = float(s.get("judge_min_score", RUT5_JUDGE_MIN_SCORE))
        RUT5_MIN_TEXT_LENGTH = int(s.get("validation_min_text_length", RUT5_MIN_TEXT_LENGTH))
        RUT5_MIN_LENGTH_RATIO = float(s.get("validation_min_length_ratio", RUT5_MIN_LENGTH_RATIO))
        RUT5_APPLY_PROMPT_LEAK_FILTER = bool(
            s.get("apply_prompt_leak_filter", RUT5_APPLY_PROMPT_LEAK_FILTER)
        )
        POST_JUDGE_RETRIES = int(s.get("post_judge_retries", POST_JUDGE_RETRIES))
        FINAL_FILL_BELOW_THRESHOLD = bool(
            s.get("final_fill_below_threshold", FINAL_FILL_BELOW_THRESHOLD)
        )
        p = s.get("paraphraser", {})
        SMALL_CLASS_SOURCE_THRESHOLD = int(
            p.get("small_class_source_threshold", SMALL_CLASS_SOURCE_THRESHOLD)
        )
        SMALL_CLASS_TEMPERATURE = float(
            p.get("small_class_temperature", SMALL_CLASS_TEMPERATURE)
        )
        TEMPERATURE_INCREASE_AFTER_ATTEMPT = int(
            p.get("temperature_increase_after_attempt", TEMPERATURE_INCREASE_AFTER_ATTEMPT)
        )
        TEMPERATURE_STEP = float(p.get("temperature_step", TEMPERATURE_STEP))
        MAX_TEMPERATURE = float(p.get("max_temperature", MAX_TEMPERATURE))

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ЭТАП 2: ruT5-парафраз с чанкованием (< {TARGET_COUNT} → {TARGET_COUNT})")
    print(f"       источник парафраза: "
          f"{'ТОЛЬКО ОРИГИНАЛЫ (stage 0)' if originals_only_sources else 'ВСЁ (legacy, каскад)'}")
    print(f"       LLM-judge threshold: {RUT5_JUDGE_MIN_SCORE}")
    print(f"       validation length: min={RUT5_MIN_TEXT_LENGTH}, "
          f"ratio={RUT5_MIN_LENGTH_RATIO}, "
          f"prompt_leak_filter={RUT5_APPLY_PROMPT_LEAK_FILTER}")
    print(f"       post-judge retries: {POST_JUDGE_RETRIES}, "
          f"final fill below threshold: {FINAL_FILL_BELOW_THRESHOLD}")
    print(f"       temperature: small≤{SMALL_CLASS_SOURCE_THRESHOLD} → {SMALL_CLASS_TEMPERATURE}, "
          f"after attempt {TEMPERATURE_INCREASE_AFTER_ATTEMPT} +{TEMPERATURE_STEP}, "
          f"max {MAX_TEMPERATURE}")
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

    total_added = 0
    total_cycles = POST_JUDGE_RETRIES + 1

    for cycle in range(1, total_cycles + 1):
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print("\n[Этап 2] Все классы уже имеют >= 35 примеров")
            break

        is_final_cycle = cycle == total_cycles
        print("\n" + "-" * 60)
        print(f"[Этап 2] Цикл добора {cycle}/{total_cycles}: "
              f"{len(classes_to_augment)} классов ещё < {TARGET_COUNT}")
        print("-" * 60)
        for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
            print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

        class_pools: dict[str, dict] = {}
        cached_pairs_by_class = _load_pairs_cache(classes_to_augment)
        paraphraser = RuT5Paraphraser.from_pipeline_config(pipeline_cfg)

        try:
            for class_name, current_count in classes_to_augment.items():
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

                cached_pairs = cached_pairs_by_class.get(class_name, [])
                if cached_pairs:
                    print(f"  [Кэш] «{class_name}»: найдено {len(cached_pairs)} "
                          f"валидированных кандидатов")

                class_pools[class_name] = {
                    "pairs": list(cached_pairs),
                    "n_needed": n_needed,
                }

                def save_current_pool(pairs, class_name=class_name):
                    class_pools[class_name]["pairs"] = list(pairs)
                    _save_pairs_cache(class_pools)

                try:
                    pairs = augment_class(
                        class_name=class_name,
                        existing_texts=existing_texts,
                        n_needed=n_needed,
                        paraphraser=paraphraser,
                        n_original=real_original_count,
                        paraphrase_sources=paraphrase_sources,
                        generation_attempt_offset=(cycle - 1) * MAX_RETRIES,
                        initial_pairs=cached_pairs,
                        cache_callback=save_current_pool,
                    )
                except Exception as e:
                    print(f"[Этап 2] Ошибка при ruT5-парафразе класса «{class_name}»: {e}")
                    print("[Этап 2] Пропускаю класс, продолжаю с остальными")
                    continue

                class_pools[class_name]["pairs"] = pairs
                _save_pairs_cache(class_pools)
        finally:
            paraphraser.unload()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not any(state.get("pairs") for state in class_pools.values()):
            print("[Этап 2] В этом цикле нет кандидатов для судьи")
            continue

        print("\n[Этап 2] Грузим LLM-судью для отбора ruT5-парафразов...")
        llm, _, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

        cycle_added = 0
        try:
            for class_name, state in class_pools.items():
                current_count = len(df[df[LABEL_COL] == class_name])
                n_needed = TARGET_COUNT - current_count
                if n_needed <= 0:
                    continue

                pairs = state["pairs"]
                if not pairs:
                    print(f"[Этап 2] Класс «{class_name}»: нет кандидатов для судьи")
                    continue

                paras = [p[0] for p in pairs]
                origs = [p[1] for p in pairs]
                fill_to_n = is_final_cycle and FINAL_FILL_BELOW_THRESHOLD

                print(f"\n[Этап 2] Класс «{class_name}»: "
                      f"{len(pairs)} кандидатов после ruT5, нужно {n_needed}, "
                      f"final_fill={fill_to_n}")

                selected = select_top_paraphrases(
                    paras, origs, class_name, llm,
                    n_needed=n_needed,
                    min_score=RUT5_JUDGE_MIN_SCORE,
                    fill_to_n=fill_to_n,
                )

                if not selected:
                    print(f"[Этап 2] Класс «{class_name}»: судья ничего не принял")
                    continue

                df = pd.concat(
                    [df, pd.DataFrame([{TEXT_COL: text, LABEL_COL: class_name} for text in selected])],
                    ignore_index=True,
                )
                cycle_added += len(selected)
                total_added += len(selected)

                print(f"[Этап 2] Класс «{class_name}»: добавлено {len(selected)} парафразов, "
                      f"теперь {len(df[df[LABEL_COL] == class_name])}/{TARGET_COUNT}")
                save_checkpoint(df, stage=STAGE)
                print(f"[Этап 2] Промежуточное сохранение: {len(df)} записей")
        finally:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cycle_added == 0:
            print(f"[Этап 2] Цикл {cycle}: новых текстов не добавлено")
        else:
            print(f"[Этап 2] Цикл {cycle}: добавлено {cycle_added} текстов")
        _archive_pairs_cache(cycle)

    remaining = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
    if remaining:
        print(f"\n[Этап 2][Внимание] После всех циклов остались классы < {TARGET_COUNT}:")
        for name, count in sorted(remaining.items(), key=lambda x: x[1]):
            print(f"  «{name}»: {count} → не хватает {TARGET_COUNT - count}")
    else:
        print(f"\n[Этап 2] Цель достигнута: все классы >= {TARGET_COUNT}")

    print(f"\n[Этап 2] Всего добавлено {total_added} текстов")
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
