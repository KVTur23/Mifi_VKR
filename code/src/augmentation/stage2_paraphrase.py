"""
stage2_paraphrase.py — Этап 2: парафраз текстов через LLM (батчевый режим)

Берём классы с 15–34 примерами и доводим до 35 через перефразирование.
Для каждого текста берём оригинал и просим LLM переписать его другими словами.

Генерация батчами через vLLM, потом валидация + LLM-судья.

Вход:  Data/data_after_stage1.csv  (или data_after_stage2.csv если чекпоинт есть)
Выход: Data/data_after_stage2.csv

Запуск:
    python src/augmentation/stage2_paraphrase.py --config configs/model_vllm.json
"""

import sys
import random
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
)
from src.augmentation.llm_utils import (
    load_llm, generate_batch,
    load_prompt_template, select_top_paraphrases,
)
from src.augmentation.validation import validate_generated_texts
from src.utils.config_loader import load_model_config


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 2
TARGET_COUNT = 35
MAX_RETRIES = 5
OVERSAMPLE_FACTOR = 5
PARAPHRASE_PROMPT = "paraphrase.txt"


def build_paraphrase_prompt(template: str, original_text: str, class_name: str) -> str:
    """Собирает промпт для перефразирования одного текста."""
    return template.format(
        original_text=original_text,
        class_name=class_name,
    )


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    llm,
    sampling_params,
    prompt_template: str,
    system_prompt: str | None = None,
    n_original: int | None = None,
    paraphrase_sources: list[str] | None = None,
) -> list[str]:
    """
    Генерирует парафразы для одного класса батчами.

    Схема: берём оригиналы по кругу → генерируем 3x с запасом →
    валидация → LLM-судья → берём лучшие. Повторяем если мало.

    Параметры:
        existing_texts:     все тексты класса (для дедупликации/валидации)
        paraphrase_sources: тексты-источники для парафраза. Если None —
                            используется existing_texts (legacy: каскад через
                            синтетику этапа 1). Если задано (например, оригиналы
                            из train_after_eda.csv) — каждый парафраз отстоит
                            от реального текста ровно на 1 LLM-операцию.
    """
    all_valid_texts = []
    current_existing = list(existing_texts)

    # По умолчанию — legacy (источник = всё, что есть, включая синтетику stage 1).
    # Передаём paraphrase_sources, чтобы фиксить каскадирование.
    if paraphrase_sources is None:
        paraphrase_sources = existing_texts

    for attempt in range(1, MAX_RETRIES + 1):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        batch_size = int(still_needed * OVERSAMPLE_FACTOR) + 1
        print(f"  [Раунд {attempt}/{MAX_RETRIES}] Нужно ещё {still_needed}, "
              f"генерируем батч из {batch_size} парафразов")

        # выбираем оригиналы равномерно по кругу — что бы не перефразировать один и тот же 10 раз
        sources = _select_sources(paraphrase_sources, batch_size)

        # собираем промпты
        prompts = [
            build_paraphrase_prompt(prompt_template, source, class_name)
            for source in sources
        ]

        # всё на GPU одним батчем
        raw_outputs = generate_batch(llm, sampling_params, prompts, system_prompt=system_prompt)

        # убираем пустые, но сохраняем связь парафраз → оригинал
        pairs = [
            (text, source)
            for text, source in zip(raw_outputs, sources)
            if text
        ]

        print(f"  [Раунд {attempt}] Получено {len(pairs)} парафразов")

        if not pairs:
            continue

        paraphrased = [p[0] for p in pairs]
        their_originals = [p[1] for p in pairs]

        # прогрессивный порог сходства: первые 2 попытки строгий (0.95),
        # потом каждую попытку +0.01, максимум 0.98
        sim_threshold = min(0.95 + max(0, attempt - 2) * 0.01, 0.98)

        # фильтры: дубликаты, длина, язык, сходство
        # n_original прокидываем — для маленьких классов (1 оригинал) порог мягче (0.98)
        valid = validate_generated_texts(
            paraphrased, current_existing, class_name,
            similarity_threshold=sim_threshold, n_original=n_original,
        )
        n_after_validation = len(valid)

        # восстанавливаем связь — какой оригинал у каждого прошедшего фильтры
        valid_set = set(valid)
        valid_with_originals = [
            (para, orig)
            for para, orig in zip(paraphrased, their_originals)
            if para in valid_set
        ]
        valid_paras = [p[0] for p in valid_with_originals]
        valid_origs = [p[1] for p in valid_with_originals]

        # LLM-судья сравнивает каждый парафраз с его конкретным оригиналом
        valid = select_top_paraphrases(
            valid_paras, valid_origs, class_name, llm,
            n_needed=still_needed,
        )

        take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:take])
        current_existing.extend(valid[:take])

        print(f"  [Раунд {attempt}] После фильтров {n_after_validation}, "
              f"после судьи {len(valid)}, принято {take}, "
              f"всего {len(all_valid_texts)}/{n_needed}")

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось получить только "
              f"{len(all_valid_texts)}/{n_needed} парафразов за {MAX_RETRIES} раундов")

    return all_valid_texts


def _select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """
    Выбирает оригиналы для перефразирования равномерно по кругу.
    Каждый оригинал используется примерно одинаковое число раз.
    """
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
    global TARGET_COUNT, MAX_RETRIES, OVERSAMPLE_FACTOR

    # Флаг originals_only_sources: если True (рекомендуется), парафраз берётся
    # только из оригинальных писем (stage 0), а stage 1 используется только
    # для дедупликации. Это устраняет каскадирование синтетики через стадии.
    # По умолчанию True — каскад был выявлен как источник плато f1_C=0.5333
    # в трёх независимых ранах fine-tune.
    originals_only_sources = True

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage2
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        # _DotDict наследует dict — используем .get() с default
        originals_only_sources = bool(s.get("originals_only_sources", True))

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ЭТАП 2: Парафраз через LLM (< {TARGET_COUNT} → {TARGET_COUNT})")
    print(f"       источник парафраза: "
          f"{'ТОЛЬКО ОРИГИНАЛЫ (stage 0)' if originals_only_sources else 'ВСЁ (legacy, каскад)'}")
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

    # грузим LLM и конфиг
    llm, sampling_params, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

    # для парафраза берём отдельный system_prompt и шаблон из конфига
    config = load_model_config(config_path)
    paraphrase_template_name = config.get("paraphrase_template", PARAPHRASE_PROMPT)
    system_prompt = config.get("paraphrase_system_prompt")
    prompt_template = load_prompt_template(paraphrase_template_name)

    # загружаем исходный датасет (до аугментации) — что бы знать реальное число оригиналов
    # после stage1 у класса может быть 15 текстов, но реально оригиналов было 1-5
    # это влияет на порог косинусного сходства в валидации
    df_original = load_dataset(stage=0)
    original_counts = df_original[LABEL_COL].value_counts().to_dict()

    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()
        real_original_count = original_counts.get(class_name, current_count)

        # Источник для парафраза: только оригиналы (фикс каскадирования).
        # Если оригиналов нет (быть не должно после EDA) — fallback на существующее.
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
              f"источников парафраза: {len(paraphrase_sources)}, "
              f"нужно ещё {n_needed}")

        try:
            generated = augment_class(
                class_name=class_name,
                existing_texts=existing_texts,
                n_needed=n_needed,
                llm=llm,
                sampling_params=sampling_params,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                n_original=real_original_count,
                paraphrase_sources=paraphrase_sources,
            )
        except Exception as e:
            print(f"[Этап 2] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 2] Пропускаю класс, продолжаю с остальными")
            continue

        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 2] Класс «{class_name}»: добавлено {len(generated)} парафразов")

        # сохраняем после каждого класса — страховка
        if new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 2] Промежуточное сохранение: "
                  f"{class_idx + 1} классов обработано, {len(df_tmp)} записей")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
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
        description="Этап 2: парафраз текстов для классов с < 35 примерами"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь до JSON-конфига модели (например, configs/model_vllm.json)",
    )
    args = parser.parse_args()

    run(args.config)
