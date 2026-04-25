"""
stage1_llm_generate.py — Этап 1: генерация текстов через LLM (батчевый режим)

Берём классы где меньше 15 примеров и догоняем до 15 через LLM.
Генерация идёт батчами через vLLM, после — валидация и оценка LLM-судьёй.

Вход:  Data/train_after_eda.csv  (или data_after_stage1.csv если чекпоинт есть)
Выход: Data/data_after_stage1.csv

Запуск:
    python src/augmentation/stage1_llm_generate.py --config configs/model_vllm.json
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
    load_llm, generate_text, generate_batch,
    load_prompt_template, select_top_half,
)
from src.augmentation.validation import validate_generated_texts


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 1
TARGET_COUNT = 15
MAX_RETRIES = 10
MAX_EXAMPLES_IN_PROMPT = 5
OVERSAMPLE_FACTOR = 5


def generate_class_context(
    class_name: str,
    examples: list[str],
    llm,
    sampling_params,
    system_prompt: str | None = None,
) -> str:
    """
    Просим LLM описать класс по имеющимся примерам.
    Это описание потом подставляется в промпт генерации —
    помогает модели лучше понять что за класс.
    """
    context_template = load_prompt_template("class_context.txt")
    examples_text = "\n---\n".join(examples)
    prompt = context_template.format(class_name=class_name, examples=examples_text)

    try:
        result = generate_text(llm, sampling_params, prompt, system_prompt=system_prompt)
        if result:
            # склеиваем абзацы в одну строку, убираем лишние переносы
            context = " ".join(result.strip().split("\n\n")).strip()
            print(f"  [Контекст] «{class_name}»:\n{context}\n")
            return context
    except Exception as e:
        print(f"  [Контекст] Ошибка генерации контекста для «{class_name}»: {e}")

    # фоллбэк если LLM не смогла
    return f"Официальные входящие письма класса «{class_name}»."


def build_prompt(template: str, class_name: str, examples: list[str], context: str = "") -> str:
    """Собирает промпт для генерации одного письма."""
    # перемешиваем примеры что бы модель не запоминала порядок
    shuffled = list(examples)
    random.shuffle(shuffled)
    selected_examples = shuffled[:MAX_EXAMPLES_IN_PROMPT]
    examples_text = "\n---\n".join(selected_examples)

    return template.format(
        class_name=class_name,
        examples=examples_text,
        context=context,
    )


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    llm,
    sampling_params,
    prompt_template: str,
    system_prompt: str | None = None,
    context: str = "",
    n_original: int | None = None,
) -> list[str]:
    """
    Генерирует тексты для одного класса батчами.

    Схема: генерируем с запасом 3x → валидация (фильтры) → LLM-судья (оценка) → берём лучшие.
    Если после всего не хватает — повторяем (до MAX_RETRIES раз).
    """
    all_valid_texts = []
    current_existing = list(existing_texts)

    for attempt in range(MAX_RETRIES):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        # генерируем 3x от нужного — часть отсеется фильтрами, часть судьёй
        batch_size = int(still_needed * OVERSAMPLE_FACTOR) + 1
        print(f"  [Генерация] Раунд {attempt + 1}/{MAX_RETRIES}: "
              f"генерируем батч из {batch_size} промптов (нужно ещё {still_needed})")

        # список промптов с разным набором примеров — для разнообразия
        prompts = [
            build_prompt(prompt_template, class_name, current_existing, context=context)
            for _ in range(batch_size)
        ]

        # всё летит на GPU одним батчем
        raw_outputs = generate_batch(llm, sampling_params, prompts, system_prompt=system_prompt)

        # выкидываем пустые ответы
        candidates = [text for text in raw_outputs if text]

        if not candidates:
            print(f"  [Генерация] Раунд {attempt + 1}: все выходы пустые, повторяю")
            continue

        # прогрессивный порог сходства: первые 2 попытки строгий (0.95),
        # потом каждую попытку +0.01, максимум 0.98
        human_attempt = attempt + 1
        sim_threshold = min(0.95 + max(0, human_attempt - 2) * 0.01, 0.98)

        # прогоняем через фильтры (дубликаты, длина, язык, сходство и т.д.)
        valid = validate_generated_texts(
            candidates, current_existing, class_name,
            similarity_threshold=sim_threshold, n_original=n_original,
        )
        n_after_validation = len(valid)

        # LLM-судья сравнивает с оригиналами + описанием класса, выкидывает слабые
        valid = select_top_half(
            valid, class_name, llm,
            n_needed=still_needed, existing_texts=existing_texts,
            context=context,
        )

        to_take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:to_take])
        current_existing.extend(valid[:to_take])

        print(f"  [Генерация] Раунд {attempt + 1}: "
              f"получено {len(candidates)}, после фильтров {n_after_validation}, "
              f"после судьи {len(valid)}, принято {to_take}, "
              f"всего {len(all_valid_texts)}/{n_needed}")

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось сгенерировать только "
              f"{len(all_valid_texts)}/{n_needed} текстов")

    return all_valid_texts


def run(config_path: str, pipeline_cfg=None) -> None:
    """Основная функция этапа 1."""
    global TARGET_COUNT, MAX_RETRIES, MAX_EXAMPLES_IN_PROMPT, OVERSAMPLE_FACTOR

    # применяем настройки из pipeline_config если переданы
    if pipeline_cfg is not None:
        s = pipeline_cfg.stage1
        TARGET_COUNT = s.target_count  #15
        MAX_RETRIES = s.max_retries     #5
        OVERSAMPLE_FACTOR = s.oversample_factor     #5
        MAX_EXAMPLES_IN_PROMPT = s.max_examples_in_prompt   #5

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ЭТАП 1: LLM-генерация (< {TARGET_COUNT} → {TARGET_COUNT})")
    print("=" * 60)

    df = load_dataset(stage=STAGE)

    # смотрим каким классам не хватает примеров
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 1] Все классы уже имеют >= 15 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 1] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    # грузим LLM один раз — дальше переиспользуем для всех классов
    llm, sampling_params, system_prompt = load_llm(config_path, pipeline_cfg=pipeline_cfg)
    prompt_template = load_prompt_template("llm_generate_one.txt")

    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 1] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

        # описание класса — генерируем один раз, потом суём в каждый промпт
        context = generate_class_context(
            class_name, existing_texts, llm, sampling_params, system_prompt,
        )

        try:
            generated = augment_class(
                class_name=class_name,
                existing_texts=existing_texts,
                n_needed=n_needed,
                llm=llm,
                sampling_params=sampling_params,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                context=context,
                n_original=current_count,
            )
        except Exception as e:
            print(f"[Этап 1] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 1] Пропускаю класс, продолжаю с остальными")
            continue

        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 1] Класс «{class_name}»: добавлено {len(generated)} текстов")

        # сохраняем после каждого класса — если colab упадёт, не потеряем
        if new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 1] Промежуточное сохранение: "
                  f"{class_idx + 1} классов обработано, {len(df_tmp)} записей")

    # финальное сохранение
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 1] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 1] Новых текстов не сгенерировано")

    save_checkpoint(df, stage=STAGE)

    # итого — что получилось
    print(f"\n[Этап 1] Итоговое распределение:")
    dist = get_class_distribution(df)
    for name, count in dist.items():
        marker = " ✓" if count >= TARGET_COUNT else " ✗"
        print(f"  «{name}»: {count}{marker}")

    print(f"\n[Этап 1] Завершён. Всего записей: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Этап 1: LLM-генерация текстов для классов с < 15 примерами"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь до JSON-конфига модели (например, configs/model_vllm.json)",
    )
    args = parser.parse_args()

    run(args.config)
