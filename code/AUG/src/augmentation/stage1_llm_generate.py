"""
stage1_llm_generate.py — Этап 1: генерация текстов через LLM (батчевый режим)

Берём классы, в которых меньше 15 примеров, и догоняем их до 15
с помощью языковой модели через vLLM (батчевый инференс).
После генерации запускается валидация — дубликаты и мусор отсеиваются.
Если после отсева не хватает примеров, генерация повторяется.

Вход:  Data/train_after_eda.csv  (или data_after_stage1.csv, если чекпоинт есть)
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
from src.augmentation.llm_utils import load_llm, generate_text, generate_batch, load_prompt_template, select_top_half
from src.augmentation.validation import validate_generated_texts


# --- Настройки этапа ---

STAGE = 1
TARGET_COUNT = 15       # Доводим каждый класс до 15 примеров
MAX_RETRIES = 5          # Сколько раундов батчевой генерации при нехватке
MAX_EXAMPLES_IN_PROMPT = 5  # Сколько примеров класса показывать модели в промпте


def generate_class_context(
    class_name: str,
    examples: list[str],
    llm,
    sampling_params,
    system_prompt: str | None = None,
) -> str:
    """
    Генерирует краткое описание класса на основе всех имеющихся примеров.
    """
    context_template = load_prompt_template("class_context.txt")
    examples_text = "\n---\n".join(examples)
    prompt = context_template.format(class_name=class_name, examples=examples_text)

    try:
        result = generate_text(llm, sampling_params, prompt, system_prompt=system_prompt)
        if result:
            context = result.split("\n\n")[0].strip()
            print(f"  [Контекст] «{class_name}»:\n{context}\n")
            return context
    except Exception as e:
        print(f"  [Контекст] Ошибка генерации контекста для «{class_name}»: {e}")

    return f"Официальные входящие письма класса «{class_name}»."


def build_prompt(template: str, class_name: str, examples: list[str], context: str = "") -> str:
    """
    Собирает промпт для генерации одного письма.
    """
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
    Генерирует новые тексты для одного класса батчами.

    За один раунд генерирует n_needed промптов одновременно,
    валидирует, и если не хватает — повторяет.
    """
    all_valid_texts = []
    current_existing = list(existing_texts)

    for attempt in range(MAX_RETRIES):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        # Генерируем с запасом 3x — часть отсеется валидацией + LLM-судья отберёт лучших
        batch_size = int(still_needed * 3) + 1
        print(f"  [Генерация] Раунд {attempt + 1}/{MAX_RETRIES}: "
              f"генерируем батч из {batch_size} промптов (нужно ещё {still_needed})")

        # Собираем батч промптов — каждый с разным набором примеров
        prompts = [
            build_prompt(prompt_template, class_name, current_existing, context=context)
            for _ in range(batch_size)
        ]

        # Батчевая генерация через vLLM
        raw_outputs = generate_batch(llm, sampling_params, prompts, system_prompt=system_prompt)

        # Собираем непустые кандидаты
        candidates = [text for text in raw_outputs if text]

        if not candidates:
            print(f"  [Генерация] Раунд {attempt + 1}: все выходы пустые, повторяю")
            continue

        # Валидируем весь батч разом
        valid = validate_generated_texts(
            candidates, current_existing, class_name, n_original=n_original,
        )

        # LLM-судья оценивает и отсеивает слабые тексты
        valid = select_top_half(
            valid, class_name, llm, sampling_params,
            n_needed=still_needed, system_prompt=system_prompt,
        )

        # Берём только сколько нужно
        to_take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:to_take])
        current_existing.extend(valid[:to_take])

        print(f"  [Генерация] Раунд {attempt + 1}: "
              f"получено {len(candidates)}, прошло валидацию {len(valid)}, "
              f"принято {to_take}, всего {len(all_valid_texts)}/{n_needed}")

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось сгенерировать только "
              f"{len(all_valid_texts)}/{n_needed} текстов")

    return all_valid_texts


def run(config_path: str) -> None:
    """
    Основная функция этапа 1 — точка входа.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 1: LLM-генерация (< 15 → 15)")
    print("=" * 60)

    # --- Загрузка данных ---
    df = load_dataset(stage=STAGE)

    # --- Какие классы нуждаются в аугментации ---
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 1] Все классы уже имеют >= 15 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 1] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    # --- Загрузка LLM ---
    llm, sampling_params, system_prompt = load_llm(config_path)
    prompt_template = load_prompt_template("llm_generate_one.txt")

    # --- Генерация по классам ---
    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 1] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

        # Генерируем контекст класса один раз
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

        # Промежуточное сохранение — страховка от вылета ядра в Colab
        if new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 1] Промежуточное сохранение: "
                  f"{class_idx + 1} классов обработано, {len(df_tmp)} записей")

    # --- Добавляем сгенерированные тексты ---
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 1] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 1] Новых текстов не сгенерировано")

    # --- Сохраняем чекпоинт ---
    save_checkpoint(df, stage=STAGE)

    # --- Итоговая статистика ---
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
