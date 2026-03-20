"""
stage1_llm_generate.py — Этап 1: генерация текстов через LLM

Берём классы, в которых меньше 15 примеров, и догоняем их до 15
с помощью языковой модели (конкретная модель задаётся через JSON-конфиг).
После генерации запускается валидация — дубликаты и мусор отсеиваются.
Если после отсева не хватает примеров, генерация повторяется.

Вход:  Data/train_after_eda.csv  (или data_after_stage1.csv, если чекпоинт есть)
Выход: Data/data_after_stage1.csv

Запуск:
    python src/augmentation/stage1_llm_generate.py --config configs/model_qwen.json
"""

import sys
import random
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
)
from src.augmentation.llm_utils import load_llm, generate_text, load_prompt_template
from src.augmentation.validation import validate_generated_texts


# --- Настройки этапа ---

STAGE = 1
TARGET_COUNT = 15       # Доводим каждый класс до 15 примеров
MAX_RETRIES = 10         # Сколько раз пробуем догенерировать, если валидация отсеяла слишком много
MAX_EXAMPLES_IN_PROMPT = 5  # Сколько примеров класса показывать модели в промпте


def generate_class_context(
    class_name: str,
    examples: list[str],
    model,
    tokenizer,
    generation_params: dict,
    system_prompt: str | None = None,
) -> str:
    """
    Генерирует краткое описание класса на основе всех имеющихся примеров.

    Один вызов LLM на класс — модель анализирует примеры и выделяет
    общие признаки: тему, тон, структуру, характерные формулировки.

    Аргументы:
        class_name:        название класса
        examples:          все имеющиеся тексты класса
        model:             загруженная LLM
        tokenizer:         токенизатор
        generation_params: параметры генерации
        system_prompt:     системный промпт (опционально)

    Возвращает:
        Текстовое описание класса (2-4 предложения)
    """
    context_template = load_prompt_template("class_context.txt")
    # Показываем все примеры (их мало — до 15)
    examples_text = "\n---\n".join(examples)
    prompt = context_template.format(class_name=class_name, examples=examples_text)

    try:
        outputs = generate_text(model, tokenizer, prompt, generation_params, system_prompt=system_prompt)
        if outputs and outputs[0].strip():
            context = outputs[0].strip()
            # Берём только первый абзац — чтобы не раздувать промпт
            context = context.split("\n\n")[0].strip()
            print(f"  [Контекст] «{class_name}»:\n{context}\n")
            return context
    except Exception as e:
        print(f"  [Контекст] Ошибка генерации контекста для «{class_name}»: {e}")

    return f"Официальные входящие письма класса «{class_name}»."


def build_prompt(template: str, class_name: str, examples: list[str], context: str = "") -> str:
    """
    Собирает промпт для генерации одного письма.

    Показывает модели контекст класса и несколько случайных примеров,
    просит написать одно новое письмо. Каждый вызов — разный набор примеров,
    поэтому генерация не застревает в повторах.

    Аргументы:
        template:    текст шаблона промпта (из prompts/llm_generate_one.txt)
        class_name:  название класса
        examples:    список существующих текстов класса
        context:     описание класса (сгенерированное LLM)

    Возвращает:
        Готовый промпт для передачи в LLM
    """
    # Перемешиваем и берём до MAX_EXAMPLES_IN_PROMPT штук —
    # чтобы каждый вызов модель видела разный набор примеров
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
    model,
    tokenizer,
    generation_params: dict,
    prompt_template: str,
    system_prompt: str | None = None,
    context: str = "",
    n_original: int | None = None,
) -> list[str]:
    """
    Генерирует промпты и новые тексты для одного класса по одному за вызов.

    Аргументы:
        class_name:        название класса
        existing_texts:    уже имеющиеся тексты этого класса
        n_needed:          сколько новых текстов нужно
        model:             загруженная LLM
        tokenizer:         токенизатор
        generation_params: параметры генерации из конфига
        prompt_template:   текст шаблона промпта
        system_prompt:     системный промпт (опционально)
        context:           описание класса (сгенерированное LLM)
        n_original:        количество оригинальных примеров (для адаптивного порога)

    Возвращает:
        Список валидных сгенерированных текстов
    """
    all_valid_texts = []
    # Тексты, которые уже есть + те, что мы уже приняли.
    # Нужно для валидации — чтобы новые тексты не дублировали уже принятые
    current_existing = list(existing_texts)

    total_attempts = 0
    # Максимум попыток — чтобы не уйти в бесконечный цикл при плохой модели
    max_total_attempts = n_needed * MAX_RETRIES

    while len(all_valid_texts) < n_needed and total_attempts < max_total_attempts:
        still_needed = n_needed - len(all_valid_texts)
        total_attempts += 1

        if total_attempts % 5 == 1:
            print(f"  [Генерация] Нужно ещё {still_needed}, попыток: {total_attempts}/{max_total_attempts}")

        # Один вызов → одно письмо: модель пишет письмо и останавливается
        prompt = build_prompt(prompt_template, class_name, current_existing, context=context)
        raw_outputs = generate_text(model, tokenizer, prompt, generation_params, system_prompt=system_prompt)
        
        # Что бы цикл не падал при ошибке генерации
        if not raw_outputs or not raw_outputs[0].strip():
            continue

        # Берём первый (и единственный) результат как одно письмо
        candidate = raw_outputs[0].strip()

        # Валидируем одно письмо против уже существующих
        valid = validate_generated_texts([candidate], current_existing, class_name, n_original=n_original)

        if valid:
            all_valid_texts.append(valid[0])
            current_existing.append(valid[0])
        # Если не прошло валидацию — просто пробуем ещё раз 

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось сгенерировать только "
              f"{len(all_valid_texts)}/{n_needed} текстов за {total_attempts} попыток")

    return all_valid_texts


def run(config_path: str) -> None:
    """
    Основная функция этапа 1 — точка входа.

    Загружает данные, определяет классы для аугментации, генерирует тексты,
    сохраняет чекпоинт.

    Аргументы:
        config_path: путь до JSON-конфига модели
    """
    # --- Фиксируем seed для воспроизводимости ---
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
    model, tokenizer, generation_params, _, system_prompt = load_llm(config_path)
    # Этап 1 всегда использует промпт «1 письмо за вызов» — это ключевое условие:
    # генерируем по одному письму, чтобы модель не добавляла мета-текст между письмами.
    # Поле prompt_template из конфига здесь не используется — оно для других этапов/экспериментов
    prompt_template = load_prompt_template("llm_generate_one.txt")

    # --- Генерация по классам ---
    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 1] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

        # Генерируем контекст класса один раз — описание общих признаков писем
        context = generate_class_context(
            class_name, existing_texts, model, tokenizer,
            generation_params, system_prompt,
        )

        try:
            generated = augment_class(
                class_name=class_name,
                existing_texts=existing_texts,
                n_needed=n_needed,
                model=model,
                tokenizer=tokenizer,
                generation_params=generation_params,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                context=context,
                n_original=current_count,
            )
        except Exception as e:
            print(f"[Этап 1] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 1] Пропускаю класс, продолжаю с остальными")
            continue

        # Формируем строки для добавления в DataFrame
        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 1] Класс «{class_name}»: добавлено {len(generated)} текстов")
        if len(generated) != 0:
            for text in generated:
                print(f"Пример сгенерированного письма:\n{'-'*50}\n{text}\n")

        # Промежуточное сохранение каждые 3 класса — страховка от вылета ядра в Colab
        if new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 1] Промежуточное сохранение: {class_idx + 1} классов обработано, {len(df_tmp)} записей в файле")

    # --- Добавляем сгенерированные тексты к датасету ---
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
        help="Путь до JSON-конфига модели (например, configs/model_qwen.json)",
    )
    args = parser.parse_args()

    run(args.config)
