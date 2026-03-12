"""
stage2_paraphrase.py — Этап 2: парафраз текстов через LLM

Берём классы с 15–34 примерами (включая дополненные на этапе 1) и доводим
до 35 через перефразирование. Для каждого недостающего примера берём случайный
оригинальный текст класса и просим LLM переформулировать его — другая структура
предложений, синонимы, другой порядок, но тот же смысл.

Вход:  Data/data_after_stage1.csv  (или data_after_stage2.csv, если чекпоинт есть)
Выход: Data/data_after_stage2.csv

Запуск:
    python src/augmentation/stage2_paraphrase.py --config configs/model_qwen.json
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
from src.utils.config_loader import load_model_config


# --- Настройки этапа ---

STAGE = 2
TARGET_COUNT = 35       # Доводим каждый класс до 35 примеров
MAX_RETRIES = 10         # Сколько раз пробуем перефразировать, если валидация отсеяла слишком много
PARAPHRASE_PROMPT = "paraphrase.txt"  # Промпт по умолчанию (для 7B+); для слабых моделей — из конфига


def build_paraphrase_prompt(template: str, original_text: str, class_name: str) -> str:
    """
    Собирает промпт для перефразирования одного текста.

    Подставляет оригинальный текст и название класса в шаблон
    из prompts/paraphrase.txt.

    Аргументы:
        template:      текст шаблона промпта
        original_text: оригинальное письмо, которое нужно перефразировать
        class_name:    название класса (для контекста модели)

    Возвращает:
        Готовый промпт для передачи в LLM
    """
    return template.format(
        original_text=original_text,
        class_name=class_name,
    )


def paraphrase_text(
    original_text: str,
    class_name: str,
    model,
    tokenizer,
    generation_params: dict,
    prompt_template: str,
    system_prompt: str | None = None,
) -> str | None:
    """
    Перефразирует один текст через LLM.

    Отправляет оригинальное письмо в модель с промптом на переформулировку.
    Возвращает первый сгенерированный вариант или None, если генерация не удалась.

    Аргументы:
        original_text:    оригинальное письмо
        class_name:       название класса
        model:            загруженная LLM
        tokenizer:        токенизатор
        generation_params: параметры генерации
        prompt_template:  текст шаблона промпта

    Возвращает:
        Перефразированный текст или None при ошибке
    """
    prompt = build_paraphrase_prompt(prompt_template, original_text, class_name)
    results = generate_text(model, tokenizer, prompt, generation_params, system_prompt=system_prompt)

    if not results:
        return None

    # Берём первый результат — для парафраза одного текста больше и не нужно
    return results[0].strip()


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    model,
    tokenizer,
    generation_params: dict,
    prompt_template: str,
    system_prompt: str | None = None,
) -> list[str]:
    """
    Генерирует новые тексты для одного класса через перефразирование.

    Для каждого недостающего текста берём случайный оригинал из класса
    и перефразируем его. Каждый оригинал стараемся использовать примерно
    одинаковое число раз, чтобы разнообразие было максимальным.
    После генерации — валидация. Если отсеяли слишком много — повторяем.

    Аргументы:
        class_name:        название класса
        existing_texts:    уже имеющиеся тексты этого класса
        n_needed:          сколько новых текстов нужно
        model:             загруженная LLM
        tokenizer:         токенизатор
        generation_params: параметры генерации
        prompt_template:   текст шаблона промпта

    Возвращает:
        Список валидных перефразированных текстов
    """
    all_valid_texts = []
    # Все тексты, которые уже есть + принятые парафразы — для валидации
    current_existing = list(existing_texts)

    for attempt in range(1, MAX_RETRIES + 1):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        print(f"  [Попытка {attempt}/{MAX_RETRIES}] Нужно ещё {still_needed} парафразов")

        # Выбираем оригиналы для перефразирования — равномерно по кругу,
        # чтобы не перефразировать один и тот же текст десять раз подряд
        sources = _select_sources(existing_texts, still_needed)

        # Перефразируем каждый выбранный текст
        paraphrased = []
        for i, source_text in enumerate(sources):
            result = paraphrase_text(
                source_text, class_name,
                model, tokenizer, generation_params, prompt_template,
                system_prompt=system_prompt,
            )
            if result:
                paraphrased.append(result)

            # Прогресс внутри попытки 
            if (i + 1) % 5 == 0 or i == len(sources) - 1:
                print(f"    Перефразировано {i + 1}/{len(sources)}", end="\r")

        print()  # Перенос строки после прогресса
        print(f"  [Попытка {attempt}] Получено {len(paraphrased)} парафразов")

        # Валидация — отсеиваем дубликаты и слишком похожие
        valid = validate_generated_texts(paraphrased, current_existing, class_name)

       
        # Берём только столько, сколько нужно
        take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:take])
        current_existing.extend(valid[:take])

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось получить только "
              f"{len(all_valid_texts)}/{n_needed} парафразов за {MAX_RETRIES} попыток")

    return all_valid_texts


def _select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """
    Выбирает оригинальные тексты для перефразирования.

    Распределяем равномерно: если оригиналов 15, а нужно 20 парафразов —
    каждый текст перефразируем по одному разу, потом оставшиеся 5 берём случайно.

    Аргументы:
        existing_texts: список оригинальных текстов класса
        n_needed:       сколько парафразов нужно

    Возвращает:
        Список текстов-источников (с возможными повторами, если n_needed > len(existing))
    """
    sources = []

    # Сначала берём каждый оригинал по разу (перемешав для разнообразия)
    shuffled = list(existing_texts)
    random.shuffle(shuffled)

    # Повторяем полные «круги» по оригиналам, пока не наберём нужное
    full_rounds = n_needed // len(shuffled)
    remainder = n_needed % len(shuffled)

    for _ in range(full_rounds):
        round_copy = list(shuffled)
        random.shuffle(round_copy)
        sources.extend(round_copy)

    # Дополняем остаток
    if remainder > 0:
        extra = list(shuffled)
        random.shuffle(extra)
        sources.extend(extra[:remainder])

    return sources


def run(config_path: str) -> None:
    """
    Основная функция этапа 2 — точка входа.

    Загружает данные после этапа 1, определяет классы с < 35 примерами,
    перефразирует тексты, сохраняет чекпоинт.

    Аргументы:
        config_path: путь до JSON-конфига модели
    """
    # --- Фиксируем seed для воспроизводимости ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 2: Парафраз через LLM (< 35 → 35)")
    print("=" * 60)

    # --- Загрузка данных ---
    df = load_dataset(stage=STAGE)

    # --- Какие классы нуждаются в аугментации ---
    # После этапа 1 все классы должны иметь >= 15, но некоторые могут быть < 35
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 2] Все классы уже имеют >= 35 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 2] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    # --- Загрузка LLM ---
    # Используем ту же модель что и для этапа 1, но промпт и system_prompt другие
    model, tokenizer, generation_params, _, _ = load_llm(config_path)
    config = load_model_config(config_path)
    paraphrase_template_name = config.get("paraphrase_template", PARAPHRASE_PROMPT)
    system_prompt = config.get("paraphrase_system_prompt")
    prompt_template = load_prompt_template(paraphrase_template_name)

    # --- Парафраз по классам ---
    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 2] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

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
            )
        except Exception as e:
            print(f"[Этап 2] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 2] Пропускаю класс, продолжаю с остальными")
            continue

        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 2] Класс «{class_name}»: добавлено {len(generated)} парафразов")

        if len(generated) != 0:
            for text in generated:
                print(f"Пример сгенерированного письма:\n{'-'*50}\n{text}\n")

        # Промежуточное сохранение каждые 3 класса — страховка от вылета ядра
        if (class_idx + 1) % 3 == 0 and new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 2] Промежуточное сохранение: {class_idx + 1} классов обработано, {len(df_tmp)} записей в файле")

    # --- Добавляем перефразированные тексты к датасету ---
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 2] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 2] Новых текстов не сгенерировано")

    # --- Сохраняем чекпоинт ---
    save_checkpoint(df, stage=STAGE)

    # --- Итоговая статистика ---
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
        help="Путь до JSON-конфига модели (например, configs/model_qwen.json)",
    )
    args = parser.parse_args()

    run(args.config)
