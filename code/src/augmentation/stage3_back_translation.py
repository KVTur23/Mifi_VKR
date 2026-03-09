"""
stage3_back_translation.py — Этап 3: обратный перевод (RU → EN → RU)

Берём классы с 35–49 примерами и доводим до 50 через обратный перевод.
Идея простая: переводим текст на английский, потом обратно на русский.
При обратном переводе неизбежно меняется формулировка — получаем
новый текст с тем же смыслом, но другими словами.

Используем лёгкие модели Helsinki-NLP (MarianMT) — они работают быстро,
не требуют GPU и хорошо справляются с деловым текстом. Обработка батчами
для скорости.

Вход:  Data/data_after_stage2.csv  (или data_after_stage3.csv, если чекпоинт есть)
Выход: Data/data_after_stage3.csv

Запуск:
    python src/augmentation/stage3_back_translation.py
"""

import sys
import random
from pathlib import Path

import pandas as pd
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
)
from src.augmentation.validation import validate_generated_texts


# --- Настройки этапа ---

STAGE = 3
TARGET_COUNT = 50       # Доводим каждый класс до 50 примеров
MAX_RETRIES = 5         # Сколько раз пробуем, если валидация отсеяла слишком много

# Helsinki-NLP модели для перевода — лёгкие, быстрые, работают без GPU
MODEL_RU_EN = "Helsinki-NLP/opus-mt-ru-en"
MODEL_EN_RU = "Helsinki-NLP/opus-mt-en-ru"

BATCH_SIZE = 8          # Размер батча для перевода — больше = быстрее, но больше памяти
MAX_LENGTH = 512        # Максимальная длина перевода в токенах


def load_translation_models() -> tuple:
    """
    Загружает обе модели перевода: RU→EN и EN→RU.

    Модели MarianMT от Helsinki-NLP — компактные (~300 МБ каждая),
    работают на CPU за разумное время. Загружаем обе сразу, чтобы
    не тратить время на повторную загрузку для каждого класса.

    Возвращает:
        Кортеж (ru_en_model, ru_en_tokenizer, en_ru_model, en_ru_tokenizer)
    """
    print(f"[Перевод] Загружаю модель RU→EN: {MODEL_RU_EN}")
    ru_en_tokenizer = MarianTokenizer.from_pretrained(MODEL_RU_EN)
    ru_en_model = MarianMTModel.from_pretrained(MODEL_RU_EN)
    ru_en_model.eval()

    print(f"[Перевод] Загружаю модель EN→RU: {MODEL_EN_RU}")
    en_ru_tokenizer = MarianTokenizer.from_pretrained(MODEL_EN_RU)
    en_ru_model = MarianMTModel.from_pretrained(MODEL_EN_RU)
    en_ru_model.eval()

    print("[Перевод] Обе модели загружены")
    return ru_en_model, ru_en_tokenizer, en_ru_model, en_ru_tokenizer


def translate_batch(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
) -> list[str]:
    """
    Переводит батч текстов через MarianMT.

    Токенизирует, прогоняет через модель, декодирует. Обрабатывает
    ошибки на уровне батча — если один батч упал, возвращаем пустые строки
    для его элементов, чтобы не терять остальные.

    Аргументы:
        texts:     список текстов для перевода
        model:     модель перевода (MarianMT)
        tokenizer: токенизатор модели

    Возвращает:
        Список переведённых текстов (того же размера, что и входной)
    """
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        outputs = model.generate(**inputs, max_length=MAX_LENGTH)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"  [Перевод] Ошибка при переводе батча: {e}")
        return [""] * len(texts)


def back_translate(
    texts: list[str],
    ru_en_model: MarianMTModel,
    ru_en_tokenizer: MarianTokenizer,
    en_ru_model: MarianMTModel,
    en_ru_tokenizer: MarianTokenizer,
) -> list[str]:
    """
    Обратный перевод списка текстов: RU → EN → RU.

    Обрабатывает батчами для скорости. Сначала весь список переводится
    на английский, потом обратно на русский. Пустые результаты (ошибки
    перевода) отбрасываются.

    Аргументы:
        texts:           список русскоязычных текстов
        ru_en_model:     модель RU→EN
        ru_en_tokenizer: токенизатор RU→EN
        en_ru_model:     модель EN→RU
        en_ru_tokenizer: токенизатор EN→RU

    Возвращает:
        Список обратно переведённых текстов (может быть короче входного,
        если часть переводов не удалась)
    """
    # --- RU → EN ---
    en_texts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="    RU→EN", leave=False):
        batch = texts[i:i + BATCH_SIZE]
        en_texts.extend(translate_batch(batch, ru_en_model, ru_en_tokenizer))

    # --- EN → RU ---
    ru_texts = []
    for i in tqdm(range(0, len(en_texts), BATCH_SIZE), desc="    EN→RU", leave=False):
        batch = en_texts[i:i + BATCH_SIZE]
        ru_texts.extend(translate_batch(batch, en_ru_model, en_ru_tokenizer))

    # Убираем пустые строки — результат ошибок перевода
    result = [t.strip() for t in ru_texts if t.strip()]
    return result


def select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """
    Выбирает тексты для обратного перевода.

    Распределяет равномерно: каждый оригинал используется примерно одинаковое
    число раз. Перемешиваем для разнообразия — при повторном переводе одного
    текста MarianMT даст тот же результат (детерминированная модель),
    поэтому повторы одного текста бесполезны. Но в сочетании с валидацией
    это нормально — дубликаты отсеются.

    Аргументы:
        existing_texts: оригинальные тексты класса
        n_needed:       сколько новых текстов нужно

    Возвращает:
        Список текстов-источников для перевода
    """
    sources = []
    shuffled = list(existing_texts)
    random.shuffle(shuffled)

    full_rounds = n_needed // len(shuffled)
    remainder = n_needed % len(shuffled)

    for _ in range(full_rounds):
        sources.extend(shuffled)

    if remainder > 0:
        sources.extend(shuffled[:remainder])

    return sources


def augment_class(
    class_name: str,
    existing_texts: list[str],
    n_needed: int,
    ru_en_model: MarianMTModel,
    ru_en_tokenizer: MarianTokenizer,
    en_ru_model: MarianMTModel,
    en_ru_tokenizer: MarianTokenizer,
) -> list[str]:
    """
    Генерирует новые тексты для одного класса через обратный перевод.

    Выбирает оригиналы, переводит RU→EN→RU, валидирует результат.
    Если после валидации не хватает — повторяет с другими оригиналами.

    Аргументы:
        class_name:        название класса
        existing_texts:    уже имеющиеся тексты класса
        n_needed:          сколько новых текстов нужно
        ru_en_model:       модель RU→EN
        ru_en_tokenizer:   токенизатор RU→EN
        en_ru_model:       модель EN→RU
        en_ru_tokenizer:   токенизатор EN→RU

    Возвращает:
        Список валидных обратно переведённых текстов
    """
    all_valid_texts = []
    current_existing = list(existing_texts)

    for attempt in range(1, MAX_RETRIES + 1):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        print(f"  [Попытка {attempt}/{MAX_RETRIES}] Нужно ещё {still_needed} текстов")

        # Выбираем оригиналы и переводим туда-обратно
        sources = select_sources(existing_texts, still_needed)
        translated = back_translate(
            sources,
            ru_en_model, ru_en_tokenizer,
            en_ru_model, en_ru_tokenizer,
        )

        print(f"  [Попытка {attempt}] Получено {len(translated)} обратных переводов")

        # Валидация — после обратного перевода особенно важна проверка языка,
        # потому что MarianMT иногда оставляет куски на английском
        valid = validate_generated_texts(translated, current_existing, class_name)

        take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:take])
        current_existing.extend(valid[:take])

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось получить только "
              f"{len(all_valid_texts)}/{n_needed} текстов за {MAX_RETRIES} попыток")

    return all_valid_texts


def run() -> None:
    """
    Основная функция этапа 3 — точка входа.

    Загружает данные после этапа 2, определяет классы с < 50 примерами,
    прогоняет обратный перевод, сохраняет чекпоинт.
    """
    # --- Фиксируем seed для воспроизводимости ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 3: Обратный перевод (< 50 → 50)")
    print("=" * 60)

    # --- Загрузка данных ---
    df = load_dataset(stage=STAGE)

    # --- Какие классы нуждаются в аугментации ---
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 3] Все классы уже имеют >= 50 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 3] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    # --- Загрузка моделей перевода ---
    ru_en_model, ru_en_tokenizer, en_ru_model, en_ru_tokenizer = load_translation_models()

    # --- Обратный перевод по классам ---
    new_rows = []

    for class_name, current_count in classes_to_augment.items():
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 3] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

        try:
            generated = augment_class(
                class_name=class_name,
                existing_texts=existing_texts,
                n_needed=n_needed,
                ru_en_model=ru_en_model,
                ru_en_tokenizer=ru_en_tokenizer,
                en_ru_model=en_ru_model,
                en_ru_tokenizer=en_ru_tokenizer,
            )
        except Exception as e:
            print(f"[Этап 3] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 3] Пропускаю класс, продолжаю с остальными")
            continue

        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 3] Класс «{class_name}»: добавлено {len(generated)} текстов")
        if len(generated) != 0:
            for text in generated:
                print(f"Пример сгенерированного письма:\n{"-"*50}\n{text}\n")

    # --- Добавляем обратные переводы к датасету ---
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 3] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 3] Новых текстов не сгенерировано")

    # --- Сохраняем чекпоинт ---
    save_checkpoint(df, stage=STAGE)

    # --- Итоговая статистика ---
    print(f"\n[Этап 3] Итоговое распределение:")
    dist = get_class_distribution(df)
    for name, count in dist.items():
        marker = " ✓" if count >= TARGET_COUNT else " ✗"
        print(f"  «{name}»: {count}{marker}")

    print(f"\n[Этап 3] Завершён. Всего записей: {len(df)}")


if __name__ == "__main__":
    run()
