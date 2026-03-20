"""
stage3_back_translation.py — Этап 3: обратный перевод (RU → EN → RU)

Берём классы с 35–49 примерами и доводим до 50 через обратный перевод.

Используем Helsinki-NLP/opus-mt (MarianMT) — две специализированные модели
для пары RU↔EN. Легче NLLB (~300 МБ каждая), обучены именно на RU-EN.

Вход:  Data/data_after_stage2.csv  (или data_after_stage3.csv, если чекпоинт есть)
Выход: Data/data_after_stage3.csv

Запуск:
    python src/augmentation/stage3_back_translation.py
"""

import sys
import gc
import random
from pathlib import Path

import torch
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

# Helsinki-NLP Opus-MT — две специализированные модели для пары RU↔EN
MODEL_RU_EN = "Helsinki-NLP/opus-mt-ru-en"
MODEL_EN_RU = "Helsinki-NLP/opus-mt-en-ru"

BATCH_SIZE = 64         # Opus-MT ~300 МБ каждая, обе влезают в T4 с запасом
MAX_LENGTH = 512        # Максимальная длина перевода в токенах
OVERSAMPLE_FACTOR = 3   # Генерируем в N раз больше, чем нужно — запас на отсев валидацией


def load_translation_models() -> tuple:
    """
    Загружает две Opus-MT модели: RU→EN и EN→RU.

    Обе модели лёгкие (~300 МБ), помещаются на GPU одновременно.
    Грузим на GPU если есть, иначе на CPU.

    Возвращает:
        Кортеж (model_ru_en, tok_ru_en, model_en_ru, tok_en_ru, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Перевод] Загружаю Opus-MT RU→EN: {MODEL_RU_EN} (устройство: {device})")
    tok_ru_en = MarianTokenizer.from_pretrained(MODEL_RU_EN)
    model_ru_en = MarianMTModel.from_pretrained(
        MODEL_RU_EN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    print(f"[Перевод] Загружаю Opus-MT EN→RU: {MODEL_EN_RU} (устройство: {device})")
    tok_en_ru = MarianTokenizer.from_pretrained(MODEL_EN_RU)
    model_en_ru = MarianMTModel.from_pretrained(
        MODEL_EN_RU,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    print("[Перевод] Обе модели загружены")
    return model_ru_en, tok_ru_en, model_en_ru, tok_en_ru, device


def load_sbert_on_gpu():
    """
    Загружает SBERT на GPU для быстрой валидации.
    Сбрасывает кэш в validation.py, чтобы не конфликтовать
    с CPU-версией, используемой на других этапах.
    """
    from src.augmentation.validation import get_sbert_model, SBERT_MODEL_NAME, _sbert_model
    from sentence_transformers import SentenceTransformer
    import src.augmentation.validation as val_module

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Если уже загружена на CPU — выгружаем
    if val_module._sbert_model is not None:
        del val_module._sbert_model
        val_module._sbert_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Валидация] Загружаю SBERT на {device}")
    sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    print(f"[Валидация] SBERT загружена на {device}")
    return sbert


def translate_batch(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    device: str,
) -> list[str]:
    """
    Переводит батч текстов через Opus-MT (MarianMT).

    Направление перевода определяется моделью (ru→en или en→ru).
    При ошибке возвращает пустые строки для элементов батча — чтобы не
    ронять весь пайплайн из-за одного проблемного батча.

    Аргументы:
        texts:     список текстов для перевода
        model:     модель MarianMT (одного направления)
        tokenizer: токенизатор MarianMT
        device:    устройство ("cuda" или "cpu")

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
        ).to(device)

        # do_sample + temperature дают разнообразие переводов: один и тот же
        # исходный текст может дать разные варианты, что снижает отсев дубликатов
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                do_sample=True,
                temperature=1.2,
                top_p=0.9,
            )

        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"  [Перевод] Ошибка при переводе батча: {e}")
        return [""] * len(texts)


def back_translate(
    texts: list[str],
    model_ru_en: MarianMTModel,
    tok_ru_en: MarianTokenizer,
    model_en_ru: MarianMTModel,
    tok_en_ru: MarianTokenizer,
    device: str,
) -> list[str]:
    """
    Обратный перевод списка текстов: RU → EN → RU.

    Две Opus-MT модели: одна переводит RU→EN, другая EN→RU.
    Обрабатывает батчами для скорости.

    Аргументы:
        texts:       список русскоязычных текстов
        model_ru_en: модель Opus-MT RU→EN
        tok_ru_en:   токенизатор RU→EN
        model_en_ru: модель Opus-MT EN→RU
        tok_en_ru:   токенизатор EN→RU
        device:      устройство ("cuda" или "cpu")

    Возвращает:
        Список обратно переведённых текстов той же длины, что и входной.
        Пустые строки на месте тех, где перевод не удался.
    """
    # --- RU → EN ---
    en_texts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="    RU→EN", leave=False):
        batch = texts[i:i + BATCH_SIZE]
        en_texts.extend(translate_batch(batch, model_ru_en, tok_ru_en, device))

    # --- EN → RU ---
    ru_texts = []
    for i in tqdm(range(0, len(en_texts), BATCH_SIZE), desc="    EN→RU", leave=False):
        batch = en_texts[i:i + BATCH_SIZE]
        ru_texts.extend(translate_batch(batch, model_en_ru, tok_en_ru, device))

    # Возвращаем результат той же длины — пустые строки остаются как маркеры ошибок
    return [t.strip() for t in ru_texts]


def select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """
    Выбирает тексты для обратного перевода.

    Распределяет равномерно: каждый оригинал используется примерно одинаковое
    число раз. Перемешиваем для разнообразия — при повторном переводе одного
    текста модель может дать тот же результат, поэтому повторы бесполезны.
    Но в сочетании с валидацией это нормально — дубликаты отсеются.

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
    model_ru_en, tok_ru_en, model_en_ru, tok_en_ru, device = load_translation_models()

    # --- Bulk-перевод: один проход RU→EN→RU для всех классов сразу ---
    # Вместо того чтобы гонять модель отдельно для каждого класса, собираем
    # все источники в один список — GPU загружен максимально, меньше накладных
    # расходов на инициализацию батчей. Потом разбиваем результат обратно по классам.

    new_rows = []


    class_state = {}
    for class_name, current_count in classes_to_augment.items():
        class_state[class_name] = {
            "existing": df[df[LABEL_COL] == class_name][TEXT_COL].tolist(),
            "n_needed": TARGET_COUNT - current_count,
            "accepted": [],
        }

    # --- Загрузка SBERT на GPU для валидации (обе модели помещаются в VRAM) ---
    sbert_model = load_sbert_on_gpu()

    for attempt in range(1, MAX_RETRIES + 1):
        # Классы, которым ещё не хватает текстов
        pending = {
            name: state for name, state in class_state.items()
            if len(state["accepted"]) < state["n_needed"]
        }
        if not pending:
            break

        print(f"\n[Этап 3] Попытка {attempt}/{MAX_RETRIES}: {len(pending)} классов ещё не доведены до цели")

        # Собираем все источники в один список — запоминаем, какому классу принадлежит каждый
        all_sources: list[str] = []
        source_class: list[str] = []  # параллельный список — метка класса для каждого источника

        for class_name, state in pending.items():
            still_needed = state["n_needed"] - len(state["accepted"])
            # Берём с запасом (OVERSAMPLE_FACTOR) — после валидации часть отсеется
            n_to_generate = still_needed * OVERSAMPLE_FACTOR
            sources = select_sources(state["existing"], n_to_generate)
            all_sources.extend(sources)
            source_class.extend([class_name] * len(sources))

        print(f"  Всего источников для перевода: {len(all_sources)}")

        # Один большой RU→EN→RU прогон по всем классам.
        all_translated = back_translate(
            all_sources, model_ru_en, tok_ru_en, model_en_ru, tok_en_ru, device
        )

        # Разбиваем результат обратно по классам, пропуская пустые переводы
        translated_by_class: dict[str, list[str]] = {name: [] for name in pending}
        for translated_text, class_name in zip(all_translated, source_class):
            if translated_text.strip():
                translated_by_class[class_name].append(translated_text.strip())

        # Валидируем и принимаем по каждому классу отдельно (SBERT уже на GPU)
        classes_completed_this_round = 0
        for class_idx, (class_name, state) in enumerate(class_state.items()):
            if class_name not in pending:
                continue

            still_needed = state["n_needed"] - len(state["accepted"])
            candidates = translated_by_class.get(class_name, [])

            if not candidates:
                continue

            # Текущие тексты класса = оригиналы + уже принятые в предыдущих попытках
            current_existing = state["existing"] + state["accepted"]
            valid = validate_generated_texts(
                candidates, current_existing, class_name,
                sbert_model=sbert_model,
            )

            take = min(len(valid), still_needed)
            state["accepted"].extend(valid[:take])

            print(f"  [{attempt}] «{class_name}»: получено {len(candidates)}, валидных {len(valid)}, "
                  f"принято {take}, итого {len(state['accepted'])}/{state['n_needed']}")

            if len(state["accepted"]) >= state["n_needed"]:
                classes_completed_this_round += 1

        # После каждого прогона сохраняем то, что уже накопили — за одну попытку
        tmp_rows = []
        for cn, st in class_state.items():
            for text in st["accepted"]:
                tmp_rows.append({TEXT_COL: text, LABEL_COL: cn})
        if tmp_rows:
            df_tmp = pd.concat([df, pd.DataFrame(tmp_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"  [Чекпоинт] Попытка {attempt}: сохранено {len(df_tmp)} записей")

    # --- Собираем принятые тексты из всех классов ---
    for class_name, state in class_state.items():
        accepted = state["accepted"]
        for text in accepted:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        if len(accepted) < state["n_needed"]:
            print(f"  [Внимание] «{class_name}»: удалось получить только "
                  f"{len(accepted)}/{state['n_needed']} за {MAX_RETRIES} попыток")

        print(f"[Этап 3] Класс «{class_name}»: добавлено {len(accepted)} текстов")
        if accepted:
            for text in accepted:
                print(f"Пример сгенерированного письма:\n{'-'*50}\n{text}\n")

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
