"""
stage3_back_translation.py — Этап 3: обратный перевод (RU → EN → RU)

Берём классы с 35–49 примерами и доводим до 50 через обратный перевод.

Используем NLLB-200 от Meta 

Вход:  Data/data_after_stage2.csv  (или data_after_stage3.csv, если чекпоинт есть)
Выход: Data/data_after_stage3.csv

Запуск:
    python src/augmentation/stage3_back_translation.py
"""

import sys
import random
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

# NLLB-200 от Meta — одна модель, переводит в обе стороны.
MODEL_NLLB = "facebook/nllb-200-distilled-600M"

# Языковые коды в формате NLLB (flores200)
LANG_RU = "rus_Cyrl"
LANG_EN = "eng_Latn"

BATCH_SIZE = 64         # T4 (16GB): NLLB-600M ~1.2GB, остаток — для батча. 
MAX_LENGTH = 512        # Максимальная длина перевода в токенах


def load_translation_models() -> tuple:
    """
    Загружает модель NLLB-200 и токенизатор.

    NLLB переводит в обе стороны — одна модель вместо двух. Направление
    перевода задаётся через forced_bos_token_id при генерации.
    Грузим на GPU если есть, иначе на CPU.

    Возвращает:
        Кортеж (model, tokenizer, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Перевод] Загружаю NLLB-200: {MODEL_NLLB} (устройство: {device})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NLLB)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NLLB,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    print("[Перевод] Модель загружена")
    return model, tokenizer, device


def translate_batch(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str,
) -> list[str]:
    """
    Переводит батч текстов через NLLB-200.

    Направление перевода задаётся парой src_lang / tgt_lang.
    При ошибке возвращает пустые строки для элементов батча — чтобы не
    ронять весь пайплайн из-за одного проблемного батча.

    Аргументы:
        texts:     список текстов для перевода
        model:     модель NLLB-200
        tokenizer: токенизатор NLLB
        src_lang:  язык источника (например, "rus_Cyrl")
        tgt_lang:  язык перевода (например, "eng_Latn")
        device:    устройство ("cuda" или "cpu")

    Возвращает:
        Список переведённых текстов (того же размера, что и входной)
    """
    try:
        # Говорим токенизатору, с какого языка переводим
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        # Говорим модели, на какой язык переводить — через forced_bos_token_id
        target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=MAX_LENGTH,
            )

        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"  [Перевод] Ошибка при переводе батча: {e}")
        return [""] * len(texts)


def back_translate(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> list[str]:
    """
    Обратный перевод списка текстов: RU → EN → RU.

    Одна NLLB-модель переводит в обе стороны — направление задаётся
    через src_lang/tgt_lang. Обрабатывает батчами для скорости.
    Пустые результаты (ошибки перевода) отбрасываются.

    Аргументы:
        texts:     список русскоязычных текстов
        model:     модель NLLB-200
        tokenizer: токенизатор NLLB
        device:    устройство ("cuda" или "cpu")

    Возвращает:
        Список обратно переведённых текстов той же длины, что и входной.
        Пустые строки на месте тех, где перевод не удался — это позволяет
        вызывающему коду сохранить соответствие индексов.
    """
    # --- RU → EN ---
    en_texts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="    RU→EN", leave=False):
        batch = texts[i:i + BATCH_SIZE]
        en_texts.extend(translate_batch(batch, model, tokenizer, LANG_RU, LANG_EN, device))

    # --- EN → RU ---
    ru_texts = []
    for i in tqdm(range(0, len(en_texts), BATCH_SIZE), desc="    EN→RU", leave=False):
        batch = en_texts[i:i + BATCH_SIZE]
        ru_texts.extend(translate_batch(batch, model, tokenizer, LANG_EN, LANG_RU, device))

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

    # --- Загрузка модели перевода ---
    model, tokenizer, device = load_translation_models()

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
            # Берём оригиналы пропорционально тому, сколько ещё нужно
            sources = select_sources(state["existing"], still_needed)
            all_sources.extend(sources)
            source_class.extend([class_name] * len(sources))

        print(f"  Всего источников для перевода: {len(all_sources)}")

        # Один большой RU→EN→RU прогон по всем классам.
        # back_translate возвращает список той же длины, что all_sources —
        # ошибочные переводы заменены пустыми строками, а не выброшены,
        # чтобы сохранить соответствие индексов.
        all_translated = back_translate(all_sources, model, tokenizer, device)

        # Разбиваем результат обратно по классам, пропуская пустые переводы
        translated_by_class: dict[str, list[str]] = {name: [] for name in pending}
        for translated_text, class_name in zip(all_translated, source_class):
            if translated_text.strip():
                translated_by_class[class_name].append(translated_text.strip())

        # Валидируем и принимаем по каждому классу отдельно
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
            valid = validate_generated_texts(candidates, current_existing, class_name)

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
