"""
stage2_paraphrase.py — Этап 2: парафраз текстов через LLM (батчевый режим)

Берём классы с 15–34 примерами (включая дополненные на этапе 1) и доводим
до 35 через перефразирование. Для каждого недостающего примера берём случайный
оригинальный текст класса и просим LLM переформулировать его.

Использует vLLM для батчевого инференса — все парафразы одного класса
генерируются за один вызов GPU.

Вход:  Data/data_after_stage1.csv  (или data_after_stage2.csv, если чекпоинт есть)
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
from src.augmentation.llm_utils import load_llm, generate_batch, load_prompt_template, select_top_half
from src.augmentation.validation import validate_generated_texts
from src.utils.config_loader import load_model_config


# --- Настройки этапа ---

STAGE = 2
TARGET_COUNT = 35       # Доводим каждый класс до 35 примеров
MAX_RETRIES = 5          # Сколько раундов батчевой генерации при нехватке
PARAPHRASE_PROMPT = "paraphrase.txt"


def build_paraphrase_prompt(template: str, original_text: str, class_name: str) -> str:
    """
    Собирает промпт для перефразирования одного текста.
    """
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
) -> list[str]:
    """
    Генерирует новые тексты для одного класса через батчевый парафраз.

    За один раунд собирает все промпты и отправляет в vLLM одним батчем.
    """
    all_valid_texts = []
    current_existing = list(existing_texts)

    for attempt in range(1, MAX_RETRIES + 1):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        # Генерируем с запасом 3x — часть отсеется валидацией + LLM-судья отберёт лучших
        batch_size = int(still_needed * 3) + 1
        print(f"  [Раунд {attempt}/{MAX_RETRIES}] Нужно ещё {still_needed}, "
              f"генерируем батч из {batch_size} парафразов")

        # Выбираем оригиналы для перефразирования — равномерно по кругу
        sources = _select_sources(existing_texts, batch_size)

        # Собираем все промпты
        prompts = [
            build_paraphrase_prompt(prompt_template, source, class_name)
            for source in sources
        ]

        # Батчевая генерация через vLLM
        raw_outputs = generate_batch(llm, sampling_params, prompts, system_prompt=system_prompt)

        # Собираем непустые кандидаты
        paraphrased = [text for text in raw_outputs if text]

        print(f"  [Раунд {attempt}] Получено {len(paraphrased)} парафразов")

        if not paraphrased:
            continue

        # Валидация
        valid = validate_generated_texts(paraphrased, current_existing, class_name)

        # LLM-судья оценивает и отсеивает слабые тексты
        valid = select_top_half(
            valid, class_name, llm, sampling_params,
            n_needed=still_needed, system_prompt=system_prompt,
        )

        # Берём только сколько нужно
        take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:take])
        current_existing.extend(valid[:take])

        print(f"  [Раунд {attempt}] Прошло валидацию {len(valid)}, "
              f"принято {take}, всего {len(all_valid_texts)}/{n_needed}")

    if len(all_valid_texts) < n_needed:
        print(f"  [Внимание] Класс «{class_name}»: удалось получить только "
              f"{len(all_valid_texts)}/{n_needed} парафразов за {MAX_RETRIES} раундов")

    return all_valid_texts


def _select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """
    Выбирает оригинальные тексты для перефразирования равномерно по кругу.
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


def run(config_path: str) -> None:
    """
    Основная функция этапа 2 — точка входа.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 2: Парафраз через LLM (< 35 → 35)")
    print("=" * 60)

    # --- Загрузка данных ---
    df = load_dataset(stage=STAGE)

    # --- Какие классы нуждаются в аугментации ---
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        print("[Этап 2] Все классы уже имеют >= 35 примеров, этап пропущен")
        save_checkpoint(df, stage=STAGE)
        return

    print(f"\n[Этап 2] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    # --- Загрузка LLM ---
    llm, sampling_params, _ = load_llm(config_path)
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
                llm=llm,
                sampling_params=sampling_params,
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

        # Промежуточное сохранение
        if new_rows:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 2] Промежуточное сохранение: "
                  f"{class_idx + 1} классов обработано, {len(df_tmp)} записей")

    # --- Добавляем перефразированные тексты ---
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 2] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 2] Новых текстов не сгенерировано")

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
        help="Путь до JSON-конфига модели (например, configs/model_vllm.json)",
    )
    args = parser.parse_args()

    run(args.config)
