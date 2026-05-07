"""
stage1_llm_generate.py - Этап 1: генерация текстов через LLM (батчевый режим)

Берём классы где меньше 15 примеров и догоняем до 15 через LLM.
Генерация идёт батчами через vLLM, после - валидация и оценка LLM-судьёй.

Вход:  Data/train_after_eda.csv  (или data_after_stage1.csv если чекпоинт есть)
Выход: Data/data_after_stage1.csv

Запуск:
    python src/augmentation/stage1_llm_generate.py --config configs/model_vllm.json
"""

import sys
import random
import argparse
import copy
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
from src.augmentation.validation import (
    SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD_LOW, validate_generated_texts,
)


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 1
TARGET_COUNT = 15
MAX_RETRIES = 5
MAX_EXAMPLES_IN_PROMPT = 5
OVERSAMPLE_FACTOR = 5

# Если у класса мало оригиналов - модель быстро упирается в "почти копии",
# поэтому при таком случае поднимаем температуру для большего разнообразия
SMALL_CLASS_SOURCE_THRESHOLD = 6
SMALL_CLASS_TEMPERATURE = 0.9

# Прогрессивный порог косинусного сходства: первые попытки строгие
# (≈0.95), а если класс упорно не набирается - постепенно ослабляем
# фильтр шагом 0.01 за попытку, но не выше 0.98
SIM_THRESHOLD_BASE = SIMILARITY_THRESHOLD
SIM_THRESHOLD_MAX = SIMILARITY_THRESHOLD_LOW
SIM_INCREASE_AFTER_ATTEMPT = 2
SIM_STEP = 0.01


def _sampling_for_source_count(sampling_params, source_count: int):
    """Если оригиналов мало - поднимаем температуру для разнообразия."""
    params = copy.copy(sampling_params)
    if source_count <= SMALL_CLASS_SOURCE_THRESHOLD:
        params.temperature = SMALL_CLASS_TEMPERATURE
        print(f"  Малое число оригиналов ({source_count}) - "
              f"повышаю температуру генерации до {params.temperature:.2f}")
    return params


def generate_class_context(
    class_name: str,
    examples: list[str],
    llm,
    sampling_params,
    system_prompt: str | None = None,
) -> str:
    """
    Просим LLM описать класс по имеющимся примерам.
    Это описание потом подставляется в промпт генерации -
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

    # если LLM выпала с ошибкой
    return f"Официальные входящие письма класса «{class_name}»."


def build_prompt(template: str, class_name: str, examples: list[str], context: str = "") -> str:
    """Собирает промпт для генерации одного письма."""
    # Перемешиваем примеры, чтобы модель не запоминала порядок
    # и не копировала структуру первого попавшегося письма
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

    Схема: генерируем с запасом OVERSAMPLE_FACTOR× -> валидация фильтрами ->
    LLM-судья оценивает -> берём лучшие. Если после всего не хватает - пробуем
    ещё раз с чуть более мягким порогом сходства (до MAX_RETRIES попыток).
    """
    all_valid_texts = []
    # Копия - потому что в каждой удачной итерации сюда добавляются принятые
    # тексты, чтобы новые кандидаты валидировались уже с учётом свежей синтетики
    current_existing = list(existing_texts)

    # Если у класса очень мало оригиналов - берём температуру повыше
    class_sampling_params = _sampling_for_source_count(
        sampling_params,
        source_count=n_original if n_original is not None else len(existing_texts),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        still_needed = n_needed - len(all_valid_texts)
        if still_needed <= 0:
            break

        # Запас на отсев: часть улетит в фильтры, часть зарежет судья
        batch_size = int(still_needed * OVERSAMPLE_FACTOR) + 1
        print(f"  [Генерация] Раунд {attempt}/{MAX_RETRIES}: "
              f"генерируем батч из {batch_size} промптов (нужно ещё {still_needed})")

        # Разные промпты - каждый со своим случайным подмножеством примеров,
        # чтобы выходы были разнообразнее
        prompts = [
            build_prompt(prompt_template, class_name, current_existing, context=context)
            for _ in range(batch_size)
        ]

        # vLLM прожёвывает всю пачку за один проход по GPU - это и есть весь смысл батча
        raw_outputs = generate_batch(llm, class_sampling_params, prompts, system_prompt=system_prompt)

        # LLM иногда возвращает пустую строку (попало в стоп-токен раньше времени) - выкидываем
        candidates = [text for text in raw_outputs if text]

        if not candidates:
            print(f"  [Генерация] Раунд {attempt}: все выходы пустые, повторяю")
            continue

        # Первые SIM_INCREASE_AFTER_ATTEMPT попыток держим строгий порог сходства,
        # дальше каждую попытку послабляем на SIM_STEP - чтобы не застрять навсегда
        # на упёртых классах, где модель крутится вокруг одних и тех же формулировок
        relax_steps = max(0, attempt - SIM_INCREASE_AFTER_ATTEMPT)
        sim_threshold = min(SIM_THRESHOLD_BASE + relax_steps * SIM_STEP, SIM_THRESHOLD_MAX)

        # Фильтры: дубликаты, длина, язык, косинусное сходство и т.д.
        valid = validate_generated_texts(
            candidates, current_existing, class_name,
            similarity_threshold=sim_threshold, n_original=n_original,
        )
        n_after_validation = len(valid)

        # Судья сравнивает кандидата с оригиналами и описанием класса -
        # important: ему отдаём только оригиналы (existing_texts), а не накопленную
        # синтетику, иначе он начнёт мерить "похоже ли на сгенерированное ранее"
        valid = select_top_half(
            valid, class_name, llm,
            n_needed=still_needed, existing_texts=existing_texts,
            context=context,
        )

        to_take = min(len(valid), still_needed)
        all_valid_texts.extend(valid[:to_take])
        current_existing.extend(valid[:to_take])

        print(f"  [Генерация] Раунд {attempt}: "
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
    global SMALL_CLASS_SOURCE_THRESHOLD, SMALL_CLASS_TEMPERATURE
    global SIM_THRESHOLD_BASE, SIM_THRESHOLD_MAX, SIM_INCREASE_AFTER_ATTEMPT, SIM_STEP

    # Если передан pipeline_config - переопределяем глобалы значениями из JSON,
    # иначе остаются дефолты, прописанные в начале файла
    if pipeline_cfg is not None:
        s = pipeline_cfg.stage1
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        MAX_EXAMPLES_IN_PROMPT = s.max_examples_in_prompt
        SMALL_CLASS_SOURCE_THRESHOLD = int(
            s.get("small_class_source_threshold", SMALL_CLASS_SOURCE_THRESHOLD)
        )
        SMALL_CLASS_TEMPERATURE = float(
            s.get("small_class_temperature", SMALL_CLASS_TEMPERATURE)
        )
        SIM_INCREASE_AFTER_ATTEMPT = int(
            s.get("similarity_increase_after_attempt", SIM_INCREASE_AFTER_ATTEMPT)
        )
        SIM_STEP = float(s.get("similarity_step", SIM_STEP))
        # Базовый и максимальный пороги сходства живут в общей секции validation -
        # одни и те же значения нужны и stage1, и stage2/3
        v = pipeline_cfg.validation
        SIM_THRESHOLD_BASE = float(v.get("similarity_threshold", SIM_THRESHOLD_BASE))
        SIM_THRESHOLD_MAX = float(v.get("similarity_threshold_max", SIM_THRESHOLD_MAX))

    # Фиксируем сиды - иначе один и тот же ран будет давать разные пары промптов
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ЭТАП 1: LLM-генерация (< {TARGET_COUNT} -> {TARGET_COUNT})")
    print(f"       oversample={OVERSAMPLE_FACTOR}×, max_retries={MAX_RETRIES}, "
          f"sim {SIM_THRESHOLD_BASE:.2f}->{SIM_THRESHOLD_MAX:.2f} "
          f"(+{SIM_STEP} после {SIM_INCREASE_AFTER_ATTEMPT}-й попытки)")
    print("=" * 60)

    df = load_dataset(stage=STAGE)

    # Считаем кому не хватает примеров - этих и догоняем
    classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)

    if not classes_to_augment:
        # Нечего догонять - выходим, файл уже на месте, переписывать не надо
        print(f"[Этап 1] Все классы уже имеют >= {TARGET_COUNT} примеров, этап пропущен")
        return

    print(f"\n[Этап 1] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} -> нужно ещё {TARGET_COUNT - count}")

    # vLLM грузим один раз - каждая повторная инициализация ~30-60s,
    # а классов десятки, так что переиспользуем под все генерации
    llm, sampling_params, system_prompt = load_llm(config_path, pipeline_cfg=pipeline_cfg)
    prompt_template = load_prompt_template("llm_generate_one.txt")

    new_rows = []

    for class_idx, (class_name, current_count) in enumerate(classes_to_augment.items()):
        n_needed = TARGET_COUNT - current_count
        existing_texts = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

        print(f"\n[Этап 1] Класс «{class_name}»: есть {current_count}, нужно ещё {n_needed}")

        # Описание класса считаем один раз и подставляем во все промпты -
        # модели проще генерировать, когда она "понимает", про что класс
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
            # Не убиваем весь этап из-за одного больного класса - пропускаем и едем дальше
            print(f"[Этап 1] Ошибка при обработке класса «{class_name}»: {e}")
            print(f"[Этап 1] Пропускаю класс, продолжаю с остальными")
            continue

        for text in generated:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 1] Класс «{class_name}»: добавлено {len(generated)} текстов")

        # Промежуточный чекпоинт после каждого класса - если Colab/SLURM
        # упадёт посередине, восстановим уже сделанное вместо повторной генерации
        is_last_class = class_idx == len(classes_to_augment) - 1
        if new_rows and not is_last_class:
            df_tmp = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_checkpoint(df_tmp, stage=STAGE)
            print(f"[Этап 1] Промежуточное сохранение: "
                  f"{class_idx + 1} классов обработано, {len(df_tmp)} записей")

    # Финальное сохранение - здесь же фиксируем итоговый df
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"\n[Этап 1] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 1] Новых текстов не сгенерировано")

    save_checkpoint(df, stage=STAGE)

    # Финальная сводка - где какие классы вышли в норму, а где не дотянули
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
