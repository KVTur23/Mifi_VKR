"""
stage3_back_translation.py - Этап 3: обратный перевод (RU -> en/de/fr -> RU)

Берём классы с 35-49 примерами и доводим до 50 через обратный перевод.
NLLB-200 переводит RU -> (en/de/fr) -> RU, после фильтры (длина, langdetect,
косинусное сходство) отбраковывают мусор, и из валидного пула берём первые
N со случайным перемешиванием. По умолчанию LLM-судья отключён: на NLLB-выходах
он стабильно ставит низкие оценки и режет ~92% и так чистого пула.
Включается через `stage3.use_judge: true` в pipeline_config.

Вход:  Data/data_after_stage2.csv  (или data_after_stage3.csv если чекпоинт есть)
Выход: Data/data_after_stage3.csv

Запуск:
    python src/augmentation/stage3_back_translation.py --config configs/model_vllm.json
"""

import re
import sys
import gc
import json
import random
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, save_checkpoint, get_class_distribution,
    get_classes_to_augment, TEXT_COL, LABEL_COL, RANDOM_SEED,
    DATA_DIR, STAGE_FILES,
)
from src.augmentation.validation import SIMILARITY_THRESHOLD, validate_generated_texts


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 3
TARGET_COUNT = 50
MAX_RETRIES = 20
MODEL_NLLB = "facebook/nllb-200-3.3B"
BATCH_SIZE = 64
OVERSAMPLE_FACTOR = 3
MIN_JUDGE_SCORE_STAGE3 = 2.5

# LLM-судья на этом этапе по факту режет почти всё (NLLB-выходы получают
# средние оценки 1.0-2.2 при пороге 2.5), а каждый его прогон - это +30s
# на загрузку vLLM × N промежуточных языков × N кругов. Поэтому по умолчанию выключен:
# кандидаты после фильтров идут прямо в чекпоинт со случайной выборкой.
USE_JUDGE_STAGE3 = False

LANG_RU = "rus_Cyrl"
LANG_EN = "eng_Latn"
LANG_DE = "deu_Latn"
LANG_FR = "fra_Latn"
PIVOT_LANGS = [LANG_EN, LANG_DE, LANG_FR]

# Сколько раундов промежуточных языков делаем. Каждый следующий раунд послабляет порог
# косинусного сходства на SIM_STEP_PER_ROUND - добираем хвост упёртых классов.
# В отличие от stage1/2 шаг применяется per-round, а не per-attempt
PIVOT_ROUNDS = 5
SIM_STEP_PER_ROUND = 0.10

# Для NLLB temperature/top_p всегда фиксированные - модель не реагирует на
# "малое число оригиналов" так, как vLLM на стейджах 1/2. Здесь это просто
# хардкод в translate_batch (1.0 / 0.9), отдельной настройки не делаем.
MAX_LENGTH = 512

# regex для NER-плейсхолдеров типа [PERSON], [ORGANIZATION], [DATE_TIME] и т.д.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:\s[A-Z_]*)?\]")

# промежуточный csv для пар фазы 1 - при рестарте подхватываем и сразу в фазу 2
_PAIRS_CSV = DATA_DIR / "_stage3_pairs_cache.csv"


def mask_placeholders(text: str) -> tuple[str, list[str]]:
    """
    Заменяет NER-плейсхолдеры на короткие маркеры <0>, <1>, ...
    NLLB их не трогает - они короткие и похожи на HTML-теги.
    Возвращает (замаскированный текст, список оригинальных плейсхолдеров).
    """
    placeholders = _PLACEHOLDER_RE.findall(text)
    masked = text
    for i, ph in enumerate(placeholders):
        # заменяем по одному вхождению за раз - на случай одинаковых плейсхолдеров
        masked = masked.replace(ph, f"<{i}>", 1)
    return masked, placeholders


def unmask_placeholders(text: str, placeholders: list[str]) -> str:
    """
    Восстанавливает оригинальные плейсхолдеры из маркеров <0>, <1>, ...
    Если NLLB потеряла маркер - пропускаем, текст останется без него.
    """
    for i, ph in enumerate(placeholders):
        text = text.replace(f"<{i}>", ph, 1)
    return text


def load_translation_models() -> tuple:
    """Грузит NLLB-200 на GPU (или CPU если нет)."""
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


def unload_from_gpu(*objects):
    """Выгружает модели из GPU и чистит память - освобождаем место для vLLM."""
    # del внутри функции убивает только локальные ссылки,
    # поэтому снаружи тоже надо занулить переменные
    for obj in objects:
        if hasattr(obj, 'cpu'):
            obj.cpu()
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[GPU] Память очищена")


def load_sbert_on_gpu():
    """Грузит SBERT на GPU для быстрой валидации."""
    from src.augmentation.validation import SBERT_MODEL_NAME
    from sentence_transformers import SentenceTransformer
    import src.augmentation.validation as val_module

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # если уже на CPU - выгружаем
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
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str,
) -> list[str]:
    """Переводит батч текстов через NLLB-200."""
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=MAX_LENGTH,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
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
    pivot_lang: str = LANG_EN,
) -> list[str]:
    """
    Обратный перевод: RU -> промежуточный -> RU с сохранением NER-плейсхолдеров.

    Перед переводом маскируем [PERSON], [ORGANIZATION] и т.д. в короткие <0>, <1>, ...
    После перевода восстанавливаем обратно.
    """
    # маскируем плейсхолдеры - NLLB их не ломает
    masked_texts = []
    all_placeholders = []
    for text in texts:
        masked, phs = mask_placeholders(text)
        masked_texts.append(masked)
        all_placeholders.append(phs)

    # RU -> промежуточный
    pivot_texts = []
    for i in tqdm(range(0, len(masked_texts), BATCH_SIZE), desc=f"    RU->{pivot_lang}", leave=False):
        batch = masked_texts[i:i + BATCH_SIZE]
        pivot_texts.extend(translate_batch(batch, model, tokenizer, LANG_RU, pivot_lang, device))

    # промежуточный -> RU
    ru_texts = []
    for i in tqdm(range(0, len(pivot_texts), BATCH_SIZE), desc=f"    {pivot_lang}->RU", leave=False):
        batch = pivot_texts[i:i + BATCH_SIZE]
        ru_texts.extend(translate_batch(batch, model, tokenizer, pivot_lang, LANG_RU, device))

    # восстанавливаем плейсхолдеры
    result = []
    for text, phs in zip(ru_texts, all_placeholders):
        result.append(unmask_placeholders(text.strip(), phs))

    return result


def select_sources(existing_texts: list[str], n_needed: int) -> list[str]:
    """Выбирает оригиналы для перевода равномерно по кругу."""
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


def run(config_path: str, pipeline_cfg=None) -> None:
    """
    Основная функция этапа 3.

    Для каждого промежуточного языка отдельно:
    1. NLLB перевод + валидация фильтрами, копим пары оригинал->перевод
    2. Выгружаем NLLB, грузим vLLM - LLM-судья отбирает лучшие
    3. Сохраняем чекпоинт и следующим промежуточным языком добираем только оставшиеся классы
    """
    global TARGET_COUNT, MAX_RETRIES, MODEL_NLLB, BATCH_SIZE, OVERSAMPLE_FACTOR, MIN_JUDGE_SCORE_STAGE3
    global PIVOT_ROUNDS, SIM_STEP_PER_ROUND, USE_JUDGE_STAGE3

    # Если передан pipeline_config - переопределяем глобалы значениями из JSON,
    # иначе остаются дефолты из шапки модуля
    if pipeline_cfg is not None:
        s = pipeline_cfg.stage3
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        MIN_JUDGE_SCORE_STAGE3 = s.min_judge_score
        MODEL_NLLB = pipeline_cfg.gpu.nllb_model
        BATCH_SIZE = pipeline_cfg.gpu.nllb_batch_size
        # _DotDict наследует dict - используем .get() с default
        PIVOT_ROUNDS = int(s.get("pivot_rounds", PIVOT_ROUNDS))
        SIM_STEP_PER_ROUND = float(s.get("similarity_step", SIM_STEP_PER_ROUND))
        USE_JUDGE_STAGE3 = bool(s.get("use_judge", USE_JUDGE_STAGE3))

    # Фиксируем сиды - иначе каждый ран будет тасовать BT-источники по-разному
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Считаем максимальный порог, до которого может дойти круг (для лога):
    # на последнем round'e он будет = base + (PIVOT_ROUNDS-1)*step. Если выше 1.0 -
    # это намеренно: косинус ограничен 1.0, так что фильтр фактически отключается
    sim_max = SIMILARITY_THRESHOLD + (PIVOT_ROUNDS - 1) * SIM_STEP_PER_ROUND

    print("=" * 60)
    print(f"ЭТАП 3: Обратный перевод (< {TARGET_COUNT} -> {TARGET_COUNT})")
    print(f"       источник BT: ТОЛЬКО ОРИГИНАЛЫ (train_after_eda.csv)")
    print(f"       NLLB: {MODEL_NLLB}, batch={BATCH_SIZE}, oversample={OVERSAMPLE_FACTOR}×")
    print(f"       промежуточные языки: {len(PIVOT_LANGS)} ({', '.join(PIVOT_LANGS)}), "
          f"rounds: {PIVOT_ROUNDS}, sim {SIMILARITY_THRESHOLD:.2f}->{sim_max:.2f} "
          f"(+{SIM_STEP_PER_ROUND} за круг)")
    print(f"       LLM-судья: "
          f"{'enabled, threshold=' + str(MIN_JUDGE_SCORE_STAGE3) if USE_JUDGE_STAGE3 else 'disabled (random pick from validated pool)'}")
    print("=" * 60)

    # ==========================================================
    # Определяем сценарий запуска:
    # 1) stage3.csv есть, все ≥ 50       -> пропуск
    # 2) stage3.csv есть, часть < 50     -> доаугментация оставшихся
    # 3) stage3.csv нет                  -> полный прогон от stage 2
    # ==========================================================

    stage3_file = DATA_DIR / STAGE_FILES[STAGE]
    has_stage3 = stage3_file.exists()

    if has_stage3:
        df = pd.read_csv(stage3_file)
        print(f"[Данные] Найден чекпоинт этапа 3: {stage3_file.name} ({len(df)} записей)")
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print(f"[Этап 3] Все классы уже ≥ {TARGET_COUNT} - этап пропущен")
            return
        print(f"[Этап 3] Чекпоинт неполный, {len(classes_to_augment)} классов < {TARGET_COUNT} - доаугментируем")
    else:
        df = load_dataset(stage=STAGE - 1)
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            # Нечего догонять - выходим, файл уже на месте
            print(f"[Этап 3] Все классы уже имеют >= {TARGET_COUNT} примеров, этап пропущен")
            return

    print(f"\n[Этап 3] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} -> нужно ещё {TARGET_COUNT - count}")

    # Загружаем оригиналы (train_after_eda.csv) - BT идёт только от них.
    # Это устраняет каскад (stage 1/2 -> stage 3), когда синтетика прошлых
    # стейджей переводилась туда-обратно и копила шум NLLB поверх шума LLM
    df_original = load_dataset(stage=0)

    # Каждый промежуточный язык - отдельный цикл: перевод -> фильтры -> отбор -> чекпоинт.
    # После первого круга en -> de -> fr запускаем следующий и повышаем
    # верхний порог косинусного сходства на SIM_STEP_PER_ROUND, чтобы добрать
    # хвост упёртых классов
    pivot_schedule = [
        (
            round_idx,
            pivot_idx,
            pivot_lang,
            SIMILARITY_THRESHOLD + (round_idx - 1) * SIM_STEP_PER_ROUND,
        )
        for round_idx in range(1, PIVOT_ROUNDS + 1)
        for pivot_idx, pivot_lang in enumerate(PIVOT_LANGS, start=1)
    ]

    # NLLB и SBERT грузим один раз на этап и держим до конца - раньше выгружали
    # между промежуточными языками, чтобы освободить GPU под vLLM-судью; теперь судьи нет
    # и эта пляска просто тратит ~30-40s на каждой загрузке (×15 итераций)
    model, tokenizer, device = load_translation_models()
    sbert_model = load_sbert_on_gpu()

    for pivot_round, pivot_idx, pivot_lang, sim_threshold in pivot_schedule:
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print("\n[Этап 3] Все классы уже ≥ 50 - оставшиеся промежуточные языки не нужны")
            break

        print(
            f"\n[Этап 3] Круг {pivot_round}/{PIVOT_ROUNDS}, "
            f"промежуточный {pivot_idx}/{len(PIVOT_LANGS)}: {pivot_lang}, "
            f"cosine_threshold={sim_threshold:.2f}"
        )
        for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
            print(f"  «{name}»: {count} -> нужно ещё {TARGET_COUNT - count}")

        class_state = {}
        for class_name, current_count in classes_to_augment.items():
            existing = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

            # Источник для BT - только оригиналы из train_after_eda.csv
            bt_sources = df_original[
                df_original[LABEL_COL] == class_name
            ][TEXT_COL].tolist()
            if not bt_sources:
                # Не должно случаться после EDA - но если случилось,
                # пропускаем класс, чем брать синтетику и тащить каскад
                print(f"  [Пропуск] Нет оригиналов для «{class_name}» в train_after_eda.csv")
                continue

            class_state[class_name] = {
                "existing": existing,
                "bt_sources": bt_sources,
                "n_needed": TARGET_COUNT - current_count,
                "accepted_pairs": [],
            }

        for attempt in range(1, MAX_RETRIES + 1):
            #  копим пул ровно до n_needed .
            pending = {
                name: state for name, state in class_state.items()
                if len(state["accepted_pairs"]) < state["n_needed"]
            }
            if not pending:
                break

            print(f"\n[Этап 3][{pivot_lang}] Попытка {attempt}/{MAX_RETRIES}: "
                  f"{len(pending)} классов ещё набирают кандидатов")

            all_sources: list[str] = []
            source_class: list[str] = []

            for class_name, state in pending.items():
                pool_gap = state["n_needed"] - len(state["accepted_pairs"])
                n_to_generate = max(pool_gap, 1) * OVERSAMPLE_FACTOR
                sources = select_sources(state["bt_sources"], n_to_generate)
                all_sources.extend(sources)
                source_class.extend([class_name] * len(sources))

            print(f"  Всего источников для перевода: {len(all_sources)}")

            all_translated = back_translate(all_sources, model, tokenizer, device, pivot_lang)

            pairs_by_class: dict[str, list[tuple[str, str]]] = {name: [] for name in pending}
            for translated, source, cls_name in zip(all_translated, all_sources, source_class):
                if translated.strip():
                    pairs_by_class[cls_name].append((translated.strip(), source))

            for class_name, state in class_state.items():
                if class_name not in pending:
                    continue

                pairs = pairs_by_class.get(class_name, [])
                if not pairs:
                    continue

                candidates = [p[0] for p in pairs]
                candidate_originals = [p[1] for p in pairs]

                current_existing = state["existing"] + [p[0] for p in state["accepted_pairs"]]
                valid = validate_generated_texts(
                    candidates, current_existing, class_name,
                    similarity_threshold=sim_threshold,
                    sbert_model=sbert_model,
                )
                n_after_validation = len(valid)

                valid_set = set(valid)
                valid_pairs = [
                    (t, o) for t, o in zip(candidates, candidate_originals)
                    if t in valid_set
                ]
                state["accepted_pairs"].extend(valid_pairs)

                print(f"  [{pivot_lang} {attempt}] «{class_name}»: получено {len(candidates)}, "
                      f"после фильтров {n_after_validation}, "
                      f"в пул +{len(valid_pairs)}, всего в пуле {len(state['accepted_pairs'])}")

            cache_rows = []
            for cn, st in class_state.items():
                for translated, original in st["accepted_pairs"]:
                    cache_rows.append({
                        LABEL_COL: cn,
                        "pivot_lang": pivot_lang,
                        "translated": translated,
                        "original": original,
                    })
            if cache_rows:
                pd.DataFrame(cache_rows).to_csv(_PAIRS_CSV, index=False)
                print(f"  [Кэш] {pivot_lang}, попытка {attempt}: сохранено {len(cache_rows)} пар")

        print(f"\n[Этап 3][{pivot_lang}] Перевод завершён, отбираем кандидатов")

        # Этап выборки. С судьёй - vLLM оценивает пары "перевод vs оригинал"
        # и берёт топ-N. Без судьи - просто перемешиваем пул и берём первые N
        # В финальной версии судбя выключен, так как режет практически все
        llm = None
        select_top_paraphrases = None
        if USE_JUDGE_STAGE3:
            print(f"\n[Этап 3][{pivot_lang}] Грузим LLM-судью...")
            # Для судьи нужно временно выгрузить NLLB+SBERT - vLLM съест VRAM
            unload_from_gpu(model, tokenizer, sbert_model)
            model = tokenizer = sbert_model = None
            import src.augmentation.validation as val_module
            val_module._sbert_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from src.augmentation.llm_utils import load_llm, select_top_paraphrases
            llm, _, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

        new_rows = []
        for class_name, state in class_state.items():
            pairs = state["accepted_pairs"]
            current_count = len(df[df[LABEL_COL] == class_name])
            n_needed = TARGET_COUNT - current_count

            if n_needed <= 0:
                continue
            if not pairs:
                print(f"[Этап 3][{pivot_lang}] Класс «{class_name}»: нет кандидатов после перевода")
                continue

            paras = [p[0] for p in pairs]
            origs = [p[1] for p in pairs]

            print(f"\n[Этап 3][{pivot_lang}] Класс «{class_name}»: "
                  f"{len(pairs)} кандидатов в пуле, нужно {n_needed}")

            if USE_JUDGE_STAGE3:
                best = select_top_paraphrases(
                    paras, origs, class_name, llm,
                    n_needed=n_needed, min_score=MIN_JUDGE_SCORE_STAGE3,
                )
            else:
                # Перемешиваем - иначе всегда брали бы пары в порядке генерации
                # и одни оригиналы доминировали бы над другими
                shuffled = list(paras)
                random.shuffle(shuffled)
                best = shuffled[:n_needed]

            for text in best:
                new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

            print(f"[Этап 3][{pivot_lang}] Класс «{class_name}»: отобрано {len(best)} текстов")

            if len(best) < n_needed:
                print(f"  [Внимание] «{class_name}»: удалось отобрать только "
                      f"{len(best)}/{n_needed}")

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"\n[Этап 3][{pivot_lang}] Добавлено {len(new_rows)} текстов")
        else:
            print(f"\n[Этап 3][{pivot_lang}] Новых текстов не добавлено")

        save_checkpoint(df, stage=STAGE)

        if llm is not None:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Если судья работал - он съел VRAM, NLLB+SBERT мы выгружали ДО него.
            # Возвращаем их обратно, чтобы следующий промежуточный язык мог переводить.
            # На USE_JUDGE_STAGE3=False этой ветки нет - модели держатся весь этап
            model, tokenizer, device = load_translation_models()
            sbert_model = load_sbert_on_gpu()

        if _PAIRS_CSV.exists():
            backup = _PAIRS_CSV.with_name(
                f"_stage3_pairs_cache_round{pivot_round}_{pivot_lang}.bak.csv"
            )
            _PAIRS_CSV.rename(backup)
            print(f"[Этап 3][{pivot_lang}] Кэш пар переименован -> {backup.name}")

    # Все промежуточные языки пройдены - выгружаем NLLB+SBERT с GPU
    print(f"\n[Этап 3] Все промежуточные языки пройдены, выгружаем NLLB и SBERT...")
    unload_from_gpu(model, tokenizer, sbert_model)
    model = tokenizer = sbert_model = None
    import src.augmentation.validation as val_module
    val_module._sbert_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[Этап 3] Итоговое распределение:")
    dist = get_class_distribution(df)
    for name, count in dist.items():
        marker = " ✓" if count >= TARGET_COUNT else " ✗"
        print(f"  «{name}»: {count}{marker}")

    print(f"\n[Этап 3] Завершён. Всего записей: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Этап 3: обратный перевод для классов с < 50 примерами"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь до JSON-конфига модели для LLM-судьи (например, configs/model_vllm.json)",
    )
    args = parser.parse_args()

    run(args.config)
