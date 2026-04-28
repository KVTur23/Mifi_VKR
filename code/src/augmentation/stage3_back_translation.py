"""
stage3_back_translation.py — Этап 3: обратный перевод (RU → EN → RU)

Берём классы с 35–49 примерами и доводим до 50 через обратный перевод.
NLLB-200 переводит RU→EN→RU, потом валидация фильтрами,
потом выгружаем NLLB и грузим vLLM — LLM-судья оценивает
каждый перевод рядом с оригиналом и отбирает лучшие.

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
from src.utils.config_loader import load_model_config
from src.augmentation.validation import validate_generated_texts


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 3
TARGET_COUNT = 50
MAX_RETRIES = 20
MODEL_NLLB = "facebook/nllb-200-3.3B"
BATCH_SIZE = 64
OVERSAMPLE_FACTOR = 3
MIN_JUDGE_SCORE_STAGE3 = 2.5

LANG_RU = "rus_Cyrl"
LANG_EN = "eng_Latn"
MAX_LENGTH = 512

# regex для NER-плейсхолдеров типа [PERSON], [ORGANIZATION], [DATE_TIME] и т.д.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:\s[A-Z_]*)?\]")

# промежуточный csv для пар — при рестарте подхватываем и сразу судьёй
_PAIRS_CSV = DATA_DIR / "_stage3_pairs_cache.csv"
# чекпоинт текстов, прошедших судью — при рестарте пропускаем уже отобранные
_JUDGED_CSV = DATA_DIR / "_stage3_judged_cache.csv"


def mask_placeholders(text: str) -> tuple[str, list[str]]:
    """
    Заменяет NER-плейсхолдеры на короткие маркеры <0>, <1>, ...
    NLLB их не трогает — они короткие и похожи на HTML-теги.
    Возвращает (замаскированный текст, список оригинальных плейсхолдеров).
    """
    placeholders = _PLACEHOLDER_RE.findall(text)
    masked = text
    for i, ph in enumerate(placeholders):
        # заменяем по одному вхождению за раз — на случай одинаковых плейсхолдеров
        masked = masked.replace(ph, f"<{i}>", 1)
    return masked, placeholders


def unmask_placeholders(text: str, placeholders: list[str]) -> str:
    """
    Восстанавливает оригинальные плейсхолдеры из маркеров <0>, <1>, ...
    Если NLLB потеряла маркер — пропускаем, текст останется без него.
    """
    for i, ph in enumerate(placeholders):
        text = text.replace(f"<{i}>", ph, 1)
    return text


# --- TranslateGemma translator backend for stage3 ---

TRANSLATEGEMMA_RU = "ru"
TRANSLATEGEMMA_EN = "en"


def get_translator_temperature_for_attempt(
    translator_cfg: dict,
    attempt: int,
) -> float:
    """Returns translator temperature for a stage3 retry attempt."""
    schedule = translator_cfg.get("temperature_schedule")
    if schedule:
        idx = min(max(attempt - 1, 0), len(schedule) - 1)
        return float(schedule[idx])
    return float(translator_cfg.get("temperature", 0.8))


def make_translator_sampling_params(
    translator_cfg: dict,
    temperature: float | None = None,
):
    """Builds vLLM SamplingParams for TranslateGemma."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=(
            float(translator_cfg.get("temperature", 0.8))
            if temperature is None else float(temperature)
        ),
        top_p=float(translator_cfg.get("top_p", 0.9)),
        max_tokens=int(translator_cfg.get("max_new_tokens", 2048)),
    )


def load_llm_translator(translator_cfg: dict, pipeline_cfg=None) -> tuple:
    """Loads TranslateGemma through vLLM."""
    from vllm import LLM

    model_name = translator_cfg["model_name"]
    max_len = int(translator_cfg.get("max_seq_length", 4096))

    gpu_mem = 0.90
    eager = True
    if pipeline_cfg is not None:
        gpu_mem = pipeline_cfg.gpu.gpu_memory_utilization
        eager = pipeline_cfg.gpu.enforce_eager
    gpu_mem = float(translator_cfg.get("gpu_memory_utilization", gpu_mem))

    print(f"[Перевод/TranslateGemma] Загружаю через vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=eager,
    )

    print(f"[Перевод/TranslateGemma] Модель загружена "
          f"(gpu_memory_utilization={gpu_mem})")
    return llm


def unload_vllm(llm) -> None:
    """Unloads a vLLM model and clears CUDA memory before loading SBERT/judge."""
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[GPU] vLLM-переводчик выгружен")


def _is_translategemma(model_name: str) -> bool:
    return "translategemma" in model_name.lower()


def _extract_vllm_text(outputs) -> list[str | None]:
    result = []
    for output in outputs:
        text = output.outputs[0].text.strip() if output.outputs else None
        result.append(text if text else None)
    return result


def translate_batch_translategemma(
    texts: list[str],
    llm,
    sampling_params,
    source_lang: str,
    target_lang: str,
) -> list[str | None]:
    """
    Translates a batch through google/translategemma-*.

    TranslateGemma needs structured chat-template content with
    source_lang_code/target_lang_code. vLLM's llm.chat() normalizes messages and
    drops those custom fields, so we apply the tokenizer chat template manually
    and pass prompt strings to llm.generate().
    """
    if not texts:
        return []

    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    for text in texts:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    try:
        outputs = llm.generate(prompts, sampling_params)
        return _extract_vllm_text(outputs)
    except Exception as e:
        print(f"[Перевод/TranslateGemma] Батч упал, пробую по одному: {e}")

    result: list[str | None] = []
    for i, prompt in enumerate(prompts, start=1):
        try:
            outputs = llm.generate([prompt], sampling_params)
            result.extend(_extract_vllm_text(outputs))
        except Exception as e:
            if i <= 3:
                print(f"[Перевод/TranslateGemma] Ошибка на тексте {i}: {e}")
            result.append(None)
    return result


def back_translate_translategemma(texts: list[str], llm, sampling_params) -> list[str]:
    """Back-translates through TranslateGemma: RU -> EN -> RU."""
    masked_texts = []
    all_placeholders = []
    for text in texts:
        masked, phs = mask_placeholders(text)
        masked_texts.append(masked)
        all_placeholders.append(phs)

    en_texts = translate_batch_translategemma(
        masked_texts,
        llm,
        sampling_params,
        TRANSLATEGEMMA_RU,
        TRANSLATEGEMMA_EN,
    )
    ru_texts = translate_batch_translategemma(
        [text or "" for text in en_texts],
        llm,
        sampling_params,
        TRANSLATEGEMMA_EN,
        TRANSLATEGEMMA_RU,
    )

    result = []
    for text, phs in zip(ru_texts, all_placeholders):
        result.append(unmask_placeholders((text or "").strip(), phs))
    return result


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
    """Выгружает модели из GPU и чистит память — освобождаем место для vLLM."""
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

    # если уже на CPU — выгружаем
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
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> list[str]:
    """
    Обратный перевод: RU → EN → RU с сохранением NER-плейсхолдеров.

    Перед переводом маскируем [PERSON], [ORGANIZATION] и т.д. в короткие <0>, <1>, ...
    После перевода восстанавливаем обратно.
    """
    # маскируем плейсхолдеры — NLLB их не ломает
    masked_texts = []
    all_placeholders = []
    for text in texts:
        masked, phs = mask_placeholders(text)
        masked_texts.append(masked)
        all_placeholders.append(phs)

    # RU → EN
    en_texts = []
    for i in tqdm(range(0, len(masked_texts), BATCH_SIZE), desc="    RU→EN", leave=False):
        batch = masked_texts[i:i + BATCH_SIZE]
        en_texts.extend(translate_batch(batch, model, tokenizer, LANG_RU, LANG_EN, device))

    # EN → RU
    ru_texts = []
    for i in tqdm(range(0, len(en_texts), BATCH_SIZE), desc="    EN→RU", leave=False):
        batch = en_texts[i:i + BATCH_SIZE]
        ru_texts.extend(translate_batch(batch, model, tokenizer, LANG_EN, LANG_RU, device))

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


def _run_judge_pass(
    class_state: dict,
    config_path: str,
    pipeline_cfg,
    min_score: float,
) -> None:
    """
    Запускает судью на accepted_pairs для классов, которым ещё нужны тексты.
    Отобранные тексты добавляются в judged_texts, accepted_pairs очищается.
    Выгружает vLLM после завершения.
    """
    has_pairs = any(
        s["accepted_pairs"] and len(s["judged_texts"]) < s["n_needed"]
        for s in class_state.values()
    )
    if not has_pairs:
        print("[Судья] Нет пар для оценки — пропускаем")
        for s in class_state.values():
            s["accepted_pairs"] = []
        return

    from src.augmentation.llm_utils import load_llm, select_top_paraphrases
    llm, _, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

    for class_name, state in class_state.items():
        pairs = state["accepted_pairs"]
        still_needed = state["n_needed"] - len(state["judged_texts"])

        if not pairs or still_needed <= 0:
            state["accepted_pairs"] = []
            continue

        paras = [p[0] for p in pairs]
        origs = [p[1] for p in pairs]

        print(f"\n[Этап 3] Класс «{class_name}»: {len(pairs)} кандидатов для судьи, "
              f"нужно ещё {still_needed}")

        best = select_top_paraphrases(
            paras, origs, class_name, llm,
            n_needed=still_needed, min_score=min_score,
        )
        state["judged_texts"].extend(best)
        state["accepted_pairs"] = []

        total_judged = len(state["judged_texts"])
        print(f"[Этап 3] Класс «{class_name}»: судья отобрал {len(best)}, "
              f"итого принято {total_judged}/{state['n_needed']}")
        if total_judged < state["n_needed"]:
            print(f"  [Внимание] «{class_name}»: ещё нужно {state['n_needed'] - total_judged}")

    unload_vllm(llm)


def _save_judged_cache(class_state: dict) -> None:
    """Сохраняет тексты, прошедшие судью, в CSV — чекпоинт для возможного рестарта."""
    rows = []
    for cn, st in class_state.items():
        for text in st["judged_texts"]:
            rows.append({LABEL_COL: cn, TEXT_COL: text})
    if rows:
        pd.DataFrame(rows).to_csv(_JUDGED_CSV, index=False)
        print(f"  [Чекпоинт судьи] Сохранено {len(rows)} текстов → {_JUDGED_CSV.name}")


def run(config_path: str, pipeline_cfg=None) -> None:
    """
    Основная функция этапа 3.

    Судья запускается сразу после каждой попытки валидации — не ждём накопления всего пула.
    При рестарте:
      - _JUDGED_CSV: тексты уже прошедшие судью — восстанавливаем без повторной оценки
      - _PAIRS_CSV:  пары ожидающие судью — запускаем судью немедленно
    """
    global TARGET_COUNT, MAX_RETRIES, MODEL_NLLB, BATCH_SIZE, OVERSAMPLE_FACTOR, MIN_JUDGE_SCORE_STAGE3

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage3
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        MIN_JUDGE_SCORE_STAGE3 = s.min_judge_score
        MODEL_NLLB = pipeline_cfg.gpu.nllb_model
        BATCH_SIZE = pipeline_cfg.gpu.nllb_batch_size

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 3: Обратный перевод (< 50 → 50)")
    print("=" * 60)

    stage3_file = DATA_DIR / STAGE_FILES[STAGE]
    has_stage3 = stage3_file.exists()

    if has_stage3:
        df = pd.read_csv(stage3_file)
        print(f"[Данные] Найден чекпоинт этапа 3: {stage3_file.name} ({len(df)} записей)")
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print("[Этап 3] Все классы уже ≥ 50 — этап пропущен")
            return
        print(f"[Этап 3] Чекпоинт неполный, {len(classes_to_augment)} классов < 50 — доаугментируем")
    else:
        df = load_dataset(stage=STAGE - 1)
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print("[Этап 3] Все классы уже имеют >= 50 примеров, этап пропущен")
            save_checkpoint(df, stage=STAGE)
            return

    print(f"\n[Этап 3] Классов для аугментации: {len(classes_to_augment)}")
    for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
        print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

    class_state = {}
    for class_name, current_count in classes_to_augment.items():
        class_state[class_name] = {
            "existing": df[df[LABEL_COL] == class_name][TEXT_COL].tolist(),
            "n_needed": TARGET_COUNT - current_count,
            "accepted_pairs": [],  # (перевод, оригинал) — кандидаты текущей попытки
            "judged_texts": [],    # тексты, отобранные судьёй
        }

    # ==========================================================
    # Загружаем чекпоинты при рестарте (только если нет stage3.csv)
    # ==========================================================

    if not has_stage3:
        if _JUDGED_CSV.exists():
            print(f"\n[Этап 3] Загружаю чекпоинт судьи: {_JUDGED_CSV.name}...")
            judged_df = pd.read_csv(_JUDGED_CSV)
            for _, row in judged_df.iterrows():
                cn = row[LABEL_COL]
                if cn in class_state:
                    class_state[cn]["judged_texts"].append(row[TEXT_COL])
            for cn, st in class_state.items():
                if st["judged_texts"]:
                    print(f"  «{cn}»: восстановлено {len(st['judged_texts'])} текстов из чекпоинта")

        if _PAIRS_CSV.exists():
            print(f"\n[Этап 3] Найден кэш пар: {_PAIRS_CSV.name} — запускаю судью сразу...")
            pairs_df = pd.read_csv(_PAIRS_CSV)
            loaded_any = False
            for _, row in pairs_df.iterrows():
                cn = row[LABEL_COL]
                if cn in class_state:
                    st = class_state[cn]
                    if len(st["judged_texts"]) < st["n_needed"]:
                        st["accepted_pairs"].append((row["translated"], row["original"]))
                        loaded_any = True
            if loaded_any:
                for cn, st in class_state.items():
                    if st["accepted_pairs"]:
                        print(f"  «{cn}»: {len(st['accepted_pairs'])} пар для немедленной оценки")
                _run_judge_pass(class_state, config_path, pipeline_cfg, MIN_JUDGE_SCORE_STAGE3)
                _save_judged_cache(class_state)
            _PAIRS_CSV.unlink(missing_ok=True)
            print(f"[Этап 3] Кэш пар обработан и удалён")

    # ==========================================================
    # Проверяем, нужна ли фаза перевода
    # ==========================================================

    needs_translation = any(
        len(st["judged_texts"]) < st["n_needed"]
        for st in class_state.values()
    )

    if not needs_translation:
        print("[Этап 3] Все классы набрали нужное количество из кэша — перевод пропущен")
    else:
        # ==========================================================
        # Фаза перевода: перевод + валидация + судья после каждой попытки
        # ==========================================================

        model_cfg_for_translation = load_model_config(config_path)
        translator_cfg = model_cfg_for_translation.get("translator") or {}
        use_translategemma = (
            translator_cfg.get("backend") == "llm"
            and _is_translategemma(translator_cfg.get("model_name", ""))
        )

        if use_translategemma:
            print("[Этап 3] Переводчик: TranslateGemma (vLLM)")
            model = tokenizer = device = sbert_model = None
        else:
            print("[Этап 3] Переводчик: NLLB-200 (HuggingFace)")
            model, tokenizer, device = load_translation_models()
            sbert_model = load_sbert_on_gpu()

        for attempt in range(1, MAX_RETRIES + 1):
            pending = {
                name: state for name, state in class_state.items()
                if len(state["judged_texts"]) < state["n_needed"]
            }
            if not pending:
                break

            print(f"\n[Этап 3] Попытка {attempt}/{MAX_RETRIES}: "
                  f"{len(pending)} классов ещё набирают тексты")

            # Собираем источники для перевода
            all_sources: list[str] = []
            source_class: list[str] = []

            for class_name, state in pending.items():
                still_needed = state["n_needed"] - len(state["judged_texts"])
                n_to_generate = max(still_needed, 1) * OVERSAMPLE_FACTOR
                sources = select_sources(state["existing"], n_to_generate)
                all_sources.extend(sources)
                source_class.extend([class_name] * len(sources))

            print(f"  Всего источников для перевода: {len(all_sources)}")

            # Перевод
            if use_translategemma:
                attempt_temp = get_translator_temperature_for_attempt(translator_cfg, attempt)
                print(f"  [Перевод/TranslateGemma] temperature={attempt_temp}, "
                      f"top_p={translator_cfg.get('top_p', 0.9)}")
                llm_tr = load_llm_translator(translator_cfg, pipeline_cfg)
                sampling_params_tr = make_translator_sampling_params(
                    translator_cfg, temperature=attempt_temp,
                )
                all_translated = back_translate_translategemma(all_sources, llm_tr, sampling_params_tr)
                unload_vllm(llm_tr)
                llm_tr = None
                sbert_model = load_sbert_on_gpu()
            else:
                all_translated = back_translate(all_sources, model, tokenizer, device)

            # Разбиваем по классам и валидируем
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
                # judged_texts уже в existing для дедупликации
                current_existing = state["existing"] + state["judged_texts"]
                valid = validate_generated_texts(
                    candidates, current_existing, class_name,
                    sbert_model=sbert_model,
                )
                n_after_validation = len(valid)
                valid_set = set(valid)
                valid_pairs = [
                    (t, o) for t, o in zip(candidates, candidate_originals)
                    if t in valid_set
                ]
                state["accepted_pairs"].extend(valid_pairs)
                print(f"  [{attempt}] «{class_name}»: получено {len(candidates)}, "
                      f"после фильтров {n_after_validation}, "
                      f"в пул +{len(valid_pairs)} (итого в пуле {len(state['accepted_pairs'])})")

            # Выгружаем SBERT перед судьёй
            if use_translategemma:
                unload_from_gpu(sbert_model)
                sbert_model = None
                import src.augmentation.validation as val_module
                val_module._sbert_model = None
            else:
                # NLLB: выгружаем переводчик и SBERT перед судьёй
                unload_from_gpu(model, tokenizer, sbert_model)
                model = tokenizer = sbert_model = None
                import src.augmentation.validation as val_module
                val_module._sbert_model = None

            # Запускаем судью сразу после валидации этой попытки
            _run_judge_pass(class_state, config_path, pipeline_cfg, MIN_JUDGE_SCORE_STAGE3)
            _save_judged_cache(class_state)

            # NLLB: перезагружаем переводчик если нужны ещё попытки
            if not use_translategemma:
                still_pending = any(
                    len(s["judged_texts"]) < s["n_needed"]
                    for s in class_state.values()
                )
                if still_pending and attempt < MAX_RETRIES:
                    model, tokenizer, device = load_translation_models()
                    sbert_model = load_sbert_on_gpu()

        print("\n[Этап 3] Перевод завершён")
        # NLLB: финальная выгрузка (если ещё загружена)
        if not use_translategemma and model is not None:
            unload_from_gpu(model, tokenizer, sbert_model)
            model = tokenizer = sbert_model = None
            import src.augmentation.validation as val_module
            val_module._sbert_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================
    # Финальное сохранение
    # ==========================================================

    new_rows = []
    for class_name, state in class_state.items():
        texts = state["judged_texts"][:state["n_needed"]]
        for text in texts:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})
        if len(texts) < state["n_needed"]:
            print(f"  [Внимание] «{class_name}»: отобрано только {len(texts)}/{state['n_needed']}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 3] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 3] Новых текстов не сгенерировано")

    save_checkpoint(df, stage=STAGE)

    # Убираем чекпоинты — переименовываем на всякий случай
    for path in [_PAIRS_CSV, _JUDGED_CSV]:
        if path.exists():
            backup = path.with_suffix(".bak.csv")
            path.rename(backup)
            print(f"[Этап 3] {path.name} → {backup.name}")

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
