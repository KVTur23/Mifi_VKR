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
MAX_NLLB_BATCH_SIZE = 16

LANG_RU = "rus_Cyrl"
LANG_EN = "eng_Latn"
MAX_LENGTH = 512
TRANSLATION_NUM_BEAMS = 2

# regex для NER-плейсхолдеров типа [PERSON], [ORGANIZATION], [DATE_TIME] и т.д.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:\s[A-Z_]*)?\]")

# промежуточный csv для пар фазы 1 — при рестарте подхватываем и сразу в фазу 2
_PAIRS_CSV = DATA_DIR / "_stage3_pairs_cache.csv"


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


# --- LLM-translator (для экспериментов где перевод делает не NLLB, а инструкционная LLM) ---

TRANSLATE_PROMPT_RU_EN = (
    "Translate the following Russian text to English. "
    "Preserve all placeholders like <0>, <1>, <2> exactly as they appear. "
    "Output only the translation, no explanations.\n\n{text}"
)
TRANSLATE_PROMPT_EN_RU = (
    "Переведи следующий английский текст на русский язык. "
    "Сохрани все плейсхолдеры вида <0>, <1>, <2> без изменений. "
    "Выведи только перевод, без пояснений.\n\n{text}"
)


def load_llm_translator(translator_cfg: dict, pipeline_cfg=None) -> tuple:
    """
    Грузит LLM-переводчик (Rosetta-20B и т.п.) через vLLM.
    Отдельная функция — чтобы не смешивать с load_llm из llm_utils, у которой другой контракт.
    """
    from vllm import LLM, SamplingParams

    model_name = translator_cfg["model_name"]
    max_len = translator_cfg.get("max_seq_length", 4096)

    gpu_mem = 0.90
    eager = True
    if pipeline_cfg is not None:
        gpu_mem = pipeline_cfg.gpu.gpu_memory_utilization
        eager = pipeline_cfg.gpu.enforce_eager

    print(f"[Перевод/LLM] Загружаю переводчик через vLLM: {model_name}")

    quantization = translator_cfg.get(
        "quantization", "awq" if "awq" in model_name.lower() else None
    )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=max_len,
        quantization=quantization,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=eager,
    )

    sp = SamplingParams(
        temperature=translator_cfg.get("temperature", 0.3),
        top_p=translator_cfg.get("top_p", 0.9),
        max_tokens=translator_cfg.get("max_new_tokens", 2048),
    )

    print(f"[Перевод/LLM] Переводчик загружен: {model_name}")
    return llm, sp


def unload_vllm(llm) -> None:
    """Выгружает vLLM-модель с GPU — чистит всё, что держит CUDA-память."""
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
    print("[GPU] vLLM-модель выгружена")


def back_translate_llm(texts: list[str], llm, sampling_params) -> list[str]:
    """
    Обратный перевод через инструкционную LLM (вместо NLLB).
    RU → EN → RU, плейсхолдеры маскируются в <N> и восстанавливаются.
    """
    from src.augmentation.llm_utils import generate_batch

    masked_texts = []
    all_placeholders = []
    for text in texts:
        masked, phs = mask_placeholders(text)
        masked_texts.append(masked)
        all_placeholders.append(phs)

    # RU → EN
    prompts_ru_en = [TRANSLATE_PROMPT_RU_EN.format(text=t) for t in masked_texts]
    en_texts = generate_batch(llm, sampling_params, prompts_ru_en)

    # EN → RU
    prompts_en_ru = [TRANSLATE_PROMPT_EN_RU.format(text=t or "") for t in en_texts]
    ru_texts = generate_batch(llm, sampling_params, prompts_en_ru)

    result = []
    for text, phs in zip(ru_texts, all_placeholders):
        result.append(unmask_placeholders((text or "").strip(), phs))

    return result


# --- llama.cpp-translator (для GGUF-квантов типа YanoljaNEXT-Rosetta-20B-GGUF) ---

def load_llamacpp_translator(translator_cfg: dict) -> tuple:
    """
    Грузит GGUF-модель через llama-cpp-python с CUDA-offload.

    Ожидает в translator_cfg одно из двух:
      - model_path: полный путь до .gguf файла (с возможным glob * в snapshot-хеше)
      - ИЛИ model_repo (например "models--mradermacher--YanoljaNEXT-Rosetta-20B-GGUF")
        + model_file ("YanoljaNEXT-Rosetta-20B.Q4_K_M.gguf")
        — тогда файл ищется в $HF_HUB_CACHE/<repo>/snapshots/*/<file>.

    Дополнительно:
      - max_seq_length (опц.): контекст (n_ctx), дефолт 4096
      - n_gpu_layers (опц.): сколько слоёв на GPU, -1 = все (дефолт -1)
    """
    import glob
    import os
    from llama_cpp import Llama

    model_path = translator_cfg.get("model_path")
    if not model_path:
        # собираем из repo+file через HF_HUB_CACHE
        repo = translator_cfg["model_repo"]
        fname = translator_cfg["model_file"]
        cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or ""
        model_path = os.path.join(cache, repo, "snapshots", "*", fname)

    # резолвим glob — llama.cpp ждёт конкретный файл
    if "*" in model_path:
        matches = glob.glob(model_path)
        if not matches:
            raise FileNotFoundError(f"GGUF не найден по шаблону: {model_path}")
        model_path = matches[0]
        print(f"[Перевод/llama.cpp] Найден GGUF: {model_path}")

    n_ctx = translator_cfg.get("max_seq_length", 4096)
    n_gpu_layers = translator_cfg.get("n_gpu_layers", -1)

    print(f"[Перевод/llama.cpp] Загружаю GGUF: {model_path}")
    print(f"[Перевод/llama.cpp] n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    gen_params = {
        "max_tokens": translator_cfg.get("max_new_tokens", 2048),
        "temperature": translator_cfg.get("temperature", 0.3),
        "top_p": translator_cfg.get("top_p", 0.9),
    }
    print(f"[Перевод/llama.cpp] GGUF-модель загружена")
    return llm, gen_params


def unload_llamacpp(llm) -> None:
    """Выгружает GGUF-модель — освобождаем VRAM под судью vLLM."""
    try:
        if hasattr(llm, "close"):
            llm.close()
        elif hasattr(llm, "__del__"):
            llm.__del__()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[GPU] llama.cpp-модель выгружена")


def back_translate_llamacpp(texts: list[str], llm, gen_params: dict) -> list[str]:
    """
    RU→EN→RU через llama-cpp-python. Без батчинга — llama.cpp идёт последовательно.
    Для ~20 классов по 50 переводов на класс это ~1000 промптов × 2 направления = 2000 вызовов,
    заметно медленнее vLLM-пути, но единственный вариант для GGUF-моделей.
    """
    masked_texts = []
    all_placeholders = []
    for text in texts:
        masked, phs = mask_placeholders(text)
        masked_texts.append(masked)
        all_placeholders.append(phs)

    # RU → EN
    en_texts = []
    for t in tqdm(masked_texts, desc="    RU→EN (llama.cpp)", leave=False):
        try:
            out = llm(TRANSLATE_PROMPT_RU_EN.format(text=t), **gen_params)
            en_texts.append(out["choices"][0]["text"].strip())
        except Exception as e:
            print(f"  [llama.cpp] Ошибка RU→EN: {e}")
            en_texts.append("")

    # EN → RU
    ru_texts = []
    for t in tqdm(en_texts, desc="    EN→RU (llama.cpp)", leave=False):
        try:
            out = llm(TRANSLATE_PROMPT_EN_RU.format(text=t or ""), **gen_params)
            ru_texts.append(out["choices"][0]["text"].strip())
        except Exception as e:
            print(f"  [llama.cpp] Ошибка EN→RU: {e}")
            ru_texts.append("")

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


def load_sbert_for_stage3():
    """Грузит SBERT на CPU, чтобы не конкурировать с NLLB за VRAM."""
    from src.augmentation.validation import SBERT_MODEL_NAME
    from sentence_transformers import SentenceTransformer
    import src.augmentation.validation as val_module

    device = "cpu"

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
    if not texts:
        return []
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
                min_length=64,
                do_sample=False,
                num_beams=TRANSLATION_NUM_BEAMS,
                # NLLB на наших анонимизированных текстах с маркерами <0>, <1>
                # и техническими аббревиатурами (НГКМ, п/о, No) сваливается в
                # дегенерацию: возвращает "< < < < <" или "Ministry Ministry Ministry".
                # repetition_penalty + no_repeat_ngram_size штрафуют повторы
                # и блокируют 3-граммы — это убирает зацикленные паттерны.
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )

        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(texts) == 1:
            print(f"  [Перевод/OOM] Не хватило памяти даже на 1 текст: {e}")
            return [""]
        mid = len(texts) // 2
        print(f"  [Перевод/OOM] Батч {len(texts)} не поместился, делю на "
              f"{mid}+{len(texts) - mid}")
        left = translate_batch(texts[:mid], model, tokenizer, src_lang, tgt_lang, device)
        right = translate_batch(texts[mid:], model, tokenizer, src_lang, tgt_lang, device)
        return left + right
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
    batch_size = min(BATCH_SIZE, MAX_NLLB_BATCH_SIZE)
    for i in tqdm(range(0, len(masked_texts), batch_size), desc="    RU→EN", leave=False):
        batch = masked_texts[i:i + batch_size]
        en_texts.extend(translate_batch(batch, model, tokenizer, LANG_RU, LANG_EN, device))

    # EN → RU
    ru_texts = []
    for i in tqdm(range(0, len(en_texts), batch_size), desc="    EN→RU", leave=False):
        batch = en_texts[i:i + batch_size]
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


def run(config_path: str, pipeline_cfg=None) -> None:
    """
    Основная функция этапа 3.

    Две фазы:
    1. NLLB перевод + валидация фильтрами (20 попыток, копим пары оригинал→перевод)
    2. Выгружаем NLLB, грузим vLLM — LLM-судья отбирает лучшие переводы
    """
    global TARGET_COUNT, MAX_RETRIES, MODEL_NLLB, BATCH_SIZE, OVERSAMPLE_FACTOR, MIN_JUDGE_SCORE_STAGE3

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage3
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        MIN_JUDGE_SCORE_STAGE3 = s.min_judge_score
        MODEL_NLLB = pipeline_cfg.gpu.nllb_model
        BATCH_SIZE = min(pipeline_cfg.gpu.nllb_batch_size, MAX_NLLB_BATCH_SIZE)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 3: Обратный перевод (< 50 → 50)")
    print("=" * 60)

    # ==========================================================
    # Определяем сценарий запуска:
    # 1) stage3.csv есть, все ≥ 50       → пропуск
    # 2) stage3.csv есть, часть < 50     → перевод + судья для оставшихся
    # 3) stage3.csv нет, кэш пар есть   → судья для всех
    # 4) stage3.csv нет, кэша нет       → полный прогон
    # ==========================================================

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

    # состояние по классам: что есть, сколько нужно, что накопили
    class_state = {}
    for class_name, current_count in classes_to_augment.items():
        class_state[class_name] = {
            "existing": df[df[LABEL_COL] == class_name][TEXT_COL].tolist(),
            "n_needed": TARGET_COUNT - current_count,
            "accepted_pairs": [],  # (перевод, оригинал) — для судьи потом
        }

    # ==========================================================
    # Проверяем кэш пар фазы 1. Если он полный — можно сразу идти к судье.
    # Если неполный — используем его как стартовое состояние и продолжаем перевод.
    # (только если stage3.csv нет — иначе кэш не актуален)
    # ==========================================================

    run_phase1 = True
    if not has_stage3 and _PAIRS_CSV.exists():
        print(f"\n[Этап 3] Найден кэш пар фазы 1: {_PAIRS_CSV.name}, загружаю...")
        pairs_df = pd.read_csv(_PAIRS_CSV)
        for _, row in pairs_df.iterrows():
            cn = row[LABEL_COL]
            if cn in class_state:
                class_state[cn]["accepted_pairs"].append(
                    (row["translated"], row["original"])
                )
        for cn, st in class_state.items():
            print(f"  «{cn}»: {len(st['accepted_pairs'])} пар в кэше")
        pending_after_cache = {
            cn: st for cn, st in class_state.items()
            if len(st["accepted_pairs"]) < st["n_needed"] * 2
        }
        if pending_after_cache:
            print(f"[Этап 3] Кэш неполный: {len(pending_after_cache)} классов "
                  f"ещё набирают кандидатов — продолжаю фазу 1")
        else:
            print("[Этап 3] Фаза 1 пропущена — кэш пар полный")
            run_phase1 = False

    if run_phase1:
        # ==========================================================
        # Фаза 1: перевод (NLLB или LLM-translator) + валидация фильтрами.
        # Копим ВСЕ пары (перевод, оригинал) что прошли фильтры —
        # судья потом отберёт лучшие по рейтингу.
        # Бэкенд переводчика выбирается по полю translator.backend в json-конфиге
        # модели: "llm" → инструкционная LLM через vLLM (Rosetta и т.п.), иначе NLLB.
        # ==========================================================

        model_cfg_for_tr = load_model_config(config_path)
        translator_cfg = model_cfg_for_tr.get("translator") or {}
        backend = translator_cfg.get("backend")
        use_llm_translator = backend == "llm"
        use_llamacpp_translator = backend == "llama_cpp"

        if use_llm_translator:
            llm_tr, sp_tr = load_llm_translator(translator_cfg, pipeline_cfg)
            translate_fn = lambda srcs: back_translate_llm(srcs, llm_tr, sp_tr)
            model = tokenizer = device = None
        elif use_llamacpp_translator:
            llm_tr, gen_params_tr = load_llamacpp_translator(translator_cfg)
            translate_fn = lambda srcs: back_translate_llamacpp(srcs, llm_tr, gen_params_tr)
            model = tokenizer = device = None
        else:
            model, tokenizer, device = load_translation_models()
            translate_fn = lambda srcs: back_translate(srcs, model, tokenizer, device)
            llm_tr = None

        sbert_model = load_sbert_for_stage3()

        for attempt in range(1, MAX_RETRIES + 1):
            # перевод останавливается когда класс набрал 2x от нужного,
            # но всё что прошло фильтры сверху — тоже в пул, судья разберётся
            pending = {
                name: state for name, state in class_state.items()
                if len(state["accepted_pairs"]) < state["n_needed"] * 2
            }
            if not pending:
                break

            print(f"\n[Этап 3] Попытка {attempt}/{MAX_RETRIES}: "
                  f"{len(pending)} классов ещё набирают кандидатов")

            # собираем все оригиналы в один список — переводим одним прогоном
            all_sources: list[str] = []
            source_class: list[str] = []

            for class_name, state in pending.items():
                pool_gap = state["n_needed"] * 2 - len(state["accepted_pairs"])
                n_to_generate = max(pool_gap, 1) * OVERSAMPLE_FACTOR
                sources = select_sources(state["existing"], n_to_generate)
                all_sources.extend(sources)
                source_class.extend([class_name] * len(sources))

            print(f"  Всего источников для перевода: {len(all_sources)}")

            # один большой RU→EN→RU прогон (NLLB или LLM-translator — зависит от конфига)
            all_translated = translate_fn(all_sources)

            # разбиваем по классам, сохраняя пары (перевод, оригинал)
            pairs_by_class: dict[str, list[tuple[str, str]]] = {name: [] for name in pending}
            for translated, source, cls_name in zip(all_translated, all_sources, source_class):
                if translated.strip():
                    pairs_by_class[cls_name].append((translated.strip(), source))

            # валидируем по классам, сохраняя связь перевод→оригинал
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
                    sbert_model=sbert_model,
                )
                n_after_validation = len(valid)

                # восстанавливаем пары для текстов что прошли фильтры
                valid_set = set(valid)
                valid_pairs = [
                    (t, o) for t, o in zip(candidates, candidate_originals)
                    if t in valid_set
                ]

                # берём всё что прошло фильтры — судья потом отберёт лучшие
                state["accepted_pairs"].extend(valid_pairs)

                print(f"  [{attempt}] «{class_name}»: получено {len(candidates)}, "
                      f"после фильтров {n_after_validation}, "
                      f"в пул +{len(valid_pairs)}, всего в пуле {len(state['accepted_pairs'])}")

            # сохраняем пары в промежуточный csv — при рестарте подхватим
            cache_rows = []
            for cn, st in class_state.items():
                for translated, original in st["accepted_pairs"]:
                    cache_rows.append({
                        LABEL_COL: cn,
                        "translated": translated,
                        "original": original,
                    })
            if cache_rows:
                pd.DataFrame(cache_rows).to_csv(_PAIRS_CSV, index=False)
                print(f"  [Кэш] Попытка {attempt}: сохранено {len(cache_rows)} пар")

        # чистим GPU — переводчик и SBERT больше не нужны
        print("\n[Этап 3] Фаза 1 завершена, выгружаем переводчик...")
        if use_llm_translator:
            unload_vllm(llm_tr)
            llm_tr = None
            unload_from_gpu(sbert_model)
            sbert_model = None
        elif use_llamacpp_translator:
            unload_llamacpp(llm_tr)
            llm_tr = None
            unload_from_gpu(sbert_model)
            sbert_model = None
        else:
            unload_from_gpu(model, tokenizer, sbert_model)
            model = tokenizer = sbert_model = None
        import src.augmentation.validation as val_module
        val_module._sbert_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================
    # Фаза 2: грузим vLLM — судья отбирает лучшие переводы
    # ==========================================================

    print("\n[Этап 3] Фаза 2: грузим LLM-судью...")

    from src.augmentation.llm_utils import load_llm, select_top_paraphrases
    llm, _, _ = load_llm(config_path, pipeline_cfg=pipeline_cfg)

    # судья оценивает каждый перевод рядом с его оригиналом
    new_rows = []

    for class_name, state in class_state.items():
        pairs = state["accepted_pairs"]
        n_needed = state["n_needed"]

        if not pairs:
            print(f"[Этап 3] Класс «{class_name}»: нет кандидатов после перевода")
            continue

        paras = [p[0] for p in pairs]
        origs = [p[1] for p in pairs]

        print(f"\n[Этап 3] Класс «{class_name}»: {len(pairs)} кандидатов в пуле, "
              f"нужно {n_needed}")

        # судья сравнивает каждый перевод с его оригиналом
        # порог ниже чем у парафраза (2.5 vs 5.0) — обратный перевод неизбежно теряет качество
        best = select_top_paraphrases(
            paras, origs, class_name, llm,
            n_needed=n_needed, min_score=MIN_JUDGE_SCORE_STAGE3,
        )

        for text in best:
            new_rows.append({TEXT_COL: text, LABEL_COL: class_name})

        print(f"[Этап 3] Класс «{class_name}»: отобрано {len(best)} текстов")

        if len(best) < n_needed:
            print(f"  [Внимание] «{class_name}»: удалось отобрать только "
                  f"{len(best)}/{n_needed}")

    # финальное сохранение
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"\n[Этап 3] Всего добавлено {len(new_rows)} текстов")
    else:
        print("\n[Этап 3] Новых текстов не сгенерировано")

    save_checkpoint(df, stage=STAGE)

    # кэш пар больше не нужен — переименовываем на всякий случай
    if _PAIRS_CSV.exists():
        backup = _PAIRS_CSV.with_suffix(".bak.csv")
        _PAIRS_CSV.rename(backup)
        print(f"[Этап 3] Кэш пар фазы 1 переименован → {backup.name}")

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
