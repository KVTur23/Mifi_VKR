"""
stage3_back_translation.py — Этап 3: обратный перевод (RU → pivot → RU)

Берём классы с 35–49 примерами и доводим до 50 через обратный перевод.
NLLB-200 переводит RU→pivot→RU, потом валидация фильтрами,
потом выгружаем NLLB и грузим vLLM — LLM-судья оценивает
каждый перевод рядом с оригиналом и отбирает лучшие.

Вход:  Data/data_after_stage2.csv  (или data_after_stage3.csv если чекпоинт есть)
Выход: Data/data_after_stage3.csv

Запуск:
    python src/augmentation/stage3_back_translation.py --config configs/model_vllm.json
"""

import sys
import gc
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
from src.augmentation.text_chunking import chunk_text, join_chunks
from src.augmentation.validation import SIMILARITY_THRESHOLD, validate_generated_texts


# --- Настройки этапа (дефолты, переопределяются через pipeline_config) ---

STAGE = 3
TARGET_COUNT = 50
MAX_RETRIES = 20
MODEL_NLLB = "facebook/nllb-200-1.3B"
BATCH_SIZE = 64
OVERSAMPLE_FACTOR = 3
MIN_JUDGE_SCORE_STAGE3 = 2.5

LANG_RU = "rus_Cyrl"
LANG_EN = "eng_Latn"
LANG_DE = "deu_Latn"
LANG_FR = "fra_Latn"
PIVOT_LANGS = [LANG_EN, LANG_DE, LANG_FR]
PIVOT_ROUNDS = 2
SIMILARITY_THRESHOLD_STEP = 0.10
TRANSLATION_CHUNK_MAX_TOKENS = 900
TRANSLATION_OUTPUT_LENGTH_FACTOR = 1.3
TRANSLATION_OUTPUT_MAX_TOKENS = 1024

# промежуточный csv для пар фазы 1 — при рестарте подхватываем и сразу в фазу 2
_PAIRS_CSV = DATA_DIR / "_stage3_pairs_cache.csv"


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
    input_token_counts: list[int],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str,
) -> list[str]:
    """Переводит батч чанков через NLLB-200 без truncation."""
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        max_length = min(
            int(max(input_token_counts) * TRANSLATION_OUTPUT_LENGTH_FACTOR) + 16,
            TRANSLATION_OUTPUT_MAX_TOKENS,
        )
        max_length = max(max_length, 32)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=max_length,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
            )

        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"  [Перевод] Ошибка при переводе батча: {e}")
        return [""] * len(texts)


def translate_texts_chunked(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str,
) -> list[str]:
    """Переводит полные тексты через tokenizer-aware чанки."""
    if not texts:
        return []

    tokenizer.src_lang = src_lang

    flat_chunks: list[str] = []
    flat_counts: list[int] = []
    owners: list[int] = []
    multi_chunk_docs = 0
    forced_splits = 0

    for doc_idx, text in enumerate(texts):
        chunks = chunk_text(text, tokenizer, TRANSLATION_CHUNK_MAX_TOKENS)
        if len(chunks) > 1:
            multi_chunk_docs += 1
        forced_splits += sum(1 for c in chunks if c.forced_split)
        for chunk in chunks:
            flat_chunks.append(chunk.text)
            flat_counts.append(chunk.token_count)
            owners.append(doc_idx)

    print(
        f"    [Chunking {src_lang}->{tgt_lang}] texts={len(texts)}, "
        f"chunks={len(flat_chunks)}, multi-chunk={multi_chunk_docs}, "
        f"forced={forced_splits}"
    )

    translated_chunks: list[str] = []
    for i in tqdm(
        range(0, len(flat_chunks), BATCH_SIZE),
        desc=f"    {src_lang}→{tgt_lang}",
        leave=False,
    ):
        batch = flat_chunks[i:i + BATCH_SIZE]
        counts = flat_counts[i:i + BATCH_SIZE]
        translated_chunks.extend(
            translate_batch(batch, counts, model, tokenizer, src_lang, tgt_lang, device)
        )

    grouped: list[list[str]] = [[] for _ in texts]
    for owner, translated in zip(owners, translated_chunks):
        grouped[owner].append(translated)

    return [join_chunks(parts) for parts in grouped]


def back_translate(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    pivot_lang: str = LANG_EN,
) -> list[str]:
    """
    Обратный перевод: RU → pivot → RU.

    Плейсхолдеры не маскируются, не восстанавливаются и не фильтруются.
    Длинные письма переводятся через чанки, чтобы не было silent truncation.
    """
    pivot_texts = translate_texts_chunked(
        texts, model, tokenizer, LANG_RU, pivot_lang, device,
    )
    return translate_texts_chunked(
        pivot_texts, model, tokenizer, pivot_lang, LANG_RU, device,
    )


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

    Для каждого pivot-языка отдельно:
    1. NLLB перевод + валидация фильтрами, копим пары оригинал→перевод
    2. Выгружаем NLLB, грузим vLLM — LLM-судья отбирает лучшие
    3. Сохраняем чекпоинт и следующим pivot добираем только оставшиеся классы
    """
    global TARGET_COUNT, MAX_RETRIES, MODEL_NLLB, BATCH_SIZE, OVERSAMPLE_FACTOR, MIN_JUDGE_SCORE_STAGE3
    global TRANSLATION_CHUNK_MAX_TOKENS, TRANSLATION_OUTPUT_LENGTH_FACTOR, TRANSLATION_OUTPUT_MAX_TOKENS

    # Флаг originals_only_sources: если True, BT берётся только из оригинальных
    # писем (stage 0). Stage 2 используется только для дедупликации.
    # Устраняет каскадирование синтетики через стадии.
    originals_only_sources = True

    if pipeline_cfg is not None:
        s = pipeline_cfg.stage3
        TARGET_COUNT = s.target_count
        MAX_RETRIES = s.max_retries
        OVERSAMPLE_FACTOR = s.oversample_factor
        MIN_JUDGE_SCORE_STAGE3 = s.min_judge_score
        MODEL_NLLB = pipeline_cfg.gpu.nllb_model
        BATCH_SIZE = pipeline_cfg.gpu.nllb_batch_size
        chunking_cfg = s.get("translation_chunking", {})
        TRANSLATION_CHUNK_MAX_TOKENS = int(
            chunking_cfg.get("chunk_max_tokens", TRANSLATION_CHUNK_MAX_TOKENS)
        )
        TRANSLATION_OUTPUT_LENGTH_FACTOR = float(
            chunking_cfg.get("output_length_factor", TRANSLATION_OUTPUT_LENGTH_FACTOR)
        )
        TRANSLATION_OUTPUT_MAX_TOKENS = int(
            chunking_cfg.get("output_max_tokens", TRANSLATION_OUTPUT_MAX_TOKENS)
        )
        # _DotDict наследует dict — используем .get() с default
        originals_only_sources = bool(s.get("originals_only_sources", True))

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("ЭТАП 3: Обратный перевод (< 50 → 50)")
    print(f"       источник BT: "
          f"{'ТОЛЬКО ОРИГИНАЛЫ (stage 0)' if originals_only_sources else 'ВСЁ (legacy, каскад)'}")
    print(f"       NLLB: {MODEL_NLLB}, chunk_max_tokens={TRANSLATION_CHUNK_MAX_TOKENS}")
    print("=" * 60)

    # ==========================================================
    # Определяем сценарий запуска:
    # 1) stage3.csv есть, все ≥ 50       → пропуск
    # 2) stage3.csv есть, часть < 50     → доаугментация оставшихся
    # 3) stage3.csv нет                  → полный прогон от stage 2
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

    # Загружаем оригиналы (stage 0) для использования как BT-источников.
    # Когда originals_only_sources=True — BT идёт только от них, что устраняет
    # каскадирование синтетики (stage 1/2 → stage 3).
    df_original = load_dataset(stage=0)

    from src.augmentation.llm_utils import load_llm, select_top_paraphrases

    # Каждый pivot — отдельный цикл: перевод → фильтры → судья → чекпоинт.
    # После первого круга en → de → fr запускаем второй круг с en и повышаем
    # верхний порог косинусного сходства на 0.10, чтобы добрать хвост классов.
    pivot_schedule = [
        (
            round_idx,
            pivot_idx,
            pivot_lang,
            SIMILARITY_THRESHOLD + (round_idx - 1) * SIMILARITY_THRESHOLD_STEP,
        )
        for round_idx in range(1, PIVOT_ROUNDS + 1)
        for pivot_idx, pivot_lang in enumerate(PIVOT_LANGS, start=1)
    ]

    for pivot_round, pivot_idx, pivot_lang, sim_threshold in pivot_schedule:
        classes_to_augment = get_classes_to_augment(df, min_count=0, max_count=TARGET_COUNT)
        if not classes_to_augment:
            print("\n[Этап 3] Все классы уже ≥ 50 — оставшиеся pivot-языки не нужны")
            break

        print(
            f"\n[Этап 3] Круг {pivot_round}/{PIVOT_ROUNDS}, "
            f"pivot {pivot_idx}/{len(PIVOT_LANGS)}: {pivot_lang}, "
            f"cosine_threshold={sim_threshold:.2f}"
        )
        for name, count in sorted(classes_to_augment.items(), key=lambda x: x[1]):
            print(f"  «{name}»: {count} → нужно ещё {TARGET_COUNT - count}")

        class_state = {}
        for class_name, current_count in classes_to_augment.items():
            existing = df[df[LABEL_COL] == class_name][TEXT_COL].tolist()

            if originals_only_sources:
                bt_sources = df_original[
                    df_original[LABEL_COL] == class_name
                ][TEXT_COL].tolist()
                if not bt_sources:
                    print(f"  [Внимание] Нет оригиналов для «{class_name}», "
                          f"fallback на существующие тексты (legacy режим)")
                    bt_sources = existing
            else:
                bt_sources = existing

            class_state[class_name] = {
                "existing": existing,
                "bt_sources": bt_sources,
                "n_needed": TARGET_COUNT - current_count,
                "accepted_pairs": [],
            }

        model, tokenizer, device = load_translation_models()
        sbert_model = load_sbert_on_gpu()

        for attempt in range(1, MAX_RETRIES + 1):
            pending = {
                name: state for name, state in class_state.items()
                if len(state["accepted_pairs"]) < state["n_needed"] * 2
            }
            if not pending:
                break

            print(f"\n[Этап 3][{pivot_lang}] Попытка {attempt}/{MAX_RETRIES}: "
                  f"{len(pending)} классов ещё набирают кандидатов")

            all_sources: list[str] = []
            source_class: list[str] = []

            for class_name, state in pending.items():
                pool_gap = state["n_needed"] * 2 - len(state["accepted_pairs"])
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

        print(f"\n[Этап 3][{pivot_lang}] Перевод завершён, выгружаем NLLB...")
        unload_from_gpu(model, tokenizer, sbert_model)
        model = tokenizer = sbert_model = None
        import src.augmentation.validation as val_module
        val_module._sbert_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[Этап 3][{pivot_lang}] Грузим LLM-судью...")
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

            best = select_top_paraphrases(
                paras, origs, class_name, llm,
                n_needed=n_needed, min_score=MIN_JUDGE_SCORE_STAGE3,
            )

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

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if _PAIRS_CSV.exists():
            backup = _PAIRS_CSV.with_name(
                f"_stage3_pairs_cache_round{pivot_round}_{pivot_lang}.bak.csv"
            )
            _PAIRS_CSV.rename(backup)
            print(f"[Этап 3][{pivot_lang}] Кэш пар переименован → {backup.name}")

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
