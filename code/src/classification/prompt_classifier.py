"""
prompt_classifier.py — Prompt-based классификация писем (Этап 3)

Загружает генеративную LLM через transformers, строит промпт
(zero/one/few-shot), генерирует ответ, извлекает название подразделения.

Модели загружаются по конфигу из pipeline_config.json → prompt_classification.prompt_models.
Работает на A100 40GB.
"""

import gc
import json
import difflib
import sys
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import TEXT_COL, LABEL_COL

# --- Пути ---
CONFIG_PATH = PROJECT_ROOT / "config_models" / "pipeline_config.json"
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "classification_prompts"
DATA_DIR = PROJECT_ROOT / "Data"


# ============================================================
# Загрузка конфига
# ============================================================

def load_prompt_config() -> dict:
    """Загружает секцию prompt_classification из pipeline_config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["prompt_classification"]


# ============================================================
# Загрузка / выгрузка модели
# ============================================================

def load_model(model_cfg: dict) -> tuple:
    """
    Загружает модель и токенизатор через transformers.

    Аргументы:
        model_cfg: словарь из stage4.prompt_models (model_name, dtype, ...)

    Возвращает:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_cfg["model_name"]
    print(f"[PromptClassifier] Загрузка модели: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Определяем dtype
    dtype_str = model_cfg.get("dtype", "float16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    # AWQ-модели загружаются без указания dtype
    if model_cfg.get("quantization") == "awq":
        pass  # autoawq определит dtype автоматически
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    print(f"[PromptClassifier] Модель загружена: {model_name}")
    return model, tokenizer


def unload_model(model, tokenizer):
    """Выгружает модель из GPU, освобождает память."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PromptClassifier] Модель выгружена, GPU память освобождена")


# ============================================================
# Построение промптов
# ============================================================

def load_prompt_template(mode: str) -> str:
    """
    Загружает шаблон промпта.

    mode: "zero_shot", "one_shot", "few_shot"
    """
    path = PROMPTS_DIR / f"{mode}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_class_list(labels: list[str]) -> str:
    """Форматирует нумерованный список классов."""
    return "\n".join(f"{i+1}. {label}" for i, label in enumerate(sorted(labels)))


def format_class_descriptions(descriptions: dict[str, str]) -> str:
    """Форматирует описания классов."""
    lines = []
    for cls in sorted(descriptions.keys()):
        lines.append(f"- {cls}: {descriptions[cls]}")
    return "\n".join(lines)


def format_examples_block(examples: dict[str, list[str]]) -> str:
    """
    Форматирует блок few-shot примеров.
    examples: {class_name: [text1, text2, ...]}
    """
    lines = []
    for cls in sorted(examples.keys()):
        for text in examples[cls]:
            # Обрезаем длинные примеры до ~500 символов
            truncated = text[:500] + "..." if len(text) > 500 else text
            lines.append(f"Письмо: {truncated}\nПодразделение: {cls}")
            lines.append("")
    return "\n".join(lines)


def build_prompt(
    text: str,
    labels: list[str],
    descriptions: dict[str, str],
    mode: str = "zero_shot",
    examples: Optional[dict[str, list[str]]] = None,
) -> str:
    """
    Собирает полный промпт для классификации одного письма.

    Аргументы:
        text:         текст письма для классификации
        labels:       список всех 36 классов
        descriptions: описания классов {class: description}
        mode:         "zero_shot" | "one_shot" | "few_shot"
        examples:     few-shot примеры {class: [texts]} (для one/few-shot)

    Возвращает:
        Готовый промпт-строку
    """
    template = load_prompt_template(mode)

    class_list = format_class_list(labels)
    class_desc = format_class_descriptions(descriptions)

    if mode == "zero_shot":
        return template.format(
            class_list=class_list,
            class_descriptions=class_desc,
            text=text,
        )

    elif mode == "one_shot":
        # Берём первый пример из первого класса
        first_cls = sorted(examples.keys())[0]
        ex_text = examples[first_cls][0] if examples[first_cls] else ""
        ex_text_trunc = ex_text[:500] + "..." if len(ex_text) > 500 else ex_text
        return template.format(
            class_list=class_list,
            class_descriptions=class_desc,
            example_text=ex_text_trunc,
            example_label=first_cls,
            text=text,
        )

    else:  # few_shot
        examples_block = format_examples_block(examples)
        return template.format(
            class_list=class_list,
            class_descriptions=class_desc,
            examples_block=examples_block,
            text=text,
        )


# ============================================================
# Генерация и извлечение предсказания
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    gen_params: dict,
    max_context: int = 8192,
) -> Optional[str]:
    """
    Генерирует ответ модели на промпт.

    Возвращает текст ответа или None если промпт не влезает в контекст.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_len = inputs["input_ids"].shape[1]

    # Проверка длины контекста
    if input_len >= max_context - 100:
        return None

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_params.get("max_new_tokens", 100),
            temperature=gen_params.get("temperature", 0.1),
            do_sample=gen_params.get("do_sample", True),
            top_p=gen_params.get("top_p", 0.9),
            repetition_penalty=gen_params.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.pad_token_id,
        )

    # Декодируем только новые токены
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def extract_prediction(
    response: str,
    labels: list[str],
    fuzzy_cutoff: float = 0.7,
) -> str:
    """
    Извлекает название подразделения из ответа модели.

    Стратегия:
      1. Точное совпадение (lower)
      2. Частичное совпадение (label содержится в ответе)
      3. Нечёткое совпадение (difflib, cutoff=0.7)
      4. "unknown" если ничего не подошло

    Возвращает: название класса или "unknown"
    """
    if not response:
        return "unknown"

    response_clean = response.strip().strip('"').strip("'").strip(".")
    response_lower = response_clean.lower()

    # 1. Точное совпадение
    for label in labels:
        if response_lower == label.lower():
            return label

    # 2. Частичное: label содержится в ответе (ищем самый длинный матч)
    partial_matches = []
    for label in labels:
        if label.lower() in response_lower:
            partial_matches.append(label)
    if partial_matches:
        # Берём самое длинное совпадение (более специфичное)
        return max(partial_matches, key=len)

    # 3. Ответ содержится в label (короткий ответ — часть полного названия)
    for label in labels:
        if response_lower in label.lower() and len(response_lower) > 5:
            return label

    # 4. Нечёткое совпадение
    labels_lower = {l.lower(): l for l in labels}
    matches = difflib.get_close_matches(
        response_lower,
        labels_lower.keys(),
        n=1,
        cutoff=fuzzy_cutoff,
    )
    if matches:
        return labels_lower[matches[0]]

    return "unknown"


# ============================================================
# Основной пайплайн классификации
# ============================================================

def classify_dataset(
    df_test: pd.DataFrame,
    model,
    tokenizer,
    labels: list[str],
    descriptions: dict[str, str],
    mode: str,
    gen_params: dict,
    max_context: int,
    examples: Optional[dict[str, list[str]]] = None,
    fuzzy_cutoff: float = 0.7,
) -> pd.DataFrame:
    """
    Классифицирует все примеры из df_test.

    Возвращает DataFrame с колонками:
      text, true_label, predicted_label, raw_response, skipped
    """
    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"Classify ({mode})"):
        text = row[TEXT_COL]
        true_label = row[LABEL_COL]

        prompt = build_prompt(text, labels, descriptions, mode, examples)
        response = generate_response(model, tokenizer, prompt, gen_params, max_context)

        if response is None:
            # Промпт не влез в контекст
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": "unknown",
                "raw_response": None,
                "skipped": True,
            })
        else:
            pred = extract_prediction(response, labels, fuzzy_cutoff)
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": pred,
                "raw_response": response,
                "skipped": False,
            })

    return pd.DataFrame(results)


def load_class_descriptions(path: str | Path | None = None) -> dict[str, str]:
    """Загружает описания классов из JSON."""
    path = Path(path) if path else DATA_DIR / "class_descriptions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
