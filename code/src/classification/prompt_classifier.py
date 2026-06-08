"""
prompt_classifier.py — Prompt-based классификация писем (Этап 3)

Загружает генеративную LLM через transformers, строит промпт
(zero/one/few-shot), генерирует ответ, извлекает название подразделения.

Модели загружаются по конфигу из pipeline_config.json → prompt_classification.prompt_models.
Работает на A100 40GB.
"""

import gc
import json
import re
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

class _VLLMWrapper:
    """Обёртка над vLLM LLM для единого интерфейса с transformers."""
    def __init__(self, llm, use_external_tokenizer: bool = False):
        self.llm = llm
        self.device = "cuda"
        # True если токенизатор загружен отдельно (skip_tokenizer_init=True в vLLM)
        self.use_external_tokenizer = use_external_tokenizer


def _resolve_hf_path(model_name: str) -> str:
    """Находит локальный snapshot для HF модели в кэше."""
    import os
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.path.expanduser("~/.cache/huggingface/hub"),
    )
    repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        # Берём первый (обычно единственный) snapshot
        dirs = sorted(snapshots.iterdir())
        if dirs:
            return str(dirs[0])
    return model_name


def load_model(model_cfg: dict) -> tuple:
    """
    Загружает модель и токенизатор.
    vLLM-модели (AWQ или use_vllm=true) грузятся через vLLM,
    остальные — через transformers.

    Возвращает:
        (model, tokenizer)
    """
    model_name = model_cfg["model_name"]
    print(f"[PromptClassifier] Загрузка модели: {model_name}")

    use_vllm = model_cfg.get("quantization") == "awq" or model_cfg.get("use_vllm")

    if use_vllm:
        from vllm import LLM
        vllm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "max_model_len": model_cfg.get("max_context", 32768),
        }
        if model_cfg.get("quantization") == "awq":
            vllm_kwargs["quantization"] = "awq"
        if model_cfg.get("dtype"):
            vllm_kwargs["dtype"] = model_cfg["dtype"]
        if model_cfg.get("tensor_parallel_size"):
            vllm_kwargs["tensor_parallel_size"] = model_cfg["tensor_parallel_size"]
        if model_cfg.get("enforce_eager"):
            vllm_kwargs["enforce_eager"] = True
        # Для моделей с кастомным конфигом токенизатора
        # vLLM не может инициализировать токенизатор сам — грузим отдельно
        skip_tok = model_cfg.get("skip_tokenizer_init", False)
        if skip_tok:
            vllm_kwargs["skip_tokenizer_init"] = True
        llm = LLM(**vllm_kwargs)
        model = _VLLMWrapper(llm, use_external_tokenizer=skip_tok)
        if skip_tok:
            from transformers import AutoTokenizer
            # Берём реальный локальный путь (vLLM разрешил HF_HUB_OFFLINE -> snapshot)
            try:
                resolved_path = llm.llm_engine.model_config.model
            except AttributeError:
                resolved_path = vllm_kwargs["model"]
            tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        else:
            tokenizer = llm.get_tokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        load_path = model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )
        except Exception:
            # Fallback для моделей с кастомным конфигом:
            # AutoTokenizer не распознаёт DeepseekConfig -> tokenizer class.
            # Читаем tokenizer_config.json и грузим класс напрямую.
            load_path = _resolve_hf_path(model_name)
            tok_cfg_path = Path(load_path) / "tokenizer_config.json"
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            tok_class_name = tok_cfg.get("tokenizer_class", "LlamaTokenizer")
            print(f"[PromptClassifier] Fallback: загружаю {tok_class_name} из {load_path}")
            import transformers
            tok_class = getattr(transformers, tok_class_name)
            tokenizer = tok_class.from_pretrained(load_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(model_cfg.get("dtype", "float16"), torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
        )
        model.eval()

    print(f"[PromptClassifier] Модель загружена: {model_name}")
    return model, tokenizer


def unload_model(model, tokenizer):
    """Выгружает модель из GPU, освобождает память."""
    if isinstance(model, _VLLMWrapper):
        del model.llm
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import time
    time.sleep(5)
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


def format_class_descriptions(descriptions: dict[str, str], max_chars: int = 200) -> str:
    """Форматирует описания классов, обрезая до max_chars символов каждое."""
    lines = []
    for cls in sorted(descriptions.keys()):
        desc = descriptions[cls]
        if len(desc) > max_chars:
            desc = desc[:max_chars].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {cls}: {desc}")
    return "\n".join(lines)


def format_examples_block(examples: dict[str, list[str]], max_chars: int = 500) -> str:
    """
    Форматирует блок few-shot примеров.
    examples: {class_name: [text1, text2, ...]}
    max_chars: максимальная длина каждого примера
    """
    lines = []
    for cls in sorted(examples.keys()):
        for text in examples[cls]:
            truncated = text[:max_chars] + "..." if len(text) > max_chars else text
            lines.append(f"Письмо: {truncated}\nПодразделение: {cls}")
            lines.append("")
    return "\n".join(lines)


def build_prompt(
    text: str,
    labels: list[str],
    descriptions: dict[str, str],
    mode: str = "zero_shot",
    examples: Optional[dict[str, list[str]]] = None,
    no_desc: bool = False,
    max_example_chars: int = 500,
) -> str:
    """
    Собирает полный промпт для классификации одного письма.

    Аргументы:
        text:             текст письма для классификации
        labels:           список всех 36 классов
        descriptions:     описания классов {class: description}
        mode:             "zero_shot" | "one_shot" | "few_shot"
        examples:         few-shot примеры {class: [texts]} (для one/few-shot)
        no_desc:          не включать описания классов в промпт
        max_example_chars: максимальная длина каждого примера в символах

    Возвращает:
        Готовый промпт-строку
    """
    class_list = format_class_list(labels)

    if mode == "zero_shot":
        template = load_prompt_template("zero_shot")
        return template.format(
            class_list=class_list,
            class_descriptions=format_class_descriptions(descriptions),
            text=text,
        )

    elif mode == "one_shot":
        template = load_prompt_template("one_shot")
        first_cls = sorted(examples.keys())[0]
        ex_text = examples[first_cls][0] if examples[first_cls] else ""
        ex_text_trunc = ex_text[:max_example_chars] + "..." if len(ex_text) > max_example_chars else ex_text
        return template.format(
            class_list=class_list,
            class_descriptions=format_class_descriptions(descriptions),
            example_text=ex_text_trunc,
            example_label=first_cls,
            text=text,
        )

    else:  # few_shot
        examples_block = format_examples_block(examples, max_chars=max_example_chars)
        if no_desc:
            template = load_prompt_template("few_shot_no_desc")
            return template.format(
                class_list=class_list,
                examples_block=examples_block,
                text=text,
            )
        else:
            template = load_prompt_template("few_shot")
            return template.format(
                class_list=class_list,
                class_descriptions=format_class_descriptions(descriptions),
                examples_block=examples_block,
                text=text,
            )


# ============================================================
# Генерация и извлечение предсказания
# ============================================================

def apply_chat_template(tokenizer, prompt: str) -> str:
    """
    Оборачивает промпт в chat template модели.

    Instruction-tuned модели (Saiga, Vikhr, Qwen) ожидают формат с ролями.
    Если у токенизатора нет chat_template — возвращает промпт как есть.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return formatted
    except Exception:
        # Fallback: модель без chat template
        return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    gen_params: dict,
    max_context: int = 8192,
) -> Optional[str]:
    """
    Генерирует ответ модели на промпт.

    Оборачивает промпт в chat template модели перед генерацией.
    Возвращает текст ответа или None если промпт не влезает в контекст.
    """
    formatted_prompt = apply_chat_template(tokenizer, prompt)

    # Проверка длины контекста
    input_len = len(tokenizer.encode(formatted_prompt))
    if input_len >= max_context - 100:
        return None

    # vLLM путь
    if isinstance(model, _VLLMWrapper):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=gen_params.get("max_new_tokens", 50),
            temperature=gen_params.get("temperature", 0.1),
            top_p=gen_params.get("top_p", 0.9),
            repetition_penalty=gen_params.get("repetition_penalty", 1.1),
        )
        if model.use_external_tokenizer:
            # skip_tokenizer_init=True: передаём token_ids и декодируем сами
            token_ids = tokenizer.encode(formatted_prompt)
            prompt = {"prompt_token_ids": token_ids}
            outputs = model.llm.generate([prompt], sampling_params)
            return tokenizer.decode(
                outputs[0].outputs[0].token_ids, skip_special_tokens=True
            ).strip()
        else:
            outputs = model.llm.generate([formatted_prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

    # transformers путь
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_params.get("max_new_tokens", 50),
            temperature=gen_params.get("temperature", 0.1),
            do_sample=gen_params.get("do_sample", True),
            top_p=gen_params.get("top_p", 0.9),
            repetition_penalty=gen_params.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _normalize(text: str) -> str:
    """Нормализация текста: lower, убираем пунктуацию, сжимаем пробелы."""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_response(response: str) -> list[str]:
    """
    Извлекает кандидатов из ответа модели.

    Обрабатывает типичные паттерны:
    - "Подразделение: Блок ..."
    - "Ответ: Блок ..."
    - "1. Блок ..."
    - Многострочные ответы (берём первую строку)
    - Кавычки, точки, лишние пробелы
    """
    candidates = []

    # Берём первую непустую строку (модели часто дают ответ + пояснение)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if not lines:
        return candidates

    for line in lines[:3]:  # Проверяем первые 3 строки
        cleaned = line

        # Убираем типичные префиксы
        prefixes = [
            r'^подразделение\s*:\s*',
            r'^ответ\s*:\s*',
            r'^результат\s*:\s*',
            r'^категория\s*:\s*',
            r'^отдел\s*:\s*',
            r'^\d+[\.\)]\s*',           # "1. " или "1) "
            r'^[-*•]\s*',               # маркеры списка
        ]
        for prefix in prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

        # Убираем кавычки, точки, скобки по краям
        cleaned = cleaned.strip().strip('"«»""\'\'').strip('.,:;!').strip()

        if cleaned:
            candidates.append(cleaned)

    return candidates


def extract_prediction(
    response: str,
    labels: list[str],
    fuzzy_cutoff: float = 0.6,
) -> str:
    """
    Извлекает название подразделения из ответа модели.

    Стратегия:
      1. Точное совпадение (normalized)
      2. Точное совпадение по cleaned кандидатам
      3. Частичное совпадение (label содержится в ответе)
      4. Ответ содержится в label
      5. Нечёткое совпадение (difflib, cutoff=0.6)
      6. "unknown" если ничего не подошло

    Возвращает: название класса или "unknown"
    """
    if not response:
        return "unknown"

    # Подготавливаем нормализованные лейблы
    label_norm = {_normalize(l): l for l in labels}

    # Получаем кандидатов из ответа
    candidates = _clean_response(response)
    if not candidates:
        return "unknown"

    for candidate in candidates:
        cand_norm = _normalize(candidate)

        # 1. Точное совпадение (normalized)
        if cand_norm in label_norm:
            return label_norm[cand_norm]

        # 2. Частичное: нормализованный label содержится в кандидате
        partial_matches = []
        for lnorm, label in label_norm.items():
            if lnorm in cand_norm:
                partial_matches.append(label)
        if partial_matches:
            return max(partial_matches, key=len)

        # 3. Кандидат содержится в label (короткий ответ — часть названия)
        if len(cand_norm) > 5:
            for lnorm, label in label_norm.items():
                if cand_norm in lnorm:
                    return label

    # 4. Нечёткое совпадение по лучшему кандидату
    best_candidate = _normalize(candidates[0])
    matches = difflib.get_close_matches(
        best_candidate,
        label_norm.keys(),
        n=1,
        cutoff=fuzzy_cutoff,
    )
    if matches:
        return label_norm[matches[0]]

    # 5. Fallback: ищем любой label в полном ответе (ненормализованный поиск)
    response_norm = _normalize(response)
    partial_in_full = []
    for lnorm, label in label_norm.items():
        if lnorm in response_norm:
            partial_in_full.append(label)
    if partial_in_full:
        return max(partial_in_full, key=len)

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
    no_desc: bool = False,
    max_example_chars: int = 500,
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

        prompt = build_prompt(text, labels, descriptions, mode, examples,
                              no_desc=no_desc, max_example_chars=max_example_chars)
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
