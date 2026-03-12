"""
llm_utils.py — Универсальная обёртка для работы с LLM через HuggingFace

Загружает любую  LM модель по JSON-конфигу из configs/, генерирует текст
по промпту. Используется в этапе 1 и 2, но спроектирован так,
чтобы работать с любой transformers-совместимой моделью.

Поддерживает:
- автоматическое распределение по устройствам (device_map="auto")
- chat-шаблоны для Instruct-моделей (Qwen, LLaMA-Chat и т.д.)
- unsloth-модели (4-bit, экономят VRAM — включается флагом use_unsloth в конфиге)

Вход:  путь до JSON-конфига модели
Выход: загруженная модель + токенизатор, готовые к генерации
"""

import sys
from pathlib import Path
try:
    import unsloth  # Unsloth должен быть импортирован до transformers
except ImportError:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Добавляем корень проекта в sys.path, чтобы импортировать утилиты
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_model_config, load_prompt



def load_llm(config_path: str) -> tuple:
    """
    Загружает LLM и токенизатор по JSON-конфигу.

    Читает конфиг, скачивает или забирает из кэша модель с HuggingFace. 

    Аргументы:
        config_path: путь до JSON-конфига (например, 'configs/model_qwen.json')

    Возвращает:
        Кортеж (model, tokenizer, generation_params, prompt_template_path, system_prompt)
        system_prompt — строка для role=system, или None если в конфиге не задан
    """
    config = load_model_config(config_path)
    model_name = config["model_name"]
    generation_params = config["generation_params"]
    prompt_template = config["prompt_template"]
    system_prompt = config.get("system_prompt")  # None если не задан в конфиге
    use_unsloth = config.get("use_unsloth", False)

    print(f"[LLM] Загружаю модель: {model_name}")

    if use_unsloth:
        # Unsloth грузит модель в 4-bit со своей оптимизацией — меньше VRAM, быстрее инференс
        from unsloth import FastLanguageModel

        max_seq_length = config.get("max_seq_length", 2048)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        # Переводим в режим инференса — без этого генерация будет медленной
        FastLanguageModel.for_inference(model)
    else:
        # --- Токенизатор ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # --- Модель ---
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.eval()

    # Заглушка, у некоторых моделей нет pad_token — без него batch-генерация падает
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LLM] Модель загружена, устройство: {model.device}")

    return model, tokenizer, generation_params, prompt_template, system_prompt


def generate_text(
    model,
    tokenizer,
    prompt: str,
    generation_params: dict,
    num_return_sequences: int = 1,
    system_prompt: str | None = None,
) -> list[str]:
    """
    Генерирует текст по промпту с заданными параметрами.

    Принимает уже загруженные модель и токенизатор, форматирует промпт
    под chat-шаблон (если модель поддерживает), прогоняет через модель
    и декодирует результат. Возвращает только сгенерированную часть (без промпта).

    Аргументы:
        model:                загруженная модель (из load_llm)
        tokenizer:            токенизатор (из load_llm)
        prompt:               текст промпта (user-сообщение)
        generation_params:    параметры генерации из конфига (temperature, top_p и т.д.)
        num_return_sequences: сколько вариантов сгенерировать за один вызов
        system_prompt:        системный промпт (role=system) — инструкции модели,
                              чётко отделённые от примеров. Если None — не используется.

    Возвращает:
        Список сгенерированных текстов. При ошибке — пустой список
        (чтобы пайплайн не падал из-за одного неудачного вызова).
    """
    # Instruct-модели ожидают специальный формат с ролями —
    # инструкции идут в system, задача с примерами — в user
    formatted_prompt = _format_prompt(tokenizer, prompt, system_prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # Берём параметры из конфига и добавляем технические
    gen_kwargs = {**generation_params}
    gen_kwargs["num_return_sequences"] = num_return_sequences
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        print(f"[LLM] Ошибка при генерации: {e}")
        # После OOM чистим кэш — иначе следующая попытка тоже упадёт
        torch.cuda.empty_cache()
        return []

    # Декодируем только сгенерированную часть — промпт отрезаем,
    # иначе получим его же в начале каждого ответа
    generated_texts = []
    for output in outputs:
        #  срезает промпт и оставляет только новые токены
        generated_tokens = output[input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if text:
            generated_texts.append(text)

    return generated_texts


def load_prompt_template(template_name: str) -> str:
    """
    Загружает шаблон промпта из папки prompts/.

    Обёртка над config_loader.load_prompt —
    Прописывает абсолютный путь до папки с промптом.

    Аргументы:
        template_name: имя файла промпта (например, 'llm_generate.txt')
                       или полный путь до файла

    Возвращает:
        Текст промпта как строку
    """
    template_path = Path(template_name)

    # Если передали просто имя файла — ищем в prompts/
    if not template_path.is_absolute() and "/" not in template_name:
        template_path = PROJECT_ROOT / "prompts" / template_name

    return load_prompt(str(template_path))


# --- Внутренние функции ---


def _format_prompt(tokenizer, prompt: str, system_prompt: str | None = None) -> str:
    """
    Форматирует промпт под chat-шаблон модели, если он есть.

    Если передан system_prompt — кладём инструкции в role=system,
    а задачу с примерами — в role=user. Так модель чётко видит границу
    между «что надо делать» и «вот данные для работы».
    """
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Если chat_template в модели не предусмотрен, просто используем сырой промпт
            pass

    return prompt


