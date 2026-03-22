"""
llm_utils.py — Универсальная обёртка для работы с LLM через vLLM

Загружает LLM модель по JSON-конфигу из configs/, генерирует текст
по промпту или батчу промптов. Используется в этапе 1 и 2.

vLLM обеспечивает батчевый инференс — все промпты обрабатываются
параллельно на GPU, что в 5-10 раз быстрее последовательной генерации.

Вход:  путь до JSON-конфига модели
Выход: загруженная модель, готовая к генерации
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_model_config, load_prompt

from vllm import LLM, SamplingParams


def load_llm(config_path: str) -> tuple:
    """
    Загружает LLM через vLLM по JSON-конфигу.

    Аргументы:
        config_path: путь до JSON-конфига (например, 'configs/model_qwen.json')

    Возвращает:
        Кортеж (llm, sampling_params, system_prompt)
        llm — объект vLLM для генерации
        sampling_params — параметры семплирования
        system_prompt — строка для role=system, или None если не задан
    """
    config = load_model_config(config_path)
    model_name = config["model_name"]
    gen_params = config["generation_params"]
    system_prompt = config.get("system_prompt")

    print(f"[LLM] Загружаю модель через vLLM: {model_name}")

    # vLLM сам управляет GPU, квантизацией, батчами
    quantization = config.get("quantization", "awq" if "awq" in model_name.lower() else None)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=config.get("max_seq_length", 4096),
        quantization=quantization,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=gen_params.get("temperature", 0.7),
        top_p=gen_params.get("top_p", 0.9),
        top_k=gen_params.get("top_k", 50),
        max_tokens=gen_params.get("max_new_tokens", 512),
        repetition_penalty=gen_params.get("repetition_penalty", 1.15),
    )

    print(f"[LLM] Модель загружена: {model_name}")

    return llm, sampling_params, system_prompt


def generate_text(
    llm: LLM,
    sampling_params: SamplingParams,
    prompt: str,
    system_prompt: str | None = None,
) -> str | None:
    """
    Генерирует текст по одному промпту (обёртка для совместимости).

    Возвращает:
        Сгенерированный текст или None при ошибке
    """
    results = generate_batch(llm, sampling_params, [prompt], system_prompt=system_prompt)
    return results[0] if results else None


def generate_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[str | None]:
    """
    Батчевая генерация — все промпты обрабатываются параллельно на GPU.

    Аргументы:
        llm:             объект vLLM
        sampling_params: параметры семплирования
        prompts:         список промптов (user-сообщения)
        system_prompt:   системный промпт (опционально)

    Возвращает:
        Список сгенерированных текстов (None для неудачных)
    """
    if not prompts:
        return []

    # Формируем chat-сообщения для каждого промпта
    conversations = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    try:
        outputs = llm.chat(conversations, sampling_params)
    except Exception as e:
        print(f"[LLM] Ошибка при батчевой генерации: {e}")
        return [None] * len(prompts)

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip() if output.outputs else None
        results.append(text if text else None)

    return results


JUDGE_PROMPT = "judge_score.txt"

# Параметры для скоринга — низкая temperature для стабильных оценок
_JUDGE_SAMPLING = dict(temperature=0.1, top_p=0.9, max_tokens=8)


def score_texts_batch(
    texts: list[str],
    class_name: str,
    llm: LLM,
    sampling_params: SamplingParams,
    system_prompt: str | None = None,
) -> list[tuple[str, float]]:
    """
    LLM-as-a-judge: оценивает каждый текст по шкале 1-10, возвращает
    список (текст, оценка) отсортированный по убыванию оценки.

    Аргументы:
        texts:           список текстов для оценки
        class_name:      название класса
        llm:             объект vLLM
        sampling_params: не используется напрямую — для скоринга свои параметры
        system_prompt:   системный промпт (опционально)

    Возвращает:
        Список кортежей (текст, оценка) от лучшего к худшему
    """
    if not texts:
        return []

    judge_template = load_prompt_template(JUDGE_PROMPT)
    judge_sp = SamplingParams(**_JUDGE_SAMPLING)

    prompts = [
        judge_template.format(text=text, class_name=class_name)
        for text in texts
    ]

    raw_scores = generate_batch(llm, judge_sp, prompts, system_prompt=system_prompt)

    scored = []
    for text, raw in zip(texts, raw_scores):
        score = _parse_score(raw)
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


MIN_JUDGE_SCORE = 5.0  # Тексты с оценкой ниже этого порога отсеиваются всегда


def select_top_half(
    texts: list[str],
    class_name: str,
    llm: LLM,
    sampling_params: SamplingParams,
    n_needed: int,
    system_prompt: str | None = None,
) -> list[str]:
    """
    Оценивает тексты через LLM и возвращает лучшие.

    Работает всегда, даже если текстов <= n_needed:
    - Отсеивает тексты с оценкой < MIN_JUDGE_SCORE (слабые)
    - Если текстов больше чем нужно — берёт топ n_needed по оценке

    Аргументы:
        texts:      кандидаты после валидации
        class_name: название класса
        llm:        объект vLLM
        sampling_params: параметры семплирования (для generate_batch)
        n_needed:   сколько текстов нужно в итоге
        system_prompt: системный промпт

    Возвращает:
        Список лучших текстов
    """
    if not texts:
        return []

    print(f"  [Судья] Оцениваю {len(texts)} текстов для «{class_name}»...")

    scored = score_texts_batch(texts, class_name, llm, sampling_params, system_prompt)

    # Отсеиваем слабые тексты (оценка < порога)
    good = [(text, score) for text, score in scored if score >= MIN_JUDGE_SCORE]
    removed = len(scored) - len(good)

    # Берём не больше n_needed
    selected = [text for text, score in good[:n_needed]]

    avg_score = sum(s for _, s in scored) / len(scored)
    avg_selected = sum(s for _, s in good[:n_needed]) / max(len(selected), 1)
    print(f"  [Судья] Средняя оценка: {avg_score:.1f}, "
          f"отсеяно < {MIN_JUDGE_SCORE}: {removed}, "
          f"отобрано {len(selected)}, средняя отобранных: {avg_selected:.1f}")

    return selected


def _parse_score(raw: str | None) -> float:
    """Извлекает числовую оценку 1-10 из ответа LLM."""
    if not raw:
        return 5.0
    match = re.search(r"\b(10|[1-9])\b", raw.strip())
    if match:
        return float(match.group(1))
    return 5.0


def load_prompt_template(template_name: str) -> str:
    """
    Загружает шаблон промпта из папки prompts/.

    Аргументы:
        template_name: имя файла промпта (например, 'llm_generate_one.txt')

    Возвращает:
        Текст промпта как строку
    """
    template_path = Path(template_name)

    if not template_path.is_absolute() and "/" not in template_name:
        template_path = PROJECT_ROOT / "prompts" / template_name

    return load_prompt(str(template_path))
