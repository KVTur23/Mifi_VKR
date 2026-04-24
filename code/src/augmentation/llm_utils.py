"""
llm_utils.py — обёртка для работы с LLM через vLLM

Загружает модель по конфигу, генерирует тексты батчами,
оценивает качество через LLM-as-a-judge.

vLLM гоняет все промпты параллельно на GPU — в 5-10 раз быстрее
чем по одному через HuggingFace.
"""

import re
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_model_config, load_prompt

from vllm import LLM, SamplingParams


def load_llm(config_path: str, pipeline_cfg=None) -> tuple:
    """
    Грузит LLM через vLLM по JSON-конфигу.
    pipeline_cfg — конфиг из load_pipeline_config(), если None — дефолты.
    Возвращает (llm, sampling_params, system_prompt).
    """
    config = load_model_config(config_path)
    model_name = config["model_name"]
    gen_params = config["generation_params"]
    system_prompt = config.get("system_prompt")

    # GPU-настройки из pipeline_config или дефолты
    gpu_mem = 0.90
    eager = True
    if pipeline_cfg is not None:
        gpu_mem = pipeline_cfg.gpu.gpu_memory_utilization
        eager = pipeline_cfg.gpu.enforce_eager

    print(f"[LLM] Загружаю модель через vLLM: {model_name}")

    # если в названии модели есть "awq" — vllm автоматом распаковывает квантизованные веса
    quantization = config.get("quantization", "awq" if "awq" in model_name.lower() else None)

    # AWQ в vLLM поддерживает только float16. bfloat16 (как в config.json у некоторых AWQ-моделей,
    # например RuadaptQwen3-32B-Instruct-AWQ) валит загрузку. Явно форсим fp16 для AWQ.
    dtype = config.get("dtype", "float16" if quantization == "awq" else "auto")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=config.get("max_seq_length", 4096),
        quantization=quantization,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=eager,
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
    """Генерирует текст по одному промпту. Обёртка над generate_batch."""
    results = generate_batch(llm, sampling_params, [prompt], system_prompt=system_prompt)
    return results[0] if results else None


def generate_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[str | None]:
    """
    Батчевая генерация — все промпты летят на GPU параллельно.
    Возвращает список текстов, None для тех что не получились.
    """
    if not prompts:
        return []

    # собираем chat-сообщения: system (если есть) + user
    conversations = []
    for prompt in prompts:
        sys_content = system_prompt or ""
        if "/no_think" not in sys_content:
            sys_content = (sys_content + "\n/no_think").strip()
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ]
        conversations.append(messages)

    try:
        outputs = llm.chat(conversations, sampling_params)
    except Exception as e:
        # если весь батч упал (например один промпт слишком длинный) —
        # пробуем по одному, что бы не терять весь батч из-за одного
        print(f"[LLM] Батч упал, пробую по одному: {e}")
        results = []
        for conv in conversations:
            try:
                out = llm.chat([conv], sampling_params)
                text = out[0].outputs[0].text.strip() if out[0].outputs else None
                results.append(text if text else None)
            except Exception:
                results.append(None)
        return results

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip() if output.outputs else None
        results.append(text if text else None)

    return results


# --- LLM-as-a-judge: оценка качества сгенерированных текстов ---

JUDGE_PROMPT = "judge_score.txt"
JUDGE_PARAPHRASE_PROMPT = "judge_paraphrase.txt"
MIN_JUDGE_SCORE = 5.0  # ниже этого — мусор, отсеиваем всегда

# для судьи нужна низкая temperature — что бы оценки были стабильные
_JUDGE_SAMPLING = dict(temperature=0.1, top_p=0.9, max_tokens=32)


MAX_JUDGE_EXAMPLES = 5  # сколько оригиналов показываем судье для сравнения


def score_texts_batch(
    texts: list[str],
    class_name: str,
    llm: LLM,
    existing_texts: list[str] | None = None,
    context: str = "",
) -> list[tuple[str, float]]:
    """
    Оценивает каждый текст по шкале 1-10.
    Возвращает список (текст, оценка), отсортированный от лучшего к худшему.

    existing_texts — примеры настоящих писем класса, судья сравнивает с ними.
    context — описание класса (что за тематика), помогает судье понять контекст.
    """
    if not texts:
        return []

    judge_template = load_prompt_template(JUDGE_PROMPT)
    judge_sp = SamplingParams(**_JUDGE_SAMPLING)

    # берём 5 случайных оригиналов для сравнения
    examples_text = ""
    if existing_texts:
        samples = random.sample(existing_texts, min(MAX_JUDGE_EXAMPLES, len(existing_texts)))
        examples_text = "\n---\n".join(samples)

    # описание класса — если нет, подставляем заглушку
    context_text = context if context else f"Входящие письма класса «{class_name}»"

    # собираем промпты для оценки каждого текста
    prompts = [
        judge_template.format(
            text=text, class_name=class_name,
            examples=examples_text, context=context_text,
        )
        for text in texts
    ]

    # батчем на GPU — ответ всего 1-2 токена, очень быстро
    raw_scores = generate_batch(llm, judge_sp, prompts, system_prompt="/no_think")

    scored = []
    for text, raw in zip(texts, raw_scores):
        score = _parse_score(raw)
        scored.append((text, score))

    # сортируем — лучшие наверх
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_top_half(
    texts: list[str],
    class_name: str,
    llm: LLM,
    n_needed: int,
    existing_texts: list[str] | None = None,
    context: str = "",
) -> list[str]:
    """
    Прогоняет тексты через LLM-судью и возвращает лучшие.

    existing_texts — оригиналы класса, судья сравнивает с ними.
    context — описание класса (генерируется на этапе 1).

    Работает всегда, даже если текстов мало:
    - тексты с оценкой < MIN_JUDGE_SCORE выкидываются в любом случае
    - если текстов больше чем нужно — берём топ по оценке
    - если мало — что прошло порог, то и берём, нехватку добьёт следующий раунд
    """
    if not texts:
        return []

    print(f"  [Судья] Оцениваю {len(texts)} текстов для «{class_name}»...")

    scored = score_texts_batch(texts, class_name, llm,
                               existing_texts=existing_texts, context=context)

    # выкидываем откровенно слабые
    good = [(text, score) for text, score in scored if score >= MIN_JUDGE_SCORE]
    removed = len(scored) - len(good)

    # берём не больше чем нужно
    selected = [text for text, score in good[:n_needed]]

    avg_score = sum(s for _, s in scored) / len(scored)
    avg_selected = sum(s for _, s in good[:n_needed]) / max(len(selected), 1)
    print(f"  [Судья] Средняя оценка: {avg_score:.1f}, "
          f"отсеяно < {MIN_JUDGE_SCORE}: {removed}, "
          f"отобрано {len(selected)}, средняя отобранных: {avg_selected:.1f}")

    return selected


def select_top_paraphrases(
    paraphrases: list[str],
    originals: list[str],
    class_name: str,
    llm: LLM,
    n_needed: int,
    min_score: float | None = None,
) -> list[str]:
    """
    Судья для парафразов — сравнивает каждый парафраз с его конкретным оригиналом.

    paraphrases и originals — параллельные списки одинаковой длины:
    paraphrases[i] — парафраз текста originals[i].

    Оценивает сохранение смысла, переформулировку, естественность.
    Выкидывает слабые (< min_score), берёт лучшие до n_needed.
    min_score: порог отсечения (по умолчанию MIN_JUDGE_SCORE=5.0).
    """
    if not paraphrases:
        return []

    print(f"  [Судья] Оцениваю {len(paraphrases)} парафразов для «{class_name}»...")

    judge_template = load_prompt_template(JUDGE_PARAPHRASE_PROMPT)
    judge_sp = SamplingParams(**_JUDGE_SAMPLING)

    # каждый парафраз оцениваем рядом с его оригиналом
    prompts = [
        judge_template.format(
            text=para, original_text=orig, class_name=class_name,
        )
        for para, orig in zip(paraphrases, originals)
    ]

    raw_scores = generate_batch(llm, judge_sp, prompts, system_prompt="/no_think")

    scored = []
    for para, raw in zip(paraphrases, raw_scores):
        score = _parse_score(raw)
        scored.append((para, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # выкидываем слабые
    threshold = min_score if min_score is not None else MIN_JUDGE_SCORE
    good = [(text, score) for text, score in scored if score >= threshold]
    removed = len(scored) - len(good)

    selected = [text for text, score in good[:n_needed]]

    avg_score = sum(s for _, s in scored) / len(scored)
    avg_selected = sum(s for _, s in good[:n_needed]) / max(len(selected), 1)
    print(f"  [Судья] Средняя оценка: {avg_score:.1f}, "
          f"отсеяно < {threshold}: {removed}, "
          f"отобрано {len(selected)}, средняя отобранных: {avg_selected:.1f}")

    return selected


def _parse_score(raw: str | None) -> float:
    """Вытаскивает число 1-10 из ответа LLM. Если не распарсилось — ставит 0."""
    if not raw:
        return 0
    match = re.search(r"\b(10|[1-9])\b", raw.strip())
    if match:
        return float(match.group(1))
    return 0


def load_prompt_template(template_name: str) -> str:
    """Грузит шаблон промпта из папки prompts/."""
    template_path = Path(template_name)

    if not template_path.is_absolute() and "/" not in template_name:
        template_path = PROJECT_ROOT / "prompts" / "aug_prompts" / template_name

    return load_prompt(str(template_path))
