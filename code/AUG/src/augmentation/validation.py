"""
validation.py — Валидация сгенерированных текстов после каждого этапа аугментации

Фильтры:
1. Точные дубликаты — убираем совпадения с существующими и между собой
2. Короткие тексты — скорее всего мусор или обрезки
3. Не русский язык — актуально после обратного перевода
4. Вырожденные тексты — повторы слов, бессмыслица
5. Иностранные символы — тексты с CJK-иероглифами (японский, китайский, корейский)
6. Промпт-утечка — LLM описала задание вместо того чтобы написать письмо
7. Косинусное сходство — слишком похожие на существующие (почти копии)
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException


# --- Настройки ---

SIMILARITY_THRESHOLD = 0.95  # Косинусное сходство выше этого — почти дубликат, отсеиваем
MIN_TEXT_LENGTH = 20         # Короче 20 символов — скорее всего мусор или обрезанное письмо
SBERT_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"  # Русскоязычная SBERT для эмбеддингов
# можно заменить на ai-forever/ru-en-RoSBERTa 
# Кэш модели на уровне модуля — грузим один раз, переиспользуем во всех вызовах
_sbert_model = None


def get_sbert_model() -> SentenceTransformer:
    """
    Возвращает загруженную SBERT-модель, кэшируя её на уровне модуля.
    Возвращает:
        SentenceTransformer — готовая к кодированию модель
    """
    global _sbert_model
    if _sbert_model is None:
        print(f"[Валидация] Загружаю SBERT-модель: {SBERT_MODEL_NAME}")
        # Грузим на CPU — GPU нужен для LLM, а для косинусного сходства CPU хватает
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
        print("[Валидация] SBERT-модель загружена (CPU)")
    return _sbert_model


def validate_generated_texts(
    new_texts: list[str],
    existing_texts: list[str],
    class_name: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_length: int = MIN_TEXT_LENGTH,
    sbert_model: SentenceTransformer | None = None,
) -> list[str]:
    """
    Главная функция — прогоняет новые тексты через все фильтры.


    Аргументы:
        new_texts:             список сгенерированных текстов
        existing_texts:        уже имеющиеся тексты этого класса в датасете
        class_name:            название класса (для логов)
        similarity_threshold:  порог косинусного сходства (по умолчанию 0.95)
        min_length:            минимальная длина текста в символах
        sbert_model:           предзагруженная SBERT-модель. Если None —
                               загрузится автоматически через кэш.

    Возвращает:
        Список текстов, прошедших все проверки
    """
    if not new_texts:
        return []

    total_before = len(new_texts)
    print(f"[Валидация] Класс «{class_name}»: проверяем {total_before} сгенерированных текстов")

    # --- Цепочка фильтров: от дешёвых к дорогим ---

    texts = remove_exact_duplicates(new_texts, existing_texts, class_name)
    texts = filter_short_texts(texts, class_name, min_length=min_length)
    texts = filter_non_russian(texts, class_name)
    texts = filter_degenerate(texts, class_name)
    texts = filter_foreign_scripts(texts, class_name)
    texts = filter_prompt_leak(texts, class_name)

    # Косинусное сходство 
    if texts and existing_texts:
        if sbert_model is None:
            sbert_model = get_sbert_model()
        texts = filter_by_cosine_similarity(
            texts, existing_texts, class_name,
            sbert_model=sbert_model,
            threshold=similarity_threshold,
        )

    total_after = len(texts)
    rejected = total_before - total_after
    print(f"[Валидация] Класс «{class_name}»: "
          f"прошло {total_after}/{total_before}, отсеяно {rejected}")

    return texts


# --- Отдельные фильтры ---


def remove_exact_duplicates(
    new_texts: list[str],
    existing_texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Убирает точные дубликаты: совпадения с существующими текстами и между собой.

    Сравниваем по нормализованному тексту (strip + lower).

    Аргументы:
        new_texts:      сгенерированные тексты
        existing_texts: тексты, уже имеющиеся в датасете для этого класса
        class_name:     название класса (для логов)

    Возвращает:
        Список текстов без дубликатов
    """
    # Множество нормализованных существующих текстов — для быстрого поиска
    existing_normalized = {t.strip().lower() for t in existing_texts}

    unique_texts = []
    seen = set()

    for text in new_texts:
        normalized = text.strip().lower()

        # Проверяем: нет ли такого среди существующих и не встречался ли уже среди новых
        if normalized not in existing_normalized and normalized not in seen:
            unique_texts.append(text)
            seen.add(normalized)

    removed = len(new_texts) - len(unique_texts)
    if removed > 0:
        print(f"  [Дубликаты] Класс «{class_name}»: удалено {removed} точных дубликатов")

    return unique_texts


def filter_short_texts(
    texts: list[str],
    class_name: str,
    min_length: int = MIN_TEXT_LENGTH,
) -> list[str]:
    """
    Убирает слишком короткие тексты.

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)
        min_length: минимальная длина в символах (по умолчанию 20)

    Возвращает:
        Список текстов длиной >= min_length
    """
    filtered = [t for t in texts if len(t.strip()) >= min_length]

    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"  [Длина] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(короче {min_length} символов)")

    return filtered


def filter_non_russian(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Проверяет, что текст написан на русском языке.

    Используем langdetect.

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов, определённых как русскоязычные
    """
    filtered = []

    for text in texts:
        try:
            lang = detect(text)
            if lang == "ru":
                filtered.append(text)
        except LangDetectException:
            # Если langdetect не смог определить язык (слишком короткий текст,
            # спецсимволы и т.д.) — пропускаем, пусть лучше потеряем один текст,
            # чем пропустим мусор
            pass

    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"  [Язык] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(не русский язык)")

    return filtered


def filter_degenerate(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Отсеивает вырожденные тексты: бессмысленные повторы, зацикливания модели.

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов, не являющихся вырожденными
    """
    filtered = []

    for text in texts:
        if _is_degenerate(text):
            continue
        filtered.append(text)

    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"  [Вырожденность] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(повторы, бессмыслица)")

    return filtered


def _is_degenerate(text: str) -> bool:
    """
    Проверяет, является ли текст вырожденным.

    Два критерия:
    1. Слишком мало уникальных слов относительно общего числа — модель зациклилась.
       Для нормального текста доля уникальных слов обычно > 0.3.
    2. Одна и та же фраза (3+ слов) повторяется подряд 3+ раз — типичный
       артефакт генерации, когда модель «залипает».
    """
    words = text.lower().split()

    if len(words) < 3:
        return False

    # Проверка 1: доля уникальных слов
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:
        return True

    # Проверка 2: повторяющиеся подряд фразы (3+ слов повторяются 3+ раз)
    # Ищем паттерн: одна и та же последовательность слов идёт подряд
    text_lower = text.lower()
    if re.search(r"(\b\w+(?:\s+\w+){2,})\s+\1\s+\1", text_lower):
        return True

    return False


# CJK-диапазоны: китайский, японский (хирагана + катакана), корейский
_FOREIGN_SCRIPTS_RE = re.compile(
    r"[\u4e00-\u9fff"    # CJK Unified Ideographs (китайский)
    r"\u3040-\u309f"     # Хирагана (японский)
    r"\u30a0-\u30ff"     # Катакана (японский)
    r"\uac00-\ud7af]"    # Хангыль (корейский)
)


def filter_foreign_scripts(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Отсеивает тексты, содержащие иероглифы CJK (китайский, японский, корейский).

    Иногда qwen генерирует такое

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов без иностранных иероглифов
    """
    filtered = [t for t in texts if not _FOREIGN_SCRIPTS_RE.search(t)]

    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"  [Иноязычные символы] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(содержат иероглифы)")

    return filtered


# Слова-маркеры промпт-утечки — LLM начинает текст с мета-описания вместо письма.
_PROMPT_LEAK_MARKERS = [
    "конечно,",
    "генерирую",
    "вот несколько",
    "вот примеры",
    "вот образцы",
    "несколько примеров",
    "пример письма:",
]


def filter_prompt_leak(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Отсеивает тексты, где LLM описала своё задание вместо письма.

    Проверяем только начало текста (первые 150 символов) — утечка всегда там.

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов без признаков промпт-утечки
    """
    def is_leak(text: str) -> bool:
        start = text.strip()[:150].lower()
        return any(marker in start for marker in _PROMPT_LEAK_MARKERS)

    filtered = [t for t in texts if not is_leak(t)]

    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"  [Промпт-утечка] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(LLM описала задание вместо письма)")

    return filtered


def filter_by_cosine_similarity(
    new_texts: list[str],
    existing_texts: list[str],
    class_name: str,
    sbert_model: SentenceTransformer,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Отсеивает тексты, слишком похожие на уже существующие.


    Аргументы:
        new_texts:      сгенерированные тексты (уже без точных дубликатов)
        existing_texts: тексты этого класса в датасете
        class_name:     название класса (для логов)
        sbert_model:    загруженная SBERT-модель
        threshold:      порог сходства (по умолчанию 0.95)

    Возвращает:
        Список текстов с косинусным сходством ниже порога
    """
    if not new_texts or not existing_texts:
        return new_texts

    # Кодируем все тексты разом — так быстрее, чем по одному
    new_embeddings = sbert_model.encode(new_texts, show_progress_bar=False)
    existing_embeddings = sbert_model.encode(existing_texts, show_progress_bar=False)

    # Матрица сходства: [новые x существующие]
    sim_matrix = cosine_similarity(new_embeddings, existing_embeddings)

    # Для каждого нового текста берём максимальное сходство с любым существующим
    max_similarities = np.max(sim_matrix, axis=1)

    filtered = []
    for i, text in enumerate(new_texts):
        if max_similarities[i] < threshold:
            filtered.append(text)

    removed = len(new_texts) - len(filtered)
    if removed > 0:
        print(f"  [Сходство] Класс «{class_name}»: отсеяно {removed} текстов "
              f"(косинусное сходство > {threshold})")

    return filtered