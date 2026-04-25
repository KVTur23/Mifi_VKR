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

SIMILARITY_THRESHOLD = 0.95  # Верхний порог косинусного сходства (слишком похож → копия)
SIMILARITY_THRESHOLD_LOW = 0.98  # Мягкий порог для классов с 1 оригинальным примером
SIMILARITY_THRESHOLD_MIN = 0.5   # Нижний порог (слишком далёк → текст исказился до неузнаваемости)
MIN_TEXT_LENGTH = 500        # Минимальная длина в символах (медиана оригинала ~1255)
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
    n_original: int | None = None,
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
        n_original:            количество оригинальных примеров класса (до аугментации).
                               Если == 1 — используется мягкий порог 0.98.

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

    threshold = similarity_threshold

    # Косинусное сходство
    if texts and existing_texts:
        if sbert_model is None:
            sbert_model = get_sbert_model()
        texts = filter_by_cosine_similarity(
            texts, existing_texts, class_name,
            sbert_model=sbert_model,
            threshold=threshold,
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


# regex для NER-плейсхолдеров типа [PERSON], [ORGANIZATION], [DATE_TIME], [WORK TYPE]
_LANG_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:\s[A-Z_]+)*\]")


def filter_non_russian(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Проверяет, что текст написан на русском языке.

    Используем langdetect, но сначала вырезаем NER-плейсхолдеры на латинице
    ([PERSON], [ORGANIZATION], ...). В наших данных их 30-60% символов —
    без вырезания langdetect путает русский текст с английским и отсевает
    почти всё (характерно для stage3 после обратного перевода).

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов, определённых как русскоязычные (оригинал, не очищенный)
    """
    filtered = []

    for text in texts:
        # вырезаем плейсхолдеры — для определения языка нужен только живой текст
        text_for_lang = _LANG_PLACEHOLDER_RE.sub(" ", text).strip()
        if not text_for_lang:
            # после очистки пусто — это шум из одних плейсхолдеров, отсекаем
            continue
        try:
            lang = detect(text_for_lang)
            if lang == "ru":
                filtered.append(text)  # сохраняем оригинал с плейсхолдерами
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


# Маркеры промпт-утечки в начале текста (первые 150 символов)
_PROMPT_LEAK_START_MARKERS = [
    "конечно,",
    "конечно!",
    "генерирую",
    "вот несколько",
    "вот примеры",
    "вот образцы",
    "вот еще одно письмо",
    "вот ещё одно письмо",
    "несколько примеров",
    "пример письма:",
    "текущий текст:",
    "как вы можете видеть",
]

# Маркеры промпт-утечки в любом месте текста — фрагменты промпта/системного сообщения
_PROMPT_LEAK_ANYWHERE_MARKERS = [
    # Фрагменты промпта
    "напиши одно письмо",
    "напиши пример письма",
    "напиши одно новое письмо",
    "на основе этих примеров",
    "tokenname:",
    "без пояснений и комментариев",
    "не используй markdown",
    "не добавляй ничего после письма",
    "перепиши это письмо другими словами",
    "переформулированное письмо:",
    "только текст письма:",
    # Мета-обсуждение задания
    "предоставьте пример",
    "предоставь пример",
    "можно ли составить",
    "какие типичные элементы",
    "письмо такого же типа",
    "такого же формата",
    "прошу вас создать",
    "пожалуйста, создайте",
    "создайте ещё одно письмо",
    "создайте еще одно письмо",
    "ваш запрос не совсем",
    "какие параметры можно изменить",
    "используя предоставленный шаблон",
    "я создам",
    "для указанного класса",
    "для класса «",
    'для класса "',
    "эти письма содержат",
    "эти метки нужно",
    # Объяснение токенов
    "[person] - имя",
    "[person] — имя",
    "[organization] - название",
    "[organization] — название",
    "[date_time] - дата",
    "[date_time] — дата",
]

# Regex-паттерны для промпт-утечки
_PROMPT_LEAK_RE = re.compile(
    r"(?:"
    r"\*\*[^*\n]{3,}?\*\*"            # Markdown bold **текст**
    r"|(?:^|\n)###?\s"                  # Markdown заголовки
    r"|(?:^|\n)\s*(?:###?\s+)?(?:[Пп]исьмо)\s+\d+\s*[:\n]"  # Нумерованные письма
    r"|(?:^|\n)user\s*\n"              # Role-маркеры (user\n)
    r"|\[[А-ЯЁ_]{4,}\]"              # Русскоязычные NER-токены [НАЗВАНИЕ_КОМПАНИИ]
    r")",
    re.MULTILINE,
)


def filter_prompt_leak(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """
    Отсеивает тексты с промпт-утечкой.

    Три уровня проверки:
    - начало текста (первые 150 символов) — мета-фразы
    - весь текст — фрагменты промпта или системного сообщения
    - regex — Markdown-разметка, нумерованные письма, русские NER-токены

    Аргументы:
        texts:      список текстов для проверки
        class_name: название класса (для логов)

    Возвращает:
        Список текстов без признаков промпт-утечки
    """
    def is_leak(text: str) -> bool:
        lower = text.strip().lower()
        # Проверка начала
        start = lower[:150]
        if any(m in start for m in _PROMPT_LEAK_START_MARKERS):
            return True
        # Проверка по всему тексту
        if any(m in lower for m in _PROMPT_LEAK_ANYWHERE_MARKERS):
            return True
        # Regex-паттерны
        if _PROMPT_LEAK_RE.search(text):
            return True
        return False

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
    threshold_min: float = SIMILARITY_THRESHOLD_MIN,
) -> list[str]:
    """
    Отсеивает тексты по косинусному сходству с существующими.

    Два порога:
    - Верхний (threshold): > порога → слишком похож, почти копия → отсеиваем
    - Нижний (threshold_min): < порога → слишком далёк, текст исказился → отсеиваем

    Аргументы:
        new_texts:      сгенерированные тексты (уже без точных дубликатов)
        existing_texts: тексты этого класса в датасете
        class_name:     название класса (для логов)
        sbert_model:    загруженная SBERT-модель
        threshold:      верхний порог сходства (по умолчанию 0.95)
        threshold_min:  нижний порог сходства (по умолчанию 0.5)

    Возвращает:
        Список текстов с косинусным сходством в диапазоне [threshold_min, threshold)
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
    removed_too_similar = 0
    removed_too_different = 0
    for i, text in enumerate(new_texts):
        if max_similarities[i] >= threshold:
            removed_too_similar += 1
        elif max_similarities[i] < threshold_min:
            removed_too_different += 1
        else:
            filtered.append(text)

    if removed_too_similar > 0:
        print(f"  [Сходство] Класс «{class_name}»: отсеяно {removed_too_similar} текстов "
              f"(косинусное сходство > {threshold})")
    if removed_too_different > 0:
        print(f"  [Искажение] Класс «{class_name}»: отсеяно {removed_too_different} текстов "
              f"(косинусное сходство < {threshold_min}, текст искажён)")

    return filtered
