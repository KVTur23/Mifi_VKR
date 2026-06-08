"""
data_cleaner.py — Предобработка сырых данных (data.json → data_after_eda.csv)

Читает data.json, чистит тексты (дубликаты, повторы слов, циклы,
обрезка приложений), сохраняет data_after_eda.csv.

Функции очистки вынесены из EDA.ipynb чтобы можно было
воспроизвести предобработку одной командой.
"""

import re
import json
import pandas as pd
from pathlib import Path

from src.utils.data_loader import DATA_DIR, TEXT_COL, LABEL_COL


# --- Исходный файл ---
RAW_FILE = "data.json"


# ===================== Функции очистки текста =====================


def remove_repeated_words(text: str) -> str:
    """Убирает подряд идущие одинаковые слова: 'работа работа работа' → 'работа'."""
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)


def remove_repeated_sequences(text: str, min_words: int = 5, max_words: int = 30) -> str:
    """Убирает длинные повторяющиеся куски (фраза повторяется 50-100 раз подряд)."""
    words = text.split()
    i = 0
    result = []

    while i < len(words):
        found_repeat = False

        for size in range(max_words, min_words - 1, -1):
            if i + size * 2 > len(words):
                continue

            phrase1 = words[i:i + size]
            phrase2 = words[i + size:i + size * 2]

            if phrase1 == phrase2:
                result.extend(phrase1)
                # пропускаем ВСЕ одинаковые повторы подряд
                j = i + size
                while j + size <= len(words) and words[j:j + size] == phrase1:
                    j += size
                i = j
                found_repeat = True
                break

        if not found_repeat:
            result.append(words[i])
            i += 1

    return " ".join(result)


def remove_duplicate_lines(text: str) -> str:
    """Убирает повторяющиеся строки (часто в логах или выгрузках)."""
    seen = set()
    cleaned_lines = []

    for line in text.split("\n"):
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            cleaned_lines.append(line.strip())
            seen.add(normalized)

    return "\n".join(cleaned_lines)


def remove_duplicate_sentences(text: str) -> str:
    """Убирает одинаковые предложения."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    result = []

    for s in sentences:
        s_clean = s.strip()
        key = s_clean.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(s_clean)

    return " ".join(result)


def remove_repeated_comma_phrases(text: str) -> str:
    """Чистит повторы через запятую."""
    pattern = r'(.{20,200}?)(?:,\s*\1){2,}'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)


def remove_cycling_numbered_lines(text: str) -> str:
    """Убирает циклы пронумерованных строк (1. текст, 2. текст, 3. тот же текст...)."""
    lines = text.split('\n')
    contents = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in lines]
    unique = list(dict.fromkeys(contents))

    # если уникальных строк сильно меньше чем всего — был цикл
    if len(unique) < len(contents) / 2:
        return '\n'.join(unique)

    return text


def remove_incremental_list(text: str) -> str:
    """Ищет паттерн: слово-число, слово-число (5+ раз подряд) и сжимает."""
    pattern = r'(\[?[\w]+\]?-\d+)(?:[,\s]+\[?[\w]+\]?-\d+){4,}'

    def shrink(match):
        tokens = re.findall(r'\[?[\w]+\]?-\d+', match.group(0))
        return f'{tokens[0]} ... {tokens[-1]}'

    return re.sub(pattern, shrink, text)


def trim_attached_documents(text: str, max_len: int = 4000) -> str:
    """Обрезает большие письма с таблицами, актами, приложениями."""
    if len(text) <= max_len:
        return text

    triggers = [
        r'[Тт]абл[а-я]*\.?',
        r'Публичная оферта',
        r'заключили настоящий Договор о нижеследующем',
        r'1\.\s*Предмет договора',
        r'Термины и определения',
        r'Акт сверки взаимных расчетов',
        r'Акт\s+№?\s*\d+\s+о\s+нарушении',
        r'Приложение\s+\d+\s+к\s+приказу',
        r'ПОЛОЖЕНИЕ\s+О\s+',
        r'1\.\s*Общие положения',
    ]

    earliest = len(text)
    for trigger in triggers:
        match = re.search(trigger, text)
        if match:
            earliest = min(earliest, match.start())

    if earliest < len(text):
        return text[:earliest].strip()

    return text


def normalize_text(text: str) -> str:
    """Нормализация пробелов."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def clean_text(text: str) -> str:
    """Полная очистка текста — прогоняет через все фильтры по порядку."""
    if not isinstance(text, str):
        return text

    text = remove_repeated_words(text)
    text = remove_repeated_sequences(text)
    text = remove_duplicate_lines(text)
    text = remove_duplicate_sentences(text)
    text = remove_repeated_comma_phrases(text)
    text = remove_cycling_numbered_lines(text)
    text = remove_incremental_list(text)
    text = trim_attached_documents(text)
    text = normalize_text(text)

    return text


# ===================== Удаление дубликатов =====================


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает точные дубликаты и межклассовые дубликаты (логика из EDA.ipynb)."""
    n_before = len(df)

    # 1) точные дубликаты (полное совпадение строк)
    df = df.drop_duplicates()
    n_after_exact = len(df)
    print(f"[Очистка] Удалено точных дубликатов: {n_before - n_after_exact}")

    # 2) межклассовые дубликаты — один текст в разных классах
    #    логика из EDA: удаляем строки из «Блок финансового директора»
    text_dupl = df[df.duplicated([TEXT_COL], keep=False)]
    if len(text_dupl) > 0:
        drop_idx = text_dupl[text_dupl[LABEL_COL] == "Блок финансового директора"].index
        df = df.drop(index=drop_idx)
        print(f"[Очистка] Удалено межклассовых дубликатов "
              f"(«Блок финансового директора»): {len(drop_idx)}")

    print(f"[Очистка] Осталось строк: {len(df)}")
    return df.reset_index(drop=True)


def remove_anomalous_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет аномальные письма: много слов, но почти все повторяются.

    Условие (из EDA): unique_word_count < 50 И word_count > 200.
    Проверяется на сыром тексте (до clean_text).
    """
    def _count_words(text):
        words = re.findall(r'\b\w+\b', text.lower()) if isinstance(text, str) else []
        return len(words), len(set(words))

    stats = df[TEXT_COL].apply(_count_words)
    word_counts = stats.apply(lambda x: x[0])
    unique_counts = stats.apply(lambda x: x[1])

    mask = (unique_counts < 50) & (word_counts > 200)
    n_anomalous = mask.sum()

    if n_anomalous > 0:
        df = df[~mask].reset_index(drop=True)
        print(f"[Очистка] Удалено аномальных писем "
              f"(unique_words < 50 & words > 200): {n_anomalous}")

    return df


# ===================== Основная функция =====================


def run(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Полный цикл очистки: data.json → data_after_eda.csv.

    1. Читает сырой JSON
    2. Удаляет дубликаты
    3. Чистит тексты
    4. Сохраняет результат
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    raw_path = data_dir / RAW_FILE

    print("=" * 60)
    print("ПРЕДОБРАБОТКА: data.json → data_after_eda.csv")
    print("=" * 60)

    # читаем сырые данные
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    print(f"[Очистка] Загружено из {RAW_FILE}: {len(df)} записей")

    # убираем лишние колонки (idx и т.д.)
    keep_cols = [TEXT_COL, LABEL_COL]
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[Очистка] Удалены колонки: {drop_cols}")

    # дубликаты
    df = remove_duplicates(df)

    # удаление аномалий: много слов, но почти все повторяются
    # (проверяется на СЫРОМ тексте, до clean_text)
    df = remove_anomalous_texts(df)

    # очистка текстов
    print(f"[Очистка] Чистим тексты...")
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    # убираем пустые после очистки
    n_before = len(df)
    df = df[df[TEXT_COL].str.strip().astype(bool)].reset_index(drop=True)
    n_empty = n_before - len(df)
    if n_empty > 0:
        print(f"[Очистка] Удалено пустых после очистки: {n_empty}")

    # сохраняем
    out_path = data_dir / "data_after_eda.csv"
    df.to_csv(out_path, index=False)
    print(f"[Очистка] Сохранено: {out_path.name} ({len(df)} записей)")

    # статистика
    print(f"\n[Очистка] Распределение по классам:")
    for cls, count in df[LABEL_COL].value_counts().items():
        print(f"  «{cls}»: {count}")

    return df
