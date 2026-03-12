"""
embeddings.py — Получение BERT-эмбеддингов для классификации

Центральный модуль для превращения текстов в числовые векторы. Используется
всеми baseline-моделями (SVM, LogReg, NaiveBayes) — они не умеют работать
с текстом напрямую, им нужны эмбеддинги.

Модель: ai-forever/sbert_large_nlu_ru — русскоязычная sentence-BERT,
обученная на задачах семантического сходства. Даёт вектор размерности 1024
для каждого текста.

Поддерживает кэширование в .npy файл — чтобы не пересчитывать эмбеддинги
при каждом запуске классификатора.
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import TEXT_COL, LABEL_COL


# --- Настройки ---

SBERT_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
DEFAULT_BATCH_SIZE = 32    # Батч для encode — 32 хорошо ложится и на CPU, и на GPU
CACHE_DIR = PROJECT_ROOT.parent / "Data" / ".embeddings_cache"


def load_embedding_model(model_name: str = SBERT_MODEL_NAME) -> SentenceTransformer:
    """
    Загружает SBERT-модель для получения эмбеддингов.

    Та же модель, что используется в валидации (validation.py),
    но здесь — для классификации. Если модель уже скачана —
    подхватится из кэша HuggingFace.

    Аргументы:
        model_name: имя модели на HuggingFace (по умолчанию ai-forever/sbert_large_nlu_ru)

    Возвращает:
        SentenceTransformer — готовая к кодированию модель
    """
    print(f"[Эмбеддинги] Загружаю модель: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[Эмбеддинги] Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def get_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Получает эмбеддинги для списка текстов.

    Кодирует тексты батчами через SentenceTransformer. Прогресс-бар
    через tqdm — на больших датасетах (~2000 текстов) это занимает
    пару минут, приятно видеть, что процесс идёт.

    Аргументы:
        texts:         список текстов для кодирования
        model:         загруженная SBERT-модель
        batch_size:    размер батча (по умолчанию 32)
        show_progress: показывать ли прогресс-бар

    Возвращает:
        numpy-массив формы (len(texts), embedding_dim)
    """
    print(f"[Эмбеддинги] Кодирую {len(texts)} текстов (батч={batch_size})")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    print(f"[Эмбеддинги] Готово, форма: {embeddings.shape}")
    return embeddings


def prepare_features(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Готовит фичи и метки для классификации из DataFrame.

    Обёртка над get_embeddings: берёт DataFrame, вытаскивает тексты,
    получает эмбеддинги, возвращает X и y. Поддерживает кэширование —
    если эмбеддинги для этого набора текстов уже считались, загружает
    из файла вместо пересчёта.

    Аргументы:
        df:         DataFrame с текстами и метками
        model:      загруженная SBERT-модель
        text_col:   название колонки с текстом
        label_col:  название колонки с метками
        batch_size: размер батча для кодирования
        use_cache:  использовать ли кэширование (по умолчанию True)

    Возвращает:
        Кортеж (X, y):
            X — numpy-массив эмбеддингов, форма (n_samples, embedding_dim)
            y — numpy-массив меток (строковых)
    """
    texts = df[text_col].tolist()
    y = df[label_col].values

    # --- Попробуем взять из кэша ---
    if use_cache:
        cached = _load_from_cache(texts)
        if cached is not None:
            print(f"[Эмбеддинги] Загружено из кэша ({cached.shape})")
            return cached, y

    # --- Считаем эмбеддинги ---
    X = get_embeddings(texts, model, batch_size=batch_size)

    # --- Сохраняем в кэш ---
    if use_cache:
        _save_to_cache(texts, X)

    return X, y


def _get_cache_path(texts: list[str]) -> Path:
    """
    Генерирует путь к файлу кэша на основе хэша текстов.

    Хэшируем содержимое и количество текстов — если датасет поменялся
    (добавились записи, изменился порядок), кэш инвалидируется автоматически.
    """
    # Хэшируем первые/последние тексты + общее количество,
    # чтобы не считать хэш от гигабайтов текста
    fingerprint = f"n={len(texts)}"
    if texts:
        fingerprint += f"|first={texts[0][:100]}|last={texts[-1][:100]}"
    cache_hash = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"embeddings_{cache_hash}.npy"


def _load_from_cache(texts: list[str]) -> np.ndarray | None:
    """
    Пытается загрузить эмбеддинги из кэша.

    Возвращает массив, если кэш найден и размер совпадает, иначе None.
    """
    cache_path = _get_cache_path(texts)
    if not cache_path.exists():
        return None

    try:
        embeddings = np.load(cache_path)
        # Проверяем, что размер совпадает — если нет, кэш протух
        if embeddings.shape[0] == len(texts):
            return embeddings
    except Exception:
        pass

    return None


def _save_to_cache(texts: list[str], embeddings: np.ndarray) -> None:
    """
    Сохраняет эмбеддинги в кэш-файл.

    Создаёт директорию кэша, если её нет. Ошибки при сохранении
    не критичны — просто в следующий раз пересчитаем.
    """
    cache_path = _get_cache_path(texts)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"[Эмбеддинги] Кэш сохранён: {cache_path.name}")
    except Exception as e:
        print(f"[Эмбеддинги] Не удалось сохранить кэш: {e}")
