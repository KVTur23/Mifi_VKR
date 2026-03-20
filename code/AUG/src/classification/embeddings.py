"""
embeddings.py — TF-IDF признаки для классификации

Центральный модуль для превращения текстов в числовые векторы.

Используется TfidfVectorizer из scikit-learn. Векторизатор обучается
на тренировочных данных и применяется к тестовым. Поддерживает кэширование
обученного векторизатора в .pkl файл.
"""

import sys
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import TEXT_COL, LABEL_COL


# --- Настройки ---

TFIDF_PARAMS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

CACHE_DIR = PROJECT_ROOT.parent / "Data" / ".tfidf_cache"


def build_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Создаёт TF-IDF векторизатор с параметрами по умолчанию.

    Аргументы:
        **kwargs: дополнительные параметры для TfidfVectorizer
                  (перезаписывают значения по умолчанию)

    Возвращает:
        TfidfVectorizer — необученный векторизатор
    """
    params = {**TFIDF_PARAMS, **kwargs}
    print(f"[TF-IDF] Параметры: max_features={params['max_features']}, "
          f"ngram_range={params['ngram_range']}")
    return TfidfVectorizer(**params)


def prepare_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    use_cache: bool = True,
    **tfidf_kwargs,
) -> tuple:
    """
    Готовит TF-IDF признаки и метки для классификации из DataFrame.

    Обучает TfidfVectorizer на train, трансформирует train и test.

    Аргументы:
        df_train:      DataFrame с тренировочными данными
        df_test:       DataFrame с тестовыми данными
        text_col:      название колонки с текстом
        label_col:     название колонки с метками
        use_cache:     использовать ли кэширование
        **tfidf_kwargs: доп. параметры для TfidfVectorizer

    Возвращает:
        Кортеж (X_train, y_train, X_test, y_test):
            X_train — разреженная матрица TF-IDF (n_train, n_features)
            y_train — numpy-массив меток
            X_test  — разреженная матрица TF-IDF (n_test, n_features)
            y_test  — numpy-массив меток
    """
    texts_train = df_train[text_col].tolist()
    texts_test = df_test[text_col].tolist()
    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    # --- Попробуем взять из кэша ---
    if use_cache:
        cached = _load_from_cache(texts_train, texts_test)
        if cached is not None:
            X_tr, X_te = cached
            print(f"[TF-IDF] Загружено из кэша: train {X_tr.shape}, test {X_te.shape}")
            return X_tr, y_train, X_te, y_test

    # --- Строим TF-IDF ---
    vectorizer = build_vectorizer(**tfidf_kwargs)

    print(f"[TF-IDF] Обучаю на {len(texts_train)} текстах...")
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    print(f"[TF-IDF] Готово: train {X_train.shape}, test {X_test.shape}")

    # --- Сохраняем в кэш ---
    if use_cache:
        _save_to_cache(texts_train, texts_test, X_train, X_test, vectorizer)

    return X_train, y_train, X_test, y_test


def _get_cache_key(texts_train: list[str], texts_test: list[str]) -> str:
    """
    Генерирует ключ кэша на основе хэша текстов.
    """
    fingerprint = f"train_n={len(texts_train)}|test_n={len(texts_test)}"
    if texts_train:
        fingerprint += f"|first={texts_train[0][:100]}|last={texts_train[-1][:100]}"
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:12]


def _load_from_cache(
    texts_train: list[str], texts_test: list[str]
) -> tuple | None:
    """
    Пытается загрузить TF-IDF матрицы из кэша.
    """
    key = _get_cache_key(texts_train, texts_test)
    train_path = CACHE_DIR / f"tfidf_train_{key}.npz"
    test_path = CACHE_DIR / f"tfidf_test_{key}.npz"

    if not train_path.exists() or not test_path.exists():
        return None

    try:
        X_train = load_npz(train_path)
        X_test = load_npz(test_path)
        if X_train.shape[0] == len(texts_train) and X_test.shape[0] == len(texts_test):
            return X_train, X_test
    except Exception:
        pass

    return None


def _save_to_cache(
    texts_train: list[str],
    texts_test: list[str],
    X_train,
    X_test,
    vectorizer: TfidfVectorizer,
) -> None:
    """
    Сохраняет TF-IDF матрицы и векторизатор в кэш.
    """
    key = _get_cache_key(texts_train, texts_test)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_npz(CACHE_DIR / f"tfidf_train_{key}.npz", X_train)
        save_npz(CACHE_DIR / f"tfidf_test_{key}.npz", X_test)
        with open(CACHE_DIR / f"tfidf_vectorizer_{key}.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"[TF-IDF] Кэш сохранён (ключ: {key})")
    except Exception as e:
        print(f"[TF-IDF] Не удалось сохранить кэш: {e}")
