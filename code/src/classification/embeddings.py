"""
embeddings.py - TF-IDF признаки для классификации

Превращает тексты в TF-IDF векторы. Векторизатор обучается на train,
применяется к test.
"""

import sys
from pathlib import Path

import pandas as pd
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


def build_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Создаёт TF-IDF векторизатор с параметрами по умолчанию.

    Аргументы:
        **kwargs: дополнительные параметры для TfidfVectorizer
                  (перезаписывают значения по умолчанию)

    Возвращает:
        TfidfVectorizer - необученный векторизатор
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
        **tfidf_kwargs: доп. параметры для TfidfVectorizer

    Возвращает:
        Кортеж (X_train, y_train, X_test, y_test):
            X_train - разреженная матрица TF-IDF (n_train, n_features)
            y_train - numpy-массив меток
            X_test  - разреженная матрица TF-IDF (n_test, n_features)
            y_test  - numpy-массив меток
    """
    texts_train = df_train[text_col].tolist()
    texts_test = df_test[text_col].tolist()
    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    vectorizer = build_vectorizer(**tfidf_kwargs)

    print(f"[TF-IDF] Обучаю на {len(texts_train)} текстах...")
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    print(f"[TF-IDF] Готово: train {X_train.shape}, test {X_test.shape}")

    return X_train, y_train, X_test, y_test
