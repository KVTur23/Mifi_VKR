"""
evaluate.py — Оценка классификаторов на тестовой выборке

Загружает train (аугментированный) и test, обучает модель,
выводит метрики на тесте.

Для представления текстов используется TF-IDF (вместо SBERT-эмбеддингов).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, load_test_set, TEXT_COL, LABEL_COL, RANDOM_SEED

STAGE = 3

# TF-IDF настройки
TFIDF_PARAMS = dict(
    max_features=10_000,
    sublinear_tf=True,
    min_df=2,
    ngram_range=(1, 2),
)


def load_data():
    """
    Загружает train/test и возвращает TF-IDF фичи + метки.

    TfidfVectorizer фитится на train, трансформирует и train и test.
    Возвращает sparse-матрицы — sklearn-классификаторы работают с ними напрямую.

    Возвращает:
        (X_train, y_train, X_test, y_test, label_names)
    """
    df_train = load_dataset(stage=STAGE)
    df_test = load_test_set()

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(df_train[TEXT_COL])
    X_test = tfidf.transform(df_test[TEXT_COL])

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[LABEL_COL])
    y_test = le.transform(df_test[LABEL_COL])

    print(f"[TF-IDF] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Словарь: {len(tfidf.vocabulary_)} токенов")

    return X_train, y_train, X_test, y_test, le.classes_


def prepare_tfidf(df_train, df_test):
    """
    Готовит TF-IDF фичи из двух DataFrame (для baseline / augmented сравнений).

    Возвращает:
        (X_train, y_train, X_test, y_test, label_names, tfidf)
    """
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(df_train[TEXT_COL])
    X_test = tfidf.transform(df_test[TEXT_COL])

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[LABEL_COL])
    y_test = le.transform(df_test[LABEL_COL])

    print(f"[TF-IDF] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Словарь: {len(tfidf.vocabulary_)} токенов")

    return X_train, y_train, X_test, y_test, le.classes_, tfidf


def evaluate_model(
    name: str,
    estimator,
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    label_names,
    param_grid: dict | None = None,
) -> dict:
    """
    Обучает модель на train, оценивает на test.

    Если передан param_grid — подбирает параметры через GridSearchCV на train.
    """
    print("=" * 60)
    print(f"КЛАССИФИКАЦИЯ: {name}")
    print("=" * 60)
    print(f"[{name}] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, "
          f"Классов: {len(label_names)}")

    # Подбор гиперпараметров на train
    if param_grid:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        grid = GridSearchCV(
            estimator, param_grid, cv=cv,
            scoring="f1_macro", n_jobs=1,
        )
        grid.fit(X_train, y_train)
        print(f"[{name}] Лучшие параметры: {grid.best_params_} "
              f"(CV macro F1 = {grid.best_score_:.4f})")
        estimator = grid.best_estimator_
    else:
        estimator.fit(X_train, y_train)

    # Оценка на тесте
    y_pred = estimator.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[{name}] Результаты на тестовой выборке:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_mac:.4f}")
    print(f"  Weighted F1: {f1_w:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

    return {"name": name, "accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_w}
