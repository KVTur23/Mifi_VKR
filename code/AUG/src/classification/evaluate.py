"""
evaluate.py — Оценка классификаторов на тестовой выборке

Загружает train (аугментированный) и test, строит TF-IDF признаки,
обучает модель, выводит метрики на тесте.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, load_test_set, RANDOM_SEED
from src.classification.embeddings import prepare_features

STAGE = 3


def load_data():
    """
    Загружает train/test и возвращает TF-IDF признаки + метки.

    Возвращает:
        (X_train, y_train, X_test, y_test, label_names)
    """
    df_train = load_dataset(stage=STAGE)
    df_test = load_test_set()

    X_train, y_train_raw, X_test, y_test_raw = prepare_features(df_train, df_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    return X_train, y_train, X_test, y_test, le.classes_


def evaluate_model(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names,
    param_grid: dict | None = None,
) -> None:
    """
    Обучает модель на train, оценивает на test.

    Если передан param_grid — подбирает параметры через GridSearchCV на train.
    """
    print("=" * 60)
    print(f"КЛАССИФИКАЦИЯ: {name}")
    print("=" * 60)
    print(f"[{name}] Train: {len(y_train)}, Test: {len(y_test)}, "
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
