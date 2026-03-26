"""
evaluate.py — Оценка классификаторов на тестовой выборке

Загружает train (аугментированный) и test, строит TF-IDF признаки,
обучает модель, выводит метрики на тесте.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
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

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n[{name}] Результаты на тестовой выборке:")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {f1_mac:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

    return {"name": name, "balanced_accuracy": bal_acc, "macro_f1": f1_mac}


# ============================================================
# Этап 4: Оценка prompt-based классификации
# ============================================================

def evaluate_prompt_classification(
    df_results: "pd.DataFrame",
    groups: dict[str, str] | None = None,
) -> dict:
    """
    Оценивает результаты prompt-based классификации.

    Аргументы:
        df_results: DataFrame с колонками true_label, predicted_label, skipped
        groups:     словарь {class_name: "A"|"B"|"C"} для per-group метрик

    Возвращает dict:
        balanced_accuracy, macro_f1, unknown_rate,
        f1_group_A, f1_group_B, f1_group_C, n_test, n_skipped,
        classification_report (str)
    """
    import pandas as pd

    # Фильтруем skipped (промпт не влез в контекст)
    n_total = len(df_results)
    n_skipped = int(df_results["skipped"].sum())

    # Работаем только с не-skipped
    df = df_results[~df_results["skipped"]].copy()

    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    # Все уникальные лейблы (из true, чтобы не потерять классы)
    all_labels = sorted(set(y_true))

    # Основные метрики
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)

    # Unknown rate
    n_unknown = sum(1 for p in y_pred if p == "unknown")
    unknown_rate = n_unknown / len(y_pred) if y_pred else 0.0

    report = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0,
    )

    result = {
        "balanced_accuracy": bal_acc,
        "macro_f1": f1_mac,
        "unknown_rate": unknown_rate,
        "n_test": n_total,
        "n_skipped": n_skipped,
        "classification_report": report,
    }

    # Per-group метрики
    if groups:
        for group_name in ("A", "B", "C"):
            group_classes = [c for c, g in groups.items() if g == group_name]
            mask = df["true_label"].isin(group_classes)
            if mask.sum() > 0:
                df_g = df[mask]
                g_labels = sorted(set(df_g["true_label"]))
                f1_g = f1_score(
                    df_g["true_label"], df_g["predicted_label"],
                    average="macro", zero_division=0, labels=g_labels,
                )
            else:
                f1_g = 0.0
            result[f"f1_group_{group_name}"] = f1_g

    # Печать
    print("=" * 60)
    print("PROMPT-BASED КЛАССИФИКАЦИЯ")
    print("=" * 60)
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {f1_mac:.4f}")
    print(f"  Unknown rate:      {unknown_rate:.4f} ({n_unknown}/{len(y_pred)})")
    print(f"  Skipped:           {n_skipped}/{n_total}")
    if groups:
        print(f"  F1 Group A:        {result.get('f1_group_A', 0):.4f}")
        print(f"  F1 Group B:        {result.get('f1_group_B', 0):.4f}")
        print(f"  F1 Group C:        {result.get('f1_group_C', 0):.4f}")
    print(f"\n{report}")

    return result
