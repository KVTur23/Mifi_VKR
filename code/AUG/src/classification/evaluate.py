"""
evaluate.py — Общая логика оценки baseline-классификаторов

Загружает данные, подбирает гиперпараметры (если есть),
прогоняет кросс-валидацию и выводит метрики + classification report.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, RANDOM_SEED
from src.classification.embeddings import load_embedding_model, prepare_features

STAGE = 3
CV_FOLDS = 5


def evaluate_model(name: str, estimator, param_grid: dict | None = None) -> None:
    """
    Единый пайплайн оценки классификатора.

    Аргументы:
        name:        название модели для вывода (например, "SVM")
        estimator:   sklearn-модель (уже с нужными параметрами кроме тех, что в param_grid)
        param_grid:  сетка для GridSearchCV (например, {"C": [0.01, 0.1, 1, 10]}).
                     Если None — просто кросс-валидация без подбора.
    """
    print("=" * 60)
    print(f"КЛАССИФИКАЦИЯ: {name}")
    print("=" * 60)

    # --- Данные ---
    df = load_dataset(stage=STAGE)
    model = load_embedding_model()
    X, y_raw = prepare_features(df, model)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"\n[{name}] Датасет: {X.shape[0]} примеров, {X.shape[1]} фичей, "
          f"{len(le.classes_)} классов")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # --- Подбор гиперпараметров (если есть) ---
    if param_grid:
        print(f"\n[{name}] GridSearchCV ({CV_FOLDS} фолдов), параметры: {param_grid}")

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X, y)

        print(f"[{name}] Лучшие параметры: {grid.best_params_} "
              f"(macro F1 = {grid.best_score_:.4f})")

        for score, params in zip(grid.cv_results_["mean_test_score"],
                                  grid.cv_results_["params"]):
            print(f"  {params} → macro F1 = {score:.4f}")

        # Обновляем estimator лучшими параметрами
        estimator = grid.best_estimator_

    # --- Кросс-валидация ---
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}

    cv_results = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
    )

    print(f"\n{'Метрика':<20} {'Среднее':>10} {'Std':>10}")
    print("-" * 42)
    for key, label in [("test_accuracy", "Accuracy"),
                        ("test_f1_macro", "Macro F1"),
                        ("test_f1_weighted", "Weighted F1")]:
        s = cv_results[key]
        print(f"  {label:<18} {s.mean():>10.4f} {s.std():>10.4f}")

    # --- Classification report (честный, через cross_val_predict) ---
    y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1)
    print(f"\n[{name}] Classification report (cross-validated):")
    print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))

    print(f"[{name}] Готово.")
