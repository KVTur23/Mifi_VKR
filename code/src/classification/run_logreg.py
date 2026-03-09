"""
run_logreg.py — Классификация писем с помощью Logistic Regression

Baseline-модель для оценки качества аугментации. Берёт финальный датасет
(после всех этапов аугментации), получает BERT-эмбеддинги, обучает
LogisticRegression с multinomial-стратегией и подбором C.

Пайплайн:
1. Загрузка data_after_stage3.csv
2. Эмбеддинги через ai-forever/sbert_large_nlu_ru
3. GridSearchCV по параметру C (StratifiedKFold, k=5)
4. Метрики: accuracy, macro F1, weighted F1, classification report

Запуск:
    python src/classification/run_logreg.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, RANDOM_SEED
from src.classification.embeddings import load_embedding_model, prepare_features


# --- Настройки ---

STAGE = 3                        # Финальный датасет после всех этапов
CV_FOLDS = 5                     # Кросс-валидация: 5 фолдов
C_VALUES = [0.01, 0.1, 1, 10]   # Сетка для подбора регуляризации
MAX_ITER = 1000                  # Для multinomial с lbfgs обычно хватает 1000


def run() -> None:
    """
    Основная функция — загружает данные, подбирает C, выводит метрики.
    """
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ: Logistic Regression (multinomial)")
    print("=" * 60)

    # --- Загрузка данных и эмбеддингов ---
    df = load_dataset(stage=STAGE)
    model = load_embedding_model()
    X, y_raw = prepare_features(df, model)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_

    print(f"\n[LogReg] Датасет: {X.shape[0]} примеров, {X.shape[1]} фичей, "
          f"{len(class_names)} классов")

    # --- Стратифицированная кросс-валидация ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # --- Подбор C через GridSearchCV ---
    print(f"\n[LogReg] Подбираю C из {C_VALUES} (GridSearchCV, {CV_FOLDS} фолдов)...")

    grid = GridSearchCV(
        estimator=LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED,
        ),
        param_grid={"C": C_VALUES},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)

    best_c = grid.best_params_["C"]
    print(f"[LogReg] Лучший C = {best_c} (macro F1 на CV = {grid.best_score_:.4f})")

    # Результаты для всех значений C
    print(f"\n[LogReg] Результаты подбора C:")
    for mean_score, params in zip(grid.cv_results_["mean_test_score"],
                                   grid.cv_results_["params"]):
        print(f"  C={params['C']:<6} → macro F1 = {mean_score:.4f}")

    # --- Финальная кросс-валидация с лучшим C ---
    print(f"\n[LogReg] Финальная оценка с C={best_c}:")

    best_model = LogisticRegression(
        C=best_c,
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(
        best_model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    # --- Вывод метрик ---
    print(f"\n{'Метрика':<20} {'Среднее':>10} {'Std':>10}")
    print("-" * 42)
    for metric_name, display_name in [
        ("test_accuracy", "Accuracy"),
        ("test_f1_macro", "Macro F1"),
        ("test_f1_weighted", "Weighted F1"),
    ]:
        scores = cv_results[metric_name]
        print(f"  {display_name:<18} {scores.mean():>10.4f} {scores.std():>10.4f}")

    # --- Classification report на всём датасете ---
    # Для наглядности — какие классы путает. Реальная оценка — cross_validate выше
    print(f"\n[LogReg] Classification report (обучение на всём датасете):")
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    print("[LogReg] Готово.")


if __name__ == "__main__":
    run()
