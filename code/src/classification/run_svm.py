"""
run_svm.py — Классификация писем с помощью Linear SVM

Baseline-модель для оценки качества аугментации. Берёт финальный датасет
(после всех этапов аугментации), получает BERT-эмбеддинги, обучает LinearSVC
с подбором гиперпараметра C через кросс-валидацию.

Пайплайн:
1. Загрузка data_after_stage3.csv
2. Эмбеддинги через ai-forever/sbert_large_nlu_ru
3. GridSearchCV по параметру C (StratifiedKFold, k=5)
4. Метрики: accuracy, macro F1, weighted F1, classification report

Запуск:
    python src/classification/run_svm.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, RANDOM_SEED
from src.classification.embeddings import load_embedding_model, prepare_features


# --- Настройки ---

STAGE = 3                        # Загружаем финальный датасет после всех этапов
CV_FOLDS = 5                     # Кросс-валидация: 5 фолдов
C_VALUES = [0.01, 0.1, 1, 10]   # Сетка для подбора параметра регуляризации C
MAX_ITER = 10000                 # Лимит итераций — на 36 классах может сходиться долго


def run() -> None:
    """
    Основная функция — загружает данные, подбирает C, выводит метрики.
    """
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ: Linear SVM (LinearSVC)")
    print("=" * 60)

    # --- Загрузка данных и эмбеддингов ---
    df = load_dataset(stage=STAGE)
    model = load_embedding_model()
    X, y_raw = prepare_features(df, model)

    # LabelEncoder — LinearSVC работает с числовыми метками,
    # а у нас строковые названия классов
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_

    print(f"\n[SVM] Датасет: {X.shape[0]} примеров, {X.shape[1]} фичей, "
          f"{len(class_names)} классов")

    # --- Стратифицированная кросс-валидация ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # --- Подбор C через GridSearchCV ---
    print(f"\n[SVM] Подбираю C из {C_VALUES} (GridSearchCV, {CV_FOLDS} фолдов)...")

    grid = GridSearchCV(
        estimator=LinearSVC(max_iter=MAX_ITER, random_state=RANDOM_SEED, dual="auto"),
        param_grid={"C": C_VALUES},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)

    best_c = grid.best_params_["C"]
    print(f"[SVM] Лучший C = {best_c} (macro F1 на CV = {grid.best_score_:.4f})")

    # Показываем результаты для всех значений C
    print(f"\n[SVM] Результаты подбора C:")
    for mean_score, params in zip(grid.cv_results_["mean_test_score"],
                                   grid.cv_results_["params"]):
        print(f"  C={params['C']:<6} → macro F1 = {mean_score:.4f}")

    # --- Финальная кросс-валидация с лучшим C ---
    # GridSearchCV уже дал нам среднюю метрику, но для полного отчёта
    # прогоним cross_validate с несколькими метриками
    print(f"\n[SVM] Финальная оценка с C={best_c}:")

    best_model = LinearSVC(C=best_c, max_iter=MAX_ITER, random_state=RANDOM_SEED, dual="auto")

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

    # --- Classification report на всём датасете (обучаем на всех данных) ---
    # Это для наглядности — чтобы увидеть, какие классы модель путает.
    # Реальная оценка — cross_validate выше
    print(f"\n[SVM] Classification report (обучение на всём датасете):")
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    print("[SVM] Готово.")


if __name__ == "__main__":
    run()
