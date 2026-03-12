"""
run_naive_bayes.py — Классификация писем с помощью Gaussian Naive Bayes

Baseline-модель для оценки качества аугментации. Используем GaussianNB,
потому что эмбеддинги из SBERT — dense-векторы с вещественными значениями.
MultinomialNB тут не подходит: он ожидает неотрицательные фичи (частоты, TF-IDF),
а у нас эмбеддинги с отрицательными значениями.

GaussianNB не имеет гиперпараметров для подбора (в отличие от SVM и LogReg),
поэтому просто прогоняем кросс-валидацию.

Пайплайн:
1. Загрузка data_after_stage3.csv
2. Эмбеддинги через ai-forever/sbert_large_nlu_ru
3. Кросс-валидация StratifiedKFold, k=5
4. Метрики: accuracy, macro F1, weighted F1, classification report

Запуск:
    python src/classification/run_naive_bayes.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, RANDOM_SEED
from src.classification.embeddings import load_embedding_model, prepare_features


# --- Настройки ---

STAGE = 3        # Финальный датасет после всех этапов
CV_FOLDS = 5     # Кросс-валидация: 5 фолдов


def run() -> None:
    """
    Основная функция — загружает данные, прогоняет кросс-валидацию, выводит метрики.
    """
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ: Gaussian Naive Bayes")
    print("=" * 60)

    # --- Загрузка данных и эмбеддингов ---
    df = load_dataset(stage=STAGE)
    model = load_embedding_model()
    X, y_raw = prepare_features(df, model)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_

    print(f"\n[NaiveBayes] Датасет: {X.shape[0]} примеров, {X.shape[1]} фичей, "
          f"{len(class_names)} классов")

    # --- Стратифицированная кросс-валидация ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # GaussianNB не имеет гиперпараметров для тюнинга —
    # просто прогоняем cross_validate
    print(f"\n[NaiveBayes] Кросс-валидация ({CV_FOLDS} фолдов)...")

    nb_model = GaussianNB()

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(
        nb_model, X, y,
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
    print(f"\n[NaiveBayes] Classification report (обучение на всём датасете):")
    nb_model.fit(X, y)
    y_pred = nb_model.predict(X)
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    print("[NaiveBayes] Готово.")


if __name__ == "__main__":
    run()
