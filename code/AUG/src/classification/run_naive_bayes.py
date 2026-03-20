"""
run_naive_bayes.py — Gaussian Naive Bayes baseline

Запуск:
    python src/classification/run_naive_bayes.py
"""

import sys
from pathlib import Path

from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.evaluate import load_data, evaluate_model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, labels = load_data()

    evaluate_model(
        name="Gaussian Naive Bayes",
        estimator=GaussianNB(),
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        label_names=labels,
    )
