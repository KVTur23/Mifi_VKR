"""
run_svm.py — Linear SVM baseline

Запуск:
    python src/classification/run_svm.py
"""

import sys
from pathlib import Path

from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import RANDOM_SEED
from src.classification.evaluate import evaluate_model


if __name__ == "__main__":
    evaluate_model(
        name="Linear SVM",
        estimator=LinearSVC(max_iter=10000, random_state=RANDOM_SEED, dual="auto"),
        param_grid={"C": [0.01, 0.1, 1, 10]},
    )
