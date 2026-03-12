"""
run_logreg.py — Logistic Regression baseline

Запуск:
    python src/classification/run_logreg.py
"""

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import RANDOM_SEED
from src.classification.evaluate import evaluate_model


if __name__ == "__main__":
    evaluate_model(
        name="Logistic Regression",
        estimator=LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        param_grid={"C": [0.01, 0.1, 1, 10]},
    )
