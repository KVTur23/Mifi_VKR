"""
run_random_forest.py — Random Forest baseline

Запуск:
    python src/classification/run_random_forest.py
"""

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import RANDOM_SEED
from src.classification.evaluate import evaluate_model


if __name__ == "__main__":
    evaluate_model(
        name="Random Forest",
        estimator=RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid={"n_estimators": [100, 300, 500], "max_depth": [None, 30, 50]},
    )
