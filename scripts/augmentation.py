#!/usr/bin/env python3
"""
augmentation.py - серверный запуск пайплайна из notebooks/augmentation.ipynb.

Из ноутбука убраны Colab/Drive, git clone/pull, pip install, magic-команды и
интерактивные графики. Остальная логика сохранена: подготовка данных,
baseline-метрики, три этапа аугментации, финальная оценка и CSV с результатами.

Примеры:
    python scripts/augmentation.py
    python scripts/augmentation.py --gpu A100_40
    python scripts/augmentation.py --config config_models/aug_configs/model_vllm_32b.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config_models" / "aug_configs" / "model_vllm_32b.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск полного augmentation pipeline на сервере.",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("AUG_GPU", "A100_40"),
        choices=["T4", "L4", "A100_40", "A100_80", "H100"],
        help="GPU-профиль из config_models/pipeline_config.json.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="Путь до aug model config. Можно абсолютный или project-relative.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_distribution(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    from src.utils.data_loader import get_class_distribution

    dist_train = get_class_distribution(df_train)
    dist_test = get_class_distribution(df_test)

    print(f"{'Класс':<70} {'Train':>6} {'Test':>5}")
    print("-" * 85)
    for cls in dist_train.index:
        tr = dist_train[cls]
        te = dist_test.get(cls, 0)
        print(f"  {cls:<68} {tr:>6} {te:>5}")
    print("-" * 85)
    print(f"  {'ИТОГО':<68} {len(df_train):>6} {len(df_test):>5}")


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.utils.data_cleaner import run as run_cleaning
    from src.utils.data_loader import (
        DATA_DIR,
        ORIGINAL_FILE,
        STAGE_FILES,
        TEST_FILE,
        split_train_test,
    )

    eda_path = DATA_DIR / "data_after_eda.csv"
    if eda_path.exists():
        print(f"data_after_eda.csv уже существует ({len(pd.read_csv(eda_path))} записей), пропускаем")
    else:
        run_cleaning()

    train_path = DATA_DIR / STAGE_FILES[0]
    test_path = DATA_DIR / TEST_FILE

    if train_path.exists() and test_path.exists():
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print("Train/test уже существуют, загружены из файлов")
    else:
        original_path = DATA_DIR / ORIGINAL_FILE
        df_original = pd.read_csv(original_path)
        print(f"Загружен оригинал: {original_path.name} ({len(df_original)} записей)")
        df_train, df_test = split_train_test(df_original)

    total = len(df_train) + len(df_test)
    print(f"\nTrain: {len(df_train)} ({len(df_train) / total * 100:.0f}%)")
    print(f"Test:  {len(df_test)} ({len(df_test) / total * 100:.0f}%)")
    print_distribution(df_train, df_test)
    return df_train, df_test


def run_baseline_metrics(df_train_orig: pd.DataFrame, df_test_baseline: pd.DataFrame) -> list[dict]:
    from src.classification.embeddings import prepare_features
    from src.classification.evaluate import evaluate_model
    from src.classification.rubert_classifier import train_and_evaluate
    from src.utils.data_loader import RANDOM_SEED

    X_train_orig, y_train_orig_raw, X_test_orig, y_test_orig_raw = prepare_features(
        df_train_orig,
        df_test_baseline,
    )

    le_baseline = LabelEncoder()
    y_train_orig = le_baseline.fit_transform(y_train_orig_raw)
    y_test_orig = le_baseline.transform(y_test_orig_raw)
    label_names_baseline = le_baseline.classes_

    print(f"Train (без аугментации): {X_train_orig.shape}")
    print(f"Test: {X_test_orig.shape}")
    print(f"Классов: {len(label_names_baseline)}")

    baseline_results = []

    print("=" * 60)
    print("BASELINE (без аугментации)")
    print("=" * 60)

    baseline_results.append(evaluate_model(
        name="[Baseline] Linear SVM",
        estimator=LinearSVC(max_iter=10000, random_state=RANDOM_SEED, dual="auto"),
        X_train=X_train_orig,
        y_train=y_train_orig,
        X_test=X_test_orig,
        y_test=y_test_orig,
        label_names=label_names_baseline,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))

    baseline_results.append(evaluate_model(
        name="[Baseline] Logistic Regression",
        estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED),
        X_train=X_train_orig,
        y_train=y_train_orig,
        X_test=X_test_orig,
        y_test=y_test_orig,
        label_names=label_names_baseline,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))

    baseline_results.append(evaluate_model(
        name="[Baseline] Multinomial Naive Bayes",
        estimator=MultinomialNB(),
        X_train=X_train_orig,
        y_train=y_train_orig,
        X_test=X_test_orig,
        y_test=y_test_orig,
        label_names=label_names_baseline,
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0]},
    ))

    rubert_configs = [
        {
            "model_name": "cointegrated/rubert-tiny2",
            "short_name": "rubert-tiny2",
            "lr": 5e-4,
            "num_epochs": 15,
            "batch_size": 32,
        },
        {
            "model_name": "DeepPavlov/rubert-base-cased",
            "short_name": "rubert-base",
            "lr": 5e-5,
            "num_epochs": 15,
            "batch_size": 32,
        },
    ]

    for cfg in rubert_configs:
        baseline_results.append(train_and_evaluate(
            df_train=df_train_orig,
            df_test=df_test_baseline,
            model_name=cfg["model_name"],
            lr=cfg["lr"],
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            name=f"[Baseline] {cfg['short_name']}",
        ))

    return baseline_results


def run_augmentation_stages(config_path: Path, pipeline_cfg) -> pd.DataFrame:
    from src.augmentation.stage1_llm_generate import run as run_stage1
    from src.augmentation.stage2_paraphrase import run as run_stage2
    from src.augmentation.stage3_back_translation import run as run_stage3
    from src.utils.data_loader import get_class_distribution, load_dataset

    print(f"Конфиг модели: {config_path}")

    run_stage1(str(config_path), pipeline_cfg=pipeline_cfg)

    df_after_s1 = load_dataset(stage=1)
    dist_s1 = get_class_distribution(df_after_s1)
    while (dist_s1 < 15).sum() != 0:
        print(f"Записей после этапа 1: {len(df_after_s1)}")
        print(f"Классов с < 15 примерами: {(dist_s1 < 15).sum()}")
        print("=" * 100)
        print("Повторяем 1й этап")
        print("=" * 100)
        run_stage1(str(config_path), pipeline_cfg=pipeline_cfg)
        df_after_s1 = load_dataset(stage=1)
        dist_s1 = get_class_distribution(df_after_s1)
    print("Этап 1: Генерация с помощью LLM полностью завершен.")

    run_stage2(str(config_path), pipeline_cfg=pipeline_cfg)

    df_after_s2 = load_dataset(stage=2)
    dist_s2 = get_class_distribution(df_after_s2)
    print(f"Записей после этапа 2: {len(df_after_s2)}")
    print(f"Классов с < 35 примерами: {(dist_s2 < 35).sum()}")

    run_stage3(str(config_path), pipeline_cfg=pipeline_cfg)

    df_final = load_dataset(stage=3)
    dist_final = get_class_distribution(df_final)
    print(f"Записей после всех этапов: {len(df_final)}")
    print(f"Классов с < 50 примерами: {(dist_final < 50).sum()}")
    print(f"\nМинимум примеров в классе: {dist_final.min()}")
    print(f"Максимум примеров в классе: {dist_final.max()}")

    return df_final


def run_augmented_metrics(
    df_final: pd.DataFrame,
    df_test_baseline: pd.DataFrame,
) -> list[dict]:
    from src.classification.evaluate import evaluate_model, load_data
    from src.classification.rubert_classifier import train_and_evaluate
    from src.utils.data_loader import RANDOM_SEED

    X_train, y_train, X_test, y_test, label_names = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Классов: {len(label_names)}")

    augmented_results = []

    augmented_results.append(evaluate_model(
        name="[Augmented] Linear SVM",
        estimator=LinearSVC(max_iter=10000, random_state=RANDOM_SEED, dual="auto"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))

    augmented_results.append(evaluate_model(
        name="[Augmented] Logistic Regression",
        estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))

    augmented_results.append(evaluate_model(
        name="[Augmented] Multinomial Naive Bayes",
        estimator=MultinomialNB(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0]},
    ))

    rubert_configs = [
        {
            "model_name": "cointegrated/rubert-tiny2",
            "short_name": "rubert-tiny2",
            "lr": 5e-4,
            "num_epochs": 10,
            "batch_size": 32,
        },
        {
            "model_name": "DeepPavlov/rubert-base-cased",
            "short_name": "rubert-base",
            "lr": 5e-5,
            "num_epochs": 14,
            "batch_size": 32,
        },
    ]

    for cfg in rubert_configs:
        augmented_results.append(train_and_evaluate(
            df_train=df_final,
            df_test=df_test_baseline,
            model_name=cfg["model_name"],
            lr=cfg["lr"],
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            name=f"[Augmented] {cfg['short_name']}",
        ))

    return augmented_results


def save_comparison_results(baseline_results: list[dict], augmented_results: list[dict]) -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    n_pairs = min(len(baseline_results), len(augmented_results))
    if len(baseline_results) != len(augmented_results):
        print(
            "[Внимание] Количество baseline и augmented результатов отличается: "
            f"{len(baseline_results)} vs {len(augmented_results)}. "
            f"Сравниваю первые {n_pairs}."
        )

    rows = []
    print(f"{'Модель':<25} | {'Метрика':<20} | {'Baseline':>10} | {'Augmented':>10} | {'Delta':>10}")
    print("-" * 83)
    for i in range(n_pairs):
        base = baseline_results[i]
        aug = augmented_results[i]
        name = base["name"].replace("[Baseline] ", "").replace("[Augmented] ", "")

        for metric, metric_label in (
            ("balanced_accuracy", "Balanced Accuracy"),
            ("macro_f1", "Macro F1"),
        ):
            b = base[metric]
            a = aug[metric]
            delta = a - b
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<23} | {metric_label:<20} | {b:>10.4f} | {a:>10.4f} | {sign}{delta:>9.4f}")
        print("-" * 83)

        rows.append({
            "stage": "baseline",
            "model": name,
            "balanced_accuracy": round(float(base["balanced_accuracy"]), 4),
            "macro_f1": round(float(base["macro_f1"]), 4),
        })
        rows.append({
            "stage": "augmented",
            "model": name,
            "balanced_accuracy": round(float(aug["balanced_accuracy"]), 4),
            "macro_f1": round(float(aug["macro_f1"]), 4),
        })

    df_results = pd.DataFrame(rows)
    csv_path = results_dir / "classification_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Сохранено: {csv_path}")
    print()
    print(df_results.to_string(index=False))


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.chdir(PROJECT_ROOT)

    from src.utils.data_loader import DATA_DIR, load_dataset, load_test_set
    from src.utils.pipeline_config import load_pipeline_config

    args = parse_args()
    config_path = resolve_project_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг модели не найден: {config_path}")

    pipeline_cfg = load_pipeline_config(args.gpu)

    print("=" * 60)
    print(f"AUGMENTATION | gpu={args.gpu} | config={config_path.name}")
    print("=" * 60)
    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка данных:   {DATA_DIR}")

    prepare_data()

    df_test_baseline = load_test_set()
    df_train_orig = load_dataset(stage=0)
    baseline_results = run_baseline_metrics(df_train_orig, df_test_baseline)

    df_final = run_augmentation_stages(config_path, pipeline_cfg)

    augmented_results = run_augmented_metrics(df_final, df_test_baseline)
    save_comparison_results(baseline_results, augmented_results)


if __name__ == "__main__":
    main()
