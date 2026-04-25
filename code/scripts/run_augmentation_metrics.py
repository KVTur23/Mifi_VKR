"""
run_augmentation_metrics.py — метрики классификации до и после аугментации.

Запуск:
    python scripts/run_augmentation_metrics.py --experiment exp4

Считает одинаковые TF-IDF классификаторы на:
  - train_after_eda.csv      (до аугментации)
  - последнем data_after_stageN.csv до --after-stage (после аугментации)

Результаты:
  - results/augmentation_metrics.csv
  - results/augmentation_metrics_report.txt
  - results/all_methods_comparison.csv (обновляется блоком augmentation)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.embeddings import prepare_features
from src.classification.rubert_classifier import train_and_evaluate
from src.utils.data_loader import (
    DATA_DIR,
    LABEL_COL,
    RANDOM_SEED,
    STAGE_FILES,
    TEXT_COL,
    load_test_set,
)


CLASSIFIERS = {
    "Linear SVM": (
        LinearSVC(max_iter=10000, random_state=RANDOM_SEED, dual="auto"),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Logistic Regression": (
        LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Multinomial Naive Bayes": (
        MultinomialNB(),
        {"alpha": [0.01, 0.1, 0.5, 1.0]},
    ),
}

RUBERT_CONFIGS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить метрики классификации до и после аугментации."
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Имя эксперимента для CSV, например exp1 или exp-aug-4.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Папка Data с train_after_eda.csv/data_after_stageN.csv/data_test.csv.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Папка для CSV/отчётов.",
    )
    parser.add_argument(
        "--after-stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Максимальный stage для датасета после аугментации.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать строки, даже если они уже есть в augmentation_metrics.csv.",
    )
    parser.add_argument(
        "--skip-rubert",
        action="store_true",
        help="Считать только TF-IDF модели, без ruBERT fine-tuning.",
    )
    return parser.parse_args()


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _experiment_name(explicit: str | None) -> str:
    if explicit:
        return explicit
    return _git_branch() or PROJECT_ROOT.parent.name


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (TEXT_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"В {path} нет колонок: {', '.join(missing)}")
    return df


def _resolve_after_dataset(data_dir: Path, max_stage: int) -> tuple[pd.DataFrame, int, Path]:
    for stage in range(max_stage, 0, -1):
        path = data_dir / STAGE_FILES[stage]
        if path.exists():
            return _load_csv(path), stage, path
    raise FileNotFoundError(
        f"Не найден data_after_stageN.csv до stage={max_stage} в {data_dir}"
    )


def _class_groups(df_original: pd.DataFrame) -> dict[str, str]:
    counts = df_original[LABEL_COL].value_counts().to_dict()
    groups = {}
    for cls, count in counts.items():
        if count >= 50:
            groups[cls] = "A"
        elif count >= 15:
            groups[cls] = "B"
        else:
            groups[cls] = "C"
    return groups


def _group_f1(y_true: list[str], y_pred: list[str], groups: dict[str, str]) -> dict[str, float]:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    out = {}
    for group_name in ("A", "B", "C"):
        group_classes = {c for c, g in groups.items() if g == group_name}
        part = df[df["true"].isin(group_classes)]
        if part.empty:
            out[f"f1_group_{group_name}"] = 0.0
            continue
        labels = sorted(part["true"].unique())
        out[f"f1_group_{group_name}"] = f1_score(
            part["true"],
            part["pred"],
            average="macro",
            labels=labels,
            zero_division=0,
        )
    return out


def _evaluate_classifier(
    model_name: str,
    estimator,
    param_grid: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    groups: dict[str, str],
) -> tuple[dict, str]:
    X_train, y_train_raw, X_test, y_test_raw = prepare_features(
        df_train,
        df_test,
        use_cache=True,
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    y_true_labels = le.inverse_transform(y_test).tolist()
    y_pred_labels = le.inverse_transform(y_pred).tolist()

    result = {
        "model": model_name,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "best_params": repr(grid.best_params_),
        "cv_macro_f1": grid.best_score_,
    }
    result.update(_group_f1(y_true_labels, y_pred_labels, groups))

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        zero_division=0,
    )
    return result, report


def _upsert_rows(csv_path: Path, rows: list[dict], keys: list[str], force: bool) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        if force:
            key_frame = new_df[keys].drop_duplicates()
            old_df = old_df.merge(key_frame, on=keys, how="left", indicator=True)
            old_df = old_df[old_df["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            old_keys = {tuple(x) for x in old_df[keys].astype(str).itertuples(index=False, name=None)}
            new_df = new_df[
                ~new_df[keys].astype(str).apply(tuple, axis=1).isin(old_keys)
            ]
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return out


def _update_all_methods(results_dir: Path, experiment: str, rows: list[dict], force: bool) -> None:
    comparison_csv = results_dir / "all_methods_comparison.csv"
    comparison_rows = []
    for row in rows:
        comparison_rows.append(
            {
                "method": "augmentation",
                "model": row["model"],
                "setting": f"{experiment}_{row['setting']}",
                "balanced_accuracy": row["balanced_accuracy"],
                "macro_f1": row["macro_f1"],
                "unknown_rate": 0.0,
            }
        )
    _upsert_rows(
        comparison_csv,
        comparison_rows,
        keys=["method", "model", "setting"],
        force=force,
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    experiment = _experiment_name(args.experiment)

    before_path = data_dir / STAGE_FILES[0]
    df_before = _load_csv(before_path)
    df_after, after_stage, after_path = _resolve_after_dataset(data_dir, args.after_stage)
    df_test = load_test_set(data_dir=data_dir)
    groups = _class_groups(df_before)

    datasets = [
        ("before_augmentation", 0, before_path, df_before),
        (f"after_stage{after_stage}", after_stage, after_path, df_after),
    ]

    rows = []
    reports = []
    run_ts = datetime.now().isoformat(timespec="seconds")

    print("=" * 60)
    print(f"МЕТРИКИ АУГМЕНТАЦИИ: {experiment}")
    print("=" * 60)
    print(f"До:    {before_path.name} ({len(df_before)} записей)")
    print(f"После: {after_path.name} ({len(df_after)} записей)")
    print(f"Тест:  {len(df_test)} записей")

    for setting, stage, train_path, df_train in datasets:
        print()
        print("#" * 60)
        print(f"# {setting}: {train_path.name} ({len(df_train)} train)")
        print("#" * 60)
        for model_name, (estimator, param_grid) in CLASSIFIERS.items():
            print(f"\n[{setting}] {model_name}")
            result, report = _evaluate_classifier(
                model_name,
                estimator,
                param_grid,
                df_train,
                df_test,
                groups,
            )
            row = {
                "experiment": experiment,
                "setting": setting,
                "stage": stage,
                "train_file": train_path.name,
                "train_size": len(df_train),
                "test_size": len(df_test),
                "n_classes": df_train[LABEL_COL].nunique(),
                "run_ts": run_ts,
                **result,
            }
            rows.append(row)
            reports.append(
                "\n".join(
                    [
                        "=" * 80,
                        f"{experiment} | {setting} | {model_name}",
                        f"train={train_path.name}, size={len(df_train)}",
                        f"best_params={result['best_params']}",
                        report,
                    ]
                )
            )
            print(
                f"  balanced_accuracy={result['balanced_accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}, "
                f"cv_macro_f1={result['cv_macro_f1']:.4f}"
            )
        if args.skip_rubert:
            continue

        for cfg in RUBERT_CONFIGS:
            model_name = cfg["short_name"]
            print(f"\n[{setting}] {model_name}")
            result = train_and_evaluate(
                df_train=df_train,
                df_test=df_test,
                model_name=cfg["model_name"],
                lr=cfg["lr"],
                num_epochs=cfg["num_epochs"],
                batch_size=cfg["batch_size"],
                name=model_name,
            )
            row = {
                "experiment": experiment,
                "setting": setting,
                "stage": stage,
                "train_file": train_path.name,
                "train_size": len(df_train),
                "test_size": len(df_test),
                "n_classes": df_train[LABEL_COL].nunique(),
                "run_ts": run_ts,
                "model": model_name,
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
                "best_params": "",
                "cv_macro_f1": "",
                "f1_group_A": "",
                "f1_group_B": "",
                "f1_group_C": "",
            }
            rows.append(row)
            reports.append(
                "\n".join(
                    [
                        "=" * 80,
                        f"{experiment} | {setting} | {model_name}",
                        f"train={train_path.name}, size={len(df_train)}",
                        "classification_report printed in SLURM log by rubert_classifier.py",
                    ]
                )
            )
            print(
                f"  balanced_accuracy={result['balanced_accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}"
            )

    metrics_csv = results_dir / "augmentation_metrics.csv"
    out_df = _upsert_rows(
        metrics_csv,
        rows,
        keys=["experiment", "setting", "model"],
        force=args.force,
    )
    _update_all_methods(results_dir, experiment, rows, force=args.force)

    report_path = results_dir / "augmentation_metrics_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))
        f.write("\n")

    print()
    print("=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"Метрики: {metrics_csv}")
    print(f"Отчёт:   {report_path}")
    print(f"Сводка:  {results_dir / 'all_methods_comparison.csv'}")
    print()
    print(out_df.tail(len(rows)).to_string(index=False))


if __name__ == "__main__":
    main()
