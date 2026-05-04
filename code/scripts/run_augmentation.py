#!/usr/bin/env python3
"""
Server runner for the augmentation pipeline from notebooks/augmentation.ipynb.

The script keeps the notebook order for data preparation and stages, but moves
all metric comparison work to the end so the expensive augmentation path is not
interleaved with evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        description="Run the full augmentation notebook pipeline on the server."
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("GPU", "A100_40"),
        choices=["T4", "L4", "A100_40", "A100_80", "H100"],
        help="GPU profile from config_models/pipeline_config.json.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "CONFIG_REL",
            "config_models/aug_configs/model_vllm_32b.json",
        ),
        help="LLM judge/generator config path, absolute or relative to project root.",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("RUN_NAME", "augmentation"),
        help="Human-readable run name used in manifests/logs.",
    )
    parser.add_argument(
        "--stage1-repeat-limit",
        type=int,
        default=20,
        help="Maximum number of Stage 1 reruns while classes remain below 15.",
    )
    parser.add_argument(
        "--metrics",
        choices=["none", "classical", "rubert", "all"],
        default=os.environ.get("AUG_METRICS", "all"),
        help="Which final metric blocks to run.",
    )
    parser.add_argument(
        "--rubert-models",
        choices=["tiny", "base", "all"],
        default=os.environ.get("AUG_RUBERT_MODELS", "all"),
        help="ruBERT configs for stage ablation metrics.",
    )
    parser.add_argument(
        "--rubert-epochs",
        type=int,
        default=int(os.environ.get("AUG_RUBERT_EPOCHS", "15")),
        help="Epochs for each ruBERT stage-ablation training run.",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def resolve_config_path(config: str) -> Path:
    path = Path(config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def git_commit() -> str | None:
    source_commit = PROJECT_ROOT / "source_commit.txt"
    if source_commit.exists():
        return source_commit.read_text(encoding="utf-8").strip()

    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_core_modules():
    from src.utils.data_loader import (
        DATA_DIR,
        LABEL_COL,
        ORIGINAL_FILE,
        RANDOM_SEED,
        STAGE_FILES,
        TEST_FILE,
        get_class_distribution,
        load_dataset,
        load_test_set,
        split_train_test,
    )

    return {
        "DATA_DIR": DATA_DIR,
        "LABEL_COL": LABEL_COL,
        "ORIGINAL_FILE": ORIGINAL_FILE,
        "RANDOM_SEED": RANDOM_SEED,
        "STAGE_FILES": STAGE_FILES,
        "TEST_FILE": TEST_FILE,
        "get_class_distribution": get_class_distribution,
        "load_dataset": load_dataset,
        "load_test_set": load_test_set,
        "split_train_test": split_train_test,
    }


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from src.utils.data_cleaner import run as run_cleaning

    core = load_core_modules()
    data_dir = core["DATA_DIR"]
    original_file = core["ORIGINAL_FILE"]
    stage_files = core["STAGE_FILES"]
    test_file = core["TEST_FILE"]
    split_train_test = core["split_train_test"]
    get_class_distribution = core["get_class_distribution"]

    section("ПОДГОТОВКА ДАННЫХ")
    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка данных:   {data_dir}")

    eda_path = data_dir / original_file
    if eda_path.exists():
        print(f"{eda_path.name} уже существует ({len(pd.read_csv(eda_path))} записей), пропускаем")
    else:
        run_cleaning()

    train_path = data_dir / stage_files[0]
    test_path = data_dir / test_file

    if train_path.exists() and test_path.exists():
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print("Train/test уже существуют, загружены из файлов")
    else:
        df_original = pd.read_csv(eda_path)
        print(f"Загружен оригинал: {eda_path.name} ({len(df_original)} записей)")
        df_train, df_test = split_train_test(df_original)

    total = len(df_train) + len(df_test)
    print(f"\nTrain: {len(df_train)} ({len(df_train) / total * 100:.0f}%)")
    print(f"Test:  {len(df_test)} ({len(df_test) / total * 100:.0f}%)")

    dist_train = get_class_distribution(df_train)
    dist_test = get_class_distribution(df_test)

    print(f"\n{'Класс':<70} {'Train':>6} {'Test':>5}")
    print("-" * 85)
    for cls in dist_train.index:
        tr = dist_train[cls]
        te = dist_test.get(cls, 0)
        print(f"  {cls:<68} {tr:>6} {te:>5}")
    print("-" * 85)
    print(f"  {'ИТОГО':<68} {len(df_train):>6} {len(df_test):>5}")

    return df_train, df_test, dist_train


def run_stage1_until_complete(config_path: Path, pipeline_cfg, repeat_limit: int) -> None:
    from src.augmentation.stage1_llm_generate import run as run_stage1

    core = load_core_modules()
    load_dataset = core["load_dataset"]
    get_class_distribution = core["get_class_distribution"]

    section("ЭТАП 1: LLM-ГЕНЕРАЦИЯ (< 15 -> 15)")
    run_stage1(str(config_path), pipeline_cfg=pipeline_cfg)

    for attempt in range(1, repeat_limit + 1):
        df_after_s1 = load_dataset(stage=1)
        dist_s1 = get_class_distribution(df_after_s1)
        missing = int((dist_s1 < 15).sum())
        if missing == 0:
            print("Этап 1: Генерация с помощью LLM полностью завершена.")
            return

        print(f"Записей после этапа 1: {len(df_after_s1)}")
        print(f"Классов с < 15 примерами: {missing}")
        print(f"Повторяем 1й этап ({attempt}/{repeat_limit})")
        run_stage1(str(config_path), pipeline_cfg=pipeline_cfg)

    df_after_s1 = load_dataset(stage=1)
    dist_s1 = get_class_distribution(df_after_s1)
    missing = int((dist_s1 < 15).sum())
    if missing:
        raise RuntimeError(
            f"Stage 1 did not reach target after {repeat_limit} reruns: "
            f"{missing} classes remain below 15"
        )


def run_augmentation_stages(config_path: Path, pipeline_cfg, repeat_limit: int) -> None:
    from src.augmentation.stage2_paraphrase import run as run_stage2
    from src.augmentation.stage3_back_translation import run as run_stage3

    core = load_core_modules()
    load_dataset = core["load_dataset"]
    get_class_distribution = core["get_class_distribution"]

    run_stage1_until_complete(config_path, pipeline_cfg, repeat_limit)

    section("ЭТАП 2: ruT5-ПАРАФРАЗ С ЧАНКОВАНИЕМ (< 35 -> 35)")
    run_stage2(str(config_path), pipeline_cfg=pipeline_cfg)
    df_after_s2 = load_dataset(stage=2)
    dist_s2 = get_class_distribution(df_after_s2)
    print(f"Записей после этапа 2: {len(df_after_s2)}")
    print(f"Классов с < 35 примерами: {int((dist_s2 < 35).sum())}")

    section("ЭТАП 3: NLLB BACK-TRANSLATION С ЧАНКОВАНИЕМ (< 50 -> 50)")
    run_stage3(str(config_path), pipeline_cfg=pipeline_cfg)

    section("ФИНАЛЬНОЕ РАСПРЕДЕЛЕНИЕ")
    df_final = load_dataset(stage=3)
    dist_final = get_class_distribution(df_final)
    print(f"Записей после всех этапов: {len(df_final)}")
    print(f"Классов с < 50 примерами: {int((dist_final < 50).sum())}")
    print(f"Минимум примеров в классе: {dist_final.min()}")
    print(f"Максимум примеров в классе: {dist_final.max()}")

    save_distribution_plot(get_class_distribution(load_dataset(stage=0)), dist_final)


def save_distribution_plot(dist_train: pd.Series, dist_final: pd.Series) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        dist_train.plot(kind="bar", ax=axes[0], color="salmon")
        axes[0].set_title("Train до аугментации")
        axes[0].set_ylabel("Количество примеров")
        axes[0].tick_params(axis="x", rotation=45)

        dist_final.plot(kind="bar", ax=axes[1], color="steelblue")
        axes[1].set_title("Train после аугментации")
        axes[1].set_ylabel("Количество примеров")
        axes[1].axhline(y=50, color="g", linestyle="--", alpha=0.5, label="Целевой минимум (50)")
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        out_path = RESULTS_DIR / "class_distribution_before_after.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"График распределения сохранён: {out_path}")
    except Exception as e:
        print(f"[WARN] Не удалось сохранить график распределения: {e}")


def classical_results(stage: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC

    from src.classification.embeddings import prepare_features
    from src.classification.evaluate import evaluate_model

    core = load_core_modules()
    random_seed = core["RANDOM_SEED"]

    prefix = "Baseline" if stage == "baseline" else "Augmented"

    X_train, y_train_raw, X_test, y_test_raw = prepare_features(
        df_train,
        df_test,
        use_cache=False,
    )
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    label_names = le.classes_

    print(f"{prefix}: Train {X_train.shape}, Test {X_test.shape}, классов: {len(label_names)}")

    results = []
    results.append(evaluate_model(
        name=f"[{prefix}] Linear SVM",
        estimator=LinearSVC(max_iter=10000, random_state=random_seed, dual="auto"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))
    results.append(evaluate_model(
        name=f"[{prefix}] Logistic Regression",
        estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_seed),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"C": [0.01, 0.1, 1, 10]},
    ))
    results.append(evaluate_model(
        name=f"[{prefix}] Multinomial Naive Bayes",
        estimator=MultinomialNB(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0]},
    ))
    return results


def clean_model_name(result_name: str) -> str:
    return result_name.replace("[Baseline] ", "").replace("[Augmented] ", "")


def save_classical_comparison(baseline_results: list[dict], augmented_results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    comparison_rows = []
    metrics = ["balanced_accuracy", "macro_f1"]

    print(f"{'Модель':<30} | {'Метрика':<20} | {'Baseline':>10} | {'Augmented':>10} | {'Delta':>10}")
    print("-" * 90)

    for baseline, augmented in zip(baseline_results, augmented_results):
        model = clean_model_name(baseline["name"])
        rows.append({
            "stage": "baseline",
            "model": model,
            "balanced_accuracy": round(float(baseline["balanced_accuracy"]), 4),
            "macro_f1": round(float(baseline["macro_f1"]), 4),
        })
        rows.append({
            "stage": "augmented",
            "model": model,
            "balanced_accuracy": round(float(augmented["balanced_accuracy"]), 4),
            "macro_f1": round(float(augmented["macro_f1"]), 4),
        })

        for metric in metrics:
            b = float(baseline[metric])
            a = float(augmented[metric])
            delta = a - b
            sign = "+" if delta >= 0 else ""
            print(f"  {model:<28} | {metric:<20} | {b:>10.4f} | {a:>10.4f} | {sign}{delta:>9.4f}")
            comparison_rows.append({
                "model": model,
                "metric": metric,
                "baseline": round(b, 4),
                "augmented": round(a, 4),
                "delta": round(delta, 4),
            })
        print("-" * 90)

    df_results = pd.DataFrame(rows)
    df_comparison = pd.DataFrame(comparison_rows)

    results_csv = RESULTS_DIR / "classification_results.csv"
    comparison_csv = RESULTS_DIR / "classification_comparison.csv"
    df_results.to_csv(results_csv, index=False)
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"Сохранено: {results_csv}")
    print(f"Сохранено: {comparison_csv}")
    print(df_results.to_string(index=False))

    model_names = [clean_model_name(r["name"]) for r in baseline_results]
    x = np.arange(len(model_names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, metric, title in zip(axes, metrics, ["Balanced Accuracy", "Macro F1"]):
        vals_b = [float(r[metric]) for r in baseline_results]
        vals_a = [float(r[metric]) for r in augmented_results]
        bars_b = ax.bar(x - width / 2, vals_b, width, label="Baseline", color="salmon")
        bars_a = ax.bar(x + width / 2, vals_a, width, label="Augmented", color="steelblue")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        for bar in list(bars_b) + list(bars_a):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Baseline vs Augmented", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "baseline_vs_augmented.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"График сравнения сохранён: {plot_path}")


def run_classical_metrics() -> None:
    core = load_core_modules()
    load_dataset = core["load_dataset"]
    load_test_set = core["load_test_set"]

    section("МЕТРИКИ: BASELINE VS AUGMENTED (TF-IDF)")
    df_train_orig = load_dataset(stage=0)
    df_train_aug = load_dataset(stage=3)
    df_test = load_test_set()

    baseline_results = classical_results("baseline", df_train_orig, df_test)
    augmented_results = classical_results("augmented", df_train_aug, df_test)
    save_classical_comparison(baseline_results, augmented_results)


def selected_rubert_configs(selection: str, epochs: int) -> list[dict]:
    if selection == "tiny":
        configs = [cfg for cfg in RUBERT_CONFIGS if cfg["short_name"] == "rubert-tiny2"]
    elif selection == "base":
        configs = [cfg for cfg in RUBERT_CONFIGS if cfg["short_name"] == "rubert-base"]
    else:
        configs = RUBERT_CONFIGS

    out = []
    for cfg in configs:
        item = dict(cfg)
        item["num_epochs"] = epochs
        out.append(item)
    return out


def set_all_seeds(seed: int) -> None:
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def run_rubert_stage_ablation(selection: str, epochs: int) -> None:
    import torch

    from src.classification.rubert_classifier import train_and_evaluate

    core = load_core_modules()
    load_dataset = core["load_dataset"]
    load_test_set = core["load_test_set"]
    random_seed = core["RANDOM_SEED"]

    section("МЕТРИКИ: RUBERT STAGE ABLATION")
    df_test = load_test_set()
    stage_datasets = [
        {"stage": "after_eda", "stage_num": 0, "df": load_dataset(stage=0)},
        {"stage": "after_stage1", "stage_num": 1, "df": load_dataset(stage=1)},
        {"stage": "after_stage2", "stage_num": 2, "df": load_dataset(stage=2)},
        {"stage": "after_stage3", "stage_num": 3, "df": load_dataset(stage=3)},
    ]

    results = []
    partial_path = RESULTS_DIR / "rubert_stage_ablation_partial.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in selected_rubert_configs(selection, epochs):
        for stage_info in stage_datasets:
            set_all_seeds(random_seed)

            stage_name = stage_info["stage"]
            df_train_stage = stage_info["df"]

            print("\n" + "=" * 90)
            print(f"MODEL: {cfg['short_name']} | STAGE: {stage_name} | TRAIN SIZE: {len(df_train_stage)}")
            print("=" * 90)

            result = train_and_evaluate(
                df_train=df_train_stage,
                df_test=df_test,
                model_name=cfg["model_name"],
                lr=cfg["lr"],
                num_epochs=cfg["num_epochs"],
                batch_size=cfg["batch_size"],
                name=f"[{stage_name}] {cfg['short_name']}",
            )

            results.append({
                "model": cfg["short_name"],
                "model_name": cfg["model_name"],
                "stage": stage_name,
                "stage_num": stage_info["stage_num"],
                "train_size": len(df_train_stage),
                "balanced_accuracy": float(result["balanced_accuracy"]),
                "macro_f1": float(result["macro_f1"]),
            })

            pd.DataFrame(results).to_csv(partial_path, index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(results).sort_values(["model", "stage_num"]).reset_index(drop=True)
    df["delta_prev_balanced_accuracy"] = df.groupby("model")["balanced_accuracy"].diff()
    df["delta_prev_macro_f1"] = df.groupby("model")["macro_f1"].diff()

    eda_baseline = (
        df[df["stage"] == "after_eda"]
        .set_index("model")[["balanced_accuracy", "macro_f1"]]
    )
    df["delta_eda_balanced_accuracy"] = df.apply(
        lambda row: row["balanced_accuracy"] - eda_baseline.loc[row["model"], "balanced_accuracy"],
        axis=1,
    )
    df["delta_eda_macro_f1"] = df.apply(
        lambda row: row["macro_f1"] - eda_baseline.loc[row["model"], "macro_f1"],
        axis=1,
    )

    out_path = RESULTS_DIR / "rubert_stage_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path}")
    print(df.to_string(index=False))


def run_final_metrics(args: argparse.Namespace) -> None:
    if args.metrics == "none":
        print("Финальные метрики пропущены (--metrics none)")
        return
    if args.metrics in {"classical", "all"}:
        run_classical_metrics()
    if args.metrics in {"rubert", "all"}:
        run_rubert_stage_ablation(args.rubert_models, args.rubert_epochs)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    args = parse_args()
    config_path = resolve_config_path(args.config)

    from src.utils.pipeline_config import load_pipeline_config

    start_time = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_name": args.run_name,
        "started_at": start_time,
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "gpu": args.gpu,
        "metrics": args.metrics,
        "rubert_models": args.rubert_models,
        "rubert_epochs": args.rubert_epochs,
        "commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_manifest(RESULTS_DIR / "augmentation_manifest.json", manifest)

    section("AUGMENTATION PIPELINE")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    pipeline_cfg = load_pipeline_config(args.gpu)
    prepare_data()
    run_augmentation_stages(config_path, pipeline_cfg, args.stage1_repeat_limit)
    run_final_metrics(args)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(RESULTS_DIR / "augmentation_manifest.json", manifest)
    section("ГОТОВО")
    print(f"Manifest: {RESULTS_DIR / 'augmentation_manifest.json'}")


if __name__ == "__main__":
    main()
