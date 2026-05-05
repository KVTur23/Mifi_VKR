#!/usr/bin/env python3
"""
Run one server finetune experiment.

This is the script version of notebooks/finetune.ipynb for Qwen3-32B QLoRA:
it builds a runtime config, validates the expected Data files, runs training,
and writes config/manifest artifacts next to the results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_CONFIG = PROJECT_ROOT / "config_models" / "finetune_configs" / "qwen3_32b_qlora.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNTIME_CONFIG_DIR = PROJECT_ROOT / "Data" / "finetune_runtime_configs"


EXPERIMENTS = {
    "cw": {
        "description": "class weights, base LoRA rank",
        "overrides": {},
    },
    "cw_r32": {
        "description": "class weights, LoRA r=32 alpha=64",
        "overrides": {
            "peft_config": {
                "r": 32,
                "lora_alpha": 64,
            },
        },
    },
    "cw_focal_g2": {
        "description": "class weights, focal loss gamma=2.0",
        "overrides": {
            "loss": "focal",
            "focal_gamma": 2.0,
        },
    },
    "no_cw_r32": {
        "description": "no class weights, LoRA r=32 alpha=64",
        "overrides": {
            "use_class_weights": False,
            "peft_config": {
                "r": 32,
                "lora_alpha": 64,
            },
        },
    },
    "cw_r32_hier_l03": {
        "description": "class weights, LoRA r=32 alpha=64, hierarchy regularizer lambda=0.3",
        "overrides": {
            "use_class_weights": True,
            "val_split": 0.10,
            "use_hierarchy": True,
            "hierarchy_lambda": 0.3,
            "peft_config": {
                "r": 32,
                "lora_alpha": 64,
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen3-32B QLoRA finetune experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--gpu",
        default="A100_40",
        choices=["T4", "L4", "A100_40", "A100_80", "H100"],
    )
    return parser.parse_args()


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


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


def validate_data() -> dict:
    data_dir = PROJECT_ROOT / "Data"
    required = [
        data_dir / "data_after_stage3.csv",
        data_dir / "data_test.csv",
        data_dir / "train_after_eda.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Finetune requires stage3 data. Missing files:\n  " + "\n  ".join(missing)
        )

    stats = {}
    for path in required:
        df = pd.read_csv(path)
        stats[path.name] = {
            "rows": int(len(df)),
            "labels": int(df["label"].nunique()) if "label" in df.columns else None,
        }
    return stats


def build_runtime_config(run_name: str, experiment: str) -> tuple[dict, Path]:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["run_key"] = run_name
    cfg["use_class_weights"] = True
    cfg["val_split"] = 0.10
    cfg["eval_test_each_epoch"] = True
    cfg.setdefault("loss", "ce")

    deep_update(cfg, EXPERIMENTS[experiment]["overrides"])

    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RUNTIME_CONFIG_DIR / f"{run_name}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return cfg, config_path


def write_manifest(manifest: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    data_stats = validate_data()
    cfg, config_path = build_runtime_config(args.run_name, args.experiment)

    manifest = {
        "run_name": args.run_name,
        "experiment": args.experiment,
        "description": EXPERIMENTS[args.experiment]["description"],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "gpu": args.gpu,
        "commit": git_commit(),
        "data_stats": data_stats,
        "config_path": str(config_path),
        "config": cfg,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_manifest(manifest)

    print("=" * 80)
    print("FINETUNE EXPERIMENT")
    print("=" * 80)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    from src.finetune.orchestrator import _load_pipeline_cfg
    from src.finetune.run_qlora import run as run_qlora

    pipeline_cfg = _load_pipeline_cfg(args.gpu)
    result = run_qlora(config_path=str(config_path), pipeline_cfg=pipeline_cfg)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["metrics"] = {
        key: value for key, value in result.items()
        if key != "classification_report"
    }
    write_manifest(manifest)

    print("=" * 80)
    print("FINETUNE DONE")
    print("=" * 80)
    print(json.dumps(manifest["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
