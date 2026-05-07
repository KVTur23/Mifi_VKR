#!/usr/bin/env python3
"""
run_finetune.py - Один прогон файнтюна.

Принимает на вход полный JSON-конфиг (метод + модель + параметры),
строит pipeline_cfg для выбранного GPU-профиля, запускает SeqClsRunner.
Результат пишется в results/finetune/<run_key>.json.

Пример:
    python scripts/run_finetune.py --config config_models/finetune_configs/qwen3_32b_qlora_cw.json
    python scripts/run_finetune.py --config qwen3_32b_qlora_cw.json --gpu A100_80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config_models" / "pipeline_config.json"
FINETUNE_CONFIGS_DIR = PROJECT_ROOT / "config_models" / "finetune_configs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Один прогон файнтюна.")
    p.add_argument(
        "--config",
        required=True,
        help="Путь до JSON-конфига. Можно абсолютный путь, project-relative "
             "или просто имя файла из config_models/finetune_configs/.",
    )
    p.add_argument(
        "--gpu",
        default="A100_40",
        choices=["T4", "L4", "A100_40", "A100_80", "H100"],
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Опционально - переопределить run_key из конфига.",
    )
    return p.parse_args()


def resolve_config(value: str) -> Path:
    """Принимает абсолютный/относительный путь или имя файла."""
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(PROJECT_ROOT / path)
        candidates.append(FINETUNE_CONFIGS_DIR / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Конфиг не найден. Проверял:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def load_pipeline_cfg(gpu: str) -> dict:
    """Читает pipeline_config.json и валидирует GPU-профиль."""
    with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "finetune" not in raw:
        raise ValueError("pipeline_config.json не содержит секции 'finetune'")
    if gpu not in raw["finetune"] or gpu == "common":
        available = [k for k in raw["finetune"].keys() if k != "common"]
        raise ValueError(f"GPU-профиль '{gpu}' не найден в finetune. Доступные: {available}")

    return {"gpu_name": gpu, "finetune": raw["finetune"]}


def main() -> None:
    # Лечит circular import torch.fx на py3.13 + torch 2.10
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    import torch  # noqa: F401
    import torch.fx  # noqa: F401

    args = parse_args()
    config_path = resolve_config(args.config)

    # Опциональный override run_key через CLI - удобно для SLURM-джоб,
    # когда один и тот же конфиг хочется запустить под разными именами
    if args.run_name:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["run_key"] = args.run_name
        runtime_dir = PROJECT_ROOT / "Data" / "finetune_runtime_configs"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        config_path = runtime_dir / f"{args.run_name}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    pipeline_cfg = load_pipeline_cfg(args.gpu)

    print("=" * 60)
    print(f"FINETUNE | gpu={args.gpu} | config={config_path.name}")
    print("=" * 60)

    from src.finetune.trainer_base import SeqClsRunner
    runner = SeqClsRunner(str(config_path), pipeline_cfg)
    runner.run()


if __name__ == "__main__":
    main()
