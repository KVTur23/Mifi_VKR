"""
run_finetune.py — CLI-обёртка над orchestrator.run_finetune

Запуск:
    python scripts/run_finetune.py                                # все 4 метода на A100_40
    python scripts/run_finetune.py --methods lora qlora
    python scripts/run_finetune.py --methods tinylora --gpu A100_80
    python scripts/run_finetune.py --force --gpu H100             # пересчёт всех
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.orchestrator import run_finetune, METHOD_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="PEFT файнтюн (LoRA/QLoRA/AdaLoRA/TinyLoRA) × разные base-модели")
    parser.add_argument(
        "--methods", nargs="+",
        choices=list(METHOD_CONFIGS.keys()),
        default=None,
        help="Композитные ключи (method+model) из METHOD_CONFIGS. По умолчанию — все.",
    )
    parser.add_argument(
        "--gpu", default="A100_40",
        choices=["T4", "L4", "A100_40", "A100_80", "H100"],
        help="GPU-профиль из pipeline_config.json.finetune.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Пересчитать методы, уже присутствующие в results/finetune_results.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_finetune(methods=args.methods, gpu=args.gpu, force=args.force)
    if not df.empty:
        print()
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
