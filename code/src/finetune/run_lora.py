"""
run_lora.py — Запуск файнтюна Qwen3-14B + LoRA (SEQ_CLS)

Тонкая обёртка над SeqClsRunner. Вся специфика метода — в JSON-конфиге.
"""

from pathlib import Path

from .trainer_base import SeqClsRunner


PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(
    PROJECT_ROOT / "config_models" / "finetune_configs" / "qwen3_14b_lora.json"
)


def run(config_path: str = CONFIG_PATH, pipeline_cfg=None):
    if pipeline_cfg is None:
        raise ValueError(
            "pipeline_cfg обязателен (собирается orchestrator.run_finetune). "
            "Ожидается dict: {'gpu_name': <profile>, 'finetune': <секция из pipeline_config.json>}"
        )
    runner = SeqClsRunner(config_path, pipeline_cfg)
    return runner.run()
