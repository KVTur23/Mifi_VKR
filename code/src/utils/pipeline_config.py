"""
pipeline_config.py — Единый конфиг пайплайна

Загружает pipeline_config.json и применяет GPU-профиль.
В ноутбуке достаточно:
    from src.utils.pipeline_config import load_pipeline_config
    cfg = load_pipeline_config("A100")
    # cfg.stage1.target_count, cfg.gpu.nllb_batch_size, ...
"""

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent.parent / "config_models" / "pipeline_config.json"


class _DotDict(dict):
    """Словарь с доступом через точку: cfg.stage1.target_count."""
    def __getattr__(self, key):
        val = self[key]
        if isinstance(val, dict):
            return _DotDict(val)
        return val

    def __setattr__(self, key, value):
        self[key] = value


_config_cache = {}


def load_pipeline_config(gpu: str = "L4") -> _DotDict:
    """
    Загружает конфиг пайплайна с выбранным GPU-профилем.

    Аргументы:
        gpu: название GPU — "T4", "L4", "A100", "H100"

    Возвращает:
        cfg с полями: cfg.gpu, cfg.stage1, cfg.stage2, cfg.stage3,
        cfg.judge, cfg.validation, cfg.training
    """
    cache_key = gpu
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gpu_upper = gpu.upper()
    profiles = raw["gpu_profiles"]
    if gpu_upper not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"GPU '{gpu}' не найден. Доступные: {available}")

    cfg = _DotDict({
        "gpu": profiles[gpu_upper],
        "stage1": raw["pipeline"]["stage1"],
        "stage2": raw["pipeline"]["stage2"],
        "stage3": raw["pipeline"]["stage3"],
        "judge": raw["judge"],
        "validation": raw["validation"],
        "training": raw["training"],
    })

    print(f"[Config] GPU: {gpu_upper} ({cfg.gpu.vram_gb}GB), "
          f"NLLB: {cfg.gpu.nllb_model}, "
          f"batch: {cfg.gpu.nllb_batch_size}, "
          f"chunk: {cfg.stage3.get('nllb_chunk_chars', 'default')}")

    _config_cache[cache_key] = cfg
    return cfg
