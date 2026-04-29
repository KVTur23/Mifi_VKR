"""
orchestrator.py — Общее ядро запуска файнтюна

`run_finetune` — единая точка для CLI (scripts/run_finetune.py) и ноутбука
(notebooks/finetune.ipynb). Вся логика оркестрации — здесь, обёртки дублировать
её не должны.

Отвечает за:
- Сборку pipeline_cfg из pipeline_config.json под выбранный GPU-профиль.
- Идемпотентность: пропускает методы, чей run_key уже есть в
  results/finetune_results.csv (если force=False).
- Последовательный вызов run_<method>.run(pipeline_cfg=...) c чисткой GPU
  между методами и логированием OOM.
- Обновление results/all_methods_comparison.csv блоком `finetune`.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch  # ранний импорт: лечит circular import torch.fx на py3.13 + torch 2.10
import torch.fx  # noqa: F401

import gc
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from .trainer_base import _model_short


PROJECT_ROOT = Path(__file__).parent.parent.parent
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config_models" / "pipeline_config.json"
FINETUNE_CONFIGS_DIR = PROJECT_ROOT / "config_models" / "finetune_configs"


METHOD_CONFIGS = {
    # Исходные Qwen3
    "lora_qwen3_14b":      FINETUNE_CONFIGS_DIR / "qwen3_14b_lora.json",
    "qlora_qwen3_32b":     FINETUNE_CONFIGS_DIR / "qwen3_32b_qlora.json",
    "adalora_qwen3_32b":   FINETUNE_CONFIGS_DIR / "qwen3_32b_adalora.json",
    "tinylora_qwen3_14b":  FINETUNE_CONFIGS_DIR / "qwen3_14b_tinylora.json",
    # Новые base-модели
    "qlora_tpro_it_21":    FINETUNE_CONFIGS_DIR / "tpro_it_21_qlora.json",
    "adalora_tpro_it_21":  FINETUNE_CONFIGS_DIR / "tpro_it_21_adalora.json",
    "lora_vikhr_nemo_12b": FINETUNE_CONFIGS_DIR / "vikhr_nemo_12b_lora.json",
    "qlora_qwen25_32b":    FINETUNE_CONFIGS_DIR / "qwen25_32b_qlora.json",
    "adalora_qwen25_32b":  FINETUNE_CONFIGS_DIR / "qwen25_32b_adalora.json",
}


def _load_pipeline_cfg(gpu: str) -> dict:
    """Читает pipeline_config.json, валидирует GPU-профиль."""
    with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "finetune" not in raw:
        raise ValueError("pipeline_config.json не содержит секции 'finetune'")
    if gpu not in raw["finetune"] or gpu == "common":
        available = [k for k in raw["finetune"].keys() if k != "common"]
        raise ValueError(f"GPU-профиль '{gpu}' не найден в finetune. Доступные: {available}")

    return {"gpu_name": gpu, "finetune": raw["finetune"]}


def _derive_run_key(config_path: Path) -> str:
    """Читает модельный JSON и строит run_key тем же способом, что и SeqClsRunner."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return f"{cfg['method']}_{_model_short(cfg['model_name'])}"


def _gpu_cleanup():
    """Освобождаем VRAM между методами (аналог prompt_classifier.unload_model)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    time.sleep(5)


def _update_all_methods_comparison(results_dir: Path):
    """
    Обновляет results/all_methods_comparison.csv блоком `finetune`
    из results/finetune_results.csv.

    Схема сводной таблицы (совместима с существующими блоками):
        method | model | setting | balanced_accuracy | macro_f1 | unknown_rate
    """
    finetune_csv = results_dir / "finetune_results.csv"
    comparison_csv = results_dir / "all_methods_comparison.csv"

    from filelock import FileLock

    if not finetune_csv.exists():
        return

    # FileLock — параллельные SLURM-джобы могут одновременно обновлять сводку.
    lock = FileLock(str(comparison_csv) + ".lock")
    with lock:
        df_ft = pd.read_csv(finetune_csv)
        ft_rows = [
            {
                "method": "finetune",
                "model": r["model"],
                "setting": r["method"],
                "balanced_accuracy": r["balanced_accuracy"],
                "macro_f1": r["macro_f1"],
                "unknown_rate": 0.0,
            }
            for _, r in df_ft.iterrows()
        ]
        df_ft_block = pd.DataFrame(ft_rows)

        if comparison_csv.exists():
            df_existing = pd.read_csv(comparison_csv)
            df_existing = df_existing[df_existing["method"] != "finetune"]
            df_out = pd.concat([df_existing, df_ft_block], ignore_index=True)
        else:
            df_out = df_ft_block

        df_out.to_csv(comparison_csv, index=False)
        print(f"[Orchestrator] Обновлено: {comparison_csv.relative_to(PROJECT_ROOT)}")


def run_finetune(methods: list[str] | None = None,
                 gpu: str = "A100_40",
                 force: bool = False) -> pd.DataFrame:
    """
    Запускает файнтюн выбранных PEFT-методов.

    Аргументы:
        methods: подмножество ключей METHOD_CONFIGS (composite method+model).
                 None → все доступные конфиги.
        gpu: GPU-профиль из pipeline_config.json.finetune (T4/L4/A100_40/A100_80/H100).
        force: если True — пересчитывает методы, даже если run_key уже в CSV.

    Возвращает:
        pd.DataFrame — актуальное содержимое results/finetune_results.csv.
    """
    if methods is None:
        methods = list(METHOD_CONFIGS.keys())

    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"Неизвестные методы: {unknown}. Доступные: {list(METHOD_CONFIGS.keys())}")

    pipeline_cfg = _load_pipeline_cfg(gpu)
    common = pipeline_cfg["finetune"]["common"]
    results_csv = Path(common["results_csv"])
    if not results_csv.is_absolute():
        results_csv = PROJECT_ROOT / results_csv
    results_dir = results_csv.parent

    done_keys = set()
    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        done_keys = set(df_existing["run_key"].astype(str).tolist())

    print(f"[Orchestrator] GPU профиль: {gpu}")
    print(f"[Orchestrator] Методы: {methods}")
    print(f"[Orchestrator] force={force}, уже готово: {sorted(done_keys) or '—'}")

    for method_key in methods:
        config_path = METHOD_CONFIGS[method_key]
        if not config_path.exists():
            print(f"[SKIP] {method_key}: конфиг не найден — {config_path}")
            continue

        # method (lora/qlora/adalora/tinylora) — из JSON, для импорта run_<method>.py
        with open(config_path, "r", encoding="utf-8") as f:
            method = json.load(f)["method"]

        run_key = _derive_run_key(config_path)

        if run_key in done_keys and not force:
            print(f"[SKIP] {method_key} ({run_key}): уже в {results_csv.name}, force=False")
            continue

        print()
        print("#" * 60)
        print(f"# {method_key}  (run_key={run_key})")
        print("#" * 60)

        try:
            _gpu_cleanup()
            from importlib import import_module
            mod = import_module(f"src.finetune.run_{method}")
            mod.run(config_path=str(config_path), pipeline_cfg=pipeline_cfg)
            done_keys.add(run_key)
        except Exception as e:
            is_oom = False
            try:
                import torch
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
            except ImportError:
                pass
            tag = "OOM" if is_oom else type(e).__name__
            print(f"[ERROR][{tag}] {method_key} упал: {e}")
            traceback.print_exc()
            print("[Orchestrator] Продолжаю со следующим методом...")
        finally:
            _gpu_cleanup()

    if results_csv.exists():
        df = pd.read_csv(results_csv)
    else:
        df = pd.DataFrame()

    _update_all_methods_comparison(results_dir)

    print()
    print("=" * 60)
    print("ИТОГО")
    print("=" * 60)
    if df.empty:
        print("(пусто — ни один метод не завершился успешно)")
    else:
        print(df.to_string(index=False))
    return df
