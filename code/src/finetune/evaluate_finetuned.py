"""
evaluate_finetuned.py - Инференс адаптера и расчёт метрик

Грузит базовую модель + PEFT-адаптер, прогоняет test, пишет per-sample preds
и сохраняет метрики в results/finetune/<run_key>.json (один файл на прогон).
Сводную таблицу собирает notebooks/aggregate_results.ipynb.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_test_set, TEXT_COL, LABEL_COL


RESULTS_DIR = PROJECT_ROOT / "results" / "finetune"


def _save_result_json(run_key: str, row: dict) -> Path:
    """Один прогон - один файл results/finetune/<run_key>.json. Перезапись свежими данными."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    return path


def load_finetuned_model(adapter_dir: str, base_model_name: str,
                         quantization_cfg: dict | None,
                         num_labels: int, id2label: dict, label2id: dict,
                         pipeline_cfg=None):
    """
    Грузит базовую модель (опционально 4bit) и поверх неё PEFT-адаптер.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    from src.finetune.peft_utils import _prepare_cuda_for_large_load, _quantized_load_controls

    load_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
    }

    if quantization_cfg:
        _prepare_cuda_for_large_load(torch)
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, quantization_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_kwargs = {
            "load_in_4bit": quantization_cfg.get("load_in_4bit", True),
            "bnb_4bit_quant_type": quantization_cfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": quantization_cfg.get("bnb_4bit_use_double_quant", True),
        }
        if quantization_cfg.get("bnb_4bit_quant_storage"):
            bnb_kwargs["bnb_4bit_quant_storage"] = getattr(torch, quantization_cfg["bnb_4bit_quant_storage"])
        load_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
        load_kwargs["device_map"] = "auto"
        load_kwargs.update(_quantized_load_controls(pipeline_cfg))

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(base_model_name, **load_kwargs)
    base.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, texts: list[str],
            batch_size: int, max_seq_length: int) -> np.ndarray:
    """Батчевый инференс. Возвращает numpy-массив предсказанных id."""
    import torch

    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_seq_length,
                padding="longest",
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(preds) if preds else np.array([], dtype=int)


def _f1_per_group(y_true: np.ndarray, y_pred: np.ndarray,
                  class_groups: dict[int, str]) -> dict:
    """F1 macro в разрезе групп A/B/C. None если в тесте нет классов группы."""
    from sklearn.metrics import f1_score

    buckets = {"A": [], "B": [], "C": []}
    for cid, g in class_groups.items():
        buckets[g].append(int(cid))

    out = {}
    for g, ids in buckets.items():
        mask = np.isin(y_true, ids)
        if mask.sum() == 0:
            out[f"f1_group_{g}"] = None
            continue
        out[f"f1_group_{g}"] = f1_score(
            y_true[mask], y_pred[mask],
            average="macro", labels=ids, zero_division=0,
        )
    return out


def evaluate(adapter_dir: str, config_path: str, pipeline_cfg,
             run_key: str,
             model=None, tokenizer=None) -> dict:
    """
    Инференс на test + метрики + per-group F1 + classification_report.

    Артефакты:
      - results/preds_<run_key>.csv          - per-sample предсказания
      - results/finetune/<run_key>.json      - метрики этого прогона

    Если model/tokenizer переданы — используем их (in-memory, после train()).
    Иначе грузим из adapter_dir. Перезагрузка экономит память и исключает OOM-hang.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

    adapter_dir = Path(adapter_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(adapter_dir / "id2label.json", "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
    id2label = {int(k): v for k, v in id2label_raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    with open(adapter_dir / "class_groups.json", "r", encoding="utf-8") as f:
        class_groups_raw = json.load(f)
    class_groups = {int(k): v for k, v in class_groups_raw.items()}

    df_test = load_test_set()

    if model is None or tokenizer is None:
        model, tokenizer = load_finetuned_model(
            adapter_dir=str(adapter_dir),
            base_model_name=cfg["model_name"],
            quantization_cfg=cfg.get("quantization"),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            pipeline_cfg=pipeline_cfg,
        )
    model.eval()

    gpu_name = pipeline_cfg["gpu_name"]
    profile = pipeline_cfg["finetune"][gpu_name]
    max_seq_length = int(profile["max_seq"])
    eval_batch = max(1, int(profile["per_device_batch"]) * 2)

    texts = df_test[TEXT_COL].tolist()
    y_pred = predict(model, tokenizer, texts, batch_size=eval_batch,
                     max_seq_length=max_seq_length)
    y_true = df_test[LABEL_COL].map(label2id).astype(int).to_numpy()

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_group = _f1_per_group(y_true, y_pred, class_groups)

    label_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        y_true, y_pred,
        labels=sorted(id2label.keys()),
        target_names=label_names,
        zero_division=0,
    )

    # preds_<run_key>.csv
    common = pipeline_cfg["finetune"]["common"]
    preds_dir = Path(common["preds_dir"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame({
        "text": df_test[TEXT_COL].values,
        "true_label": df_test[LABEL_COL].values,
        "predicted_label": [id2label[int(p)] for p in y_pred],
    })
    preds_df["correct"] = (preds_df["true_label"] == preds_df["predicted_label"]).astype(int)
    preds_df.to_csv(preds_dir / f"preds_{run_key}.csv", index=False)

    # metadata.json (может не быть, если evaluate вызван отдельно)
    meta_path = adapter_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    row = {
        "run_key": run_key,
        "method": cfg["method"],
        "model": cfg["model_name"],
        "balanced_accuracy": bal_acc,
        "macro_f1": f1_mac,
        "f1_group_A": per_group["f1_group_A"],
        "f1_group_B": per_group["f1_group_B"],
        "f1_group_C": per_group["f1_group_C"],
        "trainable_params": meta.get("trainable_params"),
        "train_time_sec": meta.get("train_time_sec"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    saved_path = _save_result_json(run_key, row)
    print(f"[Eval] Метрики сохранены: {saved_path.relative_to(PROJECT_ROOT)}")

    print("=" * 60)
    print(f"FINETUNE EVAL: {run_key}")
    print("=" * 60)
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {f1_mac:.4f}")
    for g in ("A", "B", "C"):
        v = per_group[f"f1_group_{g}"]
        print(f"  F1 Group {g}:        {v:.4f}" if v is not None else f"  F1 Group {g}:        N/A")
    print(f"\n{report}")

    result = dict(row)
    result["classification_report"] = report
    return result
