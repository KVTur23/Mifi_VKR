"""
trainer_base.py — SeqClsRunner: общий оркестратор обучения

Сливает модельный JSON с GPU-профилем из pipeline_config.finetune[gpu],
готовит данные и адаптер, запускает HF Trainer, сохраняет артефакты.
"""

import json
import math
import time
from pathlib import Path


def _model_short(model_name: str) -> str:
    """'Qwen/Qwen3-14B' → 'qwen3_14b'."""
    tail = model_name.split("/")[-1]
    return tail.lower().replace("-", "_").replace(".", "_")


def _merge_gpu_profile(cfg: dict, pipeline_cfg) -> dict:
    """
    Переопределяет ключи модельного training_params и max_seq_length
    значениями из pipeline_cfg.finetune[gpu_name].

    Паттерн тот же, что у `load_llm` с gpu_memory_utilization/enforce_eager.
    Возвращает новую cfg (исходная не мутируется).
    """
    cfg = json.loads(json.dumps(cfg))  # глубокая копия через JSON (cfg уже json-like)

    gpu_name = pipeline_cfg["gpu_name"]
    profile = pipeline_cfg["finetune"][gpu_name]

    tp = cfg.get("training_params", {})
    tp["per_device_train_batch_size"] = profile["per_device_batch"]
    tp["per_device_eval_batch_size"] = max(1, profile["per_device_batch"] * 2)
    tp["gradient_accumulation_steps"] = profile["grad_accum"]
    tp["bf16"] = bool(profile["bf16"])
    tp["fp16"] = bool(profile["fp16"])
    cfg["training_params"] = tp

    cfg["max_seq_length"] = profile["max_seq"]
    return cfg


class SeqClsRunner:
    """
    Универсальный runner. Все отличия методов — в JSON-конфиге модели.
    """

    def __init__(self, config_path: str, pipeline_cfg):
        self.config_path = str(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        self.pipeline_cfg = pipeline_cfg
        self.cfg = _merge_gpu_profile(raw_cfg, pipeline_cfg)

        self.method = self.cfg["method"]
        self.model_name = self.cfg["model_name"]
        self.run_key = f"{self.method}_{_model_short(self.model_name)}"

        common = pipeline_cfg["finetune"]["common"]
        self.output_dir = Path(common["output_dir"]) / self.run_key
        self.early_stopping_patience = int(common.get("early_stopping_patience", 2))

        # Заполняются в prepare() / train()
        self.model = None
        self.tokenizer = None
        self.train_ds = None
        self.eval_ds = None
        self.collator = None
        self.label2id = None
        self.id2label = None
        self.class_groups = None
        self.trainable_params = None
        self.train_time_sec = None

    def prepare(self):
        from .data_prep import (
            load_finetune_data, build_label_mapping, compute_class_groups,
            encode_labels, tokenize_dataset, get_collator,
        )
        from .peft_utils import load_base_model, build_peft_config, wrap_with_peft

        df_train, df_test, orig_counts = load_finetune_data()

        self.label2id, self.id2label = build_label_mapping(df_train)
        n_classes = len(self.label2id)
        expected = int(self.cfg["num_labels"])
        if n_classes != expected:
            raise ValueError(
                f"Число классов в train ({n_classes}) != num_labels в конфиге ({expected})"
            )

        self.class_groups = compute_class_groups(orig_counts, self.label2id)

        df_train = encode_labels(df_train, self.label2id)
        df_test = encode_labels(df_test, self.label2id)

        self.model, self.tokenizer = load_base_model(
            self.cfg, self.pipeline_cfg,
            num_labels=n_classes,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        max_seq = int(self.cfg["max_seq_length"])
        self.train_ds = tokenize_dataset(df_train, self.tokenizer, max_seq)
        self.eval_ds = tokenize_dataset(df_test, self.tokenizer, max_seq)
        self.collator = get_collator(self.tokenizer)

        tp = self.cfg["training_params"]
        effective_batch = int(tp["per_device_train_batch_size"]) * int(tp["gradient_accumulation_steps"])
        total_step = math.ceil(len(self.train_ds) / max(1, effective_batch)) * int(tp["num_train_epochs"])

        peft_config = build_peft_config(self.cfg, total_step=total_step)
        is_quantized = bool(self.cfg.get("quantization"))
        self.model = wrap_with_peft(self.model, peft_config, is_quantized)

        self.trainable_params = int(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )

    def train(self):
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

        from .peft_utils import save_adapter

        tp = dict(self.cfg["training_params"])

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            report_to="none",
            **tp,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "balanced_accuracy": balanced_accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            }

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            processing_class=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience
            )],
        )

        t0 = time.time()
        trainer.train()
        self.train_time_sec = float(time.time() - t0)

        save_adapter(
            self.model, str(self.output_dir), self.tokenizer,
            self.id2label, self.class_groups,
        )

        meta = {
            "method": self.method,
            "model_name": self.model_name,
            "run_key": self.run_key,
            "trainable_params": self.trainable_params,
            "train_time_sec": self.train_time_sec,
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def run(self):
        import gc
        import torch

        from .evaluate_finetuned import evaluate

        self.prepare()
        self.train()

        # Не перезагружаем базовую модель для eval — это давало OOM/hang на
        # 32B QLoRA и 12B LoRA. load_best_model_at_end=True гарантирует, что
        # self.model уже содержит лучший чекпоинт.
        self.train_ds = None
        self.eval_ds = None
        self.collator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return evaluate(
            adapter_dir=str(self.output_dir),
            config_path=self.config_path,
            pipeline_cfg=self.pipeline_cfg,
            run_key=self.run_key,
            model=self.model,
            tokenizer=self.tokenizer,
        )
