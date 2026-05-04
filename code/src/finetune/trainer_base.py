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
        self.run_key = self.cfg.get("run_key") or f"{self.method}_{_model_short(self.model_name)}"

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
        self.df_test = None  # сохраняется в prepare() для опционального TestEvalCallback

    def prepare(self):
        from .data_prep import (
            load_finetune_data, build_label_mapping, compute_class_groups,
            compute_class_weights, encode_labels, tokenize_dataset, get_collator,
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

        # Веса классов для CrossEntropy (по умолчанию ON, выключается в JSON)
        self.use_class_weights = bool(self.cfg.get("use_class_weights", True))
        if self.use_class_weights:
            import torch
            weights_np = compute_class_weights(df_train, self.label2id)
            self.class_weights = torch.tensor(weights_np, dtype=torch.float32)
            print(f"[ClassWeights] enabled. range=[{weights_np.min():.3f}, {weights_np.max():.3f}], "
                  f"mean={weights_np.mean():.3f}")
        else:
            self.class_weights = None

        df_train = encode_labels(df_train, self.label2id)
        df_test = encode_labels(df_test, self.label2id)

        # Сохраняем df_test для опционального TestEvalCallback (per-epoch test eval).
        # Тестовый сет НЕ участвует ни в обучении, ни в выборе чекпоинта,
        # callback пишет метрики только в test_curve.csv для диагностики.
        self.df_test = df_test

        # Чтобы test не участвовал в выборе чекпоинта (selection bias),
        # отрезаем val из аугментированного train. Stratified — чтобы все 36 классов
        # были в val. Доля val настраивается через JSON: "val_split": 0.10 (default 0.10).
        from sklearn.model_selection import train_test_split
        val_split = float(self.cfg.get("val_split", 0.10))
        if val_split > 0:
            df_train, df_val = train_test_split(
                df_train,
                test_size=val_split,
                stratify=df_train["label_id"],
                random_state=42,
            )
            print(f"[Split] train={len(df_train)} | val={len(df_val)} (val_split={val_split}) | "
                  f"test={len(df_test)} (held out)")
        else:
            df_val = df_test
            print(f"[Split] val_split=0 → используется df_test как val (legacy, имеет selection bias)")

        self.model, self.tokenizer = load_base_model(
            self.cfg, self.pipeline_cfg,
            num_labels=n_classes,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        max_seq = int(self.cfg["max_seq_length"])
        self.train_ds = tokenize_dataset(df_train, self.tokenizer, max_seq)
        self.eval_ds = tokenize_dataset(df_val, self.tokenizer, max_seq)
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
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import pandas as pd
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

        from .peft_utils import save_adapter
        from .data_prep import class_groups_to_array, TEXT_COL, LABEL_COL

        tp = dict(self.cfg["training_params"])
        # Для PEFT-результата нужен адаптер, а не optimizer.pt/scheduler.pt.
        # На Colab/Drive optimizer checkpoint для 32B легко бьётся с
        # `inline_container.cc unexpected pos` из-за квоты/iostream errors.
        tp.setdefault("save_only_model", True)

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

        # ===== Новые конфигурируемые лоссы =====
        loss_type = str(self.cfg.get("loss", "ce")).lower()  # "ce" | "focal"
        focal_gamma = float(self.cfg.get("focal_gamma", 2.0))
        use_hierarchy = bool(self.cfg.get("use_hierarchy", False))
        hierarchy_lambda = float(self.cfg.get("hierarchy_lambda", 0.3))

        # class_to_group тензор для hierarchy-регуляризатора
        class_to_group = None
        if use_hierarchy:
            ctg_arr = class_groups_to_array(self.class_groups)
            class_to_group = torch.tensor(ctg_arr, dtype=torch.long)
            print(f"[Hierarchy] enabled, lambda={hierarchy_lambda}, "
                  f"groups distribution: A={int((ctg_arr==0).sum())}, "
                  f"B={int((ctg_arr==1).sum())}, C={int((ctg_arr==2).sum())}")

        if loss_type == "focal":
            print(f"[Loss] Focal (gamma={focal_gamma})")
        else:
            print(f"[Loss] CrossEntropy")

        # ===== Per-epoch test eval callback =====
        eval_test_each_epoch = bool(self.cfg.get("eval_test_each_epoch", False))

        class TestEvalCallback(TrainerCallback):
            """После каждой эпохи прогоняет test и пишет метрики в test_curve.csv.

            НЕ влияет на выбор best checkpoint — `metric_for_best_model` остаётся
            `eval_macro_f1` (val). Эта callback нужна только для диагностики:
            видеть реальную test-кривую по эпохам и понимать, где настоящий пик.

            Стоимость: ~2-3 мин на эпоху (test ≈ 341 пример).
            """
            def __init__(self, df_test, label2id, class_groups,
                         output_path, max_seq_length, batch_size):
                super().__init__()
                self.df_test = df_test
                self.label2id = label2id
                self.class_groups = class_groups
                self.output_path = output_path
                self.max_seq_length = max_seq_length
                self.batch_size = batch_size
                self.history = []

            def on_epoch_end(self, args, state, control, **kwargs):
                from .evaluate_finetuned import predict, _f1_per_group

                model = kwargs.get("model")
                tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
                if model is None or tokenizer is None:
                    return

                was_training = model.training
                model.eval()
                try:
                    texts = self.df_test[TEXT_COL].tolist()
                    y_pred = predict(
                        model, tokenizer, texts,
                        batch_size=self.batch_size,
                        max_seq_length=self.max_seq_length,
                    )
                    y_true = self.df_test["label_id"].astype(int).to_numpy()

                    bal_acc = balanced_accuracy_score(y_true, y_pred)
                    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                    per_group = _f1_per_group(y_true, y_pred, self.class_groups)

                    entry = {
                        "epoch": float(state.epoch),
                        "global_step": int(state.global_step),
                        "test_balanced_accuracy": bal_acc,
                        "test_macro_f1": macro_f1,
                        "test_f1_group_A": per_group["f1_group_A"],
                        "test_f1_group_B": per_group["f1_group_B"],
                        "test_f1_group_C": per_group["f1_group_C"],
                    }
                    self.history.append(entry)

                    pd.DataFrame(self.history).to_csv(self.output_path, index=False)

                    def _fmt(v):
                        return f"{v:.4f}" if v is not None else "—"

                    print(f"[TestEval] E{state.epoch:.1f}: "
                          f"bal_acc={bal_acc:.4f}, macro_f1={macro_f1:.4f}, "
                          f"f1_A={_fmt(per_group['f1_group_A'])}, "
                          f"f1_B={_fmt(per_group['f1_group_B'])}, "
                          f"f1_C={_fmt(per_group['f1_group_C'])}")
                finally:
                    if was_training:
                        model.train()

        if eval_test_each_epoch:
            print(f"[TestEval] enabled — добавляет ~2-3 мин на эпоху, "
                  f"пишет в {self.output_dir}/test_curve.csv")

        class ConfigurableLossTrainer(Trainer):
            """Trainer с настраиваемым лоссом:
            - loss_type='ce' | 'focal'
            - class_weights — sklearn-balanced (опционально)
            - hierarchy regularizer — штраф за вероятностную массу классов из
              "неправильной" группы (A/B/C). Идея: модель не должна путать группы.

            Все веса/маски лениво переносятся на device логитов на forward.
            """
            def __init__(self, *targs,
                         loss_type="ce",
                         focal_gamma=2.0,
                         class_weights=None,
                         use_hierarchy=False,
                         hierarchy_lambda=0.3,
                         class_to_group=None,
                         **tkwargs):
                super().__init__(*targs, **tkwargs)
                self._loss_type = loss_type
                self._focal_gamma = focal_gamma
                self._class_weights = class_weights
                self._use_hierarchy = use_hierarchy
                self._hierarchy_lambda = hierarchy_lambda
                self._class_to_group = class_to_group

            def _focal_loss(self, logits, labels, weight):
                """Focal loss = -(1-pt)^gamma * log(pt) * alpha_t.

                Совместима с class_weights (alpha_t берётся из weight[label]).
                """
                log_probs = F.log_softmax(logits, dim=-1)
                log_pt = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                pt = log_pt.exp()
                focal_term = (1 - pt) ** self._focal_gamma
                loss = -focal_term * log_pt
                if weight is not None:
                    loss = loss * weight[labels]
                return loss.mean()

            def _hierarchy_penalty(self, logits, labels):
                """Сумма softmax-вероятностей классов, чья группа != группа label.

                Для каждого примера: считаем prob mass на классы из других групп
                и штрафуем. Заставляет модель сначала научиться различать
                группы A/B/C, и только потом — конкретный класс внутри группы.
                """
                ctg = self._class_to_group.to(logits.device)
                probs = F.softmax(logits, dim=-1)
                sample_groups = ctg[labels]                              # (B,)
                wrong_group_mask = ctg.unsqueeze(0) != sample_groups.unsqueeze(1)  # (B, C)
                wrong_prob = (probs * wrong_group_mask.to(probs.dtype)).sum(dim=-1)
                return wrong_prob.mean()

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                weight = None
                if self._class_weights is not None:
                    weight = self._class_weights.to(logits.device, dtype=logits.dtype)

                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)

                if self._loss_type == "focal":
                    loss = self._focal_loss(flat_logits, flat_labels, weight)
                else:
                    loss = nn.CrossEntropyLoss(weight=weight)(flat_logits, flat_labels)

                if self._use_hierarchy and self._class_to_group is not None:
                    h = self._hierarchy_penalty(flat_logits, flat_labels)
                    loss = loss + self._hierarchy_lambda * h

                return (loss, outputs) if return_outputs else loss

        # Сборка callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience
            ),
        ]
        if eval_test_each_epoch and self.df_test is not None:
            callbacks.append(TestEvalCallback(
                df_test=self.df_test,
                label2id=self.label2id,
                class_groups=self.class_groups,
                output_path=self.output_dir / "test_curve.csv",
                max_seq_length=int(self.cfg["max_seq_length"]),
                batch_size=max(1, int(tp.get("per_device_eval_batch_size", 4))),
            ))

        # Решаем, нужен ли кастомный Trainer.
        # Если ничего из новых опций не включено и class_weights выкл — стандартный.
        needs_custom = (
            self.use_class_weights
            or loss_type == "focal"
            or use_hierarchy
        )

        if needs_custom:
            trainer = ConfigurableLossTrainer(
                model=self.model,
                args=args,
                train_dataset=self.train_ds,
                eval_dataset=self.eval_ds,
                processing_class=self.tokenizer,
                data_collator=self.collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                class_weights=self.class_weights if self.use_class_weights else None,
                use_hierarchy=use_hierarchy,
                hierarchy_lambda=hierarchy_lambda,
                class_to_group=class_to_group,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=self.train_ds,
                eval_dataset=self.eval_ds,
                processing_class=self.tokenizer,
                data_collator=self.collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )

        t0 = time.time()
        recovered_from_save_error = False
        try:
            trainer.train()
        except RuntimeError as e:
            msg = str(e)
            is_checkpoint_io_error = (
                "inline_container.cc" in msg
                or "iostream error" in msg
                or "unexpected pos" in msg
            )
            if not is_checkpoint_io_error:
                raise
            recovered_from_save_error = True
            print("[CheckpointSaveError] Trainer упал на записи checkpoint-а. "
                  "Сохраняю текущий PEFT-адаптер из памяти и продолжаю eval.")
        finally:
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
            "use_class_weights": bool(self.use_class_weights),
            "loss": loss_type,
            "focal_gamma": focal_gamma if loss_type == "focal" else None,
            "use_hierarchy": use_hierarchy,
            "hierarchy_lambda": hierarchy_lambda if use_hierarchy else None,
            "eval_test_each_epoch": eval_test_each_epoch,
            "recovered_from_checkpoint_save_error": recovered_from_save_error,
        }
        if self.use_class_weights and self.class_weights is not None:
            meta["class_weights_stats"] = {
                "min":  float(self.class_weights.min()),
                "max":  float(self.class_weights.max()),
                "mean": float(self.class_weights.mean()),
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
