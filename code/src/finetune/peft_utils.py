"""
peft_utils.py — Загрузка базовой модели и применение PEFT-адаптера

Фабрика PEFT-конфигов для LoRA / QLoRA / AdaLoRA / TinyLoRA.
Всё — из JSON-конфига модели, без хардкода.
"""

import json
from pathlib import Path


def load_base_model(cfg: dict, pipeline_cfg, num_labels: int,
                    id2label: dict, label2id: dict):
    """
    Загружает AutoModelForSequenceClassification + токенизатор.

    - Если cfg.quantization не None → BitsAndBytesConfig.
    - padding_side="left" (Qwen3 + classification head: пулинг по последнему non-pad токену).
    - Если у токенизатора нет pad_token → pad_token = eos_token.
    - model.config.pad_token_id синхронизируется с токенизатором.
    - gradient_checkpointing включается, если задан в training_params.

    Возвращает (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
    }

    qcfg = cfg.get("quantization")
    tp = cfg.get("training_params", {})

    if qcfg:
        # QLoRA: bnb сам управляет dtype через bnb_4bit_compute_dtype.
        # НЕ передаём torch_dtype — иначе веса грузятся в bf16 на GPU перед
        # квантизацией, пиковое потребление 2× (для 32B на L4 = OOM).
        from transformers import BitsAndBytesConfig
        compute_dtype_str = qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str)
        bnb = BitsAndBytesConfig(
            load_in_4bit=qcfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
        )
        load_kwargs["quantization_config"] = bnb
        load_kwargs["device_map"] = "auto"  # пошаговая загрузка по слоям, без пика
    else:
        # bf16/fp16 для нерубленых LoRA-моделей
        if tp.get("bf16"):
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif tp.get("fp16"):
            load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    if tp.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_peft_config(cfg: dict, total_step: int | None = None):
    """
    Фабрика. По cfg.method возвращает нужный PEFT-конфиг с task_type=SEQ_CLS.

    - lora / qlora  → LoraConfig(**peft_config)
    - adalora       → AdaLoraConfig(**peft_config), total_step подставляется динамически
    - tinylora      → TinyLoraConfig с выборочной передачей полей
    """
    from peft import TaskType

    method = cfg["method"]
    pc = cfg["peft_config"]

    if method in ("lora", "qlora"):
        from peft import LoraConfig
        return LoraConfig(task_type=TaskType.SEQ_CLS, **pc)

    if method == "adalora":
        from peft import AdaLoraConfig
        pc = dict(pc)
        if pc.get("total_step") is None:
            if total_step is None:
                raise ValueError("AdaLoRA: total_step required")
            pc["total_step"] = total_step
        return AdaLoraConfig(task_type=TaskType.SEQ_CLS, **pc)

    if method == "tinylora":
        from peft import TinyLoraConfig
        return TinyLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=pc["r"],
            u=pc["u"],
            weight_tying=pc.get("weight_tying", 0.0),
            target_modules=pc["target_modules"],
            projection_seed=pc.get("projection_seed", 42),
            modules_to_save=pc.get("modules_to_save"),
        )

    raise ValueError(f"Unknown method: {method}")


def wrap_with_peft(model, peft_config, is_quantized: bool):
    """
    Оборачивает модель PEFT-адаптером.

    - quantized → prepare_model_for_kbit_training (сам делает enable_input_require_grads).
    - обычный   → enable_input_require_grads вручную (нужно для gradient_checkpointing + LoRA).

    Проверяет, что классификационная голова 'score' попала в trainable —
    иначе голова останется случайной.
    """
    from peft import get_peft_model, prepare_model_for_kbit_training

    if is_quantized:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not any("score" in n for n in trainable):
        raise RuntimeError(
            "Classification head 'score' не попала в trainable params. "
            "Проверь modules_to_save в peft_config или имя головы в base model."
        )

    return model


def save_adapter(model, output_dir, tokenizer, id2label: dict, class_groups: dict):
    """
    Сохраняет: адаптер, токенизатор, id2label.json, class_groups.json.
    JSON-ключи приводятся к строкам (json не умеет int-ключи).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f,
                  ensure_ascii=False, indent=2)

    with open(output_dir / "class_groups.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in class_groups.items()}, f,
                  ensure_ascii=False, indent=2)
