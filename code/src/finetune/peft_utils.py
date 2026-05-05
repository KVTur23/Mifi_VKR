"""
peft_utils.py — Загрузка базовой модели и применение PEFT-адаптера

Фабрика PEFT-конфигов для LoRA / QLoRA / AdaLoRA / TinyLoRA.
Всё — из JSON-конфига модели, без хардкода.
"""

import json
import gc
import os
from pathlib import Path


def _profile_vram_gb(pipeline_cfg) -> int | None:
    """Возвращает VRAM из имени профиля вида A100_40/A100_80/H100/L4/T4."""
    if not pipeline_cfg:
        return None

    gpu_name = str(pipeline_cfg.get("gpu_name", ""))
    if gpu_name.endswith("_40"):
        return 40
    if gpu_name.endswith("_80") or gpu_name == "H100":
        return 80
    if gpu_name == "L4":
        return 24
    if gpu_name == "T4":
        return 16
    return None


def _prepare_cuda_for_large_load(torch):
    """Минимизирует пик VRAM перед загрузкой большой quantized-модели."""
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Transformers 5.x грузит safetensors через небольшой ThreadPool. Для 32B
    # 4bit это может одновременно материализовать несколько bf16-тензоров на
    # GPU перед bnb-квантизацией и добить A100 40GB на последних слоях.
    try:
        import transformers.core_model_loading as core_model_loading
        core_model_loading.GLOBAL_WORKERS = 1
    except Exception:
        pass


def _quantized_load_controls(pipeline_cfg) -> dict:
    """Дополнительные kwargs для from_pretrained при QLoRA-загрузке."""
    vram_gb = _profile_vram_gb(pipeline_cfg)
    if vram_gb is None:
        return {}

    reserve_gb = 4 if vram_gb <= 40 else 8
    usable_gb = max(1, vram_gb - reserve_gb)
    return {"max_memory": {0: f"{usable_gb}GiB"}}


def _resolve_hf_snapshot(model_name: str) -> str:
    """Возвращает локальный snapshot из HF cache, если online lookup отключён."""
    cache_dir = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
    )
    if not cache_dir:
        return model_name

    repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return model_name

    dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if not dirs:
        return model_name
    return str(dirs[-1])


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
    load_path = _resolve_hf_snapshot(model_name)
    if load_path != model_name:
        print(f"[ModelLoad] local snapshot: {model_name} -> {load_path}")

    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    tokenizer_kwargs = {}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(load_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
    }
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    qcfg = cfg.get("quantization")
    tp = cfg.get("training_params", {})

    if qcfg:
        _prepare_cuda_for_large_load(torch)
        # QLoRA: bnb сам управляет dtype через bnb_4bit_compute_dtype.
        # НЕ передаём torch_dtype — иначе веса грузятся в bf16 на GPU перед
        # квантизацией, пиковое потребление 2× (для 32B на L4 = OOM).
        from transformers import BitsAndBytesConfig
        compute_dtype_str = qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str)
        bnb_kwargs = {
            "load_in_4bit": qcfg.get("load_in_4bit", True),
            "bnb_4bit_quant_type": qcfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": qcfg.get("bnb_4bit_use_double_quant", True),
        }
        if qcfg.get("bnb_4bit_quant_storage"):
            bnb_kwargs["bnb_4bit_quant_storage"] = getattr(torch, qcfg["bnb_4bit_quant_storage"])
        bnb = BitsAndBytesConfig(**bnb_kwargs)
        load_kwargs["quantization_config"] = bnb
        load_kwargs["device_map"] = "auto"  # пошаговая загрузка по слоям, без пика
        load_kwargs.update(_quantized_load_controls(pipeline_cfg))
        print(f"[ModelLoad] 4bit QLoRA: device_map=auto, max_memory={load_kwargs.get('max_memory')}, "
              "async_load=off")
    else:
        # bf16/fp16 для нерубленых LoRA-моделей
        if tp.get("bf16"):
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif tp.get("fp16"):
            load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(load_path, **load_kwargs)
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
