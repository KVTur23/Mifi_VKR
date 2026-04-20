# План: Модуль файнтюна Qwen3 для классификации писем

> **Контекст для Claude Code**: этот план расширяет существующий проект. Референсы — `README.md` и структура проекта из `text.md`. Цель — добавить блок `src/finetune/`, CLI-скрипт `scripts/run_finetune.py` и ноутбук `notebooks/finetune.ipynb` для файнтюна Qwen3-14B и Qwen3-32B разными PEFT-методами с сохранением архитектурного стиля проекта: конфиг-driven, GPU-профили, чекпоинты, идемпотентность, единые артефакты в `results/`.
>
> **Главное правило**: ничего не хардкодить в коде. Все модели, гиперпараметры, ранги, target_modules, пути — только из JSON-конфигов, по образцу `config_models/aug_configs/`.

---

## 0. Зафиксированные решения

| Пункт | Значение |
|---|---|
| Задача | **Sequence classification**, 36 классов. `AutoModelForSequenceClassification`, `num_labels=36`, loss — cross-entropy. |
| Train-данные | `Data/data_after_stage3.csv` (финальный аугментированный датасет — аугментация сделана именно для того, чтобы файнтюн на ней учился) |
| Test-данные | `Data/data_test.csv` (фиксированный 20% сплит, не трогать) |
| Модели | Qwen3-14B (LoRA), Qwen3-32B (QLoRA / AdaLoRA / TinyLoRA) |
| PEFT task type | `TaskType.SEQ_CLS` |
| Метрики общие | `balanced_accuracy`, `macro_f1`, `classification_report` |
| Метрики per-group | `f1_group_A/B/C` — только для разреза в `all_methods_comparison.csv`, чтобы сравнивать с prompt-классификацией |
| Группы A/B/C | Только для отчёта. Границы по числу **оригинальных** писем в train (A ≥ 50, B ∈ [15,49], C < 15) — как в `few_shot_examples.py`. На обучение не влияют. |
| Артефакты | `results/finetune_results.csv`, `results/preds_<method>_<model>.csv` — схема как у `prompt_results.csv` |
| Точки запуска | `scripts/run_finetune.py` (CLI) и `notebooks/finetune.ipynb` (Colab) — обе используют одну и ту же функцию-оркестратор |

> **Важно про Qwen3 + SEQ_CLS**: Qwen3 — causal-модель, но `AutoModelForSequenceClassification` её поддерживает — добавляет классификационную голову поверх последнего скрытого состояния **последнего не-паддинг токена**. Нужно обязательно выставить `tokenizer.padding_side = "left"` и `model.config.pad_token_id = tokenizer.pad_token_id` (у Qwen3 по умолчанию pad_token может отсутствовать — использовать `eos_token` как pad).

---

## 1. Структура новых файлов

```
code/
├── config_models/
│   ├── pipeline_config.json                 # ← РАСШИРИТЬ (секция "finetune")
│   └── finetune_configs/                    # ← НОВОЕ
│       ├── qwen3_14b_lora.json
│       ├── qwen3_32b_qlora.json
│       ├── qwen3_32b_adalora.json
│       └── qwen3_32b_tinylora.json
├── src/
│   └── finetune/                            # ← НОВОЕ
│       ├── __init__.py
│       ├── data_prep.py                     # Токенизация + encode меток + collator + группы A/B/C
│       ├── peft_utils.py                    # Загрузка базы, PEFT-фабрика (LoRA/AdaLoRA/TinyLoRA)
│       ├── trainer_base.py                  # Базовый SeqClsRunner
│       ├── run_lora.py                      # Qwen3-14B + LoRA
│       ├── run_qlora.py                     # Qwen3-32B + QLoRA
│       ├── run_adalora.py                   # Qwen3-32B + AdaLoRA
│       ├── run_tinylora.py                  # Qwen3-32B + TinyLoRA
│       ├── evaluate_finetuned.py            # Инференс + метрики + per-group F1
│       └── orchestrator.py                  # run_finetune(methods, gpu, force) — общее ядро для CLI и ноутбука
├── scripts/
│   └── run_finetune.py                      # ← НОВОЕ: CLI-обёртка над orchestrator.run_finetune
├── notebooks/
│   └── finetune.ipynb                       # ← НОВОЕ: Colab-обёртка над orchestrator.run_finetune
└── requirements.txt                         # ← ДОПОЛНИТЬ: peft, bitsandbytes, accelerate
```

Промпты / шаблоны SFT не нужны — sequence classification подаёт чистый текст.

---

## 2. Конфиги моделей (`config_models/finetune_configs/`)

### Общая схема JSON

Самодостаточный конфиг по образцу `aug_configs/model_vllm.json`. Код читает ключи и передаёт в HF/PEFT как есть.

```json
{
  "model_name": "Qwen/Qwen3-14B",
  "method": "lora",
  "task_type": "SEQ_CLS",
  "num_labels": 36,
  "max_seq_length": 2048,

  "peft_config": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "modules_to_save": ["score"]
  },

  "quantization": null,

  "training_params": {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optim": "adamw_torch",
    "gradient_checkpointing": true,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": true,
    "metric_for_best_model": "macro_f1",
    "greater_is_better": true,
    "save_total_limit": 2,
    "bf16": true,
    "seed": 42
  }
}
```

**Критично для classification-головы**: `modules_to_save: ["score"]` — иначе новая классификационная голова останется случайной. У HF-обёртки для causal-моделей голова называется `score` (проверить при реализации, поправить если окажется иначе).

### Отличия по методам — `peft_config`

**LoRA** (`qwen3_14b_lora.json`) — `method: "lora"`, `quantization: null`:
```json
"peft_config": {
  "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
  "modules_to_save": ["score"]
}
```

**QLoRA** (`qwen3_32b_qlora.json`) — `method: "qlora"`, та же структура `peft_config` что и у LoRA, но с 4bit-базой:
```json
"quantization": {
  "load_in_4bit": true,
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_use_double_quant": true
}
```

**AdaLoRA** (`qwen3_32b_adalora.json`) — `method: "adalora"`, используется `AdaLoraConfig` из `peft`:
```json
"peft_config": {
  "init_r": 12,
  "target_r": 8,
  "beta1": 0.85,
  "beta2": 0.85,
  "tinit": 200,
  "tfinal": 1000,
  "deltaT": 10,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
  "modules_to_save": ["score"],
  "total_step": null
}
```
> `total_step: null` — подставляется динамически в `trainer_base.prepare()` как `ceil(len(train_dataset) / effective_batch) × num_train_epochs`. В JSON держим `null` как явный маркер.

**TinyLoRA** (`qwen3_32b_tinylora.json`) — `method: "tinylora"`, используется `TinyLoraConfig` из `peft`:
```json
"peft_config": {
  "r": 2,
  "u": 64,
  "weight_tying": 0.0,
  "target_modules": ["q_proj", "v_proj"],
  "projection_seed": 42,
  "modules_to_save": ["score"]
}
```
> Это отдельный PEFT-метод на основе SVD (не «LoRA с низким рангом»). Параметры по рекомендации авторов: `r=2` (ранг SVD), `u=64` (размерность обучаемого вектора), `weight_tying=0.0` (каждый модуль получает свой вектор `v`). **Важно**: проверить, что `TinyLoraConfig` принимает `modules_to_save` и `task_type` — если нет, применять их альтернативным способом (см. §8.1).

### Отличия по методам — `training_params`

У TinyLoRA параметров обучения очень мало (порядка тысяч), поэтому LR может быть выше. Рекомендации:

| Метод | `learning_rate` | `num_train_epochs` |
|---|---|---|
| LoRA | 2e-4 | 5 |
| QLoRA | 2e-4 | 5 |
| AdaLoRA | 5e-4 | 5-7 (нужно время на бюджет-расписание) |
| TinyLoRA | 1e-3 | 5-10 |

Это стартовые значения, финальные подбираются после первого прогона.

---

## 3. Расширение `pipeline_config.json`

Добавить секцию `finetune` с разбивкой по GPU-профилям — **все пять профилей**, которые уже используются в проекте:

```json
"finetune": {
  "T4":       {"per_device_batch": 1, "grad_accum": 16, "max_seq": 1024, "bf16": false, "fp16": true},
  "L4":       {"per_device_batch": 1, "grad_accum": 16, "max_seq": 1536, "bf16": true,  "fp16": false},
  "A100_40":  {"per_device_batch": 2, "grad_accum": 8,  "max_seq": 1536, "bf16": true,  "fp16": false},
  "A100_80":  {"per_device_batch": 4, "grad_accum": 4,  "max_seq": 2048, "bf16": true,  "fp16": false},
  "H100":     {"per_device_batch": 4, "grad_accum": 4,  "max_seq": 2048, "bf16": true,  "fp16": false},
  "common": {
    "output_dir": "Data/finetune_checkpoints",
    "results_csv": "results/finetune_results.csv",
    "preds_dir":   "results/",
    "early_stopping_patience": 2
  }
}
```

> Профили — `T4 / L4 / A100_40 / A100_80 / H100`, как в актуальном коде (A100_40 и A100_80 разведены, т.к. имеют разный VRAM и разные допустимые `max_seq`/`per_device_batch`).

**Логика слияния**: `pipeline_config.finetune[gpu]` **переопределяет** соответствующие ключи в `training_params` модельного JSON — тот же паттерн, что у `load_llm` с `gpu_memory_utilization`/`enforce_eager`.

**Как течёт `max_seq_length`** (важно — Claude Code должен прописать это явно):
```
qwen3_14b_lora.json: "max_seq_length": 2048     ← желаемое (из модельного конфига)
                         ↓
pipeline_config.finetune[gpu].max_seq = 1024    ← физический лимит GPU (T4)
                         ↓
effective_max_seq = 1024                        ← GPU-профиль переопределяет модельное
                         ↓
tokenize_dataset(df, tokenizer, max_seq=1024)   ← truncation до 1024
```
Хардкод `max_seq_length=2048` в коде — запрещён. Всегда читается из слитого конфига.

---

## 4. Модули

### 4.1 `src/finetune/data_prep.py`

| Функция | Описание |
|---|---|
| `load_finetune_data()` | Грузит `data_after_stage3.csv` через `data_loader.load_dataset(stage=3)` + test через `load_test_set()`. Дополнительно грузит `train_after_eda.csv` (stage=0) **только для подсчёта оригинальных примеров на класс** — нужно для отчётного разреза A/B/C, на обучение не влияет. Возвращает `(df_train, df_test, orig_counts)`. |
| `build_label_mapping(df_train)` | Строит `label2id` и `id2label` из уникальных значений `label` в train, **отсортированных через `sorted()`** (детерминированный порядок — как в `data_loader.split_train_test`). Возвращает `(label2id, id2label)`. |
| `compute_class_groups(orig_counts, label2id)` | Возвращает `dict[int, str]` — `class_id → "A" / "B" / "C"` по числу **оригинальных** писем (A ≥ 50, B ∈ [15,49], C < 15). Те же границы, что в `few_shot_examples.py`. Сохраняется рядом с адаптером для переиспользования при eval. |
| `encode_labels(df, label2id)` | Добавляет колонку `label_id` с числовыми id. |
| `tokenize_dataset(df, tokenizer, max_seq_length)` | Превращает `df` в `datasets.Dataset` с полями `input_ids`, `attention_mask`, `labels`. `truncation=True` + `max_length=max_seq_length` — длинные письма обрезаются, но `padding` **не указан** — тексты сохраняются своей родной длины. |
| `get_collator(tokenizer)` | Возвращает `DataCollatorWithPadding(tokenizer, padding="longest")`. Паддит каждый батч до длины самого длинного примера **в этом батче**. |

**Зачем такое разделение (dynamic padding)**. Если паддить все тексты сразу до `max_seq_length=2048` при токенизации — батч из 4 коротких писем по 300 токенов всё равно займёт 2048×4 токенов VRAM. С динамическим паддингом в коллаторе тот же батч займёт 300×4 — в ~7 раз меньше памяти и быстрее. Это стандартный паттерн HuggingFace, особенно важный для 32B-моделей на A100_40.

**Про `padding_side`**: для Qwen3 + classification head **обязательно** `tokenizer.padding_side = "left"` — пулинг берётся с последнего non-pad токена. Выставляется в `peft_utils.load_base_model`.

### 4.2 `src/finetune/peft_utils.py`

| Функция | Описание |
|---|---|
| `load_base_model(cfg, pipeline_cfg, num_labels, id2label, label2id)` | `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=..., id2label=..., label2id=...)`. Если `cfg.quantization` — применяет `BitsAndBytesConfig`. Загружает токенизатор, выставляет `padding_side="left"`, `pad_token = eos_token` если pad отсутствует, синхронизирует `model.config.pad_token_id`. Включает `gradient_checkpointing`. Возвращает `(model, tokenizer)`. |
| `build_peft_config(cfg, total_step=None)` | Фабрика. Принимает `cfg.method` и `cfg.peft_config`. Возвращает соответствующий PEFT-конфиг с `task_type=TaskType.SEQ_CLS`. Реализация ниже. |
| `wrap_with_peft(model, peft_config, is_quantized)` | Если quantized → `prepare_model_for_kbit_training(model)`. Иначе — `model.enable_input_require_grads()` (нужно для gradient_checkpointing + LoRA). Затем `get_peft_model(model, peft_config)`. Печатает `print_trainable_parameters()` и **проверяет** что в trainable есть `score.*`. Возвращает модель. |
| `save_adapter(model, output_dir, id2label, class_groups)` | Сохраняет адаптер + tokenizer + `id2label.json` + `class_groups.json`. |

**`build_peft_config` — ключевая фабрика. Псевдокод:**

```python
from peft import LoraConfig, AdaLoraConfig, TinyLoraConfig, TaskType

def build_peft_config(cfg, total_step=None):
    method = cfg["method"]
    pc = cfg["peft_config"]

    if method in ("lora", "qlora"):
        return LoraConfig(task_type=TaskType.SEQ_CLS, **pc)

    if method == "adalora":
        pc = dict(pc)  # копия, чтобы не мутировать
        if pc.get("total_step") is None:
            if total_step is None:
                raise ValueError("AdaLoRA: total_step required")
            pc["total_step"] = total_step
        return AdaLoraConfig(task_type=TaskType.SEQ_CLS, **pc)

    if method == "tinylora":
        # TinyLoraConfig может не принимать все стандартные LoraConfig-поля.
        # Передаём только то, что он точно понимает:
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
```

### 4.3 `src/finetune/trainer_base.py`

```python
class SeqClsRunner:
    def __init__(self, config_path: str, pipeline_cfg):
        # 1. Загрузка модельного JSON через config_loader.load_model_config
        # 2. Слияние training_params с pipeline_cfg.finetune[gpu]
        #    (per_device_batch, grad_accum, max_seq, bf16/fp16 переопределяют модельные)
        # 3. Вычисление output_dir: {pipeline_cfg.finetune.common.output_dir}/{method}_{model_short}/
        # 4. Сохранение run_key = f"{method}_{model_short}" для результатов в CSV
        ...

    def prepare(self):
        # 1. load_finetune_data() → df_train, df_test, orig_counts
        # 2. build_label_mapping(df_train) → label2id, id2label (assert len == num_labels == 36)
        # 3. compute_class_groups(orig_counts, label2id) → class_groups
        # 4. load_base_model(...) → self.model, self.tokenizer
        # 5. tokenize train/test
        # 6. Вычислить total_step (для AdaLoRA) = ceil(len(train_ds) / effective_batch) * epochs
        # 7. build_peft_config(cfg, total_step) → wrap_with_peft
        ...

    def train(self):
        # TrainingArguments из self.cfg.training_params (+ output_dir, +report_to="none")
        # EarlyStoppingCallback(patience=pipeline_cfg.finetune.common.early_stopping_patience)
        # Trainer(model, args, train, eval, tokenizer, collator, compute_metrics=_compute_metrics)
        # trainer.train()
        # save_adapter(...)
        ...

    def run(self):
        self.prepare()
        self.train()
        from .evaluate_finetuned import evaluate
        return evaluate(self.output_dir, self.config_path, self.pipeline_cfg, self.run_key)
```

`_compute_metrics` для Trainer (используется в eval во время обучения — только базовые метрики, без per-group):

```python
def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
```

`metric_for_best_model: "macro_f1"` — с этим ключом совпадает.

### 4.4 Runners

Каждый файл — тонкая обёртка. Сигнатура **идентична** `stage1_llm_generate.run`:

```python
# run_lora.py
from .trainer_base import SeqClsRunner

CONFIG_PATH = "config_models/finetune_configs/qwen3_14b_lora.json"

def run(config_path: str = CONFIG_PATH, pipeline_cfg=None):
    runner = SeqClsRunner(config_path, pipeline_cfg)
    return runner.run()
```

Всё различие между методами — **в JSON-конфигах**. `run_qlora.py`, `run_adalora.py`, `run_tinylora.py` отличаются только значением `CONFIG_PATH`.

### 4.5 `src/finetune/evaluate_finetuned.py`

| Функция | Описание |
|---|---|
| `load_finetuned_model(adapter_dir, base_model_name, quantization_cfg, num_labels, id2label, label2id)` | `AutoModelForSequenceClassification.from_pretrained(base)` + `PeftModel.from_pretrained(model, adapter_dir)`. Режим `eval()`. |
| `predict(model, tokenizer, texts, batch_size, max_seq_length)` | Батчевый инференс: токенизация → forward → `argmax(logits, dim=-1)`. Возвращает numpy-массив предсказанных id. |
| `evaluate(adapter_dir, config_path, pipeline_cfg, run_key)` | Грузит test, читает `id2label.json` и `class_groups.json` из `adapter_dir`. Предиктит, считает: `balanced_accuracy`, `macro_f1`, `f1_group_A/B/C`, `classification_report(target_names=label_names)`. **Пишет артефакты в `results/`** (см. §5). Возвращает dict с метриками. |

**Per-group F1** (как в `evaluate_prompt_classification`):

```python
def _f1_per_group(y_true, y_pred, class_groups):
    # class_groups: {class_id: "A"/"B"/"C"}
    groups = {"A": [], "B": [], "C": []}
    for cid, g in class_groups.items():
        groups[g].append(cid)
    out = {}
    for g, ids in groups.items():
        mask = np.isin(y_true, ids)
        if mask.sum() == 0:
            out[f"f1_group_{g}"] = None
            continue
        out[f"f1_group_{g}"] = f1_score(
            y_true[mask], y_pred[mask], average="macro",
            labels=ids, zero_division=0
        )
    return out
```

### 4.6 `src/finetune/orchestrator.py` — общее ядро

Функция `run_finetune` — **единая точка** для CLI и ноутбука, чтобы не дублировать логику оркестрации.

```python
def run_finetune(methods: list[str] = None, gpu: str = "A100_40",
                 force: bool = False) -> pd.DataFrame:
    """
    Запускает файнтюн выбранных методов.
    methods: список из ["lora", "qlora", "adalora", "tinylora"]. None = все.
    gpu: "T4" / "L4" / "A100_40" / "A100_80" / "H100".
    force: если True — пересчитывает уже готовые run_key.
    Возвращает DataFrame с итоговыми метриками.
    """
    ...
```

**Что делает:**
1. `pipeline_cfg = load_pipeline_config(gpu=gpu)`.
2. **Идемпотентность** (как в `run_few_shot.py`): читает `results/finetune_results.csv`. Если `run_key = "{method}_{model_short}"` уже есть — метод пропускается, если `force=False`.
3. Для каждого не пропущенного метода:
   - Импортирует `src.finetune.run_<method>`, вызывает `.run(pipeline_cfg=pipeline_cfg)`.
   - Перехватывает `torch.cuda.OutOfMemoryError` → логирует, продолжает.
   - Между методами — выгрузка в стиле `prompt_classifier.unload_model`: `del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(5)`.
4. После всех методов обновляет `results/all_methods_comparison.csv`, добавляя блок `finetune`.
5. Возвращает сводную таблицу.

### 4.7 `scripts/run_finetune.py` — CLI-обёртка

```python
import argparse
from src.finetune.orchestrator import run_finetune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        choices=["lora", "qlora", "adalora", "tinylora"])
    parser.add_argument("--gpu", default="A100_40",
                        choices=["T4", "L4", "A100_40", "A100_80", "H100"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = run_finetune(methods=args.methods, gpu=args.gpu, force=args.force)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
```

Примеры использования:
```
python scripts/run_finetune.py
python scripts/run_finetune.py --methods lora qlora
python scripts/run_finetune.py --methods tinylora --gpu A100_80
python scripts/run_finetune.py --force --gpu H100
```

### 4.8 `notebooks/finetune.ipynb` — Colab-обёртка

Структура ноутбука — **по образцу `notebooks/augmentation.ipynb`**:

**Ячейка 1**: подтягивание кода (git clone / mount drive + cd в проект), `pip install -r requirements.txt`.

**Ячейка 2**: импорты и GPU-профиль — **одна строка конфигурации**, как в главном ноутбуке проекта:
```python
from src.finetune.orchestrator import run_finetune

GPU = "A100_40"  # варианты: "T4", "L4", "A100_40", "A100_80", "H100"
```

**Ячейка 3**: выбор методов — комментарии/переключение:
```python
# Полный прогон всех методов:
METHODS = None

# Или конкретный набор:
# METHODS = ["lora"]
# METHODS = ["qlora", "adalora", "tinylora"]
```

**Ячейка 4**: запуск:
```python
df = run_finetune(methods=METHODS, gpu=GPU, force=False)
df
```

**Ячейка 5**: показ сводки из `results/all_methods_comparison.csv`:
```python
import pandas as pd
pd.read_csv("results/all_methods_comparison.csv")
```

Вся логика — в `orchestrator.run_finetune`. Ноутбук и CLI — тонкие обёртки, которые **не содержат дублирующейся логики**.

---

## 5. Артефакты в `results/` — единая схема проекта

**`results/finetune_results.csv`** (инкрементальный, аналог `prompt_results.csv`):

| Колонка | Пример |
|---|---|
| `run_key` | `lora_qwen3_14b` |
| `method` | `lora` / `qlora` / `adalora` / `tinylora` |
| `model` | `Qwen/Qwen3-14B` |
| `balanced_accuracy` | 0.723 |
| `macro_f1` | 0.689 |
| `f1_group_A` | 0.812 |
| `f1_group_B` | 0.670 |
| `f1_group_C` | 0.450 |
| `trainable_params` | 42598400 |
| `train_time_sec` | 3421 |
| `timestamp` | 2026-04-20T13:00:00 |

**`results/preds_<method>_<model_short>.csv`** (per-sample, аналог `preds_<model>_exp<k>.csv`):
колонки `text, true_label, predicted_label, correct`.

**`results/all_methods_comparison.csv`** — добавить ветку `finetune` с общими колонками со существующими блоками `baseline` / `augmented` / `prompt`. Обновляется в конце `run_finetune`.

---

## 6. Порядок реализации (для Claude Code)

**Фаза 1 — конфиги:**
1. Создать `config_models/finetune_configs/` + 4 JSON из §2.
2. Расширить `pipeline_config.json` секцией `finetune` (§3). Профили — `T4/L4/A100_40/A100_80/H100` строго как в проекте.
3. Дополнить `requirements.txt`: `peft>=0.11`, `bitsandbytes>=0.43`, `accelerate>=0.30`.

**Фаза 2 — общий код:**
4. `src/finetune/__init__.py` (пустой).
5. `src/finetune/data_prep.py`.
6. `src/finetune/peft_utils.py` — **сразу проверить** в питон-REPL, что `from peft import TinyLoraConfig, AdaLoraConfig` импортируется; если нет — обновить peft.
7. `src/finetune/trainer_base.py`.
8. `src/finetune/evaluate_finetuned.py`.

**Фаза 3 — runners:**
9. `run_lora.py` — **smoke-test** на 1 эпохе, 100 шагов, T4/L4. Проверить:
   - Конфиг читается, `num_labels=36`, `id2label` в модели совпадает с `label2id` из датасета.
   - `print_trainable_parameters()` показывает ненулевой процент и среди trainable есть `score.*`.
   - `results/finetune_results.csv` создаётся с корректной строкой.
   - `f1_group_{A,B,C}` не `None`.
10. `run_qlora.py`, `run_adalora.py`, `run_tinylora.py`.

**Фаза 4 — оркестратор + точки запуска:**
11. `src/finetune/orchestrator.py` — `run_finetune()` со всей логикой.
12. `scripts/run_finetune.py` — argparse-обёртка, ~15 строк.
13. `notebooks/finetune.ipynb` — пять ячеек по §4.8, по образцу `augmentation.ipynb`.
14. Проверить идемпотентность (повторный запуск с теми же методами → `skipped`) и в CLI, и в ноутбуке.

**Фаза 5 — полный прогон:**
15. На целевом GPU запустить все 4 метода. Проверить, что `all_methods_comparison.csv` содержит новый блок `finetune` и сопоставим с `baseline`/`augmented`/`prompt`.

---

## 7. Чеклист соответствия архитектуре проекта

- [ ] **Ни одна модель и ни один гиперпараметр не захардкожены** — только JSON в `config_models/finetune_configs/`.
- [ ] Все runners имеют сигнатуру `run(config_path, pipeline_cfg=None)` — как `stage1_llm_generate.run`.
- [ ] GPU-профили названы `T4/L4/A100_40/A100_80/H100` — как в существующем коде.
- [ ] GPU-настройки переопределяют модельные через слияние — паттерн `load_llm`.
- [ ] `max_seq_length` берётся из слитого конфига, никогда не хардкодится.
- [ ] Данные — только через `src/utils/data_loader.py`.
- [ ] Конфиги грузятся через `src/utils/config_loader.load_model_config()`.
- [ ] Обучение — на `data_after_stage3.csv` (аугментированный train, как и задумано проектом).
- [ ] `label2id` детерминирован (`sorted()`), `id2label.json` и `class_groups.json` сохраняются рядом с адаптером.
- [ ] `modules_to_save` включает `score` (classification head).
- [ ] `tokenizer.padding_side = "left"` и `pad_token_id` синхронизирован с моделью.
- [ ] Группы A/B/C — только для per-group F1 в отчёте. Считаются по **оригинальным** письмам (из `train_after_eda.csv`), границы `A ≥ 50`, `B ∈ [15,49]`, `C < 15` — совпадают с `few_shot_examples.py`. На обучение не влияют.
- [ ] Идемпотентность: повторный запуск пропускает уже выполненные `run_key`, как в `run_few_shot.py`.
- [ ] Артефакты пишутся в `results/` с той же схемой, что у prompt-классификации.
- [ ] `all_methods_comparison.csv` обновляется и содержит блок `finetune`.
- [ ] Между методами — выгрузка по паттерну `prompt_classifier.unload_model` (`gc.collect` → `empty_cache` → `synchronize` → `sleep(5)`).
- [ ] CLI (`scripts/run_finetune.py`) и ноутбук (`notebooks/finetune.ipynb`) — тонкие обёртки над `orchestrator.run_finetune`, без дублирующейся логики.
- [ ] `requirements.txt` обновлён.

---

## 8. Потенциальные грабли

1. **TinyLoRA API и `modules_to_save`**. `TinyLoraConfig` может **не принимать** все стандартные LoRA-поля (проверить сигнатуру перед реализацией). Если `modules_to_save` там не поддерживается — разморозить голову вручную до `get_peft_model`:
   ```python
   for name, p in base_model.named_parameters():
       if "score" in name:
           p.requires_grad = True
   ```
   Это alternative-path в `wrap_with_peft`.

2. **Classification head под PEFT**. По умолчанию `get_peft_model` замораживает всё. Без `modules_to_save: ["score"]` голова останется случайной → метрики ~1/36. Обязательная проверка на smoke-тесте: `print_trainable_parameters()` должен показывать `score.*` в списке trainable.

3. **Qwen3 + padding**. У Qwen3 pad_token может быть не определён: `tokenizer.pad_token = tokenizer.eos_token`, `model.config.pad_token_id = tokenizer.pad_token_id`. Classification head берёт пулинг по последнему non-pad токену → **обязательно** `padding_side="left"`.

4. **AdaLoRA `total_step`**. Вычислять динамически в `trainer_base.prepare()` как `ceil(len(train_dataset) / effective_batch) × num_train_epochs` и подставлять в конфиг **до** `get_peft_model`. В JSON оставляем `null` как явный маркер.

5. **Gradient checkpointing + PEFT без квантизации**. Требует `model.enable_input_require_grads()` **до** `get_peft_model`, иначе градиенты не потекут через замороженную базу к адаптерам. `prepare_model_for_kbit_training` делает это само для quantized, для обычного LoRA — вручную.

6. **Длина текстов**. Письма бывают длинные (см. `trim_attached_documents` в `data_cleaner.py` — уже в проекте). `max_seq_length=2048` на A100_80/H100, `1536` на L4/A100_40, `1024` на T4. Проверить распределение длин после токенизации и при агрессивном обрезании хвоста — поднять `trim_attached_documents` или `max_seq`.

7. **32B + 4bit на A100_40**. Qwen3-32B в 4bit ~18-20GB + активации + оптимизатор + градиенты адаптеров. На A100_80/H100 проходит комфортно. На A100_40 с `per_device_batch=2`, `grad_accum=8`, `max_seq=1536`, `gradient_checkpointing=true` должно влезть, но близко к границе — если OOM, уменьшить `max_seq` до 1024 и увеличить `grad_accum` до 16. На L4 24GB — под большим вопросом, ограничить эту модель профилем `A100_40+`. На T4 16GB — 32B не влезет, доступен только LoRA на 14B.

8. **Группы A/B/C — только отчётный разрез**. Границы считаются по оригинальным письмам в `train_after_eda.csv` — это нужно для честного сравнения с prompt-классификацией в `all_methods_comparison.csv`. На сам процесс обучения (loss, батчи, семплирование) это никак не влияет — файнтюн видит весь аугментированный `data_after_stage3.csv` полностью, для этого аугментация и делалась.
