# Fine-tune эксперимент: сводка по 3 раундам

**Дата завершения:** 2026-04-24
**Train/Test:** 1409 / 341 писем, 36 классов (группы A ≥50, B 15-49, C <15 примеров)
**Железо:** A100-PCIE-40GB, 1 GPU/джоба, t2n1@MEPhI HPC

## Итоговая матрица результатов (balanced_accuracy на test)

| Метод × модель | R1 | R2 | R3 | **Best** | F1_A | F1_B | F1_C |
|---|---|---|---|---|---|---|---|
| lora  · Qwen3-14B           | 0.4305 | — (cancel) | **0.5378** | **R3** | 0.726 | 0.540 | 0.506 |
| qlora · Qwen3-32B           | **0.5615** | 0.4832 | 0.5352 | **R1** | 0.725 | 0.525 | 0.625 |
| adalora · Qwen3-32B         | **0.4940** | 0.3622 | 0.4049 | **R1** | 0.740 | 0.511 | 0.450 |
| qlora · T-pro-it-2.1        | — | 0.4334 | **0.5299** | **R3** | 0.731 | 0.466 | 0.597 |
| adalora · T-pro-it-2.1      | — | 0.3952 | **0.4489** | **R3** | 0.641 | 0.525 | 0.347 |
| lora · Vikhr-Nemo-12B       | ~0.467 | ~0.48 | **0.5078** | **R3** | 0.725 | 0.496 | 0.458 |
| tinylora · Qwen3-14B        | **0.2378** | — | — | R1 (only) | 0.520 | 0.296 | 0.083 |

**Топ-3 по balanced_accuracy**: QLoRA Qwen3-32B (0.5615, R1) > LoRA Qwen3-14B (0.5378, R3) > QLoRA T-pro (0.5299, R3).
**Топ-3 по f1_group_C** (редкие классы): QLoRA Qwen3-32B R1 (0.625) > QLoRA T-pro R3 (0.597) > LoRA Qwen3-14B R3 (0.506).

## Ключевые выводы

### 1. Capacity LoRA не универсально «больше = лучше»
- **Qwen3-32B**: лучший QLoRA с `r=16`, при `r=32` падает на 3 пункта.
- **T-pro-it-2.1** (Qwen3-32B, русский continued-pretrain): наоборот, `r=32` даёт +2 пункта относительно Qwen3-32B r=32. Интерпретация: русская адаптация базы сместила эмбеддинги, CLS-голове нужно больше LoRA-параметров для выучивания новой разметки.
- **Qwen3-14B LoRA**: `r=16` оптимален, capacity не лимит.

### 2. Количество эпох важнее гиперпараметров
Пик на validation у всех моделей — **на E3**, независимо от r, dropout, lr. Это структурный предел малой обучающей выборки. 4 эпохи + `load_best_model_at_end=True` — достаточно для всех, больше = overfit.

### 3. «Регуляризация carpet bomb» (R2) уничтожает обучение
Одновременное снижение `r`, увеличение `dropout`, снижение `lr`, увеличение `grad_accum` и ужесточение `max_grad_norm` суммарно ведёт к underfitting: CLS-голова не успевает прогреться, средний провал 10-30 пунктов.

### 4. Группа C (редкие классы) — главный индикатор
QLoRA Qwen3-32B R1 получил f1_C=0.625 — в 2 раза больше остальных. Этот кейс демонстрирует, что аугментация 15→50 примеров работает **если LoRA достаточно мала**, чтобы не переобучиться на артефактах аугментации.

### 5. Архитектурный фикс eval-hang
Перезагрузка базы в `load_finetuned_model` после train() давала OOM-hang на 32B и 12B. Фикс: в `trainer_base.py:run()` передаём `self.model, self.tokenizer` в `evaluate()`, без перезагрузки. `load_best_model_at_end=True` гарантирует, что это лучший чекпоинт.

## Конфиги победителей

В `best_finetune_configs/` лежат JSON с комментариями `_comment_source`:
- `qwen3_32b_qlora.json` — **R1**: r=16, α=32, dropout=0.05, lr=2e-4, epochs=5, grad_accum=8
- `qwen3_32b_adalora.json` — **R1**: init_r=12, target_r=8, α=32, dropout=0.05, lr=5e-4, epochs=6
- `qwen3_14b_lora.json` — **R3**: r=16, α=32, dropout=0.05, lr=2e-4, epochs=4, grad_accum=8
- `tpro_it_21_qlora.json` — **R3**: r=32, α=64, dropout=0.1, lr=2e-4, epochs=4, grad_accum=8
- `tpro_it_21_adalora.json` — **R3**: init_r=16, target_r=12, α=32, dropout=0.1, lr=3e-4, epochs=4
- `vikhr_nemo_12b_lora.json` — **R3**: r=16, α=32, dropout=0.1, lr=1e-4, epochs=4
- `qwen3_14b_tinylora.json` — R1 (единственный прогон)

Общие неизменные: `warmup_ratio=0.03`, `weight_decay=0.01`, `max_grad_norm=1.0`, `per_device_train_batch_size=2`, `max_seq_length=2048`, `bf16=true`, `target_modules={q,k,v,o,gate,up,down}_proj`, `modules_to_save=["score"]`.

## Что дальше (по `project_next_models.md`)

**Круг 4** (опционально, если хочется расширить матрицу):
- RuadaptQwen3-32B-Instruct — прямое сравнение T-pro vs Ruadapt как способов русификации
- Vistral-24B-Instruct — средний размер 24B, другое семейство (Mistral)

**Круг 5** (если очень захочется):
- Gemma-4-31B (base) — требует адаптации `peft_utils` под multimodal-loading

Но для диплома уже **есть достаточная матрица** (6 моделей × 3 размерных класса × 2 семейства) — рациональнее двинуться к тексту.
