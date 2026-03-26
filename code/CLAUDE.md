# CLAUDE.md — Контекст проекта для Claude Code

## Проект

ВКР: "Методы классификации текстовых документов на основе дообучения большой языковой модели". Пайплайн аугментации и классификации датасета деловых электронных писем на русском языке (36 классов, ~1750 писем). Задача — маршрутизация входящей корреспонденции нефтегазовой компании по подразделениям.

---

## Структура проекта

```
code/
├── config_models/
│   ├── pipeline_config.json              # Конфигурация + GPU-профили + prompt_classification
│   └── aug_configs/                      # Конфигурации LLM-моделей для аугментации
├── prompts/
│   ├── aug_prompts/                      # Промпты аугментации (этап 2)
│   └── classification_prompts/           # Промпты классификации (этап 3)
│       ├── zero_shot.txt
│       ├── one_shot.txt
│       └── few_shot.txt
├── src/
│   ├── augmentation/                     # Этап 2: Аугментация (под-этапы 2.1-2.3, завершены)
│   │   ├── llm_utils.py                  # vLLM обёртка + LLM-судья
│   │   ├── stage1_llm_generate.py        # 2.1: Генерация LLM (классы <15 → 15)
│   │   ├── stage2_paraphrase.py          # 2.2: Перефразирование (15-34 → 35)
│   │   ├── stage3_back_translation.py    # 2.3: Обратный перевод NLLB (35-49 → 50)
│   │   └── validation.py                 # 7 фильтров валидации
│   ├── classification/                   # Классификация
│   │   ├── embeddings.py                 # TF-IDF с кэшированием
│   │   ├── evaluate.py                   # Общая логика оценки
│   │   ├── rubert_classifier.py          # RuBERT-tiny2 fine-tune
│   │   ├── run_svm.py                    # Бейзлайн: LinearSVC
│   │   ├── run_logreg.py                 # Бейзлайн: LogisticRegression
│   │   ├── run_naive_bayes.py            # Бейзлайн: MultinomialNB
│   │   ├── prompt_classifier.py          # ← ЭТАП 3: prompt-based классификация
│   │   └── few_shot_examples.py          # ← ЭТАП 3: подготовка few-shot примеров
│   └── utils/
│       ├── data_loader.py                # Загрузка данных + чекпоинты
│       ├── data_cleaner.py               # Предобработка
│       ├── config_loader.py              # Парсинг конфигов
│       └── pipeline_config.py            # GPU-профили
├── Data/
│   ├── data_test.csv                     # Тестовый набор (343 примера, ФИКСИРОВАН)
│   ├── train_after_eda.csv               # Оригинальный train (1412 примеров)
│   ├── data_after_stage3.csv             # Аугментированный train
│   ├── class_descriptions.json           # ← ЭТАП 3: описания классов
│   └── few_shot_examples.json            # ← ЭТАП 3: примеры (K=1,3,5)
├── notebooks/
│   ├── augmentation.ipynb
│   └── zer0_one_few_shot.ipynb    # ← ЭТАП 3
├── results/
│   ├── classification_results.csv          # Результаты baseline (существует)
│   ├── prompt_results.csv                # ← ЭТАП 3
│   └── all_methods_comparison.csv        # ← ЭТАП 3
├── EDA/
│   └── EDA.ipynb
└── CLAUDE.md                             # Этот файл
```

---

## Общий поток данных

```
data.json → data_cleaner → data_after_eda.csv → split 80/20
  ├── train_after_eda.csv (1412)
  ├── data_test.csv (343) ← ЕДИНЫЙ ТЕСТ ДЛЯ ВСЕХ МЕТОДОВ
  │
  ├── Этап 1: EDA ✅ ГОТОВО
  │
  ├── Бейзлайн до аугментации: SVM / LogReg / NB / RuBERT-tiny2 ✅ ГОТОВО
  │
  ├── Этап 2: Аугментация (2.1 → 2.2 → 2.3) → data_after_stage3.csv ✅ ГОТОВО
  │
  ├── Бейзлайн после аугментации ✅ ГОТОВО
  │
  ├── Этап 3: Prompt-based Zero/One/Few-shot ← ТЕКУЩИЙ ЭТАП
  │
  ├── Этап 4: Fine-tuning LLM (следующий)
  │
  └── Итоговое сравнение
```

---

## Ключевые соглашения

- **Тестовый набор:** `data_test.csv`, 343 примера. Один и тот же для ВСЕХ методов. Никогда не менять.
- **Метрики:** `balanced_accuracy`, `macro_f1` — единые для всех этапов.
- **Seed:** `RANDOM_SEED = 42` из `data_loader.py`. Использовать везде.
- **Колонки:** `TEXT_COL = "text"`, `LABEL_COL = "label"` из `data_loader.py`.
- **GPU:** A100 40GB. Профиль `A100_40` в `pipeline_config.json`.
- **Выгрузка моделей:** между моделями всегда `del model; gc.collect(); torch.cuda.empty_cache()`.

---

## Этап 3: Prompt-based классификация (ТЕКУЩИЙ)

### Суть

Классификация писем через промпты с генеративными LLM. Модель получает инструкцию + список классов + (опционально) примеры → генерирует название подразделения.

### Модели

| Модель | VRAM | Контекст | Загрузка |
|---|---|---|---|
| `IlyaGusev/saiga_llama3_8b` | ~16 GB FP16 | 8K | transformers |
| `google/gemma-2-9b-it` | ~18 GB BF16 | 8K | transformers |
| `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | ~24 GB FP16 | 32K | transformers |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB BF16 | 32K+ | transformers |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~18-20 GB 4-bit | 32K+ | transformers (AWQ) |

### Группы классов

Влияют на выбор few-shot примеров:

- **Группа A** (9 классов, 50+ оригинальных train): few-shot только из оригинальных.
- **Группа B** (15 классов, 15–49 оригинальных): few-shot только из оригинальных.
- **Группа C** (12 классов, <15 оригинальных): few-shot из оригинальных + аугментированных.

Аугментированные данные НИКОГДА не попадают в тест. Используются только как few-shot примеры для микро-классов группы C.

### Подготовка (выполнить один раз)

1. **Описания классов** — `generate_class_context()` из `stage1_llm_generate.py` с Qwen-32B для всех 36 классов. Одна модель генерирует описания для всех экспериментов. Сохранить → `Data/class_descriptions.json`.

2. **Few-shot примеры** — для K=1, 3, 5. Группы A/B из оригинального train, группа C из оригинального + аугментированного. Сохранить → `Data/few_shot_examples.json`.

3. **Промпты** — три шаблона в `prompts/classification_prompts/`: `zero_shot.txt`, `one_shot.txt`, `few_shot.txt`. Плейсхолдеры: `{class_list}`, `{class_descriptions}`, `{text}`, `{examples_block}`, `{example_text}`, `{example_label}`.

### Ограничения контекстного окна

| Режим | ~Токенов | Saiga/Gemma (8K) | Vikhr/Qwen (32K+) |
|---|---|---|---|
| zero-shot | ~2000 | ✅ | ✅ |
| one-shot | ~9000 | ⚠️ | ✅ |
| few-shot K=3 | ~25000 | ❌ skip | ✅ |
| few-shot K=5 | ~40000 | ❌ skip | ⚠️ |

Перед запуском проверять `len(tokenizer.encode(prompt)) < max_context - 100`. Если не влезает → `skipped=true` в результатах.

### Матрица экспериментов

| Модель | K=0 | K=1 | K=3 | K=5 |
|---|---|---|---|---|
| Saiga-LLaMA3-8B | ✅ | ⚠️ | skip | skip |
| Gemma-2-9B | ✅ | ⚠️ | skip | skip |
| Vikhr-Nemo-12B | ✅ | ✅ | ✅ | ⚠️ |
| Qwen2.5-14B | ✅ | ✅ | ✅ | ⚠️ |
| Qwen2.5-32B-AWQ | ✅ | ✅ | ✅ | ⚠️ |

### Извлечение предсказания (extract_prediction)

Модель генерирует свободный текст → нужно извлечь название подразделения:
1. Точное совпадение (lower)
2. Частичное (label в ответе)
3. Нечёткое (`difflib.get_close_matches`, cutoff=0.7)
4. `"unknown"` — если ничего не сработало

Протестировать на 10-20 примерах вручную перед массовым запуском.

### Параметры генерации

```python
temperature=0.1       # Почти детерминированный
do_sample=True
top_p=0.9
repetition_penalty=1.1
max_new_tokens=100
```

### Метрики

Расширить `evaluate.py` функцией `evaluate_prompt_classification()`:
- `balanced_accuracy`, `macro_f1` — как в baseline
- `unknown_rate` — доля нераспознанных ответов
- Per-group metrics (A/B/C) — отдельно для каждой группы классов

### Результаты

**`results/prompt_results.csv`** — одна строка на эксперимент:
```
model, model_size, k_shots, balanced_accuracy, macro_f1,
unknown_rate, f1_group_A, f1_group_B, f1_group_C, prompt_tokens, skipped, n_test
```

**`results/all_methods_comparison.csv`** — сводка ВСЕХ методов:
```
method, model, setting, balanced_accuracy, macro_f1, unknown_rate
```
Включает baseline (SVM, LogReg, NB, RuBERT) + prompt-based + будущий fine-tuning.

### Визуализации

1. Барплот всех методов (baseline + prompt) по macro_f1
2. Кривая F1 от K (0→1→3→5) для каждой модели
3. Метрики по группам A/B/C
4. Confusion matrix лучшего метода
5. Per-class F1 с выделением групп цветом
6. Unknown rate по моделям

---

## Справочник существующих модулей

### data_loader.py

```python
load_dataset(stage)      # stage=0: train_after_eda, stage=3: data_after_stage3
load_test_set()          # data_test.csv (343 примера)
split_train_test(df)     # Стратифицированное 80/20
get_class_distribution(df)
TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_SEED = 42
```

### evaluate.py

```python
load_data()              # train(stage3) + test, TF-IDF, label encoding
evaluate_model(name, estimator, X_train, y_train, X_test, y_test, label_names, param_grid)
# Возвращает dict: {"name", "balanced_accuracy", "macro_f1"}, classification_report печатается
```

### stage1_llm_generate.py

```python
generate_class_context(class_name, examples, llm, sampling_params, system_prompt)
# Автогенерация описания класса по его примерам. Возвращает текст 2-4 предложения.
```

### llm_utils.py

```python
load_llm(config_path, pipeline_cfg)          # Загрузка vLLM
generate_batch(llm, sampling_params, prompts) # Батчевая генерация
load_prompt_template(template_name)           # Загрузка промпта из файла
```

### pipeline_config.py

```python
load_pipeline_config(gpu="A100_40")  # Возвращает _DotDict с настройками (этап 2)
# Этап 3 загружает конфиг напрямую: prompt_classifier.load_prompt_config()
```

---

## Чеклист этапа 3

- [ ] `Data/class_descriptions.json` — описания 36 классов через `generate_class_context()`
- [ ] `Data/few_shot_examples.json` — примеры K=1,3,5 с учётом групп A/B/C
- [x] `prompts/classification_prompts/` — три шаблона промптов
- [x] `pipeline_config.json` — секция prompt_classification с моделями и параметрами
- [x] `src/classification/few_shot_examples.py` — подготовка примеров
- [x] `src/classification/prompt_classifier.py` — инференс и extract_prediction
- [ ] Все 5 моделей проверены на загрузку в A100 40GB
- [ ] Длина промптов проверена для каждой комбинации модель × K
- [ ] `extract_prediction()` протестирован вручную
- [x] `evaluate.py` расширен функцией `evaluate_prompt_classification()`
- [ ] Результаты baseline подготовлены для сводной таблицы
- [ ] `results/prompt_results.csv` заполнен
- [ ] `results/all_methods_comparison.csv` собран
- [ ] Визуализации сгенерированы
