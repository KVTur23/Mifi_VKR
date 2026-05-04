# Аугментация и классификация корпоративных писем

Пайплайн аугментации несбалансированного датасета входящих электронных писем (36 классов).

Исходный датасет (~1750 писем) сильно несбалансирован: от 1 до 200 примеров на класс.
Три этапа аугментации доводят каждый класс до минимум 50 примеров, после чего модели оценивают эффект.

## Структура

```
code/
├── config_models/
│   ├── aug_configs/                      # JSON-конфиги LLM для аугментации
│   │   ├── model_vllm.json              #   Qwen2.5-14B-Instruct-AWQ (основной)
│   │   ├── model_vllm_32b.json          #   Qwen2.5-32B-Instruct-AWQ
│   │   ├── model_qwen.json              #   Qwen2.5-7B-Instruct
│   │   ├── model_qwen_3b.json           #   Qwen2.5-3B
│   │   └── model_qwen_14b_unsloth.json  #   Qwen2.5-14B unsloth (устаревший)
│   └── pipeline_config.json             # Настройки пайплайна + GPU-профили
├── prompts/
│   ├── aug_prompts/                     # Промпты для аугментации
│   │   ├── llm_generate_one.txt         #   Генерация одного письма (этап 1)
│   │   ├── class_context.txt            #   Описание класса для промпта
│   │   ├── judge_score.txt              #   LLM-судья: оценка генерации
│   │   └── judge_paraphrase.txt         #   LLM-судья: оценка парафраза
│   └── classification_prompts/          # Промпты для prompt-based классификации
│       ├── zero_shot.txt                #   0-shot: список + описания классов
│       ├── one_shot.txt                 #   1-shot: +1 пример первого класса
│       ├── few_shot.txt                 #   K-shot: +блок примеров, с описаниями
│       └── few_shot_no_desc.txt         #   K-shot без описаний классов
├── src/
│   ├── augmentation/
│   │   ├── stage1_llm_generate.py       #   Этап 1: LLM-генерация (< 15 → 15)
│   │   ├── stage2_paraphrase.py         #   Этап 2: ruT5-парафраз (< 35 → 35)
│   │   ├── stage3_back_translation.py   #   Этап 3: chunked back-translation (< 50 → 50)
│   │   ├── rut5_paraphraser.py          #   ruT5-large paraphraser + chunking
│   │   ├── text_chunking.py             #   tokenizer-aware разбиение длинных писем
│   │   ├── validation.py                #   Фильтры для сгенерированных текстов
│   │   └── llm_utils.py                 #   Обёртка vLLM + LLM-as-a-judge
│   ├── classification/
│   │   ├── evaluate.py                  #   Общая логика оценки + метрики prompt-классификации
│   │   ├── embeddings.py                #   TF-IDF признаки с кэшированием
│   │   ├── rubert_classifier.py         #   Fine-tuning rubert-tiny2 / rubert-base
│   │   ├── run_svm.py                   #   LinearSVC
│   │   ├── run_logreg.py                #   LogisticRegression
│   │   ├── run_naive_bayes.py           #   MultinomialNB
│   │   ├── prompt_classifier.py         #   Prompt-based: загрузка LLM, построение промптов, извлечение
│   │   └── few_shot_examples.py         #   Отбор few-shot примеров (K=1/3/5)
│   ├── utils/
│   │   ├── data_loader.py               #   Загрузка данных, чекпоинты, сплит
│   │   ├── data_cleaner.py              #   Предобработка data.json → data_after_eda.csv
│   │   ├── config_loader.py             #   Парсинг JSON-конфигов модели
│   │   └── pipeline_config.py           #   Загрузчик pipeline_config.json + GPU-профили
│   └── augmentation_main.ipynb          # Основной ноутбук (Google Colab)
├── scripts/
│   └── run_few_shot.py                  # CLI-запуск prompt-классификации (матрица модель × эксперимент)
├── notebooks/
│   ├── augmentation.ipynb               # Прогон аугментации
│   └── zer0_one_few_shot.ipynb          # Prompt-классификация интерактивно
├── Data/
│   ├── data.json                        # Сырые данные (исходный)
│   ├── data_after_eda.csv               # После предобработки
│   ├── train_after_eda.csv              # Train-часть (после стратифицированного разбиения)
│   ├── data_test.csv                    # Test-часть (не аугментируется)
│   ├── data_after_stage1.csv            # Train после этапа 1
│   ├── data_after_stage2.csv            # Train после этапа 2
│   ├── data_after_stage3.csv            # Train после этапа 3
│   ├── class_descriptions.json          # Сгенерированные LLM-описания 36 классов (кэш)
│   └── few_shot_examples.json           # Отобранные few-shot примеры для K=1/3/5 + группы A/B/C
├── results/
│   ├── preds_<model>_exp<k>.csv         # Предсказания каждого прогона (text, true, pred, raw, skipped)
│   ├── prompt_results.csv               # Сводка метрик prompt-классификации (инкрементально)
│   ├── classification_results.csv       # Метрики традиционных классификаторов
│   └── all_methods_comparison.csv       # Единая таблица: baseline / augmented / prompt
├── EDA/
│   └── EDA.ipynb                        # Разведочный анализ данных
└── requirements.txt
```

---

## Пайплайн

Перед аугментацией данные разбиваются на train/test (80/20, стратифицированно). Аугментируется только train, оценка — на test.

| Группа | Примеров | Цель | Метод |
|--------|----------|------|-------|
| A | < 15 | 15 | LLM-генерация через vLLM |
| B | 15–34 | 35 | ruT5-парафраз с чанкованием |
| C | 35–49 | 50 | Chunked back-translation (NLLB-200) |
| D | ≥ 50 | — | не трогаем |

Группы пересчитываются после каждого этапа: класс из A, доведённый до 15, попадает в B и участвует в этапе 2.

### Предобработка (data_cleaner.py)

Очистка сырых данных из `data.json`:
- Удаление дубликатов (точных и межклассовых)
- Удаление повторяющихся слов, фраз, предложений
- Обрезка приложений (таблицы, акты, договоры)
- Нормализация пробелов

### Этап 1 — LLM-генерация

Для классов с < 15 примерами модель генерирует новые письма через vLLM (батчевый режим). Контекст класса генерируется автоматически. Генерация с запасом 5x → фильтры → LLM-судья отбирает лучшие.

### Этап 2 — Парафраз

Для классов с < 35 примерами модель переформулирует существующие тексты. Отслеживаются пары (парафраз, оригинал) — LLM-судья сравнивает каждый парафраз с конкретным оригиналом.

### Этап 3 — Обратный перевод

Для классов с < 50 примерами: RU → EN → RU через NLLB-200. NER-плейсхолдеры (`[PERSON]`, `[ORGANIZATION]`) маскируются перед переводом и восстанавливаются после. Две фазы: NLLB-перевод + фильтры → выгрузка → vLLM-судья отбирает лучшие.

Каждый этап сохраняет чекпоинт — прерванный прогон продолжается с того места, где остановился.

### Прогрессивный порог сходства

На первых 2 попытках генерации/парафраза используется строгий порог косинусного сходства (0.95). С 3-й попытки порог повышается на 0.01 за попытку (до 0.98) — если фильтры отсеивают слишком много, даём больше свободы.

---

## LLM-as-a-judge

После валидации фильтрами тексты оцениваются LLM-судьёй по шкале 1-10:

- **Генерация** (`judge_score.txt`) — судья видит 5 примеров класса + описание класса, оценивает естественность, связность, соответствие классу
- **Парафраз/перевод** (`judge_paraphrase.txt`) — судья сравнивает парафраз с конкретным оригиналом, оценивает сохранение смысла и переформулировку

Порог отсечения: 5.0 для генерации/парафраза, 2.5 для обратного перевода (NLLB неизбежно теряет качество).

---

## Валидация (validation.py)

Все сгенерированные тексты проходят фильтры (от дешёвых к дорогим):

1. **Точные дубликаты** — совпадения с существующими и между собой
2. **Минимальная длина** — короче 500 символов
3. **Язык** — отсеиваем не-русский (langdetect)
4. **Вырожденность** — повторяющиеся слова/фразы
5. **Иностранные символы** — CJK-иероглифы (артефакт Qwen)
6. **Промпт-утечка** — мета-фразы, Markdown-разметка, фрагменты промпта
7. **Косинусное сходство** — SBERT (`ai-forever/sbert_large_nlu_ru`), порог 0.95–0.98

---

## GPU-профили и конфигурация

Все настройки пайплайна в `config_models/pipeline_config.json`. В ноутбуке одна строка:

```python
GPU = "A100"  # варианты: "T4", "L4", "A100", "H100"
```

| Параметр | T4 (16GB) | L4 (24GB) | A100 (80GB) | H100 (80GB) |
|----------|-----------|-----------|-------------|-------------|
| NLLB модель | 1.3B | 1.3B | 1.3B | 1.3B |
| NLLB batch | 32 | 32 | 64 | 64 |
| GPU memory | 0.90 | 0.90 | 0.95 | 0.95 |
| enforce_eager | true | true | true | true |

Настройки этапов (target_count, retries, oversample) — тоже в `pipeline_config.json`.

---

## Конфигурация LLM-моделей

Каждая модель — отдельный JSON в `config_models/aug_configs/`. Основные поля:

- `model_name` — модель с HuggingFace (AWQ-квантизация для vLLM)
- `generation_params` — temperature, top_p, top_k, max_new_tokens
- `prompt_template` — шаблон промпта из `prompts/aug_prompts/`
- `system_prompt` — системный промпт для генерации на этапе 1

Параметры ruT5-парафраза и NLLB-чанкования теперь задаются в `pipeline_config.json`.

---

## Классификация

Модели для оценки качества аугментации. Обучение на аугментированном train, оценка на отложенном test.

### Традиционные классификаторы

| Модуль | Модель |
|--------|--------|
| `run_svm.py` | LinearSVC (TF-IDF) |
| `run_logreg.py` | LogisticRegression (TF-IDF) |
| `run_naive_bayes.py` | MultinomialNB (TF-IDF) |
| `rubert_classifier.py` | rubert-tiny2 / rubert-base (fine-tuning) |

---

## Prompt-based классификация

Альтернатива обучению: готовая instruction-tuned LLM получает промпт со списком классов, описаниями и (опционально) примерами, а затем возвращает название подразделения. Ничего не обучается — сравниваем, насколько хорошо модели «из коробки» справляются с задачей, и измеряем вклад описаний и few-shot примеров.

### Модели ([`pipeline_config.json → prompt_classification.prompt_models`](config_models/pipeline_config.json))

| Ключ | HF-модель | Контекст | VRAM | Движок |
|------|-----------|----------|------|--------|
| `saiga_8b` | `IlyaGusev/saiga_llama3_8b` | 8 192 | ~16 GB | transformers, float16 |
| `t_lite_8b` | `t-tech/T-lite-it-1.0` | 32 768 | ~16 GB | transformers, bfloat16 |
| `vikhr_12b` | `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | 32 768 | ~24 GB | transformers, float16 |
| `qwen_14b` | `Qwen/Qwen2.5-14B-Instruct` | 32 768 | ~28 GB | transformers, bfloat16 |
| `qwen_32b` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 32 768 | ~20 GB | vLLM, AWQ, tensor_parallel=2 |

AWQ-модели автоматически грузятся через vLLM (флаг `quantization: "awq"` или `use_vllm: true` в конфиге), остальные — через transformers с `device_map="auto"`. Промпт оборачивается в `chat_template` токенизатора; при его отсутствии передаётся как есть.

Параметры генерации общие для всех моделей (блок `generation_params`):

```json
{"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.1, "max_new_tokens": 50, "do_sample": true}
```

### Режимы (эксперименты)

Каждый режим — кортеж `(K, mode, no_desc, max_example_chars)` в [scripts/run_few_shot.py](scripts/run_few_shot.py):

| Ключ | K | Шаблон | Описания классов | Длина примера | Назначение |
|------|---|--------|------------------|---------------|------------|
| `0` | 0 | `zero_shot.txt` | да | — | Baseline без примеров |
| `1` | 1 | `one_shot.txt` | да | 500 | Один пример первого класса |
| `3` | 3 | `few_shot.txt` | да | 500 | 3 примера на каждый из 36 классов |
| `3nd` | 3 | `few_shot_no_desc.txt` | **нет** | 500 | Проверка вклада описаний при K=3 |
| `5` | 5 | `few_shot.txt` | да | 500 | 5 примеров — у 8B-моделей часто не влезает |
| `5s` | 5 | `few_shot.txt` | да | 300 | Укороченные примеры чтобы влезть в контекст |
| `5snd` | 5 | `few_shot_no_desc.txt` | **нет** | 300 | K=5 короткие без описаний |

Матрица запусков (`DEFAULT_EXPERIMENT_MATRIX`): `saiga_8b` — только `[0, 1]` (контекст 8k); остальные — полный набор `[0, 1, 3, 3nd, 5s, 5snd]`. Режим `5` с длинными примерами помечен, но не входит в матрицу по умолчанию — почти всегда превышает 32k токенов. Перед каждым прогоном промпт для `df_test.iloc[0]` проверяется на длину: если `tokens ≥ max_context − 100`, эксперимент целиком помечается `skipped`.

### Структура промпта

Общий каркас ([`prompts/classification_prompts/`](prompts/classification_prompts/)):

1. Роль: «система маршрутизации корреспонденции нефтегазовой компании».
2. `{class_list}` — отсортированный нумерованный список всех 36 классов.
3. `{class_descriptions}` — по одной строке на класс, обрезка до 200 символов по последнему пробелу (только в `zero_shot`, `one_shot`, `few_shot`).
4. Правила: ровно один вариант из списка, не выдумывать новые, всегда делать выбор.
5. `{examples_block}` — для K≥1: пары `Письмо: … / Подразделение: …` по всем классам, текст обрезается до `max_example_chars` с многоточием.
6. `Письмо: {text}` + `Подразделение:` — якорь для короткого ответа модели.

### Отбор few-shot примеров ([`few_shot_examples.py`](src/classification/few_shot_examples.py))

- Источники: `train_after_eda.csv` (оригинал) и `data_after_stage3.csv` (аугментированный).
- Правило: если у класса **оригинальных** писем ≥ K — берутся только оригиналы; если меньше — добираются из аугментации с дедупликацией по тексту. Приоритет всегда у оригиналов.
- Перемешивание с фиксированным `RANDOM_SEED` — одни и те же примеры между прогонами.
- Результат сохраняется в `Data/few_shot_examples.json` для K=1, 3, 5 и кэшируется между запусками.
- Группы классов по числу **оригинальных** примеров в train: `A ≥ 50`, `B ∈ [15, 49]`, `C < 15`. Используются только для per-group метрик, на отбор примеров не влияют.

### Описания классов

[`ensure_class_descriptions`](scripts/run_few_shot.py) проверяет `Data/class_descriptions.json`. Если файла нет — поднимается vLLM с `model_vllm_32b.json`, для каждого класса берутся первые 5 писем train и генерируется описание через [`generate_class_context`](src/augmentation/stage1_llm_generate.py) (промпт `class_context.txt`). Результат кэшируется, модель после генерации выгружается.

### Извлечение предсказания ([`extract_prediction`](src/classification/prompt_classifier.py))

Ответ модели проходит каскад стратегий (нормализация: lower-case, только `[а-яёa-z0-9 ]`, сжатие пробелов):

1. Отделение первых трёх строк ответа, срезание префиксов «Подразделение:», «Ответ:», «Категория:», «Отдел:», маркеров списка «1.» / «-» / «•», а также кавычек и точек по краям.
2. Точное совпадение нормализованного кандидата с нормализованным названием класса.
3. Название класса содержится в кандидате → берём самое длинное (устойчивость к приставкам «Блок по …»).
4. Кандидат (> 5 символов) содержится в названии класса.
5. Нечёткое совпадение `difflib.get_close_matches` с `fuzzy_cutoff = 0.6`.
6. Fallback: поиск любого названия класса в полном (ненормализованном) ответе.
7. Если ничего не подошло — `unknown`.

Слишком длинный промпт (не влез в контекст) → `skipped=True`, предсказание `unknown`, но в знаменатель метрик такие примеры не попадают.

### Метрики ([`evaluate_prompt_classification`](src/classification/evaluate.py))

- **balanced_accuracy**, **macro_f1** — по всем не-skipped примерам, метки берутся из `set(y_true)` чтобы не потерять редкие классы.
- **unknown_rate** — доля `predicted_label == "unknown"`. Растёт, когда модель уклоняется от ответа или ломает формат.
- **f1_group_{A,B,C}** — macro-F1 в пределах каждой группы. Для группы C показывает, справляется ли модель с редкими классами, у которых в few-shot большинство примеров — аугментированные.
- **skipped / n_test** — сколько примеров выбросили из-за переполнения контекста.

### Артефакты

- `results/preds_<model>_exp<key>.csv` — построчные предсказания: `text, true_label, predicted_label, raw_response, skipped`. Raw-ответ нужен для ручного разбора `unknown`/ошибок в [`log_raw_responses`](scripts/run_few_shot.py).
- `results/prompt_results.csv` — сводка по каждому (model, exp_key): метрики, длина промпта, размер модели. Обновляется после **каждого** эксперимента, что даёт возобновление с места падения: уже посчитанные пары `(model, exp_key)` пропускаются.
- `results/all_methods_comparison.csv` — единая таблица: traditional (`baseline`/`augmented`) + prompt. Строится в конце `run_few_shot.py`, если есть `classification_results.csv`.

### Управление памятью

Перед запуском следующей модели выполняется [`unload_model`](src/classification/prompt_classifier.py): удаление vLLM-движка (`del model.llm`) или transformers-модели, `gc.collect`, `torch.cuda.empty_cache`, `cuda.synchronize`, `sleep(5)` — это обязательно для A100 40GB, иначе следующая 14B/32B-модель не влезает из-за несвободенного кэша.

---

## Запуск

### Google Colab (рекомендуется)

Открыть `src/augmentation_main.ipynb`, выбрать GPU-профиль и запустить все ячейки.

### Локально / CLI

```bash
pip install -r requirements.txt

# Предобработка
python -c "from src.utils.data_cleaner import run; run()"

# Аугментация
python src/augmentation/stage1_llm_generate.py --config config_models/aug_configs/model_vllm.json
python src/augmentation/stage2_paraphrase.py   --config config_models/aug_configs/model_vllm.json
python src/augmentation/stage3_back_translation.py --config config_models/aug_configs/model_vllm.json

# Классификация (традиционная)
python src/classification/run_svm.py
python src/classification/run_logreg.py
python src/classification/run_naive_bayes.py

# Prompt-based классификация
python scripts/run_few_shot.py                              # все модели, все эксперименты из матрицы
python scripts/run_few_shot.py --models qwen_14b            # одна модель, все её эксперименты
python scripts/run_few_shot.py --models qwen_14b qwen_32b \
                               --experiments 0 1 3 3nd      # конкретные модели и режимы
```

Прогон идемпотентен: при повторном запуске из `results/prompt_results.csv` подтягиваются уже завершённые пары `(model, exp_key)` — пересчитываются только недостающие. Чтобы пересчитать эксперимент заново, удалите соответствующую строку из `prompt_results.csv` и файл `preds_<model>_exp<key>.csv`.
