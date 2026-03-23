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
│   └── aug_prompts/                     # Промпты для аугментации
│       ├── llm_generate_one.txt         #   Генерация одного письма (этап 1)
│       ├── paraphrase.txt               #   Парафраз (этап 2)
│       ├── class_context.txt            #   Описание класса для промпта
│       ├── judge_score.txt              #   LLM-судья: оценка генерации
│       └── judge_paraphrase.txt         #   LLM-судья: оценка парафраза
├── src/
│   ├── augmentation/
│   │   ├── stage1_llm_generate.py       #   Этап 1: LLM-генерация (< 15 → 15)
│   │   ├── stage2_paraphrase.py         #   Этап 2: парафраз через LLM (< 35 → 35)
│   │   ├── stage3_back_translation.py   #   Этап 3: обратный перевод (< 50 → 50)
│   │   ├── validation.py                #   Фильтры для сгенерированных текстов
│   │   └── llm_utils.py                 #   Обёртка vLLM + LLM-as-a-judge
│   ├── classification/
│   │   ├── evaluate.py                  #   Общая логика оценки классификаторов
│   │   ├── embeddings.py                #   TF-IDF признаки с кэшированием
│   │   ├── rubert_classifier.py         #   Fine-tuning rubert-tiny2 / rubert-base
│   │   ├── run_svm.py                   #   LinearSVC
│   │   ├── run_logreg.py                #   LogisticRegression
│   │   └── run_naive_bayes.py           #   MultinomialNB
│   ├── utils/
│   │   ├── data_loader.py               #   Загрузка данных, чекпоинты, сплит
│   │   ├── data_cleaner.py              #   Предобработка data.json → data_after_eda.csv
│   │   ├── config_loader.py             #   Парсинг JSON-конфигов модели
│   │   └── pipeline_config.py           #   Загрузчик pipeline_config.json + GPU-профили
│   └── augmentation_main.ipynb          # Основной ноутбук (Google Colab)
├── Data/
│   ├── data.json                        # Сырые данные (исходный)
│   ├── data_after_eda.csv               # После предобработки
│   ├── train_after_eda.csv              # Train-часть (после стратифицированного разбиения)
│   ├── data_test.csv                    # Test-часть (не аугментируется)
│   ├── data_after_stage1.csv            # Train после этапа 1
│   ├── data_after_stage2.csv            # Train после этапа 2
│   └── data_after_stage3.csv            # Train после этапа 3
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
| B | 15–34 | 35 | Парафраз через LLM |
| C | 35–49 | 50 | Обратный перевод (NLLB-200) |
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
| NLLB модель | 600M | 3.3B | 3.3B | 3.3B |
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
- `system_prompt` / `paraphrase_system_prompt` — системные промпты для этапов 1 и 2
- `paraphrase_template` — шаблон промпта для этапа 2

---

## Классификация

Модели для оценки качества аугментации. Обучение на аугментированном train, оценка на отложенном test.

| Модуль | Модель |
|--------|--------|
| `run_svm.py` | LinearSVC (TF-IDF) |
| `run_logreg.py` | LogisticRegression (TF-IDF) |
| `run_naive_bayes.py` | MultinomialNB (TF-IDF) |
| `rubert_classifier.py` | rubert-tiny2 / rubert-base (fine-tuning) |

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

# Классификация
python src/classification/run_svm.py
python src/classification/run_logreg.py
python src/classification/run_naive_bayes.py
```
