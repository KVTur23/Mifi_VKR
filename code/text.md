# Описание проекта: Пайплайн аугментации и классификации электронных писем

## Обзор

Проект представляет собой пайплайн аугментации и классификации датасета деловых электронных писем на русском языке (36 классов, ~1750 писем). Многие классы содержат мало примеров (от 1 до 200), что делает классификацию затруднительной. Пайплайн доводит все классы минимум до 50 примеров за три стадии аугментации с контролем качества через LLM-судью и 7 валидационных фильтров.

---

## Структура проекта

```
code/
├── config_models/
│   ├── pipeline_config.json              # Конфигурация пайплайна + GPU-профили
│   └── aug_configs/                      # Конфигурации LLM-моделей
│       ├── model_vllm.json               # Qwen2.5-14B-Instruct-AWQ (основная)
│       ├── model_vllm_32b.json           # Qwen2.5-32B-Instruct-AWQ
│       ├── model_qwen.json               # Qwen2.5-7B-Instruct
│       ├── model_qwen_3b.json            # Qwen2.5-3B-Instruct
│       └── model_qwen_14b_unsloth.json   # unsloth-вариант (устаревший)
├── prompts/
│   └── aug_prompts/                      # Шаблоны промптов
│       ├── llm_generate_one.txt          # Генерация нового письма
│       ├── class_context.txt             # Автоописание класса
│       ├── judge_score.txt               # Оценка сгенерированных писем
│       └── judge_paraphrase.txt          # Оценка перефразировок/переводов
├── src/
│   ├── augmentation/                     # Модули аугментации
│   │   ├── __init__.py
│   │   ├── llm_utils.py                  # Обёртка над vLLM + LLM-судья
│   │   ├── stage1_llm_generate.py        # Этап 1: генерация LLM
│   │   ├── stage2_paraphrase.py          # Этап 2: ruT5-парафраз
│   │   ├── stage3_back_translation.py    # Этап 3: chunked обратный перевод
│   │   ├── rut5_paraphraser.py           # ruT5-large paraphraser
│   │   ├── text_chunking.py              # tokenizer-aware chunking
│   │   └── validation.py                 # 7 фильтров валидации
│   ├── classification/                   # Модули классификации
│   │   ├── __init__.py
│   │   ├── embeddings.py                 # TF-IDF с кэшированием
│   │   ├── evaluate.py                   # Общая логика оценки моделей
│   │   ├── rubert_classifier.py          # Файнтюнинг rubert-tiny2
│   │   ├── run_svm.py                    # Бейзлайн: LinearSVC
│   │   ├── run_logreg.py                 # Бейзлайн: LogisticRegression
│   │   └── run_naive_bayes.py            # Бейзлайн: MultinomialNB
│   ├── utils/                            # Утилиты
│   │   ├── __init__.py
│   │   ├── data_loader.py                # Загрузка данных + чекпоинты
│   │   ├── data_cleaner.py               # Предобработка data.json → data_after_eda.csv
│   │   ├── config_loader.py              # Парсинг конфигов моделей
│   │   └── pipeline_config.py            # Загрузка pipeline_config + GPU-профили
│   └── augmentation_main.ipynb           # Основной ноутбук (Colab)
├── Data/                                 # Данные (data.json, чекпоинты, кэши)
├── EDA/
│   └── EDA.ipynb                         # Разведочный анализ данных
├── README.md
└── requirements.txt
```

---

## Модули и функции

---

### `src/augmentation/llm_utils.py` — Обёртка над vLLM и LLM-судья

Центральный модуль для работы с LLM через vLLM. Обеспечивает батчевую генерацию текстов и оценку качества через LLM-судью.

**Константы:**
- `MIN_JUDGE_SCORE = 5.0` — минимальная оценка для принятия текста
- `MAX_JUDGE_EXAMPLES = 5` — максимум примеров для контекста судьи
- `JUDGE_PROMPT = "judge_score.txt"` — промпт для оценки генерации
- `JUDGE_PARAPHRASE_PROMPT = "judge_paraphrase.txt"` — промпт для оценки перефразировок

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_llm(config_path, pipeline_cfg=None)` | Загружает LLM через vLLM из JSON-конфига. Возвращает `(llm, sampling_params, system_prompt)`. GPU-настройки берутся из `pipeline_cfg` (gpu_memory_utilization, enforce_eager). Автоматически определяет AWQ-квантизацию по имени модели. |
| `generate_text(llm, sampling_params, prompt, system_prompt=None)` | Генерация одного текста. Обёртка над `generate_batch`. |
| `generate_batch(llm, sampling_params, prompts, system_prompt=None)` | Батчевая генерация текстов на GPU (все промпты параллельно). При ошибке батча — фоллбэк на поштучную генерацию. Возвращает `list[str | None]`. |
| `score_texts_batch(texts, class_name, llm, existing_texts=None, context="")` | Оценка качества текстов судьёй по шкале 1–10. Сравнивает с существующими примерами и описанием класса. Возвращает отсортированный список `[(text, score)]`. |
| `select_top_half(texts, class_name, llm, n_needed, existing_texts=None, context="")` | Фильтрует через судью, отсекает оценки ниже 5.0, возвращает лучшие `n_needed` текстов. |
| `select_top_paraphrases(paraphrases, originals, class_name, llm, n_needed, min_score=None)` | Судья оценивает перефразировки по парам (оригинал ↔ перефразировка). Порог: 5.0 по умолчанию, 2.5 для этапа 3. Возвращает лучшие `n_needed`. |
| `load_prompt_template(template_name)` | Загружает шаблон промпта из `prompts/aug_prompts/`. |
| `_parse_score(raw)` | Извлекает числовую оценку 1–10 из ответа LLM (regex). По умолчанию 5.0, если не распознано. |

---

### `src/augmentation/stage1_llm_generate.py` — Этап 1: Генерация LLM

Генерирует новые письма для классов с менее чем 15 примерами. LLM получает описание класса и примеры, создаёт новое письмо с нуля.

**Константы:**
- `STAGE = 1`
- `TARGET_COUNT = 15` — целевое количество примеров
- `MAX_RETRIES = 5` — максимум раундов генерации
- `MAX_EXAMPLES_IN_PROMPT = 5` — максимум примеров в промпте
- `OVERSAMPLE_FACTOR = 5` — множитель избыточной генерации

Все константы переопределяются через `pipeline_cfg.stage1.*`.

**Функции:**

| Функция | Описание |
|---------|----------|
| `generate_class_context(class_name, examples, llm, sampling_params, system_prompt=None)` | Автогенерация описания класса по его примерам. Использует промпт `class_context.txt`. LLM анализирует примеры и выдаёт 2–4 предложения о теме, стиле и характерных формулировках. Возвращает первый абзац ответа. |
| `build_prompt(template, class_name, examples, context="")` | Формирует промпт: случайно перемешивает примеры, выбирает до 5, подставляет в шаблон `llm_generate_one.txt` вместе с именем класса и его описанием. |
| `augment_class(class_name, existing_texts, n_needed, llm, sampling_params, prompt_template, system_prompt=None, context="", n_original=None)` | Многораундовая генерация для одного класса. В каждом раунде: генерирует `5 × n_needed` текстов → прогоняет через 7 фильтров → LLM-судья оценивает → отбирает лучшие. Порог косинусного сходства прогрессивно ослабляется: 0.95 (попытки 1–2) → +0.01 за попытку (до 0.98). Повторяет до `MAX_RETRIES` раз. |
| `run(config_path, pipeline_cfg=None)` | Точка входа. Загружает датасет, определяет классы с < 15 примерами, обрабатывает каждый класс, сохраняет чекпоинт `data_after_stage1.csv` после каждого класса. |

**Логика раунда генерации:**
1. Сформировать промпт с примерами
2. Сгенерировать `OVERSAMPLE_FACTOR × n_needed` текстов (батчево)
3. Провести через 7 фильтров валидации
4. Оценить LLM-судьёй (порог 5.0)
5. Добавить прошедшие тексты в пул
6. Повторить, если не хватает

---

### `src/augmentation/stage2_paraphrase.py` — Этап 2: ruT5-парафраз

Перефразирует оригинальные тексты для классов с 15–34 примерами, доводя до 35.
Использует `fyaronskiy/ruT5-large-paraphraser`; длинные письма режутся на
tokenizer-aware чанки и собираются обратно перед валидацией.

**Константы:**
- `STAGE = 2`
- `TARGET_COUNT = 35`
- `MAX_RETRIES = 5`
- `OVERSAMPLE_FACTOR = 5`
- `RUT5_JUDGE_MIN_SCORE = 4.5`

**Функции:**

| Функция | Описание |
|---------|----------|
| `_select_sources(existing_texts, n_needed)` | Выбирает оригиналы для перефразирования round-robin: каждый оригинал используется примерно одинаковое количество раз. |
| `augment_class(class_name, existing_texts, n_needed, paraphraser, n_original=None, paraphrase_sources=None)` | Набирает валидированный пул ruT5-парафразов. Прогрессивный порог сходства: 0.95 → 0.98. |
| `run(config_path, pipeline_cfg=None)` | Точка входа. Сначала грузит ruT5 и собирает пулы кандидатов, затем выгружает ruT5, грузит vLLM-судью и отбирает лучшие пары с порогом 4.5. |

---

### `src/augmentation/stage3_back_translation.py` — Этап 3: Обратный перевод

Обратный перевод (RU → pivot → RU) через NLLB-200 для классов с 35–49 примерами, доводит до 50. Двухфазная архитектура: сначала chunked перевод + валидация, затем LLM-судья.

**Константы:**
- `STAGE = 3`
- `TARGET_COUNT = 50`
- `MAX_RETRIES = 20`
- `MODEL_NLLB = "facebook/nllb-200-1.3B"` — модель перевода
- `BATCH_SIZE = 64` — размер батча NLLB
- `OVERSAMPLE_FACTOR = 3`
- `MIN_JUDGE_SCORE_STAGE3 = 2.5` — порог судьи (ниже чем на этапах 1–2)
- `LANG_RU = "rus_Cyrl"`, `LANG_EN = "eng_Latn"` — коды языков
- `TRANSLATION_CHUNK_MAX_TOKENS = 900` — максимальная длина чанка
- `TRANSLATION_OUTPUT_MAX_TOKENS = 1024` — максимум токенов выхода на чанк
- `_PAIRS_CSV = DATA_DIR / "_stage3_pairs_cache.csv"` — промежуточный кэш пар

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_translation_models()` | Загружает NLLB-200 (размер зависит от GPU-профиля). Возвращает `(model, tokenizer, device)`. |
| `unload_from_gpu(*objects)` | Выгружает модели из GPU, вызывает `gc.collect()` и `torch.cuda.empty_cache()`. |
| `load_sbert_on_gpu()` | Загружает SBERT (`ai-forever/sbert_large_nlu_ru`) для валидации. |
| `translate_batch(texts, input_token_counts, model, tokenizer, src_lang, tgt_lang, device)` | Батчевый перевод чанков через NLLB без truncation. |
| `translate_texts_chunked(texts, model, tokenizer, src_lang, tgt_lang, device)` | Разбивает полные письма на tokenizer-aware чанки, переводит и собирает обратно. |
| `back_translate(texts, model, tokenizer, device)` | Пайплайн RU → pivot → RU через chunked перевод. Плейсхолдеры не маскируются и не проверяются. |
| `select_sources(existing_texts, n_needed)` | Round-robin выбор оригиналов для перевода. |
| `run(config_path, pipeline_cfg=None)` | Точка входа, двухфазная обработка (см. ниже). |

**Двухфазная архитектура:**

**Фаза 1 — Перевод + валидация (NLLB на GPU):**
1. Загрузить NLLB-200 и SBERT
2. Для каждого класса с 35–49 примерами:
   - Выбрать оригиналы round-robin
   - Перевести RU → pivot → RU через чанки
   - Прогнать через 7 фильтров валидации
   - Собрать ВСЕ прошедшие пары (без ограничения пула), но остановить перевод при 2× n_needed
   - Повторять до `MAX_RETRIES` (20) раз
3. Сохранить пары в `_stage3_pairs_cache.csv` (промежуточный кэш, переживает рестарт Colab)

**Фаза 2 — LLM-судья (vLLM на GPU):**
1. Выгрузить NLLB и SBERT из GPU
2. Загрузить vLLM
3. Судья оценивает каждую пару (оригинал ↔ перевод), порог 2.5
4. Отобрать лучшие `n_needed` текстов по рейтингу
5. Сохранить `data_after_stage3.csv`
6. Переименовать кэш в `.bak.csv` (не удалять)

---

### `src/augmentation/validation.py` — 7 фильтров валидации

Последовательная фильтрация сгенерированных текстов. Каждый фильтр — отдельная функция, прогоняется в цепочке.

**Константы:**
- `SIMILARITY_THRESHOLD = 0.95` — верхний порог сходства (слишком похоже = копия)
- `SIMILARITY_THRESHOLD_LOW = 0.98` — мягкий порог для классов с 1 оригиналом
- `SIMILARITY_THRESHOLD_MIN = 0.5` — нижний порог (слишком далеко = мусор)
- `MIN_TEXT_LENGTH = 500` — минимальная длина текста в символах
- `SBERT_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"` — модель эмбеддингов

**Фильтры (в порядке применения):**

| # | Функция | Что делает |
|---|---------|-----------|
| 1 | `remove_exact_duplicates(new_texts, existing_texts, class_name)` | Убирает точные дубликаты (нормализация: strip + lower). Проверяет и против существующих текстов, и внутри новых. |
| 2 | `filter_short_texts(texts, class_name, min_length=500)` | Убирает тексты короче 500 символов. |
| 3 | `filter_non_russian(texts, class_name)` | Определяет язык через langdetect. Оставляет только русские тексты. |
| 4 | `filter_degenerate(texts, class_name)` | Проверяет `_is_degenerate()`: доля уникальных слов < 0.2 или фраза из 3+ слов повторяется 3+ раза. |
| 5 | `filter_foreign_scripts(texts, class_name)` | Убирает тексты с иероглифами: китайские (U+4E00–U+9FFF), хирагана, катакана, хангыль. |
| 6 | `filter_prompt_leak(texts, class_name)` | Обнаруживает утечку промпта (LLM описала задачу вместо генерации письма). Три уровня: маркеры начала ("конечно,", "генерирую"), маркеры в тексте ("напиши пример письма"), regex-паттерны (Markdown-разметка, нумерованные письма). |
| 7 | `filter_by_cosine_similarity(new_texts, existing_texts, class_name, sbert_model, threshold=0.95, threshold_min=0.5)` | Косинусное сходство через SBERT. Верхний порог (0.95): текст слишком похож на существующий (копия) → убрать. Нижний порог (0.5): текст слишком далёк (повреждён) → убрать. |

**Главная функция:**

| Функция | Описание |
|---------|----------|
| `validate_generated_texts(new_texts, existing_texts, class_name, similarity_threshold=0.95, min_length=500, sbert_model=None, n_original=None)` | Прогоняет тексты через все 7 фильтров последовательно. Автоматически загружает SBERT при первом вызове (кэш на уровне модуля). Возвращает отфильтрованный список. |
| `get_sbert_model()` | Ленивая загрузка SBERT на CPU (кэшируется глобально). |

---

### `src/utils/data_loader.py` — Загрузка данных и чекпоинты

Единая точка доступа к данным. Управляет чекпоинтами (файлами промежуточных этапов) и стратифицированным разбиением.

**Константы:**
- `TEXT_COL = "text"`, `LABEL_COL = "label"` — имена колонок
- `RANDOM_SEED = 42` — глобальный seed для воспроизводимости
- `DATA_DIR = Path(__file__).parent.parent.parent / "Data"` — путь к папке данных
- `STAGE_FILES = {3: "data_after_stage3.csv", 2: "data_after_stage2.csv", 1: "data_after_stage1.csv", 0: "train_after_eda.csv"}`

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_dataset(stage, data_dir=None)` | Загружает чекпоинт для этапа `stage` (0–3). Фоллбэк: если файл этапа N отсутствует, пробует N-1, N-2, ... до 0. Валидирует наличие колонок text/label. |
| `save_checkpoint(df, stage, data_dir=None)` | Сохраняет DataFrame в `data_after_stageN.csv`. Только для этапов 1–3. |
| `get_class_distribution(df)` | Возвращает `pd.Series` — value_counts по классам (по убыванию). |
| `get_classes_to_augment(df, min_count, max_count)` | Возвращает `{class_name: count}` для классов с `min_count ≤ count < max_count`. |
| `split_train_test(df, test_size=0.2, data_dir=None)` | Стратифицированное разбиение 80/20, гарантирует ≥1 примера каждого класса в каждой выборке. Использует `sorted()` для детерминированного порядка классов. Сохраняет `train_after_eda.csv` и `data_test.csv`. |
| `load_test_set(data_dir=None)` | Загружает тестовую выборку `data_test.csv`. |

---

### `src/utils/data_cleaner.py` — Предобработка данных

Извлечённая логика из EDA.ipynb. Преобразует сырой `data.json` в чистый `data_after_eda.csv`.

**Функции очистки текста:**

| Функция | Описание |
|---------|----------|
| `remove_repeated_words(text)` | Убирает подряд идущие одинаковые слова: «работа работа работа» → «работа». |
| `remove_repeated_sequences(text, min_words=5, max_words=30)` | Убирает длинные повторяющиеся куски (фраза из 5–30 слов, повторённая многократно). |
| `remove_duplicate_lines(text)` | Убирает повторяющиеся строки (нормализация: strip + lower). |
| `remove_duplicate_sentences(text)` | Убирает одинаковые предложения (разбиение по `.!?`). |
| `remove_repeated_comma_phrases(text)` | Чистит повторы через запятую (фраза 20–200 символов, повторённая 2+ раза). |
| `remove_cycling_numbered_lines(text)` | Убирает циклы пронумерованных строк (1. текст, 2. текст, 3. тот же текст...). Если уникальных строк < половины всех — был цикл. |
| `remove_incremental_list(text)` | Сжимает списки вида «слово-1, слово-2, ... слово-100» → «слово-1 ... слово-100». |
| `trim_attached_documents(text, max_len=4000)` | Обрезает письма с приложенными договорами, актами, таблицами. Ищет триггеры («Предмет договора», «Акт сверки», «Публичная оферта» и т.п.) и обрезает текст до первого найденного. |
| `normalize_text(text)` | Нормализация пробелов: множественные пробелы → один, множественные пустые строки → одна. |
| `clean_text(text)` | Цепочка всех функций очистки в правильном порядке. |

**Функции удаления дубликатов и аномалий:**

| Функция | Описание |
|---------|----------|
| `remove_duplicates(df)` | 1) Удаляет точные дубликаты строк (`drop_duplicates`). 2) Удаляет межклассовые дубликаты: один и тот же текст в разных классах — удаляется из «Блок финансового директора» (логика из EDA.ipynb). |
| `remove_anomalous_texts(df)` | Удаляет аномальные письма: `unique_word_count < 50` И `word_count > 200` (много слов, но почти все повторяются — мусорные тексты). |

**Главная функция:**

| Функция | Описание |
|---------|----------|
| `run(data_dir=None)` | Полный цикл: загрузить `data.json` → удалить лишние колонки → удалить дубликаты → удалить аномалии → очистить тексты → удалить пустые → сохранить `data_after_eda.csv`. Выводит статистику по классам. |

---

### `src/utils/config_loader.py` — Парсинг конфигов моделей

Загрузка и валидация JSON-конфигов LLM-моделей.

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_model_config(config_path)` | Парсит JSON-конфиг модели. Валидирует обязательные поля: `model_name`, `generation_params`, `prompt_template`. Разрешает относительные пути промптов (добавляет `prompts/aug_prompts/`). |
| `load_prompt(prompt_path)` | Читает текстовый файл промпта и возвращает содержимое. |

---

### `src/utils/pipeline_config.py` — Конфигурация пайплайна и GPU-профили

Загрузка централизованной конфигурации пайплайна с выбором GPU-профиля.

**Классы:**

| Класс | Описание |
|-------|----------|
| `_DotDict(dict)` | Словарь с доступом через точку: `cfg.stage1.target_count` вместо `cfg["stage1"]["target_count"]`. Рекурсивно оборачивает вложенные словари. |

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_pipeline_config(gpu="L4")` | Загружает `pipeline_config.json`, выбирает GPU-профиль (T4/L4/A100_40/A100_80/H100). Возвращает `_DotDict` с секциями: `gpu`, `stage1`, `stage2`, `stage3`, `judge`, `validation`, `training`. Кэширует результат. |

---

### `src/classification/embeddings.py` — TF-IDF эмбеддинги с кэшированием

Построение TF-IDF признаков для классических ML-моделей. Кэширование на диск для ускорения повторных запусков.

**Константы:**
- `TFIDF_PARAMS = {max_features: 50000, ngram_range: (1, 2), sublinear_tf: True, min_df: 2}`
- `CACHE_DIR = PROJECT_ROOT / "Data" / ".tfidf_cache"`

**Функции:**

| Функция | Описание |
|---------|----------|
| `build_vectorizer(**kwargs)` | Создаёт `TfidfVectorizer` с параметрами по умолчанию (50k признаков, биграммы, sublinear_tf). |
| `prepare_features(df_train, df_test, text_col, label_col, use_cache=True, **tfidf_kwargs)` | Обучает TF-IDF на train, трансформирует train + test. Кэширует результат (ключ — MD5-хэш текстов). Возвращает `(X_train, y_train, X_test, y_test)`. |
| `_get_cache_key(texts_train, texts_test)` | MD5-хэш fingerprint текстов (длины + первые/последние 100 символов). |
| `_load_from_cache(texts_train, texts_test)` | Загрузка из .npz файлов, если кэш валиден. |
| `_save_to_cache(texts_train, texts_test, X_train, X_test, vectorizer)` | Сохранение матриц (.npz) и векторайзера (.pkl). |

---

### `src/classification/evaluate.py` — Общая логика оценки моделей

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_data()` | Загружает train (этап 3) + test, строит TF-IDF признаки, кодирует метки. Возвращает `(X_train, y_train, X_test, y_test, label_names)`. |
| `evaluate_model(name, estimator, X_train, y_train, X_test, y_test, label_names, param_grid=None)` | Если `param_grid` задан — запускает `GridSearchCV` со `StratifiedKFold(3)`, оптимизация по `f1_macro`. Вычисляет: `balanced_accuracy`, `macro_f1`, `classification_report`. |

---

### `src/classification/rubert_classifier.py` — Файнтюнинг RuBERT-tiny2

**Константы:**
- `DEFAULT_MODEL = "cointegrated/rubert-tiny2"`
- `MAX_LENGTH = 256`, `BATCH_SIZE = 32`, `LEARNING_RATE = 5e-4`, `NUM_EPOCHS = 15`

**Классы:**

| Класс | Описание |
|-------|----------|
| `TextDataset(Dataset)` | PyTorch Dataset: токенизирует текст (max_length=256), возвращает `{input_ids, attention_mask, labels}`. |

**Функции:**

| Функция | Описание |
|---------|----------|
| `train_and_evaluate(df_train, df_test, text_col, label_col, model_name, num_epochs, batch_size, lr, name)` | Полный цикл: загрузка модели → создание DataLoader → обучение (AdamW, linear warmup, gradient clipping) → оценка на тесте. Возвращает `{name, balanced_accuracy, macro_f1}`. |

---

### `src/classification/run_svm.py` — Бейзлайн: LinearSVC

- Модель: `LinearSVC(max_iter=10000, random_state=42, dual="auto")`
- Гиперпараметры: `C ∈ [0.01, 0.1, 1, 10]`

### `src/classification/run_logreg.py` — Бейзлайн: LogisticRegression

- Модель: `LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)`
- Гиперпараметры: `C ∈ [0.01, 0.1, 1, 10]`

### `src/classification/run_naive_bayes.py` — Бейзлайн: MultinomialNB

- Модель: `MultinomialNB()`
- Гиперпараметры: `alpha ∈ [0.01, 0.1, 0.5, 1.0]`

Все три используют `evaluate.load_data()` и `evaluate.evaluate_model()`.

---

## Конфигурация

### `config_models/pipeline_config.json`

**GPU-профили:**

| Профиль | VRAM | NLLB Batch | RuBERT Batch | Memory Util | enforce_eager | NLLB-модель |
|---------|------|-----------|-------------|-------------|---------------|-------------|
| T4 | 16 ГБ | 32 | 16 | 0.90 | true | nllb-200-1.3B |
| L4 | 24 ГБ | 32 | 32 | 0.90 | true | nllb-200-1.3B |
| A100_40 | 40 ГБ | 64 | 64 | 0.92 | true | nllb-200-1.3B |
| A100_80 | 80 ГБ | 64 | 64 | 0.95 | true | nllb-200-1.3B |
| H100 | 80 ГБ | 64 | 64 | 0.95 | true | nllb-200-1.3B |

**Настройки пайплайна по этапам:**

| Параметр | Этап 1 | Этап 2 | Этап 3 |
|----------|--------|--------|--------|
| target_count | 15 | 35 | 50 |
| max_retries | 5 | 5 | 20 |
| oversample_factor | 5 | 5 | 3 |
| min_judge_score | 5.0 | 5.0 | 2.5 |

### Конфиги моделей (`config_models/aug_configs/`)

Каждый JSON содержит:
- `model_name` — HuggingFace идентификатор модели
- `max_seq_length` — максимальная длина контекста
- `system_prompt` — системный промпт для генерации
- `generation_params` — параметры генерации (temperature, top_p, top_k, max_new_tokens, repetition_penalty)

---

## Промпты (`prompts/aug_prompts/`)

| Файл | Назначение |
|------|-----------|
| `llm_generate_one.txt` | Этап 1: промпт для генерации нового письма. Включает название класса, его описание и примеры реальных писем. |
| `class_context.txt` | Автогенерация описания класса: LLM анализирует примеры и описывает тему, стиль, структуру и характерные формулировки. |
| `judge_score.txt` | Оценка сгенерированных писем (1–10). Критерии: естественность, связность, соответствие классу, полнота. |
| `judge_paraphrase.txt` | Оценка перефразировок и переводов (1–10). Критерии: сохранение смысла, реальность переформулировки, естественность, полнота. |

---

## Общий поток данных

```
data.json (~1774 письма, 36 классов)
    │
    ▼
data_cleaner.run()
  ├─ Удаление дубликатов (точных + межклассовых)
  ├─ Удаление аномалий (unique_words < 50 & words > 200)
  ├─ 8 функций очистки текста
  └─ Удаление пустых после очистки
    │
    ▼
data_after_eda.csv (~1750 писем)
    │
    ▼
split_train_test() — стратифицированное 80/20
    │                       │
    ▼                       ▼
train_after_eda.csv    data_test.csv (фиксирован)
    │
    ▼
Бейзлайн: SVM / LogReg / NB / RuBERT-tiny2
    │
    ▼
Этап 1: LLM-генерация (классы < 15 → 15)
  └─ Генерация → 7 фильтров → LLM-судья (≥ 5.0)
    │
    ▼
data_after_stage1.csv (все классы ≥ 15)
    │
    ▼
Этап 2: Перефразирование (классы 15–34 → 35)
  └─ Перефразировка → 7 фильтров → LLM-судья (≥ 5.0)
    │
    ▼
data_after_stage2.csv (все классы ≥ 35)
    │
    ▼
Этап 3: Обратный перевод (классы 35–49 → 50)
  ├─ Фаза 1: NLLB RU→EN→RU + маскирование NER + 7 фильтров
  │    └─ Промежуточный кэш: _stage3_pairs_cache.csv
  └─ Фаза 2: LLM-судья (≥ 2.5)
    │
    ▼
data_after_stage3.csv (все классы ≥ 50)
    │
    ▼
Классификация на аугментированных данных
    │
    ▼
Сравнение метрик: бейзлайн vs аугментация
```

---

## Ключевые архитектурные решения

1. **Система чекпоинтов** — каждый этап сохраняет промежуточный результат; при перезапуске загружается последний доступный чекпоинт.

2. **Многораундовая генерация** — в каждом раунде: избыточная генерация (3–5×) → фильтрация → оценка судьёй → отбор лучших. Повторяется до достижения цели или исчерпания попыток.

3. **Прогрессивный порог сходства** — начинает со строгого порога (0.95), постепенно ослабляет до 0.98 с каждой попыткой. Позволяет принимать чуть более похожие тексты для трудных классов.

4. **Двухфазная архитектура этапа 3** — NLLB и vLLM не помещаются в GPU одновременно. Сначала все переводы + фильтрация, сохранение в кэш; затем выгрузка NLLB, загрузка vLLM для судьи.

5. **Маскирование NER-плейсхолдеров** — NLLB-200 портит `[PERSON]` → `[POIRSON]`. Замена на `<0>`, `<1>` перед переводом с восстановлением после.

6. **LLM-как-судья** — два промпта: для генерации (сравнение с примерами класса) и для перефразировок (сравнение с оригиналом). Низкая температура (0.1) для стабильности оценок.

7. **GPU-профили** — одна настройка `GPU = "A100_40"` автоматически подбирает batch size, модель NLLB, memory utilization и enforce_eager.

8. **Кэширование TF-IDF** — MD5-хэш текстов как ключ кэша. При повторном запуске с теми же данными пропускает обучение векторайзера.
