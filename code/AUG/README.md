# AUG — Аугментация датасета для классификации писем

Пайплайн аугментации несбалансированного датасета входящих писем.

Исходный датасет (~1700 писем) сильно несбалансирован: от 2 до 200+ примеров на класс.
Три этапа аугментации доводят каждый класс до минимум 50 примеров, после чего baseline-модели оценивают эффект.

## Структура

```
AUG/
├── configs/                           # JSON-конфиги LLM-моделей
│   ├── model_qwen.json                #   Qwen2.5-7B-Instruct
│   ├── model_qwen_3b.json             #   Qwen2.5-3B
│   └── model_qwen_14b_unsloth.json    #   Qwen2.5-14B unsloth 4-bit (основной)
├── prompts/                           # Шаблоны промптов
│   ├── llm_generate.txt               #   Генерация нескольких писем
│   ├── llm_generate_one.txt           #   Генерация одного письма (этап 1)
│   ├── llm_generate_simple.txt        #   Упрощённый вариант (для 3B)
│   ├── paraphrase.txt                 #   Парафраз (этап 2)
│   └── paraphrase_simple.txt          #   Упрощённый парафраз (для 3B)
├── src/
│   ├── augmentation/
│   │   ├── stage1_llm_generate.py     #   Этап 1: LLM-генерация (< 15 → 15)
│   │   ├── stage2_paraphrase.py       #   Этап 2: парафраз через LLM (< 35 → 35)
│   │   ├── stage3_back_translation.py #   Этап 3: обратный перевод (< 50 → 50)
│   │   ├── validation.py              #   7 фильтров для сгенерированных текстов
│   │   └── llm_utils.py               #   Обёртка для HuggingFace LLM
│   ├── classification/
│   │   ├── evaluate.py                #   Общая логика оценки классификаторов
│   │   ├── embeddings.py              #   SBERT-эмбеддинги с кэшированием
│   │   ├── run_svm.py                 #   Baseline: LinearSVC
│   │   ├── run_logreg.py              #   Baseline: LogisticRegression
│   │   ├── run_naive_bayes.py         #   Baseline: GaussianNB
│   │   └── run_random_forest.py       #   Baseline: RandomForest
│   ├── utils/
│   │   ├── data_loader.py             #   Загрузка данных и чекпоинты
│   │   └── config_loader.py           #   Парсинг JSON-конфигов
│   └── augmentation_main.ipynb        # Ноутбук для Google Colab
├── requirements.txt
└── README.md
```

Данные лежат в `../Data/` (уровнем выше `AUG/`):

```
Data/
├── data_after_eda.csv       # Оригинальный датасет (не трогается)
├── train_after_eda.csv      # Train-часть (после разбиения)
├── data_test.csv            # Test-часть (не аугментируется)
├── data_after_stage1.csv    # Train после этапа 1 (≥ 15)
├── data_after_stage2.csv    # Train после этапа 2 (≥ 35)
└── data_after_stage3.csv    # Train после этапа 3 (≥ 50)
```

---

## Пайплайн

Перед аугментацией данные разбиваются на train/test (80/20, стратифицированно). Аугментируется только train, оценка — на test.

| Группа | Примеров | Цель | Метод |
|--------|----------|------|-------|
| A | < 15 | 15 | LLM-генерация |
| B | 15–34 | 35 | Парафраз через LLM |
| C | 35–49 | 50 | Обратный перевод |
| D | ≥ 50 | — | не трогаем |

Группы пересчитываются после каждого этапа: класс из A, доведённый до 15, попадает в B и участвует в этапе 2.

### Этап 1 — LLM-генерация

Для классов с < 15 примерами модель генерирует новые письма по одному за вызов, ориентируясь на существующие образцы.

### Этап 2 — Парафраз

Для классов с < 35 примерами модель переформулирует существующие тексты: меняет структуру, подбирает синонимы, сохраняет смысл.

### Этап 3 — Обратный перевод

Для классов с < 50 примерами: RU → EN → RU через `facebook/nllb-200-distilled-600M`. Перевод неизбежно меняет формулировки — получается новый текст с тем же смыслом. GPU не требуется.

Каждый этап сохраняет чекпоинт — прерванный прогон продолжается с того места, где остановился.

---

## Валидация (`validation.py`)

Все сгенерированные тексты проходят 7 фильтров (от дешёвых к дорогим):

1. **Точные дубликаты** — совпадения с оригиналами и между собой
2. **Минимальная длина** — короче 20 символов = мусор
3. **Язык** — отсеиваем не-русский (актуально после обратного перевода)
4. **Вырожденность** — повторяющиеся слова/символы
5. **Иностранные символы** — CJK и другие нелатинские/некириллические скрипты
6. **Промпт-утечка** — тексты, начинающиеся с мета-фраз вроде «вот пример письма»
7. **Косинусное сходство** — эмбеддинги через `ai-forever/sbert_large_nlu_ru`, порог 0.95

Если после валидации нужное количество не набрано — генерация повторяется (до 5–10 попыток).

---

## Конфигурация моделей

Каждая модель — отдельный JSON в `configs/`. Основные поля:

- `model_name` — модель с HuggingFace
- `generation_params` — параметры генерации (temperature, top_p и т.д.)
- `prompt_template` — шаблон промпта из `prompts/`
- `use_unsloth` — загрузка через unsloth в 4-bit (для больших моделей)
- `system_prompt` / `paraphrase_system_prompt` — системные промпты для этапов 1 и 2 (содержат инструкцию не использовать Markdown-разметку)
- `paraphrase_template` — шаблон промпта для этапа 2

---

## Классификация

Baseline-модели для оценки качества аугментации. Обучение на аугментированном train, оценка на отложенном test. Общая логика в `evaluate.py`: SBERT-эмбеддинги → GridSearchCV на train (если есть параметры) → метрики на test.

| Модуль | Модель |
|--------|--------|
| `run_svm.py` | LinearSVC |
| `run_logreg.py` | LogisticRegression |
| `run_naive_bayes.py` | GaussianNB |
| `run_random_forest.py` | RandomForestClassifier |

---

## Запуск

```bash
pip install -r requirements.txt
cd AUG/

# Аугментация
python src/augmentation/stage1_llm_generate.py --config configs/model_qwen_14b_unsloth.json
python src/augmentation/stage2_paraphrase.py   --config configs/model_qwen_14b_unsloth.json
python src/augmentation/stage3_back_translation.py

# Классификация
python src/classification/run_svm.py
python src/classification/run_logreg.py
python src/classification/run_naive_bayes.py
python src/classification/run_random_forest.py
```

Для Google Colab — `src/augmentation_main.ipynb`.
