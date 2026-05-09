# Аугментация и классификация корпоративных писем

Пайплайн аугментации несбалансированного датасета входящих электронных писем (36 классов).

Исходный датасет (~1750 писем) сильно несбалансирован: от 1 до 200 примеров на класс. Три
этапа аугментации доводят каждый класс до минимум 50 примеров, после чего три семейства
классификаторов (классические модели на TF-IDF, prompt-based LLM, PEFT fine-tune больших
LLM) оценивают эффект.

## Структура

```
code/
├── config_models/
│   ├── aug_configs/                       # JSON-конфиги LLM для аугментации
│   │   ├── model_vllm_32b.json            #   Qwen2.5-32B-Instruct-AWQ (основной)
│   │   ├── model_vllm.json                #   Qwen2.5-14B-Instruct-AWQ
│   │   ├── model_qwen.json                #   Qwen2.5-7B-Instruct
│   │   ├── model_qwen_3b.json             #   Qwen2.5-3B
│   │   └── model_qwen_14b_unsloth.json    #   unsloth-вариант (legacy)
│   ├── finetune_configs/                  # JSON-конфиги PEFT-экспериментов (один файл = один прогон)
│   │   ├── qwen3_32b_qlora_cw.json        #   QLoRA r=16, class_weights ON
│   │   ├── qwen3_32b_qlora_no_cw.json     #   QLoRA r=16, class_weights OFF
│   │   ├── qwen3_32b_qlora_cw_r32.json    #   QLoRA r=32, class_weights ON
│   │   ├── qwen3_32b_qlora_no_cw_r32.json #   QLoRA r=32, class_weights OFF
│   │   ├── ruadapt_qwen3_32b_qlora_cw*    #   аналог для RuadaptQwen3-32B
│   │   ├── vistral_24b_qlora_cw*          #   аналог для Vistral-24B
│   │   ├── tpro_it_21_*                   #   T-pro-it-2.1 (qlora / adalora)
│   │   ├── qwen3_14b_lora.json            #   LoRA r=16 на 14B
│   │   ├── qwen3_14b_tinylora.json        #   tiny-LoRA
│   │   └── qwen3_32b_adalora.json         #   AdaLoRA
│   └── pipeline_config.json               # GPU-профили + настройки этапов + finetune-таблица
├── prompts/
│   ├── aug_prompts/                       # Промпты для аугментации
│   │   ├── llm_generate_one.txt           #   Этап 1: генерация одного письма
│   │   ├── paraphrase.txt                 #   Этап 2: парафраз
│   │   ├── class_context.txt              #   Описание класса для промпта
│   │   ├── judge_score.txt                #   LLM-судья: оценка генерации
│   │   └── judge_paraphrase.txt           #   LLM-судья: оценка парафраза
│   └── classification_prompts/            # Промпты для prompt-based классификации
│       ├── zero_shot.txt
│       ├── one_shot.txt
│       ├── few_shot.txt
│       └── few_shot_no_desc.txt
├── src/
│   ├── augmentation/
│   │   ├── stage1_llm_generate.py         #   Этап 1: LLM-генерация (<15 → 15)
│   │   ├── stage2_paraphrase.py           #   Этап 2: парафраз через LLM (<35 → 35)
│   │   ├── stage3_back_translation.py     #   Этап 3: обратный перевод (<50 → 50)
│   │   ├── validation.py                  #   7 фильтров для сгенерированных текстов
│   │   └── llm_utils.py                   #   Обёртка vLLM + LLM-as-a-judge
│   ├── classification/
│   │   ├── embeddings.py                  #   TF-IDF признаки
│   │   ├── evaluate.py                    #   evaluate_model + load_data + few-shot метрики
│   │   ├── rubert_classifier.py           #   Fine-tuning rubert-tiny2 / rubert-base
│   │   ├── run_svm.py                     #   LinearSVC (CLI)
│   │   ├── run_logreg.py                  #   LogisticRegression (CLI)
│   │   ├── run_naive_bayes.py             #   MultinomialNB (CLI)
│   │   ├── prompt_classifier.py           #   Prompt-based: загрузка LLM + извлечение ответа
│   │   └── few_shot_examples.py           #   Отбор few-shot примеров (K=1/3/5)
│   ├── finetune/
│   │   ├── trainer_base.py                #   SeqClsRunner: сборка модели + Trainer + eval
│   │   ├── data_prep.py                   #   compute_class_weights, group_by_class
│   │   ├── peft_utils.py                  #   load_base_model, build_peft_config, save_adapter
│   │   └── evaluate_finetuned.py          #   Инференс адаптера на test, метрики, per-run JSON
│   └── utils/
│       ├── data_loader.py                 #   Чекпоинты, train/test split, helpers
│       ├── data_cleaner.py                #   data.json → data_after_eda.csv
│       ├── config_loader.py               #   JSON-конфиги моделей
│       └── pipeline_config.py             #   Загрузчик pipeline_config.json + GPU-профили
├── scripts/
│   ├── augmentation.py                    # CLI: полный аугментационный пайплайн (этапы 1-3 + метрики)
│   ├── run_finetune.py                    # CLI: один PEFT-прогон по конфигу
│   └── run_few_shot.py                    # CLI: матрица prompt-classification (модели × режимы)
├── notebooks/
│   ├── augmentation.ipynb                 # Интерактивный прогон аугментации (Colab)
│   ├── finetune.ipynb                     # Интерактивный fine-tune через CLI (Colab)
│   ├── aggregate_results.ipynb            # Сборка results/finetune/*.json + top-N графики
│   └── zer0_one_few_shot.ipynb            # Prompt-классификация интерактивно
├── Data/
│   ├── data.json                          # Сырые данные
│   ├── data_after_eda.csv                 # После предобработки
│   ├── train_after_eda.csv                # Train (стратифицированно 80/20)
│   ├── data_test.csv                      # Test (не аугментируется)
│   ├── data_after_stage{1,2,3}.csv        # Чекпоинты после каждого этапа аугментации
│   ├── _stage3_pairs_cache.csv            # Кэш валидных пар BT (для рестарта)
│   ├── class_descriptions.json            # LLM-сгенерированные описания 36 классов (кэш)
│   └── few_shot_examples.json             # Отобранные few-shot примеры
├── results/
│   ├── classification_results.csv         # Метрики традиционных классификаторов (baseline + augmented)
│   ├── prompt_results.csv                 # Сводка prompt-classification (инкрементально)
│   ├── preds_<model>_exp<k>.csv           # Per-sample предсказания few-shot
│   ├── finetune/<run_key>.json            # Один файл на PEFT-прогон с метриками
│   ├── preds_<run_key>.csv                # Per-sample предсказания PEFT-прогона
│   └── all_methods_comparison.csv         # Сводная таблица по всем методам
├── EDA/
│   └── EDA.ipynb                          # Разведочный анализ
└── requirements.txt
```

---

## 1. Аугментация

Перед аугментацией данные разбиваются на train/test (80/20, стратифицированно). Аугментируется
**только train**, оценка метрик — **на test**.

| Группа | Примеров | Цель | Метод |
|--------|----------|------|-------|
| C      | < 15     | 15   | LLM-генерация через vLLM |
| B      | 15-34    | 35   | Парафраз через LLM (тот же vLLM) |
| -      | 35-49    | 50   | Обратный перевод через NLLB-200 |
| A      | ≥ 50     | -    | не трогаем |

Группы пересчитываются после каждого этапа: класс из C, доведённый до 15, попадает в B и
участвует в этапе 2.

### Предобработка ([`src/utils/data_cleaner.py`](src/utils/data_cleaner.py))

Очистка сырых данных из `data.json`:
- Удаление точных и межклассовых дубликатов
- Удаление повторяющихся слов / фраз / предложений
- Обрезка приложений (таблицы, акты, договоры) по эвристикам
- Нормализация пробелов

### Этап 1 - LLM-генерация ([`stage1_llm_generate.py`](src/augmentation/stage1_llm_generate.py))

Для классов с менее чем 15 примерами модель генерирует новые письма через vLLM (батчевый
режим). Перед генерацией для класса один раз создаётся **описание класса** через
`generate_class_context` (промпт `class_context.txt`) - оно подставляется в каждый промпт
вместе с несколькими случайными существующими примерами. Это помогает модели "понять"
о чём класс.

Схема одного раунда: генерация с запасом `OVERSAMPLE_FACTOR=5` от нужного количества →
фильтры валидации → LLM-судья (`judge_score.txt`) → берём прошедшие. Если не хватило,
повторяем до `MAX_RETRIES=5` раз.

Особенность: для классов с очень малым числом источников (`<= small_class_source_threshold`)
автоматически поднимается температура генерации (`small_class_temperature = 0.9`) - модель
быстро упирается в "почти копии" если ей дать только 1-3 примера.

### Этап 2 - Парафраз ([`stage2_paraphrase.py`](src/augmentation/stage2_paraphrase.py))

Для классов с < 35 примерами модель переформулирует **оригиналы** (из `train_after_eda.csv`,
не из синтетики этапа 1 - чтобы не накапливать каскад артефактов). Парафраз использует
**тот же vLLM-движок** что и этап 1, но с отдельным `paraphrase_system_prompt` и шаблоном
`paraphrase.txt` из конфига модели.

Каждый парафраз сохраняется в паре с **его конкретным оригиналом** - чтобы потом (если
включён судья) сравнивать именно эту пару, а не парафраз с произвольным письмом класса.

LLM-судья на этапе 2 **по умолчанию выключен** (`use_judge: false` в `pipeline_config.json`):
фильтры валидации обычно достаточны, а судья на парафразах добавляет ~30-60s на батч.

### Этап 3 - Обратный перевод ([`stage3_back_translation.py`](src/augmentation/stage3_back_translation.py))

Для классов с < 50 примерами: RU → промежуточный язык → RU через NLLB-200. Используются три
промежуточных языка (`eng_Latn`, `deu_Latn`, `fra_Latn`) и несколько кругов с прогрессивным
ослаблением порога косинусного сходства - чтобы добрать упёртые классы.

NLLB и SBERT грузятся **один раз** в начале этапа и держатся до конца (раньше была
выгрузка между промежуточными языками для освобождения VRAM под судью; сейчас судья по
умолчанию отключён, и эта пляска убрана - экономия 7-9 минут на этап).

LLM-судья на stage3 **по умолчанию выключен** (`use_judge: false`) - на NLLB-выходах он
стабильно занижает оценки и режет ~92% и без того прошедшего фильтры пула. Без судьи
кандидаты после фильтров берутся через `random.shuffle(...)[:n_needed]`.

Каждый этап сохраняет чекпоинт - прерванный прогон продолжается с того места, где
остановился. Кэш валидных пар BT (`_stage3_pairs_cache.csv`) тоже переживает рестарт.

### Прогрессивный порог сходства

Косинусный фильтр SBERT в начале строгий (порог 0.95), а потом постепенно ослабляется -
если фильтры режут слишком много, даём больше свободы:

- **Этапы 1 и 2** - per-attempt: первые `similarity_increase_after_attempt = 2` попыток порог
  держится на `validation.similarity_threshold = 0.95`, дальше каждую попытку растёт на
  `similarity_step = 0.01`, потолок `validation.similarity_threshold_max = 0.98`.
- **Этап 3** - per-round: на каждом круге порог растёт на `similarity_step = 0.10`. На 2-м
  круге уже `1.05` (выше 1.0 → косинусный фильтр фактически отключён, так как косинус
  ограничен 1.0 - это намеренно, чтобы добрать хвост).

### LLM-as-a-judge

Когда судья включён, тексты после фильтров оцениваются по шкале 1-10:

- **Генерация** (`judge_score.txt`) - судья видит несколько примеров класса + описание
  класса, оценивает естественность, связность, соответствие классу. Порог по умолчанию 5.0.
- **Парафраз** (`judge_paraphrase.txt`) - судья сравнивает парафраз с конкретным оригиналом,
  оценивает сохранение смысла и переформулировку. Порог по умолчанию 4.5.
- **Обратный перевод** (тот же `judge_paraphrase.txt`) - порог 2.5 (NLLB неизбежно теряет
  качество, поэтому планка ниже).

По умолчанию судья включён только на этапе 1 (там он реально помогает отсеивать
"галлюцинированные" письма). На этапах 2 и 3 - выключен.

### Валидация ([`validation.py`](src/augmentation/validation.py))

Все сгенерированные тексты проходят 7 фильтров (от дешёвых к дорогим):

1. **Точные дубликаты** - совпадения с существующими и между собой
2. **Минимальная длина** - короче `validation_min_text_length` символов (250 для парафраза/
   BT, 500 в общих настройках)
3. **Доля от длины оригинала** - для парафраза/BT: текст не должен быть короче `min_length_ratio`
   от длины источника (защита от "обрезанных" ответов модели)
4. **Язык** - не-русский отсеивается через `langdetect`
5. **Вырожденность** - повторяющиеся слова / фразы (модель зациклилась)
6. **CJK-символы** - артефакт Qwen-моделей; иероглифы → отброс
7. **Косинусное сходство** - SBERT (`ai-forever/sbert_large_nlu_ru`) против всех
   существующих текстов класса. Порог сужающийся (см. выше).

Также есть опциональный **prompt-leak filter** (мета-фразы, Markdown-разметка, фрагменты
промпта) - по умолчанию выключен в этапах 2/3, потому что система-промпт этапа 1 уже
прямо запрещает Markdown.

### GPU-профили

Все настройки в [`config_models/pipeline_config.json`](config_models/pipeline_config.json).

| Параметр              | T4 (16GB) | L4 (24GB) | A100_40 | A100_80 / H100 |
|-----------------------|-----------|-----------|---------|----------------|
| NLLB модель           | 600M      | 600M      | 3.3B    | 3.3B           |
| NLLB batch size       | 32        | 32        | 64      | 64             |
| GPU memory util       | 0.90      | 0.90      | 0.92    | 0.95           |
| enforce_eager (vLLM)  | true      | true      | true    | true           |

Конфигурация одной строкой при запуске: `--gpu A100_40` (для CLI) или `GPU = 'A100_40'`
(в ноутбуке).

### Конфигурация LLM-моделей для аугментации

Каждая модель - отдельный JSON в `config_models/aug_configs/`. Основные поля:

- `model_name` - модель с HuggingFace (AWQ-квантизация для vLLM)
- `generation_params` - temperature, top_p, top_k, max_new_tokens
- `prompt_template` - шаблон промпта из `prompts/aug_prompts/` (этап 1)
- `paraphrase_template` - шаблон промпта (этап 2)
- `system_prompt` / `paraphrase_system_prompt` - системные промпты для этапов 1 и 2

---

## 2. Few-shot prompt-классификация

Альтернатива обучению: готовая instruction-tuned LLM получает промпт со списком классов,
описаниями и (опционально) примерами, и возвращает название подразделения. Ничего не
обучается - сравниваем, насколько хорошо разные модели "из коробки" справляются с задачей,
и измеряем вклад описаний и few-shot примеров.

### Модели ([`pipeline_config.json → prompt_classification.prompt_models`](config_models/pipeline_config.json))

| Ключ | HF-модель | Контекст | VRAM | Движок |
|------|-----------|----------|------|--------|
| `saiga_8b` | `IlyaGusev/saiga_llama3_8b` | 8 192 | ~16 GB | transformers, float16 |
| `t_lite_8b` | `t-tech/T-lite-it-1.0` | 32 768 | ~16 GB | transformers, bfloat16 |
| `vikhr_12b` | `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | 32 768 | ~24 GB | transformers, float16 |
| `qwen_14b` | `Qwen/Qwen2.5-14B-Instruct` | 32 768 | ~28 GB | transformers, bfloat16 |
| `qwen_32b` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 32 768 | ~20 GB | vLLM, AWQ, tensor_parallel=2 |

AWQ-модели автоматически грузятся через vLLM (`quantization: "awq"`), остальные - через
transformers с `device_map="auto"`. Промпт оборачивается в `chat_template` токенизатора.

### Режимы (эксперименты)

Каждый режим - кортеж `(K, mode, no_desc, max_example_chars)` в
[`scripts/run_few_shot.py`](scripts/run_few_shot.py):

| Ключ | K | Шаблон | Описания | Длина примера | Назначение |
|------|---|--------|----------|---------------|------------|
| `0`    | 0 | `zero_shot.txt`         | да | -   | Baseline без примеров |
| `1`    | 1 | `one_shot.txt`          | да | 500 | Один пример первого класса |
| `3`    | 3 | `few_shot.txt`          | да | 500 | 3 примера на каждый из 36 классов |
| `3nd`  | 3 | `few_shot_no_desc.txt`  | **нет** | 500 | Проверка вклада описаний |
| `5s`   | 5 | `few_shot.txt`          | да | 300 | Укороченные чтоб влезть в 32k |
| `5snd` | 5 | `few_shot_no_desc.txt`  | **нет** | 300 | K=5 без описаний |

`saiga_8b` гоняется только на `[0, 1]` (контекст 8k), остальные модели - на полном наборе.
Перед каждым прогоном промпт для `df_test.iloc[0]` проверяется на длину: если
`tokens ≥ max_context − 100`, эксперимент целиком помечается `skipped`.

### Структура промпта

Общий каркас ([`prompts/classification_prompts/`](prompts/classification_prompts/)):

1. Роль: "система маршрутизации корреспонденции нефтегазовой компании"
2. `{class_list}` - отсортированный нумерованный список всех 36 классов
3. `{class_descriptions}` - по одной строке на класс, обрезка до 200 символов (только в
   режимах с описаниями)
4. Правила: ровно один вариант из списка, не выдумывать новые, всегда делать выбор
5. `{examples_block}` - для K≥1: пары `Письмо: ... / Подразделение: ...`, обрезка до
   `max_example_chars`
6. `Письмо: {text}` + `Подразделение:` - якорь для короткого ответа модели

### Отбор few-shot примеров ([`few_shot_examples.py`](src/classification/few_shot_examples.py))

- Источники: `train_after_eda.csv` (оригинал) и `data_after_stage3.csv` (аугментированный)
- Правило: если у класса оригинальных писем ≥ K - берём только оригиналы; если меньше -
  добираем из аугментации с дедупликацией по тексту
- Перемешивание с фиксированным `RANDOM_SEED` - одни и те же примеры между прогонами
- Кэш в `Data/few_shot_examples.json`
- Группы классов по числу **оригинальных** писем в train: `A ≥ 50`, `B ∈ [15, 49]`,
  `C < 15`. Используются только для per-group метрик

### Описания классов

При первом запуске `run_few_shot.py` поднимается vLLM с `model_vllm_32b.json`, для каждого
из 36 классов берутся первые 5 писем train и через `generate_class_context` (промпт
`class_context.txt`) генерируется описание. Результат кэшируется в
`Data/class_descriptions.json`, модель после генерации выгружается.

### Извлечение предсказания ([`prompt_classifier.py`](src/classification/prompt_classifier.py))

Ответ модели проходит каскад стратегий (нормализация: lower-case, только `[а-яёa-z0-9 ]`,
сжатие пробелов):

1. Срезание префиксов "Подразделение:", "Ответ:", "Категория:", "Отдел:", маркеров списка
2. Точное совпадение нормализованного кандидата с названием класса
3. Название класса содержится в кандидате → берём самое длинное (устойчивость к "Блок по ...")
4. Кандидат (>5 символов) содержится в названии класса
5. Нечёткое совпадение `difflib.get_close_matches` с `fuzzy_cutoff = 0.6`
6. Fallback: поиск любого названия класса в полном (ненормализованном) ответе
7. Если ничего не подошло - `unknown`

Слишком длинный промпт → `skipped=True`, в знаменатель метрик не попадает.

### Метрики ([`evaluate_prompt_classification`](src/classification/evaluate.py))

- **balanced_accuracy**, **macro_f1** - по всем не-skipped примерам
- **unknown_rate** - доля `predicted_label == "unknown"`
- **f1_group_{A,B,C}** - macro-F1 в пределах каждой группы; для C показывает, справляется
  ли модель с редкими классами
- **skipped / n_test** - сколько примеров выбросили из-за переполнения контекста

### Артефакты

- `results/preds_<model>_exp<key>.csv` - построчные предсказания: `text, true_label,
  predicted_label, raw_response, skipped`
- `results/prompt_results.csv` - сводка `(model, exp_key) → метрики`. Обновляется после
  **каждого** эксперимента, что даёт возобновление с места падения: уже посчитанные
  пары пропускаются
- `results/all_methods_comparison.csv` - единая таблица: traditional + prompt + finetune

### Управление памятью

Перед запуском следующей модели вызывается `unload_model`: удаление vLLM-движка
(`del model.llm`) или transformers-модели, `gc.collect`, `torch.cuda.empty_cache`,
`cuda.synchronize`, `sleep(5)` - обязательно для A100-40GB, иначе следующая 14B/32B-модель
не влезает.

---

## 3. PEFT fine-tune

PEFT-адаптер (LoRA / QLoRA / AdaLoRA / TinyLoRA) дообучается на аугментированных данных.
Ядро всей логики - `SeqClsRunner` в [`src/finetune/trainer_base.py`](src/finetune/trainer_base.py):
сборка модели, PEFT-обвязка, HuggingFace `Trainer`, eval, сохранение адаптера и метрик.

### Архитектура

**Один JSON-конфиг = один прогон.** Конфиг полностью самодостаточен: содержит модель,
метод PEFT, параметры тренировки, флаги класс-весов / val-сплита и т.п. Никаких overrides
поверх "базы", никакого dict'а экспериментов в Python - всё в `config_models/finetune_configs/`.

Запуск через единый CLI [`scripts/run_finetune.py`](scripts/run_finetune.py):

```bash
python scripts/run_finetune.py --config qwen3_32b_qlora_cw.json --gpu A100_40
```

Опционально:
- `--config <abs_path>` или просто имя файла (ищется в `config_models/finetune_configs/`)
- `--run-name <X>` - переопределяет `run_key` из конфига (удобно для SLURM-ферм, где
  один и тот же конфиг хочется запустить под разными именами); создаёт runtime-копию
  в `Data/finetune_runtime_configs/`

### Поля конфига

```json
{
  "run_key": "qwen3_32b_qlora_cw",
  "model_name": "Qwen/Qwen3-32B",
  "method": "qlora",                    // qlora / lora / adalora / tinylora
  "task_type": "SEQ_CLS",
  "num_labels": 36,
  "max_seq_length": 2048,

  "use_class_weights": true,            // балансировка loss по обратной частоте классов
  "val_split": 0.10,                    // отрезаем 10% train → val для early stopping
  "eval_test_each_epoch": true,         // дополнительный test-eval каждую эпоху → test_curve.csv

  "peft_config":  { ... },              // r, lora_alpha, target_modules, lora_dropout
  "quantization": { ... },              // только для QLoRA: NF4 + double quant
  "training_params": { ... }            // num_train_epochs, lr, scheduler, warmup, ...
}
```

GPU-специфика (`per_device_batch`, `grad_accum`, `max_seq`, `bf16/fp16`) живёт в
`pipeline_config.json → finetune.<gpu_profile>` и накладывается поверх `training_params`
автоматически - конфиг прогона не зависит от железа.

### Class weights ([`data_prep.py`](src/finetune/data_prep.py))

При `use_class_weights: true` считается вектор весов по формуле
`w[c] = n_samples / (n_classes * count[c])` (sklearn-balanced) и подставляется в
`CrossEntropyLoss(weight=...)`. Это заставляет модель обращать внимание на редкие классы:
ошибка на крошечном классе "стоит" в десятки раз дороже ошибки на крупном.

При `use_class_weights: false` используется обычный HF `Trainer`. Оба варианта поддержаны
кастомным `ClassWeightedTrainer`.

### Eval per epoch ([`trainer_base.py:TestEvalCallback`](src/finetune/trainer_base.py))

Если `eval_test_each_epoch: true`, после каждой эпохи отдельно прогоняется test и
метрики пишутся в `<adapter>/test_curve.csv`. Это **не** влияет на выбор best-checkpoint
(он остаётся по `eval_macro_f1` на val), но даёт честную траекторию по test - видно где
реальный пик и нет ли расхождения val/test (=overfit на val-сплит).

Стоимость: ~2-3 минуты на эпоху (test ≈ 341 пример). Опционально, можно выключить.

### Early stopping + best checkpoint

В `pipeline_config.json → finetune.common.early_stopping_patience: 2` - тренировка
останавливается, если `eval_macro_f1` не улучшался 2 эпохи подряд. После окончания
автоматически загружается лучший checkpoint
(`load_best_model_at_end: true, metric_for_best_model: macro_f1`), и финальный eval на
test делается уже на этих весах. `save_total_limit: 2` хранит только последние 2
checkpoint'а на диске, но best всегда защищён HF Trainer'ом.

### Артефакты прогона

После одного `python scripts/run_finetune.py --config X.json`:

- `Data/finetune_checkpoints/<run_key>/` - PEFT-адаптер (`adapter_model.safetensors`,
  `adapter_config.json`, `id2label.json`, `class_groups.json`, `tokenizer*`,
  `metadata.json`, `test_curve.csv` если включено)
- `results/finetune/<run_key>.json` - агрегированные метрики (один прогон = один файл):
  `run_key, method, model, balanced_accuracy, macro_f1, f1_group_{A,B,C},
  trainable_params, train_time_sec, timestamp`
- `results/preds_<run_key>.csv` - per-sample предсказания (`text, true_label,
  predicted_label, correct`)

Каждый прогон **перезаписывает** свой JSON в `results/finetune/` - запуск с тем же
`run_key` обновит метрики свежими цифрами.

### Сводка прогонов ([`notebooks/aggregate_results.ipynb`](notebooks/aggregate_results.ipynb))

Отдельный ноутбук проходит по всем `results/finetune/*.json`, собирает их в один CSV
`results/all_methods_comparison.csv` (с округлением метрик до 4 знаков) и рисует две
столбчатые диаграммы: top-10 по `balanced_accuracy` и top-10 по `macro_f1`. Удобно
запускать после каждой пачки прогонов чтобы быстро увидеть лидеров.

### Готовые эксперимент-конфиги

| Конфиг | Модель | r | lora_alpha | class_weights |
|--------|--------|---|------------|---------------|
| `qwen3_32b_qlora_cw.json`         | Qwen3-32B | 16 | 32 | ON  |
| `qwen3_32b_qlora_no_cw.json`      | Qwen3-32B | 16 | 32 | OFF |
| `qwen3_32b_qlora_cw_r32.json`     | Qwen3-32B | 32 | 64 | ON  |
| `qwen3_32b_qlora_no_cw_r32.json`  | Qwen3-32B | 32 | 64 | OFF |
| `ruadapt_qwen3_32b_qlora_cw*`     | RuadaptQwen3-32B | 16/32 | 32/64 | ON |
| `vistral_24b_qlora_cw*`           | Vistral-24B | 16/32 | 32/64 | ON |
| `tpro_it_21_qlora.json`           | T-pro-it-2.1 | 16 | 32 | OFF (legacy) |
| `qwen3_14b_lora.json`             | Qwen3-14B | 16 | 32 | OFF (legacy) |

Базовые `*_qlora.json` без суффикса - "сырые" конфиги без `use_class_weights`/`val_split`,
от которых исторически наследовались эксперимент-конфиги. Сейчас они используются только
как референс при добавлении новой модели в матрицу.

---

## 4. Запуск

### Установка

```bash
pip install -r requirements.txt
```

### Полный аугментационный пайплайн

CLI-обёртка над всеми этапами + baseline/augmented метриками:

```bash
python scripts/augmentation.py --gpu A100_40
python scripts/augmentation.py --gpu A100_40 --config config_models/aug_configs/model_vllm.json
```

Что делает по порядку: предобработка `data.json`, train/test split (если ещё нет),
baseline-метрики на оригинале (LinearSVC, LogisticRegression, MultinomialNB, rubert-tiny2,
rubert-base), три этапа аугментации, augmented-метрики на расширенном train, сводка в
`results/classification_results.csv`.

Можно вызывать этапы по отдельности:

```bash
python -c "from src.utils.data_cleaner import run; run()"

python src/augmentation/stage1_llm_generate.py     --config config_models/aug_configs/model_vllm_32b.json
python src/augmentation/stage2_paraphrase.py       --config config_models/aug_configs/model_vllm_32b.json
python src/augmentation/stage3_back_translation.py --config config_models/aug_configs/model_vllm_32b.json
```

Интерактивно: [`notebooks/augmentation.ipynb`](notebooks/augmentation.ipynb).

### Few-shot prompt-классификация

```bash
python scripts/run_few_shot.py                                   # все модели × все режимы из матрицы
python scripts/run_few_shot.py --models qwen_14b                 # одна модель, все её режимы
python scripts/run_few_shot.py --models qwen_14b qwen_32b \
                               --experiments 0 1 3 3nd           # подмножество модели и режимов
```

Прогон идемпотентен: при повторном запуске из `results/prompt_results.csv` подтягиваются
уже завершённые пары `(model, exp_key)` - пересчитываются только недостающие. Чтобы
пересчитать эксперимент заново, удалите соответствующую строку из `prompt_results.csv` и
файл `preds_<model>_exp<key>.csv`.

Интерактивно: [`notebooks/zer0_one_few_shot.ipynb`](notebooks/zer0_one_few_shot.ipynb).

### PEFT fine-tune

Один прогон:

```bash
python scripts/run_finetune.py --config qwen3_32b_qlora_cw.json --gpu A100_40
python scripts/run_finetune.py --config qwen3_32b_qlora_no_cw_r32.json --gpu A100_40 --run-name my_test
```

Несколько прогонов - просто несколько вызовов CLI (последовательно или через любую
систему очередей). После завершения каждого результат лежит в
`results/finetune/<run_key>.json`.

Сборка сводки и графиков top-N: открыть и прогнать ячейки в
[`notebooks/aggregate_results.ipynb`](notebooks/aggregate_results.ipynb).

Интерактивно один прогон в Colab: [`notebooks/finetune.ipynb`](notebooks/finetune.ipynb).

### Традиционные классификаторы (отдельно)

Если нужно прогнать только их (без всего пайплайна):

```bash
python src/classification/run_svm.py
python src/classification/run_logreg.py
python src/classification/run_naive_bayes.py
```

Они читают `data_after_stage3.csv` + `data_test.csv` и пишут в
`results/classification_results.csv`.
