"""
============================================================
ЭТАП 1: Генерация синтетических данных и их валидация
============================================================
Пайплайн:
1. Разделение данных на блоки по количеству примеров
2. Генерация новых примеров с помощью LLM для малых (<25) и средних (25–49) блоков
3. Каскадная валидация: перплексия → дедупликация → классификатор-фильтр
============================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ
# ============================================================

def analyze_data(data: pd.DataFrame, target_col: str = "label"):
    """
    Анализируем распределение классов и делим блоки на малые, средние и большие.
    Малые блоки (<25 примеров) и средние блоки (25–49 примеров) пойдут на генерацию.
    """
    # Считаем количество примеров в каждом блоке
    class_counts = data[target_col].value_counts()

    print("=" * 60)
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    print("=" * 60)
    print(f"Всего классов: {len(class_counts)}")
    print(f"Всего примеров: {len(data)}")
    print(f"Медиана: {class_counts.median():.0f}")
    print(f"Минимум: {class_counts.min()} ({class_counts.idxmin()})")
    print(f"Максимум: {class_counts.max()} ({class_counts.idxmax()})")
    print()

    # Пороги: малые — <25, средние — 25–49, большие — >=50
    SMALL_THRESHOLD = 25
    MEDIUM_THRESHOLD = 50

    small_blocks = class_counts[class_counts < SMALL_THRESHOLD]
    medium_blocks = class_counts[(class_counts >= SMALL_THRESHOLD) & (class_counts < MEDIUM_THRESHOLD)]
    large_blocks = class_counts[class_counts >= MEDIUM_THRESHOLD]

    print(f"Малых блоков (<{SMALL_THRESHOLD}): {len(small_blocks)}")
    print(f"Средних блоков ({SMALL_THRESHOLD}–{MEDIUM_THRESHOLD - 1}): {len(medium_blocks)}")
    print(f"Больших блоков (>={MEDIUM_THRESHOLD}): {len(large_blocks)}")
    print()

    # Показываем малые блоки — именно для них будем генерировать
    print("МАЛЫЕ БЛОКИ (нужна генерация):")
    print("-" * 40)
    for label, count in small_blocks.items():
        print(f"  {label}: {count} примеров")

    print()
    print("СРЕДНИЕ БЛОКИ (нужна генерация):")
    print("-" * 40)
    for label, count in medium_blocks.items():
        print(f"  {label}: {count} примеров")

    return small_blocks, medium_blocks, large_blocks


# ============================================================
# 2. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ ЧЕРЕЗ LLM
# ============================================================

def load_generator(model_name: str = "ai-forever/ruGPT-3.5-13B"):
    """
    Загружаем генеративную модель.

    Варианты моделей для русского языка:
    - ai-forever/ruGPT-3.5-13B — хорошее качество, нужно ~28GB VRAM
    - ai-forever/rugpt3large_based_on_gpt2 — легче, ~4GB VRAM
    - IlyaGusev/saiga_llama3_8b — Llama 3 дообученная на русском, ~16GB VRAM

    Выбирай модель исходя из доступных ресурсов.
    """
    print(f"Загружаем модель генерации: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Половинная точность для экономии памяти
        device_map="auto"           # Автоматическое распределение по GPU
    )

    return model, tokenizer


def build_prompt(label: str, examples: list[str], n_generate: int = 5) -> str:
    """
    Собираем промпт для генерации новых примеров.

    Структура промпта:
    1. Системная инструкция — что делать
    2. Описание класса
    3. Оригинальные примеры (few-shot)
    4. Запрос на генерацию N новых примеров
    """
    # Берём не больше 5 примеров для промпта (чтобы не переполнить контекст)
    selected_examples = examples[:5]

    prompt = f"""Ты — ассистент для генерации обучающих данных.
Задача: сгенерировать новые примеры деловых писем для категории "{label}".

Требования к генерации:
- Письма должны быть на русском языке
- Стиль — деловая переписка
- Каждое письмо должно быть уникальным, не копировать примеры
- Сохраняй тематику и стиль категории "{label}"
- Варьируй длину, формулировки, детали

Примеры писем этой категории:
"""
    # Добавляем оригинальные примеры как образцы
    for i, example in enumerate(selected_examples, 1):
        prompt += f"\nПример {i}:\n{example}\n"

    prompt += f"\nСгенерируй {n_generate} новых уникальных писем этой категории. Каждое письмо начинай с маркера [ПИСЬМО]:\n"

    return prompt


def generate_examples(
    model,
    tokenizer,
    label: str,
    examples: list[str],
    n_generate: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> list[str]:
    """
    Генерируем новые примеры для одного класса.

    Параметры генерации:
    - temperature: 0.8 — баланс между разнообразием и качеством
    - top_p: 0.9 — nucleus sampling, отсекаем маловероятные токены
    """
    prompt = build_prompt(label, examples, n_generate)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,        # Максимум новых токенов
            temperature=temperature,     # Разнообразие генерации
            top_p=top_p,                # Nucleus sampling
            do_sample=True,             # Включаем сэмплирование (не жадный поиск)
            repetition_penalty=1.2,     # Штраф за повторения
            no_repeat_ngram_size=3      # Запрет повторения 3-грамм
        )

    # Декодируем и извлекаем сгенерированные письма
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Убираем промпт из ответа
    generated_text = generated_text[len(prompt):]

    # Разбиваем по маркеру [ПИСЬМО]
    raw_examples = generated_text.split("[ПИСЬМО]")

    # Чистим каждый пример от лишних пробелов и пустых строк
    cleaned = []
    for text in raw_examples:
        text = text.strip()
        if len(text) > 50:  # Отсекаем слишком короткие обрезки
            cleaned.append(text)

    print(f"  Класс '{label}': сгенерировано {len(cleaned)} примеров")
    return cleaned


def generate_for_small_blocks(
    data: pd.DataFrame,
    small_blocks: pd.Series,
    model,
    tokenizer,
    target_count: int = 50,
    target_col: str = "label",
    text_col: str = "text"
) -> pd.DataFrame:
    """
    Генерируем примеры для всех малых блоков.
    Цель — довести каждый малый блок до target_count примеров.
    """
    all_generated = []

    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 60)

    for label, count in tqdm(small_blocks.items(), desc="Генерация"):
        # Сколько примеров нужно добавить
        n_needed = target_count - count

        if n_needed <= 0:
            continue

        # Достаём оригинальные примеры этого класса
        original_examples = data[data[target_col] == label][text_col].tolist()

        # Генерируем новые примеры
        generated = generate_examples(
            model, tokenizer, label, original_examples, n_generate=n_needed
        )

        # Сохраняем с пометкой что это синтетика
        for text in generated:
            all_generated.append({
                text_col: text,
                target_col: label,
                "is_synthetic": True  # Флаг синтетических данных
            })

    generated_df = pd.DataFrame(all_generated)
    print(f"\nВсего сгенерировано: {len(generated_df)} примеров")

    return generated_df


# ============================================================
# 3. КАСКАДНАЯ ВАЛИДАЦИЯ
# ============================================================

# --- 3.1 Фильтр по перплексии ---

def filter_by_perplexity(
    texts: list[str],
    model_name: str = "ai-forever/rugpt3large_based_on_gpt2",
    max_perplexity: float = 150.0
) -> list[bool]:
    """
    Считаем перплексию каждого текста.
    Высокая перплексия = неестественный текст = отбрасываем.

    Порог max_perplexity подбирается экспериментально:
    - Посчитай перплексию на оригинальных данных
    - Возьми 95-й перцентиль как ориентир
    """
    print("\n--- Фильтр 1: Перплексия ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    passed = []

    for text in tqdm(texts, desc="Перплексия"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Перплексия = exp(средний лосс)
            perplexity = torch.exp(outputs.loss).item()

        passed.append(perplexity <= max_perplexity)

    n_passed = sum(passed)
    print(f"  Прошло: {n_passed}/{len(texts)} ({n_passed/len(texts)*100:.1f}%)")

    return passed


# --- 3.2 Фильтр по дедупликации ---

def filter_duplicates(
    generated_texts: list[str],
    original_texts: list[str],
    embed_model_name: str = "cointegrated/rubert-tiny2",
    similarity_threshold: float = 0.9
) -> list[bool]:
    """
    Убираем тексты, которые слишком похожи на оригиналы
    или друг на друга.

    Используем эмбеддинги и косинусное сходство.
    Порог 0.9 — это почти дословное совпадение.
    """
    print("\n--- Фильтр 2: Дедупликация ---")

    # Загружаем модель для эмбеддингов
    embed_model = SentenceTransformer(embed_model_name)

    # Считаем эмбеддинги
    gen_embeddings = embed_model.encode(generated_texts, show_progress_bar=True)
    orig_embeddings = embed_model.encode(original_texts, show_progress_bar=True)

    passed = []

    for i, gen_emb in enumerate(gen_embeddings):
        gen_emb = gen_emb.reshape(1, -1)

        # Проверка 1: сходство с оригинальными примерами
        sim_with_original = cosine_similarity(gen_emb, orig_embeddings).max()

        # Проверка 2: сходство с уже принятыми сгенерированными примерами
        sim_with_generated = 0
        if i > 0:
            prev_embeddings = gen_embeddings[:i]
            sim_with_generated = cosine_similarity(gen_emb, prev_embeddings).max()

        # Пример проходит если не слишком похож ни на оригиналы, ни на другие
        is_unique = (
            sim_with_original < similarity_threshold and
            sim_with_generated < similarity_threshold
        )
        passed.append(is_unique)

    n_passed = sum(passed)
    print(f"  Прошло: {n_passed}/{len(generated_texts)} ({n_passed/len(generated_texts)*100:.1f}%)")

    return passed


# --- 3.3 Фильтр по классификатору ---

def filter_by_classifier(
    texts: list[str],
    expected_labels: list[str],
    classifier_model_name: str = "your-finetuned-classifier",
    confidence_threshold: float = 0.5
) -> list[bool]:
    """
    Прогоняем сгенерированные примеры через классификатор.
    Если классификатор не уверен что пример принадлежит целевому классу —
    отбрасываем.

    ВАЖНО: перед использованием нужно дообучить классификатор
    на оригинальных 1700 примерах (например, ruBERT + классификационная голова).
    """
    print("\n--- Фильтр 3: Классификатор ---")

    # Загружаем дообученный классификатор
    clf = pipeline(
        "text-classification",
        model=classifier_model_name,
        tokenizer=classifier_model_name,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None  # Получаем вероятности всех классов
    )

    passed = []

    for text, expected_label in tqdm(
        zip(texts, expected_labels), total=len(texts), desc="Классификатор"
    ):
        # Получаем предсказания
        predictions = clf(text, truncation=True, max_length=512)

        # Ищем вероятность целевого класса
        target_prob = 0.0
        for pred in predictions[0]:
            if pred["label"] == expected_label:
                target_prob = pred["score"]
                break

        # Пример проходит если уверенность выше порога
        passed.append(target_prob >= confidence_threshold)

    n_passed = sum(passed)
    print(f"  Прошло: {n_passed}/{len(texts)} ({n_passed/len(texts)*100:.1f}%)")

    return passed


# --- 3.4 Альтернатива: LLM как арбитр (для блоков с 2-10 примерами) ---

def filter_by_llm_arbiter(
    texts: list[str],
    expected_labels: list[str],
    all_labels: list[str],
    model,
    tokenizer
) -> list[bool]:
    """
    Для блоков с очень малым числом примеров (2-10) классификатор
    будет работать плохо. Вместо него используем LLM:
    показываем ей текст и список классов, просим выбрать.
    """
    print("\n--- Фильтр 3 (альтернатива): LLM-арбитр ---")

    # Формируем список классов для промпта
    labels_str = "\n".join([f"- {label}" for label in all_labels])

    passed = []

    for text, expected_label in tqdm(
        zip(texts, expected_labels), total=len(texts), desc="LLM-арбитр"
    ):
        prompt = f"""Определи категорию следующего делового письма.

Письмо:
{text}

Возможные категории:
{labels_str}

Ответь ТОЛЬКО названием одной категории, без пояснений:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Низкая температура — хотим чёткий ответ
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Проверяем совпадает ли ответ с ожидаемым классом
        passed.append(expected_label.lower() in response.lower())

    n_passed = sum(passed)
    print(f"  Прошло: {n_passed}/{len(texts)} ({n_passed/len(texts)*100:.1f}%)")

    return passed


# ============================================================
# 4. ПОЛНЫЙ КАСКАД ВАЛИДАЦИИ
# ============================================================

def validate_cascade(
    generated_df: pd.DataFrame,
    original_data: pd.DataFrame,
    small_blocks: pd.Series,
    model=None,
    tokenizer=None,
    classifier_model_name: str = None,
    target_col: str = "label",
    text_col: str = "text"
) -> pd.DataFrame:
    """
    Прогоняем сгенерированные данные через каскад фильтров.

    Порядок фильтров (от быстрого к медленному):
    1. Перплексия — отсекает мусорные тексты
    2. Дедупликация — убирает копии и почти-копии
    3. Классификатор / LLM-арбитр — проверяет принадлежность классу
    """
    print("\n" + "=" * 60)
    print("КАСКАДНАЯ ВАЛИДАЦИЯ")
    print("=" * 60)

    texts = generated_df[text_col].tolist()
    labels = generated_df[target_col].tolist()
    initial_count = len(texts)

    # --- Фильтр 1: Перплексия ---
    perplexity_mask = filter_by_perplexity(texts)
    generated_df = generated_df[perplexity_mask].reset_index(drop=True)
    texts = generated_df[text_col].tolist()
    labels = generated_df[target_col].tolist()

    # --- Фильтр 2: Дедупликация (для каждого класса отдельно) ---
    dedup_mask = []
    for label in generated_df[target_col].unique():
        # Индексы сгенерированных примеров этого класса
        gen_mask = generated_df[target_col] == label
        gen_texts = generated_df[gen_mask][text_col].tolist()

        # Оригинальные примеры этого класса
        orig_texts = original_data[original_data[target_col] == label][text_col].tolist()

        # Проверяем уникальность
        class_dedup = filter_duplicates(gen_texts, orig_texts)
        dedup_mask.extend(class_dedup)

    # Тут нужно аккуратно — порядок может сбиться если классы идут не подряд
    # Поэтому пересоберём маску в правильном порядке
    ordered_dedup_mask = []
    idx = 0
    for label in generated_df[target_col].unique():
        gen_mask = generated_df[target_col] == label
        n_class = gen_mask.sum()
        class_results = dedup_mask[idx:idx + n_class]
        ordered_dedup_mask.extend(class_results)
        idx += n_class

    generated_df = generated_df[ordered_dedup_mask].reset_index(drop=True)
    texts = generated_df[text_col].tolist()
    labels = generated_df[target_col].tolist()

    # --- Фильтр 3: Классификатор или LLM-арбитр ---
    all_labels = original_data[target_col].unique().tolist()
    classifier_mask = []

    for label in generated_df[target_col].unique():
        gen_mask = generated_df[target_col] == label
        gen_texts = generated_df[gen_mask][text_col].tolist()
        gen_labels = generated_df[gen_mask][target_col].tolist()
        original_count = small_blocks.get(label, 0)

        if original_count <= 10 and model is not None:
            # Мало примеров — классификатор не справится, используем LLM
            class_filter = filter_by_llm_arbiter(
                gen_texts, gen_labels, all_labels, model, tokenizer
            )
        elif classifier_model_name:
            # Достаточно примеров — используем классификатор
            class_filter = filter_by_classifier(
                gen_texts, gen_labels, classifier_model_name
            )
        else:
            # Если нет ни модели, ни классификатора — пропускаем все
            class_filter = [True] * len(gen_texts)

        classifier_mask.extend(class_filter)

    # Аналогично пересобираем маску
    ordered_clf_mask = []
    idx = 0
    for label in generated_df[target_col].unique():
        gen_mask = generated_df[target_col] == label
        n_class = gen_mask.sum()
        class_results = classifier_mask[idx:idx + n_class]
        ordered_clf_mask.extend(class_results)
        idx += n_class

    generated_df = generated_df[ordered_clf_mask].reset_index(drop=True)

    # --- Итоги ---
    final_count = len(generated_df)
    print("\n" + "=" * 60)
    print("ИТОГИ ВАЛИДАЦИИ")
    print("=" * 60)
    print(f"Было сгенерировано: {initial_count}")
    print(f"Прошло валидацию:   {final_count}")
    print(f"Отсеяно:            {initial_count - final_count}")
    print(f"Доля прошедших:     {final_count/initial_count*100:.1f}%")
    print()

    # Статистика по классам
    print("По классам:")
    for label in generated_df[target_col].unique():
        count = (generated_df[target_col] == label).sum()
        print(f"  {label}: {count} прошло валидацию")

    return generated_df


# ============================================================
# 5. СБОРКА ФИНАЛЬНОГО ДАТАСЕТА
# ============================================================

def build_final_dataset(
    original_data: pd.DataFrame,
    validated_generated: pd.DataFrame,
    target_col: str = "label",
    text_col: str = "text"
) -> pd.DataFrame:
    """
    Объединяем оригинальные и сгенерированные данные.
    Добавляем флаг is_synthetic для отслеживания.
    """
    # Помечаем оригинальные данные
    original_data = original_data.copy()
    original_data["is_synthetic"] = False

    # Объединяем
    final = pd.concat([original_data, validated_generated], ignore_index=True)

    # Перемешиваем
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ ДАТАСЕТ")
    print("=" * 60)
    print(f"Оригинальных: {(~final['is_synthetic']).sum()}")
    print(f"Синтетических: {final['is_synthetic'].sum()}")
    print(f"Всего: {len(final)}")
    print()

    # Новое распределение
    print("Распределение классов после аугментации:")
    counts = final[target_col].value_counts()
    for label, count in counts.items():
        synthetic = ((final[target_col] == label) & final["is_synthetic"]).sum()
        print(f"  {label}: {count} (из них синтетических: {synthetic})")

    return final


# ============================================================
# 6. РУЧНАЯ ВАЛИДАЦИЯ — ВЫБОРКА ДЛЯ ПРОВЕРКИ
# ============================================================

def prepare_manual_validation(
    validated_generated: pd.DataFrame,
    n_per_class: int = 10,
    target_col: str = "label",
    text_col: str = "text",
    output_path: str = "manual_validation_sample.csv"
):
    """
    Готовим выборку для ручной проверки.
    Берём до n_per_class примеров из каждого класса.

    Создаём CSV с колонками для разметки:
    - correct_class: 1/0 — пример соответствует классу?
    - natural_text: 1/0 — текст выглядит естественно?
    - is_copy: 1/0 — это явная копия оригинала?
    - comment: свободный комментарий
    """
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА РУЧНОЙ ВАЛИДАЦИИ")
    print("=" * 60)

    samples = []

    for label in validated_generated[target_col].unique():
        class_data = validated_generated[validated_generated[target_col] == label]
        # Берём min(n_per_class, сколько есть)
        n_sample = min(n_per_class, len(class_data))
        class_sample = class_data.sample(n=n_sample, random_state=42)
        samples.append(class_sample)

    sample_df = pd.concat(samples, ignore_index=True)

    # Добавляем колонки для разметки
    sample_df["correct_class"] = ""   # Соответствует классу? (1/0)
    sample_df["natural_text"] = ""    # Естественный текст? (1/0)
    sample_df["is_copy"] = ""         # Копия оригинала? (1/0)
    sample_df["comment"] = ""         # Свободный комментарий

    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Сохранено {len(sample_df)} примеров в {output_path}")
    print(f"Заполни колонки correct_class, natural_text, is_copy, comment")

    return sample_df


# ============================================================
# 7. ЗАПУСК ПАЙПЛАЙНА
# ============================================================

def run_pipeline(data: pd.DataFrame):
    """
    Основная точка входа.
    Запускает весь пайплайн от анализа до финального датасета.
    """

    # Шаг 1: Анализ данных
    small_blocks, medium_blocks, _ = analyze_data(data)
    blocks_to_augment = small_blocks  # Генерация только для малых блоков (<25)

    # Шаг 2: Загрузка модели генерации
    # ЗАМЕНИ на свою модель — выбор зависит от доступных ресурсов
    model, tokenizer = load_generator(
        model_name="ai-forever/rugpt3large_based_on_gpt2"  # Лёгкая модель для начала
    )

    # Шаг 3: Генерация
    generated_df = generate_for_small_blocks(
        data, blocks_to_augment, model, tokenizer,
        target_count=50  # Довести каждый малый/средний блок до 50 примеров
    )

    # Шаг 4: Каскадная валидация
    validated_df = validate_cascade(
        generated_df,
        original_data=data,
        small_blocks=blocks_to_augment,
        model=model,                      # Для LLM-арбитра
        tokenizer=tokenizer,
        classifier_model_name=None,       # Укажи путь к дообученному классификатору
    )

    # Шаг 5: Финальный датасет
    final_data = build_final_dataset(data, validated_df)

    # Шаг 6: Выборка для ручной валидации
    prepare_manual_validation(validated_df)

    # Сохраняем результат
    final_data.to_csv("augmented_dataset.csv", index=False, encoding="utf-8-sig")
    print("\nДатасет сохранён в augmented_dataset.csv")

    return final_data


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    data = pd.read_json("../Data/data.json")

    run_pipeline(data)
