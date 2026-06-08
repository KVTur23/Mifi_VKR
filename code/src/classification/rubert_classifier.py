"""
rubert_classifier.py - Fine-tuning BERT-моделей для классификации писем

Берём предобученный BERT (по умолчанию rubert-tiny2, можно подсунуть rubert-base
и т.п.), дообучаем его на наших данных и считаем метрики на test.
Возвращаем dict с balanced_accuracy и macro_f1 для сравнения с baseline.

В отличие от TF-IDF + классические модели, BERT понимает порядок слов и
контекст, поэтому на наших несбалансированных классах обычно показывает
себя лучше всего.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Дефолты - переопределяются через аргументы train_and_evaluate
DEFAULT_MODEL = "cointegrated/rubert-tiny2"
# Сколько токенов берём из письма. Письма у нас длинные (до нескольких КБ),
# но BERT всё равно режет по 512 максимум - берём 256 как компромисс между
# скоростью и тем, сколько контекста влезает
MAX_LENGTH = 256
BATCH_SIZE = 32
# Высокий LR для tiny-модели (мало параметров, можно учить агрессивно).
# Для rubert-base снижаем до 5e-5 - у неё веса предобучены тоньше,
# при высоком LR loss начинает скакать и обучение разваливается
LEARNING_RATE = 5e-4
NUM_EPOCHS = 15


class TextDataset(Dataset):
    """
    Обёртка над текстами + метками, чтобы PyTorch DataLoader мог их батчить.
    Токенизация делается один раз в __init__, дальше при обращении [idx]
    просто отдаём готовые тензоры - это сильно быстрее, чем токенизировать
    на каждый батч.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        # truncation=True - режем длинные письма до max_length токенов
        # padding=True - добиваем короткие до длины самого длинного в батче
        # return_tensors="pt" - сразу отдать как PyTorch-тензоры
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        # int64 - стандарт для индексов классов в CrossEntropyLoss
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # input_ids - сами токены (id из словаря), attention_mask - где
        # реальные токены, а где padding (модель игнорирует padding)
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    model_name: str = DEFAULT_MODEL,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    name: str = "rubert-tiny2",
) -> dict:
    """
    Fine-tune BERT-модели на train и оценка на test.

    Что происходит:
    1. Кодируем строковые метки в числа (LabelEncoder)
    2. Грузим предобученные веса с HuggingFace и добавляем сверху
       классификационную голову на num_labels классов
    3. Учим модель num_epochs эпох с AdamW + warmup-шедулером
    4. На test предсказываем и считаем метрики

    Возвращает:
        dict с ключами: name, balanced_accuracy, macro_f1
    """
    # Если есть видеокарта - используем её, иначе CPU (на CPU будет очень долго)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] Device: {device}, Model: {model_name}")

    # Метки у нас строки (например, "Блок технического директора"), а
    # CrossEntropyLoss хочет int. LabelEncoder делает это маппирование строк
    # в индексы 0..N-1, и мы запоминаем le для обратного преобразования
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[label_col].values)
    y_test = le.transform(df_test[label_col].values)
    num_labels = len(le.classes_)

    texts_train = df_train[text_col].tolist()
    texts_test = df_test[text_col].tolist()

    # Качаем токенизатор и модель с HuggingFace (если уже в локальном кэше -
    # подхватит оттуда). num_labels определяет размер выходного слоя классификатора:
    # модель сама добавит линейный слой [hidden_size -> num_labels] поверх BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(device)

    # Готовим Dataset'ы и DataLoader'ы. shuffle=True на train - чтобы модель
    # не запоминала порядок писем (а то выучит "после класса A всегда B").
    # На test shuffle не нужен - оценка от порядка не зависит
    train_ds = TextDataset(texts_train, y_train, tokenizer)
    test_ds = TextDataset(texts_test, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # AdamW - стандарт для fine-tune BERT (отличается от обычного Adam тем,
    # что weight_decay применяется правильно, не как L2-regularization).
    # weight_decay=0.01 - небольшая регуляризация, чтобы веса не разносило
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup scheduler: первую эпоху LR плавно растёт от 0 до lr,
    # потом линейно падает до 0 к концу обучения. Warmup нужен, чтобы
    # модель не "сошла с ума" на первых батчах с большими градиентами
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader)  # 1 эпоха warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"[{name}] Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Классов: {num_labels}, Эпох: {num_epochs}")

    # Тренировочный цикл. model.train() переводит модель в режим обучения
    # (включает dropout, batch norm считается по батчу - это влияет на
    # forward, отличие от eval-режима ниже)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Перетаскиваем тензоры на GPU (если есть)
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward: модель сама посчитает loss, т.к. мы передали labels
            outputs = model(**batch)
            loss = outputs.loss
            # Backward: градиенты по всем параметрам
            loss.backward()
            # Обрезаем норму градиентов до 1.0 - защита от gradient explosion,
            # когда из-за длинных текстов loss иногда даёт огромные градиенты
            # и модель "взрывается" (loss=NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Шаг оптимизатора + шедулера, обнуляем градиенты на следующий батч
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}")

    # Eval-режим: выключает dropout, фиксирует batch norm. Это критично -
    # без model.eval() предсказания будут случайно отличаться от запуска к запуску.
    # torch.no_grad() выключает построение графа вычислений: экономит память
    # и ускоряет, т.к. backward здесь не нужен
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # logits - сырые скоры по классам, argmax даёт индекс класса
            # с максимальным скором (это и есть предсказание)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    # balanced_accuracy - средний recall по классам (учитывает дисбаланс,
    # каждый класс весит одинаково независимо от размера)
    # macro_f1 - средний F1 по классам без взвешивания, тоже учитывает дисбаланс
    bal_acc = balanced_accuracy_score(y_test, all_preds)
    f1_mac = f1_score(y_test, all_preds, average="macro", zero_division=0)

    print(f"\n[{name}] Результаты на тестовой выборке:")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {f1_mac:.4f}")
    # Подробный отчёт по каждому классу - precision/recall/f1/support.
    # zero_division=0 - чтобы не ругалось на классы где модель ничего не предсказала
    print(f"\n{classification_report(y_test, all_preds, target_names=le.classes_, zero_division=0)}")

    # Чистим GPU - модель + датасеты могут весить сотни МБ. Без этого при
    # последовательном вызове train_and_evaluate для нескольких моделей
    # (tiny + base подряд) рискуем словить OOM
    del model, optimizer, train_ds, test_ds
    torch.cuda.empty_cache()

    return {"name": name, "balanced_accuracy": bal_acc, "macro_f1": f1_mac}
