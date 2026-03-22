"""
rubert_classifier.py — Fine-tuning cointegrated/rubert-tiny2 для классификации

Дообучает rubert-tiny2 на train, оценивает на test.
Возвращает dict с метриками для сравнения с baseline-классификаторами.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 15


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    name: str = "rubert-tiny2",
) -> dict:
    """
    Fine-tune rubert-tiny2 на train, оценка на test.

    Возвращает:
        dict с ключами: name, accuracy, macro_f1, weighted_f1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] Device: {device}")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[label_col].values)
    y_test = le.transform(df_test[label_col].values)
    num_labels = len(le.classes_)

    texts_train = df_train[text_col].tolist()
    texts_test = df_test[text_col].tolist()

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
    ).to(device)

    # Datasets
    train_ds = TextDataset(texts_train, y_train, tokenizer)
    test_ds = TextDataset(texts_test, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader)  # 1 эпоха warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Training
    print(f"[{name}] Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Классов: {num_labels}, Эпох: {num_epochs}")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    acc = accuracy_score(y_test, all_preds)
    bal_acc = balanced_accuracy_score(y_test, all_preds)
    f1_mac = f1_score(y_test, all_preds, average="macro", zero_division=0)

    print(f"\n[{name}] Результаты на тестовой выборке:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {f1_mac:.4f}")
    print(f"\n{classification_report(y_test, all_preds, target_names=le.classes_, zero_division=0)}")

    # Cleanup GPU
    del model, optimizer, train_ds, test_ds
    torch.cuda.empty_cache()

    return {"name": name, "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1_mac}
