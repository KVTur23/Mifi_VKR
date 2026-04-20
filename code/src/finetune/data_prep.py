"""
data_prep.py — Подготовка данных для файнтюна sequence classification

Грузит train/test через единый data_loader, строит детерминированный label2id,
считает группы A/B/C по оригинальным письмам (только для отчётного разреза),
токенизирует и выдаёт коллатор с динамическим паддингом.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, load_test_set, TEXT_COL, LABEL_COL,
)


# Границы групп — совпадают с few_shot_examples.py
GROUP_A_MIN = 50
GROUP_B_MIN = 15


def load_finetune_data(data_dir=None):
    """
    Грузит train (stage=3, аугментированный) и test + оригинальный train (stage=0)
    только для подсчёта числа оригинальных писем на класс (для отчётных групп A/B/C).

    Возвращает: (df_train, df_test, orig_counts)
    где orig_counts — dict {label: int} по оригинальному train_after_eda.csv.
    """
    df_train = load_dataset(stage=3, data_dir=data_dir)
    df_test = load_test_set(data_dir=data_dir)
    df_orig = load_dataset(stage=0, data_dir=data_dir)

    orig_counts = df_orig[LABEL_COL].value_counts().to_dict()
    return df_train, df_test, orig_counts


def build_label_mapping(df_train: pd.DataFrame) -> tuple[dict, dict]:
    """
    Строит детерминированный label2id через sorted() — как в data_loader.split_train_test.

    Возвращает (label2id, id2label): {label: int}, {int: label}.
    """
    labels = sorted(df_train[LABEL_COL].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


def compute_class_groups(orig_counts: dict, label2id: dict) -> dict[int, str]:
    """
    Определяет группу A/B/C для каждого class_id по числу ОРИГИНАЛЬНЫХ писем.

    A ≥ 50, B ∈ [15, 49], C < 15. Классы отсутствующие в orig_counts трактуются как 0.
    Используется только для per-group F1 в отчётах, на обучение не влияет.
    """
    groups = {}
    for label, cid in label2id.items():
        n = int(orig_counts.get(label, 0))
        if n >= GROUP_A_MIN:
            groups[cid] = "A"
        elif n >= GROUP_B_MIN:
            groups[cid] = "B"
        else:
            groups[cid] = "C"
    return groups


def encode_labels(df: pd.DataFrame, label2id: dict) -> pd.DataFrame:
    """Добавляет колонку label_id с числовыми id."""
    df = df.copy()
    df["label_id"] = df[LABEL_COL].map(label2id)
    if df["label_id"].isna().any():
        unknown = df[df["label_id"].isna()][LABEL_COL].unique().tolist()
        raise ValueError(f"Неизвестные метки в df, отсутствуют в label2id: {unknown}")
    df["label_id"] = df["label_id"].astype(int)
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_seq_length: int):
    """
    Превращает df в datasets.Dataset с полями input_ids, attention_mask, labels.
    truncation до max_seq_length, БЕЗ паддинга — паддинг делает коллатор динамически.
    """
    from datasets import Dataset

    sub = df[[TEXT_COL, "label_id"]].rename(columns={"label_id": "labels"})
    ds = Dataset.from_pandas(sub, preserve_index=False)

    def _tok(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=max_seq_length,
        )

    ds = ds.map(_tok, batched=True, remove_columns=[TEXT_COL])
    return ds


def get_collator(tokenizer):
    """DataCollatorWithPadding — паддит каждый батч до длины самого длинного в батче."""
    from transformers import DataCollatorWithPadding
    return DataCollatorWithPadding(tokenizer, padding="longest")
