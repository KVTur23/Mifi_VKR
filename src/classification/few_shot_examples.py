"""
few_shot_examples.py — Подготовка few-shot примеров для prompt-based классификации

Логика отбора:
  - Если оригинальных примеров класса >= K: берём только оригинальные.
  - Если оригинальных < K: добираем аугментированными (приоритет оригинальным).

Группы A/B/C сохраняются в JSON только для подсчёта per-group метрик,
на отбор примеров не влияют.

Сохраняет примеры в Data/few_shot_examples.json для K=1, 3, 5.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    load_dataset, TEXT_COL, LABEL_COL, RANDOM_SEED,
)

DATA_DIR = PROJECT_ROOT / "Data"

# Пороги групп (из pipeline_config.json → stage4.group_thresholds)
GROUP_A_MIN = 50
GROUP_B_MIN = 15


def classify_groups(df_original: pd.DataFrame) -> dict[str, str]:
    """
    Определяет группу (A/B/C) для каждого класса по количеству
    оригинальных примеров в train.

    Возвращает: {class_name: "A"|"B"|"C"}
    """
    counts = df_original[LABEL_COL].value_counts()
    groups = {}
    for cls, n in counts.items():
        if n >= GROUP_A_MIN:
            groups[cls] = "A"
        elif n >= GROUP_B_MIN:
            groups[cls] = "B"
        else:
            groups[cls] = "C"
    return groups


def select_examples(
    df_original: pd.DataFrame,
    df_augmented: pd.DataFrame,
    k: int,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """
    Отбирает K примеров для каждого класса.

    Если оригинальных примеров >= K — берём только их.
    Если меньше K — добираем из аугментированных (приоритет оригинальным).

    Возвращает: {class_name: [text1, text2, ..., textK]}
    """
    rng = np.random.RandomState(seed)
    examples = {}

    all_classes = sorted(df_original[LABEL_COL].unique())

    for cls in all_classes:
        orig_texts = df_original[df_original[LABEL_COL] == cls][TEXT_COL].tolist()

        if len(orig_texts) >= k:
            pool = orig_texts
        else:
            # Оригинальных не хватает — добираем аугментированными
            aug_texts = df_augmented[
                (df_augmented[LABEL_COL] == cls)
                & (~df_augmented[TEXT_COL].isin(orig_texts))
            ][TEXT_COL].tolist()
            pool = orig_texts + aug_texts

        # Перемешиваем и берём K штук
        pool_shuffled = list(pool)
        rng.shuffle(pool_shuffled)
        examples[cls] = pool_shuffled[:k]

    return examples


def prepare_few_shot_examples(
    k_values: list[int] = None,
    save_path: str | Path | None = None,
) -> dict:
    """
    Готовит few-shot примеры для всех K и сохраняет в JSON.

    Возвращает структуру:
    {
        "groups": {"Блок ...": "A", ...},
        "examples": {
            "1": {"Блок ...": ["text1"], ...},
            "3": {"Блок ...": ["text1", "text2", "text3"], ...},
            "5": {"Блок ...": ["text1", ..., "text5"], ...}
        }
    }
    """
    if k_values is None:
        k_values = [1, 3, 5]

    save_path = Path(save_path) if save_path else DATA_DIR / "few_shot_examples.json"

    # Загружаем данные
    df_original = load_dataset(stage=0)
    df_augmented = load_dataset(stage=3)

    # Определяем группы
    groups = classify_groups(df_original)

    print(f"[Few-shot] Группы: A={sum(1 for g in groups.values() if g == 'A')}, "
          f"B={sum(1 for g in groups.values() if g == 'B')}, "
          f"C={sum(1 for g in groups.values() if g == 'C')}")

    # Собираем примеры для каждого K
    result = {
        "groups": groups,
        "examples": {},
    }

    for k in k_values:
        examples = select_examples(df_original, df_augmented, k)
        result["examples"][str(k)] = examples

        # Статистика
        total = sum(len(v) for v in examples.values())
        aug_used = sum(
            1 for cls in examples
            if len(df_original[df_original[LABEL_COL] == cls]) < k
        )
        print(f"[Few-shot] K={k}: {total} примеров, "
              f"классов с подмешиванием аугментированных: {aug_used}")

    # Сохраняем
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Few-shot] Сохранено: {save_path}")
    return result


def load_few_shot_examples(path: str | Path | None = None) -> dict:
    """Загружает подготовленные few-shot примеры из JSON."""
    path = Path(path) if path else DATA_DIR / "few_shot_examples.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    prepare_few_shot_examples()
