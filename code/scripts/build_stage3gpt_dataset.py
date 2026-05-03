"""
Build a cleaned augmented train set from the existing augmentation checkpoints.

Output:
  Data/data_after_stage3gpt.csv
  Data/data_after_stage3gpt_audit.csv

The script keeps original train rows by default and removes only exact duplicate
originals plus synthetic rows with deterministic, high-confidence artifacts.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
TEXT_COL = "text"
LABEL_COL = "label"

CHECKPOINTS = [
    ("stage0", DATA_DIR / "train_after_eda.csv", 0),
    ("stage1", DATA_DIR / "data_after_stage1.csv", 1),
    ("stage2", DATA_DIR / "data_after_stage2.csv", 2),
    ("stage3", DATA_DIR / "data_after_stage3.csv", 3),
]

OUT_CSV = DATA_DIR / "data_after_stage3gpt.csv"
AUDIT_CSV = DATA_DIR / "data_after_stage3gpt_audit.csv"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def row_key(row: pd.Series) -> tuple[str, str]:
    return normalize_text(row[TEXT_COL]), str(row[LABEL_COL])


def load_checkpoint(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    return df[[TEXT_COL, LABEL_COL]].copy()


def tag_rows_by_stage() -> pd.DataFrame:
    tagged_parts: list[pd.DataFrame] = []
    previous = None

    for name, path, stage in CHECKPOINTS:
        current = load_checkpoint(path)
        if previous is None:
            part = current.copy()
            part["source_stage"] = stage
            part["source_checkpoint"] = name
            part["source_row"] = range(len(part))
            tagged_parts.append(part)
        else:
            prev_counts = Counter(row_key(row) for _, row in previous.iterrows())
            added_rows = []
            for idx, row in current.iterrows():
                key = row_key(row)
                if prev_counts[key] > 0:
                    prev_counts[key] -= 1
                else:
                    added_rows.append(
                        {
                            TEXT_COL: row[TEXT_COL],
                            LABEL_COL: row[LABEL_COL],
                            "source_stage": stage,
                            "source_checkpoint": name,
                            "source_row": idx,
                        }
                    )
            if added_rows:
                tagged_parts.append(pd.DataFrame(added_rows))
        previous = current

    return pd.concat(tagged_parts, ignore_index=True)


def placeholder_set(texts: pd.Series) -> set[str]:
    found: set[str] = set()
    for text in texts.astype(str):
        found.update(re.findall(r"\[[A-Z][A-Z_]*\]", text))
    return found


def synthetic_rejection_reasons(text: str, source_stage: int, allowed_placeholders: set[str]) -> list[str]:
    reasons: list[str] = []
    raw = str(text)
    lower = raw.lower()
    first_800 = raw[:800]

    placeholders = set(re.findall(r"\[[A-Z][A-Z_]*\]", raw))
    unknown_placeholders = placeholders - allowed_placeholders
    if unknown_placeholders:
        reasons.append("unknown_placeholder")

    if re.search(r"(?:^|\n)\s*Класс письма\s*:", raw, flags=re.IGNORECASE):
        reasons.append("class_letter_header")

    if re.search(r"<\d+>|\b<\d+\b|\b\d+>", raw):
        reasons.append("unrestored_angle_placeholder")

    if re.search(r"\|\s*№\s*\|", raw):
        reasons.append("markdown_table")

    if re.search(r"(?:^|\n)\s*[-*]\s+", raw) and source_stage in {1, 2}:
        reasons.append("markdown_bullets")

    field_hits = re.findall(
        r"(?:^|\n)\s*(?:Тема|От|Дата|Адрес|Телефон|E-?mail|Документ)\s*:",
        first_800,
        flags=re.IGNORECASE,
    )
    if len(field_hits) >= 2 and source_stage in {1, 2}:
        reasons.append("template_fields")

    prompt_markers = [
        "на основании предоставленных",
        "для того чтобы правильно определить",
        "рассмотрим примеры",
        "я буду использовать",
        "ниже приведены соответствия",
        "переформулированное письмо",
        "для указанного класса",
    ]
    if any(marker in lower for marker in prompt_markers):
        reasons.append("prompt_meta")

    broken_translation_markers = [
        "честное [person]",
        "тэль:",
        "издание no",
        "нет sp объекта",
        "документ подлинная электронной",
        "подлинная электронной подлинности",
        "этот документ был подписан",
    ]
    if any(marker in lower for marker in broken_translation_markers):
        reasons.append("broken_translation")

    if re.search(r"(.{20,80})\1\1", raw, flags=re.DOTALL):
        reasons.append("repeated_fragment")

    return reasons


def build_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    tagged = tag_rows_by_stage()
    original = tagged[tagged["source_stage"] == 0]
    allowed_placeholders = placeholder_set(original[TEXT_COL])

    audit_rows = []
    seen_keys: set[tuple[str, str]] = set()
    keep_mask = []

    for idx, row in tagged.iterrows():
        key = row_key(row)
        reasons: list[str] = []

        if key in seen_keys:
            reasons.append("exact_duplicate")
        else:
            seen_keys.add(key)

        if int(row["source_stage"]) > 0:
            reasons.extend(
                synthetic_rejection_reasons(
                    row[TEXT_COL],
                    int(row["source_stage"]),
                    allowed_placeholders,
                )
            )

        keep = len(reasons) == 0
        keep_mask.append(keep)
        audit_rows.append(
            {
                "row_id": idx,
                LABEL_COL: row[LABEL_COL],
                "source_stage": row["source_stage"],
                "source_checkpoint": row["source_checkpoint"],
                "source_row": row["source_row"],
                "kept": keep,
                "reasons": ";".join(sorted(set(reasons))),
                "text_len": len(str(row[TEXT_COL])),
                TEXT_COL: row[TEXT_COL],
            }
        )

    audit = pd.DataFrame(audit_rows)
    cleaned = tagged.loc[keep_mask, [TEXT_COL, LABEL_COL]].reset_index(drop=True)
    return cleaned, audit


def main() -> None:
    cleaned, audit = build_clean_dataset()
    cleaned.to_csv(OUT_CSV, index=False)
    audit.to_csv(AUDIT_CSV, index=False)

    print(f"Saved: {OUT_CSV} ({len(cleaned)} rows)")
    print(f"Saved: {AUDIT_CSV} ({len(audit)} audited rows)")
    print("\nRows by stage:")
    print(audit.groupby(["source_stage", "kept"]).size().unstack(fill_value=0))

    rejected = audit[~audit["kept"]]
    if not rejected.empty:
        reason_counts = Counter()
        for value in rejected["reasons"]:
            reason_counts.update(str(value).split(";"))
        print("\nRejected reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

    counts = cleaned[LABEL_COL].value_counts()
    print("\nClass count summary:")
    print(counts.describe().to_string())
    print("\nClasses below 50:")
    below_50 = counts[counts < 50].sort_values()
    print(below_50.to_string() if not below_50.empty else "none")


if __name__ == "__main__":
    main()
