#!/usr/bin/env python3
"""Create a cleaned training CSV for quick metric comparisons."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def cyrillic_ratio(text: str) -> float:
    text = str(text)
    if not text:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / len(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter noisy rows from an augmented train CSV."
    )
    parser.add_argument("--input", default="Data/data_after_stage3.csv")
    parser.add_argument("--output", default="Data/data_after_stage3_clean.csv")
    parser.add_argument("--min-len", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=5000)
    parser.add_argument("--min-cyr-ratio", type=float, default=0.35)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    before = len(df)

    text = df[args.text_col].fillna("").astype(str)
    lengths = text.str.len()
    ratios = text.map(cyrillic_ratio)

    mask = (
        (lengths >= args.min_len)
        & (lengths <= args.max_len)
        & (ratios >= args.min_cyr_ratio)
        & df[args.label_col].notna()
    )

    cleaned = df.loc[mask].copy()
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=[args.text_col, args.label_col])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    removed_by_length_low = int((lengths < args.min_len).sum())
    removed_by_length_high = int((lengths > args.max_len).sum())
    removed_by_cyr = int((ratios < args.min_cyr_ratio).sum())
    removed_by_dedup = before_dedup - len(cleaned)

    counts = cleaned[args.label_col].value_counts()

    print(f"Input:  {input_path} ({before} rows)")
    print(f"Output: {output_path} ({len(cleaned)} rows)")
    print(f"Removed total: {before - len(cleaned)}")
    print(f"  short < {args.min_len}: {removed_by_length_low}")
    print(f"  long > {args.max_len}: {removed_by_length_high}")
    print(f"  cyr_ratio < {args.min_cyr_ratio}: {removed_by_cyr}")
    print(f"  duplicates after filters: {removed_by_dedup}")
    print(
        "Class counts after clean: "
        f"classes={len(counts)}, min={counts.min()}, "
        f"median={counts.median():.0f}, max={counts.max()}"
    )


if __name__ == "__main__":
    main()
