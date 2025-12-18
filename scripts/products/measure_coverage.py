#!/usr/bin/env python3
"""Report Hoodie product measure coverage before/after normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_catalog.transformers.products.products_normalization import ProductNormalizer


MISSING_SENTINELS = {"", "nan", "none", "null"}


def count_present(series: pd.Series) -> int:
    """Count entries that contain a usable value."""
    text = series.astype(str).str.strip().str.lower()
    mask = text.isin(MISSING_SENTINELS)
    return int(len(series) - int(mask.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hoodie measure coverage before and after normalization."
    )
    parser.add_argument(
        "--hoodie",
        default="data/snapshots/products/products_hoodie.csv",
        type=Path,
        help="Path to the Hoodie products CSV snapshot.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.hoodie, dtype=str)
    normalizer = ProductNormalizer()
    norm_df = normalizer.process(hoodie_df=df)

    total_rows = len(df)
    original_measure = count_present(df["measure"])
    normalized_measure = count_present(norm_df["measure"]) if "measure" in norm_df.columns else 0
    measure_mg = count_present(norm_df["measure_mg"]) if "measure_mg" in norm_df.columns else 0

    print("Hoodie products measure coverage")
    print("--------------------------------")
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with raw `measure`: {original_measure:,}")
    print(f"Rows with normalized `measure`: {normalized_measure:,}")
    print(f"Rows with normalized `measure_mg`: {measure_mg:,}")


if __name__ == "__main__":
    main()
