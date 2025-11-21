"""
Simplified fuzzy matcher scaffolding for product candidate pairs.

This module mirrors the structure of the legacy matcher but keeps the logic minimal
so we can iterate without historical baggage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class FuzzyMatcherConfig:
    """Weights and knobs controlling the similarity calculation."""

    threshold: float = 0.75
    include_description: bool = False
    name_weight: float = 0.8
    description_weight: float = 0.1
    uom_weight: float = 0.05
    measure_weight: float = 0.05


def run_fuzzy_matching(
    df_norm: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame],
    cfg: FuzzyMatcherConfig,
) -> pd.DataFrame:
    """Compute fuzzy similarity for provided candidate pairs."""
    if pairs_df is None or pairs_df.empty:
        return _empty_matches()
    # TODO(products): implement actual similarity logic using cfg weights.
    return _empty_matches()


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "left_source",
            "right_source",
            "left_product_name",
            "right_product_name",
            "similarity",
            "match_type",
        ]
    )
