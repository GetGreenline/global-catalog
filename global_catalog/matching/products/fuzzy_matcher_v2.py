"""
Simplified fuzzy matcher scaffolding for product candidate pairs.

This module mirrors the structure of the legacy matcher but keeps the logic minimal
so we can iterate without historical baggage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


@dataclass
class FuzzyMatcherConfig:
    """Weights and knobs controlling the similarity calculation."""

    threshold: float = 0.75
    name_weight: float = 0.9
    uom_weight: float = 0.05
    measure_weight: float = 0.05
    strict_measure_threshold: float = 0.78
    present_vs_null_threshold: float = 0.86
    # Require stronger name agreement by default.
    name_min_score: float = 0.75
    # When True, drop pairs if either side is missing a measure_mg value.
    require_measure_match: bool = False


def run_fuzzy_matching(
    df_norm: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame],
    cfg: FuzzyMatcherConfig,
) -> pd.DataFrame:
    """Compute fuzzy similarity for provided candidate pairs."""
    if pairs_df is None or pairs_df.empty:
        return _empty_matches()
    name_col = "product_name_norm" if "product_name_norm" in df_norm.columns else "normalized_product_name"
    brand_col = "brand_name_norm" if "brand_name_norm" in df_norm.columns else "brand_name"

    left_idx_arr = pairs_df["left_index"].to_numpy()
    right_idx_arr = pairs_df["right_index"].to_numpy()
    unique_idxs = pd.Index(pd.unique(np.concatenate([left_idx_arr, right_idx_arr])))
    df_sub = df_norm.loc[unique_idxs]
    idx_to_pos = {idx: pos for pos, idx in enumerate(df_sub.index)}

    names = df_sub[name_col].fillna("").astype(str).to_numpy()
    uoms = df_sub.get("uom_norm", "").fillna("").astype(str).str.lower().to_numpy()
    measures = df_sub.get("measure_mg_int")
    if measures is None:
        measures = df_sub.get("measure_mg", pd.Series([None] * len(df_sub), index=df_sub.index))
    measures = measures.to_numpy()
    brands = df_sub.get(brand_col, pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()
    packages = (
        df_sub.get("package_size", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    sources = df_sub.get("source", pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()

    records = []
    for left_idx, right_idx in zip(left_idx_arr, right_idx_arr):
        li = idx_to_pos.get(left_idx)
        rj = idx_to_pos.get(right_idx)
        if li is None or rj is None:
            continue

        if not _nonempty_equal(brands[li], brands[rj]):
            continue

        left_pkg = _normalize_package(packages[li])
        right_pkg = _normalize_package(packages[rj])
        if left_pkg and right_pkg and left_pkg != right_pkg:
            continue

        left_name = names[li]
        right_name = names[rj]
        name_score = fuzz.ratio(left_name, right_name) / 100.0

        if name_score < cfg.name_min_score:
            continue

        uom_score = 1.0 if uoms[li] and uoms[li] == uoms[rj] else 0.0
        measure_penalty, measure_score = _measure_enforce_and_score(
            measures[li], measures[rj], cfg.require_measure_match
        )
        if measure_penalty:
            continue

        similarity = (
            cfg.name_weight * name_score
            + cfg.uom_weight * uom_score
            + cfg.measure_weight * measure_score
        )
        if similarity < cfg.threshold:
            continue

        records.append(
            {
                "left_index": int(left_idx),
                "right_index": int(right_idx),
                "left_source": sources[li],
                "right_source": sources[rj],
                "left_product_name": left_name,
                "right_product_name": right_name,
                "left_brand_name": brands[li],
                "right_brand_name": brands[rj],
                "similarity": round(float(similarity), 4),
                "name_score": round(float(name_score), 4),
                "final_score": round(float(similarity), 4),
                "match_type": "fuzzy_v2",
            }
        )

    if not records:
        return _empty_matches()

    return pd.DataFrame(records).sort_values("similarity", ascending=False).reset_index(drop=True)


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "left_index",
            "right_index",
            "left_source",
            "right_source",
            "left_product_name",
            "right_product_name",
            "similarity",
            "name_score",
            "final_score",
            "match_type",
        ]
    )


def _safe_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _normalize_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def _normalize_package(value) -> str:
    text = _normalize_label(value)
    if not text:
        return ""
    try:
        num = float(text)
    except Exception:
        return text
    if num.is_integer():
        return str(int(num))
    return str(num)


def _nonempty_equal(left, right) -> bool:
    left_norm = _normalize_label(left)
    right_norm = _normalize_label(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm


def normalize_measure_value(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(round(value))
    text = str(value).strip().lower()
    if text == "" or text == "each":
        return None
    text = text.replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*-\s*milligrams?$", text)
    if m:
        return int(round(float(m.group(1))))
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(mg|milligrams?)$", text)
    if m:
        return int(round(float(m.group(1))))
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(g|grams?)$", text)
    if m:
        return int(round(float(m.group(1)) * 1000))
    try:
        return int(round(float(text)))
    except Exception:
        return None


def _measure_enforce_and_score(left, right, require_measure_match: bool = False):
    left_norm = normalize_measure_value(left)
    right_norm = normalize_measure_value(right)
    if left_norm is None or right_norm is None:
        if require_measure_match:
            return True, 0.0
        # Lenient: allow when one side is missing, but down-weight the score.
        return False, 0.5
    if left_norm != right_norm:
        return True, 0.0
    return False, 1.0
