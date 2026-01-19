"""Token-set fuzzy matcher prioritizing order-invariant name comparisons."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from global_catalog.matching.products.fuzzy_matcher_v2 import (
    FuzzyMatcherConfig,
    _empty_matches,
    _measure_enforce_and_score,
)
from global_catalog.common.logger import Logger


LOGGER = Logger("ProductFuzzyV3Token")


def _token_jaccard(left_name: str, right_name: str) -> float:
    """Return Jaccard similarity across whitespace-delimited tokens."""
    left_tokens = {token for token in left_name.split() if token}
    right_tokens = {token for token in right_name.split() if token}
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _normalize_label(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def _normalize_package(value: str) -> str:
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


def _nonempty_equal(left: str, right: str) -> bool:
    left_norm = _normalize_label(left)
    right_norm = _normalize_label(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm


def run_fuzzy_matching_token_set(
    df_norm: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame],
    cfg: FuzzyMatcherConfig,
    name_min_score: Optional[float] = None,
    require_disambiguator: bool = False,
) -> pd.DataFrame:
    """Compute fuzzy similarity using token-set ratios for name comparisons."""
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
    uoms = (
        df_sub.get("uom_norm", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
    )
    measures = df_sub.get("measure_mg_int")
    if measures is None:
        measures = df_sub.get("measure_mg", pd.Series([None] * len(df_sub), index=df_sub.index))
    measures = measures.to_numpy()
    brands = df_sub.get(brand_col, pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()
    sources = df_sub.get("source", pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()
    packages = (
        df_sub.get("package_size", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    extract_types = (
        df_sub.get("extract_type_norm", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    strain_or_flavor = (
        df_sub.get("strain_or_flavor_norm", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    product_lines = (
        df_sub.get("product_line_norm", pd.Series([""] * len(df_sub), index=df_sub.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    records = []
    min_name_score = cfg.name_min_score if name_min_score is None else name_min_score
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

        token_score = fuzz.token_set_ratio(names[li], names[rj]) / 100.0
        jaccard_score = _token_jaccard(names[li], names[rj])
        name_score = token_score * jaccard_score

        if name_score < min_name_score:
            continue

        uom_score = 1.0 if uoms[li] and uoms[li] == uoms[rj] else 0.0
        measure_penalty, measure_score = _measure_enforce_and_score(
            measures[li], measures[rj], cfg.require_measure_match
        )
        if measure_penalty:
            continue

        similarity = (
            cfg.name_weight * token_score
            + cfg.uom_weight * uom_score
            + cfg.measure_weight * measure_score
        )
        if similarity < cfg.threshold:
            continue

        extract_match = _nonempty_equal(extract_types[li], extract_types[rj])
        strain_match = _nonempty_equal(strain_or_flavor[li], strain_or_flavor[rj])
        product_line_match = _nonempty_equal(product_lines[li], product_lines[rj])
        bonus = 0.0
        if extract_match:
            bonus += 0.05
        if strain_match:
            bonus += 0.05
        if product_line_match:
            bonus += 0.03
        if require_disambiguator and bonus <= 0.0:
            continue
        final_score = similarity + bonus

        records.append(
            {
                "left_index": int(left_idx),
                "right_index": int(right_idx),
                "left_source": sources[li],
                "right_source": sources[rj],
                "left_product_name": names[li],
                "right_product_name": names[rj],
                "left_brand_name": brands[li],
                "right_brand_name": brands[rj],
                "similarity": round(float(similarity), 4),
                "name_score": round(float(name_score), 4),
                "final_score": round(float(final_score), 4),
                "match_type": "fuzzy_v3_token",
            }
        )

    if not records:
        return _empty_matches()

    LOGGER.info(f"Token-set matcher retained {len(records)} matches out of {len(pairs_df)} candidates.")
    return pd.DataFrame(records).sort_values("similarity", ascending=False).reset_index(drop=True)
