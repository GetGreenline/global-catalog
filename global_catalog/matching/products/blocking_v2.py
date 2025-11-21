"""
Experimental blocking strategies for product matching.

This module intentionally exposes multiple builder functions so we can iterate on
different candidate generation techniques without entangling them with the original logic.
Each strategy should accept the normalized dataframe plus tuning knobs and return a
DataFrame describing candidate pairs (left/right indices and metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


@dataclass
class BlockingConfig:
    """Configuration shared by blocking strategies."""

    threshold: float = 0.6
    max_per_left: int = 200
    include_description: bool = False
    description_token_limit: int = 0
    enforce_uom: bool = False
    measure_tol: float = 0.0


def _safe_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["left_index", "right_index"])
    return df


def blocking_strategy_one(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Brand-aware blocker with two passes (strict and lenient UOM handling)."""
    print("[blocking_v2] strategy_one: preparing dataframe")
    df = _prep_sources(df)
    if df.empty:
        print("[blocking_v2] strategy_one: empty dataframe, nothing to block")
        return _safe_frame(None)

    print("[blocking_v2] strategy_one: strict pass (enforce_uom=True)")
    strict = _brand_blocking(df, enforce_uom=True, tag="strict_uom")
    print(f"[blocking_v2] strategy_one: strict pass generated {len(strict)} pairs")
    print("[blocking_v2] strategy_one: lenient pass (allow missing UOM)")
    lenient = _brand_blocking(df, enforce_uom=False, tag="lenient_uom")
    print(f"[blocking_v2] strategy_one: lenient pass generated {len(lenient)} pairs")
    out = pd.concat([strict, lenient], ignore_index=True)
    if out.empty:
        print("[blocking_v2] strategy_one: no pairs found after both passes")
        return _safe_frame(None)
    out.drop_duplicates(subset=["left_index", "right_index"], inplace=True)
    print(f"[blocking_v2] strategy_one: total unique pairs {len(out)}")
    return out.reset_index(drop=True)


def blocking_strategy_two(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Placeholder for strategy #2 (e.g., multi-pass thresholds)."""
    # TODO(products): implement second blocking approach.
    return _safe_frame(None)


def blocking_strategy_three(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Placeholder for strategy #3 (e.g., token-based jaccard)."""
    # TODO(products): implement third blocking approach.
    return _safe_frame(None)


def build_candidates(
    df: pd.DataFrame,
    cfg: BlockingConfig,
    strategy: Callable[[pd.DataFrame, BlockingConfig], pd.DataFrame],
) -> Dict[str, Any]:
    """Convenience wrapper to run whichever strategy is being tested."""
    pairs_df = strategy(df, cfg)
    baseline = _all_pairs_count(df)
    candidate_count = 0 if pairs_df is None else len(pairs_df)
    reduction_ratio = 1.0 if baseline == 0 else 1.0 - (candidate_count / baseline)
    return {
        "pairs": pairs_df,
        "metrics": {
            "baseline_pairs": baseline,
            "candidate_pairs": candidate_count,
            "reduction_ratio": reduction_ratio,
        },
    }


# --------------------------------------------------------------------------- #
# Helpers

def _prep_sources(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    required = ["brand_name_norm", "source", "uom_norm"]
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df.copy()


def _brand_blocking(df: pd.DataFrame, enforce_uom: bool, tag: str) -> pd.DataFrame:
    rows = []
    grouped = df[df["brand_name_norm"].astype(str).str.strip().ne("")]
    if grouped.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "brand_name_norm", "pass_tag"])

    for brand, group in grouped.groupby("brand_name_norm"):
        print(f"[blocking_v2] {_pass_desc(tag)} brand='{brand}' rows={len(group)}")
        sources = group["source"].dropna().unique()
        if len(sources) < 2:
            continue
        for src_left, src_right in combinations(sources, 2):
            left = group[group["source"] == src_left]
            right = group[group["source"] == src_right]
            if left.empty or right.empty:
                continue
            for li in left.index:
                for rj in right.index:
                    u_left = _clean_str(left.at[li, "uom_norm"])
                    u_right = _clean_str(right.at[rj, "uom_norm"])
                    if enforce_uom:
                        if not u_left or not u_right or u_left != u_right:
                            continue
                    else:
                        if u_left and u_right and u_left != u_right:
                            continue
                        if u_left and u_right:
                            # Already emitted by strict pass
                            continue
                    rows.append(
                        {
                            "left_index": li,
                            "right_index": rj,
                            "brand_name_norm": brand,
                            "pass_tag": tag,
                        }
                    )
    return pd.DataFrame(rows)


def _clean_str(val: Any) -> str:
    return str(val).strip().lower() if isinstance(val, str) else (str(val).strip().lower() if pd.notna(val) else "")


def _all_pairs_count(df: pd.DataFrame) -> int:
    src_counts = df.groupby("source").size().to_dict()
    items = list(src_counts.values())
    if len(items) < 2:
        return 0
    total = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total += items[i] * items[j]
    return total


def _pass_desc(tag: str) -> str:
    return f"pass[{tag}]"
