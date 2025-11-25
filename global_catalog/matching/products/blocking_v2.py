"""
Experimental blocking strategies for product matching.

This module intentionally exposes multiple builder functions so we can iterate on
different candidate generation techniques without entangling them with the original logic.
Each strategy should accept the normalized dataframe plus tuning knobs and return a
DataFrame describing candidate pairs (left/right indices and metadata).
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BlockingConfig:
    """Configuration shared by blocking strategies."""

    threshold: float = 0.3
    max_per_left: int = 200
    include_description: bool = False
    description_token_limit: int = 25
    enforce_uom: bool = False
    measure_tol: float = 0.2
    blocking_key_specs: Optional[List[List[str]]] = None
    window_size: int = 40
    lenient_measure_pass: bool = True
    strategy_name: str = "strategy_one"
    use_local_pairs: bool = False
    pairs_dir: Optional[str] = None


def _safe_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["left_index", "right_index"])
    return df


def blocking_strategy_one(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Brand-aware blocker enforcing identical measure_mg values first, allowing NULLs second."""
    print("[blocking_v2] strategy_one: preparing dataframe")
    df = _prep_sources(df)
    if df.empty:
        print("[blocking_v2] strategy_one: empty dataframe, nothing to block")
        return _safe_frame(None)

    print("[blocking_v2] strategy_one: strict pass (measure match required)")
    strict = _brand_blocking(df, enforce_uom=True, tag="strict_measure")
    print(f"[blocking_v2] strategy_one: strict pass generated {len(strict)} pairs")

    if not cfg.lenient_measure_pass:
        strict.drop_duplicates(subset=["left_index", "right_index"], inplace=True)
        print(f"[blocking_v2] strategy_one: strict-only total unique pairs {len(strict)}")
        return strict.reset_index(drop=True)

    print("[blocking_v2] strategy_one: lenient pass (allow missing measure)")
    lenient = _brand_blocking(df, enforce_uom=False, tag="lenient_measure")
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
    """Multi-key standard blocking leveraging schema-aware specs."""
    return build_product_blocking_pairs(
        df=df,
        threshold=cfg.threshold,
        max_per_left=cfg.max_per_left,
        min_jaccard=cfg.threshold,
        measure_tol=cfg.measure_tol,
        enforce_uom=cfg.enforce_uom,
        use_states=True,
        use_country=True,
        use_strain=True,
        include_description=cfg.include_description,
        description_token_limit=cfg.description_token_limit,
        blocking_key_specs=cfg.blocking_key_specs,
    )


def blocking_strategy_three(
    df: pd.DataFrame,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Sorted neighborhood blocking within brand (or provided blocks)."""
    return build_product_blocking_pairs_sorted_neighborhood(
        df=df,
        threshold=cfg.threshold,
        window_size=cfg.window_size,
        max_per_left=cfg.max_per_left,
        measure_tol=cfg.measure_tol,
        enforce_uom=cfg.enforce_uom,
        include_description=cfg.include_description,
        description_token_limit=cfg.description_token_limit,
        group_cols=cfg.blocking_key_specs,
    )


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
                    m_left = _measure_key(left.at[li, "measure_mg"])
                    m_right = _measure_key(right.at[rj, "measure_mg"])
                    if enforce_uom:
                        if m_left is None or m_right is None or m_left != m_right:
                            continue
                    else:
                        if m_left is not None and m_right is not None and m_left != m_right:
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


def _measure_key(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    text = str(val).strip().lower()
    if not text or text == "each":
        return None
    if text.endswith("-milligrams"):
        try:
            return int(text.split("-")[0])
        except Exception:
            return None
    try:
        return int(float(text))
    except Exception:
        return None


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


# --------------------------------------------------------------------------- #
# Legacy-style blocking with multi-key support

def _tokenize(text):
    if not isinstance(text, str):
        return []
    return [t for t in re.split(r"[^\w]+", text.lower()) if t]


def _try_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return [s]
    return []


def _set_nonempty(x):
    return {str(z).strip().lower() for z in _try_list(x) if str(z).strip()}


def _to_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def _document_frequency_counter(tokens_series: pd.Series):
    c = Counter()
    for tokens in tokens_series:
        if isinstance(tokens, list):
            c.update(set(tokens))
    return c


def _is_missing_measure(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s == "" or s == "each"


def _measure_close(a, b, tol=0.15):
    a = _to_numeric(a)
    b = _to_numeric(b)
    if pd.isna(a) or pd.isna(b):
        return True
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) <= tol * max(a, b)


def _measure_rule_passes(left_mg, right_mg, tol=0.0):
    left_missing = _is_missing_measure(left_mg)
    right_missing = _is_missing_measure(right_mg)
    if left_missing or right_missing:
        return True
    a = _to_numeric(left_mg)
    b = _to_numeric(right_mg)
    if pd.isna(a) or pd.isna(b):
        return True
    return a == b


def _build_inverted_index(df_right):
    postings = defaultdict(list)
    right_token_sets = []
    right_index_to_pos = {}
    for pos, (j, row) in enumerate(df_right.iterrows()):
        toks = row["__blocking_tokens__"] if isinstance(row.get("__blocking_tokens__"), list) else []
        token_set = set(toks)
        right_token_sets.append(token_set)
        right_index_to_pos[j] = pos
        for token in token_set:
            postings[token].append(j)
    return postings, right_token_sets, right_index_to_pos


def _candidate_pairs_within_block(
    block_df: pd.DataFrame,
    threshold=0.60,
    max_per_left=60,
    min_jaccard=None,
    measure_tol=0.0,
    enforce_uom=False,
    use_states=True,
    use_country=True,
    use_strain=True,
):
    right = block_df[block_df["source"] == "hoodie"]
    left = block_df[block_df["source"] == "weedmaps"]
    if left.empty or right.empty:
        return []

    postings, right_token_sets, right_index_to_pos = _build_inverted_index(right)

    out = []
    min_j = threshold if min_jaccard is None else min_jaccard

    for i, row in left.iterrows():
        toks = row["__blocking_tokens__"] if isinstance(row.get("__blocking_tokens__"), list) else []
        left_token_set = set(toks)
        if not left_token_set:
            continue

        pool = set()
        for token in left_token_set:
            for j in postings.get(token, []):
                pool.add(j)
        if not pool:
            continue

        prelim = []
        for j in pool:
            right_token_set = right_token_sets[right_index_to_pos[j]]
            inter = len(left_token_set & right_token_set)
            denom = len(left_token_set) + len(right_token_set) - inter
            jaccard = 0.0 if denom == 0 else inter / denom
            if jaccard < min_j:
                continue

            rrow = right.loc[j]
            if not _measure_rule_passes(row.get("measure_mg"), rrow.get("measure_mg"), tol=measure_tol):
                continue

            prelim.append((i, j, jaccard))

        if not prelim:
            continue

        prelim.sort(key=lambda x: x[2], reverse=True)
        for i_idx, j_idx, jaccard in prelim[:max_per_left]:
            out.append((i_idx, j_idx, float(jaccard)))
    return out


def _build_block_key(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    parts = []
    for col in columns:
        if col not in df.columns:
            parts.append(pd.Series([""] * len(df), index=df.index))
        else:
            parts.append(df[col].fillna("").astype(str).str.strip().str.lower())
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    if len(parts) == 1:
        return parts[0]
    stacked = pd.concat(parts, axis=1)
    return stacked.apply(lambda row: "|".join(row.values), axis=1)


def build_product_blocking_pairs(
    df: pd.DataFrame,
    threshold=0.60,
    max_per_left=60,
    min_jaccard=None,
    measure_tol=0.0,
    enforce_uom=True,
    use_states=True,
    use_country=True,
    use_strain=True,
    include_description=False,
    description_token_limit=25,
    blocking_key_specs: Optional[List[List[str]]] = None,
):
    print(f"[build_product_blocking_pairs] Starting with {len(df)} records, threshold={threshold}, max_per_left={max_per_left}, enforce_uom={enforce_uom}, use_states={use_states}, use_country={use_country}, use_strain={use_strain}, include_description={include_description}")
    df = df.copy()
    df = df[df["source"].isin(["hoodie", "weedmaps"])]
    df = df[pd.notna(df.get("brand_name_norm"))]
    print(f"[build_product_blocking_pairs] Filtered to {len(df)} records with valid source and brand_name_norm")
    if df.empty:
        print("[build_product_blocking_pairs] No records to process after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"])

    if include_description:
        print("[build_product_blocking_pairs] Including description tokens.")
        df["__blocking_tokens__"] = (
            df["product_name_norm"].apply(_tokenize)
            + df["description_norm"].apply(_tokenize).apply(lambda xs: xs[:description_token_limit])
        )
    else:
        df["__blocking_tokens__"] = df["product_name_norm"].apply(_tokenize)

    specs = blocking_key_specs or [["brand_name_norm"]]
    if blocking_key_specs:
        print(f"[build_product_blocking_pairs] Using blocking_key_specs with {len(blocking_key_specs)} configs")
    else:
        brand_count = df["brand_name_norm"].nunique()
        print(f"[build_product_blocking_pairs] Processing {brand_count} unique brands.")

    out_rows = []
    for spec_idx, spec_cols in enumerate(specs):
        missing = [c for c in spec_cols if c not in df.columns]
        spec_label = ", ".join(spec_cols)
        if missing:
            print(f"[build_product_blocking_pairs] Skipping blocking spec {spec_label}: missing columns {missing}")
            continue
        key_col = f"__block_key_{spec_idx}"
        df[key_col] = _build_block_key(df, spec_cols)
        groups = df
        print(f"[build_product_blocking_pairs] Blocking with spec {spec_label}")
        for block_value, group in groups.groupby(key_col, sort=False):
            if group["source"].nunique() < 2:
                continue
            if not blocking_key_specs and spec_cols == ["brand_name_norm"]:
                print(f"[build_product_blocking_pairs] Blocking within brand: {block_value} ({len(group)} records)")
            pairs = _candidate_pairs_within_block(
                group,
                threshold=threshold,
                max_per_left=max_per_left,
                min_jaccard=min_jaccard,
                measure_tol=measure_tol,
                enforce_uom=enforce_uom,
                use_states=use_states,
                use_country=use_country,
                use_strain=use_strain,
            )
            if pairs:
                if not blocking_key_specs and spec_cols == ["brand_name_norm"]:
                    print(f"[build_product_blocking_pairs] Found {len(pairs)} candidate pairs for brand: {block_value}")
                for i, j, jac in pairs:
                    brand_val = group.loc[i].get("brand_name_norm", "")
                    out_rows.append((i, j, brand_val, jac))
        df.drop(columns=[key_col], inplace=True)

    print(f"[build_product_blocking_pairs] Total candidate pairs found: {len(out_rows)}")
    if not out_rows:
        print("[build_product_blocking_pairs] No candidate pairs found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"])

    print("[build_product_blocking_pairs] Returning DataFrame of candidate pairs.")
    out = pd.DataFrame(out_rows, columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"]).drop_duplicates(
        subset=["left_index", "right_index"]
    )
    return out.reset_index(drop=True)


def build_product_blocking_pairs_multipass(
    df: pd.DataFrame,
    threshold_first_pass=0.60,
    threshold_second_pass=0.50,
    max_per_left=60,
    measure_tol=0.0,
    enforce_uom=True,
    use_states=False,
    use_country=False,
    use_strain=False,
    include_description=False,
    description_token_limit=25,
    blocking_key_specs: Optional[List[List[str]]] = None,
):
    print(f"[build_product_blocking_pairs_multipass] Starting multipass blocking with {len(df)} records.")
    df_first = df.copy()
    print(f"[build_product_blocking_pairs_multipass] First pass: threshold={threshold_first_pass}")
    first_pass = build_product_blocking_pairs(
        df=df_first,
        threshold=threshold_first_pass,
        max_per_left=max_per_left,
        min_jaccard=threshold_first_pass,
        measure_tol=measure_tol,
        enforce_uom=enforce_uom,
        use_states=use_states,
        use_country=use_country,
        use_strain=use_strain,
        include_description=include_description,
        description_token_limit=description_token_limit,
        blocking_key_specs=blocking_key_specs,
    )
    print(f"[build_product_blocking_pairs_multipass] First pass found {len(first_pass)} pairs.")

    matched_right = set(first_pass["right_index"])
    all_right = set(df[df["source"] == "hoodie"].index)
    unmatched_right = list(all_right - matched_right)
    print(f"[build_product_blocking_pairs_multipass] Unmatched hoodie records for second pass: {len(unmatched_right)}")

    if not unmatched_right:
        print("[build_product_blocking_pairs_multipass] All hoodie records matched in first pass. Returning results.")
        return first_pass

    df_second = pd.concat([
        df.loc[unmatched_right],
        df[df["source"] == "weedmaps"]
    ])
    print(f"[build_product_blocking_pairs_multipass] Second pass: threshold={threshold_second_pass}, records={len(df_second)}")
    second_pass = build_product_blocking_pairs(
        df=df_second,
        threshold=threshold_second_pass,
        max_per_left=max_per_left,
        min_jaccard=threshold_second_pass,
        measure_tol=measure_tol,
        enforce_uom=enforce_uom,
        use_states=use_states,
        use_country=use_country,
        use_strain=use_strain,
        include_description=include_description,
        description_token_limit=description_token_limit,
        blocking_key_specs=blocking_key_specs,
    )
    print(f"[build_product_blocking_pairs_multipass] Second pass found {len(second_pass)} pairs.")

    combined = pd.concat([first_pass, second_pass]).drop_duplicates([
        "left_index", "right_index"
    ]).reset_index(drop=True)
    print(f"[build_product_blocking_pairs_multipass] Combined total pairs after deduplication: {len(combined)}")
    print("[build_product_blocking_pairs_multipass] Multipass blocking complete. Returning results.")
    return combined


def build_product_blocking_pairs_sorted_neighborhood(
    df: pd.DataFrame,
    threshold: float = 0.60,
    window_size: int = 10,
    max_per_left: int = 60,
    measure_tol: float = 0.0,
    enforce_uom: bool = True,
    include_description: bool = False,
    description_token_limit: int = 25,
    group_cols: Optional[List[List[str]]] = None,
):
    """Sorted neighborhood blocking within schema-aware blocks."""
    print(
        f"[build_product_blocking_pairs_sorted_neighborhood] Starting with {len(df)} records, "
        f"threshold={threshold}, window_size={window_size}, max_per_left={max_per_left}"
    )
    df = df.copy()
    df = df[df["source"].isin(["hoodie", "weedmaps"])]
    df = df[pd.notna(df.get("brand_name_norm"))]
    print(
        "[build_product_blocking_pairs_sorted_neighborhood] Filtered to "
        f"{len(df)} records with valid source and brand_name_norm"
    )
    if df.empty:
        print("[build_product_blocking_pairs_sorted_neighborhood] No records to process. Returning empty DataFrame.")
        return pd.DataFrame(columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"])

    if include_description:
        df["__blocking_tokens__"] = (
            df["product_name_norm"].apply(_tokenize)
            + df["description_norm"].apply(_tokenize).apply(lambda xs: xs[:description_token_limit])
        )
    else:
        df["__blocking_tokens__"] = df["product_name_norm"].apply(_tokenize)

    specs = group_cols or [["brand_name_norm"]]
    if group_cols:
        print(
            f"[build_product_blocking_pairs_sorted_neighborhood] Using group_cols with {len(group_cols)} configs"
        )
    else:
        print(
            f"[build_product_blocking_pairs_sorted_neighborhood] Processing {df['brand_name_norm'].nunique()} brands"
        )

    all_pairs = []
    for spec_idx, spec in enumerate(specs):
        missing = [c for c in spec if c not in df.columns]
        spec_label = ", ".join(spec)
        if missing:
            print(
                f"[build_product_blocking_pairs_sorted_neighborhood] Skipping spec {spec_label} due to missing columns {missing}"
            )
            continue
        key_col = f"__sn_key_{spec_idx}"
        df[key_col] = _build_block_key(df, spec)
        block_df = df
        print(
            f"[build_product_blocking_pairs_sorted_neighborhood] Running sorted neighborhood for spec {spec_label}"
        )
        for block_value, group in block_df.groupby(key_col, sort=False):
            if group["source"].nunique() < 2:
                continue
            pairs = _sorted_neighborhood_pairs(
                group,
                block_value,
                threshold=threshold,
                window_size=window_size,
                max_per_left=max_per_left,
                measure_tol=measure_tol,
            )
            if pairs:
                all_pairs.extend(pairs)
        df.drop(columns=[key_col], inplace=True)

    if not all_pairs:
        print("[build_product_blocking_pairs_sorted_neighborhood] No candidate pairs found.")
        return pd.DataFrame(columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"])

    result = pd.DataFrame(
        all_pairs,
        columns=["left_index", "right_index", "brand_name_norm", "name_jaccard"],
    ).drop_duplicates(["left_index", "right_index"])
    print(
        f"[build_product_blocking_pairs_sorted_neighborhood] Generated {len(result)} candidate pairs total"
    )
    return result.reset_index(drop=True)


def _sorted_neighborhood_pairs(
    block_df: pd.DataFrame,
    block_value: str,
    threshold: float,
    window_size: int,
    max_per_left: int,
    measure_tol: float,
):
    left_mask = block_df["source"].eq("weedmaps")
    right_mask = block_df["source"].eq("hoodie")
    if not left_mask.any() or not right_mask.any():
        return []

    df_sorted = block_df.copy()
    df_sorted["__sort_key"] = df_sorted["product_name_norm"].fillna("").astype(str).str.strip().str.lower()
    df_sorted.sort_values("__sort_key", inplace=True, kind="mergesort")

    rows = df_sorted[["__blocking_tokens__", "measure_mg", "source", "brand_name_norm"]]
    index_list = df_sorted.index.tolist()
    pairs = []
    per_left_counts: Dict[int, int] = {}
    n = len(df_sorted)
    for pos in range(n):
        i_idx = index_list[pos]
        row_i = rows.iloc[pos]
        if row_i["source"] != "weedmaps":
            continue
        token_i = set(row_i["__blocking_tokens__"] or [])
        if not token_i:
            continue
        limit = min(n, pos + window_size + 1)
        per_left = per_left_counts.get(i_idx, 0)
        candidates = []
        for pos2 in range(pos + 1, limit):
            j_idx = index_list[pos2]
            row_j = rows.iloc[pos2]
            if row_j["source"] == "weedmaps":
                continue
            token_j = set(row_j["__blocking_tokens__"] or [])
            if not token_j:
                continue
            inter = len(token_i & token_j)
            denom = len(token_i) + len(token_j) - inter
            jaccard = 0.0 if denom == 0 else inter / denom
            if jaccard < threshold:
                continue
            if not _measure_rule_passes(row_i.get("measure_mg"), row_j.get("measure_mg"), tol=measure_tol):
                continue
            candidates.append((i_idx, j_idx, float(jaccard), row_i.get("brand_name_norm", "")))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[2], reverse=True)
        for cand in candidates:
            if per_left >= max_per_left:
                break
            pairs.append((cand[0], cand[1], cand[3], cand[2]))
            per_left += 1
        per_left_counts[i_idx] = per_left

    return pairs
