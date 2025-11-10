import math
import re
import ast
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

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

def _document_frequency_counter(tokens_series: pd.Series) -> Counter:
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
    if _is_missing_measure(right_mg):
        return True
    a = _to_numeric(left_mg)
    b = _to_numeric(right_mg)
    if pd.isna(a) or pd.isna(b):
        return True
    return _measure_close(a, b, tol=tol)

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

def _candidate_pairs_within_brand(
    block_df,
    threshold=0.60,
    max_per_left=60,
    min_jaccard=None,
    measure_tol=0.0,
    enforce_uom=False,
    use_states=True,
    use_country=True,
    use_strain=True
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
    description_token_limit=25
):
    print(f"[build_product_blocking_pairs] Starting with {len(df)} records, threshold={threshold}, max_per_left={max_per_left}, enforce_uom={enforce_uom}, use_states={use_states}, use_country={use_country}, use_strain={use_strain}, include_description={include_description}")
    df = df.copy()
    df = df[df["source"].isin(["hoodie", "weedmaps"])]
    df = df[pd.notna(df.get("brand_name_norm"))]
    print(f"[build_product_blocking_pairs] Filtered to {len(df)} records with valid source and brand_name_norm")
    if df.empty:
        print("[build_product_blocking_pairs] No records to process after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=["left_index","right_index","brand_name_norm","name_jaccard"])

    if include_description:
        print("[build_product_blocking_pairs] Including description tokens.")
        df["__blocking_tokens__"] = (
            df["product_name_norm"].apply(_tokenize)
            + df["description_norm"].apply(_tokenize).apply(lambda xs: xs[:description_token_limit])
        )
    else:
        df["__blocking_tokens__"] = df["product_name_norm"].apply(_tokenize)

    out_rows = []
    brand_count = df["brand_name_norm"].nunique()
    print(f"[build_product_blocking_pairs] Processing {brand_count} unique brands.")
    for brand_name_norm, group in df.groupby("brand_name_norm", sort=False):
        print(f"[build_product_blocking_pairs] Blocking within brand: {brand_name_norm} ({len(group)} records)")
        pairs = _candidate_pairs_within_brand(
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
        print(f"[build_product_blocking_pairs] Found {len(pairs)} candidate pairs for brand: {brand_name_norm}")
        for i, j, jac in pairs:
            out_rows.append((i, j, brand_name_norm, jac))

    print(f"[build_product_blocking_pairs] Total candidate pairs found: {len(out_rows)}")
    if not out_rows:
        print("[build_product_blocking_pairs] No candidate pairs found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["left_index","right_index","brand_name_norm","name_jaccard"])

    print("[build_product_blocking_pairs] Returning DataFrame of candidate pairs.")
    return pd.DataFrame(out_rows, columns=["left_index","right_index","brand_name_norm","name_jaccard"]).reset_index(drop=True)

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
    )
    print(f"[build_product_blocking_pairs_multipass] Second pass found {len(second_pass)} pairs.")

    combined = pd.concat([first_pass, second_pass]).drop_duplicates([
        "left_index", "right_index"
    ]).reset_index(drop=True)
    print(f"[build_product_blocking_pairs_multipass] Combined total pairs after deduplication: {len(combined)}")
    print("[build_product_blocking_pairs_multipass] Multipass blocking complete. Returning results.")
    return combined
