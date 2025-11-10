import pandas as pd

def _compact(levels):
    return [v for v in levels if v is not None and str(v).strip() != ""]

def _is_parent_child(a_levels, b_levels) -> bool:
    A = _compact(a_levels)
    B = _compact(b_levels)
    if not A or not B or len(A) == len(B):
        return False
    short, long = (A, B) if len(A) < len(B) else (B, A)
    return short == long[:len(short)]

def drop_intra_parent_child(pairs: pd.DataFrame, df_norm: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    idx = df_norm.set_index("id")[["l1_norm","l2_norm","l3_norm"]].to_dict(orient="index")
    keep = []
    for _, r in pairs.iterrows():
        if r["match_scope"] != "intra":
            keep.append(True); continue
        a = idx.get(r["left_id"]); b = idx.get(r["right_d"])
        if not a or not b:
            keep.append(True); continue
        a_levels = [a["l1_norm"], a["l2_norm"], a["l3_norm"]]
        b_levels = [b["l1_norm"], b["l2_norm"], b["l3_norm"]]
        keep.append(not _is_parent_child(a_levels, b_levels))
    return pairs.loc[keep].reset_index(drop=True)


def drop_different_last_child(pairs: pd.DataFrame, df_norm: pd.DataFrame) -> pd.DataFrame:
    # Drops pairs where the last non-empty level is different between the two categories
    if pairs.empty:
        return pairs
    idx = df_norm.set_index("id")[["l1_norm","l2_norm","l3_norm"]].to_dict(orient="index")
    def last_non_empty(levels):
        vals = [v for v in levels if v is not None and str(v).strip() != ""]
        return vals[-1] if vals else None
    keep = []
    for _, r in pairs.iterrows():
        a = idx.get(r["left_id"])
        b = idx.get(r["right_id"])
        if not a or not b:
            keep.append(True); continue
        a_last = last_non_empty([a["l1_norm"], a["l2_norm"], a["l3_norm"]])
        b_last = last_non_empty([b["l1_norm"], b["l2_norm"], b["l3_norm"]])
        keep.append(a_last == b_last)
    return pairs.loc[keep].reset_index(drop=True)
