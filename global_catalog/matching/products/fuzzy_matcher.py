import pandas as pd
from difflib import SequenceMatcher
import re
import numpy as np

def _seq_ratio(a, b):
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if not a and not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def _norm_mg_value(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return int(round(float(x)))
    s = str(x).strip().lower()
    if s == "" or s == "each":
        return None
    m = re.match(r"^(\d+)\s*-\s*milligrams$", s) or re.match(r"^(\d+)\s*mg$", s)
    if m:
        return int(m.group(1))
    try:
        return int(round(float(s)))
    except Exception:
        return None


def _measure_enforce_and_score(a, b):
    va = _norm_mg_value(a)
    vb = _norm_mg_value(b)
    if va is None or vb is None:
        return (False, 0.5)
    if va == vb:
        return (False, 1.0)
    return (True, 0.0)


def fuzzy_matches_from_pairs(
    df_norm: pd.DataFrame,
    pairs_df: pd.DataFrame,
    threshold: float = 0.6,
    include_description: bool = False,
    weights: dict | None = None,
):

    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=[
            "left_source","right_source",
            "left_product_id","right_product_id",
            "left_brand_name","right_brand_name",
            "left_product_name","right_product_name",
            "left_uom","right_uom",
            "left_measure_mg","right_measure_mg",
            "similarity","match_type","pair_name_score"
        ])


    w = {
        "name_seq": 0.80,
        "name_tok": 0.00,
        "desc_seq": 0.05 if include_description else 0.0,
        "uom": 0.05,
        "measure": 0.10,
    }
    if weights:
        w.update(weights)

    name_col = "product_name_norm" if "product_name_norm" in df_norm.columns else "normalized_product_name"
    desc_col = "description_norm" if "description_norm" in df_norm.columns else "normalized_description"
    uom_col  = "uom_norm" if "uom_norm" in df_norm.columns else "uom"
    brand_name_col = next((c for c in ["brand_name_norm", "brand_name", "brand"] if c in df_norm.columns), None)

    name_score_map = {}
    if "name_score" in pairs_df.columns:
        name_score_map = {(int(l), int(r)): float(s)
                          for l, r, s in zip(pairs_df["left_index"], pairs_df["right_index"], pairs_df["name_score"])}

    rows = []
    for i, j in zip(pairs_df["left_index"], pairs_df["right_index"]):
        li = df_norm.loc[i]
        rj = df_norm.loc[j]


        ln = li.get(name_col, "") or ""
        rn = rj.get(name_col, "") or ""
        name_seq = _seq_ratio(ln, rn)

        desc_seq = _seq_ratio(li.get(desc_col, "") or "", rj.get(desc_col, "") or "") if include_description else 0.0


        uomi = (li.get(uom_col) or "").strip().lower()
        uomj = (rj.get(uom_col) or "").strip().lower()
        uom_sim = 1.0 if uomi and uomj and uomi == uomj else 0.0


        must_drop, meas_sim = _measure_enforce_and_score(li.get("measure_mg"), rj.get("measure_mg"))
        if must_drop:
            continue


        sim = (
            w["name_seq"] * name_seq +
            w["desc_seq"] * desc_seq +
            w["uom"] * uom_sim +
            w["measure"] * meas_sim
        )


        if sim >= threshold:
            rows.append({
                "left_source": li.get("source",""),
                "right_source": rj.get("source",""),
                "left_brand_name": li.get(brand_name_col,"") if brand_name_col else "",
                "right_brand_name": rj.get(brand_name_col,"") if brand_name_col else "",
                "left_product_name": li.get("product_name_norm",""),
                "right_product_name": rj.get("product_name_norm",""),
                "left_uom": li.get(uom_col,""),
                "right_uom": rj.get(uom_col,""),
                "left_measure_mg": li.get("measure_mg",""),
                "right_measure_mg": rj.get("measure_mg",""),
                "similarity": round(float(sim), 4),
                "match_type": "fuzzy",
                "pair_name_score": name_score_map.get((int(i), int(j)), np.nan),
            })

    # Return sorted DataFrame of matches
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out
