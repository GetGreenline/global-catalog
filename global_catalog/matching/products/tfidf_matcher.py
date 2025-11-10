import re
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _must_drop_measure(a, b) -> bool:
    va = _norm_mg_value(a)
    vb = _norm_mg_value(b)
    if va is None or vb is None:
        return False
    return va != vb


def _strict_uom_eq(u1, u2) -> bool:
    s1 = ("" if u1 is None else str(u1)).strip().lower()
    s2 = ("" if u2 is None else str(u2)).strip().lower()
    if not s1 or not s2:
        return True
    return s1 == s2


def _cosine_from_rows(X, i_pos: int, j_pos: int, pre_l2_normalized: bool = True) -> float:
    if pre_l2_normalized:

        return float(X[i_pos].multiply(X[j_pos]).sum())
    xi = X[i_pos]
    xj = X[j_pos]
    num = float(xi.multiply(xj).sum())
    den = np.sqrt(xi.multiply(xi).sum()) * np.sqrt(xj.multiply(xj).sum())
    return num / den if den > 0 else 0.0


def tfidf_matches_from_pairs(
    df_norm: pd.DataFrame,
    pairs_df: pd.DataFrame,
    threshold: float = 0.8,
    include_description: bool = False,
    weights: Optional[Dict[str, float]] = None,
):

    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=[
            "left_source","right_source",
            "left_product_id","right_product_id",
            "left_brand_name","right_brand_name",
            "left_product_name","right_product_name",
            "left_uom","right_uom",
            "left_measure_mg","right_measure_mg",
            "similarity","match_type","pair_name_cosine"
        ])


    name_col = "product_name_norm" if "product_name_norm" in df_norm.columns else "normalized_product_name"
    desc_col = "description_norm" if "description_norm" in df_norm.columns else "normalized_description"
    uom_col  = "uom_norm" if "uom_norm" in df_norm.columns else "uom"
    brand_name_col = next((c for c in ["brand_name_norm", "brand_name", "brand"] if c in df_norm.columns), None)

    idxs = pd.Index(sorted(set(pairs_df["left_index"].tolist() + pairs_df["right_index"].tolist())))
    df_sub = df_norm.loc[idxs]


    vect_name = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1)
    X_name = vect_name.fit_transform(df_sub[name_col].fillna("").astype(str))


    vect_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    X_char = vect_char.fit_transform(df_sub[name_col].fillna("").astype(str))


    X_desc = None
    if include_description:
        vect_desc = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)
        X_desc = vect_desc.fit_transform(df_sub[desc_col].fillna("").astype(str))

    pos_map = {idx: pos for pos, idx in enumerate(df_sub.index)}


    if include_description:
        w = {"name": 0.7, "char": 0.2, "desc": 0.1}
    else:
        w = {"name": 0.8, "char": 0.2, "desc": 0.0}
    if weights:
        w.update(weights)

    rows = []

    for i, j in zip(pairs_df["left_index"], pairs_df["right_index"]):
        li = df_norm.loc[i]
        rj = df_norm.loc[j]


        if _must_drop_measure(li.get("measure_mg"), rj.get("measure_mg")):
            continue


        pi = pos_map[i]
        pj = pos_map[j]

        cos_name = _cosine_from_rows(X_name, pi, pj, pre_l2_normalized=True)
        cos_char = _cosine_from_rows(X_char, pi, pj, pre_l2_normalized=True)
        cos_desc = _cosine_from_rows(X_desc, pi, pj, pre_l2_normalized=True) if (include_description and X_desc is not None) else 0.0


        sim = w["name"] * cos_name + w["char"] * cos_char + w["desc"] * cos_desc
        if sim >= threshold:
            rows.append({
                "left_source": li.get("source",""),
                "right_source": rj.get("source",""),
                "left_brand_name": li.get(brand_name_col,"") if brand_name_col else "",
                "right_brand_name": rj.get(brand_name_col,"") if brand_name_col else "",
                "left_product_name": li.get("product_name_norm",""),
                "right_product_name": rj.get("product_name_norm",""),
                "left_measure_mg": li.get("measure_mg",""),
                "right_measure_mg": rj.get("measure_mg",""),
                "similarity": round(float(sim), 4),
                "match_type": "tfidf",
                "pair_name_cosine": round(float(cos_name), 4),
            })


    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out
