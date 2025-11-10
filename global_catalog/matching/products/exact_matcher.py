import pandas as pd

def exact_matches(df_left: pd.DataFrame, df_right: pd.DataFrame, pairs):
    rows = []
    for i, j, block_type in pairs:
        l = df_left.loc[i]
        r = df_right.loc[j]
        if l["brand_key"] != r["brand_key"]:
            continue
        if l["name_norm"] != r["name_norm"]:
            continue
        size_ok = (l["size_val_norm"] == r["size_val_norm"] and l["size_unit_norm"] == r["size_unit_norm"]) or (l["size_unit_norm"] == "" or r["size_unit_norm"] == "")
        if not size_ok:
            continue
        rows.append({
            "left_source": l["source"],
            "right_source": r["source"],
            "left_product_id": l["product_id"],
            "right_product_id": r["product_id"],
            "left_brand": l["brand"],
            "right_brand": r["brand"],
            "left_product_name": l["product_name"],
            "right_product_name": r["product_name"],
            "left_size": l.get("size",""),
            "right_size": r.get("size",""),
            "similarity": 1.0,
            "match_type": "exact",
            "block_type": block_type
        })
    return pd.DataFrame(rows)
