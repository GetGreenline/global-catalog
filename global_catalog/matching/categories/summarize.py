import pandas as pd
from global_catalog.core.path_ops import path_pretty


def attach_pretty_paths(matches: pd.DataFrame, df_like: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches

    if "id" not in df_like.columns:
        raise ValueError("df_like must contain an 'id' column.")
    pretty_map = df_like.set_index("id").apply(
        lambda r: path_pretty(r["level_one"], r["level_two"], r["level_three"]), axis=1
    )

    out = matches.copy()
    if "left_id" not in out.columns or "right_id" not in out.columns:
        raise ValueError("matches must contain 'left_id' and 'right_id' columns.")

    out["left_path_pretty"] = out["left_id"].map(pretty_map)
    out["right_path_pretty"] = out["right_id"].map(pretty_map)
    return out


def summarize_per_category(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["id", "source", "has_intra", "intra_count", "has_cross", "cross_count"])

    def accum(side_prefix: str):
        partner = "right_id" if side_prefix == "left" else "left_id"
        source_col = f"{side_prefix}_source"
        id_col = f"{side_prefix}_id"
        grp = (
            matches.groupby([id_col, source_col, "match_scope"])[partner]
                   .agg(lambda s: set(s))
                   .reset_index()
        )
        intra = grp[grp["match_scope"] == "intra"].set_index([id_col, source_col])[partner].to_dict()
        cross = grp[grp["match_scope"] == "cross"].set_index([id_col, source_col])[partner].to_dict()
        return intra, cross

    intra_l, cross_l = accum("left")
    intra_r, cross_r = accum("right")

    left_df = matches[["left_id", "left_source"]].rename(columns={"left_id": "id", "left_source": "source"})
    right_df = matches[["right_id", "right_source"]].rename(columns={"right_id": "id", "right_source": "source"})
    base = pd.concat([left_df, right_df], ignore_index=True).drop_duplicates()

    def get_counts(row):
        key = (row["id"], row["source"])
        intra_set = intra_l.get(key, set()) | intra_r.get(key, set())
        cross_set = cross_l.get(key, set()) | cross_r.get(key, set())
        return pd.Series({
            "intra_count": len(intra_set),
            "cross_count": len(cross_set),
            "has_intra": len(intra_set) > 0,
            "has_cross": len(cross_set) > 0,
        })

    summary = base.copy()
    summary[["intra_count", "cross_count", "has_intra", "has_cross"]] = summary.apply(get_counts, axis=1)
    return summary


def attach_summary_flags(matches: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or summary.empty:
        return matches

    L = summary.rename(columns={
        "id": "left_id",
        "source": "left_source",
        "has_intra": "left_has_intra",
        "intra_count": "left_intra_count",
        "has_cross": "left_has_cross",
        "cross_count": "left_cross_count",
    })[
        ["left_id", "left_source", "left_has_intra", "left_intra_count", "left_has_cross", "left_cross_count"]
    ]

    R = summary.rename(columns={
        "id": "right_id",
        "source": "right_source",
        "has_intra": "right_has_intra",
        "intra_count": "right_intra_count",
        "has_cross": "right_has_cross",
        "cross_count": "right_cross_count",
    })[
        ["right_id", "right_source", "right_has_intra", "right_intra_count", "right_has_cross", "right_cross_count"]
    ]

    out = matches.merge(L, on=["left_id", "left_source"], how="left")
    out = out.merge(R, on=["right_id", "right_source"], how="left")

    for c in ["left_intra_count", "left_cross_count", "right_intra_count", "right_cross_count"]:
        out[c] = out[c].fillna(0).astype(int)

    for c in ["left_has_intra", "left_has_cross", "right_has_intra", "right_has_cross"]:
        out[c] = out[c].fillna(False).astype(bool)

    return out
