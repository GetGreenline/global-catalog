import argparse
from pathlib import Path
from hashlib import md5
from uuid import UUID
from datetime import datetime, timezone

import re
import numpy as np
import pandas as pd


def _uuid_from_hash(s: str) -> str:
    return str(UUID(md5(s.encode("utf-8")).hexdigest()))

def _ascii_lower_collapse(val) -> str:
    if val is None:
        s = ""
    else:
        s = str(val)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _build_row_id(source: str, l1: str, l2: str, l3: str) -> str:
    key = f"v1|{_ascii_lower_collapse(source)}|{_ascii_lower_collapse(l1)}|{_ascii_lower_collapse(l2)}|{_ascii_lower_collapse(l3)}"
    return md5(key.encode("utf-8")).hexdigest()

_NA_MARKERS = {"", "nan", "none", "null", "uncategorized"}

def _split_pretty(pp):
    if pp is None or (isinstance(pp, float) and pd.isna(pp)):
        return None, None, None
    parts = [p.strip() for p in str(pp).split("/")][:3]
    parts += [""] * (3 - len(parts))

    def _norm(x):
        s = "" if x is None else str(x).strip().lower()
        return None if s in _NA_MARKERS else x

    a, b, c = parts[0], parts[1], parts[2]
    return _norm(a), _norm(b), _norm(c)

def _count_levels(pp):
    l1, l2, l3 = _split_pretty(pp)
    return sum(v is not None and str(v).strip() != "" for v in (l1, l2, l3))

def _winner_side(row):
    ls = str(row.get("left_source", "")).strip().lower()
    rs = str(row.get("right_source", "")).strip().lower()
    if ls == "left" and rs == "right":
        return "left"
    if ls == "right" and rs == "left":
        return "right"
    lc = _count_levels(row["left_path_pretty"])
    rc = _count_levels(row["right_path_pretty"])
    if lc > rc:
        return "left"
    if rc > lc:
        return "right"
    if str(row["left_source"]) < str(row["right_source"]):
        return "left"
    if str(row["right_source"]) < str(row["left_source"]):
        return "right"
    return "left" if str(row.get("left_id", "")) <= str(row.get("right_id", "")) else "right"

def _ensure_cats_contract(cats_df: pd.DataFrame) -> pd.DataFrame:
    df = cats_df.copy()

    for c in ("source", "level_one", "level_two", "level_three"):
        if c not in df.columns:
            df[c] = ""

    if "category_id" not in df.columns:
        df["category_id"] = ""
    else:
        df["category_id"] = df["category_id"].astype(object)
        df["category_id"] = df["category_id"].where(df["category_id"].notna(), "")
        df["category_id"] = df["category_id"].astype(str).str.strip()
        df.loc[df["category_id"].str.lower().isin(["nan"]), "category_id"] = ""

    if "global_id" not in df.columns:
        df["global_id"] = ""
    else:
        df["global_id"] = df["global_id"].astype(object)
        df["global_id"] = df["global_id"].where(df["global_id"].notna(), "")
        df["global_id"] = df["global_id"].astype(str).str.strip()
        df.loc[df["global_id"].str.lower().isin(["nan", "none"]), "global_id"] = ""

    if "updated_at" not in df.columns:
        df["updated_at"] = pd.Timestamp.now()
    else:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        # Replace NaT values with current timestamp
        nat_mask = pd.isna(df["updated_at"])
        if nat_mask.any():
            current_time = pd.Timestamp.now()
            df.loc[nat_mask, "updated_at"] = current_time

    if "id" not in df.columns:
        df["id"] = ""
    need_id = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
    if need_id.any():
        df.loc[need_id, "id"] = df.loc[need_id].apply(
            lambda r: _build_row_id(
                r.get("source_raw", r.get("source", "")),
                r.get("level_one", ""),
                r.get("level_two", ""),
                r.get("level_three", ""),
            ),
            axis=1,
        )

    return df


def build_resolution_from_pairs(pairs: pd.DataFrame, cats_df: pd.DataFrame) -> pd.DataFrame:
    p = pairs[pairs["match_scope"] == "cross"].copy()
    if p.empty:
        return pd.DataFrame(columns=[
            "id", "category_id", "level_one", "level_two", "level_three", "source",
            "pretty_path", "dropped_pretty_path", "dropped_source", "dropped_id", "dropped_category_id",
            "updated_at", "pair_similarity", "pair_match_type"
        ])

    if "left_path_pretty" not in p.columns or "right_path_pretty" not in p.columns:
        raise ValueError("pairs must contain left_path_pretty and right_path_pretty")

    cats_df = _ensure_cats_contract(cats_df)

    id_to_cat = (
        cats_df.drop_duplicates(subset=["id"])
               .set_index("id")[["category_id", "updated_at"]]
               .to_dict(orient="index")
    )

    p["pair_key"] = p.apply(
        lambda r: "|".join(sorted([
            f"{r['left_source']}::{r['left_id']}",
            f"{r['right_source']}::{r['right_id']}"
        ])),
        axis=1
    )
    p = p.sort_values(["pair_key", "similarity"], ascending=[True, False]).drop_duplicates("pair_key")
    # Left side is always the anchor for global_id attribution.
    p["winner"] = "left"

    keep_cols = {"left": ["left_id", "left_source", "left_path_pretty"],
                 "right": ["right_id", "right_source", "right_path_pretty"]}
    drop_cols = {"left": ["right_id", "right_source", "right_path_pretty"],
                 "right": ["left_id", "left_source", "left_path_pretty"]}

    rows = []
    for _, r in p.iterrows():
        w = r["winner"]
        kid, ks, kp = r[keep_cols[w][0]], r[keep_cols[w][1]], r[keep_cols[w][2]]
        did, ds, dp = r[drop_cols[w][0]], r[drop_cols[w][1]], r[drop_cols[w][2]]

        wl1, wl2, wl3 = _split_pretty(kp)

        kc = id_to_cat.get(kid, {}).get("category_id", "")
        dc = id_to_cat.get(did, {}).get("category_id", "")
        kupd = id_to_cat.get(kid, {}).get("updated_at", pd.Timestamp.now())

        rows.append({
            "id": kid,
            "category_id": kc,
            "level_one": wl1,
            "level_two": wl2,
            "level_three": wl3,
            "source": ks,
            "pretty_path": kp,
            "dropped_pretty_path": dp,
            "dropped_source": ds,
            "dropped_id": did,
            "dropped_category_id": dc,
            "pair_similarity": r.get("similarity", np.nan),
            "pair_match_type": r.get("match_type", None),
            "updated_at": kupd,
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["id", "pair_similarity"], ascending=[True, False]).drop_duplicates("id")

    res = res[[
        "id", "category_id", "level_one", "level_two", "level_three", "source",
        "pretty_path", "dropped_pretty_path", "dropped_source", "dropped_id", "dropped_category_id",
        "updated_at", "pair_similarity", "pair_match_type"
    ]]
    return res


def build_intra_resolution_from_pairs(pairs: pd.DataFrame, cats_df: pd.DataFrame):
    p = pairs[pairs["match_scope"] == "intra"].copy()
    cols = ["id", "category_id", "level_one", "level_two", "level_three", "source",
            "pretty_path", "dropped_pretty_path", "dropped_source", "dropped_id", "dropped_category_id",
            "updated_at", "intra_policy"]
    if p.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    cats_df = _ensure_cats_contract(cats_df)

    pretty = {}
    for _, r in p.iterrows():
        if r["left_id"] not in pretty:
            pretty[r["left_id"]] = r.get("left_path_pretty")
        if r["right_id"] not in pretty:
            pretty[r["right_id"]] = r.get("right_path_pretty")

    id_to_cat = (
        cats_df.drop_duplicates(subset=["id"])
               .set_index("id")[["category_id", "updated_at"]]
               .to_dict(orient="index")
    )

    def score(cid):
        pp = pretty.get(cid)
        ts = id_to_cat.get(cid, {}).get("updated_at", pd.Timestamp.now())
        return (_count_levels(pp or ""), pd.to_datetime(ts) if pd.notna(ts) else pd.Timestamp.now(), str(cid))

    keep_rows = []
    pairwise_rows = []

    for src, grp in p.groupby("left_source"):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for _, r in grp.iterrows():
            union(r["left_id"], r["right_id"])

        comp = {}
        ids = set(pd.concat([grp["left_id"], grp["right_id"]], ignore_index=True).tolist())
        for cid in ids:
            root = find(cid)
            comp.setdefault(root, set()).add(cid)

        for members in comp.values():
            if len(members) < 2:
                continue
            best = sorted(
                members,
                key=lambda c: (
                    -score(c)[0],
                    -(score(c)[1].timestamp() if (isinstance(score(c)[1], pd.Timestamp) and not pd.isna(score(c)[1])) else -1e18),
                    score(c)[2]
                )
            )[0]
            wl1, wl2, wl3 = _split_pretty(pretty.get(best))
            dropped = [c for c in members if c != best]
            dropped_paths = [pretty.get(c) for c in dropped]

            keep_rows.append({
                "id": best,
                "category_id": id_to_cat.get(best, {}).get("category_id", ""),
                "level_one": wl1,
                "level_two": wl2,
                "level_three": wl3,
                "source": src,
                "pretty_path": pretty.get(best),
                "dropped_pretty_path": " | ".join([x for x in dropped_paths if x]),
                "dropped_source": src,
                "dropped_id": " | ".join(map(str, dropped)),
                "dropped_category_id": " | ".join([id_to_cat.get(x, {}).get("category_id", "") for x in dropped]),
                "updated_at": id_to_cat.get(best, {}).get("updated_at", pd.Timestamp.now()),
                "intra_policy": "most_complete_then_newest"
            })
            for dc, dp in zip(dropped, dropped_paths):
                pairwise_rows.append({
                    "id": best,
                    "category_id": id_to_cat.get(best, {}).get("category_id", ""),
                    "level_one": wl1,
                    "level_two": wl2,
                    "level_three": wl3,
                    "source": src,
                    "pretty_path": pretty.get(best),
                    "dropped_pretty_path": dp,
                    "dropped_source": src,
                    "dropped_id": str(dc),
                    "dropped_category_id": id_to_cat.get(dc, {}).get("category_id", ""),
                    "updated_at": id_to_cat.get(best, {}).get("updated_at", pd.Timestamp.now()),
                    "intra_policy": "most_complete_then_newest"
                })

    intra_agg = pd.DataFrame(keep_rows)[cols] if keep_rows else pd.DataFrame(columns=cols)
    intra_pairwise = pd.DataFrame(pairwise_rows)[cols] if pairwise_rows else pd.DataFrame(columns=cols)
    return intra_agg, intra_pairwise


def global_category_id_map(cats_df: pd.DataFrame, resolution: pd.DataFrame) -> pd.DataFrame:
    cats = _ensure_cats_contract(cats_df).copy()

    edges = {}
    if isinstance(resolution, pd.DataFrame) and not resolution.empty and {"id", "dropped_id"}.issubset(resolution.columns):
        for row in resolution.itertuples(index=False):
            dropped = getattr(row, "dropped_id", None)
            kept = getattr(row, "id", None)
            if dropped and kept and dropped != kept:
                edges[str(dropped)] = str(kept)

    def _find_anchor_id(_id: str) -> str:
        seen = set()
        cur = str(_id)
        while cur in edges and cur not in seen:
            seen.add(cur)
            cur = edges[cur]
        return cur

    cats["anchor_id"] = cats["id"].astype(str).apply(_find_anchor_id)

    # Include ALL categories in the mapping, not just winners
    # This ensures dropped categories can still be looked up by their original category_id
    all_cats = cats.copy()

    all_cats["updated_at"] = pd.to_datetime(all_cats["updated_at"], errors="coerce")
    # Replace any remaining NaT values with current timestamp
    nat_mask = pd.isna(all_cats["updated_at"])
    if nat_mask.any():
        current_time = pd.Timestamp.now()
        all_cats.loc[nat_mask, "updated_at"] = current_time
    all_cats.sort_values(["id", "updated_at"], ascending=[True, False], inplace=True)
    all_cats = all_cats.drop_duplicates(subset=["id"], keep="first")

    # Prefer existing global_id from the anchor when present, otherwise mint a new one.
    gid_map = (
        all_cats[["id", "global_id"]]
        .copy()
        .assign(global_id=lambda d: d["global_id"].astype(str).str.strip())
        .set_index("id")["global_id"]
        .to_dict()
    )
    all_cats["global_id"] = all_cats["anchor_id"].map(gid_map)
    # Note: entries in gid_map that normalize to an empty string are intentionally treated
    # as missing here, so they will cause new global_ids to be minted from anchor_id.
    missing_gid = all_cats["global_id"].isna() | (all_cats["global_id"].astype(str).str.strip() == "")
    if missing_gid.any():
        all_cats.loc[missing_gid, "global_id"] = all_cats.loc[missing_gid, "anchor_id"].astype(str).apply(_uuid_from_hash)

    all_cats["source"] = all_cats["source"].astype(str).str.strip().str.lower()
    all_cats["category_id"] = (
        all_cats["category_id"]
        .astype(object).where(all_cats["category_id"].notna(), "")
        .astype(str).str.strip()
    )
    all_cats.loc[all_cats["category_id"].str.lower().isin(["nan", "none"]), "category_id"] = ""
    out = all_cats[["global_id", "category_id", "source", "updated_at"]].copy()
    # Note: global_id is no longer unique since dropped categories share the same global_id as their winner
    # Note: load_timestamp column excluded from output parquet file
    return out[["global_id", "category_id", "source", "updated_at"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-parquet", required=True)
    ap.add_argument("--categories-csv", required=True)
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--emit-intra", action="store_true", default=False)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.pairs_parquet)
    cats = pd.read_csv(args.categories_csv)

    res = build_resolution_from_pairs(pairs, cats)

    base = str(Path(args.pairs_parquet).with_suffix(""))
    out_parquet = args.out_parquet or f"{base}.resolution.parquet"
    out_csv = args.out_csv or f"{base}.resolution.csv"
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)

    res.to_parquet(out_parquet, index=False)
    res.to_csv(out_csv, index=False)

    if args.emit_intra:
        intra_agg, intra_pairwise = build_intra_resolution_from_pairs(pairs, cats)
        intra_agg.to_parquet(f"{base}.intra_resolution.parquet", index=False)
        intra_pairwise.to_parquet(f"{base}.intra_resolution_pairwise.parquet", index=False)

    gmap = global_category_id_map(cats, res)
    gmap_out = args.out_csv or f"{base}.global_id_map.csv"
    gmap.to_csv(gmap_out, index=False)

    print(f"Wrote: {out_parquet} and {out_csv} and {gmap_out}")


if __name__ == "__main__":
    main()
