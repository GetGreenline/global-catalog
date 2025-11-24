import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def exact_cross(df_norm: pd.DataFrame) -> pd.DataFrame:
    print("path tfidf df_norm columns:", list(df_norm.columns))

    print('start exact cross...')
    t0 = time.perf_counter() if 'time' in globals() else None

    df = df_norm.loc[:, ["source", "path_norm", "id"]].copy()
    df = df.dropna(subset=["source", "path_norm", "id"])
    df["id"] = df["id"].astype(str)

    sources = list(df["source"].dropna().unique())
    rows = []

    grouped = (
        df.drop_duplicates(["source", "path_norm", "id"])
          .groupby(["source", "path_norm"])["id"]
          .apply(list)
          .reset_index(name="id_list")
    )

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]

            L = grouped[grouped["source"] == sL][["path_norm", "id_list"]].rename(columns={"id_list": "left_id_list"})
            R = grouped[grouped["source"] == sR][["path_norm", "id_list"]].rename(columns={"id_list": "right_id_list"})

            m = L.merge(R, on="path_norm", how="inner")
            if m.empty:
                continue

            m = m.explode("left_id_list").explode("right_id_list")
            m = m.rename(columns={"left_id_list": "left_id", "right_id_list": "right_id"})
            m["left_source"] = sL
            m["right_source"] = sR
            m["similarity"] = 1.0
            m["match_scope"] = "cross"
            m["match_type"] = "exact"

            rows.append(m[[
                "left_source", "right_source",
                "left_id", "right_id",
                "similarity", "match_scope", "match_type"
            ]])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
        "left_source", "right_source", "left_id", "right_id",
        "similarity", "match_scope", "match_type"
    ])

    if t0 is not None:
        print(f"exact_cross: pairs={out.shape[0]} from sources={sources} took={round(time.perf_counter()-t0,3)}s")

    return out


def exact_intra(df_norm: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df_norm.groupby(["source", "path_norm"])["id"]
               .apply(list)
               .reset_index(name="id_list")
    )
    rows = []
    for _, rr in grp.iterrows():
        id_list = rr["id_list"]
        if len(id_list) < 2:
            continue
        for a, b in combinations(id_list, 2):
            rows.append({
                "left_source": rr["source"], "right_source": rr["source"],
                "left_id": a, "right_id": b,
                "similarity": 1.0, "match_scope": "intra", "match_type": "exact",
            })
    return pd.DataFrame(rows)


def _expand_pairs_from_path(
    df_norm: pd.DataFrame,
    s_left: str, s_right: str,
    left_path: str, right_path: str,
    simv: float, scope: str
):
    ids_map = (df_norm.groupby(["source", "path_norm"])["id"].apply(list).to_dict())
    rows = []
    l_ids = ids_map.get((s_left, left_path), [])
    r_ids = ids_map.get((s_right, right_path), [])
    for lid in l_ids:
        for rid in r_ids:
            if scope == "intra" and lid >= rid:
                continue
            rows.append({
                "left_source": s_left, "right_source": s_right,
                "left_id": lid, "right_id": rid,
                "similarity": float(simv), "match_scope": scope, "match_type": "tfidf",
            })
    return rows


def tfidf_cross(df_norm: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ds = (
        df_norm.groupby(["source", "path_norm"])
               .agg(n=("id", "count"))
               .reset_index()
    )
    sources = list(ds["source"].dropna().unique())
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    rows = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]
            L = ds[ds["source"] == sL].reset_index(drop=True)
            R = ds[ds["source"] == sR].reset_index(drop=True)
            if L.empty or R.empty:
                continue
            corpus = pd.concat([L["path_norm"], R["path_norm"]], ignore_index=True)
            X = vec.fit_transform(corpus.values)
            S = cosine_similarity(X[:len(L)], X[len(L):])
            best_idx = S.argmax(axis=1)
            best_sim = S[np.arange(S.shape[0]), best_idx]
            keep = best_sim >= threshold
            Lk = L.loc[keep].reset_index(drop=True)
            Rk = R.iloc[best_idx[keep]].reset_index(drop=True)
            for k in range(len(Lk)):
                rows += _expand_pairs_from_path(
                    df_norm, sL, sR,
                    Lk.loc[k, "path_norm"], Rk.loc[k, "path_norm"],
                    best_sim[np.where(keep)[0][k]], "cross"
                )
    return pd.DataFrame(rows)


def tfidf_intra(df_norm: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ds = (
        df_norm.groupby(["source", "path_norm"])
               .agg(n=("id", "count"))
               .reset_index()
    )
    # This is more useful in case we have a source that is not garanteed to
    # be a clean one (does not apply to CategoriesV1 case)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    rows = []
    for src in ds["source"].dropna().unique():
        Xdf = ds[ds["source"] == src].reset_index(drop=True)
        if len(Xdf) < 2:
            continue
        X = vec.fit_transform(Xdf["path_norm"].values)
        S = cosine_similarity(X, X)
        for i in range(len(Xdf)):
            for j in range(i + 1, len(Xdf)):
                if S[i, j] >= threshold:
                    rows += _expand_pairs_from_path(
                        df_norm, src, src,
                        Xdf.loc[i, "path_norm"], Xdf.loc[j, "path_norm"],
                        float(S[i, j]), "intra"
                    )
    return pd.DataFrame(rows)


def unordered_exact_cross(df_norm: pd.DataFrame) -> pd.DataFrame:
    def key_row(r):
        vals = [v for v in [r["l1_norm"], r["l2_norm"], r["l3_norm"]] if isinstance(v, str) and v.strip()]
        return "||".join(sorted(set(vals)))

    tmp = df_norm.copy()
    tmp["vals_key"] = tmp.apply(key_row, axis=1)
    grp = tmp.groupby(["source", "vals_key"])["id"].apply(list).reset_index(name="id_list")
    sources = sorted(tmp["source"].dropna().unique())
    rows = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]
            L = grp[grp["source"] == sL][["vals_key", "id_list"]].rename(columns={"id_list": "left_id_list"})
            R = grp[grp["source"] == sR][["vals_key", "id_list"]].rename(columns={"id_list": "right_id_list"})
            m = L.merge(R, on="vals_key", how="inner")
            for _, rr in m.iterrows():
                for a in rr["left_id_list"]:
                    for b in rr["right_id_list"]:
                        rows.append({
                            "left_source": sL, "right_source": sR,
                            "left_id": a, "right_id": b,
                            "similarity": 1.0, "match_scope": "cross", "match_type": "unordered_exact",
                        })
    return pd.DataFrame(rows)


def unordered_exact_intra(df_norm: pd.DataFrame) -> pd.DataFrame:
    def key_row(r):
        vals = [v for v in [r["l1_norm"], r["l2_norm"], r["l3_norm"]] if isinstance(v, str) and v.strip()]
        return "||".join(sorted(set(vals)))

    tmp = df_norm.copy()
    tmp["vals_key"] = tmp.apply(key_row, axis=1)
    grp = tmp.groupby(["source", "vals_key"])["id"].apply(list).reset_index(name="id_list")
    rows = []
    for _, rr in grp.iterrows():
        id_list = rr["id_list"]
        if len(id_list) < 2:
            continue
        for a, b in combinations(id_list, 2):
            rows.append({
                "left_source": rr["source"], "right_source": rr["source"],
                "left_id": a, "right_id": b,
                "similarity": 1.0, "match_scope": "intra", "match_type": "unordered_exact",
            })
    return pd.DataFrame(rows)


def tfidf_cross_mutual(df_norm: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ds = (
        df_norm.groupby(["source", "path_norm"])
               .agg(n=("id", "count"))
               .reset_index()
    )
    sources = sorted(list(ds["source"].dropna().unique()))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    rows = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]
            L = ds[ds["source"] == sL].reset_index(drop=True)
            R = ds[ds["source"] == sR].reset_index(drop=True)
            if L.empty or R.empty:
                continue
            corpus = pd.concat([L["path_norm"], R["path_norm"]], ignore_index=True)
            X = vec.fit_transform(corpus.values)
            S = cosine_similarity(X[:len(L)], X[len(L):])
            best_L = S.argmax(axis=1)
            best_R = S.argmax(axis=0)
            for li in range(len(L)):
                ri = best_L[li]
                if best_R[ri] == li and S[li, ri] >= threshold:
                    rows += _expand_pairs_from_path(
                        df_norm, sL, sR,
                        L.loc[li, "path_norm"], R.loc[ri, "path_norm"],
                        float(S[li, ri]), "cross"
                    )
    return pd.DataFrame(rows)


def _expand_pairs_from_path_typed(
    df_norm: pd.DataFrame,
    s_left: str, s_right: str,
    left_path: str, right_path: str,
    simv: float, scope: str, match_type: str
):
    ids_map = (df_norm.groupby(["source", "path_norm"])["id"].apply(list).to_dict())
    rows = []
    l_ids = ids_map.get((s_left, left_path), [])
    r_ids = ids_map.get((s_right, right_path), [])
    for lid in l_ids:
        for rid in r_ids:
            if scope == "intra" and lid >= rid:
                continue
            rows.append({
                "left_source": s_left,
                "right_source": s_right,
                "left_id": lid,
                "right_id": rid,
                "left_path_pretty": left_path,
                "right_path_pretty": right_path,
                "similarity": float(simv),
                "match_scope": scope,
                "match_type": match_type,
            })
    return rows


def _parts3(p):
    parts = [x.strip() for x in str(p).split("/") if str(x).strip() != ""]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def cross_perm12_fallback(df_norm: pd.DataFrame, base_pairs: pd.DataFrame) -> pd.DataFrame:
    ds = (
        df_norm.groupby(["source", "path_norm"])
               .agg(id_list=("id", "unique"))
               .reset_index()
    )
    sources = sorted(list(ds["source"].dropna().unique()))
    rows = []

    id_paths = (
        df_norm.dropna(subset=["source", "id", "path_norm"])
               .astype({"id": str})
               .groupby(["source", "id"])["path_norm"]
               .unique().to_dict()
    )

    def used_paths_for(base: pd.DataFrame, src: str, side: str):
        if base.empty:
            return set()
        key_id = f"{side}_id"
        key_src = f"{side}_source"
        ids = base.loc[base[key_src] == src, key_id].astype(str).tolist()
        used = set()
        for cid in ids:
            used |= set(id_paths.get((src, cid), []))
        return used

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]
            Lpaths = set(ds.loc[ds["source"] == sL, "path_norm"].tolist())
            Rpaths = set(ds.loc[ds["source"] == sR, "path_norm"].tolist())

            baseL = used_paths_for(base_pairs, sL, "left") | used_paths_for(base_pairs, sL, "right")
            baseR = used_paths_for(base_pairs, sR, "left") | used_paths_for(base_pairs, sR, "right")

            candidates = []
            for pR in Rpaths:
                tri = _parts3(pR)
                if tri is None:
                    continue
                b, a, c = tri
                pL = "/".join([a, b, c])
                amb = "/".join([b, c, a])
                if pL in Lpaths and amb not in Lpaths:
                    if (pL not in baseL) and (pR not in baseR):
                        candidates.append((pL, pR))

            L_to_R = {}
            R_to_L = {}
            for l, r in candidates:
                L_to_R.setdefault(l, set()).add(r)
                R_to_L.setdefault(r, set()).add(l)

            unique_pairs = [(l, r) for (l, r) in candidates if len(L_to_R[l]) == 1 and len(R_to_L[r]) == 1]

            for l, r in unique_pairs:
                rows += _expand_pairs_from_path_typed(
                    df_norm, sL, sR, l, r, 1.0, "cross", "perm12_exact"
                )

    return pd.DataFrame(rows, columns=[
        "left_source", "right_source", "left_id", "right_id",
        "left_path_pretty", "right_path_pretty", "similarity", "match_scope", "match_type"
    ])


def tfidf_cross_with_perm12(df_norm: pd.DataFrame, threshold: float) -> pd.DataFrame:
    base = tfidf_cross_mutual(df_norm, threshold)
    extra = cross_perm12_fallback(df_norm, base)
    if extra.empty:
        return base

    base2 = base.copy()
    base2["pair_key"] = base2.apply(
        lambda r: "|".join(sorted([
            f"{r['left_source']}::{str(r['left_id'])}",
            f"{r['right_source']}::{str(r['right_id'])}",
        ])), axis=1
    )
    extra2 = extra.copy()
    extra2["pair_key"] = extra2.apply(
        lambda r: "|".join(sorted([
            f"{r['left_source']}::{str(r['left_id'])}",
            f"{r['right_source']}::{str(r['right_id'])}",
        ])), axis=1
    )

    all_pairs = pd.concat([base2, extra2], ignore_index=True)
    all_pairs["match_rank"] = all_pairs["match_type"].map({"tfidf": 0, "perm12_exact": 1}).fillna(2).astype(int)
    all_pairs = (
        all_pairs.sort_values(["pair_key", "match_rank", "similarity"], ascending=[True, True, False])
                .drop_duplicates("pair_key")
                .drop(columns=["pair_key", "match_rank"])
    )
    return all_pairs


def _vals_key_from_path(p: str) -> str:
    parts = [x.strip() for x in str(p).split("/") if x and str(x).strip()]
    return "||".join(sorted(set(parts)))

def l2_l3_cross(df_norm: pd.DataFrame) -> pd.DataFrame:
    tmp = df_norm.dropna(subset=["source", "id"]).copy()
    tmp["id"] = tmp["id"].astype(str)

    tmp["l2_norm"] = tmp["l2_norm"].astype(str).str.strip()
    tmp["l3_norm"] = tmp["l3_norm"].astype(str).str.strip()

    # valid, not empty, not 'uncategorized'
    mask_l2 = (tmp["l2_norm"] != "") & (tmp["l2_norm"].str.lower() != "uncategorized")
    mask_l3 = (tmp["l3_norm"] != "") & (tmp["l3_norm"].str.lower() != "uncategorized")

    tmp_l2 = tmp[mask_l2]
    tmp_l3 = tmp[mask_l3]

    if tmp_l2.empty or tmp_l3.empty:
        return pd.DataFrame(columns=[
            "left_source", "right_source", "left_id", "right_id",
            "similarity", "match_scope", "match_type"
        ])

    grp_l2 = (
        tmp_l2.groupby(["source", "l2_norm"])["id"]
              .apply(list)
              .reset_index(name="id_list")
    )
    grp_l3 = (
        tmp_l3.groupby(["source", "l3_norm"])["id"]
              .apply(list)
              .reset_index(name="id_list")
    )

    sources = sorted(tmp["source"].dropna().unique())
    rows = []

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]

            # left l2 == right l3
            L2 = grp_l2[grp_l2["source"] == sL][["l2_norm", "id_list"]].rename(
                columns={"id_list": "left_id_list", "l2_norm": "key"}
            )
            R3 = grp_l3[grp_l3["source"] == sR][["l3_norm", "id_list"]].rename(
                columns={"id_list": "right_id_list", "l3_norm": "key"}
            )
            m1 = L2.merge(R3, on="key", how="inner")

            for _, rr in m1.iterrows():
                for a in rr["left_id_list"]:
                    for b in rr["right_id_list"]:
                        rows.append({
                            "left_source": sL,
                            "right_source": sR,
                            "left_id": a,
                            "right_id": b,
                            "similarity": 1.0,
                            "match_scope": "cross",
                            "match_type": "l2_l3_exact",
                        })

    cols = ["left_source", "right_source", "left_id", "right_id",
            "similarity", "match_scope", "match_type"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def l3_l3_cross(df_norm: pd.DataFrame) -> pd.DataFrame:
    tmp = df_norm.dropna(subset=["source", "id", "l3_norm"]).copy()
    tmp["id"] = tmp["id"].astype(str)
    tmp["l3_norm"] = tmp["l3_norm"].astype(str).str.strip()

    # drop empty and 'uncategorized'
    mask_valid = tmp["l3_norm"] != ""
    mask_not_uncat = tmp["l3_norm"].str.lower() != "uncategorized"
    tmp = tmp[mask_valid & mask_not_uncat]

    if tmp.empty:
        return pd.DataFrame(columns=[
            "left_source", "right_source", "left_id", "right_id",
            "similarity", "match_scope", "match_type"
        ])

    grp = (
        tmp.groupby(["source", "l3_norm"])["id"]
           .apply(list)
           .reset_index(name="id_list")
    )

    sources = sorted(tmp["source"].dropna().unique())
    rows = []

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]

            L = grp[grp["source"] == sL][["l3_norm", "id_list"]].rename(columns={"id_list": "left_id_list"})
            R = grp[grp["source"] == sR][["l3_norm", "id_list"]].rename(columns={"id_list": "right_id_list"})

            m = L.merge(R, on="l3_norm", how="inner")
            if m.empty:
                continue

            for _, rr in m.iterrows():
                for a in rr["left_id_list"]:
                    for b in rr["right_id_list"]:
                        rows.append({
                            "left_source": sL,
                            "right_source": sR,
                            "left_id": a,
                            "right_id": b,
                            "similarity": 1.0,
                            "match_scope": "cross",
                            "match_type": "l3_exact",
                        })

    cols = ["left_source", "right_source", "left_id", "right_id",
            "similarity", "match_scope", "match_type"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)



def unordered_exact_fallback(df_norm: pd.DataFrame, base_pairs: pd.DataFrame) -> pd.DataFrame:
    tmp = df_norm.copy()
    tmp = tmp.dropna(subset=["source", "id", "path_norm"])
    tmp["id"] = tmp["id"].astype(str)
    tmp["vals_key"] = tmp["path_norm"].map(_vals_key_from_path)

    id2paths = (tmp.groupby(["source", "id"])["path_norm"].unique().to_dict())

    def pick_path(src: str, cid: str, key: str) -> str:
        cand = [p for p in id2paths.get((src, cid), []) if _vals_key_from_path(p) == key]
        if not cand:
            cand = list(id2paths.get((src, cid), []))
        return sorted(map(str, cand))[0] if cand else None

    grp = (
        tmp.groupby(["source", "vals_key"])["id"]
           .apply(lambda s: sorted(map(str, pd.unique(s))))
           .reset_index(name="id_list")
    )

    sources = sorted(tmp["source"].dropna().unique())

    def _used_sid(df):
        if df is None or df.empty:
            return set()
        L = set(zip(df["left_source"].astype(str), df["left_id"].astype(str)))
        R = set(zip(df["right_source"].astype(str), df["right_id"].astype(str)))
        return L | R

    used_ids = _used_sid(base_pairs)

    def _pair_keys(df):
        if df is None or df.empty:
            return set()
        return set(df.apply(lambda r: "|".join(sorted([
            f"{r['left_source']}::{str(r['left_id'])}",
            f"{r['right_source']}::{str(r['right_id'])}",
        ])), axis=1).tolist())

    existing_pairs = _pair_keys(base_pairs)

    rows = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sL, sR = sources[i], sources[j]
            L = grp[grp["source"] == sL][["vals_key", "id_list"]].rename(columns={"id_list": "left_id_list"})
            R = grp[grp["source"] == sR][["vals_key", "id_list"]].rename(columns={"id_list": "right_id_list"})
            m = L.merge(R, on="vals_key", how="inner")
            if m.empty:
                continue

            taken_L, taken_R = set(), set()
            for _, rr in m.iterrows():
                key = rr["vals_key"]
                left_ids = [x for x in rr["left_id_list"] if (sL, x) not in used_ids]
                right_ids = [x for x in rr["right_id_list"] if (sR, x) not in used_ids]
                for a in sorted(left_ids):
                    if a in taken_L:
                        continue
                    for b in sorted(right_ids):
                        if b in taken_R:
                            continue
                        pk = "|".join(sorted([f"{sL}::{a}", f"{sR}::{b}"]))
                        if pk in existing_pairs:
                            continue
                        lp = pick_path(sL, a, key)
                        rp = pick_path(sR, b, key)
                        rows.append({
                            "left_source": sL,
                            "right_source": sR,
                            "left_id": a,
                            "right_id": b,
                            "left_path_pretty": lp,
                            "right_path_pretty": rp,
                            "similarity": 1.0,
                            "match_scope": "cross",
                            "match_type": "unordered_exact",
                        })
                        taken_L.add(a)
                        taken_R.add(b)
                        break

    cols = [
        "left_source", "right_source", "left_id", "right_id",
        "left_path_pretty", "right_path_pretty", "similarity", "match_scope", "match_type"
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def tfidf_cross_with_perm12_and_unordered(df_norm: pd.DataFrame, threshold: float) -> pd.DataFrame:
    base = tfidf_cross_with_perm12(df_norm, threshold)
    extra = unordered_exact_fallback(df_norm, base)
    if extra.empty:
        return base

    base["pair_key"] = base.apply(lambda r: "|".join(sorted([
        f"{r['left_source']}::{str(r['left_id'])}",
        f"{r['right_source']}::{str(r['right_id'])}",
    ])), axis=1)
    extra["pair_key"] = extra.apply(lambda r: "|".join(sorted([
        f"{r['left_source']}::{str(r['left_id'])}",
        f"{r['right_source']}::{str(r['right_id'])}",
    ])), axis=1)

    all_pairs = pd.concat([base, extra], ignore_index=True)
    rank = {"tfidf": 0, "perm12_exact": 1, "unordered_exact": 2}
    all_pairs["match_rank"] = all_pairs["match_type"].map(rank).fillna(99).astype(int)
    all_pairs = (
        all_pairs.sort_values(["pair_key", "match_rank", "similarity"], ascending=[True, True, False])
                .drop_duplicates("pair_key")
                .drop(columns=["pair_key", "match_rank"])
    )
    return all_pairs
