import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _ensure_norm_cols(df_norm: pd.DataFrame):
    req = {"l1_norm","l2_norm","l3_norm","path_norm","source","category_id"}
    missing = [c for c in req if c not in df_norm.columns]
    if missing:
        raise KeyError(f"Missing columns in df_norm: {missing}")

def _prep_ds(df_norm: pd.DataFrame) -> pd.DataFrame:
    _ensure_norm_cols(df_norm)
    ds = (
        df_norm.groupby(["source","path_norm"])
        .agg(
            n=("category_id","count"),
            l1_norm=("l1_norm","first"),
            l2_norm=("l2_norm","first"),
            l3_norm=("l3_norm","first"),
        )
        .reset_index()
    )
    return ds

def _make_block_key_col(ds: pd.DataFrame, block_levels: tuple[str, ...]) -> pd.Series:
    if not block_levels:
        return pd.Series(["__ALL__"] * len(ds), index=ds.index)
    for c in block_levels:
        if c not in ds.columns:
            raise KeyError(f"Block column '{c}' not found. Available columns: {list(ds.columns)}")
    return ds.apply(lambda r: tuple(r[c] for c in block_levels), axis=1)

def _expand_pairs_from_path(df_norm: pd.DataFrame, s_left: str, s_right: str, left_path: str, right_path: str, simv: float, scope: str):
    ids_map = df_norm.groupby(["source","path_norm"])["category_id"].apply(list).to_dict()
    rows = []
    l_ids = ids_map.get((s_left, left_path), [])
    r_ids = ids_map.get((s_right, right_path), [])
    for lid in l_ids:
        for rid in r_ids:
            if scope == "intra" and lid >= rid:
                continue
            rows.append({
                "left_source": s_left, "right_source": s_right,
                "left_category_id": lid, "right_category_id": rid,
                "similarity": float(simv), "match_scope": scope, "match_type": "tfidf",
            })
    return rows

def tfidf_cross_blocked(df_norm: pd.DataFrame, threshold: float, block_levels: tuple[str, ...]) -> pd.DataFrame:
    ds = _prep_ds(df_norm)
    ds["block_key"] = _make_block_key_col(ds, block_levels)
    sources = list(ds["source"].dropna().unique())
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    out_rows = []
    if len(sources) < 2:
        return pd.DataFrame(out_rows)
    for i in range(len(sources)):
        for j in range(i+1, len(sources)):
            sL, sR = sources[i], sources[j]
            Lall = ds[ds["source"] == sL]
            Rall = ds[ds["source"] == sR]
            if Lall.empty or Rall.empty:
                continue
            shared_blocks = sorted(set(Lall["block_key"]).intersection(set(Rall["block_key"])))
            for bk in shared_blocks:
                L = Lall[Lall["block_key"] == bk].reset_index(drop=True)
                R = Rall[Rall["block_key"] == bk].reset_index(drop=True)
                if L.empty or R.empty:
                    continue
                corpus = pd.concat([L["path_norm"], R["path_norm"]], ignore_index=True)
                X = vec.fit_transform(corpus.values)
                S = cosine_similarity(X[:len(L)], X[len(L):])
                best_idx = S.argmax(axis=1)
                best_sim = S[np.arange(S.shape[0]), best_idx]
                keep = best_sim >= threshold
                if not keep.any():
                    continue
                Lk = L.loc[keep].reset_index(drop=True)
                Rk = R.iloc[best_idx[keep]].reset_index(drop=True)
                for k in range(len(Lk)):
                    out_rows += _expand_pairs_from_path(
                        df_norm, sL, sR,
                        Lk.loc[k,"path_norm"], Rk.loc[k,"path_norm"],
                        float(best_sim[np.where(keep)[0][k]]), "cross"
                    )
    return pd.DataFrame(out_rows)

def tfidf_intra_blocked(df_norm: pd.DataFrame, threshold: float, block_levels: tuple[str, ...]) -> pd.DataFrame:
    ds = _prep_ds(df_norm)
    ds["block_key"] = _make_block_key_col(ds, block_levels)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    out_rows = []
    for src in ds["source"].dropna().unique():
        Xsrc = ds[ds["source"] == src]
        if Xsrc.empty:
            continue
        for bk, grp in Xsrc.groupby("block_key"):
            G = grp.reset_index(drop=True)
            if len(G) < 2:
                continue
            X = vec.fit_transform(G["path_norm"].values)
            S = cosine_similarity(X, X)
            for i in range(len(G)):
                for j in range(i+1, len(G)):
                    if S[i, j] >= threshold:
                        out_rows += _expand_pairs_from_path(
                            df_norm, src, src,
                            G.loc[i,"path_norm"], G.loc[j,"path_norm"],
                            float(S[i, j]), "intra"
                        )
    return pd.DataFrame(out_rows)


def sorted_neighborhood_candidates(df_norm: pd.DataFrame, window: int = 5):
    ds = _prep_ds(df_norm).copy()
    ds["sn_key"] = ds["path_norm"].str.replace(r"[\s/\-&]+","",regex=True)
    ds = ds.sort_values(["source","sn_key","path_norm"]).reset_index(drop=True)
    out = []
    for src, g in ds.groupby("source"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            jmax = min(len(g), i + window)
            for j in range(i+1, jmax):
                out.append((src, g.loc[i,"path_norm"], g.loc[j,"path_norm"]))
    return out
