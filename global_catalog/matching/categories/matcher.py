from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple
import time
import pandas as pd

from global_catalog.transformers.categories.category_normalizer import CategoryNormalizer

from global_catalog.matching.categories.path_tfidf import (
    exact_cross, exact_intra, tfidf_intra,
    unordered_exact_cross, unordered_exact_intra,
    tfidf_cross_mutual, tfidf_cross_with_perm12, unordered_exact_fallback
)

from global_catalog.matching.categories.path_tfidf import (
    exact_cross, exact_intra, tfidf_cross, tfidf_intra,
    unordered_exact_cross, unordered_exact_intra,
    l3_l3_cross, l2_l3_cross,
)
from global_catalog.matching.categories.summarize import (
    attach_pretty_paths, summarize_per_category, attach_summary_flags
)
from global_catalog.matching.categories.blocking import (
    tfidf_cross_blocked, tfidf_intra_blocked
)

@dataclass
class CategoryMatchConfig:
    tfidf_threshold: float = 0.78
    block_by: Literal["none","l1","l1l2"] = "none"
    include_intra: bool = True
    include_unordered_exact: bool = True
    synonyms_path: str | None = None
    sample_limit: int = 2000


class CategoriesMatcher:
    def __init__(self, cfg: CategoryMatchConfig):
        self.cfg = cfg
        self.proc = CategoryNormalizer(synonyms_path=cfg.synonyms_path)

    def normalize(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("NORMALIZE: schema gate")
        df = df_raw
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

        print("NORMALIZE: run text rules")
        t = time.perf_counter()
        df_norm = self.proc.process(df)
        print(
            f"NORMALIZE: df_norm.shape={df_norm.shape} sources={sorted(df_norm['source'].dropna().unique())} took={round(time.perf_counter() - t, 3)}s"
        )

        df_pretty_like = df_norm.copy()
        df_pretty_like["level_one"] = df_norm["l1_norm"]
        df_pretty_like["level_two"] = df_norm["l2_norm"]
        df_pretty_like["level_three"] = df_norm["l3_norm"]

        return df_norm, df_pretty_like

    def generate_pairs(self, df_norm: pd.DataFrame) -> pd.DataFrame:
        t_all = time.perf_counter()
        print("MATCH: generate_pairs start")

        ds = (
            df_norm.groupby(["source", "path_norm"])
            .agg(n=("id", "count"))
            .reset_index()
        )
        by_src = ds.groupby("source")["path_norm"].nunique().to_dict()
        print(f"MATCH: unique path_norm per source = {by_src}")

        parts = []

        t = time.perf_counter()
        ec = exact_cross(df_norm)  # emits left_id/right_id
        ec_rows = 0 if ec is None else getattr(ec, "shape", (0, 0))[0]
        print(f"MATCH: exact_cross rows={ec_rows} took={round(time.perf_counter() - t, 3)}s")
        if ec is not None and not ec.empty:
            parts.append(ec)

        if self.cfg.block_by == "none":
            t = time.perf_counter()
            tc = tfidf_cross_with_perm12(df_norm, threshold=self.cfg.tfidf_threshold)  # emits left_id/right_id
            tc_rows = 0 if tc is None else getattr(tc, "shape", (0, 0))[0]
            print(f"MATCH: tfidf_cross_with_perm12 rows={tc_rows} took={round(time.perf_counter() - t, 3)}s")
            if tc is not None and not tc.empty:
                parts.append(tc)
            base_for_unordered = tc if tc is not None else pd.DataFrame()
        else:
            blk = {"l1": ("l1_norm",), "l1l2": ("l1_norm", "l2_norm")}[self.cfg.block_by]
            print(f"MATCH: using blocking by={self.cfg.block_by} levels={blk}")
            t = time.perf_counter()
            tcb = tfidf_cross_blocked(df_norm, threshold=self.cfg.tfidf_threshold,
                                      block_levels=blk)  # emits left_id/right_id
            tcb_rows = 0 if tcb is None else getattr(tcb, "shape", (0, 0))[0]
            print(f"MATCH: tfidf_cross_blocked rows={tcb_rows} took={round(time.perf_counter() - t, 3)}s")
            if tcb is not None and not tcb.empty:
                parts.append(tcb)
            base_for_unordered = tcb if tcb is not None else pd.DataFrame()

        if self.cfg.include_unordered_exact:
            t = time.perf_counter()
            uef = unordered_exact_fallback(df_norm, base_pairs=base_for_unordered)
            uef_rows = 0 if uef is None else getattr(uef, "shape", (0, 0))[0]
            print(f"MATCH: unordered_exact_fallback rows={uef_rows} took={round(time.perf_counter() - t, 3)}s")
            if uef is not None and not uef.empty:
                parts.append(uef)

        t = time.perf_counter()
        l3c = l3_l3_cross(df_norm)
        l3c_rows = 0 if l3c is None else getattr(l3c, "shape", (0, 0))[0]
        print(f"MATCH: l3_l3_cross rows={l3c_rows} took={round(time.perf_counter() - t, 3)}s")
        if l3c is not None and not l3c.empty:
            parts.append(l3c)

        t = time.perf_counter()
        l2l3 = l2_l3_cross(df_norm)
        l2l3_rows = 0 if l2l3 is None else getattr(l2l3, "shape", (0, 0))[0]
        print(f"MATCH: l2_l3_cross rows={l2l3_rows} took={round(time.perf_counter() - t, 3)}s")
        if l2l3 is not None and not l2l3.empty:
            parts.append(l2l3)


        parts = [p for p in parts if p is not None and not p.empty]

        combined = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        before = combined.shape[0]
        if not combined.empty:

            subset = [c for c in ["left_source", "right_source", "left_id", "right_id", "match_scope", "match_type"] if
                      c in combined.columns]
            if subset:
                combined.drop_duplicates(subset=subset, inplace=True)
            else:
                combined.drop_duplicates(inplace=True)
        after = combined.shape[0]
        print(
            f"MATCH: concat rows_before={before} rows_after_dedup={after} took_total={round(time.perf_counter() - t_all, 3)}s")
        return combined

    def filter_pairs(self, pairs: pd.DataFrame, df_norm: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            print("FILTER: no pairs, skip")
            return pairs
        t = time.perf_counter()

        print(f"FILTER: rows={pairs.shape[0]} took={round(time.perf_counter()-t,3)}s")
        return pairs

    def summarize(self, pairs: pd.DataFrame, df_pretty_like: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if pairs.empty:
            print("SUMMARY: no pairs, skip")
            return pairs, pd.DataFrame()
        t = time.perf_counter()
        pairs = attach_pretty_paths(pairs, df_pretty_like)
        summary = summarize_per_category(pairs)
        pairs = attach_summary_flags(pairs, summary)
        print(f"SUMMARY: pairs={pairs.shape} summary={summary.shape} took={round(time.perf_counter()-t,3)}s")
        return pairs, summary

    def metrics(self, df_norm: pd.DataFrame, pairs: pd.DataFrame, t0: float) -> Dict[str, Any]:
        def describe_similarity(s: pd.Series) -> dict:
            if s.empty:
                return {}
            return {
                "count": int(s.shape[0]),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
            }
        return {
            "records_total": int(df_norm.shape[0]),
            "sources": sorted([s for s in df_norm["source"].dropna().unique()]),
            "matches_total": int(pairs.shape[0]),
            "similarity_stats": describe_similarity(pairs["similarity"]) if ("similarity" in pairs.columns and not pairs.empty) else {},
            "timing_seconds": round(time.perf_counter() - t0, 3),
            "threshold": self.cfg.tfidf_threshold,
            "block_by": self.cfg.block_by,
        }

    def run(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        t0 = time.perf_counter()
        print("RUN: normalize")
        df_norm, df_pretty_like = self.normalize(df_raw)
        print("RUN: generate_pairs")
        pairs = self.generate_pairs(df_norm)
        print("RUN: filter_pairs")
        pairs = self.filter_pairs(pairs, df_norm)
        print("RUN: summarize")
        pairs, summary = self.summarize(pairs, df_pretty_like)
        if pairs.empty:
            sample = pairs
        else:
            sample = pairs.sort_values(["match_scope","match_type","similarity"], ascending=[True, True, False]).head(self.cfg.sample_limit)
        meta = self.metrics(df_norm, pairs, t0)
        print(f"RUN: done pairs={pairs.shape} summary={summary.shape} took={meta['timing_seconds']}s")
        return {"norm": df_norm, "pairs": pairs, "summary": summary, "sample": sample, "metrics": meta}
