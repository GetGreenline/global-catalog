from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from global_catalog.pipelines.common import new_run_id, prepare_run_dir


class ProductMatchResolver:
    """Persist product matching in artifacts for initial result analysis"""

    def __init__(self, out_root: str = "artifacts/products", run_label: str | None = None):
        self.out_root = Path(out_root)
        self.run_label = run_label

    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        tag_parts = ["products"]
        if self.run_label:
            tag_parts.append(self._slug(self.run_label))
        strategy_name = match_results.get("blocking_strategy") if match_results else None
        matcher_name = match_results.get("matcher_name") if match_results else None
        if strategy_name:
            tag_parts.append(self._slug(strategy_name))
        if matcher_name:
            tag_parts.append(self._slug(matcher_name))
        tag = "_".join(tag_parts)
        run_id = new_run_id(tag)
        run_dir = prepare_run_dir(str(self.out_root), run_id)

        pairs_df = match_results.get("pairs")
        if pairs_df is None:
            pairs_df = pd.DataFrame()
        candidate_df = match_results.get("candidate_pairs")
        if candidate_df is None:
            candidate_df = pd.DataFrame()
        metrics = match_results.get("metrics") or {}

        pairs_df = self._attach_context_columns(pairs_df, normalized_data)
        candidate_df = self._attach_context_columns(candidate_df, normalized_data)

        resolved_df = self._resolve_one_to_one(pairs_df, raw_data)

        pairs_path = run_dir / "pairs.parquet"
        candidates_path = run_dir / "candidate_pairs.parquet"
        metrics_path = run_dir / "metrics.json"
        resolved_parquet_path = run_dir / "resolved_pairs.parquet"
        resolved_csv_path = run_dir / "resolved_pairs.csv"

        pairs_df.to_parquet(pairs_path, index=False)
        candidate_df.to_parquet(candidates_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        resolved_df.to_parquet(resolved_parquet_path, index=False)
        resolved_df.to_csv(resolved_csv_path, index=False)

        print(f"[ProductMatchResolver] Wrote pairs to {pairs_path}")
        print(f"[ProductMatchResolver] Wrote candidate pairs to {candidates_path}")
        print(f"[ProductMatchResolver] Wrote metrics to {metrics_path}")
        print(f"[ProductMatchResolver] Wrote resolved 1:1 pairs to {resolved_parquet_path} and {resolved_csv_path}")

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "pairs_path": str(pairs_path),
            "candidate_pairs_path": str(candidates_path),
            "metrics_path": str(metrics_path),
            "resolved_pairs_path": str(resolved_parquet_path),
            "resolved_pairs_csv": str(resolved_csv_path),
        }

    def _slug(self, value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "tag"

    def _attach_context_columns(self, df: pd.DataFrame, normalized_data: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        for prefix in ("left", "right"):
            idx_col = f"{prefix}_index"
            if idx_col not in df.columns:
                continue
            idx_values = df[idx_col].astype(int)
            subset = normalized_data.loc[idx_values]
            default_none = pd.Series([None] * len(subset), index=subset.index)
            default_empty = pd.Series([""] * len(subset), index=subset.index)
            df[f"{prefix}_measure_mg"] = subset.get("measure_mg", default_none).to_list()
            df[f"{prefix}_description_norm"] = subset.get("description_norm", default_empty).to_list()
            df[f"{prefix}_product_id"] = subset.get("product_id", default_none).to_list()
        return df

    # --- Resolution helpers -------------------------------------------------

    def _resolve_one_to_one(self, pairs_df: pd.DataFrame, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Select 1:1 cross-source pairs by highest name_score, then enrich winner with missing loser fields."""
        if pairs_df is None or pairs_df.empty:
            return pd.DataFrame()

        required_cols = {"left_source", "right_source", "left_product_id", "right_product_id", "name_score", "similarity"}
        if not required_cols.issubset(pairs_df.columns):
            return pd.DataFrame()

        pairs = pairs_df[pairs_df["left_source"] != pairs_df["right_source"]].copy()
        if pairs.empty:
            return pd.DataFrame()

        pairs["_name_score_sort"] = pd.to_numeric(pairs["name_score"], errors="coerce").fillna(-1.0)
        pairs["_similarity_sort"] = pd.to_numeric(pairs["similarity"], errors="coerce").fillna(-1.0)
        pairs.sort_values(["_name_score_sort", "_similarity_sort"], ascending=False, inplace=True)

        used_left = set()
        used_right = set()
        selected_rows = []
        for _, row in pairs.iterrows():
            lpid_raw = row.get("left_product_id")
            rpid_raw = row.get("right_product_id")
            if self._is_missing(lpid_raw) or self._is_missing(rpid_raw):
                continue
            lpid = str(lpid_raw)
            rpid = str(rpid_raw)
            if lpid in used_left or rpid in used_right:
                continue
            selected_rows.append(row)
            used_left.add(lpid)
            used_right.add(rpid)

        if not selected_rows:
            return pd.DataFrame()

        raw_lookup = self._build_raw_lookup(raw_data)
        records = []
        all_attr_cols = set()

        for row in selected_rows:
            lpid = row.get("left_product_id")
            rpid = row.get("right_product_id")
            left_src = row.get("left_source")
            right_src = row.get("right_source")
            left_raw = self._get_raw(raw_lookup, left_src, lpid)
            right_raw = self._get_raw(raw_lookup, right_src, rpid)

            winner_side = self._pick_winner(left_raw, right_raw, left_src, right_src)
            winner_raw = left_raw if winner_side == "left" else right_raw
            loser_raw = right_raw if winner_side == "left" else left_raw
            merged_raw = self._merge_records(winner_raw, loser_raw)

            all_attr_cols.update(winner_raw.keys())
            all_attr_cols.update(loser_raw.keys())
            all_attr_cols.update(merged_raw.keys())

            records.append(
                {
                    "left_index": row.get("left_index"),
                    "right_index": row.get("right_index"),
                    "left_source": left_src,
                    "right_source": right_src,
                    "left_product_id": lpid,
                    "right_product_id": rpid,
                    "left_product_name": row.get("left_product_name"),
                    "right_product_name": row.get("right_product_name"),
                    "left_brand_name": row.get("left_brand_name"),
                    "right_brand_name": row.get("right_brand_name"),
                    "match_name_score": row.get("name_score"),
                    "match_similarity": row.get("similarity"),
                    "match_type": row.get("match_type"),
                    "winner_side": winner_side,
                    "winner_source": winner_raw.get("source", left_src if winner_side == "left" else right_src),
                    "winner_product_id": winner_raw.get("product_id", lpid if winner_side == "left" else rpid),
                    "loser_source": loser_raw.get("source", right_src if winner_side == "left" else left_src),
                    "loser_product_id": loser_raw.get("product_id", rpid if winner_side == "left" else lpid),
                    "winner_info_count": self._info_score(winner_raw)[0],
                    "loser_info_count": self._info_score(loser_raw)[0],
                    "_winner_raw": winner_raw,
                    "_loser_raw": loser_raw,
                    "_merged_raw": merged_raw,
                }
            )

        # Flatten winner/loser/merged attributes into prefixed columns.
        flat_rows = []
        sorted_cols = sorted(all_attr_cols)
        for rec in records:
            base = {k: v for k, v in rec.items() if not k.startswith("_")}
            w = rec["_winner_raw"]
            l = rec["_loser_raw"]
            m = rec["_merged_raw"]
            for col in sorted_cols:
                base[f"winner_{col}"] = w.get(col)
                base[f"loser_{col}"] = l.get(col)
                base[f"resolved_{col}"] = m.get(col)
            flat_rows.append(base)

        return pd.DataFrame(flat_rows)

    def _build_raw_lookup(self, raw_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        lookup: Dict[str, pd.DataFrame] = {}
        for src, df in (raw_data or {}).items():
            if not isinstance(df, pd.DataFrame) or "product_id" not in df.columns:
                continue
            tmp = df.copy()
            tmp["__product_id_key__"] = tmp["product_id"].astype(str)
            lookup[src] = tmp.set_index("__product_id_key__", drop=False)
        return lookup

    def _get_raw(self, lookup: Dict[str, pd.DataFrame], source: Any, product_id: Any) -> Dict[str, Any]:
        src = str(source)
        key = str(product_id)
        table = lookup.get(src)
        if table is None:
            return {}
        try:
            row = table.loc[key]
        except Exception:
            return {}
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        data = row.to_dict()
        data.pop("__product_id_key__", None)
        return data

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        s = str(value).strip()
        return s == "" or s.lower() in {"nan", "none", "null"}

    def _info_score(self, record: Dict[str, Any]) -> tuple[int, int]:
        if not record:
            return (0, 0)
        count = 0
        total_len = 0
        for val in record.values():
            if not self._is_missing(val):
                count += 1
                total_len += len(str(val))
        return (count, total_len)

    def _pick_winner(self, left_record: Dict[str, Any], right_record: Dict[str, Any], left_src: Any, right_src: Any) -> str:
        l_count, l_len = self._info_score(left_record)
        r_count, r_len = self._info_score(right_record)
        if l_count > r_count:
            return "left"
        if r_count > l_count:
            return "right"
        if l_len > r_len:
            return "left"
        if r_len > l_len:
            return "right"
        # deterministic tie-breaker
        return "left" if str(left_src) <= str(right_src) else "right"

    def _merge_records(self, winner: Dict[str, Any], loser: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(winner) if winner else {}
        loser = loser or {}
        for k, v in loser.items():
            if k not in merged or self._is_missing(merged.get(k)):
                if not self._is_missing(v):
                    merged[k] = v
        return merged
