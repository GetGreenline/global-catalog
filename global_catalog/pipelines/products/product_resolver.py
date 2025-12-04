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

        pairs_path = run_dir / "pairs.parquet"
        candidates_path = run_dir / "candidate_pairs.parquet"
        metrics_path = run_dir / "metrics.json"

        pairs_df.to_parquet(pairs_path, index=False)
        candidate_df.to_parquet(candidates_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"[ProductMatchResolver] Wrote pairs to {pairs_path}")
        print(f"[ProductMatchResolver] Wrote candidate pairs to {candidates_path}")
        print(f"[ProductMatchResolver] Wrote metrics to {metrics_path}")

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "pairs_path": str(pairs_path),
            "candidate_pairs_path": str(candidates_path),
            "metrics_path": str(metrics_path),
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
