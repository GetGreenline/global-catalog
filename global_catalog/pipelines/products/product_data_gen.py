from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_pipeline import ProductPipeline


@dataclass
class DatasetGeneratorConfig:
    """Knobs controlling dataset sampling for training pairs."""

    left_source: str = "weedmaps"
    right_source: str = "hoodie"
    sample_left: int = 5000
    positive_threshold: float = 0.8
    hard_negative_min: float = 0.4
    hard_negatives_per_left: int = 5
    very_negatives_per_left: int = 2
    very_negative_brand_mismatch: bool = True
    very_negative_measure_mismatch: bool = True
    random_seed: int = 42
    output_path: str = "artifacts/products/datasets/product_pairs.parquet"
    max_pairs: Optional[int] = None


class ProductDataGenerator(ProductPipeline):
    """Pipeline specialization that emits labeled training pairs instead of resolution artifacts."""

    def __init__(
        self,
        *,
        dataset_config: Optional[DatasetGeneratorConfig] = None,
        fuzzy_config: Optional[FuzzyMatcherConfig] = None,
        **pipeline_kwargs,
    ):
        cfg = fuzzy_config or FuzzyMatcherConfig()
        super().__init__(fuzzy_config=cfg, **pipeline_kwargs)
        self.dataset_config = dataset_config or DatasetGeneratorConfig()
        self._sampled_left_indices: Optional[pd.Index] = None

    def reset_context(self) -> None:
        super().reset_context()
        self._sampled_left_indices = None

    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        df = super().normalize(raw_data)
        cfg = self.dataset_config
        left_source = cfg.left_source.lower()
        sources = df.get("source", pd.Series([], dtype=str))
        left_mask = sources.astype(str).str.lower() == left_source
        left_df = df[left_mask]
        if left_df.empty or cfg.sample_left <= 0:
            self._sampled_left_indices = left_df.index
            return df
        sample_n = min(cfg.sample_left, len(left_df))
        sample_indices = left_df.sample(
            n=sample_n,
            replace=False,
            random_state=cfg.random_seed,
        ).index
        self._sampled_left_indices = sample_indices
        mask = (~left_mask) | df.index.isin(sample_indices)
        return df.loc[mask].copy()

    # The resolver stage becomes "build dataset"
    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: Any,
    ) -> Dict[str, Any]:
        dataset_df = self._build_dataset(match_results, normalized_data, raw_data)
        return {
            "dataset": dataset_df,
            "config": self.dataset_config,
            "pair_counts": dataset_df["label"].value_counts().to_dict() if not dataset_df.empty else {},
        }

    # --- Dataset building logic ----------------------------------------

    def _build_dataset(
        self,
        match_results: Dict[str, Any],
        normalized_data: Any,
        raw_data: Dict[str, Any],
    ) -> pd.DataFrame:
        cfg = self.dataset_config
        if not isinstance(normalized_data, pd.DataFrame):
            return pd.DataFrame()
        matches = match_results.get("pairs") if match_results else None
        matches_df = matches.copy() if isinstance(matches, pd.DataFrame) else pd.DataFrame()
        if matches_df.empty:
            return pd.DataFrame()

        left_source = cfg.left_source.lower()
        right_source = cfg.right_source.lower()
        df_left = normalized_data[normalized_data["source"].str.lower() == left_source]
        if df_left.empty:
            return pd.DataFrame()
        df_right = normalized_data[normalized_data["source"].str.lower() == right_source]
        if df_right.empty:
            return pd.DataFrame()

        raw_lookup = self._build_raw_lookup(raw_data)

        if self._sampled_left_indices is not None and len(self._sampled_left_indices) > 0:
            left_sample = df_left.loc[df_left.index.intersection(self._sampled_left_indices)]
        else:
            left_sample = df_left
        if left_sample.empty:
            return pd.DataFrame()

        rng = np.random.default_rng(cfg.random_seed)

        hard_max = cfg.positive_threshold
        hard_min = cfg.hard_negative_min

        matches_df = matches_df.copy()
        matches_df["left_source"] = matches_df["left_source"].astype(str).str.lower()
        matches_df["right_source"] = matches_df["right_source"].astype(str).str.lower()

        dataset_rows: List[Dict[str, Any]] = []
        max_pairs = cfg.max_pairs

        for left_idx in left_sample.index:
            if max_pairs and len(dataset_rows) >= max_pairs:
                break
            pos_row = self._pick_positive(matches_df, left_idx, left_source, right_source, cfg.positive_threshold)
            if pos_row is not None:
                dataset_rows.append(
                    self._record_from_pair(
                        normalized_data,
                        pos_row["left_index"],
                        pos_row["right_index"],
                        label=1,
                        similarity=pos_row["similarity"],
                        reason="positive",
                        enforce_order=True,
                        left_source=left_source,
                        right_source=right_source,
                        raw_lookup=raw_lookup,
                    )
                )
                if max_pairs and len(dataset_rows) >= max_pairs:
                    break

            hard_rows = self._pick_hard_negatives(matches_df, left_idx, left_source, right_source, hard_min, hard_max)
            for row in hard_rows[: cfg.hard_negatives_per_left]:
                if max_pairs and len(dataset_rows) >= max_pairs:
                    break
                dataset_rows.append(
                    self._record_from_pair(
                        normalized_data,
                        row["left_index"],
                        row["right_index"],
                        label=0,
                        similarity=row.get("similarity"),
                        reason="hard_negative",
                        enforce_order=True,
                        left_source=left_source,
                        right_source=right_source,
                        raw_lookup=raw_lookup,
                    )
                )
                if max_pairs and len(dataset_rows) >= max_pairs:
                    break

            very_needed = cfg.very_negatives_per_left
            if very_needed > 0:
                sampled = self._sample_very_negatives(
                    normalized_data,
                    left_idx,
                    df_right.index.to_list(),
                    very_needed,
                    rng,
                )
                for right_idx in sampled:
                    if max_pairs and len(dataset_rows) >= max_pairs:
                        break
                    dataset_rows.append(
                        self._record_from_pair(
                            normalized_data,
                            left_idx,
                            right_idx,
                            label=0,
                            similarity=None,
                            reason="very_negative",
                            enforce_order=True,
                            left_source=left_source,
                            right_source=right_source,
                            raw_lookup=raw_lookup,
                        )
                    )
                    if max_pairs and len(dataset_rows) >= max_pairs:
                        break

        if not dataset_rows:
            return pd.DataFrame()

        return pd.DataFrame(dataset_rows)

    def _pick_positive(
        self,
        matches_df: pd.DataFrame,
        left_idx: Any,
        left_source: str,
        right_source: str,
        threshold: float,
    ) -> Optional[pd.Series]:
        rows = matches_df[
            (matches_df["left_index"] == left_idx)
            & (matches_df["left_source"] == left_source)
            & (matches_df["right_source"] == right_source)
            & (matches_df["similarity"] >= threshold)
        ]
        if rows.empty:
            return None
        return rows.sort_values("similarity", ascending=False).iloc[0]

    def _pick_hard_negatives(
        self,
        matches_df: pd.DataFrame,
        left_idx: Any,
        left_source: str,
        right_source: str,
        min_score: float,
        max_score: float,
    ) -> List[pd.Series]:
        rows = matches_df[
            (matches_df["left_index"] == left_idx)
            & (matches_df["left_source"] == left_source)
            & (matches_df["right_source"] == right_source)
            & (matches_df["similarity"] >= min_score)
            & (matches_df["similarity"] < max_score)
        ]
        if rows.empty:
            return []
        rows = rows.sort_values("similarity", ascending=False)
        return [row for _, row in rows.iterrows()]

    def _sample_very_negatives(
        self,
        normalized_data: pd.DataFrame,
        left_idx: Any,
        right_indices: List[Any],
        count: int,
        rng: np.random.Generator,
    ) -> List[Any]:
        cfg = self.dataset_config
        left_row = normalized_data.loc[left_idx]
        left_brand = str(left_row.get("brand_name_norm", "") or "")
        left_measure = str(left_row.get("measure_mg", "") or "")

        pool = []
        for idx in right_indices:
            right_row = normalized_data.loc[idx]
            brand = str(right_row.get("brand_name_norm", "") or "")
            measure = str(right_row.get("measure_mg", "") or "")
            if cfg.very_negative_brand_mismatch and brand and left_brand and brand == left_brand:
                continue
            if cfg.very_negative_measure_mismatch and measure and left_measure and measure == left_measure:
                continue
            pool.append(idx)

        if not pool:
            return []
        if len(pool) <= count:
            rng.shuffle(pool)
            return pool
        return list(rng.choice(pool, size=count, replace=False))

    def _record_from_pair(
        self,
        normalized_data: pd.DataFrame,
        left_index: Any,
        right_index: Any,
        *,
        label: int,
        similarity: Optional[float],
        reason: str,
        enforce_order: bool,
        left_source: str,
        right_source: str,
        raw_lookup: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        left_row, right_row = self._order_rows(
            normalized_data,
            left_index,
            right_index,
            enforce_order=enforce_order,
            left_source=left_source,
            right_source=right_source,
        )
        return {
            "label": int(label),
            "score": float(similarity) if similarity is not None else None,
            "pair_type": reason,
            **self._flatten_row(left_row, suffix="w", raw_lookup=raw_lookup),
            **self._flatten_row(right_row, suffix="h", raw_lookup=raw_lookup),
        }

    def _order_rows(
        self,
        normalized_data: pd.DataFrame,
        left_index: Any,
        right_index: Any,
        *,
        enforce_order: bool,
        left_source: str,
        right_source: str,
    ) -> Tuple[pd.Series, pd.Series]:
        row_left = normalized_data.loc[left_index]
        row_right = normalized_data.loc[right_index]
        if not enforce_order:
            return row_left, row_right
        if str(row_left.get("source", "")).lower() == left_source:
            return row_left, row_right
        if str(row_right.get("source", "")).lower() == left_source:
            return row_right, row_left
        return row_left, row_right

    def _flatten_row(self, row: pd.Series, suffix: str, raw_lookup: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        source = row.get("source")
        product_id = row.get("product_id")
        raw_row = self._get_raw(raw_lookup, source, product_id)
        brand_raw = raw_row.get("brand_name") if raw_row else None
        product_raw = raw_row.get("product_name") if raw_row else None
        desc_raw = raw_row.get("description") if raw_row else None
        return {
            f"source_{suffix}": source,
            f"product_id_{suffix}": product_id,
            f"brand_name_{suffix}": brand_raw or row.get("brand_name_norm"),
            f"product_name_{suffix}": product_raw or row.get("product_name_norm"),
            f"measure_mg_{suffix}": row.get("measure_mg"),
            f"description_{suffix}": desc_raw or row.get("description_norm"),
        }

    def _build_raw_lookup(self, raw_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        lookup: Dict[str, pd.DataFrame] = {}
        for src, df in (raw_data or {}).items():
            if not isinstance(df, pd.DataFrame) or "product_id" not in df.columns:
                continue
            src_key = str(src).lower()
            tmp = df.copy()
            tmp["__product_id_key__"] = tmp["product_id"].astype(str)
            lookup[src_key] = tmp.set_index("__product_id_key__", drop=False)
        return lookup

    def _get_raw(self, lookup: Dict[str, pd.DataFrame], source: Any, product_id: Any) -> Dict[str, Any]:
        if source is None or product_id is None:
            return {}
        table = lookup.get(str(source).lower())
        if table is None:
            return {}
        key = str(product_id)
        try:
            row = table.loc[key]
        except Exception:
            return {}
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        data = row.to_dict()
        data.pop("__product_id_key__", None)
        return data
