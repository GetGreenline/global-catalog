# File: global_catalog/pipelines/products/product_data_gen.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_pipeline import ProductPipeline


@dataclass
class DatasetGeneratorConfig:
    """Knobs controlling dataset sampling for training/eval pairs."""

    # Which catalog is the "anchor/query" side (we sample from this side)
    left_source: str = "weedmaps"
    right_source: str = "hoodie"

    # Sampling / sizing
    sample_left: int = 5000
    hard_negatives_per_left: int = 5
    very_negatives_per_left: int = 2
    max_pairs: Optional[int] = None
    random_seed: int = 42

    # Silver-labeling thresholds (based on matcher similarity from match_results)
    positive_threshold: float = 0.80
    hard_negative_min: float = 0.40

    # Strong "very negative" rules (prefer guarantees over score thresholds)
    very_negative_uom_mismatch: bool = True
    very_negative_brand_mismatch: bool = True
    very_negative_measure_far: bool = True
    very_negative_measure_ratio: float = 3.0  # e.g. 10mg vs 50mg isn't "very"; 10mg vs 100mg often is

    # Optional gating to make hard negatives truly "hard"
    hard_negative_require_uom_match: bool = True
    hard_negative_prefer_brand_match: bool = False  # if True, only keep hard negs where brand matches (when present)

    # Output formatting
    output_path: str = "artifacts/products/datasets/product_pairs.parquet"
    desc_max_chars: int = 800  # for text fields; keep bounded for reproducibility


class ProductDataGenerator(ProductPipeline):
    """
    Pipeline specialization that emits labeled pairs (wide rows) instead of resolved artifacts.

    Important: labels produced here are "silver" unless you wire positives from a trusted source
    (e.g., resolver winners or manually-labeled pairs).
    """

    def __init__(
        self,
        *,
        dataset_config: Optional[DatasetGeneratorConfig] = None,
        fuzzy_config: Optional[FuzzyMatcherConfig] = None,
        **pipeline_kwargs: Any,
    ):
        cfg = fuzzy_config or FuzzyMatcherConfig()
        super().__init__(fuzzy_config=cfg, **pipeline_kwargs)
        self.dataset_config = dataset_config or DatasetGeneratorConfig()
        self._sampled_left_indices: Optional[pd.Index] = None

    def reset_context(self) -> None:
        super().reset_context()
        self._sampled_left_indices = None

    # --- Stage overrides ------------------------------------------------

    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Normalize as usual, but downsample ONLY the left_source rows so we control pair explosion.
        We keep all right_source rows to allow matching into the full catalog.
        """
        df = super().normalize(raw_data)
        if not isinstance(df, pd.DataFrame) or df.empty:
            self._sampled_left_indices = pd.Index([])
            return df

        cfg = self.dataset_config
        left_source = cfg.left_source.lower()

        # Be safe with nulls
        df = df.copy()
        df["source"] = df.get("source", "").astype(str)

        left_mask = df["source"].str.lower() == left_source
        left_df = df[left_mask]

        if left_df.empty or cfg.sample_left <= 0:
            self._sampled_left_indices = left_df.index
            return df

        sample_n = min(cfg.sample_left, len(left_df))
        sample_indices = left_df.sample(n=sample_n, replace=False, random_state=cfg.random_seed).index
        self._sampled_left_indices = sample_indices

        # Keep: all non-left rows + sampled left rows
        keep_mask = (~left_mask) | df.index.isin(sample_indices)
        return df.loc[keep_mask].copy()

    def resolve(
        self,
        match_results: Union[Dict[str, Any], pd.DataFrame, None],
        raw_data: Dict[str, Any],
        normalized_data: Any,
    ) -> Dict[str, Any]:
        """
        Replace the normal resolver stage with dataset creation.
        """
        dataset_df = self._build_dataset(match_results, normalized_data, raw_data)
        return {
            "dataset": dataset_df,
            "config": self.dataset_config,
            "pair_counts": dataset_df["label"].value_counts().to_dict() if not dataset_df.empty else {},
        }

    # --- Dataset building logic ----------------------------------------

    def _build_dataset(
        self,
        match_results: Union[Dict[str, Any], pd.DataFrame, None],
        normalized_data: Any,
        raw_data: Dict[str, Any],
    ) -> pd.DataFrame:
        cfg = self.dataset_config

        if not isinstance(normalized_data, pd.DataFrame) or normalized_data.empty:
            return pd.DataFrame()

        # Extract scored pairs DataFrame robustly
        matches_df = self._extract_matches_df(match_results)
        if matches_df.empty:
            return pd.DataFrame()

        # Normalize source columns safely
        normalized_data = normalized_data.copy()
        normalized_data["source"] = normalized_data.get("source", "").astype(str)

        matches_df = matches_df.copy()
        for col in ("left_source", "right_source"):
            if col in matches_df.columns:
                matches_df[col] = matches_df[col].astype(str).str.lower()

        left_source = cfg.left_source.lower()
        right_source = cfg.right_source.lower()

        if {"left_source", "right_source"}.issubset(matches_df.columns):
            matches_df = matches_df[
                (matches_df["left_source"] == left_source) & (matches_df["right_source"] == right_source)
            ]
            if matches_df.empty:
                return pd.DataFrame()

        df_left = normalized_data[normalized_data["source"].str.lower() == left_source]
        df_right = normalized_data[normalized_data["source"].str.lower() == right_source]
        if df_left.empty or df_right.empty:
            return pd.DataFrame()

        raw_lookup = self._build_raw_lookup(raw_data)
        rng = np.random.default_rng(cfg.random_seed)

        dataset_rows: List[Dict[str, Any]] = []
        max_pairs = cfg.max_pairs

        # Small perf helper: group matches by left_index once
        if "left_index" not in matches_df.columns:
            return pd.DataFrame()
        grouped = matches_df.groupby("left_index", sort=False)

        left_indices = list(grouped.indices.keys())
        if self._sampled_left_indices is not None and len(self._sampled_left_indices) > 0:
            sampled_set = set(self._sampled_left_indices)
            left_indices = [idx for idx in left_indices if idx in sampled_set]
        if not left_indices:
            return pd.DataFrame()

        right_indices: List[Any] = []
        if cfg.very_negatives_per_left > 0:
            right_indices = df_right.index.to_list()

        for left_idx in left_indices:
            if max_pairs and len(dataset_rows) >= max_pairs:
                break

            left_rows = grouped.get_group(left_idx) if left_idx in grouped.indices else pd.DataFrame()
            if left_rows.empty:
                continue

            # 1) Positive (silver) = top-scoring >= positive_threshold
            pos = self._pick_positive(left_rows, cfg.positive_threshold)
            pos_right_idx = None
            if pos is not None:
                pos_right_idx = pos["right_index"]
                dataset_rows.append(
                    self._record_from_pair(
                        normalized_data,
                        left_index=pos["left_index"],
                        right_index=pos["right_index"],
                        label=1,
                        similarity=pos.get("similarity"),
                        pair_type="positive",
                        left_source=left_source,
                        right_source=right_source,
                        raw_lookup=raw_lookup,
                    )
                )
                if max_pairs and len(dataset_rows) >= max_pairs:
                    break

            # 2) Hard negatives = high-ish scores but below positive threshold
            hard_rows = self._pick_hard_negatives(
                left_rows,
                min_score=cfg.hard_negative_min,
                max_score=cfg.positive_threshold,
                normalized_data=normalized_data,
                left_idx=left_idx,
                exclude_right=pos_right_idx,
                left_source=left_source,
                right_source=right_source,
            )
            for row in hard_rows.head(cfg.hard_negatives_per_left).itertuples(index=False):
                if max_pairs and len(dataset_rows) >= max_pairs:
                    break
                dataset_rows.append(
                    self._record_from_pair(
                        normalized_data,
                        left_index=row.left_index,
                        right_index=row.right_index,
                        label=0,
                        similarity=getattr(row, "similarity", None),
                        pair_type="hard_negative",
                        left_source=left_source,
                        right_source=right_source,
                        raw_lookup=raw_lookup,
                    )
                )
            if max_pairs and len(dataset_rows) >= max_pairs:
                break

            # 3) Very negatives = rule-guaranteed non-matches sampled from right universe
            if cfg.very_negatives_per_left > 0 and right_indices:
                sampled_rights = self._sample_very_negatives(
                    normalized_data=normalized_data,
                    left_idx=left_idx,
                    right_indices=right_indices,
                    count=cfg.very_negatives_per_left,
                    rng=rng,
                )
                for right_idx in sampled_rights:
                    if max_pairs and len(dataset_rows) >= max_pairs:
                        break
                    dataset_rows.append(
                        self._record_from_pair(
                            normalized_data,
                            left_index=left_idx,
                            right_index=right_idx,
                            label=0,
                            similarity=None,
                            pair_type="very_negative",
                            left_source=left_source,
                            right_source=right_source,
                            raw_lookup=raw_lookup,
                        )
                    )

        if not dataset_rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(dataset_rows)

        # Optional: dedupe exact duplicates
        if {"source_w", "product_id_w", "source_h", "product_id_h"}.issubset(df_out.columns):
            df_out["pair_id"] = (
                df_out["source_w"].astype(str)
                + "|"
                + df_out["product_id_w"].astype(str)
                + "||"
                + df_out["source_h"].astype(str)
                + "|"
                + df_out["product_id_h"].astype(str)
            )
            df_out = df_out.drop_duplicates(subset=["pair_id", "label", "pair_type"], keep="first")

        return df_out

    # --- Match rows selection ------------------------------------------

    @staticmethod
    def _pick_positive(left_rows: pd.DataFrame, threshold: float) -> Optional[pd.Series]:
        if left_rows is None or left_rows.empty or "similarity" not in left_rows.columns:
            return None
        rows = left_rows[left_rows["similarity"] >= threshold]
        if rows.empty:
            return None
        return rows.sort_values("similarity", ascending=False).iloc[0]

    def _pick_hard_negatives(
        self,
        left_rows: pd.DataFrame,
        *,
        min_score: float,
        max_score: float,
        normalized_data: pd.DataFrame,
        left_idx: Any,
        exclude_right: Optional[Any],
        left_source: str,
        right_source: str,
    ) -> pd.DataFrame:
        cfg = self.dataset_config
        if left_rows is None or left_rows.empty or "similarity" not in left_rows.columns:
            return pd.DataFrame()

        rows = left_rows[(left_rows["similarity"] >= min_score) & (left_rows["similarity"] < max_score)].copy()
        if "left_source" in rows.columns:
            rows = rows[rows["left_source"] == left_source]
        if "right_source" in rows.columns:
            rows = rows[rows["right_source"] == right_source]
        if rows.empty:
            return pd.DataFrame()

        if exclude_right is not None and "right_index" in rows.columns:
            rows = rows[rows["right_index"] != exclude_right]
            if rows.empty:
                return pd.DataFrame()

        # Optional gating to make these truly "hard"
        if cfg.hard_negative_require_uom_match:
            left_uom = str(normalized_data.loc[left_idx].get("uom_norm") or "")
            if left_uom:
                def _uom_match(ridx: Any) -> bool:
                    try:
                        ruom = str(normalized_data.loc[ridx].get("uom_norm") or "")
                    except Exception:
                        return False
                    return (not ruom) or (ruom == left_uom)

                rows = rows[rows["right_index"].apply(_uom_match)]
                if rows.empty:
                    return pd.DataFrame()

        if cfg.hard_negative_prefer_brand_match:
            left_brand = str(normalized_data.loc[left_idx].get("brand_name_norm") or "")
            if left_brand:
                def _brand_match(ridx: Any) -> bool:
                    try:
                        rbrand = str(normalized_data.loc[ridx].get("brand_name_norm") or "")
                    except Exception:
                        return False
                    return rbrand == left_brand

                rows = rows[rows["right_index"].apply(_brand_match)]
                if rows.empty:
                    return pd.DataFrame()

        return rows.sort_values("similarity", ascending=False)

    # --- Very negatives -------------------------------------------------

    def _sample_very_negatives(
        self,
        *,
        normalized_data: pd.DataFrame,
        left_idx: Any,
        right_indices: List[Any],
        count: int,
        rng: np.random.Generator,
    ) -> List[Any]:
        cfg = self.dataset_config
        if count <= 0 or not right_indices:
            return []

        left_row = normalized_data.loc[left_idx]
        left_brand = str(left_row.get("brand_name_norm") or "")
        left_uom = str(left_row.get("uom_norm") or "")
        left_measure_val = self._measure_to_float(left_row.get("measure_mg"))

        picked: List[Any] = []
        seen = set()
        max_attempts = max(200, count * 200)
        n = len(right_indices)

        for _ in range(max_attempts):
            if len(picked) >= count:
                break

            ridx = right_indices[int(rng.integers(0, n))]
            if ridx in seen:
                continue
            seen.add(ridx)
            try:
                right_row = normalized_data.loc[ridx]
            except Exception:
                continue

            right_brand = str(right_row.get("brand_name_norm") or "")
            right_uom = str(right_row.get("uom_norm") or "")
            right_measure_val = self._measure_to_float(right_row.get("measure_mg"))

            # 1) Strongest: UOM mismatch
            if cfg.very_negative_uom_mismatch and left_uom and right_uom and left_uom != right_uom:
                picked.append(ridx)
                continue

            # 2) Brand mismatch
            brand_mismatch = False
            if cfg.very_negative_brand_mismatch and left_brand and right_brand and left_brand != right_brand:
                brand_mismatch = True

            # 3) Measure far apart (ratio)
            measure_far = False
            if cfg.very_negative_measure_far and left_measure_val and right_measure_val:
                big = max(left_measure_val, right_measure_val)
                small = min(left_measure_val, right_measure_val)
                if small > 0 and (big / small) >= float(cfg.very_negative_measure_ratio):
                    measure_far = True

            # Consider it "very negative" if we have strong evidence
            if brand_mismatch and measure_far:
                picked.append(ridx)

        return picked

    @staticmethod
    def _measure_to_float(measure: Any) -> Optional[float]:
        """
        Extract leading numeric part from measure_mg formats like:
          - 28000-milligrams
          - "28000"
          - 28000
        Returns float or None.
        """
        if measure is None:
            return None
        if isinstance(measure, (int, float)) and not pd.isna(measure):
            val = float(measure)
            return val if val > 0 else None
        s = str(measure)
        if not s or s.lower() in ("nan", "none"):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if not m:
            return None
        try:
            val = float(m.group(1))
        except Exception:
            return None
        return val if val > 0 else None

    # --- Row flattening / serialization --------------------------------

    def _record_from_pair(
        self,
        normalized_data: pd.DataFrame,
        *,
        left_index: Any,
        right_index: Any,
        label: int,
        similarity: Optional[float],
        pair_type: str,
        left_source: str,
        right_source: str,
        raw_lookup: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        left_row, right_row = self._order_rows(
            normalized_data,
            left_index=left_index,
            right_index=right_index,
            left_source=left_source,
            right_source=right_source,
        )
        out: Dict[str, Any] = {
            "label": int(label),
            "score": float(similarity) if similarity is not None else None,
            "pair_type": str(pair_type),
            "left_index": left_index,
            "right_index": right_index,
        }
        out.update(self._flatten_row(left_row, suffix="w", raw_lookup=raw_lookup))
        out.update(self._flatten_row(right_row, suffix="h", raw_lookup=raw_lookup))
        return out

    @staticmethod
    def _order_rows(
        normalized_data: pd.DataFrame,
        *,
        left_index: Any,
        right_index: Any,
        left_source: str,
        right_source: str,
    ) -> Tuple[pd.Series, pd.Series]:
        row_a = normalized_data.loc[left_index]
        row_b = normalized_data.loc[right_index]

        src_a = str(row_a.get("source", "")).lower()
        src_b = str(row_b.get("source", "")).lower()

        # Ensure returned order is (left_source row, right_source row)
        if src_a == left_source and src_b == right_source:
            return row_a, row_b
        if src_b == left_source and src_a == right_source:
            return row_b, row_a

        # Fallback (shouldn't happen often)
        return row_a, row_b

    def _flatten_row(self, row: pd.Series, *, suffix: str, raw_lookup: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        cfg = self.dataset_config
        source = row.get("source")
        product_id = row.get("product_id")

        raw_row = self._get_raw(raw_lookup, source, product_id)

        brand_raw = raw_row.get("brand_name") if raw_row else None
        product_raw = raw_row.get("product_name") if raw_row else None
        desc_raw = raw_row.get("description") if raw_row else None

        brand = (brand_raw or row.get("brand_name_norm") or "").strip()
        name = (product_raw or row.get("product_name_norm") or "").strip()
        desc = (desc_raw or row.get("description_norm") or "").strip()
        uom = (row.get("uom_norm") or "").strip()
        measure = row.get("measure_mg")

        # Bound description for reproducibility
        if isinstance(desc, str) and cfg.desc_max_chars and len(desc) > cfg.desc_max_chars:
            desc = desc[: cfg.desc_max_chars]

        text = " ".join([str(x).strip() for x in [brand, name, measure, uom] if x not in (None, "", "nan")])
        if desc:
            text = f"{text} || {desc}"

        return {
            f"source_{suffix}": source,
            f"product_id_{suffix}": product_id,
            f"brand_name_{suffix}": brand or None,
            f"product_name_{suffix}": name or None,
            f"measure_mg_{suffix}": measure,
            f"uom_{suffix}": uom or None,
            f"description_{suffix}": desc or None,
            f"text_{suffix}": text or None,
        }

    # --- Raw lookups ----------------------------------------------------

    @staticmethod
    def _build_raw_lookup(raw_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        raw_data is typically {source_name: df_raw}. We index by product_id (string).
        """
        lookup: Dict[str, pd.DataFrame] = {}
        if not isinstance(raw_data, dict):
            return lookup

        for src, df in raw_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "product_id" not in df.columns:
                continue

            src_key = str(src).lower()
            tmp = df.copy()
            tmp["__product_id_key__"] = tmp["product_id"].astype(str)
            lookup[src_key] = tmp.set_index("__product_id_key__", drop=False)
        return lookup

    @staticmethod
    def _get_raw(lookup: Dict[str, pd.DataFrame], source: Any, product_id: Any) -> Dict[str, Any]:
        if source is None or product_id is None:
            return {}
        table = lookup.get(str(source).lower())
        if table is None or table.empty:
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

    # --- Helpers --------------------------------------------------------

    @staticmethod
    def _extract_matches_df(match_results: Union[Dict[str, Any], pd.DataFrame, None]) -> pd.DataFrame:
        """
        Try hard to find the scored pairs DataFrame from the pipeline's match stage output.
        """
        if match_results is None:
            return pd.DataFrame()

        if isinstance(match_results, pd.DataFrame):
            return match_results

        if isinstance(match_results, dict):
            for key in ("pairs", "matches", "match_pairs", "scored_pairs", "candidate_pairs"):
                val = match_results.get(key)
                if isinstance(val, pd.DataFrame) and not val.empty:
                    return val

        return pd.DataFrame()
