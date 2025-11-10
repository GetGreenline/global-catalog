import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import pandas as pd

from global_catalog.pipelines.entity_pipeline import EntityPipeline
from global_catalog.matching.products.fuzzy_matcher import fuzzy_matches_from_pairs
from global_catalog.pipelines.products.products_writer import write_pairs, summarize
from global_catalog.transformers.products.products_normalization import ProductNormalizer

class ProductPipeline(EntityPipeline):
    def __init__(
        self,
        repo,
        matcher,
        resolver,
        publisher_fn: Optional[Callable] = None,
        normalizer: Optional["ProductNormalizer"] = None,
        blocker: Optional[Callable] = None,
        include_description: bool = True,
        description_token_limit: int = 20,
        enforce_uom: bool = False,
        use_states: bool = False,
        use_country: bool = False,
        use_strain: bool = False,
        measure_tol: float = 0,
    ):
        super().__init__(repo=repo, matcher=matcher, resolver=resolver, publisher_fn=publisher_fn)
        self.normalizer = normalizer or ProductNormalizer()
        self.blocker = blocker
        self.include_description = include_description
        self.description_token_limit = description_token_limit
        self.enforce_uom = enforce_uom
        self.use_states = use_states
        self.use_country = use_country
        self.use_strain = use_strain
        self.measure_tol = measure_tol

    def _read_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for c in df.columns:
            df[c] = df[c].astype(str)
        return df

    def _ensure_blocking_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "product_name_norm" not in df.columns and "normalized_product_name" in df.columns:
            df["product_name_norm"] = df["normalized_product_name"]
        if "description_norm" not in df.columns and "normalized_description" in df.columns:
            df["description_norm"] = df["normalized_description"]
        if "uom_norm" not in df.columns and "uom" in df.columns:
            df["uom_norm"] = df["uom"]
        for col in ["states_norm", "country_norm", "strain_type_norm"]:
            if col not in df.columns:
                df[col] = ""
        return df

    def run_local(
        self,
        hoodie_path: str,
        weedmaps_path: str,
        brands_path: Optional[str],
        local_out_root: str,
        out_pairs_csv: str,
        threshold: float = 0.55,
        max_per_left: int = 200,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.blocker is None:
            raise ValueError("A blocker callable must be provided.")
        t0 = time.perf_counter()

        df_hoodie_raw = self._read_csv(hoodie_path)
        df_weedmaps_raw = self._read_csv(weedmaps_path)
        brand_df = self._read_csv(brands_path) if brands_path else None

        df_norm = self.normalizer.process(
            hoodie_df=df_hoodie_raw,
            weedmaps_df=df_weedmaps_raw,
            brand_df=brand_df,
        )
        df_norm = self._ensure_blocking_columns(df_norm)

        df_r = df_norm[df_norm["source"] == "hoodie"].copy()
        df_l = df_norm[df_norm["source"] == "weedmaps"].copy()

        pairs_df = self.blocker(
            df_norm,
            threshold=threshold,
            max_per_left=max_per_left,
            min_jaccard=threshold,
            measure_tol=self.measure_tol,
            enforce_uom=self.enforce_uom,
            use_states=self.use_states,
            use_country=self.use_country,
            use_strain=self.use_strain,
            include_description=self.include_description,
            description_token_limit=self.description_token_limit,
        )

        if pairs_df is None or pairs_df.empty:
            out = pd.DataFrame(columns=[
                "left_source", "right_source",
                "left_product_id", "right_product_id",
                "left_brand_name", "right_brand_name",
                "left_product_name", "right_product_name",
                "left_uom", "right_uom",
                "left_measure_mg", "right_measure_mg",
                "similarity", "match_type", "pair_name_jaccard"
            ])
            pairs_generated = 0
        else:
            out = fuzzy_matches_from_pairs(
                df_norm=df_norm,
                pairs_df=pairs_df,
                threshold=0.75,
                include_description=self.include_description,
            )
            pairs_generated = int(len(pairs_df))

        out = out.sort_values(["similarity"], ascending=[False]).reset_index(drop=True)
        Path(local_out_root).mkdir(parents=True, exist_ok=True)
        write_pairs(out, out_pairs_csv)

        run_id = uuid.uuid4().hex[:8]
        metrics = summarize(pd.DataFrame(), out, len(df_l), len(df_r), pairs_generated, threshold)
        metrics_full = {
            "run_id": run_id,
            "timing_seconds_total": round(time.perf_counter() - t0, 3),
            "inputs": {
                "hoodie_path": hoodie_path,
                "weedmaps_path": weedmaps_path,
                "brands_path": brands_path,
                "threshold": threshold,
                "max_per_left": max_per_left,
                "include_description": self.include_description,
                "description_token_limit": self.description_token_limit,
                "enforce_uom": self.enforce_uom,
                "use_states": self.use_states,
                "use_country": self.use_country,
                "use_strain": self.use_strain,
                "measure_tol": self.measure_tol,
            },
            "outputs": {
                "pairs_csv": out_pairs_csv,
                "local_out_root": local_out_root,
            },
            "matching_metrics": metrics,
            "pairs_generated": pairs_generated,
        }
        if extra_metrics:
            metrics_full.update(extra_metrics)

        (Path(local_out_root) / "metrics.json").write_text(json.dumps(metrics_full, indent=2), encoding="utf-8")

        published = {}
        if callable(self.publisher_fn):
            published = self.publisher_fn(run_id=run_id, local_artifacts={"pairs_csv": out_pairs_csv}) or {}
            metrics_full["published"] = published

        return {"pairs": out, "pairs_count": int(len(out)), "metrics": metrics_full, "published": published}
