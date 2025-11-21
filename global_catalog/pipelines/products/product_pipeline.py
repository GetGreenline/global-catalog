from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from global_catalog.pipelines.entity_pipeline import EntityPipeline
from global_catalog.transformers.products.products_normalization import ProductNormalizer


class ProductPipeline(EntityPipeline):
    """Pipeline shell that loads snapshots and prepares normalized product data."""

    def __init__(
        self,
        repo,
        matcher=None,
        resolver=None,
        publisher_fn=None,
        snapshot_root: str = "data/snapshots/products",
        source_files: Optional[Dict[str, str]] = None,
        brand_file: Optional[str] = "brands_w.csv",
        normalizer: Optional[ProductNormalizer] = None,
    ):
        super().__init__(repo=repo, matcher=matcher, resolver=resolver, publisher_fn=publisher_fn)
        self.blocker = None  # TODO(products): wire in blocking strategy once finalized.
        self.normalizer = normalizer or ProductNormalizer()
        self.snapshot_root = Path(snapshot_root)
        default_sources = {
            "weedmaps": "products_weedmaps.csv",
            "hoodie": "products_hoodie.csv",
        }
        self.source_files = source_files or default_sources
        self.brand_file = Path(brand_file) if brand_file else None

    def ingest(self) -> Dict[str, Any]:
        """Load source datasets and auxiliary brand data from snapshots."""
        payload: Dict[str, pd.DataFrame] = {}
        for source, rel_path in self.source_files.items():
            csv_path = self._resolve_path(rel_path)
            payload[source] = self._load_products_snapshot(csv_path, source)

        brand_df = self._load_brand_snapshot()
        if brand_df is not None:
            payload["brands"] = brand_df
        return payload

    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Apply product-specific normalization prior to matching."""
        df = self.normalizer.process(
            hoodie_df=raw_data.get("hoodie"),
            weedmaps_df=raw_data.get("weedmaps"),
            brand_df=raw_data.get("brands"),
        )
        df = self._ensure_blocking_columns(df)
        return df

    def match(self, normalized_data: Any, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate product pairs and metrics."""
        # TODO(products): integrate blocker + matcher orchestration.
        raise NotImplementedError("TODO(products): implement match step.")

    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: Any,
    ) -> Optional[Any]:
        """Finalize product matches into publishable artifacts."""
        # TODO(products): implement resolver once artifact contract is decided.
        raise NotImplementedError("TODO(products): implement resolve step.")

    def _resolve_path(self, rel_path: str | Path) -> Path:
        """Resolve user-supplied paths, supporting absolute or repo-relative inputs."""
        path = Path(rel_path)
        if path.is_absolute():
            return path
        # Prefer explicit repo-relative paths if they exist.
        if path.exists():
            return path
        candidate = self.snapshot_root / path
        return candidate

    def _load_products_snapshot(self, path: Path, source_name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Product snapshot not found: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "source" not in df.columns:
            df["source"] = source_name
        return df

    def _load_brand_snapshot(self) -> Optional[pd.DataFrame]:
        if not self.brand_file:
            return None
        path = self._resolve_path(self.brand_file)
        if not path.exists():
            raise FileNotFoundError(f"Brand snapshot not found: {path}")
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def _ensure_blocking_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee columns downstream blockers/matchers rely on are present."""
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
