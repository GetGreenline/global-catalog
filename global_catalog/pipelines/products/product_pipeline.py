import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from global_catalog.pipelines.entity_pipeline import EntityPipeline
from global_catalog.transformers.products.products_normalization import ProductNormalizer
from global_catalog.matching.products.blocking_v2 import (
    BlockingConfig,
    blocking_strategy_one,
    blocking_strategy_two,
    blocking_strategy_three,
    blocking_strategy_four,
    blocking_strategy_five,
    build_candidates,
)
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig, run_fuzzy_matching


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
        blocking_config: Optional[BlockingConfig] = None,
        fuzzy_config: Optional[FuzzyMatcherConfig] = None,
        local_run: bool = True,
        pairs_cache_dir: Optional[str] = "artifacts/products/pairs",
    ):
        super().__init__(repo=repo, matcher=matcher, resolver=resolver, publisher_fn=publisher_fn)
        self.normalizer = normalizer or ProductNormalizer()
        self.snapshot_root = Path(snapshot_root)
        default_sources = {
            "weedmaps": "products_weedmaps.csv",
            "hoodie": "products_hoodie.csv",
        }
        self.source_files = source_files or default_sources
        self.brand_file = Path(brand_file) if brand_file else None
        self.blocking_config = blocking_config or BlockingConfig()
        self.blocking_strategy = self._resolve_blocking_strategy(self.blocking_config.strategy_name)
        self.fuzzy_config = fuzzy_config or FuzzyMatcherConfig()
        cfg_pairs_dir = self.blocking_config.pairs_dir or pairs_cache_dir
        self.pairs_cache_dir = Path(cfg_pairs_dir) if cfg_pairs_dir else None
        cfg_use_local = getattr(self.blocking_config, "use_local_pairs", False)
        self.local_run = bool(local_run or cfg_use_local)

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
        strategy_name = self._blocking_strategy_name()
        matcher_name = self._matcher_name()
        candidate_pairs, blocking_metrics = self._resolve_candidate_pairs(strategy_name, normalized_data)

        candidate_count = 0 if candidate_pairs is None else len(candidate_pairs)
        self.logger.info(f"Blocking produced {candidate_count} candidate pairs using {strategy_name}.")

        self.logger.info("Running single-pass fuzzy matcher.")
        fuzzy_matches = run_fuzzy_matching(normalized_data, candidate_pairs, self.fuzzy_config)
        fuzzy_breakdown = {"matches": int(len(fuzzy_matches))}
        self.logger.info(f"Fuzzy matcher retained {len(fuzzy_matches)} matches.")

        return {
            "pairs": fuzzy_matches,
            "candidate_pairs": candidate_pairs,
            "metrics": {
                "blocking": blocking_metrics,
                "fuzzy_pairs": int(len(fuzzy_matches)),
                "fuzzy_breakdown": fuzzy_breakdown,
            },
            "blocking_strategy": strategy_name,
            "matcher_name": matcher_name,
        }

    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: Any,
    ) -> Optional[Any]:
        """Finalize product matches into publishable artifacts."""
        if self.resolver is None:
            return match_results
        resolver_fn = getattr(self.resolver, "resolve", None)
        if callable(resolver_fn):
            return resolver_fn(match_results, raw_data, normalized_data)
        raise ValueError("Resolver must provide a callable `resolve` method.")

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

    def _blocking_strategy_name(self) -> str:
        strategy_fn = getattr(self, "blocking_strategy", None)
        if strategy_fn is None:
            return "blocking_strategy"
        return getattr(strategy_fn, "__name__", strategy_fn.__class__.__name__)

    def _resolve_blocking_strategy(self, strategy_name: Optional[str]):
        strategy_map = {
            "strategy_one": blocking_strategy_one,
            "strategy_two": blocking_strategy_two,
            "strategy_three": blocking_strategy_three,
            "strategy_four": blocking_strategy_four,
            "strategy_five": blocking_strategy_five,
        }
        if not strategy_name:
            return blocking_strategy_one
        normalized = str(strategy_name).strip().lower()
        if normalized not in strategy_map:
            raise ValueError(
                f"Unknown blocking strategy '{strategy_name}'. Expected one of: {', '.join(strategy_map.keys())}."
            )
        return strategy_map[normalized]

    def _matcher_name(self) -> str:
        matcher = getattr(self, "matcher", None)
        if matcher is None:
            return "run_fuzzy_matching"
        name = getattr(matcher, "__name__", None)
        if name:
            return name
        matcher_cls = getattr(matcher, "__class__", None)
        if matcher_cls is not None and hasattr(matcher_cls, "__name__"):
            return matcher_cls.__name__
        return str(matcher)

    def _resolve_candidate_pairs(
        self, strategy_name: str, normalized_data: pd.DataFrame
    ) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        if self.local_run:
            cached_pairs, cached_metrics = self._load_cached_pairs(strategy_name)
            if cached_pairs is not None:
                self.logger.info(f"Using cached candidate pairs for {strategy_name}.")
                return cached_pairs, cached_metrics
        self.logger.info(f"No cached pairs found; running blocking strategy {strategy_name}.")
        blocking_out = build_candidates(normalized_data, self.blocking_config, self.blocking_strategy)
        return blocking_out.get("pairs"), blocking_out.get("metrics", {})

    def _load_cached_pairs(self, strategy_name: str) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        if not self.local_run or self.pairs_cache_dir is None:
            return None, {}

        candidate_files = [
            path
            for path in self.pairs_cache_dir.rglob("*.parquet")
            if strategy_name in path.stem or strategy_name in path.as_posix()
        ]
        if not candidate_files:
            candidate_files = list(self.pairs_cache_dir.rglob("*.parquet"))
        if not candidate_files:
            return None, {}

        latest = max(candidate_files, key=lambda p: p.stat().st_mtime)
        try:
            pairs_df = pd.read_parquet(latest)
        except Exception as exc:
            self.logger.warning(f"Failed to load cached pairs from {latest}: {exc}")
            return None, {}

        metrics: Dict[str, Any] = {}
        meta_path = latest.with_suffix(".json")
        if meta_path.exists():
            try:
                with meta_path.open() as fh:
                    metadata = json.load(fh)
                metrics = metadata.get("metrics", {})
            except Exception as exc:
                self.logger.warning(f"Failed to read pair metadata {meta_path}: {exc}")

        return pairs_df, metrics
