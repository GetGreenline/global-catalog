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
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.matching.products.fuzzy_matcher_v3 import run_fuzzy_matching_token_set
from global_catalog.matching.products.transformer_matcher import TransformerMatcher
from global_catalog.matching.products.cross_encoder_matcher import CrossEncoderMatcher


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
        normalizer: Optional[ProductNormalizer] = None,
        blocking_config: Optional[BlockingConfig] = None,
        fuzzy_config: Optional[FuzzyMatcherConfig] = None,
        local_run: bool = False,
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
        self.blocking_config = blocking_config or BlockingConfig()
        self.blocking_strategy = self._resolve_blocking_strategy(self.blocking_config.strategy_name)
        self.fuzzy_config = fuzzy_config or FuzzyMatcherConfig()
        cfg_pairs_dir = self.blocking_config.pairs_dir or pairs_cache_dir
        self.pairs_cache_dir = Path(cfg_pairs_dir) if cfg_pairs_dir else None
        cfg_use_local = getattr(self.blocking_config, "use_local_pairs", False)
        # Only use cached pairs when explicitly requested via both the pipeline flag and the config flag.
        self.local_run = bool(local_run and cfg_use_local)

    def ingest(self) -> Dict[str, Any]:
        """Load product source datasets from snapshots."""
        payload: Dict[str, pd.DataFrame] = {}
        for source, rel_path in self.source_files.items():
            csv_path = self._resolve_path(rel_path)
            payload[source] = self._load_products_snapshot(csv_path, source)

        return payload

    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Apply product-specific normalization prior to matching."""
        df = self.normalizer.process(
            hoodie_df=raw_data.get("hoodie"),
            weedmaps_df=raw_data.get("weedmaps"),
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

        run_id = None
        run_dir = None
        matcher_cfg = getattr(self.matcher, "cfg", None)
        if matcher_cfg is not None:
            pair_chunk_size = getattr(matcher_cfg, "pair_chunk_size", 0)
            chunk_output_dir = getattr(matcher_cfg, "chunk_output_dir", None)
            if pair_chunk_size and not chunk_output_dir:
                prepare_fn = getattr(self.resolver, "prepare_run_dir", None)
                if callable(prepare_fn):
                    run_id, run_dir = prepare_fn(strategy_name, matcher_name)
                    matcher_cfg.chunk_output_dir = str(Path(run_dir) / "chunks")

        matches = self._run_matcher(normalized_data, candidate_pairs, raw_data)
        breakdown: Dict[str, int] = {}
        fuzzy_breakdown = dict(breakdown)
        fuzzy_breakdown.setdefault("matches", int(len(matches)))
        self.logger.info(f"Matcher retained {len(matches)} matches using {matcher_name}.")

        return {
            "pairs": matches,
            "candidate_pairs": candidate_pairs,
            "metrics": {
                "blocking": blocking_metrics,
                "fuzzy_pairs": int(len(matches)),
                "fuzzy_breakdown": fuzzy_breakdown,
            },
            "blocking_strategy": strategy_name,
            "matcher_name": matcher_name,
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else None,
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

    def _run_matcher(self, normalized_data: Any, candidate_pairs: Any, raw_data: Optional[Dict[str, Any]] = None):
        """Dispatch to configured matcher or default fuzzy matcher."""
        if isinstance(self.matcher, CrossEncoderMatcher):
            return self.matcher(normalized_data, candidate_pairs, raw_data=raw_data)
        if isinstance(self.matcher, TransformerMatcher):
            return self.matcher(normalized_data, candidate_pairs, None)
        if callable(self.matcher):
            return self.matcher(normalized_data, candidate_pairs, getattr(self, "fuzzy_config", None))
        return run_fuzzy_matching_token_set(normalized_data, candidate_pairs, self.fuzzy_config)

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
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.ParserError as exc:
            self.logger.info(
                f"Failed to read {path} with default CSV parser ({exc}). Retrying while skipping bad lines."
            )
            df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip", engine="python")
        if "source" not in df.columns:
            df["source"] = source_name
        return df

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
            return "run_fuzzy_matching_token_set"
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
        if self.local_run and self.pairs_cache_dir is not None:
            cached_pairs, cached_metrics = self._load_cached_pairs(strategy_name)
            if cached_pairs is not None:
                self.logger.info(f"Using cached candidate pairs for {strategy_name}.")
                return cached_pairs, cached_metrics
        self.logger.info(f"No cached pairs used; running blocking strategy {strategy_name}.")
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
