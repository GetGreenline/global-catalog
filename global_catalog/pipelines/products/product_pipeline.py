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
    blocking_strategy_all_pairs,
    build_candidates,
)
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.matching.products.fuzzy_matcher_v3 import run_fuzzy_matching_token_set


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
        combined_source_path: Optional[str] = None,
        normalizer: Optional[ProductNormalizer] = None,
        blocking_config: Optional[BlockingConfig] = None,
        fuzzy_config: Optional[FuzzyMatcherConfig] = None,
        local_run: bool = True,
        pairs_cache_dir: Optional[str] = "artifacts/products/pairs",
    ):
        super().__init__(repo=repo, matcher=matcher, resolver=resolver, publisher_fn=publisher_fn)
        self.normalizer = normalizer or ProductNormalizer()
        self.snapshot_root = Path(snapshot_root)
        self.combined_source_path = Path(combined_source_path) if combined_source_path else None
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
        self.local_run = bool(local_run or cfg_use_local)

    def ingest(self) -> Dict[str, Any]:
        """Load product source datasets from snapshots."""
        payload: Dict[str, pd.DataFrame] = {}
        if self.combined_source_path:
            combined_df = self._load_products_snapshot(self.combined_source_path, "combined")
            if "source" not in combined_df.columns:
                raise ValueError("Combined products file must include a 'source' column.")
            combined_df["source"] = combined_df["source"].astype(str).str.strip().str.lower()
            for src in ("weedmaps", "hoodie"):
                subset = combined_df[combined_df["source"] == src]
                payload[src] = subset.reset_index(drop=True)
            return payload
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

        self.logger.info("Running token-set fuzzy matcher (pass 1).")
        fuzzy_matches_pass1 = run_fuzzy_matching_token_set(normalized_data, candidate_pairs, self.fuzzy_config)
        matched_left_ids, matched_right_ids = self._select_matched_product_ids(
            fuzzy_matches_pass1, normalized_data
        )

        pass2_candidates = self._filter_pass2_pairs(
            candidate_pairs, normalized_data, matched_left_ids, matched_right_ids
        )
        pass2_candidate_count = 0 if pass2_candidates is None else len(pass2_candidates)
        self.logger.info(f"Pass 2 candidate pairs after strict filters: {pass2_candidate_count}.")

        self.logger.info("Running token-set fuzzy matcher (pass 2).")
        fuzzy_matches_pass2 = run_fuzzy_matching_token_set(
            normalized_data,
            pass2_candidates,
            self.fuzzy_config,
            # Stricter name floor for the second pass to cut lenient matches.
            name_min_score=0.60,
            require_disambiguator=True,
        )

        fuzzy_matches = self._dedupe_pairs(
            pd.concat([fuzzy_matches_pass1, fuzzy_matches_pass2], ignore_index=True)
        )
        fuzzy_breakdown = {
            "pass1_matches": int(len(fuzzy_matches_pass1)),
            "pass2_matches": int(len(fuzzy_matches_pass2)),
            "pass2_candidates": int(pass2_candidate_count),
            "matches": int(len(fuzzy_matches)),
        }
        self.logger.info(f"Fuzzy matcher retained {len(fuzzy_matches)} matches across both passes.")

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
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.ParserError as exc:
            self.logger.warning(
                f"Failed to read {path} with default CSV parser ({exc}). Retrying while skipping bad lines."
            )
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                engine="python",
            )
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
        if "measure_mg_int" not in df.columns:
            df["measure_mg_int"] = None
        if "package_size" not in df.columns:
            df["package_size"] = ""
        for col in [
            "states_norm",
            "country_norm",
            "strain_type_norm",
            "extract_type_norm",
            "strain_or_flavor_norm",
            "product_line_norm",
        ]:
            if col not in df.columns:
                df[col] = ""
        return df

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        s = str(value).strip()
        return s == "" or s.lower() in {"nan", "none", "null"}

    def _clean_str_series(self, series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.strip()
        mask = s.str.lower().isin({"", "nan", "none", "null"})
        return s.mask(mask, "")

    def _normalize_package_value(self, value: Any) -> str:
        if self._is_missing(value):
            return ""
        s = str(value).strip().lower()
        if s == "":
            return ""
        try:
            num = float(s)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except Exception:
            return s

    def _select_matched_product_ids(
        self, pairs_df: Optional[pd.DataFrame], normalized_data: pd.DataFrame
    ) -> tuple[set[str], set[str]]:
        if pairs_df is None or pairs_df.empty:
            return set(), set()
        pairs = pairs_df.copy()
        if {"left_source", "right_source"}.issubset(pairs.columns):
            pairs = pairs[pairs["left_source"] != pairs["right_source"]]
        if pairs.empty:
            return set(), set()

        product_ids = normalized_data.get("product_id")
        if product_ids is None:
            return set(), set()
        pairs["left_product_id"] = product_ids.reindex(pairs["left_index"]).to_numpy()
        pairs["right_product_id"] = product_ids.reindex(pairs["right_index"]).to_numpy()
        pairs = pairs[
            ~pairs["left_product_id"].apply(self._is_missing)
            & ~pairs["right_product_id"].apply(self._is_missing)
        ]
        if pairs.empty:
            return set(), set()

        final_scores = (
            pairs["final_score"]
            if "final_score" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        name_scores = (
            pairs["name_score"]
            if "name_score" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        similarity_scores = (
            pairs["similarity"]
            if "similarity" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        pairs["_final_score_sort"] = pd.to_numeric(final_scores, errors="coerce").fillna(-1.0)
        pairs["_name_score_sort"] = pd.to_numeric(name_scores, errors="coerce").fillna(-1.0)
        pairs["_similarity_sort"] = pd.to_numeric(similarity_scores, errors="coerce").fillna(-1.0)
        pairs.sort_values(
            ["_final_score_sort", "_name_score_sort", "_similarity_sort"], ascending=False, inplace=True
        )

        used_left: set[str] = set()
        used_right: set[str] = set()
        for _, row in pairs.iterrows():
            lpid = str(row.get("left_product_id"))
            rpid = str(row.get("right_product_id"))
            if lpid in used_left or rpid in used_right:
                continue
            used_left.add(lpid)
            used_right.add(rpid)
        return used_left, used_right

    def _filter_pass2_pairs(
        self,
        candidate_pairs: Optional[pd.DataFrame],
        normalized_data: pd.DataFrame,
        matched_left_ids: set[str],
        matched_right_ids: set[str],
    ) -> Optional[pd.DataFrame]:
        if candidate_pairs is None or candidate_pairs.empty:
            return candidate_pairs

        pairs = candidate_pairs.copy()
        product_ids = normalized_data.get("product_id")
        if product_ids is not None and (matched_left_ids or matched_right_ids):
            left_ids = self._align_series_to_pairs(
                self._clean_str_series(product_ids), pairs["left_index"], pairs.index
            )
            right_ids = self._align_series_to_pairs(
                self._clean_str_series(product_ids), pairs["right_index"], pairs.index
            )
            mask = pd.Series([True] * len(pairs), index=pairs.index)
            if matched_left_ids:
                mask &= ~left_ids.isin(matched_left_ids)
            if matched_right_ids:
                mask &= ~right_ids.isin(matched_right_ids)
            pairs = pairs[mask]

        if pairs.empty:
            return pairs

        brand_series = (
            normalized_data["brand_name_norm"]
            if "brand_name_norm" in normalized_data.columns
            else pd.Series([""] * len(normalized_data), index=normalized_data.index)
        )
        left_brand = self._align_series_to_pairs(
            self._clean_str_series(brand_series), pairs["left_index"], pairs.index
        )
        right_brand = self._align_series_to_pairs(
            self._clean_str_series(brand_series), pairs["right_index"], pairs.index
        )
        brand_match = (left_brand != "") & (left_brand == right_brand)

        measure_series = (
            normalized_data["measure_mg_int"]
            if "measure_mg_int" in normalized_data.columns
            else pd.Series([None] * len(normalized_data), index=normalized_data.index)
        )
        require_measure_match = getattr(self.fuzzy_config, "require_measure_match", False)
        left_measure = pd.to_numeric(
            self._align_series_to_pairs(measure_series, pairs["left_index"], pairs.index), errors="coerce"
        )
        right_measure = pd.to_numeric(
            self._align_series_to_pairs(measure_series, pairs["right_index"], pairs.index), errors="coerce"
        )
        if require_measure_match:
            measure_match = left_measure.notna() & right_measure.notna() & (left_measure == right_measure)
        else:
            # Lenient: allow matches when a measure is missing on either side, but still require equality when present.
            measure_match = (left_measure.notna() & right_measure.notna() & (left_measure == right_measure)) | (
                left_measure.isna() | right_measure.isna()
            )

        package_series = (
            normalized_data["package_size"]
            if "package_size" in normalized_data.columns
            else pd.Series([""] * len(normalized_data), index=normalized_data.index)
        )
        left_package = self._align_series_to_pairs(package_series, pairs["left_index"], pairs.index).apply(
            self._normalize_package_value
        )
        right_package = self._align_series_to_pairs(package_series, pairs["right_index"], pairs.index).apply(
            self._normalize_package_value
        )
        package_match = (left_package != "") & (left_package == right_package)

        return pairs[brand_match & measure_match & package_match]

    def _dedupe_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        if pairs_df is None or pairs_df.empty:
            return pairs_df
        pairs = pairs_df.copy()
        final_scores = (
            pairs["final_score"]
            if "final_score" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        name_scores = (
            pairs["name_score"]
            if "name_score" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        similarity_scores = (
            pairs["similarity"]
            if "similarity" in pairs.columns
            else pd.Series([None] * len(pairs), index=pairs.index)
        )
        pairs["_final_score_sort"] = pd.to_numeric(final_scores, errors="coerce").fillna(-1.0)
        pairs["_name_score_sort"] = pd.to_numeric(name_scores, errors="coerce").fillna(-1.0)
        pairs["_similarity_sort"] = pd.to_numeric(similarity_scores, errors="coerce").fillna(-1.0)
        pairs.sort_values(
            ["_final_score_sort", "_name_score_sort", "_similarity_sort"], ascending=False, inplace=True
        )
        pairs = pairs.drop_duplicates(subset=["left_index", "right_index"], keep="first")
        return pairs.drop(columns=["_final_score_sort", "_name_score_sort", "_similarity_sort"], errors="ignore")

    def _align_series_to_pairs(
        self, series: pd.Series, index_values: pd.Series, target_index: pd.Index
    ) -> pd.Series:
        aligned = series.reindex(index_values).reset_index(drop=True)
        aligned.index = target_index
        return aligned

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
            "all_pairs": blocking_strategy_all_pairs,
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
