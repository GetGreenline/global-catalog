from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import boto3

import global_catalog.config.settings as settings
from global_catalog.pipelines.categories.resolve_category_pairs import global_category_id_map
from global_catalog.pipelines.entity_pipeline import EntityPipeline


@dataclass
class CategoriesRunConfig:
    date_prefix: str
    sources: list[str]
    local_out_root: str
    s3_run_prefix: Optional[str] = None
    s3_latest_prefix: Optional[str] = None


class CategoryPipeline(EntityPipeline):
    def __init__(self, repo, matcher, resolver, publisher_fn=None):
        super().__init__(repo, matcher, resolver, publisher_fn)
        profile = getattr(settings, "GC_S3_PROFILE", None) or os.getenv("AWS_PROFILE")
        self._session = boto3.Session(profile_name=profile, region_name=settings.AWS_REGION)
        self._run_cfg: Optional[CategoriesRunConfig] = None
        self._run_started_at: Optional[float] = None
        self._matcher_started_at: Optional[float] = None

    def run_categories_pipeline(self, config: CategoriesRunConfig) -> Dict[str, Any]:
        self._run_cfg = config
        self._run_started_at = time.perf_counter()
        try:
            return super().run()
        finally:
            self._run_cfg = None
            self._run_started_at = None
            self._matcher_started_at = None

    def ingest(self) -> Dict[str, Any]:
        cfg = self._require_run_config()
        self._log_identity()
        print(
            f"READ: s3://{self.repo.bucket}/{self.repo.prefix}/<source>/raw/{cfg.date_prefix}/categories.csv"
        )
        df_raw = self.repo.read_categories_raw(sources=cfg.sources, date_prefix=cfg.date_prefix)
        dedup_metrics = {
            "duplicates_removed": 0,
            "input_rows": int(len(df_raw)),
            "unique_rows": int(len(df_raw)),
        }
        print(
            f"READ: df_raw.shape(after_dedupe)={df_raw.shape} removed={dedup_metrics.get('duplicates_removed')}"
        )
        return {
            "df_raw": df_raw,
            "dedup_metrics": dedup_metrics,
            "date_prefix": cfg.date_prefix,
            "sources": cfg.sources,
        }

    def normalize(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        df_raw = raw_data["df_raw"]
        self._matcher_started_at = time.perf_counter()
        df_norm, df_pretty_like = self.matcher.normalize(df_raw)
        return {"df_norm": df_norm, "df_pretty_like": df_pretty_like}

    def match(self, normalized_data: Dict[str, Any], raw_data: Dict[str, Any]) -> Dict[str, Any]:
        df_norm = normalized_data["df_norm"]
        df_pretty_like = normalized_data["df_pretty_like"]

        pairs = self.matcher.generate_pairs(df_norm)
        pairs = self.matcher.filter_pairs(pairs, df_norm)
        pairs, summary = self.matcher.summarize(pairs, df_pretty_like)
        if pairs.empty:
            sample = pairs
        else:
            sample = (
                pairs.sort_values(
                    ["match_scope", "match_type", "similarity"],
                    ascending=[True, True, False],
                ).head(self.matcher.cfg.sample_limit)
            )
        metrics = self.matcher.metrics(
            df_norm,
            pairs,
            self._matcher_started_at or time.perf_counter(),
        )
        print(
            f"MATCH: pairs.shape={getattr(pairs, 'shape', None)} summary.shape={getattr(summary, 'shape', None)}"
        )
        return {
            "pairs": pairs,
            "summary": summary,
            "sample": sample,
            "metrics": metrics,
        }

    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = self._require_run_config()
        df_raw = raw_data["df_raw"]
        pairs = match_results.get("pairs")
        print("RESOLVE: start")
        resolution = self.resolver.resolve(pairs, df_raw)
        category_global_id_map_df = global_category_id_map(df_raw, resolution)
        run_id = f"{cfg.date_prefix}_{uuid.uuid4().hex[:8]}"
        run_dir = Path(cfg.local_out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        total_seconds = round(time.perf_counter() - (self._run_started_at or time.perf_counter()), 3)
        print(
            f"RESOLVE: resolution.shape={getattr(resolution, 'shape', None)} run_dir={run_dir}"
        )
        s3_run_prefix = (
            f"{cfg.s3_run_prefix.rstrip('/')}/{run_id}"
            if cfg.s3_run_prefix
            else None
        )
        s3_latest_prefix = cfg.s3_latest_prefix.rstrip('/') if cfg.s3_latest_prefix else None
        metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "timing_seconds_total": total_seconds,
            "inputs": {
                "bucket": self.repo.bucket,
                "prefix": self.repo.prefix,
                "sources": cfg.sources,
                "date_prefix": cfg.date_prefix,
            },
            "dedup_metrics": raw_data.get("dedup_metrics", {}),
            "s3_run_prefix": s3_run_prefix,
            "s3_latest_prefix": s3_latest_prefix,
        }
        return {
            "resolution": resolution,
            "category_global_id_map": category_global_id_map_df,
            "run_metadata": metadata,
        }

    def _require_run_config(self) -> CategoriesRunConfig:
        if self._run_cfg is None:
            raise RuntimeError("Categories run configuration must be set before running the pipeline")
        return self._run_cfg

    def _log_identity(self) -> None:
        sts = self._session.client("sts")
        ident = sts.get_caller_identity()
        print(f"AWS identity: account={ident['Account']} arn={ident['Arn']}")
