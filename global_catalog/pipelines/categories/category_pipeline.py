from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import boto3
import pandas as pd

import global_catalog.config.settings as settings
from global_catalog.pipelines.categories.resolve_category_pairs import global_category_id_map
from global_catalog.pipelines.entity_pipeline import EntityPipeline
from global_catalog.repositories.redshift_repo import RedShiftRepo


@dataclass
class CategoriesRunConfig:
    date_prefix: Optional[str]
    sources: list[str]
    local_out_root: str
    ingest_source: Literal["s3", "redshift", "csv"] = "s3"
    redshift_sql_path: Optional[str] = None
    redshift_left_sql_path: Optional[str] = None
    redshift_right_sql_paths: Optional[list[str]] = None
    use_snapshot: bool = False
    snapshot_csv: Optional[str] = None
    csv_path: Optional[str] = None
    s3_run_prefix: Optional[str] = None
    s3_latest_prefix: Optional[str] = None


class CategoryPipeline(EntityPipeline):
    def __init__(self, repo, matcher, resolver, publisher_fn=None, match_publisher_fn=None):
        super().__init__(repo, matcher, resolver, publisher_fn)
        profile = getattr(settings, "GC_S3_PROFILE", None) or os.getenv("AWS_PROFILE")
        self._session = boto3.Session(profile_name=profile, region_name=settings.AWS_REGION)
        self._run_cfg: Optional[CategoriesRunConfig] = None
        self._run_started_at: Optional[float] = None
        self._matcher_started_at: Optional[float] = None
        self.match_publisher_fn = match_publisher_fn

    def run_categories_pipeline(self, config: CategoriesRunConfig) -> Dict[str, Any]:
        self._run_cfg = config
        self._run_started_at = time.perf_counter()
        try:
            payload = self.run_match_stage()
            ctx = self._context
            self.logger.info("PIPELINE: resolve")
            ctx.resolution = resolution = self.resolve(
                payload["match_results"],
                payload["raw_data"],
                payload["normalized_data"],
            )
            ctx.publish_result = self.publish(ctx)
            self._last_run = {
                "raw_data": payload["raw_data"],
                "normalized": payload["normalized_data"],
                "match_results": payload["match_results"],
                "run_metadata": payload["run_metadata"],
                "match_publish_result": payload.get("match_publish_result"),
                "resolution": resolution,
                "publish_result": ctx.publish_result,
            }
            self.logger.info("PIPELINE: done")
            return self._last_run
        finally:
            self._run_cfg = None
            self._run_started_at = None
            self._matcher_started_at = None

    def run_match_stage(self, config: Optional[CategoriesRunConfig] = None) -> Dict[str, Any]:
        owns_config = False
        if config is not None:
            self._run_cfg = config
            owns_config = True
        if self._run_started_at is None:
            self._run_started_at = time.perf_counter()
        try:
            self.reset_context()
            ctx = self._context
            self.logger.info("PIPELINE: ingest")
            ctx.raw_data = raw_data = self.ingest()
            run_metadata = self._build_run_metadata(raw_data.get("dedup_metrics", {}))
            raw_data["run_metadata"] = run_metadata
            raw_data["date_prefix"] = run_metadata.get("inputs", {}).get("date_prefix")

            self.logger.info("PIPELINE: normalize")
            ctx.normalized = normalized = self.normalize(raw_data)
            self.logger.info("PIPELINE: match")
            ctx.match_results = match_results = self.match(normalized, raw_data)
            run_metadata["timing_seconds_total"] = round(
                time.perf_counter() - (self._run_started_at or time.perf_counter()), 3
            )
            match_publish_result = self.publish_match(ctx)
            payload = {
                "raw_data": raw_data,
                "normalized_data": normalized,
                "match_results": match_results,
                "run_metadata": run_metadata,
                "match_publish_result": match_publish_result,
            }
            return payload
        finally:
            if owns_config:
                self._run_cfg = None
                self._run_started_at = None

    def publish_match(self, context: Any) -> Optional[Any]:
        if not callable(self.match_publisher_fn):
            return None
        return self.match_publisher_fn(context=context)

    def _build_run_metadata(self, dedup_metrics: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._require_run_config()
        date_prefix = cfg.date_prefix or pd.Timestamp.utcnow().strftime("%Y%m%d")
        run_id = f"{date_prefix}_{uuid.uuid4().hex[:8]}"
        run_dir = Path(cfg.local_out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        s3_run_prefix = (
            f"{cfg.s3_run_prefix.rstrip('/')}/{run_id}"
            if cfg.s3_run_prefix
            else None
        )
        s3_latest_prefix = cfg.s3_latest_prefix.rstrip('/') if cfg.s3_latest_prefix else None
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "timing_seconds_total": 0.0,
            "inputs": {
                "bucket": getattr(self.repo, "bucket", None),
                "prefix": getattr(self.repo, "prefix", None),
                "sources": cfg.sources,
                "date_prefix": date_prefix,
                "ingest_source": cfg.ingest_source,
            },
            "dedup_metrics": dedup_metrics or {},
            "s3_run_prefix": s3_run_prefix,
            "s3_latest_prefix": s3_latest_prefix,
        }

    def ingest(self) -> Dict[str, Any]:
        cfg = self._require_run_config()
        self._log_identity()
        if cfg.ingest_source == "s3":
            if not cfg.date_prefix:
                raise ValueError("date_prefix is required when ingest_source='s3'")
            print(
                f"READ: s3://{self.repo.bucket}/{self.repo.prefix}/<source>/raw/{cfg.date_prefix}/categories.csv"
            )
            df_raw = self.repo.read_categories_raw(sources=cfg.sources, date_prefix=cfg.date_prefix)
        elif cfg.ingest_source == "redshift":
            rs = RedShiftRepo()
            if cfg.use_snapshot and cfg.snapshot_csv:
                snapshot_path = Path(cfg.snapshot_csv)
                if snapshot_path.exists():
                    df_raw = pd.read_csv(snapshot_path)
                    print(f"READ: redshift snapshot rows={len(df_raw)} path={snapshot_path}")
                else:
                    df_raw = None
            else:
                df_raw = None

            if df_raw is None:
                if cfg.redshift_left_sql_path or cfg.redshift_right_sql_paths:
                    if not cfg.redshift_left_sql_path:
                        raise ValueError("redshift_left_sql_path is required when using separate left/right ingestion")
                    left_path = Path(cfg.redshift_left_sql_path)
                    left_sql = left_path.read_text()
                    if not left_sql.strip():
                        raise ValueError(f"redshift_left_sql_path is empty: {left_path}")
                    left_df = rs.read_sql(left_sql)
                    left_df = left_df.copy()
                    left_df["match_side"] = "left"

                    right_paths = cfg.redshift_right_sql_paths or []
                    if not right_paths:
                        raise ValueError("redshift_right_sql_paths must include at least one path for right ingestion")
                    right_frames = []
                    for right_sql_path in right_paths:
                        rpath = Path(right_sql_path)
                        rsql = rpath.read_text()
                        if not rsql.strip():
                            raise ValueError(f"redshift_right_sql_paths entry is empty: {rpath}")
                        rdf = rs.read_sql(rsql)
                        rdf = rdf.copy()
                        rdf["match_side"] = "right"
                        right_frames.append(rdf)

                    df_raw = pd.concat([left_df] + right_frames, ignore_index=True)
                    print(
                        f"READ: redshift left_rows={len(left_df)} right_rows={sum(len(r) for r in right_frames)} "
                        f"total_rows={len(df_raw)} left_sql={left_path}"
                    )
                else:
                    if not cfg.redshift_sql_path:
                        raise ValueError("redshift_sql_path is required when ingest_source='redshift'")
                    sql_path = Path(cfg.redshift_sql_path)
                    sql = sql_path.read_text()
                    if not sql.strip():
                        raise ValueError(f"redshift_sql_path is empty: {sql_path}")
                    df_raw = rs.read_sql(sql)
                    print(f"READ: redshift rows={len(df_raw)} sql_path={sql_path}")

                if cfg.use_snapshot and cfg.snapshot_csv:
                    snapshot_path = Path(cfg.snapshot_csv)
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    df_raw.to_csv(snapshot_path, index=False)
                    print(f"READ: redshift snapshot saved path={snapshot_path}")
        elif cfg.ingest_source == "csv":
            if not cfg.csv_path:
                raise ValueError("csv_path is required when ingest_source='csv'")
            df_raw = pd.read_csv(cfg.csv_path)
            print(f"READ: csv rows={len(df_raw)} path={cfg.csv_path}")
        else:
            raise ValueError(f"Unsupported ingest_source: {cfg.ingest_source}")
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
        if not pairs.empty and "source_raw" in df_norm.columns:
            source_map = (
                df_norm[["id", "source_raw"]]
                .drop_duplicates(subset=["id"])
                .set_index("id")["source_raw"]
            )
            pairs = pairs.copy()
            pairs["left_source_raw"] = pairs["left_id"].map(source_map)
            pairs["right_source_raw"] = pairs["right_id"].map(source_map)
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
        run_metadata = raw_data.get("run_metadata") or {}
        run_id = run_metadata.get("run_id")
        run_dir_str = run_metadata.get("run_dir")
        if run_id and run_dir_str:
            run_dir = Path(run_dir_str)
            run_dir.mkdir(parents=True, exist_ok=True)
            date_prefix = run_metadata.get("inputs", {}).get("date_prefix")
        else:
            date_prefix = cfg.date_prefix or pd.Timestamp.utcnow().strftime("%Y%m%d")
            run_id = f"{date_prefix}_{uuid.uuid4().hex[:8]}"
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
        metadata = dict(run_metadata) if run_metadata else {}
        metadata.update(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "timing_seconds_total": total_seconds,
                "inputs": {
                    "bucket": getattr(self.repo, "bucket", None),
                    "prefix": getattr(self.repo, "prefix", None),
                    "sources": cfg.sources,
                    "date_prefix": date_prefix,
                    "ingest_source": cfg.ingest_source,
                },
                "dedup_metrics": raw_data.get("dedup_metrics", {}),
                "s3_run_prefix": s3_run_prefix,
                "s3_latest_prefix": s3_latest_prefix,
            }
        )
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
