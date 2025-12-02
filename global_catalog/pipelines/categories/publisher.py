from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from global_catalog.pipelines.entity_pipeline import EntityPipelineContext
from global_catalog.publishers.s3_publisher import mirror_artifacts_to_s3


class CategoriesPublisher:
    #Serialize pipeline artifacts locally and mirror them to S3

    def __init__(self, mirror_fn: Callable[[str, str, str], None] = mirror_artifacts_to_s3):
        self._mirror_fn = mirror_fn

    def __call__(self, context: EntityPipelineContext) -> Optional[Dict[str, Any]]:
        if context is None or context.resolution is None:
            return None
        resolution_payload = context.resolution
        run_metadata = resolution_payload.get("run_metadata", {})
        run_dir_str = run_metadata.get("run_dir")
        if not run_dir_str:
            raise RuntimeError("Run metadata must include run_dir before publishing")
        run_dir = Path(run_dir_str).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        match_results = context.match_results or {}
        outputs = self._write_local_artifacts(run_dir, resolution_payload, match_results)
        metrics_payload = self._build_metrics(run_metadata, match_results, outputs)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        outputs["metrics_local"] = str(metrics_path)

        #NOT MIRRORING FOR NOW
        #self._mirror_artifacts(run_dir, run_metadata)
        return {"outputs": outputs, "run_id": run_metadata.get("run_id"), "metrics": metrics_payload}

    def _write_local_artifacts(
        self,
        run_dir: Path,
        resolution_payload: Dict[str, Any],
        match_results: Dict[str, Any],
    ) -> Dict[str, str]:
        outputs: Dict[str, str] = {"run_dir": str(run_dir)}

        category_global_id_map_df = resolution_payload.get("category_global_id_map")
        staging_parquet = run_dir / "staging_categories_id_mapping.parquet"
        staging_csv = run_dir / "staging_categories_id_mapping.csv"
        if category_global_id_map_df is not None:
            staged_df = self._format_category_global_id_map(category_global_id_map_df)
            staged_df.to_parquet(staging_parquet, index=False)
            staged_df.to_csv(staging_csv, index=False)
        outputs["category_global_id_map_local"] = str(staging_parquet)
        outputs["category_global_id_map_csv_local"] = str(staging_csv)
        outputs["staging_categories_id_mapping_parquet"] = str(staging_parquet)
        outputs["staging_categories_id_mapping_csv"] = str(staging_csv)

        pairs = match_results.get("pairs")
        summary = match_results.get("summary")
        sample = match_results.get("sample")
        pairs_path = run_dir / "pairs.parquet"
        summary_path = run_dir / "summary.parquet"
        sample_path = run_dir / "sample.csv"
        if pairs is not None:
            pairs.to_parquet(pairs_path, index=False)
        if summary is not None:
            summary.to_parquet(summary_path, index=False)
        if sample is not None:
            sample.to_csv(sample_path, index=False)
        outputs["pairs_local"] = str(pairs_path)
        outputs["summary_local"] = str(summary_path)
        outputs["sample_local"] = str(sample_path)

        resolution_df = resolution_payload.get("resolution")
        resolution_parquet = run_dir / "resolution.parquet"
        resolution_csv = run_dir / "resolution.csv"
        if resolution_df is not None:
            resolution_df.to_parquet(resolution_parquet, index=False)
            resolution_df.to_csv(resolution_csv, index=False)
        outputs["resolution_parquet_local"] = str(resolution_parquet)
        outputs["resolution_csv_local"] = str(resolution_csv)

        return outputs

    def _format_category_global_id_map(self, df):
        staged = df.copy()
        if "category_id" in staged.columns:
            staged = staged.rename(columns={"category_id": "external_id"})
        if "load_timestamp" in staged.columns:
            staged = staged.drop(columns=["load_timestamp"])
        return staged

    def _build_metrics(
        self,
        run_metadata: Dict[str, Any],
        match_results: Dict[str, Any],
        outputs: Dict[str, str],
    ) -> Dict[str, Any]:
        metrics = dict(match_results.get("metrics") or {})
        metrics.update(
            {
                "run_id": run_metadata.get("run_id"),
                "inputs": run_metadata.get("inputs"),
                "dedup_metrics": run_metadata.get("dedup_metrics"),
                "outputs": outputs,
                "timing_seconds_total": run_metadata.get("timing_seconds_total"),
            }
        )
        return metrics

    def _mirror_artifacts(self, run_dir: Path, run_metadata: Dict[str, Any]) -> None:
        s3_run_prefix = run_metadata.get("s3_run_prefix")
        s3_latest_prefix = run_metadata.get("s3_latest_prefix")
        if not s3_run_prefix or not s3_latest_prefix:
            return
        self._mirror_fn(str(run_dir), s3_run_prefix, s3_latest_prefix)
