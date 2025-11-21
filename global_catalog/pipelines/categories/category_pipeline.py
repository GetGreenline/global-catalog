import os
import time
import uuid
import json
from pathlib import Path
import boto3

import global_catalog.config.settings as settings
from global_catalog.pipelines.categories.resolve_category_pairs import global_category_id_map
from global_catalog.pipelines.entity_pipeline import EntityPipeline


class CategoryPipeline(EntityPipeline):
    def __init__(self, repo, matcher, resolver, publisher_fn=None):
        super().__init__(repo, matcher, resolver, publisher_fn)
        profile = getattr(settings, "GC_S3_PROFILE", None) or os.getenv("AWS_PROFILE")
        self._session = boto3.Session(profile_name=profile, region_name=settings.AWS_REGION)

    def _whoami(self):
        sts = self._session.client("sts")
        ident = sts.get_caller_identity()
        print(f"AWS identity: account={ident['Account']} arn={ident['Arn']}")

    def _upload_file_to_s3(self, local_path: Path, bucket: str, key: str) -> None:
        s3 = self._session.client("s3")
        s3.upload_file(str(local_path), bucket, key)

    def run_categories_from_s3(self, date_prefix: str, sources: list[str], local_out_root: str, s3_out_base: str, bucket: str) -> dict:
        t0 = time.perf_counter()
        print(f"START run_categories_from_s3 date_prefix={date_prefix} sources={sources}")
        self._whoami()

        print(f"READ: s3://{self.repo.bucket}/{self.repo.prefix}/<source>/raw/{date_prefix}/categories.csv")
        df_raw = self.repo.read_categories_raw(sources=sources, date_prefix=date_prefix)
        dedup_metrics = {
            "duplicates_removed": 0,
            "input_rows": int(len(df_raw)),
            "unique_rows": int(len(df_raw)),
        }
        print(f"READ: df_raw.shape(after_dedupe)={df_raw.shape} removed={dedup_metrics.get('duplicates_removed')}")

        print("MATCH: start")
        print(" entity pipelinedf_aw columns:", list(df_raw.columns))

        match_out = self.matcher.run(df_raw)
        pairs = match_out.get("pairs")
        summary = match_out.get("summary")
        sample = match_out.get("sample")
        metrics = match_out.get("metrics", {})
        print(f"MATCH: pairs.shape={getattr(pairs, 'shape', None)} summary.shape={getattr(summary, 'shape', None)}")

        print("RESOLVE: start")
        resolution = self.resolver.resolve(pairs, df_raw)
        print(f"RESOLVE: resolution.shape={getattr(resolution, 'shape', None)}")

        run_id = f"{date_prefix}_{uuid.uuid4().hex[:8]}"
        run_dir = Path(local_out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"WRITE: local run_dir={run_dir}")

        category_global_id_map_df = global_category_id_map(df_raw, resolution)
        map_parquet = run_dir / "category_global_id_map.parquet"
        map_csv = run_dir / "category_global_id_map.csv"
        category_global_id_map_df.to_parquet(map_parquet, index=False)
        category_global_id_map_df.to_csv(map_csv, index=False)
        print("WRITE: category_global_id_map.{parquet,csv}")

        pairs_path = run_dir / "pairs.parquet"
        summary_path = run_dir / "summary.parquet"
        sample_path = run_dir / "sample.csv"
        resolution_parquet = run_dir / "resolution.parquet"
        resolution_csv = run_dir / "resolution.csv"
        metrics_path = run_dir / "metrics.json"
        pairs.to_parquet(pairs_path, index=False)
        summary.to_parquet(summary_path, index=False)
        sample.to_csv(sample_path, index=False)
        resolution.to_parquet(resolution_parquet, index=False)
        resolution.to_csv(resolution_csv, index=False)
        print("WRITE: pairs/summary/sample/resolution")

        full_metrics = {
            **metrics,
            "run_id": run_id,
            "inputs": {"bucket": self.repo.bucket, "prefix": self.repo.prefix, "sources": sources, "date_prefix": date_prefix},
            "outputs": {
                "category_global_id_map_local": str(map_parquet),
                "category_global_id_map_csv_local": str(map_csv),
                "pairs_local": str(pairs_path),
                "summary_local": str(summary_path),
                "sample_local": str(sample_path),
                "resolution_parquet_local": str(resolution_parquet),
                "resolution_csv_local": str(resolution_csv),
                "run_dir": str(run_dir),
            },
            "timing_seconds_total": round(time.perf_counter() - t0, 3),
        }
        metrics_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
        print(f"DONE run_id={run_id} total_seconds={full_metrics['timing_seconds_total']}")
        return {"run_id": run_id, "run_dir": str(run_dir), **match_out, "resolution": resolution, "metrics": full_metrics}


    def ingest(self):

