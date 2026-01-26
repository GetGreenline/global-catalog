from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_catalog.config import settings
from global_catalog.matching.categories.matcher import CategoryMatchConfig, CategoriesMatcher
from global_catalog.pipelines.categories.category_pipeline import (
    CategoriesRunConfig,
    CategoryPipeline,
)
from global_catalog.pipelines.categories.match_publisher import CategoriesMatchPublisher
from global_catalog.pipelines.categories.publisher import CategoriesPublisher
from global_catalog.repositories.s3_repo import S3Repo


class NoopResolver:
    def resolve(self, pairs_df, df_raw):
        return None


class NullRepo:
    bucket = None
    prefix = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run categories ingest+normalize+match (and optional resolve)")
    parser.add_argument(
        "--ingest-source",
        choices=["s3", "redshift", "csv"],
        default="s3",
        help="Ingestion source for categories data",
    )
    parser.add_argument("--date-prefix", default=None, help="YYYYMMDD partition (required for s3)")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[],
        help="List of sources to ingest (s3) or tag in metadata",
    )
    parser.add_argument("--out-root", default=settings.GC_OUT_ROOT, help="Local output directory")
    parser.add_argument(
        "--redshift-sql",
        default=None,
        help="Single SQL file to ingest from Redshift",
    )
    parser.add_argument(
        "--redshift-left-sql",
        default=None,
        help="Left-side Redshift SQL file (match_side=left)",
    )
    parser.add_argument(
        "--redshift-right-sql",
        nargs="+",
        default=None,
        help="One or more right-side Redshift SQL files (match_side=right)",
    )
    parser.add_argument("--use-snapshot", action="store_true", default=False)
    parser.add_argument("--snapshot-csv", default=None, help="Local CSV snapshot path")
    parser.add_argument("--csv-path", default=None, help="CSV file to ingest directly")
    parser.add_argument("--resolve", action="store_true", default=False, help="Run resolve after matching")
    return parser.parse_args()


def normalize_date_prefix(date_prefix: str | None) -> str | None:
    if not date_prefix:
        return None
    stripped = re.sub(r"[^0-9]", "", date_prefix or "")
    if len(stripped) != 8:
        raise ValueError(f"date_prefix must resolve to YYYYMMDD; got '{date_prefix}'")
    datetime.strptime(stripped, "%Y%m%d")
    return stripped


def main() -> None:
    args = parse_args()
    ingest_source = args.ingest_source
    normalized_prefix = normalize_date_prefix(args.date_prefix)

    if ingest_source == "s3":
        if not normalized_prefix:
            raise ValueError("--date-prefix is required for s3 ingestion")
        if not args.sources:
            raise ValueError("--sources is required for s3 ingestion")
        repo = S3Repo(
            bucket=settings.GC_S3_BUCKET,
            prefix=settings.GC_S3_PREFIX,
            profile=settings.GC_S3_PROFILE,
            region=settings.AWS_REGION,
        )
    else:
        repo = NullRepo()

    cfg = CategoryMatchConfig(
        tfidf_threshold=settings.GC_TFIDF_THRESHOLD,
        block_by=settings.GC_BLOCK_BY,
        synonyms_path="global_catalog/normalization/rules/categories.synonyms.yml",
    )
    matcher = CategoriesMatcher(cfg)
    resolver = NoopResolver()

    match_publisher = CategoriesMatchPublisher()
    resolution_publisher = CategoriesPublisher() if args.resolve else None
    pipe = CategoryPipeline(
        repo=repo,
        matcher=matcher,
        resolver=resolver,
        publisher_fn=resolution_publisher,
        match_publisher_fn=match_publisher,
    )

    run_cfg = CategoriesRunConfig(
        date_prefix=normalized_prefix,
        sources=args.sources,
        local_out_root=args.out_root,
        ingest_source=ingest_source,
        redshift_sql_path=args.redshift_sql,
        redshift_left_sql_path=args.redshift_left_sql,
        redshift_right_sql_paths=args.redshift_right_sql,
        use_snapshot=args.use_snapshot,
        snapshot_csv=args.snapshot_csv,
        csv_path=args.csv_path,
    )

    if args.resolve:
        result = pipe.run_categories_pipeline(run_cfg)
        publish_outputs = (result.get("publish_result") or {}).get("outputs") or {}
        if publish_outputs:
            run_dir = publish_outputs.get("run_dir")
            if run_dir:
                print(f"Artifacts directory: {run_dir}")
            else:
                print("Artifacts written; inspect metrics for file details.")
    else:
        result = pipe.run_match_stage(run_cfg)
        publish_outputs = (result.get("match_publish_result") or {}).get("outputs") or {}
        if publish_outputs:
            run_dir = publish_outputs.get("run_dir")
            if run_dir:
                print(f"Match artifacts directory: {run_dir}")
            else:
                print("Match artifacts written; inspect run_metadata for file details.")


if __name__ == "__main__":
    main()
