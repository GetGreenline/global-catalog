from __future__ import annotations

import argparse
import re
from datetime import datetime

from global_catalog.config import settings
from global_catalog.matching.categories.matcher import CategoryMatchConfig, CategoriesMatcher
from global_catalog.pipelines.categories.resolve_category_pairs import build_resolution_from_pairs
from global_catalog.pipelines.categories.category_pipeline import (
    CategoriesRunConfig,
    CategoryPipeline,
)
from global_catalog.pipelines.categories.publisher import CategoriesPublisher
from global_catalog.repositories.s3_repo import S3Repo


class CategoryResolver:
    def resolve(self, pairs_df, df_raw):
        return build_resolution_from_pairs(pairs_df, df_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the categories entity pipeline from S3 sources")
    parser.add_argument("--date-prefix", required=True, help="YYYYMMDD partition to ingest (e.g. 20251031)")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of sources to ingest (e.g. --sources weedmaps hoodie)",
    )
    parser.add_argument(
        "--out-root",
        default=settings.GC_OUT_ROOT,
        help="Local directory where run artifacts land",
    )
    parser.add_argument(
        "--s3-run-prefix",
        default=None,
        help="Base s3://bucket/prefix for versioned run uploads",
    )
    parser.add_argument(
        "--s3-latest-prefix",
        default=None,
        help="Base s3://bucket/prefix for mirroring the latest run",
    )
    return parser.parse_args()


def normalize_date_prefix(date_prefix: str) -> str:

    stripped = re.sub(r"[^0-9]", "", date_prefix or "")
    if len(stripped) != 8:
        raise ValueError(f"date_prefix must resolve to YYYYMMDD; got '{date_prefix}'")

    datetime.strptime(stripped, "%Y%m%d")
    return stripped


def main():
    args = parse_args()
    normalized_prefix = normalize_date_prefix(args.date_prefix)

    repo = S3Repo(
        bucket=settings.GC_S3_BUCKET,
        prefix=settings.GC_S3_PREFIX,
        profile=settings.GC_S3_PROFILE,
        region=settings.AWS_REGION,
    )

    cfg = CategoryMatchConfig(
        tfidf_threshold=settings.GC_TFIDF_THRESHOLD,
        block_by=settings.GC_BLOCK_BY,
        synonyms_path="global_catalog/normalization/rules/categories.synonyms.yml",
    )
    matcher = CategoriesMatcher(cfg)
    resolver = CategoryResolver()

    publisher = CategoriesPublisher()
    pipe = CategoryPipeline(
        repo=repo,
        matcher=matcher,
        resolver=resolver,
        publisher_fn=publisher,
    )

    # TODO(categories-s3): wire default S3 prefixes once uploads should resume.
    cfg = CategoriesRunConfig(
        date_prefix=normalized_prefix,
        sources=args.sources,
        local_out_root=args.out_root,
        s3_run_prefix=args.s3_run_prefix,
        s3_latest_prefix=args.s3_latest_prefix,
    )

    result = pipe.run_categories_pipeline(cfg)

    publish_outputs = (result.get("publish_result") or {}).get("outputs") or {}
    if publish_outputs:
        run_dir = publish_outputs.get("run_dir")
        if run_dir:
            print(f"Artifacts directory: {run_dir}")
        else:
            print("Artifacts written; inspect metrics for file details.")

if __name__ == "__main__":
    main()
