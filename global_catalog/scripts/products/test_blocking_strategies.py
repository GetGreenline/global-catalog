"""
Ad-hoc runner to exercise the ingest + normalize + blocking flow for products.

This script loads local snapshots, runs the ProductPipeline through the first two stages,
and evaluates the new blocking strategies. Useful while iterating on blocker logic.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from global_catalog.pipelines.products.product_pipeline import ProductPipeline
from global_catalog.matching.products.blocking_v2 import (
    BlockingConfig,
    blocking_strategy_one,
    build_candidates,
)


@dataclass
class DummyRepo:
    """Minimal repo placeholder to satisfy ProductPipeline."""

    def __init__(self, snapshot_root: str):
        self.snapshot_root = snapshot_root


def main():
    ap = argparse.ArgumentParser(description="Test product blocking strategies.")
    ap.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory containing snapshot CSVs.")
    args = ap.parse_args()

    repo = DummyRepo(snapshot_root=args.snapshot_root)
    pipeline = ProductPipeline(repo=repo, matcher=None, resolver=None, snapshot_root=args.snapshot_root)

    t0 = time.perf_counter()
    raw = pipeline.ingest()
    t_ingest = time.perf_counter()
    normalized = pipeline.normalize(raw)
    t_normalize = time.perf_counter()

    cfg = BlockingConfig()
    result = build_candidates(normalized, cfg, blocking_strategy_one)
    t_block = time.perf_counter()
    pairs = result["pairs"]
    metrics = result["metrics"]

    timing = {
        "ingest_seconds": round(t_ingest - t0, 3),
        "normalize_seconds": round(t_normalize - t_ingest, 3),
        "blocking_seconds": round(t_block - t_normalize, 3),
        "total_seconds": round(t_block - t0, 3),
    }

    print("Blocking metrics:", metrics)
    print("Timing:", timing)
    print(f"Total candidate pairs: {metrics['candidate_pairs']}")
    print(f"Sample pairs:\n{pairs.head()}")


if __name__ == "__main__":
    main()
