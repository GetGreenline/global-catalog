"""
Ad-hoc runner to exercise the ingest + normalize + blocking flow for products.

This script loads local snapshots, runs the ProductPipeline through the first two stages,
and evaluates the new blocking strategies. Useful while iterating on blocker logic.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from global_catalog.pipelines.products.product_pipeline import ProductPipeline
from global_catalog.matching.products.blocking_v2 import (
    BlockingConfig,
    blocking_strategy_one,
    blocking_strategy_two,
    blocking_strategy_three,
    blocking_strategy_four,
    blocking_strategy_five,
    build_candidates,
)


@dataclass
class DummyRepo:
    """Minimal repo placeholder to satisfy ProductPipeline."""

    def __init__(self, snapshot_root: str):
        self.snapshot_root = snapshot_root


def _persist_pairs(pairs_df, metrics, output_dir: Path, strategy_name: str, run_id: str, cfg: BlockingConfig):
    if pairs_df is None or pairs_df.empty:
        print(f"No pairs generated for {strategy_name}; skipping save.")
        return None
    strategy_dir = output_dir / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run_id}.parquet"
    out_path = strategy_dir / filename
    pairs_df.to_parquet(out_path, index=False)
    meta = {
        "strategy": strategy_name,
        "run_id": run_id,
        "blocking_config": asdict(cfg),
        "metrics": metrics,
        "pair_file": out_path.name,
    }
    with (out_path.with_suffix(".json")).open("w") as fh:
        json.dump(meta, fh, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Test product blocking strategies.")
    ap.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory containing snapshot CSVs.")
    ap.add_argument(
        "--strategy",
        default="strategy_one",
        choices=[
            "strategy_one",
            "strategy_two",
            "strategy_three",
            "strategy_four",
            "strategy_five",
            "all",
        ],
        help="Which blocking strategy to execute. Use 'all' to run every strategy sequentially.",
    )
    ap.add_argument(
        "--blocking-key-specs",
        default=None,
        help="JSON array of blocking key specs (only used by strategy_two).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard threshold for strategies 2 and 3.",
    )
    ap.add_argument(
        "--measure-tol",
        type=float,
        default=0.0,
        help="Relative tolerance for measure comparisons (e.g., 0.2 = 20%).",
    )
    ap.add_argument(
        "--max-per-left",
        type=int,
        default=50,
        help="Maximum candidate pairs per left record.",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="Window size for sorted-neighborhood (strategy_three).",
    )
    ap.add_argument(
        "--include-description",
        action="store_true",
        help="Include description tokens when blocking.",
    )
    ap.add_argument(
        "--description-token-limit",
        type=int,
        default=25,
        help="Maximum description tokens to include if enabled.",
    )
    ap.add_argument(
        "--strict-measure-only",
        action="store_true",
        help="Run blocking strategy one without the lenient measure pass.",
    )
    ap.add_argument(
        "--pairs-dir",
        default="artifacts/products/pairs",
        help="Directory where generated candidate pairs will be stored as parquet files.",
    )
    ap.add_argument(
        "--skip-save",
        action="store_true",
        help="If provided, do not persist candidate pairs to disk.",
    )
    args = ap.parse_args()

    repo = DummyRepo(snapshot_root=args.snapshot_root)
    pipeline = ProductPipeline(repo=repo, matcher=None, resolver=None, snapshot_root=args.snapshot_root)

    t0 = time.perf_counter()
    raw = pipeline.ingest()
    t_ingest = time.perf_counter()
    normalized = pipeline.normalize(raw)
    t_normalize = time.perf_counter()

    specs = None
    if args.blocking_key_specs:
        specs = json.loads(args.blocking_key_specs)
    cfg = BlockingConfig(
        threshold=args.threshold,
        max_per_left=args.max_per_left,
        include_description=args.include_description,
        description_token_limit=args.description_token_limit,
        enforce_uom=False,
        measure_tol=args.measure_tol,
        blocking_key_specs=specs,
        window_size=args.window_size,
        lenient_measure_pass=(not args.strict_measure_only),
    )
    strategy_map = {
        "strategy_one": blocking_strategy_one,
        "strategy_two": blocking_strategy_two,
        "strategy_three": blocking_strategy_three,
        "strategy_four": blocking_strategy_four,
        "strategy_five": blocking_strategy_five,
    }
    selected = list(strategy_map.keys()) if args.strategy == "all" else [args.strategy]
    output_dir = Path(args.pairs_dir)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for strategy_name in selected:
        strategy_fn = strategy_map[strategy_name]
        print(f"Running blocking strategy: {strategy_name}")
        result = build_candidates(normalized, cfg, strategy_fn)
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

        if not args.skip_save:
            saved_path = _persist_pairs(
                pairs_df=pairs,
                metrics=metrics,
                output_dir=output_dir,
                strategy_name=strategy_name,
                run_id=run_id,
                cfg=cfg,
            )
            if saved_path:
                print(f"Saved candidate pairs to {saved_path}")


if __name__ == "__main__":
    main()
