import argparse
import time

from global_catalog.matching.products.blocking_v2 import BlockingConfig
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_pipeline import ProductPipeline
from global_catalog.pipelines.products.product_resolver import ProductMatchResolver


def parse_args():
    parser = argparse.ArgumentParser(description="Run product matching pipeline on local snapshots.")
    parser.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory with product CSV snapshots.")
    parser.add_argument("--out-root", default="artifacts/products", help="Artifacts output directory.")
    parser.add_argument("--block-threshold", type=float, default=0.3, help="Blocking Jaccard threshold.")
    parser.add_argument("--match-threshold", type=float, default=0.75, help="Fuzzy matcher threshold.")
    parser.add_argument("--max-per-left", type=int, default=200, help="Max candidates per left record.")
    parser.add_argument("--description-token-limit", type=int, default=25, help="Max description tokens if enabled.")
    parser.add_argument("--strict-measure-only", action="store_true", help="Skip lenient measure blocking pass.")
    parser.add_argument(
        "--blocking-strategy",
        default="strategy_one",
        choices=["strategy_one", "strategy_two", "strategy_three", "strategy_four", "strategy_five"],
        help="Which blocking strategy to execute.",
    )
    parser.add_argument(
        "--use-local-pairs",
        action="store_true",
        help="Load candidate pairs from local artifacts/products/pairs instead of rerunning blocking.",
    )
    parser.add_argument(
        "--pairs-dir",
        default="artifacts/products/pairs",
        help="Directory to read/write cached candidate pair parquet files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    blocking_cfg = BlockingConfig(
        threshold=args.block_threshold,
        max_per_left=args.max_per_left,
        include_description=False,
        description_token_limit=args.description_token_limit,
        lenient_measure_pass=(not args.strict_measure_only),
        strategy_name=args.blocking_strategy,
        use_local_pairs=args.use_local_pairs,
        pairs_dir=args.pairs_dir,
    )
    fuzzy_cfg = FuzzyMatcherConfig(
        threshold=args.match_threshold,
    )

    run_label = "strict" if args.strict_measure_only else "lenient"
    pipeline = ProductPipeline(
        repo=object(),
        snapshot_root=args.snapshot_root,
        blocking_config=blocking_cfg,
        fuzzy_config=fuzzy_cfg,
        resolver=ProductMatchResolver(out_root=args.out_root, run_label=run_label),
        pairs_cache_dir=args.pairs_dir,
        local_run=args.use_local_pairs,
    )

    t0 = time.perf_counter()
    result = pipeline.run()
    total = round(time.perf_counter() - t0, 3)
    print(f"[run_products_matches] Completed run in {total}s")
    print(f"[run_products_matches] Artifacts: {result.get('resolution', {}).get('run_dir')}")


if __name__ == "__main__":
    main()
