import argparse
import time

from global_catalog.matching.products.blocking_v2 import BlockingConfig
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_pipeline import ProductPipeline
from global_catalog.pipelines.products.product_resolver import ProductMatchResolver
from global_catalog.transformers.products.products_normalization import (
    ProductNormalizer,
    EnrichedProductNormalizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run product matching pipeline on local snapshots.")
    parser.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory with product CSV snapshots.")
    parser.add_argument(
        "--combined",
        default=None,
        help="Single CSV/JSON/JSONL containing both sources with a 'source' column.",
    )
    parser.add_argument("--weedmaps", default=None, help="Override weedmaps snapshot path (CSV/JSON/JSONL).")
    parser.add_argument("--hoodie", default=None, help="Override hoodie snapshot path (CSV/JSON/JSONL).")
    parser.add_argument("--out-root", default="artifacts/products", help="Artifacts output directory.")
    parser.add_argument("--block-threshold", type=float, default=0.3, help="Blocking Jaccard threshold.")
    parser.add_argument("--match-threshold", type=float, default=0.75, help="Fuzzy matcher threshold.")
    parser.add_argument("--max-per-left", type=int, default=200, help="Max candidates per left record.")
    parser.add_argument("--description-token-limit", type=int, default=25, help="Max description tokens if enabled.")
    parser.add_argument("--strict-measure-only", action="store_true", help="Skip lenient measure blocking pass.")
    parser.add_argument(
        "--require-measure-match",
        action="store_true",
        help="Require measure_mg to be present on both sides and equal; defaults to lenient.",
    )
    parser.add_argument(
        "--schema",
        default="normal",
        choices=["normal", "enriched"],
        help="Normalization schema to use.",
    )
    parser.add_argument(
        "--blocking-strategy",
        default="strategy_one",
        choices=[
            "strategy_one",
            "strategy_two",
            "strategy_three",
            "strategy_four",
            "strategy_five",
            "all_pairs",
        ],
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
    parser.add_argument(
        "--skip-candidate-pairs",
        action="store_true",
        help="Skip writing candidate_pairs.parquet to reduce memory usage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_files = None
    combined_path = None
    if args.combined:
        combined_path = args.combined
    else:
        if args.hoodie or args.weedmaps:
            source_files = {
                "weedmaps": args.weedmaps or "products_weedmaps.csv",
                "hoodie": args.hoodie or "products_hoodie.csv",
            }
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
        require_measure_match=(args.require_measure_match or args.strict_measure_only),
    )

    run_label = "strict" if args.strict_measure_only else "lenient"
    normalizer = EnrichedProductNormalizer() if args.schema == "enriched" else ProductNormalizer()
    pipeline = ProductPipeline(
        repo=object(),
        snapshot_root=args.snapshot_root,
        source_files=source_files,
        combined_source_path=combined_path,
        normalizer=normalizer,
        blocking_config=blocking_cfg,
        fuzzy_config=fuzzy_cfg,
        resolver=ProductMatchResolver(
            out_root=args.out_root,
            run_label=run_label,
            skip_candidate_pairs=args.skip_candidate_pairs,
        ),
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
