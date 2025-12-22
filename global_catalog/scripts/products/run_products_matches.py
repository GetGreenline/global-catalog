import argparse
import time

from global_catalog.matching.products.blocking_v2 import BlockingConfig
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_pipeline import ProductPipeline
from global_catalog.pipelines.products.product_resolver import ProductMatchResolver
from global_catalog.matching.products.transformer_matcher import TransformerMatcher, TransformerMatcherConfig
from global_catalog.matching.products.cross_encoder_matcher import CrossEncoderMatcher, CrossEncoderMatcherConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run product matching pipeline on local snapshots.")
    parser.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory with product CSV snapshots.")
    parser.add_argument("--out-root", default="artifacts/products", help="Artifacts output directory.")
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Write candidate_pairs.parquet (slow; skipped by default).",
    )
    parser.add_argument(
        "--skip-candidate-context",
        action="store_true",
        help="Skip attaching context columns to candidate pairs (faster).",
    )
    parser.add_argument(
        "--write-resolved-csv",
        action="store_true",
        help="Write resolved_pairs.csv (slow; skipped by default).",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=100000,
        help="Max rows for resolved CSV when --write-resolved-csv is set (0 disables limit).",
    )
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
    parser.add_argument(
        "--matcher",
        default="fuzzy",
        choices=["fuzzy", "transformer", "cross_encoder"],
        help="Which matcher to use after blocking.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformer model to use when --matcher=transformer.",
    )
    parser.add_argument(
        "--transformer-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for transformer matcher.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=64,
        help="Batch size for encoding text in the transformer matcher.",
    )
    parser.add_argument(
        "--transformer-chunk-size",
        type=int,
        default=10000,
        help="Number of candidate pairs to score per chunk for the transformer matcher.",
    )
    parser.add_argument(
        "--transformer-chunk-output-dir",
        default=None,
        help="Optional output directory for transformer chunk outputs; defaults to run_dir/chunks.",
    )
    parser.add_argument(
        "--max-desc-tokens",
        type=int,
        default=128,
        help="Maximum number of description tokens to include in transformer inputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for transformer inference (e.g., 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--cross-encoder-path",
        default="artifacts/products/datasets/artifacts/products/models/cross_encoder_v1",
        help="Local path to a sentence-transformers cross-encoder model.",
    )
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=32,
        help="Batch size for cross-encoder scoring.",
    )
    parser.add_argument(
        "--cross-encoder-chunk-size",
        type=int,
        default=10000,
        help="Number of candidate pairs to score per chunk for the cross-encoder.",
    )
    parser.add_argument(
        "--cross-encoder-max-length",
        type=int,
        default=256,
        help="Max sequence length for the cross-encoder tokenizer.",
    )
    parser.add_argument(
        "--cross-encoder-threshold",
        type=float,
        default=0.0,
        help="Score threshold for cross-encoder matches (use a negative value to disable).",
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

    matcher = None
    if args.matcher == "transformer":
        transformer_cfg = TransformerMatcherConfig(
            model_name=args.model_name,
            batch_size=args.transformer_batch_size,
            device=args.device,
            threshold=args.transformer_threshold,
            max_desc_tokens=args.max_desc_tokens,
            pair_chunk_size=args.transformer_chunk_size,
            chunk_output_dir=args.transformer_chunk_output_dir,
        )
        matcher = TransformerMatcher(cfg=transformer_cfg)
    elif args.matcher == "cross_encoder":
        threshold = None if args.cross_encoder_threshold < 0 else args.cross_encoder_threshold
        cross_cfg = CrossEncoderMatcherConfig(
            model_path=args.cross_encoder_path,
            batch_size=args.cross_encoder_batch_size,
            pair_chunk_size=args.cross_encoder_chunk_size,
            device=args.device,
            max_length=args.cross_encoder_max_length,
            threshold=threshold,
        )
        matcher = CrossEncoderMatcher(cfg=cross_cfg)

    run_label = "strict" if args.strict_measure_only else "lenient"
    max_csv_rows = None if args.max_csv_rows <= 0 else args.max_csv_rows
    pipeline = ProductPipeline(
        repo=object(),
        snapshot_root=args.snapshot_root,
        blocking_config=blocking_cfg,
        fuzzy_config=fuzzy_cfg,
        matcher=matcher,
        resolver=ProductMatchResolver(
            out_root=args.out_root,
            run_label=run_label,
            write_candidates=args.write_candidates,
            include_candidate_context=(not args.skip_candidate_context),
            write_resolved_csv=args.write_resolved_csv,
            max_csv_rows=max_csv_rows,
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
