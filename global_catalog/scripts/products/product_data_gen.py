# File: global_catalog/scripts/products/generate_product_pairs.py

import argparse
from pathlib import Path

from global_catalog.matching.products.blocking_v2 import BlockingConfig
from global_catalog.matching.products.fuzzy_matcher_v2 import FuzzyMatcherConfig
from global_catalog.pipelines.products.product_data_gen import (
    DatasetGeneratorConfig,
    ProductDataGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate labeled product pairs for training/eval.")
    parser.add_argument("--snapshot-root", default="data/snapshots/products", help="Directory with product CSV snapshots.")
    parser.add_argument("--out-path", default="artifacts/products/datasets/product_pairs.parquet", help="Output parquet path.")

    parser.add_argument("--blocking-strategy", default="strategy_five", help="Blocking strategy to use for candidate pairs.")
    parser.add_argument("--pairs-dir", default=None, help="Optional directory containing cached candidate pair files.")
    parser.add_argument("--use-local-pairs", action="store_true", help="Use cached candidate pairs instead of running blocking.")

    parser.add_argument("--sample-left", type=int, default=5000, help="Number of left-source products to sample.")
    parser.add_argument("--positive-threshold", type=float, default=0.80, help="Similarity threshold for positive pairs.")
    parser.add_argument("--hard-negative-min", type=float, default=0.40, help="Minimum similarity for hard negatives.")
    parser.add_argument("--hard-negatives-per-left", type=int, default=5, help="Hard negatives per sampled left record.")
    parser.add_argument("--very-negatives-per-left", type=int, default=2, help="Very negative pairs per sampled left record.")

    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.20,
        help="Minimum similarity retained from fuzzy matcher (keep <= hard-negative-min).",
    )

    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Stop once this many labeled pairs are collected.")

    return parser.parse_args()


def main():
    args = parse_args()

    blocking_cfg = BlockingConfig(
        strategy_name=args.blocking_strategy,
        use_local_pairs=args.use_local_pairs,
        pairs_dir=args.pairs_dir,
    )

    # Keep enough low-scoring pairs so hard negatives aren't filtered out
    fuzzy_cfg = FuzzyMatcherConfig(
        threshold=min(args.min_similarity, args.hard_negative_min),
    )

    dataset_cfg = DatasetGeneratorConfig(
        sample_left=args.sample_left,
        positive_threshold=args.positive_threshold,
        hard_negative_min=args.hard_negative_min,
        hard_negatives_per_left=args.hard_negatives_per_left,
        very_negatives_per_left=args.very_negatives_per_left,
        random_seed=args.random_seed,
        output_path=args.out_path,
        max_pairs=args.max_pairs,
    )

    pipeline = ProductDataGenerator(
        repo=object(),  # ProductPipeline expects a repo; your local runners often pass a stub
        snapshot_root=args.snapshot_root,
        blocking_config=blocking_cfg,
        fuzzy_config=fuzzy_cfg,
        dataset_config=dataset_cfg,
        pairs_cache_dir=args.pairs_dir,
        local_run=args.use_local_pairs,
    )

    result = pipeline.run()
    dataset = (result or {}).get("resolution", {}).get("dataset")

    if dataset is None or dataset.empty:
        raise SystemExit(
            "No dataset rows generated. "
            "Try lowering --positive-threshold, lowering --hard-negative-min, "
            "or increasing --sample-left, and ensure your matcher returns a 'similarity' column."
        )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)

    print(f"Wrote dataset with {len(dataset)} pairs to {out_path}")
    print("Label counts:", dataset["label"].value_counts().to_dict())
    print("Pair-type counts:", dataset["pair_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
