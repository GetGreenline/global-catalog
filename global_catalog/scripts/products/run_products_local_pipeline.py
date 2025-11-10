import argparse
from pathlib import Path
import pandas as pd

from global_catalog.matching.products.fuzzy_matcher import fuzzy_matches_from_pairs
from global_catalog.transformers.products.products_normalization import ProductNormalizer
from global_catalog.matching.products.blocking import build_product_blocking_pairs, \
    build_product_blocking_pairs_multipass
from global_catalog.matching.products.tfidf_matcher import tfidf_matches_from_pairs
from global_catalog.pipelines.products.products_writer import write_pairs, summarize

def read_csv_flex(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        df[c] = df[c].astype(str)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoodie", default="data/snapshots/products/products_hoodie.csv")
    ap.add_argument("--weedmaps", default="data/snapshots/products/products_weedmaps.csv")
    ap.add_argument("--brands", default="data/snapshots/products/brands_w.csv")
    ap.add_argument("--out_root", default="data/matches/")
    ap.add_argument("--out", default="data/matches/products_pairs_tfidf_wo_description.parquet")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--block_threshold", type=float, default=0.35)
    ap.add_argument("--max_per_left", type=int, default=300)                
    ap.add_argument("--include_description", action="store_true")
    ap.add_argument("--no_enforce_uom", action="store_true")
    ap.add_argument("--use_states", action="store_true")
    ap.add_argument("--use_country", action="store_true")
    ap.add_argument("--use_strain", action="store_true")
    ap.add_argument("--block_threshold_secondary", type=float, default=0.30)
    ap.add_argument("--measure_tol", type=float, default=0.15)
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    df_l = read_csv_flex(args.weedmaps)
    df_r = read_csv_flex(args.hoodie)
    brand_df = read_csv_flex(args.brands) if args.brands else None

    normalizer = ProductNormalizer()
    df_norm = normalizer.process(hoodie_df=df_r, weedmaps_df=df_l, brand_df=brand_df)

    pairs_df = build_product_blocking_pairs_multipass(
        df=df_norm,
        threshold_first_pass=args.block_threshold,
        threshold_second_pass=0.60,
        max_per_left=args.max_per_left,
        measure_tol=args.measure_tol,
        enforce_uom=not args.no_enforce_uom,
        use_states=False,
        use_country=False,
        use_strain=False,
        include_description=False,
        description_token_limit=0,
    )

    out = fuzzy_matches_from_pairs(
        df_norm=df_norm,
        pairs_df=pairs_df,
        threshold=args.threshold,
        include_description=args.include_description,
    )

    df_ln = df_norm[df_norm["source"].astype(str).str.lower().eq("weedmaps")]
    df_rn = df_norm[df_norm["source"].astype(str).str.lower().eq("hoodie")]

    out = out.sort_values(["similarity"], ascending=[False]).reset_index(drop=True)
    write_pairs(out, args.out)

    metrics = summarize(pd.DataFrame(), out, len(df_ln), len(df_rn), len(pairs_df), args.threshold)
    print(metrics)
    print(f"wrote: {args.out}")

if __name__ == "__main__":
    main()

