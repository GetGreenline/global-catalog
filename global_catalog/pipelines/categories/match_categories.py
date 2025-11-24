import argparse
import json
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

from global_catalog.matching.categories.path_tfidf import (
    unordered_exact_cross, unordered_exact_intra
)

from global_catalog.matching.categories.blocking import (
    tfidf_cross_blocked, tfidf_intra_blocked
)

from global_catalog.matching.categories.path_tfidf import (
    exact_cross, exact_intra, tfidf_cross, tfidf_intra
)
from global_catalog.matching.categories.pair_filters import (
    drop_intra_parent_child
)
from global_catalog.matching.categories.summarize import (
    attach_pretty_paths, summarize_per_category, attach_summary_flags
)
from global_catalog.pipelines.categories.resolve_category_pairs import (
    build_resolution_from_pairs, build_intra_resolution_from_pairs
)
from global_catalog.transformers.categories.category_normalizer import CategoryNormalizer

EXPECTED_RAW = ["level_one","level_two","level_three","source","updated_at"]


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    if "external_id" in cols and "source_category_id" not in df.columns:
        df.rename(columns={cols["external_id"]: "source_category_id"}, inplace=True)
    missing = [c for c in EXPECTED_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Present: {list(df.columns)}")
    if "category_id" not in df.columns:
        df["category_id"] = df.get("source_category_id", pd.Series(range(len(df)))).astype(str)
    out_cols = ["category_id", "source_category_id", "level_one", "level_two", "level_three", "source", "updated_at"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None
    return df[out_cols]

def format_run_id(user_run_id: str | None, tag: str | None) -> str:
    rid = user_run_id.strip() if (user_run_id and user_run_id.strip()) else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if tag and tag.strip():
        rid = f"{rid}_{tag.strip()}"
    return rid

def prepare_run_dir(out_root: str, run_id: str, make_latest_symlink: bool = True) -> Path:
    root = Path(out_root)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if make_latest_symlink:
        latest = root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except Exception:
            pass
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="global_catalog/data/snapshots/categories_deduped.csv")
    ap.add_argument("--out-root", default="artifacts")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-latest-symlink", action="store_true", default=False)
    ap.add_argument("--sample-limit", type=int, default=2000)
    ap.add_argument("--tfidf-threshold", type=float, default=0.80)
    ap.add_argument("--synonyms", default="global_catalog/normalization/rules/categories.synonyms.yml")
    ap.add_argument("--emit-intra-resolution", action="store_true", default=False)
    ap.add_argument("--block-by", choices=["none", "l1", "l1l2"], default="none")
    args = ap.parse_args()

    t0 = time.perf_counter()

    run_id = format_run_id(args.run_id, args.tag)
    run_dir = prepare_run_dir(args.out_root, run_id, make_latest_symlink=(not args.no_latest_symlink))

    out_pairs_path   = run_dir / "pairs.parquet"
    out_summary_path = run_dir / "summary.parquet"
    out_sample_path  = run_dir / "sample.csv"
    out_metrics_path = run_dir / "metrics.json"
    out_args_path    = run_dir / "args.json"
    out_resolution_parquet = run_dir / "resolution.parquet"
    out_resolution_csv     = run_dir / "resolution.csv"

    df = pd.read_csv(args.in_csv)
    df = ensure_expected_columns(df)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    proc = CategoryNormalizer(synonyms_path=args.synonyms)
    df_norm = proc.process(df)

    df_pretty_like = df.copy()
    df_pretty_like["level_one"]   = df_norm["l1_norm"]
    df_pretty_like["level_two"]   = df_norm["l2_norm"]
    df_pretty_like["level_three"] = df_norm["l3_norm"]

    exact_cross_df = exact_cross(df_norm)
    exact_intra_df = exact_intra(df_norm)

    block_map = {
        "none": (),
        "l1": ("l1_norm",),
        "l1l2": ("l1_norm", "l2_norm"),
    }
    blk = block_map[args.block_by]

    if args.block_by == "none":
        tfidf_cross_df = tfidf_cross(df_norm, threshold=args.tfidf_threshold)
        tfidf_intra_df = tfidf_intra(df_norm, threshold=args.tfidf_threshold)
    else:
        tfidf_cross_df = tfidf_cross_blocked(df_norm, threshold=args.tfidf_threshold, block_levels=blk)
        tfidf_intra_df = tfidf_intra_blocked(df_norm, threshold=args.tfidf_threshold, block_levels=blk)

    unordered_cross_df = unordered_exact_cross(df_norm)
    unordered_intra_df = unordered_exact_intra(df_norm)

    combined = pd.concat(
        [exact_cross_df, tfidf_cross_df, unordered_cross_df,
         exact_intra_df, tfidf_intra_df, unordered_intra_df],
        ignore_index=True
    )
    combined.drop_duplicates(
        subset=["left_source","right_source","left_category_id","right_category_id","match_scope","match_type"],
        inplace=True
    )

    combined = drop_intra_parent_child(combined, df_norm)

    combined = attach_pretty_paths(combined, df_pretty_like)
    summary = summarize_per_category(combined)
    combined = attach_summary_flags(combined, summary)

    combined.to_parquet(out_pairs_path, index=False)
    summary.to_parquet(out_summary_path, index=False)
    (combined.sort_values(["match_scope","match_type","similarity"], ascending=[True, True, False])
            .head(args.sample_limit)
            .to_csv(out_sample_path, index=False))

    resolution = build_resolution_from_pairs(combined, df)
    resolution.to_parquet(out_resolution_parquet, index=False)
    resolution.to_csv(out_resolution_csv, index=False)

    intra_rows = 0
    if args.emit_intra_resolution:
        intra_agg, intra_pair = build_intra_resolution_from_pairs(combined, df)
        intra_agg.to_parquet(run_dir / "intra_resolution.parquet", index=False)
        intra_pair.to_parquet(run_dir / "intra_resolution_pairwise.parquet", index=False)
        intra_rows = int(intra_agg.shape[0])

    metrics = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "records_total": int(df_norm.shape[0]),
        "sources": sorted([s for s in df_norm["source"].dropna().unique()]),
        "matches_total": int(combined.shape[0]),
        "exact_cross": int(exact_cross_df.shape[0]),
        "tfidf_cross": int(tfidf_cross_df.shape[0]),
        "unordered_cross": int(unordered_cross_df.shape[0]),
        "exact_intra": int(exact_intra_df.shape[0]),
        "tfidf_intra": int(tfidf_intra_df.shape[0]),
        "unordered_intra": int(unordered_intra_df.shape[0]),
        "resolution_rows": int(resolution.shape[0]),
        "intra_resolution_rows": intra_rows,
        "similarity_stats": {},
        "timing_seconds": round(time.perf_counter() - t0, 3),
        "threshold": args.tfidf_threshold,
        "outputs": {
            "pairs": str(out_pairs_path),
            "summary": str(out_summary_path),
            "sample": str(out_sample_path),
            "resolution_parquet": str(out_resolution_parquet),
            "resolution_csv": str(out_resolution_csv),
            "metrics": str(out_metrics_path),
            "args": str(out_args_path),
            "intra_resolution_parquet": str(run_dir / "intra_resolution.parquet") if args.emit_intra_resolution else None,
            "intra_resolution_pairwise_parquet": str(run_dir / "intra_resolution_pairwise.parquet") if args.emit_intra_resolution else None,
        },
        "inputs": {"in_csv": args.in_csv}
    }
    with open(out_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir} (symlink: {'enabled' if not args.no_latest_symlink else 'disabled'})")
    print(f"Wrote pairs parquet:   {out_pairs_path}")
    print(f"Wrote summary parquet: {out_summary_path}")
    print(f"Wrote sample CSV:      {out_sample_path}")
    print(f"Wrote resolution:      {out_resolution_parquet} and {out_resolution_csv}")
    if args.emit_intra_resolution:
        print(f"Wrote intra resolution: {run_dir / 'intra_resolution.parquet'} and {run_dir / 'intra_resolution_pairwise.parquet'}")


if __name__ == "__main__":
    main()
