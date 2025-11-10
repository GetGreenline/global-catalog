from pathlib import Path
import time
import json
import pandas as pd
from global_catalog.repositories.redshift_repo import RedShiftRepo
from global_catalog.matching.categories.path_tfidf import TfidfCategoryMatcher

SNAPSHOT_PATH = Path("data/snapshots/categories/categories.csv")
REFRESH_SNAPSHOT = False
USE_WAREHOUSE_SNAPSHOT = True

DEDUPED_CSV_PATH = Path("data/snapshots/categories/categories_deduped.csv")

def load_or_create_snapshot(snapshot_path: Path, refresh: bool = False) -> pd.DataFrame:
    if snapshot_path.exists() and not refresh:
        return pd.read_parquet(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    repo = RedShiftRepo()
    sql = """

    """
    df = repo.read_sql(sql)
    df.to_parquet(snapshot_path, index=False)
    return df


def describe_similarity(sim_series: pd.Series) -> dict:
    if sim_series.empty:
        return {}
    return {
        "count": int(sim_series.shape[0]),
        "min": float(sim_series.min()),
        "max": float(sim_series.max()),
        "mean": float(sim_series.mean()),
        "std": float(sim_series.std(ddof=0))
    }


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'external_id' in df.columns:
        df.rename(columns={"external_id": "category_id",
                           "l1": "level_one",
                           "l2": "level_two",
                           "l3": "level_three"}, inplace=True)

    expected_cols = ["category_id", "level_one", "level_two", "level_three", "source", 'updated_at']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}. Present: {list(df.columns)}")
    return df[expected_cols].copy()


def calculate_usage_counts() -> dict:

    print("ahhhh")
    return {}


def main():
    t0 = time.perf_counter()

    print(f"Loading deduped categories from: {DEDUPED_CSV_PATH}")
    deduped_df_raw = pd.read_csv(DEDUPED_CSV_PATH)

    deduped_df = ensure_expected_columns(deduped_df_raw)
    deduped_df['updated_at'] = pd.to_datetime(deduped_df['updated_at'])
    print(f"Loaded {deduped_df.shape[0]} unique category records to be matched.")

    usage_counts = calculate_usage_counts()

    t_load = time.perf_counter()

    print(f"Matching categories across {deduped_df['source'].nunique()} sources...")
    matcher = TfidfCategoryMatcher(threshold=0.84)
    matches = matcher.match_across_sources(deduped_df)
    t_match = time.perf_counter()
    print(f"Found {matches.shape[0]} potential matches.")

    print("Resolving matches to select the best candidate...")
    resolved_matches = matcher.resolve_matches(matches)
    t_resolve = time.perf_counter()
    print(f"Resolved {resolved_matches.shape[0]} matches.")

    # CategoriesFileRepo().save_matches(resolved_matches,
    # f"global_catalog/artifacts/category_matches_resolved.parquet")
    resolved_matches.to_csv("global_catalog/artifacts/category_matches_resolved_sample.csv", index=False)

    metrics = {
        "records_total": int(deduped_df.shape[0]),
        "sources": sorted([s for s in deduped_df["source"].dropna().unique()]),
        "candidates_matched": int(resolved_matches.shape[0]),
        "similarity_stats": describe_similarity(resolved_matches["similarity"]),
        "timing_seconds": {
            "load_and_prep": round(t_load - t0, 4),
            "match": round(t_match - t_load, 4),
            "resolve": round(t_resolve - t_match, 4),
            "total": round(t_resolve - t0, 4)
        }
    }
    # CategoriesFileRepo().save_metrics(metrics, "global_catalog/artifacts/category_match_metrics.json")

    print("\n--- Match Complete ---")
    print(json.dumps(metrics, indent=2))
    # print("Saved: artifacts/category_matches_resolved.parquet")
    print("Saved: global_catalog/artifacts/category_matches_resolved_sample.csv")
    # print("Saved: artifacts/category_match_metrics.json")


if __name__ == "__main__":
    main()

