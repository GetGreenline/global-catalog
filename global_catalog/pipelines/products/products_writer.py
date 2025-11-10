import pandas as pd
from pathlib import Path

def write_pairs(df: pd.DataFrame, out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_csv, index=False)

def summarize(df_exact: pd.DataFrame, df_fuzzy: pd.DataFrame, n_left: int, n_right: int, n_candidates: int, threshold: float):
    return {
        "input_left": n_left,
        "input_right": n_right,
        "candidates": n_candidates,
        "exact": len(df_exact),
        "fuzzy": len(df_fuzzy),
        "matches_total": len(pd.concat([df_exact, df_fuzzy], ignore_index=True)),
        "threshold": threshold
    }
