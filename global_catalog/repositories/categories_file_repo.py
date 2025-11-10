import pandas as pd
from pathlib import Path
from datetime import datetime
import json

class CategoriesFileRepo:
    def __init__(self, csv_path: str = "artifacts/stg_us_categories_dedup_with_dupes.csv"):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        expected = {"level_one", "level_two", "level_three", "source", "updated_at"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def _timestamped_path(self, base_path: str, ext: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path(base_path)
        name = p.stem + f"_{ts}" + ext
        return p.with_name(name)

    def save_matches(self, df: pd.DataFrame, out_path: str = "artifacts/category_matches.parquet"):
        out_path = self._timestamped_path(out_path, ".parquet")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved matches to {out_path}")

    def save_metrics(self, metrics: dict, out_path: str = "artifacts/category_match_metrics.json"):
        out_path = self._timestamped_path(out_path, ".json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out_path}")
