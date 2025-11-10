from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence
import pandas as pd
import awswrangler as wr


class ExtractorConfig:
    sql_path: Optional[str] = None
    snapshot_csv: Optional[str] = None
    use_snapshot: bool = True
    s3_snapshot_uri: Optional[str] = None
    required_columns: Sequence[str] = ()

class BaseExtractor(ABC):
    def __init__(self, config: ExtractorConfig):
        self.config = config

    @abstractmethod
    def _read_live(self) -> pd.DataFrame:
        ...

    def run(self) -> pd.DataFrame:
        df = self._load_snapshot_if_requested()
        if df is None:
            df = self._read_live()
            self._write_snapshot(df)
        self._validate_columns(df)
        return df

    def _load_snapshot_if_requested(self) -> Optional[pd.DataFrame]:
        p = self.config.snapshot_csv
        if self.config.use_snapshot and p and Path(p).exists():
            return pd.read_csv(p)
        return None

    def _write_snapshot(self, df: pd.DataFrame) -> None:
        if self.config.snapshot_csv:
            Path(self.config.snapshot_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.config.snapshot_csv, index=False)
        if self.config.s3_snapshot_uri:
            wr.s3.to_csv(df=df, path=self.config.s3_snapshot_uri)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if not self.config.required_columns:
            return
        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
