from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional


class CategoriesMatchPublisher:
    #serialize match-stage artifacts locally (no resolve outputs, global-id attribution later)

    def __call__(self, context) -> Optional[Dict[str, Any]]:
        if context is None or context.match_results is None or context.raw_data is None:
            return None

        raw_data = context.raw_data or {}
        normalized = context.normalized or {}
        match_results = context.match_results or {}
        run_metadata = raw_data.get("run_metadata") or {}
        run_dir_str = run_metadata.get("run_dir")
        if not run_dir_str:
            raise RuntimeError("Run metadata must include run_dir before publishing match outputs")
        run_dir = Path(run_dir_str).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, str] = {"run_dir": str(run_dir)}

        df_raw = raw_data.get("df_raw")
        raw_parquet = run_dir / "df_raw.parquet"
        if df_raw is not None:
            df_raw.to_parquet(raw_parquet, index=False)
        outputs["df_raw_parquet_local"] = str(raw_parquet)

        df_norm = normalized.get("df_norm")
        norm_parquet = run_dir / "df_norm.parquet"
        if df_norm is not None:
            df_norm.to_parquet(norm_parquet, index=False)
        outputs["df_norm_parquet_local"] = str(norm_parquet)

        pairs = match_results.get("pairs")
        summary = match_results.get("summary")
        sample = match_results.get("sample")

        pairs_parquet = run_dir / "pairs.parquet"
        pairs_csv = run_dir / "pairs.csv"
        if pairs is not None:
            pairs.to_parquet(pairs_parquet, index=False)
            pairs.to_csv(pairs_csv, index=False)
        outputs["pairs_parquet_local"] = str(pairs_parquet)
        outputs["pairs_csv_local"] = str(pairs_csv)

        summary_parquet = run_dir / "summary.parquet"
        if summary is not None:
            summary.to_parquet(summary_parquet, index=False)
        outputs["summary_parquet_local"] = str(summary_parquet)

        sample_csv = run_dir / "sample.csv"
        if sample is not None:
            sample.to_csv(sample_csv, index=False)
        outputs["sample_csv_local"] = str(sample_csv)

        metadata_path = run_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        outputs["run_metadata_local"] = str(metadata_path)

        return {"outputs": outputs, "run_id": run_metadata.get("run_id")}
