
# Global Catalog

Data pipelines for normalizing and matching catalog entities (categories and products). The project currently focuses on the categories flow that reads raw snapshots from S3, normalizes them, performs TF‑IDF-based matching, resolves winning category IDs, and writes run artifacts locally.

## Setup

1. Create/activate a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Ensure you can assume the sandbox AWS role (the raw categories live in `s3://blaze-sandbox-global-catalog-service-staging-bucket/global-catalog/categories/...`).

## Running Categories From S3

CLI entrypoint: `python -m global_catalog.scripts.categories.run_categories_from_s3`

Parameters:
- `--date-prefix`: Partition under `<source>/raw/<date>/categories.csv`. Accepts `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY/MM/DD`.
- `--sources`: One or more source folders, e.g. `weedmaps hoodie`.
- `--out-root`: Local directory where run folders (`<date>_<uuid>/`) are created. Defaults to `GC_OUT_ROOT` (see `global_catalog/config/settings.py`).
- `--s3-run-prefix`, `--s3-latest-prefix`: Optional destinations when mirroring artifacts to S3 (not used yet but wired into the pipeline config).

Environment:
- `AWS_PROFILE` / `GC_S3_PROFILE`: profile that can read the sandbox bucket.
- `AWS_REGION`: defaults to `us-east-1`.
- Override any defaults in `global_catalog/config/settings.py` as needed.

Example command (mirrors the latest successful run):

```bash
AWS_PROFILE=sandbox-admin AWS_REGION=us-east-1 \
python -m global_catalog.scripts.categories.run_categories_from_s3 \
  --date-prefix 20251031 \
  --sources weedmaps hoodie \
  --out-root artifacts
```

Expected output:
- Console logs for each pipeline step (ingest/normalize/match/resolve) plus a final `Artifacts directory: /path/to/run`.
- Local run dir contains `pairs.parquet`, `summary.parquet`, `sample.csv`, `resolution.{parquet,csv}`, `staging_categories_id_mapping.{parquet,csv}` (with `external_id` replacing `category_id`), and `metrics.json`.

## Products V1 (In Progress)

Product ingestion/matching is being bootstrapped; CLI and documentation will be added once the first end-to-end path is ready.
