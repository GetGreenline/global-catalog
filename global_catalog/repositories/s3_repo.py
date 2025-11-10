import boto3
import awswrangler as wr
import pandas as pd

class S3Repo:
    def __init__(self, bucket: str, prefix: str, profile: str, region: str):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.session = boto3.Session(profile_name=profile, region_name=region)

    def _path(self, source: str, date_prefix: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{source}/raw/{date_prefix}/categories.csv"

    def read_categories_raw(self, sources: list[str], date_prefix: str) -> pd.DataFrame:
        dfs = []
        for src in sources:
            uri = self._path(src, date_prefix)
            df = wr.s3.read_csv(uri, boto3_session=self.session)
            if "source" not in df.columns:
                df["source"] = src

            # Rename external_id to category_id for weedmaps only
            src_key = str(src).strip().lower()
            if src_key == "weedmaps" and "external_id" in df.columns and "category_id" not in df.columns:
                df = df.rename(columns={"external_id": "category_id"})

            dfs.append(df)
        out = pd.concat(dfs, ignore_index=True)
        return out
