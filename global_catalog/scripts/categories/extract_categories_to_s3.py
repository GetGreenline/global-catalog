import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import pandas as pd
import boto3
import awswrangler as wr
from global_catalog.repositories.redshift_repo import RedShiftRepo
from global_catalog.config import settings


def write_source_to_s3(source: str, sql_path: str, s3_folder: str):
    sql = Path(sql_path).read_text()
    df = RedShiftRepo().read_sql(sql)

    s3_profile = settings.GC_S3_PROFILE
    s3_region = settings.AWS_REGION
    s3_session = boto3.Session(profile_name=s3_profile, region_name=s3_region)

    date_prefix = pd.Timestamp.utcnow().strftime("%Y%m%d")
    s3_uri = f"s3://{settings.GC_S3_BUCKET}/global-catalog/categories/{source}/raw/{date_prefix}/categories.csv"

    print(f"Writing {source} extract to {s3_uri} using profile {s3_profile}")
    wr.s3.to_csv(df=df, path=s3_uri, index=False, boto3_session=s3_session)
    print(f"Wrote {source} snapshot to: {s3_uri}")


def main():
    source_sql_map = {
        "hoodie": "global_catalog/sql/categories/hoodie_extract.sql",
        "weedmaps": "global_catalog/sql/categories/weedmaps_extract.sql",
    }
    for source, sql_path in source_sql_map.items():
        write_source_to_s3(source, sql_path, s3_folder=f"categories/{source}/raw")


if __name__ == "__main__":
    main()
