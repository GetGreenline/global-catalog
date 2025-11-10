import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import pandas as pd
import boto3
import awswrangler as wr
from global_catalog.repositories.redshift_repo import RedShiftRepo
from global_catalog.config import settings


def write_table_to_s3(source: str, table_name: str):
    query = f"SELECT * FROM {table_name}"
    df = RedShiftRepo().read_sql(query)

    s3_profile = settings.GC_S3_PROFILE
    s3_region = settings.AWS_REGION
    s3_session = boto3.Session(profile_name=s3_profile, region_name=s3_region)

    date_prefix = pd.Timestamp.utcnow().strftime("%Y%m%d")
    s3_uri = f"s3://{settings.GC_S3_BUCKET}/global-catalog/products/{source}/raw/{date_prefix}/products.csv"

    print(f"Writing {source} products from {table_name} to {s3_uri} using profile {s3_profile}")
    wr.s3.to_csv(df=df, path=s3_uri, index=False, boto3_session=s3_session)
    print(f"Wrote {source} snapshot to: {s3_uri}")


def main():

    source_table_map = {
        "hoodie": "stg_us_hoodie_products",
        "weedmaps": "stg_us_weedmaps_products",
    }

    for source, table in source_table_map.items():
        write_table_to_s3(source, table)


if __name__ == "__main__":
    main()
