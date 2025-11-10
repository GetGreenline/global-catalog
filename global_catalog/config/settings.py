import os

OUT_STAGE = 'local'

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "prod-developer")

GC_S3_BUCKET = os.getenv("GC_S3_BUCKET", "blaze-sandbox-global-catalog-service-staging-bucket")
GC_S3_PREFIX = os.getenv("GC_S3_PREFIX", "global-catalog/categories")
GC_S3_PROFILE = os.getenv("GC_S3_PROFILE", "sandbox-admin")

REDSHIFT_HOST = os.getenv("REDSHIFT_HOST", "localhost")
REDSHIFT_PORT = int(os.getenv("REDSHIFT_PORT", "5439"))
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE", "dev_data_warehouse")

REDSHIFT_CLUSTER_ID = os.getenv("REDSHIFT_CLUSTER_ID", "prod-blaze-data-pipelines-redshift-cluster")

GC_OUT_ROOT = os.getenv("GC_OUT_ROOT", "artifacts")
GC_TFIDF_THRESHOLD = float(os.getenv("GC_TFIDF_THRESHOLD", "0.80"))
GC_BLOCK_BY = os.getenv("GC_BLOCK_BY", "none")

GC_CATEGORIES_SNAPSHOT_CSV = os.getenv("GC_CATEGORIES_SNAPSHOT_CSV", "global_catalog/data/snapshots/categories_deduped.csv")

