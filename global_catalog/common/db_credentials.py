import boto3
from global_catalog.config import settings

def get_redshift_local_connection_params():
    region = settings.AWS_REGION
    profile = settings.AWS_PROFILE
    database = settings.REDSHIFT_DATABASE
    db_user = settings.REDSHIFT_DB_USER
    cluster_id = settings.REDSHIFT_CLUSTER_ID
    host = settings.REDSHIFT_HOST
    port = settings.REDSHIFT_PORT

    sess = boto3.Session(profile_name=profile, region_name=region)
    rs = sess.client("redshift", region_name=region)
    creds = rs.get_cluster_credentials(
        DbUser=db_user,
        DbName=database,
        ClusterIdentifier=cluster_id,
        AutoCreate=False,
        DurationSeconds=900,
    )

    return {
        "host": host,
        "port": port,
        "database": database,
        "user": creds["DbUser"],
        "password": creds["DbPassword"],
    }
