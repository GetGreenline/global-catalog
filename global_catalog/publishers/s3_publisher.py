from pathlib import Path
import boto3

def mirror_artifacts_to_s3(local_run_dir: str, s3_run_prefix: str, s3_latest_prefix: str):
    s3 = boto3.client("s3")
    run_dir = Path(local_run_dir)
    bucket_run, prefix_run = s3_run_prefix.replace("s3://","").split("/",1)
    bucket_latest, prefix_latest = s3_latest_prefix.replace("s3://","").split("/",1)
    for p in run_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(run_dir).as_posix()
            s3.upload_file(str(p), bucket_run, f"{prefix_run}/{rel}")
            s3.upload_file(str(p), bucket_latest, f"{prefix_latest}/{rel}")
