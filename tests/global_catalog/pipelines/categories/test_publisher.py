from pathlib import Path

import pandas as pd
import boto3

from global_catalog.config import settings
from global_catalog.pipelines.categories.publisher import CategoriesPublisher
from global_catalog.pipelines.entity_pipeline import EntityPipelineContext


def _build_context(run_dir: Path):
    run_metadata = {"run_dir": str(run_dir), "run_id": "run123", "inputs": {}}
    resolution = {
        "run_metadata": run_metadata,
        "category_global_id_map": pd.DataFrame(
            {
                "global_id": ["gid1"],
                "category_id": ["cat1"],
                "source": ["left"],
                "updated_at": ["2025-01-01"],
            }
        ),
        "resolution": pd.DataFrame(),
    }
    return EntityPipelineContext(
        raw_data={"run_metadata": run_metadata},
        match_results={},
        resolution=resolution,
    )


def test_publisher_uses_stage_in_mapping_name(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STAGE", "prod", raising=False)
    monkeypatch.setattr(settings, "GC_MAPPING_S3_BUCKET", "dummy-bucket", raising=False)
    monkeypatch.setattr(settings, "GC_MAPPING_S3_PREFIX", "categories_mapping", raising=False)

    pub = CategoriesPublisher()
    monkeypatch.setattr(pub, "_upload_category_mapping", lambda *args, **kwargs: None)

    ctx = _build_context(tmp_path)
    result = pub(ctx)

    out_path = Path(result["outputs"]["categories_id_mapping_parquet"])
    assert out_path.name == "prod_categories_id_mapping.parquet"
    assert out_path.exists()


def test_publisher_uses_aws_s3_profile_when_set(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AWS_PROFILE", "default-profile", raising=False)
    monkeypatch.setattr(settings, "AWS_S3_PROFILE", "blaze_development", raising=False)
    monkeypatch.setattr(settings, "GC_MAPPING_S3_BUCKET", "dummy-bucket", raising=False)
    monkeypatch.setattr(settings, "GC_MAPPING_S3_PREFIX", "categories_mapping", raising=False)

    seen = {}

    def fake_session(profile_name=None, region_name=None):
        seen["profile"] = profile_name

        class DummyClient:
            def upload_file(self, filename, bucket, key):
                seen["bucket"] = bucket
                seen["key"] = key

        class DummySession:
            def client(self, name):
                return DummyClient()

        return DummySession()

    monkeypatch.setattr(boto3, "Session", fake_session)

    pub = CategoriesPublisher()
    parquet_path = tmp_path / "file.parquet"
    parquet_path.write_text("x")

    pub._upload_category_mapping(parquet_path, "staging_categories_id_mapping")

    assert seen["profile"] == "blaze_development"
