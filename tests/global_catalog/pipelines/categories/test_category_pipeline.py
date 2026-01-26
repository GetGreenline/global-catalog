import pandas as pd

from global_catalog.pipelines.categories.category_pipeline import CategoriesRunConfig, CategoryPipeline


class DummyMatcher:
    def normalize(self, df_raw):
        return df_raw.copy(), df_raw.copy()

    def generate_pairs(self, df_norm):
        return pd.DataFrame()

    def filter_pairs(self, pairs, df_norm):
        return pairs

    def summarize(self, pairs, df_pretty_like):
        return pairs, pd.DataFrame()

    def metrics(self, df_norm, pairs, t0):
        return {}


class DummyResolver:
    def resolve(self, pairs_df, df_raw):
        return pd.DataFrame(columns=["id", "dropped_id"])


class DummyRepo:
    bucket = None
    prefix = None


def test_run_dir_reuse_between_match_and_resolve(tmp_path):
    df = pd.DataFrame(
        {
            "category_id": ["c1"],
            "source": ["left"],
            "level_one": ["Flower"],
            "level_two": ["Indica"],
            "level_three": ["Purple"],
            "updated_at": ["2025-01-01"],
        }
    )
    csv_path = tmp_path / "cats.csv"
    df.to_csv(csv_path, index=False)

    pipe = CategoryPipeline(
        repo=DummyRepo(),
        matcher=DummyMatcher(),
        resolver=DummyResolver(),
        publisher_fn=None,
    )
    pipe._log_identity = lambda: None

    cfg = CategoriesRunConfig(
        date_prefix=None,
        sources=["left"],
        local_out_root=str(tmp_path),
        ingest_source="csv",
        csv_path=str(csv_path),
    )

    result = pipe.run_categories_pipeline(cfg)
    run_dir_match = result["run_metadata"]["run_dir"]
    run_dir_resolve = result["resolution"]["run_metadata"]["run_dir"]

    assert run_dir_match == run_dir_resolve
