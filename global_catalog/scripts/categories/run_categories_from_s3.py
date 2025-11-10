from global_catalog.config import settings
from global_catalog.matching.categories.matcher import CategoryMatchConfig, CategoriesMatcher
from global_catalog.pipelines.categories.resolve_category_pairs import build_resolution_from_pairs
from global_catalog.pipelines.categories.category_pipeline import CategoryPipeline
from global_catalog.publishers.s3_publisher import mirror_artifacts_to_s3
from global_catalog.repositories.s3_repo import S3Repo


class CategoryResolver:
    def resolve(self, pairs_df, df_raw):
        return build_resolution_from_pairs(pairs_df, df_raw)

def main():
    repo = S3Repo(
        bucket=settings.GC_S3_BUCKET,
        prefix=settings.GC_S3_PREFIX,
        profile=settings.GC_S3_PROFILE,
        region=settings.AWS_REGION,
    )

    cfg = CategoryMatchConfig(
        tfidf_threshold=settings.GC_TFIDF_THRESHOLD,
        block_by=settings.GC_BLOCK_BY,
        synonyms_path="global_catalog/normalization/rules/categories.synonyms.yml",
    )
    matcher = CategoriesMatcher(cfg)
    resolver = CategoryResolver()

    pipe = CategoryPipeline(
        repo=repo,
        matcher=matcher,
        resolver=resolver,
        publisher_fn=mirror_artifacts_to_s3,
    )

    out = pipe.run_categories_from_s3(
        date_prefix="20251031",
        sources=["weedmaps","hoodie"],
        local_out_root=settings.GC_OUT_ROOT,
        s3_out_base="global-catalog/categories/unified",
        bucket=settings.GC_S3_BUCKET,
    )

    print(out["metrics"]["outputs"])

if __name__ == "__main__":
    main()
