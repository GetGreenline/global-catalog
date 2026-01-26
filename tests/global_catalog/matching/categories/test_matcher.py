import pandas as pd

from global_catalog.matching.categories.matcher import CategoryMatchConfig, CategoriesMatcher


def test_normalize_uses_match_side_for_source() -> None:
    df_raw = pd.DataFrame(
        {
            "id": ["L1", "R1"],
            "category_id": ["cat_left", "cat_right"],
            "source": ["v1", "weedmaps"],
            "match_side": ["left", "right"],
            "level_one": ["Flower", "Flower"],
            "level_two": ["Indica", "Indica"],
            "level_three": ["Purple", "Purple"],
            "updated_at": ["2025-01-01", "2025-01-02"],
        }
    )

    matcher = CategoriesMatcher(CategoryMatchConfig())
    df_norm, _ = matcher.normalize(df_raw)

    assert set(df_norm["source"].unique()) == {"left", "right"}
    assert df_norm.loc[df_norm["id"] == "L1", "source_raw"].iloc[0] == "v1"
    assert df_norm.loc[df_norm["id"] == "R1", "source_raw"].iloc[0] == "weedmaps"


def test_generate_pairs_crosses_left_right() -> None:
    df_raw = pd.DataFrame(
        {
            "id": ["L1", "R1"],
            "category_id": ["cat_left", "cat_right"],
            "source": ["v1", "weedmaps"],
            "match_side": ["left", "right"],
            "level_one": ["Flower", "Flower"],
            "level_two": ["Indica", "Indica"],
            "level_three": ["Purple", "Purple"],
            "updated_at": ["2025-01-01", "2025-01-02"],
        }
    )

    matcher = CategoriesMatcher(CategoryMatchConfig())
    df_norm, _ = matcher.normalize(df_raw)
    pairs = matcher.generate_pairs(df_norm)

    assert not pairs.empty
    assert set(pairs["left_source"].unique()) == {"left"}
    assert set(pairs["right_source"].unique()) == {"right"}
