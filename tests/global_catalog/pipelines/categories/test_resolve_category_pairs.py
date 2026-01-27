import pandas as pd
import pytest
from typing import Tuple, Dict, Any
from pandas import DataFrame, Timestamp

from global_catalog.pipelines.categories.resolve_category_pairs import (
    _ensure_cats_contract,
    build_resolution_from_pairs,
    build_intra_resolution_from_pairs,
    global_category_id_map,
    _split_pretty,
    _count_levels,
    _winner_side,
    _uuid_from_hash,
    _ascii_lower_collapse,
    _build_row_id
)


class TestResolveCategoryPairs:

    @pytest.fixture
    def sample_cats_data(self) -> DataFrame:
        """sample data categories data for testing."""
        return pd.DataFrame({
            'id': ['1', '2', '3'],
            'category_id': ['cat1', 'cat2', 'cat3'],
            'source': ['weedmaps', 'hoodie', 'weedmaps'],
            'level_one': ['Flowers', 'Edibles', 'CBD'],
            'level_two': ['Indica', 'Gummies', 'Oil'],
            'level_three': ['Purple', 'Fruit', 'Tincture'],
            'updated_at': [
                '2025-09-08 18:55:00',
                'invalid_timestamp',
                None
            ]
        })

    @pytest.fixture
    def sample_pairs_data(self) -> DataFrame:
        """Sample pairs data for testing."""
        return pd.DataFrame({
            'left_id': ['1', '2'],
            'right_id': ['3', '4'],
            'left_source': ['weedmaps', 'hoodie'],
            'right_source': ['hoodie', 'weedmaps'],
            'left_path_pretty': ['Flowers/Indica/Purple', 'Edibles/Gummies/Fruit'],
            'right_path_pretty': ['CBD/Oil/Tincture', 'Flowers/Sativa/Green'],
            'similarity': [0.8, 0.75],
            'match_type': ['tfidf', 'exact'],
            'match_scope': ['cross', 'intra']
        })

    def test_ensure_cats_contract_fixes_nat_timestamps(self, sample_cats_data: DataFrame) -> None:
        """Test _ensure_cats_contract replaces NaT timestamps."""
        result: DataFrame = _ensure_cats_contract(sample_cats_data)

        # No NaT values should remain
        nat_count: int = pd.isna(result['updated_at']).sum()
        assert nat_count == 0

        # Valid timestamp should be preserved
        expected_timestamp: Timestamp = pd.to_datetime('2025-09-08 18:55:00')
        assert result['updated_at'].iloc[0] == expected_timestamp

        # Should have all required columns
        required_cols = ['id', 'category_id', 'source', 'level_one', 'level_two', 'level_three',
                         'updated_at']
        for col in required_cols:
            assert col in result.columns

    def test_ensure_cats_contract_adds_missing_columns(self) -> None:
        """Test _ensure_cats_contract adds missing columns with defaults."""
        minimal_data: DataFrame = pd.DataFrame({'some_col': ['value1', 'value2']})

        result: DataFrame = _ensure_cats_contract(minimal_data)

        # Should add missing columns
        expected_cols = ['source', 'level_one', 'level_two', 'level_three', 'category_id', 'updated_at', 'id']
        for col in expected_cols:
            assert col in result.columns

        # updated_at should be current timestamp
        assert pd.api.types.is_datetime64_any_dtype(result['updated_at'])

    def test_build_resolution_from_pairs_empty_cross_pairs(self, sample_cats_data: DataFrame) -> None:
        """Test build_resolution_from_pairs with no cross pairs."""
        empty_pairs: DataFrame = pd.DataFrame({
            'match_scope': ['intra', 'intra'],
            'left_path_pretty': ['a', 'b'],
            'right_path_pretty': ['c', 'd']
        })

        result: DataFrame = build_resolution_from_pairs(empty_pairs, sample_cats_data)

        assert result.empty
        # Should have expected columns
        expected_cols = ['id', 'category_id', 'level_one', 'level_two', 'level_three', 'source',
                         'pretty_path', 'dropped_pretty_path', 'dropped_source', 'dropped_id',
                         'dropped_category_id', 'updated_at', 'pair_similarity', 'pair_match_type']
        assert list(result.columns) == expected_cols

    def test_build_resolution_from_pairs_with_cross_pairs(self, sample_cats_data: DataFrame,
                                                          sample_pairs_data: DataFrame) -> None:
        """Test build_resolution_from_pairs with cross pairs."""
        # Filter to only cross pairs
        cross_pairs: DataFrame = sample_pairs_data[sample_pairs_data['match_scope'] == 'cross'].copy()

        if not cross_pairs.empty:
            result: DataFrame = build_resolution_from_pairs(cross_pairs, sample_cats_data)

            if not result.empty:
                # Should have no NaT values
                nat_count: int = pd.isna(result['updated_at']).sum()
                assert nat_count == 0

                # Should have proper datetime type
                assert pd.api.types.is_datetime64_any_dtype(result['updated_at'])

    def test_build_intra_resolution_from_pairs_empty_pairs(self, sample_cats_data: DataFrame) -> None:
        """Test build_intra_resolution_from_pairs with empty pairs."""
        empty_pairs: DataFrame = pd.DataFrame(columns=['match_scope', 'left_path_pretty', 'right_path_pretty'])

        keep_result: DataFrame
        pairwise_result: DataFrame
        keep_result, pairwise_result = build_intra_resolution_from_pairs(empty_pairs, sample_cats_data)

        assert keep_result.empty
        assert pairwise_result.empty

        # Should have expected columns
        expected_cols = ['id', 'category_id', 'level_one', 'level_two', 'level_three', 'source',
                         'pretty_path', 'dropped_pretty_path', 'dropped_source', 'dropped_id',
                         'dropped_category_id', 'updated_at', 'intra_policy']
        assert list(keep_result.columns) == expected_cols
        assert list(pairwise_result.columns) == expected_cols

    def test_global_category_id_map_no_nat_values(self, sample_cats_data: DataFrame) -> None:
        """Test global_category_id_map produces no NaT values."""
        empty_resolution: DataFrame = pd.DataFrame(columns=['id', 'dropped_id'])

        result: DataFrame = global_category_id_map(sample_cats_data, empty_resolution)

        # Should have no NaT values
        nat_count: int = pd.isna(result['updated_at']).sum()
        assert nat_count == 0

        # Should have global_id column
        assert 'global_id' in result.columns
        assert result['global_id'].notna().all()

        # Should have expected columns
        expected_cols = ['global_id', 'category_id', 'source', 'updated_at']
        assert list(result.columns) == expected_cols

    def test_split_pretty_function(self) -> None:
        """Test _split_pretty parses path correctly."""
        # Valid path
        result: Tuple[str, str, str] = _split_pretty("Flowers/Indica/Purple")
        assert result == ("Flowers", "Indica", "Purple")

        # Partial path
        result = _split_pretty("Flowers/Indica")
        assert result == ("Flowers", "Indica", None)

        # None input
        result = _split_pretty(None)
        assert result == (None, None, None)

        # Empty/uncategorized
        result = _split_pretty("uncategorized")
        assert result == (None, None, None)

    def test_count_levels_function(self) -> None:
        """Test _count_levels counts non-empty levels."""
        assert _count_levels("Flowers/Indica/Purple") == 3
        assert _count_levels("Flowers/Indica") == 2
        assert _count_levels("Flowers") == 1
        assert _count_levels("") == 0
        assert _count_levels(None) == 0

    def test_winner_side_function(self) -> None:
        """Test _winner_side chooses correct side."""
        # More levels wins
        row: Dict[str, Any] = {
            "left_path_pretty": "Flowers/Indica/Purple",
            "right_path_pretty": "Flowers",
            "left_source": "weedmaps",
            "right_source": "hoodie",
            "left_id": "1",
            "right_id": "2"
        }
        assert _winner_side(row) == "left"

        # Equal levels, alphabetical source
        row["right_path_pretty"] = "Edibles/Gummies/Fruit"
        row["left_source"] = "hoodie"
        row["right_source"] = "weedmaps"
        assert _winner_side(row) == "left"  # hoodie < weedmaps

        # Explicit left/right override beats level counts
        row = {
            "left_path_pretty": "Flowers",
            "right_path_pretty": "Flowers/Indica/Purple",
            "left_source": "left",
            "right_source": "right",
            "left_id": "l1",
            "right_id": "r1",
        }
        assert _winner_side(row) == "left"

    def test_helper_functions(self) -> None:
        """Test utility helper functions."""
        # Test _uuid_from_hash
        uuid_result: str = _uuid_from_hash("test_string")
        assert len(uuid_result) == 36  # UUID format
        assert isinstance(uuid_result, str)

        # Test _ascii_lower_collapse
        assert _ascii_lower_collapse("  Test  String  ") == "test string"
        assert _ascii_lower_collapse(None) == ""

        # Test _build_row_id
        row_id: str = _build_row_id("weedmaps", "Flowers", "Indica", "Purple")
        assert isinstance(row_id, str)
        assert len(row_id) == 32  # MD5 hash length

    @pytest.mark.parametrize("invalid_timestamp", [None, '', 'invalid', 'NaT', '2025-13-99'])
    def test_ensure_cats_contract_handles_invalid_timestamps(self, invalid_timestamp: str) -> None:
        """Test _ensure_cats_contract handles various invalid timestamp formats."""
        test_data: DataFrame = pd.DataFrame({
            'id': ['1'],
            'category_id': ['cat1'],
            'source': ['test'],
            'level_one': ['Test'],
            'level_two': ['Test'],
            'level_three': ['Test'],
            'updated_at': [invalid_timestamp]
        })

        result: DataFrame = _ensure_cats_contract(test_data)

        #should replace with valid timestamp
        timestamp_value: Any = result['updated_at'].iloc[0]
        assert pd.notna(timestamp_value)
        assert isinstance(timestamp_value, pd.Timestamp)

    def test_global_category_id_map_with_resolution(self, sample_cats_data: DataFrame) -> None:
        """Test global_category_id_map with resolution data."""
        resolution: DataFrame = pd.DataFrame({
            'id': ['1'],
            'dropped_id': ['2']
        })

        result: DataFrame = global_category_id_map(sample_cats_data, resolution)

        # Should handle resolution properly
        assert not result.empty
        assert 'global_id' in result.columns

        # Should have no NaT values
        nat_count: int = pd.isna(result['updated_at']).sum()
        assert nat_count == 0

    def test_global_category_id_map_propagates_existing_global_id(self) -> None:
        """Existing global_id on left should propagate to matched right."""
        cats = pd.DataFrame({
            "id": ["L1", "R1"],
            "category_id": ["cat_left", "cat_right"],
            "source": ["left", "right"],
            "level_one": ["Flower", "Flower"],
            "level_two": ["Indica", "Indica"],
            "level_three": ["Purple", "Purple"],
            "updated_at": ["2025-01-01", "2025-01-02"],
            "global_id": ["gid-left", ""],
        })
        resolution = pd.DataFrame({
            "id": ["L1"],
            "dropped_id": ["R1"],
        })

        result = global_category_id_map(cats, resolution)
        left_gid = result.loc[result["category_id"] == "cat_left", "global_id"].iloc[0]
        right_gid = result.loc[result["category_id"] == "cat_right", "global_id"].iloc[0]
        assert left_gid == "gid-left"
        assert right_gid == "gid-left"

    def test_global_category_id_map_mints_and_shares_gid_when_missing(self) -> None:
        """When both sides lack global_id, matched right should receive left's minted global_id."""
        cats = pd.DataFrame({
            "id": ["L1", "R1"],
            "category_id": ["cat_left", "cat_right"],
            "source": ["left", "right"],
            "level_one": ["Flower", "Flower"],
            "level_two": ["Indica", "Indica"],
            "level_three": ["Purple", "Purple"],
            "updated_at": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
            "global_id": ["", ""],
        })
        pairs = pd.DataFrame({
            "left_id": ["L1"],
            "right_id": ["R1"],
            "left_source": ["left"],
            "right_source": ["right"],
            "left_path_pretty": ["Flower/Indica/Purple"],
            "right_path_pretty": ["Flower/Indica/Purple"],
            "similarity": [1.0],
            "match_type": ["exact"],
            "match_scope": ["cross"],
        })
        resolution = build_resolution_from_pairs(pairs, cats)
        result = global_category_id_map(cats, resolution)
        left_gid = result.loc[result["category_id"] == "cat_left", "global_id"].iloc[0]
        right_gid = result.loc[result["category_id"] == "cat_right", "global_id"].iloc[0]
        assert left_gid != ""
        assert left_gid == right_gid
