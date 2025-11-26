import datetime

import pandas as pd
import pytest
from unittest.mock import Mock
from typing import Tuple, Any, Optional
from pandas import DataFrame, Timestamp

from global_catalog.matching.categories.matcher import CategoriesMatcher, CategoryMatchConfig


class TestCategoriesMatcherNormalize:
    """Simple, focused tests for CategoriesMatcher.normalize function."""

    @pytest.fixture
    def matcher(self) -> CategoriesMatcher:
        """Create matcher with mocked processor."""
        config: CategoryMatchConfig = CategoryMatchConfig()
        matcher: CategoriesMatcher = CategoriesMatcher(config)
        matcher.proc = Mock()
        return matcher

    @pytest.fixture
    def test_data(self) -> DataFrame:
        """Test data with timestamp issues."""
        return pd.DataFrame({
            'id': ['1', '2', '3'],
            'source': ['weedmaps', 'hoodie', 'weedmaps'],
            'level_one': ['Flowers', 'Edibles', 'CBD'],
            'level_two': ['Indica', 'Gummies', 'Oil'],
            'level_three': ['Purple', 'Fruit', 'Tincture'],
            'updated_at': [
                '2025-09-08 18:55:00.706330',  # Valid
                'invalid_timestamp',           # Invalid - should be replaced
                None                          # None should be replaced
            ]
        })

    @staticmethod
    def _mock_processor(df: DataFrame) -> DataFrame:
        """Simple mock processor that returns the expected structure."""
        return pd.DataFrame({
            'id': df['id'],
            'source': df['source'],
            'updated_at': df['updated_at'],
            'l1_norm': df['level_one'].str.lower(),
            'l2_norm': df['level_two'].str.lower(),
            'l3_norm': df['level_three'].str.lower()
        })

    def test_normalize_returns_correct_structure(self, matcher: CategoriesMatcher, test_data: DataFrame) -> None:
        """Test normalize returns tuple of two DataFrames."""
        matcher.proc.process.side_effect = self._mock_processor

        result: Tuple[DataFrame, DataFrame] = matcher.normalize(test_data)

        assert isinstance(result, tuple)
        assert len(result) == 2
        df_norm: DataFrame
        df_pretty: DataFrame
        df_norm, df_pretty = result
        assert isinstance(df_norm, pd.DataFrame)
        assert isinstance(df_pretty, pd.DataFrame)

    def test_normalize_fixes_nat_timestamps(self, matcher: CategoriesMatcher, test_data: DataFrame) -> None:
        """Test normalize replaces NaT values with current timestamps."""
        matcher.proc.process.side_effect = self._mock_processor

        before_test = datetime.datetime.now()
        df_norm: DataFrame
        df_pretty: DataFrame
        df_norm, df_pretty = matcher.normalize(test_data)
        after_test = datetime.datetime.now()

        # No NaT values should remain
        nat_count_norm: int = pd.isna(df_norm['updated_at']).sum()
        nat_count_pretty: int = pd.isna(df_pretty['updated_at']).sum()
        assert nat_count_norm == 0
        assert nat_count_pretty == 0

        # Valid timestamp should be preserved
        expected_timestamp: Timestamp = pd.to_datetime('2025-09-08 18:55:00.706330')
        assert df_norm['updated_at'].iloc[0] == expected_timestamp

        # Invalid timestamps should be replaced with the current time
        invalid_indices: list[int] = [1, 2]  # Invalid timestamp indices
        for idx in invalid_indices:
            timestamp: Timestamp = df_norm['updated_at'].iloc[idx]
            assert before_test <= timestamp <= after_test

    def test_normalize_creates_pretty_dataframe(self, matcher: CategoriesMatcher, test_data: DataFrame) -> None:
        """Test normalize creates a pretty DataFrame with level_* columns."""
        matcher.proc.process.side_effect = self._mock_processor

        df_norm: DataFrame
        df_pretty: DataFrame
        df_norm, df_pretty = matcher.normalize(test_data)

        # Pretty DataFrame should have level_* columns mapped from l*_norm
        assert 'level_one' in df_pretty.columns
        assert 'level_two' in df_pretty.columns
        assert 'level_three' in df_pretty.columns

        # Check mapping is correct
        assert df_pretty['level_one'].iloc[0] == df_norm['l1_norm'].iloc[0]
        assert df_pretty['level_two'].iloc[0] == df_norm['l2_norm'].iloc[0]
        assert df_pretty['level_three'].iloc[0] == df_norm['l3_norm'].iloc[0]

    def test_normalize_calls_processor(self, matcher: CategoriesMatcher, test_data: DataFrame) -> None:
        """Test normalize calls processor with timestamp-fixed data."""
        captured_input: Optional[DataFrame] = None

        def capture_processor(df: DataFrame) -> DataFrame:
            nonlocal captured_input
            captured_input = df.copy()
            return self._mock_processor(df)

        matcher.proc.process.side_effect = capture_processor

        matcher.normalize(test_data)

        # Processor should be called once
        matcher.proc.process.assert_called_once()

        # Input to the processor should have no NaT values
        assert captured_input is not None
        nat_count: int = pd.isna(captured_input['updated_at']).sum()
        assert nat_count == 0
        assert pd.api.types.is_datetime64_any_dtype(captured_input['updated_at'])

    @pytest.mark.parametrize("invalid_value", [None, '', 'invalid', 'NaT'])
    def test_normalize_handles_various_invalid_timestamps(self, matcher: CategoriesMatcher, invalid_value: Optional[str]) -> None:
        """Test normalize handles various invalid timestamp formats."""
        test_data: DataFrame = pd.DataFrame({
            'id': ['1'],
            'source': ['test'],
            'level_one': ['Test'],
            'level_two': ['Test'],
            'level_three': ['Test'],
            'updated_at': [invalid_value]
        })

        matcher.proc.process.side_effect = self._mock_processor

        df_norm: DataFrame
        df_pretty: DataFrame
        df_norm, df_pretty = matcher.normalize(test_data)

        # Should replace it with a valid timestamp
        timestamp_value: Any = df_norm['updated_at'].iloc[0]
        assert pd.notna(timestamp_value)
        assert isinstance(timestamp_value, pd.Timestamp)

