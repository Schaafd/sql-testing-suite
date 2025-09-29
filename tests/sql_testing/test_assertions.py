"""
Comprehensive tests for the SQL testing assertion framework.

This module tests all assertion types, statistical comparisons,
and custom assertion functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from sqltest.modules.sql_testing.assertions import (
    SQLTestAssertionEngine,
    AssertionResult
)
from sqltest.modules.sql_testing.models import AssertionType


class TestSQLTestAssertionEngine:
    """Test the main assertion engine functionality."""

    @pytest.fixture
    def assertion_engine(self):
        """Create assertion engine instance."""
        return SQLTestAssertionEngine()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0],
            'active': [True, False, True]
        })

    def test_equals_assertion_single_value(self, assertion_engine):
        """Test equals assertion with single value."""
        data = pd.DataFrame({'result': [42]})

        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            data,
            42
        )

        assert result.passed is True
        assert result.actual == 42
        assert result.expected == 42

    def test_equals_assertion_single_value_failure(self, assertion_engine):
        """Test equals assertion failure with single value."""
        data = pd.DataFrame({'result': [42]})

        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            data,
            100
        )

        assert result.passed is False
        assert result.actual == 42
        assert result.expected == 100

    def test_equals_assertion_with_tolerance(self, assertion_engine):
        """Test equals assertion with numeric tolerance."""
        data = pd.DataFrame({'result': [42.01]})

        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            data,
            42.0,
            tolerance=0.1
        )

        assert result.passed is True
        assert result.tolerance == 0.1

    def test_equals_assertion_list_of_records(self, assertion_engine, sample_dataframe):
        """Test equals assertion with list of records."""
        expected = [
            {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 50000.0, 'active': True},
            {'id': 2, 'name': 'Bob', 'age': 30, 'salary': 60000.0, 'active': False},
            {'id': 3, 'name': 'Charlie', 'age': 35, 'salary': 70000.0, 'active': True}
        ]

        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            sample_dataframe,
            expected
        )

        assert result.passed is True

    def test_equals_assertion_ignore_order(self, assertion_engine):
        """Test equals assertion ignoring record order."""
        data = pd.DataFrame({
            'id': [2, 1, 3],
            'name': ['Bob', 'Alice', 'Charlie']
        })

        expected = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'},
            {'id': 3, 'name': 'Charlie'}
        ]

        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            data,
            expected,
            ignore_order=True
        )

        assert result.passed is True

    def test_not_equals_assertion(self, assertion_engine):
        """Test not equals assertion."""
        data = pd.DataFrame({'result': [42]})

        result = assertion_engine.execute_assertion(
            AssertionType.NOT_EQUALS,
            data,
            100
        )

        assert result.passed is True

    def test_contains_assertion_string_search(self, assertion_engine, sample_dataframe):
        """Test contains assertion with string search."""
        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            sample_dataframe,
            'Alice'
        )

        assert result.passed is True

    def test_contains_assertion_record_match(self, assertion_engine, sample_dataframe):
        """Test contains assertion with record matching."""
        expected_record = {'name': 'Bob', 'age': 30}

        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            sample_dataframe,
            expected_record
        )

        assert result.passed is True

    def test_contains_assertion_value_search(self, assertion_engine, sample_dataframe):
        """Test contains assertion with value search."""
        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            sample_dataframe,
            25
        )

        assert result.passed is True

    def test_not_contains_assertion(self, assertion_engine, sample_dataframe):
        """Test not contains assertion."""
        result = assertion_engine.execute_assertion(
            AssertionType.NOT_CONTAINS,
            sample_dataframe,
            'NonExistent'
        )

        assert result.passed is True

    def test_empty_assertion_with_empty_data(self, assertion_engine):
        """Test empty assertion with empty DataFrame."""
        empty_data = pd.DataFrame()

        result = assertion_engine.execute_assertion(
            AssertionType.EMPTY,
            empty_data,
            None
        )

        assert result.passed is True

    def test_empty_assertion_with_data(self, assertion_engine, sample_dataframe):
        """Test empty assertion with non-empty DataFrame."""
        result = assertion_engine.execute_assertion(
            AssertionType.EMPTY,
            sample_dataframe,
            None
        )

        assert result.passed is False

    def test_not_empty_assertion(self, assertion_engine, sample_dataframe):
        """Test not empty assertion."""
        result = assertion_engine.execute_assertion(
            AssertionType.NOT_EMPTY,
            sample_dataframe,
            None
        )

        assert result.passed is True

    def test_row_count_assertion(self, assertion_engine, sample_dataframe):
        """Test row count assertion."""
        result = assertion_engine.execute_assertion(
            AssertionType.ROW_COUNT,
            sample_dataframe,
            3
        )

        assert result.passed is True
        assert result.actual == 3

    def test_row_count_assertion_with_tolerance(self, assertion_engine, sample_dataframe):
        """Test row count assertion with tolerance."""
        result = assertion_engine.execute_assertion(
            AssertionType.ROW_COUNT,
            sample_dataframe,
            5,
            tolerance=2
        )

        assert result.passed is True

    def test_column_count_assertion(self, assertion_engine, sample_dataframe):
        """Test column count assertion."""
        result = assertion_engine.execute_assertion(
            AssertionType.COLUMN_COUNT,
            sample_dataframe,
            5
        )

        assert result.passed is True
        assert result.actual == 5

    def test_schema_match_assertion_dict(self, assertion_engine, sample_dataframe):
        """Test schema match assertion with dictionary."""
        expected_schema = {
            'id': 'int64',
            'name': 'object',
            'age': 'int64',
            'salary': 'float64',
            'active': 'bool'
        }

        result = assertion_engine.execute_assertion(
            AssertionType.SCHEMA_MATCH,
            sample_dataframe,
            expected_schema
        )

        assert result.passed is True

    def test_schema_match_assertion_columns(self, assertion_engine, sample_dataframe):
        """Test schema match assertion with column names."""
        expected_columns = ['id', 'name', 'age', 'salary', 'active']

        result = assertion_engine.execute_assertion(
            AssertionType.SCHEMA_MATCH,
            sample_dataframe,
            expected_columns
        )

        assert result.passed is True

    def test_schema_match_assertion_ignore_order(self, assertion_engine, sample_dataframe):
        """Test schema match assertion ignoring column order."""
        expected_columns = ['active', 'salary', 'age', 'name', 'id']

        result = assertion_engine.execute_assertion(
            AssertionType.SCHEMA_MATCH,
            sample_dataframe,
            expected_columns,
            ignore_order=True
        )

        assert result.passed is True

    def test_custom_assertion_simple(self, assertion_engine, sample_dataframe):
        """Test custom assertion with simple function."""
        custom_function = """
# Check if all ages are above 20
result = {
    'passed': all(data['age'] > 20),
    'actual': data['age'].tolist(),
    'message': 'All ages should be above 20'
}
"""

        result = assertion_engine.execute_assertion(
            AssertionType.CUSTOM,
            sample_dataframe,
            None,
            custom_function=custom_function
        )

        assert result.passed is True

    def test_custom_assertion_with_expected_value(self, assertion_engine, sample_dataframe):
        """Test custom assertion with expected value."""
        custom_function = """
# Check if average age matches expected
avg_age = data['age'].mean()
result = {
    'passed': abs(avg_age - expected) < 1,
    'actual': avg_age,
    'message': f'Average age is {avg_age}, expected {expected}'
}
"""

        result = assertion_engine.execute_assertion(
            AssertionType.CUSTOM,
            sample_dataframe,
            30,  # Expected average age
            custom_function=custom_function
        )

        assert result.passed is True

    def test_custom_assertion_error_handling(self, assertion_engine, sample_dataframe):
        """Test custom assertion error handling."""
        custom_function = """
# Intentionally cause an error
result = data.nonexistent_method()
"""

        result = assertion_engine.execute_assertion(
            AssertionType.CUSTOM,
            sample_dataframe,
            None,
            custom_function=custom_function
        )

        assert result.passed is False
        assert "Custom assertion failed" in result.message

    def test_statistical_assertion_mean(self, assertion_engine, sample_dataframe):
        """Test statistical assertion for mean."""
        assertion_config = {
            'statistical_type': 'mean',
            'column': 'age',
            'expected_value': 30.0,
            'tolerance': 0.1
        }

        result = assertion_engine.validate_statistical_assertion(
            sample_dataframe,
            assertion_config
        )

        assert result.passed is True
        assert result.actual == 30.0

    def test_statistical_assertion_median(self, assertion_engine, sample_dataframe):
        """Test statistical assertion for median."""
        assertion_config = {
            'statistical_type': 'median',
            'column': 'salary',
            'expected_value': 60000.0,
            'tolerance': 1.0
        }

        result = assertion_engine.validate_statistical_assertion(
            sample_dataframe,
            assertion_config
        )

        assert result.passed is True

    def test_statistical_assertion_count(self, assertion_engine, sample_dataframe):
        """Test statistical assertion for count."""
        assertion_config = {
            'statistical_type': 'count',
            'column': 'id',
            'expected_value': 3,
            'tolerance': 0
        }

        result = assertion_engine.validate_statistical_assertion(
            sample_dataframe,
            assertion_config
        )

        assert result.passed is True

    def test_statistical_assertion_missing_column(self, assertion_engine, sample_dataframe):
        """Test statistical assertion with missing column."""
        assertion_config = {
            'statistical_type': 'mean',
            'column': 'nonexistent',
            'expected_value': 100,
            'tolerance': 1.0
        }

        result = assertion_engine.validate_statistical_assertion(
            sample_dataframe,
            assertion_config
        )

        assert result.passed is False
        assert "not found in data" in result.message

    def test_statistical_assertion_invalid_type(self, assertion_engine, sample_dataframe):
        """Test statistical assertion with invalid type."""
        assertion_config = {
            'statistical_type': 'invalid_stat',
            'column': 'age',
            'expected_value': 30,
            'tolerance': 1.0
        }

        result = assertion_engine.validate_statistical_assertion(
            sample_dataframe,
            assertion_config
        )

        assert result.passed is False
        assert "Unknown statistical type" in result.message

    def test_assertion_with_none_data(self, assertion_engine):
        """Test assertions with None data."""
        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            None,
            None
        )

        assert result.passed is True

    def test_assertion_with_none_data_expecting_value(self, assertion_engine):
        """Test assertions with None data expecting a value."""
        result = assertion_engine.execute_assertion(
            AssertionType.EQUALS,
            None,
            42
        )

        assert result.passed is False

    def test_unknown_assertion_type(self, assertion_engine, sample_dataframe):
        """Test handling of unknown assertion type."""
        # Create a mock assertion type that doesn't exist
        unknown_type = "UNKNOWN_ASSERTION"

        result = assertion_engine.execute_assertion(
            unknown_type,
            sample_dataframe,
            None
        )

        assert result.passed is False
        assert "Unknown assertion type" in result.message

    def test_assertion_result_details(self, assertion_engine, sample_dataframe):
        """Test that assertion results include proper details."""
        result = assertion_engine.execute_assertion(
            AssertionType.ROW_COUNT,
            sample_dataframe,
            5,
            tolerance=2
        )

        assert result.details is not None
        assert 'tolerance' in result.details
        assert 'difference' in result.details
        assert result.details['tolerance'] == 2
        assert result.details['difference'] == -2  # 3 actual - 5 expected

    def test_record_comparison_with_tolerance(self, assertion_engine):
        """Test record comparison with tolerance for numeric values."""
        expected = {'value': 10.0, 'name': 'test'}
        actual = {'value': 10.05, 'name': 'test'}

        # Test with sufficient tolerance
        result = assertion_engine._compare_records(expected, actual, tolerance=0.1)
        assert result is True

        # Test with insufficient tolerance
        result = assertion_engine._compare_records(expected, actual, tolerance=0.01)
        assert result is False

    def test_record_comparison_different_keys(self, assertion_engine):
        """Test record comparison with different keys."""
        expected = {'a': 1, 'b': 2}
        actual = {'a': 1, 'c': 2}

        result = assertion_engine._compare_records(expected, actual)
        assert result is False


class TestAssertionEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def assertion_engine(self):
        return SQLTestAssertionEngine()

    def test_assertion_with_nan_values(self, assertion_engine):
        """Test assertions with NaN values."""
        data = pd.DataFrame({'value': [1, np.nan, 3]})

        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            data,
            1
        )

        assert result.passed is True

    def test_assertion_with_mixed_types(self, assertion_engine):
        """Test assertions with mixed data types."""
        data = pd.DataFrame({
            'mixed': [1, 'string', 3.14, True, None]
        })

        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            data,
            'string'
        )

        assert result.passed is True

    def test_large_dataset_performance(self, assertion_engine):
        """Test assertion performance with large datasets."""
        # Create a large DataFrame
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000)
        })

        import time
        start_time = time.time()

        result = assertion_engine.execute_assertion(
            AssertionType.ROW_COUNT,
            large_data,
            10000
        )

        execution_time = time.time() - start_time

        assert result.passed is True
        assert execution_time < 1.0  # Should complete within 1 second

    def test_unicode_string_assertions(self, assertion_engine):
        """Test assertions with Unicode strings."""
        data = pd.DataFrame({
            'unicode_text': ['Hello 世界', 'こんにちは', '🌟⭐️✨']
        })

        result = assertion_engine.execute_assertion(
            AssertionType.CONTAINS,
            data,
            '世界'
        )

        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])