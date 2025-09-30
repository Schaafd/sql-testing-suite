"""Comprehensive assertion library for SQL unit testing.

Provides rich assertions for:
- Query results and row counts
- Schema validation
- Performance metrics
- Data quality checks
- Custom assertions
"""

import logging
from typing import Any, Dict, List, Optional, Callable
import pandas as pd

logger = logging.getLogger(__name__)


class AssertionError(Exception):
    """Raised when an assertion fails."""
    pass


class SQLAssertions:
    """Collection of assertions for SQL testing."""

    @staticmethod
    def assert_row_count(actual: int, expected: int, message: Optional[str] = None):
        """Assert query returned expected number of rows."""
        if actual != expected:
            msg = message or f"Expected {expected} rows, but got {actual}"
            raise AssertionError(msg)

    @staticmethod
    def assert_row_count_greater_than(actual: int, minimum: int, message: Optional[str] = None):
        """Assert row count exceeds minimum."""
        if actual <= minimum:
            msg = message or f"Expected more than {minimum} rows, but got {actual}"
            raise AssertionError(msg)

    @staticmethod
    def assert_not_empty(row_count: int, message: Optional[str] = None):
        """Assert result set is not empty."""
        if row_count == 0:
            msg = message or "Expected non-empty result set"
            raise AssertionError(msg)

    @staticmethod
    def assert_empty(row_count: int, message: Optional[str] = None):
        """Assert result set is empty."""
        if row_count > 0:
            msg = message or f"Expected empty result set, but got {row_count} rows"
            raise AssertionError(msg)

    @staticmethod
    def assert_columns_exist(columns: List[str], expected_columns: List[str],
                            message: Optional[str] = None):
        """Assert expected columns exist in result."""
        missing = set(expected_columns) - set(columns)
        if missing:
            msg = message or f"Missing columns: {missing}"
            raise AssertionError(msg)

    @staticmethod
    def assert_column_values(df: pd.DataFrame, column: str,
                            expected_values: List[Any],
                            message: Optional[str] = None):
        """Assert column contains expected values."""
        actual_values = df[column].tolist()
        if set(actual_values) != set(expected_values):
            msg = message or f"Column {column} values don't match. Expected: {expected_values}, Got: {actual_values}"
            raise AssertionError(msg)

    @staticmethod
    def assert_no_nulls(df: pd.DataFrame, column: str, message: Optional[str] = None):
        """Assert column contains no NULL values."""
        null_count = df[column].isnull().sum()
        if null_count > 0:
            msg = message or f"Column {column} contains {null_count} NULL values"
            raise AssertionError(msg)

    @staticmethod
    def assert_unique_values(df: pd.DataFrame, column: str, message: Optional[str] = None):
        """Assert all values in column are unique."""
        duplicates = df[column].duplicated().sum()
        if duplicates > 0:
            msg = message or f"Column {column} contains {duplicates} duplicate values"
            raise AssertionError(msg)

    @staticmethod
    def assert_values_in_range(df: pd.DataFrame, column: str,
                               min_value: Any, max_value: Any,
                               message: Optional[str] = None):
        """Assert all values in column are within range."""
        out_of_range = ((df[column] < min_value) | (df[column] > max_value)).sum()
        if out_of_range > 0:
            msg = message or f"Column {column} has {out_of_range} values outside range [{min_value}, {max_value}]"
            raise AssertionError(msg)

    @staticmethod
    def assert_execution_time_under(actual_ms: float, max_ms: float,
                                   message: Optional[str] = None):
        """Assert query execution time is under threshold."""
        if actual_ms > max_ms:
            msg = message or f"Query took {actual_ms:.2f}ms, expected under {max_ms}ms"
            raise AssertionError(msg)

    @staticmethod
    def assert_data_quality_score(score: float, min_score: float,
                                  message: Optional[str] = None):
        """Assert data quality score meets minimum."""
        if score < min_score:
            msg = message or f"Data quality score {score:.2f} below minimum {min_score}"
            raise AssertionError(msg)

    @staticmethod
    def assert_referential_integrity(child_df: pd.DataFrame, child_column: str,
                                    parent_df: pd.DataFrame, parent_column: str,
                                    message: Optional[str] = None):
        """Assert referential integrity between tables."""
        orphans = set(child_df[child_column]) - set(parent_df[parent_column])
        if orphans:
            msg = message or f"Found {len(orphans)} orphaned records"
            raise AssertionError(msg)

    @staticmethod
    def assert_schema_matches(actual_schema: Dict[str, str],
                             expected_schema: Dict[str, str],
                             message: Optional[str] = None):
        """Assert table schema matches expected."""
        if actual_schema != expected_schema:
            differences = {
                k: (actual_schema.get(k), expected_schema.get(k))
                for k in set(actual_schema.keys()) | set(expected_schema.keys())
                if actual_schema.get(k) != expected_schema.get(k)
            }
            msg = message or f"Schema mismatch: {differences}"
            raise AssertionError(msg)

    @staticmethod
    def assert_custom(condition: bool, message: str):
        """Assert custom condition."""
        if not condition:
            raise AssertionError(message)


# Convenience function for quick assertions
def assert_that(value: Any) -> 'AssertionBuilder':
    """Start an assertion chain."""
    return AssertionBuilder(value)


class AssertionBuilder:
    """Fluent assertion builder for readable tests."""

    def __init__(self, value: Any):
        self.value = value
        self._message: Optional[str] = None

    def with_message(self, message: str) -> 'AssertionBuilder':
        """Set custom error message."""
        self._message = message
        return self

    def equals(self, expected: Any) -> 'AssertionBuilder':
        """Assert value equals expected."""
        if self.value != expected:
            msg = self._message or f"Expected {expected}, but got {self.value}"
            raise AssertionError(msg)
        return self

    def not_equals(self, not_expected: Any) -> 'AssertionBuilder':
        """Assert value does not equal."""
        if self.value == not_expected:
            msg = self._message or f"Expected value different from {not_expected}"
            raise AssertionError(msg)
        return self

    def is_greater_than(self, minimum: Any) -> 'AssertionBuilder':
        """Assert value is greater than minimum."""
        if self.value <= minimum:
            msg = self._message or f"Expected value > {minimum}, but got {self.value}"
            raise AssertionError(msg)
        return self

    def is_less_than(self, maximum: Any) -> 'AssertionBuilder':
        """Assert value is less than maximum."""
        if self.value >= maximum:
            msg = self._message or f"Expected value < {maximum}, but got {self.value}"
            raise AssertionError(msg)
        return self

    def is_between(self, min_value: Any, max_value: Any) -> 'AssertionBuilder':
        """Assert value is in range."""
        if not (min_value <= self.value <= max_value):
            msg = self._message or f"Expected value between {min_value} and {max_value}, but got {self.value}"
            raise AssertionError(msg)
        return self

    def is_not_none(self) -> 'AssertionBuilder':
        """Assert value is not None."""
        if self.value is None:
            msg = self._message or "Expected non-None value"
            raise AssertionError(msg)
        return self

    def is_none(self) -> 'AssertionBuilder':
        """Assert value is None."""
        if self.value is not None:
            msg = self._message or f"Expected None, but got {self.value}"
            raise AssertionError(msg)
        return self

    def contains(self, item: Any) -> 'AssertionBuilder':
        """Assert collection contains item."""
        if item not in self.value:
            msg = self._message or f"Expected {self.value} to contain {item}"
            raise AssertionError(msg)
        return self

    def has_length(self, expected_length: int) -> 'AssertionBuilder':
        """Assert collection has expected length."""
        actual_length = len(self.value)
        if actual_length != expected_length:
            msg = self._message or f"Expected length {expected_length}, but got {actual_length}"
            raise AssertionError(msg)
        return self

    def is_empty(self) -> 'AssertionBuilder':
        """Assert collection is empty."""
        if len(self.value) > 0:
            msg = self._message or f"Expected empty collection, but got {len(self.value)} items"
            raise AssertionError(msg)
        return self

    def is_not_empty(self) -> 'AssertionBuilder':
        """Assert collection is not empty."""
        if len(self.value) == 0:
            msg = self._message or "Expected non-empty collection"
            raise AssertionError(msg)
        return self

    def satisfies(self, condition: Callable[[Any], bool]) -> 'AssertionBuilder':
        """Assert value satisfies custom condition."""
        if not condition(self.value):
            msg = self._message or f"Value {self.value} does not satisfy condition"
            raise AssertionError(msg)
        return self