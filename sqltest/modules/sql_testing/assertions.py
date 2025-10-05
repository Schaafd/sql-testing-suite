"""Advanced assertion engine for SQL unit testing framework.

This module provides comprehensive assertion capabilities for SQL test validation
including statistical comparisons, schema validation, and custom assertions.
"""
import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .models import AssertionType

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """Result of an assertion execution."""
    assertion_type: AssertionType
    passed: bool
    expected: Any
    actual: Any
    message: str
    error: Optional[str] = None
    tolerance: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SQLTestAssertionEngine:
    """Enterprise-grade assertion engine for SQL test validation."""

    def __init__(self):
        """Initialize the assertion engine."""
        self.assertion_handlers = {
            AssertionType.EQUALS: self._assert_equals,
            AssertionType.NOT_EQUALS: self._assert_not_equals,
            AssertionType.CONTAINS: self._assert_contains,
            AssertionType.NOT_CONTAINS: self._assert_not_contains,
            AssertionType.EMPTY: self._assert_empty,
            AssertionType.NOT_EMPTY: self._assert_not_empty,
            AssertionType.ROW_COUNT: self._assert_row_count,
            AssertionType.COLUMN_COUNT: self._assert_column_count,
            AssertionType.SCHEMA_MATCH: self._assert_schema_match,
            AssertionType.CUSTOM: self._assert_custom
        }

    def execute_assertion(self,
                         assertion_type: AssertionType,
                         data: Optional[pd.DataFrame],
                         expected: Any,
                         tolerance: Optional[float] = None,
                         ignore_order: bool = False,
                         custom_function: Optional[str] = None,
                         message: Optional[str] = None) -> AssertionResult:
        """Execute an assertion and return detailed results."""

        try:
            handler = self.assertion_handlers.get(assertion_type)
            if not handler:
                return AssertionResult(
                    assertion_type=assertion_type,
                    passed=False,
                    expected=expected,
                    actual=None,
                    message=message or f"Unknown assertion type: {assertion_type}",
                    error=f"No handler for assertion type: {assertion_type}"
                )

            # Execute the assertion
            if assertion_type == AssertionType.CUSTOM:
                result = handler(data, expected, custom_function)
            else:
                result = handler(data, expected, tolerance, ignore_order)

            details = result.get('details')
            return AssertionResult(
                assertion_type=assertion_type,
                passed=bool(result.get('passed')),
                expected=self._to_native(expected),
                actual=self._to_native(result.get('actual')),
                message=message or result.get('message', f"{assertion_type} assertion"),
                tolerance=tolerance,
                details=self._to_native(details)
                if isinstance(details, (dict, list, tuple))
                else details,
            )

        except Exception as e:
            logger.error(f"Assertion execution failed: {e}")
            return AssertionResult(
                assertion_type=assertion_type,
                passed=False,
                expected=expected,
                actual=None,
                message=message or f"{assertion_type} assertion failed",
                error=str(e)
            )

    def _assert_equals(self,
                      data: Optional[pd.DataFrame],
                      expected: Any,
                      tolerance: Optional[float] = None,
                      ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data equals expected value."""
        if data is None:
            return {
                'passed': expected is None,
                'actual': None,
                'message': 'Data is None'
            }

        # Handle different expected types
        if isinstance(expected, (int, float, str)):
            # Single value comparison
            if len(data) == 1 and len(data.columns) == 1:
                actual_value = data.iloc[0, 0]
                if tolerance is not None and isinstance(actual_value, (int, float)):
                    passed = abs(actual_value - expected) <= tolerance
                else:
                    passed = actual_value == expected
                return {
                    'passed': passed,
                    'actual': actual_value,
                    'message': f'Expected {expected}, got {actual_value}'
                }
            else:
                return {
                    'passed': False,
                    'actual': f'{len(data)} rows, {len(data.columns)} columns',
                    'message': f'Expected single value {expected}, got multi-dimensional data'
                }

        elif isinstance(expected, list):
            # List of records comparison
            actual_records = data.to_dict('records')
            if ignore_order:
                # Sort both lists for comparison
                try:
                    expected_sorted = sorted(expected, key=lambda x: str(x))
                    actual_sorted = sorted(actual_records, key=lambda x: str(x))
                    passed = expected_sorted == actual_sorted
                except Exception:
                    # Fallback to set comparison if sorting fails
                    passed = set(str(x) for x in expected) == set(str(x) for x in actual_records)
            else:
                passed = expected == actual_records

            return {
                'passed': passed,
                'actual': actual_records,
                'message': f'Expected {len(expected)} records, got {len(actual_records)} records',
                'details': {
                    'expected_count': len(expected),
                    'actual_count': len(actual_records),
                    'ignore_order': ignore_order
                }
            }

        elif isinstance(expected, dict):
            # Single record comparison
            if len(data) == 1:
                actual_record = data.iloc[0].to_dict()
                passed = self._compare_records(expected, actual_record, tolerance)
                return {
                    'passed': passed,
                    'actual': actual_record,
                    'message': f'Record comparison: {passed}'
                }
            else:
                return {
                    'passed': False,
                    'actual': f'{len(data)} rows',
                    'message': f'Expected single record, got {len(data)} rows'
                }

        else:
            return {
                'passed': False,
                'actual': data.to_dict('records') if not data.empty else [],
                'message': f'Unsupported expected type: {type(expected)}'
            }

    def _assert_not_equals(self,
                          data: Optional[pd.DataFrame],
                          expected: Any,
                          tolerance: Optional[float] = None,
                          ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data does not equal expected value."""
        equals_result = self._assert_equals(data, expected, tolerance, ignore_order)
        return {
            'passed': not equals_result['passed'],
            'actual': equals_result['actual'],
            'message': f'Should not equal {expected}'
        }

    def _assert_contains(self,
                        data: Optional[pd.DataFrame],
                        expected: Any,
                        tolerance: Optional[float] = None,
                        ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data contains expected value."""
        if data is None or data.empty:
            return {
                'passed': False,
                'actual': None,
                'message': 'Cannot check contains on empty data'
            }

        # Convert data to searchable format
        if isinstance(expected, str):
            # String search in all string columns
            string_cols = data.select_dtypes(include=['object']).columns
            found = any(
                data[col].astype(str).str.contains(expected, na=False, regex=False).any()
                for col in string_cols
            )
            return {
                'passed': found,
                'actual': data.to_dict('records'),
                'message': f'String "{expected}" {"found" if found else "not found"} in data'
            }

        elif isinstance(expected, dict):
            # Check if any row contains all key-value pairs from expected
            for _, row in data.iterrows():
                if all(row.get(k) == v for k, v in expected.items()):
                    return {
                        'passed': True,
                        'actual': row.to_dict(),
                        'message': f'Found matching record'
                    }

            return {
                'passed': False,
                'actual': data.to_dict('records'),
                'message': f'No record contains {expected}'
            }

        else:
            # Value search in all columns
            found = data.isin([expected]).any().any()
            return {
                'passed': found,
                'actual': data.to_dict('records'),
                'message': f'Value {expected} {"found" if found else "not found"} in data'
            }

    def _assert_not_contains(self,
                            data: Optional[pd.DataFrame],
                            expected: Any,
                            tolerance: Optional[float] = None,
                            ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data does not contain expected value."""
        contains_result = self._assert_contains(data, expected, tolerance, ignore_order)
        return {
            'passed': not contains_result['passed'],
            'actual': contains_result['actual'],
            'message': f'Should not contain {expected}'
        }

    def _assert_empty(self,
                     data: Optional[pd.DataFrame],
                     expected: Any,
                     tolerance: Optional[float] = None,
                     ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data is empty."""
        is_empty = data is None or data.empty
        return {
            'passed': is_empty,
            'actual': 0 if is_empty else len(data),
            'message': f'Data is {"empty" if is_empty else f"not empty ({len(data)} rows)"}'
        }

    def _assert_not_empty(self,
                         data: Optional[pd.DataFrame],
                         expected: Any,
                         tolerance: Optional[float] = None,
                         ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data is not empty."""
        is_empty = data is None or data.empty
        return {
            'passed': not is_empty,
            'actual': 0 if is_empty else len(data),
            'message': f'Data is {"empty" if is_empty else f"not empty ({len(data)} rows)"}'
        }

    def _assert_row_count(self,
                         data: Optional[pd.DataFrame],
                         expected: Any,
                         tolerance: Optional[float] = None,
                         ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data has expected number of rows."""
        actual_count = 0 if data is None or data.empty else len(data)

        if tolerance is not None:
            passed = abs(actual_count - expected) <= tolerance
        else:
            passed = actual_count == expected

        return {
            'passed': passed,
            'actual': actual_count,
            'message': f'Expected {expected} rows, got {actual_count} rows',
            'details': {
                'tolerance': tolerance,
                'difference': actual_count - expected
            }
        }

    def _assert_column_count(self,
                            data: Optional[pd.DataFrame],
                            expected: Any,
                            tolerance: Optional[float] = None,
                            ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data has expected number of columns."""
        actual_count = 0 if data is None else len(data.columns)
        passed = actual_count == expected

        return {
            'passed': passed,
            'actual': actual_count,
            'message': f'Expected {expected} columns, got {actual_count} columns',
            'details': {
                'column_names': list(data.columns) if data is not None else []
            }
        }

    def _assert_schema_match(self,
                            data: Optional[pd.DataFrame],
                            expected: Any,
                            tolerance: Optional[float] = None,
                            ignore_order: bool = False) -> Dict[str, Any]:
        """Assert that data schema matches expected schema."""
        if data is None:
            return {
                'passed': False,
                'actual': None,
                'message': 'Cannot validate schema on None data'
            }

        actual_schema = {col: str(dtype) for col, dtype in data.dtypes.items()}

        if isinstance(expected, dict):
            # Exact schema match
            passed = actual_schema == expected
            return {
                'passed': passed,
                'actual': actual_schema,
                'message': f'Schema {"matches" if passed else "does not match"}',
                'details': {
                    'expected_schema': expected,
                    'actual_schema': actual_schema,
                    'missing_columns': set(expected.keys()) - set(actual_schema.keys()),
                    'extra_columns': set(actual_schema.keys()) - set(expected.keys())
                }
            }

        elif isinstance(expected, list):
            # Column names match
            actual_columns = list(data.columns)
            if ignore_order:
                passed = set(actual_columns) == set(expected)
            else:
                passed = actual_columns == expected

            return {
                'passed': passed,
                'actual': actual_columns,
                'message': f'Columns {"match" if passed else "do not match"}',
                'details': {
                    'expected_columns': expected,
                    'actual_columns': actual_columns,
                    'ignore_order': ignore_order
                }
            }

        else:
            return {
                'passed': False,
                'actual': actual_schema,
                'message': f'Invalid expected schema type: {type(expected)}'
            }

    def _assert_custom(self,
                      data: Optional[pd.DataFrame],
                      expected: Any,
                      custom_function: Optional[str]) -> Dict[str, Any]:
        """Execute custom assertion function."""
        if not custom_function:
            return {
                'passed': False,
                'actual': None,
                'message': 'No custom function provided'
            }

        try:
            # Create safe execution environment
            namespace = {
                'data': data,
                'expected': expected,
                'pd': pd,
                'np': np,
                'len': len,
                'isinstance': isinstance,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'any': any,
                'all': all,
            }

            # Execute the custom function
            exec(custom_function, namespace)

            # Look for result variable
            if 'result' in namespace:
                result = namespace['result']
                if isinstance(result, dict):
                    return result
                else:
                    return {
                        'passed': bool(result),
                        'actual': data.to_dict('records') if data is not None else None,
                        'message': f'Custom assertion returned: {result}'
                    }
            else:
                return {
                    'passed': False,
                    'actual': None,
                    'message': 'Custom function did not return a result'
                }

        except Exception as e:
            return {
                'passed': False,
                'actual': None,
                'message': f'Custom assertion failed: {str(e)}'
            }

    def _to_native(self, value: Any) -> Any:
        """Convert numpy scalar/bool types to native Python equivalents."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [self._to_native(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_native(v) for v in value)
        if isinstance(value, dict):
            return {k: self._to_native(v) for k, v in value.items()}
        return value

    def _compare_records(self, expected: Dict, actual: Dict, tolerance: Optional[float] = None) -> bool:
        """Compare two records with optional tolerance for numeric values."""
        if set(expected.keys()) != set(actual.keys()):
            return False

        for key, expected_value in expected.items():
            actual_value = actual.get(key)

            if tolerance is not None and isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                if abs(actual_value - expected_value) > tolerance:
                    return False
            else:
                if actual_value != expected_value:
                    return False

        return True

    def validate_statistical_assertion(self,
                                     data: pd.DataFrame,
                                     assertion_config: Dict[str, Any]) -> AssertionResult:
        """Execute statistical assertions on data."""
        stat_type = assertion_config.get('statistical_type')
        column = assertion_config.get('column')
        expected_value = assertion_config.get('expected_value')
        tolerance = assertion_config.get('tolerance', 0.01)

        if data is None or data.empty:
            return AssertionResult(
                assertion_type=AssertionType.CUSTOM,
                passed=False,
                expected=expected_value,
                actual=None,
                message="Cannot perform statistical assertion on empty data"
            )

        if column not in data.columns:
            return AssertionResult(
                assertion_type=AssertionType.CUSTOM,
                passed=False,
                expected=expected_value,
                actual=None,
                message=f"Column '{column}' not found in data"
            )

        try:
            column_data = data[column].dropna()

            if stat_type == 'mean':
                actual_value = column_data.mean()
            elif stat_type == 'median':
                actual_value = column_data.median()
            elif stat_type == 'std':
                actual_value = column_data.std()
            elif stat_type == 'min':
                actual_value = column_data.min()
            elif stat_type == 'max':
                actual_value = column_data.max()
            elif stat_type == 'count':
                actual_value = len(column_data)
            elif stat_type == 'sum':
                actual_value = column_data.sum()
            else:
                return AssertionResult(
                    assertion_type=AssertionType.CUSTOM,
                    passed=False,
                    expected=expected_value,
                    actual=None,
                    message=f"Unknown statistical type: {stat_type}"
                )

            # Compare with tolerance
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                passed = bool(abs(actual_value - expected_value) <= tolerance)
            else:
                passed = bool(actual_value == expected_value)

            return AssertionResult(
                assertion_type=AssertionType.CUSTOM,
                passed=passed,
                expected=expected_value,
                actual=self._to_native(actual_value),
                message=f"Statistical {stat_type}: expected {expected_value}, got {actual_value}",
                tolerance=tolerance,
                details={
                    'statistical_type': stat_type,
                    'column': column,
                    'sample_size': len(column_data)
                }
            )

        except Exception as e:
            return AssertionResult(
                assertion_type=AssertionType.CUSTOM,
                passed=False,
                expected=expected_value,
                actual=None,
                message=f"Statistical assertion failed: {str(e)}",
                error=str(e)
            )
