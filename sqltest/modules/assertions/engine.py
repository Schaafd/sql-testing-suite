"""Assertion execution engine for SQLTest Pro."""

import re
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Dict, Union
from datetime import datetime
import time
import hashlib

from ...exceptions import AssertionError as SQLTestAssertionError, DatabaseError
from .models import (
    Assertion,
    AssertionType,
    AssertionResult,
    AssertionLevel,
    QueryAssertionResult,
    TableAssertionResult
)


class AssertionEngine:
    """Core engine for executing assertions on SQL results."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize the assertion engine.
        
        Args:
            strict_mode: If True, stop on first assertion failure
        """
        self.strict_mode = strict_mode
        self.executors = {
            AssertionType.EQUALS: self._execute_equals,
            AssertionType.NOT_EQUALS: self._execute_not_equals,
            AssertionType.GREATER_THAN: self._execute_greater_than,
            AssertionType.LESS_THAN: self._execute_less_than,
            AssertionType.GREATER_EQUAL: self._execute_greater_equal,
            AssertionType.LESS_EQUAL: self._execute_less_equal,
            AssertionType.CONTAINS: self._execute_contains,
            AssertionType.NOT_CONTAINS: self._execute_not_contains,
            AssertionType.STARTS_WITH: self._execute_starts_with,
            AssertionType.ENDS_WITH: self._execute_ends_with,
            AssertionType.MATCHES_REGEX: self._execute_matches_regex,
            AssertionType.IS_NULL: self._execute_is_null,
            AssertionType.IS_NOT_NULL: self._execute_is_not_null,
            AssertionType.IS_EMPTY: self._execute_is_empty,
            AssertionType.IS_NOT_EMPTY: self._execute_is_not_empty,
            AssertionType.HAS_LENGTH: self._execute_has_length,
            AssertionType.HAS_MIN_LENGTH: self._execute_has_min_length,
            AssertionType.HAS_MAX_LENGTH: self._execute_has_max_length,
            AssertionType.IN_RANGE: self._execute_in_range,
            AssertionType.NOT_IN_RANGE: self._execute_not_in_range,
            AssertionType.IS_UNIQUE: self._execute_is_unique,
            AssertionType.HAS_DUPLICATES: self._execute_has_duplicates,
            AssertionType.ROW_COUNT: self._execute_row_count,
            AssertionType.COLUMN_COUNT: self._execute_column_count,
            AssertionType.SCHEMA_MATCHES: self._execute_schema_matches,
            AssertionType.CUSTOM: self._execute_custom
        }
    
    def execute_assertions_on_query_result(
        self,
        query: str,
        result_data: pd.DataFrame,
        assertions: List[Assertion],
        execution_time_ms: float = 0.0
    ) -> QueryAssertionResult:
        """Execute assertions on query result data.
        
        Args:
            query: SQL query that produced the results
            result_data: DataFrame containing query results
            assertions: List of assertions to execute
            execution_time_ms: Query execution time in milliseconds
            
        Returns:
            QueryAssertionResult with all assertion results
            
        Raises:
            SQLTestAssertionError: If assertion fails in strict mode
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        assertion_results = []
        total_start_time = time.time()
        
        for assertion in assertions:
            try:
                start_time = time.time()
                result = self._execute_assertion(assertion, result_data)
                end_time = time.time()
                result.execution_time_ms = (end_time - start_time) * 1000
                
                assertion_results.append(result)
                
                if not result.passed and result.level == AssertionLevel.ERROR and self.strict_mode:
                    raise SQLTestAssertionError(
                        f"Assertion '{assertion.name}' failed: {result.message}"
                    )
                    
            except Exception as e:
                if not isinstance(e, SQLTestAssertionError) and self.strict_mode:
                    raise SQLTestAssertionError(
                        f"Failed to execute assertion '{assertion.name}': {str(e)}"
                    ) from e
                
                # Create error result for failed assertion execution
                error_result = AssertionResult(
                    assertion_name=assertion.name,
                    assertion_type=assertion.assertion_type,
                    passed=False,
                    level=assertion.level,
                    message=f"Assertion execution failed: {str(e)}",
                    context={"error": str(e)}
                )
                assertion_results.append(error_result)
        
        total_end_time = time.time()
        total_execution_time_ms = (total_end_time - total_start_time) * 1000
        
        return QueryAssertionResult(
            query=query,
            query_hash=query_hash,
            assertions=assertion_results,
            total_execution_time_ms=total_execution_time_ms + execution_time_ms
        )
    
    def execute_assertions_on_table(
        self,
        table_name: str,
        database_name: str,
        table_data: pd.DataFrame,
        assertions: List[Assertion],
        schema_name: Optional[str] = None,
        execution_time_ms: float = 0.0
    ) -> TableAssertionResult:
        """Execute assertions on table data.
        
        Args:
            table_name: Name of the table
            database_name: Name of the database
            table_data: DataFrame containing table data
            assertions: List of assertions to execute
            schema_name: Optional schema name
            execution_time_ms: Data fetch execution time in milliseconds
            
        Returns:
            TableAssertionResult with all assertion results
        """
        assertion_results = []
        total_start_time = time.time()
        
        for assertion in assertions:
            try:
                start_time = time.time()
                result = self._execute_assertion(assertion, table_data)
                end_time = time.time()
                result.execution_time_ms = (end_time - start_time) * 1000
                
                assertion_results.append(result)
                
                if not result.passed and result.level == AssertionLevel.ERROR and self.strict_mode:
                    raise SQLTestAssertionError(
                        f"Table assertion '{assertion.name}' failed: {result.message}"
                    )
                    
            except Exception as e:
                if not isinstance(e, SQLTestAssertionError) and self.strict_mode:
                    raise SQLTestAssertionError(
                        f"Failed to execute table assertion '{assertion.name}': {str(e)}"
                    ) from e
                
                error_result = AssertionResult(
                    assertion_name=assertion.name,
                    assertion_type=assertion.assertion_type,
                    passed=False,
                    level=assertion.level,
                    message=f"Assertion execution failed: {str(e)}",
                    context={"error": str(e)}
                )
                assertion_results.append(error_result)
        
        total_end_time = time.time()
        total_execution_time_ms = (total_end_time - total_start_time) * 1000
        
        return TableAssertionResult(
            table_name=table_name,
            database_name=database_name,
            schema_name=schema_name,
            assertions=assertion_results,
            total_execution_time_ms=total_execution_time_ms + execution_time_ms
        )
    
    def _execute_assertion(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute a single assertion on data."""
        executor = self.executors.get(assertion.assertion_type)
        if not executor:
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=False,
                level=assertion.level,
                message=f"Unknown assertion type: {assertion.assertion_type}"
            )
        
        return executor(assertion, data)
    
    # Basic comparison assertions
    def _execute_equals(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute equals assertion."""
        expected_value = assertion.parameters.get("expected_value")
        
        # Handle different data contexts
        if len(data) == 1 and len(data.columns) == 1:
            # Single value comparison
            actual_value = data.iloc[0, 0]
            passed = actual_value == expected_value
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=expected_value,
                actual=actual_value
            )
        else:
            # For multi-row/column data, check if all values equal expected
            matches = 0
            total = 0
            
            for col in data.columns:
                for val in data[col]:
                    total += 1
                    if val == expected_value:
                        matches += 1
            
            passed = matches == total if total > 0 else False
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=expected_value,
                actual=f"{matches}/{total} values match",
                context={"matches": matches, "total": total}
            )
    
    def _execute_not_equals(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute not equals assertion."""
        expected_value = assertion.parameters.get("expected_value")
        
        if len(data) == 1 and len(data.columns) == 1:
            actual_value = data.iloc[0, 0]
            passed = actual_value != expected_value
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"not {expected_value}",
                actual=actual_value
            )
        else:
            # For multi-value data, check that no values equal the expected value
            matches = 0
            total = 0
            
            for col in data.columns:
                for val in data[col]:
                    total += 1
                    if val == expected_value:
                        matches += 1
            
            passed = matches == 0
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"no values equal to {expected_value}",
                actual=f"{matches}/{total} values match",
                context={"matches": matches, "total": total}
            )
    
    def _execute_greater_than(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute greater than assertion."""
        threshold = assertion.parameters.get("expected_value")
        
        if len(data) == 1 and len(data.columns) == 1:
            actual_value = data.iloc[0, 0]
            try:
                passed = float(actual_value) > float(threshold)
            except (ValueError, TypeError):
                passed = False
                
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"> {threshold}",
                actual=actual_value
            )
        else:
            # Check all numeric values are greater than threshold
            valid_comparisons = 0
            passed_comparisons = 0
            
            for col in data.columns:
                for val in data[col]:
                    if pd.notna(val):
                        try:
                            if float(val) > float(threshold):
                                passed_comparisons += 1
                            valid_comparisons += 1
                        except (ValueError, TypeError):
                            continue
            
            passed = passed_comparisons == valid_comparisons if valid_comparisons > 0 else False
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"all values > {threshold}",
                actual=f"{passed_comparisons}/{valid_comparisons} values pass",
                context={"passed": passed_comparisons, "total": valid_comparisons}
            )
    
    def _execute_less_than(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute less than assertion."""
        threshold = assertion.parameters.get("expected_value")
        
        if len(data) == 1 and len(data.columns) == 1:
            actual_value = data.iloc[0, 0]
            try:
                passed = float(actual_value) < float(threshold)
            except (ValueError, TypeError):
                passed = False
                
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"< {threshold}",
                actual=actual_value
            )
        else:
            valid_comparisons = 0
            passed_comparisons = 0
            
            for col in data.columns:
                for val in data[col]:
                    if pd.notna(val):
                        try:
                            if float(val) < float(threshold):
                                passed_comparisons += 1
                            valid_comparisons += 1
                        except (ValueError, TypeError):
                            continue
            
            passed = passed_comparisons == valid_comparisons if valid_comparisons > 0 else False
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"all values < {threshold}",
                actual=f"{passed_comparisons}/{valid_comparisons} values pass",
                context={"passed": passed_comparisons, "total": valid_comparisons}
            )
    
    def _execute_greater_equal(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute greater than or equal assertion.""" 
        threshold = assertion.parameters.get("expected_value")
        
        if len(data) == 1 and len(data.columns) == 1:
            actual_value = data.iloc[0, 0]
            try:
                passed = float(actual_value) >= float(threshold)
            except (ValueError, TypeError):
                passed = False
                
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f">= {threshold}",
                actual=actual_value
            )
        else:
            valid_comparisons = 0
            passed_comparisons = 0
            
            for col in data.columns:
                for val in data[col]:
                    if pd.notna(val):
                        try:
                            if float(val) >= float(threshold):
                                passed_comparisons += 1
                            valid_comparisons += 1
                        except (ValueError, TypeError):
                            continue
            
            passed = passed_comparisons == valid_comparisons if valid_comparisons > 0 else False
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"all values >= {threshold}",
                actual=f"{passed_comparisons}/{valid_comparisons} values pass",
                context={"passed": passed_comparisons, "total": valid_comparisons}
            )
    
    def _execute_less_equal(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute less than or equal assertion."""
        threshold = assertion.parameters.get("expected_value")
        
        if len(data) == 1 and len(data.columns) == 1:
            actual_value = data.iloc[0, 0]
            try:
                passed = float(actual_value) <= float(threshold)
            except (ValueError, TypeError):
                passed = False
                
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"<= {threshold}",
                actual=actual_value
            )
        else:
            valid_comparisons = 0
            passed_comparisons = 0
            
            for col in data.columns:
                for val in data[col]:
                    if pd.notna(val):
                        try:
                            if float(val) <= float(threshold):
                                passed_comparisons += 1
                            valid_comparisons += 1
                        except (ValueError, TypeError):
                            continue
            
            passed = passed_comparisons == valid_comparisons if valid_comparisons > 0 else False
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"all values <= {threshold}",
                actual=f"{passed_comparisons}/{valid_comparisons} values pass",
                context={"passed": passed_comparisons, "total": valid_comparisons}
            )
    
    # String assertions
    def _execute_contains(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute contains assertion."""
        expected_substring = assertion.parameters.get("expected_value")
        case_sensitive = assertion.parameters.get("case_sensitive", True)
        
        matches = 0
        total = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    val_str = str(val)
                    expected_str = str(expected_substring)
                    
                    if not case_sensitive:
                        val_str = val_str.lower()
                        expected_str = expected_str.lower()
                    
                    if expected_str in val_str:
                        matches += 1
                    total += 1
        
        passed = matches > 0 if total > 0 else False
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"contains '{expected_substring}'",
            actual=f"{matches}/{total} values contain substring",
            context={"matches": matches, "total": total}
        )
    
    def _execute_not_contains(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute not contains assertion."""
        expected_substring = assertion.parameters.get("expected_value")
        case_sensitive = assertion.parameters.get("case_sensitive", True)
        
        matches = 0
        total = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    val_str = str(val)
                    expected_str = str(expected_substring)
                    
                    if not case_sensitive:
                        val_str = val_str.lower()
                        expected_str = expected_str.lower()
                    
                    if expected_str not in val_str:
                        matches += 1
                    total += 1
        
        passed = matches == total if total > 0 else True
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"does not contain '{expected_substring}'",
            actual=f"{matches}/{total} values do not contain substring",
            context={"matches": matches, "total": total}
        )
    
    def _execute_starts_with(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute starts with assertion."""
        expected_prefix = assertion.parameters.get("expected_value")
        case_sensitive = assertion.parameters.get("case_sensitive", True)
        
        matches = 0
        total = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    val_str = str(val)
                    expected_str = str(expected_prefix)
                    
                    if not case_sensitive:
                        val_str = val_str.lower()
                        expected_str = expected_str.lower()
                    
                    if val_str.startswith(expected_str):
                        matches += 1
                    total += 1
        
        passed = matches > 0 if total > 0 else False
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"starts with '{expected_prefix}'",
            actual=f"{matches}/{total} values start with prefix",
            context={"matches": matches, "total": total}
        )
    
    def _execute_ends_with(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute ends with assertion."""
        expected_suffix = assertion.parameters.get("expected_value")
        case_sensitive = assertion.parameters.get("case_sensitive", True)
        
        matches = 0
        total = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    val_str = str(val)
                    expected_str = str(expected_suffix)
                    
                    if not case_sensitive:
                        val_str = val_str.lower()
                        expected_str = expected_str.lower()
                    
                    if val_str.endswith(expected_str):
                        matches += 1
                    total += 1
        
        passed = matches > 0 if total > 0 else False
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"ends with '{expected_suffix}'",
            actual=f"{matches}/{total} values end with suffix",
            context={"matches": matches, "total": total}
        )
    
    def _execute_matches_regex(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute regex match assertion."""
        pattern = assertion.parameters.get("pattern")
        flags = assertion.parameters.get("flags", 0)
        
        if not pattern:
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=False,
                level=assertion.level,
                message="Regex pattern not provided"
            )
        
        compiled_pattern = re.compile(pattern, flags)
        matches = 0
        total = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    val_str = str(val)
                    if compiled_pattern.match(val_str):
                        matches += 1
                    total += 1
        
        passed = matches > 0 if total > 0 else False
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"matches regex '{pattern}'",
            actual=f"{matches}/{total} values match pattern",
            context={"matches": matches, "total": total, "pattern": pattern}
        )
    
    # Null/empty assertions
    def _execute_is_null(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute is null assertion."""
        null_count = data.isnull().sum().sum()
        total_count = data.size
        
        passed = null_count > 0
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="contains null values",
            actual=f"{null_count}/{total_count} null values",
            context={"null_count": null_count, "total_count": total_count}
        )
    
    def _execute_is_not_null(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute is not null assertion."""
        null_count = data.isnull().sum().sum()
        total_count = data.size
        
        passed = null_count == 0
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="no null values",
            actual=f"{null_count}/{total_count} null values",
            context={"null_count": null_count, "total_count": total_count}
        )
    
    def _execute_is_empty(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute is empty assertion."""
        passed = len(data) == 0
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="empty result set",
            actual=f"{len(data)} rows",
            context={"row_count": len(data)}
        )
    
    def _execute_is_not_empty(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute is not empty assertion."""
        passed = len(data) > 0
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="non-empty result set",
            actual=f"{len(data)} rows",
            context={"row_count": len(data)}
        )
    
    # Length assertions
    def _execute_has_length(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute has specific length assertion."""
        expected_length = assertion.parameters.get("expected_length")
        
        if len(data) == 1 and len(data.columns) == 1:
            # Single value length check
            actual_value = data.iloc[0, 0]
            actual_length = len(str(actual_value)) if pd.notna(actual_value) else 0
            passed = actual_length == expected_length
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"length = {expected_length}",
                actual=f"length = {actual_length}",
                context={"expected_length": expected_length, "actual_length": actual_length}
            )
        else:
            # Result set row count check
            actual_length = len(data)
            passed = actual_length == expected_length
            
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=passed,
                level=assertion.level,
                message=assertion.failure_message if not passed else assertion.expected_message,
                expected=f"{expected_length} rows",
                actual=f"{actual_length} rows",
                context={"expected_length": expected_length, "actual_length": actual_length}
            )
    
    def _execute_has_min_length(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute has minimum length assertion."""
        min_length = assertion.parameters.get("min_length")
        actual_length = len(data)
        passed = actual_length >= min_length
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"at least {min_length} rows",
            actual=f"{actual_length} rows",
            context={"min_length": min_length, "actual_length": actual_length}
        )
    
    def _execute_has_max_length(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute has maximum length assertion."""
        max_length = assertion.parameters.get("max_length")
        actual_length = len(data)
        passed = actual_length <= max_length
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"at most {max_length} rows",
            actual=f"{actual_length} rows",
            context={"max_length": max_length, "actual_length": actual_length}
        )
    
    # Range assertions
    def _execute_in_range(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute in range assertion."""
        min_val = assertion.parameters.get("min_value")
        max_val = assertion.parameters.get("max_value")
        inclusive = assertion.parameters.get("inclusive", True)
        
        in_range_count = 0
        total_count = 0
        
        for col in data.columns:
            for val in data[col]:
                if pd.notna(val):
                    try:
                        num_val = float(val)
                        in_range = True
                        
                        if min_val is not None:
                            if inclusive:
                                in_range = in_range and num_val >= min_val
                            else:
                                in_range = in_range and num_val > min_val
                        
                        if max_val is not None:
                            if inclusive:
                                in_range = in_range and num_val <= max_val
                            else:
                                in_range = in_range and num_val < max_val
                        
                        if in_range:
                            in_range_count += 1
                        total_count += 1
                    except (ValueError, TypeError):
                        continue
        
        passed = in_range_count == total_count if total_count > 0 else False
        
        range_desc = f"[{min_val}, {max_val}]" if inclusive else f"({min_val}, {max_val})"
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"all values in range {range_desc}",
            actual=f"{in_range_count}/{total_count} values in range",
            context={"in_range": in_range_count, "total": total_count, "range": range_desc}
        )
    
    def _execute_not_in_range(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute not in range assertion."""
        # Reuse in_range logic but invert the result
        result = self._execute_in_range(assertion, data)
        result.assertion_type = AssertionType.NOT_IN_RANGE
        result.passed = not result.passed
        
        # Update expected message
        range_desc = result.context.get("range", "range")
        result.expected = f"all values outside range {range_desc}"
        
        return result
    
    # Uniqueness assertions
    def _execute_is_unique(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute is unique assertion."""
        duplicate_count = 0
        total_count = 0
        
        for col in data.columns:
            col_data = data[col].dropna()
            duplicates = col_data.duplicated().sum()
            duplicate_count += duplicates
            total_count += len(col_data)
        
        passed = duplicate_count == 0
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="all values unique",
            actual=f"{duplicate_count} duplicate values out of {total_count}",
            context={"duplicates": duplicate_count, "total": total_count}
        )
    
    def _execute_has_duplicates(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute has duplicates assertion."""
        result = self._execute_is_unique(assertion, data)
        result.assertion_type = AssertionType.HAS_DUPLICATES
        result.passed = not result.passed
        result.expected = "contains duplicate values"
        
        return result
    
    # Count assertions
    def _execute_row_count(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute row count assertion."""
        expected_count = assertion.parameters.get("expected_count")
        min_count = assertion.parameters.get("min_count")
        max_count = assertion.parameters.get("max_count")
        actual_count = len(data)
        
        passed = True
        expectations = []
        
        if expected_count is not None:
            passed = passed and actual_count == expected_count
            expectations.append(f"exactly {expected_count} rows")
        
        if min_count is not None:
            passed = passed and actual_count >= min_count
            expectations.append(f"at least {min_count} rows")
        
        if max_count is not None:
            passed = passed and actual_count <= max_count
            expectations.append(f"at most {max_count} rows")
        
        expected_desc = " and ".join(expectations) if expectations else "row count check"
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=expected_desc,
            actual=f"{actual_count} rows",
            context={
                "expected_count": expected_count,
                "min_count": min_count,
                "max_count": max_count,
                "actual_count": actual_count
            }
        )
    
    def _execute_column_count(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute column count assertion."""
        expected_count = assertion.parameters.get("expected_count")
        min_count = assertion.parameters.get("min_count")
        max_count = assertion.parameters.get("max_count")
        actual_count = len(data.columns)
        
        passed = True
        expectations = []
        
        if expected_count is not None:
            passed = passed and actual_count == expected_count
            expectations.append(f"exactly {expected_count} columns")
        
        if min_count is not None:
            passed = passed and actual_count >= min_count
            expectations.append(f"at least {min_count} columns")
        
        if max_count is not None:
            passed = passed and actual_count <= max_count
            expectations.append(f"at most {max_count} columns")
        
        expected_desc = " and ".join(expectations) if expectations else "column count check"
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=expected_desc,
            actual=f"{actual_count} columns",
            context={
                "expected_count": expected_count,
                "min_count": min_count,
                "max_count": max_count,
                "actual_count": actual_count
            }
        )
    
    def _execute_schema_matches(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute schema matches assertion."""
        expected_columns = assertion.parameters.get("expected_columns", [])
        expected_types = assertion.parameters.get("expected_types", {})
        strict_order = assertion.parameters.get("strict_order", False)
        allow_extra_columns = assertion.parameters.get("allow_extra_columns", True)
        
        actual_columns = list(data.columns)
        passed = True
        issues = []
        
        # Check column presence
        if strict_order:
            # Check exact order
            min_len = min(len(expected_columns), len(actual_columns))
            for i in range(min_len):
                if expected_columns[i] != actual_columns[i]:
                    passed = False
                    issues.append(f"Column {i}: expected '{expected_columns[i]}', got '{actual_columns[i]}'")
        else:
            # Check presence regardless of order
            missing_columns = set(expected_columns) - set(actual_columns)
            if missing_columns:
                passed = False
                issues.append(f"Missing columns: {list(missing_columns)}")
        
        # Check extra columns
        if not allow_extra_columns:
            extra_columns = set(actual_columns) - set(expected_columns)
            if extra_columns:
                passed = False
                issues.append(f"Extra columns: {list(extra_columns)}")
        
        # Check data types if provided
        for col, expected_type in expected_types.items():
            if col in actual_columns:
                actual_type = str(data[col].dtype)
                if expected_type.lower() not in actual_type.lower():
                    passed = False
                    issues.append(f"Column '{col}': expected type '{expected_type}', got '{actual_type}'")
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=passed,
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected=f"schema with columns: {expected_columns}",
            actual=f"schema with columns: {actual_columns}",
            context={
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "issues": issues,
                "expected_types": expected_types
            }
        )
    
    def _execute_custom(self, assertion: Assertion, data: pd.DataFrame) -> AssertionResult:
        """Execute custom assertion."""
        custom_func = assertion.custom_function
        
        if not custom_func or not callable(custom_func):
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=False,
                level=assertion.level,
                message="Custom function not provided or not callable"
            )
        
        try:
            passed = custom_func(data)
        except Exception as e:
            return AssertionResult(
                assertion_name=assertion.name,
                assertion_type=assertion.assertion_type,
                passed=False,
                level=assertion.level,
                message=f"Custom function failed: {str(e)}",
                context={"error": str(e)}
            )
        
        return AssertionResult(
            assertion_name=assertion.name,
            assertion_type=assertion.assertion_type,
            passed=bool(passed),
            level=assertion.level,
            message=assertion.failure_message if not passed else assertion.expected_message,
            expected="custom condition satisfied",
            actual="custom condition result"
        )
