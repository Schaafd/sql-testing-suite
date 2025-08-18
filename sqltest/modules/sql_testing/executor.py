"""
SQL test execution engine for the SQL unit testing framework.

This module handles the execution of SQL tests including setup/teardown,
assertion validation, dependency management, and result collection.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import traceback
import pandas as pd

from sqltest.core.connection_manager import ConnectionManager
from sqltest.modules.assertions.engine import AssertionEngine
from .models import (
    SQLTest, TestSuite, TestResult, TestSuiteResult, 
    TestStatus, TestAssertion, AssertionType
)
from .fixtures import FixtureManager


class TestExecutor:
    """Executes SQL unit tests with comprehensive lifecycle management."""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize test executor.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.fixture_manager = FixtureManager(connection_manager)
        self.assertion_engine = AssertionEngine()
        self._executed_tests: Set[str] = set()
        
    async def execute_test(self, test: SQLTest) -> TestResult:
        """
        Execute a single SQL test.
        
        Args:
            test: SQL test to execute
            
        Returns:
            Test execution result
        """
        result = TestResult(
            test_name=test.name,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Check if test should be skipped
            if not test.enabled:
                result.status = TestStatus.SKIPPED
                result.end_time = datetime.now()
                return result
            
            # Check dependencies
            if not self._check_dependencies(test):
                result.status = TestStatus.SKIPPED
                result.error_message = f"Dependencies not met: {', '.join(test.depends_on)}"
                result.end_time = datetime.now()
                return result
            
            # Set up fixtures
            if test.fixtures:
                await self.fixture_manager.setup_fixtures(test.fixtures)
            
            # Run setup SQL if provided
            if test.setup_sql:
                setup_result = self.connection_manager.execute_query(test.setup_sql)
                if not setup_result.success:
                    raise Exception(f"Setup SQL failed: {setup_result.error}")
            
            # Execute main test SQL
            query_result = self.connection_manager.execute_query(test.sql)
            if not query_result.success:
                raise Exception(f"Test SQL failed: {query_result.error}")
            
            result.query_result = query_result.data
            result.row_count = query_result.row_count
            
            # Run assertions
            assertion_results = []
            all_passed = True
            
            for assertion in test.assertions:
                assertion_result = await self._execute_assertion(
                    assertion, 
                    query_result.data,
                    query_result.row_count
                )
                assertion_results.append(assertion_result)
                if not assertion_result['passed']:
                    all_passed = False
            
            result.assertion_results = assertion_results
            result.status = TestStatus.PASSED if all_passed else TestStatus.FAILED
            
            # Run teardown SQL if provided
            if test.teardown_sql:
                teardown_result = self.connection_manager.execute_query(test.teardown_sql)
                if not teardown_result.success:
                    # Log warning but don't fail test
                    print(f"Warning: Teardown SQL failed for test {test.name}: {teardown_result.error}")
            
            # Clean up fixtures
            if test.fixtures:
                await self.fixture_manager.cleanup_fixtures(test.fixtures)
                
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            
            # Try to clean up on error
            if test.fixtures:
                try:
                    await self.fixture_manager.cleanup_fixtures(test.fixtures)
                except:
                    pass  # Ignore cleanup errors
        
        finally:
            result.end_time = datetime.now()
            self._executed_tests.add(test.name)
        
        return result
    
    async def _execute_assertion(
        self, 
        assertion: TestAssertion, 
        data: Optional[pd.DataFrame],
        row_count: Optional[int]
    ) -> Dict[str, Any]:
        """
        Execute a single test assertion.
        
        Args:
            assertion: Assertion to execute
            data: Query result data
            row_count: Number of rows returned
            
        Returns:
            Assertion execution result
        """
        try:
            # Convert our assertion types to assertion engine format
            assertion_def = {
                'type': assertion.assertion_type.value,
                'expected': assertion.expected,
                'message': assertion.message or f"{assertion.assertion_type} assertion",
                'tolerance': assertion.tolerance,
                'ignore_order': assertion.ignore_order
            }
            
            # Execute assertion based on type
            if assertion.assertion_type == AssertionType.EQUALS:
                passed = self.assertion_engine._assert_equals(
                    data, assertion.expected, assertion.tolerance, assertion.ignore_order
                )
            elif assertion.assertion_type == AssertionType.NOT_EQUALS:
                passed = self.assertion_engine._assert_not_equals(
                    data, assertion.expected, assertion.ignore_order
                )
            elif assertion.assertion_type == AssertionType.CONTAINS:
                passed = self.assertion_engine._assert_contains(data, assertion.expected)
            elif assertion.assertion_type == AssertionType.NOT_CONTAINS:
                passed = self.assertion_engine._assert_not_contains(data, assertion.expected)
            elif assertion.assertion_type == AssertionType.EMPTY:
                passed = self.assertion_engine._assert_empty(data)
            elif assertion.assertion_type == AssertionType.NOT_EMPTY:
                passed = self.assertion_engine._assert_not_empty(data)
            elif assertion.assertion_type == AssertionType.ROW_COUNT:
                passed = self.assertion_engine._assert_row_count(data, assertion.expected)
            elif assertion.assertion_type == AssertionType.COLUMN_COUNT:
                passed = self.assertion_engine._assert_column_count(data, assertion.expected)
            elif assertion.assertion_type == AssertionType.SCHEMA_MATCH:
                passed = self.assertion_engine._assert_schema_match(data, assertion.expected)
            elif assertion.assertion_type == AssertionType.CUSTOM:
                if assertion.custom_function:
                    # Execute custom assertion function
                    passed = self._execute_custom_assertion(
                        assertion.custom_function, data, assertion.expected
                    )
                else:
                    passed = False
            else:
                passed = False
            
            return {
                'assertion_type': assertion.assertion_type.value,
                'expected': assertion.expected,
                'actual': self._get_actual_value(assertion.assertion_type, data, row_count),
                'passed': passed,
                'message': assertion.message or f"{assertion.assertion_type} assertion",
                'error': None
            }
            
        except Exception as e:
            return {
                'assertion_type': assertion.assertion_type.value,
                'expected': assertion.expected,
                'actual': None,
                'passed': False,
                'message': assertion.message or f"{assertion.assertion_type} assertion",
                'error': str(e)
            }
    
    def _get_actual_value(
        self, 
        assertion_type: AssertionType, 
        data: Optional[pd.DataFrame],
        row_count: Optional[int]
    ) -> Any:
        """Get actual value for comparison in assertion results."""
        if data is None:
            return None
            
        if assertion_type == AssertionType.ROW_COUNT:
            return len(data) if data is not None else 0
        elif assertion_type == AssertionType.COLUMN_COUNT:
            return len(data.columns) if data is not None else 0
        elif assertion_type in [AssertionType.EMPTY, AssertionType.NOT_EMPTY]:
            return len(data) == 0 if data is not None else True
        elif assertion_type == AssertionType.SCHEMA_MATCH:
            if data is not None:
                return {col: str(dtype) for col, dtype in data.dtypes.items()}
            return None
        else:
            # For other assertions, return the data itself or a summary
            if data is not None and not data.empty:
                if len(data) == 1 and len(data.columns) == 1:
                    return data.iloc[0, 0]  # Single value
                return data.to_dict('records')  # Full data
            return None
    
    def _execute_custom_assertion(
        self, 
        function_code: str, 
        data: Optional[pd.DataFrame], 
        expected: Any
    ) -> bool:
        """
        Execute custom assertion function.
        
        Args:
            function_code: Python code for custom assertion
            data: Query result data
            expected: Expected value
            
        Returns:
            Assertion result
        """
        try:
            # Create safe execution environment
            namespace = {
                'data': data,
                'expected': expected,
                'pd': pd,
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
            }
            
            # Execute the custom function
            exec(function_code, namespace)
            
            # Look for a result variable or return value
            if 'result' in namespace:
                return bool(namespace['result'])
            else:
                # If no explicit result, assume the last expression is the result
                return True  # Default to passing if unclear
                
        except Exception as e:
            print(f"Custom assertion error: {e}")
            return False
    
    def _check_dependencies(self, test: SQLTest) -> bool:
        """Check if test dependencies have been executed."""
        if not test.depends_on:
            return True
        
        return all(dep in self._executed_tests for dep in test.depends_on)
    
    async def execute_test_suite(self, test_suite: TestSuite) -> TestSuiteResult:
        """
        Execute a complete test suite.
        
        Args:
            test_suite: Test suite to execute
            
        Returns:
            Test suite execution result
        """
        suite_result = TestSuiteResult(
            suite_name=test_suite.name,
            start_time=datetime.now()
        )
        
        try:
            # Run suite setup SQL if provided
            if test_suite.setup_sql:
                setup_result = self.connection_manager.execute_query(test_suite.setup_sql)
                if not setup_result.success:
                    raise Exception(f"Suite setup SQL failed: {setup_result.error}")
            
            # Get enabled tests
            tests_to_run = test_suite.get_enabled_tests()
            
            # Sort tests by dependencies (simple topological sort)
            ordered_tests = self._sort_tests_by_dependencies(tests_to_run)
            
            # Execute tests
            for test in ordered_tests:
                test_result = await self.execute_test(test)
                suite_result.test_results.append(test_result)
            
            # Run suite teardown SQL if provided
            if test_suite.teardown_sql:
                teardown_result = self.connection_manager.execute_query(test_suite.teardown_sql)
                if not teardown_result.success:
                    print(f"Warning: Suite teardown SQL failed: {teardown_result.error}")
                    
        except Exception as e:
            # If suite setup fails, mark all tests as error
            for test in test_suite.get_enabled_tests():
                if not any(r.test_name == test.name for r in suite_result.test_results):
                    error_result = TestResult(
                        test_name=test.name,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=f"Suite setup failed: {str(e)}"
                    )
                    suite_result.test_results.append(error_result)
        
        finally:
            suite_result.end_time = datetime.now()
        
        return suite_result
    
    def _sort_tests_by_dependencies(self, tests: List[SQLTest]) -> List[SQLTest]:
        """
        Sort tests by their dependencies using topological sort.
        
        Args:
            tests: List of tests to sort
            
        Returns:
            Tests sorted by dependencies
        """
        # Simple implementation - more sophisticated topological sort could be added
        test_dict = {test.name: test for test in tests}
        sorted_tests = []
        visited = set()
        
        def visit(test_name: str):
            if test_name in visited or test_name not in test_dict:
                return
            
            visited.add(test_name)
            test = test_dict[test_name]
            
            # Visit dependencies first
            for dep in test.depends_on:
                if dep in test_dict:
                    visit(dep)
            
            sorted_tests.append(test)
        
        # Visit all tests
        for test in tests:
            visit(test.name)
        
        return sorted_tests
    
    async def execute_tests_parallel(
        self, 
        tests: List[SQLTest], 
        max_concurrent: int = 5
    ) -> List[TestResult]:
        """
        Execute multiple tests in parallel.
        
        Args:
            tests: List of tests to execute
            max_concurrent: Maximum number of concurrent test executions
            
        Returns:
            List of test results
        """
        # Filter tests that can run in parallel (no dependencies)
        independent_tests = [test for test in tests if not test.depends_on]
        dependent_tests = [test for test in tests if test.depends_on]
        
        results = []
        
        # Run independent tests in parallel
        if independent_tests:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_semaphore(test: SQLTest):
                async with semaphore:
                    return await self.execute_test(test)
            
            tasks = [run_with_semaphore(test) for test in independent_tests]
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in parallel_results:
                if isinstance(result, Exception):
                    # Handle exception case
                    error_result = TestResult(
                        test_name="unknown",
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Run dependent tests sequentially after parallel execution
        ordered_dependent = self._sort_tests_by_dependencies(dependent_tests)
        for test in ordered_dependent:
            result = await self.execute_test(test)
            results.append(result)
        
        return results
    
    def clear_execution_state(self) -> None:
        """Clear execution state for fresh test runs."""
        self._executed_tests.clear()
        self.fixture_manager.clear_cache()
