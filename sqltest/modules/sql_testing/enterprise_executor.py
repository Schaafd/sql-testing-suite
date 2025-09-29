"""Enterprise SQL test execution engine with advanced isolation and parallel processing.

This module provides enterprise-grade SQL test execution with features including:
- Advanced test isolation (schema, transaction, database levels)
- Parallel execution with dependency management
- Comprehensive metrics collection
- Performance monitoring and optimization
- Resource cleanup and management
"""
import asyncio
import contextlib
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
import traceback
import pandas as pd
from collections import defaultdict

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, DatabaseError
from .models import (
    SQLTest, TestSuite, TestResult, TestSuiteResult,
    TestStatus, TestAssertion, AssertionType, TestIsolationLevel
)
from .fixtures import FixtureManager
from .assertions import SQLTestAssertionEngine

logger = logging.getLogger(__name__)


class TestIsolationManager:
    """Manages test isolation with temporary schemas and transaction control."""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._isolation_contexts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_isolation_context(self,
                               test_name: str,
                               isolation_level: TestIsolationLevel = TestIsolationLevel.SCHEMA) -> str:
        """Create an isolated context for test execution."""
        context_id = f"test_{test_name}_{uuid.uuid4().hex[:8]}"

        with self._lock:
            self._isolation_contexts[context_id] = {
                'test_name': test_name,
                'isolation_level': isolation_level,
                'created_at': datetime.now(),
                'temporary_schema': None,
                'transaction_savepoint': None,
                'created_objects': [],
                'cleanup_required': True
            }

        return context_id

    async def setup_isolation(self, context_id: str, database_name: str) -> Dict[str, Any]:
        """Set up test isolation based on the configured level."""
        context = self._isolation_contexts.get(context_id)
        if not context:
            raise ValueError(f"Invalid isolation context: {context_id}")

        adapter = self.connection_manager.get_adapter(database_name)
        isolation_level = context['isolation_level']

        try:
            if isolation_level == TestIsolationLevel.SCHEMA:
                # Create temporary schema
                schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"
                await adapter.execute_query(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                context['temporary_schema'] = schema_name

                # Set search path to use temporary schema
                await adapter.execute_query(f"SET search_path TO {schema_name}, public")

            elif isolation_level == TestIsolationLevel.TRANSACTION:
                # Start transaction and create savepoint
                await adapter.execute_query("BEGIN")
                savepoint_name = f"test_savepoint_{uuid.uuid4().hex[:8]}"
                await adapter.execute_query(f"SAVEPOINT {savepoint_name}")
                context['transaction_savepoint'] = savepoint_name

            elif isolation_level == TestIsolationLevel.DATABASE:
                # Create temporary database (more complex, database-specific)
                db_name = f"test_db_{uuid.uuid4().hex[:8]}"
                await adapter.execute_query(f"CREATE DATABASE {db_name}")
                context['temporary_database'] = db_name

            return {
                'schema': context.get('temporary_schema'),
                'savepoint': context.get('transaction_savepoint'),
                'database': context.get('temporary_database')
            }

        except Exception as e:
            logger.error(f"Failed to setup test isolation: {e}")
            await self.cleanup_isolation(context_id, database_name)
            raise

    async def cleanup_isolation(self, context_id: str, database_name: str):
        """Clean up test isolation resources."""
        context = self._isolation_contexts.get(context_id)
        if not context:
            return

        adapter = self.connection_manager.get_adapter(database_name)
        isolation_level = context['isolation_level']

        try:
            if isolation_level == TestIsolationLevel.SCHEMA:
                schema_name = context.get('temporary_schema')
                if schema_name:
                    await adapter.execute_query(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")

            elif isolation_level == TestIsolationLevel.TRANSACTION:
                savepoint_name = context.get('transaction_savepoint')
                if savepoint_name:
                    await adapter.execute_query(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                await adapter.execute_query("ROLLBACK")

            elif isolation_level == TestIsolationLevel.DATABASE:
                db_name = context.get('temporary_database')
                if db_name:
                    await adapter.execute_query(f"DROP DATABASE IF EXISTS {db_name}")

        except Exception as e:
            logger.warning(f"Failed to cleanup test isolation: {e}")
        finally:
            with self._lock:
                self._isolation_contexts.pop(context_id, None)


class TestMetricsCollector:
    """Collects and aggregates test execution metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_execution_time(self, test_name: str, duration_ms: float):
        """Record test execution time."""
        with self._lock:
            self.metrics[f"{test_name}_execution_time"].append(duration_ms)
            self.metrics["all_execution_times"].append(duration_ms)

    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter metric."""
        with self._lock:
            self.counters[counter_name] += value

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            stats = {}

            # Execution time statistics
            if "all_execution_times" in self.metrics:
                times = self.metrics["all_execution_times"]
                stats['execution_times'] = {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times) if times else 0,
                    'min_ms': min(times) if times else 0,
                    'max_ms': max(times) if times else 0,
                    'total_ms': sum(times)
                }

            # Counter statistics
            stats['counters'] = dict(self.counters)

            return stats


class EnterpriseTestExecutor:
    """Enterprise-grade SQL unit test executor with advanced features."""

    def __init__(self,
                 connection_manager: ConnectionManager,
                 max_workers: int = 5,
                 default_isolation_level: TestIsolationLevel = TestIsolationLevel.SCHEMA,
                 enable_metrics: bool = True,
                 test_timeout: int = 300):
        """
        Initialize enterprise test executor.

        Args:
            connection_manager: Database connection manager
            max_workers: Maximum concurrent test executions
            default_isolation_level: Default test isolation level
            enable_metrics: Enable performance metrics collection
            test_timeout: Default test timeout in seconds
        """
        self.connection_manager = connection_manager
        self.max_workers = max_workers
        self.default_isolation_level = default_isolation_level
        self.test_timeout = test_timeout

        # Initialize components
        self.fixture_manager = FixtureManager(connection_manager)
        self.assertion_engine = SQLTestAssertionEngine()
        self.isolation_manager = TestIsolationManager(connection_manager)
        self.metrics = TestMetricsCollector() if enable_metrics else None

        # Execution state
        self._executed_tests: Set[str] = set()
        self._test_execution_lock = threading.RLock()
        self._active_contexts: Dict[str, str] = {}  # test_name -> context_id

        logger.info(f"Enterprise SQL Test Executor initialized with {max_workers} workers")

    async def execute_test(self,
                          test: SQLTest,
                          database_name: str = None,
                          isolation_level: TestIsolationLevel = None) -> TestResult:
        """
        Execute a single SQL test with enterprise features.

        Args:
            test: SQL test to execute
            database_name: Target database name
            isolation_level: Override isolation level

        Returns:
            Test execution result
        """
        start_time = datetime.now()
        result = TestResult(
            test_name=test.name,
            status=TestStatus.RUNNING,
            start_time=start_time
        )

        # Determine database and isolation level
        db_name = database_name or getattr(test, 'database_name', 'default')
        iso_level = isolation_level or getattr(test, 'isolation_level', self.default_isolation_level)

        # Create isolation context
        context_id = self.isolation_manager.create_isolation_context(test.name, iso_level)

        try:
            with self._test_execution_lock:
                self._active_contexts[test.name] = context_id

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

            # Set up test isolation
            isolation_info = await self.isolation_manager.setup_isolation(context_id, db_name)

            # Set up fixtures within isolated context
            if test.fixtures:
                await self.fixture_manager.setup_fixtures(test.fixtures, isolation_info)

            # Run setup SQL if provided
            if test.setup_sql:
                setup_result = await self.connection_manager.get_adapter(db_name).execute_query(test.setup_sql)
                if not setup_result.success:
                    raise Exception(f"Setup SQL failed: {setup_result.error}")

            # Execute main test SQL
            query_result = await self.connection_manager.get_adapter(db_name).execute_query(test.sql)
            if not query_result.success:
                raise Exception(f"Test SQL failed: {query_result.error}")

            result.query_result = query_result.data
            result.row_count = len(query_result.data) if query_result.data is not None else 0

            # Run assertions
            assertion_results = []
            all_passed = True

            for assertion in test.assertions:
                assertion_result = self.assertion_engine.execute_assertion(
                    assertion_type=assertion.assertion_type,
                    data=query_result.data,
                    expected=assertion.expected,
                    tolerance=assertion.tolerance,
                    ignore_order=assertion.ignore_order,
                    custom_function=assertion.custom_function,
                    message=assertion.message
                )

                assertion_results.append({
                    'assertion_type': assertion_result.assertion_type.value,
                    'expected': assertion_result.expected,
                    'actual': assertion_result.actual,
                    'passed': assertion_result.passed,
                    'message': assertion_result.message,
                    'error': assertion_result.error,
                    'details': assertion_result.details
                })

                if not assertion_result.passed:
                    all_passed = False

            result.assertion_results = assertion_results
            result.status = TestStatus.PASSED if all_passed else TestStatus.FAILED

            # Run teardown SQL if provided
            if test.teardown_sql:
                teardown_result = await self.connection_manager.get_adapter(db_name).execute_query(test.teardown_sql)
                if not teardown_result.success:
                    logger.warning(f"Teardown SQL failed for test {test.name}: {teardown_result.error}")

            # Clean up fixtures
            if test.fixtures:
                await self.fixture_manager.cleanup_fixtures(test.fixtures)

            # Record metrics
            if self.metrics:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics.record_execution_time(test.name, duration_ms)
                self.metrics.increment_counter("tests_executed")
                if result.status == TestStatus.PASSED:
                    self.metrics.increment_counter("tests_passed")
                elif result.status == TestStatus.FAILED:
                    self.metrics.increment_counter("tests_failed")
                elif result.status == TestStatus.ERROR:
                    self.metrics.increment_counter("tests_error")

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

            # Clean up isolation context
            try:
                await self.isolation_manager.cleanup_isolation(context_id, db_name)
            except Exception as cleanup_error:
                logger.warning(f"Isolation cleanup failed for test {test.name}: {cleanup_error}")

            # Remove from active contexts
            with self._test_execution_lock:
                self._active_contexts.pop(test.name, None)

        return result

    def _check_dependencies(self, test: SQLTest) -> bool:
        """Check if test dependencies have been executed."""
        if not test.depends_on:
            return True

        return all(dep in self._executed_tests for dep in test.depends_on)

    async def execute_test_suite_with_isolation(self,
                                              test_suite: TestSuite,
                                              database_name: str = None,
                                              parallel: bool = False,
                                              fail_fast: bool = False) -> TestSuiteResult:
        """Execute test suite with advanced isolation and parallel execution."""
        suite_result = TestSuiteResult(
            suite_name=test_suite.name,
            start_time=datetime.now()
        )

        db_name = database_name or 'default'

        try:
            # Run suite setup
            if test_suite.setup_sql:
                setup_result = await self.connection_manager.get_adapter(db_name).execute_query(
                    test_suite.setup_sql
                )
                if not setup_result.success:
                    raise Exception(f"Suite setup failed: {setup_result.error}")

            # Get enabled tests
            tests_to_run = test_suite.get_enabled_tests()

            if parallel:
                # Execute tests in parallel with controlled concurrency
                results = await self._execute_tests_parallel_isolated(
                    tests_to_run, db_name, fail_fast
                )
            else:
                # Execute tests sequentially with isolation
                results = await self._execute_tests_sequential_isolated(
                    tests_to_run, db_name, fail_fast
                )

            suite_result.test_results.extend(results)

            # Run suite teardown
            if test_suite.teardown_sql:
                teardown_result = await self.connection_manager.get_adapter(db_name).execute_query(
                    test_suite.teardown_sql
                )
                if not teardown_result.success:
                    logger.warning(f"Suite teardown failed: {teardown_result.error}")

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            # Mark remaining tests as error
            for test in test_suite.get_enabled_tests():
                if not any(r.test_name == test.name for r in suite_result.test_results):
                    error_result = TestResult(
                        test_name=test.name,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=f"Suite execution failed: {str(e)}"
                    )
                    suite_result.test_results.append(error_result)

        finally:
            suite_result.end_time = datetime.now()

        return suite_result

    async def _execute_tests_parallel_isolated(self,
                                             tests: List[SQLTest],
                                             database_name: str,
                                             fail_fast: bool = False) -> List[TestResult]:
        """Execute tests in parallel with proper isolation."""
        # Separate independent tests from dependent ones
        independent_tests = [test for test in tests if not test.depends_on]
        dependent_tests = [test for test in tests if test.depends_on]

        results = []

        # Execute independent tests in parallel
        if independent_tests:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def run_with_semaphore(test: SQLTest):
                async with semaphore:
                    return await self.execute_test(test, database_name)

            tasks = [run_with_semaphore(test) for test in independent_tests]
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    error_result = TestResult(
                        test_name=independent_tests[i].name,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)

                    # Check fail-fast condition
                    if fail_fast and result.status in [TestStatus.ERROR, TestStatus.FAILED]:
                        logger.warning(f"Stopping execution due to fail-fast mode: {result.test_name}")
                        return results

        # Execute dependent tests sequentially
        if dependent_tests and not (fail_fast and any(r.status in [TestStatus.ERROR, TestStatus.FAILED] for r in results)):
            sequential_results = await self._execute_tests_sequential_isolated(
                dependent_tests, database_name, fail_fast
            )
            results.extend(sequential_results)

        return results

    async def _execute_tests_sequential_isolated(self,
                                                tests: List[SQLTest],
                                                database_name: str,
                                                fail_fast: bool = False) -> List[TestResult]:
        """Execute tests sequentially with proper dependency ordering."""
        ordered_tests = self._sort_tests_by_dependencies(tests)
        results = []

        for test in ordered_tests:
            result = await self.execute_test(test, database_name)
            results.append(result)

            # Check fail-fast condition
            if fail_fast and result.status in [TestStatus.ERROR, TestStatus.FAILED]:
                logger.warning(f"Stopping execution due to fail-fast mode: {result.test_name}")
                break

        return results

    def _sort_tests_by_dependencies(self, tests: List[SQLTest]) -> List[SQLTest]:
        """Sort tests by their dependencies using topological sort."""
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

    def clear_execution_state(self) -> None:
        """Clear execution state for fresh test runs."""
        self._executed_tests.clear()
        self.fixture_manager.clear_cache()
        if self.metrics:
            self.metrics = TestMetricsCollector()

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics."""
        if not self.metrics:
            return {"metrics_enabled": False}

        stats = self.metrics.get_summary_stats()
        stats.update({
            "metrics_enabled": True,
            "active_contexts": len(self._active_contexts),
            "executed_tests": len(self._executed_tests),
            "max_workers": self.max_workers,
            "default_isolation_level": self.default_isolation_level.value
        })

        return stats