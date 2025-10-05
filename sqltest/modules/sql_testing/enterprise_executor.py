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
import inspect
import logging
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, DatabaseError
from .models import (
    SQLTest, TestSuite, TestResult, TestSuiteResult,
    TestStatus, TestAssertion, AssertionType, TestIsolationLevel
)
from .fixtures import FixtureManager
from .assertions import SQLTestAssertionEngine

logger = logging.getLogger(__name__)


@dataclass
class TestExecutionContext:
    """Represents an isolated execution context for a SQL test."""

    test_name: str
    isolation_level: TestIsolationLevel = TestIsolationLevel.SCHEMA
    cleanup_resources: List[str] = field(default_factory=list)
    database_name: Optional[str] = None
    temporary_schema: Optional[str] = None
    transaction_savepoint: Optional[str] = None
    temporary_database: Optional[str] = None
    created_objects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestMetricEntry:
    """Execution metrics captured for a single test."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: Optional[TestStatus] = None
    error_message: Optional[str] = None
    assertion_count: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0
    total_assertion_time: float = 0.0

    @property
    def duration(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def finalize(self, status: TestStatus, error_message: Optional[str] = None) -> None:
        self.end_time = datetime.now()
        self.status = status
        self.error_message = error_message


@dataclass
class PerformanceSummary:
    """Aggregated view of test execution performance."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    total_execution_time: float


class TestIsolationManager:
    """Manages test isolation with temporary schemas and transaction control."""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._isolation_contexts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_isolation_context(
        self,
        test_name: str,
        isolation_level: TestIsolationLevel = TestIsolationLevel.SCHEMA,
    ) -> str:
        """Create an isolated context identifier for legacy workflows."""
        context_id = f"test_{test_name}_{uuid.uuid4().hex[:8]}"

        with self._lock:
            self._isolation_contexts[context_id] = {
                "test_name": test_name,
                "isolation_level": isolation_level,
                "created_at": datetime.now(),
                "temporary_schema": None,
                "transaction_savepoint": None,
                "created_objects": [],
                "cleanup_required": True,
            }

        return context_id

    async def setup_isolation(
        self,
        context: Union[TestExecutionContext, str],
        database_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set up test isolation based on the configured level."""

        if isinstance(context, TestExecutionContext):
            return await self._setup_context_object(context, database_name)

        context_id = context
        if database_name is None:
            raise ValueError(
                "database_name must be provided when using legacy isolation context identifiers"
            )

        context_data = self._isolation_contexts.get(context_id)
        if not context_data:
            raise ValueError(f"Invalid isolation context: {context_id}")

        adapter = self.connection_manager.get_adapter(database_name)
        isolation_level = context_data["isolation_level"]

        try:
            if isolation_level == TestIsolationLevel.SCHEMA:
                schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"
                await self._call_method(adapter, "execute_query", f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                context_data["temporary_schema"] = schema_name
                await self._call_method(adapter, "execute_query", f"SET search_path TO {schema_name}, public")

            elif isolation_level == TestIsolationLevel.TRANSACTION:
                await self._call_method(adapter, "execute_query", "BEGIN")
                savepoint_name = f"test_savepoint_{uuid.uuid4().hex[:8]}"
                await self._call_method(adapter, "execute_query", f"SAVEPOINT {savepoint_name}")
                context_data["transaction_savepoint"] = savepoint_name

            elif isolation_level == TestIsolationLevel.DATABASE:
                db_name = f"test_db_{uuid.uuid4().hex[:8]}"
                await self._call_method(adapter, "execute_query", f"CREATE DATABASE {db_name}")
                context_data["temporary_database"] = db_name

            return {
                "schema": context_data.get("temporary_schema"),
                "savepoint": context_data.get("transaction_savepoint"),
                "database": context_data.get("temporary_database"),
            }

        except Exception as exc:
            logger.error(f"Failed to setup test isolation: {exc}")
            await self.cleanup_isolation(context_id, database_name)
            raise

    async def cleanup_isolation(
        self,
        context: Union[TestExecutionContext, str],
        database_name: Optional[str] = None,
    ) -> None:
        """Clean up test isolation resources."""

        if isinstance(context, TestExecutionContext):
            await self._cleanup_context_object(context, database_name)
            return

        context_id = context
        if database_name is None:
            raise ValueError(
                "database_name must be provided when using legacy isolation context identifiers"
            )

        context_data = self._isolation_contexts.get(context_id)
        if not context_data:
            return

        adapter = self.connection_manager.get_adapter(database_name)
        isolation_level = context_data["isolation_level"]

        try:
            if isolation_level == TestIsolationLevel.SCHEMA:
                schema_name = context_data.get("temporary_schema")
                if schema_name:
                    await self._call_method(
                        adapter,
                        "execute_query",
                        f"DROP SCHEMA IF EXISTS {schema_name} CASCADE",
                    )

            elif isolation_level == TestIsolationLevel.TRANSACTION:
                savepoint_name = context_data.get("transaction_savepoint")
                if savepoint_name:
                    await self._call_method(
                        adapter,
                        "execute_query",
                        f"ROLLBACK TO SAVEPOINT {savepoint_name}",
                    )
                await self._call_method(adapter, "execute_query", "ROLLBACK")

            elif isolation_level == TestIsolationLevel.DATABASE:
                db_name = context_data.get("temporary_database")
                if db_name:
                    await self._call_method(
                        adapter,
                        "execute_query",
                        f"DROP DATABASE IF EXISTS {db_name}",
                    )

        except Exception as exc:
            logger.warning(f"Failed to cleanup test isolation: {exc}")
        finally:
            with self._lock:
                self._isolation_contexts.pop(context_id, None)

    async def _setup_context_object(
        self,
        context: TestExecutionContext,
        database_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        adapter = self._resolve_adapter(database_name or context.database_name)
        level = context.isolation_level

        if level == TestIsolationLevel.SCHEMA:
            schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"
            try:
                await self._call_method(
                    adapter,
                    "execute_query",
                    f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
                )
                await self._call_method(
                    adapter,
                    "execute_query",
                    f"SET search_path TO {schema_name}, public",
                )
                context.temporary_schema = schema_name
                return {"schema": schema_name}
            except Exception as exc:
                logger.debug("Schema isolation not supported: %s", exc)
                return {}

        if level == TestIsolationLevel.TRANSACTION:
            if hasattr(adapter, "begin_transaction"):
                await self._call_method(adapter, "begin_transaction")
            else:
                await self._call_method(adapter, "execute_query", "BEGIN")

            savepoint_name = f"test_savepoint_{uuid.uuid4().hex[:8]}"
            try:
                await self._call_method(
                    adapter,
                    "execute_query",
                    f"SAVEPOINT {savepoint_name}",
                )
                context.transaction_savepoint = savepoint_name
                return {"savepoint": savepoint_name}
            except Exception as exc:
                logger.debug("Savepoint creation skipped: %s", exc)
                return {}

        if level == TestIsolationLevel.DATABASE:
            db_name = f"test_db_{uuid.uuid4().hex[:8]}"
            await self._call_method(adapter, "execute_query", f"CREATE DATABASE {db_name}")
            context.temporary_database = db_name
            return {"database": db_name}

        return {}

    async def _cleanup_context_object(
        self,
        context: TestExecutionContext,
        database_name: Optional[str] = None,
    ) -> None:
        adapter = self._resolve_adapter(database_name or context.database_name)
        level = context.isolation_level

        if level == TestIsolationLevel.SCHEMA and context.temporary_schema:
            await self._call_method(
                adapter,
                "execute_query",
                f"DROP SCHEMA IF EXISTS {context.temporary_schema} CASCADE",
            )
            context.temporary_schema = None

        elif level == TestIsolationLevel.TRANSACTION:
            if context.transaction_savepoint:
                try:
                    await self._call_method(
                        adapter,
                        "execute_query",
                        f"ROLLBACK TO SAVEPOINT {context.transaction_savepoint}",
                    )
                except Exception as exc:
                    logger.debug("Rollback to savepoint failed: %s", exc)
            if hasattr(adapter, "rollback_transaction"):
                await self._call_method(adapter, "rollback_transaction")
            else:
                await self._call_method(adapter, "execute_query", "ROLLBACK")
            context.transaction_savepoint = None

        elif level == TestIsolationLevel.DATABASE and context.temporary_database:
            await self._call_method(
                adapter,
                "execute_query",
                f"DROP DATABASE IF EXISTS {context.temporary_database}",
            )
            context.temporary_database = None

    def _resolve_adapter(self, database_name: Optional[str]):
        if hasattr(self.connection_manager, "begin_transaction"):
            return self.connection_manager
        if hasattr(self.connection_manager, "get_adapter"):
            return self.connection_manager.get_adapter(database_name)
        return self.connection_manager

    async def _call_method(self, target: Any, method_name: str, *args, **kwargs):
        method = getattr(target, method_name, None)
        if method is None:
            return None
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


class TestMetricsCollector:
    """Collects and aggregates test execution metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.test_metrics: Dict[str, TestMetricEntry] = {}
        self._lock = threading.Lock()

    def start_test(self, test_name: str) -> None:
        with self._lock:
            self.test_metrics[test_name] = TestMetricEntry()
            self.counters["tests_started"] += 1

    def end_test(self, test_name: str, status: TestStatus, error_message: Optional[str] = None) -> None:
        with self._lock:
            entry = self.test_metrics.setdefault(test_name, TestMetricEntry())
            entry.finalize(status, error_message)

            duration_ms = entry.duration * 1000
            self.metrics[f"{test_name}_execution_time"].append(duration_ms)
            self.metrics["all_execution_times"].append(duration_ms)

            if status == TestStatus.PASSED:
                self.counters["tests_passed"] += 1
            elif status == TestStatus.FAILED:
                self.counters["tests_failed"] += 1
            elif status == TestStatus.ERROR:
                self.counters["tests_error"] += 1

    def record_assertion(self, test_name: str, passed: bool, duration: float = 0.0) -> None:
        with self._lock:
            entry = self.test_metrics.setdefault(test_name, TestMetricEntry())
            entry.assertion_count += 1
            if passed:
                entry.passed_assertions += 1
            else:
                entry.failed_assertions += 1
            entry.total_assertion_time += duration

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

            if "all_execution_times" in self.metrics:
                times = self.metrics["all_execution_times"]
                stats['execution_times'] = {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times) if times else 0,
                    'min_ms': min(times) if times else 0,
                    'max_ms': max(times) if times else 0,
                    'total_ms': sum(times)
                }

            stats['counters'] = dict(self.counters)
            stats['tests'] = {
                name: {
                    'status': entry.status.name if entry.status else None,
                    'duration': entry.duration,
                    'assertions': entry.assertion_count,
                    'passed_assertions': entry.passed_assertions,
                    'failed_assertions': entry.failed_assertions,
                }
                for name, entry in self.test_metrics.items()
            }

            return stats

    def get_performance_summary(self) -> PerformanceSummary:
        with self._lock:
            total_tests = len(self.test_metrics)
            passed_tests = sum(1 for entry in self.test_metrics.values() if entry.status == TestStatus.PASSED)
            failed_tests = sum(1 for entry in self.test_metrics.values() if entry.status == TestStatus.FAILED)
            total_execution_time = sum(entry.duration for entry in self.test_metrics.values())

        return PerformanceSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_execution_time=total_execution_time,
        )


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

        if self.metrics:
            self.metrics.start_test(test.name)

        # Determine database and isolation level
        db_name = database_name or getattr(test, 'database_name', 'default')
        iso_level = isolation_level or getattr(test, 'isolation_level', self.default_isolation_level)

        # Create isolation context
        context = TestExecutionContext(
            test_name=test.name,
            isolation_level=iso_level,
            database_name=db_name,
        )

        try:
            with self._test_execution_lock:
                self._active_contexts[test.name] = context

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
            isolation_info = await self.isolation_manager.setup_isolation(context, db_name)

            # Set up fixtures within isolated context
            if test.fixtures:
                await self.fixture_manager.setup_fixtures(test.fixtures, isolation_info)

            # Run setup SQL if provided
            if test.setup_sql:
                success, error = await self._execute_sql_block(
                    test.setup_sql,
                    db_name,
                    fetch_results=False,
                )
                if not success:
                    raise Exception(error or "Setup SQL failed")

            # Execute main test SQL
            query_result = await self._call_adapter(
                self.connection_manager,
                "execute_query",
                test.sql,
                db_name=db_name,
            )
            if not self._is_successful_result(query_result):
                result.status = TestStatus.FAILED
                result.error_message = getattr(query_result, "error", "Test SQL execution failed")
                result.end_time = datetime.now()
                return result

            result.data = getattr(query_result, "data", None)
            result.row_count = len(result.data) if result.data is not None else 0

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

                if self.metrics:
                    self.metrics.record_assertion(test.name, assertion_result.passed)

                if not assertion_result.passed:
                    all_passed = False

            result.assertion_results = assertion_results
            result.status = TestStatus.PASSED if all_passed else TestStatus.FAILED

            # Run teardown SQL if provided
            if test.teardown_sql:
                success, error = await self._execute_sql_block(
                    test.teardown_sql,
                    db_name,
                    fetch_results=False,
                )
                if not success:
                    logger.warning(f"Teardown SQL failed for test {test.name}: {error}")

            # Clean up fixtures
            if test.fixtures:
                await self.fixture_manager.cleanup_fixtures(test.fixtures)

        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)

            # Try to clean up on error
            if test.fixtures:
                try:
                    await self.fixture_manager.cleanup_fixtures(test.fixtures)
                except:
                    pass  # Ignore cleanup errors

        finally:
            result.end_time = datetime.now()
            if result.end_time and result.start_time:
                result.execution_time = (result.end_time - result.start_time).total_seconds()
            self._executed_tests.add(test.name)

            if self.metrics:
                self.metrics.end_test(test.name, result.status, result.error_message)

            # Clean up isolation context
            try:
                await self.isolation_manager.cleanup_isolation(context, db_name)
            except Exception as cleanup_error:
                logger.warning(f"Isolation cleanup failed for test {test.name}: {cleanup_error}")

            # Remove from active contexts
            with self._test_execution_lock:
                self._active_contexts.pop(test.name, None)

            close_connections = getattr(self.connection_manager, "close_all_connections", None)
            if callable(close_connections):
                try:
                    close_connections()
                except Exception:
                    logger.debug("Failed to close database connections for %s", test.name, exc_info=True)

        return result

    def _check_dependencies(self, test: SQLTest) -> bool:
        """Check if test dependencies have been executed."""
        if not test.depends_on:
            return True

        return all(dep in self._executed_tests for dep in test.depends_on)

    async def _call_adapter(self, target: Any, method_name: str, *args, **kwargs):
        """Invoke adapter methods that may be synchronous or asynchronous."""

        method = getattr(target, method_name, None)
        if method is None:
            return None

        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _is_successful_result(result: Any) -> bool:
        """Determine whether an adapter call succeeded."""
        if result is None:
            return True
        success = getattr(result, "success", None)
        if success is None:
            return True
        return bool(success)

    async def _execute_sql_block(
        self,
        sql: str,
        db_name: str,
        *,
        fetch_results: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Execute one or more SQL statements sequentially."""

        if not sql:
            return True, None

        statements = [statement.strip() for statement in sql.split(';') if statement.strip()]
        if not statements:
            return True, None

        for statement in statements:
            result = await self._call_adapter(
                self.connection_manager,
                "execute_query",
                statement,
                db_name=db_name,
                fetch_results=fetch_results,
            )

            if not self._is_successful_result(result):
                return False, getattr(result, "error", "SQL execution failed")

        return True, None

    async def execute_test_suite(
        self,
        test_suite: TestSuite,
        *,
        parallel: Optional[bool] = None,
        fail_fast: Optional[bool] = None,
        max_workers: Optional[int] = None,
        database_name: Optional[str] = None,
    ) -> List[TestResult]:
        """Execute a test suite and return individual test results."""

        effective_parallel = parallel if parallel is not None else getattr(test_suite, "parallel", False)
        effective_fail_fast = fail_fast if fail_fast is not None else getattr(test_suite, "fail_fast", False)
        target_database = database_name or getattr(test_suite, "database", None) or "default"

        previous_workers = self.max_workers
        requested_workers = max_workers if max_workers is not None else getattr(test_suite, "max_workers", previous_workers)
        suite_result: Optional[TestSuiteResult] = None

        previous_executed = self._executed_tests.copy()
        self._executed_tests = set()

        try:
            self.max_workers = requested_workers
            suite_result = await self.execute_test_suite_with_isolation(
                test_suite,
                database_name=target_database,
                parallel=effective_parallel,
                fail_fast=effective_fail_fast,
            )

            suite_result.total_tests = len(suite_result.test_results)
            suite_result.passed_tests = sum(1 for r in suite_result.test_results if r.status == TestStatus.PASSED)
            suite_result.failed_tests = sum(1 for r in suite_result.test_results if r.status == TestStatus.FAILED)
            suite_result.skipped_tests = sum(1 for r in suite_result.test_results if r.status == TestStatus.SKIPPED)
            suite_result.error_tests = sum(1 for r in suite_result.test_results if r.status == TestStatus.ERROR)

            return suite_result.test_results
        finally:
            self.max_workers = previous_workers
            if suite_result is None:
                suite_executed = self._executed_tests
            else:
                suite_executed = {result.test_name for result in suite_result.test_results}
            self._executed_tests = previous_executed.union(suite_executed)

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
                success, error_msg = await self._execute_sql_block(
                    test_suite.setup_sql,
                    db_name,
                    fetch_results=False,
                )
                if not success:
                    raise Exception(error_msg or "Suite setup failed")

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
                success, error_msg = await self._execute_sql_block(
                    test_suite.teardown_sql,
                    db_name,
                    fetch_results=False,
                )
                if not success:
                    logger.warning(f"Suite teardown failed: {error_msg}")

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

    def get_execution_metrics(self) -> PerformanceSummary:
        """Return aggregate performance summary for executed tests."""
        if not self.metrics:
            return PerformanceSummary(0, 0, 0, 0.0)

        return self.metrics.get_performance_summary()
