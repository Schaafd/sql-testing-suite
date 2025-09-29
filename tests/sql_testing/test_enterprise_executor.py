"""
Comprehensive tests for the enterprise SQL test execution engine.

This module tests all aspects of the enterprise test executor including
isolation management, parallel execution, and performance optimization.
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from sqltest.modules.sql_testing.enterprise_executor import (
    EnterpriseTestExecutor,
    TestIsolationManager,
    TestMetricsCollector,
    TestExecutionContext
)
from sqltest.modules.sql_testing.models import (
    TestSuite,
    TestCase,
    TestIsolationLevel,
    TestResult,
    TestStatus
)
from sqltest.db.connection import ConnectionManager


class TestIsolationManager:
    """Test the test isolation management functionality."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        manager.execute_query = AsyncMock()
        manager.begin_transaction = AsyncMock()
        manager.commit_transaction = AsyncMock()
        manager.rollback_transaction = AsyncMock()
        return manager

    @pytest.fixture
    def isolation_manager(self, mock_connection_manager):
        """Create isolation manager with mocked dependencies."""
        return TestIsolationManager(mock_connection_manager)

    @pytest.mark.asyncio
    async def test_transaction_isolation_setup(self, isolation_manager, mock_connection_manager):
        """Test transaction-level isolation setup."""
        context = TestExecutionContext(
            test_name="test_1",
            isolation_level=TestIsolationLevel.TRANSACTION,
            cleanup_resources=[]
        )

        await isolation_manager.setup_isolation(context)

        mock_connection_manager.begin_transaction.assert_called_once()
        assert context.isolation_level == TestIsolationLevel.TRANSACTION

    @pytest.mark.asyncio
    async def test_schema_isolation_setup(self, isolation_manager, mock_connection_manager):
        """Test schema-level isolation setup."""
        context = TestExecutionContext(
            test_name="test_schema",
            isolation_level=TestIsolationLevel.SCHEMA,
            cleanup_resources=[]
        )

        mock_connection_manager.execute_query.return_value = Mock(success=True)

        await isolation_manager.setup_isolation(context)

        # Should create temporary schema
        assert mock_connection_manager.execute_query.call_count >= 2  # CREATE SCHEMA + USE SCHEMA
        assert context.temporary_schema is not None
        assert context.temporary_schema.startswith("test_schema_")

    @pytest.mark.asyncio
    async def test_isolation_cleanup_transaction(self, isolation_manager, mock_connection_manager):
        """Test transaction isolation cleanup."""
        context = TestExecutionContext(
            test_name="test_1",
            isolation_level=TestIsolationLevel.TRANSACTION,
            cleanup_resources=[]
        )

        await isolation_manager.cleanup_isolation(context)

        mock_connection_manager.rollback_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_isolation_cleanup_schema(self, isolation_manager, mock_connection_manager):
        """Test schema isolation cleanup."""
        context = TestExecutionContext(
            test_name="test_schema",
            isolation_level=TestIsolationLevel.SCHEMA,
            cleanup_resources=[],
            temporary_schema="test_schema_123"
        )

        mock_connection_manager.execute_query.return_value = Mock(success=True)

        await isolation_manager.cleanup_isolation(context)

        # Should drop temporary schema
        mock_connection_manager.execute_query.assert_called()
        call_args = mock_connection_manager.execute_query.call_args_list
        drop_schema_call = any("DROP SCHEMA" in str(call) for call in call_args)
        assert drop_schema_call


class TestMetricsCollector:
    """Test the test metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return TestMetricsCollector()

    def test_start_test_metrics(self, metrics_collector):
        """Test starting test metrics collection."""
        metrics_collector.start_test("test_1")

        assert "test_1" in metrics_collector.test_metrics
        assert metrics_collector.test_metrics["test_1"].start_time is not None

    def test_end_test_metrics_success(self, metrics_collector):
        """Test ending test metrics collection for successful test."""
        metrics_collector.start_test("test_1")
        metrics_collector.end_test("test_1", TestStatus.PASSED)

        test_metrics = metrics_collector.test_metrics["test_1"]
        assert test_metrics.end_time is not None
        assert test_metrics.duration > 0
        assert test_metrics.status == TestStatus.PASSED

    def test_end_test_metrics_failure(self, metrics_collector):
        """Test ending test metrics collection for failed test."""
        metrics_collector.start_test("test_1")
        metrics_collector.end_test("test_1", TestStatus.FAILED, "Assertion failed")

        test_metrics = metrics_collector.test_metrics["test_1"]
        assert test_metrics.status == TestStatus.FAILED
        assert test_metrics.error_message == "Assertion failed"

    def test_record_assertion_metrics(self, metrics_collector):
        """Test recording assertion metrics."""
        metrics_collector.start_test("test_1")
        metrics_collector.record_assertion("test_1", True, 0.01)
        metrics_collector.record_assertion("test_1", False, 0.02)

        test_metrics = metrics_collector.test_metrics["test_1"]
        assert test_metrics.assertion_count == 2
        assert test_metrics.passed_assertions == 1
        assert test_metrics.failed_assertions == 1
        assert test_metrics.total_assertion_time == 0.03

    def test_get_performance_summary(self, metrics_collector):
        """Test getting performance summary."""
        # Create some test metrics
        metrics_collector.start_test("test_1")
        metrics_collector.end_test("test_1", TestStatus.PASSED)

        metrics_collector.start_test("test_2")
        metrics_collector.end_test("test_2", TestStatus.FAILED)

        summary = metrics_collector.get_performance_summary()

        assert summary.total_tests == 2
        assert summary.passed_tests == 1
        assert summary.failed_tests == 1
        assert summary.total_execution_time > 0


class TestEnterpriseTestExecutor:
    """Test the main enterprise test executor functionality."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        manager.execute_query = AsyncMock()
        manager.begin_transaction = AsyncMock()
        manager.commit_transaction = AsyncMock()
        manager.rollback_transaction = AsyncMock()
        return manager

    @pytest.fixture
    def sample_test_case(self):
        """Create a sample test case."""
        return TestCase(
            name="sample_test",
            description="A sample test",
            sql="SELECT 1 as result",
            assertions=[],
            fixtures=[],
            isolation_level=TestIsolationLevel.TRANSACTION
        )

    @pytest.fixture
    def sample_test_suite(self, sample_test_case):
        """Create a sample test suite."""
        return TestSuite(
            name="sample_suite",
            description="A sample test suite",
            database="test_db",
            tests=[sample_test_case]
        )

    @pytest.fixture
    def executor(self, mock_connection_manager):
        """Create enterprise test executor."""
        return EnterpriseTestExecutor(mock_connection_manager)

    @pytest.mark.asyncio
    async def test_execute_single_test_success(self, executor, sample_test_case, mock_connection_manager):
        """Test executing a single test successfully."""
        # Mock successful query execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({'result': [1]})
        mock_connection_manager.execute_query.return_value = mock_result

        result = await executor.execute_test(sample_test_case)

        assert isinstance(result, TestResult)
        assert result.test_name == "sample_test"
        assert result.status == TestStatus.PASSED
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_execute_single_test_failure(self, executor, sample_test_case, mock_connection_manager):
        """Test executing a single test with failure."""
        # Mock failed query execution
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "SQL syntax error"
        mock_connection_manager.execute_query.return_value = mock_result

        result = await executor.execute_test(sample_test_case)

        assert result.status == TestStatus.FAILED
        assert "SQL syntax error" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_test_suite_sequential(self, executor, sample_test_suite, mock_connection_manager):
        """Test executing a test suite sequentially."""
        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({'result': [1]})
        mock_connection_manager.execute_query.return_value = mock_result

        results = await executor.execute_test_suite(sample_test_suite, parallel=False)

        assert len(results) == 1
        assert results[0].status == TestStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_test_suite_parallel(self, executor, mock_connection_manager):
        """Test executing a test suite in parallel."""
        # Create multiple test cases
        test_cases = [
            TestCase(
                name=f"test_{i}",
                sql="SELECT 1 as result",
                assertions=[],
                fixtures=[],
                isolation_level=TestIsolationLevel.TRANSACTION
            )
            for i in range(3)
        ]

        test_suite = TestSuite(
            name="parallel_suite",
            database="test_db",
            tests=test_cases
        )

        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({'result': [1]})
        mock_connection_manager.execute_query.return_value = mock_result

        results = await executor.execute_test_suite(test_suite, parallel=True, max_workers=2)

        assert len(results) == 3
        assert all(result.status == TestStatus.PASSED for result in results)

    @pytest.mark.asyncio
    async def test_test_dependency_resolution(self, executor, mock_connection_manager):
        """Test that test dependencies are properly resolved."""
        # Create tests with dependencies
        test1 = TestCase(
            name="test_1",
            sql="SELECT 1",
            assertions=[],
            fixtures=[],
            depends_on=[]
        )

        test2 = TestCase(
            name="test_2",
            sql="SELECT 2",
            assertions=[],
            fixtures=[],
            depends_on=["test_1"]
        )

        test3 = TestCase(
            name="test_3",
            sql="SELECT 3",
            assertions=[],
            fixtures=[],
            depends_on=["test_1", "test_2"]
        )

        test_suite = TestSuite(
            name="dependency_suite",
            database="test_db",
            tests=[test3, test1, test2]  # Intentionally out of order
        )

        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({'result': [1]})
        mock_connection_manager.execute_query.return_value = mock_result

        results = await executor.execute_test_suite(test_suite, parallel=False)

        # Check that tests were executed in dependency order
        result_names = [result.test_name for result in results]
        assert result_names.index("test_1") < result_names.index("test_2")
        assert result_names.index("test_2") < result_names.index("test_3")

    def test_metrics_collection_integration(self, executor):
        """Test that metrics are properly collected during execution."""
        metrics = executor.get_execution_metrics()

        assert hasattr(metrics, 'total_tests')
        assert hasattr(metrics, 'passed_tests')
        assert hasattr(metrics, 'failed_tests')
        assert hasattr(metrics, 'total_execution_time')

    @pytest.mark.asyncio
    async def test_isolation_level_enforcement(self, executor, mock_connection_manager):
        """Test that different isolation levels are properly enforced."""
        test_cases = [
            TestCase(
                name="test_none",
                sql="SELECT 1",
                assertions=[],
                fixtures=[],
                isolation_level=TestIsolationLevel.NONE
            ),
            TestCase(
                name="test_transaction",
                sql="SELECT 2",
                assertions=[],
                fixtures=[],
                isolation_level=TestIsolationLevel.TRANSACTION
            ),
            TestCase(
                name="test_schema",
                sql="SELECT 3",
                assertions=[],
                fixtures=[],
                isolation_level=TestIsolationLevel.SCHEMA
            )
        ]

        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({'result': [1]})
        mock_connection_manager.execute_query.return_value = mock_result

        for test_case in test_cases:
            result = await executor.execute_test(test_case)
            assert result.status == TestStatus.PASSED

        # Verify that different isolation methods were called
        assert mock_connection_manager.execute_query.called
        if any(tc.isolation_level == TestIsolationLevel.TRANSACTION for tc in test_cases):
            assert mock_connection_manager.begin_transaction.called

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, executor, sample_test_case, mock_connection_manager):
        """Test error handling and recovery mechanisms."""
        # Mock an exception during execution
        mock_connection_manager.execute_query.side_effect = Exception("Database connection lost")

        result = await executor.execute_test(sample_test_case)

        assert result.status == TestStatus.FAILED
        assert "Database connection lost" in result.error_message

        # Verify that cleanup still occurs
        assert mock_connection_manager.rollback_transaction.called


class TestIntegrationScenarios:
    """Integration tests for complex real-world scenarios."""

    @pytest.fixture
    def temp_db_file(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_test_execution(self, temp_db_file):
        """Test complete end-to-end test execution with real database."""
        from sqltest.db.connection import ConnectionManager
        from sqltest.config.models import DatabaseConfig

        # Setup real database connection
        db_config = DatabaseConfig(
            driver="sqlite",
            database=temp_db_file
        )

        connection_manager = ConnectionManager(db_config)
        executor = EnterpriseTestExecutor(connection_manager)

        # Create a realistic test case
        test_case = TestCase(
            name="integration_test",
            description="End-to-end integration test",
            sql="SELECT 42 as answer, 'hello' as greeting",
            assertions=[],
            fixtures=[],
            setup_sql="CREATE TABLE IF NOT EXISTS test_table (id INTEGER, name TEXT)",
            teardown_sql="DROP TABLE IF EXISTS test_table"
        )

        result = await executor.execute_test(test_case)

        assert result.status == TestStatus.PASSED
        assert result.execution_time > 0
        assert result.data is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, temp_db_file):
        """Test performance with larger datasets."""
        from sqltest.db.connection import ConnectionManager
        from sqltest.config.models import DatabaseConfig

        db_config = DatabaseConfig(
            driver="sqlite",
            database=temp_db_file
        )

        connection_manager = ConnectionManager(db_config)
        executor = EnterpriseTestExecutor(connection_manager)

        # Create test with larger dataset
        setup_sql = """
        CREATE TABLE large_table (
            id INTEGER PRIMARY KEY,
            value TEXT,
            category INTEGER
        );
        INSERT INTO large_table (value, category)
        VALUES """ + ", ".join([f"('value_{i}', {i % 10})" for i in range(1000)])

        test_case = TestCase(
            name="performance_test",
            sql="SELECT category, COUNT(*) as count FROM large_table GROUP BY category",
            assertions=[],
            fixtures=[],
            setup_sql=setup_sql,
            teardown_sql="DROP TABLE large_table"
        )

        result = await executor.execute_test(test_case)

        assert result.status == TestStatus.PASSED
        assert len(result.data) == 10  # 10 categories
        # Performance should be reasonable (under 5 seconds for this dataset)
        assert result.execution_time < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])