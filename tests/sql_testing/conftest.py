"""
Pytest configuration and fixtures for SQL testing framework tests.

This module provides common fixtures and configuration for testing
the SQL testing framework components.
"""
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import pandas as pd

from sqltest.db.connection import ConnectionManager
from sqltest.config.models import DatabaseConfig


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager for testing."""
    manager = Mock(spec=ConnectionManager)
    manager.execute_query = AsyncMock()
    manager.begin_transaction = AsyncMock()
    manager.commit_transaction = AsyncMock()
    manager.rollback_transaction = AsyncMock()

    # Default successful response
    def default_execute_query(sql, params=None):
        result = Mock()
        result.success = True
        result.data = pd.DataFrame({'result': [1]})
        result.error = None
        return result

    manager.execute_query.side_effect = default_execute_query
    return manager


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Marketing'],
        'active': [True, True, False, True, True]
    })


@pytest.fixture
def sqlite_test_config():
    """Create a test database configuration for SQLite."""
    return DatabaseConfig(
        driver="sqlite",
        database=":memory:"
    )


@pytest.fixture
def sample_test_data():
    """Create sample test data for fixtures."""
    return [
        {'id': 1, 'username': 'alice', 'email': 'alice@example.com', 'created_at': '2023-01-01'},
        {'id': 2, 'username': 'bob', 'email': 'bob@example.com', 'created_at': '2023-01-02'},
        {'id': 3, 'username': 'charlie', 'email': 'charlie@example.com', 'created_at': '2023-01-03'}
    ]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "database: Tests requiring database")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Add performance marker to performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid.lower() for keyword in ["large", "bulk", "stress"]):
            item.add_marker(pytest.mark.slow)

        # Add database marker to tests that use real databases
        if "real_db" in item.fixturenames or "sqlite_test_config" in item.fixturenames:
            item.add_marker(pytest.mark.database)


@pytest.fixture(scope="session")
def test_data_directory():
    """Create a session-scoped test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create sample CSV file
        csv_content = """id,name,department,salary
1,Alice,Engineering,50000
2,Bob,Marketing,60000
3,Charlie,Sales,70000"""

        csv_file = test_dir / "sample_data.csv"
        with open(csv_file, 'w') as f:
            f.write(csv_content)

        # Create sample JSON file
        import json
        json_data = [
            {"id": 1, "product": "Widget A", "price": 10.99},
            {"id": 2, "product": "Widget B", "price": 15.99},
            {"id": 3, "product": "Widget C", "price": 20.99}
        ]

        json_file = test_dir / "sample_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        yield test_dir


# Custom assertions for testing
class SQLTestAssertions:
    """Custom assertions for SQL testing framework tests."""

    @staticmethod
    def assert_test_result_passed(test_result):
        """Assert that a test result indicates success."""
        assert test_result.status.value == 'passed', f"Test failed: {test_result.error_message}"

    @staticmethod
    def assert_test_result_failed(test_result):
        """Assert that a test result indicates failure."""
        assert test_result.status.value == 'failed', "Test was expected to fail but passed"

    @staticmethod
    def assert_dataframes_equal(df1, df2, ignore_order=False):
        """Assert that two DataFrames are equal."""
        if ignore_order:
            df1 = df1.sort_index().reset_index(drop=True)
            df2 = df2.sort_index().reset_index(drop=True)

        pd.testing.assert_frame_equal(df1, df2)

    @staticmethod
    def assert_execution_time_reasonable(execution_time, max_time=10.0):
        """Assert that execution time is reasonable."""
        assert 0 < execution_time < max_time, f"Execution time {execution_time}s is not reasonable"


@pytest.fixture
def sql_test_assertions():
    """Provide custom assertions for SQL testing."""
    return SQLTestAssertions()


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Provide a performance timing context manager."""
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer():
        start_time = time.time()
        yield lambda: time.time() - start_time

    return timer


# Async testing utilities
@pytest.fixture
def async_test_runner():
    """Provide utilities for running async tests."""
    class AsyncTestRunner:
        @staticmethod
        async def run_with_timeout(coro, timeout=10.0):
            """Run coroutine with timeout."""
            return await asyncio.wait_for(coro, timeout=timeout)

        @staticmethod
        async def run_concurrent(coros, max_concurrent=5):
            """Run multiple coroutines concurrently."""
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_semaphore(coro):
                async with semaphore:
                    return await coro

            return await asyncio.gather(*[run_with_semaphore(coro) for coro in coros])

    return AsyncTestRunner()


# Error simulation utilities
@pytest.fixture
def error_simulator():
    """Provide utilities for simulating various error conditions."""
    class ErrorSimulator:
        @staticmethod
        def connection_error():
            """Simulate database connection error."""
            return Exception("Database connection failed")

        @staticmethod
        def timeout_error():
            """Simulate timeout error."""
            return asyncio.TimeoutError("Operation timed out")

        @staticmethod
        def sql_error():
            """Simulate SQL execution error."""
            return Exception("SQL syntax error")

        @staticmethod
        def permission_error():
            """Simulate permission error."""
            return PermissionError("Access denied")

    return ErrorSimulator()
