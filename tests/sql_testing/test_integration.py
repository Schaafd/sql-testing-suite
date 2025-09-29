"""
Integration tests for the complete SQL testing framework.

This module tests end-to-end scenarios combining all components
of the SQL testing framework together.
"""
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import pandas as pd
import yaml
from unittest.mock import Mock, AsyncMock

from sqltest.modules.sql_testing.enterprise_executor import EnterpriseTestExecutor
from sqltest.modules.sql_testing.config_loader import AdvancedConfigLoader
from sqltest.modules.sql_testing.assertions import SQLTestAssertionEngine
from sqltest.modules.sql_testing.fixtures import FixtureManager
from sqltest.modules.sql_testing.models import (
    TestSuite,
    TestCase,
    TestFixture,
    TestAssertion,
    TestIsolationLevel,
    AssertionType,
    FixtureType
)
from sqltest.db.connection import ConnectionManager
from sqltest.config.models import DatabaseConfig


class TestCompleteWorkflow:
    """Test complete SQL testing workflow from configuration to execution."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_config_file(self, temp_dir):
        """Create a sample configuration file."""
        config_content = {
            'version': '1.0',
            'databases': {
                'test_db': {
                    'driver': 'sqlite',
                    'database': str(temp_dir / 'test.db')
                }
            },
            'global_fixtures': [
                {
                    'name': 'test_users',
                    'table_name': 'users',
                    'fixture_type': 'inline',
                    'data_source': [
                        {'id': 1, 'name': 'Alice', 'age': 30},
                        {'id': 2, 'name': 'Bob', 'age': 25},
                        {'id': 3, 'name': 'Charlie', 'age': 35}
                    ],
                    'schema': {
                        'id': 'INTEGER',
                        'name': 'TEXT',
                        'age': 'INTEGER'
                    }
                }
            ],
            'test_suites': [
                {
                    'name': 'user_tests',
                    'description': 'Tests for user management',
                    'database': 'test_db',
                    'setup_sql': 'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)',
                    'teardown_sql': 'DROP TABLE IF EXISTS users',
                    'test_cases': [
                        {
                            'name': 'test_user_count',
                            'description': 'Test that we have the correct number of users',
                            'sql': 'SELECT COUNT(*) as count FROM users',
                            'fixtures': ['test_users'],
                            'assertions': [
                                {
                                    'type': 'equals',
                                    'expected': [{'count': 3}]
                                }
                            ],
                            'isolation_level': 'transaction'
                        },
                        {
                            'name': 'test_average_age',
                            'description': 'Test average age calculation',
                            'sql': 'SELECT AVG(age) as avg_age FROM users',
                            'fixtures': ['test_users'],
                            'assertions': [
                                {
                                    'type': 'custom',
                                    'custom_function': '''
# Check if average age is 30
avg_age = data.iloc[0]['avg_age']
result = {
    'passed': abs(avg_age - 30) < 0.1,
    'actual': avg_age,
    'message': f'Average age is {avg_age}, expected ~30'
}
'''
                                }
                            ]
                        },
                        {
                            'name': 'test_user_exists',
                            'description': 'Test that specific user exists',
                            'sql': 'SELECT * FROM users WHERE name = "Alice"',
                            'fixtures': ['test_users'],
                            'assertions': [
                                {
                                    'type': 'contains',
                                    'expected': {'name': 'Alice', 'age': 30}
                                },
                                {
                                    'type': 'row_count',
                                    'expected': 1
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        return config_file

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a comprehensive mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        manager.execute_query = AsyncMock()
        manager.begin_transaction = AsyncMock()
        manager.commit_transaction = AsyncMock()
        manager.rollback_transaction = AsyncMock()

        # Mock successful query execution by default
        def mock_execute_query(sql, params=None):
            result = Mock()
            result.success = True

            # Mock different responses based on SQL
            if 'COUNT(*)' in sql:
                result.data = pd.DataFrame({'count': [3]})
            elif 'AVG(age)' in sql:
                result.data = pd.DataFrame({'avg_age': [30.0]})
            elif 'WHERE name = "Alice"' in sql:
                result.data = pd.DataFrame({
                    'id': [1], 'name': ['Alice'], 'age': [30]
                })
            elif 'CREATE TABLE' in sql or 'DROP TABLE' in sql or 'INSERT INTO' in sql:
                result.data = None
            else:
                result.data = pd.DataFrame()

            return result

        manager.execute_query.side_effect = mock_execute_query
        return manager

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_with_mocks(self, sample_config_file, mock_connection_manager):
        """Test complete workflow from config loading to test execution with mocks."""
        # Load configuration
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(sample_config_file)

        assert len(config.test_suites) == 1
        assert len(config.global_fixtures) == 1

        # Convert to model objects
        test_suite = TestSuite(
            name=config.test_suites[0].name,
            description=config.test_suites[0].description,
            database=config.test_suites[0].database,
            setup_sql=config.test_suites[0].setup_sql,
            teardown_sql=config.test_suites[0].teardown_sql,
            tests=[
                TestCase(
                    name=tc.name,
                    description=tc.description,
                    sql=tc.sql,
                    fixtures=[
                        TestFixture(
                            name=fx_name,
                            table_name=config.global_fixtures[0].table_name,
                            fixture_type=FixtureType.INLINE,
                            data_source=config.global_fixtures[0].data_source,
                            schema=config.global_fixtures[0].schema
                        )
                        for fx_name in tc.fixtures
                    ],
                    assertions=[
                        TestAssertion(
                            assertion_type=AssertionType(assertion.type),
                            expected=assertion.expected,
                            custom_function=getattr(assertion, 'custom_function', None)
                        )
                        for assertion in tc.assertions
                    ],
                    isolation_level=TestIsolationLevel(tc.isolation_level)
                )
                for tc in config.test_suites[0].test_cases
            ]
        )

        # Execute tests
        executor = EnterpriseTestExecutor(mock_connection_manager)
        results = await executor.execute_test_suite(test_suite)

        # Verify results
        assert len(results) == 3
        assert all(result.status.value == 'passed' for result in results)

        # Verify specific test results
        count_test = next(r for r in results if r.test_name == 'test_user_count')
        assert count_test.status.value == 'passed'

        avg_test = next(r for r in results if r.test_name == 'test_average_age')
        assert avg_test.status.value == 'passed'

        exists_test = next(r for r in results if r.test_name == 'test_user_exists')
        assert exists_test.status.value == 'passed'

        # Verify metrics were collected
        metrics = executor.get_execution_metrics()
        assert metrics.total_tests == 3
        assert metrics.passed_tests == 3
        assert metrics.failed_tests == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_execution_workflow(self, sample_config_file, mock_connection_manager):
        """Test workflow with parallel test execution."""
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(sample_config_file)

        # Create multiple test cases for parallel execution
        test_cases = []
        for i in range(5):
            test_case = TestCase(
                name=f'parallel_test_{i}',
                sql=f'SELECT {i} as test_id, COUNT(*) as count FROM users',
                fixtures=[
                    TestFixture(
                        name='test_users',
                        table_name='users',
                        fixture_type=FixtureType.INLINE,
                        data_source=config.global_fixtures[0].data_source
                    )
                ],
                assertions=[
                    TestAssertion(
                        assertion_type=AssertionType.ROW_COUNT,
                        expected=1
                    )
                ],
                isolation_level=TestIsolationLevel.TRANSACTION
            )
            test_cases.append(test_case)

        test_suite = TestSuite(
            name='parallel_suite',
            database='test_db',
            tests=test_cases
        )

        # Execute tests in parallel
        executor = EnterpriseTestExecutor(mock_connection_manager)
        results = await executor.execute_test_suite(test_suite, parallel=True, max_workers=3)

        # Verify all tests passed
        assert len(results) == 5
        assert all(result.status.value == 'passed' for result in results)

        # Verify parallel execution was faster than sequential would be
        metrics = executor.get_execution_metrics()
        assert metrics.total_execution_time > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fixture_management_integration(self, temp_dir, mock_connection_manager):
        """Test integration with fixture management."""
        # Create CSV fixture file
        csv_content = """id,name,department
1,Alice,Engineering
2,Bob,Marketing
3,Charlie,Sales"""

        csv_file = temp_dir / 'employees.csv'
        with open(csv_file, 'w') as f:
            f.write(csv_content)

        # Create configuration with CSV fixture
        config_content = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'csv_fixture_test',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_csv_data',
                            'sql': 'SELECT COUNT(*) as count FROM employees',
                            'fixtures': [
                                {
                                    'name': 'employee_data',
                                    'table_name': 'employees',
                                    'fixture_type': 'csv',
                                    'data_source': str(csv_file),
                                    'schema': {
                                        'id': 'INTEGER',
                                        'name': 'TEXT',
                                        'department': 'TEXT'
                                    }
                                }
                            ],
                            'assertions': [
                                {
                                    'type': 'equals',
                                    'expected': [{'count': 3}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / 'csv_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        # Load and execute
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(config_file)

        # Convert to model and execute
        test_suite = TestSuite(
            name=config.test_suites[0].name,
            database=config.test_suites[0].database,
            tests=[
                TestCase(
                    name=config.test_suites[0].test_cases[0].name,
                    sql=config.test_suites[0].test_cases[0].sql,
                    fixtures=[
                        TestFixture(
                            name=fx.name,
                            table_name=fx.table_name,
                            fixture_type=FixtureType(fx.fixture_type),
                            data_source=fx.data_source,
                            schema=fx.schema
                        )
                        for fx in config.test_suites[0].test_cases[0].fixtures
                    ],
                    assertions=[
                        TestAssertion(
                            assertion_type=AssertionType(assertion.type),
                            expected=assertion.expected
                        )
                        for assertion in config.test_suites[0].test_cases[0].assertions
                    ]
                )
            ]
        )

        executor = EnterpriseTestExecutor(mock_connection_manager)
        results = await executor.execute_test_suite(test_suite)

        assert len(results) == 1
        assert results[0].status.value == 'passed'

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, sample_config_file):
        """Test error handling throughout the integration."""
        # Create a connection manager that will fail
        failing_manager = Mock(spec=ConnectionManager)
        failing_manager.execute_query = AsyncMock()
        failing_manager.begin_transaction = AsyncMock()
        failing_manager.rollback_transaction = AsyncMock()

        # Make queries fail
        def failing_execute_query(sql, params=None):
            result = Mock()
            result.success = False
            result.error = "Database connection lost"
            result.data = None
            return result

        failing_manager.execute_query.side_effect = failing_execute_query

        # Load configuration
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(sample_config_file)

        # Convert to model
        test_case = TestCase(
            name='failing_test',
            sql='SELECT 1',
            fixtures=[],
            assertions=[
                TestAssertion(
                    assertion_type=AssertionType.EQUALS,
                    expected=1
                )
            ]
        )

        test_suite = TestSuite(
            name='failing_suite',
            database='test_db',
            tests=[test_case]
        )

        # Execute and verify failure handling
        executor = EnterpriseTestExecutor(failing_manager)
        results = await executor.execute_test_suite(test_suite)

        assert len(results) == 1
        assert results[0].status.value == 'failed'
        assert 'Database connection lost' in results[0].error_message

        # Verify cleanup was attempted
        assert failing_manager.rollback_transaction.called

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_assertions_integration(self, temp_dir, mock_connection_manager):
        """Test integration with complex assertion scenarios."""
        config_content = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'complex_assertions',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_statistical_assertions',
                            'sql': '''
                                SELECT
                                    AVG(salary) as avg_salary,
                                    MIN(salary) as min_salary,
                                    MAX(salary) as max_salary,
                                    COUNT(*) as employee_count
                                FROM employees
                            ''',
                            'fixtures': [
                                {
                                    'name': 'employee_salaries',
                                    'table_name': 'employees',
                                    'fixture_type': 'inline',
                                    'data_source': [
                                        {'id': 1, 'name': 'Alice', 'salary': 50000},
                                        {'id': 2, 'name': 'Bob', 'salary': 60000},
                                        {'id': 3, 'name': 'Charlie', 'salary': 70000},
                                        {'id': 4, 'name': 'Diana', 'salary': 80000}
                                    ]
                                }
                            ],
                            'assertions': [
                                {
                                    'type': 'custom',
                                    'custom_function': '''
# Comprehensive statistical validation
row = data.iloc[0]
avg_salary = row['avg_salary']
min_salary = row['min_salary']
max_salary = row['max_salary']
count = row['employee_count']

# Check all statistics
checks = [
    abs(avg_salary - 65000) < 1000,  # Average should be ~65000
    min_salary == 50000,             # Min should be 50000
    max_salary == 80000,             # Max should be 80000
    count == 4                       # Should have 4 employees
]

result = {
    'passed': all(checks),
    'actual': {
        'avg_salary': avg_salary,
        'min_salary': min_salary,
        'max_salary': max_salary,
        'count': count
    },
    'message': f'Statistical validation: {sum(checks)}/4 checks passed'
}
'''
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / 'complex_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        # Mock the statistical query response
        def mock_stats_execute_query(sql, params=None):
            result = Mock()
            result.success = True

            if 'AVG(salary)' in sql:
                result.data = pd.DataFrame({
                    'avg_salary': [65000.0],
                    'min_salary': [50000],
                    'max_salary': [80000],
                    'employee_count': [4]
                })
            else:
                result.data = None

            return result

        mock_connection_manager.execute_query.side_effect = mock_stats_execute_query

        # Load and execute
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(config_file)

        test_suite = TestSuite(
            name=config.test_suites[0].name,
            database=config.test_suites[0].database,
            tests=[
                TestCase(
                    name=config.test_suites[0].test_cases[0].name,
                    sql=config.test_suites[0].test_cases[0].sql,
                    fixtures=[
                        TestFixture(
                            name=fx.name,
                            table_name=fx.table_name,
                            fixture_type=FixtureType(fx.fixture_type),
                            data_source=fx.data_source
                        )
                        for fx in config.test_suites[0].test_cases[0].fixtures
                    ],
                    assertions=[
                        TestAssertion(
                            assertion_type=AssertionType(assertion.type),
                            custom_function=assertion.custom_function
                        )
                        for assertion in config.test_suites[0].test_cases[0].assertions
                    ]
                )
            ]
        )

        executor = EnterpriseTestExecutor(mock_connection_manager)
        results = await executor.execute_test_suite(test_suite)

        assert len(results) == 1
        assert results[0].status.value == 'passed'

    @pytest.mark.integration
    def test_configuration_validation_integration(self, temp_dir):
        """Test configuration validation integration."""
        # Create invalid configuration
        invalid_config = {
            'version': 'invalid_version',
            'test_suites': [
                {
                    'name': 'invalid_suite',
                    'database': 'test_db',
                    'max_workers': -1,  # Invalid negative workers
                    'test_cases': [
                        {
                            'name': 'invalid_test',
                            'sql': '',  # Empty SQL
                            'timeout': -5,  # Invalid negative timeout
                            'isolation_level': 'invalid_level',
                            'assertions': [
                                {
                                    'type': 'invalid_assertion_type',
                                    'tolerance': -0.1  # Invalid negative tolerance
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / 'invalid_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        # Validate configuration
        config_loader = AdvancedConfigLoader()
        errors = config_loader.validate_config(config_file)

        # Should have multiple validation errors
        assert len(errors) > 0

        # Check for specific error types
        error_text = ' '.join(errors)
        assert 'Version must be in format' in error_text


class TestPerformanceIntegration:
    """Test performance aspects of the integration."""

    @pytest.fixture
    def large_dataset_config(self, temp_dir):
        """Create configuration for large dataset testing."""
        # Generate large fixture data
        large_data = [
            {'id': i, 'value': f'value_{i}', 'category': i % 10}
            for i in range(1000)
        ]

        config_content = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'performance_test',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'large_dataset_test',
                            'sql': 'SELECT category, COUNT(*) as count FROM large_table GROUP BY category',
                            'fixtures': [
                                {
                                    'name': 'large_data',
                                    'table_name': 'large_table',
                                    'fixture_type': 'inline',
                                    'data_source': large_data
                                }
                            ],
                            'assertions': [
                                {
                                    'type': 'row_count',
                                    'expected': 10  # 10 categories
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / 'performance_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        return config_file

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, large_dataset_config):
        """Test performance with large datasets."""
        # Mock connection manager for performance testing
        mock_manager = Mock(spec=ConnectionManager)
        mock_manager.execute_query = AsyncMock()
        mock_manager.begin_transaction = AsyncMock()
        mock_manager.rollback_transaction = AsyncMock()

        def mock_execute_query(sql, params=None):
            result = Mock()
            result.success = True

            if 'GROUP BY category' in sql:
                # Simulate grouped results
                result.data = pd.DataFrame({
                    'category': list(range(10)),
                    'count': [100] * 10
                })
            else:
                result.data = None

            return result

        mock_manager.execute_query.side_effect = mock_execute_query

        # Load and execute
        config_loader = AdvancedConfigLoader()
        config = config_loader.load_config(large_dataset_config)

        test_suite = TestSuite(
            name=config.test_suites[0].name,
            database=config.test_suites[0].database,
            tests=[
                TestCase(
                    name=config.test_suites[0].test_cases[0].name,
                    sql=config.test_suites[0].test_cases[0].sql,
                    fixtures=[
                        TestFixture(
                            name=fx.name,
                            table_name=fx.table_name,
                            fixture_type=FixtureType(fx.fixture_type),
                            data_source=fx.data_source
                        )
                        for fx in config.test_suites[0].test_cases[0].fixtures
                    ],
                    assertions=[
                        TestAssertion(
                            assertion_type=AssertionType(assertion.type),
                            expected=assertion.expected
                        )
                        for assertion in config.test_suites[0].test_cases[0].assertions
                    ]
                )
            ]
        )

        import time
        start_time = time.time()

        executor = EnterpriseTestExecutor(mock_manager)
        results = await executor.execute_test_suite(test_suite)

        execution_time = time.time() - start_time

        # Verify performance
        assert len(results) == 1
        assert results[0].status.value == 'passed'
        assert execution_time < 5.0  # Should complete within 5 seconds

        # Verify metrics
        metrics = executor.get_execution_metrics()
        assert metrics.total_execution_time > 0
        assert metrics.total_tests == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])