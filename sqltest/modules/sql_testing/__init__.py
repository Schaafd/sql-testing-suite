"""
SQL Unit Testing Framework for SQLTest Pro.

This module provides comprehensive SQL unit testing capabilities including:
- Test fixtures with multiple data sources (CSV, JSON, SQL, inline, generated)
- Assertion system with multiple assertion types
- Test execution with dependency management and parallel execution
- YAML configuration support with environment variable interpolation
- Test coverage and reporting
"""
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import asyncio

from sqltest.db.connection import ConnectionManager
from .models import (
    SQLTest, TestSuite, TestResult, TestSuiteResult, TestStatus,
    TestFixture, TestAssertion, FixtureType, AssertionType,
    TestCoverage
)
from .executor import TestExecutor
from .config_loader import TestConfigLoader, create_test_suite_from_yaml
from .fixtures import FixtureManager


class SQLTestRunner:
    """
    Main interface for running SQL unit tests.
    
    Provides high-level API for executing tests, managing configurations,
    and generating reports.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize SQL test runner.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.executor = TestExecutor(connection_manager)
        self.config_loader = TestConfigLoader()
        
    async def run_test(self, test: SQLTest) -> TestResult:
        """
        Run a single SQL test.
        
        Args:
            test: SQL test to execute
            
        Returns:
            Test execution result
        """
        return await self.executor.execute_test(test)
    
    async def run_test_suite(self, test_suite: TestSuite) -> TestSuiteResult:
        """
        Run a complete test suite.
        
        Args:
            test_suite: Test suite to execute
            
        Returns:
            Test suite execution result
        """
        return await self.executor.execute_test_suite(test_suite)
    
    async def run_tests_from_config(
        self, 
        config_path: Union[str, Path]
    ) -> TestSuiteResult:
        """
        Load and run tests from YAML configuration file.
        
        Args:
            config_path: Path to YAML test configuration
            
        Returns:
            Test suite execution result
        """
        # Load configuration
        config = self.config_loader.load_test_suite(config_path)
        
        # Validate configuration
        validation_errors = self.config_loader.validate_configuration(config)
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
        
        # Convert to dataclasses and run
        test_suite = config.to_dataclass()
        return await self.run_test_suite(test_suite)
    
    async def run_tests_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "**/*test*.yaml",
        parallel: bool = False
    ) -> List[TestSuiteResult]:
        """
        Discover and run all test configurations in a directory.
        
        Args:
            directory: Directory to search for test configurations
            pattern: Glob pattern for matching test files
            parallel: Whether to run test suites in parallel
            
        Returns:
            List of test suite execution results
        """
        # Discover configurations
        configs = self.config_loader.discover_test_configurations(directory, pattern)
        
        results = []
        
        if parallel:
            # Run test suites in parallel
            tasks = []
            for config in configs:
                test_suite = config.to_dataclass()
                tasks.append(self.run_test_suite(test_suite))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = TestSuiteResult(
                        suite_name=configs[i].name,
                        start_time=asyncio.get_event_loop().time()
                    )
                    error_result.error_message = str(result)
                    results[i] = error_result
        else:
            # Run test suites sequentially
            for config in configs:
                try:
                    test_suite = config.to_dataclass()
                    result = await self.run_test_suite(test_suite)
                    results.append(result)
                except Exception as e:
                    error_result = TestSuiteResult(
                        suite_name=config.name,
                        start_time=asyncio.get_event_loop().time()
                    )
                    error_result.error_message = str(e)
                    results.append(error_result)
        
        return results
    
    async def run_tests_by_tags(
        self, 
        test_suite: TestSuite, 
        tags: List[str],
        match_all: bool = False
    ) -> TestSuiteResult:
        """
        Run tests filtered by tags.
        
        Args:
            test_suite: Test suite containing tests
            tags: List of tags to filter by
            match_all: If True, test must have all tags; if False, any tag matches
            
        Returns:
            Test suite execution result
        """
        # Filter tests by tags
        filtered_tests = []
        for test in test_suite.tests:
            if match_all:
                # Test must have all specified tags
                if all(tag in test.tags for tag in tags):
                    filtered_tests.append(test)
            else:
                # Test must have at least one of the specified tags
                if any(tag in test.tags for tag in tags):
                    filtered_tests.append(test)
        
        # Create filtered test suite
        filtered_suite = TestSuite(
            name=f"{test_suite.name}_filtered",
            description=f"Filtered by tags: {', '.join(tags)}",
            tests=filtered_tests,
            setup_sql=test_suite.setup_sql,
            teardown_sql=test_suite.teardown_sql,
            tags=test_suite.tags
        )
        
        return await self.run_test_suite(filtered_suite)
    
    def validate_configuration(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate test configuration without running tests.
        
        Args:
            config_path: Path to YAML test configuration
            
        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            config = self.config_loader.load_test_suite(config_path)
            return self.config_loader.validate_configuration(config)
        except Exception as e:
            return [f"Failed to load configuration: {str(e)}"]
    
    def generate_test_coverage(
        self, 
        results: List[TestSuiteResult]
    ) -> TestCoverage:
        """
        Generate test coverage metrics from test results.
        
        Args:
            results: List of test suite results
            
        Returns:
            Test coverage metrics
        """
        coverage = TestCoverage()
        
        # Count total and covered elements
        covered_tables = set()
        total_statements = 0
        covered_statements = 0
        
        for suite_result in results:
            for test_result in suite_result.test_results:
                total_statements += 1
                if test_result.passed:
                    covered_statements += 1
                
                # Extract table names from SQL (simple heuristic)
                if test_result.query_result is not None:
                    # This could be enhanced with SQL parsing
                    covered_tables.add("table_from_sql")  # Placeholder
        
        coverage.total_statements = total_statements
        coverage.covered_statements = covered_statements
        coverage.total_tables = len(covered_tables) if covered_tables else 1
        coverage.covered_tables = len(covered_tables)
        
        return coverage
    
    def create_test_summary(
        self, 
        results: List[TestSuiteResult]
    ) -> Dict[str, Any]:
        """
        Create a summary of test execution results.
        
        Args:
            results: List of test suite results
            
        Returns:
            Dictionary containing execution summary
        """
        total_suites = len(results)
        total_tests = sum(r.total_tests for r in results)
        total_passed = sum(r.passed_tests for r in results)
        total_failed = sum(r.failed_tests for r in results)
        total_skipped = sum(r.skipped_tests for r in results)
        total_errors = sum(r.error_tests for r in results)
        
        # Calculate total execution time
        total_execution_time = sum(
            r.execution_time for r in results 
            if r.execution_time is not None
        )
        
        # Calculate overall success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_suites': total_suites,
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_failed,
            'skipped_tests': total_skipped,
            'error_tests': total_errors,
            'success_rate': round(success_rate, 2),
            'total_execution_time': round(total_execution_time, 2),
            'suite_results': [
                {
                    'name': r.suite_name,
                    'tests': r.total_tests,
                    'passed': r.passed_tests,
                    'failed': r.failed_tests,
                    'success_rate': round(r.success_rate * 100, 2),
                    'execution_time': r.execution_time
                }
                for r in results
            ]
        }
    
    def clear_execution_state(self) -> None:
        """Clear execution state for fresh test runs."""
        self.executor.clear_execution_state()


# Convenience functions for common operations
async def run_sql_test(
    connection_manager: ConnectionManager,
    test: SQLTest
) -> TestResult:
    """
    Convenience function to run a single SQL test.
    
    Args:
        connection_manager: Database connection manager
        test: SQL test to execute
        
    Returns:
        Test execution result
    """
    runner = SQLTestRunner(connection_manager)
    return await runner.run_test(test)


async def run_tests_from_yaml(
    connection_manager: ConnectionManager,
    config_path: Union[str, Path]
) -> TestSuiteResult:
    """
    Convenience function to run tests from YAML configuration.
    
    Args:
        connection_manager: Database connection manager
        config_path: Path to YAML test configuration
        
    Returns:
        Test suite execution result
    """
    runner = SQLTestRunner(connection_manager)
    return await runner.run_tests_from_config(config_path)


# Export main classes and functions
__all__ = [
    # Main classes
    'SQLTestRunner',
    
    # Models
    'SQLTest',
    'TestSuite', 
    'TestResult',
    'TestSuiteResult',
    'TestStatus',
    'TestFixture',
    'TestAssertion',
    'FixtureType',
    'AssertionType',
    'TestCoverage',
    
    # Core components
    'TestExecutor',
    'FixtureManager',
    'TestConfigLoader',
    
    # Convenience functions
    'run_sql_test',
    'run_tests_from_yaml',
    'create_test_suite_from_yaml',
]
