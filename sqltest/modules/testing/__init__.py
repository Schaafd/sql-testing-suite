"""SQL unit testing framework with comprehensive test execution, assertions, and coverage.

Components:
- TestExecutionEngine: Execute tests with isolation and parallelization
- SQLAssertions: Rich assertion library for SQL testing
- MockDataGenerator: Generate realistic test data
- CoverageTracker: Track query and table coverage
- TestReportGenerator: Generate HTML, JUnit XML, and JSON reports
- CICDIntegration: Integrate with CI/CD platforms
"""

from sqltest.modules.testing.test_runner import (
    TestExecutionEngine,
    TestCase,
    TestSuite,
    TestResult,
    TestStatus,
    TestPriority,
    TestAssertion,
)

from sqltest.modules.testing.assertions import (
    SQLAssertions,
    AssertionError,
    assert_that,
)

from sqltest.modules.testing.fixtures import (
    MockDataGenerator,
    FixtureDefinition,
    FixtureManager,
)

from sqltest.modules.testing.reporting import (
    CoverageTracker,
    CoverageMetrics,
    TestReportGenerator,
    CICDIntegration,
    TestHistoryTracker,
)


__all__ = [
    # Test Execution
    'TestExecutionEngine',
    'TestCase',
    'TestSuite',
    'TestResult',
    'TestStatus',
    'TestPriority',
    'TestAssertion',

    # Assertions
    'SQLAssertions',
    'AssertionError',
    'assert_that',

    # Fixtures
    'MockDataGenerator',
    'FixtureDefinition',
    'FixtureManager',

    # Coverage & Reporting
    'CoverageTracker',
    'CoverageMetrics',
    'TestReportGenerator',
    'CICDIntegration',
    'TestHistoryTracker',
]