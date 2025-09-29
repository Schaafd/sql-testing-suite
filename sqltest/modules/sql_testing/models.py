"""
Core models for SQL unit testing framework.

This module defines the fundamental data structures for SQL unit testing including
test definitions, fixtures, results, and execution metadata.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, validator


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FixtureType(str, Enum):
    """Types of test fixtures."""
    CSV = "csv"
    JSON = "json"
    SQL = "sql"
    INLINE = "inline"
    GENERATED = "generated"


class AssertionType(str, Enum):
    """Types of assertions for SQL tests."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EMPTY = "empty"
    NOT_EMPTY = "not_empty"
    ROW_COUNT = "row_count"
    COLUMN_COUNT = "column_count"
    SCHEMA_MATCH = "schema_match"
    CUSTOM = "custom"


class TestIsolationLevel(str, Enum):
    """Test isolation levels for database testing."""
    NONE = "none"           # No isolation
    TRANSACTION = "transaction"  # Transaction-level isolation with rollback
    SCHEMA = "schema"        # Schema-level isolation with temporary schemas
    DATABASE = "database"    # Database-level isolation (most isolated)


@dataclass
class TestFixture:
    """Represents test data fixture."""
    name: str
    table_name: str
    fixture_type: FixtureType
    data_source: Union[str, Dict, List[Dict]]  # File path, inline data, or generator config
    schema: Optional[Dict[str, str]] = None  # Column name -> type mapping
    cleanup: bool = True  # Whether to clean up after test
    
    def __post_init__(self):
        """Validate fixture configuration."""
        if self.fixture_type in [FixtureType.CSV, FixtureType.JSON, FixtureType.SQL]:
            if not isinstance(self.data_source, str):
                raise ValueError(f"Fixture type {self.fixture_type} requires string data_source (file path)")
        elif self.fixture_type == FixtureType.INLINE:
            if not isinstance(self.data_source, (list, dict)):
                raise ValueError("Inline fixture requires list or dict data_source")


@dataclass
class TestAssertion:
    """Represents a test assertion."""
    assertion_type: AssertionType
    expected: Any = None
    message: Optional[str] = None
    tolerance: Optional[float] = None  # For numeric comparisons
    ignore_order: bool = False  # For result set comparisons
    custom_function: Optional[str] = None  # For custom assertions
    
    def __post_init__(self):
        """Validate assertion configuration."""
        if self.assertion_type == AssertionType.CUSTOM and not self.custom_function:
            raise ValueError("Custom assertion requires custom_function")


@dataclass
class SQLTest:
    """Represents a complete SQL unit test."""
    name: str
    description: str
    sql: str
    fixtures: List[TestFixture] = field(default_factory=list)
    assertions: List[TestAssertion] = field(default_factory=list)
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    timeout: int = 30  # Seconds
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)  # Test dependencies
    
    def __post_init__(self):
        """Validate test configuration."""
        if not self.sql.strip():
            raise ValueError("Test SQL cannot be empty")
        if not self.assertions:
            raise ValueError("Test must have at least one assertion")


@dataclass
class TestResult:
    """Represents the result of a single test execution."""
    test_name: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None  # Seconds
    error_message: Optional[str] = None
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    query_result: Optional[pd.DataFrame] = None
    row_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate execution time if both timestamps are available."""
        if self.end_time and self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in [TestStatus.FAILED, TestStatus.ERROR]


@dataclass
class TestSuite:
    """Represents a collection of SQL tests."""
    name: str
    description: str
    tests: List[SQLTest]
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def get_tests_by_tag(self, tag: str) -> List[SQLTest]:
        """Get all tests with a specific tag."""
        return [test for test in self.tests if tag in test.tags]
    
    def get_enabled_tests(self) -> List[SQLTest]:
        """Get all enabled tests."""
        return [test for test in self.tests if test.enabled]


@dataclass
class TestSuiteResult:
    """Represents the results of executing a test suite."""
    suite_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    test_results: List[TestResult] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate test counts from results."""
        if self.test_results:
            self.total_tests = len(self.test_results)
            self.passed_tests = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
            self.failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
            self.skipped_tests = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
            self.error_tests = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total execution time."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# Pydantic models for configuration parsing
class TestFixtureConfig(BaseModel):
    """Pydantic model for test fixture configuration."""
    name: str
    table_name: str
    fixture_type: FixtureType
    data_source: Union[str, Dict, List[Dict]]
    schema: Optional[Dict[str, str]] = None
    cleanup: bool = True
    
    def to_dataclass(self) -> TestFixture:
        """Convert to dataclass."""
        return TestFixture(**self.dict())


class TestAssertionConfig(BaseModel):
    """Pydantic model for test assertion configuration."""
    assertion_type: AssertionType
    expected: Optional[Any] = None
    message: Optional[str] = None
    tolerance: Optional[float] = None
    ignore_order: bool = False
    custom_function: Optional[str] = None
    
    def to_dataclass(self) -> TestAssertion:
        """Convert to dataclass."""
        return TestAssertion(**self.dict())


class SQLTestConfig(BaseModel):
    """Pydantic model for SQL test configuration."""
    name: str
    description: str
    sql: str
    fixtures: List[TestFixtureConfig] = Field(default_factory=list)
    assertions: List[TestAssertionConfig] = Field(default_factory=list)
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timeout: int = 30
    enabled: bool = True
    depends_on: List[str] = Field(default_factory=list)
    
    def to_dataclass(self) -> SQLTest:
        """Convert to dataclass."""
        return SQLTest(
            name=self.name,
            description=self.description,
            sql=self.sql,
            fixtures=[f.to_dataclass() for f in self.fixtures],
            assertions=[a.to_dataclass() for a in self.assertions],
            setup_sql=self.setup_sql,
            teardown_sql=self.teardown_sql,
            tags=self.tags,
            timeout=self.timeout,
            enabled=self.enabled,
            depends_on=self.depends_on
        )


class TestSuiteConfig(BaseModel):
    """Pydantic model for test suite configuration."""
    name: str
    description: str
    tests: List[SQLTestConfig]
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def to_dataclass(self) -> TestSuite:
        """Convert to dataclass."""
        return TestSuite(
            name=self.name,
            description=self.description,
            tests=[t.to_dataclass() for t in self.tests],
            setup_sql=self.setup_sql,
            teardown_sql=self.teardown_sql,
            tags=self.tags
        )


# Mock data generation configuration
@dataclass
class MockDataConfig:
    """Configuration for generating mock test data."""
    table_name: str
    rows: int
    columns: Dict[str, Dict[str, Any]]  # column_name -> {type, faker_provider, constraints}
    relationships: Optional[Dict[str, str]] = None  # foreign_key -> referenced_table.column
    seed: Optional[int] = None  # For reproducible data generation
    
    def __post_init__(self):
        """Validate mock data configuration."""
        if self.rows <= 0:
            raise ValueError("Row count must be positive")
        if not self.columns:
            raise ValueError("Must specify at least one column")


# Test coverage models
@dataclass
class TestCoverage:
    """Represents test coverage metrics."""
    total_statements: int = 0
    covered_statements: int = 0
    total_tables: int = 0
    covered_tables: int = 0
    total_columns: int = 0
    covered_columns: int = 0
    
    @property
    def statement_coverage(self) -> float:
        """Calculate statement coverage percentage."""
        if self.total_statements == 0:
            return 0.0
        return (self.covered_statements / self.total_statements) * 100
    
    @property
    def table_coverage(self) -> float:
        """Calculate table coverage percentage."""
        if self.total_tables == 0:
            return 0.0
        return (self.covered_tables / self.total_tables) * 100
    
    @property
    def column_coverage(self) -> float:
        """Calculate column coverage percentage."""
        if self.total_columns == 0:
            return 0.0
        return (self.covered_columns / self.total_columns) * 100
    
    @property
    def overall_coverage(self) -> float:
        """Calculate overall coverage as average of all metrics."""
        coverages = [self.statement_coverage, self.table_coverage, self.column_coverage]
        return sum(coverages) / len(coverages)
