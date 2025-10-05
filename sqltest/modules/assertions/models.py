"""Assertion models and types for SQLTest Pro."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import pandas as pd


class AssertionType(str, Enum):
    """Types of assertions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    HAS_LENGTH = "has_length"
    HAS_MIN_LENGTH = "has_min_length"
    HAS_MAX_LENGTH = "has_max_length"
    IN_RANGE = "in_range"
    NOT_IN_RANGE = "not_in_range"
    IS_UNIQUE = "is_unique"
    HAS_DUPLICATES = "has_duplicates"
    ROW_COUNT = "row_count"
    COLUMN_COUNT = "column_count"
    SCHEMA_MATCHES = "schema_matches"
    CUSTOM = "custom"


class AssertionLevel(str, Enum):
    """Assertion severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AssertionResult:
    """Result of an assertion check."""
    assertion_name: str
    assertion_type: AssertionType
    passed: bool
    level: AssertionLevel
    message: str
    expected: Any = None
    actual: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    execution_timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0


@dataclass
class QueryAssertionResult:
    """Result of assertions on a query result set."""
    query: str
    query_hash: str
    assertions: List[AssertionResult] = field(default_factory=list)
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time_ms: float = 0.0
    
    @property
    def passed_count(self) -> int:
        """Number of assertions that passed."""
        return sum(1 for assertion in self.assertions if assertion.passed)
    
    @property
    def failed_count(self) -> int:
        """Number of assertions that failed."""
        return sum(1 for assertion in self.assertions if not assertion.passed)
    
    @property
    def total_count(self) -> int:
        """Total number of assertions."""
        return len(self.assertions)
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.passed_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def has_failures(self) -> bool:
        """Check if any assertions failed."""
        return any(not assertion.passed for assertion in self.assertions)
    
    @property
    def error_count(self) -> int:
        """Number of error-level failures."""
        return sum(
            1 for assertion in self.assertions 
            if not assertion.passed and assertion.level == AssertionLevel.ERROR
        )
    
    @property
    def warning_count(self) -> int:
        """Number of warning-level failures."""
        return sum(
            1 for assertion in self.assertions
            if not assertion.passed and assertion.level == AssertionLevel.WARNING
        )


@dataclass
class TableAssertionResult:
    """Result of assertions on a database table."""
    table_name: str
    database_name: str
    schema_name: Optional[str] = None
    assertions: List[AssertionResult] = field(default_factory=list)
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time_ms: float = 0.0
    
    @property
    def passed_count(self) -> int:
        """Number of assertions that passed."""
        return sum(1 for assertion in self.assertions if assertion.passed)
    
    @property
    def failed_count(self) -> int:
        """Number of assertions that failed."""
        return sum(1 for assertion in self.assertions if not assertion.passed)
    
    @property
    def total_count(self) -> int:
        """Total number of assertions."""
        return len(self.assertions)
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.passed_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def has_failures(self) -> bool:
        """Check if any assertions failed."""
        return any(not assertion.passed for assertion in self.assertions)


@dataclass
class Assertion:
    """Definition of an assertion to be executed."""
    name: str
    assertion_type: AssertionType = AssertionType.CUSTOM
    description: str = ""
    level: AssertionLevel = AssertionLevel.ERROR
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_function: Optional[Callable[[Any], bool]] = None
    expected_message: str = ""
    failure_message: str = ""
    
    def __post_init__(self):
        """Set default messages if not provided."""
        if not self.expected_message:
            self.expected_message = f"Assertion '{self.name}' should pass"
        if not self.failure_message:
            self.failure_message = f"Assertion '{self.name}' failed"


@dataclass
class EqualsAssertion(Assertion):
    """Assertion that a value equals expected value."""
    expected_value: Any = None
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.EQUALS
        self.parameters = {"expected_value": self.expected_value}


@dataclass
class ContainsAssertion(Assertion):
    """Assertion that a value contains expected substring/element."""
    expected_value: Any = None
    case_sensitive: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.CONTAINS
        self.parameters = {
            "expected_value": self.expected_value,
            "case_sensitive": self.case_sensitive
        }


@dataclass
class RangeAssertion(Assertion):
    """Assertion that a value is within a range."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    inclusive: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.IN_RANGE
        self.parameters = {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive
        }


@dataclass
class RowCountAssertion(Assertion):
    """Assertion about the number of rows in a result set."""
    expected_count: Optional[int] = None
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.ROW_COUNT
        self.parameters = {
            "expected_count": self.expected_count,
            "min_count": self.min_count,
            "max_count": self.max_count
        }


@dataclass
class ColumnCountAssertion(Assertion):
    """Assertion about the number of columns in a result set."""
    expected_count: Optional[int] = None
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.COLUMN_COUNT
        self.parameters = {
            "expected_count": self.expected_count,
            "min_count": self.min_count,
            "max_count": self.max_count
        }


@dataclass
class SchemaAssertion(Assertion):
    """Assertion about the schema/structure of a result set."""
    expected_columns: List[str] = field(default_factory=list)
    expected_types: Dict[str, str] = field(default_factory=dict)
    strict_order: bool = False
    allow_extra_columns: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.assertion_type = AssertionType.SCHEMA_MATCHES
        self.parameters = {
            "expected_columns": self.expected_columns,
            "expected_types": self.expected_types,
            "strict_order": self.strict_order,
            "allow_extra_columns": self.allow_extra_columns
        }


@dataclass
class AssertionSet:
    """Collection of assertions to be executed together."""
    name: str
    description: str
    assertions: List[Assertion] = field(default_factory=list)
    
    def add_assertion(self, assertion: Assertion) -> None:
        """Add an assertion to the set."""
        self.assertions.append(assertion)
    
    def remove_assertion(self, assertion_name: str) -> None:
        """Remove an assertion by name."""
        self.assertions = [a for a in self.assertions if a.name != assertion_name]
    
    def get_assertion(self, assertion_name: str) -> Optional[Assertion]:
        """Get an assertion by name."""
        for assertion in self.assertions:
            if assertion.name == assertion_name:
                return assertion
        return None


# Pre-built common assertions
NOT_EMPTY_ASSERTION = Assertion(
    name="not_empty",
    assertion_type=AssertionType.IS_NOT_EMPTY,
    description="Result set should not be empty",
    failure_message="Result set is empty"
)

HAS_ROWS_ASSERTION = RowCountAssertion(
    name="has_rows",
    min_count=1,
    description="Result set should have at least one row",
    failure_message="Result set has no rows"
)

NO_NULLS_ASSERTION = Assertion(
    name="no_nulls",
    assertion_type=AssertionType.IS_NOT_NULL,
    description="Values should not be null",
    failure_message="Found null values"
)

IS_UNIQUE_ASSERTION = Assertion(
    name="is_unique",
    assertion_type=AssertionType.IS_UNIQUE,
    description="Values should be unique",
    failure_message="Found duplicate values"
)
