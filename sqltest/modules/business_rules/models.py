"""Business rule validator models for SQLTest Pro."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field, validator


class RuleType(str, Enum):
    """Types of business rules."""
    DATA_QUALITY = "data_quality"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_LOGIC = "business_logic"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    CUSTOM = "custom"


class RuleSeverity(str, Enum):
    """Severity levels for business rule violations."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleStatus(str, Enum):
    """Status of rule execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class ValidationScope(str, Enum):
    """Scope of validation for business rules."""
    TABLE = "table"
    DATABASE = "database"
    CROSS_DATABASE = "cross_database"
    QUERY = "query"
    COLUMN = "column"
    RECORD = "record"


@dataclass
class RuleViolation:
    """Details of a business rule violation."""
    rule_name: str
    violation_id: str
    severity: RuleSeverity
    message: str
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    row_identifier: Optional[str] = None
    violation_count: int = 1
    sample_values: List[Any] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RuleResult:
    """Result of business rule execution."""
    rule_name: str
    rule_type: RuleType
    status: RuleStatus
    severity: RuleSeverity
    scope: ValidationScope
    passed: bool
    message: str
    violations: List[RuleViolation] = field(default_factory=list)
    execution_time_ms: float = 0.0
    rows_evaluated: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return sum(v.violation_count for v in self.violations)
    
    @property
    def critical_violations(self) -> int:
        """Number of critical violations."""
        return sum(v.violation_count for v in self.violations if v.severity == RuleSeverity.CRITICAL)
    
    @property
    def error_violations(self) -> int:
        """Number of error violations."""
        return sum(v.violation_count for v in self.violations if v.severity == RuleSeverity.ERROR)
    
    @property
    def warning_violations(self) -> int:
        """Number of warning violations."""
        return sum(v.violation_count for v in self.violations if v.severity == RuleSeverity.WARNING)


@dataclass
class ValidationContext:
    """Context for business rule validation."""
    database_name: str
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessRule:
    """Definition of a business rule."""
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    scope: ValidationScope
    sql_query: Optional[str] = None
    custom_function: Optional[Callable[[pd.DataFrame, ValidationContext], RuleResult]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    expected_violation_count: Optional[int] = None
    max_violation_count: Optional[int] = None
    timeout_seconds: float = 300.0
    
    def __post_init__(self):
        """Validate rule definition."""
        if not self.sql_query and not self.custom_function:
            raise ValueError(f"Rule '{self.name}' must have either sql_query or custom_function")
        
        if self.sql_query and self.custom_function:
            raise ValueError(f"Rule '{self.name}' cannot have both sql_query and custom_function")


# Pydantic models for validation and configuration loading
class BusinessRulePydantic(BaseModel):
    """Pydantic model for business rule validation."""
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity = RuleSeverity.ERROR
    scope: ValidationScope
    sql_query: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    enabled: bool = True
    tags: Set[str] = Field(default_factory=set)
    expected_violation_count: Optional[int] = None
    max_violation_count: Optional[int] = None
    timeout_seconds: float = Field(default=300.0, ge=0)
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Rule name cannot be empty')
        return v.strip()
    
    @validator('sql_query')
    def validate_sql_query(cls, v, values):
        if not v and 'custom_function' not in values:
            raise ValueError('Rule must have either sql_query or custom_function')
        return v
    
    def to_business_rule(self) -> BusinessRule:
        """Convert to BusinessRule dataclass."""
        return BusinessRule(
            name=self.name,
            description=self.description,
            rule_type=self.rule_type,
            severity=self.severity,
            scope=self.scope,
            sql_query=self.sql_query,
            parameters=self.parameters,
            dependencies=self.dependencies,
            enabled=self.enabled,
            tags=self.tags,
            expected_violation_count=self.expected_violation_count,
            max_violation_count=self.max_violation_count,
            timeout_seconds=self.timeout_seconds
        )


@dataclass
class RuleSet:
    """Collection of business rules to be executed together."""
    name: str
    description: str
    rules: List[BusinessRule] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    parallel_execution: bool = False
    max_concurrent_rules: int = 5
    
    def add_rule(self, rule: BusinessRule) -> None:
        """Add a rule to the set."""
        # Check for duplicate names
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"Rule '{rule.name}' already exists in rule set '{self.name}'")
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        original_length = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_length
    
    def get_rule(self, rule_name: str) -> Optional[BusinessRule]:
        """Get a rule by name."""
        return next((r for r in self.rules if r.name == rule_name), None)
    
    def get_enabled_rules(self) -> List[BusinessRule]:
        """Get all enabled rules."""
        return [r for r in self.rules if r.enabled]
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[BusinessRule]:
        """Get rules by type."""
        return [r for r in self.rules if r.rule_type == rule_type]
    
    def get_rules_by_severity(self, severity: RuleSeverity) -> List[BusinessRule]:
        """Get rules by severity."""
        return [r for r in self.rules if r.severity == severity]
    
    def get_rules_by_tag(self, tag: str) -> List[BusinessRule]:
        """Get rules by tag."""
        return [r for r in self.rules if tag in r.tags]


class RuleSetPydantic(BaseModel):
    """Pydantic model for rule set validation."""
    name: str
    description: str
    rules: List[BusinessRulePydantic] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    enabled: bool = True
    tags: Set[str] = Field(default_factory=set)
    parallel_execution: bool = False
    max_concurrent_rules: int = Field(default=5, ge=1, le=50)
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Rule set name cannot be empty')
        return v.strip()
    
    @validator('rules')
    def validate_unique_rule_names(cls, v):
        names = [rule.name for rule in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f'Duplicate rule names found: {list(set(duplicates))}')
        return v
    
    def to_rule_set(self) -> RuleSet:
        """Convert to RuleSet dataclass."""
        rule_set = RuleSet(
            name=self.name,
            description=self.description,
            dependencies=self.dependencies,
            enabled=self.enabled,
            tags=self.tags,
            parallel_execution=self.parallel_execution,
            max_concurrent_rules=self.max_concurrent_rules
        )
        
        for rule_model in self.rules:
            rule_set.add_rule(rule_model.to_business_rule())
        
        return rule_set


@dataclass
class ValidationSummary:
    """Summary of business rule validation results."""
    validation_name: str
    rule_set_name: str
    validation_context: ValidationContext
    start_time: datetime
    end_time: datetime
    total_rules: int
    rules_executed: int
    rules_passed: int
    rules_failed: int
    rules_error: int
    rules_skipped: int
    total_violations: int
    critical_violations: int
    error_violations: int
    warning_violations: int
    info_violations: int
    results: List[RuleResult] = field(default_factory=list)
    
    @property
    def execution_time_ms(self) -> float:
        """Total execution time in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.rules_passed / self.rules_executed * 100) if self.rules_executed > 0 else 0.0
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical violations found."""
        return self.critical_violations > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if any error violations found."""
        return self.error_violations > 0
    
    def get_results_by_status(self, status: RuleStatus) -> List[RuleResult]:
        """Get results filtered by status."""
        return [r for r in self.results if r.status == status]
    
    def get_results_by_type(self, rule_type: RuleType) -> List[RuleResult]:
        """Get results filtered by rule type."""
        return [r for r in self.results if r.rule_type == rule_type]
    
    def get_results_by_severity(self, severity: RuleSeverity) -> List[RuleResult]:
        """Get results filtered by severity."""
        return [r for r in self.results if r.severity == severity]


# Pre-built common business rules
def create_not_null_rule(table_name: str, column_name: str, severity: RuleSeverity = RuleSeverity.ERROR) -> BusinessRule:
    """Create a not null validation rule."""
    return BusinessRule(
        name=f"{table_name}_{column_name}_not_null",
        description=f"Validate that {column_name} in {table_name} does not contain null values",
        rule_type=RuleType.DATA_QUALITY,
        severity=severity,
        scope=ValidationScope.COLUMN,
        sql_query=f"""
            SELECT COUNT(*) as violation_count,
                   '{table_name}' as table_name,
                   '{column_name}' as column_name,
                   'Null values found' as message
            FROM {table_name}
            WHERE {column_name} IS NULL
            HAVING COUNT(*) > 0
        """,
        parameters={
            "table_name": table_name,
            "column_name": column_name
        }
    )


def create_uniqueness_rule(table_name: str, column_name: str, severity: RuleSeverity = RuleSeverity.ERROR) -> BusinessRule:
    """Create a uniqueness validation rule."""
    return BusinessRule(
        name=f"{table_name}_{column_name}_unique",
        description=f"Validate that {column_name} in {table_name} contains only unique values",
        rule_type=RuleType.UNIQUENESS,
        severity=severity,
        scope=ValidationScope.COLUMN,
        sql_query=f"""
            SELECT COUNT(*) - COUNT(DISTINCT {column_name}) as violation_count,
                   '{table_name}' as table_name,
                   '{column_name}' as column_name,
                   'Duplicate values found' as message
            FROM {table_name}
            HAVING COUNT(*) - COUNT(DISTINCT {column_name}) > 0
        """,
        parameters={
            "table_name": table_name,
            "column_name": column_name
        }
    )


def create_referential_integrity_rule(
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
    severity: RuleSeverity = RuleSeverity.CRITICAL
) -> BusinessRule:
    """Create a referential integrity validation rule."""
    return BusinessRule(
        name=f"{child_table}_{child_column}_references_{parent_table}_{parent_column}",
        description=f"Validate referential integrity: {child_table}.{child_column} -> {parent_table}.{parent_column}",
        rule_type=RuleType.REFERENTIAL_INTEGRITY,
        severity=severity,
        scope=ValidationScope.CROSS_DATABASE,
        sql_query=f"""
            SELECT COUNT(*) as violation_count,
                   '{child_table}' as table_name,
                   '{child_column}' as column_name,
                   'Orphaned records found' as message
            FROM {child_table} c
            LEFT JOIN {parent_table} p ON c.{child_column} = p.{parent_column}
            WHERE c.{child_column} IS NOT NULL
              AND p.{parent_column} IS NULL
            HAVING COUNT(*) > 0
        """,
        parameters={
            "child_table": child_table,
            "child_column": child_column,
            "parent_table": parent_table,
            "parent_column": parent_column
        }
    )


def create_range_rule(
    table_name: str,
    column_name: str,
    min_value: Union[int, float, str],
    max_value: Union[int, float, str],
    severity: RuleSeverity = RuleSeverity.WARNING
) -> BusinessRule:
    """Create a range validation rule."""
    return BusinessRule(
        name=f"{table_name}_{column_name}_range",
        description=f"Validate that {column_name} in {table_name} is within range [{min_value}, {max_value}]",
        rule_type=RuleType.VALIDITY,
        severity=severity,
        scope=ValidationScope.COLUMN,
        sql_query=f"""
            SELECT COUNT(*) as violation_count,
                   '{table_name}' as table_name,
                   '{column_name}' as column_name,
                   'Values outside valid range' as message
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
              AND ({column_name} < {min_value} OR {column_name} > {max_value})
            HAVING COUNT(*) > 0
        """,
        parameters={
            "table_name": table_name,
            "column_name": column_name,
            "min_value": min_value,
            "max_value": max_value
        }
    )


def create_completeness_rule(table_name: str, required_columns: List[str], severity: RuleSeverity = RuleSeverity.ERROR) -> BusinessRule:
    """Create a completeness validation rule."""
    conditions = " OR ".join([f"{col} IS NULL" for col in required_columns])
    columns_str = ", ".join(required_columns)
    
    return BusinessRule(
        name=f"{table_name}_completeness",
        description=f"Validate completeness of required columns in {table_name}: {columns_str}",
        rule_type=RuleType.COMPLETENESS,
        severity=severity,
        scope=ValidationScope.TABLE,
        sql_query=f"""
            SELECT COUNT(*) as violation_count,
                   '{table_name}' as table_name,
                   'Incomplete records found' as message
            FROM {table_name}
            WHERE {conditions}
            HAVING COUNT(*) > 0
        """,
        parameters={
            "table_name": table_name,
            "required_columns": required_columns
        }
    )


# Export all model classes and utility functions
__all__ = [
    # Enums
    'RuleType',
    'RuleSeverity',
    'RuleStatus',
    'ValidationScope',
    
    # Data classes
    'RuleViolation',
    'RuleResult',
    'ValidationContext',
    'BusinessRule',
    'RuleSet',
    'ValidationSummary',
    
    # Pydantic models
    'BusinessRulePydantic',
    'RuleSetPydantic',
    
    # Utility functions
    'create_not_null_rule',
    'create_uniqueness_rule',
    'create_referential_integrity_rule',
    'create_range_rule',
    'create_completeness_rule'
]
