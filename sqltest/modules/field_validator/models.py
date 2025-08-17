"""Field validation models for SQLTest Pro."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Callable
from enum import Enum


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning" 
    INFO = "info"


class ValidationRuleType(str, Enum):
    """Types of validation rules."""
    REGEX = "regex"
    RANGE = "range"
    NULL_CHECK = "null_check"
    LENGTH = "length"
    DATA_TYPE = "data_type"
    UNIQUE = "unique"
    CUSTOM = "custom"
    FORMAT = "format"
    ENUM = "enum"


@dataclass
class ValidationRule:
    """Definition of a field validation rule."""
    name: str
    rule_type: ValidationRuleType
    description: str = ""
    level: ValidationLevel = ValidationLevel.ERROR
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_function: Optional[Callable[[Any], bool]] = None
    error_message: str = ""
    
    def __post_init__(self):
        """Set default error message if not provided."""
        if not self.error_message:
            self.error_message = f"Field validation '{self.name}' failed"


@dataclass
class RegexValidationRule(ValidationRule):
    """Regex pattern validation rule."""
    pattern: str = ""
    flags: int = 0
    
    def __post_init__(self):
        if not hasattr(self, 'rule_type') or self.rule_type is None:
            self.rule_type = ValidationRuleType.REGEX
        super().__post_init__()
        self.parameters = {
            "pattern": self.pattern,
            "flags": self.flags
        }


@dataclass 
class RangeValidationRule(ValidationRule):
    """Range validation rule for numeric values."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    inclusive: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.rule_type = ValidationRuleType.RANGE
        self.parameters = {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive
        }


@dataclass
class LengthValidationRule(ValidationRule):
    """Length validation rule for strings."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    exact_length: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.rule_type = ValidationRuleType.LENGTH
        self.parameters = {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "exact_length": self.exact_length
        }


@dataclass
class NullValidationRule(ValidationRule):
    """Null/not null validation rule."""
    allow_null: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.rule_type = ValidationRuleType.NULL_CHECK
        self.parameters = {
            "allow_null": self.allow_null
        }


@dataclass
class EnumValidationRule(ValidationRule):
    """Enumeration validation rule."""
    allowed_values: Set[Any] = field(default_factory=set)
    case_sensitive: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.rule_type = ValidationRuleType.ENUM
        self.parameters = {
            "allowed_values": list(self.allowed_values),
            "case_sensitive": self.case_sensitive
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    column_name: str
    passed: bool
    level: ValidationLevel
    message: str
    value: Any = None
    row_number: Optional[int] = None
    validation_timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class FieldValidationResult:
    """Complete validation result for a field."""
    column_name: str
    table_name: str
    total_rows: int
    validation_results: List[ValidationResult] = field(default_factory=list)
    passed_rules: int = 0
    failed_rules: int = 0
    warnings: int = 0
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.passed_rules + self.failed_rules
        return (self.passed_rules / total * 100) if total > 0 else 0.0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level failures."""
        return any(
            not r.passed and r.level == ValidationLevel.ERROR 
            for r in self.validation_results
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(
            not r.passed and r.level == ValidationLevel.WARNING
            for r in self.validation_results
        )


@dataclass
class TableValidationResult:
    """Complete validation result for a table."""
    table_name: str
    database_name: str
    field_results: List[FieldValidationResult] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_rules(self) -> int:
        """Total number of validation rules executed."""
        return sum(
            len(field_result.validation_results)
            for field_result in self.field_results
        )
    
    @property
    def passed_rules(self) -> int:
        """Number of rules that passed."""
        return sum(
            field_result.passed_rules 
            for field_result in self.field_results
        )
    
    @property
    def failed_rules(self) -> int:
        """Number of rules that failed."""
        return sum(
            field_result.failed_rules
            for field_result in self.field_results
        )
    
    @property
    def warnings(self) -> int:
        """Number of warnings."""
        return sum(
            field_result.warnings
            for field_result in self.field_results
        )
    
    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all fields."""
        total = self.passed_rules + self.failed_rules
        return (self.passed_rules / total * 100) if total > 0 else 0.0
    
    @property
    def has_errors(self) -> bool:
        """Check if any field has errors."""
        return any(field_result.has_errors for field_result in self.field_results)
    
    @property 
    def has_warnings(self) -> bool:
        """Check if any field has warnings."""
        return any(field_result.has_warnings for field_result in self.field_results)


@dataclass
class ValidationRuleSet:
    """Collection of validation rules for a column or table."""
    name: str
    description: str
    rules: List[ValidationRule] = field(default_factory=list)
    apply_to_columns: Optional[List[str]] = None
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the set."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None


# Common pre-built validation rules
EMAIL_REGEX_RULE = RegexValidationRule(
    name="email_format",
    rule_type=ValidationRuleType.REGEX,
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    description="Validate email format",
    error_message="Invalid email format"
)

PHONE_REGEX_RULE = RegexValidationRule(
    name="phone_format",
    rule_type=ValidationRuleType.REGEX,
    pattern=r'^\+?1?[-.\s]?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$',
    description="Validate US phone number format",
    error_message="Invalid phone number format"
)

SSN_REGEX_RULE = RegexValidationRule(
    name="ssn_format",
    rule_type=ValidationRuleType.REGEX, 
    pattern=r'^\d{3}-\d{2}-\d{4}$',
    description="Validate SSN format (XXX-XX-XXXX)",
    error_message="Invalid SSN format"
)

ZIP_CODE_RULE = RegexValidationRule(
    name="zip_code_format",
    rule_type=ValidationRuleType.REGEX,
    pattern=r'^\d{5}(-\d{4})?$', 
    description="Validate US ZIP code format",
    error_message="Invalid ZIP code format"
)

NOT_NULL_RULE = NullValidationRule(
    name="not_null",
    rule_type=ValidationRuleType.NULL_CHECK,
    allow_null=False,
    description="Field cannot be null",
    error_message="Field cannot be null"
)

POSITIVE_NUMBER_RULE = RangeValidationRule(
    name="positive_number",
    rule_type=ValidationRuleType.RANGE,
    min_value=0,
    inclusive=False,
    description="Value must be positive",
    error_message="Value must be greater than 0"
)
