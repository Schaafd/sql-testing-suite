"""Field validation engine for SQLTest Pro."""

import re
import pandas as pd
from typing import Any, List, Optional, Set
from datetime import datetime

from ...exceptions import ValidationError
from .models import (
    ValidationRule,
    ValidationRuleType,
    ValidationResult,
    FieldValidationResult,
    TableValidationResult,
    ValidationLevel,
    RegexValidationRule,
    RangeValidationRule,
    LengthValidationRule,
    NullValidationRule,
    EnumValidationRule
)


class FieldValidator:
    """Core field validation engine."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize the validator.
        
        Args:
            strict_mode: If True, stop on first error. If False, collect all errors.
        """
        self.strict_mode = strict_mode
        self.validators = {
            ValidationRuleType.REGEX: self._validate_regex,
            ValidationRuleType.RANGE: self._validate_range,
            ValidationRuleType.LENGTH: self._validate_length,
            ValidationRuleType.NULL_CHECK: self._validate_null,
            ValidationRuleType.ENUM: self._validate_enum,
            ValidationRuleType.CUSTOM: self._validate_custom,
            ValidationRuleType.UNIQUE: self._validate_unique
        }
    
    def validate_column(
        self, 
        data: pd.Series, 
        rules: List[ValidationRule],
        column_name: str,
        table_name: str = ""
    ) -> FieldValidationResult:
        """Validate a single column against a set of rules.
        
        Args:
            data: Pandas Series containing the column data
            rules: List of validation rules to apply
            column_name: Name of the column being validated
            table_name: Name of the table (for reporting)
            
        Returns:
            FieldValidationResult with detailed results
            
        Raises:
            ValidationError: If validation fails in strict mode
        """
        results = []
        passed_count = 0
        failed_count = 0
        warning_count = 0
        
        for rule in rules:
            try:
                rule_results = self._execute_rule(data, rule, column_name)
                results.extend(rule_results)
                
                # Count results by type
                for result in rule_results:
                    if result.passed:
                        passed_count += 1
                    else:
                        if result.level == ValidationLevel.ERROR:
                            failed_count += 1
                            if self.strict_mode:
                                raise ValidationError(
                                    f"Validation failed for column '{column_name}': {result.message}"
                                )
                        elif result.level == ValidationLevel.WARNING:
                            warning_count += 1
                            
            except Exception as e:
                # Handle validator errors
                error_result = ValidationResult(
                    rule_name=rule.name,
                    column_name=column_name,
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Validation error: {str(e)}"
                )
                results.append(error_result)
                failed_count += 1
                
                if self.strict_mode:
                    raise ValidationError(
                        f"Validation error for column '{column_name}': {str(e)}"
                    )
        
        return FieldValidationResult(
            column_name=column_name,
            table_name=table_name,
            total_rows=len(data),
            validation_results=results,
            passed_rules=passed_count,
            failed_rules=failed_count,
            warnings=warning_count
        )
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame,
        column_rules: dict[str, List[ValidationRule]],
        table_name: str = ""
    ) -> TableValidationResult:
        """Validate multiple columns in a DataFrame.
        
        Args:
            df: DataFrame to validate
            column_rules: Dictionary mapping column names to validation rules
            table_name: Name of the table being validated
            
        Returns:
            TableValidationResult with results for all columns
        """
        field_results = []
        
        for column_name, rules in column_rules.items():
            if column_name not in df.columns:
                # Create error result for missing column
                error_result = ValidationResult(
                    rule_name="column_exists",
                    column_name=column_name,
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Column '{column_name}' not found in data"
                )
                
                field_result = FieldValidationResult(
                    column_name=column_name,
                    table_name=table_name,
                    total_rows=0,
                    validation_results=[error_result],
                    failed_rules=1
                )
                field_results.append(field_result)
                continue
            
            # Validate the column
            field_result = self.validate_column(
                data=df[column_name],
                rules=rules,
                column_name=column_name,
                table_name=table_name
            )
            field_results.append(field_result)
        
        return TableValidationResult(
            table_name=table_name,
            database_name="",  # Will be set by caller
            field_results=field_results
        )
    
    def _execute_rule(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Execute a single validation rule on column data.
        
        Args:
            data: Column data to validate
            rule: Validation rule to execute
            column_name: Name of the column
            
        Returns:
            List of ValidationResult objects
        """
        validator_func = self.validators.get(rule.rule_type)
        if not validator_func:
            raise ValidationError(f"Unknown validation rule type: {rule.rule_type}")
        
        return validator_func(data, rule, column_name)
    
    def _validate_regex(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate data against regex pattern."""
        results = []
        pattern = rule.parameters.get("pattern", "")
        flags = rule.parameters.get("flags", 0)
        
        if not pattern:
            raise ValidationError(f"Regex rule '{rule.name}' missing pattern parameter")
        
        compiled_pattern = re.compile(pattern, flags)
        
        for idx, value in data.items():
            # Skip null values unless rule specifically checks for them
            if pd.isna(value):
                continue
                
            value_str = str(value)
            matches = bool(compiled_pattern.match(value_str))
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=matches,
                level=rule.level,
                message=rule.error_message if not matches else f"Pattern match successful",
                value=value,
                row_number=idx + 1  # 1-based row numbering
            )
            results.append(result)
        
        return results
    
    def _validate_range(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate numeric data against range constraints."""
        results = []
        min_val = rule.parameters.get("min_value")
        max_val = rule.parameters.get("max_value")
        inclusive = rule.parameters.get("inclusive", True)
        
        for idx, value in data.items():
            # Skip null values
            if pd.isna(value):
                continue
            
            # Try to convert to numeric
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                result = ValidationResult(
                    rule_name=rule.name,
                    column_name=column_name,
                    passed=False,
                    level=rule.level,
                    message=f"Value '{value}' is not numeric",
                    value=value,
                    row_number=idx + 1
                )
                results.append(result)
                continue
            
            # Check range constraints
            valid = True
            if min_val is not None:
                if inclusive:
                    valid = valid and numeric_value >= min_val
                else:
                    valid = valid and numeric_value > min_val
            
            if max_val is not None:
                if inclusive:
                    valid = valid and numeric_value <= max_val
                else:
                    valid = valid and numeric_value < max_val
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=valid,
                level=rule.level,
                message=rule.error_message if not valid else "Range validation passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results
    
    def _validate_length(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate string length constraints."""
        results = []
        min_len = rule.parameters.get("min_length")
        max_len = rule.parameters.get("max_length") 
        exact_len = rule.parameters.get("exact_length")
        
        for idx, value in data.items():
            # Skip null values
            if pd.isna(value):
                continue
                
            value_str = str(value)
            length = len(value_str)
            valid = True
            
            if exact_len is not None:
                valid = length == exact_len
            else:
                if min_len is not None:
                    valid = valid and length >= min_len
                if max_len is not None:
                    valid = valid and length <= max_len
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=valid,
                level=rule.level,
                message=rule.error_message if not valid else "Length validation passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results
    
    def _validate_null(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate null/not null constraints."""
        results = []
        allow_null = rule.parameters.get("allow_null", False)
        
        for idx, value in data.items():
            is_null = pd.isna(value)
            valid = allow_null or not is_null
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=valid,
                level=rule.level,
                message=rule.error_message if not valid else "Null check passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results
    
    def _validate_enum(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate enumeration constraints."""
        results = []
        allowed_values = set(rule.parameters.get("allowed_values", []))
        case_sensitive = rule.parameters.get("case_sensitive", True)
        
        if not case_sensitive:
            allowed_values = {str(v).lower() for v in allowed_values}
        
        for idx, value in data.items():
            # Skip null values
            if pd.isna(value):
                continue
            
            check_value = str(value)
            if not case_sensitive:
                check_value = check_value.lower()
            
            valid = check_value in allowed_values
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=valid,
                level=rule.level,
                message=rule.error_message if not valid else "Enum validation passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results
    
    def _validate_custom(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate using custom function."""
        results = []
        custom_func = rule.custom_function
        
        if not custom_func or not callable(custom_func):
            raise ValidationError(f"Custom rule '{rule.name}' missing or invalid custom_function")
        
        for idx, value in data.items():
            try:
                valid = custom_func(value)
            except Exception as e:
                valid = False
                error_msg = f"Custom validation error: {str(e)}"
            else:
                error_msg = rule.error_message
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=valid,
                level=rule.level,
                message=error_msg if not valid else "Custom validation passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results
    
    def _validate_unique(
        self, 
        data: pd.Series, 
        rule: ValidationRule, 
        column_name: str
    ) -> List[ValidationResult]:
        """Validate uniqueness constraint."""
        # Find duplicates
        duplicate_mask = data.duplicated(keep=False)
        duplicate_values = set(data[duplicate_mask].dropna())
        
        results = []
        for idx, value in data.items():
            # Skip null values for uniqueness check
            if pd.isna(value):
                continue
                
            is_duplicate = value in duplicate_values
            
            result = ValidationResult(
                rule_name=rule.name,
                column_name=column_name,
                passed=not is_duplicate,
                level=rule.level,
                message=rule.error_message if is_duplicate else "Uniqueness validation passed",
                value=value,
                row_number=idx + 1
            )
            results.append(result)
        
        return results


def validate_field(
    data: pd.Series,
    rules: List[ValidationRule], 
    column_name: str,
    table_name: str = "",
    strict_mode: bool = True
) -> FieldValidationResult:
    """Convenience function to validate a field."""
    validator = FieldValidator(strict_mode=strict_mode)
    return validator.validate_column(data, rules, column_name, table_name)


def validate_table(
    df: pd.DataFrame,
    column_rules: dict[str, List[ValidationRule]],
    table_name: str = "",
    strict_mode: bool = True
) -> TableValidationResult:
    """Convenience function to validate a table."""
    validator = FieldValidator(strict_mode=strict_mode)
    return validator.validate_dataframe(df, column_rules, table_name)
