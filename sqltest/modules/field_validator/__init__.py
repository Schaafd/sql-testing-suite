"""Field validation module for SQLTest Pro."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, DatabaseError
from .models import (
    ValidationRule,
    ValidationRuleType,
    ValidationLevel,
    ValidationResult,
    FieldValidationResult,
    TableValidationResult,
    ValidationRuleSet,
    RegexValidationRule,
    RangeValidationRule,
    LengthValidationRule,
    NullValidationRule,
    EnumValidationRule,
    # Pre-built rules
    EMAIL_REGEX_RULE,
    PHONE_REGEX_RULE,
    SSN_REGEX_RULE,
    ZIP_CODE_RULE,
    NOT_NULL_RULE,
    POSITIVE_NUMBER_RULE
)
from .validator import FieldValidator, validate_field, validate_table
from .config import (
    ValidationConfigLoader,
    load_validation_config,
    save_sample_config,
    create_sample_config
)


class TableFieldValidator:
    """High-level interface for field validation integrated with database operations."""
    
    def __init__(self, connection_manager: ConnectionManager, strict_mode: bool = False):
        """Initialize the table field validator.
        
        Args:
            connection_manager: Database connection manager
            strict_mode: If True, stop on first error. If False, collect all errors.
        """
        self.connection_manager = connection_manager
        self.validator = FieldValidator(strict_mode=strict_mode)
        self.rule_sets = {}
    
    def load_validation_rules(self, config_path: str) -> None:
        """Load validation rules from a YAML configuration file.
        
        Args:
            config_path: Path to the validation configuration file
        """
        self.rule_sets = load_validation_config(config_path)
    
    def add_rule_set(self, rule_set: ValidationRuleSet) -> None:
        """Add a validation rule set."""
        self.rule_sets[rule_set.name] = rule_set
    
    def validate_table_data(
        self,
        table_name: str,
        rule_set_name: str,
        database_name: Optional[str] = None,
        where_clause: Optional[str] = None,
        sample_rows: Optional[int] = None
    ) -> TableValidationResult:
        """Validate data in a database table using a rule set.
        
        Args:
            table_name: Name of the table to validate
            rule_set_name: Name of the rule set to apply
            database_name: Database name (uses default if not provided)
            where_clause: Optional WHERE clause to filter data
            sample_rows: Limit number of rows to validate
            
        Returns:
            TableValidationResult with detailed validation results
            
        Raises:
            ValidationError: If rule set not found or validation fails
            DatabaseError: If database operations fail
        """
        # Get rule set
        rule_set = self.rule_sets.get(rule_set_name)
        if not rule_set:
            raise ValidationError(f"Rule set '{rule_set_name}' not found")
        
        # Build query to fetch table data
        columns_to_validate = rule_set.apply_to_columns
        if not columns_to_validate:
            # If no specific columns are specified, apply to all columns that have rules
            columns_to_validate = list(set(
                col for rule in rule_set.rules
                for col in (getattr(rule, 'apply_to_columns', [table_name + '.*']) or [])
            ))
            if not columns_to_validate:
                # Default to all columns
                columns_to_validate = ['*']
        
        column_list = ", ".join(columns_to_validate) if columns_to_validate != ['*'] else "*"
        query = f"SELECT {column_list} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if sample_rows:
            # Add database-specific LIMIT clause
            adapter = self.connection_manager.get_adapter(database_name)
            if adapter.get_driver_name() == "sqlite":
                query += f" LIMIT {sample_rows}"
            elif adapter.get_driver_name() in ["psycopg2", "pymysql"]:
                query += f" LIMIT {sample_rows}"
            else:
                query += f" LIMIT {sample_rows}"
        
        # Execute query and get data
        try:
            result = self.connection_manager.execute_query(query, db_name=database_name)
            
            if result.is_empty:
                raise ValidationError(f"No data found in table '{table_name}'")
                
            # Prepare column rules mapping
            column_rules = {}
            
            if columns_to_validate == ['*']:
                # Apply all rules to all columns
                for column in result.data.columns:
                    column_rules[column] = rule_set.rules.copy()
            else:
                # Apply rules to specific columns
                for column in columns_to_validate:
                    if column in result.data.columns:
                        column_rules[column] = rule_set.rules.copy()
            
            # Validate the data
            table_result = self.validator.validate_dataframe(
                df=result.data,
                column_rules=column_rules,
                table_name=table_name
            )
            
            # Set database name
            table_result.database_name = database_name or self.connection_manager.config.default_database
            
            return table_result
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to validate table '{table_name}': {e}") from e
    
    def validate_query_results(
        self,
        query: str,
        rule_set_name: str,
        database_name: Optional[str] = None
    ) -> TableValidationResult:
        """Validate the results of a SQL query using a rule set.
        
        Args:
            query: SQL query to execute and validate
            rule_set_name: Name of the rule set to apply
            database_name: Database name (uses default if not provided)
            
        Returns:
            TableValidationResult with validation results
            
        Raises:
            ValidationError: If rule set not found or validation fails
            DatabaseError: If query execution fails
        """
        # Get rule set
        rule_set = self.rule_sets.get(rule_set_name)
        if not rule_set:
            raise ValidationError(f"Rule set '{rule_set_name}' not found")
        
        # Execute query
        try:
            result = self.connection_manager.execute_query(query, db_name=database_name)
            
            if result.is_empty:
                raise ValidationError("Query returned no results to validate")
            
            # Apply rules to all columns in the result
            column_rules = {}
            for column in result.data.columns:
                column_rules[column] = rule_set.rules.copy()
            
            # Validate the data
            table_result = self.validator.validate_dataframe(
                df=result.data,
                column_rules=column_rules,
                table_name="query_result"
            )
            
            # Set database name
            table_result.database_name = database_name or self.connection_manager.config.default_database
            
            return table_result
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to validate query results: {e}") from e
    
    def validate_column_data(
        self,
        table_name: str,
        column_name: str,
        rules: List[ValidationRule],
        database_name: Optional[str] = None,
        where_clause: Optional[str] = None,
        sample_rows: Optional[int] = None
    ) -> FieldValidationResult:
        """Validate a specific column with custom rules.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to validate
            rules: List of validation rules to apply
            database_name: Database name (uses default if not provided)
            where_clause: Optional WHERE clause to filter data
            sample_rows: Limit number of rows to validate
            
        Returns:
            FieldValidationResult with validation results
        """
        # Build query to fetch column data
        query = f"SELECT {column_name} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if sample_rows:
            adapter = self.connection_manager.get_adapter(database_name)
            if adapter.get_driver_name() == "sqlite":
                query += f" LIMIT {sample_rows}"
            elif adapter.get_driver_name() in ["psycopg2", "pymysql"]:
                query += f" LIMIT {sample_rows}"
            else:
                query += f" LIMIT {sample_rows}"
        
        # Execute query and validate
        try:
            result = self.connection_manager.execute_query(query, db_name=database_name)
            
            if result.is_empty or column_name not in result.data.columns:
                raise ValidationError(f"Column '{column_name}' not found in table '{table_name}'")
            
            # Validate the column
            field_result = self.validator.validate_column(
                data=result.data[column_name],
                rules=rules,
                column_name=column_name,
                table_name=table_name
            )
            
            return field_result
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to validate column '{column_name}': {e}") from e
    
    def get_validation_summary(
        self,
        table_results: List[TableValidationResult]
    ) -> Dict[str, Any]:
        """Get a summary of validation results across multiple tables.
        
        Args:
            table_results: List of table validation results
            
        Returns:
            Dictionary with validation summary statistics
        """
        total_tables = len(table_results)
        total_rules = sum(result.total_rules for result in table_results)
        total_passed = sum(result.passed_rules for result in table_results)
        total_failed = sum(result.failed_rules for result in table_results)
        total_warnings = sum(result.warnings for result in table_results)
        
        tables_with_errors = sum(1 for result in table_results if result.has_errors)
        tables_with_warnings = sum(1 for result in table_results if result.has_warnings)
        
        return {
            'total_tables': total_tables,
            'total_rules': total_rules,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_warnings': total_warnings,
            'overall_success_rate': (total_passed / total_rules * 100) if total_rules > 0 else 0.0,
            'tables_with_errors': tables_with_errors,
            'tables_with_warnings': tables_with_warnings,
            'tables_clean': total_tables - tables_with_errors - tables_with_warnings
        }


# Convenience functions
def validate_table_with_connection(
    connection_manager: ConnectionManager,
    table_name: str,
    column_rules: Dict[str, List[ValidationRule]],
    database_name: Optional[str] = None,
    strict_mode: bool = False
) -> TableValidationResult:
    """Convenience function to validate a table with database connection."""
    validator = TableFieldValidator(connection_manager, strict_mode)
    
    # Create a temporary rule set
    rule_set = ValidationRuleSet(
        name="temp_rules",
        description="Temporary rule set for validation"
    )
    
    # Add all rules from column_rules
    for column, rules in column_rules.items():
        for rule in rules:
            rule_set.add_rule(rule)
    
    validator.add_rule_set(rule_set)
    
    # Build query and validate
    columns = list(column_rules.keys())
    column_list = ", ".join(columns)
    query = f"SELECT {column_list} FROM {table_name}"
    
    result = connection_manager.execute_query(query, database_name=database_name)
    
    if result.is_empty:
        raise ValidationError(f"No data found in table '{table_name}'")
    
    table_result = validate_table(
        df=result.data,
        column_rules=column_rules,
        table_name=table_name,
        strict_mode=strict_mode
    )
    
    table_result.database_name = database_name or connection_manager.config.default_database
    return table_result


# Export main classes and functions
__all__ = [
    # Core classes
    'TableFieldValidator',
    'FieldValidator',
    
    # Models
    'ValidationRule',
    'ValidationRuleType', 
    'ValidationLevel',
    'ValidationResult',
    'FieldValidationResult',
    'TableValidationResult',
    'ValidationRuleSet',
    'RegexValidationRule',
    'RangeValidationRule',
    'LengthValidationRule',
    'NullValidationRule',
    'EnumValidationRule',
    
    # Pre-built rules
    'EMAIL_REGEX_RULE',
    'PHONE_REGEX_RULE',
    'SSN_REGEX_RULE',
    'ZIP_CODE_RULE',
    'NOT_NULL_RULE',
    'POSITIVE_NUMBER_RULE',
    
    # Configuration
    'ValidationConfigLoader',
    'load_validation_config',
    'save_sample_config',
    'create_sample_config',
    
    # Convenience functions
    'validate_field',
    'validate_table',
    'validate_table_with_connection'
]
