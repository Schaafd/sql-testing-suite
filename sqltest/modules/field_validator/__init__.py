"""Field validation module for SQLTest Pro."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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


VALIDATION_CHUNK_SIZE = 5000


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
        self._chunk_size = VALIDATION_CHUNK_SIZE
    
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
        
        columns_to_validate = list(dict.fromkeys(columns_to_validate))
        if any(col == '*' or col.endswith('.*') for col in columns_to_validate):
            columns_to_validate = ['*']

        column_list = "*" if columns_to_validate == ['*'] else ", ".join(columns_to_validate)
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
        
        try:
            def prepare_column_rules(available_columns: List[str]) -> Tuple[Dict[str, List[ValidationRule]], List[FieldValidationResult]]:
                column_rules: Dict[str, List[ValidationRule]] = {}
                missing_results: List[FieldValidationResult] = []

                if columns_to_validate == ['*']:
                    for column in available_columns:
                        column_rules[column] = rule_set.rules.copy()
                    return column_rules, missing_results

                available_set = set(available_columns)
                for column in columns_to_validate:
                    if column in available_set:
                        column_rules[column] = rule_set.rules.copy()
                    else:
                        missing_results.append(self._create_missing_column_result(column, table_name))
                return column_rules, missing_results

            table_result = self._stream_validate(
                query=query,
                database_name=database_name,
                table_name=table_name,
                prepare_column_rules=prepare_column_rules,
            )

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
            def prepare_column_rules(available_columns: List[str]) -> Tuple[Dict[str, List[ValidationRule]], List[FieldValidationResult]]:
                column_rules = {column: rule_set.rules.copy() for column in available_columns}
                return column_rules, []

            table_result = self._stream_validate(
                query=query,
                database_name=database_name,
                table_name="query_result",
                prepare_column_rules=prepare_column_rules,
            )

            table_result.database_name = database_name or self.connection_manager.config.default_database
            return table_result

        except DatabaseError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to validate query results: {e}") from e

    def _stream_validate(
        self,
        *,
        query: str,
        database_name: Optional[str],
        table_name: str,
        prepare_column_rules: Callable[[List[str]], Tuple[Dict[str, List[ValidationRule]], List[FieldValidationResult]]]
    ) -> TableValidationResult:
        """Stream query results and validate them chunk by chunk."""
        result = self.connection_manager.execute_query(
            query,
            db_name=database_name,
            stream_results=True,
            chunk_size=self._chunk_size,
        )

        iterator = result.iter_chunks()
        aggregate_results: Dict[str, FieldValidationResult] = {}
        rows_processed = 0
        column_rules: Optional[Dict[str, List[ValidationRule]]] = None

        try:
            for chunk in iterator:
                if chunk.empty:
                    continue

                if column_rules is None:
                    column_rules, missing_results = prepare_column_rules(chunk.columns.tolist())
                    for missing in missing_results:
                        aggregate_results[missing.column_name] = missing

                if not column_rules:
                    # Nothing to validate for this dataset
                    rows_processed += len(chunk)
                    continue

                chunk.index = range(rows_processed, rows_processed + len(chunk))
                rows_processed += len(chunk)

                chunk_result = self.validator.validate_dataframe(
                    df=chunk,
                    column_rules=column_rules,
                    table_name=table_name,
                )
                self._merge_field_results(aggregate_results, chunk_result.field_results)

        finally:
            close_fn = getattr(iterator, 'close', None)
            if callable(close_fn):
                close_fn()

        if rows_processed == 0:
            raise ValidationError(f"No data found for validation on '{table_name}'")

        if column_rules is None:
            column_rules, missing_results = prepare_column_rules([])
            for missing in missing_results:
                aggregate_results[missing.column_name] = missing

        field_results = list(aggregate_results.values())
        field_results.sort(key=lambda fr: fr.column_name)

        return TableValidationResult(
            table_name=table_name,
            database_name="",
            field_results=field_results,
        )

    def _merge_field_results(
        self,
        accumulator: Dict[str, FieldValidationResult],
        chunk_results: List[FieldValidationResult]
    ) -> None:
        for field_result in chunk_results:
            existing = accumulator.get(field_result.column_name)
            if existing is None:
                accumulator[field_result.column_name] = FieldValidationResult(
                    column_name=field_result.column_name,
                    table_name=field_result.table_name,
                    total_rows=field_result.total_rows,
                    validation_results=list(field_result.validation_results),
                    passed_rules=field_result.passed_rules,
                    failed_rules=field_result.failed_rules,
                    warnings=field_result.warnings,
                )
            else:
                existing.total_rows += field_result.total_rows
                existing.validation_results.extend(field_result.validation_results)
                existing.passed_rules += field_result.passed_rules
                existing.failed_rules += field_result.failed_rules
                existing.warnings += field_result.warnings

    @staticmethod
    def _create_missing_column_result(column_name: str, table_name: str) -> FieldValidationResult:
        error_result = ValidationResult(
            rule_name="column_exists",
            column_name=column_name,
            passed=False,
            level=ValidationLevel.ERROR,
            message=f"Column '{column_name}' not found in data",
        )

        return FieldValidationResult(
            column_name=column_name,
            table_name=table_name,
            total_rows=0,
            validation_results=[error_result],
            failed_rules=1,
        )

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
