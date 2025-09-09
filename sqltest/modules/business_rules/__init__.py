"""Business rule validator for SQLTest Pro.

This module provides comprehensive business rule validation capabilities including:
- Data quality rules
- Referential integrity checks
- Business logic validation
- Custom rule support
- Parallel execution
- Comprehensive reporting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
import json

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, ConfigurationError
from .models import (
    BusinessRule,
    RuleSet,
    RuleResult,
    ValidationContext,
    ValidationSummary,
    RuleType,
    RuleSeverity,
    RuleStatus,
    ValidationScope,
    create_not_null_rule,
    create_uniqueness_rule,
    create_referential_integrity_rule,
    create_range_rule,
    create_completeness_rule
)
from .engine import BusinessRuleEngine
from .config_loader import BusinessRuleConfigLoader, create_sample_business_rules

logger = logging.getLogger(__name__)


class BusinessRuleValidator:
    """Main interface for business rule validation."""
    
    def __init__(self, connection_manager: ConnectionManager, max_workers: int = 5):
        """Initialize the business rule validator.
        
        Args:
            connection_manager: Database connection manager for executing queries
            max_workers: Maximum number of worker threads for parallel execution
        """
        self.connection_manager = connection_manager
        self.engine = BusinessRuleEngine(connection_manager, max_workers)
        self.config_loader = BusinessRuleConfigLoader()
        self._rule_sets: Dict[str, RuleSet] = {}
        
    def load_rule_set_from_file(self, file_path: Union[str, Path]) -> str:
        """Load a rule set from a YAML configuration file.
        
        Args:
            file_path: Path to the rule set configuration file
            
        Returns:
            Name of the loaded rule set
            
        Raises:
            ConfigurationError: If file cannot be loaded
            ValidationError: If configuration is invalid
        """
        rule_set = self.config_loader.load_rule_set_from_file(file_path)
        self._rule_sets[rule_set.name] = rule_set
        logger.info(f"Loaded rule set '{rule_set.name}' with {len(rule_set.rules)} rules")
        return rule_set.name
    
    def load_rule_sets_from_directory(
        self,
        directory_path: Union[str, Path],
        pattern: str = "*.yaml",
        recursive: bool = False
    ) -> List[str]:
        """Load multiple rule sets from a directory.
        
        Args:
            directory_path: Directory containing rule set files
            pattern: File name pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of loaded rule set names
        """
        rule_sets = self.config_loader.load_rule_sets_from_directory(
            directory_path, pattern, recursive
        )
        
        loaded_names = []
        for rule_set in rule_sets:
            self._rule_sets[rule_set.name] = rule_set
            loaded_names.append(rule_set.name)
        
        logger.info(f"Loaded {len(loaded_names)} rule sets from {directory_path}")
        return loaded_names
    
    def add_rule_set(self, rule_set: RuleSet) -> None:
        """Add a rule set programmatically.
        
        Args:
            rule_set: Rule set to add
        """
        self._rule_sets[rule_set.name] = rule_set
        logger.info(f"Added rule set '{rule_set.name}' with {len(rule_set.rules)} rules")
    
    def get_rule_set(self, name: str) -> Optional[RuleSet]:
        """Get a rule set by name.
        
        Args:
            name: Rule set name
            
        Returns:
            RuleSet if found, None otherwise
        """
        return self._rule_sets.get(name)
    
    def list_rule_sets(self) -> List[str]:
        """List all loaded rule set names.
        
        Returns:
            List of rule set names
        """
        return list(self._rule_sets.keys())
    
    def validate_with_rule_set(
        self,
        rule_set_name: str,
        database_name: str,
        schema_name: Optional[str] = None,
        parallel: Optional[bool] = None,
        fail_fast: bool = False,
        tags: Optional[Set[str]] = None
    ) -> ValidationSummary:
        """Validate using a named rule set.
        
        Args:
            rule_set_name: Name of the rule set to execute
            database_name: Database to validate
            schema_name: Optional schema name
            parallel: Override parallel execution setting
            fail_fast: Stop on first critical failure
            tags: Filter rules by tags (only rules with any of these tags will run)
            
        Returns:
            ValidationSummary with results
            
        Raises:
            ValidationError: If rule set not found or validation fails
        """
        rule_set = self._rule_sets.get(rule_set_name)
        if not rule_set:
            raise ValidationError(f"Rule set not found: {rule_set_name}")
        
        # Filter rules by tags if specified
        if tags:
            filtered_rules = [r for r in rule_set.rules if tags.intersection(r.tags)]
            if not filtered_rules:
                logger.warning(f"No rules match the specified tags: {tags}")
                filtered_rules = []
            
            # Create temporary rule set with filtered rules
            filtered_rule_set = RuleSet(
                name=f"{rule_set.name}_filtered",
                description=f"Filtered version of {rule_set.name}",
                rules=filtered_rules,
                parallel_execution=rule_set.parallel_execution,
                max_concurrent_rules=rule_set.max_concurrent_rules
            )
            rule_set = filtered_rule_set
        
        context = ValidationContext(
            database_name=database_name,
            schema_name=schema_name,
            metadata={"rule_set_name": rule_set_name}
        )
        
        return self.engine.execute_rule_set(rule_set, context, parallel, fail_fast)
    
    def validate_table(
        self,
        database_name: str,
        table_name: str,
        rules: List[BusinessRule],
        schema_name: Optional[str] = None,
        parallel: bool = False,
        fail_fast: bool = False
    ) -> ValidationSummary:
        """Validate a specific table with given rules.
        
        Args:
            database_name: Database name
            table_name: Table name
            rules: List of business rules to apply
            schema_name: Optional schema name
            parallel: Whether to execute rules in parallel
            fail_fast: Stop on first critical failure
            
        Returns:
            ValidationSummary with results
        """
        return self.engine.validate_table(
            database_name=database_name,
            table_name=table_name,
            rules=rules,
            schema_name=schema_name
        )
    
    def validate_query(
        self,
        database_name: str,
        query: str,
        rules: List[BusinessRule],
        query_name: str = "custom_query",
        parallel: bool = False,
        fail_fast: bool = False
    ) -> ValidationSummary:
        """Validate query results with given rules.
        
        Args:
            database_name: Database name
            query: SQL query to validate
            rules: List of business rules to apply
            query_name: Name for the query (for reporting)
            parallel: Whether to execute rules in parallel
            fail_fast: Stop on first critical failure
            
        Returns:
            ValidationSummary with results
        """
        return self.engine.validate_query(
            database_name=database_name,
            query=query,
            rules=rules,
            query_name=query_name
        )
    
    def batch_validate(
        self,
        validations: List[Dict[str, Any]],
        parallel_validations: bool = False,
        fail_fast: bool = False
    ) -> List[ValidationSummary]:
        """Run multiple validations in batch.
        
        Args:
            validations: List of validation configurations
                Each dict should contain:
                - rule_set_name: Name of rule set to use
                - database_name: Database to validate
                - schema_name: Optional schema name
                - tags: Optional set of tags to filter rules
            parallel_validations: Whether to run validations in parallel
            fail_fast: Stop batch on first critical failure
            
        Returns:
            List of ValidationSummary objects
        """
        results = []
        
        if parallel_validations:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=self.engine.max_workers) as executor:
                future_to_config = {
                    executor.submit(self._run_single_validation, config): config
                    for config in validations
                }
                
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if fail_fast and result.has_critical_issues:
                            logger.warning(f"Stopping batch validation due to critical issues in {config}")
                            # Cancel remaining futures
                            for f in future_to_config:
                                if f != future:
                                    f.cancel()
                            break
                            
                    except Exception as e:
                        logger.error(f"Validation failed for {config}: {str(e)}")
                        if fail_fast:
                            break
        else:
            for config in validations:
                try:
                    result = self._run_single_validation(config)
                    results.append(result)
                    
                    if fail_fast and result.has_critical_issues:
                        logger.warning(f"Stopping batch validation due to critical issues in {config}")
                        break
                        
                except Exception as e:
                    logger.error(f"Validation failed for {config}: {str(e)}")
                    if fail_fast:
                        break
        
        return results
    
    def _run_single_validation(self, config: Dict[str, Any]) -> ValidationSummary:
        """Run a single validation from batch configuration."""
        rule_set_name = config.get("rule_set_name")
        database_name = config.get("database_name")
        schema_name = config.get("schema_name")
        tags = config.get("tags")
        parallel = config.get("parallel")
        fail_fast = config.get("fail_fast", False)
        
        if not rule_set_name or not database_name:
            raise ValidationError("Both rule_set_name and database_name are required")
        
        return self.validate_with_rule_set(
            rule_set_name=rule_set_name,
            database_name=database_name,
            schema_name=schema_name,
            parallel=parallel,
            fail_fast=fail_fast,
            tags=set(tags) if tags else None
        )
    
    def generate_data_quality_rules(
        self,
        database_name: str,
        table_name: str,
        schema_name: Optional[str] = None,
        include_not_null: bool = True,
        include_uniqueness: bool = True,
        include_referential_integrity: bool = False,
        primary_key_columns: Optional[List[str]] = None
    ) -> RuleSet:
        """Generate common data quality rules for a table.
        
        Args:
            database_name: Database name
            table_name: Table name
            schema_name: Optional schema name
            include_not_null: Include not null checks for non-nullable columns
            include_uniqueness: Include uniqueness checks for primary keys
            include_referential_integrity: Include foreign key checks
            primary_key_columns: List of primary key column names
            
        Returns:
            Generated RuleSet
        """
        # For now, we'll create a simplified version without schema introspection
        # TODO: Add schema introspection support when needed
        full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
        columns = []  # Empty for now - would need schema introspection
        
        rule_set = RuleSet(
            name=f"{table_name}_auto_generated_rules",
            description=f"Auto-generated data quality rules for {full_table_name}",
            tags={"auto_generated", "data_quality"}
        )
        
        # Generate not null rules
        if include_not_null and columns:
            for column_info in columns:
                column_name = column_info.get("name")
                nullable = column_info.get("nullable", True)
                
                if column_name and not nullable:
                    rule = create_not_null_rule(table_name, column_name)
                    rule_set.add_rule(rule)
        
        # Generate uniqueness rules for primary keys
        if include_uniqueness and primary_key_columns:
            for pk_column in primary_key_columns:
                rule = create_uniqueness_rule(table_name, pk_column)
                rule_set.add_rule(rule)
        
        logger.info(f"Generated {len(rule_set.rules)} rules for table {full_table_name}")
        return rule_set
    
    def create_rule_set_template(self, output_path: Union[str, Path]) -> None:
        """Create a template rule set configuration file.
        
        Args:
            output_path: Path where template will be created
        """
        self.config_loader.create_rule_set_template(output_path)
    
    def export_results_to_json(self, summary: ValidationSummary, output_path: Union[str, Path]) -> None:
        """Export validation results to JSON file.
        
        Args:
            summary: Validation summary to export
            output_path: Output file path
        """
        # Convert summary to JSON-serializable format
        results_data = {
            "validation_name": summary.validation_name,
            "rule_set_name": summary.rule_set_name,
            "validation_context": {
                "database_name": summary.validation_context.database_name,
                "schema_name": summary.validation_context.schema_name,
                "table_name": summary.validation_context.table_name,
                "query": summary.validation_context.query,
                "parameters": summary.validation_context.parameters,
                "metadata": summary.validation_context.metadata,
                "validation_timestamp": summary.validation_context.validation_timestamp.isoformat()
            },
            "execution_summary": {
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat(),
                "execution_time_ms": summary.execution_time_ms,
                "total_rules": summary.total_rules,
                "rules_executed": summary.rules_executed,
                "rules_passed": summary.rules_passed,
                "rules_failed": summary.rules_failed,
                "rules_error": summary.rules_error,
                "rules_skipped": summary.rules_skipped,
                "success_rate": summary.success_rate,
                "has_critical_issues": summary.has_critical_issues,
                "has_errors": summary.has_errors
            },
            "violation_summary": {
                "total_violations": summary.total_violations,
                "critical_violations": summary.critical_violations,
                "error_violations": summary.error_violations,
                "warning_violations": summary.warning_violations,
                "info_violations": summary.info_violations
            },
            "rule_results": []
        }
        
        # Add individual rule results
        for result in summary.results:
            rule_data = {
                "rule_name": result.rule_name,
                "rule_type": result.rule_type.value,
                "status": result.status.value,
                "severity": result.severity.value,
                "scope": result.scope.value,
                "passed": result.passed,
                "message": result.message,
                "execution_time_ms": result.execution_time_ms,
                "rows_evaluated": result.rows_evaluated,
                "timestamp": result.timestamp.isoformat(),
                "context": result.context,
                "violations": []
            }
            
            # Add violation details
            for violation in result.violations:
                violation_data = {
                    "violation_id": violation.violation_id,
                    "severity": violation.severity.value,
                    "message": violation.message,
                    "table_name": violation.table_name,
                    "column_name": violation.column_name,
                    "row_identifier": violation.row_identifier,
                    "violation_count": violation.violation_count,
                    "sample_values": violation.sample_values,
                    "context": violation.context,
                    "timestamp": violation.timestamp.isoformat()
                }
                rule_data["violations"].append(violation_data)
            
            results_data["rule_results"].append(rule_data)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Exported validation results to: {output_path}")
    
    def get_validation_statistics(self, summaries: List[ValidationSummary]) -> Dict[str, Any]:
        """Get aggregate statistics from multiple validation summaries.
        
        Args:
            summaries: List of validation summaries
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not summaries:
            return {}
        
        total_rules = sum(s.total_rules for s in summaries)
        total_executed = sum(s.rules_executed for s in summaries)
        total_passed = sum(s.rules_passed for s in summaries)
        total_failed = sum(s.rules_failed for s in summaries)
        total_errors = sum(s.rules_error for s in summaries)
        total_skipped = sum(s.rules_skipped for s in summaries)
        
        total_violations = sum(s.total_violations for s in summaries)
        critical_violations = sum(s.critical_violations for s in summaries)
        error_violations = sum(s.error_violations for s in summaries)
        warning_violations = sum(s.warning_violations for s in summaries)
        info_violations = sum(s.info_violations for s in summaries)
        
        avg_execution_time = sum(s.execution_time_ms for s in summaries) / len(summaries)
        overall_success_rate = (total_passed / total_executed * 100) if total_executed > 0 else 0.0
        
        return {
            "validation_count": len(summaries),
            "rule_statistics": {
                "total_rules": total_rules,
                "rules_executed": total_executed,
                "rules_passed": total_passed,
                "rules_failed": total_failed,
                "rules_error": total_errors,
                "rules_skipped": total_skipped,
                "overall_success_rate": overall_success_rate
            },
            "violation_statistics": {
                "total_violations": total_violations,
                "critical_violations": critical_violations,
                "error_violations": error_violations,
                "warning_violations": warning_violations,
                "info_violations": info_violations
            },
            "execution_statistics": {
                "average_execution_time_ms": avg_execution_time,
                "total_execution_time_ms": sum(s.execution_time_ms for s in summaries),
                "fastest_validation_ms": min(s.execution_time_ms for s in summaries),
                "slowest_validation_ms": max(s.execution_time_ms for s in summaries)
            },
            "quality_indicators": {
                "has_any_critical_issues": any(s.has_critical_issues for s in summaries),
                "has_any_errors": any(s.has_errors for s in summaries),
                "validations_with_issues": sum(1 for s in summaries if s.has_errors or s.has_critical_issues),
                "clean_validations": sum(1 for s in summaries if not s.has_errors and not s.has_critical_issues)
            }
        }


# Convenience functions for common use cases
def quick_validate_table_data_quality(
    connection_manager: ConnectionManager,
    database_name: str,
    table_name: str,
    schema_name: Optional[str] = None,
    primary_key_columns: Optional[List[str]] = None
) -> ValidationSummary:
    """Quick validation of basic data quality for a table.
    
    Args:
        connection_manager: Database connection manager
        database_name: Database name
        table_name: Table name
        schema_name: Optional schema name
        primary_key_columns: List of primary key columns
        
    Returns:
        ValidationSummary with results
    """
    validator = BusinessRuleValidator(connection_manager)
    rule_set = validator.generate_data_quality_rules(
        database_name=database_name,
        table_name=table_name,
        schema_name=schema_name,
        primary_key_columns=primary_key_columns or []
    )
    
    context = ValidationContext(
        database_name=database_name,
        schema_name=schema_name,
        table_name=table_name
    )
    
    return validator.engine.execute_rule_set(rule_set, context)


def create_validator_with_sample_rules(connection_manager: ConnectionManager) -> BusinessRuleValidator:
    """Create a validator with sample rules loaded.
    
    Args:
        connection_manager: Database connection manager
        
    Returns:
        BusinessRuleValidator with sample rules
    """
    validator = BusinessRuleValidator(connection_manager)
    sample_rule_set = create_sample_business_rules()
    validator.add_rule_set(sample_rule_set)
    return validator


# Export key classes and functions
__all__ = [
    'BusinessRuleValidator',
    'BusinessRule',
    'RuleSet', 
    'RuleResult',
    'ValidationContext',
    'ValidationSummary',
    'RuleType',
    'RuleSeverity',
    'RuleStatus',
    'ValidationScope',
    'BusinessRuleEngine',
    'BusinessRuleConfigLoader',
    'create_not_null_rule',
    'create_uniqueness_rule',
    'create_referential_integrity_rule', 
    'create_range_rule',
    'create_completeness_rule',
    'create_sample_business_rules',
    'quick_validate_table_data_quality',
    'create_validator_with_sample_rules'
]
