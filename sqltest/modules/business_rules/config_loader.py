"""Configuration loader for business rules in SQLTest Pro."""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import logging
from pydantic import ValidationError as PydanticValidationError

from ...exceptions import ConfigurationError, ValidationError
from .models import (
    BusinessRulePydantic,
    RuleSetPydantic,
    RuleSet,
    BusinessRule,
    RuleType,
    RuleSeverity,
    ValidationScope
)

logger = logging.getLogger(__name__)


class BusinessRuleConfigLoader:
    """Loads and validates business rule configurations from YAML files."""
    
    def __init__(self, interpolate_env: bool = True, validate_syntax: bool = True):
        """Initialize the configuration loader.
        
        Args:
            interpolate_env: Whether to interpolate environment variables
            validate_syntax: Whether to validate SQL syntax in rules
        """
        self.interpolate_env = interpolate_env
        self.validate_syntax = validate_syntax
        self._loaded_files: Set[str] = set()
        self._include_stack: List[str] = []
        
    def load_rule_set_from_file(self, file_path: Union[str, Path]) -> RuleSet:
        """Load a rule set from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            RuleSet loaded from the file
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
            ValidationError: If configuration is invalid
        """
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        if not file_path.is_file():
            raise ConfigurationError(f"Path is not a file: {file_path}")
        
        logger.info(f"Loading business rule configuration from: {file_path}")
        
        try:
            # Clear loaded files tracking for new file
            self._loaded_files.clear()
            self._include_stack.clear()
            
            # Load and parse YAML
            config_data = self._load_yaml_with_includes(file_path)
            
            # Interpolate environment variables
            if self.interpolate_env:
                config_data = self._interpolate_environment_variables(config_data)
            
            # Validate and convert to RuleSet
            rule_set_model = RuleSetPydantic(**config_data)
            rule_set = rule_set_model.to_rule_set()
            
            # Additional validation
            self._validate_rule_set(rule_set)
            
            logger.info(f"Successfully loaded rule set '{rule_set.name}' with {len(rule_set.rules)} rules")
            return rule_set
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error in {file_path}: {str(e)}") from e
        except PydanticValidationError as e:
            raise ValidationError(f"Configuration validation error in {file_path}: {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration from {file_path}: {str(e)}") from e
    
    def load_rule_sets_from_directory(
        self,
        directory_path: Union[str, Path],
        pattern: str = "*.yaml",
        recursive: bool = False
    ) -> List[RuleSet]:
        """Load multiple rule sets from a directory.
        
        Args:
            directory_path: Directory containing rule set files
            pattern: File name pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of RuleSet objects loaded from files
            
        Raises:
            ConfigurationError: If directory cannot be accessed
        """
        directory_path = Path(directory_path).resolve()
        
        if not directory_path.exists():
            raise ConfigurationError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ConfigurationError(f"Path is not a directory: {directory_path}")
        
        logger.info(f"Loading rule sets from directory: {directory_path}")
        
        # Find matching files
        if recursive:
            yaml_files = list(directory_path.rglob(pattern))
        else:
            yaml_files = list(directory_path.glob(pattern))
        
        if not yaml_files:
            logger.warning(f"No files matching pattern '{pattern}' found in {directory_path}")
            return []
        
        rule_sets = []
        errors = []
        
        for yaml_file in sorted(yaml_files):
            try:
                rule_set = self.load_rule_set_from_file(yaml_file)
                rule_sets.append(rule_set)
            except (ConfigurationError, ValidationError) as e:
                logger.error(f"Failed to load {yaml_file}: {str(e)}")
                errors.append(f"{yaml_file}: {str(e)}")
        
        if errors:
            raise ConfigurationError(f"Failed to load some rule sets:\n" + "\n".join(errors))
        
        logger.info(f"Successfully loaded {len(rule_sets)} rule sets from {directory_path}")
        return rule_sets
    
    def create_rule_set_template(self, output_path: Union[str, Path]) -> None:
        """Create a template rule set configuration file.
        
        Args:
            output_path: Path where template will be created
        """
        template = {
            "name": "example_rule_set",
            "description": "Example business rule set configuration",
            "enabled": True,
            "tags": ["data_quality", "validation"],
            "parallel_execution": False,
            "max_concurrent_rules": 5,
            "rules": [
                {
                    "name": "customers_email_not_null",
                    "description": "Validate that customer email addresses are not null",
                    "rule_type": "data_quality",
                    "severity": "error",
                    "scope": "column",
                    "sql_query": """
                        SELECT COUNT(*) as violation_count,
                               'customers' as table_name,
                               'email' as column_name,
                               'Email addresses cannot be null' as message
                        FROM customers 
                        WHERE email IS NULL
                        HAVING COUNT(*) > 0
                    """,
                    "enabled": True,
                    "timeout_seconds": 60.0
                },
                {
                    "name": "orders_total_positive",
                    "description": "Validate that order totals are positive",
                    "rule_type": "business_logic",
                    "severity": "error",
                    "scope": "column",
                    "sql_query": """
                        SELECT COUNT(*) as violation_count,
                               'orders' as table_name,
                               'total' as column_name,
                               'Order total must be positive' as message,
                               MIN(total) as min_total
                        FROM orders 
                        WHERE total <= 0
                        HAVING COUNT(*) > 0
                    """,
                    "parameters": {
                        "table_name": "orders",
                        "column_name": "total"
                    },
                    "enabled": True,
                    "tags": ["business_logic", "financial"]
                },
                {
                    "name": "referential_integrity_orders_customers",
                    "description": "Validate referential integrity between orders and customers",
                    "rule_type": "referential_integrity",
                    "severity": "critical",
                    "scope": "cross_database",
                    "sql_query": """
                        SELECT COUNT(*) as violation_count,
                               'orders' as table_name,
                               'customer_id' as column_name,
                               'Orphaned orders found' as message
                        FROM orders o
                        LEFT JOIN customers c ON o.customer_id = c.customer_id
                        WHERE o.customer_id IS NOT NULL
                          AND c.customer_id IS NULL
                        HAVING COUNT(*) > 0
                    """,
                    "dependencies": ["customers_email_not_null"],
                    "enabled": True,
                    "max_violation_count": 0
                }
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Created rule set template at: {output_path}")
    
    def _load_yaml_with_includes(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with support for includes."""
        file_path_str = str(file_path)
        
        # Check for circular includes
        if file_path_str in self._include_stack:
            raise ConfigurationError(f"Circular include detected: {' -> '.join(self._include_stack + [file_path_str])}")
        
        if file_path_str in self._loaded_files:
            logger.warning(f"File already loaded, skipping: {file_path}")
            return {}
        
        self._include_stack.append(file_path_str)
        self._loaded_files.add(file_path_str)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process includes before YAML parsing
            content = self._process_includes(content, file_path.parent)
            
            # Parse YAML
            data = yaml.safe_load(content) or {}
            
            return data
            
        finally:
            self._include_stack.pop()
    
    def _process_includes(self, content: str, base_dir: Path) -> str:
        """Process include directives in YAML content."""
        include_pattern = re.compile(r'^\s*!include\s+["\']?([^"\'\\n]+)["\']?\s*$', re.MULTILINE)
        
        def replace_include(match):
            include_path = match.group(1).strip()
            
            # Resolve relative paths
            if not os.path.isabs(include_path):
                include_path = base_dir / include_path
            else:
                include_path = Path(include_path)
            
            try:
                included_data = self._load_yaml_with_includes(include_path)
                # Convert to YAML string for inclusion
                return yaml.dump(included_data, default_flow_style=False, indent=2)
            except Exception as e:
                raise ConfigurationError(f"Failed to include file {include_path}: {str(e)}") from e
        
        return include_pattern.sub(replace_include, content)
    
    def _interpolate_environment_variables(self, data: Any) -> Any:
        """Recursively interpolate environment variables in configuration data."""
        if isinstance(data, str):
            return self._interpolate_string(data)
        elif isinstance(data, dict):
            return {key: self._interpolate_environment_variables(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_environment_variables(item) for item in data]
        else:
            return data
    
    def _interpolate_string(self, value: str) -> str:
        """Interpolate environment variables in a string value."""
        # Pattern for ${VAR_NAME} or ${VAR_NAME:default_value}
        env_pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
        
        def replace_env_var(match):
            var_name = match.group(1).strip()
            default_value = match.group(2) if match.group(2) is not None else ""
            
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default_value:
                return default_value
            else:
                logger.warning(f"Environment variable '{var_name}' not found and no default provided")
                return match.group(0)  # Return original if no value found
        
        return env_pattern.sub(replace_env_var, value)
    
    def _validate_rule_set(self, rule_set: RuleSet) -> None:
        """Perform additional validation on loaded rule set."""
        # Check for circular dependencies
        self._check_circular_dependencies(rule_set.rules)
        
        # Validate SQL queries if enabled
        if self.validate_syntax:
            self._validate_sql_queries(rule_set.rules)
        
        # Check rule name uniqueness (should be caught by Pydantic, but double-check)
        rule_names = [rule.name for rule in rule_set.rules]
        if len(rule_names) != len(set(rule_names)):
            duplicates = [name for name in rule_names if rule_names.count(name) > 1]
            raise ValidationError(f"Duplicate rule names found: {list(set(duplicates))}")
    
    def _check_circular_dependencies(self, rules: List[BusinessRule]) -> None:
        """Check for circular dependencies in rules."""
        rule_names = {rule.name for rule in rules}
        dependency_graph = {rule.name: set(rule.dependencies) & rule_names for rule in rules}
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for rule_name in dependency_graph:
            if rule_name not in visited:
                if has_cycle(rule_name):
                    raise ValidationError(f"Circular dependency detected in rules starting from: {rule_name}")
    
    def _validate_sql_queries(self, rules: List[BusinessRule]) -> None:
        """Perform basic SQL query validation."""
        for rule in rules:
            if rule.sql_query:
                self._validate_sql_query(rule.name, rule.sql_query)
    
    def _validate_sql_query(self, rule_name: str, sql_query: str) -> None:
        """Validate a single SQL query for basic syntax issues."""
        # Basic validation - check for required elements in validation queries
        query_lower = sql_query.lower().strip()
        
        # Should be a SELECT query
        if not query_lower.startswith('select'):
            logger.warning(f"Rule '{rule_name}' query should start with SELECT")
        
        # Should have violation_count column for proper result processing
        if 'violation_count' not in query_lower:
            logger.warning(f"Rule '{rule_name}' query should include 'violation_count' column for proper violation tracking")
        
        # Should use HAVING clause for filtering violations
        if 'having' not in query_lower:
            logger.warning(f"Rule '{rule_name}' query should use HAVING clause to filter out non-violations")
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter', 'truncate']
        for keyword in dangerous_keywords:
            if f' {keyword} ' in query_lower or query_lower.startswith(keyword):
                raise ValidationError(f"Rule '{rule_name}' contains dangerous SQL operation: {keyword.upper()}")


def create_sample_business_rules() -> RuleSet:
    """Create a sample rule set for demonstration purposes."""
    from .models import (
        create_not_null_rule,
        create_uniqueness_rule,
        create_referential_integrity_rule,
        create_range_rule,
        create_completeness_rule
    )
    
    rule_set = RuleSet(
        name="sample_data_quality_rules",
        description="Sample data quality rules for demonstration",
        tags={"sample", "data_quality"}
    )
    
    # Add various types of rules
    rule_set.add_rule(create_not_null_rule("customers", "email"))
    rule_set.add_rule(create_uniqueness_rule("customers", "customer_id"))
    rule_set.add_rule(create_referential_integrity_rule("orders", "customer_id", "customers", "customer_id"))
    rule_set.add_rule(create_range_rule("products", "price", 0.01, 10000.00))
    rule_set.add_rule(create_completeness_rule("orders", ["customer_id", "order_date", "total"]))
    
    return rule_set
