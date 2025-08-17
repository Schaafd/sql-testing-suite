"""YAML configuration support for field validation rules."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import (
    ValidationRule,
    ValidationRuleType,
    ValidationLevel,
    ValidationRuleSet,
    RegexValidationRule,
    RangeValidationRule,
    LengthValidationRule,
    NullValidationRule,
    EnumValidationRule
)
from ...exceptions import ConfigurationError


class ValidationConfigLoader:
    """Load validation rules from YAML configuration."""
    
    def __init__(self):
        """Initialize the config loader."""
        self.rule_factories = {
            ValidationRuleType.REGEX: self._create_regex_rule,
            ValidationRuleType.RANGE: self._create_range_rule,
            ValidationRuleType.LENGTH: self._create_length_rule,
            ValidationRuleType.NULL_CHECK: self._create_null_rule,
            ValidationRuleType.ENUM: self._create_enum_rule,
            ValidationRuleType.CUSTOM: self._create_custom_rule,
            ValidationRuleType.UNIQUE: self._create_unique_rule
        }
    
    def load_from_file(self, file_path: str) -> Dict[str, ValidationRuleSet]:
        """Load validation rules from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Dictionary mapping rule set names to ValidationRuleSet objects
            
        Raises:
            ConfigurationError: If the file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        
        return self._parse_config(config_data)
    
    def load_from_dict(self, config_data: Dict[str, Any]) -> Dict[str, ValidationRuleSet]:
        """Load validation rules from a dictionary.
        
        Args:
            config_data: Dictionary containing validation configuration
            
        Returns:
            Dictionary mapping rule set names to ValidationRuleSet objects
        """
        return self._parse_config(config_data)
    
    def _parse_config(self, config_data: Dict[str, Any]) -> Dict[str, ValidationRuleSet]:
        """Parse configuration data into ValidationRuleSet objects."""
        rule_sets = {}
        
        # Handle different configuration formats
        if 'validation_rules' in config_data:
            # New format with explicit validation_rules section
            rules_config = config_data['validation_rules']
        elif 'rules' in config_data:
            # Alternative format
            rules_config = config_data['rules']
        else:
            # Assume the entire config is rules
            rules_config = config_data
        
        for rule_set_name, rule_set_config in rules_config.items():
            rule_set = self._parse_rule_set(rule_set_name, rule_set_config)
            rule_sets[rule_set_name] = rule_set
        
        return rule_sets
    
    def _parse_rule_set(self, name: str, config: Dict[str, Any]) -> ValidationRuleSet:
        """Parse a single rule set from configuration."""
        description = config.get('description', '')
        apply_to_columns = config.get('apply_to_columns')
        rules_config = config.get('rules', [])
        
        rule_set = ValidationRuleSet(
            name=name,
            description=description,
            apply_to_columns=apply_to_columns
        )
        
        for rule_config in rules_config:
            rule = self._parse_rule(rule_config)
            rule_set.add_rule(rule)
        
        return rule_set
    
    def _parse_rule(self, config: Dict[str, Any]) -> ValidationRule:
        """Parse a single validation rule from configuration."""
        rule_name = config.get('name', '')
        if not rule_name:
            raise ConfigurationError("Validation rule missing 'name' field")
        
        rule_type_str = config.get('type', '')
        if not rule_type_str:
            raise ConfigurationError(f"Rule '{rule_name}' missing 'type' field")
        
        try:
            rule_type = ValidationRuleType(rule_type_str)
        except ValueError:
            raise ConfigurationError(f"Invalid rule type '{rule_type_str}' for rule '{rule_name}'")
        
        # Parse common fields
        description = config.get('description', '')
        level_str = config.get('level', 'error')
        error_message = config.get('error_message', '')
        
        try:
            level = ValidationLevel(level_str)
        except ValueError:
            raise ConfigurationError(f"Invalid validation level '{level_str}' for rule '{rule_name}'")
        
        # Create rule using appropriate factory
        factory = self.rule_factories.get(rule_type)
        if not factory:
            raise ConfigurationError(f"Unsupported rule type '{rule_type}' for rule '{rule_name}'")
        
        return factory(rule_name, config, description, level, error_message)
    
    def _create_regex_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> RegexValidationRule:
        """Create a regex validation rule."""
        pattern = config.get('pattern', '')
        if not pattern:
            raise ConfigurationError(f"Regex rule '{name}' missing 'pattern' field")
        
        flags = config.get('flags', 0)
        
        return RegexValidationRule(
            name=name,
            rule_type=ValidationRuleType.REGEX,
            pattern=pattern,
            flags=flags,
            description=description,
            level=level,
            error_message=error_message or f"Value does not match pattern: {pattern}"
        )
    
    def _create_range_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> RangeValidationRule:
        """Create a range validation rule."""
        min_value = config.get('min_value')
        max_value = config.get('max_value')
        inclusive = config.get('inclusive', True)
        
        if min_value is None and max_value is None:
            raise ConfigurationError(f"Range rule '{name}' must specify min_value and/or max_value")
        
        return RangeValidationRule(
            name=name,
            rule_type=ValidationRuleType.RANGE,
            min_value=min_value,
            max_value=max_value,
            inclusive=inclusive,
            description=description,
            level=level,
            error_message=error_message or f"Value not in valid range"
        )
    
    def _create_length_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> LengthValidationRule:
        """Create a length validation rule."""
        min_length = config.get('min_length')
        max_length = config.get('max_length')
        exact_length = config.get('exact_length')
        
        if min_length is None and max_length is None and exact_length is None:
            raise ConfigurationError(
                f"Length rule '{name}' must specify min_length, max_length, and/or exact_length"
            )
        
        return LengthValidationRule(
            name=name,
            rule_type=ValidationRuleType.LENGTH,
            min_length=min_length,
            max_length=max_length,
            exact_length=exact_length,
            description=description,
            level=level,
            error_message=error_message or f"Invalid length"
        )
    
    def _create_null_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> NullValidationRule:
        """Create a null validation rule."""
        allow_null = config.get('allow_null', False)
        
        return NullValidationRule(
            name=name,
            rule_type=ValidationRuleType.NULL_CHECK,
            allow_null=allow_null,
            description=description,
            level=level,
            error_message=error_message or ("Field cannot be null" if not allow_null else "Field validation")
        )
    
    def _create_enum_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> EnumValidationRule:
        """Create an enum validation rule."""
        allowed_values = config.get('allowed_values', [])
        if not allowed_values:
            raise ConfigurationError(f"Enum rule '{name}' missing 'allowed_values' field")
        
        case_sensitive = config.get('case_sensitive', True)
        
        return EnumValidationRule(
            name=name,
            rule_type=ValidationRuleType.ENUM,
            allowed_values=set(allowed_values),
            case_sensitive=case_sensitive,
            description=description,
            level=level,
            error_message=error_message or f"Value not in allowed list: {allowed_values}"
        )
    
    def _create_custom_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> ValidationRule:
        """Create a custom validation rule."""
        # Custom rules need to be created programmatically
        # This method creates the base rule structure
        return ValidationRule(
            name=name,
            rule_type=ValidationRuleType.CUSTOM,
            description=description,
            level=level,
            parameters=config.get('parameters', {}),
            error_message=error_message or "Custom validation failed"
        )
    
    def _create_unique_rule(
        self, 
        name: str, 
        config: Dict[str, Any], 
        description: str, 
        level: ValidationLevel, 
        error_message: str
    ) -> ValidationRule:
        """Create a uniqueness validation rule."""
        return ValidationRule(
            name=name,
            rule_type=ValidationRuleType.UNIQUE,
            description=description,
            level=level,
            parameters={},
            error_message=error_message or "Value is not unique"
        )


def create_sample_config() -> Dict[str, Any]:
    """Create a sample validation configuration."""
    return {
        'validation_rules': {
            'user_validation': {
                'description': 'Validation rules for user data',
                'apply_to_columns': ['email', 'phone', 'age'],
                'rules': [
                    {
                        'name': 'email_format',
                        'type': 'regex',
                        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                        'description': 'Validate email format',
                        'level': 'error',
                        'error_message': 'Invalid email format'
                    },
                    {
                        'name': 'not_null',
                        'type': 'null_check',
                        'allow_null': False,
                        'description': 'Field cannot be null',
                        'level': 'error'
                    },
                    {
                        'name': 'age_range',
                        'type': 'range',
                        'min_value': 0,
                        'max_value': 150,
                        'inclusive': True,
                        'description': 'Age must be between 0 and 150',
                        'level': 'error'
                    }
                ]
            },
            'product_validation': {
                'description': 'Validation rules for product data',
                'rules': [
                    {
                        'name': 'product_name_length',
                        'type': 'length',
                        'min_length': 3,
                        'max_length': 100,
                        'description': 'Product name length validation',
                        'level': 'warning'
                    },
                    {
                        'name': 'status_enum',
                        'type': 'enum',
                        'allowed_values': ['active', 'inactive', 'pending'],
                        'case_sensitive': False,
                        'description': 'Product status validation',
                        'level': 'error'
                    }
                ]
            }
        }
    }


def save_sample_config(file_path: str) -> None:
    """Save a sample configuration to a file."""
    config = create_sample_config()
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_validation_config(file_path: str) -> Dict[str, ValidationRuleSet]:
    """Convenience function to load validation configuration."""
    loader = ValidationConfigLoader()
    return loader.load_from_file(file_path)
