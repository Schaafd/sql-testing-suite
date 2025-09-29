"""
Advanced configuration loading for SQL unit testing framework.

This module provides enterprise-grade configuration loading with features including
environment variable substitution, template inheritance, and validation.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pydantic import BaseModel, validator, Field
import logging

from ...config.models import DatabaseConfig
from .models import TestCase, TestSuite, TestFixture, AssertionType

logger = logging.getLogger(__name__)


@dataclass
class ConfigContext:
    """Context for configuration loading and template rendering."""
    environment: str
    base_path: Path
    variables: Dict[str, Any]
    include_stack: List[str]


class TestAssertionConfig(BaseModel):
    """Configuration model for test assertions."""
    type: AssertionType
    expected: Any
    tolerance: Optional[float] = None
    ignore_order: bool = False
    custom_function: Optional[str] = None
    message: Optional[str] = None

    @validator('tolerance')
    def validate_tolerance(cls, v):
        if v is not None and v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v


class TestFixtureConfig(BaseModel):
    """Configuration model for test fixtures."""
    name: str
    table_name: str
    fixture_type: str
    data_source: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    schema: Optional[Dict[str, str]] = None
    cleanup: bool = True

    @validator('fixture_type')
    def validate_fixture_type(cls, v):
        valid_types = ['csv', 'json', 'sql', 'inline', 'generated']
        if v not in valid_types:
            raise ValueError(f"Invalid fixture type. Must be one of: {valid_types}")
        return v


class TestCaseConfig(BaseModel):
    """Configuration model for individual test cases."""
    name: str
    description: Optional[str] = None
    sql: str
    fixtures: List[TestFixtureConfig] = Field(default_factory=list)
    assertions: List[TestAssertionConfig] = Field(default_factory=list)
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    timeout: int = 30
    isolation_level: str = "transaction"
    tags: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)

    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @validator('isolation_level')
    def validate_isolation_level(cls, v):
        valid_levels = ['none', 'transaction', 'schema', 'database']
        if v not in valid_levels:
            raise ValueError(f"Invalid isolation level. Must be one of: {valid_levels}")
        return v


class TestSuiteConfig(BaseModel):
    """Configuration model for test suites."""
    name: str
    description: Optional[str] = None
    database: str
    setup_sql: Optional[str] = None
    teardown_sql: Optional[str] = None
    fixtures: List[TestFixtureConfig] = Field(default_factory=list)
    test_cases: List[TestCaseConfig] = Field(default_factory=list)
    parallel: bool = False
    max_workers: int = 4
    isolation_level: str = "transaction"
    tags: List[str] = Field(default_factory=list)

    @validator('max_workers')
    def validate_max_workers(cls, v):
        if v <= 0:
            raise ValueError("Max workers must be positive")
        return v


class TestConfigSchema(BaseModel):
    """Root configuration schema for SQL unit tests."""
    version: str = "1.0"
    includes: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    databases: Dict[str, DatabaseConfig] = Field(default_factory=dict)
    global_fixtures: List[TestFixtureConfig] = Field(default_factory=list)
    test_suites: List[TestSuiteConfig] = Field(default_factory=list)

    @validator('version')
    def validate_version(cls, v):
        if not re.match(r'^\d+\.\d+$', v):
            raise ValueError("Version must be in format 'major.minor'")
        return v


class EnvironmentVariableResolver:
    """Resolves environment variables in configuration with defaults and validation."""

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    @classmethod
    def resolve(cls, value: Any, context: ConfigContext) -> Any:
        """Resolve environment variables in any value type."""
        if isinstance(value, str):
            return cls._resolve_string(value, context)
        elif isinstance(value, dict):
            return {k: cls.resolve(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item, context) for item in value]
        else:
            return value

    @classmethod
    def _resolve_string(cls, value: str, context: ConfigContext) -> str:
        """Resolve environment variables in a string value."""
        def replacer(match):
            var_expr = match.group(1)

            # Handle default values: ${VAR_NAME:default_value}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name = var_expr
                default_value = None

            # Look in context variables first, then environment
            if var_name in context.variables:
                return str(context.variables[var_name])
            elif var_name in os.environ:
                return os.environ[var_name]
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(f"Environment variable '{var_name}' not found and no default provided")

        return cls.ENV_VAR_PATTERN.sub(replacer, value)


class ConfigTemplateEngine:
    """Template engine for configuration inheritance and composition."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._template_cache: Dict[str, Dict[str, Any]] = {}

    def load_template(self, template_path: str, context: ConfigContext) -> Dict[str, Any]:
        """Load and process a configuration template."""
        full_path = self._resolve_path(template_path, context.base_path)

        # Check for circular includes
        if str(full_path) in context.include_stack:
            raise ValueError(f"Circular include detected: {' -> '.join(context.include_stack + [str(full_path)])}")

        # Load from cache if available
        cache_key = f"{full_path}:{context.environment}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Load and process template
        with open(full_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Create new context for this template
        template_context = ConfigContext(
            environment=context.environment,
            base_path=full_path.parent,
            variables=context.variables.copy(),
            include_stack=context.include_stack + [str(full_path)]
        )

        # Process includes first
        processed_config = self._process_includes(raw_config, template_context)

        # Resolve environment variables
        processed_config = EnvironmentVariableResolver.resolve(processed_config, template_context)

        # Cache the result
        self._template_cache[cache_key] = processed_config

        return processed_config

    def _process_includes(self, config: Dict[str, Any], context: ConfigContext) -> Dict[str, Any]:
        """Process include directives in configuration."""
        if 'includes' not in config:
            return config

        result = {}

        # Load and merge included configurations
        for include_path in config['includes']:
            included_config = self.load_template(include_path, context)
            result = self._deep_merge(result, included_config)

        # Merge current configuration on top
        current_config = {k: v for k, v in config.items() if k != 'includes'}
        result = self._deep_merge(result, current_config)

        return result

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value

        return result

    def _resolve_path(self, path: str, base_path: Path) -> Path:
        """Resolve a path relative to base path."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return base_path / path_obj


class AdvancedConfigLoader:
    """Advanced configuration loader with enterprise features."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.template_engine = ConfigTemplateEngine(self.base_path)
        self._schema_cache: Dict[str, TestConfigSchema] = {}

    def load_config(self,
                   config_path: Union[str, Path],
                   environment: str = "default",
                   variables: Optional[Dict[str, Any]] = None) -> TestConfigSchema:
        """
        Load and validate test configuration from file.

        Args:
            config_path: Path to configuration file
            environment: Environment name for variable resolution
            variables: Additional variables for template rendering

        Returns:
            Validated configuration schema

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.base_path / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Create loading context
        context = ConfigContext(
            environment=environment,
            base_path=config_path.parent,
            variables=variables or {},
            include_stack=[]
        )

        # Load configuration with template processing
        try:
            config_data = self.template_engine.load_template(str(config_path), context)

            # Validate and create schema
            config_schema = TestConfigSchema(**config_data)

            # Cache for future use
            cache_key = f"{config_path}:{environment}"
            self._schema_cache[cache_key] = config_schema

            logger.info(f"Loaded configuration from {config_path} for environment '{environment}'")
            return config_schema

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise ValueError(f"Configuration loading failed: {e}") from e

    def load_from_dict(self,
                      config_data: Dict[str, Any],
                      environment: str = "default",
                      variables: Optional[Dict[str, Any]] = None) -> TestConfigSchema:
        """
        Load and validate configuration from dictionary.

        Args:
            config_data: Configuration data as dictionary
            environment: Environment name for variable resolution
            variables: Additional variables for template rendering

        Returns:
            Validated configuration schema
        """
        context = ConfigContext(
            environment=environment,
            base_path=self.base_path,
            variables=variables or {},
            include_stack=[]
        )

        # Resolve environment variables
        resolved_data = EnvironmentVariableResolver.resolve(config_data, context)

        # Validate and create schema
        return TestConfigSchema(**resolved_data)

    def validate_config(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate configuration file without loading.

        Args:
            config_path: Path to configuration file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            self.load_config(config_path)
        except FileNotFoundError as e:
            errors.append(str(e))
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return errors

    def generate_schema_documentation(self) -> str:
        """Generate documentation for the configuration schema."""
        docs = []
        docs.append("# SQL Unit Test Configuration Schema\n")

        docs.append("## Root Configuration\n")
        docs.append("```yaml")
        docs.append("version: '1.0'  # Configuration schema version")
        docs.append("includes:       # List of configuration files to include")
        docs.append("  - common.yaml")
        docs.append("variables:      # Variables for template substitution")
        docs.append("  db_host: localhost")
        docs.append("databases:      # Database connection configurations")
        docs.append("  test_db:")
        docs.append("    driver: postgresql")
        docs.append("    host: ${DB_HOST:localhost}")
        docs.append("    database: ${DB_NAME}")
        docs.append("```\n")

        docs.append("## Test Suites\n")
        docs.append("```yaml")
        docs.append("test_suites:")
        docs.append("  - name: user_tests")
        docs.append("    description: Tests for user management")
        docs.append("    database: test_db")
        docs.append("    parallel: true")
        docs.append("    max_workers: 4")
        docs.append("    isolation_level: transaction")
        docs.append("```\n")

        docs.append("## Environment Variables\n")
        docs.append("Supported syntax:")
        docs.append("- `${VAR_NAME}` - Required variable")
        docs.append("- `${VAR_NAME:default}` - Variable with default value\n")

        return "\n".join(docs)

    def clear_cache(self):
        """Clear all caches."""
        self._schema_cache.clear()
        self.template_engine._template_cache.clear()


# Convenience functions for common operations
def load_test_config(config_path: Union[str, Path],
                    environment: str = "default",
                    variables: Optional[Dict[str, Any]] = None) -> TestConfigSchema:
    """Convenience function to load test configuration."""
    loader = AdvancedConfigLoader()
    return loader.load_config(config_path, environment, variables)


def validate_test_config(config_path: Union[str, Path]) -> List[str]:
    """Convenience function to validate test configuration."""
    loader = AdvancedConfigLoader()
    return loader.validate_config(config_path)
