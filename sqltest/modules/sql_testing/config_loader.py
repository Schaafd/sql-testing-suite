"""
Advanced configuration loading for SQL unit testing framework.

This module provides enterprise-grade configuration loading with features including
environment variable substitution, template inheritance, and validation.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, validator, Field
import logging

from ...config.models import DatabaseConfig
from .models import (
    TestCase,
    TestSuite,
    TestFixture,
    TestAssertion,
    AssertionType,
    TestIsolationLevel,
    FixtureType,
)

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

    def to_dataclass(self) -> TestAssertion:
        """Convert assertion config to dataclass."""
        return TestAssertion(
            assertion_type=self.type,
            expected=self.expected,
            tolerance=self.tolerance,
            ignore_order=self.ignore_order,
            custom_function=self.custom_function,
            message=self.message,
        )


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

    def to_dataclass(self) -> TestFixture:
        """Convert fixture config to dataclass."""
        fixture_type = FixtureType(self.fixture_type)
        return TestFixture(
            name=self.name,
            table_name=self.table_name,
            fixture_type=fixture_type,
            data_source=self.data_source,
            schema=self.schema,
            cleanup=self.cleanup,
        )


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

    def to_dataclass(self) -> TestCase:
        """Convert case config to dataclass."""
        isolation_level = TestIsolationLevel(self.isolation_level)
        return TestCase(
            name=self.name,
            description=self.description or "",
            sql=self.sql,
            fixtures=[fixture.to_dataclass() for fixture in self.fixtures],
            assertions=[assertion.to_dataclass() for assertion in self.assertions],
            setup_sql=self.setup_sql,
            teardown_sql=self.teardown_sql,
            timeout=self.timeout,
            depends_on=self.depends_on,
            tags=self.tags,
            isolation_level=isolation_level,
        )


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
    fail_fast: bool = False
    tags: List[str] = Field(default_factory=list)

    @validator('max_workers')
    def validate_max_workers(cls, v):
        if v <= 0:
            raise ValueError("Max workers must be positive")
        return v

    def to_dataclass(self) -> TestSuite:
        """Convert suite config to dataclass."""
        isolation_level = TestIsolationLevel(self.isolation_level)
        tests = [case.to_dataclass() for case in self.test_cases]
        return TestSuite(
            name=self.name,
            description=self.description or "",
            tests=tests,
            setup_sql=self.setup_sql,
            teardown_sql=self.teardown_sql,
            tags=self.tags,
            database=self.database,
            parallel=self.parallel,
            max_workers=self.max_workers,
            isolation_level=isolation_level,
            fail_fast=self.fail_fast,
        )


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
        """Resolve environment variables in any value type, including keys."""

        if isinstance(value, str):
            return cls._resolve_string(value, context)

        if isinstance(value, dict):
            resolved_dict: Dict[Any, Any] = {}
            for key, item in value.items():
                resolved_key = cls.resolve(key, context) if isinstance(key, str) else key
                resolved_dict[resolved_key] = cls.resolve(item, context)
            return resolved_dict

        if isinstance(value, list):
            return [cls.resolve(item, context) for item in value]

        return value

    @classmethod
    def _resolve_string(cls, value: str, context: ConfigContext) -> str:
        """Resolve environment variables in a string value."""

        result: List[str] = []
        index = 0
        while index < len(value):
            start = value.find('${', index)
            if start == -1:
                result.append(value[index:])
                break

            result.append(value[index:start])
            end = cls._find_expression_end(value, start + 2)
            if end == -1:
                raise ValueError(f"Unclosed environment variable expression in '{value}'")

            expression = value[start + 2:end]
            resolved = cls._resolve_expression(expression, context)
            result.append(resolved)
            index = end + 1

        return ''.join(result)

    @classmethod
    def _resolve_expression(cls, expression: str, context: ConfigContext) -> str:
        """Resolve a single ${...} expression."""

        var_name, default_value = cls._split_expression(expression)

        if var_name in context.variables:
            resolved: Any = context.variables[var_name]
        elif var_name in os.environ:
            resolved = os.environ[var_name]
        elif default_value is not None:
            resolved = cls._resolve_string(default_value, context)
        else:
            raise ValueError(
                f"Environment variable '{var_name}' not found and no default provided"
            )

        if isinstance(resolved, str) and '${' in resolved:
            return cls._resolve_string(resolved, context)

        return str(resolved)

    @classmethod
    def _split_expression(cls, expression: str) -> Tuple[str, Optional[str]]:
        """Split an expression into variable name and default value."""

        depth = 0
        for idx, char in enumerate(expression):
            if expression.startswith('${', idx):
                depth += 1
                continue
            if char == '}':
                if depth == 0:
                    raise ValueError(
                        f"Unexpected closing brace in environment expression: '{expression}'"
                    )
                depth -= 1
                continue
            if char == ':' and depth == 0:
                var_name = expression[:idx]
                default_value = expression[idx + 1:]
                return var_name, default_value if default_value != '' else ''

        return expression, None

    @staticmethod
    def _find_expression_end(value: str, start: int) -> int:
        """Find the closing brace for an expression, accounting for nested placeholders."""

        depth = 0
        index = start
        while index < len(value):
            if value.startswith('${', index):
                depth += 1
                index += 2
                continue
            if value[index] == '}':
                if depth == 0:
                    return index
                depth -= 1
            index += 1
        return -1


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

        # Include environment and variable fingerprint to avoid stale cache entries
        variables_fingerprint = tuple(sorted((str(k), repr(v)) for k, v in context.variables.items()))
        cache_key = f"{full_path}:{context.environment}:{hash(variables_fingerprint)}"
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

        # Merge template-defined variables into the resolution context before substitution
        config_variables = processed_config.get('variables')
        if isinstance(config_variables, dict):
            template_context.variables = {
                **config_variables,
                **template_context.variables,
            }

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


class TestConfigLoader(AdvancedConfigLoader):
    """Compatibility shim preserving legacy loader name."""

    def __init__(self, base_path: Optional[Path] = None):
        super().__init__(base_path)


def create_test_suite_from_yaml(
    config_path: Union[str, Path],
    *,
    suite_name: Optional[str] = None,
    environment: str = "default",
    variables: Optional[Dict[str, Any]] = None,
) -> TestSuite:
    """Load a test suite definition from YAML and return it as a dataclass."""

    config_path = Path(config_path)
    loader = TestConfigLoader(base_path=config_path.parent)
    schema = loader.load_config(config_path, environment=environment, variables=variables)

    if not schema.test_suites:
        raise ValueError("No test suites defined in configuration")

    if suite_name is None:
        suite_config = schema.test_suites[0]
    else:
        suite_config = next((suite for suite in schema.test_suites if suite.name == suite_name), None)
        if suite_config is None:
            available = ", ".join(suite.name for suite in schema.test_suites)
            raise ValueError(f"Test suite '{suite_name}' not found. Available suites: {available}")

    return suite_config.to_dataclass()


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
