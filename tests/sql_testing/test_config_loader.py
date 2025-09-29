"""
Comprehensive tests for the advanced configuration loader.

This module tests configuration loading, template inheritance,
environment variable resolution, and validation.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from sqltest.modules.sql_testing.config_loader import (
    AdvancedConfigLoader,
    EnvironmentVariableResolver,
    ConfigTemplateEngine,
    ConfigContext,
    TestConfigSchema,
    load_test_config,
    validate_test_config
)


class TestEnvironmentVariableResolver:
    """Test environment variable resolution functionality."""

    def test_resolve_simple_variable(self):
        """Test resolving simple environment variable."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={},
            include_stack=[]
        )

        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            result = EnvironmentVariableResolver.resolve('${TEST_VAR}', context)
            assert result == 'test_value'

    def test_resolve_variable_with_default(self):
        """Test resolving variable with default value."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={},
            include_stack=[]
        )

        result = EnvironmentVariableResolver.resolve('${NONEXISTENT:default_val}', context)
        assert result == 'default_val'

    def test_resolve_variable_from_context(self):
        """Test resolving variable from context variables."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={'CONTEXT_VAR': 'context_value'},
            include_stack=[]
        )

        result = EnvironmentVariableResolver.resolve('${CONTEXT_VAR}', context)
        assert result == 'context_value'

    def test_resolve_missing_variable_error(self):
        """Test error when variable is missing and no default."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={},
            include_stack=[]
        )

        with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not found"):
            EnvironmentVariableResolver.resolve('${MISSING_VAR}', context)

    def test_resolve_complex_string(self):
        """Test resolving complex strings with multiple variables."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={'VAR1': 'value1'},
            include_stack=[]
        )

        with patch.dict(os.environ, {'VAR2': 'value2'}):
            result = EnvironmentVariableResolver.resolve(
                'prefix_${VAR1}_middle_${VAR2}_suffix',
                context
            )
            assert result == 'prefix_value1_middle_value2_suffix'

    def test_resolve_dict(self):
        """Test resolving variables in dictionary."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={'KEY': 'resolved_key', 'VALUE': 'resolved_value'},
            include_stack=[]
        )

        input_dict = {
            '${KEY}': '${VALUE}',
            'static': 'unchanged'
        }

        result = EnvironmentVariableResolver.resolve(input_dict, context)
        assert result == {
            'resolved_key': 'resolved_value',
            'static': 'unchanged'
        }

    def test_resolve_list(self):
        """Test resolving variables in list."""
        context = ConfigContext(
            environment="test",
            base_path=Path("."),
            variables={'ITEM': 'resolved_item'},
            include_stack=[]
        )

        input_list = ['${ITEM}', 'static', '${ITEM:default}']

        result = EnvironmentVariableResolver.resolve(input_list, context)
        assert result == ['resolved_item', 'static', 'resolved_item']


class TestConfigTemplateEngine:
    """Test configuration template engine functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def template_engine(self, temp_dir):
        """Create template engine with temporary directory."""
        return ConfigTemplateEngine(temp_dir)

    def test_load_simple_template(self, template_engine, temp_dir):
        """Test loading simple template file."""
        # Create test template file
        template_content = {
            'name': 'test_suite',
            'database': 'test_db'
        }

        template_file = temp_dir / "test_template.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(template_content, f)

        context = ConfigContext(
            environment="test",
            base_path=temp_dir,
            variables={},
            include_stack=[]
        )

        result = template_engine.load_template(str(template_file), context)
        assert result == template_content

    def test_template_with_includes(self, template_engine, temp_dir):
        """Test template with include directives."""
        # Create base template
        base_content = {
            'databases': {
                'test_db': {'driver': 'sqlite', 'database': ':memory:'}
            }
        }

        base_file = temp_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_content, f)

        # Create main template with includes
        main_content = {
            'includes': ['base.yaml'],
            'test_suites': [
                {'name': 'suite1', 'database': 'test_db'}
            ]
        }

        main_file = temp_dir / "main.yaml"
        with open(main_file, 'w') as f:
            yaml.dump(main_content, f)

        context = ConfigContext(
            environment="test",
            base_path=temp_dir,
            variables={},
            include_stack=[]
        )

        result = template_engine.load_template(str(main_file), context)

        assert 'databases' in result
        assert 'test_suites' in result
        assert result['databases'] == base_content['databases']
        assert result['test_suites'] == main_content['test_suites']

    def test_circular_include_detection(self, template_engine, temp_dir):
        """Test detection of circular includes."""
        # Create circular include files
        file1_content = {'includes': ['file2.yaml'], 'data': 'file1'}
        file2_content = {'includes': ['file1.yaml'], 'data': 'file2'}

        file1 = temp_dir / "file1.yaml"
        file2 = temp_dir / "file2.yaml"

        with open(file1, 'w') as f:
            yaml.dump(file1_content, f)
        with open(file2, 'w') as f:
            yaml.dump(file2_content, f)

        context = ConfigContext(
            environment="test",
            base_path=temp_dir,
            variables={},
            include_stack=[]
        )

        with pytest.raises(ValueError, match="Circular include detected"):
            template_engine.load_template(str(file1), context)

    def test_deep_merge_functionality(self, template_engine):
        """Test deep merge functionality."""
        base = {
            'databases': {
                'db1': {'driver': 'sqlite'},
                'db2': {'driver': 'postgres'}
            },
            'global_settings': {
                'timeout': 30,
                'retries': 3
            }
        }

        overlay = {
            'databases': {
                'db1': {'database': ':memory:'},  # Add to existing
                'db3': {'driver': 'mysql'}  # New database
            },
            'global_settings': {
                'timeout': 60  # Override existing
            },
            'new_section': {
                'value': 'new'
            }
        }

        result = template_engine._deep_merge(base, overlay)

        expected = {
            'databases': {
                'db1': {'driver': 'sqlite', 'database': ':memory:'},
                'db2': {'driver': 'postgres'},
                'db3': {'driver': 'mysql'}
            },
            'global_settings': {
                'timeout': 60,
                'retries': 3
            },
            'new_section': {
                'value': 'new'
            }
        }

        assert result == expected

    def test_template_caching(self, template_engine, temp_dir):
        """Test template caching functionality."""
        template_content = {'name': 'cached_template'}

        template_file = temp_dir / "cached.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(template_content, f)

        context = ConfigContext(
            environment="test",
            base_path=temp_dir,
            variables={},
            include_stack=[]
        )

        # Load template first time
        result1 = template_engine.load_template(str(template_file), context)

        # Load template second time (should use cache)
        result2 = template_engine.load_template(str(template_file), context)

        assert result1 == result2
        assert id(result1) == id(result2)  # Same object from cache


class TestAdvancedConfigLoader:
    """Test the main advanced configuration loader."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config_loader(self, temp_dir):
        """Create config loader with temporary directory."""
        return AdvancedConfigLoader(temp_dir)

    def test_load_valid_config(self, config_loader, temp_dir):
        """Test loading valid configuration file."""
        config_content = {
            'version': '1.0',
            'databases': {
                'test_db': {
                    'driver': 'sqlite',
                    'database': ':memory:'
                }
            },
            'test_suites': [
                {
                    'name': 'sample_suite',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'sample_test',
                            'sql': 'SELECT 1',
                            'assertions': [
                                {
                                    'type': 'equals',
                                    'expected': 1
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config_file = temp_dir / "valid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        result = config_loader.load_config(config_file)

        assert isinstance(result, TestConfigSchema)
        assert result.version == '1.0'
        assert len(result.test_suites) == 1
        assert result.test_suites[0].name == 'sample_suite'

    def test_load_config_with_variables(self, config_loader, temp_dir):
        """Test loading config with environment variables."""
        config_content = {
            'version': '1.0',
            'variables': {
                'db_name': 'test_database'
            },
            'databases': {
                'test_db': {
                    'driver': 'sqlite',
                    'database': '${DB_FILE:${db_name}.db}'
                }
            }
        }

        config_file = temp_dir / "config_with_vars.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        variables = {'db_name': 'custom_db'}

        result = config_loader.load_config(
            config_file,
            environment="development",
            variables=variables
        )

        assert result.databases['test_db'].database == 'custom_db.db'

    def test_load_config_from_dict(self, config_loader):
        """Test loading configuration from dictionary."""
        config_data = {
            'version': '1.0',
            'test_suites': []
        }

        result = config_loader.load_from_dict(config_data)

        assert isinstance(result, TestConfigSchema)
        assert result.version == '1.0'

    def test_validate_config_valid(self, config_loader, temp_dir):
        """Test validation of valid configuration."""
        config_content = {
            'version': '1.0',
            'test_suites': []
        }

        config_file = temp_dir / "valid.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        errors = config_loader.validate_config(config_file)
        assert len(errors) == 0

    def test_validate_config_invalid(self, config_loader, temp_dir):
        """Test validation of invalid configuration."""
        config_content = {
            'version': 'invalid_version',  # Invalid version format
            'test_suites': []
        }

        config_file = temp_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        errors = config_loader.validate_config(config_file)
        assert len(errors) > 0
        assert any("Version must be in format" in error for error in errors)

    def test_validate_config_missing_file(self, config_loader):
        """Test validation of missing configuration file."""
        errors = config_loader.validate_config("nonexistent.yaml")
        assert len(errors) > 0
        assert any("not found" in error for error in errors)

    def test_generate_schema_documentation(self, config_loader):
        """Test schema documentation generation."""
        docs = config_loader.generate_schema_documentation()

        assert "# SQL Unit Test Configuration Schema" in docs
        assert "## Root Configuration" in docs
        assert "## Test Suites" in docs
        assert "## Environment Variables" in docs
        assert "${VAR_NAME}" in docs

    def test_config_caching(self, config_loader, temp_dir):
        """Test configuration caching functionality."""
        config_content = {
            'version': '1.0',
            'test_suites': []
        }

        config_file = temp_dir / "cached_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        # Load config first time
        result1 = config_loader.load_config(config_file)

        # Load config second time (should use cache)
        result2 = config_loader.load_config(config_file)

        assert result1.version == result2.version
        # Note: Due to Pydantic, objects won't be identical but should have same content

        # Clear cache and verify it works
        config_loader.clear_cache()
        result3 = config_loader.load_config(config_file)
        assert result3.version == result1.version

    def test_complex_configuration_scenario(self, config_loader, temp_dir):
        """Test complex configuration scenario with multiple features."""
        # Create base configuration
        base_config = {
            'databases': {
                'default_db': {
                    'driver': 'sqlite',
                    'database': '${TEST_DB_PATH:/tmp/test.db}'
                }
            },
            'global_fixtures': [
                {
                    'name': 'users',
                    'table_name': 'users',
                    'fixture_type': 'csv',
                    'data_source': 'fixtures/users.csv'
                }
            ]
        }

        base_file = temp_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)

        # Create main configuration with includes
        main_config = {
            'version': '1.0',
            'includes': ['base.yaml'],
            'variables': {
                'suite_name': 'integration_tests'
            },
            'test_suites': [
                {
                    'name': '${suite_name}',
                    'database': 'default_db',
                    'parallel': True,
                    'max_workers': 4,
                    'test_cases': [
                        {
                            'name': 'test_user_count',
                            'sql': 'SELECT COUNT(*) as count FROM users',
                            'assertions': [
                                {
                                    'type': 'row_count',
                                    'expected': 1
                                }
                            ],
                            'isolation_level': 'transaction',
                            'timeout': 30
                        }
                    ]
                }
            ]
        }

        main_file = temp_dir / "main.yaml"
        with open(main_file, 'w') as f:
            yaml.dump(main_config, f)

        # Load with environment variables
        with patch.dict(os.environ, {'TEST_DB_PATH': '/custom/path/test.db'}):
            result = config_loader.load_config(main_file, environment="test")

        # Verify complex configuration was processed correctly
        assert result.version == '1.0'
        assert len(result.databases) == 1
        assert result.databases['default_db'].database == '/custom/path/test.db'
        assert len(result.global_fixtures) == 1
        assert result.global_fixtures[0].name == 'users'
        assert len(result.test_suites) == 1
        assert result.test_suites[0].name == 'integration_tests'
        assert result.test_suites[0].parallel is True
        assert result.test_suites[0].max_workers == 4
        assert len(result.test_suites[0].test_cases) == 1
        assert result.test_suites[0].test_cases[0].isolation_level == 'transaction'


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_load_test_config_function(self, temp_dir):
        """Test load_test_config convenience function."""
        config_content = {
            'version': '1.0',
            'test_suites': []
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        result = load_test_config(config_file)

        assert isinstance(result, TestConfigSchema)
        assert result.version == '1.0'

    def test_validate_test_config_function(self, temp_dir):
        """Test validate_test_config convenience function."""
        config_content = {
            'version': '1.0',
            'test_suites': []
        }

        config_file = temp_dir / "valid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        errors = validate_test_config(config_file)
        assert len(errors) == 0


class TestConfigurationValidation:
    """Test configuration validation edge cases."""

    @pytest.fixture
    def config_loader(self):
        return AdvancedConfigLoader()

    def test_invalid_version_format(self, config_loader):
        """Test invalid version format validation."""
        config_data = {
            'version': 'invalid',
            'test_suites': []
        }

        with pytest.raises(ValueError, match="Version must be in format"):
            config_loader.load_from_dict(config_data)

    def test_invalid_fixture_type(self, config_loader):
        """Test invalid fixture type validation."""
        config_data = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'test_suite',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_case',
                            'sql': 'SELECT 1',
                            'fixtures': [
                                {
                                    'name': 'invalid_fixture',
                                    'table_name': 'test_table',
                                    'fixture_type': 'invalid_type',
                                    'data_source': 'test.csv'
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        with pytest.raises(ValueError, match="Invalid fixture type"):
            config_loader.load_from_dict(config_data)

    def test_invalid_isolation_level(self, config_loader):
        """Test invalid isolation level validation."""
        config_data = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'test_suite',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_case',
                            'sql': 'SELECT 1',
                            'isolation_level': 'invalid_level'
                        }
                    ]
                }
            ]
        }

        with pytest.raises(ValueError, match="Invalid isolation level"):
            config_loader.load_from_dict(config_data)

    def test_negative_timeout(self, config_loader):
        """Test negative timeout validation."""
        config_data = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'test_suite',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_case',
                            'sql': 'SELECT 1',
                            'timeout': -1
                        }
                    ]
                }
            ]
        }

        with pytest.raises(ValueError, match="Timeout must be positive"):
            config_loader.load_from_dict(config_data)

    def test_negative_tolerance(self, config_loader):
        """Test negative tolerance validation."""
        config_data = {
            'version': '1.0',
            'test_suites': [
                {
                    'name': 'test_suite',
                    'database': 'test_db',
                    'test_cases': [
                        {
                            'name': 'test_case',
                            'sql': 'SELECT 1',
                            'assertions': [
                                {
                                    'type': 'equals',
                                    'expected': 1,
                                    'tolerance': -0.1
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        with pytest.raises(ValueError, match="Tolerance must be non-negative"):
            config_loader.load_from_dict(config_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])