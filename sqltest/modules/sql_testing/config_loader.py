"""
YAML configuration loader for SQL unit testing framework.

This module handles loading and parsing SQL test suite configurations from YAML files,
including test definitions, fixtures, assertions, and suite-level settings.
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import os
import re

from .models import TestSuiteConfig, SQLTestConfig, TestFixtureConfig, TestAssertionConfig


class TestConfigLoader:
    """Loads and parses SQL test configurations from YAML files."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            base_path: Base path for resolving relative file paths
        """
        self.base_path = base_path or Path.cwd()
    
    def load_test_suite(self, config_path: Union[str, Path]) -> TestSuiteConfig:
        """
        Load test suite configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Parsed test suite configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = self._resolve_path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Test configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Interpolate environment variables
        config_data = self._interpolate_env_vars(raw_config)
        
        # Handle includes
        config_data = self._process_includes(config_data, config_path.parent)
        
        # Parse and validate configuration
        return TestSuiteConfig(**config_data)
    
    def load_multiple_test_suites(
        self, 
        config_paths: List[Union[str, Path]]
    ) -> List[TestSuiteConfig]:
        """
        Load multiple test suite configurations.
        
        Args:
            config_paths: List of paths to YAML configuration files
            
        Returns:
            List of parsed test suite configurations
        """
        test_suites = []
        
        for config_path in config_paths:
            try:
                suite = self.load_test_suite(config_path)
                test_suites.append(suite)
            except Exception as e:
                raise ValueError(f"Failed to load test suite from {config_path}: {e}")
        
        return test_suites
    
    def discover_test_configurations(
        self, 
        directory: Union[str, Path],
        pattern: str = "**/*test*.yaml"
    ) -> List[TestSuiteConfig]:
        """
        Discover and load all test configurations in a directory.
        
        Args:
            directory: Directory to search for test configurations
            pattern: Glob pattern for matching test files
            
        Returns:
            List of discovered test suite configurations
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found or not a directory: {directory}")
        
        # Find all matching configuration files
        config_files = list(directory.glob(pattern))
        
        return self.load_multiple_test_suites(config_files)
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve file path relative to base path."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return self.base_path / path_obj
    
    def _interpolate_env_vars(self, data: Any) -> Any:
        """
        Recursively interpolate environment variables in configuration data.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(data, dict):
            return {key: self._interpolate_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._interpolate_string(data)
        else:
            return data
    
    def _interpolate_string(self, text: str) -> str:
        """Interpolate environment variables in a string."""
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match):
            var_spec = match.group(1)
            
            # Check if default value is specified
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
            else:
                var_name = var_spec
                default_value = None
            
            # Get environment variable value
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(f"Environment variable {var_name} not found and no default provided")
        
        return re.sub(pattern, replace_env_var, text)
    
    def _process_includes(self, config_data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """
        Process include directives in configuration.
        
        Supports including other YAML files or specific sections.
        """
        if 'include' not in config_data:
            return config_data
        
        includes = config_data.pop('include')
        if not isinstance(includes, list):
            includes = [includes]
        
        # Process each include
        for include_spec in includes:
            if isinstance(include_spec, str):
                # Simple file include
                include_path = base_dir / include_spec
                included_data = self._load_yaml_file(include_path)
                config_data = self._merge_configs(config_data, included_data)
            elif isinstance(include_spec, dict):
                # Advanced include with options
                file_path = include_spec.get('file')
                section = include_spec.get('section')
                
                if not file_path:
                    raise ValueError("Include specification must have 'file' key")
                
                include_path = base_dir / file_path
                included_data = self._load_yaml_file(include_path)
                
                # Extract specific section if specified
                if section:
                    if section not in included_data:
                        raise ValueError(f"Section '{section}' not found in {include_path}")
                    included_data = {section: included_data[section]}
                
                config_data = self._merge_configs(config_data, included_data)
        
        return config_data
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Include file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Recursively process includes and environment variables
        data = self._interpolate_env_vars(data)
        data = self._process_includes(data, file_path.parent)
        
        return data
    
    def _merge_configs(self, base_config: Dict[str, Any], include_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries.
        
        Lists are extended, dictionaries are merged recursively.
        """
        result = base_config.copy()
        
        for key, value in include_config.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key].extend(value)
                else:
                    result[key] = value  # Override
            else:
                result[key] = value
        
        return result
    
    def validate_configuration(self, config: TestSuiteConfig) -> List[str]:
        """
        Validate test suite configuration and return list of issues.
        
        Args:
            config: Test suite configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check for duplicate test names
        test_names = [test.name for test in config.tests]
        if len(test_names) != len(set(test_names)):
            duplicates = [name for name in set(test_names) if test_names.count(name) > 1]
            errors.append(f"Duplicate test names found: {', '.join(duplicates)}")
        
        # Validate individual tests
        for test in config.tests:
            test_errors = self._validate_test(test)
            errors.extend([f"Test '{test.name}': {error}" for error in test_errors])
        
        # Check dependencies
        all_test_names = set(test_names)
        for test in config.tests:
            for dep in test.depends_on:
                if dep not in all_test_names:
                    errors.append(f"Test '{test.name}' depends on non-existent test '{dep}'")
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies(config.tests)
        if circular_deps:
            errors.append(f"Circular dependencies detected: {circular_deps}")
        
        return errors
    
    def _validate_test(self, test: SQLTestConfig) -> List[str]:
        """Validate individual test configuration."""
        errors = []
        
        # Check SQL is not empty
        if not test.sql.strip():
            errors.append("SQL cannot be empty")
        
        # Check at least one assertion
        if not test.assertions:
            errors.append("Test must have at least one assertion")
        
        # Validate fixture names are unique
        fixture_names = [f.name for f in test.fixtures]
        if len(fixture_names) != len(set(fixture_names)):
            duplicates = [name for name in set(fixture_names) if fixture_names.count(name) > 1]
            errors.append(f"Duplicate fixture names: {', '.join(duplicates)}")
        
        # Validate assertions
        for i, assertion in enumerate(test.assertions):
            if assertion.assertion_type == 'custom' and not assertion.custom_function:
                errors.append(f"Assertion {i + 1}: Custom assertion requires custom_function")
        
        return errors
    
    def _find_circular_dependencies(self, tests: List[SQLTestConfig]) -> Optional[str]:
        """Find circular dependencies in test configurations."""
        # Build dependency graph
        graph = {}
        for test in tests:
            graph[test.name] = test.depends_on
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> Optional[List[str]]:
            if node not in graph:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    cycle = has_cycle(neighbor)
                    if cycle:
                        return [node] + cycle
                elif neighbor in rec_stack:
                    return [node, neighbor]
            
            rec_stack.remove(node)
            return None
        
        for test_name in graph:
            if test_name not in visited:
                cycle = has_cycle(test_name)
                if cycle:
                    return " -> ".join(cycle)
        
        return None


# Factory function for creating test objects from YAML
def create_test_suite_from_yaml(yaml_path: Union[str, Path]) -> TestSuiteConfig:
    """
    Convenience function to create test suite from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Test suite configuration object
    """
    loader = TestConfigLoader()
    return loader.load_test_suite(yaml_path)
