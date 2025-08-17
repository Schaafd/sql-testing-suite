"""Configuration parser for SQLTest Pro."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from sqltest.config.models import SQLTestConfig, EnvironmentSettings
from sqltest.exceptions import ConfigurationError


class ConfigParser:
    """Configuration parser with environment variable interpolation."""
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self) -> None:
        """Initialize the configuration parser."""
        self.env_settings = EnvironmentSettings()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> SQLTestConfig:
        """Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, looks for default locations.
            
        Returns:
            Validated SQLTestConfig instance.
            
        Raises:
            ConfigurationError: If configuration is invalid or file not found.
        """
        config_file = self._find_config_file(config_path)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
                
            if not raw_config:
                raise ConfigurationError(f"Configuration file '{config_file}' is empty")
                
            # Process environment variables
            processed_config = self._process_env_vars(raw_config)
            
            # Handle includes
            if 'include' in processed_config:
                processed_config = self._process_includes(processed_config, config_file)
            
            # Validate and create config
            return SQLTestConfig(**processed_config)
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file '{config_file}' not found")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in '{config_file}': {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find configuration file in default locations.
        
        Args:
            config_path: Explicit path to configuration file.
            
        Returns:
            Path to configuration file.
            
        Raises:
            ConfigurationError: If no configuration file is found.
        """
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise ConfigurationError(f"Configuration file '{config_path}' not found")
        
        # Check environment variable
        if self.env_settings.config_file:
            path = Path(self.env_settings.config_file)
            if path.exists():
                return path
                
        # Check default locations
        default_locations = [
            Path.cwd() / "sqltest.yaml",
            Path.cwd() / "sqltest.yml", 
            Path.cwd() / "config" / "sqltest.yaml",
            Path.cwd() / "configs" / "database.yaml",
            Path.cwd() / "examples" / "configs" / "database.yaml",
        ]
        
        for location in default_locations:
            if location.exists():
                return location
                
        raise ConfigurationError(
            f"No configuration file found in default locations: {default_locations}"
        )
    
    def _process_env_vars(self, config: Any) -> Any:
        """Recursively process environment variables in configuration.
        
        Args:
            config: Configuration data (dict, list, or primitive).
            
        Returns:
            Configuration with environment variables resolved.
        """
        if isinstance(config, dict):
            return {key: self._process_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config
    
    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in a string.
        
        Args:
            value: String potentially containing environment variables.
            
        Returns:
            String with environment variables substituted.
            
        Raises:
            ConfigurationError: If required environment variable is not set.
        """
        def replace_var(match):
            var_expr = match.group(1)
            
            # Handle default values: ${VAR:-default}
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.getenv(var_name.strip(), default.strip())
            
            # Handle required variables: ${VAR}
            var_name = var_expr.strip()
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ConfigurationError(f"Required environment variable '{var_name}' is not set")
            return env_value
        
        return self.ENV_VAR_PATTERN.sub(replace_var, value)
    
    def _process_includes(self, config: Dict[str, Any], base_path: Union[str, Path]) -> Dict[str, Any]:
        """Process include directives in configuration.
        
        Args:
            config: Configuration dictionary.
            base_path: Base path for resolving relative includes.
            
        Returns:
            Configuration with includes processed.
        """
        if 'include' not in config:
            return config
            
        base_dir = Path(base_path).parent
        includes = config.pop('include')
        
        if not isinstance(includes, list):
            includes = [includes]
            
        # Load and merge included files
        for include_file in includes:
            include_path = base_dir / include_file
            
            try:
                with open(include_path, 'r', encoding='utf-8') as file:
                    included_config = yaml.safe_load(file)
                    
                if included_config:
                    # Process environment variables in included config
                    included_config = self._process_env_vars(included_config)
                    
                    # Merge configurations (included files have lower priority)
                    config = self._merge_configs(included_config, config)
                    
            except FileNotFoundError:
                raise ConfigurationError(f"Included file '{include_path}' not found")
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Invalid YAML in included file '{include_path}': {e}")
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.
        
        Args:
            base: Base configuration (lower priority).
            override: Override configuration (higher priority).
            
        Returns:
            Merged configuration.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """Validate a configuration file without loading it.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            True if configuration is valid.
            
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        try:
            self.load_config(config_path)
            return True
        except ConfigurationError:
            raise
    
    def create_sample_config(self, output_path: Union[str, Path]) -> None:
        """Create a sample configuration file.
        
        Args:
            output_path: Path where to create the sample configuration.
        """
        sample_config = {
            'databases': {
                'dev': {
                    'type': 'postgresql',
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'myapp_dev',
                    'username': 'dev_user',
                    'password': '${DEV_DB_PASSWORD:-dev_password}',
                    'options': {
                        'sslmode': 'prefer',
                        'connect_timeout': 10
                    }
                },
                'test': {
                    'type': 'sqlite',
                    'path': './test.db'
                }
            },
            'connection_pools': {
                'default': {
                    'min_connections': 1,
                    'max_connections': 10,
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1.0
                }
            },
            'default_database': 'dev',
            'validation_settings': {
                'fail_fast': False,
                'parallel_execution': True,
                'max_workers': 4,
                'report_format': 'html',
                'output_dir': './validation_reports'
            },
            'test_settings': {
                'isolation_level': 'READ_COMMITTED',
                'timeout': 300,
                'continue_on_failure': True,
                'generate_coverage': True,
                'temp_schema': 'test_temp'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(sample_config, file, default_flow_style=False, sort_keys=False)


# Global configuration instance
_config_parser = ConfigParser()
_loaded_config: Optional[SQLTestConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None, reload: bool = False) -> SQLTestConfig:
    """Get the global configuration instance.
    
    Args:
        config_path: Path to configuration file.
        reload: Force reload of configuration.
        
    Returns:
        Global SQLTestConfig instance.
    """
    global _loaded_config
    
    if _loaded_config is None or reload:
        _loaded_config = _config_parser.load_config(config_path)
    
    return _loaded_config


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Validate a configuration file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        True if valid.
    """
    return _config_parser.validate_config_file(config_path)


def create_sample_config(output_path: Union[str, Path]) -> None:
    """Create a sample configuration file.
    
    Args:
        output_path: Output path for sample configuration.
    """
    _config_parser.create_sample_config(output_path)
