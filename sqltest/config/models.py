"""Pydantic models for SQLTest Pro configuration."""

import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"


class ConnectionPoolConfig(BaseModel):
    """Connection pool configuration."""
    min_connections: int = Field(default=1, ge=0, le=100)
    max_connections: int = Field(default=10, ge=1, le=1000)
    timeout: int = Field(default=30, ge=1, le=3600)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)

    @model_validator(mode='after')
    def validate_connection_limits(self):
        """Ensure min_connections <= max_connections."""
        if self.min_connections > self.max_connections:
            raise ValueError("min_connections must be <= max_connections")
        return self


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    path: Optional[str] = None  # For SQLite
    options: Dict[str, Any] = Field(default_factory=dict)
    
    # Snowflake-specific fields
    account: Optional[str] = None
    warehouse: Optional[str] = None
    schema_name: Optional[str] = Field(None, alias='schema')  # Renamed to avoid BaseModel conflict
    role: Optional[str] = None

    @field_validator('port')
    def validate_port(cls, v):
        """Validate port number range."""
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @model_validator(mode='after')
    def validate_database_config(self):
        """Validate database-specific required fields."""
        if self.type == DatabaseType.SQLITE:
            if not self.path:
                raise ValueError("SQLite databases require a 'path' field")
        elif self.type == DatabaseType.SNOWFLAKE:
            required_fields = ['account', 'warehouse', 'database', 'username', 'password']
            for field in required_fields:
                if not getattr(self, field):
                    raise ValueError(f"Snowflake databases require '{field}' field")
        else:
            # Standard SQL databases
            required_fields = ['host', 'database', 'username', 'password']
            for field in required_fields:
                if not getattr(self, field):
                    raise ValueError(f"{self.type.value} databases require '{field}' field")
        return self


class ValidationSettings(BaseModel):
    """Global validation settings."""
    fail_fast: bool = Field(default=False, description="Stop on first validation failure")
    parallel_execution: bool = Field(default=True, description="Run validations in parallel")
    max_workers: int = Field(default=4, ge=1, le=32, description="Maximum parallel workers")
    report_format: str = Field(default="html", pattern="^(html|json|csv)$")
    output_dir: str = Field(default="./validation_reports")
    timeout: int = Field(default=300, ge=1, le=3600, description="Timeout in seconds")


class TestSettings(BaseModel):
    """Unit test settings."""
    isolation_level: str = Field(default="READ_COMMITTED")
    timeout: int = Field(default=300, ge=1, le=3600)
    continue_on_failure: bool = Field(default=True)
    generate_coverage: bool = Field(default=True)
    temp_schema: Optional[str] = Field(default="test_temp")
    

class SQLTestConfig(BaseModel):
    """Main configuration model for SQLTest Pro."""
    databases: Dict[str, DatabaseConfig]
    connection_pools: Dict[str, ConnectionPoolConfig] = Field(
        default_factory=lambda: {"default": ConnectionPoolConfig()}
    )
    default_database: Optional[str] = None
    validation_settings: ValidationSettings = Field(default_factory=ValidationSettings)
    test_settings: TestSettings = Field(default_factory=TestSettings)
    
    @model_validator(mode='after')
    def validate_default_database(self):
        """Ensure default_database exists in databases."""
        if self.default_database and self.default_database not in self.databases:
            raise ValueError(f"default_database '{self.default_database}' not found in databases")
        return self

    @model_validator(mode='after') 
    def set_default_database(self):
        """Set default database if not specified."""
        if not self.default_database and self.databases:
            self.default_database = next(iter(self.databases))
        return self


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings."""
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    config_file: Optional[str] = Field(default=None)
    
    class Config:
        env_prefix = "SQLTEST_"
        case_sensitive = False
