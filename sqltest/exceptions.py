"""Core exceptions for SQLTest Pro."""

from typing import Any, Dict, Optional


class SQLTestError(Exception):
    """Base exception for all SQLTest Pro errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(SQLTestError):
    """Raised when there's an error in configuration parsing or validation."""
    pass


class DatabaseError(SQLTestError):
    """Raised when there's an error connecting to or querying a database."""
    
    def __init__(
        self, 
        message: str, 
        database_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.database_type = database_type
        self.connection_string = connection_string


class ValidationError(SQLTestError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        table: Optional[str] = None,
        column: Optional[str] = None,
        rule_type: Optional[str] = None,
        failed_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.table = table
        self.column = column
        self.rule_type = rule_type
        self.failed_count = failed_count


class TestExecutionError(SQLTestError):
    """Raised when unit test execution fails."""
    
    def __init__(
        self,
        message: str,
        test_name: Optional[str] = None,
        test_group: Optional[str] = None,
        sql_query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.test_name = test_name
        self.test_group = test_group
        self.sql_query = sql_query


class ProfilingError(SQLTestError):
    """Raised when data profiling encounters an error."""
    pass
