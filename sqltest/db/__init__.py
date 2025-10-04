"""Database connectivity and query execution."""

from sqltest.db.base import BaseAdapter, QueryResult, AggregateSpec
from sqltest.db.connection import (
    ConnectionManager,
    AdapterFactory,
    get_connection_manager,
    set_connection_manager,
)
from sqltest.db.adapters import (
    PostgreSQLAdapter,
    MySQLAdapter,
    SQLiteAdapter,
)

__all__ = [
    # Base classes
    "BaseAdapter",
    "QueryResult",
    "AggregateSpec",
    # Connection management
    "ConnectionManager",
    "AdapterFactory",
    "get_connection_manager",
    "set_connection_manager",
    # Database adapters
    "PostgreSQLAdapter",
    "MySQLAdapter",
    "SQLiteAdapter",
]
