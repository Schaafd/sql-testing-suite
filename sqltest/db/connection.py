"""Database connection management and adapter factory."""

import time
from typing import Dict, Optional, Type, Union

from sqltest.config.models import DatabaseConfig, DatabaseType, ConnectionPoolConfig, SQLTestConfig
from sqltest.db.base import BaseAdapter, QueryResult
from sqltest.db.adapters.postgresql import PostgreSQLAdapter
from sqltest.db.adapters.mysql import MySQLAdapter
from sqltest.db.adapters.sqlite import SQLiteAdapter
from sqltest.exceptions import DatabaseError


class AdapterFactory:
    """Factory for creating database adapters."""
    
    _adapters: Dict[DatabaseType, Type[BaseAdapter]] = {
        DatabaseType.POSTGRESQL: PostgreSQLAdapter,
        DatabaseType.MYSQL: MySQLAdapter,
        DatabaseType.SQLITE: SQLiteAdapter,
    }
    
    @classmethod
    def create_adapter(
        self,
        config: DatabaseConfig,
        pool_config: Optional[ConnectionPoolConfig] = None,
    ) -> BaseAdapter:
        """Create a database adapter based on configuration.
        
        Args:
            config: Database configuration.
            pool_config: Connection pool configuration.
            
        Returns:
            Database adapter instance.
            
        Raises:
            DatabaseError: If database type is not supported.
        """
        adapter_class = self._adapters.get(config.type)
        if not adapter_class:
            supported_types = list(self._adapters.keys())
            raise DatabaseError(
                f"Unsupported database type: {config.type}. "
                f"Supported types: {supported_types}"
            )
        
        return adapter_class(config, pool_config)
    
    @classmethod
    def register_adapter(cls, db_type: DatabaseType, adapter_class: Type[BaseAdapter]) -> None:
        """Register a custom database adapter.
        
        Args:
            db_type: Database type.
            adapter_class: Adapter class to register.
        """
        cls._adapters[db_type] = adapter_class
    
    @classmethod
    def get_supported_types(cls) -> list[DatabaseType]:
        """Get list of supported database types."""
        return list(cls._adapters.keys())


class ConnectionManager:
    """Manages database connections and adapters."""
    
    def __init__(self, config: SQLTestConfig) -> None:
        """Initialize connection manager.
        
        Args:
            config: SQLTest configuration.
        """
        self.config = config
        self._adapters: Dict[str, BaseAdapter] = {}
        self._factory = AdapterFactory()
    
    def get_adapter(self, db_name: Optional[str] = None) -> BaseAdapter:
        """Get database adapter by name.
        
        Args:
            db_name: Database connection name. If None, uses default database.
            
        Returns:
            Database adapter instance.
            
        Raises:
            DatabaseError: If database connection is not found or creation fails.
        """
        # Use default database if not specified
        if db_name is None:
            db_name = self.config.default_database
            
        if not db_name:
            raise DatabaseError("No database specified and no default database configured")
        
        # Check if database exists in configuration
        if db_name not in self.config.databases:
            available_dbs = list(self.config.databases.keys())
            raise DatabaseError(
                f"Database '{db_name}' not found in configuration. "
                f"Available databases: {available_dbs}"
            )
        
        # Return existing adapter if available
        if db_name in self._adapters:
            return self._adapters[db_name]
        
        # Create new adapter
        try:
            db_config = self.config.databases[db_name]
            pool_config = self.config.connection_pools.get("default", ConnectionPoolConfig())
            
            adapter = self._factory.create_adapter(db_config, pool_config)
            self._adapters[db_name] = adapter
            
            return adapter
            
        except Exception as e:
            raise DatabaseError(f"Failed to create adapter for database '{db_name}': {e}") from e
    
    def test_connection(self, db_name: Optional[str] = None) -> Dict[str, any]:
        """Test database connection.
        
        Args:
            db_name: Database connection name.
            
        Returns:
            Connection test result with timing and status information.
        """
        start_time = time.time()
        
        try:
            adapter = self.get_adapter(db_name)
            adapter.test_connection()
            
            end_time = time.time()
            
            return {
                'database': db_name or self.config.default_database,
                'status': 'success',
                'message': 'Connection successful',
                'response_time': round((end_time - start_time) * 1000, 2),  # milliseconds
                'driver': adapter.get_driver_name(),
                'database_type': adapter.config.type.value,
            }
            
        except DatabaseError as e:
            end_time = time.time()
            return {
                'database': db_name or self.config.default_database,
                'status': 'failed',
                'message': str(e),
                'response_time': round((end_time - start_time) * 1000, 2),
                'error': type(e).__name__,
            }
    
    def test_all_connections(self) -> Dict[str, Dict[str, any]]:
        """Test all configured database connections.
        
        Returns:
            Dictionary of connection test results for each database.
        """
        results = {}
        
        for db_name in self.config.databases.keys():
            results[db_name] = self.test_connection(db_name)
        
        return results
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, any]] = None,
        db_name: Optional[str] = None,
        fetch_results: bool = True,
        timeout: Optional[int] = None,
    ) -> QueryResult:
        """Execute SQL query on specified database.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            db_name: Database connection name.
            fetch_results: Whether to fetch result data.
            timeout: Query timeout in seconds.
            
        Returns:
            QueryResult instance.
        """
        adapter = self.get_adapter(db_name)
        return adapter.execute_query(query, params, fetch_results, timeout)
    
    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> Dict[str, any]:
        """Get table information from specified database.
        
        Args:
            table_name: Name of the table.
            schema: Schema name (optional).
            db_name: Database connection name.
            
        Returns:
            Table information dictionary.
        """
        adapter = self.get_adapter(db_name)
        return adapter.get_table_info(table_name, schema)
    
    def get_database_info(self, db_name: Optional[str] = None) -> Dict[str, any]:
        """Get general information about the database.
        
        Args:
            db_name: Database connection name.
            
        Returns:
            Database information dictionary.
        """
        adapter = self.get_adapter(db_name)
        db_config = adapter.config
        
        info = {
            'database_name': db_name or self.config.default_database,
            'database_type': db_config.type.value,
            'driver': adapter.get_driver_name(),
            'host': getattr(db_config, 'host', None),
            'port': getattr(db_config, 'port', None),
            'database': getattr(db_config, 'database', None),
            'username': getattr(db_config, 'username', None),
            'path': getattr(db_config, 'path', None),  # For SQLite
        }
        
        # Add database-specific information
        try:
            if hasattr(adapter, 'get_table_names'):
                info['table_count'] = len(adapter.get_table_names())
            if hasattr(adapter, 'get_view_names'):
                info['view_count'] = len(adapter.get_view_names())
            if hasattr(adapter, 'get_schema_names'):
                info['schema_count'] = len(adapter.get_schema_names())
        except Exception:
            # Ignore errors when getting additional info
            pass
        
        return info
    
    def close_all_connections(self) -> None:
        """Close all database connections and cleanup resources."""
        for adapter in self._adapters.values():
            try:
                adapter.close()
            except Exception:
                # Ignore errors when closing connections
                pass
        
        self._adapters.clear()
    
    def close_connection(self, db_name: str) -> None:
        """Close specific database connection.
        
        Args:
            db_name: Database connection name.
        """
        if db_name in self._adapters:
            try:
                self._adapters[db_name].close()
            except Exception:
                # Ignore errors when closing connection
                pass
            
            del self._adapters[db_name]
    
    def get_connection_status(self) -> Dict[str, any]:
        """Get status of all database connections.
        
        Returns:
            Dictionary with connection status information.
        """
        status = {
            'total_configured': len(self.config.databases),
            'total_active': len(self._adapters),
            'default_database': self.config.default_database,
            'connections': {},
        }
        
        for db_name in self.config.databases.keys():
            is_active = db_name in self._adapters
            status['connections'][db_name] = {
                'active': is_active,
                'type': self.config.databases[db_name].type.value,
            }
        
        return status


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager(config: Optional[SQLTestConfig] = None) -> ConnectionManager:
    """Get the global connection manager instance.
    
    Args:
        config: SQLTest configuration. If None, attempts to load from global config.
        
    Returns:
        Global ConnectionManager instance.
        
    Raises:
        DatabaseError: If no configuration is available.
    """
    global _connection_manager
    
    if _connection_manager is None:
        if config is None:
            # Try to get config from config module
            try:
                from sqltest.config import get_config
                config = get_config()
            except Exception as e:
                raise DatabaseError("No configuration available for connection manager") from e
        
        _connection_manager = ConnectionManager(config)
    
    return _connection_manager


def set_connection_manager(manager: ConnectionManager) -> None:
    """Set the global connection manager instance.
    
    Args:
        manager: ConnectionManager instance to set as global.
    """
    global _connection_manager
    _connection_manager = manager
