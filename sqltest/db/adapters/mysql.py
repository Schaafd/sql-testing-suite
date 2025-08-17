"""MySQL database adapter."""

from typing import Any, Dict, Optional
from urllib.parse import quote_plus

from sqltest.config.models import DatabaseConfig, ConnectionPoolConfig
from sqltest.db.base import BaseAdapter
from sqltest.exceptions import DatabaseError


class MySQLAdapter(BaseAdapter):
    """MySQL database adapter."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        pool_config: Optional[ConnectionPoolConfig] = None,
    ) -> None:
        """Initialize MySQL adapter."""
        super().__init__(config, pool_config)
        
        # Set default port if not specified
        if self.config.port is None:
            self.config.port = 3306
    
    def get_driver_name(self) -> str:
        """Get the driver name for MySQL."""
        return "pymysql"
    
    def build_connection_string(self) -> str:
        """Build MySQL connection string.
        
        Returns:
            MySQL connection string.
            
        Raises:
            DatabaseError: If required configuration is missing.
        """
        if not all([self.config.host, self.config.database, self.config.username, self.config.password]):
            raise DatabaseError("MySQL requires host, database, username, and password")
        
        # URL encode password to handle special characters
        password_encoded = quote_plus(self.config.password)
        
        connection_string = (
            f"mysql+pymysql://{self.config.username}:{password_encoded}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        # Add connection options
        options = self.config.options.copy()
        
        # Set default charset if not specified
        if 'charset' not in options:
            options['charset'] = 'utf8mb4'
        
        if options:
            option_string = "&".join([f"{k}={v}" for k, v in options.items()])
            connection_string += f"?{option_string}"
        
        return connection_string
    
    def _get_engine_options(self) -> Dict[str, Any]:
        """Get MySQL-specific engine options."""
        return {
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'connect_args': {
                'connect_timeout': self.config.options.get('connect_timeout', 10),
                'charset': self.config.options.get('charset', 'utf8mb4'),
                'autocommit': True,
            }
        }
    
    def _get_column_info_query(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get MySQL-specific query for column information."""
        database_condition = f"AND table_schema = '{schema}'" if schema else f"AND table_schema = '{self.config.database}'"
        
        return f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            ordinal_position,
            column_type,
            column_key,
            extra
        FROM information_schema.columns
        WHERE table_name = '{table_name}' {database_condition}
        ORDER BY ordinal_position
        """
    
    def get_database_names(self) -> list[str]:
        """Get list of database names on the MySQL server."""
        query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
        ORDER BY schema_name
        """
        result = self.execute_query(query)
        return [row['schema_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_table_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of table names in the database or schema."""
        database_condition = f"table_schema = '{schema}'" if schema else f"table_schema = '{self.config.database}'"
        
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE {database_condition} AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        result = self.execute_query(query)
        return [row['table_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_view_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of view names in the database or schema."""
        database_condition = f"table_schema = '{schema}'" if schema else f"table_schema = '{self.config.database}'"
        
        query = f"""
        SELECT table_name
        FROM information_schema.views
        WHERE {database_condition}
        ORDER BY table_name
        """
        result = self.execute_query(query)
        return [row['table_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_function_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of function names in the database or schema."""
        database_condition = f"routine_schema = '{schema}'" if schema else f"routine_schema = '{self.config.database}'"
        
        query = f"""
        SELECT routine_name, routine_type
        FROM information_schema.routines
        WHERE {database_condition}
        ORDER BY routine_name
        """
        result = self.execute_query(query)
        return [
            f"{row['routine_name']} ({row['routine_type'].lower()})"
            for row in result.data.to_dict('records')
        ] if not result.is_empty else []
    
    def get_trigger_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of trigger names in the database or schema."""
        database_condition = f"trigger_schema = '{schema}'" if schema else f"trigger_schema = '{self.config.database}'"
        
        query = f"""
        SELECT trigger_name, event_object_table
        FROM information_schema.triggers
        WHERE {database_condition}
        ORDER BY trigger_name
        """
        result = self.execute_query(query)
        return [
            f"{row['trigger_name']} (on {row['event_object_table']})"
            for row in result.data.to_dict('records')
        ] if not result.is_empty else []
    
    def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Run OPTIMIZE TABLE to defragment and reclaim space."""
        try:
            result = self.execute_query(f"OPTIMIZE TABLE {table_name}")
            return {
                'status': 'success',
                'message': f'Table {table_name} optimized successfully',
                'details': result.data.to_dict('records') if not result.is_empty else []
            }
        except Exception as e:
            raise DatabaseError(f"Failed to optimize table {table_name}: {e}") from e
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Run ANALYZE TABLE to update key distribution statistics."""
        try:
            result = self.execute_query(f"ANALYZE TABLE {table_name}")
            return {
                'status': 'success',
                'message': f'Table {table_name} analyzed successfully',
                'details': result.data.to_dict('records') if not result.is_empty else []
            }
        except Exception as e:
            raise DatabaseError(f"Failed to analyze table {table_name}: {e}") from e
    
    def show_table_status(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Show table status information."""
        try:
            if table_name:
                query = f"SHOW TABLE STATUS LIKE '{table_name}'"
            else:
                query = "SHOW TABLE STATUS"
                
            result = self.execute_query(query)
            return {
                'status': 'success',
                'tables': result.data.to_dict('records') if not result.is_empty else []
            }
        except Exception as e:
            raise DatabaseError(f"Failed to show table status: {e}") from e
    
    def show_processlist(self) -> Dict[str, Any]:
        """Show current MySQL processes."""
        try:
            result = self.execute_query("SHOW PROCESSLIST")
            return {
                'status': 'success',
                'processes': result.data.to_dict('records') if not result.is_empty else []
            }
        except Exception as e:
            raise DatabaseError(f"Failed to show processlist: {e}") from e
