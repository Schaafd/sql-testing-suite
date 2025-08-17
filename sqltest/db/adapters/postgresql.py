"""PostgreSQL database adapter."""

from typing import Any, Dict, Optional
from urllib.parse import quote_plus

from sqltest.config.models import DatabaseConfig, ConnectionPoolConfig
from sqltest.db.base import BaseAdapter
from sqltest.exceptions import DatabaseError


class PostgreSQLAdapter(BaseAdapter):
    """PostgreSQL database adapter."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        pool_config: Optional[ConnectionPoolConfig] = None,
    ) -> None:
        """Initialize PostgreSQL adapter."""
        super().__init__(config, pool_config)
        
        # Set default port if not specified
        if self.config.port is None:
            self.config.port = 5432
    
    def get_driver_name(self) -> str:
        """Get the driver name for PostgreSQL."""
        return "psycopg2"
    
    def build_connection_string(self) -> str:
        """Build PostgreSQL connection string.
        
        Returns:
            PostgreSQL connection string.
            
        Raises:
            DatabaseError: If required configuration is missing.
        """
        if not all([self.config.host, self.config.database, self.config.username, self.config.password]):
            raise DatabaseError("PostgreSQL requires host, database, username, and password")
        
        # URL encode password to handle special characters
        password_encoded = quote_plus(self.config.password)
        
        connection_string = (
            f"postgresql+psycopg2://{self.config.username}:{password_encoded}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        # Add connection options
        options = self.config.options.copy()
        
        # Set default SSL mode if not specified
        if 'sslmode' not in options:
            options['sslmode'] = 'prefer'
        
        if options:
            option_string = "&".join([f"{k}={v}" for k, v in options.items()])
            connection_string += f"?{option_string}"
        
        return connection_string
    
    def _get_engine_options(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific engine options."""
        return {
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'connect_args': {
                'connect_timeout': self.config.options.get('connect_timeout', 10),
                'application_name': self.config.options.get('application_name', 'sqltest_pro'),
            }
        }
    
    def _get_column_info_query(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get PostgreSQL-specific query for column information."""
        schema_condition = f"AND table_schema = '{schema}'" if schema else "AND table_schema = 'public'"
        
        return f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            ordinal_position
        FROM information_schema.columns
        WHERE table_name = '{table_name}' {schema_condition}
        ORDER BY ordinal_position
        """
    
    def get_schema_names(self) -> list[str]:
        """Get list of schema names in the database."""
        query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        ORDER BY schema_name
        """
        result = self.execute_query(query)
        return [row['schema_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_table_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of table names in the database or schema."""
        schema_condition = f"table_schema = '{schema}'" if schema else "table_schema = 'public'"
        
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE {schema_condition} AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        result = self.execute_query(query)
        return [row['table_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_view_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of view names in the database or schema."""
        schema_condition = f"table_schema = '{schema}'" if schema else "table_schema = 'public'"
        
        query = f"""
        SELECT table_name
        FROM information_schema.views
        WHERE {schema_condition}
        ORDER BY table_name
        """
        result = self.execute_query(query)
        return [row['table_name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_function_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of function names in the database or schema."""
        schema_condition = f"routine_schema = '{schema}'" if schema else "routine_schema = 'public'"
        
        query = f"""
        SELECT routine_name, routine_type
        FROM information_schema.routines
        WHERE {schema_condition}
        ORDER BY routine_name
        """
        result = self.execute_query(query)
        return [
            f"{row['routine_name']} ({row['routine_type'].lower()})"
            for row in result.data.to_dict('records')
        ] if not result.is_empty else []
    
    def analyze_table(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Run ANALYZE on a table to update statistics."""
        full_table_name = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
        
        try:
            self.execute_query(f"ANALYZE {full_table_name}", fetch_results=False)
            return {'status': 'success', 'message': f'Table {full_table_name} analyzed successfully'}
        except Exception as e:
            raise DatabaseError(f"Failed to analyze table {full_table_name}: {e}") from e
