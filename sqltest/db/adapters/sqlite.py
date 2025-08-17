"""SQLite database adapter."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from sqltest.config.models import DatabaseConfig, ConnectionPoolConfig
from sqltest.db.base import BaseAdapter
from sqltest.exceptions import DatabaseError


class SQLiteAdapter(BaseAdapter):
    """SQLite database adapter."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        pool_config: Optional[ConnectionPoolConfig] = None,
    ) -> None:
        """Initialize SQLite adapter."""
        super().__init__(config, pool_config)
        
        if not self.config.path:
            raise DatabaseError("SQLite requires a database file path")
    
    def get_driver_name(self) -> str:
        """Get the driver name for SQLite."""
        return "sqlite"
    
    def build_connection_string(self) -> str:
        """Build SQLite connection string.
        
        Returns:
            SQLite connection string.
            
        Raises:
            DatabaseError: If database path is invalid.
        """
        if not self.config.path:
            raise DatabaseError("SQLite requires a database file path")
        
        # Convert relative paths to absolute paths
        db_path = Path(self.config.path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        
        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        connection_string = f"sqlite:///{db_path}"
        
        return connection_string
    
    def _get_engine_options(self) -> Dict[str, Any]:
        """Get SQLite-specific engine options."""
        return {
            'pool_pre_ping': True,
            'pool_recycle': -1,  # No recycling for SQLite
            'connect_args': {
                'check_same_thread': False,  # Allow multi-threading
                'timeout': self.config.options.get('timeout', 30),
            }
        }
    
    def _get_column_info_query(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get SQLite-specific query for column information."""
        # SQLite uses PRAGMA to get table info
        return f"PRAGMA table_info({table_name})"
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a table (SQLite-specific implementation)."""
        try:
            # Get column information using PRAGMA
            column_result = self.execute_query(f"PRAGMA table_info({table_name})")
            
            # Check if table exists
            if column_result.is_empty:
                raise DatabaseError(f"Table '{table_name}' does not exist")
            
            # Transform PRAGMA result to match standard format
            columns = []
            for row in column_result.data.to_dict('records'):
                columns.append({
                    'column_name': row['name'],
                    'data_type': row['type'],
                    'is_nullable': 'YES' if row['notnull'] == 0 else 'NO',
                    'column_default': row['dflt_value'],
                    'ordinal_position': row['cid'] + 1,
                    'primary_key': row['pk'] == 1,
                })
            
            # Get row count
            try:
                count_result = self.execute_query(f"SELECT COUNT(*) as row_count FROM {table_name}")
                row_count = int(count_result.data.iloc[0]['row_count']) if not count_result.is_empty else 0
            except Exception:
                row_count = 0
            
            return {
                'table_name': table_name,
                'schema': schema,
                'columns': columns,
                'row_count': row_count,
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get table info for '{table_name}': {e}") from e
    
    def get_table_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of table names in the database."""
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        result = self.execute_query(query)
        return [row['name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_view_names(self, schema: Optional[str] = None) -> list[str]:
        """Get list of view names in the database."""
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view'
        ORDER BY name
        """
        result = self.execute_query(query)
        return [row['name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_index_names(self, table_name: Optional[str] = None) -> list[str]:
        """Get list of index names in the database."""
        if table_name:
            query = f"""
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND tbl_name = '{table_name}'
            ORDER BY name
            """
        else:
            query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        
        result = self.execute_query(query)
        return [row['name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def get_trigger_names(self, table_name: Optional[str] = None) -> list[str]:
        """Get list of trigger names in the database."""
        if table_name:
            query = f"""
            SELECT name
            FROM sqlite_master
            WHERE type = 'trigger' AND tbl_name = '{table_name}'
            ORDER BY name
            """
        else:
            query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'trigger'
            ORDER BY name
            """
        
        result = self.execute_query(query)
        return [row['name'] for row in result.data.to_dict('records')] if not result.is_empty else []
    
    def vacuum_database(self) -> Dict[str, Any]:
        """Run VACUUM to optimize the database."""
        try:
            self.execute_query("VACUUM", fetch_results=False)
            return {'status': 'success', 'message': 'Database vacuumed successfully'}
        except Exception as e:
            raise DatabaseError(f"Failed to vacuum database: {e}") from e
    
    def get_database_size(self) -> Dict[str, Any]:
        """Get database file size information."""
        try:
            db_path = Path(self.config.path)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
                
            if db_path.exists():
                size_bytes = db_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                
                return {
                    'path': str(db_path),
                    'size_bytes': size_bytes,
                    'size_mb': round(size_mb, 2),
                    'exists': True,
                }
            else:
                return {
                    'path': str(db_path),
                    'size_bytes': 0,
                    'size_mb': 0,
                    'exists': False,
                }
        except Exception as e:
            raise DatabaseError(f"Failed to get database size: {e}") from e
    
    def analyze_database(self) -> Dict[str, Any]:
        """Run ANALYZE to update statistics for the entire database."""
        try:
            self.execute_query("ANALYZE", fetch_results=False)
            return {'status': 'success', 'message': 'Database analyzed successfully'}
        except Exception as e:
            raise DatabaseError(f"Failed to analyze database: {e}") from e
