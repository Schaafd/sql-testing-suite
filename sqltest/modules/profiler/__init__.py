"""Data profiler module for SQLTest Pro."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...db.connection import ConnectionManager
from ...exceptions import ProfilerError, DatabaseError
from .analyzer import DataAnalyzer
from .models import TableProfile, QueryProfile, ColumnStatistics


class DataProfiler:
    """Main interface for data profiling operations."""
    
    def __init__(self, connection_manager: ConnectionManager, sample_size: int = 10000):
        """Initialize the profiler.
        
        Args:
            connection_manager: Database connection manager
            sample_size: Maximum rows to sample for analysis
        """
        self.connection_manager = connection_manager
        self.analyzer = DataAnalyzer(sample_size=sample_size)
    
    def profile_table(
        self, 
        table_name: str, 
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        sample_rows: Optional[int] = None
    ) -> TableProfile:
        """Profile a database table.
        
        Args:
            table_name: Name of the table to profile
            database_name: Database name (uses default if not provided)
            schema_name: Schema name (database-specific)
            columns: Specific columns to profile (all if not provided)
            where_clause: Optional WHERE clause to filter data
            sample_rows: Limit number of rows to analyze
            
        Returns:
            TableProfile with complete analysis
            
        Raises:
            ProfilerError: If profiling fails
            DatabaseError: If database operations fail
        """
        try:
            # Build query
            column_list = ", ".join(columns) if columns else "*"
            query = f"SELECT {column_list} FROM {table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
                
            if sample_rows:
                # Add database-specific LIMIT clause
                adapter = self.connection_manager.get_adapter(database_name)
                if adapter.get_driver_name() == "sqlite":
                    query += f" LIMIT {sample_rows}"
                elif adapter.get_driver_name() in ["psycopg2", "pymysql"]:
                    query += f" LIMIT {sample_rows}"
                else:
                    # For other databases, may need different syntax
                    query += f" LIMIT {sample_rows}"
            
            # Execute query and get results
            result = self.connection_manager.execute_query(query, database_name=database_name)
            
            if result.is_empty:
                raise ProfilerError(f"No data found in table '{table_name}'")
            
            # Profile the data
            db_name = database_name or self.connection_manager.config.default_database
            profile = self.analyzer.profile_dataframe(
                df=result.data,
                table_name=table_name,
                database_name=db_name,
                schema_name=schema_name
            )
            
            # Enhance with database metadata if available
            try:
                table_info = self.connection_manager.get_table_info(table_name, schema_name, database_name)
                
                # Extract primary key information from column metadata
                primary_keys = []
                for col_info in table_info.get('columns', []):
                    if col_info.get('primary_key') or col_info.get('column_key') == 'PRI':
                        primary_keys.append(col_info.get('column_name', col_info.get('name', '')))
                
                profile.primary_keys = primary_keys
                
            except Exception as e:
                # Metadata enhancement failed, but profiling succeeded
                print(f"Warning: Could not enhance profile with metadata: {e}")
            
            return profile
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilerError(f"Failed to profile table '{table_name}': {e}") from e
    
    def profile_query(
        self, 
        query: str, 
        database_name: Optional[str] = None
    ) -> QueryProfile:
        """Profile the results of a SQL query.
        
        Args:
            query: SQL query to execute and profile
            database_name: Database name (uses default if not provided)
            
        Returns:
            QueryProfile with analysis results
            
        Raises:
            ProfilerError: If profiling fails
            DatabaseError: If query execution fails
        """
        try:
            # Execute query and measure time
            result = self.connection_manager.execute_query(query, database_name=database_name)
            
            if result.is_empty:
                # Create empty profile for queries with no results
                import hashlib
                query_hash = hashlib.md5(query.encode()).hexdigest()
                return QueryProfile(
                    query=query,
                    query_hash=query_hash,
                    execution_time=result.execution_time,
                    rows_returned=0,
                    columns_returned=0,
                    profile_timestamp=datetime.now(),
                    columns={}
                )
            
            # Profile the query results
            profile = self.analyzer.profile_query_result(
                df=result.data,
                query=query,
                execution_time=result.execution_time
            )
            
            return profile
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilerError(f"Failed to profile query: {e}") from e
    
    def profile_column(
        self, 
        table_name: str, 
        column_name: str,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        sample_rows: Optional[int] = None
    ) -> ColumnStatistics:
        """Profile a specific column in detail.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to profile
            database_name: Database name (uses default if not provided)
            schema_name: Schema name (database-specific)
            sample_rows: Limit number of rows to analyze
            
        Returns:
            ColumnStatistics with detailed analysis
            
        Raises:
            ProfilerError: If profiling fails
        """
        try:
            # Build query for single column
            query = f"SELECT {column_name} FROM {table_name}"
            
            if sample_rows:
                adapter = self.connection_manager.get_adapter(database_name)
                if adapter.get_driver_name() == "sqlite":
                    query += f" LIMIT {sample_rows}"
                elif adapter.get_driver_name() in ["psycopg2", "pymysql"]:
                    query += f" LIMIT {sample_rows}"
                else:
                    query += f" LIMIT {sample_rows}"
            
            # Execute query and get results
            result = self.connection_manager.execute_query(query, database_name=database_name)
            
            if result.is_empty or column_name not in result.data.columns:
                raise ProfilerError(f"Column '{column_name}' not found in table '{table_name}'")
            
            # Analyze the specific column
            column_stats = self.analyzer.analyze_column(result.data[column_name], column_name)
            
            return column_stats
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilerError(f"Failed to profile column '{column_name}' in table '{table_name}': {e}") from e
    
    def get_database_profile_summary(
        self, 
        database_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get a summary profile of an entire database.
        
        Args:
            database_name: Database name (uses default if not provided)
            
        Returns:
            Dictionary with database-level statistics
        """
        try:
            adapter = self.connection_manager.get_adapter(database_name)
            
            # Get list of tables
            tables = adapter.get_table_names()
            
            summary = {
                'database_name': database_name or self.connection_manager.config.default_database,
                'total_tables': len(tables),
                'tables': [],
                'analysis_timestamp': datetime.now()
            }
            
            # Get basic info for each table
            for table_name in tables[:20]:  # Limit to first 20 tables for performance
                try:
                    table_info = self.connection_manager.get_table_info(table_name, database_name=database_name)
                    summary['tables'].append({
                        'table_name': table_name,
                        'row_count': table_info.get('row_count', 0),
                        'column_count': len(table_info.get('columns', []))
                    })
                except Exception as e:
                    print(f"Warning: Could not get info for table '{table_name}': {e}")
                    continue
            
            return summary
            
        except Exception as e:
            raise ProfilerError(f"Failed to get database profile summary: {e}") from e


# Convenience functions for direct use
def profile_table(
    connection_manager: ConnectionManager,
    table_name: str,
    database_name: Optional[str] = None,
    **kwargs
) -> TableProfile:
    """Convenience function to profile a table."""
    profiler = DataProfiler(connection_manager)
    return profiler.profile_table(table_name, database_name, **kwargs)


def profile_query(
    connection_manager: ConnectionManager,
    query: str,
    database_name: Optional[str] = None
) -> QueryProfile:
    """Convenience function to profile a query."""
    profiler = DataProfiler(connection_manager)
    return profiler.profile_query(query, database_name)


# Export main classes and functions
__all__ = [
    'DataProfiler',
    'TableProfile', 
    'QueryProfile',
    'ColumnStatistics',
    'profile_table',
    'profile_query'
]
