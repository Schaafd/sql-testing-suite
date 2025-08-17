"""Base database adapter and connection management."""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError

from sqltest.config.models import DatabaseConfig, ConnectionPoolConfig
from sqltest.exceptions import DatabaseError


class QueryResult:
    """Container for query results with metadata."""
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        rows_affected: Optional[int] = None,
        execution_time: Optional[float] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize query result.
        
        Args:
            data: Result data as DataFrame.
            rows_affected: Number of rows affected by query.
            execution_time: Query execution time in seconds.
            columns: Column names for the result.
        """
        self.data = data
        self.rows_affected = rows_affected or 0
        self.execution_time = execution_time or 0.0
        self.columns = columns or []
        
    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return self.data is None or self.data.empty
        
    @property
    def row_count(self) -> int:
        """Get number of rows in result."""
        return len(self.data) if self.data is not None else 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'data': self.data.to_dict('records') if self.data is not None else [],
            'rows_affected': self.rows_affected,
            'execution_time': self.execution_time,
            'columns': self.columns,
            'row_count': self.row_count,
            'is_empty': self.is_empty,
        }


class BaseAdapter(ABC):
    """Base class for database adapters."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        pool_config: Optional[ConnectionPoolConfig] = None,
    ) -> None:
        """Initialize database adapter.
        
        Args:
            config: Database configuration.
            pool_config: Connection pool configuration.
        """
        self.config = config
        self.pool_config = pool_config or ConnectionPoolConfig()
        self._engine: Optional[Engine] = None
        
    @abstractmethod
    def build_connection_string(self) -> str:
        """Build database connection string.
        
        Returns:
            Database connection string.
        """
        pass
    
    @abstractmethod
    def get_driver_name(self) -> str:
        """Get the driver name for this adapter.
        
        Returns:
            Driver name string.
        """
        pass
    
    def get_engine(self) -> Engine:
        """Get or create SQLAlchemy engine.
        
        Returns:
            SQLAlchemy engine instance.
            
        Raises:
            DatabaseError: If engine creation fails.
        """
        if self._engine is None:
            try:
                connection_string = self.build_connection_string()
                
                # Build engine arguments
                engine_args = {
                    'pool_size': self.pool_config.max_connections,
                    'max_overflow': 0,  # Don't allow overflow beyond pool_size
                    'pool_timeout': self.pool_config.timeout,
                    'pool_recycle': 3600,  # Recycle connections after 1 hour
                    'pool_pre_ping': True,  # Validate connections before use
                    'echo': False,  # Set to True for SQL debugging
                }
                
                # Add database-specific options
                engine_args.update(self._get_engine_options())
                
                self._engine = create_engine(connection_string, **engine_args)
                
            except Exception as e:
                raise DatabaseError(f"Failed to create database engine: {e}") from e
                
        return self._engine
    
    def _get_engine_options(self) -> Dict[str, Any]:
        """Get database-specific engine options.
        
        Returns:
            Dictionary of engine options.
        """
        return {}
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Get database connection with automatic cleanup.
        
        Yields:
            SQLAlchemy connection.
            
        Raises:
            DatabaseError: If connection fails.
        """
        engine = self.get_engine()
        connection = None
        
        try:
            connection = engine.connect()
            yield connection
            connection.commit()  # Ensure transactions are committed
            
        except SQLAlchemyError as e:
            if connection:
                try:
                    connection.rollback()
                except Exception:
                    pass  # Ignore rollback errors
            raise DatabaseError(f"Database connection error: {e}") from e
            
        except Exception as e:
            if connection:
                try:
                    connection.rollback()
                except Exception:
                    pass  # Ignore rollback errors
            raise DatabaseError(f"Unexpected database error: {e}") from e
            
        finally:
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass  # Ignore close errors
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful.
            
        Raises:
            DatabaseError: If connection test fails.
        """
        try:
            with self.get_connection() as conn:
                # Execute a simple query to test connection
                result = conn.execute(text("SELECT 1 as test"))
                result.fetchone()
                return True
                
        except Exception as e:
            raise DatabaseError(f"Connection test failed: {e}") from e
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True,
        timeout: Optional[int] = None,
    ) -> QueryResult:
        """Execute SQL query and return results.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            fetch_results: Whether to fetch result data.
            timeout: Query timeout in seconds.
            
        Returns:
            QueryResult instance.
            
        Raises:
            DatabaseError: If query execution fails.
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                # Set query timeout if specified
                if timeout:
                    conn = conn.execution_options(autocommit=False, isolation_level="AUTOCOMMIT")
                
                # Execute query with parameters
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                execution_time = time.time() - start_time
                
                # Handle different query types
                if result.returns_rows and fetch_results:
                    # SELECT query - fetch data
                    rows = result.fetchall()
                    columns = list(result.keys())
                    
                    # Debug: print what we got
                    # print(f"Debug - Rows: {rows}, Columns: {columns}")
                    
                    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
                    return QueryResult(
                        data=df,
                        rows_affected=len(rows),
                        execution_time=execution_time,
                        columns=columns,
                    )
                else:
                    # INSERT/UPDATE/DELETE/DDL query - get affected rows
                    rows_affected = result.rowcount if result.rowcount >= 0 else 0
                    return QueryResult(
                        rows_affected=rows_affected,
                        execution_time=execution_time,
                    )
                    
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            raise DatabaseError(
                f"Query execution failed after {execution_time:.2f}s: {e}"
            ) from e
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise DatabaseError(
                f"Unexpected error during query execution after {execution_time:.2f}s: {e}"
            ) from e
    
    def execute_script(
        self,
        script: str,
        delimiter: str = ";",
        continue_on_error: bool = False,
    ) -> List[QueryResult]:
        """Execute SQL script with multiple statements.
        
        Args:
            script: SQL script string.
            delimiter: Statement delimiter.
            continue_on_error: Continue execution if a statement fails.
            
        Returns:
            List of QueryResult instances.
            
        Raises:
            DatabaseError: If script execution fails and continue_on_error is False.
        """
        statements = [stmt.strip() for stmt in script.split(delimiter) if stmt.strip()]
        results = []
        
        for i, statement in enumerate(statements):
            try:
                result = self.execute_query(statement, fetch_results=False)
                results.append(result)
                
            except DatabaseError as e:
                if not continue_on_error:
                    raise DatabaseError(
                        f"Script execution failed at statement {i + 1}: {e}"
                    ) from e
                else:
                    # Create error result
                    results.append(QueryResult(
                        rows_affected=0,
                        execution_time=0.0,
                    ))
                    
        return results
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a table.
        
        Args:
            table_name: Name of the table.
            schema: Schema name (optional).
            
        Returns:
            Dictionary with table information.
            
        Raises:
            DatabaseError: If table info retrieval fails.
        """
        try:
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            # Get column information
            column_query = self._get_column_info_query(table_name, schema)
            column_result = self.execute_query(column_query)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {full_table_name}"
            count_result = self.execute_query(count_query)
            
            return {
                'table_name': table_name,
                'schema': schema,
                'columns': column_result.data.to_dict('records') if column_result.data is not None else [],
                'row_count': count_result.data.iloc[0]['row_count'] if not count_result.is_empty else 0,
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get table info for '{table_name}': {e}") from e
    
    @abstractmethod
    def _get_column_info_query(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get database-specific query for column information.
        
        Args:
            table_name: Name of the table.
            schema: Schema name (optional).
            
        Returns:
            SQL query string.
        """
        pass
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
