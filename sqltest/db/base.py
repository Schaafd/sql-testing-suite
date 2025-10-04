"""Base database adapter and connection management."""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

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
        chunk_generator: Optional[Callable[[], Iterator[pd.DataFrame]]] = None,
    ) -> None:
        """Initialize query result.

        Args:
            data: Result data as DataFrame.
            rows_affected: Number of rows affected by query.
            execution_time: Query execution time in seconds.
            columns: Column names for the result.
            chunk_generator: Callable returning an iterator that yields PD DataFrames
                in chunks when streaming is requested.
        """
        self.data = data
        self.rows_affected = rows_affected or 0
        self.execution_time = execution_time or 0.0
        self.columns = columns or []
        self._chunk_generator = chunk_generator

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

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Iterate over result rows in chunks when streaming is enabled.

        Yields:
            DataFrame chunks respecting the configured chunk size.

        Notes:
            If streaming is not enabled, this yields the full DataFrame once.
        """
        if self._chunk_generator is None:
            if self.data is not None:
                yield self.data
            return

        iterator = self._chunk_generator()
        for chunk in iterator:
            if not self.columns:
                self.columns = list(chunk.columns)
            yield chunk


@dataclass(frozen=True)
class AggregateSpec:
    """Defines aggregate operations for a specific column."""

    column: str
    operations: List[str]
    bins: int = 10


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
        chunk_size: Optional[int] = None,
        stream_results: bool = False,
        result_processor: Optional[Callable[[pd.DataFrame], Any]] = None,
    ) -> QueryResult:
        """Execute SQL query and return results.

        Args:
            query: SQL query string.
            params: Query parameters.
            fetch_results: Whether to fetch result data.
            timeout: Query timeout in seconds.
            chunk_size: Number of rows per chunk when streaming.
            stream_results: Stream results instead of loading entire dataset.
            result_processor: Optional callable invoked for each chunk.

        Returns:
            QueryResult instance.

        Raises:
            DatabaseError: If query execution fails.
        """
        start_time = time.time()

        try:
            if fetch_results and (stream_results or chunk_size or result_processor):
                effective_chunk_size = max(1, chunk_size or 10000)

                def chunk_generator() -> Iterator[pd.DataFrame]:
                    with self.get_connection() as conn:
                        exec_conn = conn
                        if timeout:
                            exec_conn = exec_conn.execution_options(autocommit=False, isolation_level="AUTOCOMMIT")

                        exec_conn = exec_conn.execution_options(stream_results=True)

                        if params:
                            cursor = exec_conn.execute(text(query), params)
                        else:
                            cursor = exec_conn.execute(text(query))

                        try:
                            columns = list(cursor.keys())
                            while True:
                                rows = cursor.fetchmany(effective_chunk_size)
                                if not rows:
                                    break
                                yield pd.DataFrame(rows, columns=columns)
                        finally:
                            cursor.close()

                if result_processor is not None:
                    processed_rows = 0
                    for chunk in chunk_generator():
                        processed_rows += len(chunk)
                        result_processor(chunk)
                    execution_time = time.time() - start_time
                    return QueryResult(
                        rows_affected=processed_rows,
                        execution_time=execution_time,
                    )

                execution_time = time.time() - start_time
                return QueryResult(
                    execution_time=execution_time,
                    chunk_generator=chunk_generator,
                )

            with self.get_connection() as conn:
                if timeout:
                    conn = conn.execution_options(autocommit=False, isolation_level="AUTOCOMMIT")

                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                execution_time = time.time() - start_time

                if result.returns_rows and fetch_results:
                    rows = result.fetchall()
                    columns = list(result.keys())
                    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
                    return QueryResult(
                        data=df,
                        rows_affected=len(rows),
                        execution_time=execution_time,
                        columns=columns,
                    )

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

    def compute_aggregates(
        self,
        table_name: str,
        specifications: List[AggregateSpec],
        schema: Optional[str] = None,
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute aggregate operations directly in the database.

        Args:
            table_name: Target table name.
            specifications: List of aggregate specifications per column.
            schema: Optional schema for the table.
            where_clause: Optional SQL filter applied to the aggregations.

        Returns:
            Dictionary containing computed aggregates and metadata.
        """
        if not specifications:
            raise DatabaseError("At least one aggregate specification is required")

        table_ref = self._build_table_reference(table_name, schema)
        base_filter = where_clause
        filter_clause = self._compose_where(base_filter)

        select_clauses = ["COUNT(*) AS total_row_count"]
        alias_map: Dict[Tuple[str, str], str] = {}

        for spec in specifications:
            operations = {op.lower() for op in spec.operations}
            column_alias = self._normalize_alias(spec.column)
            column_expr = self._quote_identifier(spec.column)

            if 'count' in operations:
                alias = f"{column_alias}_count"
                alias_map[(spec.column, 'count')] = alias
                select_clauses.append(f"COUNT({column_expr}) AS {alias}")

            if 'min' in operations:
                alias = f"{column_alias}_min"
                alias_map[(spec.column, 'min')] = alias
                select_clauses.append(f"MIN({column_expr}) AS {alias}")

            if 'max' in operations:
                alias = f"{column_alias}_max"
                alias_map[(spec.column, 'max')] = alias
                select_clauses.append(f"MAX({column_expr}) AS {alias}")

        aggregate_sql = f"SELECT {', '.join(select_clauses)} FROM {table_ref}{filter_clause}"

        aggregate_result = self.execute_query(aggregate_sql, fetch_results=True)
        if aggregate_result.is_empty:
            return {'row_count': 0, 'columns': {}}

        row = aggregate_result.data.iloc[0]
        output: Dict[str, Any] = {
            'row_count': int(row['total_row_count']) if 'total_row_count' in row else 0,
            'columns': {},
        }

        for spec in specifications:
            operations = {op.lower() for op in spec.operations}
            column_result: Dict[str, Any] = {}
            column_expr = self._quote_identifier(spec.column)

            if (spec.column, 'count') in alias_map:
                alias = alias_map[(spec.column, 'count')]
                column_result['count'] = int(row.get(alias, 0))

            min_value = None
            max_value = None

            if (spec.column, 'min') in alias_map:
                alias = alias_map[(spec.column, 'min')]
                min_value = row.get(alias)
                column_result['min'] = min_value

            if (spec.column, 'max') in alias_map:
                alias = alias_map[(spec.column, 'max')]
                max_value = row.get(alias)
                column_result['max'] = max_value

            if 'histogram' in operations:
                if min_value is None or max_value is None:
                    min_value, max_value = self._fetch_min_max(table_ref, column_expr, base_filter)
                    if min_value is not None:
                        column_result.setdefault('min', min_value)
                    if max_value is not None:
                        column_result.setdefault('max', max_value)

                histogram = self._build_histogram(
                    table_ref=table_ref,
                    column_expr=column_expr,
                    min_value=min_value,
                    max_value=max_value,
                    bins=max(1, spec.bins),
                    base_filter=base_filter,
                )
                column_result['histogram'] = histogram

            output['columns'][spec.column] = column_result

        return output

    def _fetch_min_max(
        self,
        table_ref: str,
        column_expr: str,
        base_filter: Optional[str],
    ) -> Tuple[Optional[float], Optional[float]]:
        query = (
            f"SELECT MIN({column_expr}) AS min_value, MAX({column_expr}) AS max_value "
            f"FROM {table_ref}{self._compose_where(base_filter)}"
        )
        result = self.execute_query(query, fetch_results=True)
        if result.is_empty:
            return None, None

        row = result.data.iloc[0]
        return row.get('min_value'), row.get('max_value')

    def _build_histogram(
        self,
        table_ref: str,
        column_expr: str,
        min_value: Optional[float],
        max_value: Optional[float],
        bins: int,
        base_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        if min_value is None or max_value is None:
            return []

        if min_value == max_value:
            where_clause = self._compose_where(base_filter, f"{column_expr} IS NOT NULL")
            query = (
                f"SELECT 0 AS bucket, COUNT(*) AS bucket_count FROM {table_ref}{where_clause}"
            )
            result = self.execute_query(query, fetch_results=True)
            count = 0 if result.is_empty else int(result.data.iloc[0].get('bucket_count', 0))
            return [{
                'bin': 0,
                'bin_start': min_value,
                'bin_end': max_value,
                'count': count,
            }]

        span = max_value - min_value
        if span == 0:
            span = 1
        bin_width = span / bins

        bucket_expression = (
            f"CASE "
            f"WHEN {column_expr} IS NULL THEN -1 "
            f"WHEN {column_expr} = {self._format_numeric(max_value)} THEN {bins - 1} "
            f"ELSE CAST(({column_expr} - {self._format_numeric(min_value)}) / {self._format_numeric(bin_width)} AS INTEGER) "
            f"END"
        )

        subquery_filter = self._compose_where(base_filter, f"{column_expr} IS NOT NULL")
        histogram_query = (
            f"SELECT bucket, COUNT(*) AS bucket_count FROM ("
            f"SELECT {bucket_expression} AS bucket FROM {table_ref}{subquery_filter}"
            f") agg GROUP BY bucket ORDER BY bucket"
        )

        result = self.execute_query(histogram_query, fetch_results=True)
        if result.is_empty:
            return []

        histogram_data: List[Dict[str, Any]] = []
        for _, row in result.data.iterrows():
            bucket_index = int(row['bucket'])
            if bucket_index < 0:
                continue
            bin_start = min_value + (bucket_index * bin_width)
            bin_end = bin_start + bin_width
            histogram_data.append({
                'bin': bucket_index,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'count': int(row['bucket_count']),
            })

        return histogram_data

    @staticmethod
    def _format_numeric(value: float) -> str:
        if isinstance(value, int):
            return str(value)
        return format(value, '.10g')

    @staticmethod
    def _normalize_alias(identifier: str) -> str:
        normalized = ''.join(ch if ch.isalnum() else '_' for ch in identifier)
        if not normalized:
            raise DatabaseError("Invalid identifier for alias generation")
        return normalized.lower()

    @staticmethod
    def _build_table_reference(table_name: str, schema: Optional[str]) -> str:
        table = BaseAdapter._quote_identifier(table_name)
        if schema:
            schema_name = BaseAdapter._quote_identifier(schema)
            return f"{schema_name}.{table}"
        return table

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        if not identifier:
            raise DatabaseError("Identifier cannot be empty")
        if not identifier[0].isalpha() and identifier[0] != '_':
            raise DatabaseError(f"Invalid identifier: {identifier}")
        for char in identifier[1:]:
            if not (char.isalnum() or char == '_'):
                raise DatabaseError(f"Invalid identifier: {identifier}")
        return f'"{identifier}"'

    @staticmethod
    def _compose_where(base_filter: Optional[str], extra_condition: Optional[str] = None) -> str:
        clauses = []
        if base_filter:
            clauses.append(f"({base_filter})")
        if extra_condition:
            clauses.append(extra_condition)
        if not clauses:
            return ""
        return " WHERE " + " AND ".join(clauses)

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
