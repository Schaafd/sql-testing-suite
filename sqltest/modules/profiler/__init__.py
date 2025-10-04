"""Data profiler module for SQLTest Pro."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd

from ...db import AggregateSpec
from ...db.connection import ConnectionManager
from ...exceptions import ProfilingError, DatabaseError
from .analyzer import DataAnalyzer
from .models import ColumnStatistics, QueryProfile, TableProfile


logger = logging.getLogger(__name__)


NUMERIC_KEYWORDS = {
    "int",
    "integer",
    "real",
    "float",
    "double",
    "decimal",
    "number",
    "numeric",
}

DEFAULT_HISTOGRAM_BINS = 20


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
        self._default_chunk_size = max(sample_size // 2, 1000)

    def _collect_sample(
        self,
        result,
        sample_limit: Optional[int],
        count_total: bool = False,
    ) -> Tuple[pd.DataFrame, int, int]:
        """Collect a sample DataFrame from a streaming query result.

        Args:
            result: QueryResult returned from execute_query with streaming enabled.
            sample_limit: Maximum number of rows to collect for profiling.
            count_total: If True, consume the entire stream to count total rows.

        Returns:
            Tuple of (sample dataframe, rows collected, total rows processed).
            When count_total is False, total rows processed equals collected rows.
        """
        collected = 0
        total_rows = 0
        frames: List[pd.DataFrame] = []

        iterator = result.iter_chunks()
        try:
            for chunk in iterator:
                chunk_size = len(chunk)
                if chunk_size == 0:
                    continue

                total_rows += chunk_size

                if sample_limit is None or collected < sample_limit:
                    take = chunk_size if sample_limit is None else max(min(sample_limit - collected, chunk_size), 0)
                    if take > 0:
                        frames.append(chunk.iloc[:take].copy())
                        collected += take

                if not count_total and sample_limit is not None and collected >= sample_limit:
                    break
        finally:
            close_fn = getattr(iterator, 'close', None)
            if callable(close_fn):
                close_fn()

        sample_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if count_total:
            return sample_df, collected, total_rows

        # When not counting total rows, treat collected rows as total processed
        return sample_df, collected, collected

    def _compute_aggregates(
        self,
        *,
        table_name: str,
        schema_name: Optional[str],
        database_name: Optional[str],
        where_clause: Optional[str],
        table_info: Optional[Dict[str, Any]],
        columns: List[str],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics directly in the database."""
        try:
            column_specs = self._build_aggregate_specs(table_info, columns)
            if not column_specs:
                return {'row_count': 0, 'columns': {}}

            aggregates = self.connection_manager.compute_aggregates(
                table_name=table_name,
                specifications=column_specs,
                db_name=database_name,
                schema=schema_name,
                where_clause=where_clause,
            )
            return aggregates
        except Exception as exc:
            logger.warning(
                "Falling back to sampled statistics for %s due to aggregate error: %s",
                table_name,
                exc,
            )
            return {'row_count': 0, 'columns': {}}

    def _build_aggregate_specs(
        self,
        table_info: Optional[Dict[str, Any]],
        columns: List[str],
    ) -> List[AggregateSpec]:
        specs: List[AggregateSpec] = []
        column_metadata: Dict[str, Dict[str, Any]] = {}

        if table_info:
            for col in table_info.get('columns', []):
                name = col.get('column_name') or col.get('name')
                if name:
                    column_metadata[name] = col

        for column in columns:
            metadata = column_metadata.get(column, {})
            if not metadata:
                continue
            data_type = str(metadata.get('data_type') or metadata.get('type') or '').lower()

            operations = ['count', 'min', 'max']
            if self._is_numeric_type(data_type):
                operations.append('histogram')

            specs.append(
                AggregateSpec(
                    column=column,
                    operations=operations,
                    bins=DEFAULT_HISTOGRAM_BINS,
                )
            )

        return specs

    @staticmethod
    def _is_numeric_type(data_type: Optional[str]) -> bool:
        if not data_type:
            return False
        normalized = data_type.lower()
        return any(keyword in normalized for keyword in NUMERIC_KEYWORDS)
    
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
            ProfilingError: If profiling fails
            DatabaseError: If database operations fail
        """
        try:
            # Build query
            column_list = ", ".join(columns) if columns else "*"
            query = f"SELECT {column_list} FROM {table_name}"

            if where_clause:
                query += f" WHERE {where_clause}"

            if sample_rows:
                adapter = self.connection_manager.get_adapter(database_name)
                if adapter.get_driver_name() in {"sqlite", "psycopg2", "pymysql"}:
                    query += f" LIMIT {sample_rows}"
                else:
                    query += f" LIMIT {sample_rows}"

            sample_limit = sample_rows or self.analyzer.sample_size

            result = self.connection_manager.execute_query(
                query,
                db_name=database_name,
                stream_results=True,
                chunk_size=max(sample_limit // 2, 1000),
            )

            sample_df, collected_rows, _ = self._collect_sample(result, sample_limit)

            if sample_df.empty:
                raise ProfilingError(f"No data found in table '{table_name}'")

            db_name = database_name or self.connection_manager.config.default_database

            table_info = None
            try:
                table_info = self.connection_manager.get_table_info(
                    table_name,
                    schema_name,
                    db_name=database_name,
                )
            except Exception as exc:
                logger.warning("Failed to fetch table metadata for %s: %s", table_name, exc)

            aggregate_metadata = self._compute_aggregates(
                table_name=table_name,
                schema_name=schema_name,
                database_name=database_name,
                where_clause=where_clause,
                table_info=table_info,
                columns=sample_df.columns.tolist(),
            )

            total_rows = aggregate_metadata.get('row_count') or collected_rows

            profile = self.analyzer.profile_dataframe(
                df=sample_df,
                table_name=table_name,
                database_name=db_name,
                schema_name=schema_name,
                total_rows=total_rows,
                aggregate_metadata=aggregate_metadata.get('columns'),
            )

            if table_info:
                try:
                    primary_keys = []
                    for col_info in table_info.get('columns', []):
                        column_name = col_info.get('column_name') or col_info.get('name')
                        if not column_name:
                            continue
                        if col_info.get('primary_key') or col_info.get('column_key') == 'PRI':
                            primary_keys.append(column_name)
                    profile.primary_keys = primary_keys
                except Exception as exc:
                    logger.warning("Failed to enrich profile metadata for %s: %s", table_name, exc)

            return profile

        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilingError(f"Failed to profile table '{table_name}': {e}") from e
    
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
            ProfilingError: If profiling fails
            DatabaseError: If query execution fails
        """
        try:
            sample_limit = self.analyzer.sample_size
            result = self.connection_manager.execute_query(
                query,
                db_name=database_name,
                stream_results=True,
                chunk_size=max(sample_limit // 2, 1000),
            )

            sample_df, collected_rows, total_rows = self._collect_sample(
                result,
                sample_limit,
                count_total=True,
            )

            if total_rows == 0:
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

            if sample_df.empty:
                # No sample captured but rows existed (possible with large LIMIT settings)
                import hashlib
                query_hash = hashlib.md5(query.encode()).hexdigest()
                return QueryProfile(
                    query=query,
                    query_hash=query_hash,
                    execution_time=result.execution_time,
                    rows_returned=total_rows,
                    columns_returned=0,
                    profile_timestamp=datetime.now(),
                    columns={}
                )

            profile = self.analyzer.profile_query_result(
                df=sample_df,
                query=query,
                execution_time=result.execution_time,
                total_rows=total_rows,
            )

            return profile

        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilingError(f"Failed to profile query: {e}") from e
    
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
            ProfilingError: If profiling fails
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
            result = self.connection_manager.execute_query(query, db_name=database_name)
            
            if result.is_empty or column_name not in result.data.columns:
                raise ProfilingError(f"Column '{column_name}' not found in table '{table_name}'")
            
            # Analyze the specific column
            column_stats = self.analyzer.analyze_column(result.data[column_name], column_name)
            
            return column_stats
            
        except DatabaseError:
            raise
        except Exception as e:
            raise ProfilingError(f"Failed to profile column '{column_name}' in table '{table_name}': {e}") from e
    
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
            raise ProfilingError(f"Failed to get database profile summary: {e}") from e


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
