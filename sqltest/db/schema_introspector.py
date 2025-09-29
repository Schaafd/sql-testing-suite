"""Automatic database schema introspection and metadata discovery."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sqlalchemy import inspect, MetaData, Table, Column
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DatabaseObjectType(str, Enum):
    """Types of database objects."""
    TABLE = "table"
    VIEW = "view"
    INDEX = "index"
    CONSTRAINT = "constraint"
    FUNCTION = "function"
    PROCEDURE = "procedure"
    TRIGGER = "trigger"
    SEQUENCE = "sequence"


class ColumnDataType(str, Enum):
    """Standardized column data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DECIMAL = "decimal"
    BINARY = "binary"
    JSON = "json"
    ARRAY = "array"
    UUID = "uuid"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: ColumnDataType
    native_type: str
    is_nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    default_value: Optional[Any] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    foreign_key_references: Optional[Tuple[str, str]] = None  # (table, column)
    comment: Optional[str] = None


@dataclass
class IndexInfo:
    """Information about a database index."""
    name: str
    table_name: str
    column_names: List[str]
    is_unique: bool
    is_primary: bool = False
    is_partial: bool = False
    index_type: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class ConstraintInfo:
    """Information about a database constraint."""
    name: str
    table_name: str
    constraint_type: str  # PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK
    column_names: List[str]
    referenced_table: Optional[str] = None
    referenced_columns: Optional[List[str]] = None
    check_condition: Optional[str] = None


@dataclass
class TableInfo:
    """Comprehensive information about a database table."""
    name: str
    schema_name: Optional[str]
    table_type: str  # TABLE, VIEW, etc.
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    indexes: Dict[str, IndexInfo] = field(default_factory=dict)
    constraints: Dict[str, ConstraintInfo] = field(default_factory=dict)
    row_count: Optional[int] = None
    table_size_bytes: Optional[int] = None
    comment: Optional[str] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)  # Tables this depends on
    dependents: Set[str] = field(default_factory=set)    # Tables that depend on this


@dataclass
class SchemaInfo:
    """Information about a database schema."""
    name: str
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    views: Dict[str, TableInfo] = field(default_factory=dict)
    functions: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    sequences: List[str] = field(default_factory=list)
    total_tables: int = 0
    total_views: int = 0
    total_indexes: int = 0
    total_constraints: int = 0


@dataclass
class DatabaseSchema:
    """Complete database schema information."""
    database_name: str
    database_type: str
    schemas: Dict[str, SchemaInfo] = field(default_factory=dict)
    introspection_timestamp: datetime = field(default_factory=datetime.now)
    introspection_duration_ms: float = 0.0
    total_objects: int = 0

    def get_all_tables(self) -> Dict[str, TableInfo]:
        """Get all tables across all schemas."""
        all_tables = {}
        for schema in self.schemas.values():
            for table_name, table_info in schema.tables.items():
                full_name = f"{schema.name}.{table_name}" if schema.name else table_name
                all_tables[full_name] = table_info
        return all_tables

    def get_all_views(self) -> Dict[str, TableInfo]:
        """Get all views across all schemas."""
        all_views = {}
        for schema in self.schemas.values():
            for view_name, view_info in schema.views.items():
                full_name = f"{schema.name}.{view_name}" if schema.name else view_name
                all_views[full_name] = view_info
        return all_views


class SchemaIntrospector:
    """Automatic database schema introspection and caching."""

    def __init__(self, cache_duration_minutes: int = 60):
        """Initialize schema introspector.

        Args:
            cache_duration_minutes: Duration to cache schema information
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._lock = Lock()
        self._schema_cache: Dict[str, Tuple[DatabaseSchema, datetime]] = {}

    def introspect_database(self, engine: Engine, database_name: str,
                          force_refresh: bool = False) -> DatabaseSchema:
        """Introspect database schema and return comprehensive information.

        Args:
            engine: SQLAlchemy engine connected to the database
            database_name: Name of the database
            force_refresh: Force refresh even if cached data exists

        Returns:
            DatabaseSchema with complete schema information
        """
        cache_key = f"{database_name}_{engine.url.host}_{engine.url.port}"

        # Check cache first
        if not force_refresh:
            with self._lock:
                if cache_key in self._schema_cache:
                    cached_schema, cache_time = self._schema_cache[cache_key]
                    if datetime.now() - cache_time < self.cache_duration:
                        logger.debug(f"Returning cached schema for {database_name}")
                        return cached_schema

        # Perform introspection
        start_time = time.perf_counter()
        logger.info(f"Starting schema introspection for database: {database_name}")

        try:
            schema = self._perform_introspection(engine, database_name)
            introspection_time = (time.perf_counter() - start_time) * 1000
            schema.introspection_duration_ms = introspection_time

            # Cache the result
            with self._lock:
                self._schema_cache[cache_key] = (schema, datetime.now())

            logger.info(f"Schema introspection completed in {introspection_time:.2f}ms "
                       f"({schema.total_objects} objects discovered)")

            return schema

        except Exception as e:
            logger.error(f"Schema introspection failed for {database_name}: {e}")
            raise

    def _perform_introspection(self, engine: Engine, database_name: str) -> DatabaseSchema:
        """Perform the actual schema introspection."""
        inspector = inspect(engine)

        # Determine database type
        database_type = engine.dialect.name

        schema = DatabaseSchema(
            database_name=database_name,
            database_type=database_type
        )

        # Get all schema names
        try:
            schema_names = inspector.get_schema_names()
        except (SQLAlchemyError, NotImplementedError):
            # Some databases don't support schemas or the method isn't implemented
            schema_names = [None]  # Use default schema

        total_objects = 0

        for schema_name in schema_names:
            logger.debug(f"Introspecting schema: {schema_name or 'default'}")

            schema_info = SchemaInfo(name=schema_name or "default")

            try:
                # Get tables
                table_names = inspector.get_table_names(schema=schema_name)
                for table_name in table_names:
                    table_info = self._introspect_table(inspector, table_name, schema_name)
                    schema_info.tables[table_name] = table_info
                    total_objects += 1

                schema_info.total_tables = len(table_names)

                # Get views
                try:
                    view_names = inspector.get_view_names(schema=schema_name)
                    for view_name in view_names:
                        view_info = self._introspect_view(inspector, view_name, schema_name)
                        schema_info.views[view_name] = view_info
                        total_objects += 1

                    schema_info.total_views = len(view_names)
                except (SQLAlchemyError, NotImplementedError):
                    logger.debug("Views introspection not supported or failed")

                # Get sequences (if supported)
                try:
                    sequences = inspector.get_sequence_names(schema=schema_name)
                    schema_info.sequences = sequences
                    total_objects += len(sequences)
                except (SQLAlchemyError, NotImplementedError):
                    logger.debug("Sequences introspection not supported")

                # Calculate totals
                for table_info in schema_info.tables.values():
                    schema_info.total_indexes += len(table_info.indexes)
                    schema_info.total_constraints += len(table_info.constraints)

                schema.schemas[schema_name or "default"] = schema_info

            except Exception as e:
                logger.error(f"Failed to introspect schema {schema_name}: {e}")
                continue

        # Build dependency relationships
        self._build_table_dependencies(schema)

        schema.total_objects = total_objects
        return schema

    def _introspect_table(self, inspector, table_name: str, schema_name: Optional[str]) -> TableInfo:
        """Introspect a single table."""
        table_info = TableInfo(
            name=table_name,
            schema_name=schema_name,
            table_type="TABLE"
        )

        try:
            # Get columns
            columns = inspector.get_columns(table_name, schema=schema_name)
            for col_data in columns:
                column_info = self._convert_column_info(col_data)
                table_info.columns[column_info.name] = column_info

            # Get primary keys
            try:
                pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name)
                if pk_constraint and pk_constraint.get('constrained_columns'):
                    pk_columns = pk_constraint['constrained_columns']
                    for col_name in pk_columns:
                        if col_name in table_info.columns:
                            table_info.columns[col_name].is_primary_key = True

                    # Add primary key constraint
                    constraint_info = ConstraintInfo(
                        name=pk_constraint.get('name', f"{table_name}_pkey"),
                        table_name=table_name,
                        constraint_type="PRIMARY KEY",
                        column_names=pk_columns
                    )
                    table_info.constraints[constraint_info.name] = constraint_info

            except (SQLAlchemyError, NotImplementedError):
                logger.debug(f"Primary key introspection failed for {table_name}")

            # Get foreign keys
            try:
                foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
                for fk in foreign_keys:
                    constrained_columns = fk.get('constrained_columns', [])
                    referred_table = fk.get('referred_table')
                    referred_columns = fk.get('referred_columns', [])

                    # Mark columns as foreign keys
                    for col_name in constrained_columns:
                        if col_name in table_info.columns:
                            table_info.columns[col_name].is_foreign_key = True
                            if referred_table and referred_columns:
                                table_info.columns[col_name].foreign_key_references = (
                                    referred_table, referred_columns[0] if referred_columns else None
                                )

                    # Add foreign key constraint
                    constraint_info = ConstraintInfo(
                        name=fk.get('name', f"{table_name}_{constrained_columns[0]}_fkey"),
                        table_name=table_name,
                        constraint_type="FOREIGN KEY",
                        column_names=constrained_columns,
                        referenced_table=referred_table,
                        referenced_columns=referred_columns
                    )
                    table_info.constraints[constraint_info.name] = constraint_info

            except (SQLAlchemyError, NotImplementedError):
                logger.debug(f"Foreign key introspection failed for {table_name}")

            # Get indexes
            try:
                indexes = inspector.get_indexes(table_name, schema=schema_name)
                for idx in indexes:
                    index_info = IndexInfo(
                        name=idx['name'],
                        table_name=table_name,
                        column_names=idx['column_names'],
                        is_unique=idx.get('unique', False)
                    )
                    table_info.indexes[index_info.name] = index_info

                    # Mark columns as indexed
                    for col_name in idx['column_names']:
                        if col_name in table_info.columns:
                            table_info.columns[col_name].is_indexed = True

            except (SQLAlchemyError, NotImplementedError):
                logger.debug(f"Index introspection failed for {table_name}")

            # Get unique constraints
            try:
                unique_constraints = inspector.get_unique_constraints(table_name, schema=schema_name)
                for uc in unique_constraints:
                    constrained_columns = uc.get('column_names', [])

                    # Mark columns as unique
                    for col_name in constrained_columns:
                        if col_name in table_info.columns:
                            table_info.columns[col_name].is_unique = True

                    # Add unique constraint
                    constraint_info = ConstraintInfo(
                        name=uc.get('name', f"{table_name}_{'_'.join(constrained_columns)}_unique"),
                        table_name=table_name,
                        constraint_type="UNIQUE",
                        column_names=constrained_columns
                    )
                    table_info.constraints[constraint_info.name] = constraint_info

            except (SQLAlchemyError, NotImplementedError):
                logger.debug(f"Unique constraint introspection failed for {table_name}")

            # Get check constraints
            try:
                check_constraints = inspector.get_check_constraints(table_name, schema=schema_name)
                for cc in check_constraints:
                    constraint_info = ConstraintInfo(
                        name=cc.get('name', f"{table_name}_check"),
                        table_name=table_name,
                        constraint_type="CHECK",
                        column_names=[],  # Check constraints may not have specific columns
                        check_condition=cc.get('sqltext')
                    )
                    table_info.constraints[constraint_info.name] = constraint_info

            except (SQLAlchemyError, NotImplementedError):
                logger.debug(f"Check constraint introspection failed for {table_name}")

            # Get table comment
            try:
                table_comment = inspector.get_table_comment(table_name, schema=schema_name)
                if table_comment and table_comment.get('text'):
                    table_info.comment = table_comment['text']
            except (SQLAlchemyError, NotImplementedError):
                pass

        except Exception as e:
            logger.error(f"Failed to introspect table {table_name}: {e}")

        return table_info

    def _introspect_view(self, inspector, view_name: str, schema_name: Optional[str]) -> TableInfo:
        """Introspect a database view."""
        view_info = TableInfo(
            name=view_name,
            schema_name=schema_name,
            table_type="VIEW"
        )

        try:
            # Get view columns
            columns = inspector.get_columns(view_name, schema=schema_name)
            for col_data in columns:
                column_info = self._convert_column_info(col_data)
                view_info.columns[column_info.name] = column_info

            # Get view definition (if supported)
            try:
                view_definition = inspector.get_view_definition(view_name, schema=schema_name)
                if view_definition:
                    view_info.comment = f"View Definition: {view_definition[:500]}..."
            except (SQLAlchemyError, NotImplementedError):
                pass

        except Exception as e:
            logger.error(f"Failed to introspect view {view_name}: {e}")

        return view_info

    def _convert_column_info(self, col_data: Dict[str, Any]) -> ColumnInfo:
        """Convert SQLAlchemy column data to ColumnInfo."""
        # Map SQLAlchemy types to standardized types
        native_type = str(col_data['type'])
        data_type = self._map_data_type(native_type)

        column_info = ColumnInfo(
            name=col_data['name'],
            data_type=data_type,
            native_type=native_type,
            is_nullable=col_data.get('nullable', True),
            default_value=col_data.get('default')
        )

        # Extract type-specific information
        col_type = col_data['type']
        if hasattr(col_type, 'length') and col_type.length is not None:
            column_info.max_length = col_type.length

        if hasattr(col_type, 'precision') and col_type.precision is not None:
            column_info.precision = col_type.precision

        if hasattr(col_type, 'scale') and col_type.scale is not None:
            column_info.scale = col_type.scale

        # Get column comment if available
        if 'comment' in col_data and col_data['comment']:
            column_info.comment = col_data['comment']

        return column_info

    def _map_data_type(self, native_type: str) -> ColumnDataType:
        """Map native database types to standardized types."""
        native_lower = native_type.lower()

        # String types
        if any(t in native_lower for t in ['varchar', 'char', 'text', 'string', 'nvarchar', 'nchar']):
            if 'text' in native_lower:
                return ColumnDataType.TEXT
            return ColumnDataType.STRING

        # Integer types
        if any(t in native_lower for t in ['int', 'integer', 'bigint', 'smallint', 'tinyint']):
            return ColumnDataType.INTEGER

        # Float types
        if any(t in native_lower for t in ['float', 'double', 'real']):
            return ColumnDataType.FLOAT

        # Decimal/Numeric types
        if any(t in native_lower for t in ['decimal', 'numeric', 'money']):
            return ColumnDataType.DECIMAL

        # Boolean types
        if any(t in native_lower for t in ['bool', 'boolean', 'bit']):
            return ColumnDataType.BOOLEAN

        # Date/Time types
        if 'datetime' in native_lower or 'timestamp' in native_lower:
            return ColumnDataType.DATETIME
        if 'date' in native_lower:
            return ColumnDataType.DATE
        if 'time' in native_lower:
            return ColumnDataType.TIME

        # Binary types
        if any(t in native_lower for t in ['binary', 'varbinary', 'blob', 'bytea']):
            return ColumnDataType.BINARY

        # JSON types
        if 'json' in native_lower:
            return ColumnDataType.JSON

        # Array types
        if 'array' in native_lower or '[]' in native_lower:
            return ColumnDataType.ARRAY

        # UUID types
        if 'uuid' in native_lower or 'uniqueidentifier' in native_lower:
            return ColumnDataType.UUID

        return ColumnDataType.UNKNOWN

    def _build_table_dependencies(self, schema: DatabaseSchema):
        """Build table dependency relationships based on foreign keys."""
        all_tables = schema.get_all_tables()

        for table_name, table_info in all_tables.items():
            for constraint in table_info.constraints.values():
                if constraint.constraint_type == "FOREIGN KEY" and constraint.referenced_table:
                    referenced_table = constraint.referenced_table

                    # Find the referenced table (may be in different schema)
                    referenced_full_name = None
                    for full_name, ref_table in all_tables.items():
                        if ref_table.name == referenced_table:
                            referenced_full_name = full_name
                            break

                    if referenced_full_name:
                        # Add dependency relationship
                        table_info.dependencies.add(referenced_full_name)
                        all_tables[referenced_full_name].dependents.add(table_name)

    def get_table_relationships(self, schema: DatabaseSchema) -> Dict[str, List[str]]:
        """Get table relationships based on foreign keys.

        Returns:
            Dictionary mapping table names to lists of related tables
        """
        relationships = {}
        all_tables = schema.get_all_tables()

        for table_name, table_info in all_tables.items():
            related_tables = list(table_info.dependencies.union(table_info.dependents))
            if related_tables:
                relationships[table_name] = related_tables

        return relationships

    def find_tables_by_column_name(self, schema: DatabaseSchema, column_name: str) -> List[str]:
        """Find all tables that contain a specific column name.

        Args:
            schema: Database schema to search
            column_name: Column name to find

        Returns:
            List of table names containing the column
        """
        matching_tables = []
        all_tables = schema.get_all_tables()

        for table_name, table_info in all_tables.items():
            if column_name.lower() in [col.lower() for col in table_info.columns.keys()]:
                matching_tables.append(table_name)

        return matching_tables

    def find_similar_tables(self, schema: DatabaseSchema, table_name: str,
                          similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find tables with similar column structures.

        Args:
            schema: Database schema to search
            table_name: Reference table name
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of tuples (table_name, similarity_score) sorted by similarity
        """
        all_tables = schema.get_all_tables()

        if table_name not in all_tables:
            return []

        reference_table = all_tables[table_name]
        reference_columns = set(reference_table.columns.keys())
        similar_tables = []

        for other_table_name, other_table in all_tables.items():
            if other_table_name == table_name:
                continue

            other_columns = set(other_table.columns.keys())

            # Calculate Jaccard similarity
            intersection = reference_columns.intersection(other_columns)
            union = reference_columns.union(other_columns)

            if union:
                similarity = len(intersection) / len(union)
                if similarity >= similarity_threshold:
                    similar_tables.append((other_table_name, similarity))

        # Sort by similarity score (descending)
        similar_tables.sort(key=lambda x: x[1], reverse=True)
        return similar_tables

    def analyze_schema_quality(self, schema: DatabaseSchema) -> Dict[str, Any]:
        """Analyze schema quality and provide recommendations.

        Returns:
            Dictionary with quality metrics and recommendations
        """
        all_tables = schema.get_all_tables()
        quality_issues = []
        metrics = {
            'total_tables': len(all_tables),
            'tables_without_primary_key': 0,
            'tables_without_indexes': 0,
            'tables_with_many_columns': 0,
            'unused_indexes': 0,
            'foreign_key_violations': 0,
            'naming_inconsistencies': 0
        }

        for table_name, table_info in all_tables.items():
            # Check for primary key
            has_primary_key = any(col.is_primary_key for col in table_info.columns.values())
            if not has_primary_key:
                metrics['tables_without_primary_key'] += 1
                quality_issues.append(f"Table {table_name} lacks a primary key")

            # Check for indexes
            if not table_info.indexes:
                metrics['tables_without_indexes'] += 1
                quality_issues.append(f"Table {table_name} has no indexes")

            # Check for too many columns
            if len(table_info.columns) > 50:
                metrics['tables_with_many_columns'] += 1
                quality_issues.append(f"Table {table_name} has {len(table_info.columns)} columns (consider normalization)")

            # Check naming conventions
            if '_' in table_name and table_name.lower() != table_name:
                metrics['naming_inconsistencies'] += 1
                quality_issues.append(f"Table {table_name} uses inconsistent naming convention")

        # Calculate quality score
        total_checks = len(all_tables) * 4  # 4 checks per table
        total_issues = len(quality_issues)
        quality_score = max(0, (total_checks - total_issues) / total_checks * 100)

        return {
            'quality_score': round(quality_score, 2),
            'metrics': metrics,
            'issues': quality_issues[:20],  # Limit to top 20 issues
            'recommendations': self._generate_schema_recommendations(metrics)
        }

    def _generate_schema_recommendations(self, metrics: Dict[str, int]) -> List[str]:
        """Generate schema improvement recommendations."""
        recommendations = []

        if metrics['tables_without_primary_key'] > 0:
            recommendations.append(
                f"Add primary keys to {metrics['tables_without_primary_key']} tables for better performance and data integrity"
            )

        if metrics['tables_without_indexes'] > 0:
            recommendations.append(
                f"Consider adding indexes to {metrics['tables_without_indexes']} tables to improve query performance"
            )

        if metrics['tables_with_many_columns'] > 0:
            recommendations.append(
                f"Review {metrics['tables_with_many_columns']} tables with many columns for potential normalization"
            )

        if metrics['naming_inconsistencies'] > 0:
            recommendations.append(
                f"Standardize naming conventions for {metrics['naming_inconsistencies']} objects"
            )

        return recommendations

    def clear_cache(self):
        """Clear the schema cache."""
        with self._lock:
            self._schema_cache.clear()
            logger.info("Schema cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached schemas."""
        with self._lock:
            cache_info = {}
            now = datetime.now()

            for cache_key, (schema, cache_time) in self._schema_cache.items():
                age_minutes = (now - cache_time).total_seconds() / 60
                cache_info[cache_key] = {
                    'database_name': schema.database_name,
                    'database_type': schema.database_type,
                    'total_objects': schema.total_objects,
                    'cached_at': cache_time.isoformat(),
                    'age_minutes': round(age_minutes, 2),
                    'expires_in_minutes': round(self.cache_duration.total_seconds() / 60 - age_minutes, 2)
                }

            return {
                'total_cached_schemas': len(self._schema_cache),
                'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
                'cached_schemas': cache_info
            }