"""Enhanced fixture management for SQL unit testing framework.

This module provides enterprise-grade fixture management with features including:
- Advanced data generation with realistic patterns
- Bulk loading optimization for large datasets
- Intelligent cleanup and resource management
- Caching and performance optimization
- Cross-database compatibility
"""
import asyncio
import json
import csv
import logging
import threading
import time
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
import pandas as pd
import numpy as np
from faker import Faker
import random
from collections import defaultdict

from ...db.connection import ConnectionManager
from .models import TestFixture, FixtureType

logger = logging.getLogger(__name__)


class DataGenerationEngine:
    """Advanced data generation engine with realistic patterns."""

    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def generate_realistic_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate realistic test data based on configuration."""
        row_count = config.get('row_count', 100)
        columns = config.get('columns', {})
        relationships = config.get('relationships', {})

        generated_data = []

        # First pass: generate primary data
        for i in range(row_count):
            row = {}
            for column_name, column_config in columns.items():
                row[column_name] = self._generate_column_value(column_config, i)
            generated_data.append(row)

        # Second pass: apply relationships and constraints
        if relationships:
            generated_data = self._apply_relationships(generated_data, relationships)

        return generated_data

    def _generate_column_value(self, config: Dict[str, Any], row_index: int) -> Any:
        """Generate a single column value based on configuration."""
        data_type = config.get('type', 'string')
        pattern = config.get('pattern')
        constraints = config.get('constraints', {})

        if pattern:
            return self._generate_pattern_value(pattern, row_index, constraints)

        return self._generate_typed_value(data_type, constraints)

    def _generate_pattern_value(self, pattern: str, row_index: int, constraints: Dict) -> Any:
        """Generate value based on pattern."""
        if pattern == 'email':
            domains = constraints.get('domains', ['example.com', 'test.org'])
            return self.faker.email().replace(self.faker.email().split('@')[1], random.choice(domains))
        elif pattern == 'phone':
            return self.faker.phone_number()
        elif pattern == 'name':
            return self.faker.name()
        elif pattern == 'address':
            return self.faker.address()
        elif pattern == 'company':
            return self.faker.company()
        elif pattern == 'sequential':
            start = constraints.get('start', 1)
            return start + row_index
        elif pattern == 'uuid':
            return str(uuid.uuid4())
        elif pattern == 'timestamp':
            days_ago = constraints.get('days_ago', 30)
            return self.faker.date_time_between(start_date=f'-{days_ago}d', end_date='now')
        elif pattern == 'status':
            values = constraints.get('values', ['active', 'inactive', 'pending'])
            weights = constraints.get('weights', None)
            return random.choices(values, weights=weights)[0] if weights else random.choice(values)
        elif pattern == 'amount':
            min_val = constraints.get('min', 0)
            max_val = constraints.get('max', 1000)
            decimals = constraints.get('decimals', 2)
            return round(random.uniform(min_val, max_val), decimals)
        else:
            return self.faker.text(max_nb_chars=50)

    def _generate_typed_value(self, data_type: str, constraints: Dict) -> Any:
        """Generate value based on data type."""
        if data_type == 'integer':
            min_val = constraints.get('min', 1)
            max_val = constraints.get('max', 1000)
            return random.randint(min_val, max_val)
        elif data_type == 'float':
            min_val = constraints.get('min', 0.0)
            max_val = constraints.get('max', 1000.0)
            return round(random.uniform(min_val, max_val), 2)
        elif data_type == 'boolean':
            true_probability = constraints.get('true_probability', 0.5)
            return random.random() < true_probability
        elif data_type == 'date':
            days_ago = constraints.get('days_ago', 365)
            return self.faker.date_between(start_date=f'-{days_ago}d', end_date='today')
        elif data_type == 'datetime':
            days_ago = constraints.get('days_ago', 365)
            return self.faker.date_time_between(start_date=f'-{days_ago}d', end_date='now')
        else:  # string
            max_length = constraints.get('max_length', 50)
            return self.faker.text(max_nb_chars=max_length)

    def _apply_relationships(self, data: List[Dict], relationships: Dict) -> List[Dict]:
        """Apply relationships between columns."""
        for relationship in relationships:
            source_column = relationship.get('source_column')
            target_column = relationship.get('target_column')
            relationship_type = relationship.get('type', 'reference')

            if relationship_type == 'reference':
                # Foreign key relationship
                source_values = [row[source_column] for row in data if source_column in row]
                if source_values:
                    for row in data:
                        if target_column in row:
                            row[target_column] = random.choice(source_values)

            elif relationship_type == 'calculated':
                # Calculated field
                formula = relationship.get('formula')
                for row in data:
                    try:
                        # Simple formula evaluation (extend as needed)
                        if formula and source_column in row:
                            if formula == 'multiply_by_2':
                                row[target_column] = row[source_column] * 2
                            elif formula == 'percentage':
                                row[target_column] = row[source_column] / 100
                    except Exception as e:
                        logger.warning(f"Failed to apply formula {formula}: {e}")

        return data


class FixtureCache:
    """Intelligent caching system for fixtures."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached fixture data."""
        with self._lock:
            if cache_key not in self._cache:
                return None

            cache_entry = self._cache[cache_key]
            cache_time = self._access_times[cache_key]

            # Check TTL
            if time.time() - cache_time > self.ttl_seconds:
                del self._cache[cache_key]
                del self._access_times[cache_key]
                return None

            # Update access time
            self._access_times[cache_key] = time.time()
            return cache_entry['data']

    def set(self, cache_key: str, data: pd.DataFrame):
        """Cache fixture data."""
        with self._lock:
            # Evict old entries if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(),
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

            self._cache[cache_key] = {'data': data}
            self._access_times[cache_key] = time.time()

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class EnhancedFixtureManager:
    """Enterprise-grade fixture manager with advanced capabilities."""

    def __init__(self,
                 connection_manager: ConnectionManager,
                 base_path: Optional[Path] = None,
                 enable_caching: bool = True,
                 bulk_insert_threshold: int = 1000):
        """
        Initialize enhanced fixture manager.

        Args:
            connection_manager: Database connection manager
            base_path: Base path for resolving relative fixture file paths
            enable_caching: Enable fixture data caching
            bulk_insert_threshold: Minimum rows for bulk insert optimization
        """
        self.connection_manager = connection_manager
        self.base_path = base_path or Path.cwd()
        self.bulk_insert_threshold = bulk_insert_threshold

        # Initialize components
        self.data_generator = DataGenerationEngine()
        self.cache = FixtureCache() if enable_caching else None

        # State tracking
        self._loaded_fixtures: Dict[str, pd.DataFrame] = {}
        self._created_tables: Set[str] = set()
        self._cleanup_registry: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        logger.info(f"Enhanced Fixture Manager initialized with caching={'enabled' if enable_caching else 'disabled'}")

    async def setup_fixtures(self,
                           fixtures: List[TestFixture],
                           isolation_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set up test fixtures with enterprise-grade optimization.

        Args:
            fixtures: List of fixtures to set up
            isolation_info: Test isolation context information

        Returns:
            Setup summary with performance metrics
        """
        setup_start = time.time()
        setup_summary = {
            'fixtures_loaded': 0,
            'tables_created': 0,
            'rows_inserted': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'bulk_operations': 0
        }

        for fixture in fixtures:
            try:
                # Load fixture data
                data = await self._load_fixture_data_async(fixture)
                if data is None or data.empty:
                    continue

                setup_summary['fixtures_loaded'] += 1
                setup_summary['rows_inserted'] += len(data)

                # Create table with optimized schema
                table_name = fixture.table_name
                if isolation_info and isolation_info.get('schema'):
                    table_name = f"{isolation_info['schema']}.{fixture.table_name}"

                await self._create_optimized_table(table_name, data, fixture)
                setup_summary['tables_created'] += 1

                # Insert data with optimization
                if len(data) >= self.bulk_insert_threshold:
                    await self._bulk_insert_data(table_name, data)
                    setup_summary['bulk_operations'] += 1
                else:
                    await self._insert_data(table_name, data)

                # Register for cleanup
                self._register_cleanup(fixture, table_name, isolation_info)

            except Exception as e:
                logger.error(f"Failed to setup fixture {fixture.name}: {e}")
                raise

        setup_time = time.time() - setup_start
        setup_summary['setup_time_seconds'] = setup_time

        logger.info(f"Fixture setup completed in {setup_time:.2f}s: {setup_summary}")
        return setup_summary

    async def _load_fixture_data_async(self, fixture: TestFixture) -> Optional[pd.DataFrame]:
        """Load fixture data with caching and async optimization."""
        cache_key = self._generate_cache_key(fixture)

        # Check cache first
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for fixture: {fixture.name}")
                return cached_data

        # Load data based on fixture type
        if fixture.fixture_type == FixtureType.CSV:
            data = await self._load_csv_async(fixture)
        elif fixture.fixture_type == FixtureType.JSON:
            data = await self._load_json_async(fixture)
        elif fixture.fixture_type == FixtureType.SQL:
            data = await self._load_sql_async(fixture)
        elif fixture.fixture_type == FixtureType.INLINE:
            data = self._load_inline_fixture(fixture)
        elif fixture.fixture_type == FixtureType.GENERATED:
            data = await self._load_generated_async(fixture)
        else:
            raise ValueError(f"Unsupported fixture type: {fixture.fixture_type}")

        # Apply schema transformations
        if fixture.schema and data is not None:
            data = self._apply_schema_optimized(data, fixture.schema)

        # Cache the result
        if self.cache and data is not None:
            self.cache.set(cache_key, data)

        return data

    async def _load_csv_async(self, fixture: TestFixture) -> pd.DataFrame:
        """Load CSV fixture asynchronously."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV fixture file not found: {file_path}")

        # Use asyncio for large files
        loop = asyncio.get_event_loop()

        def load_csv():
            return pd.read_csv(
                file_path,
                dtype_backend='numpy_nullable',  # Use nullable dtypes for better performance
                engine='c'  # Use faster C engine
            )

        return await loop.run_in_executor(None, load_csv)

    async def _load_json_async(self, fixture: TestFixture) -> pd.DataFrame:
        """Load JSON fixture asynchronously."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON fixture file not found: {file_path}")

        loop = asyncio.get_event_loop()

        def load_json():
            with open(file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError("JSON fixture must contain list or dict")

        return await loop.run_in_executor(None, load_json)

    async def _load_sql_async(self, fixture: TestFixture) -> pd.DataFrame:
        """Load SQL fixture asynchronously."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"SQL fixture file not found: {file_path}")

        with open(file_path, 'r') as f:
            sql = f.read()

        # Execute SQL asynchronously
        result = await self.connection_manager.get_adapter().execute_query(sql)
        if result.success and result.data is not None:
            return result.data
        else:
            raise ValueError(f"SQL fixture query failed: {result.error}")

    async def _load_generated_async(self, fixture: TestFixture) -> pd.DataFrame:
        """Generate fixture data asynchronously."""
        if not isinstance(fixture.data_source, dict):
            raise ValueError("Generated fixture requires dict data_source")

        loop = asyncio.get_event_loop()

        def generate_data():
            data = self.data_generator.generate_realistic_data(fixture.data_source)
            return pd.DataFrame(data)

        return await loop.run_in_executor(None, generate_data)

    def _load_inline_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Load inline fixture data."""
        data = fixture.data_source

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("Inline fixture data must be list or dict")

    async def _create_optimized_table(self,
                                    table_name: str,
                                    data: pd.DataFrame,
                                    fixture: TestFixture):
        """Create table with optimized schema and indexes."""
        # Generate optimized CREATE TABLE statement
        columns = []
        for col_name, dtype in data.dtypes.items():
            sql_type = self._pandas_to_sql_type(dtype, fixture.schema)
            columns.append(f"{col_name} {sql_type}")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"

        # Add indexes if specified
        if hasattr(fixture, 'indexes') and fixture.indexes:
            for index_config in fixture.indexes:
                index_sql = self._generate_index_sql(table_name, index_config)
                create_sql += f"; {index_sql}"

        await self.connection_manager.get_adapter().execute_query(create_sql)
        self._created_tables.add(table_name)

    async def _bulk_insert_data(self, table_name: str, data: pd.DataFrame):
        """Perform optimized bulk insert."""
        # Use database-specific bulk insert methods
        adapter = self.connection_manager.get_adapter()

        if hasattr(adapter, 'bulk_insert'):
            await adapter.bulk_insert(table_name, data)
        else:
            # Fallback to chunked inserts
            chunk_size = 1000
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                await self._insert_data_chunk(table_name, chunk)

    async def _insert_data(self, table_name: str, data: pd.DataFrame):
        """Insert data using standard method."""
        await self._insert_data_chunk(table_name, data)

    async def _insert_data_chunk(self, table_name: str, data: pd.DataFrame):
        """Insert a chunk of data."""
        if data.empty:
            return

        columns = ', '.join(data.columns)
        placeholders = ', '.join(['%s'] * len(data.columns))

        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        # Convert to list of tuples for insertion
        values = [tuple(row) for row in data.values]

        await self.connection_manager.get_adapter().execute_many(insert_sql, values)

    async def cleanup_fixtures(self, fixtures: List[TestFixture]):
        """Clean up fixtures with optimized resource management."""
        cleanup_start = time.time()

        for fixture in fixtures:
            if not fixture.cleanup:
                continue

            cleanup_info = self._cleanup_registry.get(fixture.name)
            if not cleanup_info:
                continue

            try:
                table_name = cleanup_info['table_name']

                # Drop table
                drop_sql = f"DROP TABLE IF EXISTS {table_name}"
                await self.connection_manager.get_adapter().execute_query(drop_sql)

                # Remove from tracking
                self._created_tables.discard(table_name)
                del self._cleanup_registry[fixture.name]

            except Exception as e:
                logger.warning(f"Failed to cleanup fixture {fixture.name}: {e}")

        cleanup_time = time.time() - cleanup_start
        logger.info(f"Fixture cleanup completed in {cleanup_time:.2f}s")

    def _generate_cache_key(self, fixture: TestFixture) -> str:
        """Generate cache key for fixture."""
        key_parts = [
            fixture.name,
            fixture.fixture_type.value,
            str(fixture.data_source),
            str(fixture.schema)
        ]
        return '_'.join(str(part) for part in key_parts)

    def _apply_schema_optimized(self, df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        """Apply schema with performance optimization."""
        for column, dtype in schema.items():
            if column not in df.columns:
                continue

            try:
                if dtype.lower() in ['int', 'integer', 'bigint']:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif dtype.lower() in ['float', 'decimal', 'numeric', 'double']:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Float64')
                elif dtype.lower() in ['bool', 'boolean']:
                    df[column] = df[column].astype('boolean')
                elif dtype.lower() in ['date']:
                    df[column] = pd.to_datetime(df[column], errors='coerce').dt.date
                elif dtype.lower() in ['datetime', 'timestamp']:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif dtype.lower() in ['str', 'string', 'varchar', 'text']:
                    df[column] = df[column].astype('string')
            except Exception as e:
                logger.warning(f"Failed to convert column {column} to {dtype}: {e}")

        return df

    def _pandas_to_sql_type(self, pandas_dtype, schema: Optional[Dict] = None) -> str:
        """Convert pandas dtype to SQL type."""
        dtype_str = str(pandas_dtype).lower()

        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'REAL'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'TEXT'
        else:
            return 'TEXT'

    def _generate_index_sql(self, table_name: str, index_config: Dict) -> str:
        """Generate index creation SQL."""
        index_name = index_config.get('name', f"idx_{table_name}_{uuid.uuid4().hex[:8]}")
        columns = index_config.get('columns', [])
        unique = index_config.get('unique', False)

        unique_clause = 'UNIQUE ' if unique else ''
        columns_clause = ', '.join(columns)

        return f"CREATE {unique_clause}INDEX {index_name} ON {table_name} ({columns_clause})"

    def _register_cleanup(self,
                         fixture: TestFixture,
                         table_name: str,
                         isolation_info: Optional[Dict]):
        """Register fixture for cleanup."""
        self._cleanup_registry[fixture.name] = {
            'table_name': table_name,
            'isolation_info': isolation_info,
            'created_at': datetime.now()
        }

    def _resolve_path(self, path: str) -> Path:
        """Resolve file path relative to base path."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return self.base_path / path_obj

    def clear_cache(self):
        """Clear fixture cache."""
        if self.cache:
            self.cache.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'loaded_fixtures': len(self._loaded_fixtures),
            'created_tables': len(self._created_tables),
            'cleanup_registry_size': len(self._cleanup_registry)
        }

        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()

        return stats


# Maintain backward compatibility
FixtureManager = EnhancedFixtureManager