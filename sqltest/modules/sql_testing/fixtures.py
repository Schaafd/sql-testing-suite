"""
Test fixture management for SQL unit testing framework.

This module handles loading test data from various sources including CSV files,
JSON files, SQL scripts, inline data, and generated mock data.
"""
import json
import csv
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
from faker import Faker
import random

from sqltest.core.connection_manager import ConnectionManager
from .models import TestFixture, FixtureType, MockDataConfig


class FixtureManager:
    """Manages test fixtures including data loading and database setup."""
    
    def __init__(self, connection_manager: ConnectionManager, base_path: Optional[Path] = None):
        """
        Initialize fixture manager.
        
        Args:
            connection_manager: Database connection manager
            base_path: Base path for resolving relative fixture file paths
        """
        self.connection_manager = connection_manager
        self.base_path = base_path or Path.cwd()
        self._loaded_fixtures: Dict[str, pd.DataFrame] = {}
        self._created_tables: List[str] = []
        
    def load_fixture_data(self, fixture: TestFixture) -> pd.DataFrame:
        """
        Load data from fixture based on its type.
        
        Args:
            fixture: Test fixture configuration
            
        Returns:
            DataFrame containing fixture data
            
        Raises:
            ValueError: If fixture type is unsupported or data is invalid
            FileNotFoundError: If fixture file doesn't exist
        """
        if fixture.name in self._loaded_fixtures:
            return self._loaded_fixtures[fixture.name]
            
        if fixture.fixture_type == FixtureType.CSV:
            data = self._load_csv_fixture(fixture)
        elif fixture.fixture_type == FixtureType.JSON:
            data = self._load_json_fixture(fixture)
        elif fixture.fixture_type == FixtureType.SQL:
            data = self._load_sql_fixture(fixture)
        elif fixture.fixture_type == FixtureType.INLINE:
            data = self._load_inline_fixture(fixture)
        elif fixture.fixture_type == FixtureType.GENERATED:
            data = self._load_generated_fixture(fixture)
        else:
            raise ValueError(f"Unsupported fixture type: {fixture.fixture_type}")
            
        # Cache the loaded data
        self._loaded_fixtures[fixture.name] = data
        return data
    
    def _load_csv_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV fixture file not found: {file_path}")
            
        # Load CSV with type inference
        df = pd.read_csv(file_path)
        
        # Apply schema if provided
        if fixture.schema:
            df = self._apply_schema(df, fixture.schema)
            
        return df
    
    def _load_json_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Load data from JSON file."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON fixture file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("JSON fixture must contain list or dict")
        
        # Apply schema if provided
        if fixture.schema:
            df = self._apply_schema(df, fixture.schema)
            
        return df
    
    def _load_sql_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Load data by executing SQL file."""
        file_path = self._resolve_path(fixture.data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"SQL fixture file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            sql = f.read()
        
        # Execute SQL and return results
        result = self.connection_manager.execute_query(sql)
        if result.success and result.data is not None:
            return result.data
        else:
            raise ValueError(f"SQL fixture query failed: {result.error}")
    
    def _load_inline_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Load data from inline configuration."""
        data = fixture.data_source
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Inline fixture data must be list or dict")
        
        # Apply schema if provided
        if fixture.schema:
            df = self._apply_schema(df, fixture.schema)
            
        return df
    
    def _load_generated_fixture(self, fixture: TestFixture) -> pd.DataFrame:
        """Generate mock data based on configuration."""
        if not isinstance(fixture.data_source, dict):
            raise ValueError("Generated fixture requires dict data_source with MockDataConfig")
            
        # Create mock data generator
        generator = MockDataGenerator(fixture.data_source)
        data = generator.generate()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Apply schema if provided
        if fixture.schema:
            df = self._apply_schema(df, fixture.schema)
            
        return df
    
    def _apply_schema(self, df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        """Apply schema type conversions to DataFrame."""
        for column, dtype in schema.items():
            if column not in df.columns:
                continue
                
            try:
                if dtype.lower() in ['int', 'integer', 'bigint']:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif dtype.lower() in ['float', 'decimal', 'numeric', 'double']:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif dtype.lower() in ['bool', 'boolean']:
                    df[column] = df[column].astype(bool)
                elif dtype.lower() in ['date']:
                    df[column] = pd.to_datetime(df[column]).dt.date
                elif dtype.lower() in ['datetime', 'timestamp']:
                    df[column] = pd.to_datetime(df[column])
                elif dtype.lower() in ['str', 'string', 'varchar', 'text']:
                    df[column] = df[column].astype(str)
            except Exception as e:
                raise ValueError(f"Failed to convert column {column} to {dtype}: {e}")
                
        return df
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve file path relative to base path."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return self.base_path / path_obj
    
    async def setup_fixtures(self, fixtures: List[TestFixture]) -> None:
        """
        Set up test fixtures by creating tables and loading data.
        
        Args:
            fixtures: List of fixtures to set up
        """
        for fixture in fixtures:
            # Load fixture data
            data = self.load_fixture_data(fixture)
            
            # Create and populate table
            await self._create_table_from_dataframe(
                fixture.table_name,
                data,
                fixture.schema
            )
            
            # Track created table for cleanup
            if fixture.table_name not in self._created_tables:
                self._created_tables.append(fixture.table_name)
    
    async def _create_table_from_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        schema: Optional[Dict[str, str]] = None
    ) -> None:
        """Create database table from DataFrame and insert data."""
        if df.empty:
            return
            
        # Generate CREATE TABLE statement
        create_sql = self._generate_create_table_sql(table_name, df, schema)
        
        # Execute CREATE TABLE
        result = self.connection_manager.execute_query(create_sql)
        if not result.success:
            raise ValueError(f"Failed to create table {table_name}: {result.error}")
        
        # Insert data
        await self._insert_dataframe_data(table_name, df)
    
    def _generate_create_table_sql(
        self,
        table_name: str,
        df: pd.DataFrame,
        schema: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate CREATE TABLE SQL from DataFrame structure."""
        columns = []
        
        for column in df.columns:
            if schema and column in schema:
                sql_type = schema[column]
            else:
                # Infer SQL type from pandas dtype
                sql_type = self._pandas_dtype_to_sql(df[column].dtype)
            
            columns.append(f"{column} {sql_type}")
        
        return f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
    
    def _pandas_dtype_to_sql(self, dtype) -> str:
        """Convert pandas dtype to SQL type."""
        dtype_str = str(dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'REAL'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        else:
            return 'TEXT'
    
    async def _insert_dataframe_data(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert DataFrame data into database table."""
        if df.empty:
            return
            
        # Convert DataFrame to list of tuples for insertion
        values = []
        for _, row in df.iterrows():
            row_values = []
            for value in row:
                if pd.isna(value):
                    row_values.append(None)
                elif isinstance(value, (datetime, date)):
                    row_values.append(value.isoformat())
                elif isinstance(value, Decimal):
                    row_values.append(float(value))
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        # Generate INSERT statement
        placeholders = ', '.join(['?' for _ in df.columns])
        columns = ', '.join(df.columns)
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        # Execute batch insert
        # Note: This would need to be adapted based on the connection manager's batch insert capabilities
        for value_tuple in values:
            result = self.connection_manager.execute_query(insert_sql, value_tuple)
            if not result.success:
                raise ValueError(f"Failed to insert data into {table_name}: {result.error}")
    
    async def cleanup_fixtures(self, fixtures: List[TestFixture]) -> None:
        """
        Clean up test fixtures by dropping created tables.
        
        Args:
            fixtures: List of fixtures to clean up
        """
        for fixture in fixtures:
            if fixture.cleanup and fixture.table_name in self._created_tables:
                drop_sql = f"DROP TABLE IF EXISTS {fixture.table_name}"
                result = self.connection_manager.execute_query(drop_sql)
                if not result.success:
                    # Log warning but don't fail test
                    print(f"Warning: Failed to drop table {fixture.table_name}: {result.error}")
                else:
                    self._created_tables.remove(fixture.table_name)
    
    def clear_cache(self) -> None:
        """Clear fixture data cache."""
        self._loaded_fixtures.clear()


class MockDataGenerator:
    """Generates mock test data using Faker and custom logic."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock data generator.
        
        Args:
            config: Mock data configuration dictionary
        """
        self.config = MockDataConfig(**config)
        self.fake = Faker()
        
        # Set seed for reproducible data
        if self.config.seed:
            Faker.seed(self.config.seed)
            random.seed(self.config.seed)
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate mock data based on configuration."""
        data = []
        
        for i in range(self.config.rows):
            row = {}
            for column_name, column_config in self.config.columns.items():
                value = self._generate_column_value(column_name, column_config, i)
                row[column_name] = value
            data.append(row)
        
        return data
    
    def _generate_column_value(self, column_name: str, config: Dict[str, Any], row_index: int) -> Any:
        """Generate a single column value based on configuration."""
        column_type = config.get('type', 'string')
        faker_provider = config.get('faker_provider')
        constraints = config.get('constraints', {})
        
        # Handle relationships (foreign keys)
        if self.config.relationships and column_name in self.config.relationships:
            # For now, generate sequential IDs for relationships
            # In a more advanced implementation, this would reference actual data
            return row_index + 1
        
        # Use custom faker provider if specified
        if faker_provider:
            try:
                method = getattr(self.fake, faker_provider)
                return method()
            except AttributeError:
                pass
        
        # Generate value based on type
        if column_type == 'integer':
            min_val = constraints.get('min', 1)
            max_val = constraints.get('max', 1000)
            return random.randint(min_val, max_val)
        
        elif column_type == 'float':
            min_val = constraints.get('min', 0.0)
            max_val = constraints.get('max', 100.0)
            return round(random.uniform(min_val, max_val), 2)
        
        elif column_type == 'string':
            if 'choices' in constraints:
                return random.choice(constraints['choices'])
            else:
                length = constraints.get('length', 10)
                return self.fake.text(max_nb_chars=length).strip()
        
        elif column_type == 'email':
            return self.fake.email()
        
        elif column_type == 'name':
            return self.fake.name()
        
        elif column_type == 'date':
            start_date = constraints.get('start_date', date(2020, 1, 1))
            end_date = constraints.get('end_date', date.today())
            return self.fake.date_between(start_date=start_date, end_date=end_date)
        
        elif column_type == 'datetime':
            start_date = constraints.get('start_date', datetime(2020, 1, 1))
            end_date = constraints.get('end_date', datetime.now())
            return self.fake.date_time_between(start_date=start_date, end_date=end_date)
        
        elif column_type == 'boolean':
            return random.choice([True, False])
        
        else:
            # Default to string
            return self.fake.word()
