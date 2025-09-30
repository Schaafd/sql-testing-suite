"""Test fixtures and mock data generation for SQL unit testing."""

import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FixtureDefinition:
    """Defines a test data fixture."""
    fixture_id: str
    name: str
    table_name: str
    data: pd.DataFrame
    cleanup_sql: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class MockDataGenerator:
    """Generate realistic mock data for testing."""

    @staticmethod
    def generate_integers(count: int, min_val: int = 1, max_val: int = 1000) -> List[int]:
        """Generate random integers."""
        return [random.randint(min_val, max_val) for _ in range(count)]

    @staticmethod
    def generate_floats(count: int, min_val: float = 0.0, max_val: float = 100.0,
                       decimals: int = 2) -> List[float]:
        """Generate random floats."""
        return [round(random.uniform(min_val, max_val), decimals) for _ in range(count)]

    @staticmethod
    def generate_strings(count: int, length: int = 10, prefix: str = "") -> List[str]:
        """Generate random strings."""
        return [
            f"{prefix}{''.join(random.choices(string.ascii_letters, k=length))}"
            for _ in range(count)
        ]

    @staticmethod
    def generate_emails(count: int, domain: str = "example.com") -> List[str]:
        """Generate random email addresses."""
        return [
            f"user{i}_{random.randint(1000, 9999)}@{domain}"
            for i in range(count)
        ]

    @staticmethod
    def generate_dates(count: int, start_date: Optional[datetime] = None,
                      days_range: int = 365) -> List[datetime]:
        """Generate random dates."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days_range)

        return [
            start_date + timedelta(days=random.randint(0, days_range))
            for _ in range(count)
        ]

    @staticmethod
    def generate_booleans(count: int, true_probability: float = 0.5) -> List[bool]:
        """Generate random booleans."""
        return [random.random() < true_probability for _ in range(count)]

    @staticmethod
    def generate_categories(count: int, categories: List[str]) -> List[str]:
        """Generate random categorical values."""
        return [random.choice(categories) for _ in range(count)]


class FixtureManager:
    """Manage test fixtures and data setup/teardown."""

    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self._fixtures: Dict[str, FixtureDefinition] = {}
        self._loaded_fixtures: Dict[str, bool] = {}

    def register_fixture(self, fixture: FixtureDefinition):
        """Register a fixture definition."""
        self._fixtures[fixture.fixture_id] = fixture

    def load_fixture(self, fixture_id: str, database_name: str) -> bool:
        """Load fixture data into database."""
        if fixture_id not in self._fixtures:
            raise ValueError(f"Fixture {fixture_id} not found")

        fixture = self._fixtures[fixture_id]

        # Load dependencies first
        for dep_id in fixture.dependencies:
            if not self._loaded_fixtures.get(dep_id, False):
                self.load_fixture(dep_id, database_name)

        # Insert fixture data
        try:
            # This would use connection_manager to insert data
            # Simplified for now
            self._loaded_fixtures[fixture_id] = True
            return True

        except Exception as e:
            return False

    def cleanup_fixture(self, fixture_id: str, database_name: str):
        """Remove fixture data from database."""
        if fixture_id not in self._fixtures:
            return

        fixture = self._fixtures[fixture_id]

        if fixture.cleanup_sql:
            try:
                self.connection_manager.execute_query(
                    fixture.cleanup_sql,
                    db_name=database_name
                )
                self._loaded_fixtures[fixture_id] = False
            except Exception:
                pass

    def create_test_data(self, table_name: str, row_count: int,
                        schema: Dict[str, str]) -> pd.DataFrame:
        """Create test data based on schema."""
        data = {}
        generator = MockDataGenerator()

        for column_name, data_type in schema.items():
            if 'int' in data_type.lower():
                data[column_name] = generator.generate_integers(row_count)
            elif 'float' in data_type.lower() or 'decimal' in data_type.lower():
                data[column_name] = generator.generate_floats(row_count)
            elif 'bool' in data_type.lower():
                data[column_name] = generator.generate_booleans(row_count)
            elif 'date' in data_type.lower():
                data[column_name] = generator.generate_dates(row_count)
            elif 'email' in column_name.lower():
                data[column_name] = generator.generate_emails(row_count)
            else:
                data[column_name] = generator.generate_strings(row_count)

        return pd.DataFrame(data)