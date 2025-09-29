"""
Advanced mocking and test doubles framework for SQL unit testing.

This module provides sophisticated mocking capabilities including database mocks,
service mocks, data generators, and intelligent test doubles for comprehensive
SQL testing scenarios.
"""
import logging
import json
import re
import random
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)


class MockType(str, Enum):
    """Types of mocks available in the framework."""
    DATABASE = "database"
    SERVICE = "service"
    DATA_SOURCE = "data_source"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    TIME_SOURCE = "time_source"


class MockBehavior(str, Enum):
    """Behavior patterns for mocks."""
    STATIC = "static"           # Returns predefined responses
    DYNAMIC = "dynamic"         # Generates responses based on input
    SEQUENCE = "sequence"       # Returns responses in sequence
    RANDOM = "random"          # Returns random responses from pool
    CONDITIONAL = "conditional" # Returns responses based on conditions
    FAILURE = "failure"        # Simulates failures
    DELAY = "delay"            # Adds artificial delays


class DataGenerationStrategy(str, Enum):
    """Strategies for generating mock data."""
    REALISTIC = "realistic"     # Use Faker for realistic data
    PATTERN = "pattern"         # Follow specific patterns
    RANDOM = "random"          # Completely random data
    TEMPLATE = "template"       # Use templates with placeholders
    HISTORICAL = "historical"   # Based on historical data patterns


@dataclass
class MockConfiguration:
    """Configuration for mock objects."""
    mock_id: str
    mock_type: MockType
    behavior: MockBehavior
    name: str
    description: Optional[str] = None
    active: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    failure_rate: float = 0.0
    delay_range: Tuple[float, float] = (0.0, 0.0)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MockInvocation:
    """Records an invocation of a mock."""
    invocation_id: str
    mock_id: str
    method: str
    parameters: Dict[str, Any]
    response: Any
    timestamp: datetime
    execution_time: float
    success: bool
    error: Optional[str] = None


class MockRegistry:
    """Central registry for managing all mocks."""

    def __init__(self):
        self.mocks: Dict[str, 'BaseMock'] = {}
        self.configurations: Dict[str, MockConfiguration] = {}
        self.invocations: List[MockInvocation] = []
        self.active_scenarios: Dict[str, str] = {}

    def register_mock(self, mock: 'BaseMock') -> None:
        """Register a mock in the registry."""
        self.mocks[mock.mock_id] = mock
        self.configurations[mock.mock_id] = mock.configuration
        logger.info(f"Registered mock: {mock.configuration.name} ({mock.mock_id})")

    def unregister_mock(self, mock_id: str) -> None:
        """Unregister a mock from the registry."""
        if mock_id in self.mocks:
            del self.mocks[mock_id]
            del self.configurations[mock_id]
            logger.info(f"Unregistered mock: {mock_id}")

    def get_mock(self, mock_id: str) -> Optional['BaseMock']:
        """Get a mock by ID."""
        return self.mocks.get(mock_id)

    def activate_scenario(self, scenario_name: str, mock_configs: Dict[str, Any]) -> None:
        """Activate a testing scenario with specific mock configurations."""
        self.active_scenarios[scenario_name] = json.dumps(mock_configs)

        for mock_id, config in mock_configs.items():
            mock = self.get_mock(mock_id)
            if mock:
                mock.update_configuration(config)

        logger.info(f"Activated scenario: {scenario_name}")

    def deactivate_scenario(self, scenario_name: str) -> None:
        """Deactivate a testing scenario."""
        if scenario_name in self.active_scenarios:
            del self.active_scenarios[scenario_name]
            logger.info(f"Deactivated scenario: {scenario_name}")

    def record_invocation(self, invocation: MockInvocation) -> None:
        """Record a mock invocation."""
        self.invocations.append(invocation)

    def get_invocation_history(self, mock_id: Optional[str] = None) -> List[MockInvocation]:
        """Get invocation history for a specific mock or all mocks."""
        if mock_id:
            return [inv for inv in self.invocations if inv.mock_id == mock_id]
        return self.invocations

    def clear_history(self, mock_id: Optional[str] = None) -> None:
        """Clear invocation history."""
        if mock_id:
            self.invocations = [inv for inv in self.invocations if inv.mock_id != mock_id]
        else:
            self.invocations.clear()

    def reset_all_mocks(self) -> None:
        """Reset all mocks to their initial state."""
        for mock in self.mocks.values():
            mock.reset()
        self.clear_history()


class BaseMock(ABC):
    """Base class for all mock implementations."""

    def __init__(self, configuration: MockConfiguration):
        self.configuration = configuration
        self.mock_id = configuration.mock_id
        self.invocation_count = 0
        self.registry: Optional[MockRegistry] = None

    def set_registry(self, registry: MockRegistry) -> None:
        """Set the mock registry."""
        self.registry = registry

    @abstractmethod
    def invoke(self, method: str, **kwargs) -> Any:
        """Invoke the mock with specified method and parameters."""
        pass

    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update mock configuration."""
        for key, value in config_updates.items():
            if hasattr(self.configuration, key):
                setattr(self.configuration, key, value)

    def reset(self) -> None:
        """Reset mock to initial state."""
        self.invocation_count = 0

    def _record_invocation(self, method: str, parameters: Dict[str, Any],
                          response: Any, execution_time: float,
                          success: bool, error: Optional[str] = None) -> None:
        """Record an invocation of this mock."""
        invocation = MockInvocation(
            invocation_id=str(uuid.uuid4()),
            mock_id=self.mock_id,
            method=method,
            parameters=parameters,
            response=response,
            timestamp=datetime.now(),
            execution_time=execution_time,
            success=success,
            error=error
        )

        if self.registry:
            self.registry.record_invocation(invocation)

    def _should_fail(self) -> bool:
        """Determine if this invocation should fail based on failure rate."""
        return random.random() < self.configuration.failure_rate

    def _get_delay(self) -> float:
        """Get delay for this invocation."""
        min_delay, max_delay = self.configuration.delay_range
        if max_delay > min_delay:
            return random.uniform(min_delay, max_delay)
        return min_delay


class DatabaseMock(BaseMock):
    """Mock for database operations."""

    def __init__(self, configuration: MockConfiguration):
        super().__init__(configuration)
        self.query_responses: Dict[str, Any] = {}
        self.table_schemas: Dict[str, List[Dict[str, str]]] = {}
        self.table_data: Dict[str, pd.DataFrame] = {}

    def add_query_response(self, query_pattern: str, response: Any) -> None:
        """Add a response for a query pattern."""
        self.query_responses[query_pattern] = response

    def add_table_schema(self, table_name: str, schema: List[Dict[str, str]]) -> None:
        """Add schema definition for a table."""
        self.table_schemas[table_name] = schema

    def add_table_data(self, table_name: str, data: pd.DataFrame) -> None:
        """Add mock data for a table."""
        self.table_data[table_name] = data

    def invoke(self, method: str, **kwargs) -> Any:
        """Invoke database mock method."""
        start_time = datetime.now()

        try:
            if self._should_fail():
                raise Exception(f"Simulated database failure for method: {method}")

            if method == "execute_query":
                result = self._execute_query(kwargs.get("query", ""), kwargs.get("parameters", {}))
            elif method == "get_table_schema":
                result = self._get_table_schema(kwargs.get("table_name", ""))
            elif method == "get_table_data":
                result = self._get_table_data(kwargs.get("table_name", ""))
            else:
                raise NotImplementedError(f"Method not implemented: {method}")

            execution_time = (datetime.now() - start_time).total_seconds()

            # Add configured delay
            import time
            delay = self._get_delay()
            if delay > 0:
                time.sleep(delay)
                execution_time += delay

            self._record_invocation(method, kwargs, result, execution_time, True)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_invocation(method, kwargs, None, execution_time, False, str(e))
            raise

    def _execute_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mock query."""
        # Find matching response pattern
        for pattern, response in self.query_responses.items():
            if re.search(pattern, query, re.IGNORECASE):
                if callable(response):
                    return response(query, parameters)
                else:
                    return response

        # Default response for unknown queries
        return {
            "success": True,
            "data": pd.DataFrame(),
            "rows_affected": 0,
            "columns": [],
            "execution_time": 0.001
        }

    def _get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get schema for a table."""
        return self.table_schemas.get(table_name, [])

    def _get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get data for a table."""
        return self.table_data.get(table_name, pd.DataFrame())


class ServiceMock(BaseMock):
    """Mock for external services and APIs."""

    def __init__(self, configuration: MockConfiguration):
        super().__init__(configuration)
        self.endpoints: Dict[str, Dict[str, Any]] = {}

    def add_endpoint(self, method: str, path: str, response: Any,
                    status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        """Add an endpoint response."""
        key = f"{method.upper()}:{path}"
        self.endpoints[key] = {
            "response": response,
            "status_code": status_code,
            "headers": headers or {}
        }

    def invoke(self, method: str, **kwargs) -> Any:
        """Invoke service mock method."""
        start_time = datetime.now()

        try:
            if self._should_fail():
                raise Exception(f"Simulated service failure for method: {method}")

            if method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                result = self._handle_http_request(method, kwargs)
            else:
                result = self._handle_custom_method(method, kwargs)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Add configured delay
            import time
            delay = self._get_delay()
            if delay > 0:
                time.sleep(delay)
                execution_time += delay

            self._record_invocation(method, kwargs, result, execution_time, True)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_invocation(method, kwargs, None, execution_time, False, str(e))
            raise

    def _handle_http_request(self, method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP request to mock service."""
        path = kwargs.get("path", "/")
        key = f"{method}:{path}"

        if key in self.endpoints:
            endpoint_config = self.endpoints[key]
            return {
                "status_code": endpoint_config["status_code"],
                "headers": endpoint_config["headers"],
                "data": endpoint_config["response"]
            }

        # Default 404 response
        return {
            "status_code": 404,
            "headers": {},
            "data": {"error": "Endpoint not found"}
        }

    def _handle_custom_method(self, method: str, kwargs: Dict[str, Any]) -> Any:
        """Handle custom service method."""
        # Implementation depends on specific service being mocked
        return {"method": method, "parameters": kwargs}


class DataSourceMock(BaseMock):
    """Mock for data sources like files, message queues, etc."""

    def __init__(self, configuration: MockConfiguration):
        super().__init__(configuration)
        self.data_store: Dict[str, Any] = {}

    def add_data(self, key: str, data: Any) -> None:
        """Add data to the mock data source."""
        self.data_store[key] = data

    def invoke(self, method: str, **kwargs) -> Any:
        """Invoke data source mock method."""
        start_time = datetime.now()

        try:
            if self._should_fail():
                raise Exception(f"Simulated data source failure for method: {method}")

            if method == "read":
                result = self._read_data(kwargs.get("key", ""))
            elif method == "write":
                result = self._write_data(kwargs.get("key", ""), kwargs.get("data"))
            elif method == "exists":
                result = self._data_exists(kwargs.get("key", ""))
            elif method == "list":
                result = self._list_data()
            else:
                raise NotImplementedError(f"Method not implemented: {method}")

            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_invocation(method, kwargs, result, execution_time, True)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_invocation(method, kwargs, None, execution_time, False, str(e))
            raise

    def _read_data(self, key: str) -> Any:
        """Read data from mock data source."""
        if key in self.data_store:
            return self.data_store[key]
        raise FileNotFoundError(f"Data not found: {key}")

    def _write_data(self, key: str, data: Any) -> bool:
        """Write data to mock data source."""
        self.data_store[key] = data
        return True

    def _data_exists(self, key: str) -> bool:
        """Check if data exists."""
        return key in self.data_store

    def _list_data(self) -> List[str]:
        """List all data keys."""
        return list(self.data_store.keys())


class AdvancedDataGenerator:
    """Advanced data generator for creating realistic test data."""

    def __init__(self, strategy: DataGenerationStrategy = DataGenerationStrategy.REALISTIC):
        self.strategy = strategy
        self.faker = Faker()
        self.templates: Dict[str, str] = {}
        self.patterns: Dict[str, Dict[str, Any]] = {}

    def generate_data(self, schema: Dict[str, Any], count: int = 100) -> pd.DataFrame:
        """Generate data based on schema definition."""
        if self.strategy == DataGenerationStrategy.REALISTIC:
            return self._generate_realistic_data(schema, count)
        elif self.strategy == DataGenerationStrategy.PATTERN:
            return self._generate_pattern_data(schema, count)
        elif self.strategy == DataGenerationStrategy.RANDOM:
            return self._generate_random_data(schema, count)
        elif self.strategy == DataGenerationStrategy.TEMPLATE:
            return self._generate_template_data(schema, count)
        else:
            return self._generate_realistic_data(schema, count)

    def _generate_realistic_data(self, schema: Dict[str, Any], count: int) -> pd.DataFrame:
        """Generate realistic data using Faker."""
        data = {}

        for column, column_config in schema.items():
            if isinstance(column_config, str):
                data_type = column_config
                constraints = {}
            else:
                data_type = column_config.get("type", "string")
                constraints = column_config.get("constraints", {})

            data[column] = self._generate_column_data(data_type, constraints, count)

        return pd.DataFrame(data)

    def _generate_column_data(self, data_type: str, constraints: Dict[str, Any], count: int) -> List[Any]:
        """Generate data for a single column."""
        values = []

        for _ in range(count):
            if data_type == "integer":
                min_val = constraints.get("min", 1)
                max_val = constraints.get("max", 1000)
                values.append(random.randint(min_val, max_val))

            elif data_type == "float":
                min_val = constraints.get("min", 0.0)
                max_val = constraints.get("max", 100.0)
                precision = constraints.get("precision", 2)
                values.append(round(random.uniform(min_val, max_val), precision))

            elif data_type == "string":
                if "choices" in constraints:
                    values.append(random.choice(constraints["choices"]))
                else:
                    length = constraints.get("length", 10)
                    values.append(self.faker.text(max_nb_chars=length).strip())

            elif data_type == "email":
                values.append(self.faker.email())

            elif data_type == "name":
                values.append(self.faker.name())

            elif data_type == "phone":
                values.append(self.faker.phone_number())

            elif data_type == "address":
                values.append(self.faker.address())

            elif data_type == "date":
                start_date = constraints.get("start_date", date(2020, 1, 1))
                end_date = constraints.get("end_date", date.today())
                values.append(self.faker.date_between(start_date=start_date, end_date=end_date))

            elif data_type == "datetime":
                start_date = constraints.get("start_date", datetime(2020, 1, 1))
                end_date = constraints.get("end_date", datetime.now())
                values.append(self.faker.date_time_between(start_date=start_date, end_date=end_date))

            elif data_type == "boolean":
                values.append(random.choice([True, False]))

            elif data_type == "uuid":
                values.append(str(uuid.uuid4()))

            else:
                values.append(self.faker.word())

        return values

    def _generate_pattern_data(self, schema: Dict[str, Any], count: int) -> pd.DataFrame:
        """Generate data following specific patterns."""
        # Implementation for pattern-based generation
        return self._generate_realistic_data(schema, count)

    def _generate_random_data(self, schema: Dict[str, Any], count: int) -> pd.DataFrame:
        """Generate completely random data."""
        # Implementation for random data generation
        return self._generate_realistic_data(schema, count)

    def _generate_template_data(self, schema: Dict[str, Any], count: int) -> pd.DataFrame:
        """Generate data using templates."""
        # Implementation for template-based generation
        return self._generate_realistic_data(schema, count)

    def add_template(self, name: str, template: str) -> None:
        """Add a data generation template."""
        self.templates[name] = template

    def add_pattern(self, name: str, pattern: Dict[str, Any]) -> None:
        """Add a data generation pattern."""
        self.patterns[name] = pattern


class MockScenarioManager:
    """Manager for handling complex mock scenarios."""

    def __init__(self, registry: MockRegistry):
        self.registry = registry
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        self.data_generator = AdvancedDataGenerator()

    def define_scenario(self, name: str, description: str,
                       mock_configurations: Dict[str, Dict[str, Any]]) -> None:
        """Define a new mock scenario."""
        self.scenarios[name] = {
            "description": description,
            "mock_configurations": mock_configurations,
            "created_at": datetime.now().isoformat()
        }

    def activate_scenario(self, name: str) -> None:
        """Activate a mock scenario."""
        if name not in self.scenarios:
            raise ValueError(f"Scenario not found: {name}")

        scenario = self.scenarios[name]
        self.registry.activate_scenario(name, scenario["mock_configurations"])

    def create_database_scenario(self, name: str, tables: Dict[str, Dict[str, Any]]) -> None:
        """Create a database mock scenario with generated data."""
        mock_config = MockConfiguration(
            mock_id=f"db_mock_{name}",
            mock_type=MockType.DATABASE,
            behavior=MockBehavior.DYNAMIC,
            name=f"Database Mock for {name}",
            description=f"Mock database for scenario: {name}"
        )

        db_mock = DatabaseMock(mock_config)

        for table_name, table_config in tables.items():
            schema = table_config.get("schema", {})
            row_count = table_config.get("row_count", 100)

            # Generate data for table
            table_data = self.data_generator.generate_data(schema, row_count)
            db_mock.add_table_data(table_name, table_data)

            # Add common query responses
            db_mock.add_query_response(
                f"SELECT.*FROM {table_name}",
                {
                    "success": True,
                    "data": table_data,
                    "rows_affected": len(table_data),
                    "columns": list(table_data.columns)
                }
            )

        self.registry.register_mock(db_mock)

        self.define_scenario(name, f"Database scenario with {len(tables)} tables", {
            db_mock.mock_id: {"active": True}
        })

    def create_service_scenario(self, name: str, services: Dict[str, Dict[str, Any]]) -> None:
        """Create a service mock scenario."""
        mock_configs = {}

        for service_name, service_config in services.items():
            mock_config = MockConfiguration(
                mock_id=f"service_mock_{service_name}",
                mock_type=MockType.SERVICE,
                behavior=MockBehavior.STATIC,
                name=f"Service Mock for {service_name}",
                description=f"Mock service: {service_name}"
            )

            service_mock = ServiceMock(mock_config)

            # Configure endpoints
            endpoints = service_config.get("endpoints", [])
            for endpoint in endpoints:
                service_mock.add_endpoint(
                    endpoint["method"],
                    endpoint["path"],
                    endpoint["response"],
                    endpoint.get("status_code", 200),
                    endpoint.get("headers", {})
                )

            self.registry.register_mock(service_mock)
            mock_configs[service_mock.mock_id] = {"active": True}

        self.define_scenario(name, f"Service scenario with {len(services)} services", mock_configs)

    def get_scenario_report(self, scenario_name: str) -> Dict[str, Any]:
        """Get a report on scenario usage."""
        if scenario_name not in self.scenarios:
            return {}

        scenario = self.scenarios[scenario_name]
        mock_ids = list(scenario["mock_configurations"].keys())

        # Collect invocation statistics
        invocation_stats = {}
        for mock_id in mock_ids:
            invocations = self.registry.get_invocation_history(mock_id)
            invocation_stats[mock_id] = {
                "total_invocations": len(invocations),
                "successful_invocations": len([i for i in invocations if i.success]),
                "failed_invocations": len([i for i in invocations if not i.success]),
                "avg_execution_time": sum(i.execution_time for i in invocations) / len(invocations) if invocations else 0
            }

        return {
            "scenario_name": scenario_name,
            "description": scenario["description"],
            "mock_count": len(mock_ids),
            "invocation_statistics": invocation_stats,
            "total_invocations": sum(stats["total_invocations"] for stats in invocation_stats.values()),
            "overall_success_rate": self._calculate_success_rate(invocation_stats)
        }

    def _calculate_success_rate(self, invocation_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall success rate across all mocks."""
        total_invocations = sum(stats["total_invocations"] for stats in invocation_stats.values())
        successful_invocations = sum(stats["successful_invocations"] for stats in invocation_stats.values())

        if total_invocations == 0:
            return 1.0

        return successful_invocations / total_invocations


# Global mock registry instance
mock_registry = MockRegistry()


# Convenience functions for creating mocks
def create_database_mock(name: str, **config) -> DatabaseMock:
    """Create a database mock with default configuration."""
    mock_config = MockConfiguration(
        mock_id=str(uuid.uuid4()),
        mock_type=MockType.DATABASE,
        behavior=MockBehavior.DYNAMIC,
        name=name,
        **config
    )

    mock = DatabaseMock(mock_config)
    mock.set_registry(mock_registry)
    mock_registry.register_mock(mock)
    return mock


def create_service_mock(name: str, **config) -> ServiceMock:
    """Create a service mock with default configuration."""
    mock_config = MockConfiguration(
        mock_id=str(uuid.uuid4()),
        mock_type=MockType.SERVICE,
        behavior=MockBehavior.STATIC,
        name=name,
        **config
    )

    mock = ServiceMock(mock_config)
    mock.set_registry(mock_registry)
    mock_registry.register_mock(mock)
    return mock


def create_data_source_mock(name: str, **config) -> DataSourceMock:
    """Create a data source mock with default configuration."""
    mock_config = MockConfiguration(
        mock_id=str(uuid.uuid4()),
        mock_type=MockType.DATA_SOURCE,
        behavior=MockBehavior.STATIC,
        name=name,
        **config
    )

    mock = DataSourceMock(mock_config)
    mock.set_registry(mock_registry)
    mock_registry.register_mock(mock)
    return mock


# Context manager for mock scenarios
class MockScenario:
    """Context manager for activating mock scenarios."""

    def __init__(self, scenario_name: str, scenario_manager: MockScenarioManager):
        self.scenario_name = scenario_name
        self.scenario_manager = scenario_manager

    def __enter__(self):
        self.scenario_manager.activate_scenario(self.scenario_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scenario_manager.registry.deactivate_scenario(self.scenario_name)
        return False