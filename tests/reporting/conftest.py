"""Test fixtures and configuration for reporting module tests."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

from sqltest.reporting.models import (
    ReportData, ReportConfiguration, ReportMetadata, ReportOptions,
    ReportFormat, ReportType, Finding, SeverityLevel, DataSource,
    ExecutionMetrics, ReportSection, ChartData
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    n_rows = 100
    return pd.DataFrame({
        'id': range(1, n_rows + 1),
        'name': [f'Item {i}' for i in range(1, n_rows + 1)],
        'value': [i * 10 + (i % 5) for i in range(1, n_rows + 1)],
        'category': (['A', 'B', 'C'] * (n_rows // 3 + 1))[:n_rows],
        'date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'flag': ([True, False] * (n_rows // 2 + 1))[:n_rows]
    })


@pytest.fixture
def sample_datasets(sample_dataframe):
    """Create sample datasets for testing."""
    return {
        'main_data': sample_dataframe,
        'secondary_data': pd.DataFrame({
            'metric': ['CPU', 'Memory', 'Disk', 'Network'],
            'value': [75.5, 82.3, 45.2, 23.8],
            'threshold': [80, 85, 70, 50],
            'status': ['OK', 'Warning', 'OK', 'OK']
        })
    }


@pytest.fixture
def sample_report_metadata():
    """Create sample report metadata."""
    return ReportMetadata(
        title="Test Report",
        description="A test report for unit testing",
        generated_at=datetime(2023, 6, 15, 10, 30, 0),
        generated_by="pytest",
        version="1.0",
        tags=["test", "sample"],
        custom_fields={"environment": "test"}
    )


@pytest.fixture
def sample_report_config():
    """Create sample report configuration."""
    return ReportConfiguration(
        report_type=ReportType.DETAILED,
        format=ReportFormat.HTML,
        title="Test Report",
        description="Test report configuration",
        include_sections=["summary", "findings"],
        parameters={"test_param": "test_value"}
    )


@pytest.fixture
def sample_report_options():
    """Create sample report options."""
    return ReportOptions(
        include_charts=True,
        include_raw_data=True,
        include_executive_summary=True,
        max_rows_per_table=100,
        chart_theme="default",
        color_scheme="blue"
    )


@pytest.fixture
def sample_findings():
    """Create sample findings."""
    return [
        Finding(
            id="finding_1",
            title="Critical Issue Found",
            description="This is a critical issue that needs immediate attention",
            severity=SeverityLevel.CRITICAL,
            category="Data Quality",
            details={"table": "users", "column": "email"},
            recommendations=["Fix data validation", "Update constraints"],
            affected_objects=["users.email"],
            created_at=datetime(2023, 6, 15, 10, 0, 0)
        ),
        Finding(
            id="finding_2",
            title="Performance Warning",
            description="Query execution time is higher than expected",
            severity=SeverityLevel.MEDIUM,
            category="Performance",
            details={"execution_time": 5.2, "query": "SELECT * FROM large_table"},
            recommendations=["Add indexes", "Optimize query"],
            affected_objects=["large_table"],
            created_at=datetime(2023, 6, 15, 10, 15, 0)
        ),
        Finding(
            id="finding_3",
            title="Info: Data Statistics",
            description="Statistical summary of the data",
            severity=SeverityLevel.INFO,
            category="Statistics",
            details={"mean": 55.5, "std": 29.0, "count": 100},
            recommendations=[],
            affected_objects=[],
            created_at=datetime(2023, 6, 15, 10, 30, 0)
        )
    ]


@pytest.fixture
def sample_data_sources():
    """Create sample data sources."""
    return [
        DataSource(
            name="main_database",
            type="PostgreSQL",
            connection_string="postgresql://localhost:5432/testdb",
            query_count=5,
            last_accessed=datetime(2023, 6, 15, 10, 0, 0),
            schema_info={"tables": ["users", "orders"], "views": ["user_summary"]}
        ),
        DataSource(
            name="analytics_db",
            type="MySQL",
            query_count=3,
            last_accessed=datetime(2023, 6, 15, 9, 45, 0),
            schema_info={"tables": ["metrics", "events"]}
        )
    ]


@pytest.fixture
def sample_execution_metrics():
    """Create sample execution metrics."""
    return ExecutionMetrics(
        execution_time=2.5,
        memory_usage=256.7,
        queries_executed=8,
        rows_processed=1000,
        cache_hit_rate=0.75,
        errors_encountered=0
    )


@pytest.fixture
def sample_chart_data():
    """Create sample chart data."""
    return [
        ChartData(
            chart_type="bar",
            title="Category Distribution",
            data={
                "labels": ["A", "B", "C"],
                "datasets": [{
                    "label": "Count",
                    "data": [34, 33, 33],
                    "backgroundColor": ["#ff6384", "#36a2eb", "#ffce56"]
                }]
            },
            options={"responsive": True},
            width=400,
            height=300
        ),
        ChartData(
            chart_type="line",
            title="Value Trend",
            data={
                "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                "datasets": [{
                    "label": "Values",
                    "data": [10, 20, 15, 25, 30],
                    "borderColor": "#36a2eb",
                    "tension": 0.1
                }]
            },
            options={"responsive": True, "scales": {"y": {"beginAtZero": True}}}
        )
    ]


@pytest.fixture
def sample_report_sections(sample_chart_data, sample_dataframe):
    """Create sample report sections."""
    return [
        ReportSection(
            id="executive_summary",
            title="Executive Summary",
            content="<p>This is the executive summary of the report.</p>",
            order=1,
            charts=[sample_chart_data[0]],
            tables=[sample_dataframe.head(10)],
            findings=[]
        ),
        ReportSection(
            id="detailed_analysis",
            title="Detailed Analysis",
            content="<p>This section contains detailed analysis results.</p>",
            order=2,
            charts=[sample_chart_data[1]],
            tables=[sample_dataframe.tail(10)],
            findings=[]
        )
    ]


@pytest.fixture
def sample_report_data(sample_report_metadata, sample_report_config,
                      sample_data_sources, sample_execution_metrics,
                      sample_findings, sample_report_sections, sample_datasets):
    """Create a complete sample ReportData object."""
    return ReportData(
        metadata=sample_report_metadata,
        configuration=sample_report_config,
        data_sources=sample_data_sources,
        execution_metrics=sample_execution_metrics,
        sections=sample_report_sections,
        findings=sample_findings,
        summary_statistics={
            "total_rows": 100,
            "total_columns": 6,
            "null_percentage": 0.05,
            "numeric_columns": 2
        },
        raw_data=sample_datasets
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    class MockConnection:
        def __init__(self):
            self.connected = True

        def execute(self, query: str) -> pd.DataFrame:
            # Return mock data based on query
            if "users" in query.lower():
                return pd.DataFrame({
                    'id': [1, 2, 3],
                    'name': ['Alice', 'Bob', 'Charlie'],
                    'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
                })
            else:
                return pd.DataFrame({'result': ['mock_data']})

        def close(self):
            self.connected = False

    return MockConnection()


@pytest.fixture
def sample_query_results(sample_dataframe):
    """Create sample query results for testing."""
    return {
        'data': {
            'query_1_result': sample_dataframe,
            'query_2_result': pd.DataFrame({
                'summary_metric': ['Total Records', 'Average Value', 'Max Value'],
                'value': [100, 55.5, 1000]
            })
        },
        'execution_info': {
            'execution_time': 1.23,
            'memory_usage': 45.6,
            'query_count': 2,
            'cache_hits': 1
        },
        'metadata': {
            'database': 'test_db',
            'schema': 'public',
            'timestamp': datetime(2023, 6, 15, 10, 0, 0)
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Any setup that should happen before each test
    yield
    # Any cleanup that should happen after each test