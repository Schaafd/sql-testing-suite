"""Tests for reporting models and data structures."""

import pytest
from datetime import datetime
from pathlib import Path

import pandas as pd

from sqltest.reporting.models import (
    ReportData, ReportConfiguration, ReportMetadata, ReportOptions,
    ReportFormat, ReportType, SeverityLevel, Finding, DataSource,
    ExecutionMetrics, ReportSection, ChartData, ReportGenerationResult
)


class TestReportConfiguration:
    """Test ReportConfiguration model."""

    def test_basic_creation(self):
        """Test basic configuration creation."""
        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Test Report"
        )

        assert config.report_type == ReportType.SUMMARY
        assert config.format == ReportFormat.JSON
        assert config.title == "Test Report"
        assert config.description is None
        assert config.output_path is None

    def test_with_output_path(self):
        """Test configuration with output path."""
        path_str = "/tmp/test_report.html"
        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.HTML,
            title="Test Report",
            output_path=path_str
        )

        assert isinstance(config.output_path, Path)
        assert str(config.output_path) == path_str

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError):
            ReportConfiguration(
                report_type="invalid_type",
                format=ReportFormat.HTML,
                title="Test"
            )


class TestReportMetadata:
    """Test ReportMetadata model."""

    def test_creation_with_defaults(self):
        """Test metadata creation with default values."""
        metadata = ReportMetadata(
            title="Test Report",
            description="Test Description",
            generated_at=datetime.now(),
            generated_by="test_user"
        )

        assert metadata.title == "Test Report"
        assert metadata.version == "1.0"
        assert isinstance(metadata.tags, list)
        assert len(metadata.tags) == 0
        assert isinstance(metadata.custom_fields, dict)

    def test_with_custom_fields(self):
        """Test metadata with custom fields."""
        custom_fields = {"environment": "test", "region": "us-east-1"}
        metadata = ReportMetadata(
            title="Test",
            description="Test",
            generated_at=datetime.now(),
            generated_by="test",
            custom_fields=custom_fields
        )

        assert metadata.custom_fields == custom_fields


class TestFinding:
    """Test Finding model."""

    def test_basic_finding(self):
        """Test basic finding creation."""
        finding = Finding(
            id="test_finding",
            title="Test Issue",
            description="This is a test issue",
            severity=SeverityLevel.MEDIUM,
            category="Test"
        )

        assert finding.id == "test_finding"
        assert finding.severity == SeverityLevel.MEDIUM
        assert isinstance(finding.created_at, datetime)

    def test_finding_with_details(self):
        """Test finding with additional details."""
        details = {"table": "users", "column": "email", "count": 5}
        recommendations = ["Fix validation", "Update schema"]
        affected_objects = ["users.email", "profiles.email"]

        finding = Finding(
            id="detailed_finding",
            title="Email Validation Issue",
            description="Invalid email formats found",
            severity=SeverityLevel.HIGH,
            category="Data Quality",
            details=details,
            recommendations=recommendations,
            affected_objects=affected_objects
        )

        assert finding.details == details
        assert finding.recommendations == recommendations
        assert finding.affected_objects == affected_objects


class TestReportSection:
    """Test ReportSection model."""

    def test_basic_section(self):
        """Test basic section creation."""
        section = ReportSection(
            id="test_section",
            title="Test Section",
            content="<p>Test content</p>",
            order=1
        )

        assert section.id == "test_section"
        assert section.order == 1
        assert len(section.subsections) == 0
        assert len(section.charts) == 0

    def test_section_with_subsections(self):
        """Test section with subsections."""
        subsection = ReportSection(
            id="subsection",
            title="Subsection",
            content="Subsection content",
            order=1
        )

        main_section = ReportSection(
            id="main_section",
            title="Main Section",
            content="Main content",
            order=1,
            subsections=[subsection]
        )

        assert len(main_section.subsections) == 1
        assert main_section.subsections[0].id == "subsection"


class TestReportData:
    """Test ReportData model."""

    def test_add_section(self, sample_report_data):
        """Test adding sections to report data."""
        initial_count = len(sample_report_data.sections)

        new_section = ReportSection(
            id="new_section",
            title="New Section",
            content="New content",
            order=99
        )

        sample_report_data.add_section(new_section)
        assert len(sample_report_data.sections) == initial_count + 1

        # Check that sections are sorted by order
        orders = [section.order for section in sample_report_data.sections]
        assert orders == sorted(orders)

    def test_get_section(self, sample_report_data):
        """Test getting section by ID."""
        section = sample_report_data.get_section("executive_summary")
        assert section is not None
        assert section.title == "Executive Summary"

        # Test non-existent section
        assert sample_report_data.get_section("nonexistent") is None

    def test_add_finding(self, sample_report_data):
        """Test adding findings to report data."""
        initial_count = len(sample_report_data.findings)

        new_finding = Finding(
            id="new_finding",
            title="New Issue",
            description="New issue description",
            severity=SeverityLevel.LOW,
            category="Test"
        )

        sample_report_data.add_finding(new_finding)
        assert len(sample_report_data.findings) == initial_count + 1

    def test_get_findings_by_severity(self, sample_report_data):
        """Test filtering findings by severity."""
        critical_findings = sample_report_data.get_findings_by_severity(SeverityLevel.CRITICAL)
        assert len(critical_findings) == 1
        assert critical_findings[0].severity == SeverityLevel.CRITICAL

        info_findings = sample_report_data.get_findings_by_severity(SeverityLevel.INFO)
        assert len(info_findings) == 1

    def test_get_critical_findings(self, sample_report_data):
        """Test getting critical findings."""
        critical_findings = sample_report_data.get_critical_findings()
        assert len(critical_findings) == 1
        assert critical_findings[0].severity == SeverityLevel.CRITICAL


class TestChartData:
    """Test ChartData model."""

    def test_basic_chart(self):
        """Test basic chart creation."""
        chart = ChartData(
            chart_type="bar",
            title="Test Chart",
            data={"labels": ["A", "B"], "values": [1, 2]}
        )

        assert chart.chart_type == "bar"
        assert chart.title == "Test Chart"
        assert chart.width is None
        assert chart.height is None

    def test_chart_with_dimensions(self):
        """Test chart with specified dimensions."""
        chart = ChartData(
            chart_type="line",
            title="Test Line Chart",
            data={"x": [1, 2, 3], "y": [1, 4, 9]},
            width=800,
            height=600
        )

        assert chart.width == 800
        assert chart.height == 600


class TestDataSource:
    """Test DataSource model."""

    def test_basic_data_source(self):
        """Test basic data source creation."""
        source = DataSource(
            name="test_db",
            type="PostgreSQL"
        )

        assert source.name == "test_db"
        assert source.type == "PostgreSQL"
        assert source.query_count == 0
        assert source.last_accessed is None

    def test_data_source_with_metadata(self):
        """Test data source with additional metadata."""
        schema_info = {"tables": ["users", "orders"], "views": ["summary"]}
        timestamp = datetime.now()

        source = DataSource(
            name="analytics_db",
            type="MySQL",
            query_count=10,
            last_accessed=timestamp,
            schema_info=schema_info
        )

        assert source.query_count == 10
        assert source.last_accessed == timestamp
        assert source.schema_info == schema_info


class TestExecutionMetrics:
    """Test ExecutionMetrics model."""

    def test_basic_metrics(self):
        """Test basic metrics creation."""
        metrics = ExecutionMetrics(
            execution_time=5.5,
            memory_usage=128.0,
            queries_executed=3,
            rows_processed=1000
        )

        assert metrics.execution_time == 5.5
        assert metrics.memory_usage == 128.0
        assert metrics.cache_hit_rate == 0.0  # Default value
        assert metrics.errors_encountered == 0  # Default value

    def test_metrics_with_all_fields(self):
        """Test metrics with all fields specified."""
        metrics = ExecutionMetrics(
            execution_time=2.3,
            memory_usage=64.5,
            queries_executed=5,
            rows_processed=2500,
            cache_hit_rate=0.85,
            errors_encountered=2
        )

        assert metrics.cache_hit_rate == 0.85
        assert metrics.errors_encountered == 2


class TestReportOptions:
    """Test ReportOptions model."""

    def test_default_options(self):
        """Test default option values."""
        options = ReportOptions()

        assert options.include_charts is True
        assert options.include_raw_data is False
        assert options.include_executive_summary is True
        assert options.max_rows_per_table == 1000
        assert options.chart_theme == "default"

    def test_custom_options(self):
        """Test custom option values."""
        options = ReportOptions(
            include_charts=False,
            include_raw_data=True,
            max_rows_per_table=500,
            chart_theme="dark",
            color_scheme="red"
        )

        assert options.include_charts is False
        assert options.include_raw_data is True
        assert options.max_rows_per_table == 500
        assert options.chart_theme == "dark"
        assert options.color_scheme == "red"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in options."""
        options = ReportOptions(
            custom_field="custom_value",
            another_field=42
        )

        # Should not raise an error due to Config.extra = "allow"
        assert hasattr(options, 'custom_field')


class TestReportGenerationResult:
    """Test ReportGenerationResult model."""

    def test_successful_result(self, temp_output_dir):
        """Test successful generation result."""
        output_path = temp_output_dir / "test_report.html"
        output_path.write_text("<html><body>Test</body></html>")

        result = ReportGenerationResult(
            success=True,
            output_path=output_path,
            format=ReportFormat.HTML,
            file_size=output_path.stat().st_size,
            generation_time=2.5
        )

        assert result.success is True
        assert result.output_path == output_path
        assert result.format == ReportFormat.HTML
        assert result.file_size > 0
        assert result.generation_time == 2.5
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed generation result."""
        result = ReportGenerationResult(
            success=False,
            error_message="Test error occurred",
            generation_time=0.1
        )

        assert result.success is False
        assert result.error_message == "Test error occurred"
        assert result.output_path is None
        assert result.format is None

    def test_result_with_warnings(self, temp_output_dir):
        """Test result with warnings."""
        output_path = temp_output_dir / "test_report.json"
        output_path.write_text('{"test": "data"}')

        warnings = ["Data quality issue detected", "Performance warning"]
        metadata = {"rows_processed": 1000, "execution_time": 5.0}

        result = ReportGenerationResult(
            success=True,
            output_path=output_path,
            format=ReportFormat.JSON,
            file_size=output_path.stat().st_size,
            generation_time=5.0,
            warnings=warnings,
            metadata=metadata
        )

        assert len(result.warnings) == 2
        assert result.metadata == metadata