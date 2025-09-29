"""Tests for report generators."""

import json
import pytest
from pathlib import Path

import pandas as pd

from sqltest.reporting.generators.json_generator import JSONReportGenerator
from sqltest.reporting.generators.csv_generator import CSVReportGenerator
from sqltest.reporting.generators.html_generator import HTMLReportGenerator
from sqltest.reporting.models import (
    ReportFormat, ReportOptions, ReportGenerationResult
)


class TestJSONReportGenerator:
    """Test JSON report generator."""

    def test_supported_format(self):
        """Test that generator supports JSON format."""
        generator = JSONReportGenerator()
        assert generator.supported_format == ReportFormat.JSON

    def test_generate_basic_report(self, sample_report_data, temp_output_dir):
        """Test generating a basic JSON report."""
        generator = JSONReportGenerator()

        # Set output path
        output_path = temp_output_dir / "test_report.json"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True
        assert result.output_path == output_path
        assert result.format == ReportFormat.JSON
        assert output_path.exists()

        # Verify JSON content
        with open(output_path, 'r') as f:
            content = json.load(f)

        assert 'metadata' in content
        assert 'configuration' in content
        assert 'sections' in content
        assert 'findings' in content
        assert content['metadata']['title'] == sample_report_data.metadata.title

    def test_generate_with_raw_data(self, sample_report_data, temp_output_dir):
        """Test generating JSON report with raw data included."""
        options = ReportOptions(include_raw_data=True)
        generator = JSONReportGenerator(options)

        output_path = temp_output_dir / "test_report_with_data.json"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True

        # Verify raw data is included
        with open(output_path, 'r') as f:
            content = json.load(f)

        assert 'raw_data' in content
        assert content['raw_data'] is not None
        assert 'main_data' in content['raw_data']

    def test_generate_compact(self, sample_report_data, temp_output_dir):
        """Test generating compact JSON report."""
        generator = JSONReportGenerator()

        output_path = temp_output_dir / "compact_report.json"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate_compact(sample_report_data)

        assert result.success is True
        assert output_path.exists()

        # Compact JSON should be smaller (no formatting)
        with open(output_path, 'r') as f:
            content = f.read()
            # Should not contain pretty-printing spaces
            assert '\n' not in content

    def test_validation_error(self, temp_output_dir):
        """Test handling of validation errors."""
        from sqltest.reporting.models import ReportData, ReportMetadata, ReportConfiguration, ReportType

        # Create invalid report data (no title)
        invalid_metadata = ReportMetadata(
            title="",  # Empty title should fail validation
            description="Test",
            generated_at=None,
            generated_by="test"
        )

        invalid_config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title=""
        )

        invalid_data = ReportData(
            metadata=invalid_metadata,
            configuration=invalid_config
        )

        generator = JSONReportGenerator()
        result = generator.generate(invalid_data)

        assert result.success is False
        assert result.error_message is not None


class TestCSVReportGenerator:
    """Test CSV report generator."""

    def test_supported_format(self):
        """Test that generator supports CSV format."""
        generator = CSVReportGenerator()
        assert generator.supported_format == ReportFormat.CSV

    def test_generate_single_dataset(self, sample_dataframe, temp_output_dir):
        """Test generating CSV from single dataset."""
        from sqltest.reporting.models import (
            ReportData, ReportMetadata, ReportConfiguration, ReportType
        )

        # Create simple report data with single dataset
        metadata = ReportMetadata(
            title="CSV Test Report",
            description="Test CSV generation",
            generated_at=None,
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.CSV,
            title="CSV Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'single_dataset': sample_dataframe}
        )

        generator = CSVReportGenerator()
        output_path = temp_output_dir / "single_dataset.csv"
        config.output_path = output_path

        result = generator.generate(report_data)

        assert result.success is True
        assert output_path.exists()

        # Verify CSV content
        df_result = pd.read_csv(output_path, comment='#')
        assert len(df_result) == len(sample_dataframe)
        assert list(df_result.columns) == list(sample_dataframe.columns)

    def test_generate_multiple_datasets(self, sample_datasets, temp_output_dir):
        """Test generating CSV from multiple datasets."""
        from sqltest.reporting.models import (
            ReportData, ReportMetadata, ReportConfiguration, ReportType
        )

        metadata = ReportMetadata(
            title="Multi CSV Test",
            description="Test multiple dataset CSV generation",
            generated_at=None,
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.CSV,
            title="Multi CSV Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data=sample_datasets
        )

        options = ReportOptions(include_raw_data=True)
        generator = CSVReportGenerator(options)

        output_path = temp_output_dir / "multi_dataset.csv"
        config.output_path = output_path

        result = generator.generate(report_data)

        assert result.success is True
        # Should create a directory with multiple CSV files
        assert output_path.parent.exists()

    def test_generate_pivot_table(self, sample_dataframe, temp_output_dir):
        """Test generating pivot table CSV."""
        from sqltest.reporting.models import (
            ReportData, ReportMetadata, ReportConfiguration, ReportType
        )

        metadata = ReportMetadata(
            title="Pivot Test",
            description="Test pivot table generation",
            generated_at=None,
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.CSV,
            title="Pivot Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'pivot_data': sample_dataframe}
        )

        generator = CSVReportGenerator()
        output_path = temp_output_dir / "pivot_test.csv"
        config.output_path = output_path

        result = generator.generate_pivot_table(
            report_data=report_data,
            dataset_name='pivot_data',
            index_cols=['category'],
            value_cols=['value'],
            aggfunc='sum'
        )

        assert result.success is True
        pivot_path = output_path.with_name("pivot_test_pivot.csv")
        assert pivot_path.exists()


class TestHTMLReportGenerator:
    """Test HTML report generator."""

    def test_supported_format(self):
        """Test that generator supports HTML format."""
        generator = HTMLReportGenerator()
        assert generator.supported_format == ReportFormat.HTML

    def test_generate_basic_html_report(self, sample_report_data, temp_output_dir):
        """Test generating a basic HTML report."""
        generator = HTMLReportGenerator()

        output_path = temp_output_dir / "test_report.html"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True
        assert result.output_path == output_path
        assert result.format == ReportFormat.HTML
        assert output_path.exists()

        # Verify HTML content
        with open(output_path, 'r') as f:
            content = f.read()

        assert '<!DOCTYPE html>' in content
        assert sample_report_data.metadata.title in content
        assert 'bootstrap' in content.lower()  # Should include Bootstrap CSS

    def test_static_assets_creation(self, sample_report_data, temp_output_dir):
        """Test that static assets are created."""
        generator = HTMLReportGenerator()

        output_path = temp_output_dir / "test_report.html"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True

        # Check that assets directory is created
        assets_dir = output_path.parent / "assets"
        assert assets_dir.exists()
        assert (assets_dir / "report.css").exists()
        assert (assets_dir / "report.js").exists()

    def test_template_functions(self):
        """Test template helper functions."""
        generator = HTMLReportGenerator()

        # Test number formatting
        assert generator._format_number(1234.567, 2) == "1,234.57"
        assert generator._format_number(1000, 0) == "1,000"

        # Test percentage formatting
        assert generator._format_percentage(0.756, 1) == "75.6%"
        assert generator._format_percentage(0.1234, 2) == "12.34%"

        # Test severity colors
        assert generator._get_severity_color("critical") == "#dc3545"
        assert generator._get_severity_color("high") == "#fd7e14"
        assert generator._get_severity_color("unknown") == "#6c757d"

    def test_dataframe_to_html(self, sample_dataframe):
        """Test DataFrame to HTML conversion."""
        generator = HTMLReportGenerator()

        html_table = generator._dataframe_to_html(sample_dataframe.head())

        assert '<table' in html_table
        assert 'table-striped' in html_table
        assert 'table-hover' in html_table
        assert sample_dataframe.columns[0] in html_table

    def test_chart_config_generation(self):
        """Test chart configuration generation."""
        from sqltest.reporting.models import ChartData

        generator = HTMLReportGenerator()

        chart_data = ChartData(
            chart_type="bar",
            title="Test Chart",
            data={"labels": ["A", "B"], "values": [1, 2]},
            options={"responsive": True}
        )

        config = generator._generate_chart_config(chart_data)

        assert config['type'] == "bar"
        assert config['data'] == chart_data.data
        assert config['options']['responsive'] is True
        assert config['options']['plugins']['title']['text'] == "Test Chart"

    def test_context_preparation(self, sample_report_data):
        """Test template context preparation."""
        generator = HTMLReportGenerator()

        context = generator._prepare_template_context(sample_report_data)

        assert 'report' in context
        assert 'metadata' in context
        assert 'findings_by_severity' in context
        assert 'chart_data' in context
        assert 'table_data' in context

        # Test findings grouping
        findings_by_severity = context['findings_by_severity']
        assert 'critical' in findings_by_severity
        assert len(findings_by_severity['critical']) == 1

    def test_html_generation_with_charts(self, sample_report_data, temp_output_dir):
        """Test HTML generation with charts enabled."""
        options = ReportOptions(include_charts=True)
        generator = HTMLReportGenerator(options)

        output_path = temp_output_dir / "chart_report.html"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True

        with open(output_path, 'r') as f:
            content = f.read()

        # Should include Chart.js
        assert 'chart.js' in content.lower()
        assert 'canvas' in content

    def test_html_generation_without_charts(self, sample_report_data, temp_output_dir):
        """Test HTML generation with charts disabled."""
        options = ReportOptions(include_charts=False)
        generator = HTMLReportGenerator(options)

        output_path = temp_output_dir / "no_chart_report.html"
        sample_report_data.configuration.output_path = output_path

        result = generator.generate(sample_report_data)

        assert result.success is True

        with open(output_path, 'r') as f:
            content = f.read()

        # Should still include basic HTML structure
        assert '<!DOCTYPE html>' in content
        assert sample_report_data.metadata.title in content


class TestGeneratorValidation:
    """Test validation across all generators."""

    def test_all_generators_implement_interface(self):
        """Test that all generators implement the required interface."""
        generators = [JSONReportGenerator, CSVReportGenerator, HTMLReportGenerator]

        for generator_class in generators:
            generator = generator_class()

            # Should have supported_format property
            assert hasattr(generator, 'supported_format')
            assert isinstance(generator.supported_format, ReportFormat)

            # Should have generate method
            assert hasattr(generator, 'generate')
            assert callable(generator.generate)

    def test_generators_handle_empty_data(self, temp_output_dir):
        """Test that generators handle empty datasets gracefully."""
        from sqltest.reporting.models import (
            ReportData, ReportMetadata, ReportConfiguration, ReportType
        )

        # Create report data with empty DataFrame
        metadata = ReportMetadata(
            title="Empty Data Test",
            description="Test with empty data",
            generated_at=None,
            generated_by="test"
        )

        empty_df = pd.DataFrame()

        generators_configs = [
            (JSONReportGenerator(), ReportFormat.JSON, "empty.json"),
            (CSVReportGenerator(), ReportFormat.CSV, "empty.csv"),
            (HTMLReportGenerator(), ReportFormat.HTML, "empty.html")
        ]

        for generator, format_type, filename in generators_configs:
            config = ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=format_type,
                title="Empty Data Test"
            )

            report_data = ReportData(
                metadata=metadata,
                configuration=config,
                raw_data={'empty': empty_df}
            )

            output_path = temp_output_dir / filename
            config.output_path = output_path

            result = generator.generate(report_data)

            # Should handle empty data gracefully
            assert result.success is True
            assert output_path.exists()

    def test_generators_with_options(self, sample_report_data, temp_output_dir):
        """Test generators with different options."""
        options = ReportOptions(
            include_charts=False,
            include_raw_data=False,
            max_rows_per_table=10
        )

        generators_configs = [
            (JSONReportGenerator(options), ReportFormat.JSON, "options.json"),
            (CSVReportGenerator(options), ReportFormat.CSV, "options.csv"),
            (HTMLReportGenerator(options), ReportFormat.HTML, "options.html")
        ]

        for generator, format_type, filename in generators_configs:
            sample_report_data.configuration.format = format_type
            output_path = temp_output_dir / filename
            sample_report_data.configuration.output_path = output_path

            result = generator.generate(sample_report_data)

            assert result.success is True
            assert result.output_path == output_path