"""Tests for the main reporting engine."""

import pytest
import tempfile
from pathlib import Path

import pandas as pd

from sqltest.reporting.engine import ReportingEngine
from sqltest.reporting.models import (
    ReportConfiguration, ReportFormat, ReportType, ReportOptions,
    ReportGenerationResult
)


class TestReportingEngine:
    """Test ReportingEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = ReportingEngine()

        assert engine.analyzer is not None
        # Should have registered default generators
        available_formats = engine.get_available_formats()
        assert ReportFormat.JSON in available_formats
        assert ReportFormat.CSV in available_formats
        assert ReportFormat.HTML in available_formats

    def test_generate_report_basic(self, sample_dataframe, temp_output_dir):
        """Test basic report generation."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Basic Test Report",
            output_path=temp_output_dir / "basic_report.json"
        )

        result = engine.generate_report(sample_dataframe, config)

        assert isinstance(result, ReportGenerationResult)
        assert result.success is True
        assert result.output_path.exists()
        assert result.format == ReportFormat.JSON

    def test_generate_report_with_dict_data(self, sample_datasets, temp_output_dir):
        """Test report generation with dictionary of DataFrames."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.HTML,
            title="Multi-Dataset Report",
            output_path=temp_output_dir / "multi_dataset.html"
        )

        options = ReportOptions(include_charts=True)

        result = engine.generate_report(sample_datasets, config, options)

        assert result.success is True
        assert result.output_path.exists()
        assert result.metadata['total_datasets'] == len(sample_datasets)

    def test_generate_report_with_analysis(self, sample_dataframe, temp_output_dir):
        """Test report generation with data analysis."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.TECHNICAL,
            format=ReportFormat.HTML,
            title="Analysis Report",
            output_path=temp_output_dir / "analysis_report.html"
        )

        result = engine.generate_report(
            sample_dataframe,
            config,
            analyze=True,
            analysis_types=['trend_analysis', 'distribution_analysis']
        )

        assert result.success is True
        assert result.metadata['analysis_performed'] is True

    def test_generate_report_without_analysis(self, sample_dataframe, temp_output_dir):
        """Test report generation without analysis."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.CSV,
            title="No Analysis Report",
            output_path=temp_output_dir / "no_analysis.csv"
        )

        result = engine.generate_report(sample_dataframe, config, analyze=False)

        assert result.success is True
        assert result.metadata['analysis_performed'] is False

    def test_generate_multi_format_report(self, sample_dataframe, temp_output_dir):
        """Test generating reports in multiple formats."""
        engine = ReportingEngine()

        base_config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,  # Will be overridden
            title="Multi-Format Report",
            output_path=temp_output_dir / "multi_format"
        )

        formats = [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV]
        results = engine.generate_multi_format_report(sample_dataframe, base_config, formats)

        assert len(results) == 3
        for format_type in formats:
            assert format_type in results
            assert results[format_type].success is True

    def test_create_report_from_query_results(self, sample_query_results, temp_output_dir):
        """Test creating report from query results."""
        engine = ReportingEngine()

        result = engine.create_report_from_query_results(
            sample_query_results,
            title="Query Results Report",
            description="Report from SQL query results"
        )

        assert result.success is True

    def test_create_dashboard_report(self, sample_datasets, temp_output_dir):
        """Test creating dashboard-style report."""
        engine = ReportingEngine()

        result = engine.create_dashboard_report(
            sample_datasets,
            title="Executive Dashboard",
            include_charts=True
        )

        assert result.success is True
        assert result.format == ReportFormat.HTML

    def test_create_technical_report(self, sample_datasets, temp_output_dir):
        """Test creating technical report."""
        engine = ReportingEngine()

        result = engine.create_technical_report(
            sample_datasets,
            title="Technical Analysis"
        )

        assert result.success is True
        assert result.format == ReportFormat.HTML

    def test_export_findings_to_csv(self, sample_report_data, temp_output_dir):
        """Test exporting findings to CSV."""
        engine = ReportingEngine()

        output_path = temp_output_dir / "findings.csv"
        success = engine.export_findings_to_csv(sample_report_data, output_path)

        assert success is True
        assert output_path.exists()

        # Verify CSV content
        df = pd.read_csv(output_path)
        assert len(df) == len(sample_report_data.findings)
        assert 'Title' in df.columns
        assert 'Severity' in df.columns

    def test_export_findings_empty(self, temp_output_dir):
        """Test exporting empty findings list."""
        from sqltest.reporting.models import ReportData, ReportMetadata, ReportConfiguration

        engine = ReportingEngine()

        # Create report data with no findings
        metadata = ReportMetadata(
            title="Empty Findings",
            description="Test",
            generated_at=None,
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Empty Test"
        )

        empty_report_data = ReportData(metadata=metadata, configuration=config)

        output_path = temp_output_dir / "empty_findings.csv"
        success = engine.export_findings_to_csv(empty_report_data, output_path)

        assert success is False  # Should return False for empty findings

    def test_validate_data_for_reporting(self, sample_dataframe):
        """Test data validation for reporting."""
        engine = ReportingEngine()

        # Test with valid data
        warnings = engine.validate_data_for_reporting(sample_dataframe)
        assert isinstance(warnings, list)

        # Test with problematic data
        problematic_df = pd.DataFrame({
            'empty_col': [None] * 100,
            'high_null_col': [1 if i < 10 else None for i in range(100)],
            'high_cardinality': [f'unique_{i}' for i in range(100)]
        })

        warnings = engine.validate_data_for_reporting(problematic_df)
        assert len(warnings) > 0

        # Should detect high null percentages and high cardinality
        warning_text = ' '.join(warnings)
        assert 'null' in warning_text.lower() or 'cardinality' in warning_text.lower()

    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        engine = ReportingEngine()

        empty_df = pd.DataFrame()
        warnings = engine.validate_data_for_reporting(empty_df)

        assert len(warnings) > 0
        assert any('empty' in warning.lower() for warning in warnings)

    def test_validate_data_non_dataframe(self):
        """Test validation with non-DataFrame data."""
        engine = ReportingEngine()

        invalid_data = {"not_a_dataframe": [1, 2, 3]}
        warnings = engine.validate_data_for_reporting(invalid_data)

        assert len(warnings) > 0
        assert any('dataframe' in warning.lower() for warning in warnings)

    def test_get_report_statistics(self, sample_report_data):
        """Test getting report statistics."""
        engine = ReportingEngine()

        stats = engine.get_report_statistics(sample_report_data)

        assert 'metadata' in stats
        assert 'data_summary' in stats
        assert 'findings_summary' in stats
        assert 'performance' in stats

        # Check specific values
        assert stats['metadata']['title'] == sample_report_data.metadata.title
        assert stats['data_summary']['total_datasets'] == len(sample_report_data.raw_data)
        assert stats['findings_summary']['total_findings'] == len(sample_report_data.findings)

    def test_prepare_report_data_single_dataframe(self, sample_dataframe):
        """Test preparing report data from single DataFrame."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Test Report"
        )

        report_data = engine._prepare_report_data(sample_dataframe, config, None)

        assert 'main_dataset' in report_data.raw_data
        assert len(report_data.data_sources) == 1
        assert report_data.metadata.title == "Test Report"

    def test_prepare_report_data_dict_dataframes(self, sample_datasets):
        """Test preparing report data from dictionary of DataFrames."""
        engine = ReportingEngine()

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.HTML,
            title="Multi Dataset Report"
        )

        report_data = engine._prepare_report_data(sample_datasets, config, None)

        assert len(report_data.raw_data) == len(sample_datasets)
        assert len(report_data.data_sources) == len(sample_datasets)

        # Check data sources have schema info
        for data_source in report_data.data_sources:
            assert 'columns' in data_source.schema_info
            assert 'shape' in data_source.schema_info

    def test_get_format_specific_path(self):
        """Test generating format-specific paths."""
        engine = ReportingEngine()

        # Test with base path having extension
        base_path = Path("/tmp/report.txt")
        json_path = engine._get_format_specific_path(base_path, ReportFormat.JSON)
        assert json_path.suffix == ".json"

        # Test with base path as directory
        base_path = Path("/tmp/reports")
        html_path = engine._get_format_specific_path(base_path, ReportFormat.HTML)
        assert html_path.name == "report.html"

        # Test with None base path
        none_path = engine._get_format_specific_path(None, ReportFormat.CSV)
        assert none_path is None

    def test_error_handling_invalid_format(self, sample_dataframe):
        """Test error handling with invalid format."""
        engine = ReportingEngine()

        # Try to use a format that doesn't exist
        with pytest.raises(ValueError):
            config = ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format="invalid_format",  # This should cause validation error
                title="Invalid Format Test"
            )

    def test_error_handling_generation_failure(self, sample_dataframe, temp_output_dir):
        """Test error handling when generation fails."""
        engine = ReportingEngine()

        # Use a path that doesn't exist and can't be created (permission issue simulation)
        invalid_path = Path("/invalid/path/that/does/not/exist/report.json")

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Failure Test",
            output_path=invalid_path
        )

        result = engine.generate_report(sample_dataframe, config)

        # Should handle the error gracefully
        assert result.success is False
        assert result.error_message is not None

    def test_concurrent_report_generation(self, sample_datasets, temp_output_dir):
        """Test that multiple reports can be generated concurrently."""
        engine = ReportingEngine()

        # Generate multiple reports
        results = []
        for i in range(3):
            config = ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title=f"Concurrent Report {i}",
                output_path=temp_output_dir / f"concurrent_{i}.json"
            )

            result = engine.generate_report(sample_datasets, config)
            results.append(result)

        # All should succeed
        for result in results:
            assert result.success is True

    def test_memory_efficiency_large_data(self):
        """Test memory efficiency with larger datasets."""
        engine = ReportingEngine()

        # Create a larger dataset
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': range(10000),
            'category': ['A', 'B', 'C'] * 3334
        })

        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Large Data Test"
        )

        options = ReportOptions(
            include_raw_data=False,  # Don't include raw data to save memory
            max_rows_per_table=100   # Limit table sizes
        )

        result = engine.generate_report(large_df, config, options, analyze=False)

        assert result.success is True
        # Should complete without memory issues

    def test_report_generation_with_custom_options(self, sample_dataframe, temp_output_dir):
        """Test report generation with custom options."""
        engine = ReportingEngine()

        custom_options = ReportOptions(
            include_charts=False,
            include_raw_data=True,
            include_executive_summary=False,
            max_rows_per_table=50,
            chart_theme="dark",
            color_scheme="green"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.HTML,
            title="Custom Options Report",
            output_path=temp_output_dir / "custom_options.html"
        )

        result = engine.generate_report(sample_dataframe, config, custom_options)

        assert result.success is True
        assert result.output_path.exists()

    def test_get_available_formats(self):
        """Test getting available report formats."""
        engine = ReportingEngine()

        formats = engine.get_available_formats()

        assert isinstance(formats, list)
        assert len(formats) >= 3  # Should have at least JSON, CSV, HTML
        assert all(isinstance(fmt, ReportFormat) for fmt in formats)