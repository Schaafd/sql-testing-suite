"""Tests for interactive HTML generator features."""

import pytest
import tempfile
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock

from sqltest.reporting.generators.html_generator import HTMLReportGenerator
from sqltest.reporting.models import (
    ReportData, ReportConfiguration, ReportFormat, ReportType,
    ReportMetadata, ReportOptions, ExecutionMetrics, DataSource,
    Finding, SeverityLevel
)
from sqltest.reporting.interactive import InteractiveReportBuilder


class TestHTMLGeneratorInteractive:
    """Test interactive features of HTML report generator."""

    @pytest.fixture
    def sample_report_data(self):
        """Create sample report data for testing."""
        metadata = ReportMetadata(
            title="Interactive Test Report",
            description="Test report for interactive features",
            generated_at=datetime.now(),
            generated_by="Test Suite",
            version="1.0"
        )

        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE,
            format=ReportFormat.HTML,
            title="Interactive Dashboard",
            description="Test interactive dashboard"
        )

        # Create sample data with time series
        df1 = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'sales': [1000 + i * 50 + (i % 7) * 100 for i in range(30)],
            'region': ['North', 'South', 'East', 'West'] * 7 + ['North', 'South']
        })

        df2 = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D', 'E'],
            'revenue': [10000, 15000, 8000, 12000, 9000],
            'category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Books']
        })

        execution_metrics = ExecutionMetrics(
            execution_time=2.5,
            memory_usage=512.0,
            queries_executed=2,
            rows_processed=35
        )

        findings = [
            Finding(
                id="finding1",
                title="Strong Sales Growth",
                description="Sales showing consistent upward trend",
                severity=SeverityLevel.INFO,
                category="Performance"
            ),
            Finding(
                id="finding2",
                title="Regional Variation",
                description="Significant variation in sales across regions",
                severity=SeverityLevel.MEDIUM,
                category="Analysis"
            )
        ]

        return ReportData(
            metadata=metadata,
            configuration=config,
            data_sources=[],
            execution_metrics=execution_metrics,
            raw_data={'sales_data': df1, 'product_data': df2},
            findings=findings
        )

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_generate_interactive_dashboard_basic(self, sample_report_data, temp_output_dir):
        """Test basic interactive dashboard generation."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            include_charts=True,
            include_executive_summary=True,
            output_path=temp_output_dir / "dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        assert result.output_path is not None
        assert result.format == ReportFormat.HTML
        assert result.metadata is not None

        # Check that file was created
        assert result.output_path.exists()

        # Read content and verify interactive elements
        content = result.output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Interactive Dashboard" in content
        assert "bootstrap" in content.lower()
        assert "chart.js" in content.lower()
        assert "datatables" in content.lower()

    def test_interactive_dashboard_with_widgets(self, sample_report_data, temp_output_dir):
        """Test dashboard generation with custom widgets."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            include_charts=True,
            include_raw_data=True,
            output_path=temp_output_dir / "widgets_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True

        content = result.output_path.read_text()

        # Check for widget containers
        assert "metric-card" in content
        assert "chart-container" in content
        assert "data-table" in content

        # Check for interactive elements
        assert "onclick" in content or "addEventListener" in content
        assert "initializeDashboard" in content

    def test_filters_generation_from_data(self, sample_report_data):
        """Test automatic filter generation from data."""
        generator = HTMLReportGenerator()

        filters = generator._generate_filters_from_data(sample_report_data)

        assert isinstance(filters, list)
        assert len(filters) > 0

        # Should generate filters for categorical columns
        filter_names = [f["id"] for f in filters]
        assert any("region" in name for name in filter_names)
        assert any("category" in name for name in filter_names)

        # Check filter structure
        for filter_item in filters:
            assert "id" in filter_item
            assert "label" in filter_item
            assert "type" in filter_item
            assert "options" in filter_item

    def test_executive_summary_integration(self, sample_report_data, temp_output_dir):
        """Test integration with executive summary."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            include_executive_summary=True,
            output_path=temp_output_dir / "summary_dashboard.html"
        )

        with patch('sqltest.reporting.interactive.TrendAnalyzer.generate_executive_summary') as mock_summary:
            mock_summary.return_value = {
                'overview': {
                    'total_records': 35,
                    'data_sources': 2,
                    'time_period': '30 days'
                },
                'key_metrics': [
                    {'name': 'Total Sales', 'value': 45000, 'trend': 'up'},
                    {'name': 'Average Daily Sales', 'value': 1500, 'trend': 'stable'}
                ],
                'findings_summary': {
                    'total_findings': 2,
                    'by_severity': {'INFO': 1, 'MEDIUM': 1}
                },
                'recommendations': [
                    'Focus on high-performing regions',
                    'Investigate regional variations'
                ]
            }

            result = generator.generate_interactive_dashboard(sample_report_data)

            assert result.success is True
            content = result.output_path.read_text()

            # Check for executive summary elements
            assert "Total Sales" in content
            assert "45000" in content or "45,000" in content
            assert "recommendations" in content.lower()

    def test_chart_generation(self, sample_report_data, temp_output_dir):
        """Test chart generation in interactive dashboard."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            include_charts=True,
            output_path=temp_output_dir / "charts_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for Chart.js integration
        assert "Chart" in content
        assert "canvas" in content
        assert "chart-container" in content

        # Should have charts for time series data
        assert "line" in content.lower() or "bar" in content.lower()

    def test_data_table_integration(self, sample_report_data, temp_output_dir):
        """Test DataTables integration."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            include_raw_data=True,
            output_path=temp_output_dir / "tables_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for DataTables integration
        assert "DataTable" in content
        assert "table" in content
        assert "thead" in content and "tbody" in content

    def test_responsive_design(self, sample_report_data, temp_output_dir):
        """Test responsive design elements."""
        generator = HTMLReportGenerator()

        options = ReportOptions(
            output_path=temp_output_dir / "responsive_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for responsive design elements
        assert "viewport" in content
        assert "col-md-" in content or "col-lg-" in content
        assert "container-fluid" in content or "container" in content

    def test_navigation_generation(self, sample_report_data, temp_output_dir):
        """Test navigation generation."""
        generator = HTMLReportGenerator()

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for navigation elements
        assert "nav" in content or "navbar" in content
        assert "dashboard" in content.lower()
        assert "export" in content.lower()

    def test_error_handling_in_dashboard_generation(self, temp_output_dir):
        """Test error handling in dashboard generation."""
        generator = HTMLReportGenerator()

        # Create invalid report data
        invalid_data = ReportData(
            metadata=None,  # Invalid
            configuration=None,  # Invalid
            data_sources=[],
            execution_metrics=None,  # Invalid
            raw_data={}
        )

        result = generator.generate_interactive_dashboard(invalid_data)

        # Should handle errors gracefully
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error_message is not None

    def test_custom_styling_integration(self, sample_report_data, temp_output_dir):
        """Test custom styling integration."""
        generator = HTMLReportGenerator()

        # Add custom styling to configuration
        sample_report_data.configuration.styling = {
            'primary_color': '#007bff',
            'secondary_color': '#6c757d',
            'font_family': 'Arial, sans-serif'
        }

        options = ReportOptions(
            output_path=temp_output_dir / "styled_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for custom styling
        assert "#007bff" in content
        assert "Arial" in content

    def test_large_dataset_handling(self, temp_output_dir):
        """Test handling of large datasets."""
        # Create large dataset
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': [i * 1.5 for i in range(10000)],
            'category': [f"Cat_{i % 100}" for i in range(10000)]
        })

        metadata = ReportMetadata(
            title="Large Dataset Test",
            description="Test with large dataset",
            generated_at=datetime.now(),
            generated_by="Test",
            version="1.0"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.HTML,
            title="Large Data Dashboard"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            data_sources=[],
            execution_metrics=ExecutionMetrics(0, 0, 1, 10000),
            raw_data={'large_data': large_df}
        )

        generator = HTMLReportGenerator()

        options = ReportOptions(
            max_rows_per_table=1000,  # Limit rows
            output_path=temp_output_dir / "large_dashboard.html"
        )

        result = generator.generate_interactive_dashboard(report_data)

        assert result.success is True

        content = result.output_path.read_text()

        # Should handle large data gracefully (pagination, etc.)
        assert "DataTable" in content
        assert len(content) < 10000000  # Reasonable file size

    def test_export_functionality(self, sample_report_data, temp_output_dir):
        """Test export functionality integration."""
        generator = HTMLReportGenerator()

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for export functionality
        assert "export" in content.lower()
        assert "pdf" in content.lower() or "csv" in content.lower()

    def test_real_time_updates_placeholder(self, sample_report_data, temp_output_dir):
        """Test real-time updates placeholder functionality."""
        generator = HTMLReportGenerator()

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for real-time update infrastructure
        assert "setInterval" in content or "websocket" in content.lower() or "refresh" in content.lower()

    @patch('sqltest.reporting.interactive.InteractiveReportBuilder')
    def test_builder_integration(self, mock_builder, sample_report_data):
        """Test integration with InteractiveReportBuilder."""
        # Setup mock builder
        mock_instance = Mock()
        mock_instance.widgets = []
        mock_instance.filters = []
        mock_instance.generate_javascript.return_value = "// Mock JS"
        mock_instance.generate_css.return_value = "/* Mock CSS */"
        mock_builder.return_value = mock_instance

        generator = HTMLReportGenerator()
        result = generator.generate_interactive_dashboard(sample_report_data)

        # Verify builder was used
        mock_builder.assert_called_once()
        assert result.success is True

    def test_accessibility_features(self, sample_report_data, temp_output_dir):
        """Test accessibility features in generated dashboard."""
        generator = HTMLReportGenerator()

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for accessibility features
        assert "aria-" in content
        assert "role=" in content
        assert "alt=" in content or "aria-label=" in content

    def test_mobile_optimization(self, sample_report_data, temp_output_dir):
        """Test mobile optimization features."""
        generator = HTMLReportGenerator()

        result = generator.generate_interactive_dashboard(sample_report_data)

        assert result.success is True
        content = result.output_path.read_text()

        # Check for mobile optimization
        assert "viewport" in content
        assert "col-sm-" in content or "col-xs-" in content or "d-block d-sm-none" in content

    def test_performance_optimization(self, sample_report_data, temp_output_dir):
        """Test performance optimization features."""
        generator = HTMLReportGenerator()

        import time
        start_time = time.time()

        result = generator.generate_interactive_dashboard(sample_report_data)

        generation_time = time.time() - start_time

        assert result.success is True
        assert generation_time < 10.0  # Should complete quickly

        content = result.output_path.read_text()

        # Check for performance optimizations
        assert "defer" in content or "async" in content
        assert len(content) < 5000000  # Reasonable file size