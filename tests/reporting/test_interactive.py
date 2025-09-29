"""Tests for the interactive reporting components."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from sqltest.reporting.interactive import InteractiveReportBuilder, TrendAnalyzer
from sqltest.reporting.models import (
    ReportData, ReportConfiguration, ReportFormat, ReportType,
    ReportMetadata, Finding, SeverityLevel, ExecutionMetrics, DataSource
)


class TestInteractiveReportBuilder:
    """Test InteractiveReportBuilder class."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = InteractiveReportBuilder()

        assert isinstance(builder.widgets, list)
        assert isinstance(builder.filters, list)
        assert len(builder.widgets) == 0
        assert len(builder.filters) == 0

    def test_add_metric_card(self):
        """Test adding metric cards."""
        builder = InteractiveReportBuilder()

        # Add integer metric
        builder.add_metric_card("metric1", "Total Users", 1250)
        assert len(builder.widgets) == 1

        widget = builder.widgets[0]
        assert widget["id"] == "metric1"
        assert widget["title"] == "Total Users"
        assert widget["value"] == 1250
        assert widget["type"] == "metric"

        # Add float metric
        builder.add_metric_card("metric2", "Average Score", 85.7)
        assert len(builder.widgets) == 2

        # Add string metric
        builder.add_metric_card("metric3", "Status", "Active")
        assert len(builder.widgets) == 3

    def test_add_chart_widget(self):
        """Test adding chart widgets."""
        builder = InteractiveReportBuilder()

        chart_data = {
            "labels": ["Jan", "Feb", "Mar"],
            "datasets": [{
                "label": "Sales",
                "data": [100, 150, 200]
            }]
        }

        builder.add_chart_widget("chart1", "Monthly Sales", "line", chart_data)
        assert len(builder.widgets) == 1

        widget = builder.widgets[0]
        assert widget["id"] == "chart1"
        assert widget["title"] == "Monthly Sales"
        assert widget["chart_type"] == "line"
        assert widget["data"] == chart_data
        assert widget["type"] == "chart"

    def test_add_data_table_widget(self):
        """Test adding data table widgets."""
        builder = InteractiveReportBuilder()

        data = [
            {"name": "John", "age": 30, "city": "NYC"},
            {"name": "Jane", "age": 25, "city": "LA"}
        ]

        builder.add_data_table_widget("table1", "User Data", data)
        assert len(builder.widgets) == 1

        widget = builder.widgets[0]
        assert widget["id"] == "table1"
        assert widget["title"] == "User Data"
        assert widget["data"] == data
        assert widget["type"] == "table"

    def test_add_progress_widget(self):
        """Test adding progress widgets."""
        builder = InteractiveReportBuilder()

        builder.add_progress_widget("progress1", "Project Progress", 75.5)
        assert len(builder.widgets) == 1

        widget = builder.widgets[0]
        assert widget["id"] == "progress1"
        assert widget["title"] == "Project Progress"
        assert widget["value"] == 75.5
        assert widget["type"] == "progress"

    def test_add_filter(self):
        """Test adding filters."""
        builder = InteractiveReportBuilder()

        # Add dropdown filter
        options = ["Option 1", "Option 2", "Option 3"]
        builder.add_filter("filter1", "Category", "dropdown", options)
        assert len(builder.filters) == 1

        filter_item = builder.filters[0]
        assert filter_item["id"] == "filter1"
        assert filter_item["label"] == "Category"
        assert filter_item["type"] == "dropdown"
        assert filter_item["options"] == options

        # Add date range filter
        builder.add_filter("filter2", "Date Range", "daterange",
                         {"start": "2023-01-01", "end": "2023-12-31"})
        assert len(builder.filters) == 2

    def test_generate_javascript(self):
        """Test JavaScript generation."""
        builder = InteractiveReportBuilder()

        # Add some widgets
        builder.add_metric_card("metric1", "Users", 100)
        builder.add_chart_widget("chart1", "Sales", "bar", {
            "labels": ["A", "B"],
            "datasets": [{"data": [1, 2]}]
        })

        js_code = builder.generate_javascript()

        assert isinstance(js_code, str)
        assert "function initializeDashboard()" in js_code
        assert "metric1" in js_code
        assert "chart1" in js_code
        assert "Chart.js" in js_code or "new Chart" in js_code

    def test_generate_css(self):
        """Test CSS generation."""
        builder = InteractiveReportBuilder()

        css_code = builder.generate_css()

        assert isinstance(css_code, str)
        assert ".widget" in css_code
        assert ".metric-card" in css_code
        assert ".chart-container" in css_code

    def test_multiple_widgets_and_filters(self):
        """Test adding multiple widgets and filters."""
        builder = InteractiveReportBuilder()

        # Add multiple widgets
        builder.add_metric_card("users", "Users", 1000)
        builder.add_metric_card("revenue", "Revenue", 50000.50)
        builder.add_chart_widget("growth", "Growth", "line", {"data": []})
        builder.add_progress_widget("completion", "Completion", 80)

        # Add multiple filters
        builder.add_filter("category", "Category", "dropdown", ["A", "B", "C"])
        builder.add_filter("date", "Date", "daterange", {})

        assert len(builder.widgets) == 4
        assert len(builder.filters) == 2

        # Check widget types
        widget_types = [w["type"] for w in builder.widgets]
        assert "metric" in widget_types
        assert "chart" in widget_types
        assert "progress" in widget_types


class TestTrendAnalyzer:
    """Test TrendAnalyzer class."""

    def test_analyze_time_series_basic(self):
        """Test basic time series analysis."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        values = [100 + i * 2 + (i % 7) * 5 for i in range(30)]  # Trending up with weekly pattern

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        result = TrendAnalyzer.analyze_time_series(df, 'date', 'value', periods=7)

        assert isinstance(result, dict)
        assert 'trend' in result
        assert 'forecast' in result
        assert 'statistics' in result
        assert 'seasonality' in result

        # Check trend
        assert result['trend']['direction'] in ['up', 'down', 'stable']
        assert isinstance(result['trend']['slope'], (int, float))

        # Check forecast
        assert isinstance(result['forecast'], list)
        assert len(result['forecast']) == 7  # periods requested

        # Check statistics
        stats = result['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_analyze_time_series_with_missing_values(self):
        """Test time series analysis with missing values."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        values = [100 + i for i in range(20)]

        # Add some NaN values
        values[5] = None
        values[15] = None

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        result = TrendAnalyzer.analyze_time_series(df, 'date', 'value')

        assert isinstance(result, dict)
        assert 'trend' in result
        assert 'forecast' in result

        # Should handle missing values gracefully
        assert result['statistics']['missing_values'] > 0

    def test_analyze_time_series_flat_data(self):
        """Test time series analysis with flat data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = [100] * 10  # Flat line

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        result = TrendAnalyzer.analyze_time_series(df, 'date', 'value')

        assert result['trend']['direction'] == 'stable'
        assert abs(result['trend']['slope']) < 0.1  # Should be near zero

    def test_detect_seasonality(self):
        """Test seasonality detection."""
        # Create data with weekly pattern
        dates = pd.date_range('2023-01-01', periods=56, freq='D')  # 8 weeks
        values = [100 + (i % 7) * 10 for i in range(56)]  # Weekly pattern

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        seasonality = TrendAnalyzer.detect_seasonality(df, 'date', 'value')

        assert isinstance(seasonality, dict)
        assert 'has_seasonality' in seasonality
        assert 'period' in seasonality
        assert 'strength' in seasonality

    def test_calculate_forecast_confidence(self):
        """Test forecast confidence calculation."""
        # Create sample data
        actual = [100, 105, 110, 115, 120]
        predicted = [98, 107, 108, 118, 122]

        confidence = TrendAnalyzer.calculate_forecast_confidence(actual, predicted)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        # Create sample report data
        metadata = ReportMetadata(
            title="Test Report",
            description="Test description",
            generated_at=datetime.now(),
            generated_by="Test",
            version="1.0"
        )

        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE,
            format=ReportFormat.HTML,
            title="Test Report"
        )

        # Create sample DataFrame
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [100, 150, 200, 120, 180],
            'date': pd.date_range('2023-01-01', periods=5, freq='D')
        })

        # Create findings
        findings = [
            Finding(
                id="finding1",
                title="High Performance",
                description="Category C shows highest performance",
                severity=SeverityLevel.INFO,
                category="Performance"
            ),
            Finding(
                id="finding2",
                title="Growth Trend",
                description="Upward trend detected",
                severity=SeverityLevel.LOW,
                category="Trend"
            )
        ]

        execution_metrics = ExecutionMetrics(
            execution_time=1.5,
            memory_usage=250.0,
            queries_executed=3,
            rows_processed=5
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            data_sources=[],
            execution_metrics=execution_metrics,
            raw_data={'main': df},
            findings=findings
        )

        summary = TrendAnalyzer.generate_executive_summary(report_data)

        assert isinstance(summary, dict)
        assert 'overview' in summary
        assert 'key_metrics' in summary
        assert 'findings_summary' in summary
        assert 'recommendations' in summary
        assert 'data_quality' in summary

        # Check overview
        overview = summary['overview']
        assert 'total_records' in overview
        assert 'data_sources' in overview
        assert 'time_period' in overview

        # Check key metrics
        metrics = summary['key_metrics']
        assert isinstance(metrics, list)

        # Check findings summary
        findings_summary = summary['findings_summary']
        assert 'total_findings' in findings_summary
        assert 'by_severity' in findings_summary
        assert 'by_category' in findings_summary

    def test_identify_key_insights(self):
        """Test key insights identification."""
        # Create sample DataFrame with interesting patterns
        df = pd.DataFrame({
            'category': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
            'value': list(range(50)) + list(range(100, 130)) + list(range(200, 220)),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })

        insights = TrendAnalyzer.identify_key_insights({'main': df})

        assert isinstance(insights, list)
        assert len(insights) > 0

        # Each insight should have required fields
        for insight in insights:
            assert 'type' in insight
            assert 'description' in insight
            assert 'confidence' in insight

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        # Create findings with different severities
        findings = [
            Finding(
                id="f1",
                title="Critical Issue",
                description="Critical performance issue",
                severity=SeverityLevel.CRITICAL,
                category="Performance"
            ),
            Finding(
                id="f2",
                title="Data Quality",
                description="Missing data detected",
                severity=SeverityLevel.HIGH,
                category="Quality"
            ),
            Finding(
                id="f3",
                title="Minor Issue",
                description="Minor formatting issue",
                severity=SeverityLevel.LOW,
                category="Format"
            )
        ]

        recommendations = TrendAnalyzer.generate_recommendations(findings)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should prioritize critical and high severity issues
        critical_recs = [r for r in recommendations if 'critical' in r.lower() or 'high' in r.lower()]
        assert len(critical_recs) > 0

    def test_analyze_data_quality(self):
        """Test data quality analysis."""
        # Create DataFrame with quality issues
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['A', 'B', 'C', None, 'E'],
            'col3': [1, 1, 1, 1, 1],  # No variation
            'col4': [1, 2, 3, 4, 5]   # Good data
        })

        quality_report = TrendAnalyzer.analyze_data_quality({'main': df})

        assert isinstance(quality_report, dict)
        assert 'overall_score' in quality_report
        assert 'issues' in quality_report
        assert 'completeness' in quality_report
        assert 'consistency' in quality_report

        # Should detect missing values
        assert quality_report['completeness'] < 100

        # Should identify quality issues
        assert len(quality_report['issues']) > 0

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        # Should handle empty data gracefully
        result = TrendAnalyzer.analyze_time_series(empty_df, 'date', 'value')
        assert result['trend']['direction'] == 'unknown'

        insights = TrendAnalyzer.identify_key_insights({'empty': empty_df})
        assert isinstance(insights, list)

        quality_report = TrendAnalyzer.analyze_data_quality({'empty': empty_df})
        assert isinstance(quality_report, dict)

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        values = [100 + i * 0.1 + (i % 30) * 2 for i in range(1000)]

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        import time
        start_time = time.time()
        result = TrendAnalyzer.analyze_time_series(df, 'date', 'value', periods=30)
        execution_time = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert execution_time < 5.0
        assert isinstance(result, dict)
        assert len(result['forecast']) == 30


class TestIntegration:
    """Integration tests for interactive components."""

    def test_builder_with_trend_analysis(self):
        """Test integration between builder and trend analyzer."""
        builder = InteractiveReportBuilder()

        # Create time series data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        values = [100 + i * 2 for i in range(30)]
        df = pd.DataFrame({'date': dates, 'value': values})

        # Analyze trends
        trend_result = TrendAnalyzer.analyze_time_series(df, 'date', 'value')

        # Add trend results to dashboard
        builder.add_metric_card("trend_direction", "Trend Direction",
                              trend_result['trend']['direction'])
        builder.add_metric_card("trend_slope", "Growth Rate",
                              f"{trend_result['trend']['slope']:.2f}")

        # Create forecast chart
        forecast_data = {
            "labels": [f"Day {i+1}" for i in range(len(trend_result['forecast']))],
            "datasets": [{
                "label": "Forecast",
                "data": trend_result['forecast']
            }]
        }
        builder.add_chart_widget("forecast_chart", "Forecast", "line", forecast_data)

        assert len(builder.widgets) == 3
        assert any(w["id"] == "trend_direction" for w in builder.widgets)
        assert any(w["id"] == "forecast_chart" for w in builder.widgets)

    def test_complete_dashboard_generation(self):
        """Test complete dashboard generation workflow."""
        builder = InteractiveReportBuilder()

        # Create comprehensive dashboard
        builder.add_metric_card("users", "Total Users", 1250)
        builder.add_metric_card("revenue", "Revenue", 50000)
        builder.add_progress_widget("completion", "Project Progress", 75)

        chart_data = {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "datasets": [{
                "label": "Sales",
                "data": [1000, 1500, 1200, 1800]
            }]
        }
        builder.add_chart_widget("quarterly_sales", "Quarterly Sales", "bar", chart_data)

        table_data = [
            {"product": "A", "sales": 1000},
            {"product": "B", "sales": 1500}
        ]
        builder.add_data_table_widget("product_table", "Product Sales", table_data)

        # Add filters
        builder.add_filter("date_range", "Date Range", "daterange", {})
        builder.add_filter("category", "Category", "dropdown", ["All", "A", "B", "C"])

        # Generate JavaScript and CSS
        js_code = builder.generate_javascript()
        css_code = builder.generate_css()

        assert len(builder.widgets) == 5
        assert len(builder.filters) == 2
        assert isinstance(js_code, str)
        assert isinstance(css_code, str)
        assert len(js_code) > 100  # Should generate substantial code
        assert len(css_code) > 100   # Should generate substantial styles