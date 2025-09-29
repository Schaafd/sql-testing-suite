"""Tests for report data analyzer."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sqltest.reporting.analyzer import ReportAnalyzer
from sqltest.reporting.models import (
    ReportData, ReportMetadata, ReportConfiguration, ReportType,
    ReportFormat, SeverityLevel, Finding, ExecutionMetrics
)


class TestReportAnalyzer:
    """Test ReportAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ReportAnalyzer()

        assert len(analyzer.insight_generators) == 8
        assert 'trend_analysis' in analyzer.insight_generators
        assert 'anomaly_detection' in analyzer.insight_generators
        assert 'correlation_analysis' in analyzer.insight_generators

    def test_analyze_report_data_basic(self, sample_report_data):
        """Test basic report data analysis."""
        analyzer = ReportAnalyzer()

        result = analyzer.analyze_report_data(sample_report_data)

        assert result is not None
        assert isinstance(result, ReportData)
        # Should have added some insights
        assert len(result.sections) >= len(sample_report_data.sections)

    def test_analyze_with_specific_types(self, sample_report_data):
        """Test analysis with specific analysis types."""
        analyzer = ReportAnalyzer()

        result = analyzer.analyze_report_data(
            sample_report_data,
            analysis_types=['trend_analysis', 'anomaly_detection']
        )

        assert result is not None
        # Should execute only specified analysis types

    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        analyzer = ReportAnalyzer()

        # Create data with clear trend
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        values = np.arange(50) + np.random.normal(0, 2, 50)  # Upward trend with noise

        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'category': ['A'] * 25 + ['B'] * 25
        })

        metadata = ReportMetadata(
            title="Trend Test",
            description="Test trend analysis",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Trend Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'trend_data': df}
        )

        analyzer._analyze_trends(report_data)

        # Should detect trend and add findings
        trend_findings = [f for f in report_data.findings if f.category == "Trend Analysis"]
        assert len(trend_findings) > 0

        # Check finding details
        trend_finding = trend_findings[0]
        assert 'trend' in trend_finding.description.lower()
        assert 'correlation' in trend_finding.details
        assert 'slope' in trend_finding.details

    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        analyzer = ReportAnalyzer()

        # Create data with clear outliers
        normal_data = np.random.normal(50, 10, 95)
        outliers = [150, -50, 200, -100, 175]  # Clear outliers
        all_data = np.concatenate([normal_data, outliers])

        df = pd.DataFrame({
            'id': range(100),
            'value': all_data,
            'category': ['normal'] * 95 + ['outlier'] * 5
        })

        metadata = ReportMetadata(
            title="Anomaly Test",
            description="Test anomaly detection",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Anomaly Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'anomaly_data': df}
        )

        analyzer._detect_anomalies(report_data)

        # Should detect anomalies
        anomaly_findings = [f for f in report_data.findings if f.category == "Anomaly Detection"]
        assert len(anomaly_findings) > 0

        # Check finding details
        anomaly_finding = anomaly_findings[0]
        assert 'anomalous' in anomaly_finding.description.lower() or 'outlier' in anomaly_finding.description.lower()
        assert 'indices' in anomaly_finding.details
        assert len(anomaly_finding.details['indices']) > 0

    def test_correlation_analysis(self):
        """Test correlation analysis functionality."""
        analyzer = ReportAnalyzer()

        # Create strongly correlated data
        x = np.random.normal(0, 1, 100)
        y = x * 2 + np.random.normal(0, 0.1, 100)  # Strong positive correlation
        z = -x * 1.5 + np.random.normal(0, 0.1, 100)  # Strong negative correlation
        w = np.random.normal(0, 1, 100)  # No correlation

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'w': w
        })

        metadata = ReportMetadata(
            title="Correlation Test",
            description="Test correlation analysis",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Correlation Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'correlation_data': df}
        )

        analyzer._analyze_correlations(report_data)

        # Should detect strong correlations
        correlation_findings = [f for f in report_data.findings if f.category == "Correlation Analysis"]
        assert len(correlation_findings) > 0

        # Check finding details
        correlation_finding = correlation_findings[0]
        assert 'correlation' in correlation_finding.description.lower()
        assert 'correlations' in correlation_finding.details
        assert len(correlation_finding.details['correlations']) > 0

    def test_distribution_analysis(self):
        """Test distribution analysis functionality."""
        analyzer = ReportAnalyzer()

        # Create data with specific distribution characteristics
        skewed_data = np.random.exponential(2, 1000)  # Right-skewed distribution
        normal_data = np.random.normal(50, 10, 1000)  # Normal distribution

        df = pd.DataFrame({
            'skewed_column': skewed_data,
            'normal_column': normal_data,
            'id': range(1000)
        })

        metadata = ReportMetadata(
            title="Distribution Test",
            description="Test distribution analysis",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Distribution Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'distribution_data': df}
        )

        analyzer._analyze_distributions(report_data)

        # Should detect distribution characteristics
        distribution_findings = [f for f in report_data.findings if f.category == "Distribution Analysis"]
        assert len(distribution_findings) > 0

        # Check for skewness detection
        skewed_finding = next((f for f in distribution_findings if 'skewed_column' in f.title), None)
        assert skewed_finding is not None
        assert 'skewness' in skewed_finding.details

    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        analyzer = ReportAnalyzer()

        # Create report data with performance issues
        metadata = ReportMetadata(
            title="Performance Test",
            description="Test performance analysis",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Performance Test"
        )

        # Create metrics that should trigger performance warnings
        execution_metrics = ExecutionMetrics(
            execution_time=45.0,  # Slow execution
            memory_usage=1500.0,  # High memory usage
            queries_executed=10,
            rows_processed=5000,
            cache_hit_rate=0.3,  # Low cache hit rate
            errors_encountered=0
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            execution_metrics=execution_metrics,
            raw_data={'test_data': pd.DataFrame({'col': [1, 2, 3]})}
        )

        analyzer._analyze_performance(report_data)

        # Should detect performance issues
        performance_findings = [f for f in report_data.findings if f.category == "Performance"]
        assert len(performance_findings) > 0

        # Check for specific performance issues
        execution_time_finding = next((f for f in performance_findings if 'slow' in f.title.lower() or 'execution' in f.title.lower()), None)
        memory_finding = next((f for f in performance_findings if 'memory' in f.title.lower()), None)
        cache_finding = next((f for f in performance_findings if 'cache' in f.title.lower()), None)

        assert execution_time_finding is not None
        assert memory_finding is not None
        assert cache_finding is not None

    def test_data_quality_assessment(self):
        """Test data quality assessment functionality."""
        analyzer = ReportAnalyzer()

        # Create data with quality issues
        df = pd.DataFrame({
            'id': range(100),
            'name': ['Name_' + str(i) if i < 90 else None for i in range(100)],  # 10% missing
            'value': [i if i < 95 else None for i in range(100)],  # 5% missing
            'mixed_type': [str(i) if i % 2 == 0 else i for i in range(100)],  # Mixed types
            'duplicate_col': [1, 2, 3] * 33 + [1]  # Some duplicates
        })

        # Add duplicate rows
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)

        metadata = ReportMetadata(
            title="Quality Test",
            description="Test data quality assessment",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Quality Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'quality_data': df}
        )

        analyzer._assess_data_quality(report_data)

        # Should detect quality issues
        quality_findings = [f for f in report_data.findings if f.category == "Data Quality"]
        assert len(quality_findings) > 0

        # Check finding details
        quality_finding = quality_findings[0]
        assert 'issues' in quality_finding.details
        issues = quality_finding.details['issues']

        # Should detect duplicate rows and missing values
        issue_types = [issue['type'] for issue in issues]
        assert 'duplicate_rows' in issue_types

    def test_pattern_recognition(self):
        """Test pattern recognition functionality."""
        analyzer = ReportAnalyzer()

        # Create categorical data with patterns
        categories = ['A'] * 60 + ['B'] * 30 + ['C'] * 5 + ['D'] * 3 + ['E'] * 2  # Dominant A, rare D,E
        df = pd.DataFrame({
            'id': range(100),
            'category': categories,
            'high_cardinality': [f'unique_{i}' for i in range(100)],  # High cardinality
            'normal_category': ['X', 'Y', 'Z'] * 33 + ['X']  # Normal distribution
        })

        metadata = ReportMetadata(
            title="Pattern Test",
            description="Test pattern recognition",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Pattern Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'pattern_data': df}
        )

        analyzer._recognize_patterns(report_data)

        # Should detect patterns
        pattern_findings = [f for f in report_data.findings if f.category == "Pattern Recognition"]
        assert len(pattern_findings) > 0

        # Check for specific pattern detection
        dominant_finding = next((f for f in pattern_findings if 'dominant' in f.description.lower()), None)
        cardinality_finding = next((f for f in pattern_findings if 'cardinality' in f.description.lower()), None)

        # Should detect either dominant category or high cardinality
        assert dominant_finding is not None or cardinality_finding is not None

    def test_comparative_analysis(self):
        """Test comparative analysis functionality."""
        analyzer = ReportAnalyzer()

        # Create two similar datasets for comparison
        df1 = pd.DataFrame({
            'id': range(50),
            'value': np.random.normal(100, 15, 50),
            'category': ['A', 'B'] * 25
        })

        df2 = pd.DataFrame({
            'id': range(50, 100),
            'value': np.random.normal(110, 20, 50),  # Different mean and std
            'category': ['A', 'B', 'C'] * 16 + ['A', 'B'],  # Different categories
            'new_column': range(50)  # Additional column
        })

        metadata = ReportMetadata(
            title="Comparison Test",
            description="Test comparative analysis",
            generated_at=datetime.now(),
            generated_by="test"
        )

        config = ReportConfiguration(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Comparison Test"
        )

        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            raw_data={'dataset1': df1, 'dataset2': df2}
        )

        analyzer._perform_comparative_analysis(report_data)

        # Should perform comparison
        comparison_findings = [f for f in report_data.findings if f.category == "Comparative Analysis"]
        assert len(comparison_findings) > 0

        # Check comparison details
        comparison_finding = comparison_findings[0]
        assert 'comparisons' in comparison_finding.details
        comparisons = comparison_finding.details['comparisons']
        assert len(comparisons) > 0

        comparison = comparisons[0]
        assert 'common_columns' in comparison
        assert 'unique_columns' in comparison

    def test_summary_insights_generation(self, sample_report_data):
        """Test summary insights generation."""
        analyzer = ReportAnalyzer()

        # Add some findings first
        sample_report_data.add_finding(Finding(
            id="critical_test",
            title="Critical Test Issue",
            description="Test critical issue",
            severity=SeverityLevel.CRITICAL,
            category="Test"
        ))

        analyzer._generate_summary_insights(sample_report_data)

        # Should add insights section
        insights_section = sample_report_data.get_section("executive_insights")
        assert insights_section is not None
        assert "critical" in insights_section.content.lower()

    def test_summary_statistics_update(self, sample_report_data):
        """Test summary statistics update."""
        analyzer = ReportAnalyzer()

        analyzer._update_summary_statistics(sample_report_data)

        stats = sample_report_data.summary_statistics
        assert 'findings_by_severity' in stats
        assert 'total_findings' in stats
        assert 'total_datasets' in stats
        assert 'total_rows' in stats
        assert 'analysis_timestamp' in stats

    def test_calculate_trend_edge_cases(self):
        """Test trend calculation with edge cases."""
        analyzer = ReportAnalyzer()

        # Test with insufficient data
        df_small = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=1),
            'value': [10]
        })

        result = analyzer._calculate_trend(df_small, 'date', 'value')
        assert result is None

        # Test with all NaN values
        df_nan = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': [np.nan] * 10
        })

        result = analyzer._calculate_trend(df_nan, 'date', 'value')
        assert result is None

    def test_find_outliers_edge_cases(self):
        """Test outlier detection with edge cases."""
        analyzer = ReportAnalyzer()

        # Test with constant values
        constant_series = pd.Series([5] * 100, name='constant')
        result = analyzer._find_outliers(constant_series, 'constant')
        assert result is None  # No outliers in constant data

        # Test with insufficient data
        small_series = pd.Series([1, 2], name='small')
        result = analyzer._find_outliers(small_series, 'small')
        assert result is None

    def test_analyze_column_distribution_edge_cases(self):
        """Test distribution analysis with edge cases."""
        analyzer = ReportAnalyzer()

        # Test with insufficient data
        small_series = pd.Series([1, 2, 3], name='small')
        result = analyzer._analyze_column_distribution(small_series, 'small')
        assert result is None

        # Test with all NaN values
        nan_series = pd.Series([np.nan] * 20, name='nan_col')
        result = analyzer._analyze_column_distribution(nan_series, 'nan_col')
        assert result is None

    def test_compare_datasets_edge_cases(self):
        """Test dataset comparison with edge cases."""
        analyzer = ReportAnalyzer()

        # Test with no common columns
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        result = analyzer._compare_datasets(df1, df2, 'df1', 'df2')
        assert result is not None
        assert len(result['common_columns']) == 0
        assert len(result['unique_columns']['dataset1']) == 1
        assert len(result['unique_columns']['dataset2']) == 1