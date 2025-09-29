"""Report data analysis and aggregation engine."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import statistics

import pandas as pd
import numpy as np

from .models import (
    ReportData, Finding, SeverityLevel, DataSource, ChartData,
    ReportSection, ExecutionMetrics
)

logger = logging.getLogger(__name__)


class ReportAnalyzer:
    """Advanced analytics engine for report data analysis and insights generation."""

    def __init__(self):
        """Initialize the analyzer."""
        self.insight_generators = {
            'trend_analysis': self._analyze_trends,
            'anomaly_detection': self._detect_anomalies,
            'correlation_analysis': self._analyze_correlations,
            'distribution_analysis': self._analyze_distributions,
            'performance_analysis': self._analyze_performance,
            'quality_assessment': self._assess_data_quality,
            'pattern_recognition': self._recognize_patterns,
            'comparative_analysis': self._perform_comparative_analysis
        }

    def analyze_report_data(self, report_data: ReportData,
                          analysis_types: Optional[List[str]] = None) -> ReportData:
        """Perform comprehensive analysis on report data.

        Args:
            report_data: The report data to analyze
            analysis_types: Specific analysis types to perform (defaults to all)

        Returns:
            Enhanced report data with analysis results
        """
        if analysis_types is None:
            analysis_types = list(self.insight_generators.keys())

        logger.info(f"Starting analysis with types: {analysis_types}")

        # Perform each requested analysis
        for analysis_type in analysis_types:
            if analysis_type in self.insight_generators:
                try:
                    self.insight_generators[analysis_type](report_data)
                    logger.debug(f"Completed {analysis_type}")
                except Exception as e:
                    logger.error(f"Error in {analysis_type}: {e}")

        # Generate summary insights
        self._generate_summary_insights(report_data)

        # Update summary statistics
        self._update_summary_statistics(report_data)

        return report_data

    def _analyze_trends(self, report_data: ReportData) -> None:
        """Analyze trends in the data and add findings."""
        for dataset_name, df in report_data.raw_data.items():
            if df.empty:
                continue

            # Look for date/time columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype in ['datetime64[ns]', 'object']:
                    try:
                        pd.to_datetime(df[col].head())
                        date_cols.append(col)
                    except:
                        continue

            if not date_cols:
                continue

            # Analyze trends for each date column
            for date_col in date_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                for numeric_col in numeric_cols:
                    trend_insight = self._calculate_trend(df, date_col, numeric_col)
                    if trend_insight:
                        finding = Finding(
                            id=f"trend_{dataset_name}_{numeric_col}",
                            title=f"Trend Analysis: {numeric_col}",
                            description=trend_insight['description'],
                            severity=trend_insight['severity'],
                            category="Trend Analysis",
                            details=trend_insight['details']
                        )
                        report_data.add_finding(finding)

    def _calculate_trend(self, df: pd.DataFrame, date_col: str, value_col: str) -> Optional[Dict[str, Any]]:
        """Calculate trend for a specific date/value column pair."""
        try:
            df_sorted = df.copy()
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
            df_sorted = df_sorted.sort_values(date_col)

            # Calculate trend using linear regression
            x = np.arange(len(df_sorted))
            y = df_sorted[value_col].values

            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None

            x_clean = x[mask]
            y_clean = y[mask]

            # Calculate slope and correlation
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]

            # Determine trend strength and direction
            trend_strength = abs(correlation)
            trend_direction = "increasing" if slope > 0 else "decreasing"

            # Assess significance
            if trend_strength > 0.8:
                severity = SeverityLevel.HIGH
                strength_desc = "strong"
            elif trend_strength > 0.5:
                severity = SeverityLevel.MEDIUM
                strength_desc = "moderate"
            else:
                severity = SeverityLevel.LOW
                strength_desc = "weak"

            return {
                'description': f"Detected {strength_desc} {trend_direction} trend in {value_col} "
                              f"(correlation: {correlation:.3f}, slope: {slope:.3f})",
                'severity': severity,
                'details': {
                    'slope': slope,
                    'correlation': correlation,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'data_points': len(y_clean),
                    'start_value': float(y_clean[0]),
                    'end_value': float(y_clean[-1]),
                    'change_percentage': ((y_clean[-1] - y_clean[0]) / y_clean[0] * 100) if y_clean[0] != 0 else 0
                }
            }

        except Exception as e:
            logger.warning(f"Error calculating trend for {value_col}: {e}")
            return None

    def _detect_anomalies(self, report_data: ReportData) -> None:
        """Detect anomalies in the data using statistical methods."""
        for dataset_name, df in report_data.raw_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                anomalies = self._find_outliers(df[col], col)
                if anomalies:
                    finding = Finding(
                        id=f"anomaly_{dataset_name}_{col}",
                        title=f"Anomalies Detected: {col}",
                        description=f"Found {len(anomalies['indices'])} anomalous values in {col}",
                        severity=SeverityLevel.MEDIUM if len(anomalies['indices']) > 5 else SeverityLevel.LOW,
                        category="Anomaly Detection",
                        details=anomalies
                    )
                    report_data.add_finding(finding)

    def _find_outliers(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Find outliers using IQR and Z-score methods."""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 4:
                return None

            # IQR method
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]

            # Z-score method
            z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
            zscore_outliers = clean_series[z_scores > 3]

            # Combine results
            outlier_indices = list(set(iqr_outliers.index) | set(zscore_outliers.index))

            if not outlier_indices:
                return None

            return {
                'indices': outlier_indices,
                'values': series.loc[outlier_indices].tolist(),
                'iqr_method': {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'count': len(iqr_outliers)
                },
                'zscore_method': {
                    'threshold': 3,
                    'count': len(zscore_outliers)
                },
                'statistics': {
                    'mean': clean_series.mean(),
                    'std': clean_series.std(),
                    'median': clean_series.median(),
                    'q1': Q1,
                    'q3': Q3
                }
            }

        except Exception as e:
            logger.warning(f"Error finding outliers in {column_name}: {e}")
            return None

    def _analyze_correlations(self, report_data: ReportData) -> None:
        """Analyze correlations between numeric variables."""
        for dataset_name, df in report_data.raw_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                continue

            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            'column1': corr_matrix.columns[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })

            if strong_correlations:
                finding = Finding(
                    id=f"correlation_{dataset_name}",
                    title=f"Strong Correlations Found",
                    description=f"Identified {len(strong_correlations)} strong correlations in {dataset_name}",
                    severity=SeverityLevel.INFO,
                    category="Correlation Analysis",
                    details={
                        'correlations': strong_correlations,
                        'correlation_matrix': corr_matrix.to_dict()
                    }
                )
                report_data.add_finding(finding)

    def _analyze_distributions(self, report_data: ReportData) -> None:
        """Analyze data distributions and identify patterns."""
        for dataset_name, df in report_data.raw_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                distribution_analysis = self._analyze_column_distribution(df[col], col)
                if distribution_analysis:
                    finding = Finding(
                        id=f"distribution_{dataset_name}_{col}",
                        title=f"Distribution Analysis: {col}",
                        description=distribution_analysis['description'],
                        severity=SeverityLevel.INFO,
                        category="Distribution Analysis",
                        details=distribution_analysis['details']
                    )
                    report_data.add_finding(finding)

    def _analyze_column_distribution(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Analyze the distribution of a single column."""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return None

            # Calculate distribution statistics
            mean_val = clean_series.mean()
            median_val = clean_series.median()
            std_val = clean_series.std()
            skewness = clean_series.skew()
            kurtosis = clean_series.kurtosis()

            # Assess distribution characteristics
            characteristics = []

            if abs(skewness) > 1:
                characteristics.append(f"highly skewed ({skewness:.2f})")
            elif abs(skewness) > 0.5:
                characteristics.append(f"moderately skewed ({skewness:.2f})")

            if kurtosis > 3:
                characteristics.append("heavy-tailed distribution")
            elif kurtosis < -1:
                characteristics.append("light-tailed distribution")

            if abs(mean_val - median_val) / std_val > 0.5:
                characteristics.append("significant mean-median difference")

            description = f"Distribution analysis for {column_name}"
            if characteristics:
                description += f": {', '.join(characteristics)}"

            return {
                'description': description,
                'details': {
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'min': clean_series.min(),
                    'max': clean_series.max(),
                    'quartiles': {
                        'q1': clean_series.quantile(0.25),
                        'q2': clean_series.quantile(0.5),
                        'q3': clean_series.quantile(0.75)
                    },
                    'characteristics': characteristics
                }
            }

        except Exception as e:
            logger.warning(f"Error analyzing distribution for {column_name}: {e}")
            return None

    def _analyze_performance(self, report_data: ReportData) -> None:
        """Analyze performance metrics and execution efficiency."""
        metrics = report_data.execution_metrics

        # Analyze execution time
        if metrics.execution_time > 30:  # Slow execution threshold
            finding = Finding(
                id="performance_execution_time",
                title="Slow Report Generation",
                description=f"Report took {metrics.execution_time:.2f} seconds to generate",
                severity=SeverityLevel.MEDIUM if metrics.execution_time > 60 else SeverityLevel.LOW,
                category="Performance",
                details={'execution_time': metrics.execution_time}
            )
            report_data.add_finding(finding)

        # Analyze memory usage
        if metrics.memory_usage > 1024:  # High memory usage threshold (MB)
            finding = Finding(
                id="performance_memory_usage",
                title="High Memory Usage",
                description=f"Report generation used {metrics.memory_usage:.2f} MB of memory",
                severity=SeverityLevel.MEDIUM if metrics.memory_usage > 2048 else SeverityLevel.LOW,
                category="Performance",
                details={'memory_usage': metrics.memory_usage}
            )
            report_data.add_finding(finding)

        # Analyze cache hit rate
        if metrics.cache_hit_rate < 0.5:  # Low cache hit rate
            finding = Finding(
                id="performance_cache_hit_rate",
                title="Low Cache Hit Rate",
                description=f"Cache hit rate is only {metrics.cache_hit_rate:.1%}",
                severity=SeverityLevel.LOW,
                category="Performance",
                details={'cache_hit_rate': metrics.cache_hit_rate}
            )
            report_data.add_finding(finding)

    def _assess_data_quality(self, report_data: ReportData) -> None:
        """Assess data quality and identify issues."""
        for dataset_name, df in report_data.raw_data.items():
            quality_issues = []

            # Check for missing values
            missing_counts = df.isnull().sum()
            high_missing = missing_counts[missing_counts > len(df) * 0.1]  # >10% missing

            if not high_missing.empty:
                quality_issues.append({
                    'type': 'missing_values',
                    'description': f"Columns with >10% missing values: {list(high_missing.index)}",
                    'details': high_missing.to_dict()
                })

            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                quality_issues.append({
                    'type': 'duplicate_rows',
                    'description': f"Found {duplicate_count} duplicate rows",
                    'details': {'count': duplicate_count, 'percentage': duplicate_count / len(df)}
                })

            # Check for data type inconsistencies
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_types = set(type(x).__name__ for x in df[col].dropna())
                    if len(unique_types) > 1:
                        quality_issues.append({
                            'type': 'mixed_types',
                            'description': f"Column {col} contains mixed data types: {unique_types}",
                            'details': {'column': col, 'types': list(unique_types)}
                        })

            if quality_issues:
                finding = Finding(
                    id=f"data_quality_{dataset_name}",
                    title=f"Data Quality Issues: {dataset_name}",
                    description=f"Identified {len(quality_issues)} data quality issues",
                    severity=SeverityLevel.MEDIUM,
                    category="Data Quality",
                    details={'issues': quality_issues}
                )
                report_data.add_finding(finding)

    def _recognize_patterns(self, report_data: ReportData) -> None:
        """Recognize patterns in categorical and text data."""
        for dataset_name, df in report_data.raw_data.items():
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns

            for col in categorical_cols:
                pattern_analysis = self._analyze_categorical_patterns(df[col], col)
                if pattern_analysis:
                    finding = Finding(
                        id=f"pattern_{dataset_name}_{col}",
                        title=f"Pattern Analysis: {col}",
                        description=pattern_analysis['description'],
                        severity=SeverityLevel.INFO,
                        category="Pattern Recognition",
                        details=pattern_analysis['details']
                    )
                    report_data.add_finding(finding)

    def _analyze_categorical_patterns(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Analyze patterns in categorical data."""
        try:
            value_counts = series.value_counts()
            total_count = len(series.dropna())

            if total_count == 0:
                return None

            patterns = []

            # Check for dominant categories (>50% of data)
            dominant = value_counts[value_counts / total_count > 0.5]
            if not dominant.empty:
                patterns.append(f"Dominant category: {dominant.index[0]} ({dominant.iloc[0]/total_count:.1%})")

            # Check for rare categories (<1% of data)
            rare = value_counts[value_counts / total_count < 0.01]
            if len(rare) > 0:
                patterns.append(f"{len(rare)} rare categories (<1% each)")

            # Check for high cardinality
            unique_ratio = len(value_counts) / total_count
            if unique_ratio > 0.9:
                patterns.append("High cardinality (mostly unique values)")

            if not patterns:
                return None

            return {
                'description': f"Categorical patterns in {column_name}: {'; '.join(patterns)}",
                'details': {
                    'unique_values': len(value_counts),
                    'total_values': total_count,
                    'unique_ratio': unique_ratio,
                    'top_values': value_counts.head(10).to_dict(),
                    'patterns': patterns
                }
            }

        except Exception as e:
            logger.warning(f"Error analyzing patterns for {column_name}: {e}")
            return None

    def _perform_comparative_analysis(self, report_data: ReportData) -> None:
        """Perform comparative analysis across datasets."""
        if len(report_data.raw_data) < 2:
            return

        dataset_names = list(report_data.raw_data.keys())
        comparisons = []

        for i in range(len(dataset_names)):
            for j in range(i+1, len(dataset_names)):
                dataset1 = report_data.raw_data[dataset_names[i]]
                dataset2 = report_data.raw_data[dataset_names[j]]

                comparison = self._compare_datasets(dataset1, dataset2, dataset_names[i], dataset_names[j])
                if comparison:
                    comparisons.append(comparison)

        if comparisons:
            finding = Finding(
                id="comparative_analysis",
                title="Dataset Comparison Analysis",
                description=f"Performed comparative analysis across {len(report_data.raw_data)} datasets",
                severity=SeverityLevel.INFO,
                category="Comparative Analysis",
                details={'comparisons': comparisons}
            )
            report_data.add_finding(finding)

    def _compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                         name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Compare two datasets for similarities and differences."""
        try:
            comparison = {
                'dataset1': name1,
                'dataset2': name2,
                'row_counts': {'dataset1': len(df1), 'dataset2': len(df2)},
                'column_counts': {'dataset1': len(df1.columns), 'dataset2': len(df2.columns)},
                'common_columns': list(set(df1.columns) & set(df2.columns)),
                'unique_columns': {
                    'dataset1': list(set(df1.columns) - set(df2.columns)),
                    'dataset2': list(set(df2.columns) - set(df1.columns))
                }
            }

            # Compare common columns if any exist
            if comparison['common_columns']:
                column_comparisons = {}
                for col in comparison['common_columns']:
                    if col in df1.columns and col in df2.columns:
                        col_comparison = self._compare_columns(df1[col], df2[col], col)
                        if col_comparison:
                            column_comparisons[col] = col_comparison

                comparison['column_comparisons'] = column_comparisons

            return comparison

        except Exception as e:
            logger.warning(f"Error comparing datasets {name1} and {name2}: {e}")
            return None

    def _compare_columns(self, series1: pd.Series, series2: pd.Series, column_name: str) -> Dict[str, Any]:
        """Compare two columns for statistical differences."""
        comparison = {
            'data_types': {'series1': str(series1.dtype), 'series2': str(series2.dtype)},
            'null_counts': {'series1': series1.isnull().sum(), 'series2': series2.isnull().sum()}
        }

        # For numeric columns, compare statistics
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            comparison['statistics'] = {
                'means': {'series1': series1.mean(), 'series2': series2.mean()},
                'stds': {'series1': series1.std(), 'series2': series2.std()},
                'medians': {'series1': series1.median(), 'series2': series2.median()}
            }

        # For categorical columns, compare value distributions
        elif series1.dtype == 'object' and series2.dtype == 'object':
            unique1 = set(series1.dropna().unique())
            unique2 = set(series2.dropna().unique())

            comparison['categorical'] = {
                'unique_counts': {'series1': len(unique1), 'series2': len(unique2)},
                'common_values': list(unique1 & unique2),
                'unique_to_series1': list(unique1 - unique2),
                'unique_to_series2': list(unique2 - unique1)
            }

        return comparison

    def _generate_summary_insights(self, report_data: ReportData) -> None:
        """Generate high-level summary insights."""
        insights = []

        # Finding summary
        finding_counts = defaultdict(int)
        for finding in report_data.findings:
            finding_counts[finding.severity] += 1

        if finding_counts:
            critical_count = finding_counts[SeverityLevel.CRITICAL]
            high_count = finding_counts[SeverityLevel.HIGH]

            if critical_count > 0:
                insights.append(f"⚠️ {critical_count} critical issues require immediate attention")
            if high_count > 0:
                insights.append(f"🔥 {high_count} high-priority issues identified")

        # Data volume summary
        total_rows = sum(len(df) for df in report_data.raw_data.values())
        if total_rows > 1000000:
            insights.append(f"📊 Large dataset analysis: {total_rows:,} total rows processed")

        # Performance summary
        if report_data.execution_metrics.execution_time > 10:
            insights.append(f"⏱️ Long execution time: {report_data.execution_metrics.execution_time:.1f} seconds")

        # Add insights as a special section
        if insights:
            summary_section = ReportSection(
                id="executive_insights",
                title="Key Insights",
                content="<ul><li>" + "</li><li>".join(insights) + "</li></ul>",
                order=0
            )
            report_data.add_section(summary_section)

    def _update_summary_statistics(self, report_data: ReportData) -> None:
        """Update the summary statistics with analysis results."""
        stats = report_data.summary_statistics

        # Add finding statistics
        finding_counts = defaultdict(int)
        for finding in report_data.findings:
            finding_counts[finding.severity.value] += 1

        stats['findings_by_severity'] = dict(finding_counts)
        stats['total_findings'] = len(report_data.findings)

        # Add data statistics
        stats['total_datasets'] = len(report_data.raw_data)
        stats['total_rows'] = sum(len(df) for df in report_data.raw_data.values())
        stats['total_columns'] = sum(len(df.columns) for df in report_data.raw_data.values())

        # Add analysis metadata
        stats['analysis_timestamp'] = datetime.now().isoformat()
        stats['analysis_version'] = "1.0"