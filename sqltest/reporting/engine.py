"""Main reporting engine that orchestrates report generation."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base import report_registry, BaseReportGenerator, ReportGeneratorError
from .models import (
    ReportData, ReportConfiguration, ReportFormat, ReportGenerationResult,
    ReportOptions, ReportMetadata, ExecutionMetrics, DataSource, SeverityLevel
)
from .analyzer import ReportAnalyzer
from .generators.json_generator import JSONReportGenerator
from .generators.csv_generator import CSVReportGenerator
from .generators.html_generator import HTMLReportGenerator
from .interactive import TrendAnalyzer

logger = logging.getLogger(__name__)


class ReportingEngine:
    """Main engine for generating comprehensive reports with multiple formats and analysis."""

    def __init__(self):
        """Initialize the reporting engine."""
        self.analyzer = ReportAnalyzer()
        self._register_default_generators()
        self._setup_logging()

    def _register_default_generators(self) -> None:
        """Register the default report generators."""
        report_registry.register_generator(ReportFormat.JSON, JSONReportGenerator)
        report_registry.register_generator(ReportFormat.CSV, CSVReportGenerator)
        report_registry.register_generator(ReportFormat.HTML, HTMLReportGenerator)

    def _setup_logging(self) -> None:
        """Set up logging for the reporting engine."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def generate_report(self,
                       data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                       config: ReportConfiguration,
                       options: Optional[ReportOptions] = None,
                       analyze: bool = True,
                       analysis_types: Optional[List[str]] = None) -> ReportGenerationResult:
        """Generate a comprehensive report from data.

        Args:
            data: Data to include in the report (DataFrame or dict of DataFrames)
            config: Report configuration
            options: Report generation options
            analyze: Whether to perform data analysis
            analysis_types: Specific analysis types to perform

        Returns:
            Result of the report generation process
        """
        start_time = time.time()

        try:
            logger.info(f"Starting report generation: {config.title}")

            # Prepare report data
            report_data = self._prepare_report_data(data, config, options)

            # Perform analysis if requested
            if analyze:
                logger.info("Performing data analysis")
                report_data = self.analyzer.analyze_report_data(report_data, analysis_types)

            # Generate the report
            generator = report_registry.get_generator(config.format, options)
            result = generator.generate(report_data)

            # Update result with additional metadata
            result.metadata = {
                'total_datasets': len(report_data.raw_data),
                'total_rows': sum(len(df) for df in report_data.raw_data.values()),
                'total_findings': len(report_data.findings),
                'analysis_performed': analyze
            }

            logger.info(f"Report generation completed in {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ReportGenerationResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time
            )

    def generate_multi_format_report(self,
                                   data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                                   base_config: ReportConfiguration,
                                   formats: List[ReportFormat],
                                   options: Optional[ReportOptions] = None) -> Dict[ReportFormat, ReportGenerationResult]:
        """Generate reports in multiple formats from the same data.

        Args:
            data: Data to include in the reports
            base_config: Base configuration (format will be overridden)
            formats: List of formats to generate
            options: Report generation options

        Returns:
            Dictionary mapping formats to generation results
        """
        results = {}

        # Prepare report data once
        report_data = self._prepare_report_data(data, base_config, options)

        # Perform analysis once
        logger.info("Performing data analysis for multi-format report")
        report_data = self.analyzer.analyze_report_data(report_data)

        # Generate each format
        for format_type in formats:
            try:
                logger.info(f"Generating {format_type.value} format")

                # Create format-specific config
                format_config = ReportConfiguration(
                    report_type=base_config.report_type,
                    format=format_type,
                    title=base_config.title,
                    description=base_config.description,
                    output_path=self._get_format_specific_path(base_config.output_path, format_type),
                    template_name=base_config.template_name,
                    include_sections=base_config.include_sections,
                    exclude_sections=base_config.exclude_sections,
                    parameters=base_config.parameters,
                    styling=base_config.styling
                )

                # Update report data config
                report_data.configuration = format_config

                # Generate report
                generator = report_registry.get_generator(format_type, options)
                result = generator.generate(report_data)
                results[format_type] = result

            except Exception as e:
                logger.error(f"Error generating {format_type.value} format: {e}")
                results[format_type] = ReportGenerationResult(
                    success=False,
                    error_message=str(e)
                )

        return results

    def create_report_from_query_results(self,
                                       query_results: Dict[str, Any],
                                       title: str,
                                       description: Optional[str] = None,
                                       format_type: ReportFormat = ReportFormat.HTML) -> ReportGenerationResult:
        """Create a report from SQL query results.

        Args:
            query_results: Dictionary containing query results and metadata
            title: Report title
            description: Report description
            format_type: Output format

        Returns:
            Result of the report generation process
        """
        # Extract data and metadata from query results
        data = query_results.get('data', {})
        execution_info = query_results.get('execution_info', {})

        # Create configuration
        config = ReportConfiguration(
            report_type="detailed",
            format=format_type,
            title=title,
            description=description or "Report generated from query results"
        )

        # Create execution metrics from query info
        execution_metrics = ExecutionMetrics(
            execution_time=execution_info.get('execution_time', 0.0),
            memory_usage=execution_info.get('memory_usage', 0.0),
            queries_executed=execution_info.get('query_count', 1),
            rows_processed=sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame))
        )

        # Generate the report
        return self.generate_report(data, config)

    def _prepare_report_data(self,
                           data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                           config: ReportConfiguration,
                           options: Optional[ReportOptions]) -> ReportData:
        """Prepare ReportData object from input data and configuration.

        Args:
            data: Input data
            config: Report configuration
            options: Report options

        Returns:
            Prepared ReportData object
        """
        # Normalize data to dict format
        if isinstance(data, pd.DataFrame):
            raw_data = {'main_dataset': data}
        else:
            raw_data = data

        # Create metadata
        metadata = ReportMetadata(
            title=config.title,
            description=config.description or "Generated report",
            generated_at=datetime.now(),
            generated_by="SQLTest Pro Reporting Engine",
            version="1.0"
        )

        # Create data sources info
        data_sources = []
        for name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                data_source = DataSource(
                    name=name,
                    type="DataFrame",
                    query_count=1,
                    last_accessed=datetime.now(),
                    schema_info={
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.astype(str).to_dict(),
                        'shape': df.shape
                    }
                )
                data_sources.append(data_source)

        # Create execution metrics placeholder
        execution_metrics = ExecutionMetrics(
            execution_time=0.0,
            memory_usage=0.0,
            queries_executed=len(data_sources),
            rows_processed=sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
        )

        # Create report data
        report_data = ReportData(
            metadata=metadata,
            configuration=config,
            data_sources=data_sources,
            execution_metrics=execution_metrics,
            raw_data=raw_data
        )

        return report_data

    def _get_format_specific_path(self, base_path: Optional[Path], format_type: ReportFormat) -> Optional[Path]:
        """Generate format-specific output path.

        Args:
            base_path: Base output path
            format_type: Report format

        Returns:
            Format-specific path
        """
        if base_path is None:
            return None

        # Change extension based on format
        if base_path.suffix:
            new_path = base_path.with_suffix(f'.{format_type.value}')
        else:
            new_path = base_path / f"report.{format_type.value}"

        return new_path

    def create_dashboard_report(self,
                              datasets: Dict[str, pd.DataFrame],
                              title: str = "Executive Dashboard",
                              include_charts: bool = True) -> ReportGenerationResult:
        """Create an executive dashboard-style report.

        Args:
            datasets: Dictionary of datasets to include
            title: Dashboard title
            include_charts: Whether to include charts

        Returns:
            Result of the report generation process
        """
        config = ReportConfiguration(
            report_type="executive",
            format=ReportFormat.HTML,
            title=title,
            description="Executive dashboard with key metrics and insights"
        )

        options = ReportOptions(
            include_charts=include_charts,
            include_executive_summary=True,
            include_raw_data=False,
            max_rows_per_table=50
        )

        return self.generate_report(datasets, config, options)

    def create_technical_report(self,
                              datasets: Dict[str, pd.DataFrame],
                              title: str = "Technical Analysis Report") -> ReportGenerationResult:
        """Create a detailed technical report.

        Args:
            datasets: Dictionary of datasets to include
            title: Report title

        Returns:
            Result of the report generation process
        """
        config = ReportConfiguration(
            report_type="technical",
            format=ReportFormat.HTML,
            title=title,
            description="Detailed technical analysis with comprehensive data insights"
        )

        options = ReportOptions(
            include_charts=True,
            include_raw_data=True,
            include_executive_summary=False,
            max_rows_per_table=500
        )

        return self.generate_report(datasets, config, options, analyze=True)

    def create_interactive_dashboard(self,
                                   datasets: Dict[str, pd.DataFrame],
                                   title: str = "Interactive Dashboard",
                                   include_trends: bool = True) -> ReportGenerationResult:
        """Create an interactive dashboard with advanced features.

        Args:
            datasets: Dictionary of datasets to include
            title: Dashboard title
            include_trends: Whether to include trend analysis

        Returns:
            Result of the dashboard generation process
        """
        config = ReportConfiguration(
            report_type="executive",
            format=ReportFormat.HTML,
            title=title,
            description="Interactive dashboard with real-time features and advanced analytics"
        )

        options = ReportOptions(
            include_charts=True,
            include_executive_summary=True,
            include_raw_data=False,
            max_rows_per_table=100
        )

        # Prepare report data
        report_data = self._prepare_report_data(datasets, config, options)

        # Perform comprehensive analysis
        if include_trends:
            report_data = self.analyzer.analyze_report_data(report_data)

        # Use HTML generator to create interactive dashboard
        generator = report_registry.get_generator(ReportFormat.HTML, options)
        if hasattr(generator, 'generate_interactive_dashboard'):
            return generator.generate_interactive_dashboard(report_data)
        else:
            return generator.generate(report_data)

    def create_executive_summary_report(self, datasets: Dict[str, pd.DataFrame],
                                      title: str = "Executive Summary") -> Dict[str, Any]:
        """Create executive summary with key insights and recommendations.

        Args:
            datasets: Dictionary of datasets to analyze
            title: Report title

        Returns:
            Dictionary containing executive summary data
        """
        # Prepare minimal report data for analysis
        config = ReportConfiguration(
            report_type="executive",
            format=ReportFormat.JSON,
            title=title
        )

        report_data = self._prepare_report_data(datasets, config, None)

        # Perform analysis to generate findings
        report_data = self.analyzer.analyze_report_data(report_data)

        # Generate executive summary
        return TrendAnalyzer.generate_executive_summary(report_data)

    def analyze_trends(self, data: pd.DataFrame, date_column: str,
                      value_column: str, periods: int = 30) -> Dict[str, Any]:
        """Analyze trends in time series data.

        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column
            periods: Number of periods to forecast

        Returns:
            Dictionary containing trend analysis results
        """
        return TrendAnalyzer.analyze_time_series(data, date_column, value_column, periods)

    def export_findings_to_csv(self, report_data: ReportData, output_path: Path) -> bool:
        """Export findings to a CSV file.

        Args:
            report_data: Report data containing findings
            output_path: Path to save the CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            findings_data = []
            for finding in report_data.findings:
                findings_data.append({
                    'ID': finding.id,
                    'Title': finding.title,
                    'Description': finding.description,
                    'Severity': finding.severity.value,
                    'Category': finding.category,
                    'Created': finding.created_at.isoformat() if finding.created_at else '',
                    'Affected Objects': ', '.join(finding.affected_objects) if finding.affected_objects else '',
                    'Recommendations': ' | '.join(finding.recommendations) if finding.recommendations else ''
                })

            if findings_data:
                df = pd.DataFrame(findings_data)
                df.to_csv(output_path, index=False)
                logger.info(f"Findings exported to {output_path}")
                return True
            else:
                logger.warning("No findings to export")
                return False

        except Exception as e:
            logger.error(f"Error exporting findings: {e}")
            return False

    def validate_data_for_reporting(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> List[str]:
        """Validate data suitability for reporting.

        Args:
            data: Data to validate

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Normalize data
        if isinstance(data, pd.DataFrame):
            datasets = {'main': data}
        else:
            datasets = data

        for name, df in datasets.items():
            if not isinstance(df, pd.DataFrame):
                warnings.append(f"Dataset '{name}' is not a pandas DataFrame")
                continue

            # Check if empty
            if df.empty:
                warnings.append(f"Dataset '{name}' is empty")

            # Check for all-null columns
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                warnings.append(f"Dataset '{name}' has columns with all null values: {null_cols}")

            # Check for very high null percentage
            null_percentages = df.isnull().mean()
            high_null_cols = null_percentages[null_percentages > 0.8].index.tolist()
            if high_null_cols:
                warnings.append(f"Dataset '{name}' has columns with >80% null values: {high_null_cols}")

            # Check for very high cardinality
            for col in df.select_dtypes(include=['object']).columns:
                cardinality_ratio = df[col].nunique() / len(df)
                if cardinality_ratio > 0.95 and len(df) > 100:
                    warnings.append(f"Dataset '{name}' column '{col}' has very high cardinality ({cardinality_ratio:.1%})")

        return warnings

    def get_available_formats(self) -> List[ReportFormat]:
        """Get list of available report formats.

        Returns:
            List of available formats
        """
        return report_registry.list_available_formats()

    def get_report_statistics(self, report_data: ReportData) -> Dict[str, Any]:
        """Get comprehensive statistics about a report.

        Args:
            report_data: Report data to analyze

        Returns:
            Dictionary containing report statistics
        """
        return {
            'metadata': {
                'title': report_data.metadata.title,
                'generated_at': report_data.metadata.generated_at,
                'version': report_data.metadata.version
            },
            'data_summary': {
                'total_datasets': len(report_data.raw_data),
                'total_rows': sum(len(df) for df in report_data.raw_data.values()),
                'total_columns': sum(len(df.columns) for df in report_data.raw_data.values()),
                'dataset_info': {
                    name: {'rows': len(df), 'columns': len(df.columns)}
                    for name, df in report_data.raw_data.items()
                }
            },
            'findings_summary': {
                'total_findings': len(report_data.findings),
                'by_severity': {
                    severity.value: len([f for f in report_data.findings if f.severity == severity])
                    for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.INFO]
                },
                'by_category': {}  # This would be populated with category counts
            },
            'performance': {
                'execution_time': report_data.execution_metrics.execution_time,
                'memory_usage': report_data.execution_metrics.memory_usage,
                'queries_executed': report_data.execution_metrics.queries_executed,
                'cache_hit_rate': report_data.execution_metrics.cache_hit_rate
            }
        }