"""Base classes and interfaces for report generation."""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from .models import (
    ReportData, ReportConfiguration, ReportFormat, ReportGenerationResult,
    ReportOptions, ReportTemplate, ExecutionMetrics, ReportMetadata
)


class ReportGeneratorError(Exception):
    """Base exception for report generation errors."""
    pass


class TemplateNotFoundError(ReportGeneratorError):
    """Raised when a report template cannot be found."""
    pass


class DataProcessingError(ReportGeneratorError):
    """Raised when there's an error processing report data."""
    pass


class OutputGenerationError(ReportGeneratorError):
    """Raised when there's an error generating the output file."""
    pass


class BaseReportGenerator(ABC):
    """Abstract base class for all report generators."""

    def __init__(self, options: Optional[ReportOptions] = None):
        """Initialize the report generator.

        Args:
            options: Configuration options for report generation
        """
        self.options = options or ReportOptions()
        self.template_paths: List[Path] = []
        self.custom_functions: Dict[str, Any] = {}

    @property
    @abstractmethod
    def supported_format(self) -> ReportFormat:
        """Return the format supported by this generator."""
        pass

    @abstractmethod
    def generate(self, report_data: ReportData) -> ReportGenerationResult:
        """Generate a report from the provided data.

        Args:
            report_data: The data to include in the report

        Returns:
            Result of the report generation process
        """
        pass

    def validate_data(self, report_data: ReportData) -> None:
        """Validate that the report data is suitable for generation.

        Args:
            report_data: The data to validate

        Raises:
            DataProcessingError: If the data is invalid or incomplete
        """
        if not report_data.metadata.title:
            raise DataProcessingError("Report title is required")

        if not report_data.sections and not report_data.raw_data:
            raise DataProcessingError("Report must contain either sections or raw data")

    def add_template_path(self, path: Path) -> None:
        """Add a path to search for templates.

        Args:
            path: Directory path to search for templates
        """
        if path.is_dir():
            self.template_paths.append(path)

    def register_custom_function(self, name: str, func: Any) -> None:
        """Register a custom function for use in templates.

        Args:
            name: Name of the function
            func: The function to register
        """
        self.custom_functions[name] = func

    def _determine_output_path(self, report_data: ReportData) -> Path:
        """Determine the output path for the report.

        Args:
            report_data: The report data containing configuration

        Returns:
            Path where the report should be saved
        """
        if report_data.configuration.output_path:
            return report_data.configuration.output_path

        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = report_data.metadata.title.replace(" ", "_").lower()
        ext = self.supported_format.value

        filename = f"{title}_{timestamp}.{ext}"
        return Path.cwd() / "reports" / filename

    def _calculate_file_size(self, file_path: Path) -> Optional[int]:
        """Calculate the size of the generated file.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes, or None if file doesn't exist
        """
        if file_path.exists():
            return file_path.stat().st_size
        return None

    def _update_execution_metrics(self, report_data: ReportData, start_time: float) -> None:
        """Update execution metrics for the report.

        Args:
            report_data: The report data to update
            start_time: When report generation started
        """
        report_data.execution_metrics.execution_time = time.time() - start_time
        report_data.execution_metrics.queries_executed = len(report_data.data_sources)
        report_data.execution_metrics.rows_processed = sum(
            df.shape[0] for df in report_data.raw_data.values()
        )


class DataProcessor:
    """Utility class for processing and transforming report data."""

    @staticmethod
    def aggregate_data(df: pd.DataFrame, group_by: List[str],
                      aggregations: Dict[str, str]) -> pd.DataFrame:
        """Aggregate data using specified grouping and aggregation functions.

        Args:
            df: DataFrame to aggregate
            group_by: Columns to group by
            aggregations: Dictionary mapping column names to aggregation functions

        Returns:
            Aggregated DataFrame
        """
        return df.groupby(group_by).agg(aggregations).reset_index()

    @staticmethod
    def calculate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing summary statistics
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        if len(numeric_cols) > 0:
            stats.update({
                'numeric_summary': df[numeric_cols].describe().to_dict(),
                'correlations': df[numeric_cols].corr().to_dict()
            })

        return stats

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'iqr') -> Dict[str, List[int]]:
        """Detect outliers in the data.

        Args:
            df: DataFrame to analyze
            columns: Columns to check for outliers (defaults to all numeric)
            method: Method to use for outlier detection ('iqr' or 'zscore')

        Returns:
            Dictionary mapping column names to lists of outlier indices
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()

        outliers = {}

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == 'zscore':
                z_scores = abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > 3
            else:
                continue

            outliers[col] = df[outlier_mask].index.tolist()

        return outliers

    @staticmethod
    def create_trend_data(df: pd.DataFrame, date_column: str,
                         value_column: str, period: str = 'D') -> pd.DataFrame:
        """Create trend data for time series analysis.

        Args:
            df: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column
            period: Period for grouping ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            DataFrame with trend data
        """
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.set_index(date_column)

        trend = df_copy[value_column].resample(period).agg(['count', 'sum', 'mean', 'std'])
        trend = trend.fillna(0).reset_index()

        return trend


class ReportRegistry:
    """Registry for managing report generators and templates."""

    def __init__(self):
        """Initialize the registry."""
        self._generators: Dict[ReportFormat, Type[BaseReportGenerator]] = {}
        self._templates: Dict[str, ReportTemplate] = {}

    def register_generator(self, format: ReportFormat,
                          generator_class: Type[BaseReportGenerator]) -> None:
        """Register a report generator for a specific format.

        Args:
            format: The report format
            generator_class: The generator class
        """
        self._generators[format] = generator_class

    def get_generator(self, format: ReportFormat,
                     options: Optional[ReportOptions] = None) -> BaseReportGenerator:
        """Get a generator instance for the specified format.

        Args:
            format: The desired report format
            options: Configuration options

        Returns:
            Generator instance

        Raises:
            ValueError: If no generator is registered for the format
        """
        if format not in self._generators:
            raise ValueError(f"No generator registered for format: {format}")

        return self._generators[format](options)

    def register_template(self, template: ReportTemplate) -> None:
        """Register a report template.

        Args:
            template: The template to register
        """
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[ReportTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template if found, None otherwise
        """
        return self._templates.get(name)

    def list_available_formats(self) -> List[ReportFormat]:
        """List all available report formats.

        Returns:
            List of supported formats
        """
        return list(self._generators.keys())

    def list_available_templates(self) -> List[str]:
        """List all available template names.

        Returns:
            List of template names
        """
        return list(self._templates.keys())


# Global registry instance
report_registry = ReportRegistry()