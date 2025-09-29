"""CSV report generator implementation."""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import StringIO

import pandas as pd

from ..base import BaseReportGenerator, DataProcessingError, OutputGenerationError
from ..models import (
    ReportData, ReportFormat, ReportGenerationResult, Finding, ReportSection
)


class CSVReportGenerator(BaseReportGenerator):
    """Generator for CSV format reports."""

    @property
    def supported_format(self) -> ReportFormat:
        """Return the CSV format."""
        return ReportFormat.CSV

    def generate(self, report_data: ReportData) -> ReportGenerationResult:
        """Generate a CSV report.

        Args:
            report_data: The data to include in the report

        Returns:
            Result of the report generation process
        """
        start_time = time.time()

        try:
            # Validate input data
            self.validate_data(report_data)

            # Determine output path
            output_path = self._determine_output_path(report_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate CSV content
            if len(report_data.raw_data) == 1:
                # Single dataset - generate single CSV
                self._generate_single_csv(report_data, output_path)
            else:
                # Multiple datasets - generate multiple CSV files or combined report
                self._generate_multi_csv(report_data, output_path)

            # Update execution metrics
            self._update_execution_metrics(report_data, start_time)

            return ReportGenerationResult(
                success=True,
                output_path=output_path,
                format=self.supported_format,
                file_size=self._calculate_file_size(output_path),
                generation_time=time.time() - start_time
            )

        except Exception as e:
            return ReportGenerationResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time
            )

    def _generate_single_csv(self, report_data: ReportData, output_path: Path) -> None:
        """Generate a single CSV file from report data.

        Args:
            report_data: The report data
            output_path: Path to save the CSV file
        """
        if report_data.raw_data:
            # Use the first (and only) dataset
            dataset_name, df = next(iter(report_data.raw_data.items()))

            # Add metadata as comments if supported
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                # Write metadata as comments
                self._write_metadata_header(f, report_data)

                # Write the dataframe
                df.to_csv(f, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            # Generate CSV from sections and findings
            self._generate_report_summary_csv(report_data, output_path)

    def _generate_multi_csv(self, report_data: ReportData, output_path: Path) -> None:
        """Generate multiple CSV files or a combined report.

        Args:
            report_data: The report data
            output_path: Base path for output files
        """
        if self.options.include_raw_data and report_data.raw_data:
            # Create a directory for multiple CSV files
            base_name = output_path.stem
            output_dir = output_path.parent / f"{base_name}_csv_export"
            output_dir.mkdir(exist_ok=True)

            # Generate individual CSV files for each dataset
            for dataset_name, df in report_data.raw_data.items():
                safe_name = self._sanitize_filename(dataset_name)
                csv_path = output_dir / f"{safe_name}.csv"

                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    self._write_dataset_header(f, dataset_name, df)
                    df.to_csv(f, index=False, quoting=csv.QUOTE_NONNUMERIC)

            # Generate summary CSV
            summary_path = output_dir / "summary.csv"
            self._generate_report_summary_csv(report_data, summary_path)

            # Update output path to point to the directory
            output_path = output_dir
        else:
            # Generate combined summary CSV
            self._generate_report_summary_csv(report_data, output_path)

    def _generate_report_summary_csv(self, report_data: ReportData, output_path: Path) -> None:
        """Generate a summary CSV with key report information.

        Args:
            report_data: The report data
            output_path: Path to save the summary CSV
        """
        summary_data = []

        # Add metadata row
        summary_data.append({
            'Type': 'Metadata',
            'Name': 'Report Title',
            'Value': report_data.metadata.title,
            'Description': report_data.metadata.description or '',
            'Generated': report_data.metadata.generated_at.isoformat() if report_data.metadata.generated_at else '',
            'Category': 'Report Info'
        })

        # Add execution metrics
        metrics = report_data.execution_metrics
        summary_data.extend([
            {
                'Type': 'Metric',
                'Name': 'Execution Time',
                'Value': f"{metrics.execution_time:.3f}s",
                'Description': 'Time taken to generate the report',
                'Generated': '',
                'Category': 'Performance'
            },
            {
                'Type': 'Metric',
                'Name': 'Queries Executed',
                'Value': str(metrics.queries_executed),
                'Description': 'Number of database queries executed',
                'Generated': '',
                'Category': 'Performance'
            },
            {
                'Type': 'Metric',
                'Name': 'Rows Processed',
                'Value': str(metrics.rows_processed),
                'Description': 'Total number of data rows processed',
                'Generated': '',
                'Category': 'Performance'
            }
        ])

        # Add data source information
        for ds in report_data.data_sources:
            summary_data.append({
                'Type': 'Data Source',
                'Name': ds.name,
                'Value': ds.type,
                'Description': f"Query count: {ds.query_count}",
                'Generated': ds.last_accessed.isoformat() if ds.last_accessed else '',
                'Category': 'Data Sources'
            })

        # Add findings summary
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            count = len([f for f in report_data.findings if f.severity.value == severity])
            if count > 0:
                summary_data.append({
                    'Type': 'Finding Summary',
                    'Name': f'{severity.title()} Findings',
                    'Value': str(count),
                    'Description': f'Number of {severity} severity findings',
                    'Generated': '',
                    'Category': 'Findings'
                })

        # Add individual findings
        for finding in report_data.findings[:50]:  # Limit to top 50 findings
            summary_data.append({
                'Type': 'Finding',
                'Name': finding.title,
                'Value': finding.severity.value,
                'Description': finding.description[:200] + ('...' if len(finding.description) > 200 else ''),
                'Generated': finding.created_at.isoformat() if finding.created_at else '',
                'Category': finding.category
            })

        # Add summary statistics
        for key, value in report_data.summary_statistics.items():
            summary_data.append({
                'Type': 'Summary Statistic',
                'Name': key,
                'Value': str(value),
                'Description': 'Calculated summary statistic',
                'Generated': '',
                'Category': 'Statistics'
            })

        # Write to CSV
        if summary_data:
            df = pd.DataFrame(summary_data)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                self._write_metadata_header(f, report_data)
                df.to_csv(f, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _write_metadata_header(self, file, report_data: ReportData) -> None:
        """Write metadata as header comments in CSV file.

        Args:
            file: File object to write to
            report_data: The report data
        """
        file.write(f"# Report: {report_data.metadata.title}\n")
        if report_data.metadata.description:
            file.write(f"# Description: {report_data.metadata.description}\n")
        file.write(f"# Generated: {report_data.metadata.generated_at}\n")
        file.write(f"# Generated by: {report_data.metadata.generated_by}\n")
        file.write(f"# Format: {report_data.configuration.format.value}\n")
        file.write("#\n")

    def _write_dataset_header(self, file, dataset_name: str, df: pd.DataFrame) -> None:
        """Write dataset information as header comments.

        Args:
            file: File object to write to
            dataset_name: Name of the dataset
            df: The dataframe
        """
        file.write(f"# Dataset: {dataset_name}\n")
        file.write(f"# Rows: {len(df)}\n")
        file.write(f"# Columns: {len(df.columns)}\n")
        file.write(f"# Data Types: {', '.join(df.dtypes.astype(str).unique())}\n")
        file.write("#\n")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system use.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')

        # Limit length
        return filename[:100]

    def generate_pivot_table(self, report_data: ReportData,
                           dataset_name: str,
                           index_cols: List[str],
                           value_cols: List[str],
                           aggfunc: str = 'sum') -> ReportGenerationResult:
        """Generate a CSV pivot table from specific data.

        Args:
            report_data: The report data
            dataset_name: Name of the dataset to pivot
            index_cols: Columns to use as index
            value_cols: Columns to aggregate
            aggfunc: Aggregation function to use

        Returns:
            Result of the report generation process
        """
        start_time = time.time()

        try:
            if dataset_name not in report_data.raw_data:
                raise DataProcessingError(f"Dataset '{dataset_name}' not found in report data")

            df = report_data.raw_data[dataset_name]

            # Create pivot table
            pivot_table = df.pivot_table(
                index=index_cols,
                values=value_cols,
                aggfunc=aggfunc,
                fill_value=0
            )

            # Determine output path
            output_path = self._determine_output_path(report_data)
            output_path = output_path.with_name(f"{output_path.stem}_pivot.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write pivot table to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# Pivot Table: {dataset_name}\n")
                f.write(f"# Index: {', '.join(index_cols)}\n")
                f.write(f"# Values: {', '.join(value_cols)}\n")
                f.write(f"# Aggregation: {aggfunc}\n")
                f.write("#\n")
                pivot_table.to_csv(f, quoting=csv.QUOTE_NONNUMERIC)

            return ReportGenerationResult(
                success=True,
                output_path=output_path,
                format=self.supported_format,
                file_size=self._calculate_file_size(output_path),
                generation_time=time.time() - start_time
            )

        except Exception as e:
            return ReportGenerationResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time
            )