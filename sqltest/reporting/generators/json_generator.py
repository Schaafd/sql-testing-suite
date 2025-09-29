"""JSON report generator implementation."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import BaseReportGenerator, DataProcessingError, OutputGenerationError
from ..models import (
    ReportData, ReportFormat, ReportGenerationResult, Finding, ReportSection
)


class JSONReportGenerator(BaseReportGenerator):
    """Generator for JSON format reports."""

    @property
    def supported_format(self) -> ReportFormat:
        """Return the JSON format."""
        return ReportFormat.JSON

    def generate(self, report_data: ReportData) -> ReportGenerationResult:
        """Generate a JSON report.

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

            # Convert report data to JSON-serializable format
            json_data = self._convert_to_json_format(report_data)

            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

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

    def _convert_to_json_format(self, report_data: ReportData) -> Dict[str, Any]:
        """Convert ReportData to JSON-serializable format.

        Args:
            report_data: The report data to convert

        Returns:
            Dictionary suitable for JSON serialization
        """
        json_data = {
            "metadata": self._serialize_metadata(report_data),
            "configuration": self._serialize_configuration(report_data),
            "execution_metrics": self._serialize_execution_metrics(report_data),
            "summary_statistics": report_data.summary_statistics,
            "data_sources": self._serialize_data_sources(report_data),
            "sections": self._serialize_sections(report_data.sections),
            "findings": self._serialize_findings(report_data.findings),
            "raw_data": self._serialize_raw_data(report_data) if self.options.include_raw_data else None
        }

        # Remove None values to keep JSON clean
        return {k: v for k, v in json_data.items() if v is not None}

    def _serialize_metadata(self, report_data: ReportData) -> Dict[str, Any]:
        """Serialize metadata to JSON format.

        Args:
            report_data: The report data

        Returns:
            Serialized metadata
        """
        metadata = report_data.metadata
        return {
            "title": metadata.title,
            "description": metadata.description,
            "generated_at": metadata.generated_at.isoformat() if metadata.generated_at else None,
            "generated_by": metadata.generated_by,
            "version": metadata.version,
            "tags": metadata.tags,
            "custom_fields": metadata.custom_fields
        }

    def _serialize_configuration(self, report_data: ReportData) -> Dict[str, Any]:
        """Serialize configuration to JSON format.

        Args:
            report_data: The report data

        Returns:
            Serialized configuration
        """
        config = report_data.configuration
        return {
            "report_type": config.report_type.value,
            "format": config.format.value,
            "title": config.title,
            "description": config.description,
            "output_path": str(config.output_path) if config.output_path else None,
            "template_name": config.template_name,
            "include_sections": config.include_sections,
            "exclude_sections": config.exclude_sections,
            "parameters": config.parameters,
            "styling": config.styling
        }

    def _serialize_execution_metrics(self, report_data: ReportData) -> Dict[str, Any]:
        """Serialize execution metrics to JSON format.

        Args:
            report_data: The report data

        Returns:
            Serialized execution metrics
        """
        metrics = report_data.execution_metrics
        return {
            "execution_time": round(metrics.execution_time, 3),
            "memory_usage": round(metrics.memory_usage, 2),
            "queries_executed": metrics.queries_executed,
            "rows_processed": metrics.rows_processed,
            "cache_hit_rate": round(metrics.cache_hit_rate, 3),
            "errors_encountered": metrics.errors_encountered
        }

    def _serialize_data_sources(self, report_data: ReportData) -> List[Dict[str, Any]]:
        """Serialize data sources to JSON format.

        Args:
            report_data: The report data

        Returns:
            List of serialized data sources
        """
        return [
            {
                "name": ds.name,
                "type": ds.type,
                "query_count": ds.query_count,
                "last_accessed": ds.last_accessed.isoformat() if ds.last_accessed else None,
                "schema_info": ds.schema_info
            }
            for ds in report_data.data_sources
        ]

    def _serialize_sections(self, sections: List[ReportSection]) -> List[Dict[str, Any]]:
        """Serialize report sections to JSON format.

        Args:
            sections: List of report sections

        Returns:
            List of serialized sections
        """
        serialized_sections = []

        for section in sections:
            section_data = {
                "id": section.id,
                "title": section.title,
                "content": section.content,
                "order": section.order,
                "charts": self._serialize_charts(section),
                "tables": self._serialize_tables(section),
                "findings": self._serialize_findings(section.findings),
                "subsections": self._serialize_sections(section.subsections)
            }
            serialized_sections.append(section_data)

        return serialized_sections

    def _serialize_charts(self, section: ReportSection) -> List[Dict[str, Any]]:
        """Serialize chart data to JSON format.

        Args:
            section: Report section containing charts

        Returns:
            List of serialized charts
        """
        if not self.options.include_charts:
            return []

        return [
            {
                "chart_type": chart.chart_type,
                "title": chart.title,
                "data": chart.data,
                "options": chart.options,
                "width": chart.width,
                "height": chart.height
            }
            for chart in section.charts
        ]

    def _serialize_tables(self, section: ReportSection) -> List[Dict[str, Any]]:
        """Serialize table data to JSON format.

        Args:
            section: Report section containing tables

        Returns:
            List of serialized tables
        """
        serialized_tables = []

        for i, table in enumerate(section.tables):
            if isinstance(table, pd.DataFrame):
                # Limit rows if specified in options
                limited_table = table.head(self.options.max_rows_per_table)

                table_data = {
                    "table_id": f"{section.id}_table_{i}",
                    "columns": limited_table.columns.tolist(),
                    "data": limited_table.to_dict('records'),
                    "total_rows": len(table),
                    "displayed_rows": len(limited_table),
                    "data_types": limited_table.dtypes.astype(str).to_dict()
                }

                # Add summary statistics if numeric columns exist
                numeric_cols = limited_table.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    table_data["summary_stats"] = limited_table[numeric_cols].describe().to_dict()

                serialized_tables.append(table_data)

        return serialized_tables

    def _serialize_findings(self, findings: List[Finding]) -> List[Dict[str, Any]]:
        """Serialize findings to JSON format.

        Args:
            findings: List of findings

        Returns:
            List of serialized findings
        """
        return [
            {
                "id": finding.id,
                "title": finding.title,
                "description": finding.description,
                "severity": finding.severity.value,
                "category": finding.category,
                "details": finding.details,
                "recommendations": finding.recommendations,
                "affected_objects": finding.affected_objects,
                "created_at": finding.created_at.isoformat() if finding.created_at else None
            }
            for finding in findings
        ]

    def _serialize_raw_data(self, report_data: ReportData) -> Dict[str, Any]:
        """Serialize raw data to JSON format.

        Args:
            report_data: The report data

        Returns:
            Dictionary containing serialized raw data
        """
        raw_data = {}

        for key, df in report_data.raw_data.items():
            if isinstance(df, pd.DataFrame):
                # Limit rows if specified in options
                limited_df = df.head(self.options.max_rows_per_table)

                raw_data[key] = {
                    "columns": limited_df.columns.tolist(),
                    "data": limited_df.to_dict('records'),
                    "total_rows": len(df),
                    "displayed_rows": len(limited_df),
                    "data_types": limited_df.dtypes.astype(str).to_dict()
                }

        return raw_data

    def generate_compact(self, report_data: ReportData) -> ReportGenerationResult:
        """Generate a compact JSON report without formatting.

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

            # Convert report data to JSON-serializable format
            json_data = self._convert_to_json_format(report_data)

            # Write compact JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, separators=(',', ':'), ensure_ascii=False, default=str)

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