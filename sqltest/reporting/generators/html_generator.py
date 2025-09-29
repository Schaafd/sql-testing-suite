"""HTML report generator implementation with Jinja2 templates."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ..base import BaseReportGenerator, TemplateNotFoundError, OutputGenerationError
from ..models import (
    ReportData, ReportFormat, ReportGenerationResult, Finding, ReportSection, ChartData
)


class HTMLReportGenerator(BaseReportGenerator):
    """Generator for HTML format reports with interactive features."""

    def __init__(self, *args, **kwargs):
        """Initialize the HTML generator."""
        super().__init__(*args, **kwargs)
        self.jinja_env: Optional[Environment] = None
        self._setup_jinja_environment()

    @property
    def supported_format(self) -> ReportFormat:
        """Return the HTML format."""
        return ReportFormat.HTML

    def _setup_jinja_environment(self) -> None:
        """Set up the Jinja2 environment for template rendering."""
        # Default template path
        default_template_path = Path(__file__).parent.parent / "templates"
        if default_template_path.exists():
            self.template_paths.append(default_template_path)

        if self.template_paths:
            self.jinja_env = Environment(
                loader=FileSystemLoader([str(p) for p in self.template_paths]),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )

            # Register custom functions
            self._register_template_functions()

    def _register_template_functions(self) -> None:
        """Register custom functions for use in templates."""
        if self.jinja_env:
            self.jinja_env.globals.update({
                'format_number': self._format_number,
                'format_percentage': self._format_percentage,
                'format_datetime': self._format_datetime,
                'severity_color': self._get_severity_color,
                'chart_config': self._generate_chart_config,
                'table_to_html': self._dataframe_to_html,
                'json_dumps': json.dumps
            })

    def generate(self, report_data: ReportData) -> ReportGenerationResult:
        """Generate an HTML report.

        Args:
            report_data: The data to include in the report

        Returns:
            Result of the report generation process
        """
        start_time = time.time()

        try:
            # Validate input data
            self.validate_data(report_data)

            # Set up Jinja environment if not already done
            if not self.jinja_env:
                self._setup_jinja_environment()

            # Determine output path
            output_path = self._determine_output_path(report_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate HTML content
            html_content = self._generate_html_content(report_data)

            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Copy static assets if needed
            self._copy_static_assets(output_path.parent)

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

    def _generate_html_content(self, report_data: ReportData) -> str:
        """Generate the HTML content for the report.

        Args:
            report_data: The report data

        Returns:
            Generated HTML content
        """
        # Determine template to use
        template_name = report_data.configuration.template_name or "default_report.html"

        # Try to load template
        template = self._load_template(template_name)

        # Prepare context data
        context = self._prepare_template_context(report_data)

        # Render template
        return template.render(**context)

    def _load_template(self, template_name: str) -> Template:
        """Load a Jinja2 template.

        Args:
            template_name: Name of the template file

        Returns:
            Loaded template

        Raises:
            TemplateNotFoundError: If template cannot be found
        """
        try:
            if self.jinja_env:
                return self.jinja_env.get_template(template_name)
            else:
                # Fallback to default template
                return Template(self._get_default_template())
        except Exception as e:
            raise TemplateNotFoundError(f"Template '{template_name}' not found: {e}")

    def _prepare_template_context(self, report_data: ReportData) -> Dict[str, Any]:
        """Prepare the context data for template rendering.

        Args:
            report_data: The report data

        Returns:
            Context dictionary for template rendering
        """
        context = {
            'report': report_data,
            'metadata': report_data.metadata,
            'config': report_data.configuration,
            'metrics': report_data.execution_metrics,
            'sections': report_data.sections,
            'findings': report_data.findings,
            'data_sources': report_data.data_sources,
            'summary_stats': report_data.summary_statistics,
            'options': self.options,
            'generation_time': datetime.now(),
            'findings_by_severity': self._group_findings_by_severity(report_data.findings),
            'chart_data': self._prepare_chart_data(report_data),
            'table_data': self._prepare_table_data(report_data)
        }

        # Add custom functions
        context.update(self.custom_functions)

        return context

    def _group_findings_by_severity(self, findings: List[Finding]) -> Dict[str, List[Finding]]:
        """Group findings by severity level.

        Args:
            findings: List of findings

        Returns:
            Dictionary mapping severity levels to findings
        """
        grouped = {}
        for finding in findings:
            severity = finding.severity.value
            if severity not in grouped:
                grouped[severity] = []
            grouped[severity].append(finding)
        return grouped

    def _prepare_chart_data(self, report_data: ReportData) -> List[Dict[str, Any]]:
        """Prepare chart data for JavaScript rendering.

        Args:
            report_data: The report data

        Returns:
            List of chart configurations
        """
        charts = []

        for section in report_data.sections:
            for chart in section.charts:
                chart_config = {
                    'id': f"chart_{section.id}_{len(charts)}",
                    'type': chart.chart_type,
                    'title': chart.title,
                    'data': chart.data,
                    'options': chart.options,
                    'width': chart.width or 400,
                    'height': chart.height or 300
                }
                charts.append(chart_config)

        return charts

    def _prepare_table_data(self, report_data: ReportData) -> List[Dict[str, Any]]:
        """Prepare table data for HTML rendering.

        Args:
            report_data: The report data

        Returns:
            List of table configurations
        """
        tables = []

        for section in report_data.sections:
            for i, table in enumerate(section.tables):
                if isinstance(table, pd.DataFrame):
                    # Limit rows for display
                    display_df = table.head(self.options.max_rows_per_table)

                    table_config = {
                        'id': f"table_{section.id}_{i}",
                        'title': f"Table {i + 1}",
                        'html': self._dataframe_to_html(display_df),
                        'total_rows': len(table),
                        'displayed_rows': len(display_df),
                        'columns': table.columns.tolist(),
                        'has_more': len(table) > self.options.max_rows_per_table
                    }
                    tables.append(table_config)

        return tables

    def _copy_static_assets(self, output_dir: Path) -> None:
        """Copy static assets (CSS, JS) to output directory.

        Args:
            output_dir: Directory to copy assets to
        """
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Create CSS file
        css_content = self._get_default_css()
        (assets_dir / "report.css").write_text(css_content, encoding='utf-8')

        # Create JavaScript file
        js_content = self._get_default_js()
        (assets_dir / "report.js").write_text(js_content, encoding='utf-8')

    # Template helper functions
    def _format_number(self, value: Any, decimals: int = 2) -> str:
        """Format a number for display."""
        try:
            if isinstance(value, (int, float)):
                return f"{value:,.{decimals}f}"
            return str(value)
        except:
            return str(value)

    def _format_percentage(self, value: Any, decimals: int = 1) -> str:
        """Format a percentage for display."""
        try:
            if isinstance(value, (int, float)):
                return f"{value * 100:.{decimals}f}%"
            return str(value)
        except:
            return str(value)

    def _format_datetime(self, value: Any, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format a datetime for display."""
        try:
            if isinstance(value, datetime):
                return value.strftime(format_str)
            elif isinstance(value, str):
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.strftime(format_str)
            return str(value)
        except:
            return str(value)

    def _get_severity_color(self, severity: str) -> str:
        """Get color code for severity level."""
        colors = {
            'critical': '#dc3545',  # Red
            'high': '#fd7e14',      # Orange
            'medium': '#ffc107',    # Yellow
            'low': '#20c997',       # Teal
            'info': '#0dcaf0'       # Cyan
        }
        return colors.get(severity.lower(), '#6c757d')

    def _generate_chart_config(self, chart_data: ChartData) -> Dict[str, Any]:
        """Generate Chart.js configuration."""
        return {
            'type': chart_data.chart_type,
            'data': chart_data.data,
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': chart_data.title
                    }
                },
                **chart_data.options
            }
        }

    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to HTML table."""
        return df.to_html(
            classes=['table', 'table-striped', 'table-hover'],
            table_id=None,
            escape=False,
            index=False
        )

    def _get_default_template(self) -> str:
        """Get the default HTML template."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/report.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="display-4">{{ metadata.title }}</h1>
                <p class="lead">{{ metadata.description or "Generated report" }}</p>
                <small class="text-muted">Generated on {{ format_datetime(metadata.generated_at) }} by {{ metadata.generated_by }}</small>
            </div>
        </div>

        <!-- Executive Summary -->
        {% if options.include_executive_summary %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h2>Executive Summary</h2>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <h3>{{ format_number(metrics.rows_processed) }}</h3>
                                    <p>Rows Processed</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <h3>{{ metrics.queries_executed }}</h3>
                                    <p>Queries Executed</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <h3>{{ format_number(metrics.execution_time, 2) }}s</h3>
                                    <p>Execution Time</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <h3>{{ findings|length }}</h3>
                                    <p>Total Findings</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Findings Summary -->
        {% if findings %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h2>Findings Summary</h2>
                    </div>
                    <div class="card-body">
                        {% for severity, severity_findings in findings_by_severity.items() %}
                        <div class="finding-group mb-3">
                            <h4 style="color: {{ severity_color(severity) }}">
                                {{ severity.title() }} ({{ severity_findings|length }})
                            </h4>
                            {% for finding in severity_findings[:5] %}
                            <div class="finding-item">
                                <strong>{{ finding.title }}</strong>
                                <p>{{ finding.description[:200] }}{% if finding.description|length > 200 %}...{% endif %}</p>
                            </div>
                            {% endfor %}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Report Sections -->
        {% for section in sections %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h2>{{ section.title }}</h2>
                    </div>
                    <div class="card-body">
                        {{ section.content|safe }}

                        <!-- Charts -->
                        {% for chart in section.charts %}
                        <div class="chart-container mb-4">
                            <canvas id="chart_{{ loop.index0 }}"></canvas>
                        </div>
                        {% endfor %}

                        <!-- Tables -->
                        {% for table in section.tables %}
                        <div class="table-container mb-4">
                            {{ table_to_html(table)|safe }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="assets/report.js"></script>
    <script>
        // Initialize charts
        {% for chart in chart_data %}
        const chart{{ loop.index0 }} = new Chart(
            document.getElementById('{{ chart.id }}'),
            {{ json_dumps(chart)|safe }}
        );
        {% endfor %}
    </script>
</body>
</html>
        '''

    def _get_default_css(self) -> str:
        """Get the default CSS styles."""
        return '''
.metric-card {
    text-align: center;
    padding: 20px;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    background: #f8f9fa;
}

.metric-card h3 {
    font-size: 2rem;
    color: #0d6efd;
    margin-bottom: 5px;
}

.metric-card p {
    color: #6c757d;
    margin: 0;
}

.finding-group {
    border-left: 4px solid #dee2e6;
    padding-left: 20px;
}

.finding-item {
    padding: 10px;
    margin: 10px 0;
    background: #f8f9fa;
    border-radius: 4px;
}

.chart-container {
    position: relative;
    height: 400px;
    margin: 20px 0;
}

.table-container {
    overflow-x: auto;
}

.table {
    margin-bottom: 0;
}

.severity-critical { color: #dc3545; }
.severity-high { color: #fd7e14; }
.severity-medium { color: #ffc107; }
.severity-low { color: #20c997; }
.severity-info { color: #0dcaf0; }
        '''

    def _get_default_js(self) -> str:
        """Get the default JavaScript code."""
        return '''
// Report utility functions
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    }
}

function exportTableToCsv(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) return;

    let csv = [];
    const rows = table.querySelectorAll('tr');

    for (let i = 0; i < rows.length; i++) {
        const row = [];
        const cols = rows[i].querySelectorAll('td, th');

        for (let j = 0; j < cols.length; j++) {
            row.push('"' + cols[j].innerText.replace(/"/g, '""') + '"');
        }

        csv.push(row.join(','));
    }

    downloadCsv(csv.join('\\n'), filename);
}

function downloadCsv(csv, filename) {
    const csvFile = new Blob([csv], { type: 'text/csv' });
    const downloadLink = document.createElement('a');

    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = 'none';

    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
        '''