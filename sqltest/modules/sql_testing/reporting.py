"""Enterprise test reporting and metrics for SQL unit testing framework.

This module provides comprehensive reporting capabilities including:
- Real-time test execution dashboards
- Performance metrics and analytics
- Test coverage analysis
- Historical trend reporting
- Multi-format report generation (HTML, JSON, XML, CSV)
"""
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
from collections import defaultdict

from .models import TestResult, TestSuiteResult, TestStatus, TestIsolationLevel

logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Comprehensive test execution metrics."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0

    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0

    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0

    coverage_percentage: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def assertion_success_rate(self) -> float:
        """Calculate assertion success rate."""
        if self.total_assertions == 0:
            return 0.0
        return (self.passed_assertions / self.total_assertions) * 100


@dataclass
class PerformanceMetrics:
    """Performance-specific metrics."""
    throughput_tests_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    database_connections_used: int = 0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0


@dataclass
class TestCoverageReport:
    """Test coverage analysis report."""
    total_tables: int = 0
    tested_tables: int = 0
    total_columns: int = 0
    tested_columns: int = 0
    total_functions: int = 0
    tested_functions: int = 0

    coverage_by_table: Dict[str, float] = None
    coverage_by_schema: Dict[str, float] = None
    untested_objects: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.coverage_by_table is None:
            self.coverage_by_table = {}
        if self.coverage_by_schema is None:
            self.coverage_by_schema = {}
        if self.untested_objects is None:
            self.untested_objects = []

    @property
    def table_coverage_percentage(self) -> float:
        """Calculate table coverage percentage."""
        if self.total_tables == 0:
            return 0.0
        return (self.tested_tables / self.total_tables) * 100

    @property
    def column_coverage_percentage(self) -> float:
        """Calculate column coverage percentage."""
        if self.total_columns == 0:
            return 0.0
        return (self.tested_columns / self.total_columns) * 100


class TestReportGenerator:
    """Enterprise-grade test report generator."""

    def __init__(self, output_dir: Path = None):
        """Initialize report generator."""
        self.output_dir = output_dir or Path('./reports')
        self.output_dir.mkdir(exist_ok=True)

        # Historical data storage
        self._execution_history: List[Dict[str, Any]] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def generate_comprehensive_report(self,
                                    suite_results: List[TestSuiteResult],
                                    performance_metrics: Optional[PerformanceMetrics] = None,
                                    coverage_report: Optional[TestCoverageReport] = None,
                                    output_format: str = 'html') -> Path:
        """Generate comprehensive test report."""

        # Calculate metrics
        test_metrics = self._calculate_test_metrics(suite_results)

        # Store in history
        self._store_execution_history(suite_results, test_metrics, performance_metrics)

        # Generate report based on format
        if output_format.lower() == 'html':
            return self._generate_html_report(suite_results, test_metrics, performance_metrics, coverage_report)
        elif output_format.lower() == 'json':
            return self._generate_json_report(suite_results, test_metrics, performance_metrics, coverage_report)
        elif output_format.lower() == 'csv':
            return self._generate_csv_report(suite_results, test_metrics)
        elif output_format.lower() == 'xml':
            return self._generate_xml_report(suite_results, test_metrics)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _calculate_test_metrics(self, suite_results: List[TestSuiteResult]) -> TestMetrics:
        """Calculate comprehensive test metrics."""
        metrics = TestMetrics()
        execution_times = []

        for suite_result in suite_results:
            for test_result in suite_result.test_results:
                metrics.total_tests += 1

                if test_result.status == TestStatus.PASSED:
                    metrics.passed_tests += 1
                elif test_result.status == TestStatus.FAILED:
                    metrics.failed_tests += 1
                elif test_result.status == TestStatus.ERROR:
                    metrics.error_tests += 1
                elif test_result.status == TestStatus.SKIPPED:
                    metrics.skipped_tests += 1

                # Execution time metrics
                if test_result.execution_time:
                    exec_time_ms = test_result.execution_time * 1000
                    execution_times.append(exec_time_ms)
                    metrics.total_execution_time_ms += exec_time_ms

                # Assertion metrics
                for assertion_result in test_result.assertion_results:
                    metrics.total_assertions += 1
                    if assertion_result.get('passed', False):
                        metrics.passed_assertions += 1
                    else:
                        metrics.failed_assertions += 1

        # Calculate derived metrics
        if execution_times:
            metrics.avg_execution_time_ms = sum(execution_times) / len(execution_times)
            metrics.min_execution_time_ms = min(execution_times)
            metrics.max_execution_time_ms = max(execution_times)

        return metrics

    def _generate_html_report(self,
                            suite_results: List[TestSuiteResult],
                            test_metrics: TestMetrics,
                            performance_metrics: Optional[PerformanceMetrics],
                            coverage_report: Optional[TestCoverageReport]) -> Path:
        """Generate interactive HTML report."""

        # Generate HTML content
        html_content = self._create_html_template(
            suite_results, test_metrics, performance_metrics, coverage_report
        )

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"test_report_{timestamp}.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _create_html_template(self,
                            suite_results: List[TestSuiteResult],
                            test_metrics: TestMetrics,
                            performance_metrics: Optional[PerformanceMetrics],
                            coverage_report: Optional[TestCoverageReport]) -> str:
        """Create HTML report template."""

        # Generate summary statistics
        summary_html = self._generate_summary_html(test_metrics, performance_metrics)

        # Generate test results table
        results_html = self._generate_results_table_html(suite_results)

        # Generate charts data
        charts_data = self._generate_charts_data(suite_results, test_metrics)

        # Generate coverage section
        coverage_html = self._generate_coverage_html(coverage_report) if coverage_report else ""

        # Main HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SQL Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .charts-section {{
            margin-bottom: 40px;
        }}
        .chart-container {{
            width: 100%;
            height: 400px;
            margin-bottom: 30px;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table th, .results-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .results-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .status-passed {{ color: #28a745; font-weight: bold; }}
        .status-failed {{ color: #dc3545; font-weight: bold; }}
        .status-error {{ color: #fd7e14; font-weight: bold; }}
        .status-skipped {{ color: #6c757d; font-weight: bold; }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            border-left: 4px solid #007acc;
            padding-left: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SQL Unit Test Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>

        {summary_html}

        <div class="charts-section">
            <h2 class="section-title">Test Results Overview</h2>
            <div class="chart-container">
                <canvas id="resultsChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        {coverage_html}

        <div class="results-section">
            <h2 class="section-title">Detailed Test Results</h2>
            {results_html}
        </div>
    </div>

    <script>
        // Charts data
        const chartsData = {charts_data};

        // Results pie chart
        const resultsCtx = document.getElementById('resultsChart').getContext('2d');
        new Chart(resultsCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Passed', 'Failed', 'Error', 'Skipped'],
                datasets: [{{
                    data: chartsData.results,
                    backgroundColor: ['#28a745', '#dc3545', '#fd7e14', '#6c757d']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Test Results Distribution'
                    }}
                }}
            }}
        }});

        // Performance chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {{
            type: 'bar',
            data: {{
                labels: chartsData.performance.labels,
                datasets: [{{
                    label: 'Execution Time (ms)',
                    data: chartsData.performance.data,
                    backgroundColor: '#007acc'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Test Execution Performance'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        return html_template

    def _generate_summary_html(self, test_metrics: TestMetrics, performance_metrics: Optional[PerformanceMetrics]) -> str:
        """Generate summary metrics HTML."""
        return f"""
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{test_metrics.total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{test_metrics.success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{test_metrics.avg_execution_time_ms:.0f}ms</div>
                <div class="metric-label">Avg Execution Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{test_metrics.total_assertions}</div>
                <div class="metric-label">Total Assertions</div>
            </div>
        </div>
        """

    def _generate_results_table_html(self, suite_results: List[TestSuiteResult]) -> str:
        """Generate test results table HTML."""
        rows = []

        for suite_result in suite_results:
            for test_result in suite_result.test_results:
                status_class = f"status-{test_result.status.value}"
                execution_time = f"{test_result.execution_time * 1000:.0f}ms" if test_result.execution_time else "N/A"

                rows.append(f"""
                <tr>
                    <td>{suite_result.suite_name}</td>
                    <td>{test_result.test_name}</td>
                    <td class="{status_class}">{test_result.status.value.upper()}</td>
                    <td>{execution_time}</td>
                    <td>{len(test_result.assertion_results)}</td>
                    <td>{test_result.error_message or ''}</td>
                </tr>
                """)

        return f"""
        <table class="results-table">
            <thead>
                <tr>
                    <th>Suite</th>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Assertions</th>
                    <th>Error Message</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_charts_data(self, suite_results: List[TestSuiteResult], test_metrics: TestMetrics) -> str:
        """Generate data for charts."""
        # Results distribution
        results_data = [
            test_metrics.passed_tests,
            test_metrics.failed_tests,
            test_metrics.error_tests,
            test_metrics.skipped_tests
        ]

        # Performance data (execution times by test)
        performance_labels = []
        performance_data = []

        for suite_result in suite_results:
            for test_result in suite_result.test_results:
                if test_result.execution_time:
                    performance_labels.append(test_result.test_name[:20])  # Truncate long names
                    performance_data.append(test_result.execution_time * 1000)

        charts_data = {
            'results': results_data,
            'performance': {
                'labels': performance_labels,
                'data': performance_data
            }
        }

        return json.dumps(charts_data)

    def _generate_coverage_html(self, coverage_report: TestCoverageReport) -> str:
        """Generate coverage report HTML."""
        return f"""
        <div class="coverage-section">
            <h2 class="section-title">Test Coverage Analysis</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value">{coverage_report.table_coverage_percentage:.1f}%</div>
                    <div class="metric-label">Table Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage_report.column_coverage_percentage:.1f}%</div>
                    <div class="metric-label">Column Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage_report.tested_tables}</div>
                    <div class="metric-label">Tables Tested</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(coverage_report.untested_objects)}</div>
                    <div class="metric-label">Untested Objects</div>
                </div>
            </div>
        </div>
        """

    def _generate_json_report(self,
                            suite_results: List[TestSuiteResult],
                            test_metrics: TestMetrics,
                            performance_metrics: Optional[PerformanceMetrics],
                            coverage_report: Optional[TestCoverageReport]) -> Path:
        """Generate JSON report."""

        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'generator': 'SQLTest Pro Enterprise'
            },
            'summary': asdict(test_metrics),
            'performance': asdict(performance_metrics) if performance_metrics else None,
            'coverage': asdict(coverage_report) if coverage_report else None,
            'test_suites': []
        }

        for suite_result in suite_results:
            suite_data = {
                'suite_name': suite_result.suite_name,
                'start_time': suite_result.start_time.isoformat(),
                'end_time': suite_result.end_time.isoformat() if suite_result.end_time else None,
                'execution_time': suite_result.execution_time,
                'tests': []
            }

            for test_result in suite_result.test_results:
                test_data = {
                    'test_name': test_result.test_name,
                    'status': test_result.status.value,
                    'start_time': test_result.start_time.isoformat(),
                    'end_time': test_result.end_time.isoformat() if test_result.end_time else None,
                    'execution_time': test_result.execution_time,
                    'error_message': test_result.error_message,
                    'assertions': test_result.assertion_results,
                    'row_count': test_result.row_count
                }
                suite_data['tests'].append(test_data)

            report_data['test_suites'].append(suite_data)

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"test_report_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report generated: {output_path}")
        return output_path

    def _generate_csv_report(self, suite_results: List[TestSuiteResult], test_metrics: TestMetrics) -> Path:
        """Generate CSV report."""

        # Flatten test results for CSV
        csv_data = []
        for suite_result in suite_results:
            for test_result in suite_result.test_results:
                csv_data.append({
                    'suite_name': suite_result.suite_name,
                    'test_name': test_result.test_name,
                    'status': test_result.status.value,
                    'execution_time_ms': test_result.execution_time * 1000 if test_result.execution_time else None,
                    'start_time': test_result.start_time.isoformat(),
                    'end_time': test_result.end_time.isoformat() if test_result.end_time else None,
                    'assertions_count': len(test_result.assertion_results),
                    'assertions_passed': sum(1 for a in test_result.assertion_results if a.get('passed', False)),
                    'row_count': test_result.row_count,
                    'error_message': test_result.error_message or ''
                })

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"test_report_{timestamp}.csv"

        df.to_csv(output_path, index=False)

        logger.info(f"CSV report generated: {output_path}")
        return output_path

    def _generate_xml_report(self, suite_results: List[TestSuiteResult], test_metrics: TestMetrics) -> Path:
        """Generate XML report (JUnit format)."""

        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_content.append(f'<testsuites tests="{test_metrics.total_tests}" failures="{test_metrics.failed_tests}" errors="{test_metrics.error_tests}" time="{test_metrics.total_execution_time_ms / 1000:.3f}">')

        for suite_result in suite_results:
            suite_time = suite_result.execution_time or 0
            xml_content.append(f'  <testsuite name="{suite_result.suite_name}" tests="{len(suite_result.test_results)}" time="{suite_time:.3f}">')

            for test_result in suite_result.test_results:
                test_time = test_result.execution_time or 0
                xml_content.append(f'    <testcase name="{test_result.test_name}" time="{test_time:.3f}">')

                if test_result.status == TestStatus.FAILED:
                    xml_content.append(f'      <failure message="{test_result.error_message or "Test failed"}">')
                    xml_content.append(f'        {test_result.error_message or "No details available"}')
                    xml_content.append('      </failure>')
                elif test_result.status == TestStatus.ERROR:
                    xml_content.append(f'      <error message="{test_result.error_message or "Test error"}">')
                    xml_content.append(f'        {test_result.error_message or "No details available"}')
                    xml_content.append('      </error>')
                elif test_result.status == TestStatus.SKIPPED:
                    xml_content.append('      <skipped/>')

                xml_content.append('    </testcase>')

            xml_content.append('  </testsuite>')

        xml_content.append('</testsuites>')

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"test_report_{timestamp}.xml"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(xml_content))

        logger.info(f"XML report generated: {output_path}")
        return output_path

    def _store_execution_history(self,
                               suite_results: List[TestSuiteResult],
                               test_metrics: TestMetrics,
                               performance_metrics: Optional[PerformanceMetrics]):
        """Store execution history for trend analysis."""
        with self._lock:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'test_metrics': asdict(test_metrics),
                'performance_metrics': asdict(performance_metrics) if performance_metrics else None,
                'suite_count': len(suite_results)
            }

            self._execution_history.append(history_entry)

            # Keep only last 100 entries
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]

    def generate_trend_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis report."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with self._lock:
            recent_history = [
                entry for entry in self._execution_history
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
            ]

        if not recent_history:
            return {'message': 'No historical data available for trend analysis'}

        # Calculate trends
        success_rates = [entry['test_metrics']['success_rate'] for entry in recent_history]
        execution_times = [entry['test_metrics']['avg_execution_time_ms'] for entry in recent_history]

        return {
            'period_days': days,
            'total_executions': len(recent_history),
            'avg_success_rate': sum(success_rates) / len(success_rates),
            'success_rate_trend': 'improving' if success_rates[-1] > success_rates[0] else 'declining',
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'performance_trend': 'improving' if execution_times[-1] < execution_times[0] else 'declining',
            'history': recent_history
        }