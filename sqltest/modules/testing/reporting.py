"""Test coverage tracking, reporting, and CI/CD integration.

Provides:
- Query and table coverage tracking
- HTML and JUnit XML report generation
- CI/CD integration (GitHub Actions, GitLab CI, Jenkins)
- Test history and trend analysis
"""

import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Test coverage metrics."""
    total_tables: int = 0
    tested_tables: int = 0
    total_queries: int = 0
    tested_queries: int = 0
    table_coverage_percent: float = 0.0
    query_coverage_percent: float = 0.0
    untested_tables: Set[str] = field(default_factory=set)
    test_count_by_table: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class CoverageTracker:
    """Track test coverage for queries and tables."""

    def __init__(self, schema_introspector):
        self.schema_introspector = schema_introspector
        self._tested_tables: Set[str] = set()
        self._tested_queries: Set[str] = set()
        self._test_count_by_table: Dict[str, int] = defaultdict(int)

    def record_test(self, query: str, tables_accessed: List[str]):
        """Record test execution for coverage."""
        # Normalize query for tracking
        query_hash = hash(query)
        self._tested_queries.add(str(query_hash))

        # Track table coverage
        for table in tables_accessed:
            self._tested_tables.add(table)
            self._test_count_by_table[table] += 1

    def get_coverage_metrics(self, database_schema) -> CoverageMetrics:
        """Calculate coverage metrics."""
        all_tables = set(database_schema.get_all_tables().keys())

        metrics = CoverageMetrics(
            total_tables=len(all_tables),
            tested_tables=len(self._tested_tables),
            total_queries=len(self._tested_queries),  # Simplified
            tested_queries=len(self._tested_queries),
            untested_tables=all_tables - self._tested_tables,
            test_count_by_table=dict(self._test_count_by_table)
        )

        if metrics.total_tables > 0:
            metrics.table_coverage_percent = (metrics.tested_tables / metrics.total_tables) * 100

        return metrics


class TestReportGenerator:
    """Generate test reports in multiple formats."""

    def generate_html_report(self, results: Dict[str, Any], output_path: Path) -> str:
        """Generate HTML test report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SQLTest Pro - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 8px; }}
        .passed {{ color: #27ae60; }}
        .failed {{ color: #e74c3c; }}
        .test-list {{ margin-top: 30px; }}
        .test-item {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .test-item.failed {{ border-left-color: #e74c3c; }}
        .timestamp {{ color: #95a5a6; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SQLTest Pro - Test Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{results.get('total_tests', 0)}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value passed">{results.get('passed', 0)}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">{results.get('failed', 0)}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{results.get('pass_rate', 0):.1f}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
        </div>

        <div class="test-list">
            <h2>Test Results</h2>
            <!-- Test items would be listed here -->
        </div>
    </div>
</body>
</html>"""

        output_path.write_text(html)
        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)

    def generate_junit_xml(self, results: Dict[str, Any], output_path: Path) -> str:
        """Generate JUnit XML report for CI/CD integration."""
        testsuites = ET.Element('testsuites')

        testsuite = ET.SubElement(testsuites, 'testsuite',
                                  name='SQLTest Pro',
                                  tests=str(results.get('total_tests', 0)),
                                  failures=str(results.get('failed', 0)),
                                  errors=str(results.get('errors', 0)),
                                  skipped=str(results.get('skipped', 0)),
                                  time=str(results.get('execution_time_seconds', 0)))

        # Add test cases
        for test_result in results.get('test_results', []):
            testcase = ET.SubElement(testsuite, 'testcase',
                                    name=test_result.get('name', ''),
                                    classname='SQLTest',
                                    time=str(test_result.get('execution_time_ms', 0) / 1000))

            if test_result.get('status') == 'failed':
                failure = ET.SubElement(testcase, 'failure',
                                      message=test_result.get('error_message', 'Test failed'))
                failure.text = test_result.get('error_details', '')

            elif test_result.get('status') == 'error':
                error = ET.SubElement(testcase, 'error',
                                    message=test_result.get('error_message', 'Test error'))
                error.text = test_result.get('error_details', '')

            elif test_result.get('status') == 'skipped':
                ET.SubElement(testcase, 'skipped')

        # Write XML
        tree = ET.ElementTree(testsuites)
        tree.write(str(output_path), encoding='utf-8', xml_declaration=True)

        logger.info(f"Generated JUnit XML report: {output_path}")
        return str(output_path)

    def generate_json_report(self, results: Dict[str, Any], output_path: Path) -> str:
        """Generate JSON report."""
        output_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Generated JSON report: {output_path}")
        return str(output_path)


class CICDIntegration:
    """CI/CD platform integration support."""

    @staticmethod
    def detect_ci_platform() -> Optional[str]:
        """Detect current CI/CD platform."""
        import os

        if os.getenv('GITHUB_ACTIONS'):
            return 'github_actions'
        elif os.getenv('GITLAB_CI'):
            return 'gitlab_ci'
        elif os.getenv('JENKINS_URL'):
            return 'jenkins'
        elif os.getenv('CIRCLECI'):
            return 'circleci'
        elif os.getenv('TRAVIS'):
            return 'travis_ci'

        return None

    @staticmethod
    def set_output(name: str, value: str):
        """Set CI output variable."""
        import os

        platform = CICDIntegration.detect_ci_platform()

        if platform == 'github_actions':
            # GitHub Actions
            github_output = os.getenv('GITHUB_OUTPUT')
            if github_output:
                with open(github_output, 'a') as f:
                    f.write(f"{name}={value}\n")

        elif platform == 'gitlab_ci':
            # GitLab CI
            print(f"{name}={value}")

    @staticmethod
    def annotate_error(message: str, file: Optional[str] = None, line: Optional[int] = None):
        """Create CI annotation for error."""
        platform = CICDIntegration.detect_ci_platform()

        if platform == 'github_actions':
            annotation = f"::error"
            if file:
                annotation += f" file={file}"
            if line:
                annotation += f",line={line}"
            annotation += f"::{message}"
            print(annotation)

    @staticmethod
    def create_summary(summary_text: str):
        """Create CI job summary."""
        import os

        platform = CICDIntegration.detect_ci_platform()

        if platform == 'github_actions':
            github_step_summary = os.getenv('GITHUB_STEP_SUMMARY')
            if github_step_summary:
                with open(github_step_summary, 'a') as f:
                    f.write(summary_text)
                    f.write("\n")


class TestHistoryTracker:
    """Track test history for trend analysis."""

    def __init__(self, history_file: Path):
        self.history_file = history_file
        self._history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        """Load historical test data."""
        if self.history_file.exists():
            try:
                self._history = json.loads(self.history_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load test history: {e}")
                self._history = []

    def add_run(self, results: Dict[str, Any]):
        """Add test run to history."""
        self._history.append({
            'timestamp': datetime.now().isoformat(),
            **results
        })

        # Keep last 100 runs
        self._history = self._history[-100:]

        # Save history
        try:
            self.history_file.write_text(json.dumps(self._history, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save test history: {e}")

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze test trends."""
        if not self._history:
            return {}

        recent_runs = self._history[-10:]

        return {
            'total_runs': len(self._history),
            'recent_pass_rate': sum(r.get('pass_rate', 0) for r in recent_runs) / len(recent_runs),
            'trend': 'improving' if len(recent_runs) > 1 and recent_runs[-1].get('pass_rate', 0) > recent_runs[0].get('pass_rate', 0) else 'stable',
            'flaky_tests': self._identify_flaky_tests()
        }

    def _identify_flaky_tests(self) -> List[str]:
        """Identify tests with inconsistent results."""
        test_results = defaultdict(list)

        for run in self._history[-20:]:
            for test_result in run.get('test_results', []):
                test_name = test_result.get('name')
                status = test_result.get('status')
                test_results[test_name].append(status)

        flaky = []
        for test_name, results in test_results.items():
            if len(set(results)) > 1 and len(results) >= 5:
                flaky.append(test_name)

        return flaky