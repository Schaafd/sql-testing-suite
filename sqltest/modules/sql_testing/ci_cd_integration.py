"""
CI/CD integration and automation capabilities for SQL unit testing framework.

This module provides comprehensive CI/CD pipeline integration including
GitHub Actions, GitLab CI, Jenkins, and Azure DevOps support with
automated test execution, reporting, and quality gates.
"""
import logging
import json
import yaml
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CIPlatform(str, Enum):
    """Supported CI/CD platforms."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    BAMBOO = "bamboo"
    TEAMCITY = "teamcity"


class QualityGateStatus(str, Enum):
    """Status of quality gate checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ArtifactType(str, Enum):
    """Types of CI/CD artifacts."""
    TEST_RESULTS = "test_results"
    COVERAGE_REPORT = "coverage_report"
    PERFORMANCE_REPORT = "performance_report"
    REGRESSION_REPORT = "regression_report"
    LOG_FILES = "log_files"
    CONFIGURATION = "configuration"


@dataclass
class QualityGate:
    """Quality gate configuration and thresholds."""
    name: str
    description: str
    enabled: bool = True
    failure_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None
    metric_type: str = "pass_rate"
    operator: str = "gte"  # gte, lte, eq, ne
    blocking: bool = True  # If true, pipeline fails when gate fails


@dataclass
class CIConfiguration:
    """CI/CD configuration for test execution."""
    platform: CIPlatform
    project_name: str
    branch_patterns: List[str] = field(default_factory=lambda: ["main", "master", "develop"])
    trigger_events: List[str] = field(default_factory=lambda: ["push", "pull_request"])
    test_suites: List[str] = field(default_factory=list)
    quality_gates: List[QualityGate] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    artifacts_to_collect: List[ArtifactType] = field(default_factory=list)
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_minutes: int = 30


@dataclass
class CIArtifact:
    """CI/CD artifact information."""
    artifact_id: str
    artifact_type: ArtifactType
    file_path: str
    size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CIExecutionResult:
    """Result of CI/CD test execution."""
    execution_id: str
    platform: CIPlatform
    build_number: Optional[str]
    commit_hash: Optional[str]
    branch: Optional[str]
    status: QualityGateStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    test_results: Dict[str, Any] = field(default_factory=dict)
    quality_gate_results: Dict[str, QualityGateStatus] = field(default_factory=dict)
    artifacts: List[CIArtifact] = field(default_factory=list)
    error_message: Optional[str] = None


class BaseCIProvider(ABC):
    """Base class for CI/CD platform providers."""

    def __init__(self, configuration: CIConfiguration):
        self.configuration = configuration

    @abstractmethod
    def generate_pipeline_config(self) -> str:
        """Generate platform-specific pipeline configuration."""
        pass

    @abstractmethod
    def parse_environment_info(self) -> Dict[str, Any]:
        """Parse CI environment information."""
        pass

    def validate_configuration(self) -> List[str]:
        """Validate CI configuration."""
        errors = []

        if not self.configuration.test_suites:
            errors.append("No test suites specified")

        if self.configuration.timeout_minutes <= 0:
            errors.append("Timeout must be positive")

        if self.configuration.max_workers <= 0:
            errors.append("Max workers must be positive")

        return errors


class GitHubActionsProvider(BaseCIProvider):
    """GitHub Actions CI/CD provider."""

    def generate_pipeline_config(self) -> str:
        """Generate GitHub Actions workflow configuration."""
        workflow = {
            "name": f"SQL Test Suite - {self.configuration.project_name}",
            "on": {
                "push": {
                    "branches": self.configuration.branch_patterns
                },
                "pull_request": {
                    "branches": self.configuration.branch_patterns
                }
            },
            "jobs": {
                "sql-tests": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": self.configuration.timeout_minutes,
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.9"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run SQL Tests",
                            "run": self._generate_test_command(),
                            "env": self.configuration.environment_variables
                        }
                    ]
                }
            }
        }

        # Add artifact collection steps
        if self.configuration.artifacts_to_collect:
            workflow["jobs"]["sql-tests"]["steps"].extend(
                self._generate_artifact_steps()
            )

        return yaml.dump(workflow, default_flow_style=False)

    def parse_environment_info(self) -> Dict[str, Any]:
        """Parse GitHub Actions environment information."""
        return {
            "platform": "github_actions",
            "build_number": os.getenv("GITHUB_RUN_NUMBER"),
            "commit_hash": os.getenv("GITHUB_SHA"),
            "branch": os.getenv("GITHUB_REF_NAME"),
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "actor": os.getenv("GITHUB_ACTOR")
        }

    def _generate_test_command(self) -> str:
        """Generate test execution command."""
        base_cmd = "sqltest ci"

        if self.configuration.test_suites:
            suites = " ".join(self.configuration.test_suites)
            base_cmd += f" --suites {suites}"

        if self.configuration.parallel_execution:
            base_cmd += f" --parallel --max-workers {self.configuration.max_workers}"

        base_cmd += " --output-format junit --output-file test-results.xml"

        return base_cmd

    def _generate_artifact_steps(self) -> List[Dict[str, Any]]:
        """Generate artifact collection steps."""
        steps = []

        if ArtifactType.TEST_RESULTS in self.configuration.artifacts_to_collect:
            steps.append({
                "name": "Upload test results",
                "uses": "actions/upload-artifact@v3",
                "if": "always()",
                "with": {
                    "name": "test-results",
                    "path": "test-results.xml"
                }
            })

        if ArtifactType.COVERAGE_REPORT in self.configuration.artifacts_to_collect:
            steps.append({
                "name": "Upload coverage report",
                "uses": "actions/upload-artifact@v3",
                "if": "always()",
                "with": {
                    "name": "coverage-report",
                    "path": "coverage-report.html"
                }
            })

        return steps


class GitLabCIProvider(BaseCIProvider):
    """GitLab CI/CD provider."""

    def generate_pipeline_config(self) -> str:
        """Generate GitLab CI pipeline configuration."""
        pipeline = {
            "stages": ["test", "report"],
            "variables": self.configuration.environment_variables,
            "sql-tests": {
                "stage": "test",
                "image": "python:3.9",
                "timeout": f"{self.configuration.timeout_minutes}m",
                "before_script": [
                    "pip install -r requirements.txt"
                ],
                "script": [
                    self._generate_test_command()
                ],
                "artifacts": {
                    "when": "always",
                    "reports": {
                        "junit": "test-results.xml"
                    },
                    "paths": self._generate_artifact_paths()
                },
                "only": {
                    "refs": self.configuration.branch_patterns
                }
            }
        }

        return yaml.dump(pipeline, default_flow_style=False)

    def parse_environment_info(self) -> Dict[str, Any]:
        """Parse GitLab CI environment information."""
        return {
            "platform": "gitlab_ci",
            "build_number": os.getenv("CI_PIPELINE_ID"),
            "commit_hash": os.getenv("CI_COMMIT_SHA"),
            "branch": os.getenv("CI_COMMIT_REF_NAME"),
            "repository": os.getenv("CI_PROJECT_PATH"),
            "pipeline_url": os.getenv("CI_PIPELINE_URL"),
            "runner_id": os.getenv("CI_RUNNER_ID")
        }

    def _generate_test_command(self) -> str:
        """Generate test execution command."""
        # Similar to GitHub Actions but with GitLab-specific adjustments
        return GitHubActionsProvider._generate_test_command(self)

    def _generate_artifact_paths(self) -> List[str]:
        """Generate artifact paths for GitLab CI."""
        paths = []

        if ArtifactType.TEST_RESULTS in self.configuration.artifacts_to_collect:
            paths.append("test-results.xml")

        if ArtifactType.COVERAGE_REPORT in self.configuration.artifacts_to_collect:
            paths.append("coverage-report.html")

        if ArtifactType.LOG_FILES in self.configuration.artifacts_to_collect:
            paths.append("logs/")

        return paths


class JenkinsProvider(BaseCIProvider):
    """Jenkins CI/CD provider."""

    def generate_pipeline_config(self) -> str:
        """Generate Jenkins pipeline configuration."""
        pipeline = f"""
pipeline {{
    agent any

    options {{
        timeout(time: {self.configuration.timeout_minutes}, unit: 'MINUTES')
    }}

    triggers {{
        pollSCM('H/5 * * * *')
    }}

    environment {{
{self._format_environment_variables()}
    }}

    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}

        stage('Setup') {{
            steps {{
                sh 'pip install -r requirements.txt'
            }}
        }}

        stage('SQL Tests') {{
            steps {{
                sh '{self._generate_test_command()}'
            }}
            post {{
                always {{
                    junit 'test-results.xml'
{self._generate_artifact_publishers()}
                }}
            }}
        }}
    }}

    post {{
        always {{
            cleanWs()
        }}
    }}
}}
"""
        return pipeline

    def parse_environment_info(self) -> Dict[str, Any]:
        """Parse Jenkins environment information."""
        return {
            "platform": "jenkins",
            "build_number": os.getenv("BUILD_NUMBER"),
            "commit_hash": os.getenv("GIT_COMMIT"),
            "branch": os.getenv("GIT_BRANCH"),
            "job_name": os.getenv("JOB_NAME"),
            "build_url": os.getenv("BUILD_URL"),
            "workspace": os.getenv("WORKSPACE")
        }

    def _format_environment_variables(self) -> str:
        """Format environment variables for Jenkins."""
        lines = []
        for key, value in self.configuration.environment_variables.items():
            lines.append(f"        {key} = '{value}'")
        return "\n".join(lines)

    def _generate_test_command(self) -> str:
        """Generate test execution command."""
        return GitHubActionsProvider._generate_test_command(self)

    def _generate_artifact_publishers(self) -> str:
        """Generate artifact publisher configuration."""
        publishers = []

        if ArtifactType.COVERAGE_REPORT in self.configuration.artifacts_to_collect:
            publishers.append("                    publishHTML([")
            publishers.append("                        allowMissing: false,")
            publishers.append("                        alwaysLinkToLastBuild: true,")
            publishers.append("                        keepAll: true,")
            publishers.append("                        reportDir: '.',")
            publishers.append("                        reportFiles: 'coverage-report.html',")
            publishers.append("                        reportName: 'Coverage Report'")
            publishers.append("                    ])")

        return "\n".join(publishers)


class QualityGateEvaluator:
    """Evaluates quality gates for CI/CD pipelines."""

    def __init__(self):
        self.default_gates = [
            QualityGate(
                name="Test Pass Rate",
                description="Minimum test pass rate threshold",
                failure_threshold=0.95,
                warning_threshold=0.98,
                metric_type="pass_rate",
                operator="gte"
            ),
            QualityGate(
                name="Performance Regression",
                description="Maximum performance regression threshold",
                failure_threshold=0.25,
                warning_threshold=0.10,
                metric_type="performance_regression",
                operator="lte"
            ),
            QualityGate(
                name="Test Coverage",
                description="Minimum test coverage threshold",
                failure_threshold=0.80,
                warning_threshold=0.85,
                metric_type="test_coverage",
                operator="gte"
            )
        ]

    def evaluate_gates(self, quality_gates: List[QualityGate],
                      test_results: Dict[str, Any]) -> Dict[str, QualityGateStatus]:
        """Evaluate all quality gates against test results."""
        results = {}

        for gate in quality_gates:
            if not gate.enabled:
                results[gate.name] = QualityGateStatus.SKIPPED
                continue

            metric_value = self._extract_metric_value(test_results, gate.metric_type)

            if metric_value is None:
                logger.warning(f"Metric {gate.metric_type} not found in test results")
                results[gate.name] = QualityGateStatus.SKIPPED
                continue

            status = self._evaluate_single_gate(gate, metric_value)
            results[gate.name] = status

        return results

    def _extract_metric_value(self, test_results: Dict[str, Any], metric_type: str) -> Optional[float]:
        """Extract metric value from test results."""
        if metric_type == "pass_rate":
            total_tests = test_results.get("total_tests", 0)
            passed_tests = test_results.get("passed_tests", 0)
            return passed_tests / total_tests if total_tests > 0 else 0

        elif metric_type == "performance_regression":
            return test_results.get("max_performance_regression", 0)

        elif metric_type == "test_coverage":
            return test_results.get("test_coverage", 0)

        elif metric_type == "failure_rate":
            total_tests = test_results.get("total_tests", 0)
            failed_tests = test_results.get("failed_tests", 0)
            return failed_tests / total_tests if total_tests > 0 else 0

        else:
            return test_results.get(metric_type)

    def _evaluate_single_gate(self, gate: QualityGate, metric_value: float) -> QualityGateStatus:
        """Evaluate a single quality gate."""
        if gate.failure_threshold is not None:
            if not self._compare_value(metric_value, gate.failure_threshold, gate.operator):
                return QualityGateStatus.FAILED

        if gate.warning_threshold is not None:
            if not self._compare_value(metric_value, gate.warning_threshold, gate.operator):
                return QualityGateStatus.WARNING

        return QualityGateStatus.PASSED

    def _compare_value(self, actual: float, threshold: float, operator: str) -> bool:
        """Compare values using specified operator."""
        if operator == "gte":
            return actual >= threshold
        elif operator == "lte":
            return actual <= threshold
        elif operator == "eq":
            return actual == threshold
        elif operator == "ne":
            return actual != threshold
        elif operator == "gt":
            return actual > threshold
        elif operator == "lt":
            return actual < threshold
        else:
            return True


class CIManager:
    """Manages CI/CD integration and execution."""

    def __init__(self):
        self.providers = {
            CIPlatform.GITHUB_ACTIONS: GitHubActionsProvider,
            CIPlatform.GITLAB_CI: GitLabCIProvider,
            CIPlatform.JENKINS: JenkinsProvider
        }
        self.quality_gate_evaluator = QualityGateEvaluator()

    def generate_pipeline_config(self, platform: CIPlatform,
                                configuration: CIConfiguration) -> str:
        """Generate platform-specific pipeline configuration."""
        provider_class = self.providers.get(platform)
        if not provider_class:
            raise ValueError(f"Unsupported platform: {platform}")

        provider = provider_class(configuration)
        return provider.generate_pipeline_config()

    def detect_ci_environment(self) -> Optional[Dict[str, Any]]:
        """Detect current CI environment and return information."""
        # Check for GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            provider = GitHubActionsProvider(CIConfiguration(CIPlatform.GITHUB_ACTIONS, "unknown"))
            return provider.parse_environment_info()

        # Check for GitLab CI
        elif os.getenv("GITLAB_CI"):
            provider = GitLabCIProvider(CIConfiguration(CIPlatform.GITLAB_CI, "unknown"))
            return provider.parse_environment_info()

        # Check for Jenkins
        elif os.getenv("JENKINS_URL"):
            provider = JenkinsProvider(CIConfiguration(CIPlatform.JENKINS, "unknown"))
            return provider.parse_environment_info()

        return None

    def execute_ci_tests(self, configuration: CIConfiguration,
                        test_executor) -> CIExecutionResult:
        """Execute tests in CI environment."""
        execution_id = f"ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        # Detect CI environment
        ci_info = self.detect_ci_environment()
        platform = CIPlatform(ci_info["platform"]) if ci_info else CIPlatform.GITHUB_ACTIONS

        result = CIExecutionResult(
            execution_id=execution_id,
            platform=platform,
            build_number=ci_info.get("build_number") if ci_info else None,
            commit_hash=ci_info.get("commit_hash") if ci_info else None,
            branch=ci_info.get("branch") if ci_info else None,
            status=QualityGateStatus.PASSED,
            start_time=start_time
        )

        try:
            # Execute test suites
            all_test_results = []

            for suite_name in configuration.test_suites:
                logger.info(f"Executing test suite: {suite_name}")

                # Load and execute test suite
                # This would integrate with your test suite loader
                suite_results = self._execute_test_suite(suite_name, test_executor)
                all_test_results.extend(suite_results)

            # Aggregate results
            result.test_results = self._aggregate_test_results(all_test_results)

            # Evaluate quality gates
            gate_results = self.quality_gate_evaluator.evaluate_gates(
                configuration.quality_gates, result.test_results
            )
            result.quality_gate_results = gate_results

            # Determine overall status
            if any(status == QualityGateStatus.FAILED for status in gate_results.values()):
                result.status = QualityGateStatus.FAILED
            elif any(status == QualityGateStatus.WARNING for status in gate_results.values()):
                result.status = QualityGateStatus.WARNING

            # Collect artifacts
            result.artifacts = self._collect_artifacts(configuration.artifacts_to_collect)

        except Exception as e:
            result.status = QualityGateStatus.FAILED
            result.error_message = str(e)
            logger.error(f"CI execution failed: {e}")

        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    def _execute_test_suite(self, suite_name: str, test_executor) -> List[Any]:
        """Execute a test suite and return results."""
        # This would integrate with your test suite execution logic
        # Placeholder implementation
        return []

    def _aggregate_test_results(self, test_results: List[Any]) -> Dict[str, Any]:
        """Aggregate individual test results into summary."""
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if getattr(r, 'status', None) == 'passed'])
        failed_tests = total_tests - passed_tests

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "failure_rate": failed_tests / total_tests if total_tests > 0 else 0
        }

    def _collect_artifacts(self, artifact_types: List[ArtifactType]) -> List[CIArtifact]:
        """Collect specified artifacts."""
        artifacts = []

        for artifact_type in artifact_types:
            artifact_path = self._get_artifact_path(artifact_type)

            if artifact_path and Path(artifact_path).exists():
                file_path = Path(artifact_path)
                artifact = CIArtifact(
                    artifact_id=f"{artifact_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    artifact_type=artifact_type,
                    file_path=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    created_at=datetime.now()
                )
                artifacts.append(artifact)

        return artifacts

    def _get_artifact_path(self, artifact_type: ArtifactType) -> Optional[str]:
        """Get file path for artifact type."""
        artifact_paths = {
            ArtifactType.TEST_RESULTS: "test-results.xml",
            ArtifactType.COVERAGE_REPORT: "coverage-report.html",
            ArtifactType.PERFORMANCE_REPORT: "performance-report.json",
            ArtifactType.REGRESSION_REPORT: "regression-report.json",
            ArtifactType.LOG_FILES: "logs/",
            ArtifactType.CONFIGURATION: "config/"
        }
        return artifact_paths.get(artifact_type)

    def create_notification_payload(self, result: CIExecutionResult) -> Dict[str, Any]:
        """Create notification payload for CI result."""
        return {
            "execution_id": result.execution_id,
            "platform": result.platform.value,
            "status": result.status.value,
            "build_number": result.build_number,
            "commit_hash": result.commit_hash,
            "branch": result.branch,
            "duration_seconds": result.duration_seconds,
            "test_summary": result.test_results,
            "quality_gates": result.quality_gate_results,
            "artifacts_count": len(result.artifacts),
            "timestamp": result.start_time.isoformat()
        }


# Convenience functions
def create_github_actions_config(project_name: str, test_suites: List[str]) -> str:
    """Create GitHub Actions configuration for SQL testing."""
    config = CIConfiguration(
        platform=CIPlatform.GITHUB_ACTIONS,
        project_name=project_name,
        test_suites=test_suites,
        artifacts_to_collect=[
            ArtifactType.TEST_RESULTS,
            ArtifactType.COVERAGE_REPORT
        ]
    )

    manager = CIManager()
    return manager.generate_pipeline_config(CIPlatform.GITHUB_ACTIONS, config)


def create_gitlab_ci_config(project_name: str, test_suites: List[str]) -> str:
    """Create GitLab CI configuration for SQL testing."""
    config = CIConfiguration(
        platform=CIPlatform.GITLAB_CI,
        project_name=project_name,
        test_suites=test_suites,
        artifacts_to_collect=[
            ArtifactType.TEST_RESULTS,
            ArtifactType.COVERAGE_REPORT
        ]
    )

    manager = CIManager()
    return manager.generate_pipeline_config(CIPlatform.GITLAB_CI, config)


def create_jenkins_pipeline(project_name: str, test_suites: List[str]) -> str:
    """Create Jenkins pipeline configuration for SQL testing."""
    config = CIConfiguration(
        platform=CIPlatform.JENKINS,
        project_name=project_name,
        test_suites=test_suites,
        artifacts_to_collect=[
            ArtifactType.TEST_RESULTS,
            ArtifactType.COVERAGE_REPORT
        ]
    )

    manager = CIManager()
    return manager.generate_pipeline_config(CIPlatform.JENKINS, config)