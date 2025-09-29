"""
Performance benchmarking and regression detection for SQL unit testing framework.

This module provides comprehensive performance monitoring, benchmarking,
regression detection, and performance optimization recommendations.
"""
import logging
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import numpy as np

from .models import TestResult, TestStatus

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Types of performance metrics tracked."""
    EXECUTION_TIME = "execution_time"
    QUERY_TIME = "query_time"
    SETUP_TIME = "setup_time"
    TEARDOWN_TIME = "teardown_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    IO_OPERATIONS = "io_operations"
    NETWORK_CALLS = "network_calls"
    ASSERTION_TIME = "assertion_time"
    FIXTURE_LOAD_TIME = "fixture_load_time"


class RegressionSeverity(str, Enum):
    """Severity levels for performance regressions."""
    CRITICAL = "critical"     # >50% degradation
    HIGH = "high"            # 25-50% degradation
    MEDIUM = "medium"        # 10-25% degradation
    LOW = "low"             # 5-10% degradation
    NONE = "none"           # <5% degradation


class BenchmarkType(str, Enum):
    """Types of performance benchmarks."""
    BASELINE = "baseline"           # Initial baseline measurement
    REGRESSION = "regression"       # Regression testing benchmark
    LOAD = "load"                  # Load testing benchmark
    STRESS = "stress"              # Stress testing benchmark
    ENDURANCE = "endurance"        # Long-running test benchmark
    COMPARATIVE = "comparative"     # Comparison between versions


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement."""
    measurement_id: str
    test_name: str
    metric_type: PerformanceMetric
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    environment: str = "default"
    version: Optional[str] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for a test."""
    test_name: str
    metric_type: PerformanceMetric
    baseline_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    created_at: datetime
    last_updated: datetime
    environment: str = "default"
    version: Optional[str] = None


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""
    alert_id: str
    test_name: str
    metric_type: PerformanceMetric
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    degradation_percent: float
    detected_at: datetime
    description: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark configuration and results."""
    benchmark_id: str
    benchmark_type: BenchmarkType
    name: str
    description: str
    test_names: List[str]
    target_metrics: List[PerformanceMetric]
    iterations: int
    warmup_iterations: int
    timeout: int
    environment: str
    results: List[PerformanceMeasurement] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Real-time performance monitoring and data collection."""

    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self.measurements: deque = deque(maxlen=10000)  # Recent measurements
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def start_measurement_session(self, session_id: str, context: Dict[str, Any]) -> None:
        """Start a performance measurement session."""
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "context": context,
            "measurements": []
        }

    def record_measurement(self, session_id: str, test_name: str,
                         metric_type: PerformanceMetric, value: float,
                         unit: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance measurement."""
        measurement = PerformanceMeasurement(
            measurement_id=f"{session_id}_{len(self.measurements)}",
            test_name=test_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            context=context or {}
        )

        self.measurements.append(measurement)

        if session_id in self.active_sessions:
            self.active_sessions[session_id]["measurements"].append(measurement)

    def end_measurement_session(self, session_id: str) -> Dict[str, Any]:
        """End a measurement session and return summary."""
        if session_id not in self.active_sessions:
            return {}

        session = self.active_sessions[session_id]
        end_time = datetime.now()
        duration = (end_time - session["start_time"]).total_seconds()

        summary = {
            "session_id": session_id,
            "duration": duration,
            "measurement_count": len(session["measurements"]),
            "metrics_collected": list(set(m.metric_type for m in session["measurements"])),
            "tests_measured": list(set(m.test_name for m in session["measurements"]))
        }

        del self.active_sessions[session_id]
        return summary

    def get_measurements(self, test_name: Optional[str] = None,
                        metric_type: Optional[PerformanceMetric] = None,
                        since: Optional[datetime] = None) -> List[PerformanceMeasurement]:
        """Get measurements with optional filtering."""
        measurements = list(self.measurements)

        if test_name:
            measurements = [m for m in measurements if m.test_name == test_name]

        if metric_type:
            measurements = [m for m in measurements if m.metric_type == metric_type]

        if since:
            measurements = [m for m in measurements if m.timestamp >= since]

        return measurements

    def create_baseline(self, test_name: str, metric_type: PerformanceMetric,
                       sample_size: int = 30, confidence_level: float = 0.95) -> PerformanceBaseline:
        """Create performance baseline from recent measurements."""
        measurements = self.get_measurements(test_name, metric_type)

        if len(measurements) < sample_size:
            raise ValueError(f"Insufficient measurements for baseline. Need {sample_size}, have {len(measurements)}")

        # Use most recent measurements
        recent_measurements = sorted(measurements, key=lambda x: x.timestamp)[-sample_size:]
        values = [m.value for m in recent_measurements]

        baseline_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_error = z_score * (std_dev / (len(values) ** 0.5))
        confidence_interval = (baseline_value - margin_error, baseline_value + margin_error)

        baseline = PerformanceBaseline(
            test_name=test_name,
            metric_type=metric_type,
            baseline_value=baseline_value,
            confidence_interval=confidence_interval,
            sample_size=len(values),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # Store baseline
        key = f"{test_name}:{metric_type.value}"
        self.baselines[key] = baseline

        logger.info(f"Created baseline for {test_name}:{metric_type.value} = {baseline_value:.3f}")
        return baseline

    def update_baseline(self, test_name: str, metric_type: PerformanceMetric,
                       adaptive: bool = True) -> Optional[PerformanceBaseline]:
        """Update existing baseline with new measurements."""
        key = f"{test_name}:{metric_type.value}"
        if key not in self.baselines:
            return None

        baseline = self.baselines[key]

        # Get measurements since last update
        recent_measurements = self.get_measurements(test_name, metric_type, baseline.last_updated)

        if len(recent_measurements) < 10:  # Need minimum measurements to update
            return baseline

        if adaptive:
            # Adaptive baseline that slowly incorporates new data
            new_values = [m.value for m in recent_measurements]
            new_mean = statistics.mean(new_values)

            # Weighted average: 80% old baseline, 20% new data
            updated_value = 0.8 * baseline.baseline_value + 0.2 * new_mean

            baseline.baseline_value = updated_value
            baseline.last_updated = datetime.now()

        return baseline


class RegressionDetector:
    """Detects performance regressions using statistical analysis."""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.regression_thresholds = {
            RegressionSeverity.CRITICAL: 0.5,   # 50% degradation
            RegressionSeverity.HIGH: 0.25,      # 25% degradation
            RegressionSeverity.MEDIUM: 0.10,    # 10% degradation
            RegressionSeverity.LOW: 0.05        # 5% degradation
        }

    def detect_regressions(self, test_results: List[TestResult]) -> List[RegressionAlert]:
        """Detect performance regressions in test results."""
        alerts = []

        for result in test_results:
            if result.execution_time is None:
                continue

            # Check execution time regression
            execution_alert = self._check_metric_regression(
                result.test_name,
                PerformanceMetric.EXECUTION_TIME,
                result.execution_time
            )

            if execution_alert:
                alerts.append(execution_alert)

            # Check assertion time regression if available
            if hasattr(result, 'assertion_time') and result.assertion_time:
                assertion_alert = self._check_metric_regression(
                    result.test_name,
                    PerformanceMetric.ASSERTION_TIME,
                    result.assertion_time
                )

                if assertion_alert:
                    alerts.append(assertion_alert)

        return alerts

    def _check_metric_regression(self, test_name: str, metric_type: PerformanceMetric,
                               current_value: float) -> Optional[RegressionAlert]:
        """Check for regression in a specific metric."""
        key = f"{test_name}:{metric_type.value}"
        baseline = self.performance_monitor.baselines.get(key)

        if not baseline:
            return None

        # Calculate degradation percentage
        if baseline.baseline_value == 0:
            return None

        degradation = (current_value - baseline.baseline_value) / baseline.baseline_value

        # Determine severity
        severity = RegressionSeverity.NONE
        for sev, threshold in self.regression_thresholds.items():
            if degradation >= threshold:
                severity = sev
                break

        if severity == RegressionSeverity.NONE:
            return None

        # Generate recommendations
        recommendations = self._generate_regression_recommendations(
            test_name, metric_type, severity, degradation
        )

        alert = RegressionAlert(
            alert_id=f"regression_{test_name}_{metric_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_name=test_name,
            metric_type=metric_type,
            severity=severity,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            degradation_percent=degradation * 100,
            detected_at=datetime.now(),
            description=f"{severity.value.upper()} regression detected in {test_name} ({metric_type.value}): "
                       f"{degradation*100:.1f}% slower than baseline",
            recommendations=recommendations
        )

        return alert

    def _generate_regression_recommendations(self, test_name: str, metric_type: PerformanceMetric,
                                          severity: RegressionSeverity, degradation: float) -> List[str]:
        """Generate recommendations for addressing regression."""
        recommendations = []

        if metric_type == PerformanceMetric.EXECUTION_TIME:
            recommendations.extend([
                "Profile test execution to identify bottlenecks",
                "Review recent changes to test logic or dependencies",
                "Check database query performance and optimization",
                "Verify test environment resources and capacity"
            ])

            if severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH]:
                recommendations.extend([
                    "Consider breaking down complex test into smaller units",
                    "Review fixture data size and generation efficiency",
                    "Implement parallel execution where possible"
                ])

        elif metric_type == PerformanceMetric.QUERY_TIME:
            recommendations.extend([
                "Analyze SQL query execution plans",
                "Check for missing or outdated database indexes",
                "Review query complexity and join strategies",
                "Monitor database server performance"
            ])

        elif metric_type == PerformanceMetric.MEMORY_USAGE:
            recommendations.extend([
                "Check for memory leaks in test code",
                "Review fixture data size and cleanup",
                "Optimize data structures and algorithms",
                "Consider streaming for large datasets"
            ])

        return recommendations

    def analyze_performance_trends(self, test_name: str, metric_type: PerformanceMetric,
                                 days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        since = datetime.now() - timedelta(days=days)
        measurements = self.performance_monitor.get_measurements(test_name, metric_type, since)

        if len(measurements) < 5:
            return {"error": "Insufficient data for trend analysis"}

        # Sort by timestamp
        measurements.sort(key=lambda x: x.timestamp)
        values = [m.value for m in measurements]
        timestamps = [m.timestamp for m in measurements]

        # Calculate trend using linear regression
        x = np.array(range(len(values)))
        y = np.array(values)

        # Simple linear regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Trend classification
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "degrading"
        else:
            trend = "improving"

        return {
            "test_name": test_name,
            "metric_type": metric_type.value,
            "trend": trend,
            "slope": slope,
            "r_squared": r_squared,
            "data_points": len(values),
            "time_range_days": days,
            "current_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": statistics.mean(values),
            "std_deviation": statistics.stdev(values) if len(values) > 1 else 0
        }


class PerformanceBenchmarkRunner:
    """Runs performance benchmarks and collects detailed metrics."""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.benchmark_results: Dict[str, PerformanceBenchmark] = {}

    async def run_benchmark(self, benchmark: PerformanceBenchmark,
                          test_executor) -> PerformanceBenchmark:
        """Run a performance benchmark."""
        logger.info(f"Starting benchmark: {benchmark.name}")

        session_id = f"benchmark_{benchmark.benchmark_id}"
        self.performance_monitor.start_measurement_session(session_id, {
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_type": benchmark.benchmark_type.value
        })

        try:
            # Warmup iterations
            if benchmark.warmup_iterations > 0:
                logger.info(f"Running {benchmark.warmup_iterations} warmup iterations")
                await self._run_iterations(
                    benchmark, test_executor, benchmark.warmup_iterations, warmup=True
                )

            # Actual benchmark iterations
            logger.info(f"Running {benchmark.iterations} benchmark iterations")
            await self._run_iterations(
                benchmark, test_executor, benchmark.iterations, warmup=False
            )

            # Generate summary
            benchmark.summary = self._generate_benchmark_summary(benchmark)

        finally:
            self.performance_monitor.end_measurement_session(session_id)

        # Store results
        self.benchmark_results[benchmark.benchmark_id] = benchmark

        logger.info(f"Benchmark completed: {benchmark.name}")
        return benchmark

    async def _run_iterations(self, benchmark: PerformanceBenchmark,
                            test_executor, iterations: int, warmup: bool = False) -> None:
        """Run benchmark iterations."""
        for iteration in range(iterations):
            logger.debug(f"{'Warmup' if warmup else 'Benchmark'} iteration {iteration + 1}/{iterations}")

            for test_name in benchmark.test_names:
                try:
                    # Load test case (this would integrate with your test loader)
                    test_case = self._load_test_case(test_name)

                    # Measure execution
                    start_time = datetime.now()
                    result = await test_executor.execute_test(test_case)
                    end_time = datetime.now()

                    execution_time = (end_time - start_time).total_seconds()

                    # Record measurements (only for actual benchmark, not warmup)
                    if not warmup:
                        for metric_type in benchmark.target_metrics:
                            value = self._extract_metric_value(result, metric_type)
                            if value is not None:
                                measurement = PerformanceMeasurement(
                                    measurement_id=f"{benchmark.benchmark_id}_{iteration}_{test_name}_{metric_type.value}",
                                    test_name=test_name,
                                    metric_type=metric_type,
                                    value=value,
                                    unit=self._get_metric_unit(metric_type),
                                    timestamp=datetime.now(),
                                    context={
                                        "benchmark_id": benchmark.benchmark_id,
                                        "iteration": iteration,
                                        "warmup": warmup
                                    }
                                )
                                benchmark.results.append(measurement)

                except Exception as e:
                    logger.error(f"Error in benchmark iteration {iteration}, test {test_name}: {e}")

    def _load_test_case(self, test_name: str):
        """Load test case by name (placeholder implementation)."""
        # This would integrate with your test case loader
        from .models import TestCase
        return TestCase(
            name=test_name,
            description=f"Benchmark test: {test_name}",
            sql="SELECT 1",
            assertions=[],
            fixtures=[]
        )

    def _extract_metric_value(self, result, metric_type: PerformanceMetric) -> Optional[float]:
        """Extract metric value from test result."""
        if metric_type == PerformanceMetric.EXECUTION_TIME:
            return getattr(result, 'execution_time', None)
        elif metric_type == PerformanceMetric.QUERY_TIME:
            return getattr(result, 'query_time', None)
        elif metric_type == PerformanceMetric.MEMORY_USAGE:
            return getattr(result, 'memory_usage', None)
        elif metric_type == PerformanceMetric.CPU_USAGE:
            return getattr(result, 'cpu_usage', None)
        else:
            return None

    def _get_metric_unit(self, metric_type: PerformanceMetric) -> str:
        """Get unit for metric type."""
        unit_map = {
            PerformanceMetric.EXECUTION_TIME: "seconds",
            PerformanceMetric.QUERY_TIME: "seconds",
            PerformanceMetric.SETUP_TIME: "seconds",
            PerformanceMetric.TEARDOWN_TIME: "seconds",
            PerformanceMetric.MEMORY_USAGE: "MB",
            PerformanceMetric.CPU_USAGE: "percent",
            PerformanceMetric.IO_OPERATIONS: "count",
            PerformanceMetric.NETWORK_CALLS: "count",
            PerformanceMetric.ASSERTION_TIME: "seconds",
            PerformanceMetric.FIXTURE_LOAD_TIME: "seconds"
        }
        return unit_map.get(metric_type, "units")

    def _generate_benchmark_summary(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Generate summary statistics for benchmark."""
        summary = {
            "total_measurements": len(benchmark.results),
            "tests_executed": len(set(m.test_name for m in benchmark.results)),
            "metrics_collected": len(set(m.metric_type for m in benchmark.results))
        }

        # Group by test and metric
        grouped_results = defaultdict(lambda: defaultdict(list))
        for measurement in benchmark.results:
            grouped_results[measurement.test_name][measurement.metric_type].append(measurement.value)

        # Calculate statistics for each test/metric combination
        test_summaries = {}
        for test_name, metrics in grouped_results.items():
            test_summary = {}
            for metric_type, values in metrics.items():
                if values:
                    test_summary[metric_type.value] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "percentile_95": np.percentile(values, 95),
                        "percentile_99": np.percentile(values, 99)
                    }
            test_summaries[test_name] = test_summary

        summary["test_summaries"] = test_summaries
        return summary

    def create_load_benchmark(self, name: str, test_names: List[str],
                            concurrent_users: int = 10, duration_minutes: int = 5) -> PerformanceBenchmark:
        """Create a load testing benchmark."""
        return PerformanceBenchmark(
            benchmark_id=f"load_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            benchmark_type=BenchmarkType.LOAD,
            name=name,
            description=f"Load test with {concurrent_users} concurrent users for {duration_minutes} minutes",
            test_names=test_names,
            target_metrics=[
                PerformanceMetric.EXECUTION_TIME,
                PerformanceMetric.QUERY_TIME,
                PerformanceMetric.CPU_USAGE,
                PerformanceMetric.MEMORY_USAGE
            ],
            iterations=concurrent_users * duration_minutes * 10,  # Approximate iterations
            warmup_iterations=10,
            timeout=300,
            environment="load_test"
        )

    def create_regression_benchmark(self, name: str, test_names: List[str]) -> PerformanceBenchmark:
        """Create a regression testing benchmark."""
        return PerformanceBenchmark(
            benchmark_id=f"regression_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            benchmark_type=BenchmarkType.REGRESSION,
            name=name,
            description="Regression benchmark for performance validation",
            test_names=test_names,
            target_metrics=[
                PerformanceMetric.EXECUTION_TIME,
                PerformanceMetric.ASSERTION_TIME
            ],
            iterations=50,
            warmup_iterations=5,
            timeout=60,
            environment="regression_test"
        )

    def export_benchmark_results(self, benchmark_id: str, format: str = "json") -> str:
        """Export benchmark results in specified format."""
        if benchmark_id not in self.benchmark_results:
            raise ValueError(f"Benchmark not found: {benchmark_id}")

        benchmark = self.benchmark_results[benchmark_id]

        if format.lower() == "json":
            return json.dumps({
                "benchmark_id": benchmark.benchmark_id,
                "benchmark_type": benchmark.benchmark_type.value,
                "name": benchmark.name,
                "description": benchmark.description,
                "iterations": benchmark.iterations,
                "warmup_iterations": benchmark.warmup_iterations,
                "created_at": benchmark.created_at.isoformat(),
                "summary": benchmark.summary,
                "results": [
                    {
                        "measurement_id": m.measurement_id,
                        "test_name": m.test_name,
                        "metric_type": m.metric_type.value,
                        "value": m.value,
                        "unit": m.unit,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in benchmark.results
                ]
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Convenience functions
def create_performance_monitor(retention_days: int = 90) -> PerformanceMonitor:
    """Create a performance monitor with specified retention."""
    return PerformanceMonitor(retention_days)


def create_regression_detector(monitor: PerformanceMonitor) -> RegressionDetector:
    """Create a regression detector with performance monitor."""
    return RegressionDetector(monitor)


def create_benchmark_runner(monitor: PerformanceMonitor) -> PerformanceBenchmarkRunner:
    """Create a benchmark runner with performance monitor."""
    return PerformanceBenchmarkRunner(monitor)