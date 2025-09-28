# SQLTest Pro Core Component Specifications

## Overview

This document provides detailed technical specifications for implementing the missing core components of SQLTest Pro, designed for enterprise-scale performance and reliability.

## 1. Business Rules Engine (Priority: CRITICAL)

### Current State: 0% Implementation
**Files to implement:**
- `sqltest/modules/business_rules/engine.py` (exists but empty)
- `sqltest/modules/business_rules/models.py` (partial)
- `sqltest/modules/business_rules/config_loader.py` (empty)

### Technical Specification

#### 1.1 Rule Execution Engine
```python
# sqltest/modules/business_rules/engine.py
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import asyncio
import time
import hashlib

@dataclass
class RuleExecutionContext:
    """Context for rule execution with shared state and metrics."""
    rule_id: str
    database_name: str
    start_time: float
    metadata: Dict[str, Any]
    dependencies: Set[str]
    priority: int = 5  # 1=highest, 10=lowest

class BusinessRuleEngine:
    """High-performance business rule execution engine."""

    def __init__(self,
                 connection_manager,
                 max_workers: int = 10,
                 cache_enabled: bool = True,
                 timeout_seconds: int = 300):
        self.connection_manager = connection_manager
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.timeout_seconds = timeout_seconds
        self.execution_cache = {}
        self.rule_registry = {}
        self.dependency_graph = {}

    async def execute_rule_set(self,
                              rule_set: 'RuleSet',
                              parallel: bool = True) -> 'ValidationSummary':
        """Execute a complete rule set with dependency management."""

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(rule_set.rules)

        # Validate no circular dependencies
        if self._has_circular_dependencies(dependency_graph):
            raise ValidationError("Circular dependencies detected in rule set")

        # Sort rules by dependencies and priority
        execution_order = self._topological_sort(dependency_graph, rule_set.rules)

        results = []
        failed_rules = set()

        if parallel:
            results = await self._execute_parallel(execution_order, failed_rules)
        else:
            results = await self._execute_sequential(execution_order, failed_rules)

        return self._create_summary(results, rule_set)

    async def _execute_parallel(self,
                               execution_order: List['BusinessRule'],
                               failed_rules: Set[str]) -> List['RuleResult']:
        """Execute rules in parallel while respecting dependencies."""

        results = []
        executing = {}  # rule_id -> Future
        completed = set()

        for batch in self._create_execution_batches(execution_order):
            # Start all rules in current batch
            batch_futures = {}

            for rule in batch:
                if rule.id not in failed_rules:
                    # Check if dependencies are satisfied
                    if self._dependencies_satisfied(rule, completed):
                        future = asyncio.create_task(self._execute_single_rule(rule))
                        batch_futures[rule.id] = future
                        executing[rule.id] = future

            # Wait for batch completion
            for rule_id, future in batch_futures.items():
                try:
                    result = await future
                    results.append(result)
                    completed.add(rule_id)

                    # If rule failed and has dependents, mark them as failed
                    if not result.passed:
                        failed_rules.update(self._get_dependent_rules(rule_id))

                except Exception as e:
                    # Handle rule execution failure
                    failed_rules.add(rule_id)
                    failed_rules.update(self._get_dependent_rules(rule_id))
                    results.append(self._create_error_result(rule_id, e))

        return results

    async def _execute_single_rule(self, rule: 'BusinessRule') -> 'RuleResult':
        """Execute a single business rule with caching and error handling."""

        # Check cache first
        if self.cache_enabled:
            cache_key = self._generate_cache_key(rule)
            if cache_key in self.execution_cache:
                cached_result = self.execution_cache[cache_key]
                if not self._is_cache_expired(cached_result):
                    return cached_result

        start_time = time.time()

        try:
            # Get database connection
            connection = await self.connection_manager.get_connection_async(
                rule.database or 'default'
            )

            # Execute rule query
            if rule.rule_type == RuleType.SQL_QUERY:
                result = await self._execute_sql_rule(connection, rule)
            elif rule.rule_type == RuleType.AGGREGATION:
                result = await self._execute_aggregation_rule(connection, rule)
            elif rule.rule_type == RuleType.COMPARISON:
                result = await self._execute_comparison_rule(connection, rule)
            else:
                raise ValueError(f"Unsupported rule type: {rule.rule_type}")

            execution_time = time.time() - start_time

            # Create result object
            rule_result = RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=result['passed'],
                message=result.get('message', ''),
                details=result.get('details', {}),
                execution_time=execution_time,
                timestamp=time.time(),
                affected_rows=result.get('affected_rows', 0)
            )

            # Cache result if appropriate
            if self.cache_enabled and self._should_cache(rule, execution_time):
                self.execution_cache[cache_key] = rule_result

            return rule_result

        except Exception as e:
            execution_time = time.time() - start_time
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                message=f"Rule execution failed: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__},
                execution_time=execution_time,
                timestamp=time.time()
            )

    def _build_dependency_graph(self, rules: List['BusinessRule']) -> Dict[str, Set[str]]:
        """Build dependency graph from rule dependencies."""
        graph = {}
        for rule in rules:
            graph[rule.id] = set(rule.depends_on or [])
        return graph

    def _topological_sort(self,
                         dependency_graph: Dict[str, Set[str]],
                         rules: List['BusinessRule']) -> List['BusinessRule']:
        """Sort rules by dependencies and priority."""

        # Implement Kahn's algorithm for topological sorting
        in_degree = {rule.id: 0 for rule in rules}
        rule_map = {rule.id: rule for rule in rules}

        # Calculate in-degrees
        for rule_id, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[rule_id] += 1

        # Priority queue with rules having no dependencies
        ready_queue = []
        for rule in rules:
            if in_degree[rule.id] == 0:
                ready_queue.append(rule)

        # Sort by priority
        ready_queue.sort(key=lambda r: r.priority)

        result = []
        while ready_queue:
            current = ready_queue.pop(0)
            result.append(current)

            # Update in-degrees for dependent rules
            for rule_id, deps in dependency_graph.items():
                if current.id in deps:
                    in_degree[rule_id] -= 1
                    if in_degree[rule_id] == 0:
                        ready_queue.append(rule_map[rule_id])
                        ready_queue.sort(key=lambda r: r.priority)

        return result
```

#### 1.2 Enhanced Rule Models
```python
# sqltest/modules/business_rules/models.py (enhancement)
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime

class RuleType(str, Enum):
    """Types of business rules supported."""
    SQL_QUERY = "sql_query"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    DATA_QUALITY = "data_quality"
    TEMPORAL_CONSISTENCY = "temporal_consistency"

class RuleSeverity(str, Enum):
    """Severity levels for rule violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ExecutionMode(str, Enum):
    """Rule execution modes."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"

class BusinessRule(BaseModel):
    """Enhanced business rule model with enterprise features."""

    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: Optional[str] = Field(None, description="Rule description")

    # Rule definition
    rule_type: RuleType = Field(..., description="Type of business rule")
    query: str = Field(..., description="SQL query or rule definition")
    expected_result: Any = Field(None, description="Expected result for comparison")

    # Execution settings
    severity: RuleSeverity = Field(RuleSeverity.MEDIUM, description="Rule severity")
    priority: int = Field(5, ge=1, le=10, description="Execution priority (1=highest)")
    timeout_seconds: int = Field(300, description="Rule execution timeout")
    retry_attempts: int = Field(0, description="Number of retry attempts on failure")

    # Dependencies and relationships
    depends_on: Optional[List[str]] = Field(None, description="Rule dependencies")
    tags: Optional[List[str]] = Field(None, description="Rule tags for organization")
    category: Optional[str] = Field(None, description="Rule category")

    # Metadata
    database: Optional[str] = Field(None, description="Target database")
    schema: Optional[str] = Field(None, description="Target schema")
    tables: Optional[List[str]] = Field(None, description="Affected tables")

    # Scheduling and execution
    execution_mode: ExecutionMode = Field(ExecutionMode.IMMEDIATE)
    schedule: Optional[str] = Field(None, description="Cron schedule for scheduled rules")

    # Caching
    cache_ttl: Optional[int] = Field(3600, description="Cache TTL in seconds")
    cache_key_fields: Optional[List[str]] = Field(None, description="Fields for cache key")

    # Alerting and notifications
    alert_on_failure: bool = Field(True, description="Send alerts on rule failure")
    notification_channels: Optional[List[str]] = Field(None, description="Notification channels")

    # Versioning and audit
    version: str = Field("1.0", description="Rule version")
    created_by: Optional[str] = Field(None, description="Rule creator")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RuleResult(BaseModel):
    """Enhanced rule execution result."""

    rule_id: str
    rule_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = {}

    # Performance metrics
    execution_time: float
    memory_usage: Optional[int] = None
    rows_processed: Optional[int] = None

    # Result data
    result_data: Optional[Any] = None
    affected_rows: int = 0

    # Timing and context
    timestamp: float
    execution_context: Optional[Dict[str, Any]] = {}

    # Error handling
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class ValidationSummary(BaseModel):
    """Summary of rule set execution."""

    total_rules: int
    passed_rules: int
    failed_rules: int
    skipped_rules: int

    # Performance metrics
    total_execution_time: float
    average_rule_time: float
    slowest_rule: Optional[str] = None

    # Results breakdown
    results_by_severity: Dict[str, int] = {}
    results_by_category: Dict[str, int] = {}

    # Execution context
    execution_start: datetime
    execution_end: datetime
    database_connections_used: List[str] = []

    # Detailed results
    rule_results: List[RuleResult] = []
```

### 1.3 Configuration Loader
```python
# sqltest/modules/business_rules/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import re

class BusinessRuleConfigLoader:
    """Load and validate business rule configurations."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.environment_cache = {}

    def load_rule_set(self, config_path: str) -> 'RuleSet':
        """Load rule set from YAML configuration."""

        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.base_path / config_file

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            raw_config = f.read()

        # Process environment variables
        processed_config = self._process_environment_variables(raw_config)

        # Parse YAML
        config_data = yaml.safe_load(processed_config)

        # Validate and create rule set
        return self._create_rule_set(config_data, config_file)

    def _process_environment_variables(self, config_text: str) -> str:
        """Process environment variable substitutions."""

        # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            var_expression = match.group(1)

            if ':' in var_expression:
                var_name, default_value = var_expression.split(':', 1)
            else:
                var_name, default_value = var_expression, None

            value = os.environ.get(var_name, default_value)

            if value is None:
                raise EnvironmentError(f"Required environment variable '{var_name}' is not set")

            return value

        return re.sub(pattern, replace_env_var, config_text)

    def _create_rule_set(self, config_data: Dict[str, Any], config_file: Path) -> 'RuleSet':
        """Create rule set from configuration data."""

        rule_set_data = config_data.get('rule_set', {})
        rules_data = config_data.get('rules', [])

        # Create rule objects
        rules = []
        for rule_data in rules_data:
            rule = BusinessRule(**rule_data)
            rules.append(rule)

        # Create rule set
        rule_set = RuleSet(
            name=rule_set_data.get('name', config_file.stem),
            description=rule_set_data.get('description'),
            rules=rules,
            metadata=rule_set_data.get('metadata', {}),
            execution_settings=rule_set_data.get('execution_settings', {})
        )

        return rule_set

def create_sample_business_rules(output_path: str):
    """Create comprehensive sample business rules configuration."""

    sample_config = {
        'rule_set': {
            'name': 'Enterprise Data Quality Rules',
            'description': 'Comprehensive business rule validation for enterprise data',
            'metadata': {
                'version': '1.0',
                'author': 'Data Quality Team',
                'created': '2024-01-01'
            },
            'execution_settings': {
                'parallel_execution': True,
                'max_workers': 10,
                'timeout_seconds': 300,
                'retry_on_failure': True
            }
        },
        'rules': [
            {
                'id': 'customer_data_integrity',
                'name': 'Customer Data Integrity Check',
                'description': 'Ensure customer data consistency across systems',
                'rule_type': 'sql_query',
                'query': '''
                    SELECT COUNT(*) as invalid_customers
                    FROM customers c
                    LEFT JOIN orders o ON c.customer_id = o.customer_id
                    WHERE c.email IS NULL
                       OR c.email NOT LIKE '%@%'
                       OR c.created_date > CURRENT_DATE
                ''',
                'expected_result': 0,
                'severity': 'critical',
                'priority': 1,
                'category': 'data_integrity',
                'tables': ['customers', 'orders'],
                'alert_on_failure': True
            },
            {
                'id': 'order_total_consistency',
                'name': 'Order Total Consistency',
                'description': 'Verify order totals match line item sums',
                'rule_type': 'comparison',
                'query': '''
                    SELECT o.order_id, o.total_amount, SUM(oi.quantity * oi.unit_price) as calculated_total
                    FROM orders o
                    JOIN order_items oi ON o.order_id = oi.order_id
                    GROUP BY o.order_id, o.total_amount
                    HAVING ABS(o.total_amount - SUM(oi.quantity * oi.unit_price)) > 0.01
                ''',
                'expected_result': 0,
                'severity': 'high',
                'priority': 2,
                'category': 'financial_integrity',
                'depends_on': ['customer_data_integrity']
            },
            {
                'id': 'inventory_balance',
                'name': 'Inventory Balance Validation',
                'description': 'Ensure inventory balances are not negative',
                'rule_type': 'aggregation',
                'query': '''
                    SELECT product_id, current_stock
                    FROM inventory
                    WHERE current_stock < 0
                ''',
                'expected_result': 0,
                'severity': 'medium',
                'priority': 3,
                'category': 'inventory_management',
                'schedule': '0 */6 * * *'  # Every 6 hours
            }
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
```

## 2. SQL Unit Testing Framework (Priority: CRITICAL)

### Current State: 0% Implementation
**Files to implement:**
- `sqltest/modules/sql_testing/executor.py` (empty)
- `sqltest/modules/sql_testing/fixtures.py` (empty)
- `sqltest/modules/sql_testing/config_loader.py` (empty)

### Technical Specification

#### 2.1 Test Execution Engine
```python
# sqltest/modules/sql_testing/executor.py
from typing import List, Dict, Any, Optional, Set
import asyncio
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
import logging

class TestExecutor:
    """High-performance SQL test execution engine."""

    def __init__(self,
                 connection_manager,
                 max_parallel_tests: int = 5,
                 isolation_level: str = "READ_COMMITTED",
                 temp_schema_prefix: str = "sqltest_temp"):

        self.connection_manager = connection_manager
        self.max_parallel_tests = max_parallel_tests
        self.isolation_level = isolation_level
        self.temp_schema_prefix = temp_schema_prefix
        self.fixture_manager = FixtureManager(connection_manager)
        self.assertion_engine = AssertionEngine()

        # Test execution tracking
        self.active_tests = {}
        self.test_results = {}
        self.coverage_data = {}

    async def execute_test_suite(self, test_suite: 'TestSuite') -> 'TestSuiteResult':
        """Execute a complete test suite with parallel execution."""

        start_time = time.time()

        # Create execution plan
        execution_plan = self._create_execution_plan(test_suite)

        # Setup test environment
        await self._setup_test_environment(test_suite)

        try:
            # Execute tests in batches
            test_results = []
            for batch in execution_plan:
                batch_results = await self._execute_test_batch(batch)
                test_results.extend(batch_results)

                # Early termination on critical failures
                if test_suite.settings.get('fail_fast', False):
                    critical_failures = [r for r in batch_results if r.status == TestStatus.FAILED and r.severity == 'critical']
                    if critical_failures:
                        break

            # Generate coverage report
            coverage_report = await self._generate_coverage_report(test_suite, test_results)

            # Create suite result
            suite_result = TestSuiteResult(
                suite_name=test_suite.name,
                total_tests=len(test_suite.tests),
                passed_tests=len([r for r in test_results if r.status == TestStatus.PASSED]),
                failed_tests=len([r for r in test_results if r.status == TestStatus.FAILED]),
                skipped_tests=len([r for r in test_results if r.status == TestStatus.SKIPPED]),
                execution_time=time.time() - start_time,
                test_results=test_results,
                coverage=coverage_report
            )

            return suite_result

        finally:
            # Cleanup test environment
            await self._cleanup_test_environment(test_suite)

    async def _execute_single_test(self, test: 'SQLTest') -> 'TestResult':
        """Execute a single SQL test with full isolation."""

        test_start = time.time()
        temp_schema = f"{self.temp_schema_prefix}_{test.id}_{int(time.time())}"

        try:
            # Get dedicated connection for test
            connection = await self.connection_manager.get_connection_async(
                test.database or 'default'
            )

            # Create isolated test schema
            await self._create_test_schema(connection, temp_schema)

            # Setup fixtures
            fixture_data = await self.fixture_manager.setup_fixtures(
                connection, test.fixtures, temp_schema
            )

            # Execute test query
            query_result = await self._execute_test_query(
                connection, test.query, temp_schema
            )

            # Run assertions
            assertion_results = await self.assertion_engine.run_assertions(
                test.assertions, query_result, fixture_data
            )

            # Determine test status
            status = TestStatus.PASSED if all(a.passed for a in assertion_results) else TestStatus.FAILED

            return TestResult(
                test_id=test.id,
                test_name=test.name,
                status=status,
                execution_time=time.time() - test_start,
                assertion_results=assertion_results,
                query_result=query_result,
                fixture_data=fixture_data,
                metadata={'temp_schema': temp_schema}
            )

        except Exception as e:
            return TestResult(
                test_id=test.id,
                test_name=test.name,
                status=TestStatus.FAILED,
                execution_time=time.time() - test_start,
                error_message=str(e),
                error_details={'exception_type': type(e).__name__}
            )

        finally:
            # Cleanup test schema
            try:
                await self._cleanup_test_schema(connection, temp_schema)
            except:
                pass

    def _create_execution_plan(self, test_suite: 'TestSuite') -> List[List['SQLTest']]:
        """Create optimized execution plan for tests."""

        # Group tests by dependencies and resource requirements
        independent_tests = []
        dependent_tests = []

        for test in test_suite.tests:
            if test.depends_on:
                dependent_tests.append(test)
            else:
                independent_tests.append(test)

        # Create batches for parallel execution
        batches = []
        batch_size = min(self.max_parallel_tests, len(independent_tests))

        # First batch: independent tests
        for i in range(0, len(independent_tests), batch_size):
            batch = independent_tests[i:i + batch_size]
            batches.append(batch)

        # Handle dependent tests
        while dependent_tests:
            ready_tests = []
            remaining_tests = []

            for test in dependent_tests:
                dependencies_satisfied = all(
                    dep_id in self.test_results and
                    self.test_results[dep_id].status == TestStatus.PASSED
                    for dep_id in test.depends_on
                )

                if dependencies_satisfied:
                    ready_tests.append(test)
                else:
                    remaining_tests.append(test)

            if ready_tests:
                # Create batches for ready tests
                for i in range(0, len(ready_tests), batch_size):
                    batch = ready_tests[i:i + batch_size]
                    batches.append(batch)

            dependent_tests = remaining_tests

            # Prevent infinite loop
            if not ready_tests and dependent_tests:
                # Mark remaining tests as skipped due to unmet dependencies
                for test in dependent_tests:
                    self.test_results[test.id] = TestResult(
                        test_id=test.id,
                        test_name=test.name,
                        status=TestStatus.SKIPPED,
                        skip_reason="Unmet dependencies"
                    )
                break

        return batches
```

## 3. Reporting System (Priority: HIGH)

### Current State: 0% Implementation

#### 3.1 Report Generation Engine
```python
# sqltest/reporting/generators/html_generator.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import json
from typing import Dict, Any, List

class HTMLReportGenerator:
    """Generate interactive HTML reports with charts and drill-down capabilities."""

    def __init__(self, template_path: Optional[Path] = None):
        template_dir = template_path or Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_validation_report(self,
                                 validation_results: 'ValidationSummary',
                                 output_path: Path) -> Path:
        """Generate comprehensive validation report."""

        template = self.env.get_template('validation_report.html')

        # Prepare data for template
        report_data = {
            'summary': validation_results,
            'charts_data': self._prepare_charts_data(validation_results),
            'detailed_results': self._prepare_detailed_results(validation_results),
            'performance_metrics': self._calculate_performance_metrics(validation_results),
            'timestamp': datetime.now().isoformat()
        }

        # Render template
        html_content = template.render(**report_data)

        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)

        return output_path

    def _prepare_charts_data(self, results: 'ValidationSummary') -> Dict[str, Any]:
        """Prepare data for interactive charts."""
        return {
            'pass_fail_chart': {
                'passed': results.passed_rules,
                'failed': results.failed_rules,
                'skipped': results.skipped_rules
            },
            'severity_breakdown': results.results_by_severity,
            'category_breakdown': results.results_by_category,
            'execution_timeline': self._create_timeline_data(results),
            'performance_distribution': self._create_performance_data(results)
        }
```

This comprehensive specification provides the foundation for implementing enterprise-grade core components with focus on performance, scalability, and reliability.