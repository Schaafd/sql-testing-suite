# SQLTest Pro API Reference

Complete API documentation for all SQLTest Pro modules.

## Table of Contents

- [Database Layer](#database-layer)
- [Testing Modules](#testing-modules)
- [Business Rules](#business-rules)
- [Field Validation](#field-validation)
- [Data Profiling](#data-profiling)
- [Reporting](#reporting)
- [Configuration](#configuration)

## Database Layer

### AdvancedConnectionManager

Enterprise-grade connection management with health monitoring and query optimization.

#### Location
`sqltest.db.advanced_connection`

#### Class Definition

```python
class AdvancedConnectionManager:
    """Advanced connection manager with enterprise features."""

    def __init__(
        self,
        config: SQLTestConfig,
        enable_query_analysis: bool = True,
        enable_result_cache: bool = True,
        cache_size_mb: int = 100
    ):
        """Initialize advanced connection manager.

        Args:
            config: SQLTest configuration
            enable_query_analysis: Enable query performance analysis
            enable_result_cache: Enable query result caching
            cache_size_mb: Cache size limit in megabytes
        """
```

#### Methods

##### `get_connection(database_name: str) -> Connection`

Get a connection from the pool with health checking.

```python
# Example usage
conn_manager = AdvancedConnectionManager(config)
connection = conn_manager.get_connection('production')
```

**Returns**: Database connection object

**Raises**:
- `DatabaseError`: If connection cannot be established
- `ConfigurationError`: If database not configured

##### `execute_query(sql: str, database_name: str, **kwargs) -> QueryResult`

Execute a query with automatic caching and analysis.

```python
# Example usage
result = conn_manager.execute_query(
    "SELECT * FROM users WHERE active = true",
    database_name='production'
)

print(f"Rows: {result.row_count}")
print(f"Execution time: {result.execution_time_ms}ms")
```

**Args**:
- `sql`: SQL query string
- `database_name`: Target database name
- `**kwargs`: Additional query parameters

**Returns**: `QueryResult` object with results and metadata

##### `get_pool_statistics(database_name: str) -> Dict[str, Any]`

Get connection pool statistics.

```python
# Example usage
stats = conn_manager.get_pool_statistics('production')
print(f"Active connections: {stats['active_connections']}")
print(f"Pool size: {stats['pool_size']}")
print(f"Overflow: {stats['overflow']}")
```

**Returns**: Dictionary containing pool metrics

---

### ConnectionMonitor

Monitors connection health and manages pool statistics.

#### Location
`sqltest.db.advanced_connection`

#### Class Definition

```python
class ConnectionMonitor:
    """Monitor connection health and pool statistics."""

    def get_health_status(self, connection_id: str) -> ConnectionHealth
    def get_overall_health_score(self) -> float
    def get_health_summary(self) -> Dict[str, Any]
```

#### Methods

##### `get_health_status(connection_id: str) -> ConnectionHealth`

Get health status for a specific connection.

**Returns**: `ConnectionHealth` object with status and metrics

##### `get_overall_health_score() -> float`

Calculate overall connection health score (0-100).

```python
# Example usage
monitor = conn_manager._monitor
health_score = monitor.get_overall_health_score()

if health_score < 80:
    print("Warning: Connection health degraded")
```

**Returns**: Health score between 0.0 and 100.0

---

### QueryAnalyzer

Analyzes query performance and provides optimization suggestions.

#### Location
`sqltest.db.query_analyzer`

#### Class Definition

```python
class QueryAnalyzer:
    """Analyze query performance and provide optimization insights."""

    def __init__(self, max_history_size: int = 50000):
        """Initialize query analyzer.

        Args:
            max_history_size: Maximum number of queries to track
        """
```

#### Methods

##### `analyze_query(sql: str, execution_time_ms: float, rows_affected: int = 0) -> QueryMetrics`

Analyze a query and track its performance.

```python
# Example usage
analyzer = QueryAnalyzer()
metrics = analyzer.analyze_query(
    sql="SELECT * FROM users WHERE created_at > NOW() - INTERVAL '1 day'",
    execution_time_ms=45.2,
    rows_affected=150
)

print(f"Query category: {metrics.category}")
print(f"Complexity: {metrics.complexity}")
```

**Args**:
- `sql`: SQL query string
- `execution_time_ms`: Query execution time in milliseconds
- `rows_affected`: Number of rows affected/returned

**Returns**: `QueryMetrics` object

##### `get_query_statistics(query_hash: str) -> QueryStatistics`

Get aggregated statistics for a specific query.

```python
# Example usage
stats = analyzer.get_query_statistics(query_hash)
print(f"Average execution time: {stats.avg_execution_time_ms:.2f}ms")
print(f"Execution count: {stats.execution_count}")
```

**Returns**: `QueryStatistics` with aggregated metrics

##### `generate_optimization_suggestions() -> List[QueryOptimizationSuggestion]`

Generate optimization suggestions based on query patterns.

```python
# Example usage
suggestions = analyzer.generate_optimization_suggestions()

for suggestion in suggestions:
    print(f"[{suggestion.suggestion_type}] {suggestion.description}")
    print(f"Impact: {suggestion.potential_impact}")
    print(f"Recommendations: {', '.join(suggestion.recommendations)}")
```

**Returns**: List of optimization suggestions

---

### QueryResultCache

Intelligent query result caching with multiple strategies.

#### Location
`sqltest.db.query_cache`

#### Class Definition

```python
class QueryResultCache:
    """Intelligent query result cache with multiple strategies."""

    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 300,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.SMART
    ):
        """Initialize query result cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl_seconds: Default time-to-live for cache entries
            eviction_policy: Cache eviction policy (LRU, LFU, FIFO, SIZE, TTL, SMART)
        """
```

#### Methods

##### `get(query: str, strategy: CacheStrategy = CacheStrategy.TTL) -> Optional[CacheEntry]`

Retrieve cached query result.

```python
# Example usage
cache = QueryResultCache(max_size_mb=100)
cached_result = cache.get("SELECT * FROM users")

if cached_result:
    print(f"Cache hit! Data: {cached_result.result}")
else:
    print("Cache miss")
```

**Args**:
- `query`: SQL query string
- `strategy`: Caching strategy to use

**Returns**: Cached entry or None if not found

##### `put(query: str, result: QueryResult, strategy: CacheStrategy = CacheStrategy.TTL)`

Store query result in cache.

```python
# Example usage
cache.put(
    "SELECT * FROM users",
    query_result,
    strategy=CacheStrategy.ADAPTIVE
)
```

**Args**:
- `query`: SQL query string
- `result`: Query result to cache
- `strategy`: Caching strategy

##### `get_statistics() -> Dict[str, Any]`

Get cache performance statistics.

```python
# Example usage
stats = cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.2f}%")
print(f"Total hits: {stats['hits']}")
print(f"Total misses: {stats['misses']}")
```

**Returns**: Dictionary with cache statistics

---

### TransactionManager

Manages distributed transactions with two-phase commit support.

#### Location
`sqltest.db.transaction_manager`

#### Class Definition

```python
class TransactionManager:
    """Manage distributed transactions with 2PC support."""

    def __init__(self, connection_manager: AdvancedConnectionManager):
        """Initialize transaction manager.

        Args:
            connection_manager: Connection manager instance
        """
```

#### Methods

##### `begin_transaction(databases: List[str], timeout_seconds: int = 300) -> str`

Begin a distributed transaction.

```python
# Example usage
txn_manager = TransactionManager(conn_manager)
txn_id = txn_manager.begin_transaction(
    databases=['production', 'analytics'],
    timeout_seconds=300
)
```

**Args**:
- `databases`: List of database names to include in transaction
- `timeout_seconds`: Transaction timeout

**Returns**: Transaction ID (UUID string)

##### `execute_operation(transaction_id: str, database_name: str, sql: str) -> QueryResult`

Execute an operation within a transaction.

```python
# Example usage
result = txn_manager.execute_operation(
    txn_id,
    'production',
    'UPDATE users SET status = "active" WHERE id = 1'
)
```

**Args**:
- `transaction_id`: Transaction ID
- `database_name`: Target database
- `sql`: SQL statement to execute

**Returns**: Query result

##### `two_phase_commit(transaction_id: str) -> bool`

Execute two-phase commit protocol.

```python
# Example usage
try:
    success = txn_manager.two_phase_commit(txn_id)
    if success:
        print("Transaction committed successfully")
    else:
        print("Transaction failed during commit")
except Exception as e:
    print(f"Commit error: {e}")
```

**Returns**: True if commit succeeded, False otherwise

##### `abort(transaction_id: str, reason: str = "User abort")`

Abort a transaction and rollback all changes.

```python
# Example usage
txn_manager.abort(txn_id, "Validation failed")
```

**Args**:
- `transaction_id`: Transaction ID
- `reason`: Reason for abort (for logging)

##### `create_savepoint(transaction_id: str, database_name: str, savepoint_name: Optional[str] = None) -> Savepoint`

Create a savepoint within a transaction.

```python
# Example usage
savepoint = txn_manager.create_savepoint(
    txn_id,
    'production',
    'before_critical_operation'
)
```

**Returns**: `Savepoint` object

##### `rollback_to_savepoint(transaction_id: str, database_name: str, savepoint_name: str) -> bool`

Rollback to a specific savepoint.

```python
# Example usage
success = txn_manager.rollback_to_savepoint(
    txn_id,
    'production',
    'before_critical_operation'
)
```

**Returns**: True if rollback succeeded

---

### QueryRouter

Routes queries to appropriate database nodes with load balancing.

#### Location
`sqltest.db.query_router`

#### Class Definition

```python
class QueryRouter:
    """Route queries to appropriate database nodes."""

    def __init__(
        self,
        connection_manager: AdvancedConnectionManager,
        default_routing_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    ):
        """Initialize query router.

        Args:
            connection_manager: Connection manager instance
            default_routing_strategy: Default load balancing strategy
        """
```

#### Methods

##### `classify_query(sql: str) -> QueryType`

Classify query as READ, WRITE, or DDL.

```python
# Example usage
router = QueryRouter(conn_manager)
query_type = router.classify_query("SELECT * FROM users")
print(f"Query type: {query_type}")  # QueryType.READ
```

**Returns**: `QueryType` enum value

##### `route_query(sql: str, database_name: str, routing_strategy: Optional[RoutingStrategy] = None) -> Tuple[str, str]`

Route query to appropriate database node.

```python
# Example usage
node_id, node_type = router.route_query(
    "SELECT * FROM users WHERE active = true",
    'production',
    routing_strategy=RoutingStrategy.LEAST_RESPONSE_TIME
)

print(f"Routing to {node_type}: {node_id}")
```

**Args**:
- `sql`: SQL query
- `database_name`: Database name
- `routing_strategy`: Load balancing strategy (optional)

**Returns**: Tuple of (node_id, node_type)

---

## Testing Modules

### TestExecutionEngine

Execute SQL unit tests with isolation and parallelization.

#### Location
`sqltest.modules.testing.test_runner`

#### Class Definition

```python
class TestExecutionEngine:
    """Execute SQL unit tests with isolation."""

    def __init__(
        self,
        connection_manager: AdvancedConnectionManager,
        transaction_manager: TransactionManager
    ):
        """Initialize test execution engine.

        Args:
            connection_manager: Connection manager instance
            transaction_manager: Transaction manager instance
        """
```

#### Methods

##### `register_suite(suite: TestSuite)`

Register a test suite for execution.

```python
# Example usage
engine = TestExecutionEngine(conn_manager, txn_manager)
suite = TestSuite(
    suite_id='test_suite_1',
    name='User Tests',
    tests=[test1, test2]
)
engine.register_suite(suite)
```

**Args**:
- `suite`: Test suite to register

##### `run_suite(suite_id: str, filter_tags: Optional[Set[str]] = None) -> Dict[str, TestResult]`

Execute all tests in a suite.

```python
# Example usage
results = engine.run_suite(
    'test_suite_1',
    filter_tags={'integration', 'critical'}
)

for test_id, result in results.items():
    print(f"{result.test_name}: {result.status}")
```

**Args**:
- `suite_id`: Suite identifier
- `filter_tags`: Only run tests with these tags
- `exclude_tags`: Skip tests with these tags

**Returns**: Dictionary mapping test_id to TestResult

##### `run_single_test(test: TestCase) -> TestResult`

Execute a single test with isolation.

```python
# Example usage
test = TestCase(
    test_id='test_001',
    name='Verify user count',
    sql_query='SELECT COUNT(*) as count FROM users',
    database_name='production',
    assertions=[{'type': 'row_count', 'expected': 1}]
)

result = engine.run_single_test(test)
print(f"Test status: {result.status}")
print(f"Execution time: {result.execution_time_ms}ms")
```

**Args**:
- `test`: Test case to execute

**Returns**: `TestResult` object

---

### SQLAssertions

Comprehensive assertion library for SQL testing.

#### Location
`sqltest.modules.testing.assertions`

#### Static Methods

##### `assert_row_count(actual: int, expected: int, message: Optional[str] = None)`

Assert query returned expected number of rows.

```python
# Example usage
SQLAssertions.assert_row_count(actual=10, expected=10)
```

**Raises**: `AssertionError` if assertion fails

##### `assert_not_empty(row_count: int, message: Optional[str] = None)`

Assert result set is not empty.

```python
# Example usage
SQLAssertions.assert_not_empty(row_count=5)
```

##### `assert_columns_exist(columns: List[str], expected_columns: List[str])`

Assert expected columns exist in result.

```python
# Example usage
SQLAssertions.assert_columns_exist(
    columns=['id', 'name', 'email'],
    expected_columns=['id', 'email']
)
```

##### `assert_no_nulls(df: pd.DataFrame, column: str, message: Optional[str] = None)`

Assert column contains no NULL values.

```python
# Example usage
SQLAssertions.assert_no_nulls(df=result_df, column='user_id')
```

##### `assert_unique_values(df: pd.DataFrame, column: str, message: Optional[str] = None)`

Assert all values in column are unique.

```python
# Example usage
SQLAssertions.assert_unique_values(df=result_df, column='email')
```

##### `assert_execution_time_under(actual_ms: float, max_ms: float)`

Assert query execution time is under threshold.

```python
# Example usage
SQLAssertions.assert_execution_time_under(actual_ms=45.2, max_ms=100.0)
```

---

### AssertionBuilder

Fluent assertion builder for readable tests.

#### Location
`sqltest.modules.testing.assertions`

#### Usage

```python
from sqltest.modules.testing.assertions import assert_that

# Example usage
assert_that(row_count).is_greater_than(0).is_less_than(1000)
assert_that(value).equals(42).is_not_none()
assert_that(email).contains('@').has_length(20)
```

#### Methods

- `with_message(message: str)` - Set custom error message
- `equals(expected: Any)` - Assert value equals expected
- `not_equals(not_expected: Any)` - Assert value does not equal
- `is_greater_than(minimum: Any)` - Assert value > minimum
- `is_less_than(maximum: Any)` - Assert value < maximum
- `is_between(min_value: Any, max_value: Any)` - Assert value in range
- `is_not_none()` - Assert value is not None
- `is_none()` - Assert value is None
- `contains(item: Any)` - Assert collection contains item
- `has_length(expected_length: int)` - Assert collection length
- `is_empty()` - Assert collection is empty
- `is_not_empty()` - Assert collection is not empty

---

### MockDataGenerator

Generate realistic mock data for testing.

#### Location
`sqltest.modules.testing.fixtures`

#### Static Methods

##### `generate_integers(count: int, min_val: int = 1, max_val: int = 1000) -> List[int]`

Generate random integers.

```python
# Example usage
ids = MockDataGenerator.generate_integers(count=100, min_val=1, max_val=1000)
```

##### `generate_emails(count: int, domain: str = "example.com") -> List[str]`

Generate random email addresses.

```python
# Example usage
emails = MockDataGenerator.generate_emails(count=50, domain="test.com")
```

##### `generate_dates(count: int, start_date: Optional[datetime] = None, days_range: int = 365) -> List[datetime]`

Generate random dates.

```python
# Example usage
dates = MockDataGenerator.generate_dates(
    count=100,
    start_date=datetime(2023, 1, 1),
    days_range=365
)
```

---

### CoverageTracker

Track test coverage for queries and tables.

#### Location
`sqltest.modules.testing.reporting`

#### Methods

##### `record_test(query: str, tables_accessed: List[str])`

Record test execution for coverage.

```python
# Example usage
tracker = CoverageTracker(schema_introspector)
tracker.record_test(
    query="SELECT * FROM users WHERE active = true",
    tables_accessed=['users']
)
```

##### `get_coverage_metrics(database_schema) -> CoverageMetrics`

Calculate coverage metrics.

```python
# Example usage
metrics = tracker.get_coverage_metrics(database_schema)
print(f"Table coverage: {metrics.table_coverage_percent:.2f}%")
print(f"Tested tables: {metrics.tested_tables}/{metrics.total_tables}")
print(f"Untested tables: {metrics.untested_tables}")
```

**Returns**: `CoverageMetrics` object

---

### TestReportGenerator

Generate test reports in multiple formats.

#### Location
`sqltest.modules.testing.reporting`

#### Methods

##### `generate_html_report(results: Dict[str, Any], output_path: Path) -> str`

Generate HTML test report.

```python
# Example usage
generator = TestReportGenerator()
report_path = generator.generate_html_report(
    results=test_results,
    output_path=Path('test_report.html')
)
```

**Returns**: Path to generated report

##### `generate_junit_xml(results: Dict[str, Any], output_path: Path) -> str`

Generate JUnit XML report for CI/CD.

```python
# Example usage
xml_path = generator.generate_junit_xml(
    results=test_results,
    output_path=Path('junit.xml')
)
```

**Returns**: Path to XML file

---

## Business Rules

### BusinessRuleEngine

Execute complex business rules with dependency management.

#### Location
`sqltest.modules.business_rules.engine`

#### Class Definition

```python
class BusinessRuleEngine:
    """Execute business rules with caching and retry logic."""

    def __init__(
        self,
        connection_manager: AdvancedConnectionManager,
        enable_caching: bool = True,
        max_workers: int = 4
    ):
        """Initialize business rule engine.

        Args:
            connection_manager: Connection manager instance
            enable_caching: Enable result caching
            max_workers: Number of parallel workers
        """
```

#### Methods

##### `execute_rule(rule: BusinessRule, database_name: str) -> RuleResult`

Execute a single business rule.

```python
# Example usage
engine = BusinessRuleEngine(conn_manager)
rule = BusinessRule(
    name="Check orphaned orders",
    rule_type=RuleType.DATA_QUALITY,
    severity=RuleSeverity.ERROR,
    scope=ValidationScope.TABLE,
    sql_query="SELECT COUNT(*) as count FROM orders o LEFT JOIN customers c ON o.customer_id = c.id WHERE c.id IS NULL"
)

result = engine.execute_rule(rule, 'production')
print(f"Rule passed: {result.passed}")
print(f"Violations: {result.violation_count}")
```

**Args**:
- `rule`: Business rule to execute
- `database_name`: Target database

**Returns**: `RuleResult` object

##### `execute_rules_parallel(rules: List[BusinessRule], database_name: str) -> List[RuleResult]`

Execute multiple rules in parallel.

```python
# Example usage
results = engine.execute_rules_parallel(
    rules=[rule1, rule2, rule3],
    database_name='production'
)

for result in results:
    print(f"{result.rule_name}: {result.status}")
```

**Args**:
- `rules`: List of business rules
- `database_name`: Target database

**Returns**: List of `RuleResult` objects

---

## Field Validation

### FieldValidator

Validate individual columns with rules.

#### Location
`sqltest.modules.field_validator.validator`

#### Methods

##### `validate_field(table_name: str, database_name: str, rule: ValidationRule) -> ValidationResult`

Validate a field against a rule.

```python
# Example usage
from sqltest.modules.field_validator import FieldValidator, ValidationRule

validator = FieldValidator(conn_manager)
rule = ValidationRule(
    column='email',
    rule_type='regex',
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

result = validator.validate_field(
    table_name='users',
    database_name='production',
    rule=rule
)

print(f"Validation passed: {result.passed}")
print(f"Violations: {result.violations}")
```

**Returns**: `ValidationResult` object

---

## Data Profiling

### DataProfiler

Comprehensive data profiling and quality analysis.

#### Location
`sqltest.modules.profiler.profiler`

#### Methods

##### `profile_table(table_name: str, database_name: str) -> ProfileResult`

Profile a table with comprehensive statistics.

```python
# Example usage
from sqltest.modules.profiler import DataProfiler

profiler = DataProfiler(conn_manager)
profile = profiler.profile_table(
    table_name='users',
    database_name='production'
)

print(f"Row count: {profile.row_count}")
print(f"Quality score: {profile.quality_score:.2f}%")
print(f"Column count: {len(profile.column_stats)}")
```

**Returns**: `ProfileResult` object

---

## Configuration

### SQLTestConfig

Main configuration class.

#### Location
`sqltest.config.models`

#### Class Definition

```python
class SQLTestConfig(BaseModel):
    """Main SQLTest configuration."""

    databases: Dict[str, DatabaseConfig]
    profiling: Optional[ProfilingConfig] = None
    validation: Optional[ValidationConfig] = None
    testing: Optional[TestingConfig] = None
    reporting: Optional[ReportingConfig] = None
```

#### Loading Configuration

```python
from sqltest.config import load_config

# Load from YAML file
config = load_config('config.yaml')

# Access configuration
db_config = config.databases['production']
print(f"Database host: {db_config.host}")
```

---

## Error Handling

### Exception Hierarchy

```python
from sqltest.exceptions import (
    SQLTestError,           # Base exception
    ConfigurationError,     # Configuration errors
    DatabaseError,          # Database connection/query errors
    ValidationError,        # Validation failures
    TestExecutionError      # Test execution errors
)
```

#### Usage

```python
try:
    result = conn_manager.execute_query(sql, 'production')
except DatabaseError as e:
    print(f"Database error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## Complete Example

```python
from sqltest.config import load_config
from sqltest.db.advanced_connection import AdvancedConnectionManager
from sqltest.db.transaction_manager import TransactionManager
from sqltest.modules.testing import (
    TestExecutionEngine,
    TestCase,
    TestSuite,
    TestPriority,
    assert_that
)

# Load configuration
config = load_config('config.yaml')

# Initialize managers
conn_manager = AdvancedConnectionManager(
    config,
    enable_query_analysis=True,
    enable_result_cache=True
)
txn_manager = TransactionManager(conn_manager)

# Create test engine
test_engine = TestExecutionEngine(conn_manager, txn_manager)

# Define tests
test1 = TestCase(
    test_id='test_001',
    name='Verify active users',
    database_name='production',
    sql_query='SELECT COUNT(*) as count FROM users WHERE active = true',
    assertions=[
        {'type': 'not_empty'},
        {'type': 'row_count', 'expected': 1}
    ],
    priority=TestPriority.HIGH
)

# Create suite
suite = TestSuite(
    suite_id='suite_001',
    name='User Tests',
    tests=[test1],
    parallel_execution=True
)

# Register and run
test_engine.register_suite(suite)
results = test_engine.run_suite('suite_001')

# Check results
for test_id, result in results.items():
    print(f"{result.test_name}: {result.status}")
    print(f"Execution time: {result.execution_time_ms}ms")
```

---

For more information, see:
- [User Guide](USER_GUIDE.md) - Detailed usage instructions
- [Configuration Guide](CONFIGURATION.md) - Configuration options
- [Testing Guide](TESTING_GUIDE.md) - Best practices

**SQLTest Pro** - Enterprise SQL Testing Framework