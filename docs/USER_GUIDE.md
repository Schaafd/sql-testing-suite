# SQLTest Pro User Guide

A comprehensive guide to using SQLTest Pro for SQL testing, validation, and data quality assurance.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Database Configuration](#database-configuration)
- [Data Profiling](#data-profiling)
- [Field Validation](#field-validation)
- [Business Rules](#business-rules)
- [SQL Unit Testing](#sql-unit-testing)
- [Reporting](#reporting)
- [CI/CD Integration](#cicd-integration)
- [Advanced Features](#advanced-features)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Database drivers for your target database(s)

### Install SQLTest Pro

```bash
# Clone the repository
git clone https://github.com/Schaafd/sql-testing-suite.git
cd sql-testing-suite

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install SQLTest Pro
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

### Install Database Drivers

```bash
# PostgreSQL
pip install psycopg2-binary

# MySQL
pip install pymysql

# SQL Server
pip install pyodbc

# Snowflake
pip install snowflake-sqlalchemy
```

### Verify Installation

```bash
# Check installation
sqltest --version

# View help
sqltest --help

# Run demo
python demo_quick.py
```

## Quick Start

### 1. Configure Your Database

Create a `database.yaml` file:

```yaml
databases:
  production:
    type: postgresql
    host: localhost
    port: 5432
    database: myapp
    username: postgres
    password: ${DB_PASSWORD}  # Environment variable

    # Connection pooling (optional)
    pool_config:
      min_connections: 2
      max_connections: 20
      pool_recycle: 3600
      pool_pre_ping: true
```

Set environment variables:

```bash
export DB_PASSWORD='your_password_here'
```

### 2. Profile Your Data

```bash
# Profile a specific table
sqltest profile --table users --database production

# Generate HTML report
sqltest profile --table sales_data --output report.html

# Profile entire database
sqltest profile --database production --all-tables
```

### 3. Run Validations

Create a `validations.yaml` file:

```yaml
field_validations:
  - table: users
    validations:
      - column: email
        rules:
          - type: regex
            pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
          - type: not_null
          - type: unique
```

Run validations:

```bash
sqltest validate --config validations.yaml
```

### 4. Execute Unit Tests

Create a `unit_tests.yaml` file:

```yaml
unit_tests:
  - test_group: "User Functions"
    tests:
      - name: "Test email validation"
        test:
          query: "SELECT validate_email('test@example.com') as valid"
          assertions:
            - type: equals
              column: valid
              expected: true
```

Run tests:

```bash
sqltest test --config unit_tests.yaml
```

## Database Configuration

### Basic Configuration

```yaml
databases:
  dev:
    type: postgresql
    host: localhost
    port: 5432
    database: myapp_dev
    username: dev_user
    password: ${DEV_DB_PASSWORD}
```

### Supported Database Types

- `postgresql` - PostgreSQL 9.6+
- `mysql` - MySQL 5.7+ / MariaDB 10.2+
- `sqlite` - SQLite 3.8+
- `mssql` - Microsoft SQL Server 2012+
- `snowflake` - Snowflake Cloud Data Warehouse

### Advanced Connection Pooling

```yaml
databases:
  production:
    type: postgresql
    host: db.example.com
    port: 5432
    database: production_db
    username: app_user
    password: ${PROD_DB_PASSWORD}

    # Advanced connection pool configuration
    pool_config:
      # Basic pool settings
      min_connections: 5          # Minimum connections in pool
      max_connections: 50         # Maximum connections
      max_overflow: 10            # Extra connections beyond max

      # Connection lifecycle
      pool_recycle: 3600          # Recycle connections after 1 hour
      pool_pre_ping: true         # Test connection before using
      max_connection_age: 7200    # Max age: 2 hours

      # Health monitoring
      health_check_interval: 60   # Check health every 60 seconds
      connection_probe_query: "SELECT 1"

      # Performance tuning
      enable_connection_events: true
      pool_reset_on_return: true
      connection_timeout: 30      # Connection timeout in seconds
```

### Read/Write Splitting

```yaml
databases:
  production:
    type: postgresql

    # Primary (write) database
    host: primary.db.example.com
    port: 5432
    database: production_db
    username: app_user
    password: ${PROD_DB_PASSWORD}

    # Read replicas
    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432

    # Routing strategy
    routing_strategy: LEAST_RESPONSE_TIME  # or ROUND_ROBIN, LEAST_CONNECTIONS, WEIGHTED, RANDOM
```

### Environment Variables

SQLTest Pro supports environment variable substitution in configuration files:

```yaml
databases:
  prod:
    password: ${DB_PASSWORD}           # Required variable
    host: ${DB_HOST:localhost}         # Default value if not set
    port: ${DB_PORT:5432}              # Type conversion happens automatically
```

Set environment variables:

```bash
# Linux/macOS
export DB_PASSWORD='secret'
export DB_HOST='db.example.com'

# Windows PowerShell
$env:DB_PASSWORD = 'secret'
$env:DB_HOST = 'db.example.com'
```

## Data Profiling

Data profiling provides comprehensive statistical analysis and data quality insights.

### Basic Profiling

```bash
# Profile a single table
sqltest profile --table users --database production

# Profile with HTML output
sqltest profile --table orders --output report.html

# Profile multiple tables
sqltest profile --tables users,orders,products --database production
```

### Profiling Configuration

Create a `profile_config.yaml`:

```yaml
profiling:
  # Tables to profile
  tables:
    - users
    - orders
    - products

  # Profiling options
  options:
    calculate_percentiles: true
    detect_patterns: true
    identify_outliers: true
    max_distinct_values: 100
    sample_size: 10000  # For large tables

  # Quality thresholds
  thresholds:
    min_completeness: 95.0      # % of non-null values
    max_null_percentage: 5.0
    max_outlier_percentage: 2.0
```

Run with configuration:

```bash
sqltest profile --config profile_config.yaml
```

### Profiling Results

The profiler generates:

- **Statistical Metrics**: Mean, median, mode, std dev, min, max, percentiles
- **Data Quality Scores**: Completeness, uniqueness, validity
- **Pattern Detection**: Common patterns, regex suggestions
- **Outlier Identification**: Statistical outliers with z-scores
- **Distribution Analysis**: Histograms and frequency tables

### Programmatic Usage

```python
from sqltest.modules.profiler import DataProfiler
from sqltest.db import ConnectionManager

# Initialize
conn_manager = ConnectionManager(config)
profiler = DataProfiler(conn_manager)

# Profile a table
profile_result = profiler.profile_table(
    table_name='users',
    database_name='production'
)

# Access results
print(f"Total rows: {profile_result.row_count}")
print(f"Data quality score: {profile_result.quality_score:.2f}%")

for column, stats in profile_result.column_stats.items():
    print(f"{column}: {stats.null_percentage:.2f}% nulls")
```

## Field Validation

Field validation provides rule-based validation for individual columns.

### Validation Rules

#### Not Null Validation

```yaml
field_validations:
  - table: users
    validations:
      - column: user_id
        rules:
          - type: not_null
            severity: error
```

#### Unique Validation

```yaml
field_validations:
  - table: users
    validations:
      - column: email
        rules:
          - type: unique
            severity: error
```

#### Regex Pattern Validation

```yaml
field_validations:
  - table: users
    validations:
      - column: email
        rules:
          - type: regex
            pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            message: "Invalid email format"

      - column: phone
        rules:
          - type: regex
            pattern: '^\+?1?\d{10,14}$'
            message: "Invalid phone number"
```

#### Range Validation

```yaml
field_validations:
  - table: products
    validations:
      - column: price
        rules:
          - type: range
            min: 0.01
            max: 999999.99
            severity: error

      - column: quantity
        rules:
          - type: range
            min: 0
            max: null  # No upper limit
```

#### Enum Validation

```yaml
field_validations:
  - table: orders
    validations:
      - column: status
        rules:
          - type: enum
            allowed_values:
              - pending
              - processing
              - shipped
              - delivered
              - cancelled
```

#### Length Validation

```yaml
field_validations:
  - table: users
    validations:
      - column: username
        rules:
          - type: length
            min: 3
            max: 50

      - column: bio
        rules:
          - type: length
            max: 500
```

### Custom Validation Rules

```yaml
field_validations:
  - table: transactions
    validations:
      - column: amount
        rules:
          - type: custom
            sql: "amount > 0 AND amount < account_balance"
            message: "Transaction amount must be positive and not exceed balance"
```

### Running Validations

```bash
# Run all validations
sqltest validate --config validations.yaml

# Run specific table validations
sqltest validate --config validations.yaml --table users

# Output to JSON
sqltest validate --config validations.yaml --output results.json

# Fail on errors (useful for CI/CD)
sqltest validate --config validations.yaml --strict
```

### Programmatic Usage

```python
from sqltest.modules.field_validator import FieldValidator, ValidationRule

# Create validator
validator = FieldValidator(conn_manager)

# Define validation
rule = ValidationRule(
    column='email',
    rule_type='regex',
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Run validation
result = validator.validate_field(
    table_name='users',
    database_name='production',
    rule=rule
)

# Check results
if result.passed:
    print("Validation passed!")
else:
    print(f"Validation failed: {result.violations} violations")
```

## Business Rules

Business rules enable complex cross-table validations and data quality checks.

### Rule Types

#### Data Quality Rules

```yaml
business_rules:
  - name: "Check for orphaned orders"
    type: DATA_QUALITY
    severity: ERROR
    scope: TABLE
    sql_query: |
      SELECT COUNT(*) as violation_count
      FROM orders o
      LEFT JOIN customers c ON o.customer_id = c.customer_id
      WHERE c.customer_id IS NULL
    expected_result: 0
```

#### Business Logic Rules

```yaml
business_rules:
  - name: "Sales amount in valid range"
    type: BUSINESS_LOGIC
    severity: WARNING
    scope: TABLE
    sql_query: |
      SELECT COUNT(*) as violation_count
      FROM sales
      WHERE amount <= 0 OR amount > 100000
    expected_result: 0
```

#### Referential Integrity Rules

```yaml
business_rules:
  - name: "Check customer references"
    type: REFERENTIAL_INTEGRITY
    severity: ERROR
    scope: DATABASE
    sql_query: |
      SELECT 'orders' as table_name, COUNT(*) as orphans
      FROM orders o
      WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.id = o.customer_id)
      UNION ALL
      SELECT 'invoices', COUNT(*)
      FROM invoices i
      WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.id = i.customer_id)
```

### Rule Execution

```bash
# Run all business rules
sqltest rules --config business_rules.yaml

# Run specific severity level
sqltest rules --config business_rules.yaml --severity ERROR

# Parallel execution
sqltest rules --config business_rules.yaml --parallel --workers 4

# Generate report
sqltest rules --config business_rules.yaml --output report.html
```

### Rule Dependencies

```yaml
business_rules:
  - name: "Base data check"
    rule_id: base_check
    type: DATA_QUALITY
    sql_query: "SELECT COUNT(*) FROM users"

  - name: "Advanced validation"
    depends_on:
      - base_check  # This rule runs only after base_check passes
    type: BUSINESS_LOGIC
    sql_query: "SELECT COUNT(*) FROM orders WHERE user_id NOT IN (SELECT id FROM users)"
```

### Programmatic Usage

```python
from sqltest.modules.business_rules import BusinessRuleEngine, BusinessRule, RuleType, RuleSeverity

# Initialize engine
engine = BusinessRuleEngine(conn_manager)

# Define rule
rule = BusinessRule(
    name="Check duplicate emails",
    rule_type=RuleType.DATA_QUALITY,
    severity=RuleSeverity.ERROR,
    scope=ValidationScope.TABLE,
    sql_query="""
        SELECT email, COUNT(*) as count
        FROM users
        GROUP BY email
        HAVING COUNT(*) > 1
    """
)

# Execute rule
result = engine.execute_rule(rule, database_name='production')

# Check result
if result.passed:
    print("Rule passed!")
else:
    print(f"Rule failed: {result.violation_count} violations")
```

## SQL Unit Testing

Comprehensive unit testing framework for SQL code with isolation and CI/CD integration.

### Test Structure

```yaml
unit_tests:
  - test_suite: "User Management Tests"
    database: production
    parallel_execution: true
    max_workers: 4

    setup_suite:
      - "CREATE TEMP TABLE test_users AS SELECT * FROM users LIMIT 0"

    teardown_suite:
      - "DROP TABLE IF EXISTS test_users"

    tests:
      - name: "Test user creation"
        priority: HIGH
        timeout_seconds: 30

        setup:
          - "DELETE FROM test_users"

        test:
          query: |
            INSERT INTO test_users (username, email)
            VALUES ('testuser', 'test@example.com')
            RETURNING *

          assertions:
            - type: row_count
              expected: 1
            - type: not_empty

        teardown:
          - "DELETE FROM test_users WHERE username = 'testuser'"
```

### Assertion Types

#### Row Count Assertions

```yaml
tests:
  - name: "Check user count"
    test:
      query: "SELECT * FROM users WHERE active = true"
      assertions:
        - type: row_count
          expected: 100

        - type: row_count_greater_than
          expected: 50
```

#### Value Assertions

```yaml
tests:
  - name: "Check total sales"
    test:
      query: "SELECT SUM(amount) as total FROM sales"
      assertions:
        - type: equals
          column: total
          expected: 50000.00

        - type: greater_than
          column: total
          expected: 0
```

#### Execution Time Assertions

```yaml
tests:
  - name: "Check query performance"
    test:
      query: "SELECT * FROM large_table WHERE indexed_column = 123"
      assertions:
        - type: execution_time
          expected: 100  # milliseconds
```

#### Schema Assertions

```yaml
tests:
  - name: "Check table structure"
    test:
      query: "SELECT * FROM users LIMIT 1"
      assertions:
        - type: columns_exist
          expected_columns:
            - user_id
            - username
            - email
            - created_at
```

### Test Fixtures

```yaml
fixtures:
  - name: "sample_users"
    table: users
    cleanup: true
    data:
      - username: "testuser1"
        email: "test1@example.com"
      - username: "testuser2"
        email: "test2@example.com"

unit_tests:
  - test_suite: "With Fixtures"
    fixtures:
      - sample_users

    tests:
      - name: "Test with fixture data"
        test:
          query: "SELECT COUNT(*) as count FROM users WHERE username LIKE 'testuser%'"
          assertions:
            - type: equals
              column: count
              expected: 2
```

### Running Tests

```bash
# Run all tests
sqltest test --config unit_tests.yaml

# Run specific test suite
sqltest test --config unit_tests.yaml --suite "User Management Tests"

# Run with specific tags
sqltest test --config unit_tests.yaml --tags integration,smoke

# Parallel execution
sqltest test --config unit_tests.yaml --parallel --workers 8

# Generate coverage report
sqltest test --config unit_tests.yaml --coverage --output coverage.html

# Generate JUnit XML for CI/CD
sqltest test --config unit_tests.yaml --junit junit.xml
```

### Test Isolation

All tests run in isolated transactions that are automatically rolled back:

```python
# Automatically handled by TestExecutionEngine
# 1. Begin transaction
# 2. Run setup SQL
# 3. Execute test query
# 4. Run assertions
# 5. Execute teardown SQL
# 6. Rollback transaction (always, even on success)
```

This ensures tests never affect your database state!

### Programmatic Usage

```python
from sqltest.modules.testing import (
    TestExecutionEngine,
    TestCase,
    TestSuite,
    TestPriority
)

# Initialize engine
engine = TestExecutionEngine(
    connection_manager=conn_manager,
    transaction_manager=txn_manager
)

# Create test
test = TestCase(
    test_id='test_001',
    name='Test user creation',
    database_name='production',
    sql_query='SELECT COUNT(*) as count FROM users',
    assertions=[
        {'type': 'row_count', 'expected': 1}
    ],
    priority=TestPriority.HIGH
)

# Create suite
suite = TestSuite(
    suite_id='suite_001',
    name='User Tests',
    tests=[test],
    parallel_execution=True
)

# Register and run
engine.register_suite(suite)
results = engine.run_suite('suite_001')

# Check results
for test_id, result in results.items():
    print(f"{result.test_name}: {result.status}")
```

## Reporting

SQLTest Pro provides comprehensive reporting capabilities.

### Report Types

#### HTML Reports

```bash
# Generate interactive HTML report
sqltest report --type html --output report.html

# Executive dashboard
sqltest report --type executive --output dashboard.html
```

#### JSON Reports

```bash
# Machine-readable JSON
sqltest report --type json --output report.json
```

#### JUnit XML (CI/CD)

```bash
# For CI/CD integration
sqltest test --junit junit.xml
```

### Automated Report Scheduling

```yaml
reporting:
  schedules:
    - name: "Daily Quality Report"
      frequency: daily
      time: "09:00"
      report_type: executive

      email:
        enabled: true
        recipients:
          - team@example.com
          - manager@example.com
        subject: "Daily Data Quality Report"
        smtp:
          host: smtp.gmail.com
          port: 587
          username: reports@example.com
          password: ${SMTP_PASSWORD}

      file:
        enabled: true
        output_dir: ./reports/
        format: html
```

Start scheduler:

```bash
sqltest scheduler --config reporting.yaml
```

## CI/CD Integration

SQLTest Pro integrates seamlessly with major CI/CD platforms.

### GitHub Actions

```yaml
name: SQL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install SQLTest Pro
        run: |
          pip install -e .

      - name: Run SQL Tests
        env:
          DB_PASSWORD: postgres
        run: |
          sqltest test --config unit_tests.yaml --junit junit.xml

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action@v1
        if: always()
        with:
          files: junit.xml
```

### GitLab CI

```yaml
stages:
  - test

sql_tests:
  stage: test
  image: python:3.9

  services:
    - postgres:14

  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DB_PASSWORD: postgres

  before_script:
    - pip install -e .

  script:
    - sqltest test --config unit_tests.yaml --junit junit.xml

  artifacts:
    reports:
      junit: junit.xml
```

### Jenkins

```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'pip install -e .'
            }
        }

        stage('SQL Tests') {
            steps {
                sh 'sqltest test --config unit_tests.yaml --junit junit.xml'
            }
        }
    }

    post {
        always {
            junit 'junit.xml'
        }
    }
}
```

## Advanced Features

### Query Caching

Enable intelligent query result caching:

```yaml
databases:
  production:
    # ... connection config ...

    caching:
      enabled: true
      max_size_mb: 100
      default_ttl_seconds: 300
      strategy: SMART  # TTL, ADAPTIVE, FREQUENCY, SIZE_BASED, SMART
      eviction_policy: SMART  # LRU, LFU, FIFO, SIZE, TTL, SMART
```

### Transaction Management

Use distributed transactions:

```python
from sqltest.db.transaction_manager import TransactionManager

txn_manager = TransactionManager(conn_manager)

# Begin transaction across multiple databases
txn_id = txn_manager.begin_transaction(
    databases=['production', 'analytics']
)

try:
    # Execute operations
    txn_manager.execute_operation(
        txn_id,
        'production',
        'UPDATE users SET last_login = NOW() WHERE id = 1'
    )

    txn_manager.execute_operation(
        txn_id,
        'analytics',
        'INSERT INTO login_events (user_id, timestamp) VALUES (1, NOW())'
    )

    # Two-phase commit
    success = txn_manager.two_phase_commit(txn_id)

except Exception as e:
    # Automatic rollback
    txn_manager.abort(txn_id, str(e))
```

### Change Data Capture (CDC)

Monitor database changes in real-time:

```python
from sqltest.db.data_operations import CDCEngine

cdc = CDCEngine(conn_manager)

# Define CDC configuration
cdc_config = CDCConfiguration(
    source_database='production',
    tables=['users', 'orders'],
    capture_inserts=True,
    capture_updates=True,
    capture_deletes=True
)

# Subscribe to changes
def handle_change(event: CDCEvent):
    print(f"Change detected: {event.operation} on {event.table_name}")
    print(f"Data: {event.data}")

cdc.subscribe(cdc_config, handle_change)

# Start capturing
cdc.start_capture(cdc_config)
```

### Schema Introspection

Automatically discover and analyze database schemas:

```python
from sqltest.db.schema_introspector import SchemaIntrospector

introspector = SchemaIntrospector(conn_manager)

# Introspect database
schema = introspector.introspect_database(
    engine=conn_manager.get_engine('production'),
    database_name='production'
)

# Analyze schema quality
quality_report = introspector.analyze_schema_quality(schema)

print(f"Schema quality score: {quality_report['overall_score']:.2f}%")
print(f"Tables with primary keys: {quality_report['tables_with_primary_keys']}")
print(f"Quality issues: {len(quality_report['quality_issues'])}")
```

## Troubleshooting

### Common Issues

#### Connection Errors

```
Error: Could not connect to database
```

**Solution**: Check your database configuration and credentials:
- Verify host, port, database name
- Check username and password
- Ensure database is running and accessible
- Test connection: `sqltest connection --test`

#### Permission Errors

```
Error: Permission denied for table 'users'
```

**Solution**: Ensure your database user has appropriate permissions:
```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_user;
```

#### Test Isolation Issues

```
Error: Test affected database state
```

**Solution**: Tests should use transactions that are automatically rolled back. Check:
- Test uses transaction manager correctly
- No DDL statements that can't be rolled back
- No explicit COMMIT statements in test code

### Getting Help

- **Documentation**: [https://github.com/Schaafd/sql-testing-suite](https://github.com/Schaafd/sql-testing-suite)
- **Issues**: [GitHub Issues](https://github.com/Schaafd/sql-testing-suite/issues)
- **Examples**: See `examples/` directory
- **Demos**: Run `python demo_comprehensive.py`

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed API documentation
- Check out the [Testing Guide](TESTING_GUIDE.md) for best practices
- Explore [Configuration Guide](CONFIGURATION.md) for advanced configuration options
- See [Database Layer](DATABASE_LAYER.md) for performance optimization techniques

---

**SQLTest Pro** - Enterprise SQL Testing & Validation Framework