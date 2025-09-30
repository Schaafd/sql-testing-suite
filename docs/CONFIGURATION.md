# SQLTest Pro Configuration Guide

Comprehensive guide to configuring SQLTest Pro for your environment.

## Table of Contents

- [Overview](#overview)
- [Database Configuration](#database-configuration)
- [Connection Pooling](#connection-pooling)
- [Query Caching](#query-caching)
- [Read/Write Splitting](#readwrite-splitting)
- [Transaction Management](#transaction-management)
- [Testing Configuration](#testing-configuration)
- [Validation Rules](#validation-rules)
- [Business Rules](#business-rules)
- [Reporting Configuration](#reporting-configuration)
- [Environment Variables](#environment-variables)

## Overview

SQLTest Pro uses YAML-based configuration files with Pydantic schema validation. Configuration supports:

- Environment variable substitution
- Multiple database connections
- Advanced connection pooling
- Query optimization settings
- Test execution parameters
- Reporting and scheduling

### Configuration File Structure

```yaml
# database.yaml
databases:
  production:
    type: postgresql
    host: localhost
    # ... connection settings ...
    pool_config:
      # ... pool settings ...

# validations.yaml
field_validations:
  - table: users
    validations:
      # ... validation rules ...

# unit_tests.yaml
unit_tests:
  - test_suite: "My Tests"
    tests:
      # ... test cases ...

# business_rules.yaml
business_rules:
  - name: "My Rule"
    # ... rule definition ...

# reporting.yaml
reporting:
  schedules:
    # ... report schedules ...
```

## Database Configuration

### Basic PostgreSQL Configuration

```yaml
databases:
  production:
    type: postgresql
    host: localhost
    port: 5432
    database: myapp_production
    username: app_user
    password: ${DB_PASSWORD}
```

### MySQL Configuration

```yaml
databases:
  mysql_db:
    type: mysql
    host: mysql.example.com
    port: 3306
    database: myapp
    username: mysql_user
    password: ${MYSQL_PASSWORD}

    # MySQL-specific options
    options:
      charset: utf8mb4
      ssl_ca: /path/to/ca-cert.pem
      ssl_cert: /path/to/client-cert.pem
      ssl_key: /path/to/client-key.pem
```

### SQL Server Configuration

```yaml
databases:
  sqlserver_db:
    type: mssql
    host: sqlserver.example.com
    port: 1433
    database: myapp
    username: sa
    password: ${MSSQL_PASSWORD}

    # SQL Server options
    options:
      driver: ODBC Driver 17 for SQL Server
      TrustServerCertificate: yes
```

### SQLite Configuration

```yaml
databases:
  local_db:
    type: sqlite
    database: /path/to/database.db

    # SQLite doesn't require host, port, username, password
```

### Snowflake Configuration

```yaml
databases:
  snowflake_db:
    type: snowflake
    account: xy12345
    warehouse: COMPUTE_WH
    database: ANALYTICS_DB
    schema: PUBLIC
    username: snowflake_user
    password: ${SNOWFLAKE_PASSWORD}

    # Snowflake-specific options
    options:
      role: ANALYST
      authenticator: snowflake
      region: us-east-1
```

### Multiple Databases

```yaml
databases:
  # Production database
  production:
    type: postgresql
    host: prod-db.example.com
    port: 5432
    database: prod_db
    username: prod_user
    password: ${PROD_DB_PASSWORD}

  # Staging database
  staging:
    type: postgresql
    host: staging-db.example.com
    port: 5432
    database: staging_db
    username: staging_user
    password: ${STAGING_DB_PASSWORD}

  # Analytics database
  analytics:
    type: snowflake
    account: xy12345
    warehouse: ANALYTICS_WH
    database: ANALYTICS_DB
    username: analyst
    password: ${ANALYTICS_PASSWORD}
```

## Connection Pooling

### Basic Pool Configuration

```yaml
databases:
  production:
    type: postgresql
    # ... connection details ...

    pool_config:
      # Basic settings
      min_connections: 2        # Minimum connections in pool
      max_connections: 20       # Maximum connections
      max_overflow: 10          # Extra connections beyond max
```

### Advanced Pool Configuration

```yaml
databases:
  production:
    type: postgresql
    # ... connection details ...

    pool_config:
      # Pool size
      min_connections: 5          # Minimum idle connections
      max_connections: 50         # Maximum total connections
      max_overflow: 10            # Allow 10 extra connections

      # Connection lifecycle
      pool_recycle: 3600          # Recycle after 1 hour (seconds)
      pool_pre_ping: true         # Test before using
      max_connection_age: 7200    # Max age: 2 hours (seconds)
      pool_timeout: 30            # Wait timeout (seconds)

      # Health monitoring
      health_check_interval: 60   # Check health every 60 seconds
      connection_probe_query: "SELECT 1"
      enable_connection_events: true

      # Performance
      pool_reset_on_return: true  # Reset connection on return
      connection_timeout: 30      # Connection establishment timeout
```

### Pool Size Recommendations

#### Small Applications (< 100 users)
```yaml
pool_config:
  min_connections: 2
  max_connections: 10
  max_overflow: 5
```

#### Medium Applications (100-1000 users)
```yaml
pool_config:
  min_connections: 5
  max_connections: 30
  max_overflow: 10
```

#### Large Applications (> 1000 users)
```yaml
pool_config:
  min_connections: 10
  max_connections: 100
  max_overflow: 20
```

#### High-Concurrency Applications
```yaml
pool_config:
  min_connections: 20
  max_connections: 200
  max_overflow: 50
  pool_pre_ping: true
  health_check_interval: 30
```

## Query Caching

### Enable Query Caching

```yaml
databases:
  production:
    # ... connection details ...

    caching:
      enabled: true
      max_size_mb: 100              # 100 MB cache
      default_ttl_seconds: 300      # 5 minutes
      strategy: SMART               # Caching strategy
      eviction_policy: SMART        # Eviction policy
```

### Caching Strategies

#### TTL Strategy (Time-To-Live)
Fixed expiration time for all cached entries.

```yaml
caching:
  strategy: TTL
  default_ttl_seconds: 300  # 5 minutes
```

#### ADAPTIVE Strategy
Adjusts TTL based on query complexity and execution time.

```yaml
caching:
  strategy: ADAPTIVE
  default_ttl_seconds: 300
  # Expensive queries get longer TTL automatically
```

#### FREQUENCY Strategy
Frequently accessed queries stay cached longer.

```yaml
caching:
  strategy: FREQUENCY
  default_ttl_seconds: 300
  # Popular queries get extended TTL
```

#### SIZE_BASED Strategy
TTL based on result set size (smaller results cached longer).

```yaml
caching:
  strategy: SIZE_BASED
  default_ttl_seconds: 300
  # Small results get longer TTL
```

#### SMART Strategy (Recommended)
Combines multiple strategies for optimal performance.

```yaml
caching:
  strategy: SMART
  default_ttl_seconds: 300
  # Intelligently combines all strategies
```

### Eviction Policies

#### LRU (Least Recently Used)
```yaml
caching:
  eviction_policy: LRU
  # Evicts least recently accessed entries
```

#### LFU (Least Frequently Used)
```yaml
caching:
  eviction_policy: LFU
  # Evicts least frequently accessed entries
```

#### FIFO (First In, First Out)
```yaml
caching:
  eviction_policy: FIFO
  # Evicts oldest entries first
```

#### SIZE (Largest First)
```yaml
caching:
  eviction_policy: SIZE
  # Evicts largest entries first
```

#### TTL (Shortest TTL First)
```yaml
caching:
  eviction_policy: TTL
  # Evicts entries closest to expiration
```

#### SMART (Recommended)
```yaml
caching:
  eviction_policy: SMART
  # Intelligently combines policies
```

### Cache Configuration Examples

#### High-Performance Application
```yaml
caching:
  enabled: true
  max_size_mb: 500          # Large cache
  default_ttl_seconds: 600  # 10 minutes
  strategy: SMART
  eviction_policy: LFU      # Keep frequently used
```

#### Memory-Constrained Environment
```yaml
caching:
  enabled: true
  max_size_mb: 50           # Small cache
  default_ttl_seconds: 120  # 2 minutes
  strategy: TTL
  eviction_policy: SIZE     # Evict large results
```

#### Read-Heavy Workload
```yaml
caching:
  enabled: true
  max_size_mb: 1000         # Very large cache
  default_ttl_seconds: 3600 # 1 hour
  strategy: FREQUENCY
  eviction_policy: LRU
```

## Read/Write Splitting

### Basic Read/Write Splitting

```yaml
databases:
  production:
    type: postgresql

    # Primary (write) database
    host: primary.db.example.com
    port: 5432
    database: production_db
    username: app_user
    password: ${DB_PASSWORD}

    # Read replicas
    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432
```

### Load Balancing Strategies

#### Round Robin (Default)
Distributes queries evenly across replicas.

```yaml
databases:
  production:
    # ... connection details ...
    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432

    routing_strategy: ROUND_ROBIN
```

#### Least Connections
Routes to replica with fewest active connections.

```yaml
databases:
  production:
    # ... connection details ...
    routing_strategy: LEAST_CONNECTIONS
```

#### Least Response Time (Recommended)
Routes to replica with fastest average response time.

```yaml
databases:
  production:
    # ... connection details ...
    routing_strategy: LEAST_RESPONSE_TIME
```

#### Weighted Load Balancing
Distribute based on replica weights (higher = more traffic).

```yaml
databases:
  production:
    # ... connection details ...
    replicas:
      - host: replica1.db.example.com
        port: 5432
        weight: 2  # Gets 2x traffic
      - host: replica2.db.example.com
        port: 5432
        weight: 1  # Gets 1x traffic

    routing_strategy: WEIGHTED
```

#### Random
Randomly select replica.

```yaml
databases:
  production:
    # ... connection details ...
    routing_strategy: RANDOM
```

### Failover Configuration

```yaml
databases:
  production:
    # ... connection details ...

    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432

    failover:
      enabled: true
      health_check_interval: 30     # Check every 30 seconds
      max_retries: 3                # Retry 3 times before failover
      retry_delay_seconds: 1        # Wait 1 second between retries
      auto_promote_replica: true    # Promote replica on primary failure
```

## Transaction Management

### Transaction Configuration

```yaml
transactions:
  # Default transaction timeout
  default_timeout_seconds: 300

  # Enable distributed transactions
  enable_distributed: true

  # Two-phase commit settings
  two_phase_commit:
    enabled: true
    prepare_timeout_seconds: 60
    commit_timeout_seconds: 60

  # Savepoint support
  enable_savepoints: true
  max_savepoints_per_transaction: 10

  # Transaction isolation level
  isolation_level: READ_COMMITTED  # or SERIALIZABLE, REPEATABLE_READ, READ_UNCOMMITTED
```

### Transaction Audit Logging

```yaml
transactions:
  audit_logging:
    enabled: true
    log_file: /var/log/sqltest/transactions.log
    log_level: INFO

    # What to log
    log_begin: true
    log_commit: true
    log_rollback: true
    log_savepoint: true
```

## Testing Configuration

### Test Suite Configuration

```yaml
testing:
  # Default settings
  default_timeout_seconds: 30
  default_parallel_execution: true
  default_max_workers: 4

  # Test isolation
  isolation:
    auto_rollback: true         # Always rollback after tests
    use_transactions: true       # Run tests in transactions
    cleanup_on_failure: true     # Cleanup even if test fails

  # Coverage tracking
  coverage:
    enabled: true
    track_tables: true
    track_queries: true
    min_coverage_percent: 80.0

  # Reporting
  reporting:
    generate_html: true
    generate_junit_xml: true
    generate_json: true
    output_directory: ./test_reports/
```

### Test Execution

```yaml
unit_tests:
  - test_suite: "User Management Tests"
    database: production

    # Execution settings
    parallel_execution: true
    max_workers: 8
    timeout_seconds: 60

    # Suite-level setup/teardown
    setup_suite:
      - "CREATE TEMP TABLE test_users AS SELECT * FROM users LIMIT 0"

    teardown_suite:
      - "DROP TABLE IF EXISTS test_users"

    # Tests
    tests:
      - name: "Test user creation"
        priority: HIGH  # or CRITICAL, MEDIUM, LOW
        timeout_seconds: 30
        tags:
          - integration
          - critical

        # Test-level setup/teardown
        setup:
          - "DELETE FROM test_users"

        teardown:
          - "DELETE FROM test_users WHERE username = 'testuser'"

        # Test query and assertions
        test:
          query: |
            INSERT INTO test_users (username, email)
            VALUES ('testuser', 'test@example.com')
            RETURNING *

          assertions:
            - type: row_count
              expected: 1
            - type: not_empty
            - type: execution_time
              expected: 100  # milliseconds
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
        active: true
      - username: "testuser2"
        email: "test2@example.com"
        active: true

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

## Validation Rules

### Field Validation Configuration

```yaml
field_validations:
  # Regex validation
  - table: users
    validations:
      - column: email
        rules:
          - type: regex
            pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            message: "Invalid email format"
            severity: ERROR

      - column: phone
        rules:
          - type: regex
            pattern: '^\+?1?\d{10,14}$'
            message: "Invalid phone number"
            severity: WARNING

  # Not null validation
  - table: orders
    validations:
      - column: customer_id
        rules:
          - type: not_null
            severity: ERROR

      - column: order_date
        rules:
          - type: not_null
            severity: ERROR

  # Unique validation
  - table: users
    validations:
      - column: email
        rules:
          - type: unique
            severity: ERROR

      - column: username
        rules:
          - type: unique
            severity: ERROR

  # Range validation
  - table: products
    validations:
      - column: price
        rules:
          - type: range
            min: 0.01
            max: 999999.99
            severity: ERROR

      - column: quantity
        rules:
          - type: range
            min: 0
            severity: ERROR

  # Enum validation
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
            severity: ERROR

  # Length validation
  - table: users
    validations:
      - column: username
        rules:
          - type: length
            min: 3
            max: 50
            severity: ERROR

  # Custom SQL validation
  - table: transactions
    validations:
      - column: amount
        rules:
          - type: custom
            sql: "amount > 0 AND amount <= account_balance"
            message: "Transaction amount invalid"
            severity: ERROR
```

### Validation Execution Settings

```yaml
validation:
  # Parallel execution
  parallel_execution: true
  max_workers: 4

  # Severity levels
  fail_on_error: true
  fail_on_warning: false

  # Reporting
  generate_report: true
  output_format: html  # or json, csv
  output_file: validation_report.html
```

## Business Rules

### Business Rules Configuration

```yaml
business_rules:
  # Data quality rule
  - name: "Check for orphaned orders"
    rule_type: DATA_QUALITY
    severity: ERROR
    scope: TABLE
    description: "Ensure all orders have valid customer references"

    sql_query: |
      SELECT COUNT(*) as violation_count
      FROM orders o
      LEFT JOIN customers c ON o.customer_id = c.customer_id
      WHERE c.customer_id IS NULL

    expected_result: 0

    # Retry configuration
    retry:
      enabled: true
      max_attempts: 3
      delay_seconds: 1
      backoff_multiplier: 2

  # Business logic rule
  - name: "Sales amount validation"
    rule_type: BUSINESS_LOGIC
    severity: WARNING
    scope: TABLE

    sql_query: |
      SELECT COUNT(*) as violation_count
      FROM sales
      WHERE amount <= 0 OR amount > 100000

    expected_result: 0

  # Referential integrity rule
  - name: "Customer references"
    rule_type: REFERENTIAL_INTEGRITY
    severity: ERROR
    scope: DATABASE

    sql_query: |
      SELECT
        'orders' as table_name,
        COUNT(*) as orphans
      FROM orders o
      WHERE NOT EXISTS (
        SELECT 1 FROM customers c
        WHERE c.id = o.customer_id
      )
      UNION ALL
      SELECT
        'invoices',
        COUNT(*)
      FROM invoices i
      WHERE NOT EXISTS (
        SELECT 1 FROM customers c
        WHERE c.id = i.customer_id
      )

  # Rule with dependencies
  - name: "Advanced validation"
    rule_type: BUSINESS_LOGIC
    severity: ERROR
    depends_on:
      - "Check for orphaned orders"
      - "Customer references"

    sql_query: |
      SELECT COUNT(*) as violations
      FROM orders
      WHERE status = 'invalid'
```

### Business Rule Execution

```yaml
business_rules_execution:
  # Parallel execution
  parallel_execution: true
  max_workers: 8

  # Caching
  enable_caching: true
  cache_ttl_seconds: 300

  # Retry logic
  enable_retry: true
  max_retries: 3
  retry_delay_seconds: 1

  # Reporting
  generate_report: true
  output_format: html
```

## Reporting Configuration

### Basic Reporting

```yaml
reporting:
  # Output formats
  formats:
    - html
    - json
    - csv

  # Output directory
  output_directory: ./reports/

  # Report types
  types:
    - technical    # Detailed technical report
    - executive    # Executive summary
    - coverage     # Test coverage report
```

### Automated Report Scheduling

```yaml
reporting:
  schedules:
    # Daily quality report
    - name: "Daily Quality Report"
      enabled: true
      frequency: daily
      time: "09:00"
      timezone: "America/New_York"
      report_type: executive

      # Email configuration
      email:
        enabled: true
        recipients:
          - team@example.com
          - manager@example.com
        subject: "Daily Data Quality Report - {{date}}"
        body_template: |
          Daily data quality report for {{date}}.

          Overall status: {{status}}

          See attached report for details.

        smtp:
          host: smtp.gmail.com
          port: 587
          use_tls: true
          username: reports@example.com
          password: ${SMTP_PASSWORD}

      # File output
      file:
        enabled: true
        output_dir: ./reports/daily/
        filename_template: "quality_report_{{date}}.html"
        format: html

    # Weekly executive summary
    - name: "Weekly Executive Summary"
      enabled: true
      frequency: weekly
      day: monday
      time: "08:00"
      report_type: executive

      email:
        enabled: true
        recipients:
          - executives@example.com
        subject: "Weekly Data Quality Summary"
        smtp:
          host: smtp.gmail.com
          port: 587
          use_tls: true
          username: reports@example.com
          password: ${SMTP_PASSWORD}
```

### Report Customization

```yaml
reporting:
  # Customize HTML reports
  html:
    template: custom_template.html
    theme: dark  # or light
    include_charts: true
    chart_library: chartjs  # or plotly

  # Customize JSON reports
  json:
    pretty_print: true
    include_metadata: true

  # Customize executive dashboards
  executive:
    include_trends: true
    trend_period_days: 30
    include_forecasts: true
    forecast_period_days: 7
```

## Environment Variables

### Using Environment Variables

SQLTest Pro supports environment variable substitution in configuration files:

```yaml
databases:
  production:
    password: ${DB_PASSWORD}              # Required
    host: ${DB_HOST:localhost}            # Default: localhost
    port: ${DB_PORT:5432}                 # Default: 5432
```

### Setting Environment Variables

#### Linux/macOS

```bash
# Set for current session
export DB_PASSWORD='secret_password'
export DB_HOST='db.example.com'
export DB_PORT='5432'

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export DB_PASSWORD="secret_password"' >> ~/.bashrc
```

#### Windows PowerShell

```powershell
# Set for current session
$env:DB_PASSWORD = 'secret_password'
$env:DB_HOST = 'db.example.com'

# Set permanently
[Environment]::SetEnvironmentVariable('DB_PASSWORD', 'secret_password', 'User')
```

#### .env File Support

Create a `.env` file in your project root:

```ini
# Database credentials
DB_PASSWORD=secret_password
DB_HOST=db.example.com
DB_PORT=5432

# SMTP credentials
SMTP_PASSWORD=email_password

# API keys
SNOWFLAKE_PASSWORD=snowflake_password
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()

# Now environment variables are available
```

### Best Practices

1. **Never commit passwords** - Use environment variables for sensitive data
2. **Use .gitignore** - Add `.env` to `.gitignore`
3. **Provide defaults** - Use `${VAR:default}` syntax for optional values
4. **Document variables** - List required environment variables in README

## Complete Configuration Example

```yaml
# complete_config.yaml
databases:
  production:
    type: postgresql
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    database: production_db
    username: app_user
    password: ${DB_PASSWORD}

    pool_config:
      min_connections: 5
      max_connections: 50
      max_overflow: 10
      pool_recycle: 3600
      pool_pre_ping: true
      health_check_interval: 60

    caching:
      enabled: true
      max_size_mb: 200
      default_ttl_seconds: 300
      strategy: SMART
      eviction_policy: SMART

    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432

    routing_strategy: LEAST_RESPONSE_TIME

    failover:
      enabled: true
      health_check_interval: 30
      auto_promote_replica: true

testing:
  default_timeout_seconds: 30
  default_parallel_execution: true
  default_max_workers: 8

  isolation:
    auto_rollback: true
    use_transactions: true

  coverage:
    enabled: true
    min_coverage_percent: 80.0

  reporting:
    generate_html: true
    generate_junit_xml: true
    output_directory: ./test_reports/

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
        smtp:
          host: smtp.gmail.com
          port: 587
          username: reports@example.com
          password: ${SMTP_PASSWORD}
```

---

For more information, see:
- [User Guide](USER_GUIDE.md) - Detailed usage instructions
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Testing Guide](TESTING_GUIDE.md) - Best practices for SQL testing

**SQLTest Pro** - Enterprise SQL Testing & Validation Framework