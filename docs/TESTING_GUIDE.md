# SQLTest Pro Testing Guide

Best practices and patterns for SQL unit testing with SQLTest Pro.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Test Isolation](#test-isolation)
- [Writing Effective Tests](#writing-effective-tests)
- [Test Organization](#test-organization)
- [Assertion Best Practices](#assertion-best-practices)
- [Fixtures and Test Data](#fixtures-and-test-data)
- [Performance Testing](#performance-testing)
- [CI/CD Integration](#cicd-integration)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Testing Philosophy

### Why Test SQL?

SQL code should be tested for the same reasons as application code:

1. **Correctness**: Verify queries return expected results
2. **Performance**: Ensure queries meet performance requirements
3. **Maintainability**: Catch regressions when refactoring
4. **Documentation**: Tests serve as executable documentation
5. **Confidence**: Deploy database changes with confidence

### Testing Principles

1. **Test Isolation**: Each test should be independent
2. **Repeatability**: Tests should produce consistent results
3. **Fast Execution**: Tests should run quickly
4. **Clear Failures**: Test failures should clearly indicate the problem
5. **Comprehensive Coverage**: Test all critical SQL code paths

## Test Structure

### Anatomy of a Good Test

```yaml
unit_tests:
  - test_suite: "User Management"
    database: production

    tests:
      - name: "Test: User creation returns new user ID"  # Clear, descriptive name
        priority: HIGH                                   # Indicate importance

        # Setup: Prepare test environment
        setup:
          - "DELETE FROM test_users WHERE username = 'testuser'"

        # Test: Execute the code being tested
        test:
          query: |
            INSERT INTO test_users (username, email, created_at)
            VALUES ('testuser', 'test@example.com', NOW())
            RETURNING user_id, username

          # Assertions: Verify expected behavior
          assertions:
            - type: row_count
              expected: 1
              message: "Should return exactly one row"

            - type: not_empty
              message: "Should return the new user data"

            - type: column_exists
              column: user_id
              message: "Should include user_id in response"

        # Teardown: Clean up test data
        teardown:
          - "DELETE FROM test_users WHERE username = 'testuser'"
```

### Test Naming Convention

Use descriptive names that explain what is being tested:

**Good Names:**
- `Test: Valid email format is accepted`
- `Test: Duplicate emails are rejected`
- `Test: User creation updates audit log`

**Poor Names:**
- `Test 1`
- `Email test`
- `Check users`

### Test Organization Pattern

```yaml
unit_tests:
  # Group 1: Happy path tests
  - test_suite: "User Creation - Happy Path"
    tests:
      - name: "Test: Create user with valid data"
      - name: "Test: Create user with minimal required fields"
      - name: "Test: Create user with all optional fields"

  # Group 2: Error handling tests
  - test_suite: "User Creation - Error Handling"
    tests:
      - name: "Test: Reject user with invalid email"
      - name: "Test: Reject user with duplicate username"
      - name: "Test: Reject user with missing required fields"

  # Group 3: Edge cases
  - test_suite: "User Creation - Edge Cases"
    tests:
      - name: "Test: Handle very long usernames"
      - name: "Test: Handle special characters in username"
      - name: "Test: Handle concurrent user creation"
```

## Test Isolation

### Why Isolation Matters

Test isolation ensures:
- Tests don't interfere with each other
- Tests can run in any order
- Tests can run in parallel
- Database state is predictable

### Automatic Transaction Rollback

SQLTest Pro automatically wraps each test in a transaction and rolls it back:

```python
# Automatic isolation (handled by TestExecutionEngine)
def run_test(test):
    txn_id = begin_transaction()
    try:
        run_setup(test.setup)
        execute_query(test.query)
        run_assertions(test.assertions)
        run_teardown(test.teardown)
    finally:
        rollback(txn_id)  # Always rollback, even on success
```

### Manual Isolation Patterns

#### Pattern 1: Use Unique Test Data

```yaml
tests:
  - name: "Test: User creation"
    setup:
      - "DELETE FROM users WHERE username LIKE 'test_%'"

    test:
      query: |
        INSERT INTO users (username, email)
        VALUES ('test_user_{{timestamp}}', 'test_{{timestamp}}@example.com')

    teardown:
      - "DELETE FROM users WHERE username LIKE 'test_%'"
```

#### Pattern 2: Use Temporary Tables

```yaml
tests:
  - name: "Test: Data transformation"
    setup:
      - "CREATE TEMP TABLE test_data AS SELECT * FROM users LIMIT 0"
      - "INSERT INTO test_data SELECT * FROM users WHERE id IN (1, 2, 3)"

    test:
      query: "SELECT * FROM test_data WHERE active = true"

    teardown:
      - "DROP TABLE IF EXISTS test_data"
```

#### Pattern 3: Use Test Schemas

```yaml
tests:
  - name: "Test: Schema operations"
    setup:
      - "CREATE SCHEMA IF NOT EXISTS test_schema"
      - "CREATE TABLE test_schema.users AS SELECT * FROM users LIMIT 0"

    test:
      query: "SELECT * FROM test_schema.users"

    teardown:
      - "DROP SCHEMA IF EXISTS test_schema CASCADE"
```

## Writing Effective Tests

### Test One Thing at a Time

**Good: Single Responsibility**
```yaml
tests:
  - name: "Test: User email validation"
    test:
      query: "SELECT validate_email('test@example.com') as valid"
      assertions:
        - type: equals
          column: valid
          expected: true

  - name: "Test: User phone validation"
    test:
      query: "SELECT validate_phone('+1234567890') as valid"
      assertions:
        - type: equals
          column: valid
          expected: true
```

**Poor: Multiple Responsibilities**
```yaml
tests:
  - name: "Test: Validation functions"
    test:
      query: |
        SELECT
          validate_email('test@example.com') as email_valid,
          validate_phone('+1234567890') as phone_valid
      assertions:
        - type: equals
          column: email_valid
          expected: true
        - type: equals
          column: phone_valid
          expected: true
```

### Use Meaningful Assertions

**Good: Specific Assertions**
```yaml
tests:
  - name: "Test: Calculate order total"
    test:
      query: "SELECT calculate_order_total(123) as total"
      assertions:
        - type: equals
          column: total
          expected: 150.00
          message: "Order 123 total should be $150.00"

        - type: greater_than
          column: total
          expected: 0
          message: "Order total must be positive"
```

**Poor: Vague Assertions**
```yaml
tests:
  - name: "Test: Calculate order total"
    test:
      query: "SELECT calculate_order_total(123) as total"
      assertions:
        - type: not_empty  # Too vague
```

### Test Edge Cases

```yaml
tests:
  # Normal case
  - name: "Test: Standard order calculation"
    test:
      query: "SELECT calculate_order_total(123) as total"
      assertions:
        - type: equals
          column: total
          expected: 150.00

  # Edge case: Empty order
  - name: "Test: Empty order returns zero"
    test:
      query: "SELECT calculate_order_total(999) as total"  # Non-existent order
      assertions:
        - type: equals
          column: total
          expected: 0.00

  # Edge case: Large order
  - name: "Test: Large order calculation"
    test:
      query: "SELECT calculate_order_total(456) as total"
      assertions:
        - type: greater_than
          column: total
          expected: 10000.00

  # Edge case: Negative values
  - name: "Test: Refunds handled correctly"
    test:
      query: "SELECT calculate_order_total(789) as total"
      assertions:
        - type: less_than
          column: total
          expected: 0.00
```

### Test Error Conditions

```yaml
tests:
  # Test valid input
  - name: "Test: Valid input accepted"
    test:
      query: "SELECT create_user('valid@example.com') as success"
      assertions:
        - type: equals
          column: success
          expected: true

  # Test invalid input
  - name: "Test: Invalid email rejected"
    test:
      query: |
        DO $$
        BEGIN
          PERFORM create_user('invalid-email');
          RAISE EXCEPTION 'Should have raised error';
        EXCEPTION
          WHEN check_violation THEN
            -- Expected error
            NULL;
        END $$;
      assertions:
        - type: no_error

  # Test duplicate input
  - name: "Test: Duplicate email rejected"
    setup:
      - "INSERT INTO users (email) VALUES ('existing@example.com')"

    test:
      query: |
        SELECT create_user('existing@example.com') as success
      assertions:
        - type: equals
          column: success
          expected: false
```

## Test Organization

### Organize by Feature

```
unit_tests/
├── authentication/
│   ├── login_tests.yaml
│   ├── logout_tests.yaml
│   └── password_reset_tests.yaml
├── user_management/
│   ├── user_creation_tests.yaml
│   ├── user_update_tests.yaml
│   └── user_deletion_tests.yaml
└── orders/
    ├── order_creation_tests.yaml
    ├── order_fulfillment_tests.yaml
    └── order_cancellation_tests.yaml
```

### Use Test Tags

```yaml
tests:
  - name: "Test: User login"
    tags:
      - authentication
      - critical
      - smoke

  - name: "Test: Password strength"
    tags:
      - authentication
      - security
      - integration

  - name: "Test: Profile update"
    tags:
      - user_management
      - integration
```

Run tests by tag:

```bash
# Run critical tests only
sqltest test --tags critical

# Run smoke tests
sqltest test --tags smoke

# Run integration tests
sqltest test --tags integration
```

### Test Priority Levels

```yaml
tests:
  - name: "Test: User authentication"
    priority: CRITICAL  # Must pass before deployment

  - name: "Test: Order creation"
    priority: HIGH      # Important functionality

  - name: "Test: Email notification"
    priority: MEDIUM    # Standard priority

  - name: "Test: UI preferences"
    priority: LOW       # Nice to have
```

## Assertion Best Practices

### Use Specific Assertions

```yaml
# Good: Specific assertion
assertions:
  - type: equals
    column: status
    expected: "active"

# Poor: Generic assertion
assertions:
  - type: not_empty  # Doesn't verify actual value
```

### Chain Assertions

```yaml
# Test multiple conditions
assertions:
  - type: row_count
    expected: 1
    message: "Should return exactly one user"

  - type: column_exists
    columns: [user_id, username, email]
    message: "Should include all required columns"

  - type: no_nulls
    columns: [user_id, username]
    message: "Required fields should not be null"

  - type: execution_time
    expected: 100
    message: "Query should execute in under 100ms"
```

### Use Fluent Assertions (Python)

```python
from sqltest.modules.testing.assertions import assert_that

# Readable assertion chains
assert_that(user_count).is_greater_than(0).is_less_than(1000)
assert_that(email).contains('@').has_length(20)
assert_that(status).equals('active').is_not_none()
```

## Fixtures and Test Data

### Creating Reusable Fixtures

```yaml
fixtures:
  # Base user fixture
  - name: "base_users"
    table: users
    cleanup: true
    data:
      - username: "testuser1"
        email: "test1@example.com"
        active: true
      - username: "testuser2"
        email: "test2@example.com"
        active: false

  # Order fixture (depends on users)
  - name: "base_orders"
    table: orders
    cleanup: true
    depends_on:
      - base_users
    data:
      - user_id: 1
        amount: 100.00
        status: "pending"
      - user_id: 2
        amount: 250.00
        status: "completed"

# Use fixtures in tests
unit_tests:
  - test_suite: "Order Tests"
    fixtures:
      - base_users
      - base_orders

    tests:
      - name: "Test: Order count"
        test:
          query: "SELECT COUNT(*) as count FROM orders"
          assertions:
            - type: equals
              column: count
              expected: 2
```

### Generate Test Data

```python
from sqltest.modules.testing.fixtures import MockDataGenerator

# Generate test data programmatically
generator = MockDataGenerator()

# 100 test users
emails = generator.generate_emails(100, domain="test.com")
usernames = generator.generate_strings(100, length=10, prefix="user_")
ages = generator.generate_integers(100, min_val=18, max_val=80)

# Create test data
test_data = pd.DataFrame({
    'username': usernames,
    'email': emails,
    'age': ages
})
```

## Performance Testing

### Test Query Performance

```yaml
tests:
  - name: "Test: User search performance"
    test:
      query: |
        SELECT * FROM users
        WHERE username LIKE 'test%'
        AND active = true
        LIMIT 100

      assertions:
        - type: execution_time
          expected: 50  # milliseconds
          message: "Query must execute in under 50ms"

        - type: row_count
          expected: 100
          message: "Should return up to 100 users"
```

### Load Testing

```yaml
tests:
  - name: "Test: Concurrent user creation"
    priority: HIGH
    parallel_execution: true
    max_workers: 10

    test:
      query: |
        INSERT INTO users (username, email, created_at)
        VALUES (
          'user_' || gen_random_uuid()::text,
          gen_random_uuid()::text || '@test.com',
          NOW()
        )
        RETURNING user_id

      assertions:
        - type: row_count
          expected: 1

        - type: execution_time
          expected: 1000  # Should handle under load
```

### Index Effectiveness Testing

```yaml
tests:
  - name: "Test: Index on email improves performance"
    setup:
      - "DROP INDEX IF EXISTS idx_users_email"

    test:
      query: "SELECT * FROM users WHERE email = 'test@example.com'"
      assertions:
        - type: execution_time
          expected: 1000  # Baseline without index

  - name: "Test: Query with index"
    setup:
      - "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"

    test:
      query: "SELECT * FROM users WHERE email = 'test@example.com'"
      assertions:
        - type: execution_time
          expected: 10  # Should be much faster
```

## CI/CD Integration

### GitHub Actions Example

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

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Run smoke tests
        run: |
          sqltest test --tags smoke --junit smoke.xml

      - name: Run integration tests
        run: |
          sqltest test --tags integration --junit integration.xml

      - name: Run all tests
        run: |
          sqltest test --junit all_tests.xml

      - name: Generate coverage report
        run: |
          sqltest test --coverage --output coverage.html

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v1
        if: always()
        with:
          files: |
            smoke.xml
            integration.xml
            all_tests.xml

      - name: Upload coverage report
        uses: actions/upload-artifact@v2
        with:
          name: coverage-report
          path: coverage.html
```

### Test Stages

```yaml
# 1. Smoke tests (fast, critical)
smoke_tests:
  - name: "Database connection"
    priority: CRITICAL
    test:
      query: "SELECT 1"
      assertions:
        - type: row_count
          expected: 1

# 2. Unit tests (isolated, fast)
unit_tests:
  - name: "User validation"
    priority: HIGH
    test:
      query: "SELECT validate_user_data(...)"

# 3. Integration tests (cross-table, slower)
integration_tests:
  - name: "Order fulfillment workflow"
    priority: MEDIUM
    test:
      query: "SELECT process_order(...)"

# 4. E2E tests (full workflow, slowest)
e2e_tests:
  - name: "Complete user journey"
    priority: LOW
    test:
      query: "CALL complete_user_flow(...)"
```

## Common Patterns

### Pattern: Test Data Builder

```yaml
fixtures:
  - name: "user_builder"
    table: users
    builder:
      base:
        active: true
        created_at: NOW()

      variants:
        - name: "admin_user"
          role: "admin"
          permissions: ["all"]

        - name: "regular_user"
          role: "user"
          permissions: ["read"]

        - name: "guest_user"
          role: "guest"
          permissions: []
```

### Pattern: Parameterized Tests

```yaml
tests:
  - name: "Test: Email validation"
    parameterized: true
    parameters:
      - email: "valid@example.com"
        expected: true
      - email: "invalid.email"
        expected: false
      - email: "missing@domain"
        expected: false

    test:
      query: "SELECT validate_email('{{email}}') as valid"
      assertions:
        - type: equals
          column: valid
          expected: "{{expected}}"
```

### Pattern: Golden Master Testing

```yaml
tests:
  - name: "Test: Report generation matches baseline"
    test:
      query: "SELECT * FROM generate_monthly_report('2024-01')"
      assertions:
        - type: matches_file
          file: "tests/golden/monthly_report_2024_01.json"
          message: "Report should match golden master"
```

## Troubleshooting

### Common Issues

#### Test Fails Intermittently

**Problem**: Test passes sometimes, fails other times.

**Solutions**:
1. Check for timing issues (use transactions, not delays)
2. Ensure test isolation (no shared state)
3. Verify fixture cleanup
4. Check for database triggers or background jobs

```yaml
# Bad: Timing-dependent test
setup:
  - "INSERT INTO queue (job) VALUES ('process')"
  - "WAIT FOR DELAY '00:00:05'"  # Timing issue

# Good: State-based test
setup:
  - "INSERT INTO queue (job) VALUES ('process')"
test:
  query: "SELECT * FROM queue WHERE status = 'completed'"
```

#### Tests Slow in CI/CD

**Problem**: Tests run fast locally but slow in CI/CD.

**Solutions**:
1. Use connection pooling
2. Reduce test data size
3. Use parallel execution
4. Optimize database configuration

```yaml
# Enable parallel execution
testing:
  parallel_execution: true
  max_workers: 8

# Optimize pool
pool_config:
  min_connections: 5
  max_connections: 20
```

#### Tests Affect Each Other

**Problem**: Tests pass individually but fail when run together.

**Solutions**:
1. Verify transaction rollback
2. Use unique test data
3. Clean up in teardown
4. Check for shared fixtures

```yaml
# Ensure cleanup
teardown:
  - "DELETE FROM users WHERE username LIKE 'test_%'"
  - "DELETE FROM orders WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'test_%')"
```

### Debugging Tests

```yaml
# Enable verbose logging
testing:
  logging:
    level: DEBUG
    log_queries: true
    log_results: true

# Add diagnostic assertions
tests:
  - name: "Debug: Check test data"
    test:
      query: "SELECT COUNT(*) as count, MAX(created_at) as latest FROM users"
      assertions:
        - type: debug_output  # Print results
```

---

## Summary

**Key Takeaways:**

1. **Isolate Tests**: Use transactions and unique data
2. **Test One Thing**: Each test should have a single purpose
3. **Use Good Names**: Descriptive test names aid understanding
4. **Assert Specifically**: Precise assertions catch more bugs
5. **Organize Well**: Structure tests by feature and priority
6. **Performance Matters**: Test query performance requirements
7. **CI/CD Integration**: Automate test execution
8. **Handle Errors**: Test both success and failure cases

For more information, see:
- [User Guide](USER_GUIDE.md) - Detailed usage instructions
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](CONFIGURATION.md) - Configuration options

**SQLTest Pro** - Enterprise SQL Testing & Validation Framework