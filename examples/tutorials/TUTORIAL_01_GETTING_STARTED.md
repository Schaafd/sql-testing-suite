# Tutorial 1: Getting Started with SQLTest Pro

Learn how to set up and run your first SQL tests with SQLTest Pro.

## Prerequisites

- Python 3.8 or higher installed
- A database to test (PostgreSQL, MySQL, SQLite, etc.)
- Basic knowledge of SQL

## Step 1: Install SQLTest Pro

```bash
# Clone the repository
git clone https://github.com/Schaafd/sql-testing-suite.git
cd sql-testing-suite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install SQLTest Pro
pip install -e .

# Verify installation
sqltest --version
```

## Step 2: Set Up Your Database Configuration

Create a file named `database.yaml`:

```yaml
databases:
  local:
    type: sqlite
    database: ./tutorial.db
```

For PostgreSQL:

```yaml
databases:
  local:
    type: postgresql
    host: localhost
    port: 5432
    database: tutorial_db
    username: postgres
    password: ${DB_PASSWORD}
```

Set environment variable:

```bash
export DB_PASSWORD='your_password'
```

## Step 3: Create Sample Database (SQLite Example)

```bash
# Create SQLite database
sqlite3 tutorial.db << EOF
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    age INTEGER,
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (username, email, age) VALUES
    ('alice', 'alice@example.com', 30),
    ('bob', 'bob@example.com', 25),
    ('charlie', 'charlie@example.com', 35);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

INSERT INTO products (name, price, stock_quantity) VALUES
    ('Widget', 19.99, 100),
    ('Gadget', 29.99, 50),
    ('Doohickey', 9.99, 200);
EOF
```

## Step 4: Write Your First Test

Create `my_first_test.yaml`:

```yaml
unit_tests:
  - test_suite: "My First Tests"
    database: local

    tests:
      # Test 1: Database connection
      - name: "Test: Database is accessible"
        test:
          query: "SELECT 1 as result"
          assertions:
            - type: row_count
              expected: 1
            - type: equals
              column: result
              expected: 1

      # Test 2: Check user count
      - name: "Test: Users table has 3 users"
        test:
          query: "SELECT COUNT(*) as count FROM users"
          assertions:
            - type: equals
              column: count
              expected: 3

      # Test 3: Verify user data
      - name: "Test: All users have valid emails"
        test:
          query: |
            SELECT COUNT(*) as invalid_count
            FROM users
            WHERE email NOT LIKE '%@%.%'
          assertions:
            - type: equals
              column: invalid_count
              expected: 0
              message: "All emails should be valid"
```

## Step 5: Run Your Tests

```bash
# Run all tests
sqltest test --config my_first_test.yaml

# Run with verbose output
sqltest test --config my_first_test.yaml --verbose

# Generate HTML report
sqltest test --config my_first_test.yaml --output report.html
```

You should see output like:

```
Running test suite: My First Tests
✓ Test: Database is accessible (2ms)
✓ Test: Users table has 3 users (5ms)
✓ Test: All users have valid emails (3ms)

Test Summary: 3/3 passed, 0/3 failed
```

## Step 6: Add Field Validations

Create `validations.yaml`:

```yaml
field_validations:
  - table: users
    validations:
      # Email validation
      - column: email
        rules:
          - type: not_null
            severity: ERROR
          - type: unique
            severity: ERROR

      # Username validation
      - column: username
        rules:
          - type: not_null
            severity: ERROR
          - type: unique
            severity: ERROR
          - type: length
            min: 3
            max: 50
            severity: ERROR

      # Age validation
      - column: age
        rules:
          - type: range
            min: 18
            max: 120
            severity: WARNING

  - table: products
    validations:
      # Price validation
      - column: price
        rules:
          - type: not_null
            severity: ERROR
          - type: range
            min: 0.01
            max: 999999.99
            severity: ERROR

      # Stock validation
      - column: stock_quantity
        rules:
          - type: range
            min: 0
            severity: ERROR
```

Run validations:

```bash
sqltest validate --config validations.yaml
```

## Step 7: Profile Your Data

```bash
# Profile users table
sqltest profile --table users --database local

# Generate HTML profile report
sqltest profile --table users --output user_profile.html
```

## Step 8: Add More Advanced Tests

Update `my_first_test.yaml` to include more tests:

```yaml
unit_tests:
  - test_suite: "Advanced Tests"
    database: local

    tests:
      # Test with setup and teardown
      - name: "Test: User creation works"
        setup:
          - "DELETE FROM users WHERE username = 'testuser'"

        test:
          query: |
            INSERT INTO users (username, email, age)
            VALUES ('testuser', 'test@example.com', 25)
            RETURNING user_id, username

          assertions:
            - type: row_count
              expected: 1
            - type: not_null
              column: user_id

        teardown:
          - "DELETE FROM users WHERE username = 'testuser'"

      # Performance test
      - name: "Test: User lookup is fast"
        test:
          query: "SELECT * FROM users WHERE user_id = 1"
          assertions:
            - type: execution_time
              expected: 50  # milliseconds

      # Data integrity test
      - name: "Test: All products have positive prices"
        test:
          query: |
            SELECT COUNT(*) as invalid_count
            FROM products
            WHERE price <= 0
          assertions:
            - type: equals
              column: invalid_count
              expected: 0
```

## Step 9: View Test Results

After running tests, check the generated reports:

```bash
# Open HTML report
open report.html  # macOS
xdg-open report.html  # Linux
start report.html  # Windows
```

The report shows:
- Pass/fail status for each test
- Execution times
- Assertion details
- Error messages (if any)

## Next Steps

Congratulations! You've completed your first SQLTest Pro tutorial. Next, try:

1. **Tutorial 2**: Write more complex tests with fixtures
2. **Tutorial 3**: Set up CI/CD integration
3. **Tutorial 4**: Configure advanced connection pooling
4. **Tutorial 5**: Implement business rules validation

## Common Issues

### Issue: "Database connection failed"

**Solution**: Check your database configuration and ensure:
- Database is running
- Credentials are correct
- Network connectivity is available

### Issue: "No tests found"

**Solution**: Verify:
- YAML file syntax is correct
- File path is correct
- `unit_tests` key exists in YAML

### Issue: "Test fails but should pass"

**Solution**: Check:
- Test data is correct
- Assertions match expected values
- Database state is as expected

## Summary

You've learned how to:
- ✓ Install SQLTest Pro
- ✓ Configure database connections
- ✓ Write basic unit tests
- ✓ Run validations
- ✓ Profile data
- ✓ Generate reports

For more information:
- [User Guide](../../docs/USER_GUIDE.md)
- [API Reference](../../docs/API_REFERENCE.md)
- [Testing Guide](../../docs/TESTING_GUIDE.md)

---

**Next Tutorial**: [Tutorial 2: Advanced Testing Patterns](TUTORIAL_02_ADVANCED_PATTERNS.md)