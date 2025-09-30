# SQLTest Pro Examples and Tutorials

Complete examples and step-by-step tutorials for SQLTest Pro.

## 📁 Directory Structure

```
examples/
├── getting_started/           # Simple examples for beginners
│   ├── simple_database_config.yaml
│   ├── simple_tests.yaml
│   └── simple_validations.yaml
│
├── advanced/                  # Enterprise-grade examples
│   ├── enterprise_config.yaml
│   ├── comprehensive_tests.yaml
│   └── business_rules.yaml
│
└── tutorials/                 # Step-by-step tutorials
    ├── TUTORIAL_01_GETTING_STARTED.md
    ├── TUTORIAL_02_ADVANCED_PATTERNS.md
    ├── TUTORIAL_03_CICD_INTEGRATION.md
    ├── TUTORIAL_04_PERFORMANCE_OPTIMIZATION.md
    └── TUTORIAL_05_BUSINESS_RULES.md
```

## 🚀 Quick Start

### 1. Getting Started Examples

Perfect for first-time users:

```bash
# Navigate to getting started examples
cd examples/getting_started/

# Run simple tests
sqltest test --config simple_tests.yaml

# Run validations
sqltest validate --config simple_validations.yaml
```

**What you'll learn:**
- Basic database configuration
- Simple unit tests
- Field validation rules

### 2. Advanced Examples

Enterprise-grade configurations:

```bash
# Navigate to advanced examples
cd examples/advanced/

# Run comprehensive test suite
sqltest test --config comprehensive_tests.yaml

# Use enterprise configuration
sqltest test --config enterprise_config.yaml --suite "Performance Tests"
```

**What you'll learn:**
- Connection pooling optimization
- Query caching strategies
- Read/write splitting
- Transaction management
- Automated reporting

### 3. Tutorials

Step-by-step learning:

```bash
cd examples/tutorials/

# Follow Tutorial 1
cat TUTORIAL_01_GETTING_STARTED.md
```

## 📚 Tutorials

### [Tutorial 1: Getting Started](tutorials/TUTORIAL_01_GETTING_STARTED.md)
**Duration**: 15 minutes
**Level**: Beginner

Learn the basics:
- Installing SQLTest Pro
- Configuring your first database
- Writing and running simple tests
- Field validation basics

### Tutorial 2: Advanced Testing Patterns
**Duration**: 30 minutes
**Level**: Intermediate

Master advanced patterns:
- Test fixtures and test data
- Parameterized tests
- Test organization and tagging
- Parallel test execution

### Tutorial 3: CI/CD Integration
**Duration**: 20 minutes
**Level**: Intermediate

Automate your testing:
- GitHub Actions setup
- GitLab CI configuration
- Jenkins pipeline integration
- Test result reporting

### Tutorial 4: Performance Optimization
**Duration**: 30 minutes
**Level**: Advanced

Optimize performance:
- Connection pool tuning
- Query caching strategies
- Read/write splitting
- Load balancing

### Tutorial 5: Business Rules Validation
**Duration**: 25 minutes
**Level**: Intermediate

Implement business rules:
- Data quality rules
- Referential integrity checks
- Custom validation logic
- Rule dependencies

## 📋 Example Files

### Getting Started

#### `simple_database_config.yaml`
Basic database configuration for SQLite and PostgreSQL.

```yaml
databases:
  local:
    type: sqlite
    database: ./test_database.db
```

#### `simple_tests.yaml`
Basic unit tests covering:
- Connection testing
- Row count assertions
- Data validation
- Performance checks

#### `simple_validations.yaml`
Field-level validation rules:
- Email format validation
- Username uniqueness
- Range validation for numbers

### Advanced

#### `enterprise_config.yaml`
Production-ready configuration with:
- Advanced connection pooling (100+ connections)
- Query result caching (500MB cache)
- Read replica configuration
- Automatic failover
- Transaction management
- Automated report scheduling

#### `comprehensive_tests.yaml`
Enterprise test suite featuring:
- Multiple test suites with priorities
- Setup/teardown at suite and test level
- Performance benchmarks
- Data integrity tests
- Transaction testing
- Test fixtures with dependencies

## 🎯 Use Cases

### Use Case 1: E-Commerce Platform

**Scenario**: Testing an e-commerce database

**Files to use:**
- `examples/advanced/enterprise_config.yaml` - For production database
- `examples/advanced/comprehensive_tests.yaml` - Adapt for order/product testing

**Key tests:**
- Order integrity (orders match line items)
- Price validation (positive prices)
- Stock quantity checks
- User authentication
- Payment processing

### Use Case 2: Financial Application

**Scenario**: Testing a banking database

**Focus areas:**
- Transaction integrity
- Balance calculations
- Audit trail validation
- Regulatory compliance

**Recommended patterns:**
- Strict isolation (always rollback)
- Savepoint usage for complex transactions
- Two-phase commit for distributed transactions
- Comprehensive audit logging

### Use Case 3: Analytics Platform

**Scenario**: Testing data warehouse queries

**Configuration:**
- Snowflake or Redshift connection
- Large query result caching
- Read-heavy optimization

**Key tests:**
- Query performance benchmarks
- Data aggregation accuracy
- Report generation validation

## 🔧 Configuration Examples

### Minimal Configuration

```yaml
databases:
  local:
    type: sqlite
    database: ./test.db
```

### Production Configuration

```yaml
databases:
  production:
    type: postgresql
    host: db.example.com
    port: 5432
    database: prod_db
    username: app_user
    password: ${DB_PASSWORD}

    pool_config:
      min_connections: 10
      max_connections: 100
      pool_pre_ping: true

    caching:
      enabled: true
      max_size_mb: 500
      strategy: SMART

    replicas:
      - host: replica1.db.example.com
        port: 5432
      - host: replica2.db.example.com
        port: 5432

    routing_strategy: LEAST_RESPONSE_TIME
```

## 🧪 Test Examples

### Basic Test

```yaml
tests:
  - name: "Test: Database connection"
    test:
      query: "SELECT 1 as result"
      assertions:
        - type: equals
          column: result
          expected: 1
```

### Advanced Test with Setup/Teardown

```yaml
tests:
  - name: "Test: User creation"
    setup:
      - "DELETE FROM test_users WHERE username = 'testuser'"

    test:
      query: |
        INSERT INTO test_users (username, email)
        VALUES ('testuser', 'test@example.com')
        RETURNING user_id, username

      assertions:
        - type: row_count
          expected: 1
        - type: not_null
          column: user_id

    teardown:
      - "DELETE FROM test_users WHERE username = 'testuser'"
```

### Performance Test

```yaml
tests:
  - name: "Test: Query performance"
    test:
      query: "SELECT * FROM users WHERE user_id = 1"
      assertions:
        - type: execution_time
          expected: 10  # milliseconds
```

## 📊 Validation Examples

### Email Validation

```yaml
field_validations:
  - table: users
    validations:
      - column: email
        rules:
          - type: regex
            pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
          - type: unique
          - type: not_null
```

### Range Validation

```yaml
field_validations:
  - table: products
    validations:
      - column: price
        rules:
          - type: range
            min: 0.01
            max: 999999.99
          - type: not_null
```

## 🚀 Running Examples

### Run All Tests

```bash
# Getting started examples
sqltest test --config examples/getting_started/simple_tests.yaml

# Advanced examples
sqltest test --config examples/advanced/comprehensive_tests.yaml

# With HTML report
sqltest test --config examples/advanced/comprehensive_tests.yaml --output report.html
```

### Run Validations

```bash
# Simple validations
sqltest validate --config examples/getting_started/simple_validations.yaml

# With strict mode (fail on warnings)
sqltest validate --config examples/getting_started/simple_validations.yaml --strict
```

### Profile Data

```bash
# Profile users table
sqltest profile --table users --database local

# Generate HTML profile
sqltest profile --table users --output profile.html
```

## 🔍 Troubleshooting Examples

### Check Database Connection

```bash
# Test connection from config
sqltest connection --config examples/getting_started/simple_database_config.yaml --test
```

### Validate Configuration

```bash
# Check YAML syntax
sqltest config --validate examples/advanced/enterprise_config.yaml
```

### Debug Tests

```bash
# Run with verbose logging
sqltest test --config examples/getting_started/simple_tests.yaml --verbose --log-level DEBUG
```

## 📖 Additional Resources

- **[User Guide](../docs/USER_GUIDE.md)** - Complete usage documentation
- **[API Reference](../docs/API_REFERENCE.md)** - API documentation
- **[Configuration Guide](../docs/CONFIGURATION.md)** - Configuration options
- **[Testing Guide](../docs/TESTING_GUIDE.md)** - Best practices
- **[Main README](../README.md)** - Project overview

## 💡 Tips

1. **Start Simple**: Begin with `simple_tests.yaml` and gradually add complexity
2. **Use Environment Variables**: Never commit passwords - use `${VAR}` syntax
3. **Tag Your Tests**: Use tags for organizing and filtering tests
4. **Test Incrementally**: Run tests frequently during development
5. **Monitor Performance**: Use execution time assertions to catch slow queries
6. **Isolate Tests**: Each test should be independent and not affect others

## 🤝 Contributing Examples

Have a useful example or tutorial? Contributions welcome!

1. Fork the repository
2. Add your example to appropriate directory
3. Update this README
4. Submit a pull request

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details

---

**SQLTest Pro** - Enterprise SQL Testing & Validation Framework

Ready to get started? Try [Tutorial 1: Getting Started](tutorials/TUTORIAL_01_GETTING_STARTED.md)!