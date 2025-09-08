# SQLTest Pro

A comprehensive Python-based testing framework for SQL code that provides unit testing capabilities, data validation, profiling, and business rule verification through an intuitive CLI and YAML configuration.

## 🚀 Features

- **📊 Data Profiling**: Analyze SQL tables with comprehensive statistics, pattern detection, and outlier identification
- **✅ Field Validation**: Rule-based validation for individual columns (regex, ranges, nulls, enums, etc.)
- **🔍 Business Rule Validation**: Complex cross-table validations with custom SQL queries
- **🎯 Data Type Validation**: Ensure type consistency and proper data type usage
- **🧪 SQL Unit Testing**: Full unit testing framework with fixtures, assertions, and coverage reporting
- **🎨 Interactive CLI**: Beautiful, engaging command-line interface with progress bars and real-time feedback
- **📄 Multiple Output Formats**: JSON, HTML, CSV reports for all testing tools
- **⚙️ YAML Configuration**: Define all tests and validations through simple YAML files

## 🏗️ Architecture

SQLTest Pro is built with a modular architecture consisting of:

- **Database Abstraction Layer**: Multi-database support (PostgreSQL, MySQL, SQLite, SQL Server, Snowflake)
- **Testing Modules**: Data profiler, validators, and unit testing framework
- **Configuration System**: YAML-based configuration with schema validation
- **CLI Interface**: Rich interactive terminal interface
- **Reporting Engine**: Multiple output formats and coverage reports

## 📁 Project Structure

```
sql-testing-suite/
├── sqltest/                     # Main package
│   ├── cli/                     # Command-line interface
│   ├── db/                      # Database abstraction layer
│   ├── modules/                 # Core testing modules
│   │   ├── profiler/           # Data profiling
│   │   ├── validators/         # Validation modules
│   │   └── testing/            # Unit testing framework
│   ├── config/                 # Configuration management
│   ├── reporting/              # Report generation
│   └── utils/                  # Utilities
├── examples/                   # Example configurations
├── docs/                       # Documentation
└── tests/                      # Test suite
```

## 🛠️ Technology Stack

- **CLI Framework**: Rich + Click
- **Database**: SQLAlchemy + native drivers
- **Configuration**: PyYAML + Pydantic
- **Reporting**: Jinja2 (HTML), pandas (data manipulation)
- **Testing**: pytest
- **Code Quality**: black, flake8, mypy

## 📋 Example Usage

### Data Profiling
```bash
# Profile a single table
sqltest profile --table users --output html

# Profile with custom query
sqltest profile --query "SELECT * FROM orders WHERE date > '2024-01-01'"
```

### Validation
```bash
# Run all validations
sqltest validate --config validations.yaml

# Run specific validation types
sqltest validate --type field_validations --table users
```

### Unit Testing
```bash
# Run all unit tests
sqltest test --config unit_tests.yaml

# Run specific test group
sqltest test --group "SQL Functions" --coverage
```

### Report Generation
```bash
# Generate HTML coverage report
sqltest report --type coverage --format html --output ./reports/
```

## ⚙️ Configuration

All testing is configured through YAML files:

### Database Connections (`database.yaml`)
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

### Validation Rules (`validations.yaml`)
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

### Unit Tests (`unit_tests.yaml`)
```yaml
unit_tests:
  - test_group: "SQL Functions"
    tests:
      - name: "Test customer_lifetime_value function"
        test:
          query: SELECT customer_lifetime_value(1) as clv
          assertions:
            - type: equals
              column: clv
              expected: 250.00
```

## 🎯 Development Status

This project is currently in the planning phase. The implementation will be done in 4 phases:

1. **Phase 1**: Foundation (Database layer, CLI framework, Configuration system)
2. **Phase 2**: Core Modules (Data profiler, Field validator, Basic reporting)
3. **Phase 3**: Advanced Features (Business rule validator, SQL unit testing, Coverage reporting)
4. **Phase 4**: Polish & Documentation (CLI enhancements, Documentation, Examples)

## 📚 Documentation

- [Project Plan](docs/PROJECT_PLAN.md) - Detailed implementation plan and architecture
- [CLI Mockup](docs/CLI_MOCKUP.md) - Visual mockups of the command-line interface
- [Example Configurations](examples/configs/) - Sample YAML configuration files

## 🤝 Contributing

This project is in early development. Contributions, ideas, and feedback are welcome!

## 📄 License

MIT License (to be added)

## 🎯 Goals

- Provide SQL developers with unit testing capabilities similar to other programming languages
- Improve data quality through comprehensive validation and profiling
- Make SQL testing accessible through intuitive YAML configuration
- Support multiple database platforms
- Enable integration with CI/CD pipelines
- Create an engaging developer experience through beautiful CLI interfaces

---

**Note**: This project is currently in the planning and design phase. Implementation will begin soon based on the detailed [PROJECT_PLAN.md](docs/PROJECT_PLAN.md).
