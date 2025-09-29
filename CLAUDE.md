# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SQLTest Pro is a comprehensive Python-based testing framework for SQL code that provides unit testing capabilities, data validation, profiling, and business rule verification through an intuitive CLI and YAML configuration.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=sqltest --cov-report=html

# Run specific test file
pytest tests/test_cli.py

# Run tests with verbose output
pytest -v
```

### Code Quality
```bash
# Format code
black sqltest/

# Check formatting without changes
black --check sqltest/

# Lint code
flake8 sqltest/

# Type checking
mypy sqltest/

# Sort imports
isort sqltest/

# Run all code quality checks
black --check sqltest/ && flake8 sqltest/ && mypy sqltest/ && isort --check-only sqltest/
```

### CLI Usage
```bash
# Show main dashboard
sqltest

# Show help
sqltest --help

# Profile data (placeholder implementation)
sqltest profile --table users --output html

# Run validations (placeholder implementation)
sqltest validate --config validations.yaml

# Run unit tests (placeholder implementation)
sqltest test --config unit_tests.yaml
```

## Architecture

### Core Package Structure
- `sqltest/cli/` - Command-line interface using Click + Rich
- `sqltest/db/` - Database abstraction layer with multi-database support
- `sqltest/modules/` - Core functionality:
  - `profiler/` - Data profiling and analysis
  - `field_validator/` - Field-level validation rules
  - `business_rules/` - Business rule validation engine
  - `testing/` - SQL unit testing framework
- `sqltest/config/` - YAML configuration management with Pydantic
- `sqltest/reporting/` - Report generation (JSON, HTML, CSV)

### Technology Stack
- **CLI**: Click + Rich for interactive terminal interface
- **Database**: SQLAlchemy with native drivers (PostgreSQL, MySQL, SQLite, SQL Server, Snowflake)
- **Configuration**: PyYAML + Pydantic for schema validation
- **Data Processing**: pandas + numpy
- **Reporting**: Jinja2 templates
- **Testing**: pytest with coverage reporting

### Configuration System
The project uses YAML-based configuration files for:
- Database connections (`database.yaml`)
- Field validation rules (`validations.yaml`)
- Business rule definitions (`business_rules.yaml`)
- Unit test specifications (`unit_tests.yaml`)

Configuration files support environment variable substitution (e.g., `${DEV_DB_PASSWORD}`).

## Development Guidelines

### Module Implementation Status
- **CLI Framework**: ✅ Complete with Rich UI and all command stubs
- **Database Layer**: 🚧 Basic structure, needs connection management implementation
- **Field Validator**: ✅ Working implementation with comprehensive rule support
- **Business Rules**: ✅ **ENTERPRISE-GRADE** - Advanced implementation with Week 2 enhancements:
  - ✅ Core rule execution engine with dependency management
  - ✅ Multi-level caching (L1 memory + L2 Redis)
  - ✅ Performance monitoring and metrics collection
  - ✅ Retry mechanisms with exponential backoff
  - ✅ Parallel execution with intelligent rule batching
  - ✅ Comprehensive error handling and resilience
  - ✅ Full YAML configuration support with environment variables
  - ✅ Enterprise-grade test coverage (79% coverage, 22 advanced tests)
  - ✅ Complete documentation and examples
- **Data Profiler**: 🚧 Basic structure, needs statistical analysis implementation
- **Unit Testing**: 🚧 Framework structure in place, needs test execution engine

### Testing Strategy
- Unit tests for all core functionality in `tests/`
- Integration tests for database operations
- CLI tests for command validation
- Coverage target: >80%
- Test markers: `unit`, `integration`, `slow`, `database`

### Code Style
- Black formatting (line length: 88)
- isort for import organization
- MyPy for type checking
- Flake8 for linting
- All code should include type hints
- Docstrings required for public interfaces

### Database Support
The framework supports multiple database platforms through SQLAlchemy:
- PostgreSQL (primary)
- MySQL/MariaDB
- SQLite (for testing/examples)
- SQL Server
- Snowflake

### Entry Points
- Main CLI: `sqltest.cli.main:cli`
- Package imports: `from sqltest import __version__`
- Core exceptions: `sqltest.exceptions`

### Error Handling
Custom exception hierarchy:
- `SQLTestError` - Base exception
- `ConfigurationError` - Configuration-related errors
- `DatabaseError` - Database connection/query errors