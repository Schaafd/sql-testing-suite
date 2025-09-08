# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

SQLTest Pro is a comprehensive Python-based testing framework for SQL code that provides data profiling, validation, and unit testing capabilities through an interactive CLI. The project is currently in the foundation phase with a complete CLI framework and project structure but core modules are not yet implemented.

## Common Development Commands

### Environment Setup
```bash
# Install the package in development mode
pip install -e .

# Install with all development dependencies
pip install -e .[dev]

# Install specific dependency groups
pip install -e .[test]  # Testing dependencies only
pip install -e .[docs]  # Documentation dependencies only
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run tests verbosely with detailed output
pytest -v

# Run tests with coverage report
pytest --cov=sqltest --cov-report=html

# Run specific test file
pytest tests/test_basic.py

# Run tests with specific markers
pytest -m "unit"          # Unit tests only
pytest -m "integration"   # Integration tests only  
pytest -m "database"      # Database-dependent tests only
```

### Code Quality
```bash
# Format code with black
black sqltest/

# Check code formatting without changes
black --check sqltest/

# Sort imports with isort
isort sqltest/

# Run type checking with mypy
mypy sqltest/

# Lint code with flake8
flake8 sqltest/

# Run all code quality checks
black --check sqltest/ && isort --check-only sqltest/ && flake8 sqltest/ && mypy sqltest/
```

### CLI Testing
```bash
# Test basic CLI functionality
sqltest --help
sqltest --version

# Show interactive dashboard (placeholder)
sqltest

# Test individual commands (all show "not yet implemented")
sqltest profile --help
sqltest validate --help
sqltest test --help
sqltest report --help
sqltest init --help
```

## Architecture and Code Structure

### Modular Architecture
The project follows a clean, modular architecture with clear separation of concerns:

- **CLI Layer** (`sqltest/cli/`) - Rich-based interactive command-line interface
- **Database Abstraction** (`sqltest/db/`) - Multi-database connection and query management
- **Core Modules** (`sqltest/modules/`) - Main functionality (profiler, validators, testing framework)
- **Configuration System** (`sqltest/config/`) - YAML-based configuration with Pydantic validation
- **Reporting Engine** (`sqltest/reporting/`) - Multi-format report generation

### Key Design Patterns

**Command Pattern**: Each CLI command is implemented as a separate function with Click decorators, making it easy to add new commands or modify existing ones.

**Database Adapter Pattern**: The `sqltest/db/adapters/` directory is structured to support multiple database types (PostgreSQL, MySQL, SQLite, SQL Server, Snowflake) through a common interface.

**Configuration-Driven Architecture**: All testing behavior is defined through YAML configuration files with extensive examples in `examples/configs/`:
- `database.yaml` - Database connections with environment variable support
- `validations.yaml` - Field validations, business rules, and data type checks
- `unit_tests.yaml` - SQL unit tests with fixtures and assertions

### Technology Stack Integration
- **CLI Framework**: Click + Rich for beautiful interactive interfaces
- **Database Layer**: SQLAlchemy 2.0+ for database abstraction
- **Configuration**: PyYAML + Pydantic for robust config management
- **Data Processing**: pandas + numpy for data analysis
- **Reporting**: Jinja2 templates for HTML reports
- **Testing Infrastructure**: pytest with coverage reporting

## Implementation Status

### ✅ Completed (Phase 1 - Foundation)
- Complete project structure and package setup
- CLI framework with Rich-based interactive dashboard
- All command stubs implemented with proper help text
- Comprehensive pytest configuration and basic test suite
- Development tooling configuration (black, flake8, mypy, isort)
- Example YAML configurations for all major features
- Exception hierarchy and package initialization

### 🚧 Next Implementation Priorities (Phase 2)
Based on `PROJECT_PLAN.md`, the next components to implement are:

1. **Database Abstraction Layer** (`sqltest/db/`)
   - Connection management with pooling
   - Query execution engine with error handling
   - Database-specific adapters for supported databases

2. **Configuration System** (`sqltest/config/`)
   - YAML parser with schema validation
   - Pydantic models for configuration objects
   - Environment variable interpolation

3. **Data Profiler** (`sqltest/modules/profiler/`)
   - Table statistics computation
   - Pattern detection algorithms
   - Outlier identification

## Configuration System

The project uses a sophisticated YAML-based configuration system:

### Database Connections
- Supports multiple database types with connection pooling
- Environment variable substitution (e.g., `${DB_PASSWORD}`)
- Connection pool configuration with retry logic
- Default database selection

### Validation Rules
- **Field Validations**: Column-level rules (regex, ranges, nulls, enums, lengths)
- **Business Rules**: Complex multi-table validation queries
- **Data Type Validations**: Schema consistency checks
- **Custom Validations**: Python functions for complex logic

### Unit Testing Framework
- Test fixtures with shared data
- Parameterized tests for multiple input scenarios
- Setup/teardown SQL execution
- Mock data generation with configurable schemas
- Coverage reporting for SQL objects

## Development Workflow

### Project Commands Integration
The CLI is designed to be the primary interface for all operations:

```bash
# Data profiling (when implemented)
sqltest profile --table users --output html

# Validation execution (when implemented)
sqltest validate --config examples/configs/validations.yaml

# Unit test execution (when implemented)
sqltest test --config examples/configs/unit_tests.yaml --coverage

# Report generation (when implemented)
sqltest report --type coverage --format html --output ./reports/
```

### Configuration File Location
All example configurations are in `examples/configs/` and serve as templates for real implementations. The CLI accepts `--config` parameters to specify custom configuration files.

### Error Handling
Custom exception hierarchy in `sqltest/exceptions.py`:
- `SQLTestError` - Base exception
- `ConfigurationError` - Configuration parsing/validation errors
- `DatabaseError` - Database connection/query errors

## Testing Strategy

### Current Test Coverage: 49%
The test suite in `tests/test_basic.py` covers:
- Package initialization and exports
- CLI command registration and help text
- Version information display

### Test Categories (Configured in pyproject.toml)
- `unit` - Unit tests for individual components
- `integration` - Integration tests across modules
- `slow` - Long-running tests
- `database` - Tests requiring database connections

### Future Testing Needs
When implementing core modules, ensure tests cover:
- Database connection handling and error scenarios
- YAML configuration parsing and validation
- Data profiling accuracy with known datasets
- Validation rule execution and reporting
- Unit test framework with various SQL constructs

## Build and Packaging

### Project Metadata
- Package name: `sqltest-pro`
- Entry point: `sqltest = sqltest.cli.main:cli`
- Python compatibility: 3.9+
- Dependencies managed through `pyproject.toml` with optional extras

### Development Dependencies
The `[dev]` extra includes all tools for development workflow:
- Code quality tools (black, flake8, mypy, isort)
- Testing framework (pytest with plugins)
- Documentation tools (sphinx)

This modular architecture allows for incremental development while maintaining a consistent interface and clear separation of concerns.
