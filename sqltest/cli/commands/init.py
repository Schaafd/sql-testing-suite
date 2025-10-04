
"""Project initialization CLI command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.table import Table

from sqltest.cli.utils import console
from sqltest.config import create_sample_config

@click.command(name="init")
@click.argument("project_name")
@click.option("--template", type=click.Choice(["basic", "advanced", "complete"]), default="basic", help="Project template")
@click.option("--with-validation", is_flag=True, help="Include sample validation rule sets")
@click.option("--with-tests", is_flag=True, help="Include sample test configurations")
@click.option("--with-examples", is_flag=True, help="Include example data and scenarios")
def init_command(project_name: str, template: str, with_validation: bool, with_tests: bool, with_examples: bool) -> None:
    """🚀 Initialize a new SQLTest Pro project.
    
    Creates a new project directory with sample configurations, validation rules,
    test cases, and documentation based on the selected template.
    """
    console.print(f"[bold green]Initializing project: {project_name}[/bold green]")
    console.print(f"Using template: [green]{template}[/green]")
    
    project_path = Path(project_name)
    if project_path.exists():
        console.print(f"[red]Error: Directory '{project_name}' already exists[/red]")
        return
    
    try:
        # Create project directory
        project_path.mkdir(parents=True)
        console.print(f"📁 Created project directory: [green]{project_path}[/green]")
        
        # Create sample configuration
        config_path = project_path / "sqltest.yaml"
        create_sample_config(config_path)
        console.print(f"⚙️  Created sample configuration: [green]{config_path}[/green]")
        
        # Create directories
        (project_path / "reports").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "rules").mkdir()
        (project_path / "templates").mkdir()
        console.print("📂 Created project directories")
        
        # Determine what to include based on template and options
        include_validation = with_validation or template in ["advanced", "complete"]
        include_tests = with_tests or template in ["advanced", "complete"]
        include_examples = with_examples or template == "complete"
        
        # Create validation rule sets
        if include_validation:
            create_validation_templates(project_path, template, include_examples)
            console.print(f"✅ Created validation rule templates")
        
        # Create test configurations
        if include_tests:
            create_test_templates(project_path, template, include_examples)
            console.print(f"🧪 Created test configuration templates")
        
        # Create example data and documentation
        if include_examples:
            create_example_data(project_path)
            console.print(f"📊 Created example data and scenarios")
        
        # Create documentation
        create_project_documentation(project_path, template, include_validation, include_tests, include_examples)
        console.print(f"📚 Created project documentation")
        
        console.print(f"\n[bold green]✅ Project '{project_name}' initialized successfully![/bold green]")
        
        # Show what was created
        show_project_summary(project_path, template, include_validation, include_tests, include_examples)
        
        # Show next steps
        console.print(f"\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(f"1. Edit [cyan]{config_path}[/cyan] to configure your databases")
        console.print(f"2. Set required environment variables (see README.md)")
        console.print(f"3. Test your configuration: [cyan]sqltest config validate[/cyan]")
        console.print(f"4. Test database connection: [cyan]sqltest db test[/cyan]")
        
        if include_validation:
            console.print(f"5. Customize validation rules in [cyan]rules/[/cyan] directory")
            console.print(f"6. Run validation: [cyan]sqltest validate --config rules/data_quality_rules.yaml[/cyan]")
        
        if include_tests:
            console.print(f"5. Customize test cases in [cyan]tests/[/cyan] directory")
            console.print(f"6. Run tests: [cyan]sqltest test --directory tests/[/cyan]")
        
        if include_examples:
            console.print(f"7. Explore example scenarios in [cyan]examples/[/cyan] directory")
        
    except Exception as e:
        console.print(f"[red]Error creating project: {e}[/red]")



# Project template creation functions
def create_validation_templates(project_path: Path, template: str, include_examples: bool) -> None:
    """Create validation rule templates."""
    rules_dir = project_path / "rules"
    
    # Basic data quality rules
    data_quality_rules = {
        "rule_set_name": "Data Quality Rules",
        "description": "Basic data quality validation rules for common data issues",
        "rules": [
            {
                "name": "null_check_required_fields",
                "type": "null_check",
                "severity": "error",
                "description": "Check for null values in required fields",
                "target": {
                    "table": "${TABLE_NAME}",
                    "columns": ["id", "email", "created_at"]
                },
                "tags": ["completeness", "critical"]
            },
            {
                "name": "email_format_validation",
                "type": "pattern_match",
                "severity": "warning",
                "description": "Validate email addresses format",
                "target": {
                    "table": "${TABLE_NAME}",
                    "column": "email"
                },
                "parameters": {
                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                },
                "tags": ["format", "data_quality"]
            },
            {
                "name": "duplicate_check_primary_key",
                "type": "uniqueness",
                "severity": "critical",
                "description": "Check for duplicate values in primary key columns",
                "target": {
                    "table": "${TABLE_NAME}",
                    "columns": ["id"]
                },
                "tags": ["uniqueness", "integrity"]
            },
            {
                "name": "date_range_validation",
                "type": "range",
                "severity": "warning",
                "description": "Validate date ranges are within reasonable bounds",
                "target": {
                    "table": "${TABLE_NAME}",
                    "column": "created_at"
                },
                "parameters": {
                    "min_date": "2020-01-01",
                    "max_date": "${CURRENT_DATE}"
                },
                "tags": ["temporal", "reasonableness"]
            }
        ]
    }
    
    # Write data quality rules
    with open(rules_dir / "data_quality_rules.yaml", 'w') as f:
        yaml.dump(data_quality_rules, f, default_flow_style=False, sort_keys=False)
    
    # Referential integrity rules for advanced templates
    if template in ["advanced", "complete"]:
        referential_rules = {
            "rule_set_name": "Referential Integrity Rules",
            "description": "Rules to validate foreign key relationships and data consistency",
            "rules": [
                {
                    "name": "foreign_key_existence",
                    "type": "foreign_key",
                    "severity": "error",
                    "description": "Validate foreign key references exist",
                    "target": {
                        "table": "orders",
                        "column": "customer_id"
                    },
                    "parameters": {
                        "reference_table": "customers",
                        "reference_column": "id"
                    },
                    "tags": ["referential_integrity", "consistency"]
                },
                {
                    "name": "orphan_record_check",
                    "type": "sql_rule",
                    "severity": "warning",
                    "description": "Check for orphaned records without valid parents",
                    "target": {
                        "sql": """
                        SELECT COUNT(*) as orphan_count 
                        FROM orders o 
                        LEFT JOIN customers c ON o.customer_id = c.id 
                        WHERE c.id IS NULL
                        """
                    },
                    "parameters": {
                        "expected_value": 0,
                        "comparison": "equals"
                    },
                    "tags": ["orphans", "data_integrity"]
                }
            ]
        }
        
        with open(rules_dir / "referential_integrity_rules.yaml", 'w') as f:
            yaml.dump(referential_rules, f, default_flow_style=False, sort_keys=False)
    
    # Business logic rules for complete template
    if template == "complete" or include_examples:
        business_rules = {
            "rule_set_name": "Business Logic Rules",
            "description": "Custom business logic validation rules",
            "rules": [
                {
                    "name": "order_total_calculation",
                    "type": "sql_rule",
                    "severity": "error",
                    "description": "Validate order totals are calculated correctly",
                    "target": {
                        "sql": """
                        SELECT o.id, o.total, 
                               SUM(oi.quantity * oi.unit_price) as calculated_total
                        FROM orders o
                        JOIN order_items oi ON o.id = oi.order_id
                        GROUP BY o.id, o.total
                        HAVING ABS(o.total - SUM(oi.quantity * oi.unit_price)) > 0.01
                        """
                    },
                    "parameters": {
                        "max_violations": 0
                    },
                    "tags": ["business_logic", "calculations"]
                },
                {
                    "name": "inventory_consistency",
                    "type": "sql_rule",
                    "severity": "warning",
                    "description": "Check inventory levels are consistent with sales",
                    "target": {
                        "sql": """
                        SELECT p.id, p.stock_quantity,
                               COALESCE(SUM(oi.quantity), 0) as total_sold
                        FROM products p
                        LEFT JOIN order_items oi ON p.id = oi.product_id
                        GROUP BY p.id, p.stock_quantity
                        HAVING p.stock_quantity < 0
                        """
                    },
                    "parameters": {
                        "max_violations": 0
                    },
                    "tags": ["inventory", "consistency"]
                }
            ]
        }
        
        with open(rules_dir / "business_logic_rules.yaml", 'w') as f:
            yaml.dump(business_rules, f, default_flow_style=False, sort_keys=False)


def create_test_templates(project_path: Path, template: str, include_examples: bool) -> None:
    """Create test configuration templates."""
    tests_dir = project_path / "tests"
    
    # Basic unit test configuration
    unit_tests = {
        "test_suite_name": "Basic Unit Tests",
        "description": "Basic SQL unit tests for database functions and procedures",
        "tests": [
            {
                "name": "test_user_creation",
                "description": "Test user creation with valid data",
                "tags": ["users", "crud", "basic"],
                "setup": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'test@example.com'"
                    ]
                },
                "execute": {
                    "sql": """
                    INSERT INTO users (name, email, created_at) 
                    VALUES ('Test User', 'test@example.com', NOW())
                    RETURNING id, name, email
                    """
                },
                "assertions": [
                    {
                        "type": "row_count",
                        "expected": 1,
                        "message": "Should insert exactly one user"
                    },
                    {
                        "type": "column_value",
                        "column": "name",
                        "expected": "Test User",
                        "message": "Name should match input"
                    },
                    {
                        "type": "not_null",
                        "column": "id",
                        "message": "ID should be generated"
                    }
                ],
                "teardown": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'test@example.com'"
                    ]
                }
            },
            {
                "name": "test_email_uniqueness",
                "description": "Test that duplicate emails are rejected",
                "tags": ["users", "constraints", "validation"],
                "setup": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'duplicate@example.com'",
                        "INSERT INTO users (name, email, created_at) VALUES ('First User', 'duplicate@example.com', NOW())"
                    ]
                },
                "execute": {
                    "sql": """
                    INSERT INTO users (name, email, created_at) 
                    VALUES ('Second User', 'duplicate@example.com', NOW())
                    """,
                    "expect_error": True
                },
                "assertions": [
                    {
                        "type": "error_occurred",
                        "expected": True,
                        "message": "Should raise constraint violation error"
                    }
                ],
                "teardown": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'duplicate@example.com'"
                    ]
                }
            }
        ]
    }
    
    with open(tests_dir / "unit_tests.yaml", 'w') as f:
        yaml.dump(unit_tests, f, default_flow_style=False, sort_keys=False)
    
    # Integration tests for advanced templates
    if template in ["advanced", "complete"]:
        integration_tests = {
            "test_suite_name": "Integration Tests",
            "description": "Integration tests for complex business workflows",
            "fixtures": {
                "customers": {
                    "type": "csv",
                    "data": [
                        {"id": 1, "name": "John Doe", "email": "john@example.com"},
                        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
                    ]
                },
                "products": {
                    "type": "csv",
                    "data": [
                        {"id": 1, "name": "Widget A", "price": 10.99, "stock": 100},
                        {"id": 2, "name": "Widget B", "price": 15.50, "stock": 50}
                    ]
                }
            },
            "tests": [
                {
                    "name": "test_order_workflow",
                    "description": "Test complete order creation and processing workflow",
                    "tags": ["orders", "workflow", "integration"],
                    "dependencies": ["customers", "products"],
                    "setup": {
                        "fixtures": ["customers", "products"],
                        "sql": [
                            "DELETE FROM order_items",
                            "DELETE FROM orders"
                        ]
                    },
                    "execute": {
                        "sql": """
                        WITH new_order AS (
                            INSERT INTO orders (customer_id, order_date, status)
                            VALUES (1, NOW(), 'pending')
                            RETURNING id
                        ),
                        order_items AS (
                            INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                            SELECT no.id, 1, 2, 10.99 FROM new_order no
                            UNION ALL
                            SELECT no.id, 2, 1, 15.50 FROM new_order no
                        )
                        SELECT o.id, o.customer_id, o.status,
                               COUNT(oi.id) as item_count,
                               SUM(oi.quantity * oi.unit_price) as total
                        FROM orders o
                        JOIN order_items oi ON o.id = oi.order_id
                        WHERE o.id = (SELECT id FROM new_order)
                        GROUP BY o.id, o.customer_id, o.status
                        """
                    },
                    "assertions": [
                        {
                            "type": "row_count",
                            "expected": 1,
                            "message": "Should create one order"
                        },
                        {
                            "type": "column_value",
                            "column": "item_count",
                            "expected": 2,
                            "message": "Should have 2 order items"
                        },
                        {
                            "type": "column_value",
                            "column": "total",
                            "expected": 37.48,
                            "message": "Total should be calculated correctly"
                        }
                    ],
                    "teardown": {
                        "sql": [
                            "DELETE FROM order_items",
                            "DELETE FROM orders"
                        ]
                    }
                }
            ]
        }
        
        with open(tests_dir / "integration_tests.yaml", 'w') as f:
            yaml.dump(integration_tests, f, default_flow_style=False, sort_keys=False)
    
    # Performance tests for complete template
    if template == "complete" or include_examples:
        performance_tests = {
            "test_suite_name": "Performance Tests",
            "description": "Performance and load testing for database operations",
            "tests": [
                {
                    "name": "test_bulk_insert_performance",
                    "description": "Test performance of bulk insert operations",
                    "tags": ["performance", "bulk_operations"],
                    "timeout_ms": 5000,
                    "setup": {
                        "sql": ["CREATE TEMPORARY TABLE temp_users AS SELECT * FROM users LIMIT 0"]
                    },
                    "execute": {
                        "sql": """
                        INSERT INTO temp_users (name, email, created_at)
                        SELECT 
                            'User ' || generate_series(1, 1000),
                            'user' || generate_series(1, 1000) || '@example.com',
                            NOW()
                        """
                    },
                    "assertions": [
                        {
                            "type": "execution_time",
                            "max_ms": 3000,
                            "message": "Bulk insert should complete within 3 seconds"
                        },
                        {
                            "type": "row_count",
                            "table": "temp_users",
                            "expected": 1000,
                            "message": "Should insert 1000 users"
                        }
                    ]
                }
            ]
        }
        
        with open(tests_dir / "performance_tests.yaml", 'w') as f:
            yaml.dump(performance_tests, f, default_flow_style=False, sort_keys=False)


def create_example_data(project_path: Path) -> None:
    """Create example data and scenarios."""
    examples_dir = project_path / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Sample data files
    sample_users_csv = """id,name,email,created_at,status
1,John Doe,john@example.com,2023-01-15 10:30:00,active
2,Jane Smith,jane@example.com,2023-01-16 14:20:00,active
3,Bob Johnson,bob@example.com,2023-01-17 09:15:00,inactive
4,Alice Brown,alice@example.com,2023-01-18 16:45:00,active
5,Charlie Wilson,charlie@example.com,2023-01-19 11:30:00,pending"""
    
    with open(examples_dir / "sample_users.csv", 'w') as f:
        f.write(sample_users_csv)
    
    # Sample schema creation script
    schema_sql = """
-- Example database schema for SQLTest Pro
-- This creates a simple e-commerce-like schema for demonstration

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES users(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    total DECIMAL(10, 2)
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);

-- Sample data
INSERT INTO users (name, email, created_at, status) VALUES
('John Doe', 'john@example.com', '2023-01-15 10:30:00', 'active'),
('Jane Smith', 'jane@example.com', '2023-01-16 14:20:00', 'active'),
('Bob Johnson', 'bob@example.com', '2023-01-17 09:15:00', 'inactive')
ON CONFLICT (email) DO NOTHING;

INSERT INTO products (name, description, price, stock_quantity) VALUES
('Widget A', 'A useful widget for various tasks', 10.99, 100),
('Widget B', 'An advanced widget with extra features', 15.50, 50),
('Gadget X', 'A revolutionary gadget', 25.00, 25)
ON CONFLICT DO NOTHING;
"""
    
    with open(examples_dir / "sample_schema.sql", 'w') as f:
        f.write(schema_sql)
    
    # Sample scenarios documentation
    scenarios_md = """
# Example Test Scenarios

This directory contains example data and test scenarios to help you get started with SQLTest Pro.

## Files

- `sample_schema.sql` - Example database schema with tables and sample data
- `sample_users.csv` - Sample user data for testing
- `scenario_*.yaml` - Various testing scenarios

## Scenarios

### 1. Data Quality Testing
- Null value validation
- Format validation (email, phone, etc.)
- Range validation for numeric fields
- Duplicate detection

### 2. Business Rule Testing  
- Order total calculations
- Inventory consistency
- Customer status transitions
- Date range validations

### 3. Integration Testing
- Multi-table operations
- Foreign key constraints
- Transaction integrity
- Workflow validation

### 4. Performance Testing
- Bulk operations
- Query performance
- Index effectiveness
- Concurrent access patterns

## Getting Started

1. Create your test database and run `sample_schema.sql`
2. Update the database configuration in `../sqltest.yaml`
3. Customize the validation rules in `../rules/`
4. Modify the test cases in `../tests/`
5. Run your first validation: `sqltest validate --config ../rules/data_quality_rules.yaml`
6. Run your first tests: `sqltest test --directory ../tests/`

## Customization

Feel free to modify these examples to match your specific:
- Database schema
- Business rules
- Data quality requirements
- Performance expectations
"""
    
    with open(examples_dir / "README.md", 'w') as f:
        f.write(scenarios_md)


def create_project_documentation(project_path: Path, template: str, include_validation: bool, 
                                include_tests: bool, include_examples: bool) -> None:
    """Create project documentation."""
    
    readme_content = f"""
# SQLTest Pro Project

Welcome to your new SQLTest Pro project! This project was created with the `{template}` template.

## 🚀 Quick Start

### 1. Configure Your Database

Edit `sqltest.yaml` to configure your database connections:

```yaml
default_database: dev_db

databases:
  dev_db:
    type: postgresql  # or mysql, sqlite
    host: localhost
    port: 5432
    database: your_database
    username: your_username
    password: ${{DEV_DB_PASSWORD}}  # Use environment variables for passwords
```

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
export DEV_DB_PASSWORD="your_secure_password"
```

### 3. Test Your Configuration

```bash
# Validate configuration
sqltest config validate sqltest.yaml

# Test database connection
sqltest db test

# List available tables
sqltest db tables
```

## 📊 Features Included

### Data Profiling
Analyze your data structure and quality:

```bash
# Profile a specific table
sqltest profile --table users

# Profile with custom query
sqltest profile --query "SELECT * FROM orders WHERE created_at > '2023-01-01'"
```
"""
    
    if include_validation:
        readme_content += """
### Validation Rules
The `rules/` directory contains validation rule sets:

- `data_quality_rules.yaml` - Basic data quality checks
"""
        if template in ["advanced", "complete"]:
            readme_content += """- `referential_integrity_rules.yaml` - Foreign key and relationship validation
"""
        if template == "complete":
            readme_content += """- `business_logic_rules.yaml` - Custom business rule validation
"""
        
        readme_content += """
```bash
# Run validation rules
sqltest validate --config rules/data_quality_rules.yaml

# Run with specific table focus
sqltest validate --config rules/data_quality_rules.yaml --table users

# Generate automatic data quality rules
sqltest validate --generate --table users
```
"""
    
    if include_tests:
        readme_content += """
### Unit Testing
The `tests/` directory contains test suites:

- `unit_tests.yaml` - Basic unit tests for database operations
"""
        if template in ["advanced", "complete"]:
            readme_content += """- `integration_tests.yaml` - Multi-table integration tests
"""
        if template == "complete":
            readme_content += """- `performance_tests.yaml` - Performance and load tests
"""
        
        readme_content += """
```bash
# Run all tests
sqltest test --directory tests/

# Run specific test file
sqltest test --config tests/unit_tests.yaml

# Run with coverage reporting
sqltest test --directory tests/ --coverage

# Run tests matching specific tags
sqltest test --directory tests/ --group "users,crud"
```
"""
    
    if include_examples:
        readme_content += """
### Example Data
The `examples/` directory contains:

- Sample database schema (`sample_schema.sql`)
- Sample data files (`sample_users.csv`)
- Documentation for various testing scenarios

See `examples/README.md` for detailed information.
"""
    
    readme_content += """
## 📄 Reporting

Generate comprehensive reports:

```bash
# Generate HTML dashboard
sqltest report --type dashboard --format html

# Generate validation report from results
sqltest validate --output validation_results.json
sqltest report --type validation --input validation_results.json --format html

# Generate test coverage report
sqltest test --directory tests/ --coverage --output test_results.json
sqltest report --type testing --input test_results.json --format html
```

## 📚 Directory Structure

```
{project_path.name}/
├── sqltest.yaml              # Main configuration
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── reports/                  # Generated reports
├── templates/                # Custom report templates
"""
    
    if include_validation:
        readme_content += """├── rules/                    # Validation rule sets
│   ├── data_quality_rules.yaml
"""
        if template in ["advanced", "complete"]:
            readme_content += """│   ├── referential_integrity_rules.yaml
"""
        if template == "complete":
            readme_content += """│   └── business_logic_rules.yaml
"""
        else:
            readme_content = readme_content.rstrip() + "\n"
    
    if include_tests:
        readme_content += """├── tests/                   # Test configurations
│   ├── unit_tests.yaml
"""
        if template in ["advanced", "complete"]:
            readme_content += """│   ├── integration_tests.yaml
"""
        if template == "complete":
            readme_content += """│   └── performance_tests.yaml
"""
        else:
            readme_content = readme_content.rstrip() + "\n"
    
    if include_examples:
        readme_content += """└── examples/               # Example data and scenarios
    ├── README.md
    ├── sample_schema.sql
    └── sample_users.csv
"""
    else:
        readme_content += """└── ...
"""
    
    readme_content += """
```

## 🔧 Customization

### Adding New Validation Rules

1. Edit existing rule files in `rules/` or create new ones
2. Follow the YAML schema for rule definitions
3. Test your rules: `sqltest validate --config rules/your_rules.yaml`

### Creating New Tests

1. Add test configurations to `tests/` directory
2. Define fixtures, setup, execution, and assertions
3. Run your tests: `sqltest test --config tests/your_tests.yaml`

### Custom Reports

1. Create custom HTML templates in `templates/`
2. Use `--template` option when generating reports
3. Customize styling and layout as needed

## 📖 Documentation

- [SQLTest Pro Documentation](https://github.com/your-org/sqltest-pro)
- [Configuration Reference](https://github.com/your-org/sqltest-pro/docs/config)
- [Rule Definition Guide](https://github.com/your-org/sqltest-pro/docs/rules)
- [Testing Framework Guide](https://github.com/your-org/sqltest-pro/docs/testing)

## 🤝 Support

If you need help:

1. Check the documentation
2. Run `sqltest --help` for CLI usage
3. Use `--verbose` flag for detailed error information
4. Create an issue on the project repository

---

**Happy Testing!** 🎉
"""
    
    with open(project_path / "README.md", 'w') as f:
        f.write(readme_content)


def show_project_summary(project_path: Path, template: str, include_validation: bool, 
                        include_tests: bool, include_examples: bool) -> None:
    """Show summary of what was created."""
    console.print(f"\n[bold cyan]📋 Project Summary[/bold cyan]")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Files Created", style="white")
    
    summary_table.add_row("Configuration", "✅ Created", "sqltest.yaml")
    summary_table.add_row("Documentation", "✅ Created", "README.md")
    summary_table.add_row("Directories", "✅ Created", "reports/, templates/")
    
    if include_validation:
        files = "data_quality_rules.yaml"
        if template in ["advanced", "complete"]:
            files += ", referential_integrity_rules.yaml"
        if template == "complete":
            files += ", business_logic_rules.yaml"
        summary_table.add_row("Validation Rules", "✅ Created", files)
    
    if include_tests:
        files = "unit_tests.yaml"
        if template in ["advanced", "complete"]:
            files += ", integration_tests.yaml"
        if template == "complete":
            files += ", performance_tests.yaml"
        summary_table.add_row("Test Configurations", "✅ Created", files)
    
    if include_examples:
        summary_table.add_row("Example Data", "✅ Created", "sample_schema.sql, sample_users.csv")
    
    console.print(summary_table)


