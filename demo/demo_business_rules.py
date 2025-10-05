#!/usr/bin/env python3
"""
Demo script for SQLTest Pro Business Rule Validator
"""

import sys
import os
from pathlib import Path

# Configure project and demo paths
DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = DEMO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = DEMO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from sqltest.db.connection import ConnectionManager
from sqltest.config.models import SQLTestConfig, DatabaseConfig, DatabaseType
from sqltest.modules.business_rules import (
    BusinessRuleValidator, 
    BusinessRule,
    RuleSet,
    RuleType,
    RuleSeverity,
    ValidationScope,
    ValidationContext,
    create_not_null_rule,
    create_uniqueness_rule,
    create_referential_integrity_rule,
    create_range_rule,
    create_completeness_rule,
    create_sample_business_rules
)
import sqlite3
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_database():
    """Create a demo SQLite database with sample data."""
    db_path = DATA_DIR / "demo_business_rules.db"

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create customers table (with relaxed constraints for demo)
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            registration_date DATE,
            status TEXT DEFAULT 'active'
        )
    """)
    
    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            total_amount DECIMAL(10,2) NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)
    
    # Insert sample customers
    customers_data = [
        (1, 'John Doe', 'john.doe@email.com', '555-0101', '2023-01-15', 'active'),
        (2, 'Jane Smith', 'jane.smith@email.com', '555-0102', '2023-02-20', 'active'),
        (3, 'Bob Johnson', 'bob.johnson@email.com', None, '2023-03-10', 'inactive'),  # Missing phone
        (4, 'Alice Brown', 'alice.brown@email.com', '555-0104', '2023-04-05', 'active'),
        (5, 'Charlie Wilson', 'charlie.wilson@email.com', '555-0105', None, 'active'),  # Missing registration date
        (6, '', 'empty.name@email.com', '555-0106', '2023-05-01', 'active'),  # Empty name - violation
        (7, None, 'null.name@email.com', '555-0107', '2023-05-02', 'active'),  # Null name - violation
    ]
    
    cursor.executemany(
        "INSERT INTO customers (customer_id, customer_name, email, phone, registration_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        customers_data
    )
    
    # Insert sample orders
    orders_data = [
        (1, 1, '2023-01-20', 150.00, 'completed'),
        (2, 2, '2023-02-25', 250.00, 'completed'),
        (3, 1, '2023-03-01', 75.50, 'pending'),
        (4, 4, '2023-04-10', 300.00, 'completed'),
        (5, 999, '2023-04-15', 100.00, 'pending'),  # Invalid customer_id - referential integrity violation
        (6, 2, '2023-05-01', -50.00, 'completed'),  # Negative amount - range violation
        (7, 3, '2023-05-05', 0.00, 'pending'),  # Zero amount - potential violation
        (8, 1, '2023-05-10', 1000000.00, 'completed'),  # Extremely high amount - potential outlier
    ]
    
    cursor.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?, ?)",
        orders_data
    )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created demo database: {db_path}")
    return db_path

def create_custom_business_rules():
    """Create custom business rules for the demo."""
    
    # Rule 1: Customer name should not be null or empty
    customer_name_rule = BusinessRule(
        name="customer_name_not_empty",
        description="Customer names must not be null or empty",
        rule_type=RuleType.DATA_QUALITY,
        severity=RuleSeverity.ERROR,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT 
                customer_id,
                customer_name,
                'Customer name is null or empty' as message,
                1 as violation_count,
                'customers' as table_name,
                'customer_name' as column_name
            FROM customers 
            WHERE customer_name IS NULL OR trim(customer_name) = ''
        """,
        enabled=True,
        tags={"data_quality", "customers", "not_null"}
    )
    
    # Rule 2: Email uniqueness check
    email_uniqueness_rule = BusinessRule(
        name="customer_email_unique",
        description="Customer emails must be unique",
        rule_type=RuleType.DATA_QUALITY,
        severity=RuleSeverity.CRITICAL,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT 
                email,
                'Duplicate email found' as message,
                COUNT(*) as violation_count,
                'customers' as table_name,
                'email' as column_name
            FROM customers 
            GROUP BY email 
            HAVING COUNT(*) > 1
        """,
        enabled=True,
        tags={"data_quality", "customers", "uniqueness"}
    )
    
    # Rule 3: Order amount validation
    order_amount_rule = BusinessRule(
        name="order_amount_positive",
        description="Order amounts must be positive",
        rule_type=RuleType.BUSINESS_LOGIC,
        severity=RuleSeverity.ERROR,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT 
                order_id,
                total_amount,
                'Order amount must be positive' as message,
                1 as violation_count,
                'orders' as table_name,
                'total_amount' as column_name
            FROM orders 
            WHERE total_amount <= 0
        """,
        enabled=True,
        tags={"business_logic", "orders", "range"}
    )
    
    # Rule 4: Referential integrity check
    referential_integrity_rule = BusinessRule(
        name="order_customer_exists",
        description="All orders must reference valid customers",
        rule_type=RuleType.REFERENTIAL_INTEGRITY,
        severity=RuleSeverity.CRITICAL,
        scope=ValidationScope.DATABASE,
        sql_query="""
            SELECT 
                o.order_id,
                o.customer_id,
                'Order references non-existent customer' as message,
                1 as violation_count,
                'orders' as table_name,
                'customer_id' as column_name
            FROM orders o
            LEFT JOIN customers c ON o.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """,
        enabled=True,
        tags={"referential_integrity", "orders", "customers"},
        dependencies=[]  # No dependencies for this rule
    )
    
    # Rule 5: High value order alert (Warning level)
    high_value_rule = BusinessRule(
        name="high_value_order_alert",
        description="Alert for unusually high value orders",
        rule_type=RuleType.BUSINESS_LOGIC,
        severity=RuleSeverity.WARNING,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT 
                order_id,
                total_amount,
                'Unusually high order value detected' as message,
                1 as violation_count,
                'orders' as table_name,
                'total_amount' as column_name
            FROM orders 
            WHERE total_amount > 500.00
        """,
        enabled=True,
        tags={"business_logic", "orders", "outlier"},
        max_violation_count=5  # Expect at most 5 high value orders
    )
    
    # Rule 6: Customer registration completeness
    registration_completeness_rule = BusinessRule(
        name="customer_registration_complete",
        description="Customer registration should be complete",
        rule_type=RuleType.DATA_QUALITY,
        severity=RuleSeverity.WARNING,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT 
                customer_id,
                customer_name,
                'Incomplete customer registration' as message,
                1 as violation_count,
                'customers' as table_name,
                CASE 
                    WHEN phone IS NULL THEN 'phone'
                    WHEN registration_date IS NULL THEN 'registration_date'
                    ELSE 'unknown'
                END as column_name
            FROM customers 
            WHERE phone IS NULL OR registration_date IS NULL
        """,
        enabled=True,
        tags={"data_quality", "customers", "completeness"}
    )
    
    # Create rule set
    rule_set = RuleSet(
        name="demo_ecommerce_rules",
        description="Comprehensive business rules for ecommerce demo",
        rules=[
            customer_name_rule,
            email_uniqueness_rule,
            order_amount_rule,
            referential_integrity_rule,
            high_value_rule,
            registration_completeness_rule
        ],
        parallel_execution=True,
        max_concurrent_rules=3,
        tags={"demo", "ecommerce", "data_quality"}
    )
    
    return rule_set

def demonstrate_business_rule_validation():
    """Demonstrate the business rule validation capabilities."""
    
    print("🔧 SQLTest Pro Business Rule Validator Demo")
    print("=" * 60)
    
    # Create demo database
    db_path = create_demo_database()
    
    # Setup connection manager with proper config
    db_config = DatabaseConfig(
        type=DatabaseType.SQLITE,
        path=db_path
    )
    
    config = SQLTestConfig(
        databases={'demo_db': db_config},
        default_database='demo_db'
    )
    
    connection_manager = ConnectionManager(config)
    
    # Create validator
    validator = BusinessRuleValidator(connection_manager, max_workers=3)
    
    # Load custom rule set
    custom_rules = create_custom_business_rules()
    validator.add_rule_set(custom_rules)
    
    # Also create sample rules for comparison
    sample_rules = create_sample_business_rules()
    validator.add_rule_set(sample_rules)
    
    print(f"\n📋 Loaded rule sets:")
    for rule_set_name in validator.list_rule_sets():
        rule_set = validator.get_rule_set(rule_set_name)
        print(f"  • {rule_set_name}: {len(rule_set.rules)} rules")
    
    # Run validation with custom rules
    print(f"\n🔍 Running Business Rule Validation...")
    print("-" * 40)
    
    try:
        # Validate with custom rules
        summary = validator.validate_with_rule_set(
            rule_set_name="demo_ecommerce_rules",
            database_name="demo_db",
            parallel=True,
            fail_fast=False
        )
        
        # Display results
        print(f"\n📊 Validation Results Summary:")
        print(f"  Rule Set: {summary.rule_set_name}")
        print(f"  Execution Time: {summary.execution_time_ms:.2f}ms")
        print(f"  Total Rules: {summary.total_rules}")
        print(f"  Rules Executed: {summary.rules_executed}")
        print(f"  Rules Passed: {summary.rules_passed} ✅")
        print(f"  Rules Failed: {summary.rules_failed} ❌")
        print(f"  Rules with Errors: {summary.rules_error} 🔥")
        print(f"  Rules Skipped: {summary.rules_skipped} ⏭️")
        print(f"  Success Rate: {summary.success_rate:.1f}%")
        
        print(f"\n🚨 Violation Summary:")
        print(f"  Total Violations: {summary.total_violations}")
        print(f"  Critical: {summary.critical_violations} 🔴")
        print(f"  Errors: {summary.error_violations} 🟠")
        print(f"  Warnings: {summary.warning_violations} 🟡")
        print(f"  Info: {summary.info_violations} 🔵")
        
        # Display individual rule results
        print(f"\n📋 Detailed Rule Results:")
        print("-" * 40)
        
        for result in summary.results:
            status_icon = "✅" if result.passed else "❌"
            severity_icon = {
                RuleSeverity.CRITICAL: "🔴",
                RuleSeverity.ERROR: "🟠", 
                RuleSeverity.WARNING: "🟡",
                RuleSeverity.INFO: "🔵"
            }.get(result.severity, "⚪")
            
            print(f"\n{status_icon} {result.rule_name} {severity_icon}")
            print(f"   Type: {result.rule_type.value}")
            print(f"   Status: {result.status.value}")
            print(f"   Message: {result.message}")
            print(f"   Execution Time: {result.execution_time_ms:.2f}ms")
            
            if result.violations:
                print(f"   Violations ({len(result.violations)}):")
                for violation in result.violations[:3]:  # Show first 3
                    print(f"     • {violation.message}")
                    if violation.sample_values:
                        print(f"       Sample: {violation.sample_values}")
                
                if len(result.violations) > 3:
                    print(f"     ... and {len(result.violations) - 3} more")
        
        # Export results
        export_path = OUTPUT_DIR / "demo_business_rules_results.json"
        validator.export_results_to_json(summary, str(export_path))
        print(f"\n💾 Results exported to: {export_path}")
        
        # Test with tags filtering
        print(f"\n🏷️  Testing Tag-based Filtering...")
        print("-" * 40)
        
        # Run only data quality rules
        dq_summary = validator.validate_with_rule_set(
            rule_set_name="demo_ecommerce_rules",
            database_name="demo_db",
            tags={"data_quality"}
        )
        
        print(f"Data Quality Rules Only: {dq_summary.rules_executed} rules executed")
        
        # Run only critical severity rules
        critical_summary = validator.validate_with_rule_set(
            rule_set_name="demo_ecommerce_rules", 
            database_name="demo_db",
            tags={"referential_integrity"}
        )
        
        print(f"Referential Integrity Rules Only: {critical_summary.rules_executed} rules executed")
        
        print(f"\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        logger.exception("Validation failed")
    
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info("Cleaned up demo database")

if __name__ == "__main__":
    demonstrate_business_rule_validation()
