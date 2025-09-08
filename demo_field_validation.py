#!/usr/bin/env python3
"""
Demo script to showcase SQLTest Pro Field Validator functionality.

This script demonstrates:
1. Loading a validation configuration
2. Connecting to a SQLite database
3. Running field validation on a table
4. Displaying results in a user-friendly format
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any
from sqltest.modules.field_validator.models import TableValidationResult

from sqltest.modules.field_validator import TableFieldValidator, ValidationConfigLoader
from sqltest.db.connection import ConnectionManager


def create_sample_database():
    """Create a sample SQLite database with test data."""
    db_path = Path("data/test_data.db")
    db_path.parent.mkdir(exist_ok=True)
    
    # Connect and create sample data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            age INTEGER,
            status TEXT,
            salary REAL
        )
    """)
    
    # Insert sample data
    sample_data = [
        (1, "Alice Johnson", "alice@example.com", 28, "active", 75000.0),
        (2, "Bob Smith", "bob.invalid-email", 35, "active", 85000.0),
        (3, "Charlie Brown", "charlie@company.org", 150, "inactive", 65000.0),  # Invalid age
        (4, "Diana Prince", "diana@superhero.net", 32, "pending", 95000.0),
        (5, "Eve Wilson", None, 25, "active", None),  # Missing email and salary
        (6, "Frank Miller", "frank@example.com", -5, "unknown", 45000.0),  # Invalid age and status
        (7, "Grace Lee", "grace@email.co", 42, "active", 1200000.0),  # Salary too high
        (8, "Henry Ford", "henry@auto.com", 29, "inactive", 55000.0),
        (9, "Ivy Chen", "ivy@tech.startup", 24, "active", 70000.0),
        (10, "Jack Black", "jack@entertainment.biz", 45, "active", 80000.0)
    ]
    
    # Clear existing data and insert fresh data
    cursor.execute("DELETE FROM users")
    cursor.executemany("""
        INSERT INTO users (id, name, email, age, status, salary)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_data)
    
    conn.commit()
    conn.close()
    print(f"✓ Sample database created at {db_path}")
    print(f"✓ Inserted {len(sample_data)} sample records")


def display_validation_results(results: TableValidationResult):
    """Display validation results in a user-friendly format."""
    print("\n" + "="*80)
    print("🔍 FIELD VALIDATION RESULTS")
    print("="*80)
    
    # Overview
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   • Table: {results.table_name}")
    print(f"   • Database: {results.database_name}")
    print(f"   • Fields Validated: {len(results.field_results)}")
    print(f"   • Total Rules: {results.total_rules}")
    print(f"   • Rules Passed: {results.passed_rules}")
    print(f"   • Rules Failed: {results.failed_rules}")
    print(f"   • Warnings: {results.warnings}")
    print(f"   • Success Rate: {results.overall_success_rate:.1f}%")
    print(f"   • Overall Status: {'✅ PASSED' if not results.has_errors else '❌ FAILED'}")
    
    # Field-by-field results
    for field_result in results.field_results:
        print(f"\n📝 FIELD: {field_result.column_name.upper()}")
        print(f"   Status: {'✅ PASSED' if not field_result.has_errors else '❌ FAILED'}")
        print(f"   Total Rows: {field_result.total_rows}")
        print(f"   Rules Passed: {field_result.passed_rules}")
        print(f"   Rules Failed: {field_result.failed_rules}")
        print(f"   Warnings: {field_result.warnings}")
        print(f"   Success Rate: {field_result.success_rate:.1f}%")
        
        # Show validation issues if any
        error_results = [r for r in field_result.validation_results if not r.passed]
        if error_results:
            print(f"   🚨 Issues:")
            for issue in error_results[:3]:  # Show first 3 issues
                level_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.level.value if hasattr(issue.level, 'value') else str(issue.level), "ℹ️")
                print(f"     {level_emoji} {issue.rule_name}: {issue.message}")
                if issue.value is not None:
                    print(f"        Value: '{issue.value}' (Row {issue.row_number or 'N/A'})")
            
            if len(error_results) > 3:
                print(f"     ... and {len(error_results) - 3} more issues")
    
    print("\n" + "="*80)


def main():
    """Main demo function."""
    print("🚀 SQLTest Pro - Field Validator Demo")
    print("="*50)
    
    try:
        # Step 1: Create sample database
        print("\n1️⃣ Creating sample database...")
        create_sample_database()
        
        # Step 2: Load validation configuration
        print("\n2️⃣ Loading validation configuration...")
        config_path = Path("examples/demo/validation_rules_corrected.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return
            
        config_loader = ValidationConfigLoader()
        config = config_loader.load_from_file(str(config_path))
        print(f"✓ Loaded validation config with {len(config)} rule sets")
        
        # Step 3: Setup database connection
        print("\n3️⃣ Setting up database connection...")
        # Simple connection manager for SQLite
        class SimpleResult:
            def __init__(self, data_tuples, columns):
                import pandas as pd
                if data_tuples:
                    self.data = pd.DataFrame(data_tuples, columns=columns)
                    self.is_empty = False
                else:
                    self.data = pd.DataFrame(columns=columns)
                    self.is_empty = True
        
        class SimpleConnectionManager:
            def __init__(self, db_path: str):
                self.db_path = db_path
                # Create a simple config object
                class SimpleConfig:
                    default_database = "sqlite"
                self.config = SimpleConfig()
            
            def get_connection(self, db_name: str = None):
                return sqlite3.connect(self.db_path)
            
            def execute_query(self, query: str, params=None, db_name: str = None):
                conn = self.get_connection()
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                results = cursor.fetchall()
                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []
                conn.close()
                return SimpleResult(results, columns)
            
            def get_adapter(self, db_name: str = None):
                # Simple adapter for SQLite
                class SimpleAdapter:
                    def get_driver_name(self):
                        return "sqlite"
                return SimpleAdapter()
        
        connection_manager = SimpleConnectionManager("data/test_data.db")
        print("✓ Database connection ready")
        
        # Step 4: Initialize and run field validator
        print("\n4️⃣ Running field validation...")
        validator = TableFieldValidator(connection_manager, strict_mode=False)
        
        # Load the validation rules into the validator
        for rule_set_name, rule_set in config.items():
            validator.add_rule_set(rule_set)
        
        # Run validation on the users table with all rule sets
        # We'll combine all results
        all_field_results = []
        
        for rule_set_name, rule_set in config.items():
            try:
                result = validator.validate_table_data('users', rule_set_name)
                all_field_results.extend(result.field_results)
            except Exception as e:
                print(f"   ⚠️ Warning: Could not validate with rule set '{rule_set_name}': {e}")
        
        # Create combined results
        from sqltest.modules.field_validator.models import TableValidationResult
        results = TableValidationResult(
            table_name="users",
            database_name="sqlite", 
            field_results=all_field_results
        )
        
        # Step 5: Display results
        print("\n5️⃣ Processing results...")
        display_validation_results(results)
        
        # Step 6: Save results to file
        print("\n6️⃣ Saving results...")
        results_path = Path("reports/demo_validation_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert dataclass to dictionary for JSON serialization
        import dataclasses
        results_dict = dataclasses.asdict(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"✓ Results saved to {results_path}")
        
        print("\n🎉 Demo completed successfully!")
        print("💡 Next steps: Try implementing Business Rule Validator!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
