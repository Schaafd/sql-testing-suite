#!/usr/bin/env python3
"""
Demo script to showcase SQLTest Pro Data Profiler functionality.

This script demonstrates:
1. Loading a sample dataset
2. Connecting to a SQLite database
3. Running comprehensive data profiling on a table
4. Profiling specific columns
5. Profiling SQL query results
6. Displaying results in a user-friendly format
7. Generating insights and recommendations
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
import random

from sqltest.modules.profiler import DataProfiler
from sqltest.modules.profiler.models import TableProfile


def create_sample_dataset():
    """Create a more comprehensive sample dataset for profiling demonstration."""
    db_path = Path("data/profiler_test_data.db")
    db_path.parent.mkdir(exist_ok=True)
    
    # Connect and create sample data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create orders table with diverse data types and quality issues
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            customer_email TEXT,
            order_date DATE,
            order_amount REAL,
            status TEXT,
            shipping_address TEXT,
            phone_number TEXT,
            product_category TEXT,
            discount_percent REAL,
            notes TEXT
        )
    """)
    
    # Clear existing data
    cursor.execute("DELETE FROM orders")
    
    # Generate sample data with various patterns and quality issues
    statuses = ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned', None]
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', None]
    
    sample_data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(1, 1001):  # 1000 sample orders
        order_date = base_date + timedelta(days=random.randint(0, 365))
        
        # Generate customer email with various formats and some invalid ones
        if i % 50 == 0:  # 2% invalid emails
            email = f"invalid-email-{i}"
        elif i % 25 == 0:  # 4% missing emails
            email = None
        else:
            domains = ['gmail.com', 'yahoo.com', 'company.com', 'example.org']
            email = f"customer{i}@{random.choice(domains)}"
        
        # Generate phone numbers with various formats
        if i % 30 == 0:  # Some missing phones
            phone = None
        elif i % 15 == 0:  # Some invalid formats
            phone = f"invalid-{i}"
        else:
            area = random.randint(200, 999)
            exchange = random.randint(200, 999)
            number = random.randint(1000, 9999)
            formats = [
                f"({area}) {exchange}-{number}",
                f"{area}-{exchange}-{number}",
                f"{area}.{exchange}.{number}",
                f"{area}{exchange}{number}"
            ]
            phone = random.choice(formats)
        
        # Generate order amounts with outliers
        if i % 100 == 0:  # 1% very high amounts
            amount = random.uniform(10000, 50000)
        elif i % 200 == 0:  # 0.5% negative amounts (data quality issue)
            amount = -random.uniform(10, 100)
        else:
            amount = random.uniform(10, 500)
        
        # Generate discount with some outliers
        if i % 150 == 0:  # Some invalid discounts > 100%
            discount = random.uniform(100, 200)
        else:
            discount = random.uniform(0, 30)
        
        # Generate addresses with varying lengths
        if i % 40 == 0:  # Some missing addresses
            address = None
        elif i % 80 == 0:  # Some very long addresses
            address = f"Very Long Address Line {i} with lots of unnecessary details that make it extremely long Street, Apartment {i}, Building Complex Name, Area {i}, City {i}, State, Country - {i:05d}"
        else:
            address = f"{i} Main St, City {i % 50}, State {i % 10:02d}"
        
        # Generate notes with varying content
        if i % 20 == 0:  # Some very long notes
            notes = f"This is a very detailed order note for order {i} with extensive information about customer preferences, special delivery instructions, gift wrapping requirements, and various other details that make this note quite lengthy and potentially indicate a data quality issue."
        elif i % 10 == 0:  # Some empty notes
            notes = ""
        elif i % 5 == 0:  # Some null notes
            notes = None
        else:
            note_templates = [
                f"Regular order {i}",
                f"Gift for customer {i}",
                f"Rush delivery needed",
                f"Customer {i} is a VIP",
                "Handle with care"
            ]
            notes = random.choice(note_templates)
        
        sample_data.append((
            i,  # order_id
            random.randint(1, 500),  # customer_id
            email,
            order_date.date(),
            amount,
            random.choice(statuses),
            address,
            phone,
            random.choice(categories),
            discount,
            notes
        ))
    
    # Insert sample data
    cursor.executemany("""
        INSERT INTO orders (order_id, customer_id, customer_email, order_date, order_amount, 
                           status, shipping_address, phone_number, product_category, discount_percent, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_data)
    
    conn.commit()
    conn.close()
    print(f"✓ Sample dataset created at {db_path}")
    print(f"✓ Inserted {len(sample_data)} order records with realistic data patterns and quality issues")


def display_profile_results(profile: TableProfile):
    """Display profiling results in a user-friendly format."""
    print("\n" + "="*80)
    print("📊 DATA PROFILING RESULTS")
    print("="*80)
    
    # Table overview
    print(f"\n📋 TABLE OVERVIEW:")
    print(f"   • Table Name: {profile.table_name}")
    print(f"   • Database: {profile.database_name}")
    if profile.schema_name:
        print(f"   • Schema: {profile.schema_name}")
    print(f"   • Total Rows: {profile.total_rows:,}")
    print(f"   • Total Columns: {profile.total_columns}")
    print(f"   • Profile Generated: {profile.profile_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   • Analysis Time: {profile.execution_time:.2f} seconds")
    
    # Data quality scores
    print(f"\n📈 DATA QUALITY SCORES:")
    print(f"   • Completeness: {profile.completeness_score:.1f}%")
    print(f"   • Uniqueness: {profile.uniqueness_score:.1f}%")
    print(f"   • Validity: {profile.validity_score:.1f}%")
    print(f"   • Consistency: {profile.consistency_score:.1f}%")
    print(f"   • Overall Quality: {profile.overall_score:.1f}%")
    
    # Column-by-column analysis
    print(f"\n📝 COLUMN ANALYSIS:")
    for column_name, stats in profile.columns.items():
        print(f"\n   🔹 {column_name.upper()}")
        print(f"      Data Type: {stats.data_type}")
        print(f"      Rows: {stats.total_rows:,} | Non-null: {stats.non_null_count:,} | Nulls: {stats.null_count:,} ({stats.null_percentage:.1f}%)")
        print(f"      Unique: {stats.unique_count:,} ({stats.unique_percentage:.1f}%)")
        
        # Numeric statistics
        if stats.min_value is not None:
            print(f"      Range: {stats.min_value:.2f} to {stats.max_value:.2f}")
            print(f"      Mean: {stats.mean_value:.2f} | Median: {stats.median_value:.2f}")
            if stats.std_deviation:
                print(f"      Std Dev: {stats.std_deviation:.2f}")
            if stats.outliers:
                print(f"      Outliers: {len(stats.outliers)} detected (e.g., {stats.outliers[0]:.2f})")
        
        # String statistics
        if stats.min_length is not None:
            print(f"      Length: {stats.min_length} to {stats.max_length} chars (avg: {stats.avg_length:.1f})")
        
        # Date statistics
        if stats.min_date is not None:
            print(f"      Date Range: {stats.min_date} to {stats.max_date}")
        
        # Pattern detection
        if stats.patterns:
            print(f"      Patterns Detected:")
            for pattern in stats.patterns[:2]:  # Show top 2 patterns
                print(f"        • {pattern['pattern_name']}: {pattern['match_percentage']:.1f}% match ({pattern['match_count']} records)")
        
        # Most frequent values
        if stats.most_frequent:
            print(f"      Top Values:")
            for freq in stats.most_frequent[:3]:  # Show top 3
                value_str = str(freq['value'])
                if len(value_str) > 30:
                    value_str = value_str[:27] + "..."
                print(f"        • '{value_str}': {freq['count']} ({freq['percentage']:.1f}%)")
    
    # Warnings and recommendations
    if profile.warnings:
        print(f"\n⚠️  DATA QUALITY WARNINGS:")
        for warning in profile.warnings:
            print(f"   • {warning}")
    
    if profile.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in profile.recommendations:
            print(f"   • {recommendation}")
    
    print("\n" + "="*80)


def main():
    """Main demo function."""
    print("🚀 SQLTest Pro - Data Profiler Demo")
    print("="*50)
    
    try:
        # Step 1: Create comprehensive sample dataset
        print("\n1️⃣ Creating sample dataset...")
        create_sample_dataset()
        
        # Step 2: Setup database connection
        print("\n2️⃣ Setting up database connection...")
        # Simple connection manager for SQLite (same as field validator demo)
        class SimpleResult:
            def __init__(self, data_tuples, columns, execution_time=0.1):
                import pandas as pd
                if data_tuples:
                    self.data = pd.DataFrame(data_tuples, columns=columns)
                    self.is_empty = False
                else:
                    self.data = pd.DataFrame(columns=columns)
                    self.is_empty = True
                self.execution_time = execution_time
        
        class SimpleConnectionManager:
            def __init__(self, db_path: str):
                self.db_path = db_path
                # Create a simple config object
                class SimpleConfig:
                    default_database = "sqlite"
                self.config = SimpleConfig()
            
            def get_connection(self, db_name: str = None):
                return sqlite3.connect(self.db_path)
            
            def execute_query(self, query: str, params=None, database_name: str = None):
                import time
                start_time = time.time()
                
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
                
                execution_time = time.time() - start_time
                return SimpleResult(results, columns, execution_time)
            
            def get_adapter(self, db_name: str = None):
                # Simple adapter for SQLite
                class SimpleAdapter:
                    def get_driver_name(self):
                        return "sqlite"
                    def get_table_names(self):
                        return ["orders"]  # Our sample table
                return SimpleAdapter()
            
            def get_table_info(self, table_name: str, schema_name: str = None, database_name: str = None):
                # Return basic table info
                return {
                    'row_count': 1000,  # Our sample size
                    'columns': []  # Will be filled by profiler
                }
        
        connection_manager = SimpleConnectionManager("data/profiler_test_data.db")
        print("✓ Database connection ready")
        
        # Step 3: Initialize Data Profiler
        print("\n3️⃣ Initializing Data Profiler...")
        profiler = DataProfiler(connection_manager, sample_size=1000)
        print("✓ Data Profiler initialized")
        
        # Step 4: Profile the complete table
        print("\n4️⃣ Profiling complete table...")
        print("   📊 Analyzing 1000 orders with 11 columns...")
        table_profile = profiler.profile_table('orders')
        
        # Step 5: Display comprehensive results
        print("\n5️⃣ Analyzing results...")
        display_profile_results(table_profile)
        
        # Step 6: Profile specific columns for detailed analysis
        print("\n6️⃣ Detailed column analysis...")
        print("\n🔍 EMAIL COLUMN ANALYSIS:")
        email_stats = profiler.profile_column('orders', 'customer_email')
        print(f"   • Total emails: {email_stats.total_rows:,}")
        print(f"   • Valid emails: {email_stats.non_null_count:,} ({100-email_stats.null_percentage:.1f}%)")
        if email_stats.patterns:
            for pattern in email_stats.patterns:
                print(f"   • {pattern['pattern_name']}: {pattern['match_percentage']:.1f}% valid format")
        
        print("\n🔍 ORDER AMOUNT ANALYSIS:")
        amount_stats = profiler.profile_column('orders', 'order_amount')
        print(f"   • Range: ${amount_stats.min_value:.2f} to ${amount_stats.max_value:.2f}")
        print(f"   • Average: ${amount_stats.mean_value:.2f}")
        if amount_stats.outliers:
            print(f"   • Outliers detected: {len(amount_stats.outliers)} (e.g., ${amount_stats.outliers[0]:.2f})")
        
        # Step 7: Profile a custom query
        print("\n7️⃣ Query result profiling...")
        query = """
        SELECT 
            status,
            COUNT(*) as order_count,
            AVG(order_amount) as avg_amount,
            MIN(order_date) as first_order,
            MAX(order_date) as last_order
        FROM orders 
        WHERE order_amount > 100 
        GROUP BY status 
        ORDER BY order_count DESC
        """
        query_profile = profiler.profile_query(query)
        
        print(f"\n📈 QUERY RESULTS ANALYSIS:")
        print(f"   • Query executed in: {query_profile.execution_time:.3f} seconds")
        print(f"   • Rows returned: {query_profile.rows_returned}")
        print(f"   • Columns returned: {query_profile.columns_returned}")
        for col_name, col_stats in query_profile.columns.items():
            print(f"   • {col_name}: {col_stats.data_type} ({col_stats.unique_count} unique values)")
        
        # Step 8: Save results
        print("\n8️⃣ Saving profiling results...")
        results_path = Path("reports/data_profiling_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert to dict for JSON serialization
        import dataclasses
        profile_dict = dataclasses.asdict(table_profile)
        
        with open(results_path, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)
        
        print(f"✓ Table profile saved to {results_path}")
        
        # Save query profile too
        query_results_path = Path("reports/query_profiling_results.json")
        query_profile_dict = dataclasses.asdict(query_profile)
        
        with open(query_results_path, 'w') as f:
            json.dump(query_profile_dict, f, indent=2, default=str)
        
        print(f"✓ Query profile saved to {query_results_path}")
        
        print("\n🎉 Data Profiler demo completed successfully!")
        print("💡 Next steps:")
        print("   • Connect the profiler to CLI commands")
        print("   • Implement automated data quality monitoring")
        print("   • Add profile comparison capabilities")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
