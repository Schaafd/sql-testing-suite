# SQLTest Pro - CLI Interface Mockup

## Main Dashboard

```
╭───────────────────────── SQLTest Pro v1.0.0 ─────────────────────────╮
│                                                                       │
│  Connected to: dev (PostgreSQL)                    ⚡ Ready           │
│                                                                       │
│  📊 Last Profile: 2024-01-15 14:32 (users table)                    │
│  ✅ Last Test Run: 2024-01-15 15:45 (42/45 passed)                  │
│  ⚠️  Active Issues: 3 critical, 5 medium                             │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯

What would you like to do?

  ❯ 📊 Profile Data
    ✓  Run Validations  
    🧪 Execute Unit Tests
    📄 Generate Reports
    ⚙️  Configure Settings
    📚 View Documentation
    🚪 Exit

Use ↑/↓ arrows to navigate, Enter to select, or type a command
```

## Data Profiling Interface

```
╭─────────────────────── Data Profiler ───────────────────────╮
│                                                              │
│  Select profiling target:                                    │
│                                                              │
│  ○ Single Table                                             │
│  ● Multiple Tables                                          │
│  ○ Custom Query                                             │
│                                                              │
│  Tables to profile:                                         │
│  ┌─────────────────────────────────────────┐               │
│  │ ☑ users                                  │               │
│  │ ☑ orders                                 │               │
│  │ ☐ products                               │               │
│  │ ☐ order_items                            │               │
│  └─────────────────────────────────────────┘               │
│                                                              │
│  Profile Options:                                            │
│  ☑ Basic Statistics    ☑ Pattern Detection                  │
│  ☑ Null Analysis      ☑ Outlier Detection                  │
│  ☑ Cardinality        ☐ Column Correlation                 │
│                                                              │
│  [Start Profiling] [Back]                                    │
╰──────────────────────────────────────────────────────────────╯
```

## Progress Display

```
Profiling Database Tables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02

📊 users table
  ├─ Analyzing structure... ✓
  ├─ Computing statistics... ✓
  ├─ Detecting patterns... ✓
  └─ Identifying outliers... ✓

📊 orders table
  ├─ Analyzing structure... ⣾ 
  ├─ Computing statistics... ⣽ [42%]
  ├─ Detecting patterns... ⣻
  └─ Identifying outliers... ⣯

┌─────────────── Live Statistics ───────────────┐
│ Rows Processed: 125,420 / 298,000            │
│ Issues Found: 3                               │
│ Elapsed Time: 00:01:42                       │
│ ETA: 00:02:15                                │
└───────────────────────────────────────────────┘
```

## Validation Results

```
╭────────────────── Validation Results ─────────────────────╮
│                                                           │
│  Validation Suite: production_checks                      │
│  Started: 2024-01-15 16:00:00                            │
│  Duration: 3m 42s                                         │
│                                                           │
│  Summary:                                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │ ✅ Passed:  142                              │        │
│  │ ❌ Failed:   8                               │        │
│  │ ⚠️  Warning:  3                              │        │
│  │ ⏭️  Skipped:  2                              │        │
│  └─────────────────────────────────────────────┘        │
│                                                           │
│  Critical Failures:                                       │
│                                                           │
│  ❌ Order totals mismatch                                │
│     Table: orders                                         │
│     Failed records: 127                                   │
│     Query: SELECT o.order_id FROM orders o...           │
│     [View Details]                                        │
│                                                           │
│  ❌ Invalid email formats                                 │
│     Table: users                                          │
│     Column: email                                         │
│     Failed records: 43                                    │
│     Pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ │
│     [View Details]                                        │
│                                                           │
│  Actions:                                                 │
│  [📄 Export Report] [📧 Email Team] [🔄 Re-run Failed]   │
│                                                           │
╰───────────────────────────────────────────────────────────╯
```

## Unit Test Execution

```
╭──────────────── SQL Unit Test Runner ────────────────╮
│                                                       │
│  Test Suite: e-commerce-tests v1.0.0                │
│  Database: test                                       │
│                                                       │
│  Running Tests...                                     │
│                                                       │
│  Stored Procedures                                    │
│  ├─ ✅ calculate_order_total ................. 0.12s │
│  ├─ ✅ update_inventory ...................... 0.23s │
│  └─ ❌ process_returns ....................... 0.45s │
│       └─ AssertionError: Expected 2 rows, got 0     │
│                                                       │
│  SQL Functions                                        │
│  ├─ ✅ get_customer_lifetime_value ........... 0.08s │
│  ├─ ✅ format_phone_number (3 cases) ......... 0.15s │
│  └─ ⏭️  calculate_tax (skipped: not implemented)     │
│                                                       │
│  Views                                                │
│  ├─ ✅ active_customers ...................... 0.05s │
│  └─ ✅ order_summary ......................... 0.11s │
│                                                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% 8/10 tests   │
│                                                       │
│  Coverage Report:                                     │
│  ┌───────────────────────────────────────┐          │
│  │ Functions:   12/15  (80.0%) ████████░ │          │
│  │ Procedures:   8/10  (80.0%) ████████░ │          │
│  │ Views:        5/5  (100.0%) ██████████ │          │
│  │ Triggers:     3/4   (75.0%) ███████▌░ │          │
│  └───────────────────────────────────────┘          │
│                                                       │
╰───────────────────────────────────────────────────────╯
```

## Interactive Error Details

```
╭────────────── Test Failure Details ──────────────╮
│                                                   │
│  Test: process_returns                           │
│  Type: Stored Procedure                          │
│                                                   │
│  Setup SQL:                                      │
│  ┌─────────────────────────────────────────┐    │
│  │ CREATE TEMP TABLE test_returns (...)    │    │
│  │ INSERT INTO test_returns VALUES (...)   │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  Test Query:                                     │
│  ┌─────────────────────────────────────────┐    │
│  │ CALL process_returns(order_id => 123)   │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  Expected Result:                                │
│  ┌─────────────────────────────────────────┐    │
│  │ return_id │ status     │ refund_amount │    │
│  │ 1         │ processed  │ 99.99         │    │
│  │ 2         │ processed  │ 149.99        │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  Actual Result:                                  │
│  ┌─────────────────────────────────────────┐    │
│  │ (empty result set)                      │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  Debug Info:                                     │
│  • Execution time: 0.45s                        │
│  • Error: No rows returned                      │
│  • Last SQL state: 00000                        │
│                                                   │
│  [🔍 Debug] [📋 Copy SQL] [↻ Re-run] [→ Next]   │
│                                                   │
╰───────────────────────────────────────────────────╯
```

## Report Generation

```
╭──────────────── Report Generator ────────────────╮
│                                                   │
│  Select Report Type:                             │
│                                                   │
│  ○ Data Quality Summary                         │
│  ● Test Coverage Report                         │
│  ○ Validation History                           │
│  ○ Performance Benchmarks                       │
│  ○ Custom Report                                │
│                                                   │
│  Report Options:                                 │
│                                                   │
│  Format:        [●] HTML  [ ] PDF  [ ] JSON     │
│  Time Range:    Last 7 days ▼                   │
│  Include:       ☑ Passed Tests                  │
│                 ☑ Failed Tests                  │
│                 ☑ Charts & Graphs               │
│                 ☐ Detailed SQL Queries          │
│                                                   │
│  Output:        ./reports/coverage_2024-01-15   │
│                                                   │
│  Preview:                                        │
│  ┌─────────────────────────────────────────┐   │
│  │ 📊 Test Coverage Report                 │   │
│  │                                         │   │
│  │ Overall Coverage: 82.5%                 │   │
│  │ ████████████████████░░░░ 82.5%        │   │
│  │                                         │   │
│  │ By Component:                           │   │
│  │ • Functions:  80.0% ████████░          │   │
│  │ • Procedures: 85.0% ████████▌          │   │
│  │ • Views:     100.0% ██████████         │   │
│  └─────────────────────────────────────────┘   │
│                                                   │
│  [Generate Report] [Preview] [Cancel]            │
│                                                   │
╰───────────────────────────────────────────────────╯
```

## Configuration Editor

```
╭────────────── Configuration Editor ──────────────╮
│                                                   │
│  Current Configuration: validations.yaml         │
│                                                   │
│  ┌─ Field Validations ─────────────────────┐    │
│  │ Table: users                             │    │
│  │ Column: email                            │    │
│  │                                          │    │
│  │ Rules:                                   │    │
│  │ ┌────────────────────────────────┐      │    │
│  │ │ ✓ not_null                     │      │    │
│  │ │ ✓ unique                       │      │    │
│  │ │ ✓ regex: email pattern         │      │    │
│  │ │ + Add Rule...                  │      │    │
│  │ └────────────────────────────────┘      │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  Quick Actions:                                   │
│  [+ Add Table] [+ Add Rule] [⚡ Validate Now]    │
│                                                   │
│  YAML Preview:                                    │
│  ┌─────────────────────────────────────────┐    │
│  │ field_validations:                      │    │
│  │   - table: users                        │    │
│  │     validations:                        │    │
│  │       - column: email                   │    │
│  │         rules:                          │    │
│  │           - type: not_null              │    │
│  │           - type: unique                │    │
│  │           - type: regex                 │    │
│  │             pattern: '^[a-zA-Z0-9...    │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  [💾 Save] [✓ Validate YAML] [↻ Reset] [← Back] │
│                                                   │
╰───────────────────────────────────────────────────╯
```

## Help System

```
╭───────────────────── SQLTest Pro Help ─────────────────────╮
│                                                             │
│  Command: sqltest profile                                   │
│                                                             │
│  Description:                                               │
│  Profile SQL tables to analyze data quality, patterns,     │
│  and statistics.                                            │
│                                                             │
│  Usage:                                                     │
│  sqltest profile [OPTIONS] [TABLE]                        │
│                                                             │
│  Options:                                                   │
│    --table TEXT       Table name to profile               │
│    --query TEXT       Custom SQL query to profile         │
│    --columns LIST     Specific columns (default: all)     │
│    --sample INT       Sample size (default: 10000)        │
│    --output FORMAT    Output format: table|json|html      │
│    --save PATH        Save results to file                │
│                                                             │
│  Examples:                                                  │
│                                                             │
│  # Profile a single table                                  │
│  $ sqltest profile --table users                          │
│                                                             │
│  # Profile specific columns                                │
│  $ sqltest profile --table orders --columns total,date    │
│                                                             │
│  # Profile with custom query                              │
│  $ sqltest profile --query "SELECT * FROM users           │
│    WHERE created_at > '2024-01-01'"                      │
│                                                             │
│  # Save results as HTML report                            │
│  $ sqltest profile --table products --output html         │
│    --save reports/products_profile.html                   │
│                                                             │
│  Related Commands:                                          │
│  • validate - Run validation rules                         │
│  • test    - Execute unit tests                          │
│  • report  - Generate reports                             │
│                                                             │
│  [← Back to Menu] [📚 Full Docs] [🔍 Search Help]        │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
```
