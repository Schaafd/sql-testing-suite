# SQLTest Pro Demo

This demo showcases core functionality using a local SQLite database:
- Database discovery (tables/views)
- Data profiling (table and query)
- Field validation via YAML rules

Quick start

1) Setup the demo database and run the demo

bash
scripts/demo.sh

2) Run individual demo commands (once DB is created)

bash
# List tables and views
uv run sqltest db tables -d local_sqlite
uv run sqltest db views -d local_sqlite

# Profile a table
uv run sqltest profile --table users -d local_sqlite

# Profile a query
uv run sqltest profile --query "SELECT id, email, age FROM users WHERE age > 30" -d local_sqlite

# Run field validation
uv run sqltest validate --config examples/demo/validation_rules.yaml --table users -d local_sqlite

What the demo creates

- SQLite DB at ./test_data.db (referenced by examples/configs/database.yaml as local_sqlite)
- users table with some intentionally invalid data to show validation
- orders table
- user_orders view (join of users and orders)

Files

- examples/demo/data.sql: schema and seed data
- examples/demo/validation_rules.yaml: sample field validation rules
- examples/demo/README.md: this guide
- scripts/demo.sh: setup + demo runner

Troubleshooting

- If you see environment variable errors while using other database configs, ensure to use the local_sqlite database for this demo.
- Re-run scripts/demo.sh to recreate the database from scratch.

