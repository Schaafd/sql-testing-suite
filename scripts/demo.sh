#!/bin/bash
# SQLTest Pro Demo Script
# Creates demo database and runs through key functionality

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Change to project root
cd "$(dirname "$0")/.."

print_step "SQLTest Pro Demo Setup"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first."
    exit 1
fi

print_info "Setting up demo database..."

# Create data directory if it doesn't exist
mkdir -p ./data

# Remove existing database if it exists
if [ -f "./data/test_data.db" ]; then
    rm "./data/test_data.db"
    print_info "Removed existing demo database"
fi

# Create the demo database using sqlite3
sqlite3 "./data/test_data.db" < "./examples/demo/data.sql"
print_success "Demo database created at ./data/test_data.db"

print_step "Demo 1: Database Discovery"

print_info "Listing all tables in the database..."
uv run sqltest --config examples/demo/database.yaml db tables

print_info "Listing all views in the database..."
uv run sqltest --config examples/demo/database.yaml db views

print_info "Describing the users table..."
uv run sqltest --config examples/demo/database.yaml db describe users

print_step "Demo 2: Data Profiling"

print_info "Profiling the users table..."
uv run sqltest --config examples/demo/database.yaml profile --table users

print_info "Profiling a custom query..."
uv run sqltest --config examples/demo/database.yaml profile --query "SELECT id, email, age, status FROM users WHERE age > 30"

print_step "Demo 3: Field Validation"

print_info "Running field validation on users table with comprehensive rules..."
uv run sqltest --config examples/demo/database.yaml validate --config examples/demo/validation_rules.yaml --table users --rule-set user_data_validation

print_info "Running field validation on orders table..."
uv run sqltest --config examples/demo/database.yaml validate --config examples/demo/validation_rules.yaml --table orders --rule-set orders_data_validation

print_step "Demo Complete!"

print_success "All demo commands completed successfully"
print_info "Demo database is available at ./data/test_data.db"
print_info "You can run individual commands from the README or explore the CLI with:"
print_info "  uv run sqltest --help"
print_info "  uv run sqltest --config examples/demo/database.yaml db --help"
print_info "  uv run sqltest --config examples/demo/database.yaml profile --help"
print_info "  uv run sqltest --config examples/demo/database.yaml validate --help"

echo
print_info "Sample commands to try:"
echo "  # List all available databases"
echo "  uv run sqltest --config examples/demo/database.yaml db info"
echo
echo "  # Profile a specific query"
echo "  uv run sqltest --config examples/demo/database.yaml profile --query \"SELECT * FROM user_orders LIMIT 5\""
echo
echo "  # Generate a sample validation config"
echo "  uv run sqltest validate --generate --output my_validation_rules.yaml"
echo
echo "For more details, see examples/demo/README.md"
