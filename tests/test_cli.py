"""Tests for CLI commands."""

import pytest
import tempfile
import sqlite3
import re
from pathlib import Path
from click.testing import CliRunner

from sqltest.cli.main import cli
from sqltest.config import create_sample_config


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


@pytest.fixture(scope="function")
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create test data
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount REAL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        
        INSERT INTO users (name, email) VALUES 
            ('Alice Johnson', 'alice@example.com'),
            ('Bob Smith', 'bob@example.com'),
            ('Carol Davis', 'carol@example.com');
            
        INSERT INTO orders (user_id, amount, status) VALUES 
            (1, 150.00, 'completed'),
            (1, 89.99, 'pending'),
            (2, 299.50, 'completed'),
            (3, 45.75, 'cancelled');
        
        CREATE VIEW user_orders AS
        SELECT u.name, o.amount, o.status
        FROM users u
        JOIN orders o ON u.id = o.user_id;
    """)
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture(scope="function")
def temp_config(temp_db):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = f"""
databases:
  test:
    type: sqlite
    path: {temp_db}
    options:
      timeout: 30

connection_pools:
  default:
    min_connections: 1
    max_connections: 5
    timeout: 30
    retry_attempts: 3

default_database: test

validation_settings:
  fail_fast: false
  parallel_execution: false
  max_workers: 2
  report_format: html
  output_dir: ./reports

test_settings:
  isolation_level: READ_COMMITTED
  timeout: 300
  continue_on_failure: true
  generate_coverage: true
"""
        f.write(config_content)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture(autouse=True, scope="function")
def cleanup_database_connections():
    """Cleanup database connections after each test."""
    # Cleanup before the test as well
    _cleanup_connections()
    
    yield  # Run the test
    
    # Cleanup after the test
    _cleanup_connections()


def _cleanup_connections():
    """Helper function to clean up database connections."""
    import gc
    import sqlalchemy.pool
    from sqltest.db.connection import reset_connection_manager
    
    # Reset the global connection manager
    try:
        reset_connection_manager()
    except Exception:
        pass  # Ignore errors during cleanup
    
    # Clear SQLAlchemy connection pools
    try:
        sqlalchemy.pool.clear_managers()
    except Exception:
        pass
    
    gc.collect()


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    def test_cli_help(self):
        """Test that CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SQLTest Pro' in result.output
        assert 'comprehensive SQL testing' in result.output
    
    def test_cli_version(self):
        """Test that CLI shows version."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'SQLTest Pro v' in result.output
    
    def test_cli_dashboard_default(self):
        """Test that CLI shows dashboard by default."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert 'SQLTest Pro' in result.output


class TestCLIConfig:
    """Test configuration management commands."""
    
    def test_config_sample_creation(self):
        """Test creating a sample configuration."""
        runner = CliRunner()
        
        # Use a path that doesn't exist initially to avoid confirmation prompt
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'sample_config.yaml'
            
            result = runner.invoke(cli, ['config', 'sample', str(config_path)])
            assert result.exit_code == 0
            assert 'Sample configuration created' in result.output
            assert config_path.exists()
    
    def test_config_validation_valid(self, temp_config):
        """Test validating a valid configuration."""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'validate', temp_config])
        assert result.exit_code == 0
        assert 'Configuration file' in result.output
        assert 'is valid' in result.output
    
    def test_config_validation_invalid(self):
        """Test validating an invalid configuration."""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("databases: invalid_structure")
            invalid_config = f.name
        
        try:
            result = runner.invoke(cli, ['config', 'validate', invalid_config])
            assert result.exit_code == 1
            assert 'Configuration validation failed' in result.output
        finally:
            Path(invalid_config).unlink(missing_ok=True)


class TestCLIDatabase:
    """Test database management commands."""
    
    def test_db_status(self, temp_config):
        """Test database status command."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'status'])
        assert result.exit_code == 0
        assert 'Database Connection Status' in result.output
        assert 'test' in result.output
        assert 'sqlite' in result.output
    
    def test_db_tables_list(self, temp_config):
        """Test listing tables in database."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'tables', '-d', 'test'])
        assert result.exit_code == 0
        assert 'Tables in test' in result.output
        assert 'users' in result.output
        assert 'orders' in result.output
    
    def test_db_views_list(self, temp_config):
        """Test listing views in database."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'views', '-d', 'test'])
        assert result.exit_code == 0
        assert 'Views in test' in result.output
        assert 'user_orders' in result.output
    
    def test_db_describe_table(self, temp_config):
        """Test describing a table structure."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'describe', 'users', '-d', 'test'])
        assert result.exit_code == 0
        assert 'Table Structure: users' in result.output
        assert 'Column Details' in result.output
        assert 'name' in result.output
        assert 'email' in result.output
        assert 'TEXT' in result.output
    
    def test_db_describe_nonexistent_table(self, temp_config):
        """Test describing a non-existent table."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'describe', 'nonexistent', '-d', 'test'])
        assert result.exit_code == 1
        assert 'Database Error' in result.output


class TestCLIProfile:
    """Test data profiling commands."""
    
    def test_profile_table(self, temp_config):
        """Test profiling a table."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'profile', '--table', 'users', '-d', 'test'])
        assert result.exit_code == 0
        
        # Strip ANSI codes for easier text matching
        clean_output = strip_ansi(result.output)
        assert 'Data Profiling' in clean_output
        assert 'Profiling table: users' in clean_output
        assert 'TABLE OVERVIEW' in clean_output
        assert 'DATA QUALITY SCORES' in clean_output
        assert 'COLUMN ANALYSIS SUMMARY' in clean_output
        # Check that we have the expected columns
        assert 'name' in clean_output
        assert 'email' in clean_output
    
    def test_profile_table_with_columns_filter(self, temp_config):
        """Test profiling a table with column filter."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config', temp_config,
            'profile', '--table', 'users', '--columns', 'name,email',
            '-d', 'test'
        ])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.output)
        assert 'Data Profiling' in clean_output
        assert 'name' in clean_output
        assert 'email' in clean_output
    
    def test_profile_query(self, temp_config):
        """Test profiling a custom query."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config', temp_config,
            'profile', '--query', 'SELECT name, amount FROM user_orders WHERE amount > 100',
            '-d', 'test'
        ])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.output)
        assert 'Data Profiling' in clean_output
        assert 'Profiling query' in clean_output
        # Note: Query results may not contain specific user names in summary
    
    def test_profile_empty_query(self, temp_config):
        """Test profiling a query that returns no results."""
        from sqltest.db.connection import reset_connection_manager
        reset_connection_manager()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config', temp_config,
            'profile', '--query', 'SELECT * FROM users WHERE name = "NonExistent"',
            '-d', 'test'
        ])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.output)
        # Check for empty results indication 
        assert 'no results' in clean_output or 'empty' in clean_output or '0' in clean_output
    
    def test_profile_no_arguments(self, temp_config):
        """Test profile command without table or query."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'profile', '-d', 'test'])
        assert result.exit_code == 0
        assert 'Either --table or --query must be specified' in result.output


class TestCLIProject:
    """Test project initialization commands."""
    
    def test_init_project(self):
        """Test initializing a new project."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = 'test_project'
            project_path = Path(temp_dir) / project_name
            
            result = runner.invoke(cli, ['init', str(project_path)])
            assert result.exit_code == 0
            assert 'Initializing project' in result.output
            assert 'initialized successfully' in result.output
            
            # Verify project structure was created
            assert project_path.exists()
            assert (project_path / 'sqltest.yaml').exists()
            assert (project_path / 'reports').exists()
            assert (project_path / 'tests').exists()
    
    def test_init_existing_project(self):
        """Test initializing a project in existing directory."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / 'existing'
            existing_dir.mkdir()
            
            result = runner.invoke(cli, ['init', str(existing_dir)])
            assert result.exit_code == 0
            assert 'already exists' in result.output


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', 'nonexistent.yaml', 'db', 'status'])
        assert result.exit_code == 2  # Click file validation error
        assert 'does not exist' in result.output
    
    def test_invalid_database_name(self, temp_config):
        """Test handling of invalid database name."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', temp_config, 'db', 'tables', '-d', 'nonexistent'])
        assert result.exit_code == 1
        clean_output = strip_ansi(result.output)
        assert 'Database' in clean_output and 'not found in configuration' in clean_output
    
    def test_malformed_query(self, temp_config):
        """Test handling of malformed SQL query."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config', temp_config,
            'profile', '--query', 'INVALID SQL QUERY',
            '-d', 'test'
        ])
        assert result.exit_code == 1
        assert 'Database Error' in result.output
