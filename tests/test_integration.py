"""Integration tests for configuration and database systems."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest

from sqltest.config.models import DatabaseConfig, DatabaseType, SQLTestConfig
from sqltest.config.parser import ConfigParser
from sqltest.db.connection import ConnectionManager
from sqltest.db.adapters.sqlite import SQLiteAdapter
from sqltest.exceptions import ConfigurationError, DatabaseError


class TestConfigurationSystem:
    """Test configuration parsing and validation."""
    
    def test_config_parser_basic(self):
        """Test basic configuration parsing."""
        config_content = {
            'databases': {
                'test': {
                    'type': 'sqlite',
                    'path': ':memory:'
                }
            },
            'default_database': 'test'
        }
        
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.safe_dump(config_content, f)
            config_path = f.name
        
        try:
            config = parser.load_config(config_path)
            
            assert isinstance(config, SQLTestConfig)
            assert 'test' in config.databases
            assert config.default_database == 'test'
            assert config.databases['test'].type == DatabaseType.SQLITE
            
        finally:
            os.unlink(config_path)
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        config_content = {
            'databases': {
                'test': {
                    'type': 'postgresql',
                    'host': 'localhost',
                    'database': 'test_db',
                    'username': 'user',
                    'password': '${TEST_DB_PASSWORD}'
                }
            }
        }
        
        parser = ConfigParser()
        
        # Set environment variable
        os.environ['TEST_DB_PASSWORD'] = 'secret123'
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                import yaml
                yaml.safe_dump(config_content, f)
                config_path = f.name
            
            try:
                config = parser.load_config(config_path)
                assert config.databases['test'].password == 'secret123'
                
            finally:
                os.unlink(config_path)
                
        finally:
            del os.environ['TEST_DB_PASSWORD']
    
    def test_environment_variable_with_default(self):
        """Test environment variable with default value."""
        config_content = {
            'databases': {
                'test': {
                    'type': 'sqlite',
                    'path': '${SQLITE_PATH:-./default.db}'
                }
            }
        }
        
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.safe_dump(config_content, f)
            config_path = f.name
        
        try:
            config = parser.load_config(config_path)
            assert config.databases['test'].path == './default.db'
            
        finally:
            os.unlink(config_path)
    
    def test_config_validation_errors(self):
        """Test configuration validation errors."""
        # Missing required field
        invalid_config = {
            'databases': {
                'test': {
                    'type': 'postgresql',
                    'host': 'localhost'
                    # Missing required fields: database, username, password
                }
            }
        }
        
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.safe_dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError):
                parser.load_config(config_path)
                
        finally:
            os.unlink(config_path)
    
    def test_sample_config_creation(self):
        """Test sample configuration creation."""
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            sample_path = f.name
        
        try:
            parser.create_sample_config(sample_path)
            
            # Load the created sample config
            config = parser.load_config(sample_path)
            
            assert isinstance(config, SQLTestConfig)
            assert len(config.databases) >= 2  # Should have at least dev and test
            assert config.default_database is not None
            
        finally:
            os.unlink(sample_path)


class TestDatabaseAdapters:
    """Test database adapter functionality."""
    
    @pytest.fixture
    def temp_sqlite_db(self):
        """Create temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_sqlite_adapter_basic_operations(self, temp_sqlite_db):
        """Test basic SQLite adapter operations."""
        config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            path=temp_sqlite_db
        )
        
        adapter = SQLiteAdapter(config)
        
        # Test connection
        assert adapter.test_connection() is True
        
        # Create table
        create_result = adapter.execute_query("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value REAL
            )
        """, fetch_results=False)
        
        assert create_result.rows_affected >= 0
        assert create_result.execution_time >= 0
        
        # Insert data
        insert_result = adapter.execute_query("""
            INSERT INTO test_table (name, value) VALUES 
            ('test1', 1.5),
            ('test2', 2.5)
        """, fetch_results=False)
        
        assert insert_result.rows_affected == 2
        
        # Query data
        select_result = adapter.execute_query("SELECT * FROM test_table ORDER BY id")
        
        assert not select_result.is_empty
        assert select_result.row_count == 2
        assert len(select_result.columns) == 3
        
        # Check table info
        table_info = adapter.get_table_info('test_table')
        
        assert table_info['table_name'] == 'test_table'
        assert len(table_info['columns']) == 3
        assert table_info['row_count'] == 2
        
        # Check utility methods
        table_names = adapter.get_table_names()
        assert 'test_table' in table_names
        
        adapter.close()
    
    def test_sqlite_adapter_error_handling(self, temp_sqlite_db):
        """Test SQLite adapter error handling."""
        config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            path=temp_sqlite_db
        )
        
        adapter = SQLiteAdapter(config)
        
        # Test invalid SQL
        with pytest.raises(DatabaseError):
            adapter.execute_query("INVALID SQL STATEMENT")
        
        # Test non-existent table
        with pytest.raises(DatabaseError):
            adapter.get_table_info('non_existent_table')
        
        adapter.close()
    
    def test_sqlite_path_handling(self):
        """Test SQLite path handling."""
        # Test relative path
        config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            path='./test_relative.db'
        )
        
        adapter = SQLiteAdapter(config)
        connection_string = adapter.build_connection_string()
        
        assert 'sqlite:///' in connection_string
        assert connection_string.endswith('test_relative.db')
        
        # Test absolute path
        abs_path = os.path.abspath('./test_absolute.db')
        config.path = abs_path
        
        connection_string = adapter.build_connection_string()
        assert abs_path in connection_string
        
        adapter.close()


class TestConnectionManager:
    """Test connection manager functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        config = SQLTestConfig(
            databases={
                'primary': DatabaseConfig(
                    type=DatabaseType.SQLITE,
                    path=db_path
                ),
                'secondary': DatabaseConfig(
                    type=DatabaseType.SQLITE,
                    path=':memory:'
                )
            },
            default_database='primary'
        )
        
        yield config, db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_connection_manager_basic(self, test_config):
        """Test basic connection manager operations."""
        config, db_path = test_config
        
        manager = ConnectionManager(config)
        
        # Test getting adapter
        adapter = manager.get_adapter('primary')
        assert isinstance(adapter, SQLiteAdapter)
        
        # Test default adapter
        default_adapter = manager.get_adapter()
        assert default_adapter is adapter  # Should be the same instance
        
        # Test connection test
        test_result = manager.test_connection('primary')
        assert test_result['status'] == 'success'
        assert test_result['database'] == 'primary'
        assert test_result['database_type'] == 'sqlite'
        assert 'response_time' in test_result
        
        # Test query execution
        result = manager.execute_query(
            "SELECT 'Hello World' as message",
            db_name='primary'
        )
        
        assert not result.is_empty
        assert result.row_count == 1
        
        # Test database info
        db_info = manager.get_database_info('primary')
        assert db_info['database_name'] == 'primary'
        assert db_info['database_type'] == 'sqlite'
        assert db_info['path'] == db_path
        
        # Test connection status
        status = manager.get_connection_status()
        assert status['total_configured'] == 2
        assert status['total_active'] == 1  # Only primary is active
        assert status['default_database'] == 'primary'
        
        manager.close_all_connections()
    
    def test_connection_manager_multiple_connections(self, test_config):
        """Test connection manager with multiple connections."""
        config, db_path = test_config
        
        manager = ConnectionManager(config)
        
        # Get both adapters
        primary_adapter = manager.get_adapter('primary')
        secondary_adapter = manager.get_adapter('secondary')
        
        assert primary_adapter is not secondary_adapter
        
        # Test all connections
        all_results = manager.test_all_connections()
        
        assert len(all_results) == 2
        assert all_results['primary']['status'] == 'success'
        assert all_results['secondary']['status'] == 'success'
        
        # Test connection status
        status = manager.get_connection_status()
        assert status['total_active'] == 2
        
        # Close specific connection
        manager.close_connection('secondary')
        
        status = manager.get_connection_status()
        assert status['total_active'] == 1
        
        manager.close_all_connections()
    
    def test_connection_manager_errors(self, test_config):
        """Test connection manager error handling."""
        config, db_path = test_config
        
        manager = ConnectionManager(config)
        
        # Test non-existent database
        with pytest.raises(DatabaseError):
            manager.get_adapter('non_existent')
        
        # Test invalid query
        with pytest.raises(DatabaseError):
            manager.execute_query("INVALID SQL", db_name='primary')
        
        manager.close_all_connections()


@pytest.mark.integration
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_config_to_database_workflow(self):
        """Test complete workflow from config loading to database operations."""
        # Create temporary config file
        config_content = {
            'databases': {
                'analytics': {
                    'type': 'sqlite',
                    'path': '${ANALYTICS_DB_PATH:-./analytics.db}'
                }
            },
            'default_database': 'analytics',
            'validation_settings': {
                'fail_fast': False,
                'parallel_execution': True,
                'max_workers': 2
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.safe_dump(config_content, f)
            config_path = f.name
        
        try:
            # Load configuration
            parser = ConfigParser()
            config = parser.load_config(config_path)
            
            # Create connection manager
            manager = ConnectionManager(config)
            
            # Set up database
            manager.execute_query("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    active BOOLEAN DEFAULT 1
                )
            """, fetch_results=False)
            
            # Insert test data
            manager.execute_query("""
                INSERT OR REPLACE INTO users (id, name, email, active) VALUES
                (1, 'Alice Johnson', 'alice@example.com', 1),
                (2, 'Bob Smith', 'bob@example.com', 1),
                (3, 'Charlie Brown', 'charlie@example.com', 0)
            """, fetch_results=False)
            
            # Perform analysis queries
            active_users = manager.execute_query(
                "SELECT COUNT(*) as count FROM users WHERE active = 1"
            )
            
            assert not active_users.is_empty
            assert active_users.data.iloc[0]['count'] == 2
            
            # Test table info
            table_info = manager.get_table_info('users')
            assert table_info['row_count'] == 3
            assert len(table_info['columns']) == 4
            
            # Test database info
            db_info = manager.get_database_info()
            assert db_info['database_type'] == 'sqlite'
            
            manager.close_all_connections()
            
        finally:
            os.unlink(config_path)
            # Clean up database file
            db_path = Path('./analytics.db')
            if db_path.exists():
                db_path.unlink()
