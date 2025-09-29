"""Comprehensive tests for business rules engine."""

import pytest
import tempfile
import os
import yaml
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from sqltest.modules.business_rules import BusinessRuleValidator
from sqltest.modules.business_rules.engine import BusinessRuleEngine
from sqltest.modules.business_rules.config_loader import BusinessRuleConfigLoader
from sqltest.modules.business_rules.models import (
    BusinessRule,
    RuleSet,
    RuleType,
    RuleSeverity,
    ValidationScope,
    ValidationContext,
    RuleResult,
    RuleStatus
)
from sqltest.db.connection import ConnectionManager
from sqltest.exceptions import ValidationError, ConfigurationError


class TestBusinessRuleEngine:
    """Test the core business rule engine functionality."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager for testing."""
        manager = Mock(spec=ConnectionManager)
        adapter = Mock()
        adapter.execute_query.return_value = Mock(data=pd.DataFrame())
        manager.get_adapter.return_value = adapter
        return manager

    @pytest.fixture
    def rule_engine(self, mock_connection_manager):
        """Create a rule engine instance for testing."""
        return BusinessRuleEngine(mock_connection_manager, max_workers=2)

    @pytest.fixture
    def sample_rule(self):
        """Create a sample business rule for testing."""
        return BusinessRule(
            name="test_rule",
            description="Test rule for validation",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT COUNT(*) as violation_count FROM users WHERE email IS NULL",
            severity=RuleSeverity.CRITICAL,
            scope=ValidationScope.TABLE,
            enabled=True,
            timeout_seconds=30,
            dependencies=[],
            expected_violation_count=0
        )

    @pytest.fixture
    def sample_rule_set(self, sample_rule):
        """Create a sample rule set for testing."""
        return RuleSet(
            name="test_rule_set",
            description="Test rule set",
            rules=[sample_rule],
            parallel_execution=True,
            max_concurrent_rules=2
        )

    @pytest.fixture
    def validation_context(self):
        """Create validation context for testing."""
        return ValidationContext(
            database_name="test_db",
            schema_name="public",
            table_name="users"
        )

    def test_engine_initialization(self, mock_connection_manager):
        """Test engine initialization."""
        engine = BusinessRuleEngine(mock_connection_manager, max_workers=5)
        assert engine.connection_manager == mock_connection_manager
        assert engine.max_workers == 5
        assert engine._rule_cache == {}
        assert engine._dependency_graph == {}

    def test_execute_single_rule_success(self, rule_engine, sample_rule, validation_context):
        """Test successful execution of a single rule."""
        # Mock adapter to return no violations
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        result = rule_engine.execute_rule(sample_rule, validation_context)

        assert isinstance(result, RuleResult)
        assert result.rule_name == "test_rule"
        assert result.status == RuleStatus.PASSED
        assert result.passed is True
        assert len(result.violations) == 0

    def test_execute_single_rule_with_violations(self, rule_engine, sample_rule, validation_context):
        """Test rule execution that finds violations."""
        # Mock adapter to return violations
        violation_data = pd.DataFrame({
            'violation_count': [5],
            'message': ['Email violations found'],
            'table_name': ['users'],
            'column_name': ['email']
        })
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = violation_data

        result = rule_engine.execute_rule(sample_rule, validation_context)

        assert result.status == RuleStatus.FAILED
        assert result.passed is False
        assert len(result.violations) == 2  # One from SQL result, one from expected count mismatch

        # First violation should be from the SQL result
        sql_violation = result.violations[0]
        assert sql_violation.violation_count == 5
        assert sql_violation.message == 'Email violations found'

        # Second violation should be from expected count check
        expected_violation = result.violations[1]
        assert "Expected 0 violations but found 5" in expected_violation.message

    def test_execute_disabled_rule(self, rule_engine, sample_rule, validation_context):
        """Test execution of disabled rule."""
        sample_rule.enabled = False

        result = rule_engine.execute_rule(sample_rule, validation_context)

        assert result.status == RuleStatus.SKIPPED
        assert result.passed is True
        assert result.message == "Rule is disabled"

    def test_execute_rule_timeout(self, rule_engine, sample_rule, validation_context):
        """Test rule execution timeout."""
        # Mock adapter to simulate timeout
        rule_engine.connection_manager.get_adapter.return_value.execute_query.side_effect = TimeoutError("Query timed out")

        result = rule_engine.execute_rule(sample_rule, validation_context)

        assert result.status == RuleStatus.ERROR
        assert result.passed is False
        assert "timed out" in result.message.lower()

    def test_execute_rule_with_exception(self, rule_engine, sample_rule, validation_context):
        """Test rule execution with exception."""
        # Mock adapter to raise exception
        rule_engine.connection_manager.get_adapter.return_value.execute_query.side_effect = Exception("Database error")

        result = rule_engine.execute_rule(sample_rule, validation_context)

        assert result.status == RuleStatus.ERROR
        assert result.passed is False
        assert "Database error" in result.message

    def test_execute_rule_set_sequential(self, rule_engine, sample_rule_set, validation_context):
        """Test sequential execution of rule set."""
        # Mock adapter to return no violations
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        summary = rule_engine.execute_rule_set(sample_rule_set, validation_context, parallel=False)

        assert summary.rule_set_name == "test_rule_set"
        assert summary.total_rules == 1
        assert summary.rules_passed == 1
        assert summary.rules_failed == 0

    def test_execute_rule_set_parallel(self, rule_engine, sample_rule_set, validation_context):
        """Test parallel execution of rule set."""
        # Mock adapter to return no violations
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        summary = rule_engine.execute_rule_set(sample_rule_set, validation_context, parallel=True)

        assert summary.rule_set_name == "test_rule_set"
        assert summary.total_rules == 1
        assert summary.rules_passed == 1

    def test_rule_dependencies(self, rule_engine, validation_context):
        """Test rule dependency management."""
        # Create rules with dependencies
        rule1 = BusinessRule(
            name="rule1",
            description="First rule",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT 1",
            severity=RuleSeverity.WARNING,
            scope=ValidationScope.TABLE,
            enabled=True,
            dependencies=[]
        )

        rule2 = BusinessRule(
            name="rule2",
            description="Second rule",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT 2",
            severity=RuleSeverity.WARNING,
            scope=ValidationScope.TABLE,
            enabled=True,
            dependencies=["rule1"]
        )

        rule_set = RuleSet(
            name="dependency_test",
            description="Test dependencies",
            rules=[rule2, rule1],  # Rule2 first to test ordering
            parallel_execution=False
        )

        # Mock successful execution
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        summary = rule_engine.execute_rule_set(rule_set, validation_context)

        # Both rules should execute successfully
        assert summary.rules_passed == 2
        assert summary.rules_failed == 0

        # Check execution order in results
        result_names = [result.rule_name for result in summary.results]
        assert result_names.index("rule1") < result_names.index("rule2")

    def test_fail_fast_execution(self, rule_engine, validation_context):
        """Test fail-fast execution mode."""
        # Create rules where first fails
        rule1 = BusinessRule(
            name="failing_rule",
            description="This rule will fail",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT 1 as violation_count",
            severity=RuleSeverity.CRITICAL,
            scope=ValidationScope.TABLE,
            enabled=True,
            expected_violation_count=0  # Expect no violations, but query returns 1
        )

        rule2 = BusinessRule(
            name="second_rule",
            description="This rule should be skipped",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT 0 as violation_count",
            severity=RuleSeverity.WARNING,
            scope=ValidationScope.TABLE,
            enabled=True
        )

        rule_set = RuleSet(
            name="fail_fast_test",
            description="Test fail-fast",
            rules=[rule1, rule2],
            parallel_execution=False
        )

        # Mock first rule to return violations
        violation_data = pd.DataFrame({'violation_count': [1]})
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = violation_data

        summary = rule_engine.execute_rule_set(rule_set, validation_context, fail_fast=True)

        # Only first rule should have executed
        assert len(summary.results) == 1
        assert summary.results[0].rule_name == "failing_rule"
        assert summary.results[0].status == RuleStatus.FAILED

    def test_validate_table_convenience_method(self, rule_engine, sample_rule):
        """Test the validate_table convenience method."""
        # Mock successful execution
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        summary = rule_engine.validate_table(
            database_name="test_db",
            table_name="users",
            rules=[sample_rule]
        )

        assert summary.rule_set_name == "users_validation"
        assert summary.total_rules == 1
        assert summary.rules_passed == 1

    def test_validate_query_convenience_method(self, rule_engine, sample_rule):
        """Test the validate_query convenience method."""
        # Mock successful execution
        rule_engine.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

        summary = rule_engine.validate_query(
            database_name="test_db",
            query="SELECT * FROM users",
            rules=[sample_rule],
            query_name="custom_validation"
        )

        assert summary.rule_set_name == "custom_validation_validation"
        assert summary.total_rules == 1


class TestBusinessRuleConfigLoader:
    """Test the business rule configuration loader."""

    @pytest.fixture
    def config_loader(self):
        """Create config loader instance."""
        return BusinessRuleConfigLoader()

    def test_environment_variable_interpolation(self, config_loader):
        """Test environment variable interpolation in config."""
        config_text = """
rule_set:
  name: ${RULE_SET_NAME:default_name}
  description: "Test with env var: ${TEST_VAR}"
rules:
  - name: test_rule
    query: "SELECT * FROM ${TABLE_NAME:users}"
"""
        # Set environment variables
        os.environ['RULE_SET_NAME'] = 'production_rules'
        os.environ['TEST_VAR'] = 'test_value'

        try:
            processed = config_loader._process_environment_variables(config_text)

            assert 'production_rules' in processed
            assert 'test_value' in processed
            assert 'users' in processed  # Default value
        finally:
            # Clean up environment variables
            os.environ.pop('RULE_SET_NAME', None)
            os.environ.pop('TEST_VAR', None)

    def test_missing_required_environment_variable(self, config_loader):
        """Test error when required environment variable is missing."""
        config_text = "name: ${REQUIRED_VAR}"

        with pytest.raises(EnvironmentError, match="Required environment variable 'REQUIRED_VAR' is not set"):
            config_loader._process_environment_variables(config_text)

    def test_load_rule_set_from_file(self, config_loader):
        """Test loading rule set from YAML file."""
        # Create temporary YAML file
        sample_config = {
            'name': 'test_rules',
            'description': 'Test rule set',
            'rules': [{
                'name': 'test_rule',
                'description': 'Test rule',
                'rule_type': 'data_quality',
                'sql_query': 'SELECT COUNT(*) FROM users',
                'severity': 'critical',
                'scope': 'table',
                'enabled': True
            }]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_file = f.name

        try:
            rule_set = config_loader.load_rule_set_from_file(temp_file)

            assert rule_set.name == 'test_rules'
            assert rule_set.description == 'Test rule set'
            assert len(rule_set.rules) == 1
            assert rule_set.rules[0].name == 'test_rule'
            assert rule_set.rules[0].rule_type == RuleType.DATA_QUALITY
        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_file(self, config_loader):
        """Test loading from nonexistent file."""
        with pytest.raises(ConfigurationError):
            config_loader.load_rule_set_from_file("/nonexistent/file.yaml")

    def test_circular_dependency_detection(self, config_loader):
        """Test detection of circular dependencies in rules."""
        sample_config = {
            'name': 'circular_test',
            'description': 'Test circular dependencies',
            'rules': [
                {
                    'name': 'rule_a',
                    'description': 'Rule A',
                    'rule_type': 'data_quality',
                    'sql_query': 'SELECT 1',
                    'severity': 'warning',
                    'scope': 'table',
                    'enabled': True,
                    'dependencies': ['rule_b']
                },
                {
                    'name': 'rule_b',
                    'description': 'Rule B',
                    'rule_type': 'data_quality',
                    'sql_query': 'SELECT 2',
                    'severity': 'warning',
                    'scope': 'table',
                    'enabled': True,
                    'dependencies': ['rule_a']
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_file = f.name

        try:
            # The config loader should detect circular dependencies and raise ConfigurationError
            with pytest.raises(ConfigurationError, match="Circular dependency detected"):
                config_loader.load_rule_set_from_file(temp_file)

        finally:
            os.unlink(temp_file)


class TestBusinessRuleValidator:
    """Test the high-level business rule validator."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        adapter = Mock()
        adapter.execute_query.return_value = Mock(data=pd.DataFrame())
        manager.get_adapter.return_value = adapter
        return manager

    @pytest.fixture
    def validator(self, mock_connection_manager):
        """Create validator instance."""
        return BusinessRuleValidator(mock_connection_manager)

    def test_validator_initialization(self, mock_connection_manager):
        """Test validator initialization."""
        validator = BusinessRuleValidator(mock_connection_manager, max_workers=10)
        assert validator.connection_manager == mock_connection_manager
        assert validator.engine.max_workers == 10

    def test_load_and_validate_from_config(self, validator):
        """Test loading rules from config and validating."""
        # Create a temporary config file
        sample_config = {
            'name': 'integration_test',
            'description': 'Integration test rules',
            'rules': [{
                'name': 'test_rule',
                'description': 'Test rule',
                'rule_type': 'data_quality',
                'sql_query': 'SELECT COUNT(*) as violation_count FROM users WHERE email IS NULL',
                'severity': 'critical',
                'scope': 'table',
                'enabled': True,
                'expected_violation_count': 0
            }]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_file = f.name

        try:
            # Mock successful query execution
            validator.connection_manager.get_adapter.return_value.execute_query.return_value.data = pd.DataFrame()

            summary = validator.validate_from_config(
                config_path=temp_file,
                database_name="test_db"
            )

            assert summary.rule_set_name == 'integration_test'
            assert summary.total_rules == 1
            assert summary.rules_passed == 1

        finally:
            os.unlink(temp_file)


class TestPerformanceAndCaching:
    """Test performance optimizations and caching."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        adapter = Mock()
        adapter.execute_query.return_value = Mock(data=pd.DataFrame())
        manager.get_adapter.return_value = adapter
        return manager

    def test_rule_caching(self, mock_connection_manager):
        """Test that rule results are cached appropriately."""
        engine = BusinessRuleEngine(mock_connection_manager, max_workers=2)

        # Enable caching (would be configurable in real implementation)
        engine._cache_enabled = True

        sample_rule = BusinessRule(
            name="cacheable_rule",
            description="Rule that should be cached",
            rule_type=RuleType.DATA_QUALITY,
            sql_query="SELECT COUNT(*) FROM users",
            severity=RuleSeverity.WARNING,
            scope=ValidationScope.TABLE,
            enabled=True
        )

        context = ValidationContext(database_name="test_db")

        # First execution
        result1 = engine.execute_rule(sample_rule, context)

        # Second execution should potentially use cache
        result2 = engine.execute_rule(sample_rule, context)

        # Both executions should succeed
        assert result1.status == RuleStatus.PASSED
        assert result2.status == RuleStatus.PASSED

    def test_parallel_execution_performance(self, mock_connection_manager):
        """Test parallel execution with multiple rules."""
        engine = BusinessRuleEngine(mock_connection_manager, max_workers=3)

        # Create multiple independent rules
        rules = []
        for i in range(5):
            rule = BusinessRule(
                name=f"parallel_rule_{i}",
                description=f"Parallel rule {i}",
                rule_type=RuleType.DATA_QUALITY,
                sql_query=f"SELECT {i} as result",
                severity=RuleSeverity.WARNING,
                scope=ValidationScope.TABLE,
                enabled=True,
                dependencies=[]
            )
            rules.append(rule)

        rule_set = RuleSet(
            name="parallel_test",
            description="Test parallel execution",
            rules=rules,
            parallel_execution=True,
            max_concurrent_rules=3
        )

        context = ValidationContext(database_name="test_db")

        # Execute with timing
        start_time = datetime.now()
        summary = engine.execute_rule_set(rule_set, context)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # All rules should pass
        assert summary.rules_passed == 5
        assert summary.rules_failed == 0

        # Execution should be reasonably fast
        assert execution_time < 10  # Should complete in under 10 seconds


def test_create_sample_business_rules():
    """Test creation of sample business rules configuration."""
    from sqltest.modules.business_rules.config_loader import create_sample_business_rules

    # The function returns a RuleSet, not creates a file
    rule_set = create_sample_business_rules()

    # Verify rule set structure
    assert rule_set.name == "sample_data_quality_rules"
    assert rule_set.description == "Sample data quality rules for demonstration"
    assert len(rule_set.rules) > 0

    # Verify structure of rules
    for rule in rule_set.rules:
        assert rule.name is not None
        assert rule.description is not None
        assert rule.rule_type is not None
        assert rule.severity is not None


class TestAdvancedBusinessRuleFeatures:
    """Test advanced features like caching, metrics, retry, and batching."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager for testing."""
        manager = Mock(spec=ConnectionManager)
        adapter = Mock()
        adapter.execute_query.return_value = Mock(data=pd.DataFrame())
        manager.get_adapter.return_value = adapter
        return manager

    @pytest.fixture
    def advanced_rule_engine(self, mock_connection_manager):
        """Create an advanced rule engine with all features enabled."""
        return BusinessRuleEngine(
            mock_connection_manager,
            max_workers=4,
            enable_caching=True,
            enable_metrics=True,
            cache_config={'l1_max_size': 100, 'l1_ttl_seconds': 60},
            retry_config={'max_retries': 2, 'base_delay': 0.1}
        )

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules for testing."""
        return [
            BusinessRule(
                name="rule_1",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.ERROR,
                scope=ValidationScope.TABLE,
                sql_query="SELECT COUNT(*) as violation_count FROM users WHERE email IS NULL",
                description="Check for null emails",
                enabled=True,
                timeout_seconds=30.0
            ),
            BusinessRule(
                name="rule_2",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.WARNING,
                scope=ValidationScope.COLUMN,
                sql_query="SELECT COUNT(*) as violation_count FROM users WHERE age < 0",
                description="Check for negative ages",
                enabled=True,
                timeout_seconds=30.0
            ),
            BusinessRule(
                name="rule_3",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.INFO,
                scope=ValidationScope.TABLE,
                sql_query="SELECT COUNT(*) as violation_count FROM users WHERE created_at > NOW()",
                description="Check for future dates",
                enabled=True,
                timeout_seconds=30.0,
                dependencies=["rule_1"]  # Depends on rule_1
            )
        ]

    def test_performance_metrics_collection(self, advanced_rule_engine, sample_rules):
        """Test performance metrics collection."""
        engine = advanced_rule_engine

        # Execute some rules
        context = ValidationContext(database_name="test_db")
        rule_set = RuleSet(name="test_set", rules=sample_rules)

        # Mock adapter to simulate execution
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.return_value = Mock(data=pd.DataFrame({'violation_count': [0]}))

        # Execute rule set
        summary = engine.execute_rule_set(rule_set, context)

        # Check metrics were collected
        assert engine.metrics is not None
        stats = engine.get_performance_stats()

        assert 'execution_metrics' in stats
        assert 'engine_config' in stats
        assert stats['engine_config']['metrics_enabled'] is True
        assert stats['engine_config']['cache_enabled'] is True

    def test_caching_functionality(self, advanced_rule_engine, sample_rules):
        """Test result caching functionality."""
        engine = advanced_rule_engine
        rule = sample_rules[0]
        context = ValidationContext(database_name="test_db")

        # Mock adapter
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.return_value = Mock(data=pd.DataFrame({'violation_count': [0]}))

        # Execute rule first time
        result1 = engine.execute_rule(rule, context)

        # Execute rule second time (should use cache)
        result2 = engine.execute_rule(rule, context)

        # Both should succeed
        assert result1.passed
        assert result2.passed

        # Check cache stats
        stats = engine.get_performance_stats()
        assert 'cache_stats' in stats

        # Invalidate cache
        engine.invalidate_cache()
        cache_stats = stats['cache_stats']
        assert 'l1_size' in cache_stats

    def test_retry_mechanism(self, advanced_rule_engine, sample_rules):
        """Test retry mechanism with exponential backoff."""
        engine = advanced_rule_engine
        rule = sample_rules[0]
        context = ValidationContext(database_name="test_db")

        # Mock adapter to fail first few times, then succeed
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.side_effect = [
            Exception("Connection failed"),
            Exception("Temporary error"),
            Mock(data=pd.DataFrame({'violation_count': [0]}))  # Third call succeeds
        ]

        # Execute rule - should succeed after retries
        result = engine.execute_rule(rule, context)

        # Should succeed after retries
        assert result.passed

        # Verify retry logic was called
        assert adapter.execute_query.call_count == 3

    def test_rule_batching(self, advanced_rule_engine, sample_rules):
        """Test rule batching functionality."""
        engine = advanced_rule_engine

        # Remove dependencies to allow batching
        for rule in sample_rules:
            rule.dependencies = []

        # Create batches
        batches = engine._create_rule_batches(sample_rules)

        # Should create appropriate number of batches
        assert len(batches) > 0

        # Test batch compatibility
        rule1, rule2 = sample_rules[0], sample_rules[1]
        assert engine._can_batch_rule(rule1, rule2) is True

        # Test with dependency - should not be batchable
        rule2.dependencies = [rule1.name]
        assert engine._can_batch_rule(rule1, rule2) is False

    def test_parallel_batched_execution(self, advanced_rule_engine, sample_rules):
        """Test parallel execution with batching."""
        engine = advanced_rule_engine
        context = ValidationContext(database_name="test_db")

        # Remove dependencies for parallel execution
        for rule in sample_rules:
            rule.dependencies = []

        rule_set = RuleSet(
            name="parallel_test",
            rules=sample_rules,
            parallel_execution=True,
            max_concurrent_rules=2
        )

        # Mock adapter
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.return_value = Mock(data=pd.DataFrame({'violation_count': [0]}))

        # Execute with batching enabled
        summary = engine.execute_rule_set(rule_set, context, enable_batching=True)

        # Should complete successfully
        assert summary.rules_executed == len(sample_rules)
        assert summary.rules_passed == len(sample_rules)

    def test_fail_fast_with_batching(self, advanced_rule_engine, sample_rules):
        """Test fail-fast behavior with batch execution."""
        engine = advanced_rule_engine
        context = ValidationContext(database_name="test_db")

        # Set first rule to critical severity
        sample_rules[0].severity = RuleSeverity.CRITICAL

        # Remove dependencies
        for rule in sample_rules:
            rule.dependencies = []

        rule_set = RuleSet(
            name="fail_fast_test",
            rules=sample_rules,
            parallel_execution=True,
            max_concurrent_rules=2
        )

        # Mock adapter - first rule fails
        adapter = engine.connection_manager.get_adapter.return_value
        side_effects = [
            Mock(data=pd.DataFrame({'violation_count': [5]})),  # First rule fails
            Mock(data=pd.DataFrame({'violation_count': [0]})),  # Second rule would pass
            Mock(data=pd.DataFrame({'violation_count': [0]}))   # Third rule would pass
        ]
        adapter.execute_query.side_effect = side_effects

        # Execute with fail-fast
        summary = engine.execute_rule_set(rule_set, context, fail_fast=True)

        # Should have failed rules due to violations
        assert summary.rules_failed > 0

    def test_cache_invalidation(self, advanced_rule_engine, sample_rules):
        """Test cache invalidation patterns."""
        engine = advanced_rule_engine
        rule = sample_rules[0]
        context = ValidationContext(database_name="test_db")

        # Mock adapter
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.return_value = Mock(data=pd.DataFrame({'violation_count': [0]}))

        # Execute rule to populate cache
        result1 = engine.execute_rule(rule, context)
        assert result1.passed

        # Invalidate specific pattern
        engine.invalidate_cache("test_rule")

        # Invalidate all cache
        engine.invalidate_cache()

        # Should still work after cache invalidation
        result2 = engine.execute_rule(rule, context)
        assert result2.passed

    def test_engine_configuration_options(self, mock_connection_manager):
        """Test various engine configuration options."""
        # Test with caching disabled
        engine_no_cache = BusinessRuleEngine(
            mock_connection_manager,
            enable_caching=False,
            enable_metrics=False
        )

        assert engine_no_cache.cache_manager is None
        assert engine_no_cache.metrics is None

        # Test with custom retry configuration
        engine_custom_retry = BusinessRuleEngine(
            mock_connection_manager,
            retry_config={'max_retries': 5, 'base_delay': 0.5}
        )

        assert engine_custom_retry.retry_manager.max_retries == 5
        assert engine_custom_retry.retry_manager.base_delay == 0.5

    @pytest.mark.integration
    def test_end_to_end_advanced_features(self, advanced_rule_engine, sample_rules):
        """End-to-end test of all advanced features together."""
        engine = advanced_rule_engine
        context = ValidationContext(database_name="test_db", table_name="users")

        # Create a complex rule set with dependencies
        complex_rules = [
            BusinessRule(
                name="data_exists",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.CRITICAL,
                scope=ValidationScope.TABLE,
                sql_query="SELECT CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END as violation_count FROM users",
                description="Ensure data exists",
                enabled=True,
                timeout_seconds=30.0
            ),
            BusinessRule(
                name="email_validation",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.ERROR,
                scope=ValidationScope.COLUMN,
                sql_query="SELECT COUNT(*) as violation_count FROM users WHERE email IS NULL OR email = ''",
                description="Validate email fields",
                enabled=True,
                timeout_seconds=30.0,
                dependencies=["data_exists"]
            )
        ]

        rule_set = RuleSet(
            name="complex_validation",
            description="Complex validation with all features",
            rules=complex_rules,
            parallel_execution=True,
            max_concurrent_rules=2
        )

        # Mock adapter responses
        adapter = engine.connection_manager.get_adapter.return_value
        adapter.execute_query.return_value = Mock(data=pd.DataFrame({'violation_count': [0]}))

        # Execute the complex rule set
        summary = engine.execute_rule_set(rule_set, context, enable_batching=True)

        # Verify execution completed successfully
        assert summary.total_rules == 2
        assert summary.rules_executed == 2
        assert summary.rules_passed == 2
        assert summary.rules_failed == 0

        # Verify performance stats are available
        stats = engine.get_performance_stats()
        assert 'execution_metrics' in stats
        assert 'cache_stats' in stats
        assert 'engine_config' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])