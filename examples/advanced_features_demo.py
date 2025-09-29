#!/usr/bin/env python3
"""
Advanced Business Rules Engine Demo

This script demonstrates all the advanced features implemented in Week 2:
- Multi-level caching (L1 memory + L2 Redis)
- Performance monitoring and metrics collection
- Retry mechanisms with exponential backoff
- Parallel execution with intelligent rule batching
- Comprehensive error handling and monitoring

Usage:
    python examples/advanced_features_demo.py [--config CONFIG_FILE] [--redis-url REDIS_URL]

Requirements:
    - SQLTest Pro with advanced features
    - Optional: Redis server for L2 caching
    - PostgreSQL database (can be adapted for other databases)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqltest.modules.business_rules import BusinessRuleValidator
from sqltest.modules.business_rules.engine import BusinessRuleEngine
from sqltest.modules.business_rules.models import (
    BusinessRule, RuleSet, RuleType, RuleSeverity, ValidationScope, ValidationContext
)
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ValidationError, ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeaturesDemo:
    """Demonstrates advanced business rules engine features."""

    def __init__(self, config_path: str = None, redis_url: str = None):
        """Initialize the demo with configuration."""
        self.config_path = config_path or "examples/advanced_business_rules_config.yaml"
        self.redis_url = redis_url
        self.connection_manager = None
        self.engine = None

    def setup(self):
        """Set up database connections and engine."""
        logger.info("Setting up Advanced Features Demo...")

        try:
            # Load configuration
            if Path(self.config_path).exists():
                config = get_config(self.config_path)
                self.connection_manager = get_connection_manager(config)
            else:
                logger.warning(f"Config file {self.config_path} not found, using mock connection")
                self.connection_manager = self._create_mock_connection_manager()

            # Create advanced engine
            self.engine = BusinessRuleEngine(
                self.connection_manager,
                max_workers=8,
                enable_caching=True,
                cache_config={
                    'l1_max_size': 500,
                    'l1_ttl_seconds': 300,
                    'l2_ttl_seconds': 1800
                },
                enable_metrics=True,
                redis_url=self.redis_url,
                retry_config={
                    'max_retries': 3,
                    'base_delay': 0.5,
                    'max_delay': 30.0,
                    'backoff_multiplier': 2.0
                }
            )

            logger.info("✅ Setup completed successfully!")

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

    def _create_mock_connection_manager(self):
        """Create a mock connection manager for demo purposes."""
        from unittest.mock import Mock
        import pandas as pd

        manager = Mock()
        adapter = Mock()

        # Mock successful responses
        adapter.execute_query.return_value = Mock(
            data=pd.DataFrame({'violation_count': [0], 'message': ['All checks passed']})
        )
        manager.get_adapter.return_value = adapter

        return manager

    def demo_caching_features(self):
        """Demonstrate caching capabilities."""
        logger.info("\n🚀 Demo 1: Caching Features")

        # Create a test rule
        rule = BusinessRule(
            name="cache_demo_rule",
            rule_type=RuleType.DATA_QUALITY,
            severity=RuleSeverity.ERROR,
            scope=ValidationScope.TABLE,
            sql_query="SELECT COUNT(*) as violation_count FROM demo_table WHERE invalid_data = 1",
            description="Demo rule for caching",
            enabled=True,
            timeout_seconds=30.0,
            cache_results=True
        )

        context = ValidationContext(database_name="demo_db", table_name="demo_table")

        # First execution (cache miss)
        logger.info("Executing rule (cache miss)...")
        start_time = time.time()
        result1 = self.engine.execute_rule(rule, context)
        first_duration = time.time() - start_time

        # Second execution (cache hit)
        logger.info("Executing rule again (cache hit)...")
        start_time = time.time()
        result2 = self.engine.execute_rule(rule, context)
        second_duration = time.time() - start_time

        # Display results
        logger.info(f"First execution: {first_duration:.3f}s")
        logger.info(f"Second execution: {second_duration:.3f}s")
        logger.info(f"Speed improvement: {first_duration/second_duration:.1f}x")

        # Check cache stats
        stats = self.engine.get_performance_stats()
        cache_stats = stats.get('cache_stats', {})
        logger.info(f"Cache statistics: {json.dumps(cache_stats, indent=2)}")

        # Demonstrate cache invalidation
        logger.info("Invalidating cache...")
        self.engine.invalidate_cache()
        logger.info("✅ Cache invalidation completed")

    def demo_retry_mechanisms(self):
        """Demonstrate retry mechanisms."""
        logger.info("\n🔄 Demo 2: Retry Mechanisms")

        # Create a rule that will initially fail
        rule = BusinessRule(
            name="retry_demo_rule",
            rule_type=RuleType.DATA_QUALITY,
            severity=RuleSeverity.ERROR,
            scope=ValidationScope.TABLE,
            sql_query="SELECT COUNT(*) as violation_count FROM unstable_table",
            description="Demo rule for retry mechanism",
            enabled=True,
            timeout_seconds=30.0
        )

        context = ValidationContext(database_name="demo_db")

        # Mock adapter to fail then succeed
        if hasattr(self.connection_manager, 'get_adapter'):
            adapter = self.connection_manager.get_adapter.return_value
            adapter.execute_query.side_effect = [
                Exception("Connection timeout"),
                Exception("Temporary failure"),
                Mock(data=pd.DataFrame({'violation_count': [0]}))  # Success on third try
            ]

        logger.info("Executing rule with simulated failures...")
        try:
            result = self.engine.execute_rule(rule, context)
            if result.passed:
                logger.info("✅ Rule succeeded after retries!")
            else:
                logger.warning("⚠️ Rule failed after all retries")

            # Check retry statistics
            stats = self.engine.get_performance_stats()
            execution_stats = stats.get('execution_metrics', {})
            logger.info(f"Execution statistics: {json.dumps(execution_stats, indent=2)}")

        except Exception as e:
            logger.error(f"❌ Retry demo failed: {e}")

    def demo_parallel_execution(self):
        """Demonstrate parallel execution with batching."""
        logger.info("\n⚡ Demo 3: Parallel Execution & Batching")

        # Create a set of rules that can be batched
        rules = [
            BusinessRule(
                name=f"parallel_rule_{i}",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.WARNING,
                scope=ValidationScope.COLUMN,
                sql_query=f"SELECT COUNT(*) as violation_count FROM table_{i} WHERE invalid = 1",
                description=f"Parallel demo rule {i}",
                enabled=True,
                timeout_seconds=30.0,
                batch_compatible=True
            )
            for i in range(1, 6)
        ]

        # Add a rule with dependencies
        dependent_rule = BusinessRule(
            name="dependent_rule",
            rule_type=RuleType.DATA_QUALITY,
            severity=RuleSeverity.ERROR,
            scope=ValidationScope.TABLE,
            sql_query="SELECT 0 as violation_count",
            description="Rule that depends on others",
            enabled=True,
            timeout_seconds=30.0,
            dependencies=["parallel_rule_1", "parallel_rule_2"]
        )
        rules.append(dependent_rule)

        rule_set = RuleSet(
            name="parallel_demo",
            description="Demo of parallel execution",
            rules=rules,
            parallel_execution=True,
            max_concurrent_rules=4
        )

        context = ValidationContext(database_name="demo_db")

        # Execute with batching
        logger.info("Executing rules with parallel batching...")
        start_time = time.time()
        summary = self.engine.execute_rule_set(rule_set, context, enable_batching=True)
        execution_time = time.time() - start_time

        logger.info(f"Parallel execution completed in {execution_time:.3f}s")
        logger.info(f"Rules executed: {summary.rules_executed}")
        logger.info(f"Rules passed: {summary.rules_passed}")
        logger.info(f"Rules failed: {summary.rules_failed}")

        # Compare with sequential execution
        logger.info("Comparing with sequential execution...")
        start_time = time.time()
        sequential_summary = self.engine.execute_rule_set(
            rule_set, context, parallel=False, enable_batching=False
        )
        sequential_time = time.time() - start_time

        speedup = sequential_time / execution_time
        logger.info(f"Sequential execution: {sequential_time:.3f}s")
        logger.info(f"Parallel speedup: {speedup:.1f}x")

    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        logger.info("\n📊 Demo 4: Performance Monitoring")

        # Execute several rules to generate metrics
        rules = [
            BusinessRule(
                name=f"monitoring_rule_{i}",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.INFO,
                scope=ValidationScope.TABLE,
                sql_query=f"SELECT {i % 2} as violation_count",  # Some pass, some fail
                description=f"Monitoring demo rule {i}",
                enabled=True,
                timeout_seconds=30.0
            )
            for i in range(10)
        ]

        rule_set = RuleSet(
            name="monitoring_demo",
            description="Demo of performance monitoring",
            rules=rules
        )

        context = ValidationContext(database_name="demo_db")

        logger.info("Executing rules to generate performance data...")
        summary = self.engine.execute_rule_set(rule_set, context)

        # Get comprehensive statistics
        stats = self.engine.get_performance_stats()

        logger.info("📈 Performance Statistics:")
        logger.info(f"  Total rules executed: {summary.rules_executed}")
        logger.info(f"  Success rate: {summary.rules_passed / summary.rules_executed:.2%}")

        # Display execution metrics
        execution_metrics = stats.get('execution_metrics', {})
        if execution_metrics:
            logger.info("⏱️ Execution Metrics:")
            for metric, data in execution_metrics.items():
                if isinstance(data, dict) and 'avg' in data:
                    logger.info(f"  {metric}: avg={data['avg']:.2f}, count={data['count']}")

        # Display cache metrics
        cache_stats = stats.get('cache_stats', {})
        if cache_stats:
            logger.info("💾 Cache Statistics:")
            for key, value in cache_stats.items():
                logger.info(f"  {key}: {value}")

        # Display engine configuration
        engine_config = stats.get('engine_config', {})
        logger.info("⚙️ Engine Configuration:")
        for key, value in engine_config.items():
            logger.info(f"  {key}: {value}")

    def demo_error_handling(self):
        """Demonstrate advanced error handling."""
        logger.info("\n🛡️ Demo 5: Error Handling & Resilience")

        # Create rules with various error conditions
        rules = [
            BusinessRule(
                name="timeout_rule",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.ERROR,
                scope=ValidationScope.TABLE,
                sql_query="SELECT pg_sleep(45), 0 as violation_count",  # Will timeout
                description="Rule that times out",
                enabled=True,
                timeout_seconds=1.0  # Short timeout to trigger error
            ),
            BusinessRule(
                name="sql_error_rule",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.WARNING,
                scope=ValidationScope.TABLE,
                sql_query="SELECT * FROM non_existent_table",  # SQL error
                description="Rule with SQL error",
                enabled=True,
                timeout_seconds=30.0
            ),
            BusinessRule(
                name="successful_rule",
                rule_type=RuleType.DATA_QUALITY,
                severity=RuleSeverity.INFO,
                scope=ValidationScope.TABLE,
                sql_query="SELECT 0 as violation_count",
                description="Successful rule",
                enabled=True,
                timeout_seconds=30.0
            )
        ]

        rule_set = RuleSet(
            name="error_handling_demo",
            description="Demo of error handling",
            rules=rules,
            parallel_execution=True
        )

        context = ValidationContext(database_name="demo_db")

        # Mock different error conditions if using mock
        if hasattr(self.connection_manager, 'get_adapter'):
            adapter = self.connection_manager.get_adapter.return_value
            adapter.execute_query.side_effect = [
                Exception("Simulated timeout"),
                Exception("Simulated SQL error"),
                Mock(data=pd.DataFrame({'violation_count': [0]}))  # Success
            ]

        logger.info("Executing rules with various error conditions...")
        try:
            summary = self.engine.execute_rule_set(rule_set, context, fail_fast=False)

            logger.info("Error handling results:")
            logger.info(f"  Rules executed: {summary.rules_executed}")
            logger.info(f"  Rules passed: {summary.rules_passed}")
            logger.info(f"  Rules failed: {summary.rules_failed}")
            logger.info(f"  Rules with errors: {summary.rules_error}")

            # Show individual results
            for result in summary.results:
                status_icon = "✅" if result.passed else "❌"
                logger.info(f"  {status_icon} {result.rule_name}: {result.status} - {result.message}")

        except Exception as e:
            logger.error(f"❌ Error handling demo failed: {e}")

    def run_comprehensive_demo(self):
        """Run all demo scenarios."""
        logger.info("🎯 Starting Comprehensive Advanced Features Demo")

        try:
            self.setup()

            # Run all demo scenarios
            self.demo_caching_features()
            self.demo_retry_mechanisms()
            self.demo_parallel_execution()
            self.demo_performance_monitoring()
            self.demo_error_handling()

            # Final performance summary
            self._display_final_summary()

            logger.info("\n🎉 All demos completed successfully!")

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

    def _display_final_summary(self):
        """Display final performance summary."""
        logger.info("\n📋 Final Performance Summary")

        if self.engine and self.engine.metrics:
            stats = self.engine.get_performance_stats()

            # Overall statistics
            execution_metrics = stats.get('execution_metrics', {})
            total_rules = execution_metrics.get('rules_executed_count', {}).get('value', 0)
            passed_rules = execution_metrics.get('rules_passed_count', {}).get('value', 0)
            failed_rules = execution_metrics.get('rules_failed_count', {}).get('value', 0)

            if total_rules > 0:
                success_rate = passed_rules / total_rules
                logger.info(f"  Total rules executed: {total_rules}")
                logger.info(f"  Overall success rate: {success_rate:.2%}")

            # Cache performance
            cache_hits = execution_metrics.get('cache_hits_count', {}).get('value', 0)
            cache_misses = execution_metrics.get('cache_misses_count', {}).get('value', 0)
            total_cache_ops = cache_hits + cache_misses

            if total_cache_ops > 0:
                cache_hit_rate = cache_hits / total_cache_ops
                logger.info(f"  Cache hit rate: {cache_hit_rate:.2%}")

            # Performance metrics
            avg_execution_time = execution_metrics.get('rule_execution_time', {}).get('avg', 0)
            if avg_execution_time > 0:
                logger.info(f"  Average execution time: {avg_execution_time:.2f}ms")

            logger.info("\n✨ Advanced features successfully demonstrated!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced Business Rules Engine Demo")
    parser.add_argument(
        '--config',
        help='Configuration file path',
        default='examples/advanced_business_rules_config.yaml'
    )
    parser.add_argument(
        '--redis-url',
        help='Redis URL for L2 caching (optional)',
        default=None
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the demo
    demo = AdvancedFeaturesDemo(args.config, args.redis_url)
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()