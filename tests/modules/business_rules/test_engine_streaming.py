"""Business rule engine streaming tests."""

from __future__ import annotations

import pytest

from sqltest.config.models import DatabaseConfig, DatabaseType, SQLTestConfig
from sqltest.db.connection import ConnectionManager
from sqltest.modules.business_rules.engine import BusinessRuleEngine
from sqltest.modules.business_rules.models import (
    BusinessRule,
    RuleSeverity,
    RuleType,
    ValidationContext,
    ValidationScope,
)


@pytest.fixture()
def engine(tmp_path):
    db_path = tmp_path / "rules.db"

    config = SQLTestConfig(
        databases={
            "analytics": DatabaseConfig(type=DatabaseType.SQLITE, path=str(db_path))
        },
        default_database="analytics",
    )

    manager = ConnectionManager(config)
    adapter = manager.get_adapter("analytics")

    adapter.execute_query(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER)",
        fetch_results=False,
    )

    insert_sql = "INSERT INTO numbers (value) VALUES (:value)"
    for i in range(2000):
        adapter.execute_query(insert_sql, params={"value": i}, fetch_results=False)

    engine = BusinessRuleEngine(manager, max_workers=1)
    yield engine
    manager.close_all_connections()


def test_execute_sql_rule_streaming_collects_all_violations(engine: BusinessRuleEngine):
    rule = BusinessRule(
        name="multiples_of_fifty",
        description="Identify values divisible by 50",
        rule_type=RuleType.DATA_QUALITY,
        severity=RuleSeverity.ERROR,
        scope=ValidationScope.TABLE,
        sql_query="""
            SELECT id, value, 1 AS violation_count
            FROM numbers
            WHERE value % 50 = 0
        """,
        parameters={"chunk_size": 128},
    )

    context = ValidationContext(database_name="analytics", table_name="numbers")

    result = engine.execute_rule(rule, context)

    assert result.rows_evaluated == 40
    assert len(result.violations) == 40
    assert result.violations[0].context["value"] == 0
