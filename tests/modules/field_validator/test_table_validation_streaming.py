"""Tests for streaming field validation."""

from __future__ import annotations

import pytest

from sqltest.config.models import DatabaseConfig, DatabaseType, SQLTestConfig
from sqltest.db.connection import ConnectionManager
from sqltest.modules.field_validator import TableFieldValidator
from sqltest.modules.field_validator.models import (
    NullValidationRule,
    ValidationRuleSet,
    ValidationRuleType,
)


@pytest.fixture()
def field_validator(tmp_path):
    db_path = tmp_path / "validation.db"

    config = SQLTestConfig(
        databases={
            "analytics": DatabaseConfig(type=DatabaseType.SQLITE, path=str(db_path))
        },
        default_database="analytics",
    )

    manager = ConnectionManager(config)
    adapter = manager.get_adapter("analytics")

    adapter.execute_query(
        "CREATE TABLE emails (id INTEGER PRIMARY KEY, email TEXT)",
        fetch_results=False,
    )

    insert_sql = "INSERT INTO emails (email) VALUES (:email)"
    for i in range(250):
        value = f"user{i}@example.com"
        if i % 60 == 0:
            value = None
        adapter.execute_query(insert_sql, params={"email": value}, fetch_results=False)

    validator = TableFieldValidator(manager, strict_mode=False)

    rule_set = ValidationRuleSet(
        name="email_rules",
        description="Validate email presence",
        rules=[
            NullValidationRule(
                name="no_null_emails",
                rule_type=ValidationRuleType.NULL_CHECK,
                description="Emails must not be null",
                allow_null=False,
                error_message="Email is required",
            )
        ],
        apply_to_columns=["email"],
    )

    validator.add_rule_set(rule_set)
    yield validator
    validator.connection_manager.close_all_connections()


def test_table_validation_streaming_aggregates_results(field_validator: TableFieldValidator):
    result = field_validator.validate_table_data("emails", "email_rules")

    assert result.table_name == "emails"
    email_result = next(fr for fr in result.field_results if fr.column_name == "email")

    assert email_result.total_rows == 250

    failures = [r for r in email_result.validation_results if not r.passed]
    assert len(failures) == 5
    assert email_result.failed_rules == 5
    assert max(f.row_number for f in failures) == 241
