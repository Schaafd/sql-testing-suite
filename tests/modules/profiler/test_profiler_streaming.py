"""Tests for DataProfiler streaming and aggregate behaviour."""

from __future__ import annotations

import pytest

from sqltest.config.models import DatabaseConfig, DatabaseType, SQLTestConfig
from sqltest.db.connection import ConnectionManager
from sqltest.modules.profiler import DataProfiler


@pytest.fixture()
def profiler(tmp_path):
    db_path = tmp_path / "profiling.db"

    config = SQLTestConfig(
        databases={
            "analytics": DatabaseConfig(type=DatabaseType.SQLITE, path=str(db_path))
        },
        default_database="analytics",
    )

    manager = ConnectionManager(config)
    adapter = manager.get_adapter("analytics")

    adapter.execute_query(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER, category TEXT)",
        fetch_results=False,
    )

    insert_sql = "INSERT INTO numbers (value, category) VALUES (:value, :category)"
    for i in range(1200):
        category = "even" if i % 2 == 0 else "odd"
        if i % 100 == 0:
            category = None
        adapter.execute_query(
            insert_sql,
            params={"value": i, "category": category},
            fetch_results=False,
        )

    profiler = DataProfiler(manager, sample_size=100)
    yield profiler
    manager.close_all_connections()


def test_profile_table_streaming_uses_aggregates(profiler: DataProfiler):
    profile = profiler.profile_table("numbers")

    assert profile.total_rows == 1200

    value_stats = profile.columns["value"]
    assert value_stats.total_rows == 1200
    assert value_stats.min_value == 0
    assert value_stats.max_value == 1199
    assert value_stats.histogram is not None and len(value_stats.histogram) > 0

    category_stats = profile.columns["category"]
    assert category_stats.total_rows == 1200
    assert category_stats.null_count == 12
    # ensure streaming sample did not mis-align percentages
    assert category_stats.null_percentage == pytest.approx((12 / 1200) * 100)
