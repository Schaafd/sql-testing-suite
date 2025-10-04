"""Tests for streaming query execution and aggregate helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sqltest.config.models import ConnectionPoolConfig, DatabaseConfig, DatabaseType
from sqltest.db.adapters.sqlite import SQLiteAdapter
from sqltest.db.base import AggregateSpec


@pytest.fixture()
def sqlite_adapter(tmp_path: Path) -> SQLiteAdapter:
    db_path = tmp_path / "streaming_test.db"
    config = DatabaseConfig(type=DatabaseType.SQLITE, path=str(db_path))
    pool_config = ConnectionPoolConfig(min_connections=0, max_connections=1, max_overflow=0)
    adapter = SQLiteAdapter(config, pool_config)

    adapter.execute_query(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER)",
        fetch_results=False,
    )

    insert_query = "INSERT INTO numbers (value) VALUES (:value)"
    for i in range(50):
        adapter.execute_query(insert_query, params={"value": i}, fetch_results=False)

    return adapter


def test_execute_query_streaming_chunks(sqlite_adapter: SQLiteAdapter) -> None:
    result = sqlite_adapter.execute_query(
        "SELECT value FROM numbers ORDER BY value",
        chunk_size=12,
        stream_results=True,
    )

    chunks = list(result.iter_chunks())

    assert result.data is None
    assert chunks, "Expected chunks when streaming is enabled"
    assert sum(len(chunk) for chunk in chunks) == 50
    assert chunks[0].iloc[0]["value"] == 0
    assert chunks[-1].iloc[-1]["value"] == 49


def test_execute_query_result_processor(sqlite_adapter: SQLiteAdapter) -> None:
    collected: list[int] = []

    def processor(chunk: pd.DataFrame) -> None:
        collected.append(int(chunk["value"].sum()))

    result = sqlite_adapter.execute_query(
        "SELECT value FROM numbers",
        chunk_size=20,
        result_processor=processor,
    )

    assert collected, "Result processor should have been invoked with streamed chunks"
    assert result.data is None
    assert result.rows_affected == 50


def test_compute_aggregates_histogram(sqlite_adapter: SQLiteAdapter) -> None:
    specs = [
        AggregateSpec(column="value", operations=["count", "min", "max", "histogram"], bins=5),
    ]

    aggregates = sqlite_adapter.compute_aggregates("numbers", specs)

    assert aggregates["row_count"] == 50
    column_stats = aggregates["columns"]["value"]

    assert column_stats["min"] == 0
    assert column_stats["max"] == 49
    assert column_stats["count"] == 50

    histogram = column_stats["histogram"]
    assert histogram, "Histogram data should not be empty"
    assert sum(bucket["count"] for bucket in histogram) == 50
