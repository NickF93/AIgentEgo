import sqlite3

import pytest

from aigentego.persistence import (
    SCHEMA_VERSION,
    connect_sqlite,
    initialize_sqlite_schema,
    open_sqlite_database,
    read_schema_version,
)


def test_open_sqlite_database_creates_parent_directory_and_schema(tmp_path) -> None:
    sqlite_path = tmp_path / "runtime" / "aigentego.sqlite3"

    connection = open_sqlite_database(sqlite_path)
    try:
        assert sqlite_path.exists()
        assert sqlite_path.parent.exists()
        assert read_schema_version(connection) == SCHEMA_VERSION
    finally:
        connection.close()


def test_schema_initialization_is_idempotent(tmp_path) -> None:
    sqlite_path = tmp_path / "aigentego.sqlite3"

    connection = open_sqlite_database(sqlite_path)
    try:
        initialize_sqlite_schema(connection)
        initialize_sqlite_schema(connection)

        rows = connection.execute("SELECT key, value FROM schema_metadata").fetchall()
    finally:
        connection.close()

    assert [(row["key"], row["value"]) for row in rows] == [
        ("schema_version", str(SCHEMA_VERSION)),
    ]


def test_schema_initialization_creates_persistence_tables(tmp_path) -> None:
    connection = open_sqlite_database(tmp_path / "aigentego.sqlite3")
    try:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
                AND name IN (
                    'schema_metadata',
                    'sessions',
                    'conversations',
                    'messages',
                    'memory_summaries'
                )
            ORDER BY name
            """,
        ).fetchall()
    finally:
        connection.close()

    assert [row["name"] for row in rows] == [
        "conversations",
        "memory_summaries",
        "messages",
        "schema_metadata",
        "sessions",
    ]


def test_connect_sqlite_enables_foreign_keys_and_row_factory(tmp_path) -> None:
    sqlite_path = tmp_path / "aigentego.sqlite3"

    connection = connect_sqlite(sqlite_path)
    try:
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO sample (id) VALUES (1)")
        row = connection.execute("SELECT id FROM sample").fetchone()
    finally:
        connection.close()

    assert foreign_keys is not None
    assert foreign_keys[0] == 1
    assert isinstance(row, sqlite3.Row)
    assert row["id"] == 1


def test_open_sqlite_database_supports_in_memory_database() -> None:
    connection = open_sqlite_database(":memory:")
    try:
        assert read_schema_version(connection) == SCHEMA_VERSION
    finally:
        connection.close()


@pytest.mark.parametrize("sqlite_path", ["", "   "])
def test_blank_sqlite_path_is_rejected(sqlite_path: str) -> None:
    with pytest.raises(ValueError, match="sqlite_path must not be blank"):
        open_sqlite_database(sqlite_path)


def test_read_schema_version_requires_initialized_metadata(tmp_path) -> None:
    connection = connect_sqlite(tmp_path / "uninitialized.sqlite3")
    try:
        with pytest.raises(
            RuntimeError,
            match="SQLite schema metadata is not initialized",
        ):
            read_schema_version(connection)
    finally:
        connection.close()
