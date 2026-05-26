"""SQLite connection and schema initialization helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA_VERSION_KEY = "schema_version"

_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def _normalize_sqlite_path(sqlite_path: str | Path) -> str:
    value = str(sqlite_path).strip()
    if not value:
        raise ValueError("sqlite_path must not be blank")
    if _is_in_memory_path(value):
        return value
    return str(Path(value).expanduser())


def _is_in_memory_path(sqlite_path: str) -> bool:
    return sqlite_path == ":memory:"


def _ensure_parent_directory(sqlite_path: str) -> None:
    if _is_in_memory_path(sqlite_path):
        return

    parent = Path(sqlite_path).expanduser().parent
    if parent == Path("."):
        return

    parent.mkdir(parents=True, exist_ok=True)


def connect_sqlite(sqlite_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with project defaults applied."""
    normalized_path = _normalize_sqlite_path(sqlite_path)
    connection = sqlite3.connect(normalized_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    """Initialize the metadata-only persistence schema idempotently."""
    connection.execute(_METADATA_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO schema_metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
    )
    connection.commit()


def open_sqlite_database(sqlite_path: str | Path) -> sqlite3.Connection:
    """Open a local SQLite database and initialize the persistence schema."""
    normalized_path = _normalize_sqlite_path(sqlite_path)
    _ensure_parent_directory(normalized_path)
    connection = connect_sqlite(normalized_path)
    initialize_sqlite_schema(connection)
    return connection


def read_schema_version(connection: sqlite3.Connection) -> int:
    """Read the initialized persistence schema version."""
    try:
        row = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = ?",
            (SCHEMA_VERSION_KEY,),
        ).fetchone()
    except sqlite3.OperationalError as error:
        raise RuntimeError("SQLite schema metadata is not initialized") from error
    if row is None:
        raise RuntimeError("SQLite schema metadata is not initialized")
    return int(row["value"])
