"""Local persistence substrate primitives."""

from aigentego.persistence.sqlite import (
    SCHEMA_VERSION,
    connect_sqlite,
    initialize_sqlite_schema,
    open_sqlite_database,
    read_schema_version,
)

__all__ = [
    "SCHEMA_VERSION",
    "connect_sqlite",
    "initialize_sqlite_schema",
    "open_sqlite_database",
    "read_schema_version",
]
