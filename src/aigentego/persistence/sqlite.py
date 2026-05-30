"""SQLite connection and schema initialization helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 3
SCHEMA_VERSION_KEY = "schema_version"

_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    created_at TEXT,
    updated_at TEXT
)
"""

_CONVERSATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    is_default INTEGER NOT NULL CHECK (is_default IN (0, 1)),
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (session_id, conversation_id),
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
)
"""

_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    sequence_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TEXT,
    UNIQUE (conversation_id, sequence_index),
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id, conversation_id)
        REFERENCES conversations (session_id, conversation_id)
        ON DELETE CASCADE
)
"""

_MEMORY_SUMMARIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_summaries (
    summary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (conversation_id, revision),
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id, conversation_id)
        REFERENCES conversations (session_id, conversation_id)
        ON DELETE CASCADE
)
"""

_NOTE_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS note_files (
    path TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    modified_time_ns INTEGER NOT NULL CHECK (modified_time_ns >= 0),
    extension TEXT NOT NULL CHECK (extension IN ('.md', '.markdown', '.txt')),
    UNIQUE (root_path, relative_path)
)
"""

_NOTE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS note_documents (
    document_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    content_length INTEGER NOT NULL CHECK (content_length >= 0),
    content_hash TEXT NOT NULL,
    FOREIGN KEY (file_path) REFERENCES note_files (path) ON DELETE CASCADE
)
"""

_NOTE_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS note_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL CHECK (content_length >= 1),
    content_hash TEXT NOT NULL,
    UNIQUE (document_id, chunk_index),
    FOREIGN KEY (document_id)
        REFERENCES note_documents (document_id)
        ON DELETE CASCADE
)
"""

_CONVERSATIONS_SESSION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_conversations_session_id
ON conversations (session_id)
"""

_MESSAGES_CONVERSATION_SEQUENCE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
ON messages (conversation_id, sequence_index)
"""

_MEMORY_SUMMARIES_CONVERSATION_REVISION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memory_summaries_conversation_revision
ON memory_summaries (conversation_id, revision DESC)
"""

_NOTE_FILES_ROOT_RELATIVE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_note_files_root_relative
ON note_files (root_path, relative_path)
"""

_NOTE_DOCUMENTS_FILE_PATH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_note_documents_file_path
ON note_documents (file_path)
"""

_NOTE_CHUNKS_DOCUMENT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_note_chunks_document_index
ON note_chunks (document_id, chunk_index)
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
    """Initialize the local persistence schema idempotently."""
    connection.execute(_METADATA_TABLE_SQL)
    connection.execute(_SESSIONS_TABLE_SQL)
    connection.execute(_CONVERSATIONS_TABLE_SQL)
    connection.execute(_MESSAGES_TABLE_SQL)
    connection.execute(_MEMORY_SUMMARIES_TABLE_SQL)
    connection.execute(_NOTE_FILES_TABLE_SQL)
    connection.execute(_NOTE_DOCUMENTS_TABLE_SQL)
    connection.execute(_NOTE_CHUNKS_TABLE_SQL)
    connection.execute(_CONVERSATIONS_SESSION_INDEX_SQL)
    connection.execute(_MESSAGES_CONVERSATION_SEQUENCE_INDEX_SQL)
    connection.execute(_MEMORY_SUMMARIES_CONVERSATION_REVISION_INDEX_SQL)
    connection.execute(_NOTE_FILES_ROOT_RELATIVE_INDEX_SQL)
    connection.execute(_NOTE_DOCUMENTS_FILE_PATH_INDEX_SQL)
    connection.execute(_NOTE_CHUNKS_DOCUMENT_INDEX_SQL)
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
