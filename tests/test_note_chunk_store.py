import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    NoteChunkStore,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
    ingest_note_path,
)


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_store_upserts_document_and_chunks(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text("abcdef", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    result = ingest_note_path(note_path, policy, max_chunk_chars=3)
    store = NoteChunkStore(sqlite_connection, policy)

    stored = store.upsert_ingested_note(result)

    assert stored == result
    assert store.get_document(result.document.document_id) == result.document
    assert store.get_document_for_file(note_path) == result.document
    assert store.list_documents() == [result.document]
    assert store.list_chunks(result.document.document_id) == list(result.chunks)


def test_store_lists_documents_in_source_order(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root_b = tmp_path / "b-notes"
    root_a = tmp_path / "a-notes"
    root_a.mkdir()
    root_b.mkdir()
    path_b = root_b / "b.md"
    path_a = root_a / "a.txt"
    path_b.write_text("b", encoding="utf-8")
    path_a.write_text("a", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root_b, root_a])
    store = NoteChunkStore(sqlite_connection, policy)

    result_b = ingest_note_path(path_b, policy)
    result_a = ingest_note_path(path_a, policy)
    store.upsert_ingested_note(result_b)
    store.upsert_ingested_note(result_a)

    assert [
        document.file_metadata.relative_path for document in store.list_documents()
    ] == ["a.txt", "b.md"]
    assert store.list_documents(root_a) == [result_a.document]


def test_reingestion_replaces_old_chunks(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text("abcdef", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    store = NoteChunkStore(sqlite_connection, policy)

    first = ingest_note_path(note_path, policy, max_chunk_chars=3)
    store.upsert_ingested_note(first)
    note_path.write_text("xy", encoding="utf-8")
    second = ingest_note_path(note_path, policy, max_chunk_chars=3)
    stored = store.upsert_ingested_note(second)

    assert stored == second
    stored_chunks = store.list_chunks(second.document.document_id)
    assert [chunk.content for chunk in stored_chunks] == ["xy"]


def test_store_handles_empty_documents_without_chunks(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "empty.txt"
    note_path.write_text("", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    result = ingest_note_path(note_path, policy)
    store = NoteChunkStore(sqlite_connection, policy)

    stored = store.upsert_ingested_note(result)

    assert stored.document.content_length == 0
    assert stored.chunks == ()
    assert store.list_chunks(result.document.document_id) == []


def test_store_rejects_out_of_policy_document(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    note_path = outside / "secret.md"
    note_path.write_text("# secret\n", encoding="utf-8")
    permissive_policy = ReadOnlyFilesystemPolicy([outside])
    strict_policy = ReadOnlyFilesystemPolicy([allowed])
    result = ingest_note_path(note_path, permissive_policy)
    store = NoteChunkStore(sqlite_connection, strict_policy)

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        store.upsert_ingested_note(result)


def test_note_chunk_schema_is_idempotent(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "aigentego.sqlite3"
    first_connection = open_sqlite_database(sqlite_path)
    first_connection.close()
    second_connection = open_sqlite_database(sqlite_path)
    try:
        rows = second_connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
                AND name IN ('note_documents', 'note_chunks')
            ORDER BY name
            """,
        ).fetchall()
    finally:
        second_connection.close()

    assert [row["name"] for row in rows] == ["note_chunks", "note_documents"]


def test_orphan_chunks_are_rejected(
    sqlite_connection: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        sqlite_connection.execute(
            """
            INSERT INTO note_chunks (
                chunk_id,
                document_id,
                chunk_index,
                content,
                content_length,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-orphan",
                "doc-missing",
                0,
                "content",
                len("content"),
                "0" * 64,
            ),
        )


def test_store_returns_json_serializable_ingestion_result(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    result = ingest_note_path(note_path, policy)
    store = NoteChunkStore(sqlite_connection, policy)

    stored = store.upsert_ingested_note(result)

    rendered = json.loads(stored.model_dump_json())
    assert rendered["document"]["file_metadata"]["relative_path"] == "note.md"
    assert rendered["chunks"][0]["content"] == "# note\n"
