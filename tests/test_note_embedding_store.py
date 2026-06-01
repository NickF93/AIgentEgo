import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    NoteChunkEmbedding,
    NoteChunkStore,
    NoteEmbeddingStore,
    ReadOnlyFilesystemPolicy,
    ingest_note_path,
)
from aigentego.retrieval.ingestion import NoteChunk


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_store_upserts_and_fetches_chunk_embedding(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunk = persist_chunks(tmp_path, sqlite_connection)[0]
    store = NoteEmbeddingStore(sqlite_connection)
    embedding = make_embedding(chunk, vector=(0.1, 0.2, 0.3))

    stored = store.upsert_embedding(embedding)

    assert stored == embedding
    assert store.get_embedding(chunk.chunk_id, "embed-model") == embedding
    assert store.list_embeddings("embed-model") == [embedding]


def test_store_lists_embeddings_in_source_order(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_ordered_chunks(tmp_path, sqlite_connection)
    store = NoteEmbeddingStore(sqlite_connection)
    embeddings = [
        make_embedding(chunks[1], vector=(0.2,)),
        make_embedding(chunks[0], vector=(0.1,)),
    ]

    store.upsert_embeddings(embeddings)

    stored_chunk_ids = [
        embedding.chunk_id
        for embedding in store.list_embeddings("embed-model")
    ]
    assert stored_chunk_ids == [chunks[0].chunk_id, chunks[1].chunk_id]


def test_store_replaces_existing_embedding(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunk = persist_chunks(tmp_path, sqlite_connection)[0]
    store = NoteEmbeddingStore(sqlite_connection)
    first = make_embedding(chunk, vector=(0.1, 0.2))
    second = make_embedding(chunk, vector=(0.3, 0.4, 0.5))

    store.upsert_embedding(first)
    stored = store.upsert_embedding(second)

    assert stored == second
    assert store.list_embeddings("embed-model") == [second]


def test_store_selects_chunks_missing_current_embeddings(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection, content="abcdef")
    store = NoteEmbeddingStore(sqlite_connection)
    current = make_embedding(chunks[0], vector=(0.1,))
    stale = make_embedding(
        chunks[1],
        vector=(0.2,),
        chunk_content_hash="0" * 64,
    )
    store.upsert_embeddings([current, stale])

    assert store.list_chunks_missing_embeddings(chunks, "embed-model") == [chunks[1]]


def test_embedding_model_rejects_invalid_vectors() -> None:
    with pytest.raises(ValidationError, match="dimensions"):
        NoteChunkEmbedding(
            chunk_id="chunk-1",
            model="embed-model",
            dimensions=2,
            vector=(0.1,),
            chunk_content_hash="0" * 64,
        )

    with pytest.raises(ValidationError, match="numeric"):
        NoteChunkEmbedding(
            chunk_id="chunk-1",
            model="embed-model",
            dimensions=1,
            vector=(True,),
            chunk_content_hash="0" * 64,
        )


def test_note_embedding_schema_is_idempotent(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "aigentego.sqlite3"
    first_connection = open_sqlite_database(sqlite_path)
    first_connection.close()
    second_connection = open_sqlite_database(sqlite_path)
    try:
        rows = second_connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'note_chunk_embeddings'
            """,
        ).fetchall()
    finally:
        second_connection.close()

    assert [row["name"] for row in rows] == ["note_chunk_embeddings"]


def test_orphan_embeddings_are_rejected(
    sqlite_connection: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        sqlite_connection.execute(
            """
            INSERT INTO note_chunk_embeddings (
                chunk_id,
                model,
                dimensions,
                vector_json,
                chunk_content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "chunk-missing",
                "embed-model",
                1,
                "[0.1]",
                "0" * 64,
            ),
        )


def test_store_returns_json_serializable_embedding(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunk = persist_chunks(tmp_path, sqlite_connection)[0]
    store = NoteEmbeddingStore(sqlite_connection)

    stored = store.upsert_embedding(make_embedding(chunk, vector=(0.1, 0.2)))

    assert json.loads(stored.model_dump_json()) == {
        "chunk_id": chunk.chunk_id,
        "model": "embed-model",
        "dimensions": 2,
        "vector": [0.1, 0.2],
        "chunk_content_hash": chunk.content_hash,
    }


def persist_chunks(
    tmp_path: Path,
    connection: sqlite3.Connection,
    *,
    content: str = "note content",
) -> list[NoteChunk]:
    root = tmp_path / "notes"
    root.mkdir(exist_ok=True)
    note_path = root / "note.md"
    note_path.write_text(content, encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    result = ingest_note_path(note_path, policy, max_chunk_chars=3)
    NoteChunkStore(connection, policy).upsert_ingested_note(result)
    return list(result.chunks)


def persist_ordered_chunks(
    tmp_path: Path,
    connection: sqlite3.Connection,
) -> list[NoteChunk]:
    root = tmp_path / "notes"
    root.mkdir()
    path_b = root / "b.md"
    path_a = root / "a.md"
    path_b.write_text("b", encoding="utf-8")
    path_a.write_text("a", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    store = NoteChunkStore(connection, policy)
    result_b = ingest_note_path(path_b, policy)
    result_a = ingest_note_path(path_a, policy)
    store.upsert_ingested_note(result_b)
    store.upsert_ingested_note(result_a)
    return [result_a.chunks[0], result_b.chunks[0]]


def make_embedding(
    chunk: NoteChunk,
    *,
    vector: tuple[float, ...],
    chunk_content_hash: str | None = None,
) -> NoteChunkEmbedding:
    return NoteChunkEmbedding(
        chunk_id=chunk.chunk_id,
        model="embed-model",
        dimensions=len(vector),
        vector=vector,
        chunk_content_hash=chunk_content_hash or chunk.content_hash,
    )
