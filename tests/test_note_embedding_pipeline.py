import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aigentego.llm import EmbeddingRequest, EmbeddingResponse, LlmConnectionError
from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    NoteChunkStore,
    NoteEmbeddingPipeline,
    NoteEmbeddingResponseError,
    NoteEmbeddingStore,
    ReadOnlyFilesystemPolicy,
    ingest_note_path,
)
from aigentego.retrieval.ingestion import NoteChunk


class FakeEmbeddingProvider:
    provider_name = "fake"

    def __init__(
        self,
        responses: list[EmbeddingResponse | Exception],
    ) -> None:
        self.responses = responses
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_pipeline_embeds_and_persists_missing_chunks(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection, content="abcdef")
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="configured-embed-model",
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model=" configured-embed-model ",
    )

    embeddings = asyncio.run(pipeline.embed_chunks(chunks))

    assert provider.requests == [
        EmbeddingRequest(
            model="configured-embed-model",
            inputs=["abc", "def"],
        ),
    ]
    assert [embedding.chunk_id for embedding in embeddings] == [
        chunk.chunk_id for chunk in chunks
    ]
    assert [embedding.dimensions for embedding in embeddings] == [2, 2]
    assert store.list_embeddings("configured-embed-model") == list(embeddings)


def test_pipeline_is_idempotent_for_unchanged_chunks(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection, content="abc")
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="embed-model",
                embeddings=[[0.1, 0.2]],
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    first = asyncio.run(pipeline.embed_chunks(chunks))
    second = asyncio.run(pipeline.embed_chunks(chunks))

    assert len(first) == 1
    assert second == ()
    assert provider.requests == [
        EmbeddingRequest(model="embed-model", inputs=["abc"]),
    ]


def test_pipeline_does_not_call_provider_without_pending_chunks(
    sqlite_connection: sqlite3.Connection,
) -> None:
    provider = FakeEmbeddingProvider([])
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    embeddings = asyncio.run(pipeline.embed_chunks([]))

    assert embeddings == ()
    assert provider.requests == []


def test_pipeline_surfaces_provider_failure_without_persisting(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection)
    provider = FakeEmbeddingProvider(
        [
            LlmConnectionError(
                "provider unavailable",
                provider="fake",
                operation="embed",
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    with pytest.raises(LlmConnectionError):
        asyncio.run(pipeline.embed_chunks(chunks))

    assert store.list_embeddings("embed-model") == []


def test_pipeline_rejects_embedding_count_mismatch_without_persisting(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection, content="abcdef")
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="embed-model",
                embeddings=[[0.1]],
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    with pytest.raises(NoteEmbeddingResponseError, match="count"):
        asyncio.run(pipeline.embed_chunks(chunks))

    assert store.list_embeddings("embed-model") == []


def test_pipeline_rejects_model_mismatch_without_persisting(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection)
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="other-model",
                embeddings=[[0.1]],
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    with pytest.raises(NoteEmbeddingResponseError, match="model"):
        asyncio.run(pipeline.embed_chunks(chunks))

    assert store.list_embeddings("embed-model") == []


def test_pipeline_rejects_invalid_vector_without_persisting(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_chunks(tmp_path, sqlite_connection)
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="embed-model",
                embeddings=[[]],
            ),
        ],
    )
    store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteEmbeddingPipeline(
        provider=provider,
        store=store,
        model="embed-model",
    )

    with pytest.raises(NoteEmbeddingResponseError, match="invalid"):
        asyncio.run(pipeline.embed_chunks(chunks))

    assert store.list_embeddings("embed-model") == []


def persist_chunks(
    tmp_path: Path,
    connection: sqlite3.Connection,
    *,
    content: str = "abc",
) -> list[NoteChunk]:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text(content, encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    result = ingest_note_path(note_path, policy, max_chunk_chars=3)
    NoteChunkStore(connection, policy).upsert_ingested_note(result)
    return list(result.chunks)
