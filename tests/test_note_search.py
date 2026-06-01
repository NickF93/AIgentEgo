import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from aigentego.llm import EmbeddingRequest, EmbeddingResponse, LlmConnectionError
from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    NoteChunk,
    NoteChunkEmbedding,
    NoteChunkStore,
    NoteEmbeddingStore,
    NoteSearchPipeline,
    NoteSearchResponseError,
    ReadOnlyFilesystemPolicy,
    ingest_note_path,
)


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


def test_search_returns_ranked_chunks_with_safe_metadata(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {"alpha.md": "alpha note", "beta.md": "beta note"},
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    embedding_store.upsert_embeddings(
        [
            make_embedding(chunks["alpha.md"], vector=(1.0, 0.0)),
            make_embedding(chunks["beta.md"], vector=(0.0, 1.0)),
        ],
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0, 0.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model=" embed-model ",
    )

    response = asyncio.run(pipeline.search(" alpha query ", top_k=5))

    assert provider.requests == [
        EmbeddingRequest(model="embed-model", inputs=["alpha query"]),
    ]
    assert response.query == "alpha query"
    assert response.candidate_count == 2
    assert response.skipped_stale_embeddings == 0
    assert response.skipped_dimension_mismatches == 0
    assert [result.relative_path for result in response.results] == [
        "alpha.md",
        "beta.md",
    ]
    assert response.results[0].chunk_id == chunks["alpha.md"].chunk_id
    assert response.results[0].score == pytest.approx(1.0)
    assert response.results[0].snippet == "alpha note"
    assert not Path(response.results[0].relative_path).is_absolute()
    rendered = json.loads(response.model_dump_json())
    assert rendered["results"][0]["relative_path"] == "alpha.md"


def test_top_k_limits_ranked_results(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {
            "alpha.md": "alpha",
            "beta.md": "beta",
            "gamma.md": "gamma",
        },
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    embedding_store.upsert_embeddings(
        [
            make_embedding(chunks["alpha.md"], vector=(1.0, 0.0)),
            make_embedding(chunks["beta.md"], vector=(0.5, 0.0)),
            make_embedding(chunks["gamma.md"], vector=(0.0, 1.0)),
        ],
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0, 0.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    response = asyncio.run(pipeline.search("rank", top_k=2))

    assert [result.relative_path for result in response.results] == [
        "alpha.md",
        "beta.md",
    ]
    assert response.top_k == 2


def test_score_ties_use_stable_source_order(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {"b.md": "b note", "a.md": "a note"},
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    embedding_store.upsert_embeddings(
        [
            make_embedding(chunks["b.md"], vector=(1.0, 0.0)),
            make_embedding(chunks["a.md"], vector=(1.0, 0.0)),
        ],
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0, 0.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    response = asyncio.run(pipeline.search("tie"))

    assert [result.relative_path for result in response.results] == [
        "a.md",
        "b.md",
    ]


def test_blank_query_and_invalid_top_k_are_rejected(
    sqlite_connection: sqlite3.Connection,
) -> None:
    provider = FakeEmbeddingProvider([])
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=NoteEmbeddingStore(sqlite_connection),
        model="embed-model",
    )

    with pytest.raises(ValidationError, match="query"):
        asyncio.run(pipeline.search("   "))

    with pytest.raises(ValidationError, match="top_k"):
        asyncio.run(pipeline.search("valid", top_k=0))

    assert provider.requests == []


def test_missing_embeddings_returns_empty_inspectable_response(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {"alpha.md": "alpha note"},
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    response = asyncio.run(pipeline.search("alpha"))

    assert response.candidate_count == 0
    assert response.results == ()
    assert provider.requests == [
        EmbeddingRequest(model="embed-model", inputs=["alpha"]),
    ]


def test_dimension_mismatches_are_skipped(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {"alpha.md": "alpha", "wide.md": "wide"},
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    embedding_store.upsert_embeddings(
        [
            make_embedding(chunks["alpha.md"], vector=(1.0, 0.0)),
            make_embedding(chunks["wide.md"], vector=(1.0, 0.0, 0.0)),
        ],
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0, 0.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    response = asyncio.run(pipeline.search("alpha"))

    assert response.candidate_count == 2
    assert response.skipped_dimension_mismatches == 1
    assert [result.relative_path for result in response.results] == ["alpha.md"]


def test_stale_embeddings_are_skipped(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    chunks = persist_single_chunk_notes(
        tmp_path,
        sqlite_connection,
        {"alpha.md": "alpha"},
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    embedding_store.upsert_embedding(
        make_embedding(
            chunks["alpha.md"],
            vector=(1.0,),
            chunk_content_hash="0" * 64,
        ),
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-model", embeddings=[[1.0]])],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    response = asyncio.run(pipeline.search("alpha"))

    assert response.candidate_count == 1
    assert response.skipped_stale_embeddings == 1
    assert response.results == ()


def test_provider_failure_propagates_without_local_side_effects(
    sqlite_connection: sqlite3.Connection,
) -> None:
    provider = FakeEmbeddingProvider(
        [
            LlmConnectionError(
                "provider unavailable",
                provider="fake",
                operation="embed",
            ),
        ],
    )
    embedding_store = NoteEmbeddingStore(sqlite_connection)
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=embedding_store,
        model="embed-model",
    )

    with pytest.raises(LlmConnectionError):
        asyncio.run(pipeline.search("alpha"))

    assert embedding_store.list_embeddings("embed-model") == []


def test_invalid_query_embedding_response_is_rejected(
    sqlite_connection: sqlite3.Connection,
) -> None:
    provider = FakeEmbeddingProvider(
        [
            EmbeddingResponse(
                model="other-model",
                embeddings=[[1.0]],
            ),
        ],
    )
    pipeline = NoteSearchPipeline(
        provider=provider,
        store=NoteEmbeddingStore(sqlite_connection),
        model="embed-model",
    )

    with pytest.raises(NoteSearchResponseError, match="model"):
        asyncio.run(pipeline.search("alpha"))


def persist_single_chunk_notes(
    tmp_path: Path,
    connection: sqlite3.Connection,
    notes: dict[str, str],
) -> dict[str, NoteChunk]:
    root = tmp_path / "notes"
    root.mkdir()
    policy = ReadOnlyFilesystemPolicy([root])
    store = NoteChunkStore(connection, policy)
    chunks: dict[str, NoteChunk] = {}
    for relative_path, content in notes.items():
        note_path = root / relative_path
        note_path.write_text(content, encoding="utf-8")
        result = ingest_note_path(note_path, policy)
        stored = store.upsert_ingested_note(result)
        chunks[relative_path] = stored.chunks[0]
    return chunks


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
