from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aigentego.api.dependencies import get_llm_provider, get_settings
from aigentego.llm import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
)
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    NoteChunkEmbedding,
    NoteChunkStore,
    NoteEmbeddingStore,
    ReadOnlyFilesystemPolicy,
    ingest_note_path,
)
from aigentego.settings import Settings


class FakeEmbeddingProvider:
    provider_name = "fake"

    def __init__(self, responses: list[EmbeddingResponse]) -> None:
        self.responses = responses
        self.embed_requests: list[EmbeddingRequest] = []
        self.chat_requests: list[ChatRequest] = []

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("notes search API must not call chat")

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embed_requests.append(request)
        return self.responses[len(self.embed_requests) - 1]


@pytest.fixture
def notes_root(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    root.mkdir()
    return root


def test_notes_search_api_returns_ranked_local_results(
    tmp_path: Path,
    notes_root: Path,
) -> None:
    sqlite_path = tmp_path / "notes-search.sqlite3"
    seed_notes(
        sqlite_path,
        notes_root,
        {
            "alpha.md": ("alpha note", (1.0, 0.0)),
            "beta.md": ("beta note", (0.0, 1.0)),
        },
    )
    provider = FakeEmbeddingProvider(
        [EmbeddingResponse(model="embed-api-model", embeddings=[[1.0, 0.0]])],
    )
    client = make_client(sqlite_path, notes_root, provider)

    response = client.post(
        "/notes/search",
        headers={REQUEST_ID_HEADER: "notes-search-request"},
        json={"query": " alpha query ", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "notes-search-request"
    body = response.json()
    assert body["request_id"] == "notes-search-request"
    assert body["search"]["query"] == "alpha query"
    assert body["search"]["model"] == "embed-api-model"
    assert body["search"]["top_k"] == 2
    assert [result["relative_path"] for result in body["search"]["results"]] == [
        "alpha.md",
        "beta.md",
    ]
    assert body["search"]["results"][0]["score"] == pytest.approx(1.0)
    assert body["search"]["results"][0]["snippet"] == "alpha note"
    assert not Path(body["search"]["results"][0]["relative_path"]).is_absolute()
    assert provider.embed_requests == [
        EmbeddingRequest(model="embed-api-model", inputs=["alpha query"]),
    ]
    assert provider.chat_requests == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": True},
        {"query": "valid", "unexpected": "field"},
    ],
)
def test_notes_search_api_rejects_invalid_payloads(
    tmp_path: Path,
    notes_root: Path,
    payload: dict[str, object],
) -> None:
    sqlite_path = tmp_path / "notes-search.sqlite3"
    provider = FakeEmbeddingProvider([])
    client = make_client(sqlite_path, notes_root, provider)

    response = client.post("/notes/search", json=payload)

    assert response.status_code == 422
    assert provider.embed_requests == []
    assert provider.chat_requests == []


def make_client(
    sqlite_path: Path,
    notes_root: Path,
    provider: FakeEmbeddingProvider,
) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        sqlite_path=str(sqlite_path),
        embedding_model="embed-api-model",
        notes_allowed_roots=(str(notes_root),),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app)


def seed_notes(
    sqlite_path: Path,
    notes_root: Path,
    notes: dict[str, tuple[str, tuple[float, ...]]],
) -> None:
    connection = open_sqlite_database(sqlite_path)
    try:
        policy = ReadOnlyFilesystemPolicy([notes_root])
        chunk_store = NoteChunkStore(connection, policy)
        embedding_store = NoteEmbeddingStore(connection)
        for relative_path, (content, vector) in notes.items():
            note_path = notes_root / relative_path
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(content, encoding="utf-8")
            ingested = chunk_store.upsert_ingested_note(
                ingest_note_path(note_path, policy),
            )
            chunk = ingested.chunks[0]
            embedding_store.upsert_embedding(
                NoteChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    model="embed-api-model",
                    dimensions=len(vector),
                    vector=vector,
                    chunk_content_hash=chunk.content_hash,
                ),
            )
    finally:
        connection.close()
