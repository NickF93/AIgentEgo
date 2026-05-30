import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aigentego.api.dependencies import get_llm_provider, get_settings
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
)
from aigentego.main import create_app
from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    FileMetadataStore,
    NoteChunkEmbedding,
    NoteChunkStore,
    NoteEmbeddingPipeline,
    NoteEmbeddingStore,
    NoteSearchPipeline,
    PathOutsideAllowedRootsError,
    RagContextLimits,
    ReadOnlyFilesystemPolicy,
    build_rag_context,
    discover_note_files,
    format_rag_context_for_prompt,
    ingest_note_file,
    ingest_note_path,
)
from aigentego.settings import Settings


class FakeRetrievalProvider:
    provider_name = "fake"

    def __init__(
        self,
        *,
        embedding_responses: list[EmbeddingResponse],
        chat_responses: list[str] | None = None,
    ) -> None:
        self.embedding_responses = embedding_responses
        self.chat_responses = chat_responses or []
        self.embed_requests: list[EmbeddingRequest] = []
        self.chat_requests: list[ChatRequest] = []

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        response = self.chat_responses[len(self.chat_requests) - 1]
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=response),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embed_requests.append(request)
        return self.embedding_responses[len(self.embed_requests) - 1]


def test_local_retrieval_pipeline_is_read_only_and_context_bounded(
    tmp_path: Path,
) -> None:
    notes_root = tmp_path / "notes"
    outside_root = tmp_path / "outside"
    notes_root.mkdir()
    outside_root.mkdir()
    alpha_path = notes_root / "alpha.md"
    beta_path = notes_root / "beta.txt"
    unsupported_path = notes_root / "image.png"
    outside_path = outside_root / "secret.md"
    alpha_path.write_text("Alpha note\r\nImportant local context.", encoding="utf-8")
    beta_path.write_text("Beta note\nReference material.", encoding="utf-8")
    unsupported_path.write_text("not a note", encoding="utf-8")
    outside_path.write_text("outside", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([notes_root])

    with pytest.raises(PathOutsideAllowedRootsError):
        policy.resolve_read_path(outside_path)

    discovered = discover_note_files(policy)

    assert [metadata.relative_path for metadata in discovered] == [
        "alpha.md",
        "beta.txt",
    ]

    connection = open_sqlite_database(":memory:")
    try:
        metadata_store = FileMetadataStore(connection, policy)
        chunk_store = NoteChunkStore(connection, policy)
        embedding_store = NoteEmbeddingStore(connection)
        metadata_store.upsert_files(discovered)

        alpha_ingestion = ingest_note_file(
            discovered[0],
            policy,
            max_chunk_chars=80,
        )
        beta_ingestion = ingest_note_file(
            discovered[1],
            policy,
            max_chunk_chars=80,
        )
        assert alpha_ingestion.document.content_length == len(
            "Alpha note\nImportant local context.",
        )
        assert [chunk.content for chunk in alpha_ingestion.chunks] == [
            "Alpha note\nImportant local context.",
        ]

        stored_alpha = chunk_store.upsert_ingested_note(alpha_ingestion)
        stored_beta = chunk_store.upsert_ingested_note(beta_ingestion)
        chunks = [stored_alpha.chunks[0], stored_beta.chunks[0]]

        embedding_provider = FakeRetrievalProvider(
            embedding_responses=[
                EmbeddingResponse(
                    model="embed-model",
                    embeddings=[[1.0, 0.0], [0.0, 1.0]],
                ),
            ],
        )
        embeddings = asyncio.run(
            NoteEmbeddingPipeline(
                provider=embedding_provider,
                store=embedding_store,
                model="embed-model",
            ).embed_chunks(chunks),
        )

        assert [embedding.chunk_id for embedding in embeddings] == [
            chunk.chunk_id for chunk in chunks
        ]
        assert embedding_provider.embed_requests == [
            EmbeddingRequest(
                model="embed-model",
                inputs=[chunk.content for chunk in chunks],
            ),
        ]

        search_provider = FakeRetrievalProvider(
            embedding_responses=[
                EmbeddingResponse(model="embed-model", embeddings=[[1.0, 0.0]]),
            ],
        )
        search = asyncio.run(
            NoteSearchPipeline(
                provider=search_provider,
                store=embedding_store,
                model="embed-model",
            ).search(" alpha query ", top_k=2),
        )

        assert [result.relative_path for result in search.results] == [
            "alpha.md",
            "beta.txt",
        ]
        assert search.results[0].score == pytest.approx(1.0)
        assert not Path(search.results[0].relative_path).is_absolute()

        context = build_rag_context(
            search.results,
            limits=RagContextLimits(max_snippets=1, max_total_characters=10),
        )

        assert context.items[0].snippet == "Alpha note"
        assert context.items[0].relative_path == "alpha.md"
        assert context.items[0].truncated is True
        assert context.source_result_count == 2
        assert context.excluded_by_limit_count == 1
        assert format_rag_context_for_prompt(context) == (
            "Retrieved local note context:\n"
            "[1] source=alpha.md chunk=0 score=1.000000 truncated\n"
            "Alpha note"
        )
        assert json.loads(search.model_dump_json())["results"][0][
            "relative_path"
        ] == "alpha.md"
        assert json.loads(context.model_dump_json())["items"][0][
            "relative_path"
        ] == "alpha.md"
    finally:
        connection.close()


def test_notes_rag_apis_are_explicit_and_runtime_paths_remain_non_rag(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    seed_embedded_notes(
        sqlite_path,
        notes_root,
        {
            "local.md": ("Local RAG source text.", (1.0, 0.0)),
            "other.md": ("Other local note.", (0.0, 1.0)),
        },
    )
    provider = FakeRetrievalProvider(
        embedding_responses=[
            EmbeddingResponse(model="embed-api-model", embeddings=[[1.0, 0.0]]),
            EmbeddingResponse(model="embed-api-model", embeddings=[[1.0, 0.0]]),
        ],
        chat_responses=[
            "Stateless chat answer.",
            "Persistent chat answer.",
            json.dumps({"tool_calls": []}),
            "Agent final answer.",
        ],
    )
    client = make_client(sqlite_path, notes_root, provider)

    search_response = client.post(
        "/notes/search",
        json={"query": "local", "top_k": 2},
    )
    context_response = client.post(
        "/rag/context",
        json={
            "query": "local",
            "top_k": 2,
            "max_snippets": 1,
            "max_total_characters": 8,
        },
    )

    assert search_response.status_code == 200
    assert context_response.status_code == 200
    search_body = search_response.json()
    context_body = context_response.json()
    assert [result["relative_path"] for result in search_body["search"]["results"]] == [
        "local.md",
        "other.md",
    ]
    assert not Path(search_body["search"]["results"][0]["relative_path"]).is_absolute()
    assert context_body["context"]["items"][0]["snippet"] == "Local RA"
    assert "formatted_context" in context_body
    assert "final_answer" not in context_body
    assert provider.chat_requests == []

    chat_response = client.post("/chat", json={"message": "Use no RAG."})
    create_conversation(client)
    persistent_chat_response = client.post(
        "/conversations/evaluation-conversation/chat",
        json={"message": "Continue without RAG."},
    )
    agent_response = client.post(
        "/agent/run",
        json={"message": "Run without RAG context."},
    )

    assert chat_response.status_code == 200
    assert persistent_chat_response.status_code == 200
    assert agent_response.status_code == 200
    assert provider.embed_requests == [
        EmbeddingRequest(model="embed-api-model", inputs=["local"]),
        EmbeddingRequest(model="embed-api-model", inputs=["local"]),
    ]
    assert [request.messages[-1].content for request in provider.chat_requests] == [
        "Use no RAG.",
        "Continue without RAG.",
        provider.chat_requests[2].messages[-1].content,
        provider.chat_requests[3].messages[-1].content,
    ]
    rendered_chat_requests = json.dumps(
        [request.model_dump(mode="json") for request in provider.chat_requests],
    )
    assert "Local RAG source text." not in rendered_chat_requests
    assert "Other local note." not in rendered_chat_requests


def make_client(
    sqlite_path: Path,
    notes_root: Path,
    provider: FakeRetrievalProvider,
) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        sqlite_path=str(sqlite_path),
        chat_model="chat-api-model",
        embedding_model="embed-api-model",
        notes_allowed_roots=(str(notes_root),),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app)


def create_conversation(client: TestClient) -> None:
    session_response = client.post(
        "/sessions",
        json={"session_id": "evaluation-session"},
    )
    conversation_response = client.post(
        "/sessions/evaluation-session/conversations",
        json={"conversation_id": "evaluation-conversation"},
    )

    assert session_response.status_code == 200
    assert conversation_response.status_code == 200


def seed_embedded_notes(
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
