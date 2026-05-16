import logging
from uuid import UUID

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
from aigentego.observability import REQUEST_ID_HEADER, configure_logging
from aigentego.settings import Settings


class MockProvider:
    provider_name = "mock"

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content="Mock response."),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


def make_client() -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="llama3.2:3b",
        ollama_embed_model="nomic-embed-text",
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: MockProvider()

    return TestClient(app)


def test_request_id_is_generated_when_header_is_missing() -> None:
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]
    assert str(UUID(request_id)) == request_id


def test_incoming_request_id_is_preserved() -> None:
    client = make_client()

    response = client.get("/health", headers={REQUEST_ID_HEADER: "external-id-123"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "external-id-123"


def test_response_contains_request_id_header() -> None:
    client = make_client()

    response = client.get("/diagnostics")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


def test_chat_response_request_id_matches_response_header() -> None:
    client = make_client()

    response = client.post(
        "/chat",
        headers={REQUEST_ID_HEADER: "chat-request-123"},
        json={"message": "Say hello."},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "chat-request-123"
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_request_logging_does_not_require_live_ollama_server(caplog) -> None:
    client = make_client()

    with caplog.at_level(
        logging.INFO,
        logger="aigentego.observability.request_context",
    ):
        response = client.post("/chat", json={"message": "Do not log this prompt."})

    assert response.status_code == 200
    records = [
        record
        for record in caplog.records
        if record.name == "aigentego.observability.request_context"
    ]
    assert records
    assert any(
        "method=POST" in record.getMessage()
        and "path=/chat" in record.getMessage()
        and "status_code=200" in record.getMessage()
        for record in records
    )
    log_messages = "\n".join(record.getMessage() for record in records)
    assert "Do not log this prompt." not in log_messages
    assert "Mock response." not in log_messages


def test_logging_setup_is_idempotent() -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level

    try:
        configure_logging("INFO")
        first_count = _aigentego_handler_count()
        configure_logging("DEBUG")
        second_count = _aigentego_handler_count()
    finally:
        root_logger.setLevel(original_level)

    assert first_count == 1
    assert second_count == 1


def _aigentego_handler_count() -> int:
    return sum(
        1
        for handler in logging.getLogger().handlers
        if bool(getattr(handler, "_aigentego_handler", False))
    )
