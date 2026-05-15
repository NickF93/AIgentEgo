from fastapi.testclient import TestClient

from aigentego.api.dependencies import get_llm_provider, get_settings
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmConnectionError,
    ModelInfo,
)
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.settings import Settings


class MockProvider:
    provider_name = "mock"

    def __init__(self, *, reachable: bool = True, fail_chat: bool = False) -> None:
        self.reachable = reachable
        self.fail_chat = fail_chat
        self.chat_request: ChatRequest | None = None

    async def health(self) -> bool:
        return self.reachable

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_request = request
        if self.fail_chat:
            raise LlmConnectionError(
                "request failed",
                provider=self.provider_name,
                operation="chat",
            )
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content="Hello from the model."),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


def make_client(
    provider: MockProvider | None = None,
    settings: Settings | None = None,
) -> tuple[TestClient, MockProvider]:
    app = create_app()
    provider = provider or MockProvider()
    settings = settings or Settings(
        _env_file=None,
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="llama3.2:3b",
        ollama_embed_model="nomic-embed-text",
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider

    return TestClient(app), provider


def test_app_can_be_created() -> None:
    app = create_app()

    assert app.title == "AIgentEgo"


def test_health_returns_expected_status_with_mocked_provider() -> None:
    client, _provider = make_client(MockProvider(reachable=True))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ollama_reachable": True,
        "chat_model": "llama3.2:3b",
    }


def test_diagnostics_returns_non_secret_configuration_fields() -> None:
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="llama3.2:3b",
        ollama_embed_model="nomic-embed-text",
    )
    client, _provider = make_client(MockProvider(reachable=False), settings)

    response = client.get("/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "mock"
    assert body["ollama_base_url"] == "http://ollama:11434"
    assert body["chat_model"] == "llama3.2:3b"
    assert body["embedding_model"] == "nomic-embed-text"
    assert body["ollama_reachable"] is False
    assert "environment" not in body
    assert "log_level" not in body


def test_chat_returns_model_response_with_request_id() -> None:
    client, provider = make_client()

    response = client.post("/chat", json={"message": " Say hello. "})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["request_id"], str)
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert body["model"] == "llama3.2:3b"
    assert body["message"] == "Hello from the model."
    assert provider.chat_request is not None
    assert provider.chat_request.messages == [
        ChatMessage(role="user", content="Say hello.")
    ]


def test_chat_maps_provider_errors_to_clear_http_errors() -> None:
    client, _provider = make_client(MockProvider(fail_chat=True))

    response = client.post("/chat", json={"message": "Say hello."})

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "llm_unavailable"


def test_chat_rejects_invalid_request_payloads() -> None:
    client, _provider = make_client()

    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
