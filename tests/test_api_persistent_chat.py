import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
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
    LlmConnectionError,
    ModelInfo,
)
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.persistence import (
    Conversation,
    MemorySummary,
    Message,
    MessageRole,
    MessageStore,
    Session,
    open_sqlite_database,
)
from aigentego.settings import Settings


class SequencedProvider:
    provider_name = "fake"

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.chat_requests: list[ChatRequest] = []

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        response = self.responses[len(self.chat_requests) - 1]
        if isinstance(response, Exception):
            raise response
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=response),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    return tmp_path / "persistent-chat.sqlite3"


def make_client(
    sqlite_path: Path,
    provider: SequencedProvider,
) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        sqlite_path=str(sqlite_path),
        chat_model="persistent-chat-model",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app)


def test_persistent_chat_stores_user_and_assistant_turn(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["Stored assistant answer."])
    client = make_client(sqlite_path, provider)
    create_conversation(client)

    response = client.post(
        "/conversations/conversation-123/chat",
        headers={REQUEST_ID_HEADER: "persistent-chat-request"},
        json={"message": " Hello persistently. "},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "persistent-chat-request"
    assert response.json() == {
        "request_id": "persistent-chat-request",
        "session_id": "session-123",
        "conversation_id": "conversation-123",
        "model": "persistent-chat-model",
        "message": "Stored assistant answer.",
    }
    assert provider.chat_requests == [
        ChatRequest(
            model="persistent-chat-model",
            messages=[ChatMessage(role="user", content="Hello persistently.")],
        ),
    ]

    messages = list_messages(sqlite_path, "conversation-123")
    assert [(message.role.value, message.content) for message in messages] == [
        ("user", "Hello persistently."),
        ("assistant", "Stored assistant answer."),
    ]
    assert {message.session_id for message in messages} == {"session-123"}
    assert {message.conversation_id for message in messages} == {
        "conversation-123",
    }


def test_persistent_chat_injects_prior_conversation_messages(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["Second answer."])
    client = make_client(sqlite_path, provider)
    seed_conversation_messages(
        sqlite_path,
        [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ],
        [
            "Use short answers.",
            "What is stored?",
            "Stored context.",
        ],
    )

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": "Continue."},
    )

    assert response.status_code == 200
    assert provider.chat_requests[0].messages == [
        ChatMessage(role="system", content="Use short answers."),
        ChatMessage(role="user", content="What is stored?"),
        ChatMessage(role="assistant", content="Stored context."),
        ChatMessage(role="user", content="Continue."),
    ]
    messages = list_messages(sqlite_path, "conversation-123")
    assert [message.content for message in messages] == [
        "Use short answers.",
        "What is stored?",
        "Stored context.",
        "Continue.",
        "Second answer.",
    ]


def test_persistent_chat_injects_latest_memory_summary(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["Memory-aware answer."])
    client = make_client(sqlite_path, provider)
    seed_conversation_messages(
        sqlite_path,
        [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ],
        [
            "Earlier question.",
            "Earlier answer.",
        ],
    )
    seed_memory_summaries(sqlite_path)

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": "Use the summary."},
    )

    assert response.status_code == 200
    assert provider.chat_requests[0].messages == [
        ChatMessage(
            role="system",
            content=(
                "Conversation memory summary:\n"
                "Latest summary for this conversation."
            ),
        ),
        ChatMessage(role="user", content="Earlier question."),
        ChatMessage(role="assistant", content="Earlier answer."),
        ChatMessage(role="user", content="Use the summary."),
    ]
    messages = list_messages(sqlite_path, "conversation-123")
    assert [message.content for message in messages] == [
        "Earlier question.",
        "Earlier answer.",
        "Use the summary.",
        "Memory-aware answer.",
    ]


def test_persistent_chat_returns_not_found_for_missing_conversation(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["unused"])
    client = make_client(sqlite_path, provider)

    response = client.post(
        "/conversations/missing-conversation/chat",
        json={"message": "Hello."},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "conversation_not_found"
    assert provider.chat_requests == []


def test_persistent_chat_rejects_blank_message(sqlite_path: Path) -> None:
    provider = SequencedProvider(["unused"])
    client = make_client(sqlite_path, provider)
    create_conversation(client)

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": "   "},
    )

    assert response.status_code == 422
    assert provider.chat_requests == []
    assert list_messages(sqlite_path, "conversation-123") == []


def test_persistent_chat_provider_failure_stores_no_turn(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(
        [
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )
    client = make_client(sqlite_path, provider)
    create_conversation(client)

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": "Do not persist on failure."},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "llm_unavailable"
    assert len(provider.chat_requests) == 1
    assert list_messages(sqlite_path, "conversation-123") == []


def test_stateless_chat_does_not_write_persistent_messages(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["Stateless answer."])
    client = make_client(sqlite_path, provider)
    seed_conversation_messages(
        sqlite_path,
        [MessageRole.USER],
        ["Persisted context that stateless chat must ignore."],
    )
    seed_memory_summaries(sqlite_path)
    before_messages = list_messages(sqlite_path, "conversation-123")

    response = client.post("/chat", json={"message": "Hello stateless."})

    assert response.status_code == 200
    assert provider.chat_requests[0].messages == [
        ChatMessage(role="user", content="Hello stateless."),
    ]
    assert list_messages(sqlite_path, "conversation-123") == before_messages


def test_agent_run_does_not_write_persistent_messages(sqlite_path: Path) -> None:
    provider = SequencedProvider(
        [
            json.dumps({"tool_calls": []}),
            "Agent answer.",
        ],
    )
    client = make_client(sqlite_path, provider)
    seed_conversation_messages(
        sqlite_path,
        [MessageRole.USER],
        ["Persisted context that agent run must ignore."],
    )
    seed_memory_summaries(sqlite_path)
    before_messages = list_messages(sqlite_path, "conversation-123")

    response = client.post("/agent/run", json={"message": "Run without memory."})

    assert response.status_code == 200
    rendered_requests = json.dumps(
        [request.model_dump(mode="json") for request in provider.chat_requests],
    )
    assert "Persisted context that agent run must ignore." not in rendered_requests
    assert "Latest summary for this conversation." not in rendered_requests
    assert list_messages(sqlite_path, "conversation-123") == before_messages


def create_conversation(client: TestClient) -> None:
    session_response = client.post("/sessions", json={"session_id": "session-123"})
    conversation_response = client.post(
        "/sessions/session-123/conversations",
        json={"conversation_id": "conversation-123"},
    )
    assert session_response.status_code == 200
    assert conversation_response.status_code == 200


def seed_conversation_messages(
    sqlite_path: Path,
    roles: list[MessageRole],
    contents: list[str],
) -> None:
    with sqlite_connection(sqlite_path) as connection:
        store = MessageStore(connection)
        store.upsert_session(Session(session_id="session-123"))
        store.upsert_conversation(
            Conversation(
                conversation_id="conversation-123",
                session_id="session-123",
            ),
        )
        for index, (role, content) in enumerate(zip(roles, contents, strict=True)):
            store.append_message(
                Message(
                    message_id=f"message-{index}",
                    session_id="session-123",
                    conversation_id="conversation-123",
                    role=role,
                    content=content,
                ),
            )


def seed_memory_summaries(sqlite_path: Path) -> None:
    with sqlite_connection(sqlite_path) as connection:
        store = MessageStore(connection)
        store.upsert_session(Session(session_id="session-123"))
        store.upsert_conversation(
            Conversation(
                conversation_id="conversation-123",
                session_id="session-123",
            ),
        )
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-old",
                session_id="session-123",
                conversation_id="conversation-123",
                content="Old summary for this conversation.",
                revision=1,
            ),
        )
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-latest",
                session_id="session-123",
                conversation_id="conversation-123",
                content="Latest summary for this conversation.",
                revision=2,
            ),
        )


def list_messages(sqlite_path: Path, conversation_id: str) -> list[Message]:
    with sqlite_connection(sqlite_path) as connection:
        return MessageStore(connection).list_messages(conversation_id)


@contextmanager
def sqlite_connection(sqlite_path: Path) -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(str(sqlite_path))
    try:
        yield connection
    finally:
        connection.close()
