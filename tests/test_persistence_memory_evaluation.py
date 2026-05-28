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
from aigentego.persistence import (
    Conversation,
    MemorySummary,
    Message,
    MessageRole,
    MessageStore,
    Session,
    open_sqlite_database,
    read_schema_version,
)
from aigentego.persistence.sqlite import SCHEMA_VERSION, initialize_sqlite_schema
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
    return tmp_path / "memory-evaluation.sqlite3"


def test_local_persistence_models_and_schema_are_json_serializable(
    sqlite_path: Path,
) -> None:
    with sqlite_connection(sqlite_path) as connection:
        initialize_sqlite_schema(connection)
        initialize_sqlite_schema(connection)
        assert read_schema_version(connection) == SCHEMA_VERSION

        store = MessageStore(connection)
        session = store.upsert_session(
            Session(session_id="session-123", title="Local planning"),
        )
        conversation = store.upsert_conversation(
            Conversation(
                conversation_id="conversation-123",
                session_id=session.session_id,
                title="MVP 0.5",
            ),
        )
        user_message = Message(
            message_id="message-user",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            role=MessageRole.USER,
            content="Remember that persistence is local.",
        )
        assistant_message = Message(
            message_id="message-assistant",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content="Stored locally.",
        )
        old_summary = MemorySummary(
            summary_id="summary-old",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            content="Older local summary.",
            revision=1,
        )
        latest_summary = MemorySummary(
            summary_id="summary-latest",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            content="Latest local summary.",
            revision=2,
        )

        store.append_messages([user_message, assistant_message])
        store.upsert_memory_summary(latest_summary)
        store.upsert_memory_summary(old_summary)

        data = {
            "session": store.get_session(session.session_id).model_dump(mode="json"),
            "conversation": store.get_conversation(
                conversation.conversation_id,
            ).model_dump(mode="json"),
            "messages": [
                message.model_dump(mode="json")
                for message in store.list_messages(conversation.conversation_id)
            ],
            "latest_summary": store.get_latest_memory_summary(
                conversation.conversation_id,
            ).model_dump(mode="json"),
        }

    assert [message["message_id"] for message in data["messages"]] == [
        "message-user",
        "message-assistant",
    ]
    assert data["latest_summary"]["summary_id"] == "summary-latest"
    assert json.loads(json.dumps(data)) == data


def test_persistent_chat_closure_flow_stores_turn_and_injects_context(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(["Closure answer."])
    client = make_client(sqlite_path, provider)
    seed_persistent_conversation(sqlite_path)

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": " Continue with memory. "},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Closure answer."
    assert provider.chat_requests[0].messages == [
        ChatMessage(
            role="system",
            content=(
                "Conversation memory summary:\n"
                "Latest local memory summary."
            ),
        ),
        ChatMessage(role="user", content="What is the stored plan?"),
        ChatMessage(role="assistant", content="Use local persistence."),
        ChatMessage(role="user", content="Continue with memory."),
    ]

    messages = list_messages(sqlite_path, "conversation-123")
    assert [(message.role.value, message.content) for message in messages] == [
        ("user", "What is the stored plan?"),
        ("assistant", "Use local persistence."),
        ("user", "Continue with memory."),
        ("assistant", "Closure answer."),
    ]


def test_persistent_chat_provider_failure_stores_no_failed_turn(
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
    seed_persistent_conversation(sqlite_path)
    before_messages = list_messages(sqlite_path, "conversation-123")

    response = client.post(
        "/conversations/conversation-123/chat",
        json={"message": "This turn should not persist."},
    )

    assert response.status_code == 503
    assert len(provider.chat_requests) == 1
    assert list_messages(sqlite_path, "conversation-123") == before_messages


def test_stateless_chat_and_agent_run_do_not_use_persistence(
    sqlite_path: Path,
) -> None:
    provider = SequencedProvider(
        [
            "Stateless answer.",
            json.dumps({"tool_calls": []}),
            "Agent answer.",
        ],
    )
    client = make_client(sqlite_path, provider)
    seed_persistent_conversation(sqlite_path)
    before_messages = list_messages(sqlite_path, "conversation-123")

    chat_response = client.post("/chat", json={"message": "Ignore memory."})
    agent_response = client.post("/agent/run", json={"message": "Ignore memory."})

    assert chat_response.status_code == 200
    assert agent_response.status_code == 200
    assert provider.chat_requests[0].messages == [
        ChatMessage(role="user", content="Ignore memory."),
    ]
    rendered_agent_requests = json.dumps(
        [
            request.model_dump(mode="json")
            for request in provider.chat_requests[1:]
        ],
    )
    assert "Latest local memory summary." not in rendered_agent_requests
    assert "What is the stored plan?" not in rendered_agent_requests
    assert list_messages(sqlite_path, "conversation-123") == before_messages


def make_client(sqlite_path: Path, provider: SequencedProvider) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        sqlite_path=str(sqlite_path),
        chat_model="memory-evaluation-model",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app)


def seed_persistent_conversation(sqlite_path: Path) -> None:
    with sqlite_connection(sqlite_path) as connection:
        store = MessageStore(connection)
        session = store.upsert_session(Session(session_id="session-123"))
        conversation = store.upsert_conversation(
            Conversation(
                conversation_id="conversation-123",
                session_id=session.session_id,
            ),
        )
        store.append_messages(
            [
                Message(
                    message_id="message-user",
                    session_id=session.session_id,
                    conversation_id=conversation.conversation_id,
                    role=MessageRole.USER,
                    content="What is the stored plan?",
                ),
                Message(
                    message_id="message-assistant",
                    session_id=session.session_id,
                    conversation_id=conversation.conversation_id,
                    role=MessageRole.ASSISTANT,
                    content="Use local persistence.",
                ),
            ],
        )
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-old",
                session_id=session.session_id,
                conversation_id=conversation.conversation_id,
                content="Old local memory summary.",
                revision=1,
            ),
        )
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-latest",
                session_id=session.session_id,
                conversation_id=conversation.conversation_id,
                content="Latest local memory summary.",
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
