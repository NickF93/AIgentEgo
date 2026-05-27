import pytest
from fastapi.testclient import TestClient

from aigentego.api.dependencies import get_settings
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.settings import Settings


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        sqlite_path=str(tmp_path / "aigentego-api.sqlite3"),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_sessions_api_creates_and_gets_session_with_request_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/sessions",
        headers={REQUEST_ID_HEADER: "session-request-123"},
        json={"session_id": " session-123 ", "title": " Planning "},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "session-request-123"
    assert response.json() == {
        "request_id": "session-request-123",
        "session": {
            "session_id": "session-123",
            "title": "Planning",
            "status": "active",
            "created_at": None,
            "updated_at": None,
        },
    }

    get_response = client.get(
        "/sessions/session-123",
        headers={REQUEST_ID_HEADER: "session-get-123"},
    )

    assert get_response.status_code == 200
    assert get_response.headers[REQUEST_ID_HEADER] == "session-get-123"
    assert get_response.json()["request_id"] == "session-get-123"
    assert get_response.json()["session"]["session_id"] == "session-123"


def test_sessions_api_lists_sessions_deterministically(client: TestClient) -> None:
    client.post("/sessions", json={"session_id": "session-b", "title": "Second"})
    client.post("/sessions", json={"session_id": "session-a", "title": "First"})

    response = client.get("/sessions")

    assert response.status_code == 200
    assert [session["session_id"] for session in response.json()["sessions"]] == [
        "session-a",
        "session-b",
    ]


def test_conversations_api_creates_gets_and_lists_conversations(
    client: TestClient,
) -> None:
    client.post("/sessions", json={"session_id": "session-123"})

    first_response = client.post(
        "/sessions/session-123/conversations",
        headers={REQUEST_ID_HEADER: "conversation-request-123"},
        json={
            "conversation_id": " conversation-b ",
            "title": " Second ",
            "is_default": True,
        },
    )
    second_response = client.post(
        "/sessions/session-123/conversations",
        json={"conversation_id": "conversation-a", "title": "First"},
    )

    assert first_response.status_code == 200
    assert first_response.headers[REQUEST_ID_HEADER] == "conversation-request-123"
    assert first_response.json() == {
        "request_id": "conversation-request-123",
        "conversation": {
            "conversation_id": "conversation-b",
            "session_id": "session-123",
            "title": "Second",
            "status": "active",
            "is_default": True,
            "created_at": None,
            "updated_at": None,
        },
    }
    assert second_response.status_code == 200

    get_response = client.get("/conversations/conversation-b")
    assert get_response.status_code == 200
    assert get_response.json()["conversation"]["conversation_id"] == "conversation-b"

    list_response = client.get("/sessions/session-123/conversations")
    assert list_response.status_code == 200
    assert [
        conversation["conversation_id"]
        for conversation in list_response.json()["conversations"]
    ] == ["conversation-a", "conversation-b"]


def test_sessions_api_returns_not_found_for_missing_records(
    client: TestClient,
) -> None:
    missing_session = client.get("/sessions/missing-session")
    missing_session_conversations = client.get(
        "/sessions/missing-session/conversations",
    )
    missing_conversation = client.get("/conversations/missing-conversation")

    assert missing_session.status_code == 404
    assert missing_session.json()["detail"]["error"] == "session_not_found"
    assert missing_session_conversations.status_code == 404
    assert (
        missing_session_conversations.json()["detail"]["error"]
        == "session_not_found"
    )
    assert missing_conversation.status_code == 404
    assert missing_conversation.json()["detail"]["error"] == "conversation_not_found"


def test_conversation_creation_requires_existing_session(client: TestClient) -> None:
    response = client.post(
        "/sessions/missing-session/conversations",
        json={"conversation_id": "conversation-123"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "session_not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": ""},
        {"session_id": "   "},
        {"session_id": "session-123", "title": "   "},
        {"session_id": "session-123", "unexpected": "field"},
    ],
)
def test_sessions_api_rejects_invalid_session_payloads(
    client: TestClient,
    payload: dict[str, str],
) -> None:
    response = client.post("/sessions", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"conversation_id": ""},
        {"conversation_id": "   "},
        {"conversation_id": "conversation-123", "title": "   "},
        {"conversation_id": "conversation-123", "unexpected": "field"},
    ],
)
def test_sessions_api_rejects_invalid_conversation_payloads(
    client: TestClient,
    payload: dict[str, str],
) -> None:
    client.post("/sessions", json={"session_id": "session-123"})

    response = client.post(
        "/sessions/session-123/conversations",
        json=payload,
    )

    assert response.status_code == 422
