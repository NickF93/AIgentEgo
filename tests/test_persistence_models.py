import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import aigentego.persistence as persistence
from aigentego.persistence import (
    Conversation,
    ConversationStatus,
    Session,
    SessionStatus,
)


def timestamp() -> datetime:
    return datetime(2026, 5, 26, 12, 30, tzinfo=UTC)


def test_valid_session_model_can_be_created() -> None:
    created_at = timestamp()
    updated_at = created_at + timedelta(minutes=5)

    session = Session(
        session_id="session-123",
        title="Planning",
        status=SessionStatus.ACTIVE,
        created_at=created_at,
        updated_at=updated_at,
    )

    assert session.session_id == "session-123"
    assert session.title == "Planning"
    assert session.status is SessionStatus.ACTIVE
    assert session.created_at == created_at
    assert session.updated_at == updated_at


def test_valid_conversation_model_can_be_created() -> None:
    conversation = Conversation(
        conversation_id="conversation-123",
        session_id="session-123",
        title="Default conversation",
        status=ConversationStatus.ACTIVE,
        is_default=True,
    )

    assert conversation.conversation_id == "conversation-123"
    assert conversation.session_id == "session-123"
    assert conversation.title == "Default conversation"
    assert conversation.status is ConversationStatus.ACTIVE
    assert conversation.is_default is True


def test_conversation_represents_session_relationship_and_default_flag() -> None:
    session = Session(session_id="session-123")
    conversation = Conversation(
        conversation_id="conversation-123",
        session_id=session.session_id,
        is_default=True,
    )

    assert conversation.session_id == session.session_id
    assert conversation.is_default is True


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": ""},
        {"session_id": "   "},
        {"session_id": "session-123", "title": "   "},
    ],
)
def test_session_rejects_blank_ids_and_optional_text(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        Session.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"conversation_id": "", "session_id": "session-123"},
        {"conversation_id": "conversation-123", "session_id": ""},
        {"conversation_id": "   ", "session_id": "session-123"},
        {"conversation_id": "conversation-123", "session_id": "   "},
        {
            "conversation_id": "conversation-123",
            "session_id": "session-123",
            "title": "   ",
        },
    ],
)
def test_conversation_rejects_blank_ids_and_optional_text(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        Conversation.model_validate(payload)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Session.model_validate(
            {
                "session_id": "session-123",
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        Conversation.model_validate(
            {
                "conversation_id": "conversation-123",
                "session_id": "session-123",
                "unexpected": "field",
            },
        )


def test_timestamps_must_be_timezone_aware() -> None:
    naive_timestamp = datetime(2026, 5, 26, 12, 30)

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        Session(session_id="session-123", created_at=naive_timestamp)

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        Conversation(
            conversation_id="conversation-123",
            session_id="session-123",
            updated_at=naive_timestamp,
        )


def test_updated_at_must_not_be_before_created_at() -> None:
    created_at = timestamp()
    updated_at = created_at - timedelta(seconds=1)

    with pytest.raises(ValidationError, match="updated_at must not be before"):
        Session(
            session_id="session-123",
            created_at=created_at,
            updated_at=updated_at,
        )

    with pytest.raises(ValidationError, match="updated_at must not be before"):
        Conversation(
            conversation_id="conversation-123",
            session_id="session-123",
            created_at=created_at,
            updated_at=updated_at,
        )


def test_models_serialize_to_json_compatible_data() -> None:
    created_at = timestamp()
    updated_at = created_at + timedelta(minutes=5)

    session = Session(
        session_id="session-123",
        title="Planning",
        status=SessionStatus.ARCHIVED,
        created_at=created_at,
        updated_at=updated_at,
    )
    conversation = Conversation(
        conversation_id="conversation-123",
        session_id=session.session_id,
        title="Default conversation",
        status=ConversationStatus.ARCHIVED,
        is_default=True,
        created_at=created_at,
        updated_at=updated_at,
    )

    data = {
        "session": session.model_dump(mode="json"),
        "conversation": conversation.model_dump(mode="json"),
    }

    assert json.loads(
        json.dumps(data),
    ) == data
    assert data == {
        "session": {
            "session_id": "session-123",
            "title": "Planning",
            "status": "archived",
            "created_at": "2026-05-26T12:30:00Z",
            "updated_at": "2026-05-26T12:35:00Z",
        },
        "conversation": {
            "conversation_id": "conversation-123",
            "session_id": "session-123",
            "title": "Default conversation",
            "status": "archived",
            "is_default": True,
            "created_at": "2026-05-26T12:30:00Z",
            "updated_at": "2026-05-26T12:35:00Z",
        },
    }


def test_message_model_and_store_are_not_exported_yet() -> None:
    assert not hasattr(persistence, "Message")
    assert not hasattr(persistence, "MessageStore")
