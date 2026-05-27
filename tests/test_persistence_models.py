import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aigentego.persistence import (
    Conversation,
    ConversationStatus,
    MemorySummary,
    Message,
    MessageRole,
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


def test_valid_message_model_can_be_created() -> None:
    created_at = timestamp()

    message = Message(
        message_id="message-123",
        session_id="session-123",
        conversation_id="conversation-123",
        role=MessageRole.USER,
        content="Hello.",
        created_at=created_at,
    )

    assert message.message_id == "message-123"
    assert message.session_id == "session-123"
    assert message.conversation_id == "conversation-123"
    assert message.role is MessageRole.USER
    assert message.content == "Hello."
    assert message.created_at == created_at


def test_valid_memory_summary_model_can_be_created() -> None:
    created_at = timestamp()
    updated_at = created_at + timedelta(minutes=5)

    summary = MemorySummary(
        summary_id="summary-123",
        session_id="session-123",
        conversation_id="conversation-123",
        content="The user is planning a local-first assistant.",
        revision=2,
        created_at=created_at,
        updated_at=updated_at,
    )

    assert summary.summary_id == "summary-123"
    assert summary.session_id == "session-123"
    assert summary.conversation_id == "conversation-123"
    assert summary.content == "The user is planning a local-first assistant."
    assert summary.revision == 2
    assert summary.created_at == created_at
    assert summary.updated_at == updated_at


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


@pytest.mark.parametrize(
    "payload",
    [
        {
            "message_id": "",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "role": "user",
            "content": "Hello.",
        },
        {
            "message_id": "message-123",
            "session_id": "",
            "conversation_id": "conversation-123",
            "role": "user",
            "content": "Hello.",
        },
        {
            "message_id": "message-123",
            "session_id": "session-123",
            "conversation_id": "",
            "role": "user",
            "content": "Hello.",
        },
        {
            "message_id": "message-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "role": "user",
            "content": "",
        },
        {
            "message_id": "   ",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "role": "user",
            "content": "Hello.",
        },
        {
            "message_id": "message-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "role": "user",
            "content": "   ",
        },
    ],
)
def test_message_rejects_blank_ids_and_content(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "summary_id": "",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "content": "Summary.",
        },
        {
            "summary_id": "summary-123",
            "session_id": "",
            "conversation_id": "conversation-123",
            "content": "Summary.",
        },
        {
            "summary_id": "summary-123",
            "session_id": "session-123",
            "conversation_id": "",
            "content": "Summary.",
        },
        {
            "summary_id": "summary-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "content": "",
        },
        {
            "summary_id": "   ",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "content": "Summary.",
        },
        {
            "summary_id": "summary-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "content": "   ",
        },
    ],
)
def test_memory_summary_rejects_blank_ids_and_content(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        MemorySummary.model_validate(payload)


def test_memory_summary_rejects_invalid_revision() -> None:
    with pytest.raises(ValidationError):
        MemorySummary(
            summary_id="summary-123",
            session_id="session-123",
            conversation_id="conversation-123",
            content="Summary.",
            revision=0,
        )


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

    with pytest.raises(ValidationError):
        Message.model_validate(
            {
                "message_id": "message-123",
                "session_id": "session-123",
                "conversation_id": "conversation-123",
                "role": "user",
                "content": "Hello.",
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        MemorySummary.model_validate(
            {
                "summary_id": "summary-123",
                "session_id": "session-123",
                "conversation_id": "conversation-123",
                "content": "Summary.",
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

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        Message(
            message_id="message-123",
            session_id="session-123",
            conversation_id="conversation-123",
            role=MessageRole.USER,
            content="Hello.",
            created_at=naive_timestamp,
        )

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        MemorySummary(
            summary_id="summary-123",
            session_id="session-123",
            conversation_id="conversation-123",
            content="Summary.",
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

    with pytest.raises(ValidationError, match="updated_at must not be before"):
        MemorySummary(
            summary_id="summary-123",
            session_id="session-123",
            conversation_id="conversation-123",
            content="Summary.",
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
    message = Message(
        message_id="message-123",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        role=MessageRole.ASSISTANT,
        content="The answer is 42.",
        created_at=created_at,
    )
    memory_summary = MemorySummary(
        summary_id="summary-123",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="The user cares about deterministic local memory.",
        revision=2,
        created_at=created_at,
        updated_at=updated_at,
    )

    data = {
        "session": session.model_dump(mode="json"),
        "conversation": conversation.model_dump(mode="json"),
        "message": message.model_dump(mode="json"),
        "memory_summary": memory_summary.model_dump(mode="json"),
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
        "message": {
            "message_id": "message-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "role": "assistant",
            "content": "The answer is 42.",
            "created_at": "2026-05-26T12:30:00Z",
        },
        "memory_summary": {
            "summary_id": "summary-123",
            "session_id": "session-123",
            "conversation_id": "conversation-123",
            "content": "The user cares about deterministic local memory.",
            "revision": 2,
            "created_at": "2026-05-26T12:30:00Z",
            "updated_at": "2026-05-26T12:35:00Z",
        },
    }


def test_message_roles_are_explicit() -> None:
    assert [role.value for role in MessageRole] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
