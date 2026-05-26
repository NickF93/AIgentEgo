import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from aigentego.persistence import (
    Conversation,
    Message,
    MessageRole,
    MessageStore,
    Session,
    open_sqlite_database,
)


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def timestamp() -> datetime:
    return datetime(2026, 5, 26, 12, 30, tzinfo=UTC)


def test_store_upserts_and_fetches_session_with_default_conversation(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    created_at = timestamp()
    session = Session(
        session_id="session-123",
        title="Planning",
        created_at=created_at,
        updated_at=created_at + timedelta(minutes=1),
    )
    conversation = Conversation(
        conversation_id="conversation-123",
        session_id=session.session_id,
        title="Default",
        is_default=True,
        created_at=created_at,
        updated_at=created_at + timedelta(minutes=2),
    )

    assert store.upsert_session(session) == session
    assert store.upsert_conversation(conversation) == conversation

    assert store.get_session(session.session_id) == session
    assert store.get_conversation(conversation.conversation_id) == conversation
    assert store.list_sessions() == [session]
    assert store.list_conversations(session.session_id) == [conversation]


def test_store_updates_session_and_conversation_records(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session = Session(session_id="session-123", title="Planning")
    conversation = Conversation(
        conversation_id="conversation-123",
        session_id=session.session_id,
        title="Default",
    )

    store.upsert_session(session)
    store.upsert_conversation(conversation)

    updated_session = Session(session_id="session-123", title="Updated planning")
    updated_conversation = Conversation(
        conversation_id="conversation-123",
        session_id=session.session_id,
        title="Updated default",
        is_default=True,
    )

    assert store.upsert_session(updated_session) == updated_session
    assert store.upsert_conversation(updated_conversation) == updated_conversation
    assert store.get_session(session.session_id) == updated_session
    assert store.get_conversation(conversation.conversation_id) == updated_conversation


def test_store_appends_messages_in_deterministic_order(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    first = Message(
        message_id="message-b",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        role=MessageRole.USER,
        content="What is 2 + 2?",
        created_at=timestamp(),
    )
    second = Message(
        message_id="message-a",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        role=MessageRole.ASSISTANT,
        content="4",
        created_at=timestamp() + timedelta(seconds=1),
    )

    assert store.append_message(first) == first
    assert store.append_message(second) == second

    messages = store.list_messages(conversation.conversation_id)
    assert [message.message_id for message in messages] == [
        "message-b",
        "message-a",
    ]
    assert messages == [first, second]


def test_store_persists_system_and_tool_message_roles(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    system_message = Message(
        message_id="message-system",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        role=MessageRole.SYSTEM,
        content="Use concise answers.",
    )
    tool_message = Message(
        message_id="message-tool",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        role=MessageRole.TOOL,
        content='{"result": 4}',
    )

    store.append_message(system_message)
    store.append_message(tool_message)

    assert [
        message.role for message in store.list_messages(conversation.conversation_id)
    ] == [MessageRole.SYSTEM, MessageRole.TOOL]


def test_store_rejects_orphan_conversations_and_messages(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_conversation(
            Conversation(
                conversation_id="conversation-orphan",
                session_id="session-missing",
            ),
        )

    session = store.upsert_session(Session(session_id="session-123"))
    with pytest.raises(sqlite3.IntegrityError):
        store.append_message(
            Message(
                message_id="message-orphan",
                session_id=session.session_id,
                conversation_id="conversation-missing",
                role=MessageRole.USER,
                content="Hello.",
            ),
        )

    conversation = store.upsert_conversation(
        Conversation(
            conversation_id="conversation-123",
            session_id=session.session_id,
        ),
    )
    other_session = store.upsert_session(Session(session_id="session-other"))
    with pytest.raises(sqlite3.IntegrityError):
        store.append_message(
            Message(
                message_id="message-wrong-session",
                session_id=other_session.session_id,
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="Hello.",
            ),
        )


def test_store_returns_json_serializable_models(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    message = store.append_message(
        Message(
            message_id="message-123",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            role=MessageRole.USER,
            content="Hello.",
            created_at=timestamp(),
        ),
    )

    data = {
        "session": store.get_session(session.session_id).model_dump(mode="json"),
        "conversation": store.get_conversation(
            conversation.conversation_id,
        ).model_dump(mode="json"),
        "messages": [
            stored_message.model_dump(mode="json")
            for stored_message in store.list_messages(conversation.conversation_id)
        ],
    }

    assert json.loads(json.dumps(data)) == data
    assert data["messages"] == [message.model_dump(mode="json")]


def persist_session_and_conversation(
    store: MessageStore,
) -> tuple[Session, Conversation]:
    session = store.upsert_session(Session(session_id="session-123"))
    conversation = store.upsert_conversation(
        Conversation(
            conversation_id="conversation-123",
            session_id=session.session_id,
            is_default=True,
        ),
    )
    return session, conversation
