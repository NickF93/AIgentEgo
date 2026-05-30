import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from aigentego.persistence import (
    Conversation,
    MemorySummary,
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


def test_store_upserts_and_fetches_memory_summary(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    summary = MemorySummary(
        summary_id="summary-123",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="The user is designing local conversation memory.",
        revision=1,
        created_at=timestamp(),
        updated_at=timestamp() + timedelta(minutes=5),
    )

    assert store.upsert_memory_summary(summary) == summary
    assert store.get_memory_summary(summary.summary_id) == summary
    assert store.get_latest_memory_summary(conversation.conversation_id) == summary
    assert store.list_memory_summaries(conversation.conversation_id) == [summary]


def test_store_replaces_memory_summary(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    original = MemorySummary(
        summary_id="summary-123",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="Original summary.",
        revision=1,
    )
    updated = MemorySummary(
        summary_id="summary-123",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="Updated summary.",
        revision=2,
    )

    store.upsert_memory_summary(original)

    assert store.upsert_memory_summary(updated) == updated
    assert store.get_memory_summary(original.summary_id) == updated
    assert store.list_memory_summaries(conversation.conversation_id) == [updated]


def test_store_fetches_latest_memory_summary_by_revision(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    first = MemorySummary(
        summary_id="summary-first",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="First summary.",
        revision=1,
    )
    second = MemorySummary(
        summary_id="summary-second",
        session_id=session.session_id,
        conversation_id=conversation.conversation_id,
        content="Second summary.",
        revision=2,
    )

    store.upsert_memory_summary(second)
    store.upsert_memory_summary(first)

    assert store.list_memory_summaries(conversation.conversation_id) == [first, second]
    assert store.get_latest_memory_summary(conversation.conversation_id) == second


def test_store_lists_memory_summaries_deterministically_across_conversations(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session = store.upsert_session(Session(session_id="session-123"))
    first_conversation = store.upsert_conversation(
        Conversation(
            conversation_id="conversation-a",
            session_id=session.session_id,
        ),
    )
    second_conversation = store.upsert_conversation(
        Conversation(
            conversation_id="conversation-b",
            session_id=session.session_id,
        ),
    )
    summaries = [
        MemorySummary(
            summary_id="summary-b2",
            session_id=session.session_id,
            conversation_id=second_conversation.conversation_id,
            content="Second conversation revision 2.",
            revision=2,
        ),
        MemorySummary(
            summary_id="summary-a1",
            session_id=session.session_id,
            conversation_id=first_conversation.conversation_id,
            content="First conversation revision 1.",
            revision=1,
        ),
        MemorySummary(
            summary_id="summary-b1",
            session_id=session.session_id,
            conversation_id=second_conversation.conversation_id,
            content="Second conversation revision 1.",
            revision=1,
        ),
    ]

    for summary in summaries:
        store.upsert_memory_summary(summary)

    assert [
        summary.summary_id for summary in store.list_memory_summaries()
    ] == [
        "summary-a1",
        "summary-b1",
        "summary-b2",
    ]


def test_store_rejects_orphan_memory_summaries(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-orphan",
                session_id="session-missing",
                conversation_id="conversation-missing",
                content="Orphan summary.",
            ),
        )

    session = store.upsert_session(Session(session_id="session-123"))
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-missing-conversation",
                session_id=session.session_id,
                conversation_id="conversation-missing",
                content="Missing conversation.",
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
        store.upsert_memory_summary(
            MemorySummary(
                summary_id="summary-wrong-session",
                session_id=other_session.session_id,
                conversation_id=conversation.conversation_id,
                content="Wrong session relationship.",
            ),
        )


def test_store_returns_json_serializable_memory_summary(
    sqlite_connection: sqlite3.Connection,
) -> None:
    store = MessageStore(sqlite_connection)
    session, conversation = persist_session_and_conversation(store)
    summary = store.upsert_memory_summary(
        MemorySummary(
            summary_id="summary-123",
            session_id=session.session_id,
            conversation_id=conversation.conversation_id,
            content="The user prefers explicit local memory.",
            revision=3,
            created_at=timestamp(),
            updated_at=timestamp() + timedelta(minutes=5),
        ),
    )

    data = {
        "summary": store.get_memory_summary(summary.summary_id).model_dump(
            mode="json",
        ),
        "latest": store.get_latest_memory_summary(
            conversation.conversation_id,
        ).model_dump(mode="json"),
    }

    assert json.loads(json.dumps(data)) == data
    assert data["summary"] == summary.model_dump(mode="json")
    assert data["latest"] == summary.model_dump(mode="json")


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
