"""Synchronous SQLite-backed persistence store."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from aigentego.persistence.models import Conversation, Message, Session


class MessageStore:
    """Explicit local store for sessions, conversations, and messages."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def upsert_session(self, session: Session) -> Session:
        """Create or update a session record."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    title,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.title,
                    session.status.value,
                    _format_datetime(session.created_at),
                    _format_datetime(session.updated_at),
                ),
            )
        stored = self.get_session(session.session_id)
        if stored is None:
            raise RuntimeError("failed to upsert session")
        return stored

    def get_session(self, session_id: str) -> Session | None:
        """Return one session by id when present."""
        row = self._connection.execute(
            """
            SELECT session_id, title, status, created_at, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _session_from_row(row)

    def list_sessions(self) -> list[Session]:
        """Return sessions in deterministic id order."""
        rows = self._connection.execute(
            """
            SELECT session_id, title, status, created_at, updated_at
            FROM sessions
            ORDER BY session_id ASC
            """,
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def upsert_conversation(self, conversation: Conversation) -> Conversation:
        """Create or update a conversation record."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    session_id,
                    title,
                    status,
                    is_default,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    title = excluded.title,
                    status = excluded.status,
                    is_default = excluded.is_default,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation.conversation_id,
                    conversation.session_id,
                    conversation.title,
                    conversation.status.value,
                    int(conversation.is_default),
                    _format_datetime(conversation.created_at),
                    _format_datetime(conversation.updated_at),
                ),
            )
        stored = self.get_conversation(conversation.conversation_id)
        if stored is None:
            raise RuntimeError("failed to upsert conversation")
        return stored

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return one conversation by id when present."""
        row = self._connection.execute(
            """
            SELECT
                conversation_id,
                session_id,
                title,
                status,
                is_default,
                created_at,
                updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return _conversation_from_row(row)

    def list_conversations(self, session_id: str | None = None) -> list[Conversation]:
        """Return conversations in deterministic order."""
        if session_id is None:
            rows = self._connection.execute(
                """
                SELECT
                    conversation_id,
                    session_id,
                    title,
                    status,
                    is_default,
                    created_at,
                    updated_at
                FROM conversations
                ORDER BY session_id ASC, conversation_id ASC
                """,
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT
                    conversation_id,
                    session_id,
                    title,
                    status,
                    is_default,
                    created_at,
                    updated_at
                FROM conversations
                WHERE session_id = ?
                ORDER BY conversation_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_conversation_from_row(row) for row in rows]

    def append_message(self, message: Message) -> Message:
        """Append a message to a conversation with deterministic ordering."""
        return self.append_messages([message])[0]

    def append_messages(self, messages: list[Message]) -> list[Message]:
        """Append messages to a conversation in one transaction."""
        if not messages:
            return []

        with self._connection:
            for message in messages:
                sequence_index = self._next_message_sequence(message.conversation_id)
                self._connection.execute(
                    """
                    INSERT INTO messages (
                        message_id,
                        session_id,
                        conversation_id,
                        sequence_index,
                        role,
                        content,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.session_id,
                        message.conversation_id,
                        sequence_index,
                        message.role.value,
                        message.content,
                        _format_datetime(message.created_at),
                    ),
                )

        stored_messages: list[Message] = []
        for message in messages:
            stored = self.get_message(message.message_id)
            if stored is None:
                raise RuntimeError("failed to append message")
            stored_messages.append(stored)
        return stored_messages

    def get_message(self, message_id: str) -> Message | None:
        """Return one message by id when present."""
        row = self._connection.execute(
            """
            SELECT
                message_id,
                session_id,
                conversation_id,
                role,
                content,
                created_at
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return _message_from_row(row)

    def list_messages(self, conversation_id: str) -> list[Message]:
        """Return conversation messages in append order."""
        rows = self._connection.execute(
            """
            SELECT
                message_id,
                session_id,
                conversation_id,
                role,
                content,
                created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence_index ASC
            """,
            (conversation_id,),
        ).fetchall()
        return [_message_from_row(row) for row in rows]

    def _next_message_sequence(self, conversation_id: str) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(sequence_index) + 1, 0) AS sequence_index
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["sequence_index"])


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session.model_validate(_row_to_dict(row))


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation.model_validate(_row_to_dict(row))


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message.model_validate(_row_to_dict(row))


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


__all__ = ["MessageStore"]
