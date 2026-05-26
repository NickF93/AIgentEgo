"""Local persistence substrate primitives."""

from aigentego.persistence.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    Session,
    SessionStatus,
)
from aigentego.persistence.sqlite import (
    SCHEMA_VERSION,
    connect_sqlite,
    initialize_sqlite_schema,
    open_sqlite_database,
    read_schema_version,
)
from aigentego.persistence.store import MessageStore

__all__ = [
    "SCHEMA_VERSION",
    "Conversation",
    "ConversationStatus",
    "Message",
    "MessageRole",
    "MessageStore",
    "Session",
    "SessionStatus",
    "connect_sqlite",
    "initialize_sqlite_schema",
    "open_sqlite_database",
    "read_schema_version",
]
