"""Local persistence substrate primitives."""

from aigentego.persistence.context import (
    DEFAULT_CONVERSATION_CONTEXT_MESSAGE_LIMIT,
    MEMORY_SUMMARY_CONTEXT_PREFIX,
    build_conversation_context_messages,
)
from aigentego.persistence.models import (
    Conversation,
    ConversationStatus,
    MemorySummary,
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
    "DEFAULT_CONVERSATION_CONTEXT_MESSAGE_LIMIT",
    "MEMORY_SUMMARY_CONTEXT_PREFIX",
    "Message",
    "MessageRole",
    "MessageStore",
    "MemorySummary",
    "Session",
    "SessionStatus",
    "build_conversation_context_messages",
    "connect_sqlite",
    "initialize_sqlite_schema",
    "open_sqlite_database",
    "read_schema_version",
]
