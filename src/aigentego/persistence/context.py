"""Conversation context construction for explicit local persistence."""

from collections.abc import Sequence
from typing import Final, Literal, cast

from aigentego.llm import ChatMessage
from aigentego.persistence.models import MemorySummary, Message, MessageRole

DEFAULT_CONVERSATION_CONTEXT_MESSAGE_LIMIT: Final[int] = 20
MEMORY_SUMMARY_CONTEXT_PREFIX: Final[str] = "Conversation memory summary:"

ChatMessageRole = Literal["system", "user", "assistant"]


def build_conversation_context_messages(
    *,
    current_user_message: str,
    prior_messages: Sequence[Message] = (),
    memory_summary: MemorySummary | None = None,
    message_limit: int = DEFAULT_CONVERSATION_CONTEXT_MESSAGE_LIMIT,
) -> list[ChatMessage]:
    """Build provider-neutral chat context for one explicit conversation turn."""
    if message_limit < 0:
        raise ValueError("message_limit must be greater than or equal to 0")

    stripped_user_message = current_user_message.strip()
    if not stripped_user_message:
        raise ValueError("current_user_message must not be blank")

    messages: list[ChatMessage] = []
    if memory_summary is not None:
        messages.append(
            ChatMessage(
                role="system",
                content=f"{MEMORY_SUMMARY_CONTEXT_PREFIX}\n{memory_summary.content}",
            ),
        )

    context_messages = [
        chat_message
        for message in prior_messages
        if (chat_message := _message_to_chat_message(message)) is not None
    ]
    if message_limit == 0:
        context_messages = []
    else:
        context_messages = context_messages[-message_limit:]

    messages.extend(context_messages)
    messages.append(ChatMessage(role="user", content=stripped_user_message))
    return messages


def _message_to_chat_message(message: Message) -> ChatMessage | None:
    if message.role is MessageRole.TOOL:
        return None
    return ChatMessage(
        role=cast(ChatMessageRole, message.role.value),
        content=message.content,
    )


__all__ = [
    "DEFAULT_CONVERSATION_CONTEXT_MESSAGE_LIMIT",
    "MEMORY_SUMMARY_CONTEXT_PREFIX",
    "build_conversation_context_messages",
]
