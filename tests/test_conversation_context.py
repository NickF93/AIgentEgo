import pytest

from aigentego.llm import ChatMessage
from aigentego.persistence import (
    MEMORY_SUMMARY_CONTEXT_PREFIX,
    MemorySummary,
    Message,
    MessageRole,
    build_conversation_context_messages,
)


def test_context_builder_returns_current_user_message_without_history() -> None:
    assert build_conversation_context_messages(
        current_user_message=" Continue. ",
    ) == [
        ChatMessage(role="user", content="Continue."),
    ]


def test_context_builder_preserves_prior_message_order() -> None:
    messages = build_conversation_context_messages(
        current_user_message="What next?",
        prior_messages=[
            message(0, MessageRole.SYSTEM, "Use short answers."),
            message(1, MessageRole.USER, "What is stored?"),
            message(2, MessageRole.ASSISTANT, "Stored context."),
        ],
    )

    assert messages == [
        ChatMessage(role="system", content="Use short answers."),
        ChatMessage(role="user", content="What is stored?"),
        ChatMessage(role="assistant", content="Stored context."),
        ChatMessage(role="user", content="What next?"),
    ]


def test_context_builder_omits_tool_messages() -> None:
    messages = build_conversation_context_messages(
        current_user_message="Continue.",
        prior_messages=[
            message(0, MessageRole.USER, "Use the calculator."),
            message(1, MessageRole.TOOL, '{"value": 42}'),
            message(2, MessageRole.ASSISTANT, "The answer is 42."),
        ],
    )

    assert messages == [
        ChatMessage(role="user", content="Use the calculator."),
        ChatMessage(role="assistant", content="The answer is 42."),
        ChatMessage(role="user", content="Continue."),
    ]


def test_context_builder_includes_memory_summary_first() -> None:
    messages = build_conversation_context_messages(
        current_user_message="Continue.",
        prior_messages=[message(0, MessageRole.USER, "Earlier turn.")],
        memory_summary=MemorySummary(
            summary_id="summary-123",
            session_id="session-123",
            conversation_id="conversation-123",
            content="The user prefers concise local-memory behavior.",
            revision=2,
        ),
    )

    assert messages == [
        ChatMessage(
            role="system",
            content=(
                f"{MEMORY_SUMMARY_CONTEXT_PREFIX}\n"
                "The user prefers concise local-memory behavior."
            ),
        ),
        ChatMessage(role="user", content="Earlier turn."),
        ChatMessage(role="user", content="Continue."),
    ]


def test_context_builder_limits_prior_messages_deterministically() -> None:
    messages = build_conversation_context_messages(
        current_user_message="Current turn.",
        prior_messages=[
            message(index, MessageRole.USER, f"Prior {index}")
            for index in range(5)
        ],
        message_limit=2,
    )

    assert messages == [
        ChatMessage(role="user", content="Prior 3"),
        ChatMessage(role="user", content="Prior 4"),
        ChatMessage(role="user", content="Current turn."),
    ]


def test_context_builder_can_omit_prior_messages_with_zero_limit() -> None:
    messages = build_conversation_context_messages(
        current_user_message="Current turn.",
        prior_messages=[message(0, MessageRole.USER, "Prior turn.")],
        memory_summary=MemorySummary(
            summary_id="summary-123",
            session_id="session-123",
            conversation_id="conversation-123",
            content="Local summary.",
        ),
        message_limit=0,
    )

    assert messages == [
        ChatMessage(
            role="system",
            content=f"{MEMORY_SUMMARY_CONTEXT_PREFIX}\nLocal summary.",
        ),
        ChatMessage(role="user", content="Current turn."),
    ]


@pytest.mark.parametrize("current_user_message", ["", "   "])
def test_context_builder_rejects_blank_current_user_message(
    current_user_message: str,
) -> None:
    with pytest.raises(ValueError, match="current_user_message must not be blank"):
        build_conversation_context_messages(
            current_user_message=current_user_message,
        )


def test_context_builder_rejects_negative_message_limit() -> None:
    with pytest.raises(ValueError, match="message_limit"):
        build_conversation_context_messages(
            current_user_message="Current turn.",
            message_limit=-1,
        )


def message(index: int, role: MessageRole, content: str) -> Message:
    return Message(
        message_id=f"message-{index}",
        session_id="session-123",
        conversation_id="conversation-123",
        role=role,
        content=content,
    )
