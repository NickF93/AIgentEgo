"""API request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aigentego.agents import AgentLoopLimits
from aigentego.persistence import Conversation, Session
from aigentego.tools import ToolDefinition, ToolErrorDetail


class HealthResponse(BaseModel):
    """Health status for the runtime API and configured LLM provider."""

    status: str
    provider_reachable: bool
    chat_model: str
    ollama_reachable: bool


class DiagnosticsResponse(BaseModel):
    """Non-secret runtime diagnostics."""

    status: str
    package_version: str
    llm_backend: str
    llm_provider: str
    provider_base_url: str
    chat_model: str
    embedding_model: str
    provider_reachable: bool
    ollama_base_url: str
    ollama_reachable: bool


class ChatApiRequest(BaseModel):
    """Public non-streaming chat request."""

    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        """Reject blank messages after trimming surrounding whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatApiResponse(BaseModel):
    """Public non-streaming chat response."""

    request_id: str
    model: str
    message: str


class PersistentChatApiResponse(BaseModel):
    """Public response for an explicitly persisted chat turn."""

    request_id: str
    session_id: str
    conversation_id: str
    model: str
    message: str


class AgentRunApiRequest(BaseModel):
    """Public bounded agent run request."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    max_steps: int | None = Field(default=None, ge=0)
    max_tool_errors: int | None = Field(default=None, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        """Reject blank messages after trimming surrounding whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped

    def to_limits(self) -> AgentLoopLimits:
        """Build bounded loop limits from supplied request overrides."""
        default_limits = AgentLoopLimits()
        return AgentLoopLimits(
            max_steps=(
                self.max_steps
                if self.max_steps is not None
                else default_limits.max_steps
            ),
            max_tool_errors=(
                self.max_tool_errors
                if self.max_tool_errors is not None
                else default_limits.max_tool_errors
            ),
            timeout_seconds=(
                self.timeout_seconds
                if self.timeout_seconds is not None
                else default_limits.timeout_seconds
            ),
        )


class SessionCreateApiRequest(BaseModel):
    """Public request for explicitly creating or updating a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    title: str | None = None

    @field_validator("session_id", "title")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional titles after trimming."""
        return _strip_optional_text(value)


class SessionApiResponse(BaseModel):
    """Public response containing one persisted session."""

    request_id: str
    session: Session


class SessionListApiResponse(BaseModel):
    """Public response containing persisted sessions."""

    request_id: str
    sessions: list[Session]


class ConversationCreateApiRequest(BaseModel):
    """Public request for explicitly creating or updating a conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    title: str | None = None
    is_default: bool = False

    @field_validator("conversation_id", "title")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional titles after trimming."""
        return _strip_optional_text(value)


class ConversationApiResponse(BaseModel):
    """Public response containing one persisted conversation."""

    request_id: str
    conversation: Conversation


class ConversationListApiResponse(BaseModel):
    """Public response containing persisted conversations."""

    request_id: str
    conversations: list[Conversation]


class ApiErrorResponse(BaseModel):
    """Structured API error body."""

    error: str
    message: str


class ToolListApiResponse(BaseModel):
    """Public deterministic tool listing response."""

    tools: list[ToolDefinition]


class ToolExecuteApiRequest(BaseModel):
    """Public manual deterministic tool execution request."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecuteApiResponse(BaseModel):
    """Public manual deterministic tool execution response."""

    request_id: str
    tool_name: str
    success: bool
    result: dict[str, Any] | None = None
    error: ToolErrorDetail | None = None


def _strip_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("text fields must not be blank")
    return stripped
