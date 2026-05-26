"""API request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aigentego.agents import AgentLoopLimits
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
