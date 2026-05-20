"""API request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aigentego.tools import ToolDefinition, ToolErrorDetail


class HealthResponse(BaseModel):
    """Health status for the runtime API and configured LLM provider."""

    status: str
    ollama_reachable: bool
    chat_model: str


class DiagnosticsResponse(BaseModel):
    """Non-secret runtime diagnostics."""

    status: str
    package_version: str
    llm_provider: str
    ollama_base_url: str
    chat_model: str
    embedding_model: str
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
