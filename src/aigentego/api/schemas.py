"""API request and response schemas."""

from pydantic import BaseModel, Field, field_validator


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
