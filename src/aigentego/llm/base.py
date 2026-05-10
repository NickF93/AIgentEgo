"""Provider-neutral LLM request and response models."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single chat message."""

    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """A provider-neutral chat request."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    """A provider-neutral chat response."""

    model: str
    message: ChatMessage
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class EmbeddingRequest(BaseModel):
    """A provider-neutral embedding request."""

    model: str
    inputs: list[str] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    """A provider-neutral embedding response."""

    model: str
    embeddings: list[list[float]]
    prompt_eval_count: int | None = None


class ModelInfo(BaseModel):
    """Minimal metadata for a locally available model."""

    name: str
    model: str | None = None
    modified_at: str | None = None
    size: int | None = None
    digest: str | None = None
    details: dict[str, Any] | None = None


class LlmProvider(Protocol):
    """Minimal asynchronous LLM provider interface."""

    async def health(self) -> bool:
        """Return whether the provider is reachable."""

    async def list_models(self) -> list[ModelInfo]:
        """Return locally available models."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a non-streaming chat response."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate text embeddings."""
