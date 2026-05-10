"""Provider-neutral LLM interfaces and clients."""

from aigentego.llm.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmProvider,
    ModelInfo,
)
from aigentego.llm.errors import (
    LlmConnectionError,
    LlmProviderError,
    LlmResponseError,
    LlmTimeoutError,
)
from aigentego.llm.ollama_client import OllamaClient

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "LlmConnectionError",
    "LlmProvider",
    "LlmProviderError",
    "LlmResponseError",
    "LlmTimeoutError",
    "ModelInfo",
    "OllamaClient",
]
