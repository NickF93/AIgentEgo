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
    LlmConfigurationError,
    LlmConnectionError,
    LlmProviderError,
    LlmResponseError,
    LlmTimeoutError,
    UnsupportedLlmBackendError,
)
from aigentego.llm.factory import build_llm_provider
from aigentego.llm.ollama_client import OllamaClient

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "LlmConfigurationError",
    "LlmConnectionError",
    "LlmProvider",
    "LlmProviderError",
    "LlmResponseError",
    "LlmTimeoutError",
    "ModelInfo",
    "OllamaClient",
    "UnsupportedLlmBackendError",
    "build_llm_provider",
]
