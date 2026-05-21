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
    ToolCallParseError,
    ToolCallValidationError,
    UnsupportedLlmBackendError,
)
from aigentego.llm.factory import build_llm_provider
from aigentego.llm.ollama_client import OllamaClient
from aigentego.llm.tool_call_parser import parse_structured_tool_calls
from aigentego.llm.tool_calls import LlmToolCall, StructuredToolCallOutput

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
    "LlmToolCall",
    "ModelInfo",
    "OllamaClient",
    "StructuredToolCallOutput",
    "ToolCallParseError",
    "ToolCallValidationError",
    "UnsupportedLlmBackendError",
    "build_llm_provider",
    "parse_structured_tool_calls",
]
