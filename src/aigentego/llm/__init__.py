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
    ToolCallFailureCode,
    ToolCallParseError,
    ToolCallValidationError,
    UnsupportedLlmBackendError,
)
from aigentego.llm.factory import build_llm_provider
from aigentego.llm.ollama_client import OllamaClient
from aigentego.llm.tool_call_parser import parse_structured_tool_calls
from aigentego.llm.tool_call_repair import (
    ToolCallFailureCategory,
    ToolCallParsingFailure,
    ToolCallRepairAction,
    ToolCallRepairDecision,
    ToolCallRepairDecisionReason,
    ToolCallRetryPolicy,
    classify_tool_call_failure,
    decide_tool_call_repair,
)
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
    "ToolCallFailureCategory",
    "ToolCallFailureCode",
    "ToolCallParsingFailure",
    "ToolCallParseError",
    "ToolCallRepairAction",
    "ToolCallRepairDecision",
    "ToolCallRepairDecisionReason",
    "ToolCallRetryPolicy",
    "ToolCallValidationError",
    "UnsupportedLlmBackendError",
    "build_llm_provider",
    "classify_tool_call_failure",
    "decide_tool_call_repair",
    "parse_structured_tool_calls",
]
