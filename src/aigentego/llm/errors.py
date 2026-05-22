"""LLM provider error types."""

from enum import StrEnum


class LlmProviderError(Exception):
    """Base error for LLM provider failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        operation: str,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        context = f"{self.provider} {self.operation}"
        if self.status_code is not None:
            context = f"{context} returned status {self.status_code}"
        return f"{context}: {message}"


class LlmConnectionError(LlmProviderError):
    """The provider could not be reached."""


class LlmResponseError(LlmProviderError):
    """The provider returned an unsuccessful or invalid response."""


class LlmTimeoutError(LlmProviderError):
    """The provider request timed out."""


class LlmConfigurationError(LlmProviderError):
    """The configured LLM provider cannot be constructed."""


class UnsupportedLlmBackendError(LlmConfigurationError):
    """The configured LLM backend is not implemented."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(
            f"unsupported LLM backend {backend!r}",
            provider="llm",
            operation="configure_provider",
        )


class ToolCallFailureCode(StrEnum):
    """Deterministic structured tool-call parse and validation failure codes."""

    INVALID_JSON = "invalid_json"
    NON_OBJECT_JSON = "non_object_json"
    SCHEMA_INVALID_JSON = "schema_invalid_json"
    UNKNOWN_TOOL = "unknown_tool"


class ToolCallParseError(LlmProviderError):
    """Structured tool-call output could not be parsed."""

    def __init__(
        self,
        message: str = "invalid structured tool-call JSON",
        *,
        failure_code: ToolCallFailureCode = ToolCallFailureCode.INVALID_JSON,
    ) -> None:
        self.failure_code = failure_code
        super().__init__(
            message,
            provider="llm",
            operation="parse_tool_calls",
        )


class ToolCallValidationError(LlmProviderError):
    """Structured tool-call output failed provider-neutral validation."""

    def __init__(
        self,
        message: str = "invalid structured tool-call output",
        *,
        failure_code: ToolCallFailureCode = ToolCallFailureCode.SCHEMA_INVALID_JSON,
        tool_name: str | None = None,
    ) -> None:
        self.failure_code = failure_code
        self.tool_name = tool_name
        super().__init__(
            message,
            provider="llm",
            operation="parse_tool_calls",
        )
