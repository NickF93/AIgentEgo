"""LLM provider error types."""


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
