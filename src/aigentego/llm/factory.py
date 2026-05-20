"""LLM provider construction."""

from aigentego.llm.base import LlmProvider
from aigentego.llm.errors import UnsupportedLlmBackendError
from aigentego.llm.ollama_client import OllamaClient
from aigentego.settings import Settings


def build_llm_provider(settings: Settings) -> LlmProvider:
    """Build the configured provider-neutral LLM adapter."""
    backend = settings.llm_backend.strip().lower()
    if backend == "ollama":
        return OllamaClient(
            base_url=settings.llm_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )

    raise UnsupportedLlmBackendError(settings.llm_backend)
