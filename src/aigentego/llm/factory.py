"""LLM provider construction."""

from aigentego.llm.base import LlmProvider
from aigentego.llm.errors import UnsupportedLlmBackendError
from aigentego.llm.ollama_client import OllamaClient
from aigentego.settings import Settings

OLLAMA_BACKEND = "ollama"
LLAMACPP_BACKEND = "llamacpp"


def normalize_llm_backend(value: str) -> str:
    """Return the deterministic backend identifier used by provider selection."""
    return value.strip().lower()


def build_llm_provider(settings: Settings) -> LlmProvider:
    """Build the configured provider-neutral LLM adapter."""
    backend = normalize_llm_backend(settings.llm_backend)
    if backend == OLLAMA_BACKEND:
        return OllamaClient(
            base_url=settings.llm_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )

    # llama.cpp support targets llama-server's OpenAI-compatible endpoints.
    raise UnsupportedLlmBackendError(settings.llm_backend)
