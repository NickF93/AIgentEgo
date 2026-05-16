"""FastAPI dependency construction."""

from typing import Annotated

from fastapi import Depends

from aigentego.llm import LlmProvider, OllamaClient
from aigentego.settings import Settings


def get_settings() -> Settings:
    """Load runtime settings."""
    return Settings()


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LlmProvider:
    """Build the configured LLM provider."""
    return OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
