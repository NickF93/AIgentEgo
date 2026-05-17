"""FastAPI dependency construction."""

from typing import Annotated

from fastapi import Depends

from aigentego.llm import LlmProvider, OllamaClient
from aigentego.settings import Settings
from aigentego.tools import CalculatorTool, ToolExecutor, ToolRegistry


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


def get_tool_registry() -> ToolRegistry:
    """Build the deterministic tool registry."""
    return ToolRegistry([CalculatorTool()])


def get_tool_executor(
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> ToolExecutor:
    """Build the deterministic tool executor."""
    return ToolExecutor(registry)
