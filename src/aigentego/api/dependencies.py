"""FastAPI dependency construction."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from aigentego.llm import LlmProvider, build_llm_provider
from aigentego.persistence import MessageStore, open_sqlite_database
from aigentego.retrieval import NoteEmbeddingStore
from aigentego.settings import Settings
from aigentego.tools import CalculatorTool, ToolExecutor, ToolRegistry


def get_settings() -> Settings:
    """Load runtime settings."""
    return Settings()


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LlmProvider:
    """Build the configured LLM provider."""
    return build_llm_provider(settings)


def get_tool_registry() -> ToolRegistry:
    """Build the deterministic tool registry."""
    return ToolRegistry([CalculatorTool()])


def get_tool_executor(
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> ToolExecutor:
    """Build the deterministic tool executor."""
    return ToolExecutor(registry)


async def get_message_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[MessageStore]:
    """Build a request-scoped local persistence store."""
    connection = open_sqlite_database(settings.sqlite_path)
    try:
        yield MessageStore(connection)
    finally:
        connection.close()


async def get_note_embedding_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[NoteEmbeddingStore]:
    """Build a request-scoped local note embedding store."""
    connection = open_sqlite_database(settings.sqlite_path)
    try:
        yield NoteEmbeddingStore(connection)
    finally:
        connection.close()
