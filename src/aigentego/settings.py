"""Environment-driven runtime settings for AIgentEgo."""

from typing import Any

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    agent_host: str = "0.0.0.0"
    agent_port: int = 8080
    llm_backend: str = "ollama"
    llm_base_url: str = Field(
        default="http://ollama:11434",
        validation_alias=AliasChoices("LLM_BASE_URL", "OLLAMA_BASE_URL"),
    )
    chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias=AliasChoices("CHAT_MODEL", "OLLAMA_CHAT_MODEL"),
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OLLAMA_EMBED_MODEL"),
    )
    sqlite_path: str = Field(
        default=".aigentego/aigentego.sqlite3",
        validation_alias=AliasChoices("SQLITE_PATH", "AIGENTEGO_SQLITE_PATH"),
    )
    request_timeout_seconds: int = 120
    log_level: str = "INFO"

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_ollama_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        legacy_fields = {
            "ollama_base_url": "llm_base_url",
            "ollama_chat_model": "chat_model",
            "ollama_embed_model": "embedding_model",
        }
        for legacy_name, canonical_name in legacy_fields.items():
            if canonical_name not in values and legacy_name in values:
                values[canonical_name] = values[legacy_name]

        return values

    @property
    def ollama_base_url(self) -> str:
        """Compatibility accessor for the current Ollama provider."""
        return self.llm_base_url

    @property
    def ollama_chat_model(self) -> str:
        """Compatibility accessor for the current Ollama chat model setting."""
        return self.chat_model

    @property
    def ollama_embed_model(self) -> str:
        """Compatibility accessor for the current Ollama embedding model setting."""
        return self.embedding_model

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
