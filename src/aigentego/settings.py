"""Environment-driven runtime settings for AIgentEgo."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    agent_host: str = "0.0.0.0"
    agent_port: int = 8080
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    request_timeout_seconds: int = 120
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
