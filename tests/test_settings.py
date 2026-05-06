from pathlib import Path

from aigentego.settings import Settings

SETTINGS_ENV_VARS = (
    "AGENT_HOST",
    "AGENT_PORT",
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    "REQUEST_TIMEOUT_SECONDS",
    "LOG_LEVEL",
)


def clear_settings_env(monkeypatch) -> None:
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_settings_can_be_instantiated(monkeypatch) -> None:
    clear_settings_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.agent_host == "0.0.0.0"
    assert settings.agent_port == 8080
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.request_timeout_seconds == 120
    assert settings.log_level == "INFO"


def test_default_model_names_are_set(monkeypatch) -> None:
    clear_settings_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.ollama_chat_model == "llama3.2:3b"
    assert settings.ollama_embed_model == "nomic-embed-text"


def test_environment_variables_override_defaults(monkeypatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("AGENT_HOST", "127.0.0.1")
    monkeypatch.setenv("AGENT_PORT", "9090")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.agent_host == "127.0.0.1"
    assert settings.agent_port == 9090
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_chat_model == "qwen2.5:7b"
    assert settings.ollama_embed_model == "mxbai-embed-large"
    assert settings.request_timeout_seconds == 30
    assert settings.log_level == "DEBUG"


def test_env_file_is_not_required(tmp_path: Path, monkeypatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.agent_port == 8080


def test_env_file_is_loaded_when_present(tmp_path: Path, monkeypatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "AGENT_PORT=7070\nREQUEST_TIMEOUT_SECONDS=45\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.agent_port == 7070
    assert settings.request_timeout_seconds == 45


def test_integer_fields_are_parsed_from_environment(monkeypatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("AGENT_PORT", "8181")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "15")

    settings = Settings(_env_file=None)

    assert settings.agent_port == 8181
    assert settings.request_timeout_seconds == 15
    assert isinstance(settings.agent_port, int)
    assert isinstance(settings.request_timeout_seconds, int)
