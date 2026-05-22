import pytest

from aigentego.llm.errors import UnsupportedLlmBackendError
from aigentego.llm.factory import (
    LLAMACPP_BACKEND,
    OLLAMA_BACKEND,
    build_llm_provider,
    normalize_llm_backend,
)
from aigentego.settings import Settings

LLAMACPP_PROVIDER_ENDPOINTS = (
    "/health",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/embeddings",
)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_ollama_remains_default_llm_backend() -> None:
    settings = make_settings()

    assert settings.llm_backend == OLLAMA_BACKEND


def test_llamacpp_backend_identifier_is_canonical() -> None:
    assert LLAMACPP_BACKEND == "llamacpp"


@pytest.mark.parametrize(
    ("raw_backend", "normalized_backend"),
    (
        ("ollama", OLLAMA_BACKEND),
        ("  OLLAMA  ", OLLAMA_BACKEND),
        ("llamacpp", LLAMACPP_BACKEND),
        ("  LLAMACPP  ", LLAMACPP_BACKEND),
    ),
)
def test_backend_normalization_is_case_and_whitespace_only(
    raw_backend: str,
    normalized_backend: str,
) -> None:
    assert normalize_llm_backend(raw_backend) == normalized_backend


@pytest.mark.parametrize("backend", ("llama.cpp", "llama_cpp"))
def test_llamacpp_alias_spellings_are_not_supported(backend: str) -> None:
    assert normalize_llm_backend(backend) != LLAMACPP_BACKEND


def test_llamacpp_settings_value_is_accepted_before_provider_exists() -> None:
    settings = make_settings(
        llm_backend=LLAMACPP_BACKEND,
        llm_base_url="http://localhost:8081",
        chat_model="local-chat",
        embedding_model="local-embed",
    )

    assert settings.llm_backend == LLAMACPP_BACKEND
    assert settings.llm_base_url == "http://localhost:8081"
    assert settings.chat_model == "local-chat"
    assert settings.embedding_model == "local-embed"


def test_llamacpp_provider_is_not_constructed_until_factory_integration() -> None:
    settings = make_settings(llm_backend=LLAMACPP_BACKEND)

    with pytest.raises(UnsupportedLlmBackendError) as exc_info:
        build_llm_provider(settings)

    assert exc_info.value.backend == LLAMACPP_BACKEND


def test_llamacpp_contract_targets_llama_server_openai_compatible_endpoints() -> None:
    assert LLAMACPP_PROVIDER_ENDPOINTS == (
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/embeddings",
    )
