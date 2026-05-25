import pytest

from aigentego.api.dependencies import get_llm_provider
from aigentego.llm import (
    LlamaCppProvider,
    OllamaClient,
    UnsupportedLlmBackendError,
    build_llm_provider,
)
from aigentego.settings import Settings


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_factory_builds_ollama_provider_by_default() -> None:
    provider = build_llm_provider(make_settings())

    assert isinstance(provider, OllamaClient)
    assert provider.provider_name == "ollama"


def test_factory_uses_provider_neutral_base_url_and_timeout() -> None:
    settings = make_settings(
        llm_base_url="http://localhost:11434/",
        request_timeout_seconds=45,
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, OllamaClient)
    assert provider._base_url == "http://localhost:11434"
    assert provider._timeout.connect == 45
    assert provider._timeout.read == 45


def test_factory_accepts_normalized_ollama_backend_name() -> None:
    provider = build_llm_provider(make_settings(llm_backend="  OLLAMA  "))

    assert isinstance(provider, OllamaClient)


def test_factory_builds_llamacpp_provider_for_backend() -> None:
    settings = make_settings(
        llm_backend="llamacpp",
        llm_base_url="http://localhost:8081/",
        request_timeout_seconds=15,
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, LlamaCppProvider)
    assert provider.provider_name == "llamacpp"
    assert provider._base_url == "http://localhost:8081"
    assert provider._timeout.connect == 15
    assert provider._timeout.read == 15


def test_factory_accepts_normalized_llamacpp_backend_name() -> None:
    provider = build_llm_provider(make_settings(llm_backend="  LLAMACPP  "))

    assert isinstance(provider, LlamaCppProvider)


def test_factory_rejects_unsupported_backend() -> None:
    settings = make_settings(llm_backend="llama.cpp")

    with pytest.raises(UnsupportedLlmBackendError) as exc_info:
        build_llm_provider(settings)

    assert exc_info.value.backend == "llama.cpp"
    assert "unsupported LLM backend" in str(exc_info.value)


def test_fastapi_dependency_uses_provider_factory() -> None:
    provider = get_llm_provider(
        make_settings(
            llm_base_url="http://dependency:11434",
            request_timeout_seconds=30,
        )
    )

    assert isinstance(provider, OllamaClient)
    assert provider._base_url == "http://dependency:11434"
    assert provider._timeout.connect == 30


def test_fastapi_dependency_uses_factory_for_llamacpp() -> None:
    provider = get_llm_provider(
        make_settings(
            llm_backend="llamacpp",
            llm_base_url="http://dependency:8081",
            request_timeout_seconds=20,
        )
    )

    assert isinstance(provider, LlamaCppProvider)
    assert provider._base_url == "http://dependency:8081"
    assert provider._timeout.connect == 20


def test_legacy_ollama_settings_still_feed_factory() -> None:
    provider = build_llm_provider(
        make_settings(ollama_base_url="http://legacy-ollama:11434")
    )

    assert isinstance(provider, OllamaClient)
    assert provider._base_url == "http://legacy-ollama:11434"
