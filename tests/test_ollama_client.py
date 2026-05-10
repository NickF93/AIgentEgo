import asyncio
import json

import httpx
import pytest

from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    LlmConnectionError,
    LlmResponseError,
    LlmTimeoutError,
    OllamaClient,
)


def test_health_returns_true_when_tags_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []}, request=request)

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(client.health()) is True


def test_health_returns_false_on_non_success_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"}, request=request)

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(client.health()) is False


def test_health_returns_false_on_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(client.health()) is False


def test_list_models_parses_model_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "llama3.2:3b",
                        "model": "llama3.2:3b",
                        "modified_at": "2026-05-10T10:00:00Z",
                        "size": 2_016,
                        "digest": "abc123",
                        "details": {
                            "family": "llama",
                            "parameter_size": "3B",
                        },
                    }
                ]
            },
            request=request,
        )

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert models[0].name == "llama3.2:3b"
    assert models[0].model == "llama3.2:3b"
    assert models[0].size == 2_016
    assert models[0].digest == "abc123"
    assert models[0].details == {"family": "llama", "parameter_size": "3B"}


def test_chat_sends_expected_payload_and_parses_response() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "message": {
                    "role": "assistant",
                    "content": "Hello from Ollama.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
            request=request,
        )

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="llama3.2:3b",
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Say hello."),
        ],
    )

    response = asyncio.run(client.chat(request))

    assert seen_payload == {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "stream": False,
    }
    assert response.model == "llama3.2:3b"
    assert response.message.role == "assistant"
    assert response.message.content == "Hello from Ollama."
    assert response.done is True
    assert response.done_reason == "stop"
    assert response.prompt_eval_count == 8
    assert response.eval_count == 4


def test_chat_raises_project_error_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "message": {"role": "assistant"},
                "done": True,
            },
            request=request,
        )

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="llama3.2:3b",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmResponseError):
        asyncio.run(client.chat(request))


def test_embed_parses_valid_embedding_response() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
                "prompt_eval_count": 3,
            },
            request=request,
        )

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = EmbeddingRequest(
        model="nomic-embed-text",
        inputs=["hello world"],
    )

    response = asyncio.run(client.embed(request))

    assert seen_payload == {
        "model": "nomic-embed-text",
        "input": ["hello world"],
    }
    assert response.model == "nomic-embed-text"
    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert response.prompt_eval_count == 3


def test_timeout_errors_are_mapped_to_project_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = EmbeddingRequest(
        model="nomic-embed-text",
        inputs=["hello world"],
    )

    with pytest.raises(LlmTimeoutError):
        asyncio.run(client.embed(request))


def test_connection_errors_are_mapped_to_project_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    client = OllamaClient(
        "http://ollama:11434",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="llama3.2:3b",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmConnectionError):
        asyncio.run(client.chat(request))
