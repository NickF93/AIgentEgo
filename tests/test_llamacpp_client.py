import asyncio
import json

import httpx
import pytest

from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    LlamaCppProvider,
    LlmConnectionError,
    LlmResponseError,
    LlmTimeoutError,
)


def test_provider_name_and_base_url_normalization() -> None:
    provider = LlamaCppProvider(
        "http://localhost:8080/",
        timeout_seconds=10,
    )

    assert provider.provider_name == "llamacpp"
    assert provider._base_url == "http://localhost:8080"


def test_health_returns_true_when_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"}, request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(provider.health()) is True


def test_health_returns_false_when_model_is_loading() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": 503,
                    "message": "Loading model",
                    "type": "unavailable_error",
                }
            },
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(provider.health()) is False


def test_health_returns_false_on_malformed_success_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "loading"}, request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(provider.health()) is False


def test_health_returns_false_on_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(provider.health()) is False


def test_list_models_parses_openai_compatible_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "local-chat",
                        "object": "model",
                        "created": 1_777_777_777,
                        "owned_by": "llama.cpp",
                    }
                ],
            },
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    models = asyncio.run(provider.list_models())

    assert len(models) == 1
    assert models[0].name == "local-chat"
    assert models[0].model == "local-chat"
    assert models[0].details == {
        "object": "model",
        "created": 1_777_777_777,
        "owned_by": "llama.cpp",
    }


def test_list_models_rejects_malformed_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"object": "model"}]},
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LlmResponseError):
        asyncio.run(provider.list_models())


def test_chat_sends_non_streaming_payload_and_parses_response() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "local-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from llama.cpp.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
            },
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Say hello."),
        ],
    )

    response = asyncio.run(provider.chat(request))

    assert seen_payload == {
        "model": "local-chat",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "stream": False,
    }
    assert response.model == "local-chat"
    assert response.message.role == "assistant"
    assert response.message.content == "Hello from llama.cpp."
    assert response.done is True
    assert response.done_reason == "stop"
    assert response.prompt_eval_count == 7
    assert response.eval_count == 5


@pytest.mark.parametrize(
    "response_json",
    (
        {"model": "local-chat", "choices": []},
        {"model": "local-chat", "choices": [{"finish_reason": "stop"}]},
        {
            "model": "local-chat",
            "choices": [{"message": {"role": "assistant"}}],
        },
    ),
)
def test_chat_rejects_malformed_responses(
    response_json: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json, request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmResponseError):
        asyncio.run(provider.chat(request))


def test_chat_maps_timeout_to_project_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmTimeoutError):
        asyncio.run(provider.chat(request))


def test_chat_maps_http_status_error_to_project_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "provider failure"}},
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmResponseError) as exc_info:
        asyncio.run(provider.chat(request))

    assert exc_info.value.status_code == 500
    assert "provider failure" not in str(exc_info.value)


def test_chat_maps_connection_failure_to_project_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmConnectionError):
        asyncio.run(provider.chat(request))


def test_invalid_json_response_maps_to_project_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(
        model="local-chat",
        messages=[ChatMessage(role="user", content="Say hello.")],
    )

    with pytest.raises(LlmResponseError):
        asyncio.run(provider.chat(request))


def test_embed_sends_payload_and_parses_response() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        seen_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "local-embed",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                    {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = EmbeddingRequest(
        model="local-embed",
        inputs=["first", "second"],
    )

    response = asyncio.run(provider.embed(request))

    assert seen_payload == {
        "model": "local-embed",
        "input": ["first", "second"],
        "encoding_format": "float",
    }
    assert response.model == "local-embed"
    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert response.prompt_eval_count == 4


def test_embed_rejects_malformed_embedding_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "local-embed",
                "data": [{"embedding": [0.1, True]}],
            },
            request=request,
        )

    provider = LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )
    request = EmbeddingRequest(
        model="local-embed",
        inputs=["hello world"],
    )

    with pytest.raises(LlmResponseError):
        asyncio.run(provider.embed(request))
