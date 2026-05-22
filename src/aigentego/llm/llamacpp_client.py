"""Asynchronous llama.cpp HTTP provider adapter."""

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from aigentego.llm.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
)
from aigentego.llm.errors import (
    LlmConnectionError,
    LlmResponseError,
    LlmTimeoutError,
)

JsonObject = dict[str, Any]


class LlamaCppProvider:
    """Provider adapter for llama-server's OpenAI-compatible HTTP API."""

    provider_name = "llamacpp"

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._transport = transport

    async def health(self) -> bool:
        """Return whether llama-server is ready."""
        try:
            async with self._client() as client:
                response = await client.get("/health")
        except httpx.RequestError:
            return False

        if not response.is_success:
            return False

        try:
            data = response.json()
        except ValueError:
            return False

        return isinstance(data, dict) and data.get("status") == "ok"

    async def list_models(self) -> list[ModelInfo]:
        """Return models exposed through llama-server."""
        data = await self._request("GET", "/v1/models", operation="list_models")
        models = data.get("data")
        if not isinstance(models, list):
            raise self._response_error("list_models", "missing model data list")

        parsed: list[ModelInfo] = []
        for model_data in models:
            if not isinstance(model_data, dict):
                raise self._response_error("list_models", "invalid model metadata")
            parsed.append(self._parse_model_info(model_data, operation="list_models"))

        return parsed

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a non-streaming chat response through llama-server."""
        payload = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
        }
        data = await self._request(
            "POST",
            "/v1/chat/completions",
            operation="chat",
            json=payload,
        )
        choice = self._first_choice(data, operation="chat")
        message_data = choice.get("message")
        if not isinstance(message_data, dict):
            raise self._response_error("chat", "missing chat message")

        try:
            message = ChatMessage.model_validate(message_data)
            return ChatResponse(
                model=self._required_str(data, "model", operation="chat"),
                message=message,
                done=True,
                done_reason=self._optional_str(choice.get("finish_reason")),
                prompt_eval_count=self._usage_int(data, "prompt_tokens"),
                eval_count=self._usage_int(data, "completion_tokens"),
            )
        except ValidationError as exc:
            raise self._response_error("chat", "invalid chat response") from exc

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings through llama-server."""
        payload = {
            "model": request.model,
            "input": request.inputs,
            "encoding_format": "float",
        }
        data = await self._request(
            "POST",
            "/v1/embeddings",
            operation="embed",
            json=payload,
        )

        return EmbeddingResponse(
            model=self._required_str(data, "model", operation="embed"),
            embeddings=self._parse_embeddings(data.get("data")),
            prompt_eval_count=self._usage_int(data, "prompt_tokens"),
        )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        try:
            async with self._client() as client:
                response = await client.request(method, path, json=json)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LlmTimeoutError(
                "request timed out",
                provider=self.provider_name,
                operation=operation,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LlmResponseError(
                "request failed",
                provider=self.provider_name,
                operation=operation,
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise LlmConnectionError(
                "request failed",
                provider=self.provider_name,
                operation=operation,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise self._response_error(operation, "invalid JSON response") from exc

        if not isinstance(data, dict):
            raise self._response_error(operation, "response must be a JSON object")

        return data

    def _parse_model_info(
        self,
        model_data: Mapping[str, Any],
        *,
        operation: str,
    ) -> ModelInfo:
        model_id = self._required_str(model_data, "id", operation=operation)
        details = {key: value for key, value in model_data.items() if key != "id"}
        try:
            return ModelInfo(
                name=model_id,
                model=model_id,
                details=details or None,
            )
        except ValidationError as exc:
            raise self._response_error(operation, "invalid model metadata") from exc

    def _first_choice(
        self,
        data: Mapping[str, Any],
        *,
        operation: str,
    ) -> Mapping[str, Any]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._response_error(operation, "missing choices list")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise self._response_error(operation, "invalid choice metadata")

        return choice

    def _parse_embeddings(self, value: object) -> list[list[float]]:
        if not isinstance(value, list):
            raise self._response_error("embed", "missing embedding data list")

        embeddings: list[list[float]] = []
        for item in value:
            if not isinstance(item, dict):
                raise self._response_error("embed", "invalid embedding metadata")
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise self._response_error("embed", "invalid embedding vector")

            vector: list[float] = []
            for number in embedding:
                if (
                    not isinstance(number, int | float)
                    or isinstance(number, bool)
                ):
                    raise self._response_error("embed", "invalid embedding value")
                vector.append(float(number))

            embeddings.append(vector)

        return embeddings

    def _required_str(
        self,
        data: Mapping[str, Any],
        key: str,
        *,
        operation: str,
    ) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            raise self._response_error(operation, f"missing string field {key}")
        return value

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise self._response_error("parse", "expected optional string field")

    def _usage_int(self, data: Mapping[str, Any], key: str) -> int | None:
        usage = data.get("usage")
        if usage is None:
            return None
        if not isinstance(usage, dict):
            raise self._response_error("parse", "invalid usage metadata")

        value = usage.get(key)
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise self._response_error("parse", "expected optional integer field")

    def _response_error(self, operation: str, message: str) -> LlmResponseError:
        return LlmResponseError(
            message,
            provider=self.provider_name,
            operation=operation,
        )
