"""Asynchronous Ollama HTTP client."""

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


class OllamaClient:
    """Minimal async client for Ollama's native HTTP API."""

    provider_name = "ollama"

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
        """Return whether Ollama is reachable."""
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
        except httpx.RequestError:
            return False

        return response.is_success

    async def list_models(self) -> list[ModelInfo]:
        """Return models available through Ollama."""
        data = await self._request("GET", "/api/tags", operation="list_models")
        models = data.get("models")
        if not isinstance(models, list):
            raise self._response_error("list_models", "missing models list")

        parsed: list[ModelInfo] = []
        for model_data in models:
            if not isinstance(model_data, dict):
                raise self._response_error("list_models", "invalid model metadata")
            parsed.append(self._parse_model_info(model_data, operation="list_models"))

        return parsed

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a non-streaming Ollama chat response."""
        payload = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
        }
        data = await self._request(
            "POST",
            "/api/chat",
            operation="chat",
            json=payload,
        )

        try:
            message = ChatMessage.model_validate(data.get("message"))
            return ChatResponse(
                model=self._required_str(data, "model", operation="chat"),
                message=message,
                done=self._required_bool(data, "done", operation="chat"),
                done_reason=self._optional_str(data.get("done_reason")),
                prompt_eval_count=self._optional_int(data.get("prompt_eval_count")),
                eval_count=self._optional_int(data.get("eval_count")),
            )
        except ValidationError as exc:
            raise self._response_error("chat", "invalid chat response") from exc

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings through Ollama."""
        payload = {
            "model": request.model,
            "input": request.inputs,
        }
        data = await self._request(
            "POST",
            "/api/embed",
            operation="embed",
            json=payload,
        )

        return EmbeddingResponse(
            model=self._required_str(data, "model", operation="embed"),
            embeddings=self._parse_embeddings(data.get("embeddings")),
            prompt_eval_count=self._optional_int(data.get("prompt_eval_count")),
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
        details = model_data.get("details")
        if details is not None and not isinstance(details, dict):
            raise self._response_error(operation, "invalid model details")

        try:
            return ModelInfo(
                name=self._required_str(model_data, "name", operation=operation),
                model=self._optional_str(model_data.get("model")),
                modified_at=self._optional_str(model_data.get("modified_at")),
                size=self._optional_int(model_data.get("size")),
                digest=self._optional_str(model_data.get("digest")),
                details=details,
            )
        except ValidationError as exc:
            raise self._response_error(operation, "invalid model metadata") from exc

    def _parse_embeddings(self, value: object) -> list[list[float]]:
        if not isinstance(value, list):
            raise self._response_error("embed", "missing embeddings list")

        embeddings: list[list[float]] = []
        for embedding in value:
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

    def _required_bool(
        self,
        data: Mapping[str, Any],
        key: str,
        *,
        operation: str,
    ) -> bool:
        value = data.get(key)
        if not isinstance(value, bool):
            raise self._response_error(operation, f"missing boolean field {key}")
        return value

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise self._response_error("parse", "expected optional string field")

    def _optional_int(self, value: object) -> int | None:
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
