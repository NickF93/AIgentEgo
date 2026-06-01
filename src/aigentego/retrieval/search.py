"""Deterministic local notes search over persisted chunk embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aigentego.llm import EmbeddingRequest, EmbeddingResponse, LlmProvider
from aigentego.retrieval.store import EmbeddedNoteChunk

DEFAULT_NOTE_SEARCH_TOP_K = 5


class NoteSearchError(Exception):
    """Base error for local note search failures."""


class NoteSearchResponseError(NoteSearchError):
    """The provider returned a query embedding that cannot be searched."""


class NoteSearchRequest(BaseModel):
    """A local notes search request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_NOTE_SEARCH_TOP_K, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject blank queries and search the normalized text."""
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, value: object) -> object:
        """Reject bools before Pydantic can coerce them into integers."""
        if isinstance(value, bool):
            raise ValueError("top_k must be an integer")
        return value


class NoteSearchResult(BaseModel):
    """One ranked local note search result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=-1.0, le=1.0)
    snippet: str = Field(min_length=1)

    @field_validator("chunk_id", "document_id", "snippet")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank result identifiers and snippets."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Expose only safe source paths relative to allowed roots."""
        if not value.strip():
            raise ValueError("relative_path must not be blank")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within the allowed root")
        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Reject non-finite scores."""
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class NoteSearchResponse(BaseModel):
    """Inspectable local note search response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    model: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    candidate_count: int = Field(ge=0)
    skipped_stale_embeddings: int = Field(ge=0)
    skipped_dimension_mismatches: int = Field(ge=0)
    results: tuple[NoteSearchResult, ...] = ()

    @field_validator("query", "model")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank response text fields."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_result_count(self) -> Self:
        """Keep response cardinality aligned with the requested limit."""
        if len(self.results) > self.top_k:
            raise ValueError("results must not exceed top_k")
        return self


class NoteSearchStoreProtocol(Protocol):
    """Store operations required by the local note search pipeline."""

    def list_embedded_chunks(self, model: str) -> list[EmbeddedNoteChunk]:
        """Return embedded chunks for the configured embedding model."""


class NoteSearchPipeline:
    """Search persisted note chunk embeddings with provider-neutral queries."""

    def __init__(
        self,
        *,
        provider: LlmProvider,
        store: NoteSearchStoreProtocol,
        model: str,
    ) -> None:
        model_name = model.strip()
        if not model_name:
            raise ValueError("embedding model must not be blank")
        self._provider = provider
        self._store = store
        self._model = model_name

    async def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_NOTE_SEARCH_TOP_K,
    ) -> NoteSearchResponse:
        """Embed a query and rank locally persisted note chunks."""
        request = NoteSearchRequest(query=query, top_k=top_k)
        response = await self._provider.embed(
            EmbeddingRequest(model=self._model, inputs=[request.query]),
        )
        query_vector = _query_vector_from_response(
            response,
            expected_model=self._model,
        )
        candidates = self._store.list_embedded_chunks(self._model)
        ranked: list[NoteSearchResult] = []
        skipped_stale = 0
        skipped_dimensions = 0

        for candidate in candidates:
            if candidate.embedding.chunk_content_hash != candidate.chunk.content_hash:
                skipped_stale += 1
                continue
            if candidate.embedding.dimensions != len(query_vector):
                skipped_dimensions += 1
                continue
            ranked.append(
                NoteSearchResult(
                    chunk_id=candidate.chunk.chunk_id,
                    document_id=candidate.chunk.document_id,
                    relative_path=candidate.relative_path,
                    chunk_index=candidate.chunk.chunk_index,
                    score=_cosine_similarity(
                        query_vector,
                        candidate.embedding.vector,
                    ),
                    snippet=candidate.chunk.content,
                ),
            )

        ranked.sort(key=_result_sort_key)
        return NoteSearchResponse(
            query=request.query,
            model=self._model,
            top_k=request.top_k,
            candidate_count=len(candidates),
            skipped_stale_embeddings=skipped_stale,
            skipped_dimension_mismatches=skipped_dimensions,
            results=tuple(ranked[: request.top_k]),
        )


def _query_vector_from_response(
    response: EmbeddingResponse,
    *,
    expected_model: str,
) -> tuple[float, ...]:
    model_name = expected_model.strip()
    if not model_name:
        raise ValueError("embedding model must not be blank")
    if response.model != model_name:
        raise NoteSearchResponseError("query embedding response model mismatch")
    if len(response.embeddings) != 1:
        raise NoteSearchResponseError("query embedding response count mismatch")
    return _validate_vector(response.embeddings[0])


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if not vector:
        raise NoteSearchResponseError("query embedding vector must not be empty")
    normalized: list[float] = []
    for number in vector:
        if not isinstance(number, int | float) or isinstance(number, bool):
            raise NoteSearchResponseError("query embedding vector must be numeric")
        parsed = float(number)
        if not math.isfinite(parsed):
            raise NoteSearchResponseError("query embedding vector must be finite")
        normalized.append(parsed)
    return tuple(normalized)


def _cosine_similarity(
    query_vector: tuple[float, ...],
    candidate_vector: tuple[float, ...],
) -> float:
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_vector))
    denominator = query_norm * candidate_norm
    if denominator == 0:
        return 0.0

    score = sum(
        query_value * candidate_value
        for query_value, candidate_value in zip(
            query_vector,
            candidate_vector,
            strict=True,
        )
    ) / denominator
    return max(-1.0, min(1.0, score))


def _result_sort_key(
    result: NoteSearchResult,
) -> tuple[float, str, int, str, str]:
    return (
        -result.score,
        result.relative_path,
        result.chunk_index,
        result.document_id,
        result.chunk_id,
    )


__all__ = [
    "DEFAULT_NOTE_SEARCH_TOP_K",
    "NoteSearchError",
    "NoteSearchPipeline",
    "NoteSearchRequest",
    "NoteSearchResponse",
    "NoteSearchResponseError",
    "NoteSearchResult",
]
