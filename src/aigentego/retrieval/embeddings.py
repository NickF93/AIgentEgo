"""Provider-neutral embedding pipeline for ingested note chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from aigentego.llm import EmbeddingRequest, EmbeddingResponse, LlmProvider
from aigentego.retrieval.ingestion import NoteChunk


class NoteEmbeddingError(Exception):
    """Base error for local note embedding failures."""


class NoteEmbeddingResponseError(NoteEmbeddingError):
    """The provider returned embeddings that cannot be mapped to chunks."""


class NoteChunkEmbedding(BaseModel):
    """A local embedding record for one ingested note chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    vector: tuple[float, ...] = Field(min_length=1)
    chunk_content_hash: str = Field(min_length=64, max_length=64)

    @field_validator("chunk_id", "model", "chunk_content_hash")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank identifiers and model names."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("chunk_content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require SHA-256 hex digest strings."""
        return _validate_sha256_hex(value)

    @field_validator("vector", mode="before")
    @classmethod
    def validate_vector_values(cls, value: object) -> object:
        """Reject non-finite or non-numeric vector values before coercion."""
        if not isinstance(value, list | tuple):
            return value
        for number in value:
            if not isinstance(number, int | float) or isinstance(number, bool):
                raise ValueError("vector values must be numeric")
            if not math.isfinite(float(number)):
                raise ValueError("vector values must be finite")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Keep recorded dimensions aligned with vector length."""
        if self.dimensions != len(self.vector):
            raise ValueError("dimensions must match vector length")
        return self


class NoteEmbeddingStoreProtocol(Protocol):
    """Store operations required by the embedding pipeline."""

    def list_chunks_missing_embeddings(
        self,
        chunks: Sequence[NoteChunk],
        model: str,
    ) -> list[NoteChunk]:
        """Return chunks that do not have current embeddings for model."""

    def upsert_embeddings(
        self,
        embeddings: Sequence[NoteChunkEmbedding],
    ) -> list[NoteChunkEmbedding]:
        """Persist embeddings and return stored records."""


class NoteEmbeddingPipeline:
    """Generate and persist embeddings for existing note chunks."""

    def __init__(
        self,
        *,
        provider: LlmProvider,
        store: NoteEmbeddingStoreProtocol,
        model: str,
    ) -> None:
        model_name = model.strip()
        if not model_name:
            raise ValueError("embedding model must not be blank")
        self._provider = provider
        self._store = store
        self._model = model_name

    async def embed_chunks(
        self,
        chunks: Sequence[NoteChunk],
    ) -> tuple[NoteChunkEmbedding, ...]:
        """Embed chunks missing current local embeddings."""
        pending_chunks = tuple(
            self._store.list_chunks_missing_embeddings(chunks, self._model),
        )
        if not pending_chunks:
            return ()

        response = await self._provider.embed(
            EmbeddingRequest(
                model=self._model,
                inputs=[chunk.content for chunk in pending_chunks],
            ),
        )
        embeddings = build_note_chunk_embeddings(
            pending_chunks,
            response,
            expected_model=self._model,
        )
        return tuple(self._store.upsert_embeddings(embeddings))


def build_note_chunk_embeddings(
    chunks: Sequence[NoteChunk],
    response: EmbeddingResponse,
    *,
    expected_model: str,
) -> tuple[NoteChunkEmbedding, ...]:
    """Map provider embeddings to chunks after deterministic validation."""
    model_name = expected_model.strip()
    if not model_name:
        raise ValueError("embedding model must not be blank")
    if response.model != model_name:
        raise NoteEmbeddingResponseError("embedding response model mismatch")
    if len(response.embeddings) != len(chunks):
        raise NoteEmbeddingResponseError("embedding response count mismatch")

    embeddings: list[NoteChunkEmbedding] = []
    for chunk, vector in zip(chunks, response.embeddings, strict=True):
        try:
            embeddings.append(
                NoteChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    model=model_name,
                    dimensions=len(vector),
                    vector=tuple(vector),
                    chunk_content_hash=chunk.content_hash,
                ),
            )
        except ValidationError as exc:
            raise NoteEmbeddingResponseError("invalid embedding vector") from exc
    return tuple(embeddings)


def _validate_sha256_hex(value: str) -> str:
    if len(value) != 64:
        raise ValueError("hash must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("hash must be a SHA-256 hex digest") from exc
    return value


__all__ = [
    "NoteChunkEmbedding",
    "NoteEmbeddingError",
    "NoteEmbeddingPipeline",
    "NoteEmbeddingResponseError",
    "build_note_chunk_embeddings",
]
