"""Read-only text and Markdown note ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aigentego.retrieval.file_discovery import (
    SUPPORTED_NOTE_EXTENSIONS,
    FileMetadata,
)
from aigentego.retrieval.filesystem_policy import (
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)

DEFAULT_CHUNK_SIZE_CHARS = 2_000


class NoteIngestionError(Exception):
    """Base error for local note ingestion failures."""


class UnsupportedNoteFileError(NoteIngestionError):
    """Requested path is not a supported note file."""


class NoteDecodeError(NoteIngestionError):
    """Requested note file could not be decoded as UTF-8 text."""


class NoteDocument(BaseModel):
    """Inspectable metadata for one ingested note document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    file_metadata: FileMetadata
    content_length: int = Field(ge=0)
    content_hash: str = Field(min_length=64, max_length=64)

    @field_validator("document_id", "content_hash")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank identifiers and hashes."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require SHA-256 hex digest strings."""
        return _validate_sha256_hex(value)


class NoteChunk(BaseModel):
    """One deterministic chunk of an ingested note document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    content_length: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)

    @field_validator("chunk_id", "document_id", "content_hash")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank identifiers and hashes."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require SHA-256 hex digest strings."""
        return _validate_sha256_hex(value)

    @model_validator(mode="after")
    def validate_content_metadata(self) -> Self:
        """Keep chunk metadata consistent with content."""
        if self.content_length != len(self.content):
            raise ValueError("content_length must match content length")
        if self.content_hash != _sha256_text(self.content):
            raise ValueError("content_hash must match content")
        return self


class NoteIngestionResult(BaseModel):
    """The document and chunks produced by one ingestion operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: NoteDocument
    chunks: tuple[NoteChunk, ...] = ()

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        """Require chunks to belong to the document and stay ordered."""
        expected_indices = tuple(range(len(self.chunks)))
        actual_indices = tuple(chunk.chunk_index for chunk in self.chunks)
        if actual_indices != expected_indices:
            raise ValueError("chunks must be ordered by contiguous chunk_index")
        for chunk in self.chunks:
            if chunk.document_id != self.document.document_id:
                raise ValueError("chunks must belong to the document")
        return self


def ingest_note_path(
    path: str | Path,
    policy: ReadOnlyFilesystemPolicy,
    *,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
) -> NoteIngestionResult:
    """Ingest one supported note path after read-policy validation."""
    metadata = _metadata_for_note_path(path, policy)
    return ingest_note_file(
        metadata,
        policy,
        max_chunk_chars=max_chunk_chars,
    )


def ingest_note_file(
    metadata: FileMetadata,
    policy: ReadOnlyFilesystemPolicy,
    *,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
) -> NoteIngestionResult:
    """Read, normalize, and chunk one policy-approved note file."""
    _validate_chunk_size(max_chunk_chars)
    validated_metadata = _validate_metadata(metadata, policy)
    content = _read_utf8_note(validated_metadata.path)
    normalized_content = normalize_note_text(content)
    document = NoteDocument(
        document_id=_document_id(validated_metadata),
        file_metadata=validated_metadata,
        content_length=len(normalized_content),
        content_hash=_sha256_text(normalized_content),
    )
    return NoteIngestionResult(
        document=document,
        chunks=chunk_note_document(
            document,
            normalized_content,
            max_chunk_chars=max_chunk_chars,
        ),
    )


def normalize_note_text(content: str) -> str:
    """Normalize note text deterministically without semantic parsing."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def chunk_note_document(
    document: NoteDocument,
    content: str,
    *,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
) -> tuple[NoteChunk, ...]:
    """Split normalized note content into bounded deterministic chunks."""
    _validate_chunk_size(max_chunk_chars)
    if not content:
        return ()

    chunks: list[NoteChunk] = []
    for chunk_index, start in enumerate(range(0, len(content), max_chunk_chars)):
        chunk_content = content[start : start + max_chunk_chars]
        chunk_hash = _sha256_text(chunk_content)
        chunks.append(
            NoteChunk(
                chunk_id=_chunk_id(
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content_hash=chunk_hash,
                ),
                document_id=document.document_id,
                chunk_index=chunk_index,
                content=chunk_content,
                content_length=len(chunk_content),
                content_hash=chunk_hash,
            ),
        )
    return tuple(chunks)


def _metadata_for_note_path(
    path: str | Path,
    policy: ReadOnlyFilesystemPolicy,
) -> FileMetadata:
    resolved_path = policy.resolve_read_path(path)
    extension = resolved_path.suffix.lower()
    if extension not in SUPPORTED_NOTE_EXTENSIONS:
        raise UnsupportedNoteFileError("unsupported note file extension")
    if not resolved_path.is_file():
        raise UnsupportedNoteFileError("candidate path is not a file")

    root_path = _matching_allowed_root(resolved_path, policy)
    stat_result = resolved_path.stat()
    return FileMetadata(
        path=resolved_path,
        root_path=root_path,
        relative_path=resolved_path.relative_to(root_path).as_posix(),
        size_bytes=stat_result.st_size,
        modified_time_ns=stat_result.st_mtime_ns,
        extension=extension,
    )


def _matching_allowed_root(
    path: Path,
    policy: ReadOnlyFilesystemPolicy,
) -> Path:
    for allowed_root in policy.allowed_roots:
        if path == allowed_root.path or allowed_root.path in path.parents:
            return allowed_root.path
    raise PathOutsideAllowedRootsError(
        "candidate path is outside allowed read-only roots",
    )


def _validate_metadata(
    metadata: FileMetadata,
    policy: ReadOnlyFilesystemPolicy,
) -> FileMetadata:
    if metadata.extension not in SUPPORTED_NOTE_EXTENSIONS:
        raise UnsupportedNoteFileError("unsupported note file extension")

    resolved_path = policy.resolve_read_path(metadata.path)
    resolved_root = policy.resolve_read_path(metadata.root_path)
    allowed_roots = {allowed_root.path for allowed_root in policy.allowed_roots}
    if resolved_root not in allowed_roots:
        raise PathOutsideAllowedRootsError(
            "file metadata root is not a configured allowed root",
        )
    return FileMetadata(
        path=resolved_path,
        root_path=resolved_root,
        relative_path=metadata.relative_path,
        size_bytes=metadata.size_bytes,
        modified_time_ns=metadata.modified_time_ns,
        extension=metadata.extension,
    )


def _read_utf8_note(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NoteDecodeError("note file must be valid UTF-8 text") from exc


def _validate_chunk_size(max_chunk_chars: int) -> None:
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be at least 1")


def _document_id(metadata: FileMetadata) -> str:
    return "doc_" + _sha256_bytes(metadata.path.as_posix().encode("utf-8"))


def _chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    key = f"{document_id}:{chunk_index}:{content_hash}".encode()
    return "chunk_" + _sha256_bytes(key)


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256_hex(value: str) -> str:
    if len(value) != 64:
        raise ValueError("hash must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("hash must be a SHA-256 hex digest") from exc
    return value


__all__ = [
    "DEFAULT_CHUNK_SIZE_CHARS",
    "NoteChunk",
    "NoteDecodeError",
    "NoteDocument",
    "NoteIngestionError",
    "NoteIngestionResult",
    "UnsupportedNoteFileError",
    "chunk_note_document",
    "ingest_note_file",
    "ingest_note_path",
    "normalize_note_text",
]
