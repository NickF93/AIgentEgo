"""SQLite-backed retrieval metadata store."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aigentego.retrieval.embeddings import NoteChunkEmbedding
from aigentego.retrieval.file_discovery import FileMetadata
from aigentego.retrieval.filesystem_policy import (
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)
from aigentego.retrieval.ingestion import (
    NoteChunk,
    NoteDocument,
    NoteIngestionResult,
)


class EmbeddedNoteChunk(BaseModel):
    """A persisted note chunk with its local embedding and safe source path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: NoteChunk
    embedding: NoteChunkEmbedding
    relative_path: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Keep retrieval source metadata relative to its configured root."""
        if not value.strip():
            raise ValueError("relative_path must not be blank")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within the allowed root")
        return value


class FileMetadataStore:
    """Persist discovered note file metadata behind the read-only policy."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        policy: ReadOnlyFilesystemPolicy,
    ) -> None:
        self._connection = connection
        self._policy = policy

    def upsert_file(self, metadata: FileMetadata) -> FileMetadata:
        """Create or update metadata for one policy-approved note file."""
        return self.upsert_files([metadata])[0]

    def upsert_files(self, metadata_items: list[FileMetadata]) -> list[FileMetadata]:
        """Create or update metadata records in one transaction."""
        if not metadata_items:
            return []

        validated_items = [
            _validate_file_metadata(self._policy, metadata)
            for metadata in metadata_items
        ]
        with self._connection:
            for metadata in validated_items:
                _upsert_file_metadata(self._connection, metadata)

        stored_items: list[FileMetadata] = []
        for metadata in validated_items:
            stored = self.get_file(metadata.path)
            if stored is None:
                raise RuntimeError("failed to upsert file metadata")
            stored_items.append(stored)
        return stored_items

    def get_file(self, path: str | Path) -> FileMetadata | None:
        """Return one metadata record by resolved path."""
        resolved_path = self._policy.resolve_read_path(path)
        row = self._connection.execute(
            """
            SELECT
                path,
                root_path,
                relative_path,
                size_bytes,
                modified_time_ns,
                extension
            FROM note_files
            WHERE path = ?
            """,
            (resolved_path.as_posix(),),
        ).fetchone()
        if row is None:
            return None
        return _metadata_from_row(row)

    def list_files(self, root_path: str | Path | None = None) -> list[FileMetadata]:
        """Return file metadata in deterministic root-relative order."""
        if root_path is None:
            rows = self._connection.execute(
                """
                SELECT
                    path,
                    root_path,
                    relative_path,
                    size_bytes,
                    modified_time_ns,
                    extension
                FROM note_files
                ORDER BY root_path ASC, relative_path ASC, path ASC
                """,
            ).fetchall()
        else:
            resolved_root = self._policy.resolve_read_path(root_path)
            rows = self._connection.execute(
                """
                SELECT
                    path,
                    root_path,
                    relative_path,
                    size_bytes,
                    modified_time_ns,
                    extension
                FROM note_files
                WHERE root_path = ?
                ORDER BY relative_path ASC, path ASC
                """,
                (resolved_root.as_posix(),),
            ).fetchall()

        return [_metadata_from_row(row) for row in rows]


class NoteChunkStore:
    """Persist ingested note documents and bounded chunks locally."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        policy: ReadOnlyFilesystemPolicy,
    ) -> None:
        self._connection = connection
        self._policy = policy

    def upsert_ingested_note(
        self,
        result: NoteIngestionResult,
    ) -> NoteIngestionResult:
        """Create or replace one ingested document and its chunks."""
        document = self._validate_document(result.document)
        chunks = tuple(self._validate_chunk(chunk, document) for chunk in result.chunks)

        with self._connection:
            _upsert_file_metadata(self._connection, document.file_metadata)
            self._connection.execute(
                """
                INSERT INTO note_documents (
                    document_id,
                    file_path,
                    content_length,
                    content_hash
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    content_length = excluded.content_length,
                    content_hash = excluded.content_hash
                """,
                (
                    document.document_id,
                    document.file_metadata.path.as_posix(),
                    document.content_length,
                    document.content_hash,
                ),
            )
            self._connection.execute(
                "DELETE FROM note_chunks WHERE document_id = ?",
                (document.document_id,),
            )
            for chunk in chunks:
                self._connection.execute(
                    """
                    INSERT INTO note_chunks (
                        chunk_id,
                        document_id,
                        chunk_index,
                        content,
                        content_length,
                        content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.content_length,
                        chunk.content_hash,
                    ),
                )

        stored_document = self.get_document(document.document_id)
        if stored_document is None:
            raise RuntimeError("failed to upsert note document")
        return NoteIngestionResult(
            document=stored_document,
            chunks=tuple(self.list_chunks(document.document_id)),
        )

    def get_document(self, document_id: str) -> NoteDocument | None:
        """Return one ingested note document by id."""
        row = self._connection.execute(
            """
            SELECT
                d.document_id AS document_id,
                d.content_length AS document_content_length,
                d.content_hash AS document_content_hash,
                f.path AS path,
                f.root_path AS root_path,
                f.relative_path AS relative_path,
                f.size_bytes AS size_bytes,
                f.modified_time_ns AS modified_time_ns,
                f.extension AS extension
            FROM note_documents AS d
            JOIN note_files AS f ON f.path = d.file_path
            WHERE d.document_id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return _document_from_row(row)

    def get_document_for_file(self, path: str | Path) -> NoteDocument | None:
        """Return the ingested document for a policy-approved file path."""
        resolved_path = self._policy.resolve_read_path(path)
        row = self._connection.execute(
            """
            SELECT
                d.document_id AS document_id,
                d.content_length AS document_content_length,
                d.content_hash AS document_content_hash,
                f.path AS path,
                f.root_path AS root_path,
                f.relative_path AS relative_path,
                f.size_bytes AS size_bytes,
                f.modified_time_ns AS modified_time_ns,
                f.extension AS extension
            FROM note_documents AS d
            JOIN note_files AS f ON f.path = d.file_path
            WHERE d.file_path = ?
            """,
            (resolved_path.as_posix(),),
        ).fetchone()
        if row is None:
            return None
        return _document_from_row(row)

    def list_documents(self, root_path: str | Path | None = None) -> list[NoteDocument]:
        """Return ingested documents in deterministic source order."""
        if root_path is None:
            rows = self._connection.execute(
                """
                SELECT
                    d.document_id AS document_id,
                    d.content_length AS document_content_length,
                    d.content_hash AS document_content_hash,
                    f.path AS path,
                    f.root_path AS root_path,
                    f.relative_path AS relative_path,
                    f.size_bytes AS size_bytes,
                    f.modified_time_ns AS modified_time_ns,
                    f.extension AS extension
                FROM note_documents AS d
                JOIN note_files AS f ON f.path = d.file_path
                ORDER BY f.root_path ASC, f.relative_path ASC, f.path ASC
                """,
            ).fetchall()
        else:
            resolved_root = self._policy.resolve_read_path(root_path)
            rows = self._connection.execute(
                """
                SELECT
                    d.document_id AS document_id,
                    d.content_length AS document_content_length,
                    d.content_hash AS document_content_hash,
                    f.path AS path,
                    f.root_path AS root_path,
                    f.relative_path AS relative_path,
                    f.size_bytes AS size_bytes,
                    f.modified_time_ns AS modified_time_ns,
                    f.extension AS extension
                FROM note_documents AS d
                JOIN note_files AS f ON f.path = d.file_path
                WHERE f.root_path = ?
                ORDER BY f.relative_path ASC, f.path ASC
                """,
                (resolved_root.as_posix(),),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def list_chunks(self, document_id: str) -> list[NoteChunk]:
        """Return chunks for one document in deterministic chunk order."""
        rows = self._connection.execute(
            """
            SELECT
                chunk_id,
                document_id,
                chunk_index,
                content,
                content_length,
                content_hash
            FROM note_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC, chunk_id ASC
            """,
            (document_id,),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _validate_document(self, document: NoteDocument) -> NoteDocument:
        metadata = _validate_file_metadata(self._policy, document.file_metadata)
        return NoteDocument(
            document_id=document.document_id,
            file_metadata=metadata,
            content_length=document.content_length,
            content_hash=document.content_hash,
        )

    def _validate_chunk(
        self,
        chunk: NoteChunk,
        document: NoteDocument,
    ) -> NoteChunk:
        if chunk.document_id != document.document_id:
            raise ValueError("chunk does not belong to document")
        return chunk


class NoteEmbeddingStore:
    """Persist provider-neutral embeddings for ingested note chunks."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def upsert_embedding(
        self,
        embedding: NoteChunkEmbedding,
    ) -> NoteChunkEmbedding:
        """Create or replace one local chunk embedding."""
        return self.upsert_embeddings([embedding])[0]

    def upsert_embeddings(
        self,
        embeddings: Sequence[NoteChunkEmbedding],
    ) -> list[NoteChunkEmbedding]:
        """Create or replace local chunk embeddings in one transaction."""
        if not embeddings:
            return []

        with self._connection:
            for embedding in embeddings:
                self._connection.execute(
                    """
                    INSERT INTO note_chunk_embeddings (
                        chunk_id,
                        model,
                        dimensions,
                        vector_json,
                        chunk_content_hash
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id, model) DO UPDATE SET
                        dimensions = excluded.dimensions,
                        vector_json = excluded.vector_json,
                        chunk_content_hash = excluded.chunk_content_hash
                    """,
                    (
                        embedding.chunk_id,
                        embedding.model,
                        embedding.dimensions,
                        _vector_to_json(embedding.vector),
                        embedding.chunk_content_hash,
                    ),
                )

        stored_embeddings: list[NoteChunkEmbedding] = []
        for embedding in embeddings:
            stored = self.get_embedding(embedding.chunk_id, embedding.model)
            if stored is None:
                raise RuntimeError("failed to upsert note chunk embedding")
            stored_embeddings.append(stored)
        return stored_embeddings

    def get_embedding(
        self,
        chunk_id: str,
        model: str,
    ) -> NoteChunkEmbedding | None:
        """Return one chunk embedding by chunk and model."""
        model_name = _validate_model(model)
        row = self._connection.execute(
            """
            SELECT
                chunk_id,
                model,
                dimensions,
                vector_json,
                chunk_content_hash
            FROM note_chunk_embeddings
            WHERE chunk_id = ? AND model = ?
            """,
            (chunk_id, model_name),
        ).fetchone()
        if row is None:
            return None
        return _embedding_from_row(row)

    def list_embeddings(self, model: str | None = None) -> list[NoteChunkEmbedding]:
        """Return local embeddings in deterministic source order."""
        if model is None:
            rows = self._connection.execute(
                """
                SELECT
                    e.chunk_id AS chunk_id,
                    e.model AS model,
                    e.dimensions AS dimensions,
                    e.vector_json AS vector_json,
                    e.chunk_content_hash AS chunk_content_hash
                FROM note_chunk_embeddings AS e
                JOIN note_chunks AS c ON c.chunk_id = e.chunk_id
                JOIN note_documents AS d ON d.document_id = c.document_id
                JOIN note_files AS f ON f.path = d.file_path
                ORDER BY
                    e.model ASC,
                    f.root_path ASC,
                    f.relative_path ASC,
                    c.chunk_index ASC,
                    e.chunk_id ASC
                """,
            ).fetchall()
        else:
            model_name = _validate_model(model)
            rows = self._connection.execute(
                """
                SELECT
                    e.chunk_id AS chunk_id,
                    e.model AS model,
                    e.dimensions AS dimensions,
                    e.vector_json AS vector_json,
                    e.chunk_content_hash AS chunk_content_hash
                FROM note_chunk_embeddings AS e
                JOIN note_chunks AS c ON c.chunk_id = e.chunk_id
                JOIN note_documents AS d ON d.document_id = c.document_id
                JOIN note_files AS f ON f.path = d.file_path
                WHERE e.model = ?
                ORDER BY
                    f.root_path ASC,
                    f.relative_path ASC,
                    c.chunk_index ASC,
                    e.chunk_id ASC
                """,
                (model_name,),
            ).fetchall()
        return [_embedding_from_row(row) for row in rows]

    def list_chunks_missing_embeddings(
        self,
        chunks: Sequence[NoteChunk],
        model: str,
    ) -> list[NoteChunk]:
        """Return chunks that need embeddings for a model."""
        model_name = _validate_model(model)
        missing_chunks: list[NoteChunk] = []
        for chunk in chunks:
            existing = self.get_embedding(chunk.chunk_id, model_name)
            if existing is None or existing.chunk_content_hash != chunk.content_hash:
                missing_chunks.append(chunk)
        return missing_chunks

    def list_embedded_chunks(self, model: str) -> list[EmbeddedNoteChunk]:
        """Return embedded chunks with safe metadata in deterministic source order."""
        model_name = _validate_model(model)
        rows = self._connection.execute(
            """
            SELECT
                c.chunk_id AS chunk_id,
                c.document_id AS document_id,
                c.chunk_index AS chunk_index,
                c.content AS content,
                c.content_length AS content_length,
                c.content_hash AS content_hash,
                f.relative_path AS relative_path,
                e.model AS embedding_model,
                e.dimensions AS embedding_dimensions,
                e.vector_json AS embedding_vector_json,
                e.chunk_content_hash AS embedding_chunk_content_hash
            FROM note_chunk_embeddings AS e
            JOIN note_chunks AS c ON c.chunk_id = e.chunk_id
            JOIN note_documents AS d ON d.document_id = c.document_id
            JOIN note_files AS f ON f.path = d.file_path
            WHERE e.model = ?
            ORDER BY
                f.root_path ASC,
                f.relative_path ASC,
                c.chunk_index ASC,
                c.document_id ASC,
                c.chunk_id ASC
            """,
            (model_name,),
        ).fetchall()
        return [_embedded_chunk_from_row(row) for row in rows]


def _metadata_from_row(row: sqlite3.Row) -> FileMetadata:
    return FileMetadata.model_validate(_row_to_dict(row))


def _document_from_row(row: sqlite3.Row) -> NoteDocument:
    metadata = FileMetadata(
        path=Path(row["path"]),
        root_path=Path(row["root_path"]),
        relative_path=row["relative_path"],
        size_bytes=row["size_bytes"],
        modified_time_ns=row["modified_time_ns"],
        extension=row["extension"],
    )
    return NoteDocument(
        document_id=row["document_id"],
        file_metadata=metadata,
        content_length=row["document_content_length"],
        content_hash=row["document_content_hash"],
    )


def _chunk_from_row(row: sqlite3.Row) -> NoteChunk:
    return NoteChunk.model_validate(_row_to_dict(row))


def _embedding_from_row(row: sqlite3.Row) -> NoteChunkEmbedding:
    return NoteChunkEmbedding(
        chunk_id=row["chunk_id"],
        model=row["model"],
        dimensions=row["dimensions"],
        vector=_vector_from_json(row["vector_json"]),
        chunk_content_hash=row["chunk_content_hash"],
    )


def _embedded_chunk_from_row(row: sqlite3.Row) -> EmbeddedNoteChunk:
    chunk = NoteChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        content_length=row["content_length"],
        content_hash=row["content_hash"],
    )
    embedding = NoteChunkEmbedding(
        chunk_id=row["chunk_id"],
        model=row["embedding_model"],
        dimensions=row["embedding_dimensions"],
        vector=_vector_from_json(row["embedding_vector_json"]),
        chunk_content_hash=row["embedding_chunk_content_hash"],
    )
    return EmbeddedNoteChunk(
        chunk=chunk,
        embedding=embedding,
        relative_path=row["relative_path"],
    )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _validate_file_metadata(
    policy: ReadOnlyFilesystemPolicy,
    metadata: FileMetadata,
) -> FileMetadata:
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


def _upsert_file_metadata(
    connection: sqlite3.Connection,
    metadata: FileMetadata,
) -> None:
    connection.execute(
        """
        INSERT INTO note_files (
            path,
            root_path,
            relative_path,
            size_bytes,
            modified_time_ns,
            extension
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            root_path = excluded.root_path,
            relative_path = excluded.relative_path,
            size_bytes = excluded.size_bytes,
            modified_time_ns = excluded.modified_time_ns,
            extension = excluded.extension
        """,
        (
            metadata.path.as_posix(),
            metadata.root_path.as_posix(),
            metadata.relative_path,
            metadata.size_bytes,
            metadata.modified_time_ns,
            metadata.extension,
        ),
    )


def _validate_model(model: str) -> str:
    model_name = model.strip()
    if not model_name:
        raise ValueError("embedding model must not be blank")
    return model_name


def _vector_to_json(vector: tuple[float, ...]) -> str:
    return json.dumps(list(vector), separators=(",", ":"), allow_nan=False)


def _vector_from_json(value: str) -> tuple[float, ...]:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("stored embedding vector must be a JSON list")
    vector: list[float] = []
    for number in data:
        if not isinstance(number, int | float) or isinstance(number, bool):
            raise ValueError("stored embedding vector values must be numeric")
        parsed = float(number)
        if not math.isfinite(parsed):
            raise ValueError("stored embedding vector values must be finite")
        vector.append(parsed)
    return tuple(vector)


__all__ = [
    "EmbeddedNoteChunk",
    "FileMetadataStore",
    "NoteChunkStore",
    "NoteEmbeddingStore",
]
