"""Local retrieval safety boundaries."""

from aigentego.retrieval.embeddings import (
    NoteChunkEmbedding,
    NoteEmbeddingError,
    NoteEmbeddingPipeline,
    NoteEmbeddingResponseError,
    build_note_chunk_embeddings,
)
from aigentego.retrieval.file_discovery import (
    SUPPORTED_NOTE_EXTENSIONS,
    FileMetadata,
    discover_note_files,
)
from aigentego.retrieval.filesystem_policy import (
    AllowedRoot,
    FilesystemAccessError,
    InvalidAllowedRootError,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)
from aigentego.retrieval.ingestion import (
    DEFAULT_CHUNK_SIZE_CHARS,
    NoteChunk,
    NoteDecodeError,
    NoteDocument,
    NoteIngestionError,
    NoteIngestionResult,
    UnsupportedNoteFileError,
    chunk_note_document,
    ingest_note_file,
    ingest_note_path,
    normalize_note_text,
)
from aigentego.retrieval.store import (
    FileMetadataStore,
    NoteChunkStore,
    NoteEmbeddingStore,
)

__all__ = [
    "AllowedRoot",
    "DEFAULT_CHUNK_SIZE_CHARS",
    "FileMetadata",
    "FileMetadataStore",
    "FilesystemAccessError",
    "InvalidAllowedRootError",
    "NoteChunk",
    "NoteChunkEmbedding",
    "NoteChunkStore",
    "NoteDecodeError",
    "NoteDocument",
    "NoteEmbeddingError",
    "NoteEmbeddingPipeline",
    "NoteEmbeddingResponseError",
    "NoteEmbeddingStore",
    "NoteIngestionError",
    "NoteIngestionResult",
    "PathOutsideAllowedRootsError",
    "ReadOnlyFilesystemPolicy",
    "SUPPORTED_NOTE_EXTENSIONS",
    "UnsupportedNoteFileError",
    "build_note_chunk_embeddings",
    "chunk_note_document",
    "discover_note_files",
    "ingest_note_file",
    "ingest_note_path",
    "normalize_note_text",
]
