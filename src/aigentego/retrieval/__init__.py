"""Local retrieval safety boundaries."""

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
from aigentego.retrieval.store import FileMetadataStore

__all__ = [
    "AllowedRoot",
    "FileMetadata",
    "FileMetadataStore",
    "FilesystemAccessError",
    "InvalidAllowedRootError",
    "PathOutsideAllowedRootsError",
    "ReadOnlyFilesystemPolicy",
    "SUPPORTED_NOTE_EXTENSIONS",
    "discover_note_files",
]
