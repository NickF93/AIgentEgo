"""Read-only discovery of local note file metadata."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aigentego.retrieval.filesystem_policy import (
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)

SUPPORTED_NOTE_EXTENSIONS = (".md", ".markdown", ".txt")


class FileMetadata(BaseModel):
    """Inspectable metadata for a supported note file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    root_path: Path
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    modified_time_ns: int = Field(ge=0)
    extension: str = Field(min_length=1)

    @field_validator("path", "root_path")
    @classmethod
    def validate_absolute_path(cls, value: Path) -> Path:
        """Require stored paths to be normalized absolute paths."""
        if not value.is_absolute():
            raise ValueError("file metadata paths must be absolute")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Keep relative paths local to their allowed root."""
        if not value.strip():
            raise ValueError("relative_path must not be blank")
        parts = Path(value).parts
        if Path(value).is_absolute() or ".." in parts:
            raise ValueError("relative_path must stay inside root_path")
        return value

    @field_validator("extension")
    @classmethod
    def validate_extension(cls, value: str) -> str:
        """Limit discovery metadata to supported note extensions."""
        extension = value.lower()
        if extension not in SUPPORTED_NOTE_EXTENSIONS:
            raise ValueError("unsupported note file extension")
        return extension

    @model_validator(mode="after")
    def validate_path_relationship(self) -> FileMetadata:
        """Keep path, root, relative path, and extension consistent."""
        try:
            relative_path = self.path.relative_to(self.root_path).as_posix()
        except ValueError as exc:
            raise ValueError("path must be inside root_path") from exc

        if self.relative_path != relative_path:
            raise ValueError("relative_path must match path relative to root_path")
        if self.extension != self.path.suffix.lower():
            raise ValueError("extension must match path suffix")
        return self


def discover_note_files(policy: ReadOnlyFilesystemPolicy) -> list[FileMetadata]:
    """Discover supported note files under policy-approved roots."""
    discovered: list[FileMetadata] = []
    for allowed_root in policy.allowed_roots:
        root_path = policy.resolve_read_path(allowed_root.path)
        discovered.extend(_discover_root(policy, root_path))
    return sorted(
        discovered,
        key=lambda metadata: (
            metadata.root_path.as_posix(),
            metadata.relative_path,
        ),
    )


def _discover_root(
    policy: ReadOnlyFilesystemPolicy,
    root_path: Path,
) -> Iterable[FileMetadata]:
    pending = [root_path]
    while pending:
        current_dir = pending.pop()
        for child in _iter_children(current_dir):
            if child.is_symlink():
                continue

            try:
                resolved_child = policy.resolve_read_path(child)
            except PathOutsideAllowedRootsError:
                continue

            if resolved_child.is_dir():
                pending.append(resolved_child)
                pending.sort(key=lambda path: path.as_posix(), reverse=True)
                continue

            if not resolved_child.is_file():
                continue
            if resolved_child.suffix.lower() not in SUPPORTED_NOTE_EXTENSIONS:
                continue

            yield _metadata_for_file(root_path=root_path, file_path=resolved_child)


def _iter_children(directory: Path) -> list[Path]:
    return sorted(directory.iterdir(), key=lambda path: path.name)


def _metadata_for_file(*, root_path: Path, file_path: Path) -> FileMetadata:
    stat_result = file_path.stat()
    return FileMetadata(
        path=file_path,
        root_path=root_path,
        relative_path=file_path.relative_to(root_path).as_posix(),
        size_bytes=stat_result.st_size,
        modified_time_ns=stat_result.st_mtime_ns,
        extension=file_path.suffix.lower(),
    )


__all__ = [
    "FileMetadata",
    "SUPPORTED_NOTE_EXTENSIONS",
    "discover_note_files",
]
