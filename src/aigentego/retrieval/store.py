"""SQLite-backed retrieval metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from aigentego.retrieval.file_discovery import FileMetadata
from aigentego.retrieval.filesystem_policy import (
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)


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
            self._validate_metadata(metadata) for metadata in metadata_items
        ]
        with self._connection:
            for metadata in validated_items:
                self._connection.execute(
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

    def _validate_metadata(self, metadata: FileMetadata) -> FileMetadata:
        resolved_path = self._policy.resolve_read_path(metadata.path)
        resolved_root = self._policy.resolve_read_path(metadata.root_path)
        allowed_roots = {
            allowed_root.path for allowed_root in self._policy.allowed_roots
        }
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


def _metadata_from_row(row: sqlite3.Row) -> FileMetadata:
    return FileMetadata.model_validate(_row_to_dict(row))


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


__all__ = ["FileMetadataStore"]
