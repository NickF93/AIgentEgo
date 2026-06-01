import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aigentego.persistence import open_sqlite_database
from aigentego.retrieval import (
    FileMetadata,
    FileMetadataStore,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
    discover_note_files,
)


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = open_sqlite_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_store_upserts_and_lists_file_metadata_deterministically(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "b.md").write_text("b\n", encoding="utf-8")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (nested / "c.markdown").write_text("c\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    metadata_items = discover_note_files(policy)
    store = FileMetadataStore(sqlite_connection, policy)

    assert store.upsert_files(list(reversed(metadata_items))) == list(
        reversed(metadata_items),
    )

    assert [metadata.relative_path for metadata in store.list_files()] == [
        "a.txt",
        "b.md",
        "nested/c.markdown",
    ]
    assert store.list_files(root.resolve()) == metadata_items
    assert store.get_file(root / "a.txt") == metadata_items[0]


def test_store_replaces_file_metadata(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    file_path = root / "note.md"
    file_path.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    metadata = discover_note_files(policy)[0]
    updated = metadata.model_copy(
        update={"size_bytes": metadata.size_bytes + 10},
    )
    store = FileMetadataStore(sqlite_connection, policy)

    store.upsert_file(metadata)

    assert store.upsert_file(updated) == updated
    assert store.get_file(file_path) == updated
    assert store.list_files() == [updated]


def test_store_handles_multiple_roots_deterministically(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root_b = tmp_path / "b-notes"
    root_a = tmp_path / "a-notes"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "b.md").write_text("b\n", encoding="utf-8")
    (root_a / "a.md").write_text("a\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root_b, root_a])
    metadata_items = discover_note_files(policy)
    store = FileMetadataStore(sqlite_connection, policy)

    store.upsert_files(metadata_items)

    assert [
        (metadata.root_path.name, metadata.relative_path)
        for metadata in store.list_files()
    ] == [
        ("a-notes", "a.md"),
        ("b-notes", "b.md"),
    ]
    assert [metadata.relative_path for metadata in store.list_files(root_a)] == [
        "a.md",
    ]


def test_store_rejects_metadata_outside_allowed_roots(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    candidate = outside / "secret.md"
    candidate.write_text("secret\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    store = FileMetadataStore(sqlite_connection, policy)
    metadata = FileMetadata(
        path=candidate.resolve(),
        root_path=outside.resolve(),
        relative_path="secret.md",
        size_bytes=1,
        modified_time_ns=1,
        extension=".md",
    )

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        store.upsert_file(metadata)


def test_store_rejects_metadata_for_unconfigured_subroot(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    subroot = root / "subroot"
    subroot.mkdir(parents=True)
    candidate = subroot / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    store = FileMetadataStore(sqlite_connection, policy)
    metadata = FileMetadata(
        path=candidate.resolve(),
        root_path=subroot.resolve(),
        relative_path="note.md",
        size_bytes=1,
        modified_time_ns=1,
        extension=".md",
    )

    with pytest.raises(PathOutsideAllowedRootsError, match="configured allowed root"):
        store.upsert_file(metadata)


def test_note_file_schema_is_idempotent(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "aigentego.sqlite3"
    first_connection = open_sqlite_database(sqlite_path)
    first_connection.close()
    second_connection = open_sqlite_database(sqlite_path)
    try:
        rows = second_connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'note_files'
            """,
        ).fetchall()
    finally:
        second_connection.close()

    assert [row["name"] for row in rows] == ["note_files"]


def test_store_returns_json_serializable_metadata(
    tmp_path: Path,
    sqlite_connection: sqlite3.Connection,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    metadata = discover_note_files(policy)[0]
    store = FileMetadataStore(sqlite_connection, policy)

    stored = store.upsert_file(metadata)

    assert stored.model_dump(mode="json") == {
        "path": str((root / "note.md").resolve()),
        "root_path": str(root.resolve()),
        "relative_path": "note.md",
        "size_bytes": len("# note\n"),
        "modified_time_ns": metadata.modified_time_ns,
        "extension": ".md",
    }
