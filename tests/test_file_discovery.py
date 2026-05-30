import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from aigentego.retrieval import (
    FileMetadata,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
    discover_note_files,
)


def test_discovers_supported_note_files_under_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    nested = root / "projects"
    nested.mkdir(parents=True)
    markdown = root / "alpha.md"
    text = root / "beta.txt"
    long_markdown = nested / "gamma.markdown"
    unsupported = root / "archive.pdf"
    markdown.write_text("# alpha\n", encoding="utf-8")
    text.write_text("beta\n", encoding="utf-8")
    long_markdown.write_text("# gamma\n", encoding="utf-8")
    unsupported.write_text("not indexed\n", encoding="utf-8")
    os.utime(markdown, ns=(1_000, 1_000))

    metadata_items = discover_note_files(ReadOnlyFilesystemPolicy([root]))

    assert [metadata.relative_path for metadata in metadata_items] == [
        "alpha.md",
        "beta.txt",
        "projects/gamma.markdown",
    ]
    assert metadata_items[0].path == markdown.resolve()
    assert metadata_items[0].root_path == root.resolve()
    assert metadata_items[0].size_bytes == len("# alpha\n")
    assert metadata_items[0].modified_time_ns == 1_000
    assert metadata_items[0].extension == ".md"


def test_discovery_order_is_deterministic_across_roots(tmp_path: Path) -> None:
    root_b = tmp_path / "b-notes"
    root_a = tmp_path / "a-notes"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "b.md").write_text("b\n", encoding="utf-8")
    (root_a / "c.txt").write_text("c\n", encoding="utf-8")
    (root_a / "a.txt").write_text("a\n", encoding="utf-8")

    metadata_items = discover_note_files(ReadOnlyFilesystemPolicy([root_b, root_a]))

    assert [
        (metadata.root_path.name, metadata.relative_path)
        for metadata in metadata_items
    ] == [
        ("a-notes", "a.txt"),
        ("a-notes", "c.txt"),
        ("b-notes", "b.md"),
    ]


def test_discovery_skips_unsupported_files_and_directories(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    nested = root / "nested.md"
    nested.mkdir(parents=True)
    (nested / "inside.tmp").write_text("ignored\n", encoding="utf-8")
    (root / "note.MD").write_text("# uppercase suffix\n", encoding="utf-8")
    (root / "image.png").write_text("ignored\n", encoding="utf-8")

    metadata_items = discover_note_files(ReadOnlyFilesystemPolicy([root]))

    assert [metadata.relative_path for metadata in metadata_items] == ["note.MD"]
    assert metadata_items[0].extension == ".md"


def test_discovery_skips_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "secret.md"
    target.write_text("secret\n", encoding="utf-8")
    link = root / "linked-secret.md"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable in this environment: {exc}")

    metadata_items = discover_note_files(ReadOnlyFilesystemPolicy([root]))

    assert metadata_items == []


def test_discovery_does_not_read_file_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError(f"read_text called for {self}")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    metadata_items = discover_note_files(policy)

    assert [metadata.relative_path for metadata in metadata_items] == ["note.md"]


def test_file_metadata_is_json_serializable(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("# note\n", encoding="utf-8")

    metadata = discover_note_files(ReadOnlyFilesystemPolicy([root]))[0]

    assert json.loads(json.dumps(metadata.model_dump(mode="json"))) == {
        "path": str((root / "note.md").resolve()),
        "root_path": str(root.resolve()),
        "relative_path": "note.md",
        "size_bytes": len("# note\n"),
        "modified_time_ns": metadata.modified_time_ns,
        "extension": ".md",
    }


def test_file_metadata_rejects_outside_root_path(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    candidate = outside / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="inside root_path"):
        FileMetadata(
            path=candidate.resolve(),
            root_path=root.resolve(),
            relative_path="note.md",
            size_bytes=1,
            modified_time_ns=1,
            extension=".md",
        )


def test_file_metadata_rejects_path_traversal_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    candidate = root / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="inside root_path"):
        FileMetadata(
            path=candidate.resolve(),
            root_path=root.resolve(),
            relative_path="../note.md",
            size_bytes=1,
            modified_time_ns=1,
            extension=".md",
        )


def test_file_metadata_rejects_unsupported_extension(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    candidate = root / "note.pdf"
    candidate.write_text("pdf\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="unsupported"):
        FileMetadata(
            path=candidate.resolve(),
            root_path=root.resolve(),
            relative_path="note.pdf",
            size_bytes=1,
            modified_time_ns=1,
            extension=".pdf",
        )


def test_discovery_requires_policy_approved_existing_roots(tmp_path: Path) -> None:
    candidate = tmp_path / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")

    with pytest.raises(PathOutsideAllowedRootsError, match="no allowed"):
        ReadOnlyFilesystemPolicy().resolve_read_path(candidate)
