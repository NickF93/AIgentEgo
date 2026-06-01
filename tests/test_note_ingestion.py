import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aigentego.retrieval import (
    NoteChunk,
    NoteDecodeError,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
    UnsupportedNoteFileError,
    ingest_note_file,
    ingest_note_path,
    normalize_note_text,
)
from aigentego.retrieval.file_discovery import discover_note_files


def test_ingests_txt_file_under_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.txt"
    note_path.write_text("alpha\nbeta\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    result = ingest_note_path(note_path, policy, max_chunk_chars=6)

    assert result.document.file_metadata.path == note_path.resolve()
    assert result.document.file_metadata.relative_path == "note.txt"
    assert result.document.file_metadata.extension == ".txt"
    assert result.document.content_length == len("alpha\nbeta\n")
    assert result.document.content_hash == _sha256_text("alpha\nbeta\n")
    assert [(chunk.chunk_index, chunk.content) for chunk in result.chunks] == [
        (0, "alpha\n"),
        (1, "beta\n"),
    ]


def test_ingests_markdown_discovery_metadata(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    nested = root / "nested"
    nested.mkdir(parents=True)
    note_path = nested / "note.markdown"
    note_path.write_text("# Title\r\nBody\r\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])
    metadata = discover_note_files(policy)[0]

    result = ingest_note_file(metadata, policy, max_chunk_chars=20)

    assert result.document.file_metadata.path == note_path.resolve()
    assert result.document.file_metadata.relative_path == "nested/note.markdown"
    assert result.document.content_length == len("# Title\nBody\n")
    assert [chunk.content for chunk in result.chunks] == ["# Title\nBody\n"]


def test_ingestion_rejects_unsupported_extension_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.pdf"
    note_path.write_text("not a note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError(f"read_text called for {self}")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(UnsupportedNoteFileError, match="unsupported"):
        ingest_note_path(note_path, policy)


def test_ingestion_rejects_outside_root_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    note_path = outside / "secret.md"
    note_path.write_text("# secret\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError(f"read_text called for {self}")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        ingest_note_path(note_path, policy)


def test_ingestion_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "secret.md"
    target.write_text("# secret\n", encoding="utf-8")
    link = root / "linked-secret.md"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable in this environment: {exc}")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        ingest_note_path(link, policy)


def test_ingestion_rejects_invalid_utf8(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "bad.md"
    note_path.write_bytes(b"# bad\n\xff")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(NoteDecodeError, match="UTF-8"):
        ingest_note_path(note_path, policy)


def test_line_endings_are_normalized_deterministically() -> None:
    assert normalize_note_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_chunk_output_is_deterministic_and_json_serializable(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text("abcdefghi", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    first = ingest_note_path(note_path, policy, max_chunk_chars=3)
    second = ingest_note_path(note_path, policy, max_chunk_chars=3)

    assert first == second
    assert [chunk.chunk_index for chunk in first.chunks] == [0, 1, 2]
    assert [chunk.content for chunk in first.chunks] == ["abc", "def", "ghi"]
    assert json.loads(first.model_dump_json())["chunks"][0]["content"] == "abc"


def test_chunk_model_rejects_inconsistent_content_metadata() -> None:
    with pytest.raises(ValidationError, match="content_length"):
        NoteChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="abc",
            content_length=2,
            content_hash=_sha256_text("abc"),
        )

    with pytest.raises(ValidationError, match="content_hash"):
        NoteChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="abc",
            content_length=3,
            content_hash=_sha256_text("def"),
        )


def test_invalid_chunk_size_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    note_path = root / "note.md"
    note_path.write_text("abc", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(ValueError, match="max_chunk_chars"):
        ingest_note_path(note_path, policy, max_chunk_chars=0)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
