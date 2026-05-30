from pathlib import Path

import pytest

from aigentego.retrieval import (
    InvalidAllowedRootError,
    PathOutsideAllowedRootsError,
    ReadOnlyFilesystemPolicy,
)


def test_empty_allowed_roots_reject_candidate_path(tmp_path: Path) -> None:
    candidate = tmp_path / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy()

    with pytest.raises(PathOutsideAllowedRootsError, match="no allowed"):
        policy.resolve_read_path(candidate)


def test_candidate_inside_allowed_root_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    candidate = root / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    assert policy.resolve_read_path(candidate) == candidate.resolve()


def test_nested_candidate_inside_allowed_root_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    nested = root / "projects" / "ai"
    nested.mkdir(parents=True)
    candidate = nested / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    assert policy.resolve_read_path(candidate) == candidate.resolve()


def test_candidate_outside_allowed_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    outside = tmp_path / "private.md"
    outside.write_text("private\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        policy.resolve_read_path(outside)


def test_path_traversal_outside_allowed_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "secret.md"
    target.write_text("secret\n", encoding="utf-8")
    traversal = root / ".." / "outside" / "secret.md"
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        policy.resolve_read_path(traversal)


def test_sibling_prefix_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    sibling = tmp_path / "notes-private"
    root.mkdir()
    sibling.mkdir()
    candidate = sibling / "secret.md"
    candidate.write_text("secret\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        policy.resolve_read_path(candidate)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
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
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="outside"):
        policy.resolve_read_path(link)


def test_multiple_allowed_roots_are_deduplicated_and_ordered(tmp_path: Path) -> None:
    root_b = tmp_path / "b"
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b.mkdir()

    policy = ReadOnlyFilesystemPolicy([root_b, root_a, root_b])

    assert [root.path for root in policy.allowed_roots] == [
        root_a.resolve(),
        root_b.resolve(),
    ]


def test_configured_roots_are_normalized(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    nested = root / "nested"
    nested.mkdir(parents=True)

    policy = ReadOnlyFilesystemPolicy([nested / ".."])

    assert policy.allowed_roots[0].path == root.resolve()


def test_relative_allowed_roots_are_resolved_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    monkeypatch.chdir(tmp_path)

    policy = ReadOnlyFilesystemPolicy(["notes"])

    assert policy.allowed_roots[0].path == root.resolve()


def test_blank_allowed_root_is_rejected() -> None:
    with pytest.raises(InvalidAllowedRootError, match="blank"):
        ReadOnlyFilesystemPolicy([" "])


def test_nonexistent_allowed_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidAllowedRootError, match="exist"):
        ReadOnlyFilesystemPolicy([tmp_path / "missing"])


def test_non_directory_allowed_root_is_rejected(tmp_path: Path) -> None:
    root_file = tmp_path / "notes.txt"
    root_file.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(InvalidAllowedRootError, match="directory"):
        ReadOnlyFilesystemPolicy([root_file])


def test_blank_candidate_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="blank"):
        policy.resolve_read_path(" ")


def test_relative_candidate_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    candidate = root / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    with pytest.raises(PathOutsideAllowedRootsError, match="absolute"):
        policy.resolve_read_path(Path("note.md"))


def test_policy_does_not_read_file_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    candidate = root / "note.md"
    candidate.write_text("# note\n", encoding="utf-8")
    policy = ReadOnlyFilesystemPolicy([root])

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError(f"read_text called for {self}")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert policy.resolve_read_path(candidate) == candidate.resolve()
