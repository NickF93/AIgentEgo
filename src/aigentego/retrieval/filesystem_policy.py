"""Read-only filesystem access policy for local retrieval."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FilesystemAccessError(Exception):
    """Base error for local filesystem policy failures."""


class InvalidAllowedRootError(FilesystemAccessError):
    """Configured read-only root is invalid."""


class PathOutsideAllowedRootsError(FilesystemAccessError):
    """Requested path is not permitted by the configured roots."""


class AllowedRoot(BaseModel):
    """Resolved read-only root that may contain retrievable files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path


AllowedRootInput = str | Path | AllowedRoot
PathInput = str | Path


class ReadOnlyFilesystemPolicy:
    """Validate candidate paths against explicit read-only roots."""

    def __init__(self, allowed_roots: Iterable[AllowedRootInput] = ()) -> None:
        resolved_roots = {
            _coerce_allowed_root(root).path for root in allowed_roots
        }
        self._allowed_roots = tuple(
            AllowedRoot(path=root)
            for root in sorted(resolved_roots, key=lambda path: path.as_posix())
        )

    @property
    def allowed_roots(self) -> tuple[AllowedRoot, ...]:
        """Return normalized allowed roots in deterministic order."""
        return self._allowed_roots

    def resolve_read_path(self, candidate_path: PathInput) -> Path:
        """Return a resolved candidate path when it is inside an allowed root."""
        if not self._allowed_roots:
            raise PathOutsideAllowedRootsError(
                "no allowed read-only roots are configured",
            )

        candidate = _coerce_candidate_path(candidate_path)
        for allowed_root in self._allowed_roots:
            if _is_relative_to(candidate, allowed_root.path):
                return candidate

        raise PathOutsideAllowedRootsError(
            "candidate path is outside allowed read-only roots",
        )


def _coerce_allowed_root(root: AllowedRootInput) -> AllowedRoot:
    if isinstance(root, AllowedRoot):
        root_path = root.path
    else:
        root_path = _coerce_path(
            root,
            blank_message="allowed root must not be blank",
            error_cls=InvalidAllowedRootError,
        )
    try:
        resolved_root = root_path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAllowedRootError("allowed root must exist") from exc

    if not resolved_root.is_dir():
        raise InvalidAllowedRootError("allowed root must be an existing directory")

    return AllowedRoot(path=resolved_root)


def _coerce_candidate_path(candidate_path: PathInput) -> Path:
    path = _coerce_path(
        candidate_path,
        blank_message="candidate path must not be blank",
        error_cls=PathOutsideAllowedRootsError,
    )
    if not path.is_absolute():
        raise PathOutsideAllowedRootsError("candidate path must be absolute")

    try:
        return path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathOutsideAllowedRootsError("candidate path must exist") from exc


def _coerce_path(
    value: PathInput,
    *,
    blank_message: str,
    error_cls: type[FilesystemAccessError],
) -> Path:
    if isinstance(value, str) and not value.strip():
        raise error_cls(blank_message)
    return Path(value)


def _is_relative_to(candidate_path: Path, allowed_root: Path) -> bool:
    return candidate_path == allowed_root or allowed_root in candidate_path.parents


__all__ = [
    "AllowedRoot",
    "FilesystemAccessError",
    "InvalidAllowedRootError",
    "PathOutsideAllowedRootsError",
    "ReadOnlyFilesystemPolicy",
]
