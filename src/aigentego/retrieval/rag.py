"""Deterministic RAG context assembly from local note search results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aigentego.retrieval.search import NoteSearchResult

DEFAULT_RAG_MAX_SNIPPETS = 5
DEFAULT_RAG_MAX_TOTAL_CHARACTERS = 4000


class RagContextLimits(BaseModel):
    """Deterministic bounds for prompt-ready RAG context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_snippets: int = Field(default=DEFAULT_RAG_MAX_SNIPPETS, ge=1)
    max_total_characters: int = Field(
        default=DEFAULT_RAG_MAX_TOTAL_CHARACTERS,
        ge=1,
    )

    @field_validator("max_snippets", "max_total_characters", mode="before")
    @classmethod
    def validate_integer_limit(cls, value: object) -> object:
        """Reject bools before Pydantic can coerce them into integers."""
        if isinstance(value, bool):
            raise ValueError("context limits must be integers")
        return value


class RagContextItem(BaseModel):
    """One source-attributed RAG context snippet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snippet: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    relative_path: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    truncated: bool = False

    @field_validator("snippet", "chunk_id", "document_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank context snippets and identifiers."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Expose only safe source paths relative to allowed roots."""
        if not value.strip():
            raise ValueError("relative_path must not be blank")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within the allowed root")
        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Reject non-finite scores."""
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RagContext(BaseModel):
    """Inspectable provider-neutral RAG context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[RagContextItem, ...] = ()
    source_result_count: int = Field(ge=0)
    excluded_by_limit_count: int = Field(ge=0)
    total_snippet_characters: int = Field(ge=0)
    limits: RagContextLimits = Field(default_factory=RagContextLimits)
    truncated: bool = False

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        """Keep context summary fields aligned with included items and limits."""
        if len(self.items) > self.limits.max_snippets:
            raise ValueError("items must not exceed max_snippets")
        if self.source_result_count < len(self.items):
            raise ValueError("source_result_count must cover included items")
        expected_excluded = self.source_result_count - len(self.items)
        if self.excluded_by_limit_count != expected_excluded:
            raise ValueError("excluded_by_limit_count must match omitted results")

        total_characters = sum(len(item.snippet) for item in self.items)
        if self.total_snippet_characters != total_characters:
            raise ValueError("total_snippet_characters must match items")
        if total_characters > self.limits.max_total_characters:
            raise ValueError("items must not exceed max_total_characters")

        expected_truncated = (
            self.excluded_by_limit_count > 0
            or any(item.truncated for item in self.items)
        )
        if self.truncated != expected_truncated:
            raise ValueError("truncated must reflect omitted or shortened snippets")
        return self


def build_rag_context(
    results: Sequence[NoteSearchResult],
    *,
    limits: RagContextLimits | None = None,
) -> RagContext:
    """Build bounded context from already-ranked local note search results."""
    context_limits = limits or RagContextLimits()
    items: list[RagContextItem] = []
    remaining_characters = context_limits.max_total_characters

    for result in results:
        if len(items) >= context_limits.max_snippets:
            break
        if remaining_characters <= 0:
            break

        snippet = result.snippet
        truncated = False
        if len(snippet) > remaining_characters:
            snippet = snippet[:remaining_characters]
            truncated = True

        items.append(
            RagContextItem(
                snippet=snippet,
                score=result.score,
                relative_path=result.relative_path,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                truncated=truncated,
            ),
        )
        remaining_characters -= len(snippet)

    return RagContext(
        items=tuple(items),
        source_result_count=len(results),
        excluded_by_limit_count=len(results) - len(items),
        total_snippet_characters=sum(len(item.snippet) for item in items),
        limits=context_limits,
        truncated=len(items) < len(results) or any(item.truncated for item in items),
    )


def format_rag_context_for_prompt(context: RagContext) -> str:
    """Render RAG context as deterministic source-attributed prompt text."""
    if not context.items:
        return ""

    sections = ["Retrieved local note context:"]
    for index, item in enumerate(context.items, start=1):
        marker = " truncated" if item.truncated else ""
        sections.extend(
            [
                (
                    f"[{index}] source={item.relative_path} "
                    f"chunk={item.chunk_index} score={item.score:.6f}{marker}"
                ),
                item.snippet,
            ],
        )
    return "\n".join(sections)


__all__ = [
    "DEFAULT_RAG_MAX_SNIPPETS",
    "DEFAULT_RAG_MAX_TOTAL_CHARACTERS",
    "RagContext",
    "RagContextItem",
    "RagContextLimits",
    "build_rag_context",
    "format_rag_context_for_prompt",
]
