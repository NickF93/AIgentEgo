import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aigentego.retrieval import (
    RagContext,
    RagContextItem,
    RagContextLimits,
    build_rag_context,
    format_rag_context_for_prompt,
)
from aigentego.retrieval.search import NoteSearchResult


def test_empty_results_produce_empty_context() -> None:
    context = build_rag_context([])

    assert context.items == ()
    assert context.source_result_count == 0
    assert context.excluded_by_limit_count == 0
    assert context.total_snippet_characters == 0
    assert context.truncated is False
    assert format_rag_context_for_prompt(context) == ""
    assert json.loads(context.model_dump_json())["items"] == []


def test_ranked_results_preserve_order_and_source_metadata() -> None:
    results = [
        make_result("second.md", score=0.80, chunk_index=2),
        make_result("first.md", score=0.99, chunk_index=0),
    ]

    context = build_rag_context(results)

    assert [item.relative_path for item in context.items] == [
        "second.md",
        "first.md",
    ]
    assert context.items[0].score == pytest.approx(0.80)
    assert context.items[0].chunk_id == "chunk-second-md"
    assert context.items[0].document_id == "doc-second-md"
    assert context.items[0].chunk_index == 2
    assert not Path(context.items[0].relative_path).is_absolute()
    assert context.source_result_count == 2
    assert context.excluded_by_limit_count == 0


def test_max_snippets_limit_is_enforced() -> None:
    context = build_rag_context(
        [
            make_result("a.md", snippet="alpha"),
            make_result("b.md", snippet="beta"),
            make_result("c.md", snippet="gamma"),
        ],
        limits=RagContextLimits(max_snippets=2, max_total_characters=100),
    )

    assert [item.relative_path for item in context.items] == ["a.md", "b.md"]
    assert context.excluded_by_limit_count == 1
    assert context.truncated is True


def test_character_budget_truncates_last_included_snippet() -> None:
    context = build_rag_context(
        [
            make_result("a.md", snippet="abcde"),
            make_result("b.md", snippet="fghij"),
            make_result("c.md", snippet="klmno"),
        ],
        limits=RagContextLimits(max_snippets=5, max_total_characters=8),
    )

    assert [item.snippet for item in context.items] == ["abcde", "fgh"]
    assert [item.truncated for item in context.items] == [False, True]
    assert context.total_snippet_characters == 8
    assert context.excluded_by_limit_count == 1
    assert context.truncated is True


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValidationError, match="max_snippets"):
        RagContextLimits(max_snippets=0)

    with pytest.raises(ValidationError, match="max_total_characters"):
        RagContextLimits(max_total_characters=0)

    with pytest.raises(ValidationError, match="integers"):
        RagContextLimits(max_snippets=True)


def test_context_items_reject_absolute_or_traversing_paths() -> None:
    with pytest.raises(ValidationError, match="allowed root"):
        RagContextItem(
            snippet="safe text",
            score=0.5,
            relative_path="/tmp/secret.md",
            chunk_id="chunk",
            document_id="doc",
            chunk_index=0,
        )

    with pytest.raises(ValidationError, match="allowed root"):
        RagContextItem(
            snippet="safe text",
            score=0.5,
            relative_path="../secret.md",
            chunk_id="chunk",
            document_id="doc",
            chunk_index=0,
        )


def test_context_contract_rejects_inconsistent_summary_fields() -> None:
    item = RagContextItem(
        snippet="safe text",
        score=0.5,
        relative_path="safe.md",
        chunk_id="chunk",
        document_id="doc",
        chunk_index=0,
    )

    with pytest.raises(ValidationError, match="total_snippet_characters"):
        RagContext(
            items=(item,),
            source_result_count=1,
            excluded_by_limit_count=0,
            total_snippet_characters=1,
            limits=RagContextLimits(),
            truncated=False,
        )


def test_context_json_serialization_is_deterministic() -> None:
    context = build_rag_context([make_result("alpha.md", snippet="alpha")])

    rendered = json.loads(context.model_dump_json())

    assert rendered == {
        "items": [
            {
                "snippet": "alpha",
                "score": 0.75,
                "relative_path": "alpha.md",
                "chunk_id": "chunk-alpha-md",
                "document_id": "doc-alpha-md",
                "chunk_index": 0,
                "truncated": False,
            },
        ],
        "source_result_count": 1,
        "excluded_by_limit_count": 0,
        "total_snippet_characters": 5,
        "limits": {
            "max_snippets": 5,
            "max_total_characters": 4000,
        },
        "truncated": False,
    }


def test_prompt_formatter_is_deterministic_and_source_aware() -> None:
    context = build_rag_context(
        [
            make_result("alpha.md", snippet="alpha note", score=0.9),
            make_result("beta.md", snippet="beta note", score=0.25),
        ],
    )

    assert format_rag_context_for_prompt(context) == (
        "Retrieved local note context:\n"
        "[1] source=alpha.md chunk=0 score=0.900000\n"
        "alpha note\n"
        "[2] source=beta.md chunk=0 score=0.250000\n"
        "beta note"
    )


def test_prompt_formatter_marks_truncated_items() -> None:
    context = build_rag_context(
        [make_result("alpha.md", snippet="abcdef")],
        limits=RagContextLimits(max_snippets=1, max_total_characters=3),
    )

    assert format_rag_context_for_prompt(context) == (
        "Retrieved local note context:\n"
        "[1] source=alpha.md chunk=0 score=0.750000 truncated\n"
        "abc"
    )


def make_result(
    relative_path: str,
    *,
    snippet: str = "source snippet",
    score: float = 0.75,
    chunk_index: int = 0,
) -> NoteSearchResult:
    safe_id = relative_path.replace("/", "-").replace(".", "-")
    return NoteSearchResult(
        chunk_id=f"chunk-{safe_id}",
        document_id=f"doc-{safe_id}",
        relative_path=relative_path,
        chunk_index=chunk_index,
        score=score,
        snippet=snippet,
    )
