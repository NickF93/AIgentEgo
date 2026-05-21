import json

import pytest
from pydantic import ValidationError

from aigentego.llm import LlmToolCall, StructuredToolCallOutput
from aigentego.tools import ToolCall


def test_empty_tool_calls_are_accepted() -> None:
    output = StructuredToolCallOutput(tool_calls=[])

    assert output.tool_calls == []
    assert output.to_tool_calls() == []


def test_single_tool_call_is_accepted() -> None:
    output = StructuredToolCallOutput(
        tool_calls=[
            LlmToolCall(
                tool_name="calculator",
                arguments={"expression": "12 * 31"},
            ),
        ],
    )

    assert output.tool_calls[0].tool_name == "calculator"
    assert output.tool_calls[0].arguments == {"expression": "12 * 31"}


def test_multiple_tool_calls_are_accepted() -> None:
    output = StructuredToolCallOutput(
        tool_calls=[
            LlmToolCall(tool_name="calculator", arguments={"expression": "1 + 1"}),
            LlmToolCall(tool_name="calculator", arguments={"expression": "2 + 2"}),
        ],
    )

    assert [tool_call.arguments["expression"] for tool_call in output.tool_calls] == [
        "1 + 1",
        "2 + 2",
    ]


def test_missing_arguments_default_to_empty_object() -> None:
    tool_call = LlmToolCall(tool_name="calculator")

    assert tool_call.arguments == {}
    assert tool_call.to_tool_call() == ToolCall(tool_name="calculator")


def test_extra_top_level_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StructuredToolCallOutput.model_validate(
            {"tool_calls": [], "unexpected": "field"},
        )


def test_extra_per_call_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StructuredToolCallOutput.model_validate(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "1 + 1"},
                        "unexpected": "field",
                    },
                ],
            },
        )


def test_blank_tool_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LlmToolCall(tool_name="   ", arguments={})


def test_non_dict_arguments_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LlmToolCall.model_validate(
            {"tool_name": "calculator", "arguments": ["not", "an", "object"]},
        )


def test_conversion_to_tool_calls_preserves_name_and_arguments() -> None:
    output = StructuredToolCallOutput(
        tool_calls=[
            LlmToolCall(
                tool_name="calculator",
                arguments={"expression": "12 * 31"},
            ),
        ],
    )

    tool_calls = output.to_tool_calls()

    assert tool_calls == [
        ToolCall(tool_name="calculator", arguments={"expression": "12 * 31"}),
    ]


def test_conversion_does_not_mutate_source_output() -> None:
    output = StructuredToolCallOutput(
        tool_calls=[
            LlmToolCall(
                tool_name="calculator",
                arguments={"nested": {"value": 1}},
            ),
        ],
    )

    tool_calls = output.to_tool_calls()
    tool_calls[0].arguments["nested"]["value"] = 2

    assert output.tool_calls[0].arguments == {"nested": {"value": 1}}


def test_structured_tool_call_output_is_json_serializable() -> None:
    output = StructuredToolCallOutput(
        tool_calls=[
            LlmToolCall(
                tool_name="calculator",
                arguments={"expression": "12 * 31"},
            ),
        ],
    )

    assert json.loads(output.model_dump_json()) == output.model_dump(mode="json")
