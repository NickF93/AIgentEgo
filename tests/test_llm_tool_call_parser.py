import json
from collections.abc import Mapping
from typing import Any

import pytest

from aigentego.llm import (
    ToolCallParseError,
    ToolCallValidationError,
    parse_structured_tool_calls,
)
from aigentego.tools import (
    CalculatorTool,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)


class NeverExecuteTool:
    def __init__(self) -> None:
        self.executed = False

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="never_execute",
            description="A parser test tool that must not execute.",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        self.executed = True
        raise AssertionError("parser must not execute tools")


def calculator_registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool()])


def test_valid_empty_tool_calls_parse_to_empty_list() -> None:
    assert parse_structured_tool_calls(
        '{"tool_calls": []}',
        calculator_registry(),
    ) == []


def test_valid_single_known_tool_call_parses_to_tool_call() -> None:
    tool_calls = parse_structured_tool_calls(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "12 * 31"},
                    },
                ],
            },
        ),
        calculator_registry(),
    )

    assert tool_calls == [
        ToolCall(tool_name="calculator", arguments={"expression": "12 * 31"}),
    ]


def test_valid_multiple_known_tool_calls_preserve_order() -> None:
    tool_calls = parse_structured_tool_calls(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "1 + 1"},
                    },
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "2 + 2"},
                    },
                ],
            },
        ),
        calculator_registry(),
    )

    assert [tool_call.arguments["expression"] for tool_call in tool_calls] == [
        "1 + 1",
        "2 + 2",
    ]


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ToolCallParseError):
        parse_structured_tool_calls(
            '{"tool_calls": [',
            calculator_registry(),
        )


@pytest.mark.parametrize("content", ["[]", '"not an object"', "1", "true"])
def test_top_level_non_object_json_is_rejected(content: str) -> None:
    with pytest.raises(ToolCallValidationError) as error_info:
        parse_structured_tool_calls(content, calculator_registry())

    assert "JSON object" in str(error_info.value)


def test_missing_tool_calls_is_rejected() -> None:
    with pytest.raises(ToolCallValidationError):
        parse_structured_tool_calls("{}", calculator_registry())


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ToolCallValidationError):
        parse_structured_tool_calls(
            '{"tool_calls": [], "unexpected": "field"}',
            calculator_registry(),
        )


def test_blank_tool_names_are_rejected() -> None:
    with pytest.raises(ToolCallValidationError):
        parse_structured_tool_calls(
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "  ",
                            "arguments": {},
                        },
                    ],
                },
            ),
            calculator_registry(),
        )


def test_non_dict_arguments_are_rejected() -> None:
    with pytest.raises(ToolCallValidationError):
        parse_structured_tool_calls(
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "calculator",
                            "arguments": ["not", "an", "object"],
                        },
                    ],
                },
            ),
            calculator_registry(),
        )


def test_unknown_tool_names_are_rejected_using_registry() -> None:
    with pytest.raises(ToolCallValidationError) as error_info:
        parse_structured_tool_calls(
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "missing_tool",
                            "arguments": {},
                        },
                    ],
                },
            ),
            calculator_registry(),
        )

    assert error_info.value.tool_name == "missing_tool"
    assert "requested tool is not registered" in str(error_info.value)


def test_parser_errors_do_not_include_raw_model_content() -> None:
    raw_content = (
        '{"tool_calls": [{"tool_name": "missing_tool", '
        '"arguments": {"secret": "do-not-log"}}]}'
    )

    with pytest.raises(ToolCallValidationError) as error_info:
        parse_structured_tool_calls(raw_content, calculator_registry())

    assert raw_content not in str(error_info.value)
    assert "do-not-log" not in str(error_info.value)


def test_parser_does_not_execute_tools() -> None:
    tool = NeverExecuteTool()
    tool_calls = parse_structured_tool_calls(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "never_execute",
                        "arguments": {"value": "kept for later execution"},
                    },
                ],
            },
        ),
        ToolRegistry([tool]),
    )

    assert tool_calls == [
        ToolCall(
            tool_name="never_execute",
            arguments={"value": "kept for later execution"},
        ),
    ]
    assert tool.executed is False
