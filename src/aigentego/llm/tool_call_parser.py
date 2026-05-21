"""Parse provider-neutral structured tool-call output."""

import json
from typing import Any

from pydantic import ValidationError

from aigentego.llm.errors import ToolCallParseError, ToolCallValidationError
from aigentego.llm.tool_calls import StructuredToolCallOutput
from aigentego.tools import ToolCall, ToolNotFoundError, ToolRegistry


def parse_structured_tool_calls(
    content: str,
    registry: ToolRegistry,
) -> list[ToolCall]:
    """Parse raw model content into validated deterministic tool calls."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ToolCallParseError() from error

    if not isinstance(parsed, dict):
        raise ToolCallValidationError(
            "structured tool-call output must be a JSON object",
        )

    output = _validate_structured_output(parsed)
    tool_calls = output.to_tool_calls()
    _validate_registered_tools(tool_calls, registry)
    return tool_calls


def _validate_structured_output(data: dict[str, Any]) -> StructuredToolCallOutput:
    try:
        return StructuredToolCallOutput.model_validate(data)
    except ValidationError as error:
        raise ToolCallValidationError() from error


def _validate_registered_tools(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
) -> None:
    for tool_call in tool_calls:
        try:
            registry.get(tool_call.tool_name)
        except ToolNotFoundError as error:
            raise ToolCallValidationError(
                "requested tool is not registered",
                tool_name=tool_call.tool_name,
            ) from error
