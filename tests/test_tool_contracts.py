import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from aigentego.tools import (
    Tool,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolError,
    ToolErrorDetail,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
)


class EchoTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Return the supplied arguments.",
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
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={
                "arguments": dict(arguments),
                "request_id": context.request_id,
            },
        )


def test_tool_definition_accepts_provider_neutral_metadata() -> None:
    definition = ToolDefinition(
        name="calculator",
        description="Evaluate a safe arithmetic expression.",
        parameters_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )

    assert definition.name == "calculator"
    assert definition.parameters_schema["required"] == ["expression"]


def test_tool_definition_rejects_unstable_names() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(name="Calculator Tool", description="Invalid name.")


def test_tool_call_defaults_to_empty_arguments() -> None:
    call = ToolCall(tool_name="calculator")

    assert call.tool_name == "calculator"
    assert call.arguments == {}


def test_tool_context_carries_request_id_and_metadata() -> None:
    context = ToolContext(request_id="req-123", metadata={"source": "test"})

    assert context.request_id == "req-123"
    assert context.metadata == {"source": "test"}


def test_tool_result_enforces_success_error_consistency() -> None:
    result = ToolResult(tool_name="calculator", success=True, result={"value": 372})

    assert result.error is None

    with pytest.raises(ValidationError):
        ToolResult(
            tool_name="calculator",
            success=True,
            error=ToolErrorDetail(code="tool_error", message="unexpected"),
        )

    with pytest.raises(ValidationError):
        ToolResult(tool_name="calculator", success=False)


def test_failed_tool_result_accepts_normalized_error_detail() -> None:
    error = ToolValidationError(
        "expression is required",
        tool_name="calculator",
    )

    result = ToolResult(
        tool_name="calculator",
        success=False,
        error=error.to_detail(),
    )

    assert result.error == ToolErrorDetail(
        code="tool_validation_error",
        message="expression is required",
        tool_name="calculator",
    )


def test_tool_protocol_matches_minimal_async_interface() -> None:
    tool = EchoTool()
    context = ToolContext(request_id="req-123")

    assert isinstance(tool, Tool)
    result = asyncio.run(tool.execute({"value": "hello"}, context))

    assert result.success is True
    assert result.result == {
        "arguments": {"value": "hello"},
        "request_id": "req-123",
    }


def test_tool_errors_preserve_context_and_normalized_details() -> None:
    error = ToolNotFoundError("calculator")

    assert isinstance(error, ToolError)
    assert str(error) == "calculator: tool is not registered"
    assert error.to_detail() == ToolErrorDetail(
        code="tool_not_found",
        message="tool is not registered",
        tool_name="calculator",
    )


def test_tool_error_subclasses_use_stable_codes() -> None:
    validation_error = ToolValidationError("invalid input", tool_name="calculator")
    execution_error = ToolExecutionError("execution failed", tool_name="calculator")

    assert validation_error.to_detail().code == "tool_validation_error"
    assert execution_error.to_detail().code == "tool_execution_error"
