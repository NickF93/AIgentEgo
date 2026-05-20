import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from aigentego.tools import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
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


class ValidationErrorTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="requires_value",
            description="Require a value argument.",
            parameters_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        raise ToolValidationError(
            "value is required",
            tool_name=self.definition.name,
        )


class PydanticValidationTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="typed_value",
            description="Validate a value argument through Pydantic.",
            parameters_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        class Arguments(BaseModel):
            value: str

        Arguments.model_validate(dict(arguments))
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={"validated": True},
        )


class FailingTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fails",
            description="Raise an unexpected exception.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        raise RuntimeError("internal failure details")


class FailedResultTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failed_result",
            description="Return a normalized failed result.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        error = ToolValidationError(
            "value is invalid",
            tool_name=self.definition.name,
        )
        return ToolResult(
            tool_name=self.definition.name,
            success=False,
            error=error.to_detail(),
        )


def test_executor_invokes_registered_tool_for_explicit_call() -> None:
    executor = ToolExecutor(ToolRegistry([EchoTool()]))
    context = ToolContext(request_id="req-123")

    result = asyncio.run(
        executor.execute(
            ToolCall(tool_name="echo", arguments={"value": "hello"}),
            context,
        ),
    )

    assert result == ToolResult(
        tool_name="echo",
        success=True,
        result={
            "arguments": {"value": "hello"},
            "request_id": "req-123",
        },
    )


def test_executor_returns_normalized_not_found_result_for_unknown_tool() -> None:
    executor = ToolExecutor(ToolRegistry())

    result = asyncio.run(executor.execute(ToolCall(tool_name="missing")))

    assert result.success is False
    assert result.tool_name == "missing"
    assert result.error is not None
    assert result.error.code == "tool_not_found"
    assert result.error.message == "tool is not registered"
    assert result.error.tool_name == "missing"


def test_executor_normalizes_tool_validation_errors() -> None:
    executor = ToolExecutor(ToolRegistry([ValidationErrorTool()]))

    result = asyncio.run(executor.execute(ToolCall(tool_name="requires_value")))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_validation_error"
    assert result.error.message == "value is required"
    assert result.error.tool_name == "requires_value"


def test_executor_normalizes_pydantic_validation_errors() -> None:
    executor = ToolExecutor(ToolRegistry([PydanticValidationTool()]))

    result = asyncio.run(executor.execute(ToolCall(tool_name="typed_value")))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_validation_error"
    assert result.error.message == "tool arguments are invalid"
    assert result.error.tool_name == "typed_value"


def test_executor_normalizes_unexpected_tool_failures_without_leaking_details() -> None:
    executor = ToolExecutor(ToolRegistry([FailingTool()]))

    result = asyncio.run(executor.execute(ToolCall(tool_name="fails")))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert result.error.message == "tool execution failed"
    assert result.error.tool_name == "fails"


def test_executor_preserves_normalized_failed_tool_results() -> None:
    executor = ToolExecutor(ToolRegistry([FailedResultTool()]))

    result = asyncio.run(executor.execute(ToolCall(tool_name="failed_result")))

    assert result == ToolResult(
        tool_name="failed_result",
        success=False,
        error=ToolValidationError(
            "value is invalid",
            tool_name="failed_result",
        ).to_detail(),
    )
