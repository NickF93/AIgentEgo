import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from fastapi.testclient import TestClient

from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.tools import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionError,
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
            description="Return a static result.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={"value": "ok"},
        )


class ValidationFailureTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="validation_failure",
            description="Raise a validation failure.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        raise ToolValidationError(
            "invalid input",
            tool_name=self.definition.name,
        )


class ExecutionFailureTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="execution_failure",
            description="Raise an execution failure.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        raise ToolExecutionError(
            "execution failed",
            tool_name=self.definition.name,
        )


def test_tool_execution_logs_success_without_payloads(caplog) -> None:
    executor = ToolExecutor(ToolRegistry([EchoTool()]))

    with caplog.at_level(logging.INFO, logger="aigentego.tools.executor"):
        result = asyncio.run(
            executor.execute(
                ToolCall(
                    tool_name="echo",
                    arguments={"secret": "do not log"},
                ),
                ToolContext(request_id="tool-request-123"),
            ),
        )

    assert result.success is True
    records = _tool_execution_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    assert record.request_id == "tool-request-123"
    message = record.getMessage()
    assert "tool_name=echo" in message
    assert "success=true" in message
    assert "duration_ms=" in message
    assert "error_code=None" in message
    assert "do not log" not in message
    assert "ok" not in message


def test_tool_execution_logs_unknown_tool_with_normalized_error_code(caplog) -> None:
    executor = ToolExecutor(ToolRegistry())

    with caplog.at_level(logging.WARNING, logger="aigentego.tools.executor"):
        result = asyncio.run(
            executor.execute(
                ToolCall(
                    tool_name="missing",
                    arguments={"secret": "do not log"},
                ),
                ToolContext(request_id="missing-request-123"),
            ),
        )

    assert result.success is False
    records = _tool_execution_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.request_id == "missing-request-123"
    message = record.getMessage()
    assert "tool_name=missing" in message
    assert "success=false" in message
    assert "duration_ms=" in message
    assert "error_code=tool_not_found" in message
    assert "do not log" not in message


def test_tool_execution_logs_validation_failure_with_normalized_error_code(
    caplog,
) -> None:
    executor = ToolExecutor(ToolRegistry([ValidationFailureTool()]))

    with caplog.at_level(logging.WARNING, logger="aigentego.tools.executor"):
        result = asyncio.run(
            executor.execute(
                ToolCall(
                    tool_name="validation_failure",
                    arguments={"secret": "do not log"},
                ),
                ToolContext(request_id="validation-request-123"),
            ),
        )

    assert result.success is False
    records = _tool_execution_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.request_id == "validation-request-123"
    message = record.getMessage()
    assert "tool_name=validation_failure" in message
    assert "success=false" in message
    assert "duration_ms=" in message
    assert "error_code=tool_validation_error" in message
    assert "do not log" not in message
    assert "invalid input" not in message


def test_tool_execution_logs_execution_failure_with_normalized_error_code(
    caplog,
) -> None:
    executor = ToolExecutor(ToolRegistry([ExecutionFailureTool()]))

    with caplog.at_level(logging.WARNING, logger="aigentego.tools.executor"):
        result = asyncio.run(
            executor.execute(
                ToolCall(
                    tool_name="execution_failure",
                    arguments={"secret": "do not log"},
                ),
                ToolContext(request_id="execution-request-123"),
            ),
        )

    assert result.success is False
    records = _tool_execution_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.request_id == "execution-request-123"
    message = record.getMessage()
    assert "tool_name=execution_failure" in message
    assert "success=false" in message
    assert "duration_ms=" in message
    assert "error_code=tool_execution_error" in message
    assert "do not log" not in message
    assert "execution failed" not in message


def test_tool_execute_api_response_request_id_matches_header() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/tools/execute",
        headers={REQUEST_ID_HEADER: "api-tool-request-123"},
        json={
            "tool_name": "calculator",
            "arguments": {"expression": "2 + 2"},
        },
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "api-tool-request-123"
    assert response.json()["request_id"] == "api-tool-request-123"


def _tool_execution_records(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "aigentego.tools.executor"
        and "tool_execution" in record.getMessage()
    ]
