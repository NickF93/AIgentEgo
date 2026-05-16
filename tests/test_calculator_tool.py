import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from aigentego.tools import (
    CalculatorTool,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


def execute_calculator(arguments: Mapping[str, Any]) -> ToolResult:
    return asyncio.run(CalculatorTool().execute(arguments, ToolContext()))


def assert_failed_result(
    result: ToolResult,
    *,
    code: str,
    message: str | None = None,
) -> None:
    assert result.tool_name == "calculator"
    assert result.success is False
    assert result.error is not None
    assert result.error.code == code
    assert result.error.tool_name == "calculator"
    if message is not None:
        assert result.error.message == message


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12 * 31", 372),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-5 + +2", -3),
        ("5 / 2", 2.5),
        ("5 % 2", 1),
        ("2 ** 3", 8),
        ("2 ** -2", 0.25),
        ("1.5 * 2", 3.0),
    ],
)
def test_calculator_evaluates_allowed_arithmetic(
    expression: str,
    expected: int | float,
) -> None:
    result = execute_calculator({"expression": expression})

    assert result == ToolResult(
        tool_name="calculator",
        success=True,
        result={"value": expected},
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "expression is required"),
        ({"expression": 12}, "expression must be a string"),
        ({"expression": "  "}, "expression must not be blank"),
    ],
)
def test_calculator_rejects_missing_or_invalid_expression_argument(
    arguments: Mapping[str, Any],
    message: str,
) -> None:
    result = execute_calculator(arguments)

    assert_failed_result(
        result,
        code="tool_validation_error",
        message=message,
    )


def test_calculator_rejects_invalid_syntax() -> None:
    result = execute_calculator({"expression": "1 +"})

    assert_failed_result(
        result,
        code="tool_validation_error",
        message="expression syntax is invalid",
    )


@pytest.mark.parametrize(
    "expression",
    [
        "name + 1",
        "__import__('os').system('id')",
        "(1).__class__",
        "[value for value in [1]]",
        "sum([1, 2])",
        "{1: 2}",
        "lambda: 1",
        "True",
    ],
)
def test_calculator_rejects_forbidden_ast_nodes(expression: str) -> None:
    result = execute_calculator({"expression": expression})

    assert_failed_result(
        result,
        code="tool_validation_error",
        message="expression contains unsupported syntax",
    )


@pytest.mark.parametrize("expression", ["1 / 0", "1 % 0"])
def test_calculator_returns_execution_error_for_arithmetic_failures(
    expression: str,
) -> None:
    result = execute_calculator({"expression": expression})

    assert_failed_result(
        result,
        code="tool_execution_error",
        message="calculator execution failed",
    )


def test_calculator_runs_through_tool_executor() -> None:
    executor = ToolExecutor(ToolRegistry([CalculatorTool()]))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                tool_name="calculator",
                arguments={"expression": "12 * 31"},
            ),
            ToolContext(request_id="req-123"),
        ),
    )

    assert result == ToolResult(
        tool_name="calculator",
        success=True,
        result={"value": 372},
    )
