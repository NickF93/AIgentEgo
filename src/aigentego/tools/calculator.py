"""Safe deterministic arithmetic calculator tool."""

import ast
import operator
from collections.abc import Callable, Mapping
from typing import Any

from aigentego.tools.base import ToolContext, ToolDefinition, ToolResult
from aigentego.tools.errors import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
)

Number = int | float

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool:
    """Evaluate a constrained arithmetic expression."""

    @property
    def definition(self) -> ToolDefinition:
        """Return provider-neutral calculator metadata."""
        return ToolDefinition(
            name="calculator",
            description="Evaluate a safe arithmetic expression.",
            parameters_schema={
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate.",
                    },
                },
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Evaluate the requested arithmetic expression."""
        try:
            expression = self._get_expression(arguments)
            value = self._evaluate_expression(expression)
        except ToolError as error:
            return self._failed_result(error)
        except ArithmeticError:
            return self._failed_result(
                ToolExecutionError(
                    "calculator execution failed",
                    tool_name=self.definition.name,
                ),
            )

        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={"value": value},
        )

    def _get_expression(self, arguments: Mapping[str, Any]) -> str:
        if "expression" not in arguments:
            raise ToolValidationError(
                "expression is required",
                tool_name=self.definition.name,
            )

        expression = arguments["expression"]
        if not isinstance(expression, str):
            raise ToolValidationError(
                "expression must be a string",
                tool_name=self.definition.name,
            )
        if expression.strip() == "":
            raise ToolValidationError(
                "expression must not be blank",
                tool_name=self.definition.name,
            )

        return expression

    def _evaluate_expression(self, expression: str) -> Number:
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ToolValidationError(
                "expression syntax is invalid",
                tool_name=self.definition.name,
            ) from error

        return self._evaluate_node(parsed.body)

    def _evaluate_node(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            return self._evaluate_constant(node.value)

        if isinstance(node, ast.BinOp):
            binary_operator = _BINARY_OPERATORS.get(type(node.op))
            if binary_operator is None:
                raise self._unsupported_syntax_error()

            result = binary_operator(
                self._evaluate_node(node.left),
                self._evaluate_node(node.right),
            )
            return self._ensure_number(result)

        if isinstance(node, ast.UnaryOp):
            unary_operator = _UNARY_OPERATORS.get(type(node.op))
            if unary_operator is None:
                raise self._unsupported_syntax_error()

            result = unary_operator(self._evaluate_node(node.operand))
            return self._ensure_number(result)

        raise self._unsupported_syntax_error()

    def _evaluate_constant(self, value: object) -> Number:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._unsupported_syntax_error()
        return value

    def _ensure_number(self, value: object) -> Number:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolExecutionError(
                "calculator execution failed",
                tool_name=self.definition.name,
            )
        return value

    def _unsupported_syntax_error(self) -> ToolValidationError:
        return ToolValidationError(
            "expression contains unsupported syntax",
            tool_name=self.definition.name,
        )

    def _failed_result(self, error: ToolError) -> ToolResult:
        return ToolResult(
            tool_name=self.definition.name,
            success=False,
            error=error.to_detail(),
        )
