"""Deterministic execution for explicit provider-neutral tool calls."""

from pydantic import ValidationError

from aigentego.tools.base import ToolCall, ToolContext, ToolResult
from aigentego.tools.errors import ToolError, ToolExecutionError, ToolValidationError
from aigentego.tools.registry import ToolRegistry


class ToolExecutor:
    """Execute explicit tool calls through a deterministic registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute a registered tool and normalize expected failure modes."""
        execution_context = context if context is not None else ToolContext()

        try:
            tool = self._registry.get(call.tool_name)
        except ToolError as error:
            return self._failed_result(call.tool_name, error)

        try:
            return await tool.execute(call.arguments.copy(), execution_context)
        except ToolError as error:
            return self._failed_result(call.tool_name, error)
        except ValidationError:
            validation_error = ToolValidationError(
                "tool arguments are invalid",
                tool_name=call.tool_name,
            )
            return self._failed_result(call.tool_name, validation_error)
        except Exception:
            execution_error = ToolExecutionError(
                "tool execution failed",
                tool_name=call.tool_name,
            )
            return self._failed_result(call.tool_name, execution_error)

    @staticmethod
    def _failed_result(tool_name: str, error: ToolError) -> ToolResult:
        detail = error.to_detail()
        if detail.tool_name is None:
            detail = detail.model_copy(update={"tool_name": tool_name})

        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=detail,
        )
