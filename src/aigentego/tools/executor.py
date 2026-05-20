"""Deterministic execution for explicit provider-neutral tool calls."""

import logging
from time import perf_counter

from pydantic import ValidationError

from aigentego.tools.base import ToolCall, ToolContext, ToolResult
from aigentego.tools.errors import ToolError, ToolExecutionError, ToolValidationError
from aigentego.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        started_at = perf_counter()

        try:
            tool = self._registry.get(call.tool_name)
        except ToolError as error:
            result = self._failed_result(call.tool_name, error)
        else:
            try:
                result = await tool.execute(call.arguments.copy(), execution_context)
            except ToolError as error:
                result = self._failed_result(call.tool_name, error)
            except ValidationError:
                validation_error = ToolValidationError(
                    "tool arguments are invalid",
                    tool_name=call.tool_name,
                )
                result = self._failed_result(call.tool_name, validation_error)
            except Exception:
                execution_error = ToolExecutionError(
                    "tool execution failed",
                    tool_name=call.tool_name,
                )
                result = self._failed_result(call.tool_name, execution_error)

        self._log_result(result, execution_context, started_at)
        return result

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

    @staticmethod
    def _log_result(
        result: ToolResult,
        context: ToolContext,
        started_at: float,
    ) -> None:
        duration_ms = (perf_counter() - started_at) * 1000
        error_code = result.error.code if result.error is not None else None
        logger.log(
            logging.INFO if result.success else logging.WARNING,
            "tool_execution tool_name=%s success=%s duration_ms=%.2f error_code=%s",
            result.tool_name,
            str(result.success).lower(),
            duration_ms,
            error_code,
            extra={"request_id": context.request_id or "-"},
        )
