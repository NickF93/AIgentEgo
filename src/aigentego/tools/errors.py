"""Deterministic tool error types."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ToolErrorDetail(BaseModel):
    """Structured, provider-neutral tool error details."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    tool_name: str | None = None


class ToolError(Exception):
    """Base error for deterministic tool failures."""

    error_code: ClassVar[str] = "tool_error"

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        self.message = message
        self.tool_name = tool_name
        super().__init__(self._format_message())

    def to_detail(self) -> ToolErrorDetail:
        """Return normalized error details for API and executor layers."""
        return ToolErrorDetail(
            code=self.error_code,
            message=self.message,
            tool_name=self.tool_name,
        )

    def _format_message(self) -> str:
        if self.tool_name is None:
            return self.message
        return f"{self.tool_name}: {self.message}"


class ToolDefinitionError(ToolError):
    """A tool definition is invalid."""

    error_code: ClassVar[str] = "tool_definition_error"


class ToolNotFoundError(ToolError):
    """The requested tool is not registered."""

    error_code: ClassVar[str] = "tool_not_found"

    def __init__(self, tool_name: str) -> None:
        super().__init__("tool is not registered", tool_name=tool_name)


class ToolValidationError(ToolError):
    """A tool call failed input validation."""

    error_code: ClassVar[str] = "tool_validation_error"


class ToolExecutionError(ToolError):
    """A tool failed while executing a validated call."""

    error_code: ClassVar[str] = "tool_execution_error"
