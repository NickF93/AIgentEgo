"""Provider-neutral deterministic tool contracts."""

from collections.abc import Mapping
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aigentego.tools.errors import ToolErrorDetail


class ToolDefinition(BaseModel):
    """Metadata and input schema for a deterministic tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1)
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """An explicit deterministic tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolContext(BaseModel):
    """Execution context passed to deterministic tools."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Normalized deterministic tool execution result."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    success: bool
    result: dict[str, Any] | None = None
    error: ToolErrorDetail | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> Self:
        """Keep success and error fields consistent."""
        if self.success and self.error is not None:
            raise ValueError("successful tool results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed tool results must include an error")
        return self


@runtime_checkable
class Tool(Protocol):
    """Minimal asynchronous deterministic tool interface."""

    @property
    def definition(self) -> ToolDefinition:
        """Return provider-neutral tool metadata."""

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a deterministic tool call."""
