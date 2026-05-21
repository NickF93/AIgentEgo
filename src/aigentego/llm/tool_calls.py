"""Provider-neutral structured tool-call output contracts."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aigentego.tools import ToolCall


class LlmToolCall(BaseModel):
    """A model-requested tool call intent."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        """Reject blank tool names before deterministic validation."""
        if not value.strip():
            raise ValueError("tool_name must not be blank")
        return value

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments(cls, value: object) -> object:
        """Require JSON-object arguments for model-requested tool calls."""
        if not isinstance(value, dict):
            raise ValueError("arguments must be a JSON object")
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("arguments must be JSON-serializable") from exc
        return value

    def to_tool_call(self) -> ToolCall:
        """Convert model intent into a deterministic tool call."""
        arguments = json.loads(json.dumps(self.arguments))
        return ToolCall(tool_name=self.tool_name, arguments=arguments)


class StructuredToolCallOutput(BaseModel):
    """Structured model output containing requested tool calls."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: list[LlmToolCall]

    def to_tool_calls(self) -> list[ToolCall]:
        """Convert all model-requested calls into deterministic tool calls."""
        return [tool_call.to_tool_call() for tool_call in self.tool_calls]
