"""Provider-neutral contracts for bounded agent runs."""

import json
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aigentego.llm.tool_call_repair import ToolCallRepairDecision
from aigentego.tools import ToolCall, ToolResult


class AgentRunStatus(StrEnum):
    """Inspectable lifecycle states for one bounded agent run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentStepType(StrEnum):
    """Kinds of steps that can appear in a bounded agent run."""

    MODEL = "model"
    TOOL = "tool"
    FINAL = "final"
    ERROR = "error"


class AgentStepStatus(StrEnum):
    """Inspectable lifecycle states for one agent step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentStep(BaseModel):
    """Inspectable state for one ordered step in a bounded agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    step_type: AgentStepType
    status: AgentStepStatus
    model_summary: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    observation: dict[str, Any] | None = None
    repair_decision: ToolCallRepairDecision | None = None
    error_detail: str | None = None

    @field_validator("model_summary", "error_detail")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Reject blank optional text fields."""
        return _validate_optional_text(value)

    @field_validator("observation", mode="before")
    @classmethod
    def validate_observation(cls, value: object) -> object:
        """Require JSON-object observations when present."""
        return _validate_optional_json_object(value, field_name="observation")

    @model_validator(mode="after")
    def validate_status_detail(self) -> Self:
        """Keep terminal step statuses consistent with failure details."""
        if self.status is AgentStepStatus.FAILED:
            if self.error_detail is None and self.repair_decision is None:
                raise ValueError(
                    "failed agent steps must include error or repair details",
                )
        if self.status in {AgentStepStatus.SUCCEEDED, AgentStepStatus.SKIPPED}:
            if self.error_detail is not None:
                raise ValueError(
                    "successful or skipped agent steps must not include errors",
                )
        return self


class AgentRun(BaseModel):
    """Inspectable provider-neutral state for one bounded agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    request_id: str | None = None
    user_message: str = Field(min_length=1)
    status: AgentRunStatus = AgentRunStatus.PENDING
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    stop_reason: str | None = None
    repair_decision: ToolCallRepairDecision | None = None
    error_detail: str | None = None

    @field_validator(
        "run_id",
        "request_id",
        "user_message",
        "final_answer",
        "stop_reason",
        "error_detail",
    )
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional details."""
        return _validate_optional_text(value)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        """Keep run status and terminal details internally consistent."""
        if self.status is AgentRunStatus.SUCCEEDED:
            if self.final_answer is None:
                raise ValueError("succeeded agent runs must include final_answer")
            if self.error_detail is not None or self.repair_decision is not None:
                raise ValueError("succeeded agent runs must not include errors")

        if self.status is AgentRunStatus.FAILED:
            if self.error_detail is None and self.repair_decision is None:
                raise ValueError(
                    "failed agent runs must include error or repair details",
                )
            if self.final_answer is not None:
                raise ValueError("failed agent runs must not include final_answer")

        if self.status is AgentRunStatus.STOPPED:
            if self.stop_reason is None:
                raise ValueError("stopped agent runs must include stop_reason")
            if self.final_answer is not None:
                raise ValueError("stopped agent runs must not include final_answer")

        if self.status in {AgentRunStatus.PENDING, AgentRunStatus.RUNNING}:
            if (
                self.final_answer is not None
                or self.stop_reason is not None
                or self.error_detail is not None
                or self.repair_decision is not None
            ):
                raise ValueError(
                    "non-terminal agent runs must not include terminal details",
                )

        return self


def _validate_optional_text(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("text fields must not be blank")
    return value


def _validate_optional_json_object(
    value: object,
    *,
    field_name: str,
) -> object:
    if value is None:
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc
    return value
