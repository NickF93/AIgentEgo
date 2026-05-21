"""Bounded repair decisions for invalid structured tool-call output."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from aigentego.llm.errors import (
    ToolCallFailureCode,
    ToolCallParseError,
    ToolCallValidationError,
)

ToolCallFailureError = ToolCallParseError | ToolCallValidationError


class ToolCallFailureCategory(StrEnum):
    """High-level failure categories for structured tool-call parsing."""

    PARSE = "parse"
    VALIDATION = "validation"


class ToolCallRepairAction(StrEnum):
    """Decision outcomes for invalid structured tool-call output."""

    RETRY = "retry"
    SAFE_FAILURE = "safe_failure"


class ToolCallRepairDecisionReason(StrEnum):
    """Deterministic reasons for repair decisions."""

    ATTEMPTS_REMAIN = "attempts_remain"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    REPAIR_DISABLED = "repair_disabled"


class ToolCallParsingFailure(BaseModel):
    """Safe, normalized representation of a structured tool-call failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ToolCallFailureCode
    category: ToolCallFailureCategory
    message: str = Field(min_length=1)


class ToolCallRepairDecision(BaseModel):
    """Bounded decision for retrying or stopping after invalid tool-call output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ToolCallRepairAction
    reason: ToolCallRepairDecisionReason
    failure: ToolCallParsingFailure
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)

    @property
    def should_retry(self) -> bool:
        """Return whether a caller may request a repair attempt."""
        return self.action is ToolCallRepairAction.RETRY

    @property
    def is_safe_failure(self) -> bool:
        """Return whether parsing should stop without tool execution."""
        return self.action is ToolCallRepairAction.SAFE_FAILURE


class ToolCallRetryPolicy(BaseModel):
    """Bounded repair policy for structured tool-call parse failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1)
    allow_repair: bool = True

    def decide(
        self,
        error: ToolCallFailureError,
        *,
        attempt: int,
    ) -> ToolCallRepairDecision:
        """Return the retry or safe-failure decision for a failed attempt."""
        if attempt < 1:
            raise ValueError("attempt must be positive")

        failure = classify_tool_call_failure(error)
        if not self.allow_repair:
            return ToolCallRepairDecision(
                action=ToolCallRepairAction.SAFE_FAILURE,
                reason=ToolCallRepairDecisionReason.REPAIR_DISABLED,
                failure=failure,
                attempt=attempt,
                max_attempts=self.max_attempts,
            )
        if attempt < self.max_attempts:
            return ToolCallRepairDecision(
                action=ToolCallRepairAction.RETRY,
                reason=ToolCallRepairDecisionReason.ATTEMPTS_REMAIN,
                failure=failure,
                attempt=attempt,
                max_attempts=self.max_attempts,
            )
        return ToolCallRepairDecision(
            action=ToolCallRepairAction.SAFE_FAILURE,
            reason=ToolCallRepairDecisionReason.MAX_ATTEMPTS_REACHED,
            failure=failure,
            attempt=attempt,
            max_attempts=self.max_attempts,
        )


def classify_tool_call_failure(
    error: ToolCallFailureError,
) -> ToolCallParsingFailure:
    """Classify parser errors into safe deterministic failure details."""
    return ToolCallParsingFailure(
        code=error.failure_code,
        category=_failure_category(error.failure_code),
        message=_SAFE_FAILURE_MESSAGES[error.failure_code],
    )


def decide_tool_call_repair(
    error: ToolCallFailureError,
    *,
    attempt: int,
    max_attempts: int,
    allow_repair: bool = True,
) -> ToolCallRepairDecision:
    """Apply a bounded retry policy to a structured tool-call failure."""
    policy = ToolCallRetryPolicy(
        max_attempts=max_attempts,
        allow_repair=allow_repair,
    )
    return policy.decide(error, attempt=attempt)


def _failure_category(code: ToolCallFailureCode) -> ToolCallFailureCategory:
    if code is ToolCallFailureCode.INVALID_JSON:
        return ToolCallFailureCategory.PARSE
    return ToolCallFailureCategory.VALIDATION


_SAFE_FAILURE_MESSAGES: dict[ToolCallFailureCode, str] = {
    ToolCallFailureCode.INVALID_JSON: "invalid structured tool-call JSON",
    ToolCallFailureCode.NON_OBJECT_JSON: (
        "structured tool-call output must be a JSON object"
    ),
    ToolCallFailureCode.SCHEMA_INVALID_JSON: "invalid structured tool-call output",
    ToolCallFailureCode.UNKNOWN_TOOL: "requested tool is not registered",
}
