import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from aigentego.llm import (
    ToolCallFailureCategory,
    ToolCallFailureCode,
    ToolCallParseError,
    ToolCallParsingFailure,
    ToolCallRepairAction,
    ToolCallRepairDecisionReason,
    ToolCallRetryPolicy,
    ToolCallValidationError,
    classify_tool_call_failure,
    decide_tool_call_repair,
    parse_structured_tool_calls,
)
from aigentego.tools import (
    CalculatorTool,
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)


class NeverExecuteTool:
    def __init__(self) -> None:
        self.executed = False

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="never_execute",
            description="A repair test tool that must not execute.",
            parameters_schema={"type": "object"},
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        self.executed = True
        raise AssertionError("repair decisions must not execute tools")


def calculator_registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool()])


def parse_failure(content: str) -> ToolCallParseError | ToolCallValidationError:
    with pytest.raises((ToolCallParseError, ToolCallValidationError)) as error_info:
        parse_structured_tool_calls(content, calculator_registry())
    return error_info.value


def test_invalid_json_maps_to_parse_failure_code() -> None:
    failure = classify_tool_call_failure(parse_failure('{"tool_calls": ['))

    assert failure == ToolCallParsingFailure(
        code=ToolCallFailureCode.INVALID_JSON,
        category=ToolCallFailureCategory.PARSE,
        message="invalid structured tool-call JSON",
    )


@pytest.mark.parametrize("content", ["[]", '"not an object"', "1", "true"])
def test_non_object_json_maps_to_validation_failure_code(content: str) -> None:
    failure = classify_tool_call_failure(parse_failure(content))

    assert failure.code is ToolCallFailureCode.NON_OBJECT_JSON
    assert failure.category is ToolCallFailureCategory.VALIDATION
    assert failure.message == "structured tool-call output must be a JSON object"


def test_schema_invalid_json_maps_to_validation_failure_code() -> None:
    failure = classify_tool_call_failure(parse_failure("{}"))

    assert failure.code is ToolCallFailureCode.SCHEMA_INVALID_JSON
    assert failure.category is ToolCallFailureCategory.VALIDATION
    assert failure.message == "invalid structured tool-call output"


def test_unknown_tool_maps_to_validation_failure_code() -> None:
    failure = classify_tool_call_failure(
        parse_failure(
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "missing_tool",
                            "arguments": {},
                        },
                    ],
                },
            ),
        ),
    )

    assert failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert failure.category is ToolCallFailureCategory.VALIDATION
    assert failure.message == "requested tool is not registered"


def test_retry_policy_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValidationError):
        ToolCallRetryPolicy(max_attempts=0)


def test_retry_allowed_before_max_attempts() -> None:
    decision = decide_tool_call_repair(
        parse_failure('{"tool_calls": ['),
        attempt=1,
        max_attempts=2,
    )

    assert decision.action is ToolCallRepairAction.RETRY
    assert decision.reason is ToolCallRepairDecisionReason.ATTEMPTS_REMAIN
    assert decision.should_retry is True
    assert decision.is_safe_failure is False
    assert decision.attempt == 1
    assert decision.max_attempts == 2


def test_failure_at_max_attempts_returns_safe_failure() -> None:
    decision = decide_tool_call_repair(
        parse_failure('{"tool_calls": ['),
        attempt=2,
        max_attempts=2,
    )

    assert decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert decision.reason is ToolCallRepairDecisionReason.MAX_ATTEMPTS_REACHED
    assert decision.should_retry is False
    assert decision.is_safe_failure is True


def test_safe_failure_when_repair_is_disabled() -> None:
    decision = decide_tool_call_repair(
        parse_failure('{"tool_calls": ['),
        attempt=1,
        max_attempts=2,
        allow_repair=False,
    )

    assert decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert decision.reason is ToolCallRepairDecisionReason.REPAIR_DISABLED
    assert decision.should_retry is False
    assert decision.is_safe_failure is True


def test_decision_rejects_invalid_attempt() -> None:
    with pytest.raises(ValueError, match="attempt must be positive"):
        decide_tool_call_repair(
            parse_failure('{"tool_calls": ['),
            attempt=0,
            max_attempts=2,
        )


def test_failure_messages_exclude_raw_model_content() -> None:
    raw_content = (
        '{"tool_calls": [{"tool_name": "missing_tool", '
        '"arguments": {"secret": "do-not-log"}}]}'
    )

    decision = decide_tool_call_repair(
        parse_failure(raw_content),
        attempt=1,
        max_attempts=1,
    )

    rendered = decision.model_dump_json()
    assert raw_content not in rendered
    assert "do-not-log" not in rendered
    assert "missing_tool" not in rendered


def test_repair_decision_does_not_execute_tools() -> None:
    tool = NeverExecuteTool()
    with pytest.raises(ToolCallValidationError) as error_info:
        parse_structured_tool_calls(
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "never_execute",
                            "arguments": ["not", "an", "object"],
                        },
                    ],
                },
            ),
            ToolRegistry([tool]),
        )

    decision = decide_tool_call_repair(
        error_info.value,
        attempt=1,
        max_attempts=1,
    )

    assert decision.is_safe_failure is True
    assert tool.executed is False
