import json

import pytest
from pydantic import ValidationError

from aigentego.agents import (
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepStatus,
    AgentStepType,
)
from aigentego.llm import (
    ToolCallFailureCategory,
    ToolCallFailureCode,
    ToolCallParsingFailure,
    ToolCallRepairAction,
    ToolCallRepairDecision,
    ToolCallRepairDecisionReason,
)
from aigentego.tools import ToolCall, ToolResult


def repair_decision() -> ToolCallRepairDecision:
    return ToolCallRepairDecision(
        action=ToolCallRepairAction.SAFE_FAILURE,
        reason=ToolCallRepairDecisionReason.MAX_ATTEMPTS_REACHED,
        failure=ToolCallParsingFailure(
            code=ToolCallFailureCode.INVALID_JSON,
            category=ToolCallFailureCategory.PARSE,
            message="invalid structured tool-call JSON",
        ),
        attempt=1,
        max_attempts=1,
    )


def tool_call() -> ToolCall:
    return ToolCall(
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )


def tool_result() -> ToolResult:
    return ToolResult(
        tool_name="calculator",
        success=True,
        result={"value": 4},
    )


def test_valid_minimal_agent_run_can_be_created() -> None:
    run = AgentRun(
        run_id="run-123",
        request_id="req-123",
        user_message="Calculate 2 + 2.",
    )

    assert run.run_id == "run-123"
    assert run.request_id == "req-123"
    assert run.user_message == "Calculate 2 + 2."
    assert run.status is AgentRunStatus.PENDING
    assert run.steps == []
    assert run.final_answer is None


def test_agent_run_with_multiple_steps_preserves_order() -> None:
    steps = [
        AgentStep(
            index=0,
            step_type=AgentStepType.MODEL,
            status=AgentStepStatus.SUCCEEDED,
            model_summary="model requested calculator",
        ),
        AgentStep(
            index=1,
            step_type=AgentStepType.TOOL,
            status=AgentStepStatus.SUCCEEDED,
            tool_call=tool_call(),
            tool_result=tool_result(),
            observation={"source": "tool_executor"},
        ),
    ]

    run = AgentRun(
        run_id="run-123",
        user_message="Calculate 2 + 2.",
        status=AgentRunStatus.RUNNING,
        steps=steps,
    )

    assert run.steps == steps
    assert [step.index for step in run.steps] == [0, 1]
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
    ]


def test_agent_step_can_represent_model_step() -> None:
    step = AgentStep(
        index=0,
        step_type=AgentStepType.MODEL,
        status=AgentStepStatus.SUCCEEDED,
        model_summary="model produced a structured tool-call request",
        tool_calls=[tool_call()],
    )

    assert step.model_summary == "model produced a structured tool-call request"
    assert step.tool_calls == [tool_call()]
    assert step.tool_call is None
    assert step.tool_result is None


def test_agent_step_can_represent_tool_step_with_result() -> None:
    step = AgentStep(
        index=1,
        step_type=AgentStepType.TOOL,
        status=AgentStepStatus.SUCCEEDED,
        tool_call=tool_call(),
        tool_result=tool_result(),
        observation={"request_id": "req-123"},
    )

    assert step.tool_call == tool_call()
    assert step.tool_result == tool_result()
    assert step.observation == {"request_id": "req-123"}


def test_agent_run_can_represent_safe_failure_details() -> None:
    decision = repair_decision()

    run = AgentRun(
        run_id="run-123",
        user_message="Use a tool.",
        status=AgentRunStatus.FAILED,
        repair_decision=decision,
        steps=[
            AgentStep(
                index=0,
                step_type=AgentStepType.ERROR,
                status=AgentStepStatus.FAILED,
                repair_decision=decision,
            ),
        ],
    )

    assert run.repair_decision == decision
    assert run.steps[0].repair_decision == decision


def test_invalid_step_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStep(
            index=-1,
            step_type=AgentStepType.MODEL,
            status=AgentStepStatus.PENDING,
        )


def test_blank_identifiers_and_messages_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentRun(run_id="   ", user_message="valid")

    with pytest.raises(ValidationError):
        AgentRun(run_id="run-123", request_id="   ", user_message="valid")

    with pytest.raises(ValidationError):
        AgentRun(run_id="run-123", user_message="   ")

    with pytest.raises(ValidationError):
        AgentStep(
            index=0,
            step_type=AgentStepType.MODEL,
            status=AgentStepStatus.SUCCEEDED,
            model_summary="   ",
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentRun.model_validate(
            {
                "run_id": "run-123",
                "user_message": "valid",
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {
                "index": 0,
                "step_type": "model",
                "status": "pending",
                "unexpected": "field",
            },
        )


def test_models_serialize_to_json_compatible_data() -> None:
    run = AgentRun(
        run_id="run-123",
        request_id="req-123",
        user_message="Calculate 2 + 2.",
        status=AgentRunStatus.SUCCEEDED,
        final_answer="The result is 4.",
        steps=[
            AgentStep(
                index=0,
                step_type=AgentStepType.TOOL,
                status=AgentStepStatus.SUCCEEDED,
                tool_call=tool_call(),
                tool_result=tool_result(),
            ),
        ],
    )

    data = run.model_dump(mode="json")

    assert json.loads(run.model_dump_json()) == data
    assert data == {
        "run_id": "run-123",
        "request_id": "req-123",
        "user_message": "Calculate 2 + 2.",
        "status": "succeeded",
        "steps": [
            {
                "index": 0,
                "step_type": "tool",
                "status": "succeeded",
                "model_summary": None,
                "tool_calls": [],
                "tool_call": {
                    "tool_name": "calculator",
                    "arguments": {"expression": "2 + 2"},
                },
                "tool_result": {
                    "tool_name": "calculator",
                    "success": True,
                    "result": {"value": 4},
                    "error": None,
                },
                "observation": None,
                "repair_decision": None,
                "error_detail": None,
            },
        ],
        "final_answer": "The result is 4.",
        "stop_reason": None,
        "repair_decision": None,
        "error_detail": None,
    }


def test_terminal_success_and_failure_consistency_is_enforced() -> None:
    with pytest.raises(ValidationError):
        AgentRun(
            run_id="run-123",
            user_message="valid",
            status=AgentRunStatus.SUCCEEDED,
        )

    with pytest.raises(ValidationError):
        AgentRun(
            run_id="run-123",
            user_message="valid",
            status=AgentRunStatus.FAILED,
        )

    with pytest.raises(ValidationError):
        AgentRun(
            run_id="run-123",
            user_message="valid",
            status=AgentRunStatus.RUNNING,
            final_answer="not terminal yet",
        )

    with pytest.raises(ValidationError):
        AgentStep(
            index=0,
            step_type=AgentStepType.ERROR,
            status=AgentStepStatus.FAILED,
        )


def test_stopped_run_requires_stop_reason() -> None:
    with pytest.raises(ValidationError):
        AgentRun(
            run_id="run-123",
            user_message="valid",
            status=AgentRunStatus.STOPPED,
        )

    run = AgentRun(
        run_id="run-123",
        user_message="valid",
        status=AgentRunStatus.STOPPED,
        stop_reason="max_steps_reached",
    )

    assert run.stop_reason == "max_steps_reached"
