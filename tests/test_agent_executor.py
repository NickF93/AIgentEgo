import asyncio
import json

import pytest
from pydantic import ValidationError

from aigentego.agents import (
    AGENT_LOOP_SKELETON_STOP_REASON,
    MAX_STEPS_REACHED_STOP_REASON,
    AgentLoopExecutor,
    AgentLoopLimits,
    AgentRun,
    AgentRunStatus,
    run_agent_loop,
)


def test_executor_creates_stopped_run_for_valid_message() -> None:
    executor = AgentLoopExecutor()

    run = asyncio.run(
        executor.run(
            "Calculate 2 + 2.",
            run_id="run-123",
        ),
    )

    assert run == AgentRun(
        run_id="run-123",
        user_message="Calculate 2 + 2.",
        status=AgentRunStatus.STOPPED,
        stop_reason=AGENT_LOOP_SKELETON_STOP_REASON,
    )
    assert run.steps == []
    assert run.final_answer is None
    assert run.error_detail is None
    assert run.repair_decision is None


def test_executor_preserves_request_id() -> None:
    run = asyncio.run(
        AgentLoopExecutor().run(
            "Use the available tools.",
            run_id="run-123",
            request_id="req-123",
        ),
    )

    assert run.request_id == "req-123"


def test_executor_rejects_blank_user_message() -> None:
    with pytest.raises(ValidationError):
        asyncio.run(AgentLoopExecutor().run("   "))


def test_executor_stops_at_zero_max_steps() -> None:
    executor = AgentLoopExecutor(limits=AgentLoopLimits(max_steps=0))

    run = asyncio.run(executor.run("Calculate 2 + 2."))

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON
    assert run.steps == []


def test_agent_loop_limits_reject_invalid_values_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentLoopLimits(max_steps=-1)

    with pytest.raises(ValidationError):
        AgentLoopLimits.model_validate(
            {
                "max_steps": 1,
                "unexpected": "field",
            },
        )


def test_run_agent_loop_convenience_function_uses_limits() -> None:
    run = asyncio.run(
        run_agent_loop(
            "Start a bounded run.",
            run_id="run-123",
            request_id="req-123",
            limits=AgentLoopLimits(max_steps=0),
        ),
    )

    assert run.run_id == "run-123"
    assert run.request_id == "req-123"
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON


def test_returned_run_is_json_serializable() -> None:
    run = asyncio.run(AgentLoopExecutor().run("Calculate 2 + 2."))

    data = run.model_dump(mode="json")

    assert json.loads(run.model_dump_json()) == data
    assert data == {
        "run_id": "agent-run",
        "request_id": None,
        "user_message": "Calculate 2 + 2.",
        "status": "stopped",
        "steps": [],
        "final_answer": None,
        "stop_reason": AGENT_LOOP_SKELETON_STOP_REASON,
        "repair_decision": None,
        "error_detail": None,
    }


def test_repeated_runs_are_structurally_deterministic() -> None:
    executor = AgentLoopExecutor()

    first = asyncio.run(executor.run("Repeatable run.", run_id="run-123"))
    second = asyncio.run(executor.run("Repeatable run.", run_id="run-123"))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
