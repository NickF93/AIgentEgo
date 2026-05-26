import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from aigentego.agents import (
    MAX_STEPS_REACHED_STOP_REASON,
    MAX_TOOL_ERRORS_REACHED_STOP_REASON,
    TIMEOUT_REACHED_STOP_REASON,
    TOOL_CALL_GENERATION_FAILED_ERROR_DETAIL,
    AgentLoopExecutor,
    AgentLoopLimits,
    AgentRunStatus,
    AgentStepStatus,
    AgentStepType,
)
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmConnectionError,
    ModelInfo,
    ToolCallFailureCode,
)
from aigentego.tools import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
)


class MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SequencedProvider:
    provider_name = "fake"

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.chat_calls = 0
        self.chat_requests: list[ChatRequest] = []

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls += 1
        self.chat_requests.append(request)
        response = self.responses[self.chat_calls - 1]
        if isinstance(response, Exception):
            raise response
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=response),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


class EchoTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Return supplied arguments.",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={
                "arguments": dict(arguments),
                "request_id": context.request_id,
            },
        )


class SlowEchoTool(EchoTool):
    def __init__(self, clock: MutableClock) -> None:
        self._clock = clock

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        self._clock.now = 31.0
        return await super().execute(arguments, context)


class FailedResultTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failed_result",
            description="Return a normalized failed result.",
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        error = ToolValidationError(
            "value is invalid",
            tool_name=self.definition.name,
        )
        return ToolResult(
            tool_name=self.definition.name,
            success=False,
            error=error.to_detail(),
        )


class RecordingExecutor(ToolExecutor):
    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.calls: list[ToolCall] = []
        self.contexts: list[ToolContext | None] = []

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext | None = None,
    ) -> ToolResult:
        self.calls.append(call)
        self.contexts.append(context)
        return await super().execute(call, context)


class RaisingExecutor(ToolExecutor):
    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.calls: list[ToolCall] = []

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext | None = None,
    ) -> ToolResult:
        self.calls.append(call)
        raise RuntimeError("private executor details")


def tool_call_response(*tool_names: str) -> str:
    return json.dumps(
        {
            "tool_calls": [
                {"tool_name": tool_name, "arguments": {"value": tool_name}}
                for tool_name in tool_names
            ],
        },
    )


def run_agent(
    provider: SequencedProvider,
    registry: ToolRegistry,
    executor: ToolExecutor,
    *,
    limits: AgentLoopLimits,
    clock: MutableClock | None = None,
):
    return asyncio.run(
        AgentLoopExecutor(
            limits=limits,
            provider=provider,
            model="agent-model",
            registry=registry,
            tool_executor=executor,
            clock=clock,
        ).run(
            "Use a tool if needed.",
            run_id="run-123",
            request_id="req-123",
        ),
    )


def test_agent_loop_limits_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        AgentLoopLimits(max_steps=-1)

    with pytest.raises(ValidationError):
        AgentLoopLimits(max_tool_errors=-1)

    with pytest.raises(ValidationError):
        AgentLoopLimits(timeout_seconds=0)

    with pytest.raises(ValidationError):
        AgentLoopLimits.model_validate({"max_steps": 1, "unexpected": "field"})


def test_max_steps_zero_stops_before_model_call() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_steps=0),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON
    assert run.steps == []
    assert provider.chat_calls == 0
    assert executor.calls == []


def test_max_steps_stops_before_tool_execution() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_steps=1),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON
    assert [step.step_type for step in run.steps] == [AgentStepType.MODEL]
    assert provider.chat_calls == 1
    assert executor.calls == []


def test_max_steps_stops_before_final_synthesis() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_steps=2),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
    ]
    assert provider.chat_calls == 1
    assert executor.calls == [ToolCall(tool_name="echo", arguments={"value": "echo"})]


def test_enough_max_steps_allows_successful_path() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "final answer"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_steps=3),
    )

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "final answer"
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    assert provider.chat_calls == 2


def test_max_tool_errors_zero_stops_after_first_failed_tool_result() -> None:
    registry = ToolRegistry([FailedResultTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("failed_result"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_tool_errors=0),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_TOOL_ERRORS_REACHED_STOP_REASON
    assert provider.chat_calls == 1
    assert len(executor.calls) == 1
    failed_step = run.steps[1]
    assert failed_step.status is AgentStepStatus.FAILED
    assert failed_step.tool_result is not None
    assert failed_step.tool_result.success is False
    assert failed_step.observation == {
        "source": "tool_executor",
        "request_id": "req-123",
        "success": False,
    }


def test_multiple_failed_tool_results_count_toward_limit() -> None:
    registry = ToolRegistry([FailedResultTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [tool_call_response("failed_result", "failed_result"), "unused"],
    )

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_tool_errors=1),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_TOOL_ERRORS_REACHED_STOP_REASON
    assert provider.chat_calls == 1
    assert len(executor.calls) == 2
    assert [step.status for step in run.steps[1:]] == [
        AgentStepStatus.FAILED,
        AgentStepStatus.FAILED,
    ]


def test_timeout_stops_before_later_phase_with_deterministic_clock() -> None:
    clock = MutableClock()
    registry = ToolRegistry([SlowEchoTool(clock)])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(timeout_seconds=30),
        clock=clock,
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == TIMEOUT_REACHED_STOP_REASON
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
    ]
    assert provider.chat_calls == 1
    assert len(executor.calls) == 1


def test_invalid_structured_output_does_not_execute_tools_or_exceed_limits() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    raw_content = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    provider = SequencedProvider([raw_content, "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_steps=1),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_STEPS_REACHED_STOP_REASON
    assert len(run.steps) == 1
    assert run.steps[0].repair_decision is not None
    assert run.steps[0].repair_decision.failure.code is (
        ToolCallFailureCode.INVALID_JSON
    )
    assert provider.chat_calls == 1
    assert executor.calls == []
    rendered = run.model_dump_json()
    assert raw_content not in rendered
    assert "do-not-leak" not in rendered


def test_provider_generation_error_returns_failed_run() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.error_detail is not None
    assert TOOL_CALL_GENERATION_FAILED_ERROR_DETAIL in run.error_detail
    assert run.steps[0].step_type is AgentStepType.MODEL
    assert run.steps[0].status is AgentStepStatus.FAILED
    assert run.steps[0].error_detail == run.error_detail
    assert provider.chat_calls == 1
    assert executor.calls == []


def test_provider_synthesis_error_returns_failed_run() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("echo"),
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.error_detail is not None
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    assert run.steps[-1].status is AgentStepStatus.FAILED
    assert provider.chat_calls == 2


def test_unexpected_tool_executor_failure_is_sanitized_and_inspectable() -> None:
    registry = ToolRegistry([EchoTool()])
    executor = RaisingExecutor(registry)
    provider = SequencedProvider([tool_call_response("echo"), "unused"])

    run = run_agent(
        provider,
        registry,
        executor,
        limits=AgentLoopLimits(max_tool_errors=0),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == MAX_TOOL_ERRORS_REACHED_STOP_REASON
    tool_step = run.steps[1]
    assert tool_step.status is AgentStepStatus.FAILED
    assert tool_step.tool_result is not None
    assert tool_step.tool_result.error is not None
    assert tool_step.tool_result.error.code == "tool_execution_error"
    assert tool_step.tool_result.error.message == "tool execution failed"
    rendered = run.model_dump_json()
    assert "private executor details" not in rendered
    assert provider.chat_calls == 1
    assert executor.calls == [ToolCall(tool_name="echo", arguments={"value": "echo"})]
