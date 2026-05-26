import asyncio
import json
from collections.abc import Mapping
from typing import Any

from aigentego.agents import (
    AGENT_LOOP_SKELETON_STOP_REASON,
    AgentLoopExecutor,
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
    ModelInfo,
    ToolCallFailureCode,
    ToolCallRepairAction,
    ToolCallRetryPolicy,
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


class FakeProvider:
    provider_name = "fake"

    def __init__(
        self,
        content: str = '{"tool_calls": []}',
        *,
        final_content: str = "final answer",
    ) -> None:
        self.contents = [content, final_content]
        self.chat_calls = 0
        self.chat_request: ChatRequest | None = None

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls += 1
        self.chat_request = request
        content = self.contents[min(self.chat_calls - 1, len(self.contents) - 1)]
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=content),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


class EchoTool:
    def __init__(self) -> None:
        self.executions: list[tuple[dict[str, Any], ToolContext]] = []

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
        recorded_arguments = dict(arguments)
        self.executions.append((recorded_arguments, context))
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={
                "arguments": recorded_arguments,
                "request_id": context.request_id,
                "sequence": len(self.executions),
            },
        )


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


def registry_with_tools() -> tuple[ToolRegistry, EchoTool]:
    echo_tool = EchoTool()
    return ToolRegistry([echo_tool, FailedResultTool()]), echo_tool


def run_with_tool_executor(
    provider: FakeProvider,
    registry: ToolRegistry,
    executor: RecordingExecutor,
    *,
    retry_policy: ToolCallRetryPolicy | None = None,
):
    return asyncio.run(
        AgentLoopExecutor(
            provider=provider,
            model="tool-model",
            registry=registry,
            retry_policy=retry_policy,
            tool_executor=executor,
        ).run(
            "Use a tool if needed.",
            run_id="run-123",
            request_id="req-123",
        ),
    )


def test_valid_single_tool_call_executes_once_and_records_observation() -> None:
    registry, echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "hello"}},
                ],
            },
        ),
    )

    run = run_with_tool_executor(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.stop_reason is None
    assert run.final_answer == "final answer"
    assert len(run.steps) == 3
    model_step, tool_step, final_step = run.steps
    assert model_step.step_type is AgentStepType.MODEL
    assert model_step.status is AgentStepStatus.SUCCEEDED
    assert model_step.tool_calls == [
        ToolCall(tool_name="echo", arguments={"value": "hello"}),
    ]
    assert tool_step.index == 1
    assert tool_step.step_type is AgentStepType.TOOL
    assert tool_step.status is AgentStepStatus.SUCCEEDED
    assert tool_step.tool_call == model_step.tool_calls[0]
    assert tool_step.tool_result == ToolResult(
        tool_name="echo",
        success=True,
        result={
            "arguments": {"value": "hello"},
            "request_id": "req-123",
            "sequence": 1,
        },
    )
    assert tool_step.observation == {
        "source": "tool_executor",
        "request_id": "req-123",
        "success": True,
    }
    assert final_step.step_type is AgentStepType.FINAL
    assert final_step.status is AgentStepStatus.SUCCEEDED
    assert executor.calls == model_step.tool_calls
    assert len(echo_tool.executions) == 1
    assert echo_tool.executions[0][1].request_id == "req-123"
    assert executor.contexts[0] is not None
    assert executor.contexts[0].request_id == "req-123"


def test_multiple_tool_calls_execute_and_record_steps_in_order() -> None:
    registry, _echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "first"}},
                    {"tool_name": "echo", "arguments": {"value": "second"}},
                ],
            },
        ),
    )

    run = run_with_tool_executor(provider, registry, executor)

    assert provider.chat_calls == 2
    assert [call.arguments["value"] for call in executor.calls] == [
        "first",
        "second",
    ]
    assert [step.index for step in run.steps] == [0, 1, 2, 3]
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    assert [
        step.tool_result.result["sequence"]
        for step in run.steps[1:-1]
        if step.tool_result is not None and step.tool_result.result is not None
    ] == [1, 2]


def test_failed_tool_result_is_recorded_without_aborting_run() -> None:
    registry, _echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "failed_result", "arguments": {}},
                ],
            },
        ),
    )

    run = run_with_tool_executor(provider, registry, executor)

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.stop_reason is None
    assert len(run.steps) == 3
    failed_step = run.steps[1]
    assert failed_step.step_type is AgentStepType.TOOL
    assert failed_step.status is AgentStepStatus.FAILED
    assert failed_step.error_detail == "tool execution returned a failed result"
    assert failed_step.tool_result is not None
    assert failed_step.tool_result.success is False
    assert failed_step.tool_result.error is not None
    assert failed_step.tool_result.error.code == "tool_validation_error"
    assert failed_step.observation == {
        "source": "tool_executor",
        "request_id": "req-123",
        "success": False,
    }


def test_invalid_json_executes_no_tools() -> None:
    registry, echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    raw_content = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    provider = FakeProvider(raw_content)

    run = run_with_tool_executor(
        provider,
        registry,
        executor,
        retry_policy=ToolCallRetryPolicy(max_attempts=2),
    )

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.repair_decision is None
    assert run.steps[0].repair_decision is not None
    assert run.steps[0].repair_decision.action is ToolCallRepairAction.RETRY
    assert run.steps[0].repair_decision.failure.code is (
        ToolCallFailureCode.INVALID_JSON
    )
    assert executor.calls == []
    assert echo_tool.executions == []
    rendered = run.model_dump_json()
    assert raw_content not in rendered
    assert "do-not-leak" not in rendered


def test_unknown_tool_executes_no_tools() -> None:
    registry, echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "missing_tool", "arguments": {"value": "hello"}},
                ],
            },
        ),
    )

    run = run_with_tool_executor(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.repair_decision is None
    assert run.steps[0].repair_decision is not None
    assert run.steps[0].repair_decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert run.steps[0].repair_decision.failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert executor.calls == []
    assert echo_tool.executions == []


def test_empty_tool_calls_executes_no_tools() -> None:
    registry, echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider('{"tool_calls": []}')

    run = run_with_tool_executor(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.stop_reason is None
    assert len(run.steps) == 2
    assert run.steps[0].tool_calls == []
    assert executor.calls == []
    assert echo_tool.executions == []


def test_generation_only_behavior_remains_when_tool_executor_is_absent() -> None:
    registry, echo_tool = registry_with_tools()
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "hello"}},
                ],
            },
        ),
    )

    run = asyncio.run(
        AgentLoopExecutor(
            provider=provider,
            model="tool-model",
            registry=registry,
        ).run("Use a tool if needed."),
    )

    assert provider.chat_calls == 1
    assert len(run.steps) == 1
    assert run.steps[0].tool_calls == [
        ToolCall(tool_name="echo", arguments={"value": "hello"}),
    ]
    assert echo_tool.executions == []


def test_skeleton_no_provider_behavior_remains_intact() -> None:
    run = asyncio.run(
        AgentLoopExecutor().run(
            "Use a tool if needed.",
            run_id="run-123",
        ),
    )

    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == AGENT_LOOP_SKELETON_STOP_REASON
    assert run.steps == []


def test_tool_observation_run_is_json_serializable() -> None:
    registry, _echo_tool = registry_with_tools()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "hello"}},
                ],
            },
        ),
    )

    run = run_with_tool_executor(provider, registry, executor)
    data = run.model_dump(mode="json")

    assert json.loads(run.model_dump_json()) == data
    assert data["final_answer"] == "final answer"
    assert data["steps"][1]["observation"] == {
        "source": "tool_executor",
        "request_id": "req-123",
        "success": True,
    }
    assert data["steps"][2]["step_type"] == "final"
