import asyncio
import json
from collections.abc import Mapping
from typing import Any

from aigentego.agents import (
    FINAL_ANSWER_SYNTHESIS_FAILED_ERROR_DETAIL,
    FINAL_ANSWER_SYNTHESIS_SUMMARY,
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
    LlmConnectionError,
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
)


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


def registry_with_echo_tool() -> tuple[ToolRegistry, EchoTool]:
    echo_tool = EchoTool()
    return ToolRegistry([echo_tool]), echo_tool


def run_with_synthesis(
    provider: SequencedProvider,
    registry: ToolRegistry,
    executor: RecordingExecutor,
    *,
    retry_policy: ToolCallRetryPolicy | None = None,
):
    return asyncio.run(
        AgentLoopExecutor(
            provider=provider,
            model="agent-model",
            registry=registry,
            retry_policy=retry_policy,
            tool_executor=executor,
        ).run(
            "Use a tool if needed.",
            run_id="run-123",
            request_id="req-123",
        ),
    )


def synthesis_payload(provider: SequencedProvider) -> dict[str, Any]:
    return json.loads(provider.chat_requests[1].messages[1].content)


def tool_call_response(*values: str) -> str:
    return json.dumps(
        {
            "tool_calls": [
                {"tool_name": "echo", "arguments": {"value": value}}
                for value in values
            ],
        },
    )


def test_valid_tool_call_executes_tool_and_synthesizes_final_answer() -> None:
    registry, echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("alpha"),
            "The deterministic result is alpha.",
        ],
    )

    run = run_with_synthesis(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "The deterministic result is alpha."
    assert run.request_id == "req-123"
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    assert run.steps[-1].status is AgentStepStatus.SUCCEEDED
    assert run.steps[-1].model_summary == FINAL_ANSWER_SYNTHESIS_SUMMARY
    assert run.steps[-1].observation == {
        "source": "llm_provider",
        "provider": "fake",
        "model": "agent-model",
        "request_id": "req-123",
    }
    assert executor.calls == [ToolCall(tool_name="echo", arguments={"value": "alpha"})]
    assert len(echo_tool.executions) == 1
    assert echo_tool.executions[0][1].request_id == "req-123"


def test_final_synthesis_request_includes_user_tool_calls_and_results() -> None:
    registry, _echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("first", "second"),
            "Both deterministic results are available.",
        ],
    )

    run_with_synthesis(provider, registry, executor)

    assert provider.chat_requests[1].model == "agent-model"
    assert [message.role for message in provider.chat_requests[1].messages] == [
        "system",
        "user",
    ]
    assert (
        "Do not request or execute tools"
        in provider.chat_requests[1].messages[0].content
    )
    payload = synthesis_payload(provider)
    assert payload["user_message"] == "Use a tool if needed."
    assert payload["request_id"] == "req-123"
    assert [call["arguments"]["value"] for call in payload["tool_calls"]] == [
        "first",
        "second",
    ]
    assert [result["result"]["sequence"] for result in payload["tool_results"]] == [
        1,
        2,
    ]
    assert [step["step_type"] for step in payload["steps"]] == [
        "model",
        "tool",
        "tool",
    ]


def test_empty_tool_calls_synthesizes_without_tool_execution() -> None:
    registry, echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            '{"tool_calls": []}',
            "No deterministic tool was needed.",
        ],
    )

    run = run_with_synthesis(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "No deterministic tool was needed."
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.FINAL,
    ]
    assert executor.calls == []
    assert echo_tool.executions == []
    payload = synthesis_payload(provider)
    assert payload["tool_calls"] == []
    assert payload["tool_results"] == []
    assert payload["repair_decision"] is None


def test_invalid_structured_output_synthesizes_from_safe_context_only() -> None:
    registry, echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    raw_model_output = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    provider = SequencedProvider(
        [
            raw_model_output,
            "I could not safely use a tool.",
        ],
    )

    run = run_with_synthesis(
        provider,
        registry,
        executor,
        retry_policy=ToolCallRetryPolicy(max_attempts=1),
    )

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "I could not safely use a tool."
    assert executor.calls == []
    assert echo_tool.executions == []
    model_step = run.steps[0]
    assert model_step.status is AgentStepStatus.FAILED
    assert model_step.repair_decision is not None
    assert model_step.repair_decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert model_step.repair_decision.failure.code is ToolCallFailureCode.INVALID_JSON
    payload = synthesis_payload(provider)
    assert payload["repair_decision"]["failure"]["code"] == "invalid_json"
    rendered_request = provider.chat_requests[1].model_dump_json()
    assert raw_model_output not in rendered_request
    assert "do-not-leak" not in rendered_request


def test_synthesis_provider_error_returns_inspectable_failed_run() -> None:
    registry, echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("alpha"),
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )

    run = run_with_synthesis(provider, registry, executor)

    assert provider.chat_calls == 2
    assert run.status is AgentRunStatus.FAILED
    assert run.final_answer is None
    assert run.error_detail is not None
    assert FINAL_ANSWER_SYNTHESIS_FAILED_ERROR_DETAIL in run.error_detail
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    final_step = run.steps[-1]
    assert final_step.status is AgentStepStatus.FAILED
    assert final_step.error_detail == run.error_detail
    assert len(echo_tool.executions) == 1


def test_synthesized_agent_run_is_json_serializable() -> None:
    registry, _echo_tool = registry_with_echo_tool()
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("alpha"),
            "The deterministic result is alpha.",
        ],
    )

    run = run_with_synthesis(provider, registry, executor)
    data = run.model_dump(mode="json")

    assert json.loads(run.model_dump_json()) == data
    assert data["status"] == "succeeded"
    assert data["final_answer"] == "The deterministic result is alpha."
