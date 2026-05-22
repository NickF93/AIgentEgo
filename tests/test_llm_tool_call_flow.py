import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

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
    run_single_step_tool_call,
)
from aigentego.tools import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(
        self,
        content: str = '{"tool_calls": []}',
        *,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.chat_calls = 0
        self.chat_request: ChatRequest | None = None

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls += 1
        self.chat_request = request
        if self.error is not None:
            raise self.error
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=self.content),
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


def run_flow(
    provider: FakeProvider,
    registry: ToolRegistry,
    executor: RecordingExecutor,
    *,
    request_id: str | None = None,
    retry_policy: ToolCallRetryPolicy | None = None,
):
    return asyncio.run(
        run_single_step_tool_call(
            provider,
            model="test-model",
            user_message="Echo the supplied value.",
            registry=registry,
            executor=executor,
            request_id=request_id,
            retry_policy=retry_policy,
        ),
    )


def make_echo_flow(content: str = '{"tool_calls": []}'):
    tool = EchoTool()
    registry = ToolRegistry([tool])
    executor = RecordingExecutor(registry)
    provider = FakeProvider(content)
    return provider, registry, executor, tool


def test_valid_single_known_tool_executes_once() -> None:
    provider, registry, executor, tool = make_echo_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "hello"},
                    },
                ],
            },
        ),
    )

    result = run_flow(provider, registry, executor, request_id="req-123")

    assert provider.chat_calls == 1
    assert provider.chat_request is not None
    assert provider.chat_request.model == "test-model"
    assert [message.role for message in provider.chat_request.messages] == [
        "system",
        "user",
    ]
    prompt_payload = json.loads(provider.chat_request.messages[1].content)
    assert prompt_payload["user_message"] == "Echo the supplied value."
    assert prompt_payload["tools"][0]["name"] == "echo"
    assert "tool_calls" in prompt_payload["expected_output"]
    assert result.parsing_succeeded is True
    assert result.execution_attempted is True
    assert result.repair_decision is None
    assert result.tool_calls == [
        ToolCall(tool_name="echo", arguments={"value": "hello"}),
    ]
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert executor.calls == result.tool_calls
    assert len(tool.executions) == 1
    assert tool.executions[0][1].request_id == "req-123"


def test_valid_multiple_known_tools_execute_in_order() -> None:
    provider, registry, executor, _tool = make_echo_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "first"},
                    },
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "second"},
                    },
                ],
            },
        ),
    )

    result = run_flow(provider, registry, executor)

    assert provider.chat_calls == 1
    assert [call.arguments["value"] for call in executor.calls] == [
        "first",
        "second",
    ]
    assert [tool_result.result["sequence"] for tool_result in result.tool_results] == [
        1,
        2,
    ]


def test_valid_empty_tool_calls_output_executes_no_tools() -> None:
    provider, registry, executor, tool = make_echo_flow('{"tool_calls": []}')

    result = run_flow(provider, registry, executor)

    assert provider.chat_calls == 1
    assert result.parsing_succeeded is True
    assert result.execution_attempted is False
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.repair_decision is None
    assert executor.calls == []
    assert tool.executions == []


def test_invalid_json_returns_repair_decision_and_executes_no_tools() -> None:
    provider, registry, executor, tool = make_echo_flow('{"tool_calls": [')

    result = run_flow(
        provider,
        registry,
        executor,
        retry_policy=ToolCallRetryPolicy(max_attempts=2),
    )

    assert provider.chat_calls == 1
    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.repair_decision is not None
    assert result.repair_decision.action is ToolCallRepairAction.RETRY
    assert result.repair_decision.failure.code is ToolCallFailureCode.INVALID_JSON
    assert executor.calls == []
    assert tool.executions == []


def test_unknown_tool_returns_safe_failure_and_executes_no_tools() -> None:
    provider, registry, executor, tool = make_echo_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "missing_tool",
                        "arguments": {"value": "hello"},
                    },
                ],
            },
        ),
    )

    result = run_flow(provider, registry, executor)

    assert provider.chat_calls == 1
    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.repair_decision is not None
    assert result.repair_decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert result.repair_decision.failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert executor.calls == []
    assert tool.executions == []


def test_failure_result_does_not_include_raw_model_content() -> None:
    raw_content = json.dumps(
        {
            "tool_calls": [
                {
                    "tool_name": "missing_tool",
                    "arguments": {"secret": "do-not-log"},
                },
            ],
        },
    )
    provider, registry, executor, _tool = make_echo_flow(raw_content)

    result = run_flow(provider, registry, executor)

    rendered = result.model_dump_json()
    assert raw_content not in rendered
    assert "do-not-log" not in rendered
    assert "missing_tool" not in rendered


def test_provider_errors_propagate_without_execution() -> None:
    provider, registry, executor, tool = make_echo_flow()
    provider.error = LlmConnectionError(
        "request failed",
        provider=provider.provider_name,
        operation="chat",
    )

    with pytest.raises(LlmConnectionError):
        run_flow(provider, registry, executor)

    assert provider.chat_calls == 1
    assert executor.calls == []
    assert tool.executions == []
