import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from aigentego.agents import (
    NO_TOOL_CALLS_GENERATED_STOP_REASON,
    TOOL_CALL_GENERATION_CONFIG_ERROR,
    TOOL_CALLS_GENERATED_STOP_REASON,
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
    ToolRegistry,
    ToolResult,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(self, content: str = '{"tool_calls": []}') -> None:
        self.content = content
        self.chat_calls = 0
        self.chat_request: ChatRequest | None = None

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls += 1
        self.chat_request = request
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=self.content),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


class EchoTool:
    def __init__(self) -> None:
        self.executions = 0

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
        self.executions += 1
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={"arguments": dict(arguments), "request_id": context.request_id},
        )


def registry_with_echo_tool() -> tuple[ToolRegistry, EchoTool]:
    tool = EchoTool()
    return ToolRegistry([tool]), tool


def run_with_generation(
    provider: FakeProvider,
    registry: ToolRegistry,
    *,
    retry_policy: ToolCallRetryPolicy | None = None,
):
    return asyncio.run(
        AgentLoopExecutor(
            provider=provider,
            model="tool-model",
            registry=registry,
            retry_policy=retry_policy,
        ).run(
            "Use a tool if needed.",
            run_id="run-123",
            request_id="req-123",
        ),
    )


def prompt_payload(provider: FakeProvider) -> dict[str, Any]:
    assert provider.chat_request is not None
    return json.loads(provider.chat_request.messages[1].content)


def test_valid_single_tool_call_records_model_step_without_execution() -> None:
    registry, tool = registry_with_echo_tool()
    provider = FakeProvider(
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

    run = run_with_generation(provider, registry)

    assert provider.chat_calls == 1
    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == TOOL_CALLS_GENERATED_STOP_REASON
    assert run.request_id == "req-123"
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.index == 0
    assert step.step_type is AgentStepType.MODEL
    assert step.status is AgentStepStatus.SUCCEEDED
    assert step.tool_calls == [
        ToolCall(tool_name="echo", arguments={"value": "hello"}),
    ]
    assert step.tool_call is None
    assert step.tool_result is None
    assert run.repair_decision is None
    assert tool.executions == 0


def test_valid_multiple_tool_calls_preserve_order() -> None:
    registry, tool = registry_with_echo_tool()
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

    run = run_with_generation(provider, registry)

    assert provider.chat_calls == 1
    assert [call.arguments["value"] for call in run.steps[0].tool_calls] == [
        "first",
        "second",
    ]
    assert tool.executions == 0


def test_empty_tool_calls_records_noop_model_step() -> None:
    registry, tool = registry_with_echo_tool()
    provider = FakeProvider('{"tool_calls": []}')

    run = run_with_generation(provider, registry)

    assert provider.chat_calls == 1
    assert run.status is AgentRunStatus.STOPPED
    assert run.stop_reason == NO_TOOL_CALLS_GENERATED_STOP_REASON
    assert run.steps[0].status is AgentStepStatus.SUCCEEDED
    assert run.steps[0].tool_calls == []
    assert run.repair_decision is None
    assert tool.executions == 0


def test_invalid_json_records_repair_decision_without_execution() -> None:
    registry, tool = registry_with_echo_tool()
    raw_content = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    provider = FakeProvider(raw_content)

    run = run_with_generation(
        provider,
        registry,
        retry_policy=ToolCallRetryPolicy(max_attempts=2),
    )

    assert provider.chat_calls == 1
    assert run.status is AgentRunStatus.FAILED
    assert run.repair_decision is not None
    assert run.repair_decision.action is ToolCallRepairAction.RETRY
    assert run.repair_decision.failure.code is ToolCallFailureCode.INVALID_JSON
    assert run.steps[0].status is AgentStepStatus.FAILED
    assert run.steps[0].repair_decision == run.repair_decision
    assert run.steps[0].tool_calls == []
    assert tool.executions == 0
    rendered = run.model_dump_json()
    assert raw_content not in rendered
    assert "do-not-leak" not in rendered


def test_unknown_tool_records_safe_failure_without_execution() -> None:
    registry, tool = registry_with_echo_tool()
    provider = FakeProvider(
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

    run = run_with_generation(provider, registry)

    assert provider.chat_calls == 1
    assert run.status is AgentRunStatus.FAILED
    assert run.repair_decision is not None
    assert run.repair_decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert run.repair_decision.failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert run.steps[0].repair_decision == run.repair_decision
    assert tool.executions == 0
    rendered = run.model_dump_json()
    assert "missing_tool" not in rendered
    assert "hello" not in rendered


def test_model_request_includes_tools_and_expected_output_shape() -> None:
    registry, _tool = registry_with_echo_tool()
    provider = FakeProvider('{"tool_calls": []}')

    run_with_generation(provider, registry)

    assert provider.chat_request is not None
    assert provider.chat_request.model == "tool-model"
    assert [message.role for message in provider.chat_request.messages] == [
        "system",
        "user",
    ]
    assert "tool_calls" in provider.chat_request.messages[0].content
    payload = prompt_payload(provider)
    assert payload["user_message"] == "Use a tool if needed."
    assert payload["tools"] == [
        {
            "name": "echo",
            "description": "Return supplied arguments.",
            "parameters_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    ]
    assert payload["expected_output"] == {
        "tool_calls": [
            {
                "tool_name": "registered_tool_name",
                "arguments": {},
            },
        ],
    }


def test_run_can_accept_generation_dependencies_per_call() -> None:
    registry, tool = registry_with_echo_tool()
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "runtime"}},
                ],
            },
        ),
    )

    run = asyncio.run(
        AgentLoopExecutor().run(
            "Use runtime dependencies.",
            provider=provider,
            model="tool-model",
            registry=registry,
        ),
    )

    assert provider.chat_calls == 1
    assert run.steps[0].tool_calls == [
        ToolCall(tool_name="echo", arguments={"value": "runtime"}),
    ]
    assert tool.executions == 0


def test_partial_generation_configuration_is_rejected() -> None:
    provider = FakeProvider()

    with pytest.raises(ValueError, match=TOOL_CALL_GENERATION_CONFIG_ERROR):
        AgentLoopExecutor(provider=provider)

    with pytest.raises(ValueError, match=TOOL_CALL_GENERATION_CONFIG_ERROR):
        asyncio.run(AgentLoopExecutor().run("Use tools.", provider=provider))


def test_generated_run_is_json_serializable() -> None:
    registry, _tool = registry_with_echo_tool()
    provider = FakeProvider(
        json.dumps(
            {
                "tool_calls": [
                    {"tool_name": "echo", "arguments": {"value": "hello"}},
                ],
            },
        ),
    )

    run = run_with_generation(provider, registry)
    data = run.model_dump(mode="json")

    assert json.loads(run.model_dump_json()) == data
    assert data["steps"][0]["tool_calls"] == [
        {
            "tool_name": "echo",
            "arguments": {"value": "hello"},
        },
    ]
