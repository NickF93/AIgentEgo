import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aigentego.agents import (
    MAX_STEPS_REACHED_STOP_REASON,
    AgentLoopExecutor,
    AgentLoopLimits,
    AgentRun,
    AgentRunStatus,
    AgentStepStatus,
    AgentStepType,
)
from aigentego.api.dependencies import get_llm_provider, get_settings
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
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.settings import Settings
from aigentego.tools import (
    CalculatorTool,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
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


class FailedResultTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failed_result",
            description="Return a deterministic failed result.",
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


def make_client(provider: SequencedProvider) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        llm_backend="ollama",
        llm_base_url="http://ollama:11434",
        chat_model="agent-eval-model",
        embedding_model="nomic-embed-text",
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider

    return TestClient(app)


def tool_call_response(tool_name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_calls": [
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
            ],
        },
    )


def run_agent(
    provider: SequencedProvider,
    registry: ToolRegistry,
    executor: RecordingExecutor,
    *,
    limits: AgentLoopLimits | None = None,
) -> AgentRun:
    return asyncio.run(
        AgentLoopExecutor(
            limits=limits,
            provider=provider,
            model="agent-eval-model",
            registry=registry,
            tool_executor=executor,
        ).run(
            "Use deterministic tools if needed.",
            run_id="eval-run",
            request_id="eval-request",
        ),
    )


def test_agent_run_api_succeeds_with_mocked_provider_and_calculator() -> None:
    provider = SequencedProvider(
        [
            tool_call_response("calculator", {"expression": "8 * 9"}),
            "The calculator result is 72.",
        ],
    )
    client = make_client(provider)

    response = client.post(
        "/agent/run",
        headers={REQUEST_ID_HEADER: "eval-api-request"},
        json={"message": " What is 8 * 9? "},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "eval-api-request"
    body = response.json()
    assert body["run_id"] == "eval-api-request"
    assert body["request_id"] == "eval-api-request"
    assert body["user_message"] == "What is 8 * 9?"
    assert body["status"] == "succeeded"
    assert body["final_answer"] == "The calculator result is 72."
    assert [step["step_type"] for step in body["steps"]] == [
        "model",
        "tool",
        "final",
    ]
    assert body["steps"][1]["tool_result"] == {
        "tool_name": "calculator",
        "success": True,
        "result": {"value": 72},
        "error": None,
    }
    assert provider.chat_calls == 2
    json.dumps(body)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "   "},
        {"message": "Run.", "max_steps": -1},
        {"message": "Run.", "max_tool_errors": -1},
        {"message": "Run.", "timeout_seconds": 0},
        {"message": "Run.", "unexpected": "field"},
    ],
)
def test_agent_run_api_rejects_invalid_payloads(
    payload: dict[str, object],
) -> None:
    provider = SequencedProvider([])
    client = make_client(provider)

    response = client.post("/agent/run", json=payload)

    assert response.status_code == 422
    assert provider.chat_calls == 0


def test_agent_loop_max_steps_stop_prevents_tool_execution() -> None:
    registry = ToolRegistry([CalculatorTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("calculator", {"expression": "2 + 2"}),
            "unused",
        ],
    )

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
    json.dumps(run.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("raw_output", "expected_code"),
    [
        (
            '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}',
            ToolCallFailureCode.INVALID_JSON,
        ),
        (
            tool_call_response("missing_tool", {}),
            ToolCallFailureCode.UNKNOWN_TOOL,
        ),
    ],
)
def test_invalid_or_unknown_tool_calls_do_not_execute_tools(
    raw_output: str,
    expected_code: ToolCallFailureCode,
) -> None:
    registry = ToolRegistry([CalculatorTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider([raw_output, "I could not use that tool."])

    run = run_agent(provider, registry, executor)

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "I could not use that tool."
    assert executor.calls == []
    assert provider.chat_calls == 2
    assert run.steps[0].status is AgentStepStatus.FAILED
    assert run.steps[0].repair_decision is not None
    assert run.steps[0].repair_decision.failure.code is expected_code
    rendered = json.dumps(run.model_dump(mode="json"))
    assert raw_output not in rendered
    assert "do-not-leak" not in rendered


def test_failed_tool_result_is_recorded_as_inspectable_observation() -> None:
    registry = ToolRegistry([FailedResultTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("failed_result", {}),
            "The deterministic tool returned a validation failure.",
        ],
    )

    run = run_agent(provider, registry, executor)

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.final_answer == "The deterministic tool returned a validation failure."
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    tool_step = run.steps[1]
    assert tool_step.status is AgentStepStatus.FAILED
    assert tool_step.error_detail == "tool execution returned a failed result"
    assert tool_step.tool_result is not None
    assert tool_step.tool_result.success is False
    assert tool_step.tool_result.error is not None
    assert tool_step.observation == {
        "source": "tool_executor",
        "request_id": "eval-request",
        "success": False,
    }
    assert executor.calls == [ToolCall(tool_name="failed_result", arguments={})]


def test_final_synthesis_failure_returns_inspectable_failed_run() -> None:
    registry = ToolRegistry([CalculatorTool()])
    executor = RecordingExecutor(registry)
    provider = SequencedProvider(
        [
            tool_call_response("calculator", {"expression": "3 * 3"}),
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )

    run = run_agent(provider, registry, executor)

    assert run.status is AgentRunStatus.FAILED
    assert run.final_answer is None
    assert run.error_detail is not None
    assert run.error_detail.startswith("final answer synthesis failed")
    assert [step.step_type for step in run.steps] == [
        AgentStepType.MODEL,
        AgentStepType.TOOL,
        AgentStepType.FINAL,
    ]
    assert run.steps[-1].status is AgentStepStatus.FAILED
    assert run.steps[-1].error_detail == run.error_detail
    assert provider.chat_calls == 2
    assert executor.calls == [
        ToolCall(tool_name="calculator", arguments={"expression": "3 * 3"}),
    ]
    json.dumps(run.model_dump(mode="json"))
