import asyncio
import json
from typing import Any

from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    SingleStepToolCallResult,
    ToolCallFailureCode,
    ToolCallRepairAction,
    ToolCallRetryPolicy,
    run_single_step_tool_call,
    synthesize_tool_answer,
)
from aigentego.tools import (
    CalculatorTool,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(
        self,
        content: str,
        *,
        response_model: str = "test-model",
    ) -> None:
        self.content = content
        self.response_model = response_model
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
            model=self.response_model,
            message=ChatMessage(role="assistant", content=self.content),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


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


def calculator_registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool()])


def run_flow(
    content: str,
    *,
    retry_policy: ToolCallRetryPolicy | None = None,
) -> tuple[FakeProvider, RecordingExecutor, SingleStepToolCallResult]:
    registry = calculator_registry()
    executor = RecordingExecutor(registry)
    provider = FakeProvider(content)
    result = asyncio.run(
        run_single_step_tool_call(
            provider,
            model="tool-model",
            user_message="Use the calculator when arithmetic is required.",
            registry=registry,
            executor=executor,
            request_id="eval-request",
            retry_policy=retry_policy,
        ),
    )
    return provider, executor, result


def prompt_payload(provider: FakeProvider) -> dict[str, Any]:
    assert provider.chat_request is not None
    return json.loads(provider.chat_request.messages[1].content)


def test_valid_calculator_tool_call_executes_once() -> None:
    provider, executor, result = run_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "12 * 31"},
                    },
                ],
            },
        ),
    )

    assert provider.chat_calls == 1
    assert result.parsing_succeeded is True
    assert result.execution_attempted is True
    assert result.tool_calls == [
        ToolCall(tool_name="calculator", arguments={"expression": "12 * 31"}),
    ]
    assert result.tool_results == [
        ToolResult(tool_name="calculator", success=True, result={"value": 372}),
    ]
    assert executor.calls == result.tool_calls
    assert executor.contexts[0] is not None
    assert executor.contexts[0].request_id == "eval-request"


def test_invalid_json_returns_repair_decision_without_execution() -> None:
    raw_content = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    _provider, executor, result = run_flow(
        raw_content,
        retry_policy=ToolCallRetryPolicy(max_attempts=2),
    )

    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.repair_decision is not None
    assert result.repair_decision.action is ToolCallRepairAction.RETRY
    assert result.repair_decision.failure.code is ToolCallFailureCode.INVALID_JSON
    assert executor.calls == []
    assert raw_content not in result.model_dump_json()
    assert "do-not-leak" not in result.model_dump_json()


def test_schema_invalid_structured_output_does_not_execute() -> None:
    _provider, executor, result = run_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": ["not", "an", "object"],
                    },
                ],
            },
        ),
    )

    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.repair_decision is not None
    assert (
        result.repair_decision.failure.code
        is ToolCallFailureCode.SCHEMA_INVALID_JSON
    )
    assert executor.calls == []


def test_unknown_tool_request_does_not_execute() -> None:
    _provider, executor, result = run_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "missing_tool",
                        "arguments": {"expression": "12 * 31"},
                    },
                ],
            },
        ),
    )

    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.repair_decision is not None
    assert result.repair_decision.failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert executor.calls == []


def test_empty_tool_calls_is_successful_noop() -> None:
    provider, executor, result = run_flow('{"tool_calls": []}')

    assert provider.chat_calls == 1
    assert result.parsing_succeeded is True
    assert result.execution_attempted is False
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.repair_decision is None
    assert executor.calls == []


def test_final_synthesis_prompt_after_successful_tool_result() -> None:
    _tool_provider, _executor, step_result = run_flow(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "12 * 31"},
                    },
                ],
            },
        ),
    )
    answer_provider = FakeProvider(
        "The calculator result is 372.",
        response_model="answer-model",
    )

    answer = asyncio.run(
        synthesize_tool_answer(
            answer_provider,
            model="answer-model",
            user_message="What is 12 * 31?",
            step_result=step_result,
        ),
    )

    assert answer.final_answer == "The calculator result is 372."
    assert answer.model == "answer-model"
    assert answer_provider.chat_calls == 1
    payload = prompt_payload(answer_provider)
    assert payload["user_message"] == "What is 12 * 31?"
    assert payload["tool_step"]["tool_results"] == [
        {
            "tool_name": "calculator",
            "success": True,
            "result": {"value": 372},
            "error": None,
        },
    ]


def test_final_synthesis_prompt_after_safe_failure_excludes_raw_model_output() -> None:
    raw_content = json.dumps(
        {
            "tool_calls": [
                {
                    "tool_name": "missing_tool",
                    "arguments": {"secret": "do-not-leak"},
                },
            ],
        },
    )
    _tool_provider, _executor, step_result = run_flow(raw_content)
    answer_provider = FakeProvider("I could not safely use that tool.")

    asyncio.run(
        synthesize_tool_answer(
            answer_provider,
            model="answer-model",
            user_message="Use the missing tool.",
            step_result=step_result,
        ),
    )

    rendered_request = answer_provider.chat_request.model_dump_json()
    payload = prompt_payload(answer_provider)
    repair_decision = payload["tool_step"]["repair_decision"]
    assert repair_decision["action"] == "safe_failure"
    assert repair_decision["failure"]["code"] == "unknown_tool"
    assert raw_content not in rendered_request
    assert "missing_tool" not in rendered_request
    assert "do-not-leak" not in rendered_request


def test_model_output_is_validated_before_any_execution() -> None:
    malformed_outputs = [
        '{"tool_calls": [',
        json.dumps({"tool_calls": [{"tool_name": "calculator"}], "extra": True}),
        json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "missing_tool",
                        "arguments": {"expression": "1 + 1"},
                    },
                ],
            },
        ),
    ]

    for content in malformed_outputs:
        _provider, executor, result = run_flow(content)

        assert result.parsing_succeeded is False
        assert result.execution_attempted is False
        assert executor.calls == []
