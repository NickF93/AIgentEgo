import asyncio
import json

import pytest

import aigentego.llm.tool_call_parser as parser_module
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmConnectionError,
    ModelInfo,
    SingleStepToolCallResult,
    ToolCallFailureCategory,
    ToolCallFailureCode,
    ToolCallParsingFailure,
    ToolCallRepairAction,
    ToolCallRepairDecision,
    ToolCallRepairDecisionReason,
    synthesize_tool_answer,
)
from aigentego.tools import ToolCall, ToolExecutor, ToolResult


class FakeProvider:
    provider_name = "fake"

    def __init__(
        self,
        content: str = "final answer",
        *,
        response_model: str = "answer-model",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.response_model = response_model
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
            model=self.response_model,
            message=ChatMessage(role="assistant", content=self.content),
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(model=request.model, embeddings=[])


def run_synthesis(
    provider: FakeProvider,
    step_result: SingleStepToolCallResult,
    *,
    user_message: str = "Use the tool result.",
):
    return asyncio.run(
        synthesize_tool_answer(
            provider,
            model="test-model",
            user_message=user_message,
            step_result=step_result,
        ),
    )


def prompt_payload(provider: FakeProvider) -> dict[str, object]:
    assert provider.chat_request is not None
    return json.loads(provider.chat_request.messages[1].content)


def successful_step(
    values: list[str],
) -> SingleStepToolCallResult:
    return SingleStepToolCallResult(
        parsing_succeeded=True,
        execution_attempted=bool(values),
        tool_calls=[
            ToolCall(tool_name="echo", arguments={"value": value})
            for value in values
        ],
        tool_results=[
            ToolResult(
                tool_name="echo",
                success=True,
                result={"value": value, "sequence": index + 1},
            )
            for index, value in enumerate(values)
        ],
    )


def test_successful_synthesis_after_one_tool_result() -> None:
    provider = FakeProvider("The value is alpha.")
    step_result = successful_step(["alpha"])

    result = run_synthesis(provider, step_result)

    assert provider.chat_calls == 1
    assert result.synthesis_succeeded is True
    assert result.final_answer == "The value is alpha."
    assert result.model == "answer-model"
    assert provider.chat_request is not None
    assert provider.chat_request.model == "test-model"
    assert [message.role for message in provider.chat_request.messages] == [
        "system",
        "user",
    ]
    payload = prompt_payload(provider)
    assert payload["user_message"] == "Use the tool result."
    tool_step = payload["tool_step"]
    assert tool_step["parsing_succeeded"] is True
    assert tool_step["execution_attempted"] is True
    assert tool_step["tool_calls"] == [
        {"tool_name": "echo", "arguments": {"value": "alpha"}},
    ]
    assert tool_step["tool_results"] == [
        {
            "tool_name": "echo",
            "success": True,
            "result": {"value": "alpha", "sequence": 1},
            "error": None,
        },
    ]


def test_successful_synthesis_preserves_multiple_tool_result_order() -> None:
    provider = FakeProvider("Both values are available.")
    step_result = successful_step(["first", "second"])

    run_synthesis(provider, step_result)

    tool_step = prompt_payload(provider)["tool_step"]
    assert [call["arguments"]["value"] for call in tool_step["tool_calls"]] == [
        "first",
        "second",
    ]
    assert [result["result"]["sequence"] for result in tool_step["tool_results"]] == [
        1,
        2,
    ]


def test_successful_synthesis_after_empty_tool_calls() -> None:
    provider = FakeProvider("No tool was needed.")
    step_result = successful_step([])

    result = run_synthesis(provider, step_result)

    assert provider.chat_calls == 1
    assert result.final_answer == "No tool was needed."
    tool_step = prompt_payload(provider)["tool_step"]
    assert tool_step["parsing_succeeded"] is True
    assert tool_step["execution_attempted"] is False
    assert tool_step["tool_calls"] == []
    assert tool_step["tool_results"] == []
    assert tool_step["repair_decision"] is None


def test_synthesis_prompt_includes_original_message_and_tool_result_data() -> None:
    provider = FakeProvider()
    step_result = successful_step(["visible-result"])

    run_synthesis(
        provider,
        step_result,
        user_message="What did the deterministic tool return?",
    )

    rendered_request = provider.chat_request.model_dump_json()
    assert "What did the deterministic tool return?" in rendered_request
    assert "visible-result" in rendered_request


def test_safe_failure_context_excludes_raw_model_output() -> None:
    provider = FakeProvider("I could not use the requested tool.")
    raw_model_output = (
        '{"tool_calls":[{"tool_name":"missing_tool",'
        '"arguments":{"secret":"do-not-leak"}}]}'
    )
    step_result = SingleStepToolCallResult(
        parsing_succeeded=False,
        execution_attempted=False,
        repair_decision=ToolCallRepairDecision(
            action=ToolCallRepairAction.SAFE_FAILURE,
            reason=ToolCallRepairDecisionReason.MAX_ATTEMPTS_REACHED,
            failure=ToolCallParsingFailure(
                code=ToolCallFailureCode.UNKNOWN_TOOL,
                category=ToolCallFailureCategory.VALIDATION,
                message="requested tool is not registered",
            ),
            attempt=1,
            max_attempts=1,
        ),
    )

    run_synthesis(provider, step_result)

    rendered_request = provider.chat_request.model_dump_json()
    payload = prompt_payload(provider)
    repair_decision = payload["tool_step"]["repair_decision"]
    assert repair_decision["action"] == "safe_failure"
    assert repair_decision["failure"]["code"] == "unknown_tool"
    assert "requested tool is not registered" in rendered_request
    assert raw_model_output not in rendered_request
    assert "missing_tool" not in rendered_request
    assert "do-not-leak" not in rendered_request


def test_provider_errors_propagate() -> None:
    provider = FakeProvider(
        error=LlmConnectionError(
            "request failed",
            provider="fake",
            operation="chat",
        ),
    )

    with pytest.raises(LlmConnectionError):
        run_synthesis(provider, successful_step(["alpha"]))

    assert provider.chat_calls == 1


def test_synthesis_does_not_parse_or_execute_tools(monkeypatch) -> None:
    parser_called = False
    executor_called = False

    def fail_parser(*_args: object, **_kwargs: object) -> list[ToolCall]:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("parser should not be called")

    async def fail_execute(
        _self: ToolExecutor,
        _call: ToolCall,
        _context: object = None,
    ) -> ToolResult:
        nonlocal executor_called
        executor_called = True
        raise AssertionError("tool executor should not be called")

    monkeypatch.setattr(
        parser_module,
        "parse_structured_tool_calls",
        fail_parser,
    )
    monkeypatch.setattr(ToolExecutor, "execute", fail_execute)

    provider = FakeProvider("Synthesis only.")
    result = run_synthesis(provider, successful_step(["alpha"]))

    assert result.final_answer == "Synthesis only."
    assert provider.chat_calls == 1
    assert parser_called is False
    assert executor_called is False
