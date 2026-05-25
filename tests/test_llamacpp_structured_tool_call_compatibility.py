import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx

from aigentego.llm import (
    LlamaCppProvider,
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


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> LlamaCppProvider:
    return LlamaCppProvider(
        "http://localhost:8080",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )


def chat_completion(content: str) -> dict[str, Any]:
    return {
        "model": "local-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            },
        ],
    }


def calculator_flow_components() -> tuple[ToolRegistry, RecordingExecutor]:
    registry = ToolRegistry([CalculatorTool()])
    return registry, RecordingExecutor(registry)


def request_payload(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode("utf-8"))


def test_llamacpp_structured_calculator_tool_call_executes() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        payload = request_payload(request)
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json=chat_completion(
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
            ),
            request=request,
        )

    registry, executor = calculator_flow_components()
    provider = make_provider(handler)

    result = asyncio.run(
        run_single_step_tool_call(
            provider,
            model="local-chat",
            user_message="What is 12 * 31?",
            registry=registry,
            executor=executor,
            request_id="llamacpp-compat",
        ),
    )

    assert provider.provider_name == "llamacpp"
    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "local-chat"
    assert seen_payloads[0]["stream"] is False
    assert [message["role"] for message in seen_payloads[0]["messages"]] == [
        "system",
        "user",
    ]
    prompt_context = json.loads(seen_payloads[0]["messages"][1]["content"])
    assert prompt_context["user_message"] == "What is 12 * 31?"
    assert prompt_context["tools"][0]["name"] == "calculator"
    assert result.parsing_succeeded is True
    assert result.execution_attempted is True
    assert result.repair_decision is None
    assert result.tool_calls == [
        ToolCall(tool_name="calculator", arguments={"expression": "12 * 31"}),
    ]
    assert result.tool_results == [
        ToolResult(tool_name="calculator", success=True, result={"value": 372}),
    ]
    assert executor.calls == result.tool_calls
    assert executor.contexts[0] is not None
    assert executor.contexts[0].request_id == "llamacpp-compat"


def test_llamacpp_invalid_structured_output_does_not_execute_tools() -> None:
    raw_content = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json=chat_completion(raw_content),
            request=request,
        )

    registry, executor = calculator_flow_components()
    result = asyncio.run(
        run_single_step_tool_call(
            make_provider(handler),
            model="local-chat",
            user_message="Use the calculator.",
            registry=registry,
            executor=executor,
            retry_policy=ToolCallRetryPolicy(max_attempts=2),
        ),
    )

    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.repair_decision is not None
    assert result.repair_decision.action is ToolCallRepairAction.RETRY
    assert result.repair_decision.failure.code is ToolCallFailureCode.INVALID_JSON
    assert executor.calls == []
    rendered_result = result.model_dump_json()
    assert raw_content not in rendered_result
    assert "do-not-leak" not in rendered_result


def test_llamacpp_unknown_tool_output_does_not_execute_tools() -> None:
    raw_content = json.dumps(
        {
            "tool_calls": [
                {
                    "tool_name": "missing_tool",
                    "arguments": {"expression": "12 * 31"},
                },
            ],
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json=chat_completion(raw_content),
            request=request,
        )

    registry, executor = calculator_flow_components()
    result = asyncio.run(
        run_single_step_tool_call(
            make_provider(handler),
            model="local-chat",
            user_message="Use the missing tool.",
            registry=registry,
            executor=executor,
        ),
    )

    assert result.parsing_succeeded is False
    assert result.execution_attempted is False
    assert result.repair_decision is not None
    assert result.repair_decision.action is ToolCallRepairAction.SAFE_FAILURE
    assert result.repair_decision.failure.code is ToolCallFailureCode.UNKNOWN_TOOL
    assert executor.calls == []
    rendered_result = result.model_dump_json()
    assert raw_content not in rendered_result
    assert "missing_tool" not in rendered_result


def test_llamacpp_final_synthesis_uses_tool_result_context() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def tool_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_completion(
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
            ),
            request=request,
        )

    registry, executor = calculator_flow_components()
    step_result = asyncio.run(
        run_single_step_tool_call(
            make_provider(tool_handler),
            model="local-chat",
            user_message="What is 12 * 31?",
            registry=registry,
            executor=executor,
        ),
    )

    def answer_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = request_payload(request)
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json=chat_completion("The calculator result is 372."),
            request=request,
        )

    answer = asyncio.run(
        synthesize_tool_answer(
            make_provider(answer_handler),
            model="local-chat",
            user_message="What is 12 * 31?",
            step_result=step_result,
        ),
    )

    assert answer.final_answer == "The calculator result is 372."
    assert answer.model == "local-chat"
    assert answer.synthesis_succeeded is True
    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "local-chat"
    assert seen_payloads[0]["stream"] is False
    payload_context = json.loads(seen_payloads[0]["messages"][1]["content"])
    assert payload_context["user_message"] == "What is 12 * 31?"
    assert payload_context["tool_step"]["tool_results"] == [
        {
            "tool_name": "calculator",
            "success": True,
            "result": {"value": 372},
            "error": None,
        },
    ]


def test_llamacpp_final_synthesis_uses_safe_failure_context() -> None:
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

    def tool_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_completion(raw_content),
            request=request,
        )

    registry, executor = calculator_flow_components()
    step_result = asyncio.run(
        run_single_step_tool_call(
            make_provider(tool_handler),
            model="local-chat",
            user_message="Use the missing tool.",
            registry=registry,
            executor=executor,
        ),
    )

    seen_payloads: list[dict[str, Any]] = []

    def answer_handler(request: httpx.Request) -> httpx.Response:
        payload = request_payload(request)
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json=chat_completion("I could not safely use that tool."),
            request=request,
        )

    answer = asyncio.run(
        synthesize_tool_answer(
            make_provider(answer_handler),
            model="local-chat",
            user_message="Use the missing tool.",
            step_result=step_result,
        ),
    )

    assert answer.final_answer == "I could not safely use that tool."
    assert executor.calls == []
    rendered_payload = json.dumps(seen_payloads[0], sort_keys=True)
    payload_context = json.loads(seen_payloads[0]["messages"][1]["content"])
    repair_decision = payload_context["tool_step"]["repair_decision"]
    assert repair_decision["action"] == "safe_failure"
    assert repair_decision["failure"]["code"] == "unknown_tool"
    assert raw_content not in rendered_payload
    assert "missing_tool" not in rendered_payload
    assert "do-not-leak" not in rendered_payload
