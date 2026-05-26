import json

import pytest
from fastapi.testclient import TestClient

from aigentego.api.dependencies import get_llm_provider, get_settings
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmConnectionError,
    ModelInfo,
)
from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER
from aigentego.settings import Settings


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


def make_client(provider: SequencedProvider) -> TestClient:
    app = create_app()
    settings = Settings(
        _env_file=None,
        llm_backend="ollama",
        llm_base_url="http://ollama:11434",
        chat_model="agent-test-model",
        embedding_model="nomic-embed-text",
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: provider

    return TestClient(app)


def calculator_tool_call_response(expression: str) -> str:
    return json.dumps(
        {
            "tool_calls": [
                {
                    "tool_name": "calculator",
                    "arguments": {"expression": expression},
                },
            ],
        },
    )


def test_agent_run_executes_calculator_and_returns_inspectable_run() -> None:
    provider = SequencedProvider(
        [
            calculator_tool_call_response("6 * 7"),
            "The answer is 42.",
        ],
    )
    client = make_client(provider)

    response = client.post(
        "/agent/run",
        headers={REQUEST_ID_HEADER: "agent-request-123"},
        json={"message": " What is 6 * 7? "},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "agent-request-123"
    body = response.json()
    assert body["run_id"] == "agent-request-123"
    assert body["request_id"] == "agent-request-123"
    assert body["user_message"] == "What is 6 * 7?"
    assert body["status"] == "succeeded"
    assert body["final_answer"] == "The answer is 42."
    assert [step["step_type"] for step in body["steps"]] == [
        "model",
        "tool",
        "final",
    ]
    assert body["steps"][1]["tool_call"] == {
        "tool_name": "calculator",
        "arguments": {"expression": "6 * 7"},
    }
    assert body["steps"][1]["tool_result"] == {
        "tool_name": "calculator",
        "success": True,
        "result": {"value": 42},
        "error": None,
    }
    assert body["steps"][1]["observation"]["request_id"] == "agent-request-123"
    assert provider.chat_calls == 2


def test_agent_run_uses_configured_model_and_calculator_tool_context() -> None:
    provider = SequencedProvider(
        [
            calculator_tool_call_response("12 / 3"),
            "The answer is 4.",
        ],
    )
    client = make_client(provider)

    response = client.post("/agent/run", json={"message": "Calculate 12 / 3."})

    assert response.status_code == 200
    assert [request.model for request in provider.chat_requests] == [
        "agent-test-model",
        "agent-test-model",
    ]
    generation_request = provider.chat_requests[0]
    assert [message.role for message in generation_request.messages] == [
        "system",
        "user",
    ]
    assert "tool_calls" in generation_request.messages[0].content
    prompt_payload = json.loads(generation_request.messages[1].content)
    assert prompt_payload["user_message"] == "Calculate 12 / 3."
    assert [tool["name"] for tool in prompt_payload["tools"]] == ["calculator"]
    assert prompt_payload["expected_output"] == {
        "tool_calls": [
            {
                "tool_name": "registered_tool_name",
                "arguments": {},
            },
        ],
    }


def test_agent_run_rejects_blank_message() -> None:
    provider = SequencedProvider([])
    client = make_client(provider)

    response = client.post("/agent/run", json={"message": "   "})

    assert response.status_code == 422
    assert provider.chat_calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Run.", "max_steps": -1},
        {"message": "Run.", "max_tool_errors": -1},
        {"message": "Run.", "timeout_seconds": 0},
        {"message": "Run.", "unexpected": "field"},
    ],
)
def test_agent_run_rejects_invalid_limit_payloads(
    payload: dict[str, object],
) -> None:
    provider = SequencedProvider([])
    client = make_client(provider)

    response = client.post("/agent/run", json=payload)

    assert response.status_code == 422
    assert provider.chat_calls == 0


def test_agent_run_safe_failure_returns_inspectable_stopped_run() -> None:
    raw_output = '{"tool_calls": [{"arguments": {"secret": "do-not-leak"}}'
    provider = SequencedProvider([raw_output, "unused final answer"])
    client = make_client(provider)

    response = client.post(
        "/agent/run",
        json={"message": "Use a tool safely.", "max_steps": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stopped"
    assert body["stop_reason"] == "max_steps_reached"
    assert body["repair_decision"]["failure"]["code"] == "invalid_json"
    assert [step["step_type"] for step in body["steps"]] == ["model"]
    assert body["steps"][0]["status"] == "failed"
    assert body["steps"][0]["repair_decision"]["failure"]["code"] == "invalid_json"
    assert provider.chat_calls == 1
    rendered = json.dumps(body)
    assert raw_output not in rendered
    assert "do-not-leak" not in rendered


def test_agent_run_provider_generation_error_returns_failed_run() -> None:
    provider = SequencedProvider(
        [
            LlmConnectionError(
                "request failed",
                provider="fake",
                operation="chat",
            ),
        ],
    )
    client = make_client(provider)

    response = client.post("/agent/run", json={"message": "Use a tool."})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_detail"].startswith("structured tool-call generation failed")
    assert [step["step_type"] for step in body["steps"]] == ["model"]
    assert body["steps"][0]["status"] == "failed"
    assert body["steps"][0]["error_detail"] == body["error_detail"]
    assert provider.chat_calls == 1
