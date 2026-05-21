"""Single-step LLM structured tool-call flow."""

import json

from pydantic import BaseModel, ConfigDict, Field

from aigentego.llm.base import ChatMessage, ChatRequest, LlmProvider
from aigentego.llm.errors import ToolCallParseError, ToolCallValidationError
from aigentego.llm.tool_call_parser import parse_structured_tool_calls
from aigentego.llm.tool_call_repair import (
    ToolCallRepairDecision,
    ToolCallRetryPolicy,
)
from aigentego.tools import (
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    serialize_tool_definitions,
)


class SingleStepToolCallResult(BaseModel):
    """Intermediate result from one structured tool-call LLM step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parsing_succeeded: bool
    execution_attempted: bool
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    repair_decision: ToolCallRepairDecision | None = None


async def run_single_step_tool_call(
    provider: LlmProvider,
    *,
    model: str,
    user_message: str,
    registry: ToolRegistry,
    executor: ToolExecutor,
    request_id: str | None = None,
    retry_policy: ToolCallRetryPolicy | None = None,
) -> SingleStepToolCallResult:
    """Request structured tool calls once and execute validated calls."""
    policy = retry_policy if retry_policy is not None else ToolCallRetryPolicy(
        max_attempts=1,
    )
    provider_response = await provider.chat(
        _build_tool_call_request(
            model=model,
            user_message=user_message,
            registry=registry,
        ),
    )

    try:
        tool_calls = parse_structured_tool_calls(
            provider_response.message.content,
            registry,
        )
    except (ToolCallParseError, ToolCallValidationError) as error:
        return SingleStepToolCallResult(
            parsing_succeeded=False,
            execution_attempted=False,
            repair_decision=policy.decide(error, attempt=1),
        )

    tool_results: list[ToolResult] = []
    context = ToolContext(request_id=request_id)
    for tool_call in tool_calls:
        tool_results.append(await executor.execute(tool_call, context))

    return SingleStepToolCallResult(
        parsing_succeeded=True,
        execution_attempted=bool(tool_calls),
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def _build_tool_call_request(
    *,
    model: str,
    user_message: str,
    registry: ToolRegistry,
) -> ChatRequest:
    tool_context = {
        "user_message": user_message,
        "tools": serialize_tool_definitions(registry.list_definitions()),
        "expected_output": {
            "tool_calls": [
                {
                    "tool_name": "registered_tool_name",
                    "arguments": {},
                },
            ],
        },
    }
    return ChatRequest(
        model=model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Return only a JSON object matching the structured tool-call "
                    'shape {"tool_calls": [{"tool_name": string, '
                    '"arguments": object}]}. Use an empty tool_calls array when '
                    "no tool should be called."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    tool_context,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ],
    )
