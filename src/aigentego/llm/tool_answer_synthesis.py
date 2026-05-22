"""Final answer synthesis for single-step tool-call results."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from aigentego.llm.base import ChatMessage, ChatRequest, LlmProvider
from aigentego.llm.tool_call_flow import SingleStepToolCallResult


class ToolAnswerSynthesisResult(BaseModel):
    """Normalized final answer synthesized from one tool-call step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    final_answer: str
    model: str
    synthesis_succeeded: bool


async def synthesize_tool_answer(
    provider: LlmProvider,
    *,
    model: str,
    user_message: str,
    step_result: SingleStepToolCallResult,
) -> ToolAnswerSynthesisResult:
    """Ask the provider once to synthesize a final answer from step output."""
    provider_response = await provider.chat(
        _build_final_answer_request(
            model=model,
            user_message=user_message,
            step_result=step_result,
        ),
    )
    return ToolAnswerSynthesisResult(
        final_answer=provider_response.message.content,
        model=provider_response.model,
        synthesis_succeeded=True,
    )


def _build_final_answer_request(
    *,
    model: str,
    user_message: str,
    step_result: SingleStepToolCallResult,
) -> ChatRequest:
    context = _build_synthesis_context(
        user_message=user_message,
        step_result=step_result,
    )
    return ChatRequest(
        model=model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Synthesize a final natural-language answer for the user "
                    "from the provider-neutral JSON context. Use tool results "
                    "when present. If structured tool-call parsing failed, use "
                    "only the safe failure details. Do not request or execute "
                    "tools."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    context,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ],
    )


def _build_synthesis_context(
    *,
    user_message: str,
    step_result: SingleStepToolCallResult,
) -> dict[str, Any]:
    repair_decision = (
        step_result.repair_decision.model_dump(mode="json")
        if step_result.repair_decision is not None
        else None
    )
    return {
        "user_message": user_message,
        "tool_step": {
            "parsing_succeeded": step_result.parsing_succeeded,
            "execution_attempted": step_result.execution_attempted,
            "tool_calls": [
                tool_call.model_dump(mode="json")
                for tool_call in step_result.tool_calls
            ],
            "tool_results": [
                tool_result.model_dump(mode="json")
                for tool_result in step_result.tool_results
            ],
            "repair_decision": repair_decision,
        },
    }
