"""Minimal provider-neutral executor for bounded agent runs."""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from aigentego.agents.contracts import (
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepStatus,
    AgentStepType,
)
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    LlmProvider,
    ToolCallParseError,
    ToolCallRetryPolicy,
    ToolCallValidationError,
    parse_structured_tool_calls,
)
from aigentego.tools import ToolRegistry, serialize_tool_definitions

AGENT_LOOP_SKELETON_STOP_REASON = "agent_loop_skeleton_has_no_step_handlers"
MAX_STEPS_REACHED_STOP_REASON = "max_steps_reached"
NO_TOOL_CALLS_GENERATED_STOP_REASON = "no_tool_calls_generated"
TOOL_CALLS_GENERATED_STOP_REASON = "tool_calls_generated_execution_not_integrated"
TOOL_CALL_GENERATION_CONFIG_ERROR = (
    "structured tool-call generation requires provider, model, and registry"
)


class AgentLoopLimits(BaseModel):
    """Safety limits accepted by the bounded agent loop skeleton."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=1, ge=0)


@dataclass(frozen=True)
class _ToolCallGenerationConfig:
    provider: LlmProvider
    model: str
    registry: ToolRegistry
    retry_policy: ToolCallRetryPolicy


class AgentLoopExecutor:
    """Create inspectable agent runs with one bounded model generation step."""

    def __init__(
        self,
        *,
        limits: AgentLoopLimits | None = None,
        provider: LlmProvider | None = None,
        model: str | None = None,
        registry: ToolRegistry | None = None,
        retry_policy: ToolCallRetryPolicy | None = None,
    ) -> None:
        self._limits = limits if limits is not None else AgentLoopLimits()
        self._provider = provider
        self._model = model
        self._registry = registry
        self._retry_policy = retry_policy if retry_policy is not None else (
            ToolCallRetryPolicy(max_attempts=1)
        )
        self._validate_generation_config(
            provider=self._provider,
            model=self._model,
            registry=self._registry,
        )

    @property
    def limits(self) -> AgentLoopLimits:
        """Return configured safety limits."""
        return self._limits

    async def run(
        self,
        user_message: str,
        *,
        run_id: str = "agent-run",
        request_id: str | None = None,
        provider: LlmProvider | None = None,
        model: str | None = None,
        registry: ToolRegistry | None = None,
        retry_policy: ToolCallRetryPolicy | None = None,
    ) -> AgentRun:
        """Start one bounded agent run and perform at most one model step."""
        generation_config = self._resolve_generation_config(
            provider=provider,
            model=model,
            registry=registry,
            retry_policy=retry_policy,
        )
        if generation_config is not None and self._limits.max_steps > 0:
            return await self._generate_tool_calls(
                generation_config,
                user_message=user_message,
                run_id=run_id,
                request_id=request_id,
            )

        return AgentRun(
            run_id=run_id,
            request_id=request_id,
            user_message=user_message,
            status=AgentRunStatus.STOPPED,
            steps=[],
            stop_reason=self._stop_reason(),
        )

    async def _generate_tool_calls(
        self,
        config: _ToolCallGenerationConfig,
        *,
        user_message: str,
        run_id: str,
        request_id: str | None,
    ) -> AgentRun:
        response = await config.provider.chat(
            _build_tool_call_generation_request(
                model=config.model,
                user_message=user_message,
                registry=config.registry,
            ),
        )

        try:
            tool_calls = parse_structured_tool_calls(
                response.message.content,
                config.registry,
            )
        except (ToolCallParseError, ToolCallValidationError) as error:
            repair_decision = config.retry_policy.decide(error, attempt=1)
            return AgentRun(
                run_id=run_id,
                request_id=request_id,
                user_message=user_message,
                status=AgentRunStatus.FAILED,
                steps=[
                    AgentStep(
                        index=0,
                        step_type=AgentStepType.MODEL,
                        status=AgentStepStatus.FAILED,
                        model_summary="structured tool-call output was invalid",
                        repair_decision=repair_decision,
                    ),
                ],
                repair_decision=repair_decision,
            )

        return AgentRun(
            run_id=run_id,
            request_id=request_id,
            user_message=user_message,
            status=AgentRunStatus.STOPPED,
            steps=[
                AgentStep(
                    index=0,
                    step_type=AgentStepType.MODEL,
                    status=AgentStepStatus.SUCCEEDED,
                    model_summary=_tool_call_generation_summary(len(tool_calls)),
                    tool_calls=tool_calls,
                ),
            ],
            stop_reason=_tool_call_generation_stop_reason(len(tool_calls)),
        )

    def _resolve_generation_config(
        self,
        *,
        provider: LlmProvider | None,
        model: str | None,
        registry: ToolRegistry | None,
        retry_policy: ToolCallRetryPolicy | None,
    ) -> _ToolCallGenerationConfig | None:
        resolved_provider = provider if provider is not None else self._provider
        resolved_model = model if model is not None else self._model
        resolved_registry = registry if registry is not None else self._registry

        self._validate_generation_config(
            provider=resolved_provider,
            model=resolved_model,
            registry=resolved_registry,
        )

        if (
            resolved_provider is None
            and resolved_model is None
            and resolved_registry is None
        ):
            return None

        if resolved_provider is None or resolved_model is None:
            raise ValueError(TOOL_CALL_GENERATION_CONFIG_ERROR)
        if resolved_registry is None:
            raise ValueError(TOOL_CALL_GENERATION_CONFIG_ERROR)

        return _ToolCallGenerationConfig(
            provider=resolved_provider,
            model=resolved_model,
            registry=resolved_registry,
            retry_policy=(
                retry_policy if retry_policy is not None else self._retry_policy
            ),
        )

    def _validate_generation_config(
        self,
        *,
        provider: LlmProvider | None,
        model: str | None,
        registry: ToolRegistry | None,
    ) -> None:
        if provider is None and model is None and registry is None:
            return
        if provider is None or model is None or registry is None:
            raise ValueError(TOOL_CALL_GENERATION_CONFIG_ERROR)
        if not model.strip():
            raise ValueError("model must not be blank")

    def _stop_reason(self) -> str:
        if self._limits.max_steps == 0:
            return MAX_STEPS_REACHED_STOP_REASON
        return AGENT_LOOP_SKELETON_STOP_REASON


async def run_agent_loop(
    user_message: str,
    *,
    run_id: str = "agent-run",
    request_id: str | None = None,
    limits: AgentLoopLimits | None = None,
    provider: LlmProvider | None = None,
    model: str | None = None,
    registry: ToolRegistry | None = None,
    retry_policy: ToolCallRetryPolicy | None = None,
) -> AgentRun:
    """Run the bounded agent loop once."""
    executor = AgentLoopExecutor(
        limits=limits,
        provider=provider,
        model=model,
        registry=registry,
        retry_policy=retry_policy,
    )
    return await executor.run(
        user_message,
        run_id=run_id,
        request_id=request_id,
    )


def _build_tool_call_generation_request(
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


def _tool_call_generation_summary(tool_call_count: int) -> str:
    if tool_call_count == 0:
        return "model produced no structured tool calls"
    if tool_call_count == 1:
        return "model produced 1 structured tool call"
    return f"model produced {tool_call_count} structured tool calls"


def _tool_call_generation_stop_reason(tool_call_count: int) -> str:
    if tool_call_count == 0:
        return NO_TOOL_CALLS_GENERATED_STOP_REASON
    return TOOL_CALLS_GENERATED_STOP_REASON


__all__ = [
    "AGENT_LOOP_SKELETON_STOP_REASON",
    "MAX_STEPS_REACHED_STOP_REASON",
    "NO_TOOL_CALLS_GENERATED_STOP_REASON",
    "TOOL_CALLS_GENERATED_STOP_REASON",
    "TOOL_CALL_GENERATION_CONFIG_ERROR",
    "AgentLoopExecutor",
    "AgentLoopLimits",
    "run_agent_loop",
]
