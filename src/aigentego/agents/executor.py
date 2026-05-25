"""Minimal provider-neutral skeleton for bounded agent runs."""

from pydantic import BaseModel, ConfigDict, Field

from aigentego.agents.contracts import AgentRun, AgentRunStatus

AGENT_LOOP_SKELETON_STOP_REASON = "agent_loop_skeleton_has_no_step_handlers"
MAX_STEPS_REACHED_STOP_REASON = "max_steps_reached"


class AgentLoopLimits(BaseModel):
    """Safety limits accepted by the bounded agent loop skeleton."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=1, ge=0)


class AgentLoopExecutor:
    """Create inspectable agent runs without step handlers yet."""

    def __init__(self, *, limits: AgentLoopLimits | None = None) -> None:
        self._limits = limits if limits is not None else AgentLoopLimits()

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
    ) -> AgentRun:
        """Start one bounded agent run and stop before any real step execution."""
        return AgentRun(
            run_id=run_id,
            request_id=request_id,
            user_message=user_message,
            status=AgentRunStatus.STOPPED,
            steps=[],
            stop_reason=self._stop_reason(),
        )

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
) -> AgentRun:
    """Run the minimal bounded agent loop skeleton once."""
    executor = AgentLoopExecutor(limits=limits)
    return await executor.run(
        user_message,
        run_id=run_id,
        request_id=request_id,
    )


__all__ = [
    "AGENT_LOOP_SKELETON_STOP_REASON",
    "MAX_STEPS_REACHED_STOP_REASON",
    "AgentLoopExecutor",
    "AgentLoopLimits",
    "run_agent_loop",
]
