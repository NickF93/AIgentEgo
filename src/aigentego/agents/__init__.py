"""Provider-neutral agent run contracts and execution skeleton."""

from aigentego.agents.contracts import (
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepStatus,
    AgentStepType,
)
from aigentego.agents.executor import (
    AGENT_LOOP_SKELETON_STOP_REASON,
    FINAL_ANSWER_SYNTHESIS_FAILED_ERROR_DETAIL,
    FINAL_ANSWER_SYNTHESIS_SUMMARY,
    MAX_STEPS_REACHED_STOP_REASON,
    MAX_TOOL_ERRORS_REACHED_STOP_REASON,
    NO_TOOL_CALLS_GENERATED_STOP_REASON,
    TIMEOUT_REACHED_STOP_REASON,
    TOOL_CALL_GENERATION_CONFIG_ERROR,
    TOOL_CALL_GENERATION_FAILED_ERROR_DETAIL,
    TOOL_CALLS_GENERATED_STOP_REASON,
    TOOL_EXECUTION_COMPLETED_STOP_REASON,
    AgentLoopExecutor,
    AgentLoopLimits,
    run_agent_loop,
)

__all__ = [
    "AGENT_LOOP_SKELETON_STOP_REASON",
    "AgentRun",
    "AgentRunStatus",
    "AgentLoopExecutor",
    "AgentLoopLimits",
    "AgentStep",
    "AgentStepStatus",
    "AgentStepType",
    "FINAL_ANSWER_SYNTHESIS_FAILED_ERROR_DETAIL",
    "FINAL_ANSWER_SYNTHESIS_SUMMARY",
    "MAX_STEPS_REACHED_STOP_REASON",
    "MAX_TOOL_ERRORS_REACHED_STOP_REASON",
    "NO_TOOL_CALLS_GENERATED_STOP_REASON",
    "TIMEOUT_REACHED_STOP_REASON",
    "TOOL_CALLS_GENERATED_STOP_REASON",
    "TOOL_EXECUTION_COMPLETED_STOP_REASON",
    "TOOL_CALL_GENERATION_CONFIG_ERROR",
    "TOOL_CALL_GENERATION_FAILED_ERROR_DETAIL",
    "run_agent_loop",
]
