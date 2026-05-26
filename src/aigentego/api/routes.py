"""Runtime API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aigentego import __version__
from aigentego.agents import AgentLoopExecutor, AgentRun
from aigentego.api.dependencies import (
    get_llm_provider,
    get_settings,
    get_tool_executor,
    get_tool_registry,
)
from aigentego.api.schemas import (
    AgentRunApiRequest,
    ApiErrorResponse,
    ChatApiRequest,
    ChatApiResponse,
    DiagnosticsResponse,
    HealthResponse,
    ToolExecuteApiRequest,
    ToolExecuteApiResponse,
    ToolListApiResponse,
)
from aigentego.llm import (
    ChatMessage,
    ChatRequest,
    LlmConnectionError,
    LlmProvider,
    LlmProviderError,
    LlmResponseError,
    LlmTimeoutError,
)
from aigentego.observability import get_request_id
from aigentego.settings import Settings
from aigentego.tools import ToolCall, ToolContext, ToolExecutor, ToolRegistry

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> HealthResponse:
    """Return API health and LLM provider reachability."""
    provider_reachable = await _provider_reachable(provider)
    return HealthResponse(
        status="ok",
        provider_reachable=provider_reachable,
        chat_model=settings.chat_model,
        ollama_reachable=provider_reachable,
    )


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> DiagnosticsResponse:
    """Return non-secret runtime diagnostics."""
    provider_reachable = await _provider_reachable(provider)
    return DiagnosticsResponse(
        status="ok",
        package_version=__version__,
        llm_backend=settings.llm_backend,
        llm_provider=provider.provider_name,
        provider_base_url=settings.llm_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        provider_reachable=provider_reachable,
        ollama_base_url=settings.llm_base_url,
        ollama_reachable=provider_reachable,
    )


@router.post(
    "/chat",
    response_model=ChatApiResponse,
    responses={
        status.HTTP_502_BAD_GATEWAY: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
    },
)
async def chat(
    request: Request,
    payload: ChatApiRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> ChatApiResponse:
    """Generate a non-streaming response for a single user message."""
    request_id = get_request_id(request)
    provider_request = ChatRequest(
        model=settings.chat_model,
        messages=[ChatMessage(role="user", content=payload.message)],
    )

    try:
        provider_response = await provider.chat(provider_request)
    except (LlmConnectionError, LlmTimeoutError) as exc:
        raise _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="llm_unavailable",
            message=str(exc),
        ) from exc
    except LlmResponseError as exc:
        raise _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error="llm_bad_response",
            message=str(exc),
        ) from exc
    except LlmProviderError as exc:
        raise _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error="llm_error",
            message=str(exc),
        ) from exc

    return ChatApiResponse(
        request_id=request_id,
        model=provider_response.model,
        message=provider_response.message.content,
    )


@router.post("/agent/run", response_model=AgentRun)
async def run_agent(
    request: Request,
    payload: AgentRunApiRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    executor: Annotated[ToolExecutor, Depends(get_tool_executor)],
) -> AgentRun:
    """Run one bounded, inspectable agent pass."""
    request_id = get_request_id(request)
    agent_executor = AgentLoopExecutor(
        limits=payload.to_limits(),
        provider=provider,
        model=settings.chat_model,
        registry=registry,
        tool_executor=executor,
    )
    return await agent_executor.run(
        payload.message,
        run_id=request_id,
        request_id=request_id,
    )


@router.get("/tools", response_model=ToolListApiResponse)
async def list_tools(
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> ToolListApiResponse:
    """List registered deterministic tools."""
    return ToolListApiResponse(tools=registry.list_definitions())


@router.post("/tools/execute", response_model=ToolExecuteApiResponse)
async def execute_tool(
    request: Request,
    payload: ToolExecuteApiRequest,
    executor: Annotated[ToolExecutor, Depends(get_tool_executor)],
) -> ToolExecuteApiResponse:
    """Execute an explicitly requested deterministic tool."""
    request_id = get_request_id(request)
    tool_result = await executor.execute(
        ToolCall(tool_name=payload.tool_name, arguments=payload.arguments),
        ToolContext(request_id=request_id),
    )

    return ToolExecuteApiResponse(
        request_id=request_id,
        tool_name=tool_result.tool_name,
        success=tool_result.success,
        result=tool_result.result,
        error=tool_result.error,
    )


async def _provider_reachable(provider: LlmProvider) -> bool:
    try:
        return await provider.health()
    except LlmProviderError:
        return False


def _api_error(*, status_code: int, error: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ApiErrorResponse(error=error, message=message).model_dump(),
    )
