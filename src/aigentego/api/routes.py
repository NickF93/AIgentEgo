"""Runtime API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aigentego import __version__
from aigentego.agents import AgentLoopExecutor, AgentRun
from aigentego.api.dependencies import (
    get_llm_provider,
    get_message_store,
    get_settings,
    get_tool_executor,
    get_tool_registry,
)
from aigentego.api.schemas import (
    AgentRunApiRequest,
    ApiErrorResponse,
    ChatApiRequest,
    ChatApiResponse,
    ConversationApiResponse,
    ConversationCreateApiRequest,
    ConversationListApiResponse,
    DiagnosticsResponse,
    HealthResponse,
    SessionApiResponse,
    SessionCreateApiRequest,
    SessionListApiResponse,
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
from aigentego.persistence import Conversation, MessageStore, Session
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


@router.post("/sessions", response_model=SessionApiResponse)
async def create_session(
    request: Request,
    payload: SessionCreateApiRequest,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> SessionApiResponse:
    """Create or update an explicit local session."""
    request_id = get_request_id(request)
    session = store.upsert_session(
        Session(
            session_id=payload.session_id,
            title=payload.title,
        ),
    )
    return SessionApiResponse(request_id=request_id, session=session)


@router.get("/sessions/{session_id}", response_model=SessionApiResponse)
async def get_session(
    request: Request,
    session_id: str,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> SessionApiResponse:
    """Return one persisted local session."""
    request_id = get_request_id(request)
    session = store.get_session(session_id)
    if session is None:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="session_not_found",
            message="session not found",
        )
    return SessionApiResponse(request_id=request_id, session=session)


@router.get("/sessions", response_model=SessionListApiResponse)
async def list_sessions(
    request: Request,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> SessionListApiResponse:
    """List persisted local sessions."""
    request_id = get_request_id(request)
    return SessionListApiResponse(
        request_id=request_id,
        sessions=store.list_sessions(),
    )


@router.post(
    "/sessions/{session_id}/conversations",
    response_model=ConversationApiResponse,
)
async def create_conversation(
    request: Request,
    session_id: str,
    payload: ConversationCreateApiRequest,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> ConversationApiResponse:
    """Create or update an explicit local conversation under a session."""
    request_id = get_request_id(request)
    if store.get_session(session_id) is None:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="session_not_found",
            message="session not found",
        )
    conversation = store.upsert_conversation(
        Conversation(
            conversation_id=payload.conversation_id,
            session_id=session_id,
            title=payload.title,
            is_default=payload.is_default,
        ),
    )
    return ConversationApiResponse(
        request_id=request_id,
        conversation=conversation,
    )


@router.get(
    "/sessions/{session_id}/conversations",
    response_model=ConversationListApiResponse,
)
async def list_session_conversations(
    request: Request,
    session_id: str,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> ConversationListApiResponse:
    """List persisted local conversations for one session."""
    request_id = get_request_id(request)
    if store.get_session(session_id) is None:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="session_not_found",
            message="session not found",
        )
    return ConversationListApiResponse(
        request_id=request_id,
        conversations=store.list_conversations(session_id),
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationApiResponse,
)
async def get_conversation(
    request: Request,
    conversation_id: str,
    store: Annotated[MessageStore, Depends(get_message_store)],
) -> ConversationApiResponse:
    """Return one persisted local conversation."""
    request_id = get_request_id(request)
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="conversation_not_found",
            message="conversation not found",
        )
    return ConversationApiResponse(
        request_id=request_id,
        conversation=conversation,
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
