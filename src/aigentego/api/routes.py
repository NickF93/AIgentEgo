"""Runtime API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aigentego import __version__
from aigentego.api.dependencies import get_llm_provider, get_settings
from aigentego.api.schemas import (
    ApiErrorResponse,
    ChatApiRequest,
    ChatApiResponse,
    DiagnosticsResponse,
    HealthResponse,
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

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> HealthResponse:
    """Return API health and LLM provider reachability."""
    return HealthResponse(
        status="ok",
        ollama_reachable=await _provider_reachable(provider),
        chat_model=settings.ollama_chat_model,
    )


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> DiagnosticsResponse:
    """Return non-secret runtime diagnostics."""
    return DiagnosticsResponse(
        status="ok",
        package_version=__version__,
        llm_provider=provider.provider_name,
        ollama_base_url=settings.ollama_base_url,
        chat_model=settings.ollama_chat_model,
        embedding_model=settings.ollama_embed_model,
        ollama_reachable=await _provider_reachable(provider),
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
        model=settings.ollama_chat_model,
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
