"""Request tracing middleware and helpers."""

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Final
from uuid import uuid4

from fastapi import Request, Response

REQUEST_ID_HEADER: Final = "X-Request-ID"
_REQUEST_ID_STATE_KEY: Final = "request_id"

RequestHandler = Callable[[Request], Awaitable[Response]]

logger = logging.getLogger(__name__)


async def request_tracing_middleware(
    request: Request,
    call_next: RequestHandler,
) -> Response:
    """Attach a request id to request state, response headers, and logs."""
    request_id = _request_id_from_headers(request)
    setattr(request.state, _REQUEST_ID_STATE_KEY, request_id)

    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = _elapsed_ms(started_at)
        logger.exception(
            "http_request method=%s path=%s status_code=500 duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
            extra={"request_id": request_id},
        )
        raise

    duration_ms = _elapsed_ms(started_at)
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.log(
        _level_for_status(response.status_code),
        "http_request method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id},
    )
    return response


def get_request_id(request: Request) -> str:
    """Return the current request id, creating one only as a fallback."""
    request_id = getattr(request.state, _REQUEST_ID_STATE_KEY, None)
    if isinstance(request_id, str) and request_id:
        return request_id

    request_id = str(uuid4())
    setattr(request.state, _REQUEST_ID_STATE_KEY, request_id)
    return request_id


def _request_id_from_headers(request: Request) -> str:
    incoming_request_id = request.headers.get(REQUEST_ID_HEADER)
    if incoming_request_id:
        stripped_request_id = incoming_request_id.strip()
        if stripped_request_id:
            return stripped_request_id
    return str(uuid4())


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _level_for_status(status_code: int) -> int:
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    return logging.INFO
