"""Standard-library logging setup for the runtime API."""

import logging
from typing import Final

_HANDLER_MARKER: Final = "_aigentego_handler"
_LOG_FORMAT: Final = (
    "%(asctime)s %(levelname)s %(name)s "
    "request_id=%(request_id)s %(message)s"
)
_DEFAULT_REQUEST_ID: Final = "-"


class RequestIdFilter(logging.Filter):
    """Ensure log records have a request_id field for formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            setattr(record, "request_id", _DEFAULT_REQUEST_ID)
        return True


def configure_logging(log_level: str) -> None:
    """Configure root logging once and update the level on later calls."""
    level = _resolve_log_level(log_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = _find_handler(root_logger)
    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        handler.addFilter(RequestIdFilter())
        root_logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))


def _find_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if bool(getattr(handler, _HANDLER_MARKER, False)):
            return handler
    return None


def _resolve_log_level(log_level: str) -> int:
    resolved_level = logging.getLevelName(log_level.upper())
    if isinstance(resolved_level, int):
        return resolved_level
    return logging.INFO
