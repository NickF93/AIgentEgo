"""Provider-neutral calendar domain contracts."""

from aigentego.calendar.contracts import (
    CalendarAvailability,
    CalendarEvent,
    CalendarQuery,
    CalendarQueryResult,
    CalendarTimeRange,
)
from aigentego.calendar.errors import (
    CalendarConfigurationError,
    CalendarError,
    CalendarErrorDetail,
    CalendarNotFoundError,
    CalendarPermissionError,
    CalendarQueryError,
    CalendarWriteNotAllowedError,
)

__all__ = [
    "CalendarAvailability",
    "CalendarConfigurationError",
    "CalendarError",
    "CalendarErrorDetail",
    "CalendarEvent",
    "CalendarNotFoundError",
    "CalendarPermissionError",
    "CalendarQuery",
    "CalendarQueryError",
    "CalendarQueryResult",
    "CalendarTimeRange",
    "CalendarWriteNotAllowedError",
]
