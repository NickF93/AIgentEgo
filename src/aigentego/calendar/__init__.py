"""Provider-neutral calendar domain contracts."""

from aigentego.calendar.adapters import CalendarAdapter, FakeCalendarAdapter
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
    "CalendarAdapter",
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
    "FakeCalendarAdapter",
]
