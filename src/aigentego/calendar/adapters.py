"""Provider-neutral calendar adapters."""

from typing import Protocol

from pydantic import ValidationError

from aigentego.calendar.contracts import (
    CalendarEvent,
    CalendarQuery,
    CalendarQueryResult,
)
from aigentego.calendar.errors import (
    CalendarConfigurationError,
    CalendarQueryError,
)


class CalendarAdapter(Protocol):
    """Read-only provider-neutral calendar adapter interface."""

    def query_events(self, query: CalendarQuery) -> CalendarQueryResult:
        """Return events matching a provider-neutral calendar query."""


class FakeCalendarAdapter:
    """Deterministic in-memory calendar adapter for local tests."""

    def __init__(self, events: tuple[CalendarEvent, ...] = ()) -> None:
        self._events = _validate_events(events)

    def query_events(self, query: CalendarQuery) -> CalendarQueryResult:
        """Return matching events without external calendar access."""
        calendar_query = _validate_query(query)
        filtered = [
            event
            for event in self._events
            if _matches_query(event, calendar_query)
        ]
        filtered.sort(key=_event_sort_key)
        return CalendarQueryResult(
            query=calendar_query,
            events=tuple(filtered[: calendar_query.limit]),
        )


def _validate_events(events: tuple[CalendarEvent, ...]) -> tuple[CalendarEvent, ...]:
    validated: list[CalendarEvent] = []
    for index, event in enumerate(events):
        try:
            validated.append(CalendarEvent.model_validate(event))
        except ValidationError as exc:
            raise CalendarConfigurationError(
                "invalid fake calendar event",
                context={"event_index": str(index)},
            ) from exc
    return tuple(validated)


def _validate_query(query: CalendarQuery) -> CalendarQuery:
    try:
        return CalendarQuery.model_validate(query)
    except ValidationError as exc:
        raise CalendarQueryError("invalid calendar query") from exc


def _matches_query(event: CalendarEvent, query: CalendarQuery) -> bool:
    if query.calendar_ids and event.calendar_id not in query.calendar_ids:
        return False
    return (
        event.time_range.start_time < query.time_range.end_time
        and event.time_range.end_time > query.time_range.start_time
    )


def _event_sort_key(event: CalendarEvent) -> tuple[object, object, str]:
    return (
        event.time_range.start_time,
        event.time_range.end_time,
        event.event_id,
    )


__all__ = [
    "CalendarAdapter",
    "FakeCalendarAdapter",
]
