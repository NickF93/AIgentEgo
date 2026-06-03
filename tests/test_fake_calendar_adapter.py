from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aigentego.calendar import (
    CalendarConfigurationError,
    CalendarEvent,
    CalendarQuery,
    CalendarQueryError,
    CalendarTimeRange,
    FakeCalendarAdapter,
)


def timestamp(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 3, hour, minute, tzinfo=UTC)


def time_range(
    start_time: datetime,
    end_time: datetime,
) -> CalendarTimeRange:
    return CalendarTimeRange(
        start_time=start_time,
        end_time=end_time,
        timezone="UTC",
    )


def event(
    event_id: str,
    start_time: datetime,
    end_time: datetime,
    *,
    calendar_id: str = "work",
) -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title=f"Event {event_id}",
        time_range=time_range(start_time, end_time),
        calendar_id=calendar_id,
    )


def query(
    start_time: datetime = timestamp(9),
    end_time: datetime = timestamp(12),
    *,
    calendar_ids: tuple[str, ...] = (),
    limit: int = 50,
) -> CalendarQuery:
    return CalendarQuery(
        time_range=time_range(start_time, end_time),
        calendar_ids=calendar_ids,
        limit=limit,
    )


def test_empty_adapter_returns_empty_query_result() -> None:
    calendar_query = query()

    result = FakeCalendarAdapter().query_events(calendar_query)

    assert result.query == calendar_query
    assert result.events == ()


def test_adapter_returns_events_inside_query_range() -> None:
    inside = event("inside", timestamp(10), timestamp(11))
    before = event("before", timestamp(7), timestamp(8))
    after = event("after", timestamp(13), timestamp(14))

    result = FakeCalendarAdapter((before, after, inside)).query_events(query())

    assert result.events == (inside,)


def test_adapter_includes_overlapping_events() -> None:
    overlaps_start = event("overlaps-start", timestamp(8), timestamp(9, 30))
    overlaps_end = event("overlaps-end", timestamp(11, 30), timestamp(13))
    encloses = event("encloses", timestamp(8), timestamp(13))

    result = FakeCalendarAdapter(
        (overlaps_end, encloses, overlaps_start),
    ).query_events(query())

    assert [calendar_event.event_id for calendar_event in result.events] == [
        "overlaps-start",
        "encloses",
        "overlaps-end",
    ]


def test_adapter_uses_half_open_boundary_behavior() -> None:
    ends_at_query_start = event("ends-at-start", timestamp(8), timestamp(9))
    starts_at_query_end = event("starts-at-end", timestamp(12), timestamp(13))
    touches_inside = event("touches-inside", timestamp(9), timestamp(12))

    result = FakeCalendarAdapter(
        (starts_at_query_end, touches_inside, ends_at_query_start),
    ).query_events(query())

    assert result.events == (touches_inside,)


def test_adapter_filters_by_calendar_id() -> None:
    work = event("work", timestamp(10), timestamp(11), calendar_id="work")
    personal = event("personal", timestamp(10), timestamp(11), calendar_id="personal")
    uncategorized = CalendarEvent(
        event_id="uncategorized",
        title="Uncategorized",
        time_range=time_range(timestamp(10), timestamp(11)),
    )

    result = FakeCalendarAdapter((personal, uncategorized, work)).query_events(
        query(calendar_ids=("work",)),
    )

    assert result.events == (work,)


def test_adapter_applies_limit_after_deterministic_ordering() -> None:
    later = event("later", timestamp(10), timestamp(11))
    earlier = event("earlier", timestamp(9), timestamp(10))
    middle = event("middle", timestamp(9, 30), timestamp(10, 30))

    result = FakeCalendarAdapter((later, middle, earlier)).query_events(
        query(limit=2),
    )

    assert result.events == (earlier, middle)


def test_adapter_orders_by_start_time_end_time_and_event_id() -> None:
    second = event("b", timestamp(10), timestamp(10, 30))
    third = event("a", timestamp(10), timestamp(11))
    first = event("a", timestamp(9), timestamp(11))
    fourth = event("c", timestamp(10), timestamp(11))

    result = FakeCalendarAdapter((fourth, third, second, first)).query_events(query())

    assert result.events == (first, second, third, fourth)


def test_adapter_wraps_invalid_fixture_data() -> None:
    invalid_event_data = {
        "event_id": "",
        "title": "Invalid",
        "time_range": time_range(timestamp(10), timestamp(11)),
    }

    with pytest.raises(CalendarConfigurationError) as exc_info:
        FakeCalendarAdapter((invalid_event_data,))  # type: ignore[arg-type]

    detail = exc_info.value.to_detail()
    assert detail.code == "calendar_configuration_error"
    assert detail.message == "invalid fake calendar event"
    assert detail.context == {"event_index": "0"}


def test_adapter_wraps_invalid_query_data() -> None:
    adapter = FakeCalendarAdapter()
    invalid_query = {
        "time_range": time_range(timestamp(9), timestamp(10)),
        "limit": 0,
    }

    with pytest.raises(CalendarQueryError) as exc_info:
        adapter.query_events(invalid_query)  # type: ignore[arg-type]

    assert exc_info.value.to_detail().code == "calendar_query_error"
    assert exc_info.value.to_detail().message == "invalid calendar query"
    assert exc_info.value.to_detail().context == {}


def test_adapter_does_not_mutate_supplied_event_objects() -> None:
    calendar_event = event("event-1", timestamp(10), timestamp(11))
    adapter = FakeCalendarAdapter((calendar_event,))

    result = adapter.query_events(query())

    assert result.events == (calendar_event,)
    with pytest.raises(ValidationError):
        result.events[0].title = "Changed"  # type: ignore[misc]
