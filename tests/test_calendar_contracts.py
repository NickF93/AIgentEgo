import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aigentego.calendar import (
    CalendarAvailability,
    CalendarEvent,
    CalendarQuery,
    CalendarQueryResult,
    CalendarTimeRange,
)


def timestamp() -> datetime:
    return datetime(2026, 6, 3, 9, 0, tzinfo=UTC)


def time_range() -> CalendarTimeRange:
    start_time = timestamp()
    return CalendarTimeRange(
        start_time=start_time,
        end_time=start_time + timedelta(hours=1),
        timezone="UTC",
    )


def event(event_id: str = "event-123") -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title="Planning review",
        time_range=time_range(),
        calendar_id="work",
        location="Office",
        description="Review implementation plan.",
        attendees=("alice@example.test", "bob@example.test"),
        availability=CalendarAvailability.BUSY,
    )


def test_valid_calendar_event_can_be_created() -> None:
    calendar_event = event()

    assert calendar_event.event_id == "event-123"
    assert calendar_event.title == "Planning review"
    assert calendar_event.time_range.start_time == timestamp()
    assert calendar_event.time_range.end_time == timestamp() + timedelta(hours=1)
    assert calendar_event.time_range.timezone == "UTC"
    assert calendar_event.calendar_id == "work"
    assert calendar_event.attendees == ("alice@example.test", "bob@example.test")
    assert calendar_event.availability is CalendarAvailability.BUSY


@pytest.mark.parametrize(
    "payload",
    [
        {"event_id": "", "title": "Title", "time_range": time_range()},
        {"event_id": "   ", "title": "Title", "time_range": time_range()},
        {"event_id": "event-123", "title": "", "time_range": time_range()},
        {"event_id": "event-123", "title": "   ", "time_range": time_range()},
        {
            "event_id": "event-123",
            "title": "Title",
            "time_range": time_range(),
            "calendar_id": "   ",
        },
        {
            "event_id": "event-123",
            "title": "Title",
            "time_range": time_range(),
            "attendees": ("   ",),
        },
    ],
)
def test_calendar_event_rejects_blank_text(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CalendarEvent.model_validate(payload)


def test_time_range_rejects_invalid_order() -> None:
    start_time = timestamp()

    with pytest.raises(ValidationError, match="end_time must be after start_time"):
        CalendarTimeRange(
            start_time=start_time,
            end_time=start_time,
        )

    with pytest.raises(ValidationError, match="end_time must be after start_time"):
        CalendarTimeRange(
            start_time=start_time,
            end_time=start_time - timedelta(seconds=1),
        )


def test_timezone_policy_is_enforced() -> None:
    naive_timestamp = datetime(2026, 6, 3, 9, 0)

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        CalendarTimeRange(
            start_time=naive_timestamp,
            end_time=timestamp() + timedelta(hours=1),
        )

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        CalendarTimeRange(
            start_time=timestamp(),
            end_time=naive_timestamp,
        )

    with pytest.raises(ValidationError, match="text fields must not be blank"):
        CalendarTimeRange(
            start_time=timestamp(),
            end_time=timestamp() + timedelta(hours=1),
            timezone="   ",
        )


def test_query_model_validates_time_range_and_limit() -> None:
    query = CalendarQuery(
        time_range=time_range(),
        calendar_ids=("work", "personal"),
        limit=2,
    )

    assert query.time_range == time_range()
    assert query.calendar_ids == ("work", "personal")
    assert query.limit == 2

    with pytest.raises(ValidationError):
        CalendarQuery(time_range=time_range(), calendar_ids=("   ",))

    with pytest.raises(ValidationError):
        CalendarQuery(time_range=time_range(), limit=0)

    with pytest.raises(ValidationError, match="limit must be an integer"):
        CalendarQuery(time_range=time_range(), limit=True)


def test_query_result_preserves_deterministic_ordering() -> None:
    first = event("event-1")
    second = event("event-2")
    query = CalendarQuery(time_range=time_range(), limit=2)

    result = CalendarQueryResult(query=query, events=(second, first))

    assert result.events == (second, first)
    assert [calendar_event.event_id for calendar_event in result.events] == [
        "event-2",
        "event-1",
    ]


def test_query_result_rejects_more_events_than_limit() -> None:
    query = CalendarQuery(time_range=time_range(), limit=1)

    with pytest.raises(ValidationError, match="events must not exceed query limit"):
        CalendarQueryResult(
            query=query,
            events=(event("event-1"), event("event-2")),
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CalendarTimeRange.model_validate(
            {
                "start_time": timestamp(),
                "end_time": timestamp() + timedelta(hours=1),
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        CalendarEvent.model_validate(
            {
                "event_id": "event-123",
                "title": "Title",
                "time_range": time_range(),
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        CalendarQuery.model_validate(
            {
                "time_range": time_range(),
                "unexpected": "field",
            },
        )

    with pytest.raises(ValidationError):
        CalendarQueryResult.model_validate(
            {
                "query": CalendarQuery(time_range=time_range()),
                "unexpected": "field",
            },
        )


def test_models_serialize_to_json_compatible_data() -> None:
    query = CalendarQuery(
        time_range=time_range(),
        calendar_ids=("work",),
        limit=5,
    )
    result = CalendarQueryResult(query=query, events=(event(),))

    data = result.model_dump(mode="json")

    assert json.loads(result.model_dump_json()) == data
    assert data == {
        "query": {
            "time_range": {
                "start_time": "2026-06-03T09:00:00Z",
                "end_time": "2026-06-03T10:00:00Z",
                "timezone": "UTC",
            },
            "calendar_ids": ["work"],
            "limit": 5,
        },
        "events": [
            {
                "event_id": "event-123",
                "title": "Planning review",
                "time_range": {
                    "start_time": "2026-06-03T09:00:00Z",
                    "end_time": "2026-06-03T10:00:00Z",
                    "timezone": "UTC",
                },
                "calendar_id": "work",
                "location": "Office",
                "description": "Review implementation plan.",
                "attendees": ["alice@example.test", "bob@example.test"],
                "availability": "busy",
            },
        ],
    }
