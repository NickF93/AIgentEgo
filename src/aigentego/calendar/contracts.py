"""Provider-neutral calendar data contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalendarAvailability(StrEnum):
    """Inspectable availability values for calendar events."""

    FREE = "free"
    BUSY = "busy"
    TENTATIVE = "tentative"


class CalendarTimeRange(BaseModel):
    """A timezone-aware interval for calendar queries and events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_time: datetime
    end_time: datetime
    timezone: str | None = None

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require timezone-aware calendar timestamps."""
        return _validate_timezone_aware_datetime(value)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        """Reject blank timezone labels."""
        return _validate_optional_text(value)

    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        """Require ranges to have a positive duration."""
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class CalendarEvent(BaseModel):
    """Provider-neutral calendar event data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    time_range: CalendarTimeRange
    calendar_id: str | None = None
    location: str | None = None
    description: str | None = None
    attendees: tuple[str, ...] = ()
    availability: CalendarAvailability = CalendarAvailability.BUSY

    @field_validator(
        "event_id",
        "title",
        "calendar_id",
        "location",
        "description",
    )
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional text fields."""
        return _validate_optional_text(value)

    @field_validator("attendees")
    @classmethod
    def validate_attendees(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank attendee labels."""
        for attendee in value:
            _validate_optional_text(attendee)
        return value


class CalendarQuery(BaseModel):
    """A read-only provider-neutral calendar query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_range: CalendarTimeRange
    calendar_ids: tuple[str, ...] = ()
    limit: int = Field(default=50, ge=1)

    @field_validator("calendar_ids")
    @classmethod
    def validate_calendar_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank calendar identifiers."""
        for calendar_id in value:
            _validate_optional_text(calendar_id)
        return value

    @field_validator("limit", mode="before")
    @classmethod
    def validate_limit(cls, value: object) -> object:
        """Reject bools before Pydantic can coerce them into integers."""
        if isinstance(value, bool):
            raise ValueError("limit must be an integer")
        return value


class CalendarQueryResult(BaseModel):
    """Inspectable read-only calendar query result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: CalendarQuery
    events: tuple[CalendarEvent, ...] = ()

    @model_validator(mode="after")
    def validate_event_count(self) -> Self:
        """Keep result cardinality aligned with the requested limit."""
        if len(self.events) > self.query.limit:
            raise ValueError("events must not exceed query limit")
        return self


def _validate_optional_text(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("text fields must not be blank")
    return value


def _validate_timezone_aware_datetime(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


__all__ = [
    "CalendarAvailability",
    "CalendarEvent",
    "CalendarQuery",
    "CalendarQueryResult",
    "CalendarTimeRange",
]
