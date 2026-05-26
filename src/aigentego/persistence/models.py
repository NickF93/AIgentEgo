"""Local-first persistence data contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SessionStatus(StrEnum):
    """Inspectable lifecycle states for a persisted user session."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationStatus(StrEnum):
    """Inspectable lifecycle states for a persisted conversation."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class Session(BaseModel):
    """Provider-neutral metadata for a local user session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    title: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("session_id", "title")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional text."""
        return _validate_optional_text(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require caller-supplied timestamps to be timezone-aware."""
        return _validate_optional_timezone_aware_datetime(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        """Keep caller-supplied updated_at after created_at."""
        _validate_timestamp_order(self.created_at, self.updated_at)
        return self


class Conversation(BaseModel):
    """Provider-neutral metadata for one local conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    title: str | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("conversation_id", "session_id", "title")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        """Reject blank identifiers and optional text."""
        return _validate_optional_text(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require caller-supplied timestamps to be timezone-aware."""
        return _validate_optional_timezone_aware_datetime(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        """Keep caller-supplied updated_at after created_at."""
        _validate_timestamp_order(self.created_at, self.updated_at)
        return self


def _validate_optional_text(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("text fields must not be blank")
    return value


def _validate_optional_timezone_aware_datetime(
    value: datetime | None,
) -> datetime | None:
    if value is not None and value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


def _validate_timestamp_order(
    created_at: datetime | None,
    updated_at: datetime | None,
) -> None:
    if created_at is not None and updated_at is not None and updated_at < created_at:
        raise ValueError("updated_at must not be before created_at")


__all__ = [
    "Conversation",
    "ConversationStatus",
    "Session",
    "SessionStatus",
]
