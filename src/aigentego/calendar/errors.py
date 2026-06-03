"""Provider-neutral calendar error types."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CalendarErrorDetail(BaseModel):
    """Structured, provider-neutral calendar error details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    context: dict[str, str] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject blank error codes and messages."""
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: dict[str, str]) -> dict[str, str]:
        """Keep context safe, simple, and deterministic."""
        for key, item in value.items():
            if not key.strip() or not item.strip():
                raise ValueError("context keys and values must not be blank")
        return value


class CalendarError(Exception):
    """Base error for provider-neutral calendar failures."""

    error_code: ClassVar[str] = "calendar_error"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, str] | None = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def to_detail(self) -> CalendarErrorDetail:
        """Return normalized safe calendar error details."""
        return CalendarErrorDetail(
            code=self.error_code,
            message=self.message,
            context=self.context,
        )


class CalendarConfigurationError(CalendarError):
    """Calendar configuration is incomplete or invalid."""

    error_code: ClassVar[str] = "calendar_configuration_error"


class CalendarQueryError(CalendarError):
    """A calendar query could not be completed."""

    error_code: ClassVar[str] = "calendar_query_error"


class CalendarPermissionError(CalendarError):
    """Calendar access is not permitted."""

    error_code: ClassVar[str] = "calendar_permission_error"


class CalendarNotFoundError(CalendarError):
    """The requested calendar or event does not exist."""

    error_code: ClassVar[str] = "calendar_not_found"


class CalendarWriteNotAllowedError(CalendarError):
    """Calendar writes are not allowed by the current boundary."""

    error_code: ClassVar[str] = "calendar_write_not_allowed"


__all__ = [
    "CalendarConfigurationError",
    "CalendarError",
    "CalendarErrorDetail",
    "CalendarNotFoundError",
    "CalendarPermissionError",
    "CalendarQueryError",
    "CalendarWriteNotAllowedError",
]
