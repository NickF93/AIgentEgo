import pytest
from pydantic import ValidationError

from aigentego.calendar import (
    CalendarConfigurationError,
    CalendarError,
    CalendarErrorDetail,
    CalendarNotFoundError,
    CalendarPermissionError,
    CalendarQueryError,
    CalendarWriteNotAllowedError,
)


def test_calendar_errors_preserve_safe_normalized_details() -> None:
    error = CalendarQueryError(
        "calendar query failed",
        context={"calendar_id": "work"},
    )

    assert isinstance(error, CalendarError)
    assert str(error) == "calendar query failed"
    assert error.to_detail() == CalendarErrorDetail(
        code="calendar_query_error",
        message="calendar query failed",
        context={"calendar_id": "work"},
    )


def test_calendar_error_subclasses_use_stable_codes() -> None:
    errors = [
        CalendarConfigurationError("invalid calendar configuration"),
        CalendarQueryError("invalid calendar query"),
        CalendarPermissionError("calendar permission denied"),
        CalendarNotFoundError("calendar not found"),
        CalendarWriteNotAllowedError("calendar writes are disabled"),
    ]

    assert [error.to_detail().code for error in errors] == [
        "calendar_configuration_error",
        "calendar_query_error",
        "calendar_permission_error",
        "calendar_not_found",
        "calendar_write_not_allowed",
    ]


def test_calendar_error_detail_rejects_blank_values() -> None:
    with pytest.raises(ValidationError):
        CalendarErrorDetail(code="", message="message")

    with pytest.raises(ValidationError):
        CalendarErrorDetail(code="calendar_error", message="   ")

    with pytest.raises(ValidationError, match="context keys and values"):
        CalendarErrorDetail(
            code="calendar_error",
            message="message",
            context={"": "value"},
        )

    with pytest.raises(ValidationError, match="context keys and values"):
        CalendarErrorDetail(
            code="calendar_error",
            message="message",
            context={"calendar_id": "   "},
        )


def test_calendar_error_detail_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CalendarErrorDetail.model_validate(
            {
                "code": "calendar_error",
                "message": "message",
                "context": {},
                "provider_payload": {"secret": "do-not-leak"},
            },
        )


def test_calendar_errors_do_not_require_provider_context() -> None:
    error = CalendarError("calendar failure")

    assert error.to_detail() == CalendarErrorDetail(
        code="calendar_error",
        message="calendar failure",
        context={},
    )
