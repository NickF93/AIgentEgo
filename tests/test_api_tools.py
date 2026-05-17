from fastapi.testclient import TestClient

from aigentego.main import create_app
from aigentego.observability import REQUEST_ID_HEADER


def make_client() -> TestClient:
    return TestClient(create_app())


def test_tools_list_returns_registered_calculator() -> None:
    client = make_client()

    response = client.get("/tools")

    assert response.status_code == 200
    body = response.json()
    assert [tool["name"] for tool in body["tools"]] == ["calculator"]
    calculator = body["tools"][0]
    assert calculator["description"] == "Evaluate a safe arithmetic expression."
    assert calculator["parameters_schema"]["required"] == ["expression"]


def test_tools_execute_calculator_returns_successful_result() -> None:
    client = make_client()

    response = client.post(
        "/tools/execute",
        headers={REQUEST_ID_HEADER: "tool-request-123"},
        json={
            "tool_name": "calculator",
            "arguments": {"expression": "12 * 31"},
        },
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "tool-request-123"
    assert response.json() == {
        "request_id": "tool-request-123",
        "tool_name": "calculator",
        "success": True,
        "result": {"value": 372},
        "error": None,
    }


def test_tools_execute_unknown_tool_returns_normalized_error() -> None:
    client = make_client()

    response = client.post(
        "/tools/execute",
        json={"tool_name": "missing", "arguments": {}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_name"] == "missing"
    assert body["success"] is False
    assert body["result"] is None
    assert body["error"] == {
        "code": "tool_not_found",
        "message": "tool is not registered",
        "tool_name": "missing",
    }


def test_tools_execute_invalid_arguments_returns_validation_error() -> None:
    client = make_client()

    response = client.post(
        "/tools/execute",
        json={"tool_name": "calculator", "arguments": {}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_name"] == "calculator"
    assert body["success"] is False
    assert body["result"] is None
    assert body["error"]["code"] == "tool_validation_error"
    assert body["error"]["tool_name"] == "calculator"


def test_tools_execute_calculator_runtime_failure_returns_execution_error() -> None:
    client = make_client()

    response = client.post(
        "/tools/execute",
        json={
            "tool_name": "calculator",
            "arguments": {"expression": "1 / 0"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_name"] == "calculator"
    assert body["success"] is False
    assert body["result"] is None
    assert body["error"]["code"] == "tool_execution_error"
    assert body["error"]["tool_name"] == "calculator"


def test_tools_execute_rejects_invalid_payload() -> None:
    client = make_client()

    response = client.post("/tools/execute", json={"arguments": {}})

    assert response.status_code == 422
