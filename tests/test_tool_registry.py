from collections.abc import Mapping
from typing import Any

import pytest

from aigentego.tools import (
    ToolContext,
    ToolDefinition,
    ToolDefinitionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)


class StaticTool:
    def __init__(
        self,
        name: str,
        description: str = "A deterministic test tool.",
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.definition.name,
            success=True,
            result={
                "arguments": dict(arguments),
                "request_id": context.request_id,
            },
        )


def test_registry_registers_and_retrieves_tool_by_name() -> None:
    tool = StaticTool("echo")
    registry = ToolRegistry()

    registry.register(tool)

    assert registry.get("echo") is tool


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry([StaticTool("echo")])

    with pytest.raises(ToolDefinitionError) as error_info:
        registry.register(StaticTool("echo", description="Another echo tool."))

    detail = error_info.value.to_detail()
    assert detail.code == "tool_definition_error"
    assert detail.message == "tool name is already registered"
    assert detail.tool_name == "echo"


def test_registry_raises_normalized_error_for_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError) as error_info:
        registry.get("missing")

    detail = error_info.value.to_detail()
    assert detail.code == "tool_not_found"
    assert detail.message == "tool is not registered"
    assert detail.tool_name == "missing"


def test_registry_lists_definitions_in_stable_name_order() -> None:
    registry = ToolRegistry(
        [
            StaticTool("zeta", description="Last tool."),
            StaticTool("alpha", description="First tool."),
        ],
    )

    definitions = registry.list_definitions()

    assert [definition.name for definition in definitions] == ["alpha", "zeta"]
    assert [definition.description for definition in definitions] == [
        "First tool.",
        "Last tool.",
    ]


def test_registry_returns_definition_copies() -> None:
    registry = ToolRegistry([StaticTool("echo")])
    definitions = registry.list_definitions()

    definitions[0].parameters_schema["properties"] = {}

    assert registry.list_definitions()[0].parameters_schema == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
