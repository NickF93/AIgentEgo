"""Deterministic registry for provider-neutral tools."""

from collections.abc import Iterable

from aigentego.tools.base import Tool, ToolDefinition
from aigentego.tools.errors import ToolDefinitionError, ToolNotFoundError


class ToolRegistry:
    """Register and retrieve deterministic tools by unique name."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._definitions: dict[str, ToolDefinition] = {}

        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool if its definition name is not already present."""
        definition = tool.definition
        name = definition.name

        if name in self._tools:
            raise ToolDefinitionError(
                "tool name is already registered",
                tool_name=name,
            )

        self._tools[name] = tool
        self._definitions[name] = definition.model_copy(deep=True)

    def get(self, name: str) -> Tool:
        """Return a registered tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def list_definitions(self) -> list[ToolDefinition]:
        """Return registered tool definitions in deterministic name order."""
        return [
            self._definitions[name].model_copy(deep=True)
            for name in sorted(self._definitions)
        ]
