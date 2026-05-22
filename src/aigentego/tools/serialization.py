"""Model-facing serialization for deterministic tool definitions."""

import json
from collections.abc import Sequence
from typing import Any

from aigentego.tools.base import ToolDefinition


def serialize_tool_definitions(
    definitions: Sequence[ToolDefinition],
) -> list[dict[str, Any]]:
    """Serialize tool definitions into provider-neutral model-facing data."""
    serialized: list[dict[str, Any]] = []

    for definition in definitions:
        data: dict[str, Any] = json.loads(definition.model_dump_json())
        serialized.append(
            {
                "name": data["name"],
                "description": data["description"],
                "parameters_schema": data["parameters_schema"],
            },
        )

    return serialized
