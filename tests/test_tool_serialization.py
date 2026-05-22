import json

from aigentego.tools import (
    CalculatorTool,
    ToolDefinition,
    ToolRegistry,
    serialize_tool_definitions,
)


def test_empty_tool_definitions_serialize_to_empty_list() -> None:
    assert serialize_tool_definitions([]) == []


def test_calculator_definition_serializes_model_facing_fields() -> None:
    definition = CalculatorTool().definition

    serialized = serialize_tool_definitions([definition])

    assert serialized == [
        {
            "name": "calculator",
            "description": "Evaluate a safe arithmetic expression.",
            "parameters_schema": {
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate.",
                    },
                },
                "additionalProperties": False,
            },
        },
    ]


def test_tool_definition_serialization_preserves_input_order() -> None:
    definitions = [
        ToolDefinition(
            name="zeta",
            description="Last supplied tool.",
            parameters_schema={"type": "object"},
        ),
        ToolDefinition(
            name="alpha",
            description="First supplied tool.",
            parameters_schema={"type": "object"},
        ),
    ]

    serialized = serialize_tool_definitions(definitions)

    assert [tool["name"] for tool in serialized] == ["zeta", "alpha"]


def test_tool_definition_serialization_does_not_expose_source_mutation() -> None:
    definition = ToolDefinition(
        name="echo",
        description="Return supplied data.",
        parameters_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    )

    serialized = serialize_tool_definitions([definition])
    serialized[0]["parameters_schema"]["properties"]["value"]["type"] = "integer"

    assert definition.parameters_schema == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }


def test_serialized_tool_definitions_are_json_serializable() -> None:
    serialized = serialize_tool_definitions([CalculatorTool().definition])

    assert json.loads(json.dumps(serialized)) == serialized


def test_registry_tool_definitions_serialize_deterministically() -> None:
    registry = ToolRegistry([CalculatorTool()])

    serialized = serialize_tool_definitions(registry.list_definitions())

    assert [tool["name"] for tool in serialized] == ["calculator"]
