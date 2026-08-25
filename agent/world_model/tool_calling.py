from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

from agent.query import extract_answer_tag


JSONSchema = Dict[str, Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JSONSchema

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCallRequest:
    tool_calls: list[ToolCall]
    reasoning_summary: str = ""
    expected_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "reasoning_summary": self.reasoning_summary,
            "expected_evidence": list(self.expected_evidence),
        }


@dataclass(frozen=True)
class ToolCallResult:
    tool: str
    status: str
    call_id: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolCallValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]):
        self._tools = {tool.name: tool for tool in tools}

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> ToolDefinition:
        return self._tools[tool_name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_prompt_schema(self) -> list[dict[str, Any]]:
        return [self._tools[name].to_dict() for name in self.names()]


def default_qcpr_tool_registry() -> ToolRegistry:
    rollout_id_description = (
        'Rollout id to inspect. Use "base" for the unedited rollout, or copy exactly a rollout_id '
        "returned by a previous simulate_edit/remove_object tool result. Do not invent rollout ids."
    )
    return ToolRegistry(
        [
            ToolDefinition(
                name="simulate_edit",
                description=(
                    "Run a query-conditioned physical rollout after a scene edit. Use this for generic "
                    "counterfactual edits such as removing an object."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["edit"],
                    "properties": {
                        "edit": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["type", "object_id"],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["remove_object"],
                                    "description": "Edit type to simulate.",
                                },
                                "object_id": {
                                    "type": "string",
                                    "description": "Internal object id affected by the edit, such as obj_1.",
                                },
                                "start_frame": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Optional frame where the edit starts.",
                                },
                            },
                        }
                    },
                },
            ),
            ToolDefinition(
                name="check_contact",
                description=(
                    "Check whether two objects contact or collide in a rollout. Returns confirmed contact, "
                    "nearest distance, and supporting evidence."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_a", "object_b"],
                    "properties": {
                        "rollout_id": {
                            "type": "string",
                            "description": rollout_id_description,
                        },
                        "object_a": {"type": "string", "description": "First internal object id, such as obj_2."},
                        "object_b": {"type": "string", "description": "Second internal object id, such as obj_4."},
                        "start_frame": {
                            "type": "integer",
                            "minimum": 0,
                            "description": (
                                "Optional inclusive start frame. Omit it to inspect the whole rollout. "
                                "Use the boundary-aware predictive start frame only for predictive questions about future events after the observed video, "
                                "not for ordinary counterfactual edited rollouts."
                            ),
                        },
                        "end_frame": {"type": "integer", "minimum": 0, "description": "Optional inclusive end frame."},
                    },
                },
            ),
            ToolDefinition(
                name="get_event_time",
                description=(
                    "Get the frame/time when two objects contact or collide in a rollout. Use this for "
                    "choices that are themselves collision/contact events."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_a", "object_b"],
                    "properties": {
                        "rollout_id": {
                            "type": "string",
                            "description": rollout_id_description,
                        },
                        "object_a": {"type": "string", "description": "First internal object id, such as obj_2."},
                        "object_b": {"type": "string", "description": "Second internal object id, such as obj_4."},
                        "start_frame": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional inclusive start frame. Omit it to inspect the whole rollout.",
                        },
                        "end_frame": {"type": "integer", "minimum": 0, "description": "Optional inclusive end frame."},
                    },
                },
            ),
            ToolDefinition(
                name="compare_rollouts",
                description="Compare object trajectories and collision events between two rollouts.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "base_rollout_id": {
                            "type": "string",
                            "description": 'Reference rollout id. Use "base" or copy exactly a previous tool result rollout_id. Defaults to base.',
                        },
                        "edited_rollout_id": {
                            "type": "string",
                            "description": "Edited rollout id. Copy exactly a rollout_id returned by simulate_edit/remove_object. Defaults to the first edited rollout.",
                        },
                        "object_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional internal object ids to compare.",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="inspect_world_state",
                description="Inspect all active objects and collision events at one video frame.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame_index"],
                    "properties": {
                        "frame_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Video frame index to inspect.",
                        }
                    },
                },
            ),
            ToolDefinition(
                name="inspect_object_trajectory",
                description="Inspect one object's corrected or simulated trajectory over a frame range.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_id"],
                    "properties": {
                        "rollout_id": {
                            "type": "string",
                            "description": rollout_id_description,
                        },
                        "object_id": {
                            "type": "string",
                            "description": "Internal object id, such as obj_1.",
                        },
                        "start_frame": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional inclusive start frame.",
                        },
                        "end_frame": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional inclusive end frame.",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="remove_object",
                description=(
                    "Remove one object and create an edited rollout."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_id"],
                    "properties": {
                        "object_id": {
                            "type": "string",
                            "description": "Internal object id to remove, such as obj_2.",
                        },
                        "start_frame": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Frame where the object removal starts. Defaults to the object's first active frame.",
                        },
                    },
                },
            ),
        ]
    )


def build_prompt_only_tool_call_instruction(registry: ToolRegistry) -> str:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_calls"],
        "properties": {
            "tool_calls": [
                {
                    "tool": "simulate_edit",
                    "arguments": {"edit": {"type": "remove_object", "object_id": "obj_2", "start_frame": 35}},
                    "call_id": "optional stable id",
                }
            ],
            "reasoning_summary": "brief reason for these tool calls",
            "expected_evidence": ["what evidence these calls should produce"],
        },
    }
    return (
        "Return only compact JSON inside <answer> </answer>.\n"
        "The expected_evidence field, when present, must be a JSON array of strings, not a single string.\n"
        'Rollout id protocol: use "base" for the unedited rollout. For edited rollouts, copy the exact '
        "rollout_id returned by a previous simulate_edit/remove_object tool result. Never invent ids such "
        "as unnamed_edit_1, edited_rollout_1, or rollout_1.\n"
        "Use this JSON response shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Available tools:\n"
        f"{json.dumps(registry.to_prompt_schema(), ensure_ascii=False, indent=2)}"
    )


def parse_prompt_only_tool_call_response(text: str, registry: ToolRegistry) -> ToolCallRequest:
    payload_text = extract_answer_tag(text) or text
    payload = _parse_json_object(payload_text)
    return validate_tool_call_request(payload, registry)


def validate_tool_call_request(payload: Mapping[str, Any], registry: ToolRegistry) -> ToolCallRequest:
    issues: list[str] = []
    if not isinstance(payload, Mapping):
        raise ToolCallValidationError(["response must be a JSON object"])

    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        issues.append("tool_calls must be a list")
        raw_calls = []
    if not raw_calls:
        issues.append("tool_calls must contain at least one tool call")

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        path = f"tool_calls[{index}]"
        if not isinstance(raw_call, Mapping):
            issues.append(f"{path} must be an object")
            continue
        tool_name = raw_call.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            issues.append(f"{path}.tool must be a non-empty string")
            continue
        tool_name = tool_name.strip()
        if tool_name not in registry:
            issues.append(f"{path}.tool {tool_name!r} is not registered; allowed tools: {registry.names()}")
            continue
        arguments = raw_call.get("arguments")
        if not isinstance(arguments, Mapping):
            issues.append(f"{path}.arguments must be an object")
            continue
        _validate_schema_value(
            value=dict(arguments),
            schema=registry.get(tool_name).parameters,
            path=f"{path}.arguments",
            issues=issues,
        )
        call_id = raw_call.get("call_id")
        if call_id is not None and not isinstance(call_id, str):
            issues.append(f"{path}.call_id must be a string when provided")
            call_id = None
        calls.append(ToolCall(tool=tool_name, arguments=dict(arguments), call_id=call_id))

    reasoning_summary = payload.get("reasoning_summary", "")
    if reasoning_summary is None:
        reasoning_summary = ""
    if not isinstance(reasoning_summary, str):
        issues.append("reasoning_summary must be a string when provided")
        reasoning_summary = ""

    expected_evidence = payload.get("expected_evidence", [])
    if expected_evidence is None:
        expected_evidence = []
    if not isinstance(expected_evidence, list) or not all(isinstance(item, str) for item in expected_evidence):
        issues.append("expected_evidence must be a list of strings when provided")
        expected_evidence = []

    if issues:
        raise ToolCallValidationError(issues)
    return ToolCallRequest(
        tool_calls=calls,
        reasoning_summary=reasoning_summary,
        expected_evidence=list(expected_evidence),
    )


def build_tool_call_repair_instruction(
    *,
    original_text: str,
    validation_error: ToolCallValidationError,
    registry: ToolRegistry,
) -> str:
    return (
        "Your previous tool-call response was invalid.\n"
        f"Validation errors: {json.dumps(validation_error.issues, ensure_ascii=False)}\n\n"
        f"Original response:\n{original_text}\n\n"
        f"{build_prompt_only_tool_call_instruction(registry)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ToolCallValidationError(["response does not contain a JSON object"]) from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ToolCallValidationError([f"response JSON parse failed: {exc.msg}"]) from None
    if not isinstance(value, dict):
        raise ToolCallValidationError(["response JSON root must be an object"])
    return value


def _validate_schema_value(*, value: Any, schema: JSONSchema, path: str, issues: list[str]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            issues.append(f"{path} must be an object")
            return
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in value:
                issues.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    issues.append(f"{path}.{key} is not allowed")
        for key, item in value.items():
            subschema = properties.get(key)
            if isinstance(subschema, Mapping):
                _validate_schema_value(value=item, schema=dict(subschema), path=f"{path}.{key}", issues=issues)
        return

    if expected_type == "string":
        if not isinstance(value, str):
            issues.append(f"{path} must be a string")
            return
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            issues.append(f"{path} must be one of {enum}")
        return

    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"{path} must be an integer")
            return
        _validate_numeric_bounds(value=float(value), schema=schema, path=path, issues=issues)
        return

    if expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            issues.append(f"{path} must be a number")
            return
        _validate_numeric_bounds(value=float(value), schema=schema, path=path, issues=issues)
        return

    if expected_type == "boolean":
        if not isinstance(value, bool):
            issues.append(f"{path} must be a boolean")
        return

    if expected_type == "array":
        if not isinstance(value, list):
            issues.append(f"{path} must be an array")
            return
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(value=item, schema=dict(item_schema), path=f"{path}[{index}]", issues=issues)
        return


def _validate_numeric_bounds(*, value: float, schema: JSONSchema, path: str, issues: list[str]) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < float(minimum):
        issues.append(f"{path} must be >= {minimum}")
    if isinstance(maximum, (int, float)) and value > float(maximum):
        issues.append(f"{path} must be <= {maximum}")
