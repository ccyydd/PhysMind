from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


def default_appearance() -> Dict[str, Any]:
    return {
        "color": "unknown",
        "material": "rubber",
        "material_confidence": None,
        "reasoning": "",
    }


@dataclass
class TargetObject:
    object_id: str
    description: str
    role: str
    geometry_type: str = "irregular"
    geometry_confidence: Optional[float] = None
    appearance: Dict[str, Any] = field(default_factory=default_appearance)
    source_track_id: Optional[str] = None
    # Physion++ bouncy_wall two-segment: when set, this object reuses the geometry of
    # the referenced object (its same-role counterpart in the other segment) instead of
    # being reconstructed on its own. Only the seg2 object carries this; the seg1 source
    # object is reconstructed normally.
    mesh_reuse_source_object_id: Optional[str] = None


@dataclass
class ObjectPlan:
    scene_index: int
    question_id: int
    question_type: str
    question: str
    choices: List[Dict[str, Any]]
    target_objects: List[TargetObject]
    reasoning: str
    scene_objects: Dict[str, Any] = field(default_factory=dict)
    special_scene: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, TargetObject):
                return asdict(value)
            if is_dataclass(value):
                return {item.name: convert(getattr(value, item.name)) for item in fields(value)}
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        payload = convert(self)
        if (
            payload.get("status") == "bootstrap"
            and isinstance(payload.get("target_objects"), list)
            and not payload["target_objects"]
        ):
            payload.pop("target_objects", None)
            payload["dynamic_objects_pending"] = True
            payload["dynamic_objects_source"] = "sam3_full_video_tracks"
        return payload


@dataclass
class ToolResult:
    tool_name: str
    status: str
    artifact_path: Optional[str] = None
    message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    elapsed_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldModelQuestionResult:
    scene_index: int
    video_filename: str
    question_id: int
    question_type: str
    question: str
    status: str
    final_answer: Optional[str]
    error_message: Optional[str]
    artifact_dir: str
    stages: List[Dict[str, Any]]
    elapsed_with_scene_reconstruction_sec: Optional[float] = None
    elapsed_without_scene_reconstruction_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
