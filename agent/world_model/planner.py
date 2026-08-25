from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from agent.query import answer_video_question, extract_answer_tag
from agent.world_model.prompts import build_scene_assessment_prompt
from agent.world_model.schemas import ObjectPlan, TargetObject, default_appearance
from utils.config import ModelConfig


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
SCENE_VLM_HORIZONTAL_ROLL_SUPPORT_BUNDLE_ROUTE = (
    "assessment.motion_camera_support_vlm"
)
STATIC_GEOMETRY_TYPES = {"plane", "wall", "ramp", "box", "irregular"}
MATERIAL_ALIASES = {
    "metal": "metal",
    "metallic": "metal",
    "rubber": "rubber",
    "rubbery": "rubber",
    "matte": "rubber",
}
MATERIAL_TYPES = {"metal", "rubber"}


def _fallback_scene_target_objects() -> List[TargetObject]:
    return [
        TargetObject(
            object_id="obj_1",
            description="visible dynamic rigid object(s) in the scene",
            role="fallback dynamic object for scene-level dry run or invalid VLM object plan",
            geometry_type="irregular",
            geometry_confidence=None,
            appearance=default_appearance(),
        )
    ]


def _parse_json_answer(text: str) -> Dict[str, Any]:
    answer = extract_answer_tag(text) or text
    match = JSON_OBJECT_PATTERN.search(answer)
    if not match:
        raise ValueError("No JSON object found in planner response.")
    return json.loads(match.group(0))


def _geometry_confidence(value: Any) -> float | None:
    try:
        if value is None:
            return None
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _appearance(value: Any, description: str = "") -> Dict[str, Any]:
    defaults = default_appearance()
    if not isinstance(value, dict):
        return defaults

    material_raw = str(value.get("material", defaults["material"])).strip()
    material = MATERIAL_ALIASES.get(material_raw.lower(), MATERIAL_ALIASES.get(material_raw, material_raw))
    if material not in MATERIAL_TYPES:
        material = defaults["material"]

    return {
        "color": str(value.get("color") or defaults["color"]).strip() or defaults["color"],
        "material": material,
        "material_confidence": _geometry_confidence(value.get("material_confidence")),
        "reasoning": str(value.get("reasoning") or "").strip() or f"inferred from visual appearance: {description}"[:200],
    }


def _static_geometry_type(value: Any, description: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in STATIC_GEOMETRY_TYPES:
        return raw
    text = description.lower()
    if "ground" in text or "floor" in text or "plane" in text:
        return "plane"
    if "wall" in text:
        return "wall"
    if "ramp" in text or "slope" in text:
        return "ramp"
    return "irregular"


def _default_scene_objects(target_objects: List[TargetObject]) -> Dict[str, Any]:
    return {
        "dynamic_objects": [item.object_id for item in target_objects],
        "static_objects": [
            {
                "object_id": "ground_plane",
                "description": "horizontal floor or support plane",
                "role": "static collision support",
                "geometry_type": "plane",
                "appearance": _appearance(
                    {
                        "color": "gray",
                        "material": "rubber",
                        "material_confidence": 0.5,
                        "reasoning": "default static support appearance",
                    },
                    "horizontal floor or support plane",
                ),
            }
        ],
    }


def _static_objects_from_support_surfaces(payload: Any) -> List[Dict[str, Any]]:
    surfaces = payload if isinstance(payload, list) else []
    static_objects = []
    for index, item in enumerate(surfaces):
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        geometry_type = _static_geometry_type(item.get("geometry_type"), description)
        static_objects.append(
            {
                "object_id": str(item.get("object_id") or f"support_{index + 1}"),
                "description": description,
                "role": str(item.get("role") or "static collision support"),
                "geometry_type": geometry_type,
                "confidence": _geometry_confidence(item.get("confidence")),
                "appearance": _appearance(item.get("appearance"), description),
            }
        )
    return static_objects


def _normalize_scene_assessment(payload: Any) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    special_scene = _special_scene(payload)
    static_objects = _static_objects_from_support_surfaces(payload.get("support_surfaces"))
    return {
        "special_scene": special_scene,
        "static_objects": static_objects,
        "reasoning": str(payload.get("reasoning") or "").strip(),
        "raw_payload": payload,
    }


def _special_scene(payload: Any) -> Dict[str, Any]:
    horizontal = {}
    roll_stabilization = {}
    if isinstance(payload, dict) and isinstance(payload.get("horizontal_plane_motion"), dict):
        horizontal = payload["horizontal_plane_motion"]
    if isinstance(payload, dict) and isinstance(payload.get("roll_stabilization"), dict):
        roll_stabilization = payload["roll_stabilization"]
    applies = horizontal.get("applies", "unknown") if isinstance(horizontal, dict) else "unknown"
    confidence = _geometry_confidence(horizontal.get("confidence")) if isinstance(horizontal, dict) else None
    roll_applies = (
        roll_stabilization.get("applies", "unknown")
        if isinstance(roll_stabilization, dict)
        else "unknown"
    )
    roll_confidence = (
        _geometry_confidence(roll_stabilization.get("confidence"))
        if isinstance(roll_stabilization, dict)
        else None
    )
    return {
        "horizontal_plane_motion": {
            "applies": applies if isinstance(applies, bool) else "unknown",
            "confidence": confidence,
            "reason": str(horizontal.get("reason", "")) if isinstance(horizontal, dict) else "",
        },
        "roll_stabilization": {
            "applies": roll_applies if isinstance(roll_applies, bool) else "unknown",
            "confidence": roll_confidence,
            "reason": str(roll_stabilization.get("reason", "")) if isinstance(roll_stabilization, dict) else "",
        },
    }


def _benchmark_name(scene: Any) -> str:
    return str(getattr(scene, "benchmark", "clevrer") or "clevrer").strip().lower()


def _scene_metadata(scene: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "benchmark": _benchmark_name(scene),
        "scene_index": int(getattr(scene, "scene_index", -1)),
    }
    for name in ("scenario", "stimulus_id", "video_filename"):
        value = getattr(scene, name, None)
        if value is not None:
            metadata[name] = str(value)
    video_path = getattr(scene, "video_path", None)
    if video_path is not None:
        metadata["video_path"] = str(video_path)
    return metadata


def build_dry_run_scene_object_plan(scene: Any) -> ObjectPlan:
    target_objects = _fallback_scene_target_objects()
    return ObjectPlan(
        scene_index=scene.scene_index,
        question_id=-1,
        question_type="scene",
        question="",
        choices=[],
        target_objects=target_objects,
        reasoning="Dry-run planner uses a scene-level fallback object plan without question text.",
        scene_objects=_default_scene_objects(target_objects),
        special_scene={
            **_special_scene(None),
            "sam3_video_tracking_prompts": [],
            "benchmark": _benchmark_name(scene),
            "scene_metadata": _scene_metadata(scene),
        },
        status="dry_run",
    )


def build_track_bootstrap_scene_object_plan(
    scene: Any,
    *,
    scene_assessment: Dict[str, Any] | None = None,
) -> ObjectPlan:
    target_objects: List[TargetObject] = []
    benchmark = _benchmark_name(scene)
    special_scene = _special_scene(None)
    scene_objects = _default_scene_objects(target_objects)
    if isinstance(scene_assessment, dict):
        assessed_special = scene_assessment.get("special_scene")
        if isinstance(assessed_special, dict):
            special_scene.update(assessed_special)
        static_objects = scene_assessment.get("static_objects")
        if isinstance(static_objects, list) and static_objects:
            scene_objects["static_objects"] = static_objects
        special_scene["scene_assessment"] = {
            "source": "vlm_video_assessment",
            "reasoning": str(scene_assessment.get("reasoning") or ""),
            "raw_payload": scene_assessment.get("raw_payload") if isinstance(scene_assessment.get("raw_payload"), dict) else {},
        }
        resolved_route = scene_assessment.get("resolved_route")
        if isinstance(resolved_route, dict):
            special_scene["scene_assessment"]["resolved_route"] = resolved_route
    special_scene["sam3_video_tracking_prompts"] = []
    special_scene["object_inventory_source"] = "sam3_full_video_tracks"
    special_scene["benchmark"] = benchmark
    special_scene["scene_metadata"] = _scene_metadata(scene)
    return ObjectPlan(
        scene_index=scene.scene_index,
        question_id=-1,
        question_type="scene",
        question="",
        choices=[],
        target_objects=target_objects,
        reasoning=(
            "Bootstrap scene plan: dynamic objects are intentionally left empty and will be derived "
            "from SAM3 full-video tracks before downstream reconstruction."
        ),
        scene_objects=scene_objects,
        special_scene=special_scene,
        status="bootstrap",
    )


def build_vlm_scene_assessment(
    config: ModelConfig,
    scene: Any,
    *,
    route: str,
) -> Dict[str, Any]:
    if route != SCENE_VLM_HORIZONTAL_ROLL_SUPPORT_BUNDLE_ROUTE:
        raise ValueError(f"unsupported scene-assessment route: {route!r}")
    prompt = build_scene_assessment_prompt()
    response = answer_video_question(
        config=config,
        prompt=prompt,
        video_path=scene.video_path,
        request_context={
            "scene_index": scene.scene_index,
            "question_id": -1,
            "question_type": "scene",
            "stage": "scene_assessment",
            "benchmark": _benchmark_name(scene),
        },
    )
    try:
        payload = _parse_json_answer(response.text)
    except Exception as exc:
        retry_prompt = (
            "Your previous reply for the same scene assessment task was not valid JSON and could not be parsed: "
            f"{exc}. Return exactly one complete JSON object matching the requested schema. "
            "Do not include markdown, code fences, or extra explanation."
        )
        retry_response = answer_video_question(
            config=config,
            prompt=prompt,
            video_path=scene.video_path,
            request_context={
                "scene_index": scene.scene_index,
                "question_id": -1,
                "question_type": "scene",
                "stage": "scene_assessment_retry",
                "benchmark": _benchmark_name(scene),
            },
            continuation_messages=[
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": retry_prompt},
            ],
        )
        try:
            payload = _parse_json_answer(retry_response.text)
        except Exception:
            return {
                **_normalize_scene_assessment({}),
                "status": "parse_error",
                "error_message": str(exc),
                "raw_response_text": response.text,
                "retry_response_text": retry_response.text,
            }
    assessment = _normalize_scene_assessment(payload)
    assessment["status"] = "ok"
    assessment["raw_response_text"] = response.text
    return assessment
