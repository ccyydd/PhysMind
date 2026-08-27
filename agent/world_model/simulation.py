from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from agent.query import answer_with_image_files, create_openai_compatible_client, extract_answer_tag
from agent.world_model.debug_artifacts import render_qcpr_plane_projection_debug
from agent.world_model.artifacts import ArtifactManager
from agent.world_model.schemas import ObjectPlan, ToolResult
from agent.world_model.route_policy import (
    RouteContext,
    RoutePolicy,
    RoutePolicyValidationError,
    load_route_policy,
)
from agent.world_model.tool_call_planner import build_tool_call_planner
from agent.world_model.tool_calling import (
    ToolCall,
    ToolCallRequest,
    ToolCallResult,
    build_prompt_only_tool_call_instruction,
    default_qcpr_tool_registry,
)
from benchmark.clevrer import ClevrerQuestion, ClevrerScene
from benchmark.physion_pp import PhysionPPScene
from utils.config import ModelConfig
from utils.terminal import terminal_print


def _log_tool(tool_name: str, message: str) -> None:
    terminal_print(f"[query-conditioned-physical-rollout] tool={tool_name} {message}", flush=True)


def _trajectory_summary(payload: Dict[str, Any]) -> str:
    trajectories = payload.get("simulated_trajectories") or payload.get("trajectories") or []
    if not trajectories and isinstance(payload.get("predictions"), list):
        trajectories = payload["predictions"]
    frame_count = 0
    for item in trajectories:
        if not isinstance(item, dict):
            continue
        frames = item.get("frames") or item.get("poses") or item.get("trajectory") or []
        frame_count += len(frames)
    return f"backend={payload.get('backend') or payload.get('tool')} trajectories={len(trajectories)} frames={frame_count}"


_CLEVRER_COLORS = {"gray", "red", "blue", "green", "brown", "yellow", "cyan", "purple"}
_REFERENCE_COLORS = _CLEVRER_COLORS | {"gold", "silver", "metallic"}
_REFERENCE_MATERIALS = {"metal", "metallic", "rubber"}
_REFERENCE_SHAPES = {"sphere", "ball", "cylinder", "cube", "box", "object"}
_COLOR_ALIASES = {
    "gold": {"gold", "yellow", "brown"},
    "silver": {"silver", "gray"},
    "gray": {"gray", "silver"},
}
_IMPULSE_ANALYTIC_ROLLOUT_API_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "world_model" / "impulse_analytic_rollout_api.py"
)
_PHYSIONPP_FRICTION_FUTURE_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "world_model" / "physionpp_friction_future_rollout.py"
)
_PHYSIONPP_FRICTION_COLLISION_FUTURE_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "world_model"
    / "physionpp_friction_collision_future_rollout.py"
)
_PHYSIONPP_MASS_COLLISION_FUTURE_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "world_model"
    / "physionpp_mass_collision_future_rollout.py"
)
_PHYSIONPP_BOUNCY_WALL_FUTURE_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "world_model" / "physionpp_bouncy_wall_future_rollout.py"
)
_PHYSIONPP_BOUNCY_PLATFORM_FUTURE_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "world_model"
    / "physionpp_bouncy_platform_future_rollout.py"
)
QUESTION_ROLLOUT_BACKEND_DECISION_ID = "ROL-001.question_rollout_backend"
QUESTION_ROLLOUT_ROUTE_BY_SCENARIO = {
    "friction_platform_pp": (
        "rollout.surface_friction_analytic"
    ),
    "bouncy_wall_pp": "rollout.wall_bounce_analytic",
    "bouncy_platform_pp": (
        "rollout.platform_bounce_analytic"
    ),
    "friction_collision_pp": (
        "rollout.collision_friction_analytic"
    ),
    "mass_collision_pp": (
        "rollout.collision_mass_analytic"
    ),
}
QUESTION_ROLLOUT_ROUTE_CONTRACTS = {
    "rollout.tool_physics": {
        "fit_backend": "swr_backend.impulse_analytic",
        "artifact_backend": "rollout.tool_physics",
    },
    "rollout.surface_friction_analytic": {
        "fit_backend": "swr_backend.surface_friction_sphere",
        "artifact_backend": "physionpp_friction_future_analytic_rollout",
    },
    "rollout.wall_bounce_analytic": {
        "fit_backend": "swr_backend.wall_bounce_sphere",
        "artifact_backend": "rollout.wall_bounce_analytic",
    },
    "rollout.platform_bounce_analytic": {
        "fit_backend": "swr_backend.platform_bounce_sphere",
        "artifact_backend": "rollout.platform_bounce_analytic",
    },
    "rollout.collision_friction_analytic": {
        "fit_backend": "swr_backend.collision_friction_spheres",
        "artifact_backend": "rollout.collision_friction_analytic",
    },
    "rollout.collision_mass_analytic": {
        "fit_backend": "swr_backend.collision_mass_spheres",
        "artifact_backend": "rollout.collision_mass_analytic",
    },
}
QUESTION_ROLLOUT_BACKEND_ROUTES = frozenset(
    QUESTION_ROLLOUT_ROUTE_CONTRACTS
)
CLEVRER_TOOL_PLANNING_DECISION_ID = "ROL-002.clevrer_tool_planning"
CLEVRER_TOOL_PLANNING_ROUTE = "planning.two_round_tools"
CLEVRER_TOOL_PLAN_REPAIR_DECISION_ID = "ROL-003.clevrer_tool_plan_repair"
CLEVRER_TOOL_PLAN_REPAIR_ROUTE = (
    "planning.json_repair_fallback"
)
SWR_VISUAL_POSE_PRESERVATION_DECISION_ID = (
    "SWR-004.visual_pose_preservation"
)
SWR_VISUAL_POSE_PRESERVATION_ROUTES = frozenset(
    {
        "visual_pose.position_only",
        "visual_pose.corrected_rotation",
    }
)
COLLISION_VISUAL_POSE_SCENARIOS = frozenset(
    {"friction_collision_pp", "mass_collision_pp"}
)


def _require_swr_visual_pose_preservation_route(
    *,
    scene: Any,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    is_physion_pp = isinstance(scene, PhysionPPScene)
    scenario = str(getattr(scene, "scenario", "") or "").strip().lower()
    is_target_scope = (
        is_physion_pp and scenario in COLLISION_VISUAL_POSE_SCENARIOS
    )
    special_scene = (
        getattr(object_plan, "special_scene", {})
        if isinstance(getattr(object_plan, "special_scene", {}), dict)
        else {}
    )
    route_record = special_scene.get("swr_visual_pose_preservation_route")
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise RoutePolicyValidationError(
                "collision Physion++ rollout is missing its "
                "SWR visual-pose-preservation route"
            )
        return None
    if not is_target_scope:
        raise RoutePolicyValidationError(
            "SWR visual-pose-preservation route is not applicable to "
            f"scenario {scenario!r}"
        )
    if (
        route_record.get("decision_id")
        != SWR_VISUAL_POSE_PRESERVATION_DECISION_ID
    ):
        raise RoutePolicyValidationError(
            "SWR visual-pose-preservation route has an unexpected decision_id"
        )
    route = route_record.get("route")
    if route not in SWR_VISUAL_POSE_PRESERVATION_ROUTES:
        raise RoutePolicyValidationError(
            f"unsupported SWR visual-pose-preservation route: {route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise RoutePolicyValidationError(
            "SWR visual-pose-preservation route is missing its context"
        )
    if context.get("benchmark") != "physion_pp":
        raise RoutePolicyValidationError(
            "SWR visual-pose-preservation benchmark context mismatch"
        )
    if context.get("scenario") != scenario:
        raise RoutePolicyValidationError(
            "SWR visual-pose-preservation scenario context mismatch"
        )
    if route == "visual_pose.corrected_rotation":
        if (
            route_record.get("option_id")
            != "option.visual_pose.corrected_rotation"
        ):
            raise RoutePolicyValidationError(
                "corrected-rotation visual pose preservation requires "
                "ALT-SWR-001"
            )
    return route_record


def _question_rollout_policy_context(
    scene: Any,
    question: Any,
) -> RouteContext | None:
    if isinstance(scene, PhysionPPScene):
        benchmark = "physion_pp"
        scenario = str(getattr(scene, "scenario", "") or "").strip() or None
    elif isinstance(scene, ClevrerScene):
        benchmark = "clevrer"
        scenario = None
    else:
        return None
    return RouteContext(
        benchmark=benchmark,
        scenario=scenario,
        question_type=(
            str(getattr(question, "question_type", "") or "").strip() or None
        ),
    )


def _resolve_question_rollout_backend_route(
    scene: Any,
    question: Any,
    *,
    policy: RoutePolicy | None = None,
) -> Dict[str, Any] | None:
    context = _question_rollout_policy_context(scene, question)
    if context is None:
        return None
    active_policy = policy or load_route_policy()
    return active_policy.resolve_record(
        QUESTION_ROLLOUT_BACKEND_DECISION_ID,
        context,
    )


def _require_question_rollout_backend_route(
    route_record: Dict[str, Any] | None,
    *,
    scene: Any,
    question: Any,
) -> Dict[str, Any] | None:
    context = _question_rollout_policy_context(scene, question)
    if context is None:
        if route_record is not None:
            raise RoutePolicyValidationError(
                "question-rollout route is not applicable to this benchmark"
            )
        return None
    if route_record is None:
        raise RoutePolicyValidationError(
            "missing ROL-001 question-rollout backend route record"
        )
    if (
        route_record.get("decision_id")
        != QUESTION_ROLLOUT_BACKEND_DECISION_ID
    ):
        raise RoutePolicyValidationError(
            "question-rollout route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    if route not in QUESTION_ROLLOUT_BACKEND_ROUTES:
        raise RoutePolicyValidationError(
            f"unsupported question-rollout backend route: {route!r}"
        )
    record_context = route_record.get("context")
    if not isinstance(record_context, dict):
        raise RoutePolicyValidationError(
            "question-rollout route is missing context"
        )
    expected_context = {
        "benchmark": context.benchmark,
        "scenario": context.scenario,
        "question_type": context.question_type,
    }
    observed_context = {
        key: record_context.get(key)
        for key in expected_context
    }
    if observed_context != expected_context:
        raise RoutePolicyValidationError(
            "question-rollout route context mismatch: "
            f"{observed_context!r} != {expected_context!r}"
        )
    expected_route = (
        "rollout.tool_physics"
        if context.benchmark == "clevrer"
        else QUESTION_ROLLOUT_ROUTE_BY_SCENARIO.get(
            str(context.scenario or "")
        )
    )
    if route != expected_route:
        raise RoutePolicyValidationError(
            "question-rollout route does not match benchmark/scenario: "
            f"{route!r} != {expected_route!r}"
        )
    return route_record


def _clevrer_tool_planning_policy_context(
    scene: Any,
    question: Any,
) -> RouteContext | None:
    if isinstance(scene, PhysionPPScene) or not isinstance(scene, ClevrerScene):
        return None
    return RouteContext(
        benchmark="clevrer",
        question_type=(
            str(getattr(question, "question_type", "") or "").strip()
            or None
        ),
    )


def _resolve_clevrer_tool_planning_route(
    scene: Any,
    question: Any,
    *,
    policy: RoutePolicy | None = None,
) -> Dict[str, Any] | None:
    context = _clevrer_tool_planning_policy_context(scene, question)
    if context is None:
        return None
    active_policy = policy or load_route_policy()
    return active_policy.resolve_record(
        CLEVRER_TOOL_PLANNING_DECISION_ID,
        context,
    )


def _require_clevrer_tool_planning_route(
    route_record: Dict[str, Any] | None,
    *,
    scene: Any,
    question: Any,
) -> Dict[str, Any] | None:
    context = _clevrer_tool_planning_policy_context(scene, question)
    if context is None:
        if route_record is not None:
            raise RoutePolicyValidationError(
                "CLEVRER tool-planning route is not applicable to this benchmark"
            )
        return None
    if route_record is None:
        raise RoutePolicyValidationError(
            "missing ROL-002 CLEVRER tool-planning route record"
        )
    if route_record.get("decision_id") != CLEVRER_TOOL_PLANNING_DECISION_ID:
        raise RoutePolicyValidationError(
            "CLEVRER tool-planning route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    if route != CLEVRER_TOOL_PLANNING_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported CLEVRER tool-planning route: {route!r}"
        )
    record_context = route_record.get("context")
    if not isinstance(record_context, dict):
        raise RoutePolicyValidationError(
            "CLEVRER tool-planning route is missing context"
        )
    expected_context = {
        "benchmark": context.benchmark,
        "scenario": context.scenario,
        "question_type": context.question_type,
    }
    observed_context = {
        key: record_context.get(key)
        for key in expected_context
    }
    if observed_context != expected_context:
        raise RoutePolicyValidationError(
            "CLEVRER tool-planning route context mismatch: "
            f"{observed_context!r} != {expected_context!r}"
        )
    return route_record


def _resolve_clevrer_tool_plan_repair_route(
    scene: Any,
    question: Any,
    *,
    policy: RoutePolicy | None = None,
) -> Dict[str, Any] | None:
    context = _clevrer_tool_planning_policy_context(scene, question)
    if context is None:
        return None
    active_policy = policy or load_route_policy()
    return active_policy.resolve_record(
        CLEVRER_TOOL_PLAN_REPAIR_DECISION_ID,
        context,
    )


def _require_clevrer_tool_plan_repair_route(
    route_record: Dict[str, Any] | None,
    *,
    scene: Any,
    question: Any,
) -> Dict[str, Any] | None:
    context = _clevrer_tool_planning_policy_context(scene, question)
    if context is None:
        if route_record is not None:
            raise RoutePolicyValidationError(
                "CLEVRER tool-plan repair route is not applicable to this benchmark"
            )
        return None
    if route_record is None:
        raise RoutePolicyValidationError(
            "missing ROL-003 CLEVRER tool-plan repair route record"
        )
    if route_record.get("decision_id") != CLEVRER_TOOL_PLAN_REPAIR_DECISION_ID:
        raise RoutePolicyValidationError(
            "CLEVRER tool-plan repair route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    if route != CLEVRER_TOOL_PLAN_REPAIR_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported CLEVRER tool-plan repair route: {route!r}"
        )
    record_context = route_record.get("context")
    if not isinstance(record_context, dict):
        raise RoutePolicyValidationError(
            "CLEVRER tool-plan repair route is missing context"
        )
    expected_context = {
        "benchmark": context.benchmark,
        "scenario": context.scenario,
        "question_type": context.question_type,
    }
    observed_context = {
        key: record_context.get(key)
        for key in expected_context
    }
    if observed_context != expected_context:
        raise RoutePolicyValidationError(
            "CLEVRER tool-plan repair route context mismatch: "
            f"{observed_context!r} != {expected_context!r}"
        )
    return route_record


def _record_clevrer_tool_planning_route(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
    *,
    require_planning_metadata: bool,
) -> None:
    existing_route = payload.get("clevrer_tool_planning_route")
    if existing_route is not None and existing_route != route_record:
        raise RoutePolicyValidationError(
            "trajectory CLEVRER tool-planning route mismatch"
        )
    payload["clevrer_tool_planning_route"] = deepcopy(route_record)
    if not require_planning_metadata:
        return
    planning = payload.get("tool_planning")
    if not isinstance(planning, dict):
        raise RoutePolicyValidationError(
            "CLEVRER trajectory is missing tool-planning metadata"
        )
    if planning.get("planner") != CLEVRER_TOOL_PLANNING_ROUTE:
        raise RoutePolicyValidationError(
            "trajectory tool-planning implementation conflicts with the "
            f"resolved route: {planning.get('planner')!r} != "
            f"{CLEVRER_TOOL_PLANNING_ROUTE!r}"
        )
    nested_route = planning.get("clevrer_tool_planning_route")
    if nested_route is not None and nested_route != route_record:
        raise RoutePolicyValidationError(
            "trajectory tool-planning metadata route mismatch"
        )
    planning["clevrer_tool_planning_route"] = deepcopy(route_record)


def _load_impulse_analytic_rollout_api() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_impulse_analytic_rollout_api_for_qcpr",
        _IMPULSE_ANALYTIC_ROLLOUT_API_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load impulse analytic rollout API: {_IMPULSE_ANALYTIC_ROLLOUT_API_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_physionpp_friction_future_rollout() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_physionpp_friction_future_rollout_for_qcpr",
        _PHYSIONPP_FRICTION_FUTURE_ROLLOUT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Physion++ future rollout helper: {_PHYSIONPP_FRICTION_FUTURE_ROLLOUT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_physionpp_friction_collision_future_rollout() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_physionpp_friction_collision_future_rollout_for_qcpr",
        _PHYSIONPP_FRICTION_COLLISION_FUTURE_ROLLOUT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "failed to load Physion++ friction-collision future rollout helper: "
            f"{_PHYSIONPP_FRICTION_COLLISION_FUTURE_ROLLOUT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_physionpp_mass_collision_future_rollout() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_physionpp_mass_collision_future_rollout_for_qcpr",
        _PHYSIONPP_MASS_COLLISION_FUTURE_ROLLOUT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "failed to load Physion++ mass-collision future rollout helper: "
            f"{_PHYSIONPP_MASS_COLLISION_FUTURE_ROLLOUT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_physionpp_bouncy_wall_future_rollout() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_physionpp_bouncy_wall_future_rollout_for_qcpr",
        _PHYSIONPP_BOUNCY_WALL_FUTURE_ROLLOUT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "failed to load Physion++ bouncy-wall future rollout helper: "
            f"{_PHYSIONPP_BOUNCY_WALL_FUTURE_ROLLOUT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_physionpp_bouncy_platform_future_rollout() -> Any:
    spec = importlib.util.spec_from_file_location(
        "physmind_physionpp_bouncy_platform_future_rollout_for_qcpr",
        _PHYSIONPP_BOUNCY_PLATFORM_FUTURE_ROLLOUT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "failed to load Physion++ bouncy-platform future rollout helper: "
            f"{_PHYSIONPP_BOUNCY_PLATFORM_FUTURE_ROLLOUT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_vector(value: Any, size: int = 3) -> list[float] | None:
    if not isinstance(value, list) or len(value) < size:
        return None
    output = []
    for item in value[:size]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        output.append(number)
    return output


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _normalize_color(value: Any, description: str = "") -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in _CLEVRER_COLORS:
        return text
    combined = f"{text} {description.lower()}"
    for color in _CLEVRER_COLORS:
        if color in combined:
            return color
    if "gold" in combined or "orange" in combined:
        return "yellow"
    if "silver" in combined:
        return "gray"
    return "gray"


def _normalize_shape(value: Any, description: str = "") -> str:
    text = str(value or "").strip().lower()
    combined = f"{text} {description.lower()}"
    if "sphere" in combined or "ball" in combined:
        return "sphere"
    if "cylinder" in combined:
        return "cylinder"
    if "cube" in combined or "box" in combined:
        return "cube"
    return "cube"


def _normalize_material(value: Any, description: str = "") -> str:
    combined = f"{value or ''} {description}".lower()
    if "metal" in combined or "metallic" in combined:
        return "metal"
    return "rubber"


def _shape_terms(value: str) -> set[str]:
    shape = _normalize_shape(value)
    if shape == "sphere":
        return {"sphere", "ball"}
    if shape == "cylinder":
        return {"cylinder"}
    if shape == "cube":
        return {"cube", "box"}
    return {shape}


def _tokenize_reference(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower().replace("-", " "))


def _reference_terms_for_object(obj: dict[str, Any]) -> dict[str, set[str]]:
    color = str(obj.get("raw_color") or obj.get("color") or "").strip().lower()
    material = str(obj.get("material") or "").strip().lower()
    shape = str(obj.get("shape") or "").strip().lower()
    description_tokens = set(_tokenize_reference(str(obj.get("description") or "")))
    colors = {color} if color else set()
    colors.update(_COLOR_ALIASES.get(color, set()))
    if "gold" in description_tokens:
        colors.update(_COLOR_ALIASES["gold"])
    if "silver" in description_tokens:
        colors.update(_COLOR_ALIASES["silver"])
    materials = {material} if material else set()
    if "metallic" in description_tokens or "metal" in description_tokens:
        materials.update({"metal", "metallic"})
    if "rubber" in description_tokens:
        materials.add("rubber")
    shapes = _shape_terms(shape)
    shapes.update(token for token in description_tokens if token in _REFERENCE_SHAPES and token != "object")
    return {
        "colors": colors,
        "materials": materials,
        "shapes": shapes,
        "description_tokens": description_tokens,
    }


def _extract_object_references(question: ClevrerQuestion) -> list[str]:
    texts = [question.question] + [choice.choice for choice in question.choices]
    refs: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        ref = " ".join(_tokenize_reference(value))
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for text in texts:
        tokens = _tokenize_reference(text)
        for index, token in enumerate(tokens):
            if token in _REFERENCE_SHAPES and token != "object":
                start = index
                while start > 0 and tokens[start - 1] in (_REFERENCE_COLORS | _REFERENCE_MATERIALS):
                    start -= 1
                add(" ".join(tokens[start : index + 1]))
            if token == "object" and index > 0 and tokens[index - 1] in (_REFERENCE_COLORS | _REFERENCE_MATERIALS):
                add(" ".join(tokens[index - 1 : index + 1]))
        for pattern in [
            r"\bif\s+(?:the\s+)?(.+?)\s+is\s+removed\b",
            r"\bwithout\s+(?:the\s+)?(.+?)(?:,|\?|$)",
        ]:
            match = re.search(pattern, text.lower())
            if match:
                value = match.group(1)
                value = re.split(r"\bwhich\b|\bwhat\b|\bwill\b", value)[0]
                add(value)
    return refs


def _score_reference_object(reference: str, obj: dict[str, Any]) -> tuple[float, list[str]]:
    tokens = set(_tokenize_reference(reference))
    terms = _reference_terms_for_object(obj)
    score = 0.0
    reasons: list[str] = []

    raw_color = str(obj.get("raw_color") or obj.get("color") or "").lower()
    exact_colors = tokens & {raw_color}
    alias_colors = (tokens & terms["colors"]) - exact_colors
    materials = tokens & terms["materials"]
    shapes = (tokens & terms["shapes"]) - {"object"}
    description_hits = tokens & terms["description_tokens"]

    if exact_colors:
        score += 4.0 * len(exact_colors)
        reasons.append(f"exact_color={sorted(exact_colors)}")
    if alias_colors:
        score += 2.0 * len(alias_colors)
        reasons.append(f"color_alias={sorted(alias_colors)}")
    if materials:
        score += 2.5 * len(materials)
        reasons.append(f"material={sorted(materials)}")
    if shapes:
        score += 3.0 * len(shapes)
        reasons.append(f"shape={sorted(shapes)}")
    if description_hits:
        score += 0.25 * len(description_hits)
    return score, reasons


def _build_object_reference_map(question: ClevrerQuestion, objects: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for reference in _extract_object_references(question):
        candidates = []
        for obj in objects:
            score, reasons = _score_reference_object(reference, obj)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "object_id": obj.get("object_id"),
                    "id": obj.get("id"),
                    "score": float(score),
                    "description": obj.get("description"),
                    "color": obj.get("color"),
                    "material": obj.get("material"),
                    "shape": obj.get("shape"),
                    "reasons": reasons,
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), str(item.get("object_id"))))
        if candidates:
            top_score = float(candidates[0]["score"])
            second_score = float(candidates[1]["score"]) if len(candidates) > 1 else None
            entries.append(
                {
                    "reference": reference,
                    "object_id": candidates[0].get("object_id"),
                    "id": candidates[0].get("id"),
                    "score": top_score,
                    "margin": None if second_score is None else top_score - second_score,
                    "candidates": candidates[:4],
                }
            )
    return {
        "status": "ok",
        "method": "deterministic_text_to_object_catalog",
        "references": entries,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    answer = extract_answer_tag(text) or text
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in VLM object reference response.")
    return json.loads(match.group(0))


def _answer_text_only(*, config: ModelConfig, prompt: str, request_context: dict[str, Any]) -> str:
    print(
        f"[request] scene_index={request_context.get('scene_index')} "
        f"question_id={request_context.get('question_id')} "
        f"question_type={request_context.get('question_type')} "
        f"provider={config.provider} model={config.model} mode=text-only timeout={config.request_timeout}s"
    )
    if config.provider == "google":
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model)
        response = model.generate_content(
            [prompt],
            generation_config={"max_output_tokens": config.max_output_tokens},
            stream=False,
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text.strip()

    client = create_openai_compatible_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=config.max_output_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _reference_attribute_payload(references: list[str]) -> dict[str, Any]:
    phrases = []
    colors: set[str] = set()
    shapes: set[str] = set()
    materials: set[str] = set()
    other_descriptors: set[str] = set()
    for reference in references:
        tokens = set(_tokenize_reference(reference))
        ref_colors = sorted(tokens & _REFERENCE_COLORS)
        ref_shapes = sorted((tokens & _REFERENCE_SHAPES) - {"object"})
        ref_materials = sorted(tokens & _REFERENCE_MATERIALS)
        ref_other = sorted((tokens & {"object"}) - set(ref_shapes))
        colors.update(ref_colors)
        shapes.update(ref_shapes)
        materials.update(ref_materials)
        other_descriptors.update(ref_other)
        phrases.append(
            {
                "phrase": reference,
                "colors": ref_colors,
                "shapes": ref_shapes,
                "materials": ref_materials,
                "other_descriptors": ref_other,
                "required_attributes": sorted(set(ref_colors + ref_shapes + ref_materials)),
            }
        )
    return {
        "phrases": phrases,
        "attribute_vocabulary": {
            "colors": sorted(colors),
            "shapes": sorted(shapes),
            "materials": sorted(materials),
            "other_descriptors": sorted(other_descriptors),
        },
    }


def _track_representative_inputs(world_model_dir: Path) -> tuple[list[dict[str, Any]], list[Path], list[str]]:
    labels_path = (
        world_model_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_track_labels"
        / "sam3_video_track_labels.json"
    )
    inputs_path = (
        world_model_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_track_labels"
        / "sam3_video_track_label_inputs.json"
    )
    if not labels_path.exists() or not inputs_path.exists():
        return [], [], []
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    inputs_payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    labels_by_track = {
        str(item.get("track_id")): item
        for item in labels_payload.get("track_labels", [])
        if isinstance(item, dict) and item.get("track_id")
    }
    tracks = []
    image_paths: list[Path] = []
    image_labels: list[str] = []
    for item in inputs_payload.get("tracks", []):
        if not isinstance(item, dict):
            continue
        track_id = str(item.get("track_id") or "")
        overlay_image = Path(str(item.get("representative_overlay_path") or item.get("representative_image_path") or ""))
        label = labels_by_track.get(track_id, {})
        object_id = str(label.get("object_id") or "")
        if not track_id or not object_id or not overlay_image.exists():
            continue
        representative_frame = item.get("representative_frame") if isinstance(item.get("representative_frame"), dict) else {}
        tracks.append(
            {
                "object_id": object_id,
                "track_id": track_id,
                "first_frame_index": item.get("first_frame_index"),
                "last_frame_index": item.get("last_frame_index"),
                "representative_frame_index": representative_frame.get("frame_index"),
                "representative_image_path": str(overlay_image),
                "representative_overlay_path": str(overlay_image),
            }
        )
        image_paths.append(overlay_image)
        image_labels.append(object_id)
    return tracks, image_paths, image_labels


def _normalize_visual_reference_map(
    *,
    payload: dict[str, Any],
    references: list[str],
    tracks: list[dict[str, Any]],
    raw_response: str,
    required_attributes_by_reference: dict[str, set[str]] | None = None,
    attributes_by_object: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    valid_by_object = {str(item["object_id"]): item for item in tracks}
    valid_track_by_object = {str(item["object_id"]): str(item["track_id"]) for item in tracks}
    entries = []
    raw_entries = payload.get("references")
    if not isinstance(raw_entries, list):
        raise ValueError("VLM object reference response must contain a references list.")
    reference_set = set(references)
    seen: set[str] = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        reference = " ".join(_tokenize_reference(str(item.get("reference") or "")))
        object_id = str(item.get("object_id") or "")
        if reference not in reference_set or reference in seen or object_id not in valid_by_object:
            continue
        required = (required_attributes_by_reference or {}).get(reference, set())
        object_attributes = (attributes_by_object or {}).get(object_id, set())
        enforce_required = bool(required) and any(
            required.issubset(candidate_attributes)
            for candidate_attributes in (attributes_by_object or {}).values()
        )
        if enforce_required and not required.issubset(object_attributes):
            continue
        seen.add(reference)
        track = valid_by_object[object_id]
        entries.append(
            {
                "reference": reference,
                "object_id": object_id,
                "id": None,
                "track_id": valid_track_by_object[object_id],
                "score": None,
                "margin": None,
                "reason": str(item.get("reason") or ""),
                "candidates": [
                    {
                        "object_id": object_id,
                        "track_id": valid_track_by_object[object_id],
                        "representative_image_path": track.get("representative_image_path"),
                        "representative_frame_index": track.get("representative_frame_index"),
                    }
                ],
            }
        )
    return {
        "status": "ok",
        "method": "vlm_visual_track_representative_image",
        "references": entries,
        "raw_response": raw_response,
    }


def _known_attribute_set(attribute_payload: dict[str, Any]) -> set[str]:
    vocabulary = attribute_payload.get("attribute_vocabulary") if isinstance(attribute_payload, dict) else {}
    if not isinstance(vocabulary, dict):
        return set()
    output: set[str] = set()
    for key in ("colors", "shapes", "materials", "other_descriptors"):
        values = vocabulary.get(key) or []
        if isinstance(values, list):
            output.update(str(value).strip().lower() for value in values if str(value).strip())
    return output


def _normalize_phrase_attribute_payload(payload: dict[str, Any], references: list[str]) -> dict[str, Any]:
    fallback = _reference_attribute_payload(references)
    reference_set = set(references)
    known = _known_attribute_set(fallback)
    phrases = []
    seen: set[str] = set()
    raw_phrases = payload.get("phrases") if isinstance(payload, dict) else None
    if isinstance(raw_phrases, list):
        for item in raw_phrases:
            if not isinstance(item, dict):
                continue
            phrase = " ".join(_tokenize_reference(str(item.get("phrase") or "")))
            if phrase not in reference_set or phrase in seen:
                continue
            raw_required = item.get("required_attributes") or []
            required = sorted({
                str(value).strip().lower()
                for value in raw_required
                if str(value).strip().lower() in known
            })
            if not required:
                required = next(
                    (
                        list(entry.get("required_attributes") or [])
                        for entry in fallback["phrases"]
                        if entry.get("phrase") == phrase
                    ),
                    [],
                )
            phrases.append({"phrase": phrase, "required_attributes": required})
            seen.add(phrase)
    for entry in fallback["phrases"]:
        phrase = str(entry.get("phrase") or "")
        if phrase not in seen:
            phrases.append({"phrase": phrase, "required_attributes": list(entry.get("required_attributes") or [])})
    return {"phrases": phrases, "attribute_vocabulary": fallback["attribute_vocabulary"], "fallback_payload": fallback}


def _normalize_object_attribute_payload(
    payload: dict[str, Any],
    *,
    tracks: list[dict[str, Any]],
    allowed_attributes: set[str],
    raw_response: str,
) -> dict[str, Any]:
    valid_ids = {str(item.get("object_id")) for item in tracks}
    objects = []
    raw_objects = payload.get("objects") if isinstance(payload, dict) else None
    if isinstance(raw_objects, list):
        for item in raw_objects:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "")
            if object_id not in valid_ids:
                continue
            raw_attributes = item.get("attributes") or []
            attributes = sorted({
                str(value).strip().lower()
                for value in raw_attributes
                if str(value).strip().lower() in allowed_attributes
            })
            objects.append(
                {
                    "object_id": object_id,
                    "attributes": attributes,
                    "reason": str(item.get("reason") or ""),
                }
            )
    seen = {str(item.get("object_id")) for item in objects}
    for track in tracks:
        object_id = str(track.get("object_id") or "")
        if object_id and object_id not in seen:
            objects.append({"object_id": object_id, "attributes": [], "reason": "missing from VLM response"})
    return {"objects": objects, "raw_response": raw_response}


def _match_phrases_from_attributes(
    *,
    phrase_attributes: dict[str, Any],
    object_attributes: dict[str, Any],
    tracks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    track_by_object = {str(item.get("object_id")): item for item in tracks}
    attributes_by_object = {
        str(item.get("object_id")): set(str(value) for value in item.get("attributes") or [])
        for item in object_attributes.get("objects", [])
        if isinstance(item, dict)
    }
    entries = []
    unresolved = []
    ambiguous = []
    for item in phrase_attributes.get("phrases", []):
        if not isinstance(item, dict):
            continue
        phrase = str(item.get("phrase") or "")
        required = set(str(value) for value in item.get("required_attributes") or [])
        if not phrase or not required:
            unresolved.append(phrase)
            continue
        candidates = [
            object_id
            for object_id, attributes in attributes_by_object.items()
            if required.issubset(attributes)
        ]
        if len(candidates) != 1:
            (ambiguous if candidates else unresolved).append(phrase)
            continue
        object_id = candidates[0]
        track = track_by_object.get(object_id, {})
        entries.append(
            {
                "reference": phrase,
                "object_id": object_id,
                "id": None,
                "track_id": track.get("track_id"),
                "score": None,
                "margin": None,
                "reason": f"matched required attributes {sorted(required)}",
                "candidates": [
                    {
                        "object_id": object_id,
                        "track_id": track.get("track_id"),
                        "representative_image_path": track.get("representative_image_path"),
                        "representative_frame_index": track.get("representative_frame_index"),
                        "attributes": sorted(attributes_by_object.get(object_id, set())),
                    }
                ],
            }
        )
    return entries, unresolved, ambiguous


def _required_attributes_for_phrase(phrase_attributes: dict[str, Any], phrase: str) -> list[str]:
    normalized = " ".join(_tokenize_reference(phrase))
    for item in phrase_attributes.get("phrases", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("phrase") or "") == normalized:
            return [str(value) for value in item.get("required_attributes") or [] if str(value)]
    return []


def _attributes_by_object(object_attributes: dict[str, Any]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for item in object_attributes.get("objects", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_id:
            output[object_id] = {str(value) for value in item.get("attributes") or []}
    return output


class QueryConditionedPhysicalRolloutAdapter:
    tool_name = "query_conditioned_physical_rollout"

    def __init__(self, artifacts: ArtifactManager, config: ModelConfig | None = None, dry_run: bool = False):
        self.artifacts = artifacts
        self.config = config
        self.dry_run = dry_run

    def _terminal_corrected_rotations_from_fit(
        self,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, List[List[float]]]:
        terminal: Dict[str, tuple[int, List[List[float]]]] = {}
        segments = world_reconstruction_fit.get("segments")
        if not isinstance(segments, list):
            return {}
        for segment in segments:
            trajectories = (
                segment.get("target_trajectories")
                if isinstance(segment, dict)
                else None
            )
            if not isinstance(trajectories, dict):
                continue
            for raw_object_id, records in trajectories.items():
                object_id = str(raw_object_id)
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    frame_index = record.get("frame_index")
                    rotation = record.get("rotation_blender_world_3x3")
                    if frame_index is None or not (
                        isinstance(rotation, list) and len(rotation) == 3
                    ):
                        continue
                    try:
                        normalized = [
                            [float(value) for value in row]
                            for row in rotation
                        ]
                    except (TypeError, ValueError):
                        continue
                    if any(len(row) != 3 for row in normalized):
                        continue
                    frame = int(frame_index)
                    previous = terminal.get(object_id)
                    if previous is None or frame > previous[0]:
                        terminal[object_id] = (frame, normalized)
        return {
            object_id: rotation
            for object_id, (_frame, rotation) in terminal.items()
        }

    def _apply_future_visual_pose_preservation(
        self,
        *,
        payload: Dict[str, Any],
        world_reconstruction_fit: Dict[str, Any],
        route_record: Dict[str, Any],
    ) -> None:
        route = str(route_record.get("route") or "")
        if route not in SWR_VISUAL_POSE_PRESERVATION_ROUTES:
            raise RoutePolicyValidationError(
                f"unsupported SWR visual-pose-preservation route: {route!r}"
            )
        payload["swr_visual_pose_preservation_route"] = deepcopy(route_record)
        future = payload.get("physion_pp_future_rollout")
        if not isinstance(future, dict):
            raise RoutePolicyValidationError(
                "collision rollout is missing physion_pp_future_rollout"
            )
        if route == "visual_pose.position_only":
            future["visual_pose_preservation"] = {
                "enabled": False,
                "rotation_policy": None,
                "affects_physics": False,
                "affects_contact_answer": False,
                "attached_future_record_count": 0,
            }
            return
        terminal_rotations = self._terminal_corrected_rotations_from_fit(
            world_reconstruction_fit
        )
        trajectories = future.get("future_trajectories")
        if not isinstance(trajectories, dict):
            raise RoutePolicyValidationError(
                "collision rollout has no future_trajectories to decorate"
            )
        decorated_count = 0
        for raw_object_id, records in trajectories.items():
            object_id = str(raw_object_id)
            if not isinstance(records, list) or not records:
                continue
            rotation = terminal_rotations.get(object_id)
            if rotation is None:
                raise RoutePolicyValidationError(
                    "future visual pose preservation is missing the terminal "
                    f"corrected rotation for object {object_id}"
                )
            for record in records:
                if not isinstance(record, dict):
                    continue
                position = record.get("position_blender_world_m")
                if not isinstance(position, list) or len(position) != 3:
                    raise RoutePolicyValidationError(
                        "future visual pose preservation found an invalid "
                        f"position for object {object_id}"
                    )
                pose = [
                    [*rotation[0], float(position[0])],
                    [*rotation[1], float(position[1])],
                    [*rotation[2], float(position[2])],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                record["rotation_blender_world_3x3"] = deepcopy(rotation)
                record["pose_blender_world_4x4"] = pose
                record["visual_pose_only"] = True
                record["visual_rotation_policy"] = (
                    "terminal_observed_corrected_rotation_held_constant"
                )
                decorated_count += 1
        future["visual_pose_preservation"] = {
            "enabled": True,
            "rotation_policy": (
                "terminal_observed_corrected_rotation_held_constant"
            ),
            "affects_physics": False,
            "affects_contact_answer": False,
            "attached_future_record_count": decorated_count,
        }

    def run(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        question_dir: Path,
        object_plan: ObjectPlan,
        stage_results: List[ToolResult],
        world_model_dir: Path | None = None,
        question_rollout_backend_route: Dict[str, Any] | None = None,
        clevrer_tool_planning_route: Dict[str, Any] | None = None,
    ) -> ToolResult:
        active_route_record = None
        active_tool_planning_record = None
        active_visual_pose_record = None
        if not self.dry_run:
            active_route_record = question_rollout_backend_route
            if active_route_record is None:
                active_route_record = _resolve_question_rollout_backend_route(
                    scene,
                    question,
                )
            active_route_record = _require_question_rollout_backend_route(
                active_route_record,
                scene=scene,
                question=question,
            )
            active_tool_planning_record = clevrer_tool_planning_route
            if active_tool_planning_record is None:
                active_tool_planning_record = (
                    _resolve_clevrer_tool_planning_route(
                        scene,
                        question,
                    )
                )
            active_tool_planning_record = (
                _require_clevrer_tool_planning_route(
                    active_tool_planning_record,
                    scene=scene,
                    question=question,
                )
            )
            if active_route_record is not None:
                is_tool_call_rollout = (
                    active_route_record.get("route")
                    == "rollout.tool_physics"
                )
                if is_tool_call_rollout != (
                    active_tool_planning_record is not None
                ):
                    raise RoutePolicyValidationError(
                        "ROL-001 rollout backend and ROL-002 tool-planning "
                        "applicability disagree"
                    )
            active_visual_pose_record = (
                _require_swr_visual_pose_preservation_route(
                    scene=scene,
                    object_plan=object_plan,
                )
            )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, "trajectory.json")
        existing = self.artifacts.read_optional(artifact_path)
        if existing:
            if active_route_record is not None:
                route = str(active_route_record["route"])
                expected_backend = str(
                    QUESTION_ROLLOUT_ROUTE_CONTRACTS[route][
                        "artifact_backend"
                    ]
                )
                existing_route = existing.get(
                    "question_rollout_backend_route"
                )
                if (
                    existing_route is not None
                    and existing_route != active_route_record
                ):
                    raise RoutePolicyValidationError(
                        "cached trajectory question-rollout route mismatch"
                    )
                if existing.get("backend") != expected_backend:
                    raise RoutePolicyValidationError(
                        "cached trajectory backend conflicts with the resolved "
                        f"route: {existing.get('backend')!r} != "
                        f"{expected_backend!r}"
                    )
                if active_visual_pose_record is not None:
                    existing_visual_route = existing.get(
                        "swr_visual_pose_preservation_route"
                    )
                    if existing_visual_route != active_visual_pose_record:
                        raise RoutePolicyValidationError(
                            "cached trajectory SWR visual-pose-preservation "
                            "route mismatch"
                        )
                existing["question_rollout_backend_route"] = deepcopy(
                    active_route_record
                )
                if active_tool_planning_record is not None:
                    _record_clevrer_tool_planning_route(
                        existing,
                        active_tool_planning_record,
                        require_planning_metadata=True,
                    )
                elif existing.get("clevrer_tool_planning_route") is not None:
                    raise RoutePolicyValidationError(
                        "non-CLEVRER trajectory contains a CLEVRER "
                        "tool-planning route"
                    )
                self.artifacts.write(artifact_path, existing)
            _log_tool(self.tool_name, f"loaded artifact={artifact_path} {_trajectory_summary(existing)}")
            return ToolResult(self.tool_name, "loaded", str(artifact_path), payload=existing)
        payload = {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [result.to_dict() for result in stage_results],
            "backend": "query_conditioned_analytic_rollout",
            "simulated_trajectories": [],
            "fit_error": None,
        }
        if world_model_dir is not None:
            payload["world_model_artifact_dir"] = str(world_model_dir)
        if self.dry_run:
            self.artifacts.write(artifact_path, payload)
            _log_tool(self.tool_name, f"dry_run artifact={artifact_path} {_trajectory_summary(payload)}")
            return ToolResult(self.tool_name, "dry_run", str(artifact_path), payload=payload)

        source_dir = world_model_dir or question_dir
        world_reconstruction_fit = self._read_optional(
            source_dir
            / "simulatable-world-reconstruction"
            / "fit"
            / "world_reconstruction_fit.json"
        )
        if world_reconstruction_fit:
            if active_visual_pose_record is not None:
                fit_visual_route = world_reconstruction_fit.get(
                    "swr_visual_pose_preservation_route"
                )
                if fit_visual_route != active_visual_pose_record:
                    raise RoutePolicyValidationError(
                        "SWR fit visual-pose-preservation route conflicts "
                        "with the object plan"
                    )
            if active_route_record is None:
                payload = self._analytic_rollout_payload(
                    scene=scene,
                    question=question,
                    object_plan=object_plan,
                    question_dir=question_dir,
                    world_reconstruction_fit=world_reconstruction_fit,
                    clevrer_tool_planning_route=(
                        active_tool_planning_record
                    ),
                )
            else:
                route = str(active_route_record["route"])
                contract = QUESTION_ROLLOUT_ROUTE_CONTRACTS[route]
                observed_fit_backend = world_reconstruction_fit.get("backend")
                if observed_fit_backend != contract["fit_backend"]:
                    raise RoutePolicyValidationError(
                        "SWR fit backend conflicts with the resolved question "
                        f"rollout route: {observed_fit_backend!r} != "
                        f"{contract['fit_backend']!r}"
                    )
                builders = {
                    "rollout.tool_physics": lambda: self._analytic_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        question_dir=question_dir,
                        world_reconstruction_fit=world_reconstruction_fit,
                        clevrer_tool_planning_route=(
                            active_tool_planning_record
                        ),
                    ),
                    "rollout.surface_friction_analytic": lambda: self._physionpp_friction_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        world_model_dir=source_dir,
                        world_reconstruction_fit=world_reconstruction_fit,
                    ),
                    "rollout.collision_friction_analytic": lambda: self._physionpp_friction_collision_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        world_reconstruction_fit=world_reconstruction_fit,
                    ),
                    "rollout.collision_mass_analytic": lambda: self._physionpp_mass_collision_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        world_reconstruction_fit=world_reconstruction_fit,
                    ),
                    "rollout.wall_bounce_analytic": lambda: self._physionpp_bouncy_wall_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        world_reconstruction_fit=world_reconstruction_fit,
                    ),
                    "rollout.platform_bounce_analytic": lambda: self._physionpp_bouncy_platform_rollout_payload(
                        scene=scene,
                        question=question,
                        object_plan=object_plan,
                        world_model_dir=source_dir,
                        world_reconstruction_fit=world_reconstruction_fit,
                    ),
                }
                payload = builders[route]()
                if payload.get("backend") != contract["artifact_backend"]:
                    raise RoutePolicyValidationError(
                        "trajectory backend conflicts with the resolved question "
                        f"rollout route: {payload.get('backend')!r} != "
                        f"{contract['artifact_backend']!r}"
                    )
                payload["question_rollout_backend_route"] = deepcopy(
                    active_route_record
                )
                if active_tool_planning_record is not None:
                    _record_clevrer_tool_planning_route(
                        payload,
                        active_tool_planning_record,
                        require_planning_metadata=True,
                    )
                if active_visual_pose_record is not None:
                    self._apply_future_visual_pose_preservation(
                        payload=payload,
                        world_reconstruction_fit=world_reconstruction_fit,
                        route_record=active_visual_pose_record,
                    )
            payload["world_model_artifact_dir"] = str(source_dir)
            self.artifacts.write(artifact_path, payload)
            self._maybe_render_plane_projection_debug(
                artifact_path=artifact_path,
                payload=payload,
                world_model_dir=source_dir,
            )
            _log_tool(self.tool_name, f"artifact={artifact_path} {_trajectory_summary(payload)}")
            return ToolResult(self.tool_name, "ok", str(artifact_path), payload=payload)

        payload["status"] = "missing_world_reconstruction_fit"
        payload["backend"] = "query_conditioned_analytic_rollout"
        payload["error_message"] = "SWR fit artifact is required for analytic query rollout"
        if active_route_record is not None:
            payload["question_rollout_backend_route"] = deepcopy(
                active_route_record
            )
        if active_tool_planning_record is not None:
            _record_clevrer_tool_planning_route(
                payload,
                active_tool_planning_record,
                require_planning_metadata=False,
            )
        self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"missing_world_reconstruction_fit artifact={artifact_path}")
        return ToolResult(self.tool_name, "tool_error", str(artifact_path), message=payload["error_message"], payload=payload)

    def _physionpp_friction_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        world_model_dir: Path,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        manifest_path = (
            world_model_dir
            / "simulatable-world-reconstruction"
            / "fit"
            / "world_reconstruction_fit_manifest.json"
        )
        sam3_tracks_path = (
            world_model_dir
            / "object-segmentation-and-event-detection"
            / "sam3_video_tracks"
            / "sam3_video_tracks.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing Physion++ SWR manifest: {manifest_path}")
        if not sam3_tracks_path.exists():
            raise FileNotFoundError(f"missing Physion++ SAM3 tracks artifact: {sam3_tracks_path}")
        helper = _load_physionpp_friction_future_rollout()
        future = helper.run_physionpp_friction_future_rollout(
            fit=world_reconstruction_fit,
            manifest=self._read_optional(manifest_path),
            object_plan=object_plan.to_dict(),
            sam3_tracks=self._read_optional(sam3_tracks_path),
            max_future_frames=int(os.getenv("PHYSMIND_PHYSIONPP_FUTURE_MAX_FRAMES", "300")),
            stationary_speed_m_per_s=float(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_SPEED_M_PER_S", "0.01")),
            stationary_window_frames=int(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_WINDOW_FRAMES", "10")),
        )
        return {
            "tool": self.tool_name,
            "status": future.get("status", "ok"),
            "backend": "physionpp_friction_future_analytic_rollout",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [],
            "fit_error": world_reconstruction_fit.get("fit_error"),
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "physion_pp_future_rollout": future,
            "predictions": [
                {
                    "what_if": -1,
                    "source": "physionpp_friction_future_analytic_rollout",
                    "patient_contact": future.get("patient_contact"),
                    "horizon": future.get("horizon"),
                }
            ],
            "notes": [
                "Physion++ friction-platform OCP uses a future analytic rollout from the observed boundary.",
                "The rollout continues for up to 300 frames unless the dynamic object becomes stationary first.",
            ],
        }

    def _physionpp_friction_collision_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        helper = _load_physionpp_friction_collision_future_rollout()
        future = helper.run_physionpp_friction_collision_future_rollout(
            fit=world_reconstruction_fit,
            max_future_frames=int(os.getenv("PHYSMIND_PHYSIONPP_FUTURE_MAX_FRAMES", "300")),
            stationary_speed_m_per_s=float(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_SPEED_M_PER_S", "0.01")),
            stationary_window_frames=int(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_WINDOW_FRAMES", "10")),
        )
        return {
            "tool": self.tool_name,
            "status": future.get("status", "ok"),
            "backend": "rollout.collision_friction_analytic",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [],
            "fit_error": world_reconstruction_fit.get("fit_error"),
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "physion_pp_future_rollout": future,
            "predictions": [
                {
                    "what_if": -1,
                    "source": "rollout.collision_friction_analytic",
                    "patient_contact": future.get("patient_contact"),
                    "horizon": future.get("horizon"),
                }
            ],
            "notes": [
                "Physion++ friction-collision OCP continues both fitted 3D sphere trajectories from the observed boundary.",
                "Contact is detected continuously in 3D and the rollout stops at first agent-patient contact.",
            ],
        }

    def _physionpp_mass_collision_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        helper = _load_physionpp_mass_collision_future_rollout()
        future = helper.run_physionpp_mass_collision_future_rollout(
            fit=world_reconstruction_fit,
            max_future_frames=int(os.getenv("PHYSMIND_PHYSIONPP_FUTURE_MAX_FRAMES", "300")),
            stationary_speed_m_per_s=float(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_SPEED_M_PER_S", "0.01")),
            stationary_window_frames=int(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_WINDOW_FRAMES", "10")),
        )
        return {
            "tool": self.tool_name,
            "status": future.get("status", "ok"),
            "backend": "rollout.collision_mass_analytic",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [],
            "fit_error": world_reconstruction_fit.get("fit_error"),
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "physion_pp_future_rollout": future,
            "predictions": [
                {
                    "what_if": -1,
                    "source": "rollout.collision_mass_analytic",
                    "patient_contact": future.get("patient_contact"),
                    "horizon": future.get("horizon"),
                }
            ],
            "notes": [
                "Physion++ mass-collision OCP continues the fitted second-segment agent and patient states.",
                "Contact is detected continuously in 3D and the rollout stops at first agent-patient contact.",
            ],
        }

    def _physionpp_bouncy_wall_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        helper = _load_physionpp_bouncy_wall_future_rollout()
        future = helper.run_physionpp_bouncy_wall_future_rollout(
            fit=world_reconstruction_fit,
            max_future_frames=int(os.getenv("PHYSMIND_PHYSIONPP_FUTURE_MAX_FRAMES", "300")),
            stationary_speed_m_per_s=float(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_SPEED_M_PER_S", "0.01")),
            stationary_window_frames=int(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_WINDOW_FRAMES", "10")),
        )
        return {
            "tool": self.tool_name,
            "status": future.get("status", "ok"),
            "backend": "rollout.wall_bounce_analytic",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [],
            "fit_error": world_reconstruction_fit.get("fit_error"),
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "physion_pp_future_rollout": future,
            "predictions": [
                {
                    "what_if": -1,
                    "source": "rollout.wall_bounce_analytic",
                    "patient_contact": future.get("patient_contact"),
                    "horizon": future.get("horizon"),
                }
            ],
            "notes": [
                "Physion++ bouncy-wall OCP continues the second test segment from its fitted terminal state.",
                "The first segment contributes shared physical properties but is not connected to the test trajectory.",
            ],
        }

    def _physionpp_bouncy_platform_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        world_model_dir: Path,
        world_reconstruction_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        manifest_path = (
            world_model_dir
            / "simulatable-world-reconstruction"
            / "fit"
            / "world_reconstruction_fit_manifest.json"
        )
        sam3_tracks_path = (
            world_model_dir
            / "object-segmentation-and-event-detection"
            / "sam3_video_tracks"
            / "sam3_video_tracks.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing Physion++ SWR manifest: {manifest_path}")
        if not sam3_tracks_path.exists():
            raise FileNotFoundError(f"missing Physion++ SAM3 tracks artifact: {sam3_tracks_path}")
        helper = _load_physionpp_bouncy_platform_future_rollout()
        future = helper.run_physionpp_bouncy_platform_future_rollout(
            fit=world_reconstruction_fit,
            manifest=self._read_optional(manifest_path),
            object_plan=object_plan.to_dict(),
            sam3_tracks=self._read_optional(sam3_tracks_path),
            max_future_frames=int(os.getenv("PHYSMIND_PHYSIONPP_FUTURE_MAX_FRAMES", "300")),
            stationary_speed_m_per_s=float(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_SPEED_M_PER_S", "0.01")),
            stationary_window_frames=int(os.getenv("PHYSMIND_PHYSIONPP_STATIONARY_WINDOW_FRAMES", "10")),
        )
        return {
            "tool": self.tool_name,
            "status": future.get("status", "ok"),
            "backend": "rollout.platform_bounce_analytic",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "input_artifacts": [],
            "fit_error": world_reconstruction_fit.get("fit_error"),
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "physion_pp_future_rollout": future,
            "predictions": [
                {
                    "what_if": -1,
                    "source": "rollout.platform_bounce_analytic",
                    "patient_contact": future.get("patient_contact"),
                    "horizon": future.get("horizon"),
                }
            ],
            "notes": [
                "Physion++ bouncy-platform OCP continues the fitted analytic trajectory from the observed boundary.",
                "The rollout continues for up to 300 frames unless the dynamic object becomes stationary first.",
            ],
        }

    def _maybe_render_plane_projection_debug(
        self,
        *,
        artifact_path: Path,
        payload: Dict[str, Any],
        world_model_dir: Path,
    ) -> None:
        if not self.artifacts.debug_artifacts:
            return
        fit_path = world_model_dir / "simulatable-world-reconstruction" / "fit" / "world_reconstruction_fit.json"
        if not fit_path.exists():
            return
        try:
            result = render_qcpr_plane_projection_debug(
                trajectory_path=artifact_path,
                world_reconstruction_fit_path=fit_path,
            )
        except Exception as exc:
            result = {
                "status": "tool_error",
                "error_message": str(exc),
                "trajectory_path": str(artifact_path),
                "world_reconstruction_fit_path": str(fit_path),
            }
        debug_artifacts = dict(payload.get("debug_artifacts") or {})
        debug_artifacts["plane_projection_debug"] = result
        payload["debug_artifacts"] = debug_artifacts
        self.artifacts.write(artifact_path, payload)

    def _read_optional(self, path: Path) -> Dict[str, Any]:
        return self.artifacts.read_optional(path) or {}

    def _extract_phrase_attributes(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        references: list[str],
    ) -> dict[str, Any]:
        choices = [
            {
                "choice_letter": chr(ord("A") + index),
                "choice_id": choice.choice_id,
                "choice": choice.choice,
            }
            for index, choice in enumerate(question.choices)
        ]
        fallback = _reference_attribute_payload(references)
        prompt = (
            "Extract descriptive visual attributes from object phrases in a CLEVRER question.\n"
            "Do not map anything to object ids. Only parse each provided phrase into required visual attributes.\n"
            "Use only colors, shapes, and materials that are explicit in the phrase. Ignore the generic word object.\n"
            "Return compact JSON inside <answer> </answer> with this schema:\n"
            '{"phrases":[{"phrase":"metal cube","required_attributes":["metal","cube"]}]}\n\n'
            f"Question: {question.question}\n"
            f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
            f"Object phrases to parse exactly: {json.dumps(references, ensure_ascii=False)}\n"
            f"Allowed attributes: {json.dumps(fallback['attribute_vocabulary'], ensure_ascii=False)}\n"
        )
        try:
            text = _answer_text_only(
                config=self.config,
                prompt=prompt,
                request_context={
                    "scene_index": scene.scene_index,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                },
            )
            parsed = _parse_json_object(text)
            payload = _normalize_phrase_attribute_payload(parsed, references)
            payload["raw_response"] = text
            payload["method"] = "vlm_phrase_attribute_extraction"
            return payload
        except Exception as exc:
            payload = _normalize_phrase_attribute_payload({}, references)
            payload["method"] = "deterministic_phrase_attribute_extraction_fallback"
            payload["error_message"] = str(exc)
            return payload

    def _assign_scene_visual_attributes(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        tracks: list[dict[str, Any]],
        image_paths: list[Path],
        image_labels: list[str],
        phrase_attributes: dict[str, Any],
    ) -> dict[str, Any]:
        vocabulary = phrase_attributes.get("attribute_vocabulary") or {}
        allowed_attributes = _known_attribute_set(phrase_attributes)
        prompt = (
            "Classify each red-outlined object relative to the other objects in this scene.\n"
            "Do not use any question text; only use the images and the allowed attribute vocabulary.\n"
            "For each object_id image label, list every allowed attribute that visibly applies to the outlined object.\n"
            "Return compact JSON inside <answer> </answer> with this schema:\n"
            '{"objects":[{"object_id":"obj_3","attributes":["metal","cube"],"reason":"brief visual reason"}]}\n\n'
            f"Allowed attribute vocabulary: {json.dumps(vocabulary, ensure_ascii=False)}\n"
            f"Object ids to classify: {json.dumps([item.get('object_id') for item in tracks], ensure_ascii=False)}\n"
        )
        response = answer_with_image_files(
            config=self.config,
            prompt=prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            request_context={
                "scene_index": scene.scene_index,
                "question_id": question.question_id,
                "question_type": question.question_type,
            },
        )
        parsed = _parse_json_object(response.text)
        payload = _normalize_object_attribute_payload(
            parsed,
            tracks=tracks,
            allowed_attributes=allowed_attributes,
            raw_response=response.text,
        )
        payload["method"] = "vlm_scene_visual_attribute_assignment"
        return payload

    def _fallback_visual_object_reference_map(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        reference: str,
        tracks: list[dict[str, Any]],
        image_paths: list[Path],
        image_labels: list[str],
        phrase_attributes: dict[str, Any],
        object_attributes: dict[str, Any],
    ) -> dict[str, Any]:
        choices = [
            {
                "choice_letter": chr(ord("A") + index),
                "choice_id": choice.choice_id,
                "choice": choice.choice,
            }
            for index, choice in enumerate(question.choices)
        ]
        prompt = (
            "Map exactly one unresolved object phrase to the tracked objects shown in images.\n"
            "Each representative image is one full-frame image for one object_id. The tracked object is outlined with a red contour line, with no text labels drawn on the image.\n"
            "Each image label is the exact object_id you may use, such as obj_3.\n"
            "Use only the images, question text, choices, and object_id labels. Do not use or assume any other unresolved phrase.\n"
            "The selected object must satisfy every required attribute for this phrase.\n"
            "Return only compact JSON inside <answer> </answer> with this schema:\n"
            '{"references":[{"reference":"yellow cube","object_id":"obj_3","reason":"brief visual reason"}]}\n\n'
            f"Question: {question.question}\n"
            f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
            f"Unresolved object phrase: {json.dumps(reference, ensure_ascii=False)}\n"
            f"Required attributes: {json.dumps(_required_attributes_for_phrase(phrase_attributes, reference), ensure_ascii=False)}\n"
        )
        response = answer_with_image_files(
            config=self.config,
            prompt=prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            request_context={
                "scene_index": scene.scene_index,
                "question_id": question.question_id,
                "question_type": question.question_type,
            },
        )
        parsed = _parse_json_object(response.text)
        payload = _normalize_visual_reference_map(
            payload=parsed,
            references=[reference],
            tracks=tracks,
            raw_response=response.text,
            required_attributes_by_reference={reference: set(_required_attributes_for_phrase(phrase_attributes, reference))},
            attributes_by_object=_attributes_by_object(object_attributes),
        )
        payload["method"] = "vlm_unresolved_visual_reference_fallback"
        return payload

    def _build_visual_object_reference_map(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        world_model_dir: Path,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        references = _extract_object_references(question)
        if not references:
            return {
                "status": "ok",
                "method": "no_question_object_references",
                "references": [],
                "fallback_object_reference_map": fallback,
            }
        if self.config is None:
            return {
                **fallback,
                "status": "fallback",
                "fallback_reason": "missing_model_config_for_visual_object_reference_map",
                "fallback_object_reference_map": fallback,
            }
        tracks, image_paths, image_labels = _track_representative_inputs(world_model_dir)
        if not tracks or not image_paths:
            return {
                **fallback,
                "status": "fallback",
                "fallback_reason": "missing_sam3_video_track_representative_images",
                "fallback_object_reference_map": fallback,
            }

        try:
            phrase_attributes = self._extract_phrase_attributes(
                scene=scene,
                question=question,
                references=references,
            )
            object_attributes = self._assign_scene_visual_attributes(
                scene=scene,
                question=question,
                tracks=tracks,
                image_paths=image_paths,
                image_labels=image_labels,
                phrase_attributes=phrase_attributes,
            )
            entries, unresolved, ambiguous = _match_phrases_from_attributes(
                phrase_attributes=phrase_attributes,
                object_attributes=object_attributes,
                tracks=tracks,
            )
            fallback_payloads = []
            fallback_references = sorted(set(unresolved + ambiguous))
            mapped = {str(item.get("reference")) for item in entries}
            for fallback_reference in fallback_references:
                fallback_payload = self._fallback_visual_object_reference_map(
                    scene=scene,
                    question=question,
                    reference=fallback_reference,
                    tracks=tracks,
                    image_paths=image_paths,
                    image_labels=image_labels,
                    phrase_attributes=phrase_attributes,
                    object_attributes=object_attributes,
                )
                fallback_payloads.append(fallback_payload)
                for item in fallback_payload.get("references", []):
                    if isinstance(item, dict) and str(item.get("reference")) not in mapped:
                        entries.append(item)
                        mapped.add(str(item.get("reference")))
            mapped_references = {str(item.get("reference")) for item in entries}
            visual_map = {
                "status": "ok" if mapped_references == set(references) else "partial",
                "method": "vlm_attribute_catalog_then_local_match",
                "references": entries,
                "phrase_attribute_extraction": phrase_attributes,
                "scene_visual_attribute_assignment": object_attributes,
                "unresolved_references": unresolved,
                "ambiguous_references": ambiguous,
            }
            if fallback_payloads:
                visual_map["unresolved_visual_fallback"] = {
                    "method": "vlm_unresolved_visual_reference_fallback_per_phrase",
                    "payloads": fallback_payloads,
                    "references": [
                        item
                        for payload in fallback_payloads
                        for item in payload.get("references", [])
                        if isinstance(item, dict)
                    ],
                }
            if mapped_references != set(references):
                visual_map["missing_references"] = sorted(set(references) - mapped_references)
            visual_map["fallback_object_reference_map"] = fallback
            return visual_map
        except Exception as exc:
            return {
                **fallback,
                "status": "fallback",
                "fallback_reason": "visual_object_reference_map_failed",
                "error_message": str(exc),
                "fallback_object_reference_map": fallback,
            }

    def _analytic_rollout_payload(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        question_dir: Path,
        world_reconstruction_fit: Dict[str, Any],
        clevrer_tool_planning_route: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        active_tool_planning_route = _require_clevrer_tool_planning_route(
            clevrer_tool_planning_route,
            scene=scene,
            question=question,
        )
        active_tool_plan_repair_route = (
            _require_clevrer_tool_plan_repair_route(
                _resolve_clevrer_tool_plan_repair_route(
                    scene,
                    question,
                ),
                scene=scene,
                question=question,
            )
        )
        object_plan_payload = object_plan.to_dict()
        object_index_by_id = {item.object_id: index for index, item in enumerate(object_plan.target_objects)}
        objects = [
            {
                "id": object_index_by_id[item.object_id],
                "object_id": item.object_id,
                "color": _normalize_color(item.appearance.get("color"), item.description),
                "raw_color": str(item.appearance.get("color") or "").strip().lower(),
                "material": _normalize_material(item.appearance.get("material"), item.description),
                "shape": _normalize_shape(item.geometry_type, item.description),
                "description": item.description,
            }
            for item in object_plan.target_objects
        ]
        forecast_extension_frames = self._forecast_extension_frames(world_reconstruction_fit)
        base_replay = self._replay_world_reconstruction_fit(
            world_reconstruction_fit=world_reconstruction_fit,
            object_index_by_id=object_index_by_id,
            objects=objects,
            edit_manifest=None,
            forecast_extension_frames=forecast_extension_frames,
        )
        if base_replay.get("status") == "ok":
            trajectory = base_replay["trajectory"]
            collisions = base_replay["collisions"]
        else:
            trajectories = self._rollout_trajectories(world_reconstruction_fit)
            trajectory = self._analytic_rollout_frames(
                trajectories=trajectories,
                object_index_by_id=object_index_by_id,
                objects=objects,
                world_reconstruction_fit=world_reconstruction_fit,
            )
            collisions = []
        in_out_events = self._in_out_events(trajectory)
        deterministic_object_reference_map = _build_object_reference_map(question, objects)
        object_reference_map = self._build_visual_object_reference_map(
            scene=scene,
            question=question,
            world_model_dir=question_dir.parent / "world-modeling",
            fallback=deterministic_object_reference_map,
        )
        source_last_frame = base_replay.get("source_last_frame")
        tool_call_request, tool_planning = self._plan_tool_calls(
            scene=scene,
            question=question,
            objects=objects,
            trajectory=trajectory,
            collisions=collisions,
            in_out_events=in_out_events,
            world_reconstruction_fit=world_reconstruction_fit,
            object_reference_map=object_reference_map,
            source_last_frame=source_last_frame,
            clevrer_tool_plan_repair_route=(
                active_tool_plan_repair_route
            ),
        )
        tool_call_results, execution_metadata = self._execute_tool_call_rounds(
            initial_request=tool_call_request,
            initial_metadata=tool_planning,
            scene=scene,
            question=question,
            objects=objects,
            trajectory=trajectory,
            collisions=collisions,
            in_out_events=in_out_events,
            world_reconstruction_fit=world_reconstruction_fit,
            object_index_by_id=object_index_by_id,
            object_reference_map=object_reference_map,
            source_last_frame=source_last_frame,
            clevrer_tool_planning_route=active_tool_planning_route,
            clevrer_tool_plan_repair_route=(
                active_tool_plan_repair_route
            ),
            base_rollout_metadata={
                "forecast_extension_frames": forecast_extension_frames,
                "replay_status": base_replay.get("status"),
                "replay_error_message": base_replay.get("error_message"),
                "source_last_frame": base_replay.get("source_last_frame"),
                "rollout_last_frame": base_replay.get("rollout_last_frame"),
            },
        )
        edited_rollouts = [
            result.result
            for result in tool_call_results
            if result.tool in {"remove_object", "simulate_edit"}
            and result.status == "ok"
            and result.result.get("removed_object_id") is not None
        ]
        predictions = [
            {
                "what_if": -1,
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
                "source": "swr_physics_rollout",
            }
        ]
        for rollout in edited_rollouts:
            predictions.append(
                {
                    "what_if": rollout.get("removed_object_index"),
                    "removed_object_id": rollout.get("removed_object_id"),
                    "trajectory": rollout.get("trajectory", []),
                    "collisions": rollout.get("collisions", []),
                    "in_out_events": rollout.get("in_out_events", []),
                    "source": rollout.get("execution_backend") or "tool_call_remove_object",
                }
            )
        return {
            "tool": self.tool_name,
            "status": "ok",
            "backend": "rollout.tool_physics",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "choices": [
                {
                    "choice_id": choice.choice_id,
                    "choice_letter": chr(ord("A") + index),
                    "choice": choice.choice,
                }
                for index, choice in enumerate(question.choices)
            ],
            "objects": objects,
            "object_reference_map": object_reference_map,
            "deterministic_object_reference_map": deterministic_object_reference_map,
            "base_rollout": {
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
                "forecast_extension_frames": forecast_extension_frames,
                "replay_status": base_replay.get("status"),
                "replay_error_message": base_replay.get("error_message"),
                "source_last_frame": base_replay.get("source_last_frame"),
                "rollout_last_frame": base_replay.get("rollout_last_frame"),
            },
            "tool_call_prompt_instruction": build_prompt_only_tool_call_instruction(default_qcpr_tool_registry()),
            "tool_planning": execution_metadata,
            **(
                {
                    "clevrer_tool_planning_route": deepcopy(
                        active_tool_planning_route
                    )
                }
                if active_tool_planning_route is not None
                else {}
            ),
            "tool_call_request": execution_metadata.get("combined_request", tool_call_request.to_dict()),
            "tool_call_results": [result.to_dict() for result in tool_call_results],
            "edited_rollouts": edited_rollouts,
            "predictions": predictions,
            "object_index_by_object_id": object_index_by_id,
            "source_world_reconstruction_fit_backend": world_reconstruction_fit.get("backend"),
            "source_world_reconstruction_fit_error": world_reconstruction_fit.get("fit_error"),
            "object_plan": object_plan_payload,
            "notes": [
                "Tool-call execution records explicit edit manifests.",
                "remove_object replays the source world reconstruction backend when a replay API is available.",
                "QCPR rollouts extend beyond the observed video by the configured default forecast horizon unless a tool call explicitly limits the inspected frame range.",
            ],
        }

    def _plan_tool_calls(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
        object_reference_map: dict[str, Any],
        source_last_frame: Any = None,
        clevrer_tool_plan_repair_route: Dict[str, Any] | None = None,
    ) -> tuple[ToolCallRequest, dict[str, Any]]:
        fallback = self._fallback_tool_call_request(question=question, objects=objects, trajectory=trajectory)
        planner = build_tool_call_planner(self.config)
        if planner is None:
            adjusted_fallback, adjustment = self._apply_reference_map_to_tool_calls(
                question=question,
                request=fallback,
                object_reference_map=object_reference_map,
            )
            adjusted_fallback, future_adjustment = self._apply_predictive_future_start_frame(
                question=question,
                request=adjusted_fallback,
                source_last_frame=source_last_frame,
            )
            metadata = {
                "status": "fallback",
                "reason": "missing_openai_compatible_model_config",
                "planner": "deterministic",
            }
            if adjustment:
                metadata["object_reference_map_adjustment"] = adjustment
            if future_adjustment:
                metadata["predictive_future_start_frame_adjustment"] = future_adjustment
            return adjusted_fallback, metadata
        request_context = {
            "benchmark": (
                "clevrer"
                if _clevrer_tool_planning_policy_context(scene, question)
                is not None
                else None
            ),
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
        }
        trajectory_summary = self._trajectory_brief(trajectory)
        if source_last_frame is not None:
            trajectory_summary["source_last_frame"] = source_last_frame
            try:
                trajectory_summary["future_start_frame"] = max(0, int(source_last_frame) - 5)
            except (TypeError, ValueError):
                pass
        request, metadata = planner.plan(
            question=question,
            objects=objects,
            trajectory_summary=trajectory_summary,
            collisions=collisions,
            in_out_events=in_out_events,
            source_fit_error=world_reconstruction_fit.get("fit_error"),
            object_reference_map=object_reference_map,
            fallback=fallback,
            request_context=request_context,
            tool_plan_repair_route=clevrer_tool_plan_repair_route,
        )
        adjusted_request, adjustment = self._apply_reference_map_to_tool_calls(
            question=question,
            request=request,
            object_reference_map=object_reference_map,
        )
        if adjustment:
            metadata = dict(metadata)
            metadata["object_reference_map_adjustment"] = adjustment
        adjusted_request, future_adjustment = self._apply_predictive_future_start_frame(
            question=question,
            request=adjusted_request,
            source_last_frame=source_last_frame,
        )
        if future_adjustment:
            metadata = dict(metadata)
            metadata["predictive_future_start_frame_adjustment"] = future_adjustment
        return adjusted_request, metadata

    def _apply_predictive_future_start_frame(
        self,
        *,
        question: ClevrerQuestion,
        request: ToolCallRequest,
        source_last_frame: Any,
    ) -> tuple[ToolCallRequest, dict[str, Any] | None]:
        if str(question.question_type).lower() != "predictive" or source_last_frame is None:
            return request, None
        try:
            future_start_frame = max(0, int(source_last_frame) - 5)
        except (TypeError, ValueError):
            return request, None
        changed = False
        tool_calls: list[ToolCall] = []
        for call in request.tool_calls:
            if call.tool not in {"check_contact", "get_event_time"}:
                tool_calls.append(call)
                continue
            rollout_id = call.arguments.get("rollout_id")
            if rollout_id not in (None, "base"):
                tool_calls.append(call)
                continue
            if call.arguments.get("start_frame") != future_start_frame:
                changed = True
                tool_calls.append(
                    ToolCall(
                        tool=call.tool,
                        arguments={**call.arguments, "rollout_id": "base", "start_frame": future_start_frame},
                        call_id=call.call_id,
                    )
                )
            else:
                tool_calls.append(call)
        if not changed:
            return request, None
        return (
            ToolCallRequest(
                tool_calls=tool_calls,
                reasoning_summary=(
                    f"{request.reasoning_summary} Predictive contact/event-time calls were constrained to "
                    f"the boundary-aware future window starting at {future_start_frame}."
                ),
                expected_evidence=request.expected_evidence,
            ),
            {
                "source_last_frame": int(source_last_frame),
                "future_start_frame": future_start_frame,
                "reason": "predictive questions use a five-frame boundary buffer before the observed cutoff",
            },
        )

    def _apply_reference_map_to_tool_calls(
        self,
        *,
        question: ClevrerQuestion,
        request: ToolCallRequest,
        object_reference_map: dict[str, Any],
    ) -> tuple[ToolCallRequest, dict[str, Any] | None]:
        removed_reference = self._removed_reference(question)
        if not removed_reference:
            return request, None
        mapped = self._mapped_object_for_reference(removed_reference, object_reference_map)
        if not mapped:
            return request, None
        changed = False
        tool_calls = []
        remove_added = False
        for call in request.tool_calls:
            if call.tool == "remove_object":
                if not remove_added:
                    if call.arguments.get("object_id") != mapped:
                        changed = True
                    tool_calls.append(
                        ToolCall(
                            tool=call.tool,
                            arguments={**call.arguments, "object_id": mapped},
                            call_id=call.call_id,
                        )
                    )
                    remove_added = True
                else:
                    changed = True
                continue
            if call.tool == "simulate_edit":
                edit = call.arguments.get("edit") if isinstance(call.arguments.get("edit"), dict) else {}
                if str(edit.get("type") or "") == "remove_object":
                    if not remove_added:
                        if edit.get("object_id") != mapped:
                            changed = True
                        tool_calls.append(
                            ToolCall(
                                tool=call.tool,
                                arguments={**call.arguments, "edit": {**edit, "object_id": mapped}},
                                call_id=call.call_id,
                            )
                        )
                        remove_added = True
                    else:
                        changed = True
                    continue
            tool_calls.append(call)
        if not remove_added:
            changed = True
            tool_calls.insert(
                0,
                ToolCall(
                    tool="simulate_edit",
                    arguments={"edit": {"type": "remove_object", "object_id": mapped}},
                    call_id=None,
                ),
            )
        if not changed:
            return request, None
        return (
            ToolCallRequest(
                tool_calls=tool_calls,
                reasoning_summary=(
                    f"{request.reasoning_summary} Reference map forced '{removed_reference}' to {mapped}."
                ),
                expected_evidence=request.expected_evidence,
            ),
            {
                "removed_reference": removed_reference,
                "forced_object_id": mapped,
                "reason": "counterfactual removed object matched by object_reference_map",
            },
        )

    def _removed_reference(self, question: ClevrerQuestion) -> str | None:
        text = str(question.question or "").lower()
        for pattern in [
            r"\bif\s+(?:the\s+)?(.+?)\s+is\s+removed\b",
            r"\bwithout\s+(?:the\s+)?(.+?)(?:,|\?|$)",
        ]:
            match = re.search(pattern, text)
            if not match:
                continue
            value = re.split(r"\bwhich\b|\bwhat\b|\bwill\b", match.group(1))[0]
            reference = " ".join(_tokenize_reference(value))
            if reference:
                return reference
        return None

    def _mapped_object_for_reference(self, reference: str, object_reference_map: dict[str, Any]) -> str | None:
        entries = object_reference_map.get("references") if isinstance(object_reference_map, dict) else []
        if not isinstance(entries, list):
            return None
        for item in entries:
            if not isinstance(item, dict):
                continue
            if str(item.get("reference") or "") == reference and item.get("object_id"):
                return str(item["object_id"])
        return None

    def _trajectory_brief(self, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        frames = [int(frame.get("frame_index", 0)) for frame in trajectory]
        object_ids = sorted({
            str(item.get("object_id"))
            for frame in trajectory
            for item in frame.get("objects", [])
            if item.get("object_id") is not None
        })
        return {
            "frame_count": len(trajectory),
            "first_frame": min(frames) if frames else None,
            "last_frame": max(frames) if frames else None,
            "object_ids": object_ids,
        }

    def _fallback_tool_call_request(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
    ) -> ToolCallRequest:
        calls: list[ToolCall] = []
        first_frame = int(trajectory[0]["frame_index"]) if trajectory else 0
        if str(question.question_type).lower() == "counterfactual":
            for item in objects:
                calls.append(
                    ToolCall(
                        tool="remove_object",
                        arguments={"object_id": item["object_id"]},
                    )
                )
        else:
            calls.append(ToolCall(tool="inspect_world_state", arguments={"frame_index": first_frame}))
        return ToolCallRequest(
            tool_calls=calls,
            reasoning_summary="deterministic fallback until VLM tool planning is enabled",
            expected_evidence=["base and edited rollout summaries"],
        )

    def _execute_tool_call_rounds(
        self,
        *,
        initial_request: ToolCallRequest,
        initial_metadata: dict[str, Any],
        scene: ClevrerScene,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        object_reference_map: dict[str, Any],
        source_last_frame: Any,
        clevrer_tool_planning_route: Dict[str, Any] | None,
        clevrer_tool_plan_repair_route: Dict[str, Any] | None = None,
        base_rollout_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[ToolCallResult], dict[str, Any]]:
        if clevrer_tool_planning_route is not None and (
            clevrer_tool_planning_route.get("decision_id")
            != CLEVRER_TOOL_PLANNING_DECISION_ID
            or clevrer_tool_planning_route.get("route")
            != CLEVRER_TOOL_PLANNING_ROUTE
        ):
            raise RoutePolicyValidationError(
                "dynamic tool-call rounds require the resolved ROL-002 route"
            )
        rollouts = self._initial_tool_rollouts(
            trajectory=trajectory,
            collisions=collisions,
            in_out_events=in_out_events,
            base_rollout_metadata=base_rollout_metadata,
        )
        first_results = self._execute_tool_calls(
            request=initial_request,
            trajectory=trajectory,
            collisions=collisions,
            in_out_events=in_out_events,
            world_reconstruction_fit=world_reconstruction_fit,
            object_index_by_id=object_index_by_id,
            objects=objects,
            rollouts=rollouts,
        )
        combined_results = list(first_results)
        rounds = [
            {
                "round_index": 1,
                "planning": initial_metadata,
                "request": initial_request.to_dict(),
                "available_rollout_ids": sorted(rollouts),
                "result_count": len(first_results),
            }
        ]
        planner = build_tool_call_planner(self.config)
        if planner is not None and self._should_plan_followup_tool_calls(first_results):
            fallback = ToolCallRequest(
                tool_calls=[],
                reasoning_summary="No follow-up tool calls were generated.",
                expected_evidence=[],
            )
            trajectory_summary = self._trajectory_brief(trajectory)
            if source_last_frame is not None:
                trajectory_summary["source_last_frame"] = source_last_frame
                try:
                    trajectory_summary["future_start_frame"] = max(0, int(source_last_frame) - 5)
                except (TypeError, ValueError):
                    pass
            followup_request, followup_metadata = planner.plan(
                question=question,
                objects=objects,
                trajectory_summary=trajectory_summary,
                collisions=collisions,
                in_out_events=in_out_events,
                source_fit_error=world_reconstruction_fit.get("fit_error"),
                object_reference_map=object_reference_map,
                available_rollout_ids=sorted(rollouts),
                previous_tool_results=self._compact_tool_results_for_planning(first_results),
                fallback=fallback,
                request_context={
                    "benchmark": (
                        "clevrer"
                        if _clevrer_tool_planning_policy_context(
                            scene,
                            question,
                        )
                        is not None
                        else None
                    ),
                    "scene_index": scene.scene_index,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "tool_round": 2,
                },
                tool_plan_repair_route=(
                    clevrer_tool_plan_repair_route
                ),
            )
            followup_request = self._filter_followup_tool_calls(followup_request)
            followup_request, future_adjustment = self._apply_predictive_future_start_frame(
                question=question,
                request=followup_request,
                source_last_frame=source_last_frame,
            )
            if future_adjustment:
                followup_metadata = dict(followup_metadata)
                followup_metadata["predictive_future_start_frame_adjustment"] = future_adjustment
            followup_results = self._execute_tool_calls(
                request=followup_request,
                trajectory=trajectory,
                collisions=collisions,
                in_out_events=in_out_events,
                world_reconstruction_fit=world_reconstruction_fit,
                object_index_by_id=object_index_by_id,
                objects=objects,
                rollouts=rollouts,
            )
            combined_results.extend(followup_results)
            rounds.append(
                {
                    "round_index": 2,
                    "planning": followup_metadata,
                    "request": followup_request.to_dict(),
                    "available_rollout_ids": sorted(rollouts),
                    "result_count": len(followup_results),
                }
            )
        combined_calls: list[ToolCall] = list(initial_request.tool_calls)
        if len(rounds) > 1:
            second_request = rounds[1].get("request") if isinstance(rounds[1].get("request"), dict) else {}
            for item in second_request.get("tool_calls", []) if isinstance(second_request.get("tool_calls"), list) else []:
                if isinstance(item, dict) and isinstance(item.get("tool"), str) and isinstance(item.get("arguments"), dict):
                    combined_calls.append(
                        ToolCall(
                            tool=str(item["tool"]),
                            arguments=dict(item["arguments"]),
                            call_id=item.get("call_id") if isinstance(item.get("call_id"), str) else None,
                        )
                    )
        combined_request = ToolCallRequest(
            tool_calls=combined_calls,
            reasoning_summary="Executed dynamic two-round tool-call plan.",
            expected_evidence=[],
        ).to_dict()
        return combined_results, {
            "status": "ok",
            "planner": "planning.two_round_tools",
            **(
                {
                    "clevrer_tool_planning_route": deepcopy(
                        clevrer_tool_planning_route
                    )
                }
                if clevrer_tool_planning_route is not None
                else {}
            ),
            "rounds": rounds,
            "combined_request": combined_request,
        }

    def _initial_tool_rollouts(
        self,
        *,
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        base_rollout_metadata: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        return {
            "base": {
                "rollout_id": "base",
                "status": "ok",
                "execution_backend": "swr_physics_rollout",
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
                **(base_rollout_metadata or {}),
            }
        }

    def _should_plan_followup_tool_calls(self, results: list[ToolCallResult]) -> bool:
        return any(
            result.tool in {"remove_object", "simulate_edit"}
            and result.status == "ok"
            and result.result.get("rollout_id")
            for result in results
        )

    def _compact_tool_results_for_planning(self, results: list[ToolCallResult]) -> list[dict[str, Any]]:
        compact = []
        for result in results:
            payload = result.result if isinstance(result.result, dict) else {}
            compact.append(
                {
                    "tool": result.tool,
                    "status": result.status,
                    "call_id": result.call_id,
                    "rollout_id": payload.get("rollout_id"),
                    "rollout_aliases": payload.get("rollout_aliases"),
                    "edit_manifest": payload.get("edit_manifest"),
                    "removed_object_id": payload.get("removed_object_id"),
                    "error_message": result.error_message,
                }
            )
        return compact

    def _filter_followup_tool_calls(self, request: ToolCallRequest) -> ToolCallRequest:
        tool_calls = [
            call
            for call in request.tool_calls
            if call.tool not in {"simulate_edit", "remove_object", "get_reliability"}
        ]
        if len(tool_calls) == len(request.tool_calls):
            return request
        return ToolCallRequest(
            tool_calls=tool_calls,
            reasoning_summary=(
                f"{request.reasoning_summary} Edit tools were ignored in follow-up planning; "
                "follow-up rounds may only inspect existing rollouts."
            ),
            expected_evidence=request.expected_evidence,
        )

    def _execute_tool_calls(
        self,
        *,
        request: ToolCallRequest,
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
        base_rollout_metadata: dict[str, Any] | None = None,
        rollouts: dict[str, dict[str, Any]] | None = None,
    ) -> list[ToolCallResult]:
        results = []
        if rollouts is None:
            rollouts = self._initial_tool_rollouts(
                trajectory=trajectory,
                collisions=collisions,
                in_out_events=in_out_events,
                base_rollout_metadata=base_rollout_metadata,
            )
        edit_rollout_count = 0

        def register_rollout_aliases(rollout: dict[str, Any], call: ToolCall) -> None:
            nonlocal edit_rollout_count
            aliases = []
            rollout_id = rollout.get("rollout_id")
            if rollout_id:
                aliases.append(str(rollout_id))
            if call.call_id:
                aliases.append(str(call.call_id))
            edit_rollout_count += 1
            sim_alias = f"sim_{edit_rollout_count}"
            aliases.append(sim_alias)
            aliases.append(f"unnamed_rollout_{edit_rollout_count}")
            if not aliases:
                return
            if not rollout_id:
                rollout["rollout_id"] = aliases[0]
            rollout["rollout_aliases"] = sorted(set([*rollout.get("rollout_aliases", []), *aliases]))
            for alias in rollout["rollout_aliases"]:
                rollouts[str(alias)] = rollout

        def unavailable_rollout_result(call: ToolCall, rollout: dict[str, Any]) -> ToolCallResult | None:
            if rollout.get("status") == "invalid_rollout_id":
                return ToolCallResult(
                    tool=call.tool,
                    call_id=call.call_id,
                    status="error",
                    result=rollout,
                    error_message=str(rollout.get("error_message") or "invalid rollout_id"),
                )
            if rollout.get("status") != "missing_rollout":
                if rollout.get("status") in {None, "ok"}:
                    return None
                result = {
                    "status": "dependency_failed",
                    "rollout_id": rollout.get("rollout_id"),
                    "failed_rollout_status": rollout.get("status"),
                    "error_message": rollout.get("error_message"),
                    "execution_backend": rollout.get("execution_backend"),
                }
                return ToolCallResult(
                    tool=call.tool,
                    call_id=call.call_id,
                    status="error",
                    result=result,
                    error_message=str(rollout.get("error_message") or rollout.get("status") or "rollout dependency failed"),
                )
            return ToolCallResult(
                tool=call.tool,
                call_id=call.call_id,
                status="error",
                result=rollout,
                error_message=str(rollout.get("error_message") or "unknown rollout_id"),
            )

        for call in request.tool_calls:
            if call.tool == "inspect_world_state":
                rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("rollout_id"))
                unavailable = unavailable_rollout_result(call, rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="ok",
                        result=self._inspect_world_state(
                            trajectory=rollout.get("trajectory", []),
                            collisions=rollout.get("collisions", []),
                            frame_index=int(call.arguments["frame_index"]),
                        ),
                    )
                )
            elif call.tool == "inspect_object_trajectory":
                rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("rollout_id"))
                unavailable = unavailable_rollout_result(call, rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="ok",
                        result=self._inspect_object_trajectory(
                            trajectory=rollout.get("trajectory", []),
                            rollout_id=str(rollout.get("rollout_id", "base")),
                            object_id=str(call.arguments["object_id"]),
                            start_frame=call.arguments.get("start_frame"),
                            end_frame=call.arguments.get("end_frame"),
                        ),
                    )
                )
            elif call.tool == "simulate_edit":
                edit = call.arguments.get("edit") if isinstance(call.arguments.get("edit"), dict) else {}
                rollout = self._simulate_edit_rollout(
                    trajectory=trajectory,
                    collisions=collisions,
                    in_out_events=in_out_events,
                    world_reconstruction_fit=world_reconstruction_fit,
                    object_index_by_id=object_index_by_id,
                    objects=objects,
                    edit=edit,
                )
                register_rollout_aliases(rollout, call)
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="ok" if rollout.get("status") == "ok" else "error",
                        result=rollout,
                        error_message=None if rollout.get("status") == "ok" else str(rollout.get("error_message") or rollout.get("status")),
                    )
                )
            elif call.tool == "check_contact":
                rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("rollout_id"))
                unavailable = unavailable_rollout_result(call, rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                result = self._check_contact(
                    rollout=rollout,
                    base_rollout=rollouts.get("base"),
                    object_a=str(call.arguments["object_a"]),
                    object_b=str(call.arguments["object_b"]),
                    start_frame=call.arguments.get("start_frame"),
                    end_frame=call.arguments.get("end_frame"),
                    world_reconstruction_fit=world_reconstruction_fit,
                )
                results.append(ToolCallResult(tool=call.tool, call_id=call.call_id, status="ok", result=result))
            elif call.tool == "get_event_time":
                rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("rollout_id"))
                unavailable = unavailable_rollout_result(call, rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                result = self._get_event_time(
                    rollout=rollout,
                    object_a=str(call.arguments["object_a"]),
                    object_b=str(call.arguments["object_b"]),
                    start_frame=call.arguments.get("start_frame"),
                    end_frame=call.arguments.get("end_frame"),
                    world_reconstruction_fit=world_reconstruction_fit,
                )
                results.append(ToolCallResult(tool=call.tool, call_id=call.call_id, status="ok", result=result))
            elif call.tool == "compare_rollouts":
                base_rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("base_rollout_id") or "base")
                edited_rollout = self._rollout_for_tool_call(rollouts, call.arguments.get("edited_rollout_id"))
                unavailable = unavailable_rollout_result(call, base_rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                unavailable = unavailable_rollout_result(call, edited_rollout)
                if unavailable is not None:
                    results.append(unavailable)
                    continue
                result = self._compare_rollouts(
                    base_rollout=base_rollout,
                    edited_rollout=edited_rollout,
                    object_ids=call.arguments.get("object_ids"),
                    world_reconstruction_fit=world_reconstruction_fit,
                )
                results.append(ToolCallResult(tool=call.tool, call_id=call.call_id, status="ok", result=result))
            elif call.tool == "get_reliability":
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="ignored",
                        result={"status": "ignored", "reason": "get_reliability is not part of the active QCPR tool set"},
                    )
                )
            elif call.tool == "remove_object":
                rollout = self._remove_object_rollout(
                    trajectory=trajectory,
                    collisions=collisions,
                    in_out_events=in_out_events,
                    world_reconstruction_fit=world_reconstruction_fit,
                    object_index_by_id=object_index_by_id,
                    objects=objects,
                    object_id=str(call.arguments["object_id"]),
                    start_frame=call.arguments.get("start_frame"),
                )
                register_rollout_aliases(rollout, call)
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="ok" if rollout.get("status") == "ok" else "error",
                        result=rollout,
                        error_message=None if rollout.get("status") == "ok" else str(rollout.get("error_message") or rollout.get("status")),
                    )
                )
            else:
                results.append(
                    ToolCallResult(
                        tool=call.tool,
                        call_id=call.call_id,
                        status="error",
                        error_message=f"unsupported tool: {call.tool}",
                    )
                )
        return results

    def _rollout_for_tool_call(self, rollouts: dict[str, dict[str, Any]], rollout_id: Any = None) -> dict[str, Any]:
        if rollout_id is not None:
            key = str(rollout_id)
            if key in rollouts:
                return rollouts[key]
            return {
                "rollout_id": key,
                "status": "invalid_rollout_id",
                "error_message": f"unknown rollout_id: {key}",
                "requested_rollout_id": key,
                "valid_rollout_ids": sorted(rollouts),
                "instruction": 'Use "base" or copy one exact rollout_id returned by a previous tool result.',
            }
        edited = [item for key, item in rollouts.items() if key != "base"]
        if rollout_id is None and edited:
            return edited[-1]
        return rollouts["base"]

    def _inspect_world_state(
        self,
        *,
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        frame_index: int,
    ) -> dict[str, Any]:
        selected = None
        for frame in trajectory:
            if int(frame.get("frame_index", -1)) == int(frame_index):
                selected = frame
                break
        if selected is None:
            return {
                "frame_index": int(frame_index),
                "status": "missing_frame",
                "objects": [],
                "collisions": [],
            }
        return {
            "frame_index": int(frame_index),
            "status": "ok",
            "objects": selected.get("objects", []),
            "collisions": [event for event in collisions if int(event.get("frame", -1)) == int(frame_index)],
        }

    def _inspect_object_trajectory(
        self,
        *,
        trajectory: list[dict[str, Any]],
        rollout_id: str,
        object_id: str,
        start_frame: Any = None,
        end_frame: Any = None,
    ) -> dict[str, Any]:
        start = int(start_frame) if start_frame is not None else None
        end = int(end_frame) if end_frame is not None else None
        records = []
        for frame in trajectory:
            frame_index = int(frame.get("frame_index", 0))
            if start is not None and frame_index < start:
                continue
            if end is not None and frame_index > end:
                continue
            for item in frame.get("objects", []):
                if str(item.get("object_id")) == object_id:
                    records.append({"frame_index": frame_index, "object": item})
        return {
            "rollout_id": rollout_id,
            "object_id": object_id,
            "start_frame": start,
            "end_frame": end,
            "record_count": len(records),
            "first_record": records[0] if records else None,
            "last_record": records[-1] if records else None,
            "records": records,
        }

    def _forecast_extension_frames(self, world_reconstruction_fit: Dict[str, Any]) -> int:
        explicit = os.getenv("PHYSMIND_QCPR_FORECAST_EXTENSION_FRAMES")
        if explicit not in {None, ""}:
            return max(int(explicit), 0)
        ratio = float(os.getenv("PHYSMIND_QCPR_FORECAST_EXTENSION_RATIO", "1.0"))
        frames = []
        for records in (world_reconstruction_fit.get("target_trajectories") or {}).values():
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict) and record.get("frame_index") is not None:
                    frames.append(int(record["frame_index"]))
        if not frames:
            return 0
        observed_count = max(frames) - min(frames) + 1
        return max(int(math.ceil(max(observed_count, 1) * max(ratio, 0.0))), 0)

    def _replay_world_reconstruction_fit(
        self,
        *,
        world_reconstruction_fit: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
        edit_manifest: dict[str, Any] | None,
        forecast_extension_frames: int,
    ) -> dict[str, Any]:
        source_backend = str(world_reconstruction_fit.get("backend") or "")
        try:
            if source_backend == "swr_backend.impulse_analytic":
                rollout_api = _load_impulse_analytic_rollout_api()
                replay = rollout_api.rollout_from_world_reconstruction_fit(
                    world_reconstruction_fit,
                    edit_manifest=edit_manifest,
                    forecast_extension_frames=forecast_extension_frames,
                )
            else:
                raise RuntimeError(f"unsupported analytic SWR backend for replay: {source_backend}")
        except Exception as exc:
            return {
                "status": "physics_rerollout_error",
                "error_message": str(exc),
                "execution_backend": f"{source_backend or 'unknown'}_rerollout",
                "edit_manifest": edit_manifest or {"edit_type": "none"},
            }
        removed_object_id = None
        if isinstance(edit_manifest, dict) and edit_manifest.get("edit_type") == "remove_object":
            removed_object_id = str(edit_manifest.get("object_id"))
        replay_object_index_by_id = {
            item_object_id: index
            for item_object_id, index in object_index_by_id.items()
            if removed_object_id is None or str(item_object_id) != removed_object_id
        }
        replay_objects = [
            item
            for item in objects
            if removed_object_id is None or str(item.get("object_id")) != removed_object_id
        ]
        replay_fit = {
            **world_reconstruction_fit,
            "physics_rollout": {
                **(world_reconstruction_fit.get("physics_rollout") or {}),
                "simulated_trajectories": replay.get("simulated_trajectories") or {},
            },
        }
        replay_trajectory = self._analytic_rollout_frames(
            trajectories=replay.get("simulated_trajectories") or {},
            object_index_by_id=replay_object_index_by_id,
            objects=replay_objects,
            world_reconstruction_fit=replay_fit,
        )
        replay_collisions = self._analytic_rerollout_collisions(
            collisions=replay.get("collisions") or [],
            object_index_by_id=replay_object_index_by_id,
            objects=replay_objects,
        )
        return {
            "status": "ok",
            "execution_backend": replay.get("backend") or "analytic_rerollout",
            "edit_manifest": edit_manifest or {"edit_type": "none"},
            "trajectory": replay_trajectory,
            "collisions": replay_collisions,
            "in_out_events": self._in_out_events(replay_trajectory),
            "physics_replay_fit_error": replay.get("fit_error"),
            "forecast_extension_frames": replay.get("forecast_extension_frames"),
            "source_last_frame": replay.get("source_last_frame"),
            "rollout_last_frame": replay.get("rollout_last_frame"),
        }

    def _simulate_edit_rollout(
        self,
        *,
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
        edit: dict[str, Any],
    ) -> dict[str, Any]:
        edit_type = str(edit.get("type") or "")
        if edit_type != "remove_object":
            return {
                "status": "unsupported_edit",
                "error_message": f"unsupported edit type: {edit_type}",
                "edit": edit,
            }
        return self._remove_object_rollout(
            trajectory=trajectory,
            collisions=collisions,
            in_out_events=in_out_events,
            world_reconstruction_fit=world_reconstruction_fit,
            object_index_by_id=object_index_by_id,
            objects=objects,
            object_id=str(edit.get("object_id")),
            start_frame=edit.get("start_frame"),
        )

    def _remove_object_rollout(
        self,
        *,
        trajectory: list[dict[str, Any]],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
        object_id: str,
        start_frame: Any = None,
    ) -> dict[str, Any]:
        object_index = self._object_index_from_id(trajectory, object_id)
        if object_index is None:
            return {
                "status": "missing_object",
                "removed_object_id": object_id,
                "removed_object_index": None,
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
            }
        start = int(start_frame) if start_frame is not None else self._first_active_frame(trajectory, object_index)
        first_active = self._first_active_frame(trajectory, object_index)
        if start != first_active:
            return {
                "status": "unsupported_delayed_removal",
                "error_message": "Analytic replay currently supports object removal from the object's first active frame only.",
                "removed_object_id": object_id,
                "removed_object_index": int(object_index),
                "requested_start_frame": int(start),
                "first_active_frame": int(first_active),
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
            }
        edit_manifest = {
            "edit_type": "remove_object",
            "object_id": object_id,
            "object_index": int(object_index),
            "start_frame": int(start),
        }
        replay = self._replay_world_reconstruction_fit(
            world_reconstruction_fit=world_reconstruction_fit,
            object_index_by_id=object_index_by_id,
            objects=objects,
            edit_manifest=edit_manifest,
            forecast_extension_frames=self._forecast_extension_frames(world_reconstruction_fit),
        )
        if replay.get("status") != "ok":
            return {
                "status": replay.get("status", "physics_rerollout_error"),
                "error_message": replay.get("error_message"),
                "execution_backend": replay.get("execution_backend") or "physics_rerollout",
                "edit_manifest": edit_manifest,
                "removed_object_id": object_id,
                "removed_object_index": int(object_index),
                "start_frame": int(start),
                "trajectory": trajectory,
                "collisions": collisions,
                "in_out_events": in_out_events,
            }
        return {
            "rollout_id": f"remove_{object_id}",
            "status": "ok",
            "execution_backend": replay.get("execution_backend") or "analytic_rerollout",
            "edit_manifest": edit_manifest,
            "removed_object_id": object_id,
            "removed_object_index": int(object_index),
            "start_frame": int(start),
            "trajectory": replay.get("trajectory", []),
            "collisions": replay.get("collisions", []),
            "in_out_events": replay.get("in_out_events", []),
            "physics_replay_fit_error": replay.get("physics_replay_fit_error"),
            "forecast_extension_frames": replay.get("forecast_extension_frames"),
            "source_last_frame": replay.get("source_last_frame"),
            "rollout_last_frame": replay.get("rollout_last_frame"),
            "notes": [
                "This rollout replays the optimized world reconstruction dynamics after removing the selected object.",
                "Collision events are emitted by the selected SWR replay backend.",
                "The rollout includes the configured QCPR forecast horizon beyond the observed video.",
            ],
        }

    def _collision_frames_for_pair(
        self,
        *,
        rollout: dict[str, Any],
        object_a: str,
        object_b: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        world_reconstruction_fit: Dict[str, Any] | None = None,
    ) -> list[int]:
        frames = []
        for event in rollout.get("collisions", []) or []:
            object_ids = {
                str(item.get("object_id"))
                for item in event.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            frame = int(event.get("frame", event.get("frame_index", 0)))
            if start_frame is not None and frame < start_frame:
                continue
            if end_frame is not None and frame > end_frame:
                continue
            if {object_a, object_b}.issubset(object_ids):
                frames.append(frame)
        if not frames and world_reconstruction_fit is not None:
            posthoc = self._posthoc_conservative_contact_scan(
                rollout=rollout,
                object_a=object_a,
                object_b=object_b,
                start_frame=start_frame,
                end_frame=end_frame,
                world_reconstruction_fit=world_reconstruction_fit,
            )
            frames = [int(frame) for frame in posthoc.get("frames", [])]
        return frames

    def _conservative_contact_radius(
        self,
        *,
        object_id: str,
        world_reconstruction_fit: Dict[str, Any],
    ) -> tuple[float, str] | None:
        physics_rollout = world_reconstruction_fit.get("physics_rollout")
        if not isinstance(physics_rollout, dict):
            return None
        geometry_by_object = {
            str(item.get("object_id")): str(item.get("geometry_type") or "").lower()
            for item in world_reconstruction_fit.get("trajectory_physics_initialization", {}).get("objects", [])
            if isinstance(item, dict) and item.get("object_id") is not None
        }
        geometry_type = geometry_by_object.get(object_id, "")
        half_extents = physics_rollout.get("shape_half_extents_2d_by_object", {}).get(object_id)
        if geometry_type in {"box", "cube"} and isinstance(half_extents, list) and len(half_extents) >= 2:
            try:
                half_x = float(half_extents[0])
                half_y = float(half_extents[1])
            except (TypeError, ValueError):
                return None
            if math.isfinite(half_x) and math.isfinite(half_y) and half_x > 0.0 and half_y > 0.0:
                return math.sqrt(half_x * half_x + half_y * half_y), "box_half_diagonal"
        dimensions = physics_rollout.get("proxy_dimensions_by_object", {}).get(object_id)
        if isinstance(dimensions, list) and len(dimensions) >= 2:
            try:
                radius = 0.5 * max(float(dimensions[0]), float(dimensions[1]))
            except (TypeError, ValueError):
                return None
            if math.isfinite(radius) and radius > 0.0:
                return radius, "native_circle"
        radius = physics_rollout.get("shape_radius_2d_by_object", {}).get(object_id)
        try:
            radius_value = float(radius)
        except (TypeError, ValueError):
            return None
        if math.isfinite(radius_value) and radius_value > 0.0:
            return radius_value, "fallback_shape_radius"
        return None

    def _posthoc_conservative_contact_scan(
        self,
        *,
        rollout: dict[str, Any],
        object_a: str,
        object_b: str,
        start_frame: int | None,
        end_frame: int | None,
        world_reconstruction_fit: Dict[str, Any],
    ) -> dict[str, Any]:
        radius_a = self._conservative_contact_radius(
            object_id=object_a,
            world_reconstruction_fit=world_reconstruction_fit,
        )
        radius_b = self._conservative_contact_radius(
            object_id=object_b,
            world_reconstruction_fit=world_reconstruction_fit,
        )
        if radius_a is None or radius_b is None:
            return {"confirmed": False, "frames": [], "events": [], "reason": "missing_contact_radius"}
        radius_a_value, radius_a_policy = radius_a
        radius_b_value, radius_b_policy = radius_b
        radius_sum = radius_a_value + radius_b_value
        best: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []
        for frame in rollout.get("trajectory", []) or []:
            if not isinstance(frame, dict):
                continue
            frame_index = int(frame.get("frame_index", 0))
            if start_frame is not None and frame_index < start_frame:
                continue
            if end_frame is not None and frame_index > end_frame:
                continue
            by_id = {
                str(item.get("object_id")): item
                for item in frame.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            if object_a not in by_id or object_b not in by_id:
                continue
            pos_a = _finite_vector(by_id[object_a].get("position"))
            pos_b = _finite_vector(by_id[object_b].get("position"))
            if pos_a is None or pos_b is None:
                continue
            plane_distance = math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
            gap = plane_distance - radius_sum
            record = {
                "frame": frame_index,
                "center_distance_m": float(plane_distance),
                "contact_residual_m": float(gap),
                "object_a_position": pos_a,
                "object_b_position": pos_b,
            }
            if best is None or gap < float(best["contact_residual_m"]):
                best = record
            if gap > 0.0:
                continue
            object_items = [dict(by_id[object_a]), dict(by_id[object_b])]
            object_indices = [
                item.get("id")
                for item in object_items
                if item.get("id") is not None
            ]
            events.append(
                {
                    "type": "collision",
                    "frame": frame_index,
                    "object": object_indices,
                    "objects": object_items,
                    "source_event": {
                        "event_type": "pair_collision_candidate",
                        "event_source": "posthoc_conservative_footprint_scan",
                        "contact_model": "conservative_footprint_circle",
                        "object_ids": [object_a, object_b],
                        "frame": frame_index,
                        "contact_residual_m": float(gap),
                        "center_distance_m": float(plane_distance),
                        "radius_sum_m": float(radius_sum),
                        "radius_by_object": {
                            object_a: float(radius_a_value),
                            object_b: float(radius_b_value),
                        },
                        "radius_policy_by_object": {
                            object_a: radius_a_policy,
                            object_b: radius_b_policy,
                        },
                    },
                }
            )
        return {
            "confirmed": bool(events),
            "frames": [int(event["frame"]) for event in events],
            "events": events,
            "first_frame": int(events[0]["frame"]) if events else None,
            "best": best,
            "radius_sum_m": float(radius_sum),
            "radius_by_object": {
                object_a: float(radius_a_value),
                object_b: float(radius_b_value),
            },
            "radius_policy_by_object": {
                object_a: radius_a_policy,
                object_b: radius_b_policy,
            },
            "notes": [
                "This scan is post-hoc evidence only; it does not change the rollout trajectory.",
                "Box/cube contacts use half the support-plane footprint diagonal as a conservative radius.",
            ],
        }

    def _check_contact(
        self,
        *,
        rollout: dict[str, Any],
        base_rollout: dict[str, Any] | None = None,
        object_a: str,
        object_b: str,
        start_frame: Any = None,
        end_frame: Any = None,
        world_reconstruction_fit: Dict[str, Any],
    ) -> dict[str, Any]:
        start = int(start_frame) if start_frame is not None else None
        end = int(end_frame) if end_frame is not None else None
        confirmed = []
        confirmation_source = "rollout_collision_events"
        for event in rollout.get("collisions", []) or []:
            object_ids = {
                str(item.get("object_id"))
                for item in event.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            frame = int(event.get("frame", event.get("frame_index", 0)))
            if start is not None and frame < start:
                continue
            if end is not None and frame > end:
                continue
            if {object_a, object_b}.issubset(object_ids):
                confirmed.append(event)
        posthoc_contact = self._posthoc_conservative_contact_scan(
            rollout=rollout,
            object_a=object_a,
            object_b=object_b,
            start_frame=start,
            end_frame=end,
            world_reconstruction_fit=world_reconstruction_fit,
        )
        if not confirmed and posthoc_contact.get("confirmed"):
            confirmed = list(posthoc_contact.get("events", []))
            confirmation_source = "posthoc_conservative_footprint_scan"
        nearest = self._nearest_object_distance(
            trajectory=rollout.get("trajectory", []) or [],
            object_a=object_a,
            object_b=object_b,
            start_frame=start,
            end_frame=end,
        )
        reliability = self._rollout_reliability(
            rollout=rollout,
            world_reconstruction_fit=world_reconstruction_fit,
            object_ids=[object_a, object_b],
        )
        source_last_frame = rollout.get("source_last_frame")
        rollout_last_frame = rollout.get("rollout_last_frame")
        confirmed_frames = [int(event.get("frame", event.get("frame_index", 0))) for event in confirmed]
        after_source = None
        before_or_at_source = None
        if source_last_frame is not None:
            source_last = int(source_last_frame)
            after_source = any(frame > source_last for frame in confirmed_frames)
            before_or_at_source = any(frame <= source_last for frame in confirmed_frames)
        same_event_tolerance = int(os.getenv("PHYSMIND_QCPR_SAME_EVENT_FRAME_TOLERANCE", "5"))
        rollout_id = str(rollout.get("rollout_id", "base"))
        base_first_frame = None
        edited_first_frame = confirmed_frames[0] if confirmed_frames else None
        event_frame_delta = None
        same_event_with_base = None
        event_identity_note = None
        if base_rollout is not None and rollout_id != "base":
            base_frames = self._collision_frames_for_pair(
                rollout=base_rollout,
                object_a=object_a,
                object_b=object_b,
                world_reconstruction_fit=world_reconstruction_fit,
            )
            base_first_frame = base_frames[0] if base_frames else None
            if base_first_frame is not None and edited_first_frame is not None:
                event_frame_delta = int(edited_first_frame) - int(base_first_frame)
                same_event_with_base = abs(event_frame_delta) <= same_event_tolerance
                if same_event_with_base:
                    event_identity_note = (
                        f"Edited rollout collision matches the original event timing "
                        f"({base_first_frame}->{edited_first_frame}, tolerance {same_event_tolerance} frames)."
                    )
                else:
                    event_identity_note = (
                        f"Edited rollout still has this collision, but its timing differs substantially "
                        f"from the original event ({base_first_frame}->{edited_first_frame}, tolerance "
                        f"{same_event_tolerance} frames), so it is not the same event."
                    )
            elif base_first_frame is not None and edited_first_frame is None:
                same_event_with_base = False
                event_identity_note = (
                    f"The original event occurs at frame {base_first_frame}, but the edited rollout has no matching collision."
                )
            elif base_first_frame is None and edited_first_frame is not None:
                same_event_with_base = False
                event_identity_note = (
                    f"The edited rollout has this collision at frame {edited_first_frame}, but no original matching event is found."
                )
        status = "confirmed" if confirmed else "no_confirmed_contact"
        return {
            "rollout_id": rollout_id,
            "object_a": object_a,
            "object_b": object_b,
            "forecast_extension_frames": rollout.get("forecast_extension_frames"),
            "source_last_frame": source_last_frame,
            "rollout_last_frame": rollout_last_frame,
            "frame_range": {"start_frame": start, "end_frame": end},
            "status": status,
            "confirmed": bool(confirmed),
            "confirmation_source": confirmation_source if confirmed else "none",
            "confirmed_frames": confirmed_frames,
            "confirmed_after_source_last_frame": after_source,
            "confirmed_before_or_at_source_last_frame": before_or_at_source,
            "confirmed_collisions": confirmed,
            "posthoc_conservative_contact": posthoc_contact,
            "base_first_frame": base_first_frame,
            "edited_first_frame": edited_first_frame,
            "event_frame_delta": event_frame_delta,
            "same_event_frame_tolerance": same_event_tolerance,
            "same_event_with_base": same_event_with_base,
            "event_identity_note": event_identity_note,
            "nearest_distance": nearest,
            "reliability": reliability,
            "notes": [
                "confirmed contact first uses rollout collision events; if absent, post-hoc conservative footprint scan may confirm contact",
                "nearest_distance is auxiliary evidence and does not by itself prove collision",
            ],
        }

    def _get_event_time(
        self,
        *,
        rollout: dict[str, Any],
        object_a: str,
        object_b: str,
        start_frame: Any = None,
        end_frame: Any = None,
        world_reconstruction_fit: Dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = int(start_frame) if start_frame is not None else None
        end = int(end_frame) if end_frame is not None else None
        matched = []
        for event in rollout.get("collisions", []) or []:
            object_ids = {
                str(item.get("object_id"))
                for item in event.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            frame = int(event.get("frame", event.get("frame_index", 0)))
            if start is not None and frame < start:
                continue
            if end is not None and frame > end:
                continue
            if {object_a, object_b}.issubset(object_ids):
                matched.append(event)
        confirmation_source = "rollout_collision_events"
        posthoc_contact = None
        if not matched and world_reconstruction_fit is not None:
            posthoc_contact = self._posthoc_conservative_contact_scan(
                rollout=rollout,
                object_a=object_a,
                object_b=object_b,
                start_frame=start,
                end_frame=end,
                world_reconstruction_fit=world_reconstruction_fit,
            )
            if posthoc_contact.get("confirmed"):
                matched = list(posthoc_contact.get("events", []))
                confirmation_source = "posthoc_conservative_footprint_scan"
        frames = [int(event.get("frame", event.get("frame_index", 0))) for event in matched]
        first_event = matched[0] if matched else None
        return {
            "rollout_id": rollout.get("rollout_id", "base"),
            "object_a": object_a,
            "object_b": object_b,
            "frame_range": {"start_frame": start, "end_frame": end},
            "status": "found" if matched else "not_found",
            "found": bool(matched),
            "confirmation_source": confirmation_source if matched else "none",
            "event_count": len(matched),
            "frames": frames,
            "first_frame": frames[0] if frames else None,
            "first_event": first_event,
            "posthoc_conservative_contact": posthoc_contact,
            "source_last_frame": rollout.get("source_last_frame"),
            "rollout_last_frame": rollout.get("rollout_last_frame"),
            "notes": [
                "Use first_frame to compare event order in causal questions.",
            ],
        }

    def _nearest_object_distance(
        self,
        *,
        trajectory: list[dict[str, Any]],
        object_a: str,
        object_b: str,
        start_frame: int | None,
        end_frame: int | None,
    ) -> dict[str, Any] | None:
        best = None
        for frame in trajectory:
            frame_index = int(frame.get("frame_index", 0))
            if start_frame is not None and frame_index < start_frame:
                continue
            if end_frame is not None and frame_index > end_frame:
                continue
            by_id = {
                str(item.get("object_id")): item
                for item in frame.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            if object_a not in by_id or object_b not in by_id:
                continue
            pos_a = _finite_vector(by_id[object_a].get("position"))
            pos_b = _finite_vector(by_id[object_b].get("position"))
            if pos_a is None or pos_b is None:
                continue
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))
            if best is None or distance < best["distance"]:
                best = {
                    "distance": float(distance),
                    "frame": frame_index,
                    "object_a_position": pos_a,
                    "object_b_position": pos_b,
                }
        return best

    def _compare_rollouts(
        self,
        *,
        base_rollout: dict[str, Any],
        edited_rollout: dict[str, Any],
        object_ids: Any = None,
        world_reconstruction_fit: Dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = {str(item) for item in object_ids} if isinstance(object_ids, list) else None
        base_collisions = self._collision_pairs_for_rollout(
            rollout=base_rollout,
            selected=selected,
            world_reconstruction_fit=world_reconstruction_fit,
        )
        edited_collisions = self._collision_pairs_for_rollout(
            rollout=edited_rollout,
            selected=selected,
            world_reconstruction_fit=world_reconstruction_fit,
        )
        return {
            "base_rollout_id": base_rollout.get("rollout_id", "base"),
            "edited_rollout_id": edited_rollout.get("rollout_id", "base"),
            "object_ids": sorted(selected) if selected else None,
            "base_collision_pairs": sorted(base_collisions),
            "edited_collision_pairs": sorted(edited_collisions),
            "added_collision_pairs": sorted(edited_collisions - base_collisions),
            "removed_collision_pairs": sorted(base_collisions - edited_collisions),
            "trajectory_delta": self._trajectory_delta_summary(base_rollout, edited_rollout, selected),
        }

    def _collision_pairs_for_rollout(
        self,
        *,
        rollout: dict[str, Any],
        selected: set[str] | None,
        world_reconstruction_fit: Dict[str, Any] | None,
    ) -> set[str]:
        output = self._collision_pairs(rollout.get("collisions", []) or [], selected)
        if world_reconstruction_fit is None:
            return output
        object_ids = sorted(self._trajectory_positions_by_object(rollout.get("trajectory", []) or []))
        for index, object_a in enumerate(object_ids[:-1]):
            for object_b in object_ids[index + 1 :]:
                if selected is not None and not selected.intersection({object_a, object_b}):
                    continue
                key = "|".join([object_a, object_b])
                if key in output:
                    continue
                posthoc = self._posthoc_conservative_contact_scan(
                    rollout=rollout,
                    object_a=object_a,
                    object_b=object_b,
                    start_frame=None,
                    end_frame=None,
                    world_reconstruction_fit=world_reconstruction_fit,
                )
                if posthoc.get("confirmed"):
                    output.add(key)
        return output

    def _collision_pairs(self, collisions: list[dict[str, Any]], selected: set[str] | None) -> set[str]:
        output = set()
        for event in collisions:
            ids = sorted(
                str(item.get("object_id"))
                for item in event.get("objects", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            )
            if len(ids) < 2:
                continue
            for index, object_a in enumerate(ids[:-1]):
                for object_b in ids[index + 1 :]:
                    pair = {object_a, object_b}
                    if selected is not None and not selected.intersection(pair):
                        continue
                    output.add("|".join([object_a, object_b]))
        return output

    def _trajectory_delta_summary(
        self,
        base_rollout: dict[str, Any],
        edited_rollout: dict[str, Any],
        selected: set[str] | None,
    ) -> list[dict[str, Any]]:
        base = self._trajectory_positions_by_object(base_rollout.get("trajectory", []) or [])
        edited = self._trajectory_positions_by_object(edited_rollout.get("trajectory", []) or [])
        object_ids = sorted((set(base) | set(edited)) if selected is None else selected)
        output = []
        for object_id in object_ids:
            common = sorted(set(base.get(object_id, {})) & set(edited.get(object_id, {})))
            if not common:
                continue
            distances = []
            for frame in common:
                pos_a = base[object_id][frame]
                pos_b = edited[object_id][frame]
                distances.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b))))
            output.append(
                {
                    "object_id": object_id,
                    "sample_count": len(distances),
                    "mean_position_delta": float(sum(distances) / max(len(distances), 1)),
                    "max_position_delta": float(max(distances) if distances else 0.0),
                }
            )
        return output

    def _trajectory_positions_by_object(self, trajectory: list[dict[str, Any]]) -> dict[str, dict[int, list[float]]]:
        output: dict[str, dict[int, list[float]]] = {}
        for frame in trajectory:
            frame_index = int(frame.get("frame_index", 0))
            for item in frame.get("objects", []):
                object_id = str(item.get("object_id"))
                position = _finite_vector(item.get("position"))
                if position is None:
                    continue
                output.setdefault(object_id, {})[frame_index] = position
        return output

    def _rollout_reliability(
        self,
        *,
        rollout: dict[str, Any],
        world_reconstruction_fit: Dict[str, Any],
        object_ids: Any = None,
    ) -> dict[str, Any]:
        selected = {str(item) for item in object_ids} if isinstance(object_ids, list) else None
        fit_error = world_reconstruction_fit.get("fit_error") if isinstance(world_reconstruction_fit.get("fit_error"), dict) else {}
        per_object = fit_error.get("per_object_translation_rmse") if isinstance(fit_error.get("per_object_translation_rmse"), dict) else {}
        selected_errors = {
            object_id: payload
            for object_id, payload in per_object.items()
            if selected is None or object_id in selected
        }
        replay_fit_error = rollout.get("physics_replay_fit_error") if isinstance(rollout.get("physics_replay_fit_error"), dict) else None
        return {
            "rollout_id": rollout.get("rollout_id", "base"),
            "source_fit_error": {
                "overall_translation_rmse": fit_error.get("overall_translation_rmse"),
                "overall_reprojection_rmse_px": fit_error.get("overall_reprojection_rmse_px"),
                "per_object_translation_rmse": selected_errors,
            },
            "replay_fit_error": replay_fit_error,
            "notes": [
                "High source or replay fit error lowers confidence in counterfactual contact claims.",
            ],
        }

    def _analytic_rerollout_collisions(
        self,
        *,
        collisions: list[dict[str, Any]],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        attrs_by_id = {str(item.get("object_id")): item for item in objects}
        output = []
        for event in collisions:
            if not isinstance(event, dict):
                continue
            object_ids = [str(item) for item in event.get("object_ids", []) if str(item) in object_index_by_id]
            if len(object_ids) < 2:
                continue
            indices = [int(object_index_by_id[object_id]) for object_id in object_ids[:2]]
            output.append(
                {
                    "type": "collision",
                    "frame": int(event.get("frame", 0)),
                    "object": indices,
                    "objects": [
                        {
                            "id": int(object_index_by_id[object_id]),
                            "color": attrs_by_id.get(object_id, {}).get("color", "unknown"),
                            "material": attrs_by_id.get(object_id, {}).get("material", "unknown"),
                            "shape": attrs_by_id.get(object_id, {}).get("shape", "unknown"),
                            "object_id": object_id,
                        }
                        for object_id in object_ids[:2]
                    ],
                    "source_event": event,
                }
            )
        return output

    def _object_index_from_id(self, trajectory: list[dict[str, Any]], object_id: str) -> int | None:
        for frame in trajectory:
            for item in frame.get("objects", []):
                if str(item.get("object_id")) == object_id:
                    return int(item["id"])
        return None

    def _first_active_frame(self, trajectory: list[dict[str, Any]], object_index: int) -> int:
        for frame in trajectory:
            if any(int(item.get("id", -1)) == int(object_index) for item in frame.get("objects", [])):
                return int(frame.get("frame_index", 0))
        return 0

    def _rollout_trajectories(self, world_reconstruction_fit: Dict[str, Any]) -> Dict[str, Any]:
        rollout = (world_reconstruction_fit.get("physics_rollout") or {}).get("simulated_trajectories")
        if isinstance(rollout, dict) and rollout:
            return rollout
        target = world_reconstruction_fit.get("target_trajectories")
        if isinstance(target, dict):
            return {key: value for key, value in target.items() if key != "objects"}
        return {}

    def _support_plane(self, world_reconstruction_fit: Dict[str, Any]) -> Dict[str, list[float]] | None:
        plane = world_reconstruction_fit.get("analytic_support_plane") or {}
        origin = _finite_vector(plane.get("origin") or plane.get("center"))
        tangent_1 = _finite_vector(plane.get("tangent_1"))
        tangent_2 = _finite_vector(plane.get("tangent_2"))
        if origin and tangent_1 and tangent_2:
            return {"origin": origin, "tangent_1": tangent_1, "tangent_2": tangent_2}
        return None

    def _plane_xy(self, position: list[float], plane: Dict[str, list[float]] | None) -> tuple[float, float]:
        if plane is None:
            return float(position[0]), float(position[1])
        delta = _sub(position, plane["origin"])
        return _dot(delta, plane["tangent_1"]), _dot(delta, plane["tangent_2"])

    def _analytic_rollout_frames(
        self,
        *,
        trajectories: Dict[str, Any],
        object_index_by_id: Dict[str, int],
        objects: list[dict[str, Any]],
        world_reconstruction_fit: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        plane = self._support_plane(world_reconstruction_fit)
        attrs_by_id = {item["object_id"]: item for item in objects}
        records_by_frame: Dict[int, list[dict[str, Any]]] = {}
        fps = self._video_fps(world_reconstruction_fit)
        previous_by_object: dict[str, tuple[int, float, float]] = {}
        sorted_items = sorted(trajectories.items(), key=lambda item: item[0])
        for object_id, records in sorted_items:
            if object_id not in object_index_by_id or not isinstance(records, list):
                continue
            previous_by_object.pop(object_id, None)
            for record in sorted(records, key=lambda item: int(item.get("frame_index", 0)) if isinstance(item, dict) else 0):
                if not isinstance(record, dict):
                    continue
                position = _finite_vector(record.get("position") or record.get("position_blender") or record.get("position_camera"))
                if position is None:
                    continue
                frame_index = int(record.get("frame_index", 0))
                x, y = self._plane_xy(position, plane)
                prev = previous_by_object.get(object_id)
                vx = vy = 0.0
                if prev is not None:
                    prev_frame, prev_x, prev_y = prev
                    dt = max((frame_index - prev_frame) / max(fps, 1e-12), 1e-12)
                    vx = (x - prev_x) / dt
                    vy = (y - prev_y) / dt
                previous_by_object[object_id] = (frame_index, x, y)
                attrs = attrs_by_id[object_id]
                records_by_frame.setdefault(frame_index, []).append(
                    {
                        "id": int(object_index_by_id[object_id]),
                        "object_id": object_id,
                        "color": attrs["color"],
                        "material": attrs["material"],
                        "shape": attrs["shape"],
                        "x": float(x),
                        "y": float(y),
                        "z": float(position[2]),
                        "vx": float(vx),
                        "vy": float(vy),
                        "position": [float(value) for value in position],
                    }
                )
        return [
            {"frame_index": int(frame_index), "objects": sorted(items, key=lambda item: item["id"])}
            for frame_index, items in sorted(records_by_frame.items())
        ]

    def _video_fps(self, world_reconstruction_fit: Dict[str, Any]) -> float:
        for source in [
            world_reconstruction_fit.get("video_metadata"),
            (world_reconstruction_fit.get("trajectory_physics_initialization") or {}),
        ]:
            if isinstance(source, dict):
                try:
                    value = float(source.get("fps"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    return value
        return 24.0

    def _in_out_events(self, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        previous: set[int] = set()
        for frame in trajectory:
            frame_index = int(frame.get("frame_index", 0))
            current = {int(obj["id"]) for obj in frame.get("objects", [])}
            for object_index in sorted(current - previous):
                output.append({"type": "in", "object": [object_index], "frame": frame_index})
            for object_index in sorted(previous - current):
                output.append({"type": "out", "object": [object_index], "frame": frame_index})
            previous = current
        return output
