from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.world_model.cross_segment_identity import select_same_object_ab
from agent.world_model.module_profiles import (
    ModuleSpec,
    TrackingModuleProfile,
    default_module_profile_policy,
)
from scripts.world_model.run_sam3 import (
    _add_sam3_to_path,
    _mask_for_image,
    _mask_stats,
    _masks_from_response,
    _patch_start_session_for_current_sam3,
)
from agent.world_model.debug_artifacts import write_sam3_video_track_overlay_videos
from utils.config import build_model_config


BENCH_CONCEPT_PROMPTS = {
    "clevrer": [
        {
            "object_id": "clevrer_dynamic_objects",
            "concept_id": "clevrer_dynamic_objects",
            "prompt": "all visible objects",
            "expected_object_ids": [],
            "reason": "CLEVRER dynamic object category",
        },
    ],
}
TRACKING_FAMILY_DECISION_ID = "TRK-001.tracking_family"
CLEVRER_TRACKING_FAMILY_ROUTE = "tracking.sam3_broad_text"
FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_role_box_point"
)
BOUNCY_WALL_TRACKING_FAMILY_ROUTE = "tracking.gdino_two_segment_point_probe"
BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_per_part_seeds"
)
FRICTION_COLLISION_TRACKING_FAMILY_ROUTE = "tracking.sam3_two_segment_text"
MASS_COLLISION_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_two_segment_boxes"
)
PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO = {
    "friction_platform_pp": FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE,
    "bouncy_wall_pp": BOUNCY_WALL_TRACKING_FAMILY_ROUTE,
    "bouncy_platform_pp": BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE,
    "friction_collision_pp": FRICTION_COLLISION_TRACKING_FAMILY_ROUTE,
    "mass_collision_pp": MASS_COLLISION_TRACKING_FAMILY_ROUTE,
}
TRACKING_FAMILY_ROUTES = frozenset(
    {CLEVRER_TRACKING_FAMILY_ROUTE}
    | set(PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO.values())
)
TEMPORAL_PARTITION_DECISION_ID = "INP-003.temporal_partition"
SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE = "temporal.single_video"
SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE = (
    "temporal.cue_two_segment"
)
TEMPORAL_PARTITION_ROUTES = frozenset(
    {
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE,
        SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
    }
)
TEMPORAL_PARTITION_ROUTE_BY_TRACKING_FAMILY = {
    CLEVRER_TRACKING_FAMILY_ROUTE: SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE,
    FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE: (
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE: (
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    BOUNCY_WALL_TRACKING_FAMILY_ROUTE: SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
    FRICTION_COLLISION_TRACKING_FAMILY_ROUTE: (
        SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    MASS_COLLISION_TRACKING_FAMILY_ROUTE: SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
}
CUE_ROLE_BINDING_DECISION_ID = "TRK-002.cue_role_binding"
CUE_ROLE_BINDING_ROUTE = "role_binding.cue_rounds"
STATIC_ROLE_ASSIGNMENT_DECISION_ID = "TRK-003.static_role_assignment"
STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO = {
    "friction_platform_pp": "static_roles.platform_displacement",
    "bouncy_wall_pp": "static_roles.wall_patient_two_segment",
    "bouncy_platform_pp": "static_roles.bounce_platform_displacement",
    "friction_collision_pp": "static_roles.collision_structure",
    "mass_collision_pp": "static_roles.mass_structure",
}
STATIC_ROLE_ASSIGNMENT_ROUTES = frozenset(
    STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO.values()
)
CROSS_SEGMENT_IDENTITY_DECISION_ID = "TRK-004.cross_segment_identity"
CROSS_SEGMENT_IDENTITY_ROUTE = "identity.vlm_ab"
CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO = {
    "bouncy_wall_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
    "friction_collision_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
    "mass_collision_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
}
CROSS_SEGMENT_IDENTITY_ROUTES = frozenset(
    CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO.values()
)
PHYSION_DEFAULT_MAIN_PROMPT = "all visible objects"
PHYSION_DEFAULT_MAIN_CONCEPT_ID = "physion_dynamic_objects"

# Friction-platform tracking detects role-specific seeds, propagates priority roles in
# separate sessions, carves overlaps by role priority, and merges only structure tracks.
# Bouncy-platform tracking uses per-part seeds at a mid-video anchor, propagates in both
# directions, removes contained duplicates, and binds the cue-flash roles.
# Two-segment scenarios exclude wipe frames and track each visible arrangement separately.
# Cross-segment physical identity is resolved by the configured VLM identity route.
PHYSION_PP_TWO_SEGMENT_TEXT_SCENARIOS = {"friction_collision_pp"}
PHYSION_PP_TEXT_CONCEPT = "small object"
PHYSION_PP_TEXT_CONCEPT_ID = "physion_pp_text_concept"
# Only the first-segment friction-collision patient is treated as a static fixture.
PHYSION_PP_BW_RGB_DIFF_THRESH = 25
PHYSION_PP_BW_CORE_MIN_DIFF_FRAC = 0.20
PHYSION_PP_BW_COL_BAND_TOP = 176
PHYSION_PP_BW_COL_OCCUPANCY = 0.35
PHYSION_PP_BW_COL_ADJACENCY = 8
PHYSION_PP_BW_STALL_FRAMES = 10
PHYSION_PP_BW_BG1_FRAMES = 45
PHYSION_PP_BW_BG2_FRAMES = 11
PHYSION_PP_BW_WALL_POS_IOU = 0.3
PHYSION_PP_BW_STATIC_PAIR_IOU = 0.6
# Mass-collision tracking detects the curtain interval from full-frame change, seeds the
# ball/agent/patient independently, binds the agent across segments through VLM identity,
# and removes border or far-depth background tracks.


def _tracking_shared_module(name: str) -> ModuleSpec:
    return default_module_profile_policy().shared_module(name)


def _video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    metadata = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def _detect_use_fa3() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 9


def _load_model(
    *,
    checkpoint: str | None,
    sam3_version: str,
    compile_model: bool,
    max_num_objects: int,
    async_loading_frames: bool,
) -> tuple[Any, bool]:
    username = os.getenv("USER", "physmind")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}")
    _add_sam3_to_path()
    import torch
    from sam3 import build_sam3_predictor
    from sam3.model_builder import build_sam3_video_predictor

    use_fa3 = _detect_use_fa3()
    if sam3_version == "sam3":
        build_kwargs: dict[str, Any] = {
            "compile": compile_model,
            "async_loading_frames": async_loading_frames,
        }
        if checkpoint:
            build_kwargs["checkpoint_path"] = checkpoint
        model = build_sam3_video_predictor(**build_kwargs)
    else:
        build_kwargs = {
            "version": "sam3.1",
            "compile": compile_model,
            "warm_up": compile_model,
            "max_num_objects": max_num_objects,
            "async_loading_frames": async_loading_frames,
            "use_fa3": use_fa3,
        }
        if checkpoint:
            build_kwargs["checkpoint_path"] = checkpoint
        model = build_sam3_predictor(**build_kwargs)
        _patch_start_session_for_current_sam3(model)
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    return model, use_fa3


def _object_prompts(object_plan: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = []
    for index, item in enumerate(object_plan.get("target_objects", []), start=1):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or f"obj_{index}")
        prompt = str(item.get("description") or item.get("role") or object_id).strip()
        if not prompt:
            prompt = object_id
        prompts.append(
            {
                "object_id": object_id,
                "sam_obj_id": index,
                "prompt": prompt,
            }
        )
    return prompts


def _generic_movable_prompt(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "object_id": "track_prompt_movable",
            "sam_obj_id": 1,
            "prompt": prompt,
        }
    ]


def _vlm_concept_prompts(object_plan: dict[str, Any]) -> list[dict[str, Any]]:
    special_scene = object_plan.get("special_scene") if isinstance(object_plan.get("special_scene"), dict) else {}
    raw_prompts = special_scene.get("sam3_video_tracking_prompts")
    if not isinstance(raw_prompts, list):
        raw_prompts = object_plan.get("sam3_video_tracking_prompts")
    if not isinstance(raw_prompts, list):
        return []
    prompts = []
    for index, item in enumerate(raw_prompts, start=1):
        if not isinstance(item, dict):
            continue
        prompt = " ".join(str(item.get("prompt") or "").strip().split())
        if not prompt:
            continue
        raw_expected = item.get("expected_object_ids")
        if not isinstance(raw_expected, list):
            raw_expected = []
        prompts.append(
            {
                "object_id": str(item.get("concept_id") or f"concept_{index}"),
                "concept_id": str(item.get("concept_id") or f"concept_{index}"),
                "sam_obj_id": index,
                "prompt": prompt,
                "expected_object_ids": [str(object_id) for object_id in raw_expected],
                "reason": str(item.get("reason", "")),
            }
        )
    return prompts


def _bench_concept_prompts(bench: str | None) -> list[dict[str, Any]]:
    if not bench:
        return []
    prompts = BENCH_CONCEPT_PROMPTS.get(str(bench).strip().lower())
    if not prompts:
        return []
    return [
        {
            **item,
            "sam_obj_id": index,
        }
        for index, item in enumerate(prompts, start=1)
    ]


def _select_prompts(
    *,
    object_plan_payload: dict[str, Any],
    mode: str,
    bench: str | None,
    generic_prompt: str,
) -> tuple[list[dict[str, Any]], str]:
    if mode == "auto":
        prompts = _bench_concept_prompts(bench)
        if prompts:
            return prompts, "bench-concepts"
        prompts = _vlm_concept_prompts(object_plan_payload)
        if prompts:
            return prompts, "vlm-concept-prompts"
        return [], "none"
    if mode == "bench-concepts":
        return _bench_concept_prompts(bench), "bench-concepts"
    if mode == "generic-movable":
        return _generic_movable_prompt(generic_prompt), "generic-movable"
    if mode == "vlm-concept-prompts":
        return _vlm_concept_prompts(object_plan_payload), "vlm-concept-prompts"
    return _object_prompts(object_plan_payload), "object-prompts"


def _is_physion_pp_scenario_name(scenario_name: str | None) -> bool:
    return str(scenario_name or "").strip().lower().endswith("_pp")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _object_plan_scene_metadata(object_plan_payload: dict[str, Any]) -> dict[str, Any]:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        return {}
    metadata = special_scene.get("scene_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _resolve_tracking_family_dispatch(
    *,
    bench: str | None,
    scenario_name: str | None,
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    route_record = special_scene.get("tracking_family_route")

    normalized_bench = str(bench or "").strip().lower()
    normalized_scenario = str(scenario_name or "").strip().lower()
    if normalized_bench == "clevrer":
        expected_route = CLEVRER_TRACKING_FAMILY_ROUTE
        expected_policy_benchmark = "clevrer"
    elif normalized_scenario in PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO:
        expected_route = PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO[
            normalized_scenario
        ]
        expected_policy_benchmark = "physion_pp"
    else:
        expected_route = None
        expected_policy_benchmark = None

    if not isinstance(route_record, dict):
        if expected_route is not None:
            raise ValueError(
                "target tracking scope is missing its tracking-family route record: "
                f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
            )
        return None
    if expected_route is None or expected_policy_benchmark is None:
        raise ValueError(
            "tracking-family route record is not applicable to this context: "
            f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != TRACKING_FAMILY_DECISION_ID:
        raise ValueError(
            "tracking-family route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in TRACKING_FAMILY_ROUTES:
        raise ValueError(f"unsupported tracking-family route: {route!r}")
    if route != expected_route:
        raise ValueError(
            "tracking-family route does not match the benchmark/scenario context: "
            f"route={route!r} expected={expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("tracking-family route record is missing its context")
    if context.get("benchmark") != expected_policy_benchmark:
        raise ValueError(
            "tracking-family route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {expected_policy_benchmark!r}"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    expected_scenario = normalized_scenario if expected_policy_benchmark == "physion_pp" else ""
    if recorded_scenario != expected_scenario:
        raise ValueError(
            "tracking-family route scenario context mismatch: "
            f"{recorded_scenario!r} != {expected_scenario!r}"
        )
    return route_record


def _resolve_temporal_partition_dispatch(
    *,
    bench: str | None,
    scenario_name: str | None,
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    route_record = special_scene.get("temporal_partition_route")

    normalized_bench = str(bench or "").strip().lower()
    normalized_scenario = str(scenario_name or "").strip().lower()
    if normalized_bench == "clevrer":
        expected_policy_benchmark = "clevrer"
    elif normalized_scenario in PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO:
        expected_policy_benchmark = "physion_pp"
    else:
        expected_policy_benchmark = None

    if not isinstance(route_record, dict):
        if expected_policy_benchmark is not None:
            raise ValueError(
                "target tracking scope is missing its temporal-partition route record: "
                f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
            )
        return None
    if expected_policy_benchmark is None:
        raise ValueError(
            "temporal-partition route record is not applicable to this context: "
            f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != TEMPORAL_PARTITION_DECISION_ID:
        raise ValueError(
            "temporal-partition route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in TEMPORAL_PARTITION_ROUTES:
        raise ValueError(f"unsupported temporal-partition route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("temporal-partition route record is missing its context")
    if context.get("benchmark") != expected_policy_benchmark:
        raise ValueError(
            "temporal-partition route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {expected_policy_benchmark!r}"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    expected_scenario = (
        normalized_scenario if expected_policy_benchmark == "physion_pp" else ""
    )
    if recorded_scenario != expected_scenario:
        raise ValueError(
            "temporal-partition route scenario context mismatch: "
            f"{recorded_scenario!r} != {expected_scenario!r}"
        )
    return route_record


def _resolve_cue_role_binding_dispatch(
    *,
    bench: str | None,
    scenario_name: str | None,
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    route_record = special_scene.get("cue_role_binding_route")

    normalized_bench = str(bench or "").strip().lower()
    normalized_scenario = str(scenario_name or "").strip().lower()
    is_target_scope = (
        normalized_scenario in PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO
    )
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target tracking scope is missing its cue-role-binding route record: "
                f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "cue-role-binding route record is not applicable to this context: "
            f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != CUE_ROLE_BINDING_DECISION_ID:
        raise ValueError(
            "cue-role-binding route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != CUE_ROLE_BINDING_ROUTE:
        raise ValueError(f"unsupported cue-role-binding route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("cue-role-binding route record is missing its context")
    if context.get("benchmark") != "physion_pp":
        raise ValueError(
            "cue-role-binding route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != 'physion_pp'"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "cue-role-binding route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_cue_role_binding_result(
    physion_tracking: dict[str, Any] | None,
    route_record: dict[str, Any],
) -> None:
    if not isinstance(physion_tracking, dict):
        raise ValueError(
            "cue-role-binding target tracking did not produce physion_tracking"
        )
    if route_record.get("decision_id") != CUE_ROLE_BINDING_DECISION_ID:
        raise ValueError(
            "cue-role-binding route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != CUE_ROLE_BINDING_ROUTE:
        raise ValueError(f"unsupported cue-role-binding route: {route!r}")
    physion_tracking["cue_role_binding_route"] = route_record


def _resolve_static_role_assignment_dispatch(
    *,
    bench: str | None,
    scenario_name: str | None,
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    route_record = special_scene.get("static_role_assignment_route")

    normalized_bench = str(bench or "").strip().lower()
    normalized_scenario = str(scenario_name or "").strip().lower()
    is_target_scope = normalized_scenario in STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target tracking scope is missing its static-role-assignment route record: "
                f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "static-role-assignment route record is not applicable to this context: "
            f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != STATIC_ROLE_ASSIGNMENT_DECISION_ID:
        raise ValueError(
            "static-role-assignment route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in STATIC_ROLE_ASSIGNMENT_ROUTES:
        raise ValueError(f"unsupported static-role-assignment route: {route!r}")
    expected_route = STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO[normalized_scenario]
    if route != expected_route:
        raise ValueError(
            "static-role-assignment route does not match the scenario context: "
            f"route={route!r} expected={expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("static-role-assignment route record is missing its context")
    if context.get("benchmark") != "physion_pp":
        raise ValueError(
            "static-role-assignment route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != 'physion_pp'"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "static-role-assignment route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_static_role_assignment_result(
    physion_tracking: dict[str, Any] | None,
    route_record: dict[str, Any],
) -> None:
    if not isinstance(physion_tracking, dict):
        raise ValueError(
            "static-role-assignment target tracking did not produce physion_tracking"
        )
    if route_record.get("decision_id") != STATIC_ROLE_ASSIGNMENT_DECISION_ID:
        raise ValueError(
            "static-role-assignment route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in STATIC_ROLE_ASSIGNMENT_ROUTES:
        raise ValueError(f"unsupported static-role-assignment route: {route!r}")
    physion_tracking["static_role_assignment_route"] = route_record


def _resolve_cross_segment_identity_dispatch(
    *,
    bench: str | None,
    scenario_name: str | None,
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    route_record = special_scene.get("cross_segment_identity_route")

    normalized_bench = str(bench or "").strip().lower()
    normalized_scenario = str(scenario_name or "").strip().lower()
    is_target_scope = normalized_scenario in CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target tracking scope is missing its cross-segment-identity route record: "
                f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "cross-segment-identity route record is not applicable to this context: "
            f"bench={normalized_bench!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != CROSS_SEGMENT_IDENTITY_DECISION_ID:
        raise ValueError(
            "cross-segment-identity route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in CROSS_SEGMENT_IDENTITY_ROUTES:
        raise ValueError(f"unsupported cross-segment-identity route: {route!r}")
    expected_route = CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO[normalized_scenario]
    if route != expected_route:
        raise ValueError(
            "cross-segment-identity route does not match the scenario context: "
            f"route={route!r} expected={expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("cross-segment-identity route record is missing its context")
    if context.get("benchmark") != "physion_pp":
        raise ValueError(
            "cross-segment-identity route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != 'physion_pp'"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "cross-segment-identity route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_cross_segment_identity_result(
    physion_tracking: dict[str, Any] | None,
    route_record: dict[str, Any],
) -> None:
    if not isinstance(physion_tracking, dict):
        raise ValueError(
            "cross-segment-identity target tracking did not produce physion_tracking"
        )
    if route_record.get("decision_id") != CROSS_SEGMENT_IDENTITY_DECISION_ID:
        raise ValueError(
            "cross-segment-identity route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in CROSS_SEGMENT_IDENTITY_ROUTES:
        raise ValueError(f"unsupported cross-segment-identity route: {route!r}")
    physion_tracking["cross_segment_identity_route"] = route_record


def _tracking_dispatch_name(
    *,
    is_physion_pp: bool,
    scenario_name: str | None,
    tracking_family_route: str | None,
    temporal_partition_route: str | None,
) -> str:
    dispatch_by_route = {
        CLEVRER_TRACKING_FAMILY_ROUTE: "joint_prompt_tracking",
        FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE: "friction_platform_tracking",
        BOUNCY_WALL_TRACKING_FAMILY_ROUTE: "bouncy_wall_tracking",
        BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE: "bouncy_platform_tracking",
        FRICTION_COLLISION_TRACKING_FAMILY_ROUTE: "friction_collision_tracking",
        MASS_COLLISION_TRACKING_FAMILY_ROUTE: "mass_collision_tracking",
    }
    if tracking_family_route is not None:
        expected_temporal_partition_route = (
            TEMPORAL_PARTITION_ROUTE_BY_TRACKING_FAMILY.get(
                tracking_family_route
            )
        )
        if expected_temporal_partition_route is None:
            raise ValueError(
                f"unsupported tracking-family route: {tracking_family_route!r}"
            )
        if temporal_partition_route != expected_temporal_partition_route:
            raise ValueError(
                "tracking-family route and temporal-partition route are inconsistent: "
                f"tracking_family={tracking_family_route!r} "
                f"temporal_partition={temporal_partition_route!r} "
                f"expected={expected_temporal_partition_route!r}"
            )
        try:
            return dispatch_by_route[tracking_family_route]
        except KeyError as exc:
            raise ValueError(
                f"unsupported tracking-family route: {tracking_family_route!r}"
            ) from exc
    if is_physion_pp:
        raise ValueError(
            "Physion++ tracking requires a JSON tracking-family route for one of "
            f"the five supported scenarios; got scenario={scenario_name!r}"
        )
    return "joint_prompt_tracking"


def _require_tracking_profile_scenario(
    module_profile: TrackingModuleProfile,
    scenario_name: str,
    *,
    frontend: str,
) -> None:
    scenario_key = str(scenario_name).strip().lower()
    if scenario_key not in module_profile.scenarios:
        raise ValueError(
            f"{frontend} does not support scenario {scenario_name!r}"
        )


def _physion_pp_scenario_name(
    *,
    object_plan_payload: dict[str, Any],
) -> str | None:
    metadata = _object_plan_scene_metadata(object_plan_payload)
    scenario = metadata.get("scenario")
    if scenario is None:
        special_scene = object_plan_payload.get("special_scene")
        if isinstance(special_scene, dict):
            scenario = special_scene.get("scenario")
    if scenario is None:
        scenario = object_plan_payload.get("scenario")
    if scenario is not None:
        name = str(scenario).strip()
        return name or None
    return None


def _physion_main_prompts(scenario_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_prompts = scenario_config.get("main_prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raw_prompts = [{"prompt": PHYSION_DEFAULT_MAIN_PROMPT, "concept_id": PHYSION_DEFAULT_MAIN_CONCEPT_ID}]
    prompts = []
    for index, item in enumerate(raw_prompts, start=1):
        prompt = _normalize_text(item.get("prompt")) or PHYSION_DEFAULT_MAIN_PROMPT
        concept_id = str(item.get("concept_id") or PHYSION_DEFAULT_MAIN_CONCEPT_ID)
        reason = str(item.get("reason") or "Physion++ main dynamic object category")
        prompts.append(
            {
                "object_id": concept_id,
                "concept_id": concept_id,
                "sam_obj_id": index,
                "prompt": prompt,
                "expected_object_ids": [],
                "reason": reason,
                "track_prefix": str(item.get("track_prefix") or ""),
            }
        )
    return prompts


def _records_by_object(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    records_by_object: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        object_id = str(record.get("object_id") or "")
        if object_id:
            records_by_object.setdefault(object_id, []).append(record)
    return records_by_object


def _read_video_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for Physion++ yellow patch validation: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read frame {frame_index} for Physion++ yellow patch validation: {video}")
        return frame
    finally:
        capture.release()


def _score_by_obj_id(response: dict[str, Any]) -> dict[int, float | None]:
    import torch

    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    obj_ids = [int(item) for item in np.asarray(obj_ids).reshape(-1)]
    scores = outputs.get("out_scores")
    if scores is None:
        scores = outputs.get("scores")
    if scores is None:
        return {obj_id: None for obj_id in obj_ids}
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    return {obj_id: (float(scores[index]) if index < len(scores) else None) for index, obj_id in enumerate(obj_ids)}


def _append_records_from_response(
    *,
    response: dict[str, Any],
    object_id_by_sam_obj_id: dict[int, str],
    concept_id_by_sam_obj_id: dict[int, str],
    prompt_by_sam_obj_id: dict[int, str],
    expected_object_ids_by_sam_obj_id: dict[int, list[str]],
    frame_index: int,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> None:
    masks_by_obj_id = _masks_from_response(response)
    scores_by_obj_id = _score_by_obj_id(response)
    for sam_obj_id, mask in sorted(masks_by_obj_id.items()):
        object_id = object_id_by_sam_obj_id.get(int(sam_obj_id), f"anonymous_track_{int(sam_obj_id):03d}")
        mask_key = f"{object_id}__frame_{frame_index:05d}__sam_obj_{int(sam_obj_id)}"
        mask_u8 = mask.astype(np.uint8)
        all_arrays[mask_key] = mask_u8
        all_records.append(
            {
                "object_id": object_id,
                "concept_id": concept_id_by_sam_obj_id.get(int(sam_obj_id)),
                "prompt": prompt_by_sam_obj_id.get(int(sam_obj_id)),
                "expected_object_ids": expected_object_ids_by_sam_obj_id.get(int(sam_obj_id), []),
                "frame_index": int(frame_index),
                "sam_object_id": int(sam_obj_id),
                "mask_key": mask_key,
                "score": scores_by_obj_id.get(int(sam_obj_id)),
                **_mask_stats(mask_u8),
            }
        )


def _track_object_id(
    *,
    resolved_mode: str,
    prompt_item: dict[str, Any],
    sam_obj_id: int,
) -> str:
    if resolved_mode in {"bench-concepts", "generic-movable", "vlm-concept-prompts"}:
        return f"anonymous_track_{sam_obj_id:03d}"
    return str(prompt_item.get("object_id") or f"object_{sam_obj_id:03d}")


def _run_joint_prompt_tracking(
    *,
    model: Any,
    video: Path,
    prompts: list[dict[str, Any]],
    resolved_mode: str,
    prompt_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    session_response = model.handle_request({"type": "start_session", "resource_path": str(video)})
    session_id = session_response["session_id"]
    object_id_by_sam_obj_id: dict[int, str] = {}
    concept_id_by_sam_obj_id: dict[int, str] = {}
    prompt_by_sam_obj_id: dict[int, str] = {}
    expected_object_ids_by_sam_obj_id: dict[int, list[str]] = {}
    prompt_responses = []
    try:
        for prompt_index, item in enumerate(prompts, start=1):
            request = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "text": item["prompt"],
            }
            if resolved_mode == "object-prompts":
                request["obj_id"] = int(item["sam_obj_id"])
            prompt_response = model.handle_request(request)
            masks_by_prompt_obj_id = _masks_from_response(prompt_response)
            for sam_obj_id in masks_by_prompt_obj_id:
                sam_obj_id_int = int(sam_obj_id)
                object_id_by_sam_obj_id[sam_obj_id_int] = _track_object_id(
                    resolved_mode=resolved_mode,
                    prompt_item=item,
                    sam_obj_id=sam_obj_id_int,
                )
                concept_id_by_sam_obj_id[sam_obj_id_int] = str(item.get("concept_id") or item.get("object_id"))
                prompt_by_sam_obj_id[sam_obj_id_int] = str(item["prompt"])
                expected_object_ids_by_sam_obj_id[sam_obj_id_int] = [
                    str(object_id) for object_id in item.get("expected_object_ids", [])
                ]
            prompt_responses.append(
                {
                    "object_id": item["object_id"],
                    "concept_id": item.get("concept_id"),
                    "sam_object_id": item["sam_obj_id"],
                    "prompt": item["prompt"],
                    "frame_index": prompt_response.get("frame_index"),
                    "mask_count": len(masks_by_prompt_obj_id),
                    "prompt_index": prompt_index,
                    "expected_object_ids": item.get("expected_object_ids", []),
                }
            )

        stream_request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": prompt_frame_index,
        }
        if max_frame_num_to_track is not None:
            stream_request["max_frame_num_to_track"] = int(max_frame_num_to_track)

        seen_frame_indices = set()
        for response in model.handle_stream_request(stream_request):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices:
                continue
            seen_frame_indices.add(frame_index)
            _append_records_from_response(
                response=response,
                object_id_by_sam_obj_id=object_id_by_sam_obj_id,
                concept_id_by_sam_obj_id=concept_id_by_sam_obj_id,
                prompt_by_sam_obj_id=prompt_by_sam_obj_id,
                expected_object_ids_by_sam_obj_id=expected_object_ids_by_sam_obj_id,
                frame_index=frame_index,
                all_records=all_records,
                all_arrays=all_arrays,
            )
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})
    return prompt_responses




def _physion_pp_bouncy_wall_gdino_prompts(
    *,
    video: Path,
    scenario_name: str,
    prompt_frame_index: int,
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build bouncy-wall role prompts from GDINO detections."""
    from scripts.world_model.run_gdino_boxes import (
        build_query,
        cleanup_boxes,
        detect_boxes,
        load_gdino,
    )

    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="bouncy-wall GDINO frontend",
    )

    detector = module_profile.module("detector")
    cleanup = _tracking_shared_module("gdino_cleanup")
    frame = _read_video_frame(video, prompt_frame_index)
    height, width = frame.shape[:2]
    processor, gdino_model, device = load_gdino()
    query = build_query(list(detector.require_strings("vocabulary")))
    raw = detect_boxes(
        processor,
        gdino_model,
        frame,
        query,
        confidence=cleanup.require_number("confidence"),
        device=device,
    )
    cleaned = cleanup_boxes(
        raw,
        image_width=width,
        image_height=height,
        area_cap_fraction=cleanup.require_number("area_cap_fraction"),
        nms_iou=cleanup.require_number("same_label_nms_iou"),
        edge_margin_px=cleanup.require_integer("edge_margin_px"),
        edge_sides_to_drop=cleanup.require_integer("edge_sides_to_drop"),
    )

    prompts = []
    for rank, det in enumerate(sorted(cleaned, key=lambda item: -item["score"]), start=1):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx = min(max((x1 + x2) / 2.0, 0.0), width - 1.0) / width
        cy = min(max((y1 + y2) / 2.0, 0.0), height - 1.0) / height
        prompts.append(
            {
                "sam_obj_id": rank,
                "class": str(det["class"]),
                "score": float(det["score"]),
                "bbox_xyxy": [int(value) for value in det["bbox_xyxy"]],
                "points": [[cx, cy]],
                "point_labels": [1],
                "prompt_mode": "point",
            }
        )
    summary = {
        "mode": "bouncy_wall_mixed_vocab_point_prompts",
        "query": query,
        "frame_index": int(prompt_frame_index),
        "image_size": [int(width), int(height)],
        "raw_box_count": len(raw),
        "cleaned_boxes": cleaned,
    }
    return prompts, summary


def _physion_pp_friction_platform_gdino_prompts(
    *,
    video: Path,
    scenario_name: str,
    prompt_frame_index: int,
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """friction_platform role-specific GDINO frontend.

    Red is a BOX prompt in a solo session, yellow is a POINT prompt in a solo
    session, and house/long-ramp/small-wedge are BOX prompts in a joint structure
    session. The five independently calibrated detector rounds are merged before
    SAM3 so cross-query duplicates cannot steal identities at prompt time.
    """
    from scripts.world_model.run_gdino_boxes import (
        build_query,
        box_iou,
        detect_boxes,
        load_gdino,
    )

    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="friction-platform GDINO frontend",
    )
    detector = module_profile.module("detector")
    detector_rounds = {
        str(item.get("name") or ""): item
        for item in detector.require_records("rounds")
    }
    expected_rounds = {"red", "yellow", "house", "long_ramp", "small_wedge"}
    if set(detector_rounds) != expected_rounds:
        raise ValueError(
            "friction-platform GDINO profile must define exactly "
            f"{sorted(expected_rounds)!r}, got {sorted(detector_rounds)!r}"
        )

    frame = _read_video_frame(video, prompt_frame_index)
    height, width = frame.shape[:2]
    processor, gdino_model, device = load_gdino()

    def detect_round(name: str, query: str, confidence: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        detections = detect_boxes(
            processor,
            gdino_model,
            frame,
            build_query([query]),
            confidence=confidence,
            device=device,
        )
        detections.sort(key=lambda item: -float(item["score"]))
        return detections, {
            "name": name,
            "query": query,
            "confidence_threshold": float(confidence),
            "detections": detections,
        }

    red_dets, red_round = detect_round(
        "red",
        str(detector_rounds["red"]["query"]),
        float(detector_rounds["red"]["confidence"]),
    )
    yellow_dets, yellow_round = detect_round(
        "yellow",
        str(detector_rounds["yellow"]["query"]),
        float(detector_rounds["yellow"]["confidence"]),
    )
    house_dets, house_round = detect_round(
        "house",
        str(detector_rounds["house"]["query"]),
        float(detector_rounds["house"]["confidence"]),
    )
    long_dets, long_round = detect_round(
        "long_ramp",
        str(detector_rounds["long_ramp"]["query"]),
        float(detector_rounds["long_ramp"]["confidence"]),
    )
    small_dets, small_round = detect_round(
        "small_wedge",
        str(detector_rounds["small_wedge"]["query"]),
        float(detector_rounds["small_wedge"]["confidence"]),
    )

    def clipped(box: list[int]) -> list[int]:
        x1, y1, x2, y2 = [int(value) for value in box]
        return [
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        ]

    def intersection_over_smaller(first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return float(intersection / min(area_a, area_b))

    def same_structure_box(first: list[int], second: list[int]) -> bool:
        return (
            box_iou(first, second)
            >= detector.require_number("cross_round_merge_iou")
            or intersection_over_smaller(first, second)
            >= detector.require_number("cross_round_merge_containment")
        )

    # One large structure exists in every scene. In no-house scenes the house query may
    # independently rediscover the long ramp, so merge the two rounds by box geometry.
    big_candidates = []
    if house_dets:
        big_candidates.append({**house_dets[0], "detector_round": "house"})
    if long_dets:
        big_candidates.append({**long_dets[0], "detector_round": "long_ramp"})
    big_structures: list[dict[str, Any]] = []
    for candidate in big_candidates:
        duplicate = next(
            (
                item
                for item in big_structures
                if same_structure_box(item["bbox_xyxy"], candidate["bbox_xyxy"])
            ),
            None,
        )
        if duplicate is None:
            big_structures.append({**candidate, "detector_rounds": [candidate["detector_round"]]})
            continue
        duplicate["detector_rounds"].append(candidate["detector_round"])
        if float(candidate["score"]) > float(duplicate["score"]):
            rounds = duplicate["detector_rounds"]
            duplicate.clear()
            duplicate.update({**candidate, "detector_rounds": rounds})

    # The small-wedge query also emits already-known big/yellow boxes. Remove those and
    # cross-query duplicates before prompting SAM3. Red is intentionally NOT a containment
    # reference: a real ramp can contain the tiny red box when the agent rests on it.
    prior_boxes = [item["bbox_xyxy"] for item in big_structures]
    if yellow_dets:
        prior_boxes.append(yellow_dets[0]["bbox_xyxy"])
    small_structures: list[dict[str, Any]] = []
    for candidate in small_dets:
        box = candidate["bbox_xyxy"]
        if any(same_structure_box(box, prior_box) for prior_box in prior_boxes):
            continue
        if any(same_structure_box(box, item["bbox_xyxy"]) for item in small_structures):
            continue
        small_structures.append({**candidate, "detector_round": "small_wedge"})

    prompts: list[dict[str, Any]] = []

    def add_prompt(
        detection: dict[str, Any], *, group: str, role: str, prompt_mode: str
    ) -> None:
        x1, y1, x2, y2 = clipped(detection["bbox_xyxy"])
        if x2 <= x1 or y2 <= y1:
            return
        if prompt_mode == "point":
            points = [[(x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height]]
            point_labels = [1]
        else:
            points = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
            point_labels = [2, 3]
        prompts.append(
            {
                "sam_obj_id": len(prompts) + 1,
                "class": str(detection["class"]),
                "score": float(detection["score"]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "points": points,
                "point_labels": point_labels,
                "prompt_mode": prompt_mode,
                "prompt_group": group,
                "role": role,
                "detector_round": detection.get("detector_round"),
                "detector_rounds": detection.get("detector_rounds"),
            }
        )

    if red_dets:
        add_prompt(red_dets[0], group="red", role="agent", prompt_mode="box")
    if yellow_dets:
        add_prompt(yellow_dets[0], group="yellow", role="patient", prompt_mode="point")
    for detection in big_structures:
        add_prompt(detection, group="structure", role="big_structure", prompt_mode="box")
    for detection in small_structures:
        add_prompt(detection, group="structure", role="small_wedge", prompt_mode="box")

    summary = {
        "frame_index": int(prompt_frame_index),
        "image_size": [int(width), int(height)],
        "mode": "friction_platform_role_specific_rounds",
        "rounds": [red_round, yellow_round, house_round, long_round, small_round],
        "selected": [
            {key: value for key, value in prompt.items() if key != "points"}
            for prompt in prompts
        ],
        "big_structure_count": len(big_structures),
        "small_structure_count": len(small_structures),
    }
    return prompts, summary


def _pp_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int((first & second).sum())
    union = int(first.sum()) + int(second.sum()) - intersection
    return intersection / union if union else 0.0


def _pp_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _pp_presence_runs(frames_sorted: list[int]) -> list[tuple[int, int]]:
    runs = []
    start = prev = frames_sorted[0]
    for frame in frames_sorted[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        runs.append((start, prev))
        start = prev = frame
    runs.append((start, prev))
    return runs


def _pp_net_displacement(frames: dict[int, np.ndarray]) -> float:
    order = sorted(frames)
    head = np.mean([_pp_centroid(frames[f]) for f in order[:3]], axis=0)
    tail = np.mean([_pp_centroid(frames[f]) for f in order[-3:]], axis=0)
    return float(np.hypot(*(tail - head)))


def _pp_same_object_evidence(
    fi: dict[int, np.ndarray],
    fj: dict[int, np.ndarray],
    *,
    evidence_kinds: frozenset[str],
) -> str | None:
    """Pairwise same-object test over per-frame masks. Evidence kinds:
    overlap = classic same-frame duplicates; relay = alternating ownership with a
    spatially continuous handover (flicker); comotion = two dynamic fragments that
    stay in contact and move rigidly together (two-piece agents)."""
    merge_profile = _tracking_shared_module("same_object_merge")
    shared = sorted(set(fi) & set(fj))
    if (
        "overlap" in evidence_kinds
        and len(shared)
        >= merge_profile.require_integer("overlap_min_shared_frames")
    ):
        step = max(1, len(shared) // 20)
        ious = [_pp_mask_iou(fi[f], fj[f]) for f in shared[::step][:20]]
        if float(np.mean(ious)) >= merge_profile.require_number("overlap_iou"):
            return "overlap"
    if (
        "relay" in evidence_kinds
        and fi
        and fj
        and len(shared)
        <= merge_profile.require_number("relay_coexist_fraction")
        * min(len(fi), len(fj))
    ):
        for a, b in ((fi, fj), (fj, fi)):
            for _, end_a in _pp_presence_runs(sorted(a)):
                for start_b, _ in _pp_presence_runs(sorted(b)):
                    if (
                        0 < start_b - end_a
                        <= merge_profile.require_integer("relay_gap_frames")
                    ):
                        ca, cb = _pp_centroid(a[end_a]), _pp_centroid(b[start_b])
                        if (
                            np.hypot(ca[0] - cb[0], ca[1] - cb[1])
                            <= merge_profile.require_number("relay_centroid_px")
                        ):
                            return "relay"
    if (
        "comotion" in evidence_kinds
        and len(shared)
        >= merge_profile.require_integer("comotion_min_shared_frames")
    ):
        if (
            _pp_net_displacement(fi)
            >= merge_profile.require_number("comotion_min_displacement_px")
            and _pp_net_displacement(fj)
            >= merge_profile.require_number("comotion_min_displacement_px")
        ):
            size = 2 * merge_profile.require_integer("comotion_dilate_px") + 1
            kernel = np.ones((size, size), np.uint8)
            contact = 0
            distances = []
            for frame in shared:
                if (cv2.dilate(fi[frame].astype(np.uint8), kernel).astype(bool) & fj[frame]).any():
                    contact += 1
                ci, cj = _pp_centroid(fi[frame]), _pp_centroid(fj[frame])
                distances.append(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
            if (
                contact / len(shared)
                >= merge_profile.require_number("comotion_contact_fraction")
                and float(np.std(distances))
                <= merge_profile.require_number("comotion_distance_std_px")
            ):
                return "comotion"
    return None


def _apply_physion_pp_same_object_merge(
    *,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    score_by_object_id: dict[str, float],
    evidence_kinds: frozenset[str],
    eligible_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Union-find same-object merge over the unified track stream. Groups are merged
    into the top-scoring member: per-frame pixel union under its object_id/sam id.
    When ``eligible_object_ids`` is supplied, only that family participates in pairwise
    merge decisions; tracks outside that family remain untouched."""
    records_by_object = _records_by_object(all_records)
    masks_by_object: dict[str, dict[int, np.ndarray]] = {}
    for object_id, records in records_by_object.items():
        if eligible_object_ids is not None and object_id not in eligible_object_ids:
            continue
        frames = {}
        for record in records:
            mask = all_arrays.get(str(record.get("mask_key")))
            if mask is not None and mask.any():
                frames[int(record["frame_index"])] = mask.astype(bool)
        if frames:
            masks_by_object[object_id] = frames

    object_ids = sorted(masks_by_object)
    parent = {object_id: object_id for object_id in object_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for index, first_id in enumerate(object_ids):
        for second_id in object_ids[index + 1:]:
            evidence = _pp_same_object_evidence(
                masks_by_object[first_id],
                masks_by_object[second_id],
                evidence_kinds=evidence_kinds,
            )
            if evidence:
                edges.append({"pair": [first_id, second_id], "evidence": evidence})
                parent[find(first_id)] = find(second_id)

    groups: dict[str, list[str]] = {}
    for object_id in object_ids:
        groups.setdefault(find(object_id), []).append(object_id)
    merged_groups = [members for members in groups.values() if len(members) > 1]
    if not merged_groups:
        return {
            "evidence_kinds": sorted(evidence_kinds),
            "merged_group_count": 0,
            "edges": [],
            "groups": [],
        }

    group_summaries = []
    for members in merged_groups:
        survivor = max(members, key=lambda oid: score_by_object_id.get(oid, 0.0))
        template = records_by_object[survivor][0]
        sam_obj_id = int(template.get("sam_object_id") or 0)
        union_frames: dict[int, np.ndarray] = {}
        for member in members:
            for frame, mask in masks_by_object[member].items():
                union_frames[frame] = (union_frames[frame] | mask) if frame in union_frames else mask.copy()
        absorbed = [member for member in members if member != survivor]
        for member in members:
            for record in records_by_object[member]:
                all_arrays.pop(str(record.get("mask_key")), None)
        survivor_records = []
        for frame in sorted(union_frames):
            mask_u8 = union_frames[frame].astype(np.uint8)
            mask_key = f"{survivor}__frame_{frame:05d}__sam_obj_{sam_obj_id}"
            all_arrays[mask_key] = mask_u8
            survivor_records.append(
                {
                    "object_id": survivor,
                    "concept_id": template.get("concept_id"),
                    "prompt": template.get("prompt"),
                    "expected_object_ids": [],
                    "frame_index": int(frame),
                    "sam_object_id": sam_obj_id,
                    "mask_key": mask_key,
                    "score": None,
                    **_mask_stats(mask_u8),
                }
            )
        member_set = set(members)
        all_records[:] = [
            record for record in all_records if str(record["object_id"]) not in member_set
        ] + survivor_records
        group_summaries.append(
            {"survivor": survivor, "absorbed": absorbed, "frame_count": len(union_frames)}
        )
    return {
        "evidence_kinds": sorted(evidence_kinds),
        "merged_group_count": len(merged_groups),
        "edges": edges,
        "groups": group_summaries,
    }


def _apply_physion_pp_structure_fragment_cleanup(
    *,
    structure_track_ids: set[str],
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> dict[str, Any]:
    """Remove only tiny 8-connected islands from post-carve structure masks.

    The largest component is always preserved. Secondary components survive when
    their area reaches the conservative absolute/relative threshold, so subtracting
    an agent that crosses a ramp cannot force a destructive largest-component-only
    policy. Empty records are removed to keep centroid-based temporal logic finite.
    """
    postprocess = module_profile.module("postprocess")
    fragment_minimum_area_px = postprocess.require_integer(
        "fragment_minimum_area_px"
    )
    fragment_minimum_largest_fraction = postprocess.require_number(
        "fragment_minimum_largest_fraction"
    )
    removed_component_count = 0
    removed_pixel_count = 0
    changed_record_count = 0
    empty_record_count = 0
    per_track: dict[str, dict[str, int]] = {}
    kept_records: list[dict[str, Any]] = []
    for record in all_records:
        track_id = str(record.get("object_id") or "")
        if track_id not in structure_track_ids:
            kept_records.append(record)
            continue
        key = str(record.get("mask_key") or "")
        source = all_arrays.get(key)
        if source is None:
            kept_records.append(record)
            continue
        mask = source.astype(bool)
        track_stats = per_track.setdefault(
            track_id,
            {"changed_records": 0, "removed_components": 0, "removed_pixels": 0, "empty_records": 0},
        )
        if not mask.any():
            all_arrays.pop(key, None)
            empty_record_count += 1
            track_stats["empty_records"] += 1
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 2:
            kept_records.append(record)
            continue
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(areas[largest_label - 1])
        min_area = max(
            fragment_minimum_area_px,
            int(np.ceil(largest_area * fragment_minimum_largest_fraction)),
        )
        keep_labels = {largest_label}
        for label, area in enumerate(areas, start=1):
            if int(area) >= min_area:
                keep_labels.add(label)
        cleaned = np.isin(labels, list(keep_labels))
        removed_labels = [label for label in range(1, count) if label not in keep_labels]
        removed_pixels = int(mask.sum() - cleaned.sum())
        if removed_labels:
            all_arrays[key] = cleaned.astype(np.uint8)
            record.update(_mask_stats(cleaned.astype(np.uint8)))
            removed_component_count += len(removed_labels)
            removed_pixel_count += removed_pixels
            changed_record_count += 1
            track_stats["changed_records"] += 1
            track_stats["removed_components"] += len(removed_labels)
            track_stats["removed_pixels"] += removed_pixels
        kept_records.append(record)
    all_records[:] = kept_records
    return {
        "applied": True,
        "connectivity": 8,
        "min_area_px": fragment_minimum_area_px,
        "min_largest_fraction": fragment_minimum_largest_fraction,
        "changed_record_count": changed_record_count,
        "removed_component_count": removed_component_count,
        "removed_pixel_count": removed_pixel_count,
        "empty_record_count": empty_record_count,
        "per_track": per_track,
    }


def _bind_physion_pp_roles(
    *,
    video: Path,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Bind merged tracks to agent/patient via the staged cue-flash regions.

    Reads the sibling *_cue_flash.npz written at staging time. If the cue metadata
    is absent, binding is skipped. One track cannot take both
    roles: the higher-coverage assignment wins and the other role re-picks."""
    artifact_identity = _tracking_shared_module("artifact_identity")
    cue_role_binding = _tracking_shared_module("cue_role_binding")
    cue_trimmed_suffix = artifact_identity.require_string("cue_trimmed_suffix")
    cue_flash_suffix = artifact_identity.require_string("cue_flash_suffix")
    minimum_coverage = cue_role_binding.require_number("minimum_coverage")
    tie_margin = cue_role_binding.require_number("tie_margin")
    if not video.name.endswith(cue_trimmed_suffix):
        return {"status": "skipped_not_trimmed_clip"}
    flash_path = video.with_name(
        video.name.replace(cue_trimmed_suffix, cue_flash_suffix)
    )
    if not flash_path.exists():
        return {"status": "skipped_no_flash_npz", "expected": str(flash_path)}
    flash = np.load(flash_path)
    regions = {
        "agent": flash["red_region"].astype(bool),
        "patient": flash["yellow_region"].astype(bool),
    }

    per_track: dict[str, dict[str, Any]] = {}
    for track_id, records in _records_by_object(all_records).items():
        ordered = sorted(records, key=lambda r: int(r["frame_index"]))
        masks = [
            all_arrays[str(r["mask_key"])].astype(bool)
            for r in ordered
            if str(r.get("mask_key")) in all_arrays
        ]
        if not masks:
            continue
        last_mask = masks[-1]
        area = max(1, int(last_mask.sum()))
        first_c, last_c = _pp_centroid(masks[0]), _pp_centroid(last_mask)
        per_track[track_id] = {
            "agent_coverage": round(float((regions["agent"] & last_mask).sum() / area), 3),
            "patient_coverage": round(float((regions["patient"] & last_mask).sum() / area), 3),
            "net_displacement_px": round(float(np.hypot(last_c[0] - first_c[0], last_c[1] - first_c[1])), 1),
        }

    def pick(role: str, exclude: str | None) -> str | None:
        key = f"{role}_coverage"
        candidates = [
            (track_id, info) for track_id, info in per_track.items()
            if track_id != exclude and info[key] >= minimum_coverage
        ]
        if not candidates:
            return None
        best_cov = max(info[key] for _, info in candidates)
        near = [
            (track_id, info) for track_id, info in candidates
            if best_cov - info[key] <= tie_margin
        ]
        # scenario prior: the red agent moves, the yellow patient mat is static
        dynamic_first = role == "agent"
        near.sort(key=lambda item: item[1]["net_displacement_px"], reverse=dynamic_first)
        return near[0][0]

    agent = pick("agent", exclude=None)
    patient = pick("patient", exclude=None)
    if agent is not None and agent == patient:
        info = per_track[agent]
        if info["agent_coverage"] >= info["patient_coverage"]:
            patient = pick("patient", exclude=agent)
        else:
            agent = pick("agent", exclude=patient)
    return {
        "status": "ok",
        "flash_regions": flash_path.name,
        "min_coverage": minimum_coverage,
        "agent_track": agent,
        "patient_track": patient,
        "per_track": per_track,
    }


def _apply_physion_pp_static_prefix(
    *,
    role_binding: dict[str, Any],
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Rename non-agent, near-static Physion++ tracks onto physion_pp_static_.

    Static test = the mask-centroid net displacement already computed by role binding;
    the agent track is always exempt. The rename rewrites records, mask keys, and the
    role-binding payload so the whole track stream stays self-consistent."""
    artifact_identity = _tracking_shared_module("artifact_identity")
    static_assignment = _tracking_shared_module("static_role_assignment")
    static_track_prefix = artifact_identity.require_string("static_track_prefix")
    maximum_displacement_px = static_assignment.require_number(
        "maximum_displacement_px"
    )
    if role_binding.get("status") != "ok":
        return {"applied": False, "reason": "role_binding_not_ok", "renamed": []}
    agent_track = str(role_binding.get("agent_track") or "")
    if not agent_track:
        return {"applied": False, "reason": "no_agent_track", "renamed": []}
    per_track = role_binding.get("per_track") or {}
    mapping: dict[str, str] = {}
    renamed = []
    static_index = 0
    for track_id in sorted(per_track):
        info = per_track.get(track_id) or {}
        displacement = float(info.get("net_displacement_px", float("inf")))
        if track_id == agent_track or displacement >= maximum_displacement_px:
            continue
        new_id = f"{static_track_prefix}{static_index:03d}"
        static_index += 1
        mapping[track_id] = new_id
        renamed.append({"from": track_id, "to": new_id, "net_displacement_px": displacement})
    if not mapping:
        return {
            "applied": False,
            "reason": "no_static_candidates",
            "max_disp_px": maximum_displacement_px,
            "renamed": [],
        }
    for record in all_records:
        new_id = mapping.get(str(record.get("object_id")))
        if not new_id:
            continue
        old_key = str(record.get("mask_key"))
        new_key = (
            f"{new_id}__frame_{int(record['frame_index']):05d}"
            f"__sam_obj_{int(record.get('sam_object_id') or 0)}"
        )
        if old_key in all_arrays:
            all_arrays[new_key] = all_arrays.pop(old_key)
        record["object_id"] = new_id
        record["mask_key"] = new_key
    role_binding["per_track"] = {mapping.get(k, k): v for k, v in per_track.items()}
    patient_track = role_binding.get("patient_track")
    if patient_track in mapping:
        role_binding["patient_track"] = mapping[patient_track]
    return {
        "applied": True,
        "max_disp_px": maximum_displacement_px,
        "renamed": renamed,
    }


def _run_physion_pp_friction_platform_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    prompt_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="friction-platform tracking",
    )
    artifact_identity = _tracking_shared_module("artifact_identity")
    mask_filter = _tracking_shared_module("sam3_mask_filter")
    concept_id = artifact_identity.require_string("concept_id")
    minimum_area_px = mask_filter.require_integer("minimum_area_px")
    dedup_iou = mask_filter.require_number("f0_dedup_iou")
    prompts, gdino_summary = _physion_pp_friction_platform_gdino_prompts(
        video=video,
        scenario_name=scenario_name,
        prompt_frame_index=prompt_frame_index,
        module_profile=module_profile,
    )
    frame0 = _read_video_frame(video, prompt_frame_index)
    session_id = model.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )["session_id"]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    def _probe_visual_prompt(obj_id: int, points: list, labels: list):
        response = model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "obj_id": obj_id,
                "points": points,
                "point_labels": labels,
            }
        )
        mask = _masks_from_response(response).get(obj_id)
        mask = _mask_for_image(mask, frame0) if mask is not None else None
        model.handle_request(
            {
                "type": "remove_object",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "obj_id": obj_id,
            }
        )
        return mask, int(mask.sum()) if mask is not None else 0

    def _propagate_group(group_session_id: str, group: list[dict[str, Any]]) -> None:
        object_id_by_local: dict[int, str] = {}
        concept_by_local: dict[int, str] = {}
        prompt_by_local: dict[int, str] = {}
        expected_by_local: dict[int, list[str]] = {}
        for local_id, k in enumerate(group, start=1):
            model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": group_session_id,
                    "frame_index": prompt_frame_index,
                    "obj_id": local_id,
                    "points": k["points"],
                    "point_labels": k["point_labels"],
                }
            )
            object_id_by_local[local_id] = k["track_id"]
            concept_by_local[local_id] = concept_id
            prompt_by_local[local_id] = f"gdino:{k['class']}"
            expected_by_local[local_id] = []
        stream_request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": group_session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": prompt_frame_index,
        }
        if max_frame_num_to_track is not None:
            stream_request["max_frame_num_to_track"] = int(max_frame_num_to_track)
        seen_frame_indices: set[int] = set()
        for response in model.handle_stream_request(stream_request):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices:
                continue
            seen_frame_indices.add(frame_index)
            _append_records_from_response(
                response=response,
                object_id_by_sam_obj_id=object_id_by_local,
                concept_id_by_sam_obj_id=concept_by_local,
                prompt_by_sam_obj_id=prompt_by_local,
                expected_object_ids_by_sam_obj_id=expected_by_local,
                frame_index=frame_index,
                all_records=all_records,
                all_arrays=all_arrays,
            )

    red_kept: list[dict[str, Any]] = []
    yellow_kept: list[dict[str, Any]] = []
    structure_kept: list[dict[str, Any]] = []
    try:
        for prompt in prompts:
            mask, area = _probe_visual_prompt(
                prompt["sam_obj_id"], prompt["points"], prompt["point_labels"]
            )
            if area < minimum_area_px:
                dropped.append({**prompt, "dropped_as": "empty", "prompt_mask_area": area})
                continue
            # Red/yellow are priority role prompts. Only structure candidates may
            # suppress one another at the SAM anchor, preventing a broad structure
            # mask from deleting a priority role.
            duplicate_of = None
            if prompt["prompt_group"] == "structure":
                duplicate_of = next(
                    (
                        k["sam_obj_id"]
                        for k in kept
                        if k["prompt_group"] == "structure"
                        and _pp_mask_iou(mask, k["_prompt_mask"])
                        >= dedup_iou
                    ),
                    None,
                )
            if duplicate_of is not None:
                dropped.append(
                    {**prompt, "dropped_as": str(duplicate_of), "prompt_mask_area": area}
                )
            else:
                kept.append({**prompt, "_prompt_mask": mask, "prompt_mask_area": area})

        for global_id, k in enumerate(kept, start=1):
            k["probe_sam_obj_id"] = k["sam_obj_id"]
            k["sam_obj_id"] = global_id
            k["track_id"] = f"anonymous_track_{global_id - 1:03d}"
        red_kept = [k for k in kept if k["prompt_group"] == "red"]
        yellow_kept = [k for k in kept if k["prompt_group"] == "yellow"]
        structure_kept = [k for k in kept if k["prompt_group"] == "structure"]
        # Reuse the anchor session for the solo-red propagation stream.
        if red_kept:
            _propagate_group(session_id, red_kept)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})

    # Yellow receives its own session, then all box-prompted structures share one
    # session. This keeps both priority role masks free from SAM3's joint-session
    # non-overlap competition while retaining efficient structure propagation.
    for group in (yellow_kept, structure_kept):
        if not group:
            continue
        group_session_id = model.handle_request(
            {"type": "start_session", "resource_path": str(video)}
        )["session_id"]
        try:
            _propagate_group(group_session_id, group)
        finally:
            model.handle_request({"type": "close_session", "session_id": group_session_id})

    # Explicit visible-pixel priority: red > yellow > structures. Then remove only
    # tiny disconnected structure islands before temporal same-object merging.
    carved_record_count = _pp_priority_carve(
        red=red_kept,
        yellow=yellow_kept,
        all_records=all_records,
        all_arrays=all_arrays,
    )
    structure_track_ids = {k["track_id"] for k in structure_kept}
    fragment_cleanup = _apply_physion_pp_structure_fragment_cleanup(
        structure_track_ids=structure_track_ids,
        all_records=all_records,
        all_arrays=all_arrays,
        module_profile=module_profile,
    )

    score_by_object_id = {k["track_id"]: float(k["score"]) for k in kept}
    merge_summary = _apply_physion_pp_same_object_merge(
        all_records=all_records,
        all_arrays=all_arrays,
        score_by_object_id=score_by_object_id,
        evidence_kinds=frozenset(
            module_profile.module("postprocess").require_strings(
                "same_object_merge_evidence"
            )
        ),
        eligible_object_ids=structure_track_ids,
    )
    role_binding = _bind_physion_pp_roles(
        video=video, all_records=all_records, all_arrays=all_arrays
    )
    static_prefix = _apply_physion_pp_static_prefix(
        role_binding=role_binding, all_records=all_records, all_arrays=all_arrays
    )
    static_rename = {item["from"]: item["to"] for item in static_prefix.get("renamed", [])}

    payload_prompts = []
    for k in kept:
        payload_prompts.append(
            {
                "object_id": static_rename.get(k["track_id"], k["track_id"]),
                "concept_id": concept_id,
                "sam_obj_id": k["sam_obj_id"],
                "prompt": f"gdino:{k['class']}",
                "gdino_class": k["class"],
                "gdino_score": k["score"],
                "gdino_bbox_xyxy": k["bbox_xyxy"],
                "points": k["points"],
                "point_labels": k["point_labels"],
                "prompt_mode": k.get("prompt_mode", "point"),
                "prompt_group": k["prompt_group"],
                "role": k["role"],
                "detector_round": k.get("detector_round"),
                "detector_rounds": k.get("detector_rounds"),
                "prompt_mask_area": k["prompt_mask_area"],
            }
        )
    payload_prompt_responses = [
        {
            "object_id": static_rename.get(k["track_id"], k["track_id"]),
            "concept_id": concept_id,
            "sam_object_id": k["sam_obj_id"],
            "prompt": f"gdino:{k['class']}",
            "frame_index": int(prompt_frame_index),
            "mask_count": 1,
            "prompt_index": k["sam_obj_id"],
            "expected_object_ids": [],
        }
        for k in kept
    ]
    tracking_summary = {
        "mode": "tracking.gdino_role_box_point",
        "scenario": scenario_name,
        "gdino": gdino_summary,
        "prompt_dedup": {
            "f0_dedup_iou": dedup_iou,
            "f0_min_area_px": minimum_area_px,
            "kept_count": len(kept),
            "dropped": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in dropped
            ],
        },
        "solo_red": {
            "red_tracks": sorted(k["track_id"] for k in red_kept),
            "prompt_mode": "box",
        },
        "solo_yellow": {
            "yellow_tracks": sorted(k["track_id"] for k in yellow_kept),
            "prompt_mode": "point",
        },
        "structures": {
            "tracks": sorted(structure_track_ids),
            "prompt_mode": "box",
        },
        "priority_carve": {
            "order": ["red", "yellow", "structures"],
            "carved_record_count": carved_record_count,
        },
        "structure_fragment_cleanup": fragment_cleanup,
        "same_object_merge": merge_summary,
        "role_binding": role_binding,
        "static_prefix": static_prefix,
    }
    return payload_prompts, payload_prompt_responses, tracking_summary


def _run_physion_pp_bouncy_wall_single_segment_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    prompt_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Track a bouncy-wall clip as one segment when no wipe is found."""
    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="bouncy-wall fallback",
    )
    artifact_identity = _tracking_shared_module("artifact_identity")
    mask_filter = _tracking_shared_module("sam3_mask_filter")
    same_object_merge = _tracking_shared_module("same_object_merge")
    sam3_priority = _tracking_shared_module("sam3_priority")
    concept_id = artifact_identity.require_string("concept_id")
    minimum_area_px = mask_filter.require_integer("minimum_area_px")
    dedup_iou = mask_filter.require_number("f0_dedup_iou")
    prompts, gdino_summary = _physion_pp_bouncy_wall_gdino_prompts(
        video=video,
        scenario_name=scenario_name,
        prompt_frame_index=prompt_frame_index,
        module_profile=module_profile,
    )
    frame0 = _read_video_frame(video, prompt_frame_index)
    session_id = model.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )["session_id"]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    def _probe_point_prompt(obj_id: int, points: list, labels: list):
        response = model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "obj_id": obj_id,
                "points": points,
                "point_labels": labels,
            }
        )
        mask = _masks_from_response(response).get(obj_id)
        mask = _mask_for_image(mask, frame0) if mask is not None else None
        model.handle_request(
            {
                "type": "remove_object",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "obj_id": obj_id,
            }
        )
        return mask, int(mask.sum()) if mask is not None else 0

    def _propagate_group(group_session_id: str, group: list[dict[str, Any]]) -> None:
        object_id_by_local: dict[int, str] = {}
        concept_by_local: dict[int, str] = {}
        prompt_by_local: dict[int, str] = {}
        expected_by_local: dict[int, list[str]] = {}
        for local_id, item in enumerate(group, start=1):
            model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": group_session_id,
                    "frame_index": prompt_frame_index,
                    "obj_id": local_id,
                    "points": item["points"],
                    "point_labels": item["point_labels"],
                }
            )
            object_id_by_local[local_id] = item["track_id"]
            concept_by_local[local_id] = concept_id
            prompt_by_local[local_id] = f"gdino:{item['class']}"
            expected_by_local[local_id] = []
        stream_request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": group_session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": prompt_frame_index,
        }
        if max_frame_num_to_track is not None:
            stream_request["max_frame_num_to_track"] = int(max_frame_num_to_track)
        seen_frame_indices: set[int] = set()
        for response in model.handle_stream_request(stream_request):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices:
                continue
            seen_frame_indices.add(frame_index)
            _append_records_from_response(
                response=response,
                object_id_by_sam_obj_id=object_id_by_local,
                concept_id_by_sam_obj_id=concept_by_local,
                prompt_by_sam_obj_id=prompt_by_local,
                expected_object_ids_by_sam_obj_id=expected_by_local,
                frame_index=frame_index,
                all_records=all_records,
                all_arrays=all_arrays,
            )

    red_kept: list[dict[str, Any]] = []
    other_kept: list[dict[str, Any]] = []
    try:
        for prompt in prompts:
            mask, area = _probe_point_prompt(
                prompt["sam_obj_id"], prompt["points"], prompt["point_labels"]
            )
            if area < minimum_area_px:
                dropped.append({**prompt, "dropped_as": "empty", "prompt_mask_area": area})
                continue
            duplicate_of = next(
                (
                    item["sam_obj_id"]
                    for item in kept
                    if _pp_mask_iou(mask, item["_prompt_mask"]) >= dedup_iou
                ),
                None,
            )
            if duplicate_of is not None:
                dropped.append(
                    {**prompt, "dropped_as": str(duplicate_of), "prompt_mask_area": area}
                )
            else:
                kept.append({**prompt, "_prompt_mask": mask, "prompt_mask_area": area})

        for global_id, item in enumerate(kept, start=1):
            item["probe_sam_obj_id"] = item["sam_obj_id"]
            item["sam_obj_id"] = global_id
            item["track_id"] = f"anonymous_track_{global_id - 1:03d}"
        red_kept = [
            item
            for item in kept
            if (
                sam3_priority.require_string("solo_red_class_keyword")
                in str(item["class"]).lower()
            )
        ]
        other_kept = [item for item in kept if item not in red_kept]
        if red_kept:
            _propagate_group(session_id, red_kept)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})

    if other_kept:
        joint_session_id = model.handle_request(
            {"type": "start_session", "resource_path": str(video)}
        )["session_id"]
        try:
            _propagate_group(joint_session_id, other_kept)
        finally:
            model.handle_request({"type": "close_session", "session_id": joint_session_id})

    carved_record_count = _pp_priority_carve(
        red=red_kept,
        yellow=[],
        all_records=all_records,
        all_arrays=all_arrays,
    )
    score_by_object_id = {item["track_id"]: float(item["score"]) for item in kept}
    merge_summary = _apply_physion_pp_same_object_merge(
        all_records=all_records,
        all_arrays=all_arrays,
        score_by_object_id=score_by_object_id,
        evidence_kinds=frozenset(
            same_object_merge.require_strings("default_evidence")
        ),
    )
    role_binding = _bind_physion_pp_roles(
        video=video, all_records=all_records, all_arrays=all_arrays
    )
    static_prefix = _apply_physion_pp_static_prefix(
        role_binding=role_binding, all_records=all_records, all_arrays=all_arrays
    )
    static_rename = {
        item["from"]: item["to"] for item in static_prefix.get("renamed", [])
    }

    payload_prompts = [
        {
            "object_id": static_rename.get(item["track_id"], item["track_id"]),
            "concept_id": concept_id,
            "sam_obj_id": item["sam_obj_id"],
            "prompt": f"gdino:{item['class']}",
            "gdino_class": item["class"],
            "gdino_score": item["score"],
            "gdino_bbox_xyxy": item["bbox_xyxy"],
            "points": item["points"],
            "point_labels": item["point_labels"],
            "prompt_mode": item.get("prompt_mode", "point"),
            "prompt_mask_area": item["prompt_mask_area"],
        }
        for item in kept
    ]
    payload_prompt_responses = [
        {
            "object_id": static_rename.get(item["track_id"], item["track_id"]),
            "concept_id": concept_id,
            "sam_object_id": item["sam_obj_id"],
            "prompt": f"gdino:{item['class']}",
            "frame_index": int(prompt_frame_index),
            "mask_count": 1,
            "prompt_index": item["sam_obj_id"],
            "expected_object_ids": [],
        }
        for item in kept
    ]
    tracking_summary = {
        "mode": "bouncy_wall_gdino_point_probe_solo_red_fallback",
        "scenario": scenario_name,
        "gdino": gdino_summary,
        "prompt_dedup": {
            "f0_dedup_iou": dedup_iou,
            "f0_min_area_px": minimum_area_px,
            "kept_count": len(kept),
            "dropped": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in dropped
            ],
        },
        "solo_red": {
            "red_tracks": sorted(item["track_id"] for item in red_kept),
            "other_track_count": len(other_kept),
            "carved_record_count": carved_record_count,
        },
        "same_object_merge": merge_summary,
        "role_binding": role_binding,
        "static_prefix": static_prefix,
    }
    return payload_prompts, payload_prompt_responses, tracking_summary


def _pp_mask_contain(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` that lies inside `outer` (|inner & outer| / |inner|)."""
    area = int(inner.sum())
    return int((inner & outer).sum()) / area if area else 0.0


def _bp_family(class_name: str, is_agent: bool) -> str:
    if is_agent:
        return "agent"
    lowered = str(class_name).lower()
    for key in ("yellow", "green", "teal"):
        if key in lowered:
            return key
    return "mat" if "mat" in lowered else "structure"


def _bp_is_agent_box(
    det: dict[str, Any],
    *,
    module_profile: TrackingModuleProfile,
) -> bool:
    x1, y1, x2, y2 = det["bbox_xyxy"]
    maximum_side_px = module_profile.module("detector").require_integer(
        "agent_max_side_px"
    )
    solo_red_class_keyword = _tracking_shared_module(
        "sam3_priority"
    ).require_string("solo_red_class_keyword")
    return (
        solo_red_class_keyword in str(det["class"]).lower()
        and (x2 - x1) <= maximum_side_px
        and (y2 - y1) <= maximum_side_px
    )


def _bp_box_area(det: dict[str, Any]) -> int:
    x1, y1, x2, y2 = det["bbox_xyxy"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bp_xyxy_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _bp_xyxy_inter(a: list[int], b: list[int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bp_enclosing_box(boxes: list[list[int]]) -> list[int] | None:
    """Min axis-aligned box enclosing every box (bottom-left -> top-right)."""
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _bp_box_contained_either(box: list[int], ref: list[int], tau: float) -> bool:
    """True if `box` contains `ref` OR is contained by `ref` at >= tau (fraction of the
    smaller lying inside the larger)."""
    inter = _bp_xyxy_inter(box, ref)
    ab, ar = _bp_xyxy_area(box), _bp_xyxy_area(ref)
    if ab == 0 or ar == 0:
        return False
    return inter / ar >= tau or inter / ab >= tau


def _physion_pp_bouncy_prompts(
    *,
    video: Path,
    scenario_name: str,
    anchor_frame_index: int,
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two-stage per-part GDINO seeding for bouncy_platform. Targeted queries at the
    mid-video anchor (agent has entered by ~f60), each with its own selection rule:
      * red agent      -- 'small red block' @0.15, size-gated top-1  -> POINT seed
                          (falls back to the nearest fallback frame if absent at anchor)
      * yellow patient -- full mainline vocab @0.25, smallest 'yellow'-class box -> BOX
      * big block (2 cubes) -- 'large dark box' @0.15, min-enclosing box -> BOX
      * guardrail/mat statics -- mainline vocab minus red/yellow words @0.25 + cleanup,
        minus any box that contains OR is contained by the agent/patient/bigblock box.
    SAM3 anchor filtering and containment dedup run after this seeding stage."""
    from scripts.world_model.run_gdino_boxes import (
        build_query,
        cleanup_boxes,
        detect_boxes,
        load_gdino,
    )

    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="bouncy-platform GDINO frontend",
    )
    detector = module_profile.module("detector")
    sam3_seed = module_profile.module("sam3_seed")
    cleanup = _tracking_shared_module("gdino_cleanup")
    processor, gdino_model, device = load_gdino()
    full_vocab = list(detector.require_strings("vocabulary"))
    static_vocab = [
        w for w in full_vocab
        if not any(
            drop in str(w).lower()
            for drop in detector.require_strings("static_drop_words")
        )
    ]
    frame_count = _video_metadata(video)["frame_count"]

    def frame_at(frame_index: int):
        if not (0 <= frame_index < frame_count):
            return None
        return _read_video_frame(video, frame_index)

    anchor_image = frame_at(anchor_frame_index)
    if anchor_image is None:
        return [], {"anchor_frame_index": int(anchor_frame_index), "reason": "anchor_out_of_range"}
    height, width = anchor_image.shape[:2]

    def detect(image, phrases, conf):
        return detect_boxes(processor, gdino_model, image, build_query(phrases),
                            confidence=conf, device=device)

    def clip_box(box):
        x1, y1, x2, y2 = box
        return [max(0, int(x1)), max(0, int(y1)), min(width - 1, int(x2)), min(height - 1, int(y2))]

    # --- red agent: dedicated query, size-gated top-1 (POINT); fallback frames ---
    def best_agent(dets):
        gated = [
            d for d in dets if _bp_is_agent_box(d, module_profile=module_profile)
        ]
        return max(gated, key=lambda d: d["score"]) if gated else None
    agent_query = detector.require_string("agent_query")
    agent_confidence = detector.require_number("agent_confidence")
    agent_det = best_agent(detect(anchor_image, [agent_query], agent_confidence))
    agent_frame = anchor_frame_index
    if agent_det is None:
        for fallback in sam3_seed.require_integers("agent_fallback_frames"):
            fimg = frame_at(fallback)
            if fimg is None:
                continue
            cand = best_agent(detect(fimg, [agent_query], agent_confidence))
            if cand is not None:
                agent_det, agent_frame = cand, fallback
                break

    # --- yellow patient: full vocab @0.25, smallest 'yellow'-class box (BOX) ---
    default_confidence = cleanup.require_number("confidence")
    cleanup_kwargs = {
        "area_cap_fraction": cleanup.require_number("area_cap_fraction"),
        "nms_iou": cleanup.require_number("same_label_nms_iou"),
        "edge_margin_px": cleanup.require_integer("edge_margin_px"),
        "edge_sides_to_drop": cleanup.require_integer("edge_sides_to_drop"),
    }
    full_dets = cleanup_boxes(
        detect(anchor_image, full_vocab, default_confidence),
        image_width=width,
        image_height=height,
        **cleanup_kwargs,
    )
    yellows = [d for d in full_dets if "yellow" in str(d["class"]).lower()]
    patient_det = min(yellows, key=_bp_box_area) if yellows else None

    # --- big block: 'large dark box' @0.15, min-enclosing box of all dets (BOX) ---
    big_block_query = detector.require_string("big_block_query")
    bigblock_dets = detect(
        anchor_image,
        [big_block_query],
        detector.require_number("big_block_confidence"),
    )
    bigblock_box = _bp_enclosing_box([d["bbox_xyxy"] for d in bigblock_dets])

    # --- guardrail/mat statics: reduced vocab @0.25 + cleanup, minus containment w/ refs ---
    static_dets = cleanup_boxes(
        detect(anchor_image, static_vocab, default_confidence),
        image_width=width,
        image_height=height,
        **cleanup_kwargs,
    )
    refs = [b for b in (
        agent_det["bbox_xyxy"] if agent_det else None,
        patient_det["bbox_xyxy"] if patient_det else None,
        bigblock_box,
    ) if b]
    static_keep = [
        d for d in static_dets
        if not any(_bp_box_contained_either(
            d["bbox_xyxy"], ref, detector.require_number("reference_containment")
        )
                   for ref in refs)
    ]

    # --- assemble prompts (agent POINT, everything else two-corner BOX) ---
    prompts: list[dict[str, Any]] = []

    def add_prompt(box, class_name, score, frame_index, *, is_agent=False, is_patient=False, is_bigblock=False):
        x1, y1, x2, y2 = clip_box(box)
        if is_agent:
            points, labels, mode = [[(x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height]], [1], "point"
        else:
            points, labels, mode = [[x1 / width, y1 / height], [x2 / width, y2 / height]], [2, 3], "box"
        prompts.append({
            "sam_obj_id": len(prompts) + 1,
            "class": str(class_name),
            "score": float(score),
            "bbox_xyxy": [x1, y1, x2, y2],
            "points": points,
            "point_labels": labels,
            "prompt_mode": mode,
            "prompt_frame": int(frame_index),
            "_is_agent": bool(is_agent),
            "_is_patient": bool(is_patient),
            "_is_bigblock": bool(is_bigblock),
        })

    if agent_det is not None:
        add_prompt(agent_det["bbox_xyxy"], agent_det["class"], agent_det["score"], agent_frame, is_agent=True)
    if patient_det is not None:
        add_prompt(patient_det["bbox_xyxy"], patient_det["class"], patient_det["score"], anchor_frame_index, is_patient=True)
    if bigblock_box is not None:
        bb_score = max((d["score"] for d in bigblock_dets), default=0.0)
        add_prompt(
            bigblock_box,
            big_block_query,
            bb_score,
            anchor_frame_index,
            is_bigblock=True,
        )
    for det in static_keep:
        add_prompt(det["bbox_xyxy"], det["class"], det["score"], anchor_frame_index)

    summary = {
        "anchor_frame_index": int(anchor_frame_index),
        "agent_frame": int(agent_frame),
        "image_size": [int(width), int(height)],
        "static_vocab": static_vocab,
        "cleaned_boxes": full_dets,
        "bigblock_box": bigblock_box,
        "n_prompts": len(prompts),
        "n_static_kept": len(static_keep),
    }
    return prompts, summary


def _bp_dedup(
    cands: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    *,
    module_profile: TrackingModuleProfile,
):
    """Mask containment keep-big dedup on the anchor probe masks. The agent (POINT) and
    the yellow patient (pre-selected at seeding via `_is_patient`) are exempt and always
    kept; every other candidate (big block + guardrail/mat statics) is processed
    largest-first and dropped if its mask is >= TAU contained inside an already-kept
    non-agent mask (keep the bigger track). Returns (kept, patient-or-None)."""
    tau = module_profile.module("sam3_seed").require_number("mask_containment")

    def record(cand, reason):
        dropped.append(
            {**{k: v for k, v in cand.items() if not k.startswith("_")}, "dropped_as": reason,
             "prompt_mask_area": cand["_area"]}
        )

    agent = [c for c in cands if c["_is_agent"]]
    patients = [c for c in cands if c.get("_is_patient")]
    ykeep = patients[0] if patients else None
    others = [c for c in cands if (not c["_is_agent"]) and not c.get("_is_patient")]
    kept = list(agent) + ([ykeep] if ykeep is not None else [])
    for c in sorted(others, key=lambda item: -item["_area"]):
        containers = [k for k in kept if not k["_is_agent"]]
        if any(_pp_mask_contain(c["_mask"], k["_mask"]) >= tau for k in containers):
            record(c, "covered_by_other")
        else:
            kept.append(c)
    return kept, ykeep


def _bp_propagate(
    *, model: Any, session_id: str, group: list[dict[str, Any]], anchor_frame_index: int,
    max_frame_num_to_track: int | None, all_records: list[dict[str, Any]], all_arrays: dict[str, np.ndarray],
) -> None:
    concept_id = _tracking_shared_module("artifact_identity").require_string(
        "concept_id"
    )
    object_id_by_local: dict[int, str] = {}
    concept_by_local: dict[int, str] = {}
    prompt_by_local: dict[int, str] = {}
    expected_by_local: dict[int, list[str]] = {}
    for k in group:
        local_id = k["_local_id"]
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": k["prompt_frame"],
                "obj_id": local_id,
                "points": k["points"],
                "point_labels": k["point_labels"],
            }
        )
        object_id_by_local[local_id] = k["track_id"]
        concept_by_local[local_id] = concept_id
        prompt_by_local[local_id] = f"gdino:{k['class']}"
        expected_by_local[local_id] = []
    stream_request: dict[str, Any] = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "both",
        "start_frame_index": anchor_frame_index,
    }
    if max_frame_num_to_track is not None:
        stream_request["max_frame_num_to_track"] = int(max_frame_num_to_track)
    seen_frame_indices: set[int] = set()
    for response in model.handle_stream_request(stream_request):
        frame_index = int(response["frame_index"])
        if frame_index in seen_frame_indices:
            continue
        seen_frame_indices.add(frame_index)
        _append_records_from_response(
            response=response,
            object_id_by_sam_obj_id=object_id_by_local,
            concept_id_by_sam_obj_id=concept_by_local,
            prompt_by_sam_obj_id=prompt_by_local,
            expected_object_ids_by_sam_obj_id=expected_by_local,
            frame_index=frame_index,
            all_records=all_records,
            all_arrays=all_arrays,
        )


def _pp_priority_carve(
    *, red: list[dict[str, Any]], yellow: list[dict[str, Any]],
    all_records: list[dict[str, Any]], all_arrays: dict[str, np.ndarray],
) -> int:
    """agent (red) wins every pixel; the yellow mat wins over the rest. Carve the red
    union out of yellow + others, and the yellow union out of others, so delivered
    masks never overlap. Shared by friction_platform and bouncy_platform."""
    def union_by_frame(tracks):
        ids = {k["track_id"] for k in tracks}
        by_frame: dict[int, np.ndarray] = {}
        for record in all_records:
            if str(record["object_id"]) not in ids:
                continue
            mask = all_arrays.get(str(record["mask_key"]))
            if mask is not None and mask.any():
                frame = int(record["frame_index"])
                bool_mask = mask.astype(bool)
                by_frame[frame] = (by_frame[frame] | bool_mask) if frame in by_frame else bool_mask
        return by_frame, ids

    red_union, red_ids = union_by_frame(red)
    yellow_union, yellow_ids = union_by_frame(yellow)
    carved = 0
    for record in all_records:
        object_id = str(record["object_id"])
        if object_id in red_ids:
            continue
        frame = int(record["frame_index"])
        if object_id in yellow_ids:
            parts = [red_union.get(frame)]
        else:
            parts = [red_union.get(frame), yellow_union.get(frame)]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        carve = parts[0]
        for part in parts[1:]:
            carve = carve | part
        key = str(record["mask_key"])
        mask = all_arrays.get(key)
        if mask is None or not (mask.astype(bool) & carve).any():
            continue
        new_mask = (mask.astype(bool) & ~carve).astype(np.uint8)
        all_arrays[key] = new_mask
        record.update(_mask_stats(new_mask))
        carved += 1
    return carved


def _run_physion_pp_bouncy_platform_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    max_frame_num_to_track: int | None,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """bouncy_platform tracking: two-stage per-part GDINO seeding (red agent / yellow
    patient / big block / guardrail-mat statics, each a targeted query + selection rule;
    statics pruned by containment vs agent/patient/bigblock) -> SAM3 anchor probe (<20px
    drop) -> mask containment keep-big dedup -> solo-red + solo-yellow + joint-others,
    both-direction from a mid-video anchor -> agent/yellow priority carve -> cue-flash
    role binding + static-prefix rename. No same-object merge / no post cleanup."""
    artifact_identity = _tracking_shared_module("artifact_identity")
    mask_filter = _tracking_shared_module("sam3_mask_filter")
    concept_id = artifact_identity.require_string("concept_id")
    minimum_area_px = mask_filter.require_integer("minimum_area_px")
    frame_count = _video_metadata(video)["frame_count"]
    containment_tau = module_profile.module("sam3_seed").require_number(
        "mask_containment"
    )
    anchor = min(
        module_profile.module("sam3_seed").require_integer("anchor_frame"),
        max(0, frame_count - 1),
    )
    prompts, gdino_summary = _physion_pp_bouncy_prompts(
        video=video,
        scenario_name=scenario_name,
        anchor_frame_index=anchor,
        module_profile=module_profile,
    )
    anchor_image = _read_video_frame(video, anchor)

    dropped: list[dict[str, Any]] = []
    session_id = model.handle_request({"type": "start_session", "resource_path": str(video)})["session_id"]
    kept: list[dict[str, Any]] = []
    ykeep: dict[str, Any] | None = None
    red: list[dict[str, Any]] = []
    yellow: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    try:
        cands: list[dict[str, Any]] = []
        for prompt in prompts:
            response = model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt["prompt_frame"],
                    "obj_id": prompt["sam_obj_id"],
                    "points": prompt["points"],
                    "point_labels": prompt["point_labels"],
                }
            )
            mask = _masks_from_response(response).get(prompt["sam_obj_id"])
            mask = _mask_for_image(mask, anchor_image) if mask is not None else None
            model.handle_request(
                {
                    "type": "remove_object",
                    "session_id": session_id,
                    "frame_index": prompt["prompt_frame"],
                    "obj_id": prompt["sam_obj_id"],
                }
            )
            area = int(mask.sum()) if mask is not None else 0
            if area < minimum_area_px:
                dropped.append(
                    {**{k: v for k, v in prompt.items() if not k.startswith("_")}, "dropped_as": "empty",
                     "prompt_mask_area": area}
                )
                continue
            cands.append({**prompt, "_mask": mask, "_area": area, "_fam": _bp_family(prompt["class"], prompt["_is_agent"])})

        kept, ykeep = _bp_dedup(
            cands,
            dropped,
            module_profile=module_profile,
        )
        for global_id, k in enumerate(kept, start=1):
            k["_local_id"] = global_id
            k["track_id"] = f"anonymous_track_{global_id - 1:03d}"
        red = [k for k in kept if k["_is_agent"]]
        yellow = [k for k in kept if k is ykeep] if ykeep is not None else []
        other = [k for k in kept if (not k["_is_agent"]) and k is not ykeep]
        if red:
            _bp_propagate(model=model, session_id=session_id, group=red, anchor_frame_index=anchor,
                          max_frame_num_to_track=max_frame_num_to_track, all_records=all_records, all_arrays=all_arrays)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})

    for group in (yellow, other):
        if not group:
            continue
        group_session_id = model.handle_request({"type": "start_session", "resource_path": str(video)})["session_id"]
        try:
            _bp_propagate(model=model, session_id=group_session_id, group=group, anchor_frame_index=anchor,
                          max_frame_num_to_track=max_frame_num_to_track, all_records=all_records, all_arrays=all_arrays)
        finally:
            model.handle_request({"type": "close_session", "session_id": group_session_id})

    carved_record_count = _pp_priority_carve(red=red, yellow=yellow, all_records=all_records, all_arrays=all_arrays)
    role_binding = _bind_physion_pp_roles(video=video, all_records=all_records, all_arrays=all_arrays)
    static_prefix = _apply_physion_pp_static_prefix(
        role_binding=role_binding, all_records=all_records, all_arrays=all_arrays
    )
    static_rename = {item["from"]: item["to"] for item in static_prefix.get("renamed", [])}

    payload_prompts = []
    payload_prompt_responses = []
    for k in kept:
        object_id = static_rename.get(k["track_id"], k["track_id"])
        payload_prompts.append(
            {
                "object_id": object_id,
                "concept_id": concept_id,
                "sam_obj_id": k["_local_id"],
                "prompt": f"gdino:{k['class']}",
                "gdino_class": k["class"],
                "gdino_score": k["score"],
                "gdino_bbox_xyxy": k["bbox_xyxy"],
                "points": k["points"],
                "point_labels": k["point_labels"],
                "prompt_mode": k.get("prompt_mode", "point"),
                "prompt_mask_area": k["_area"],
            }
        )
        payload_prompt_responses.append(
            {
                "object_id": object_id,
                "concept_id": concept_id,
                "sam_object_id": k["_local_id"],
                "prompt": f"gdino:{k['class']}",
                "frame_index": int(k["prompt_frame"]),
                "mask_count": 1,
                "prompt_index": k["_local_id"],
                "expected_object_ids": [],
            }
        )
    tracking_summary = {
        "mode": "tracking.gdino_per_part_seeds",
        "scenario": scenario_name,
        "anchor_frame_index": int(anchor),
        "gdino": gdino_summary,
        "prompt_dedup": {
            "containment_tau": containment_tau,
            "f0_min_area_px": minimum_area_px,
            "kept_count": len(kept),
            "dropped": [{key: value for key, value in item.items() if not key.startswith("_")} for item in dropped],
        },
        "solo_red": {"red_tracks": sorted(k["track_id"] for k in red)},
        "solo_yellow": {"yellow_track": (ykeep["track_id"] if ykeep is not None else None)},
        "priority_carve": {"carved_record_count": carved_record_count},
        "role_binding": role_binding,
        "static_prefix": static_prefix,
    }
    return payload_prompts, payload_prompt_responses, tracking_summary


def _is_physion_pp_two_segment_text_scenario(scenario_name: str | None) -> bool:
    return str(scenario_name or "").strip().lower() in PHYSION_PP_TWO_SEGMENT_TEXT_SCENARIOS


def _bw_read_all_frames(
    video: Path,
    *,
    minimum_frame_count: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for two-segment tracking: {video}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < minimum_frame_count:
        raise ValueError(f"Video too short for two-segment tracking ({len(frames)} frames): {video}")
    return frames


def _bw_tracked_bound(
    col_prof: np.ndarray,
    start: int,
    step: int,
    eps: float,
    *,
    column_adjacency: int,
    stall_frames: int,
) -> int:
    """Walk from the wipe core while board columns stay column-adjacent AND keep
    moving; a frozen column range is a landed object, never the sweeping board."""
    cur = np.nonzero(col_prof[start] >= eps)[0]
    if not len(cur):
        return start
    t, last = start, start
    stall, last_moving = 0, start
    while True:
        t += step
        if t < 0 or t >= len(col_prof):
            break
        cand = np.nonzero(col_prof[t] >= eps)[0]
        if not len(cand):
            break
        lo = cur.min() - column_adjacency
        hi = cur.max() + column_adjacency
        kept = cand[(cand >= lo) & (cand <= hi)]
        if not len(kept):
            break
        if abs(int(kept.min()) - int(cur.min())) <= 1 and abs(int(kept.max()) - int(cur.max())) <= 1:
            stall += 1
            if stall >= stall_frames:
                return last_moving
        else:
            stall, last_moving = 0, t
        cur, last = kept, t
    return last


def _bw_detect_transition(
    frames: list[np.ndarray],
    *,
    rgb_difference_threshold: int,
    core_minimum_difference_fraction: float,
    column_band_top: int,
    column_occupancy: float,
    column_adjacency: int,
    stall_frames: int,
    background_1_frames: int,
    background_2_frames: int,
) -> dict[str, Any] | None:
    """Return {'a': first wipe frame, 'b': last wipe frame, ...} or None."""
    stack = np.stack(frames).astype(np.int16)
    t_last = len(frames) - 1
    diff_f0 = np.abs(stack - stack[0]).max(axis=3) > rgb_difference_threshold
    diff_ft = np.abs(stack - stack[t_last]).max(axis=3) > rgb_difference_threshold
    core_signal = np.minimum(diff_f0.mean(axis=(1, 2)), diff_ft.mean(axis=(1, 2)))
    core = np.nonzero(core_signal > core_minimum_difference_fraction)[0]
    if not len(core):
        return None
    bg1 = np.median(stack[:background_1_frames], axis=0).astype(np.int16)
    bg2 = np.median(stack[-background_2_frames:], axis=0).astype(np.int16)
    band = slice(column_band_top, None)
    prof1 = (np.abs(stack - bg1).max(axis=3) > rgb_difference_threshold)[:, band, :].mean(axis=1)
    prof2 = (np.abs(stack - bg2).max(axis=3) > rgb_difference_threshold)[:, band, :].mean(axis=1)
    a = _bw_tracked_bound(
        prof1,
        int(core[0]),
        -1,
        column_occupancy,
        column_adjacency=column_adjacency,
        stall_frames=stall_frames,
    )
    b = _bw_tracked_bound(
        prof2,
        int(core[-1]),
        +1,
        column_occupancy,
        column_adjacency=column_adjacency,
        stall_frames=stall_frames,
    )
    if not (0 < a <= b < t_last):
        return None
    return {
        "a": int(a), "b": int(b),
        "core_start": int(core[0]), "core_end": int(core[-1]),
        "seg1": [0, int(a) - 1], "seg2": [int(b) + 1, t_last],
    }


def _bw_window_box_filter(
    prompts: list[dict[str, Any]],
    frame0: np.ndarray,
    frame_last: np.ndarray,
    *,
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    temporal = module_profile.module("temporal_partition")
    postprocess = module_profile.module("postprocess")
    cross = (
        np.abs(frame_last.astype(np.int16) - frame0.astype(np.int16)).max(axis=2)
        > temporal.require_integer("rgb_difference_threshold")
    )
    height, width = frame0.shape[:2]
    kept, dropped = [], []
    for prompt in prompts:
        x1, y1, x2, y2 = prompt["bbox_xyxy"]
        x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        frac = float(cross[y1:y2, x1:x2].mean()) if area else 0.0
        info = {**prompt, "cross_diff_frac": round(frac, 3), "box_area_px": int(area)}
        if (
            frac < postprocess.require_number("window_cross_static")
            and area < postprocess.require_integer("window_max_area_px")
        ):
            dropped.append(info)
        else:
            kept.append(info)
    return kept, dropped


def _bw_probe_and_propagate_segment(
    *,
    model: Any,
    video: Path,
    anchor_frame_index: int,
    anchor_image: np.ndarray,
    prompts: list[dict[str, Any]],
    direction: str,
    keep_lo: int,
    keep_hi: int,
    seg_tag: str,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate anchor masks and propagate one segment."""
    artifact_identity = _tracking_shared_module("artifact_identity")
    mask_filter = _tracking_shared_module("sam3_mask_filter")
    concept_id = artifact_identity.require_string("concept_id")
    minimum_area_px = mask_filter.require_integer("minimum_area_px")
    dedup_iou = mask_filter.require_number("f0_dedup_iou")
    session_id = model.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )["session_id"]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    def _propagate_group(group_session_id: str, group: list[dict[str, Any]]) -> None:
        object_id_by_local: dict[int, str] = {}
        concept_by_local: dict[int, str] = {}
        prompt_by_local: dict[int, str] = {}
        expected_by_local: dict[int, list[str]] = {}
        for local_id, k in enumerate(group, start=1):
            model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": group_session_id,
                    "frame_index": anchor_frame_index,
                    "obj_id": local_id,
                    "points": k["points"],
                    "point_labels": k["point_labels"],
                }
            )
            k["_local_id"] = local_id
            object_id_by_local[local_id] = k["track_id"]
            concept_by_local[local_id] = concept_id
            prompt_by_local[local_id] = f"gdino:{k['class']}"
            expected_by_local[local_id] = []
        seen_frame_indices: set[int] = set()
        for response in model.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": group_session_id,
                "propagation_direction": direction,
                "start_frame_index": anchor_frame_index,
                "max_frame_num_to_track": int(keep_hi - keep_lo + 3),
            }
        ):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices or not (keep_lo <= frame_index <= keep_hi):
                continue
            seen_frame_indices.add(frame_index)
            _append_records_from_response(
                response=response,
                object_id_by_sam_obj_id=object_id_by_local,
                concept_id_by_sam_obj_id=concept_by_local,
                prompt_by_sam_obj_id=prompt_by_local,
                expected_object_ids_by_sam_obj_id=expected_by_local,
                frame_index=frame_index,
                all_records=all_records,
                all_arrays=all_arrays,
            )
        # backward propagation never emits the anchor frame; the probe mask IS the
        # anchor-frame mask, so seed it for any track missing that record
        have_anchor = {
            str(r["object_id"]) for r in all_records if int(r["frame_index"]) == anchor_frame_index
        }
        for k in group:
            if k["track_id"] in have_anchor:
                continue
            mask_u8 = k["_prompt_mask"].astype(np.uint8)
            mask_key = f"{k['track_id']}__frame_{anchor_frame_index:05d}__sam_obj_{k['_local_id']}"
            all_arrays[mask_key] = mask_u8
            all_records.append(
                {
                    "object_id": k["track_id"],
                    "concept_id": concept_id,
                    "prompt": f"gdino:{k['class']}",
                    "expected_object_ids": [],
                    "frame_index": int(anchor_frame_index),
                    "sam_object_id": int(k["_local_id"]),
                    "mask_key": mask_key,
                    "score": float(k["score"]),
                    **_mask_stats(mask_u8),
                }
            )

    try:
        for probe_id, prompt in enumerate(prompts, start=1):
            response = model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor_frame_index,
                    "obj_id": probe_id,
                    "points": prompt["points"],
                    "point_labels": prompt["point_labels"],
                }
            )
            mask = _masks_from_response(response).get(probe_id)
            mask = _mask_for_image(mask, anchor_image) if mask is not None else None
            model.handle_request(
                {
                    "type": "remove_object",
                    "session_id": session_id,
                    "frame_index": anchor_frame_index,
                    "obj_id": probe_id,
                }
            )
            area = int(mask.sum()) if mask is not None else 0
            if area < minimum_area_px:
                dropped.append({**prompt, "dropped_as": "empty", "prompt_mask_area": area})
                continue
            duplicate_of = next(
                (
                    k["track_id"]
                    for k in kept
                    if _pp_mask_iou(mask, k["_prompt_mask"]) >= dedup_iou
                ),
                None,
            )
            if duplicate_of is not None:
                dropped.append({**prompt, "dropped_as": duplicate_of, "prompt_mask_area": area})
                continue
            kept.append(
                {
                    **prompt,
                    "_prompt_mask": mask,
                    "prompt_mask_area": area,
                    "track_id": f"{seg_tag}_track_{len(kept):03d}",
                    "solo_small": any(
                        word in str(prompt["class"])
                        for word in module_profile.module(
                            "sam3_seed"
                        ).require_strings("solo_class_words")
                    ),
                }
            )
        solo = [k for k in kept if k["solo_small"]]
        joint = [k for k in kept if not k["solo_small"]]
        if solo:
            _propagate_group(session_id, solo)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})
    if joint:
        joint_session_id = model.handle_request(
            {"type": "start_session", "resource_path": str(video)}
        )["session_id"]
        try:
            _propagate_group(joint_session_id, joint)
        finally:
            model.handle_request({"type": "close_session", "session_id": joint_session_id})

    # solo priority within the segment: the small mover wins pixels, fixture masks
    # are carved by the per-frame solo union (same rationale as friction solo-red)
    solo_ids = {k["track_id"] for k in solo}
    solo_union: dict[int, np.ndarray] = {}
    for record in all_records:
        if str(record["object_id"]) in solo_ids:
            mask = all_arrays.get(str(record["mask_key"]))
            if mask is not None and mask.any():
                frame = int(record["frame_index"])
                bool_mask = mask.astype(bool)
                solo_union[frame] = (
                    (solo_union[frame] | bool_mask) if frame in solo_union else bool_mask
                )
    joint_ids = {k["track_id"] for k in joint}
    for record in all_records:
        if str(record["object_id"]) not in joint_ids:
            continue
        union = solo_union.get(int(record["frame_index"]))
        if union is None:
            continue
        key = str(record["mask_key"])
        mask = all_arrays.get(key)
        if mask is None or not (mask.astype(bool) & union).any():
            continue
        carved = (mask.astype(bool) & ~union).astype(np.uint8)
        all_arrays[key] = carved
        record.update(_mask_stats(carved))
    return kept, dropped


def _pp_vlm_pick_seg1_agent(
    *,
    frames: list[np.ndarray],
    transition: dict[str, Any],
    t_last: int,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    seg1_masks: dict[str, np.ndarray],
    seg2_agent_tid: str,
    anchors2: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Ask one focused VLM question to bind the seg2 agent to one of two seg1 tracks."""
    tids = sorted(seg1_masks)
    if len(tids) != 2:
        raise ValueError(
            f"TRK-004 VLM identity requires exactly two seg1 candidates, got {tids}"
        )
    transition_b = transition.get("b")
    ref_frame_index = (
        min(int(transition_b) + 2, t_last)
        if transition_b is not None
        else t_last
    )
    ref_mask = None
    for record in all_records:
        if (
            str(record["object_id"]) == seg2_agent_tid
            and int(record["frame_index"]) == ref_frame_index
        ):
            ref_mask = all_arrays.get(str(record["mask_key"]))
            break
    if ref_mask is None:
        ref_frame_index, ref_mask = t_last, anchors2.get(seg2_agent_tid)
    if ref_mask is None:
        raise ValueError(
            f"TRK-004 VLM identity is missing the seg2 agent mask: {seg2_agent_tid}"
        )
    decision = select_same_object_ab(
        config=build_model_config(),
        reference_image=frames[ref_frame_index],
        reference_mask=np.asarray(ref_mask).astype(bool),
        candidate_images=[frames[0], frames[0]],
        candidate_masks=[
            seg1_masks[tids[0]].astype(bool),
            seg1_masks[tids[1]].astype(bool),
        ],
        candidate_ids=tids,
        request_context={
            "stage": "cross_segment_identity",
            "decision_id": CROSS_SEGMENT_IDENTITY_DECISION_ID,
        },
    )
    decision["reference_frame_index"] = ref_frame_index
    decision["candidate_frame_indices"] = {"A": 0, "B": 0}
    return decision


def _pp_mask_median_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image[np.asarray(mask).astype(bool)]
    if not len(pixels):
        raise ValueError("TRK-004 candidate reduction received an empty mask")
    return np.median(pixels.astype(np.float64), axis=0)


def _pp_reduce_seg1_identity_candidates(
    *,
    seg1_image: np.ndarray,
    seg2_image: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    seg2_agent_mask: np.ndarray,
    seg2_patient_mask: np.ndarray,
) -> dict[str, Any]:
    """Reduce extra seg1 tracks to one agent/patient pair before VLM A/B identity.

    Cue masks already identify the two seg2 roles.  The reduction jointly assigns
    two distinct seg1 candidates to those roles by minimum total median-colour L1
    cost; the focused VLM still makes the final agent identity decision.
    """

    candidate_ids = sorted(candidate_masks)
    if len(candidate_ids) <= 2:
        raise ValueError(
            "TRK-004 candidate reduction requires more than two seg1 candidates"
        )

    candidate_colors = {
        track_id: _pp_mask_median_color(seg1_image, candidate_masks[track_id])
        for track_id in candidate_ids
    }
    role_colors = {
        "agent": _pp_mask_median_color(seg2_image, seg2_agent_mask),
        "patient": _pp_mask_median_color(seg2_image, seg2_patient_mask),
    }
    assignments = []
    for agent_track in candidate_ids:
        for patient_track in candidate_ids:
            if patient_track == agent_track:
                continue
            agent_cost = float(
                np.abs(candidate_colors[agent_track] - role_colors["agent"]).sum()
            )
            patient_cost = float(
                np.abs(
                    candidate_colors[patient_track] - role_colors["patient"]
                ).sum()
            )
            assignments.append(
                (agent_cost + patient_cost, agent_track, patient_track)
            )
    total_cost, agent_track, patient_track = min(assignments)
    selected_tracks = [agent_track, patient_track]
    return {
        "method": "joint_agent_patient_median_color_prune",
        "input_tracks": candidate_ids,
        "selected_tracks": selected_tracks,
        "selected_role_hypothesis": {
            "agent": agent_track,
            "patient": patient_track,
        },
        "pruned_tracks": [
            track_id
            for track_id in candidate_ids
            if track_id not in selected_tracks
        ],
        "total_l1_cost": total_cost,
    }


def _pp_reduce_identity_candidates_by_reference(
    *,
    candidate_image: np.ndarray,
    reference_image: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    reference_mask: np.ndarray,
) -> dict[str, Any]:
    """Keep the two candidates closest to one cue-bound reference for VLM A/B."""

    candidate_ids = sorted(candidate_masks)
    if len(candidate_ids) <= 2:
        raise ValueError(
            "TRK-004 reference reduction requires more than two candidates"
        )
    reference_color = _pp_mask_median_color(reference_image, reference_mask)
    ranked = sorted(
        (
            float(
                np.abs(
                    _pp_mask_median_color(candidate_image, candidate_masks[track_id])
                    - reference_color
                ).sum()
            ),
            track_id,
        )
        for track_id in candidate_ids
    )
    selected_tracks = [track_id for _, track_id in ranked[:2]]
    return {
        "method": "reference_median_color_prune",
        "input_tracks": candidate_ids,
        "selected_tracks": selected_tracks,
        "pruned_tracks": [
            track_id
            for track_id in candidate_ids
            if track_id not in selected_tracks
        ],
        "candidate_l1_costs": {
            track_id: cost for cost, track_id in ranked
        },
    }


def _bw_assign_roles_and_purge(
    *,
    video: Path,
    frames: list[np.ndarray],
    transition: dict[str, Any],
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    bind_minimum_coverage: float,
    wall_position_iou: float,
    static_pair_iou: float,
    seg2_patient_dynamic: bool = False,
    all_patients_dynamic: bool = False,
) -> dict[str, Any]:
    """Bind cue roles, identify the wall, and remove unassigned tracks.

    The wall is confirmed by cross-segment position, static pairs are purged, and
    focused VLM decisions bind physical identities across segments.
    seg2_patient_dynamic keeps the seg2 patient a dynamic track instead of a static
    fixture (friction_collision only, where the agent can knock it before the cut).
    all_patients_dynamic keeps BOTH segments' patients dynamic (friction_collision:
    the whole scene is reconstructed the CLEVRER way -- per-frame FoundationPose on
    every object -- so no track may take the constant-pose static-fixture route)."""
    artifact_identity = _tracking_shared_module("artifact_identity")
    cue_trimmed_suffix = artifact_identity.require_string("cue_trimmed_suffix")
    cue_flash_suffix = artifact_identity.require_string("cue_flash_suffix")
    static_track_prefix = artifact_identity.require_string("static_track_prefix")
    t_last = len(frames) - 1
    flash_path = video.with_name(
        video.name.replace(cue_trimmed_suffix, cue_flash_suffix)
    )
    if not video.name.endswith(cue_trimmed_suffix) or not flash_path.exists():
        return {"status": "skipped_no_flash_npz", "expected": str(flash_path)}
    flash = np.load(flash_path)
    regions = {"agent": flash["red_region"].astype(bool), "patient": flash["yellow_region"].astype(bool)}

    def anchor_mask(track_id: str, frame_index: int) -> np.ndarray | None:
        for record in all_records:
            if str(record["object_id"]) == track_id and int(record["frame_index"]) == frame_index:
                mask = all_arrays.get(str(record["mask_key"]))
                if mask is not None and mask.any():
                    return mask.astype(bool)
        return None

    seg_tracks = {"seg1": set(), "seg2": set()}
    for record in all_records:
        object_id = str(record["object_id"])
        if object_id.startswith("seg1_"):
            seg_tracks["seg1"].add(object_id)
        elif object_id.startswith("seg2_"):
            seg_tracks["seg2"].add(object_id)
    anchors1 = {tid: anchor_mask(tid, 0) for tid in seg_tracks["seg1"]}
    anchors2 = {tid: anchor_mask(tid, t_last) for tid in seg_tracks["seg2"]}
    anchors1 = {tid: m for tid, m in anchors1.items() if m is not None}
    anchors2 = {tid: m for tid, m in anchors2.items() if m is not None}

    assign: dict[str, dict[str, str]] = {"seg1": {}, "seg2": {}}
    detail: dict[str, Any] = {}
    for role, region in (("agent", regions["agent"]), ("patient", regions["patient"])):
        best, best_tid = 0.0, None
        for tid, mask in anchors2.items():
            coverage = float((mask & region).sum() / max(1, mask.sum()))
            if coverage > best:
                best, best_tid = coverage, tid
        if best_tid is not None and best >= bind_minimum_coverage:
            assign["seg2"][role] = best_tid
        detail[f"bind_{role}"] = {"track": best_tid, "coverage": round(best, 3)}

    rest2 = [tid for tid in anchors2 if tid not in assign["seg2"].values()]
    if rest2:
        wall2 = max(rest2, key=lambda tid: int(anchors2[tid].sum()))
        position_ious = {tid: _pp_mask_iou(mask, anchors2[wall2]) for tid, mask in anchors1.items()}
        if position_ious:
            wall1, pos_iou = max(position_ious.items(), key=lambda item: item[1])
            detail["wall"] = {"seg2": wall2, "seg1": wall1, "position_iou": round(pos_iou, 3)}
            if pos_iou >= wall_position_iou:
                assign["seg2"]["wall"] = wall2
                assign["seg1"]["wall"] = wall1

    rest1 = [tid for tid in anchors1 if tid not in assign["seg1"].values()]
    rest2 = [tid for tid in anchors2 if tid not in assign["seg2"].values()]
    static_pairs = []
    for tid1 in list(rest1):
        for tid2 in rest2:
            pair_iou = _pp_mask_iou(anchors1[tid1], anchors2[tid2])
            if pair_iou >= static_pair_iou:
                rest1.remove(tid1)
                static_pairs.append({"seg1": tid1, "seg2": tid2, "iou": round(pair_iou, 3)})
                break
    detail["static_pairs_purged"] = static_pairs

    seg2_agent = assign["seg2"].get("agent")
    if seg2_agent and rest1:
        if len(rest1) > 2:
            seg2_patient = assign["seg2"].get("patient")
            if seg2_patient is None:
                raise ValueError(
                    "TRK-004 candidate reduction requires the seg2 patient role"
                )
            candidate_reduction = _pp_reduce_seg1_identity_candidates(
                seg1_image=frames[0],
                seg2_image=frames[t_last],
                candidate_masks={tid: anchors1[tid] for tid in rest1},
                seg2_agent_mask=anchors2[seg2_agent],
                seg2_patient_mask=anchors2[seg2_patient],
            )
            rest1 = list(candidate_reduction["selected_tracks"])
            detail["seg1_candidate_reduction"] = candidate_reduction
        if len(rest1) == 1:
            identity_decision = {
                "method": "single_candidate",
                "selected_track": rest1[0],
                "candidate_tracks": list(rest1),
            }
        elif len(rest1) == 2:
            identity_decision = _pp_vlm_pick_seg1_agent(
                frames=frames,
                transition=transition,
                t_last=t_last,
                all_records=all_records,
                all_arrays=all_arrays,
                seg1_masks={tid: anchors1[tid] for tid in rest1},
                seg2_agent_tid=seg2_agent,
                anchors2=anchors2,
            )
        else:
            raise ValueError(
                f"TRK-004 identity has invalid reduced seg1 candidates: {rest1}"
            )
        seg1_agent = str(identity_decision["selected_track"])
        assign["seg1"]["agent"] = seg1_agent
        rest1.remove(seg1_agent)
        if assign["seg2"].get("patient") and rest1:
            assign["seg1"]["patient"] = rest1.pop(0)
        detail["seg1_link"] = identity_decision["method"]
        detail["seg1_identity"] = identity_decision
    elif seg2_agent:
        raise ValueError("TRK-004 identity has no valid seg1 candidate")
    else:
        detail["seg1_link"] = "missing_seg2_agent"

    assigned_ids = {tid for seg in assign.values() for tid in seg.values()}
    purged = sorted((seg_tracks["seg1"] | seg_tracks["seg2"]) - assigned_ids)
    if purged:
        purged_set = set(purged)
        for record in [r for r in all_records if str(r["object_id"]) in purged_set]:
            all_arrays.pop(str(record["mask_key"]), None)
            all_records.remove(record)

    # static-fixture prefix for the four fixtures so downstream routes them to the
    # constant-pose treatment; agents keep their segment track ids
    mapping: dict[str, str] = {}
    static_index = 0
    for seg in ("seg1", "seg2"):
        for role in ("patient", "wall"):
            tid = assign[seg].get(role)
            if not tid:
                continue
            if role == "patient" and (
                all_patients_dynamic or (seg2_patient_dynamic and seg == "seg2")
            ):
                continue  # dynamic patients keep their segment track ids
            mapping[tid] = f"{static_track_prefix}{static_index:03d}"
            static_index += 1
    detail["seg2_patient_dynamic"] = seg2_patient_dynamic
    detail["all_patients_dynamic"] = all_patients_dynamic
    for record in all_records:
        new_id = mapping.get(str(record["object_id"]))
        if not new_id:
            continue
        old_key = str(record["mask_key"])
        new_key = (
            f"{new_id}__frame_{int(record['frame_index']):05d}"
            f"__sam_obj_{int(record.get('sam_object_id') or 0)}"
        )
        if old_key in all_arrays:
            all_arrays[new_key] = all_arrays.pop(old_key)
        record["object_id"] = new_id
        record["mask_key"] = new_key
    assign = {
        seg: {role: mapping.get(tid, tid) for role, tid in roles.items()}
        for seg, roles in assign.items()
    }

    missing = [
        f"{seg}/{role}"
        for seg in ("seg1", "seg2")
        for role in ("agent", "patient", "wall")
        if role not in assign[seg]
    ]
    return {
        "status": "ok" if not missing else "partial",
        "flash_regions": flash_path.name,
        "agent_track": assign["seg2"].get("agent"),
        "patient_track": assign["seg2"].get("patient"),
        "assignments": assign,
        "purged_tracks": purged,
        "missing_roles": missing,
        "detail": detail,
    }


def _bw_write_segment_debug_videos(
    *, video: Path, frames: list[np.ndarray], transition: dict[str, Any], output: Path
) -> dict[str, str]:
    debug_dir = output.parent / "debug" / "videos"
    debug_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for seg in ("seg1", "seg2"):
        lo, hi = transition[seg]
        target = debug_dir / f"{video.stem}_{seg}.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        try:
            for frame_index in range(lo, hi + 1):
                writer.write(frames[frame_index])
        finally:
            writer.release()
        written[seg] = str(target)
    return written


def _pp_text_track_segment(
    *,
    model: Any,
    video: Path,
    anchor_frame_index: int,
    anchor_image: np.ndarray,
    concept: str,
    direction: str,
    keep_lo: int,
    keep_hi: int,
    seg_tag: str,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> list[str]:
    """SAM3 text-concept detect+track inside one segment (no GDINO). Every SAM3 instance obj_id
    becomes a track ``f"{seg_tag}_track_{obj_id:03d}"``; the concept response seeds the anchor frame
    (propagation may skip it) and propagation fills the rest of the segment. Returns the track ids."""
    session_id = model.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )["session_id"]
    track_ids: set[str] = set()
    minimum_area_px = _tracking_shared_module(
        "sam3_mask_filter"
    ).require_integer("minimum_area_px")

    def _emit(sam_obj_id: int, frame_index: int, mask: np.ndarray) -> None:
        mask_u8 = _mask_for_image(mask, anchor_image).astype(np.uint8)
        if int(mask_u8.sum()) < minimum_area_px:
            return
        track_id = f"{seg_tag}_track_{int(sam_obj_id):03d}"
        mask_key = f"{track_id}__frame_{frame_index:05d}__sam_obj_{int(sam_obj_id)}"
        all_arrays[mask_key] = mask_u8
        all_records.append(
            {
                "object_id": track_id,
                "concept_id": PHYSION_PP_TEXT_CONCEPT_ID,
                "prompt": f"text:{concept}",
                "expected_object_ids": [],
                "frame_index": int(frame_index),
                "sam_object_id": int(sam_obj_id),
                "mask_key": mask_key,
                "score": 1.0,
                **_mask_stats(mask_u8),
            }
        )
        track_ids.add(track_id)

    try:
        response = model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor_frame_index,
                "text": concept,
            }
        )
        for sam_obj_id, mask in _masks_from_response(response).items():
            _emit(sam_obj_id, anchor_frame_index, mask)
        seen_frame_indices: set[int] = {anchor_frame_index}
        for response in model.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
                "start_frame_index": anchor_frame_index,
                "max_frame_num_to_track": int(keep_hi - keep_lo + 3),
            }
        ):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices or not (keep_lo <= frame_index <= keep_hi):
                continue
            seen_frame_indices.add(frame_index)
            for sam_obj_id, mask in _masks_from_response(response).items():
                _emit(sam_obj_id, frame_index, mask)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})
    return sorted(track_ids)


def _run_physion_pp_two_segment_text_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    output: Path,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """friction_collision recipe: two-segment wipe split + SAM3 text-concept tracking (no GDINO).

    Uses the shared transition detector, same-object merge, and cue-flash role binding.
    Each segment is tracked through one ``PHYSION_PP_TEXT_CONCEPT`` pass
    (GDINO/window-filter/point-box probe are all skipped). If no wipe is found the whole clip is
    tracked as one (seg2) segment so role binding still has a t_last anchor for the flash regions.
    """
    same_object_merge = _tracking_shared_module("same_object_merge")
    cue_role_binding = _tracking_shared_module("cue_role_binding")
    frames = _bw_read_all_frames(
        video,
        minimum_frame_count=PHYSION_PP_BW_BG1_FRAMES + PHYSION_PP_BW_BG2_FRAMES,
    )
    t_last = len(frames) - 1
    transition = _bw_detect_transition(
        frames,
        rgb_difference_threshold=PHYSION_PP_BW_RGB_DIFF_THRESH,
        core_minimum_difference_fraction=PHYSION_PP_BW_CORE_MIN_DIFF_FRAC,
        column_band_top=PHYSION_PP_BW_COL_BAND_TOP,
        column_occupancy=PHYSION_PP_BW_COL_OCCUPANCY,
        column_adjacency=PHYSION_PP_BW_COL_ADJACENCY,
        stall_frames=PHYSION_PP_BW_STALL_FRAMES,
        background_1_frames=PHYSION_PP_BW_BG1_FRAMES,
        background_2_frames=PHYSION_PP_BW_BG2_FRAMES,
    )
    if transition is None:
        segment_specs = (("seg2", t_last, "backward", (0, t_last)),)
    else:
        segment_specs = (
            ("seg1", 0, "forward", tuple(transition["seg1"])),
            ("seg2", t_last, "backward", tuple(transition["seg2"])),
        )

    tracks_by_segment: dict[str, list[str]] = {}
    for seg_tag, anchor, direction, (keep_lo, keep_hi) in segment_specs:
        tracks_by_segment[seg_tag] = _pp_text_track_segment(
            model=model,
            video=video,
            anchor_frame_index=anchor,
            anchor_image=frames[anchor],
            concept=PHYSION_PP_TEXT_CONCEPT,
            direction=direction,
            keep_lo=keep_lo,
            keep_hi=keep_hi,
            seg_tag=seg_tag,
            all_records=all_records,
            all_arrays=all_arrays,
        )

    merge_summary = _apply_physion_pp_same_object_merge(
        all_records=all_records,
        all_arrays=all_arrays,
        score_by_object_id={str(record["object_id"]): 1.0 for record in all_records},
        evidence_kinds=frozenset(
            same_object_merge.require_strings("default_evidence")
        ),
    )
    role_assignment = _bw_assign_roles_and_purge(
        video=video,
        frames=frames,
        transition=transition or {"seg1": [0, -1], "seg2": [0, t_last]},
        all_records=all_records,
        all_arrays=all_arrays,
        bind_minimum_coverage=cue_role_binding.require_number(
            "minimum_coverage"
        ),
        wall_position_iou=PHYSION_PP_BW_WALL_POS_IOU,
        static_pair_iou=PHYSION_PP_BW_STATIC_PAIR_IOU,
        seg2_patient_dynamic=True,
        all_patients_dynamic=True,
    )

    debug_segment_videos: dict[str, str] = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1" and transition is not None:
        debug_segment_videos = _bw_write_segment_debug_videos(
            video=video, frames=frames, transition=transition, output=output
        )

    tracking = {
        "mode": "tracking.sam3_two_segment_text",
        "scenario": scenario_name,
        "two_segment": transition
        if transition is not None
        else {"status": "transition_not_found_fallback_single_segment"},
        "text_concept": PHYSION_PP_TEXT_CONCEPT,
        "tracks_by_segment": tracks_by_segment,
        "same_object_merge": merge_summary,
        "role_binding": role_assignment,
        "debug_segment_videos": debug_segment_videos,
    }
    payload_prompts = [{"concept": PHYSION_PP_TEXT_CONCEPT, "prompt_mode": "text"}]
    return payload_prompts, [], tracking


def _mc_detect_transition(
    frames: list[np.ndarray],
    *,
    module_profile: TrackingModuleProfile,
) -> dict[str, Any] | None:
    """Split a mass-collision clip using the curtain's full-frame coverage signal.

    The per-frame minimum difference from the first and last arrangements locates the
    wipe core; a fixed margin includes the curtain's entering and leaving edges.
    """
    temporal = module_profile.module("temporal_partition")
    stack = np.stack(frames).astype(np.int16)
    t_last = len(frames) - 1
    cov = np.minimum(
        (
            np.abs(stack - stack[0]).max(axis=3)
            > temporal.require_integer("rgb_difference_threshold")
        ).mean(axis=(1, 2)),
        (
            np.abs(stack - stack[t_last]).max(axis=3)
            > temporal.require_integer("rgb_difference_threshold")
        ).mean(axis=(1, 2)),
    )
    core = np.nonzero(
        cov > temporal.require_number("core_minimum_difference_fraction")
    )[0]
    if not len(core):
        return None
    core_start, core_end = int(core[0]), int(core[-1])
    edge_margin_frames = temporal.require_integer("edge_margin_frames")
    a = max(1, core_start - edge_margin_frames)
    b = min(t_last - 1, core_end + edge_margin_frames)
    if not (0 < a <= b < t_last):
        return None
    return {
        "a": a,
        "b": b,
        "core_start": core_start,
        "core_end": core_end,
        "seg1": [0, a - 1],
        "seg2": [b + 1, t_last],
    }


_MC_MOGE_MODELS: dict[str, Any] = {}


def _mc_get_moge_model(model_name: str) -> Any:
    """Lazily load MoGe-2 once and cache it (a shared single-frame depth resource; only the
    mass_collision window-removal step calls it, but it is not exclusive to that scenario)."""
    if model_name not in _MC_MOGE_MODELS:
        moge_root = PROJECT_ROOT / "third_party" / "MoGe"
        if str(moge_root) not in sys.path:
            sys.path.insert(0, str(moge_root))
        import torch  # noqa: F401
        from moge.model.v2 import MoGeModel

        _MC_MOGE_MODELS[model_name] = (
            MoGeModel.from_pretrained(model_name).to("cuda").eval()
        )
    return _MC_MOGE_MODELS[model_name]


def _mc_moge_depth(frame_bgr: np.ndarray, *, model_name: str) -> np.ndarray:
    import torch

    model = _mc_get_moge_model(model_name)
    image = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    image = image.permute(2, 0, 1).to("cuda")
    with torch.no_grad():
        result = model.infer(image)
    return result["depth"].detach().cpu().numpy().astype(np.float32)


def _mc_depth_rank(depth: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the image nearer than the region's median depth (near ~0, far background ~1)."""
    valid = np.isfinite(depth)
    region = depth[mask & valid]
    if region.size == 0 or valid.sum() == 0:
        return 1.0
    return float((depth[valid] < float(np.median(region))).mean())


def _mc_mask_touches_border(mask: np.ndarray, *, margin: int) -> bool:
    return bool(
        mask[:margin, :].any() or mask[-margin:, :].any()
        or mask[:, :margin].any() or mask[:, -margin:].any()
    )


def _mc_ball_priority_carve(
    *,
    ball_track_ids: set[str],
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> int:
    """Give ball masks pixel priority by carving their union from other tracks."""
    ball_union: dict[int, np.ndarray] = {}
    for record in all_records:
        if str(record["object_id"]) not in ball_track_ids:
            continue
        mask = all_arrays.get(str(record["mask_key"]))
        if mask is not None and mask.any():
            frame = int(record["frame_index"])
            bool_mask = mask.astype(bool)
            ball_union[frame] = (ball_union[frame] | bool_mask) if frame in ball_union else bool_mask
    carved = 0
    for record in all_records:
        if str(record["object_id"]) in ball_track_ids:
            continue
        carve = ball_union.get(int(record["frame_index"]))
        if carve is None:
            continue
        key = str(record["mask_key"])
        mask = all_arrays.get(key)
        if mask is None or not (mask.astype(bool) & carve).any():
            continue
        new_mask = (mask.astype(bool) & ~carve).astype(np.uint8)
        all_arrays[key] = new_mask
        record.update(_mask_stats(new_mask))
        carved += 1
    return carved


def _mc_assign_roles_and_cleanup(
    *,
    video: Path,
    frames: list[np.ndarray],
    transition: dict[str, Any] | None,
    ball_tracks: dict[str, str | None],
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> dict[str, Any]:
    """Bind mass-collision roles and remove background tracks.

    Cue regions bind segment-two agent and patient tracks, VLM identity binds the agent
    across segments, and seeded solo tracks provide ball identity. Border and far-depth
    unbound tracks are removed from segment one.
    """
    artifact_identity = _tracking_shared_module("artifact_identity")
    cue_role_binding = _tracking_shared_module("cue_role_binding")
    postprocess = module_profile.module("postprocess")
    cue_trimmed_suffix = artifact_identity.require_string("cue_trimmed_suffix")
    cue_flash_suffix = artifact_identity.require_string("cue_flash_suffix")
    minimum_coverage = cue_role_binding.require_number("minimum_coverage")
    edge_touch_margin_px = postprocess.require_integer("edge_touch_margin_px")
    moge_model = postprocess.require_string("moge_model")
    depth_rank_threshold = postprocess.require_number("depth_rank_threshold")
    t_last = len(frames) - 1
    flash_path = video.with_name(
        video.name.replace(cue_trimmed_suffix, cue_flash_suffix)
    )
    if not video.name.endswith(cue_trimmed_suffix) or not flash_path.exists():
        return {"status": "skipped_no_flash_npz", "expected": str(flash_path)}
    flash = np.load(flash_path)
    regions = {"agent": flash["red_region"].astype(bool), "patient": flash["yellow_region"].astype(bool)}

    records_by_track: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        records_by_track.setdefault(str(record["object_id"]), []).append(record)

    def mask_at(track_id: str, frame_index: int) -> np.ndarray | None:
        for record in records_by_track.get(track_id, []):
            if int(record["frame_index"]) == frame_index:
                mask = all_arrays.get(str(record["mask_key"]))
                if mask is not None and mask.any():
                    return mask.astype(bool)
        return None

    def settled_mask(track_id: str) -> tuple[int, np.ndarray] | None:
        for record in sorted(records_by_track.get(track_id, []),
                             key=lambda r: -int(r["frame_index"])):
            mask = all_arrays.get(str(record["mask_key"]))
            if mask is not None and mask.any():
                return int(record["frame_index"]), mask.astype(bool)
        return None

    ball1, ball2 = ball_tracks.get("seg1"), ball_tracks.get("seg2")
    seg1_tracks = sorted(tid for tid in records_by_track if tid.startswith("seg1_"))
    seg2_tracks = sorted(tid for tid in records_by_track if tid.startswith("seg2_"))
    anchors2 = {tid: mask_at(tid, t_last) for tid in seg2_tracks if tid != ball2}
    anchors2 = {tid: m for tid, m in anchors2.items() if m is not None}
    anchors1 = {tid: mask_at(tid, 0) for tid in seg1_tracks if tid != ball1}
    anchors1 = {tid: m for tid, m in anchors1.items() if m is not None}

    assign: dict[str, dict[str, str]] = {"seg1": {}, "seg2": {}}
    detail: dict[str, Any] = {}
    for role in ("agent", "patient"):
        region = regions[role]
        best, best_tid = 0.0, None
        for tid, mask in anchors2.items():
            coverage = float((mask & region).sum() / max(1, mask.sum()))
            if coverage > best:
                best, best_tid = coverage, tid
        if best_tid is not None and best >= minimum_coverage:
            assign["seg2"][role] = best_tid
        detail[f"bind_{role}"] = {"track": best_tid, "coverage": round(best, 3)}
    agent2 = assign["seg2"].get("agent")
    if agent2 is not None and anchors1:
        candidate_tracks = sorted(anchors1)
        if len(candidate_tracks) > 2:
            candidate_reduction = _pp_reduce_identity_candidates_by_reference(
                candidate_image=frames[0],
                reference_image=frames[t_last],
                candidate_masks={tid: anchors1[tid] for tid in candidate_tracks},
                reference_mask=anchors2[agent2],
            )
            candidate_tracks = list(candidate_reduction["selected_tracks"])
            detail["seg1_candidate_reduction"] = candidate_reduction
        if len(candidate_tracks) == 1:
            identity_decision = {
                "method": "single_candidate",
                "selected_track": candidate_tracks[0],
                "candidate_tracks": candidate_tracks,
            }
        elif len(candidate_tracks) == 2:
            identity_decision = _pp_vlm_pick_seg1_agent(
                frames=frames,
                transition=transition or {},
                t_last=t_last,
                all_records=all_records,
                all_arrays=all_arrays,
                seg1_masks={tid: anchors1[tid] for tid in candidate_tracks},
                seg2_agent_tid=agent2,
                anchors2=anchors2,
            )
        else:
            raise ValueError(
                f"TRK-004 identity has invalid reduced MC seg1 candidates: {candidate_tracks}"
            )
        assign["seg1"]["agent"] = str(identity_decision["selected_track"])
        detail["seg1_agent"] = identity_decision

    # --- cleanup ---
    keep2 = {tid for tid in (assign["seg2"].get("agent"), assign["seg2"].get("patient"), ball2) if tid}
    removed: list[dict[str, Any]] = []
    for tid in seg2_tracks:
        if tid not in keep2:
            removed.append({"track": tid, "seg": "seg2", "reason": "seg2_unbound"})
    depth_cache: dict[int, np.ndarray] = {}
    kept_extras: list[dict[str, Any]] = []
    bound1 = {tid for tid in (assign["seg1"].get("agent"), ball1) if tid}
    for tid in seg1_tracks:
        if tid in bound1:
            continue
        settled = settled_mask(tid)
        if settled is None:
            removed.append({"track": tid, "seg": "seg1", "reason": "seg1_empty"})
            continue
        frame_index, mask = settled
        if _mc_mask_touches_border(
            mask,
            margin=edge_touch_margin_px,
        ):
            removed.append({"track": tid, "seg": "seg1", "reason": "seg1_edge"})
            continue
        if frame_index not in depth_cache:
            depth_cache[frame_index] = _mc_moge_depth(
                frames[frame_index],
                model_name=moge_model,
            )
        rank = round(_mc_depth_rank(depth_cache[frame_index], mask), 3)
        if rank >= depth_rank_threshold:
            removed.append({"track": tid, "seg": "seg1", "reason": "seg1_window", "depth_rank": rank})
            continue
        kept_extras.append({"track": tid, "settled_frame": frame_index, "depth_rank": rank})
    if removed:
        removed_ids = {item["track"] for item in removed}
        for record in [r for r in all_records if str(r["object_id"]) in removed_ids]:
            all_arrays.pop(str(record["mask_key"]), None)
            all_records.remove(record)

    missing = [f"seg2/{role}" for role in ("agent", "patient") if role not in assign["seg2"]]
    if ball2 is None:
        missing.append("seg2/ball")
    return {
        "status": "ok" if not missing else "partial",
        "flash_regions": flash_path.name,
        "agent_track": assign["seg2"].get("agent"),
        "patient_track": assign["seg2"].get("patient"),
        "assignments": assign,
        "ball": {"seg1": ball1, "seg2": ball2, "source": "gdino_seed"},
        "removed_tracks": removed,
        "kept_extra_tracks": kept_extras,
        "missing_roles": missing,
        "detail": detail,
    }


def _mc_box_sv(frame: np.ndarray, box: list[int]) -> tuple[float, float]:
    """Median HSV (S, V) inside a det box, clamped to the frame (degenerate boxes safe)."""
    height, width = frame.shape[:2]
    x1 = min(max(0, int(box[0])), width - 2)
    x2 = min(max(x1 + 1, int(box[2])), width - 1)
    y1 = min(max(0, int(box[1])), height - 2)
    y2 = min(max(y1 + 1, int(box[3])), height - 1)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 1])), float(np.median(hsv[:, :, 2]))


def _mc_contain_frac(inner: list[int], outer: list[int]) -> float:
    return _bp_xyxy_inter(inner, outer) / max(1, _bp_xyxy_area(inner))


def _mc_box_related(
    det_box: list[int],
    ref_box: list[int],
    *,
    detector: ModuleSpec,
) -> bool:
    """det IS the ref object, or a group box swallowing it (either-way containment / IoU)."""
    from scripts.world_model.run_gdino_boxes import box_iou

    return (
        _mc_contain_frac(ref_box, det_box)
        >= detector.require_number("containment")
        or _mc_contain_frac(det_box, ref_box)
        >= detector.require_number("same_object")
        or box_iou(det_box, ref_box)
        >= detector.require_number("same_object")
    )


def _mc_edge_gap(a: list[int], b: list[int]) -> int:
    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3], 0)


def _mc_is_window_colored(
    frame: np.ndarray,
    box: list[int],
    *,
    detector: ModuleSpec,
) -> bool:
    sat, val = _mc_box_sv(frame, box)
    return (
        sat < detector.require_number("window_saturation_maximum")
        and val > detector.require_number("window_value_minimum")
    )


def _mc_gdino_flow(
    frame: np.ndarray,
    dets_by_query: dict[str, list[dict[str, Any]]],
    *,
    module_profile: TrackingModuleProfile,
) -> dict[str, Any]:
    """Select one box per role on an anchor frame: ball -> windows ->
    agent (ball/window dedup + score-band largest + colour-gated gap merge) -> patient (same
    dedup + agent containment suppression, top-1). Returns one box per part (or None)."""
    detector = module_profile.module("detector")
    cleanup_profile = _tracking_shared_module("gdino_cleanup")
    ball_query = detector.require_string("ball_query")
    agent_query = detector.require_string("agent_query")
    patient_query = detector.require_string("patient_query")
    window_query = detector.require_string("window_query")
    ball_dets = dets_by_query[ball_query]
    ball_box = ball_dets[0]["bbox_xyxy"] if ball_dets else None
    windows = [
        det["bbox_xyxy"]
        for det in dets_by_query[window_query]
        if _mc_is_window_colored(
            frame,
            det["bbox_xyxy"],
            detector=detector,
        )
    ]

    height, width = frame.shape[:2]

    def clean(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from scripts.world_model.run_gdino_boxes import _edge_touch_count

        kept = []
        for det in cands:
            box = det["bbox_xyxy"]
            if ball_box and _mc_box_related(box, ball_box, detector=detector):
                continue
            # Remove window-colored detections even if the window query misses an
            # edge-cropped window.
            if _mc_is_window_colored(frame, box, detector=detector):
                continue
            if any(_mc_box_related(box, w, detector=detector) for w in windows):
                continue
            # Frame-spanning stray (>=3 edges touched) = background junk, e.g. the giant
            # patient pick on absent-patient seg1 scenes; zero true picks touch 3 edges.
            if (
                _edge_touch_count(
                    box,
                    width,
                    height,
                    edge_margin_px=cleanup_profile.require_integer(
                        "edge_margin_px"
                    ),
                )
                >= cleanup_profile.require_integer("edge_sides_to_drop")
            ):
                continue
            kept.append(det)
        return kept

    def median_bgr(box: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.median(
            frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].reshape(-1, 3), axis=0
        )

    agent_cands = clean(dets_by_query[agent_query])
    agent: dict[str, Any] | None = None
    if agent_cands:
        smax = max(det["score"] for det in agent_cands)
        band = [
            det
            for det in agent_cands
            if det["score"] >= detector.require_number("score_band") * smax
        ]
        pick = max(band, key=lambda det: _bp_xyxy_area(det["bbox_xyxy"]))
        agent_box, base = list(pick["bbox_xyxy"]), median_bgr(pick["bbox_xyxy"])
        for det in band:
            box = det["bbox_xyxy"]
            if (
                box == pick["bbox_xyxy"]
                or _mc_edge_gap(box, agent_box)
                > detector.require_integer("merge_gap_px")
            ):
                continue
            if (
                np.abs(median_bgr(box) - base).max()
                > detector.require_number("merge_color_distance")
            ):
                continue
            agent_box = [min(agent_box[0], box[0]), min(agent_box[1], box[1]),
                         max(agent_box[2], box[2]), max(agent_box[3], box[3])]
        agent = {"bbox_xyxy": agent_box, "score": float(pick["score"])}
    patient_cands = [
        det for det in clean(dets_by_query[patient_query])
        if not (
            agent
            and _mc_contain_frac(det["bbox_xyxy"], agent["bbox_xyxy"])
            >= detector.require_number("agent_suppression")
        )
    ]
    patient = patient_cands[0] if patient_cands else None
    return {
        "ball": {"bbox_xyxy": list(ball_box), "score": float(ball_dets[0]["score"])} if ball_box else None,
        "agent": agent,
        "patient": {"bbox_xyxy": list(patient["bbox_xyxy"]), "score": float(patient["score"])} if patient else None,
        "window_boxes": windows,
    }


def _mc_select_seg2_anchor(
    *,
    flow_at: Any,
    seg2_lo: int,
    seg2_hi: int,
    module_profile: TrackingModuleProfile,
) -> tuple[int, dict[str, Any]]:
    """Select the segment-two anchor from configured offsets near the final frame.

    Candidates are tried in profile order. A confident post-suppression patient pick
    wins, followed by any patient pick and then the primary offset.
    """
    sam3_seed = module_profile.module("sam3_seed")
    candidates = []
    for offset in sam3_seed.require_integers("anchor_offsets"):
        t = max(seg2_lo, seg2_hi - offset)
        if t not in candidates:
            candidates.append(t)
    scan = []
    for t in candidates:
        picks = flow_at(t)
        scan.append({
            "t": t,
            "patient_score": round(picks["patient"]["score"], 4) if picks["patient"] else 0.0,
        })
    for tier, status in (
        (
            lambda s: s["patient_score"]
            >= sam3_seed.require_number("anchor_patient_minimum_score"),
            "ok",
        ),
        (lambda s: s["patient_score"] > 0, "fallback_weak_patient"),
    ):
        for s in scan:
            if tier(s):
                return s["t"], {"status": status, "scan": scan}
    return candidates[0], {"status": "fallback_no_patient", "scan": scan}


_MC_GDINO_MODEL: Any | None = None


def _mc_get_gdino() -> Any:
    """Lazily load Grounding DINO once per process and cache it (the MC frontend runs ~60
    queries per scene across the anchor scan; a per-scene reload wastes ~8s each)."""
    global _MC_GDINO_MODEL
    if _MC_GDINO_MODEL is None:
        from scripts.world_model.run_gdino_boxes import load_gdino

        _MC_GDINO_MODEL = load_gdino()
    return _MC_GDINO_MODEL


def _physion_pp_mc_gdino_prompts(
    *,
    frames: list[np.ndarray],
    seg2_lo: int,
    seg2_hi: int,
    module_profile: TrackingModuleProfile,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """mass_collision GDINO seeding: flow picks at seg1 f0 and at the runtime seg2 anchor t2*.
    Returns ({"seg1": [prompt...], "seg2": [prompt...]}, summary); each prompt is a two-corner
    BOX in the _bp_propagate format (points normalized, labels [2, 3])."""
    from scripts.world_model.run_gdino_boxes import detect_boxes

    detector = module_profile.module("detector")
    processor, gdino_model, device = _mc_get_gdino()

    def detect(image: np.ndarray, query: str) -> list[dict[str, Any]]:
        dets = detect_boxes(
            processor, gdino_model, image, query + " .",
            confidence=detector.require_number("confidence"), device=device,
        )
        dets.sort(key=lambda det: -det["score"])
        return dets[:detector.require_integer("top_k")]

    all_queries = (
        detector.require_string("ball_query"),
        detector.require_string("agent_query"),
        detector.require_string("patient_query"),
        detector.require_string("window_query"),
    )
    flow_cache: dict[int, dict[str, Any]] = {}

    def flow_at(frame_index: int) -> dict[str, Any]:
        if frame_index not in flow_cache:
            frame = frames[frame_index]
            flow_cache[frame_index] = _mc_gdino_flow(
                frame,
                {query: detect(frame, query) for query in all_queries},
                module_profile=module_profile,
            )
        return flow_cache[frame_index]

    t2, anchor_diag = _mc_select_seg2_anchor(
        flow_at=flow_at,
        seg2_lo=seg2_lo,
        seg2_hi=seg2_hi,
        module_profile=module_profile,
    )
    height, width = frames[0].shape[:2]
    prompts_by_segment: dict[str, list[dict[str, Any]]] = {}
    picks_by_segment: dict[str, dict[str, Any]] = {}
    query_by_part = {
        "ball": detector.require_string("ball_query"),
        "agent": detector.require_string("agent_query"),
        "patient": detector.require_string("patient_query"),
    }
    for seg_tag, anchor in (("seg1", 0), ("seg2", t2)):
        picks = flow_at(anchor)
        picks_by_segment[seg_tag] = picks
        prompts = []
        for sam_obj_id, part in enumerate(("ball", "agent", "patient"), start=1):
            pick = picks[part]
            if pick is None:
                continue
            x1, y1, x2, y2 = pick["bbox_xyxy"]
            if part == "ball":
                # Use a point seed for the small ball and propagate it in a solo
                # session with pixel priority.
                points = [[(x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height]]
                point_labels, prompt_mode = [1], "point"
            else:
                points = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
                point_labels, prompt_mode = [2, 3], "box"
            prompts.append({
                "sam_obj_id": sam_obj_id,
                "part": part,
                "class": query_by_part[part],
                "score": pick["score"],
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "points": points,
                "point_labels": point_labels,
                "prompt_mode": prompt_mode,
                "prompt_frame": int(anchor),
            })
        prompts_by_segment[seg_tag] = prompts
    summary = {
        "seg2_anchor": int(t2),
        "seg2_anchor_diag": anchor_diag,
        "queries": dict(
            query_by_part,
            window=detector.require_string("window_query"),
        ),
        "confidence": detector.require_number("confidence"),
        "picks": {
            seg: {part: (picks[part] or None) for part in ("ball", "agent", "patient")}
            for seg, picks in picks_by_segment.items()
        },
        "window_boxes": {seg: picks["window_boxes"] for seg, picks in picks_by_segment.items()},
    }
    return prompts_by_segment, summary


def _mc_gdino_track_segment(
    *,
    model: Any,
    video: Path,
    anchor_frame_index: int,
    anchor_image: np.ndarray,
    prompts: list[dict[str, Any]],
    direction: str,
    keep_lo: int,
    keep_hi: int,
    seg_tag: str,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> list[str]:
    """SAM3 box-prompt seeding + propagation inside one segment (the GDINO twin of
    ``_pp_text_track_segment``). Track ids keep the ``f"{seg_tag}_track_{obj:03d}"`` shape the
    downstream merge/role layers expect; the prompt field carries the part identity."""
    if not prompts:
        return []
    artifact_identity = _tracking_shared_module("artifact_identity")
    mask_filter = _tracking_shared_module("sam3_mask_filter")
    concept_id = artifact_identity.require_string("concept_id")
    minimum_area_px = mask_filter.require_integer("minimum_area_px")
    session_id = model.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )["session_id"]
    track_ids: set[str] = set()
    prompt_by_obj = {int(p["sam_obj_id"]): p for p in prompts}

    def _emit(sam_obj_id: int, frame_index: int, mask: np.ndarray) -> None:
        prompt = prompt_by_obj.get(int(sam_obj_id))
        if prompt is None:
            return
        mask_u8 = _mask_for_image(mask, anchor_image).astype(np.uint8)
        if int(mask_u8.sum()) < minimum_area_px:
            return
        track_id = f"{seg_tag}_track_{int(sam_obj_id):03d}"
        mask_key = f"{track_id}__frame_{frame_index:05d}__sam_obj_{int(sam_obj_id)}"
        all_arrays[mask_key] = mask_u8
        all_records.append(
            {
                "object_id": track_id,
                "concept_id": concept_id,
                "prompt": f"gdino:{prompt['part']}:{prompt['class']}",
                "expected_object_ids": [],
                "frame_index": int(frame_index),
                "sam_object_id": int(sam_obj_id),
                "mask_key": mask_key,
                "score": float(prompt["score"]),
                **_mask_stats(mask_u8),
            }
        )
        track_ids.add(track_id)

    try:
        for prompt in prompts:
            response = model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(prompt["prompt_frame"]),
                    "obj_id": int(prompt["sam_obj_id"]),
                    "points": prompt["points"],
                    "point_labels": prompt["point_labels"],
                }
            )
            for sam_obj_id, mask in _masks_from_response(response).items():
                _emit(sam_obj_id, int(prompt["prompt_frame"]), mask)
        seen_frame_indices: set[int] = {anchor_frame_index}
        for response in model.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
                "start_frame_index": anchor_frame_index,
                "max_frame_num_to_track": int(keep_hi - keep_lo + 3),
            }
        ):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices or not (keep_lo <= frame_index <= keep_hi):
                continue
            seen_frame_indices.add(frame_index)
            for sam_obj_id, mask in _masks_from_response(response).items():
                _emit(sam_obj_id, frame_index, mask)
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})
    return sorted(track_ids)


def _run_physion_pp_mass_collision_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    output: Path,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """mass_collision recipe: curtain-coverage wipe split (``_mc_detect_transition``) + GDINO
    seeding (``_physion_pp_mc_gdino_prompts``: ball POINT / agent+patient BOX prompts, seg1
    anchored at f0, seg2 at the fixed t2* < t_last) + SAM3 per-segment tracking with the ball
    in a SOLO session and top pixel priority (``_mc_ball_priority_carve``) + same-object merge
    over the non-ball tracks + the role/cleanup layer (``_mc_assign_roles_and_cleanup``)."""
    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="mass-collision tracking",
    )
    temporal = module_profile.module("temporal_partition")
    same_object_merge = _tracking_shared_module("same_object_merge")
    concept_id = _tracking_shared_module("artifact_identity").require_string(
        "concept_id"
    )
    frames = _bw_read_all_frames(
        video,
        minimum_frame_count=(
            temporal.require_integer("background_1_frames")
            + temporal.require_integer("background_2_frames")
        ),
    )
    t_last = len(frames) - 1
    transition = _mc_detect_transition(frames, module_profile=module_profile)
    if transition is None:
        seg2_lo, seg2_hi = 0, t_last
    else:
        seg2_lo, seg2_hi = transition["seg2"]
    prompts_by_segment, gdino_summary = _physion_pp_mc_gdino_prompts(
        frames=frames,
        seg2_lo=seg2_lo,
        seg2_hi=seg2_hi,
        module_profile=module_profile,
    )
    seg2_anchor = int(gdino_summary["seg2_anchor"])
    if transition is None:
        segment_specs = (("seg2", seg2_anchor, "both", (0, t_last)),)
    else:
        segment_specs = (
            ("seg1", 0, "forward", tuple(transition["seg1"])),
            ("seg2", seg2_anchor, "both", tuple(transition["seg2"])),
        )

    tracks_by_segment: dict[str, dict[str, list[str]]] = {}
    ball_tracks: dict[str, str | None] = {}
    for seg_tag, anchor, direction, (keep_lo, keep_hi) in segment_specs:
        seg_prompts = prompts_by_segment.get(seg_tag, [])
        groups = (
            ("solo_ball", [p for p in seg_prompts if p["part"] == "ball"]),
            ("joint_others", [p for p in seg_prompts if p["part"] != "ball"]),
        )
        seg_tracks: dict[str, list[str]] = {}
        for group_tag, group_prompts in groups:
            seg_tracks[group_tag] = _mc_gdino_track_segment(
                model=model,
                video=video,
                anchor_frame_index=anchor,
                anchor_image=frames[anchor],
                prompts=group_prompts,
                direction=direction,
                keep_lo=keep_lo,
                keep_hi=keep_hi,
                seg_tag=seg_tag,
                all_records=all_records,
                all_arrays=all_arrays,
            )
        tracks_by_segment[seg_tag] = seg_tracks
        ball_tracks[seg_tag] = seg_tracks["solo_ball"][0] if seg_tracks["solo_ball"] else None

    ball_track_ids = {tid for tid in ball_tracks.values() if tid}
    carved_record_count = _mc_ball_priority_carve(
        ball_track_ids=ball_track_ids, all_records=all_records, all_arrays=all_arrays
    )
    # The ball is safe from the union-find merge without any filtering: the priority carve
    # just made its pixels disjoint from every other track, so no overlap evidence can link it.
    merge_summary = _apply_physion_pp_same_object_merge(
        all_records=all_records,
        all_arrays=all_arrays,
        score_by_object_id={str(record["object_id"]): 1.0 for record in all_records},
        evidence_kinds=frozenset(
            same_object_merge.require_strings("default_evidence")
        ),
    )
    role_assignment = _mc_assign_roles_and_cleanup(
        video=video,
        frames=frames,
        transition=transition,
        ball_tracks=ball_tracks,
        all_records=all_records,
        all_arrays=all_arrays,
        module_profile=module_profile,
    )

    debug_segment_videos: dict[str, str] = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1" and transition is not None:
        debug_segment_videos = _bw_write_segment_debug_videos(
            video=video, frames=frames, transition=transition, output=output
        )

    tracking = {
        "mode": "tracking.gdino_two_segment_boxes",
        "scenario": scenario_name,
        "two_segment": transition
        if transition is not None
        else {"status": "transition_not_found_fallback_single_segment"},
        "gdino": gdino_summary,
        "tracks_by_segment": tracks_by_segment,
        "ball_priority_carve": {"carved_record_count": carved_record_count},
        "same_object_merge": merge_summary,
        "role_binding": role_assignment,
        "debug_segment_videos": debug_segment_videos,
    }
    payload_prompts = [
        {
            "object_id": f"{seg_tag}_track_{prompt['sam_obj_id']:03d}",
            "concept_id": concept_id,
            "sam_obj_id": prompt["sam_obj_id"],
            "prompt": f"gdino:{prompt['part']}:{prompt['class']}",
            "gdino_class": prompt["class"],
            "gdino_score": prompt["score"],
            "gdino_bbox_xyxy": prompt["bbox_xyxy"],
            "points": prompt["points"],
            "point_labels": prompt["point_labels"],
            "prompt_mode": prompt["prompt_mode"],
            "prompt_frame": prompt["prompt_frame"],
            "segment": seg_tag,
        }
        for seg_tag, prompts in prompts_by_segment.items()
        for prompt in prompts
    ]
    return payload_prompts, [], tracking


def _run_physion_pp_two_segment_tracking(
    *,
    model: Any,
    video: Path,
    scenario_name: str,
    output: Path,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
    module_profile: TrackingModuleProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _require_tracking_profile_scenario(
        module_profile,
        scenario_name,
        frontend="bouncy-wall two-segment tracking",
    )
    temporal = module_profile.module("temporal_partition")
    role_binding = module_profile.module("role_binding")
    cue_role_binding = _tracking_shared_module("cue_role_binding")
    same_object_merge = _tracking_shared_module("same_object_merge")
    frames = _bw_read_all_frames(
        video,
        minimum_frame_count=(
            temporal.require_integer("background_1_frames")
            + temporal.require_integer("background_2_frames")
        ),
    )
    t_last = len(frames) - 1
    transition = _bw_detect_transition(
        frames,
        rgb_difference_threshold=temporal.require_integer(
            "rgb_difference_threshold"
        ),
        core_minimum_difference_fraction=temporal.require_number(
            "core_minimum_difference_fraction"
        ),
        column_band_top=temporal.require_integer("column_band_top"),
        column_occupancy=temporal.require_number("column_occupancy"),
        column_adjacency=temporal.require_integer("column_adjacency"),
        stall_frames=temporal.require_integer("stall_frames"),
        background_1_frames=temporal.require_integer("background_1_frames"),
        background_2_frames=temporal.require_integer("background_2_frames"),
    )
    if transition is None:
        # no wipe found: fall back to the single-segment friction recipe rather
        # than failing the whole scene
        prompts, responses, tracking = _run_physion_pp_bouncy_wall_single_segment_tracking(
            model=model,
            video=video,
            scenario_name=scenario_name,
            prompt_frame_index=0,
            propagation_direction="forward",
            max_frame_num_to_track=None,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=module_profile,
        )
        tracking["two_segment"] = {"status": "transition_not_found_fallback_single_segment"}
        return prompts, responses, tracking

    segment_specs = (
        ("seg1", 0, "forward", transition["seg1"]),
        ("seg2", t_last, "backward", transition["seg2"]),
    )
    payload_prompts: list[dict[str, Any]] = []
    gdino_summaries: dict[str, Any] = {}
    window_drops: dict[str, Any] = {}
    kept_by_segment: dict[str, list[dict[str, Any]]] = {}
    dropped_by_segment: dict[str, list[dict[str, Any]]] = {}
    for seg_tag, anchor, direction, (keep_lo, keep_hi) in segment_specs:
        prompts, gdino_summary = _physion_pp_bouncy_wall_gdino_prompts(
            video=video,
            scenario_name=scenario_name,
            prompt_frame_index=anchor,
            module_profile=module_profile,
        )
        prompts, window_dropped = _bw_window_box_filter(
            prompts,
            frames[0],
            frames[t_last],
            module_profile=module_profile,
        )
        gdino_summaries[seg_tag] = gdino_summary
        window_drops[seg_tag] = [
            {k: v for k, v in d.items() if k != "_prompt_mask"} for d in window_dropped
        ]
        kept, dropped = _bw_probe_and_propagate_segment(
            model=model,
            video=video,
            anchor_frame_index=anchor,
            anchor_image=frames[anchor],
            prompts=prompts,
            direction=direction,
            keep_lo=keep_lo,
            keep_hi=keep_hi,
            seg_tag=seg_tag,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=module_profile,
        )
        kept_by_segment[seg_tag] = kept
        dropped_by_segment[seg_tag] = dropped
        payload_prompts.extend(
            {k: v for k, v in item.items() if not k.startswith("_")} for item in kept
        )

    score_by_object_id = {
        k["track_id"]: float(k["score"]) for kept in kept_by_segment.values() for k in kept
    }
    merge_summary = _apply_physion_pp_same_object_merge(
        all_records=all_records,
        all_arrays=all_arrays,
        score_by_object_id=score_by_object_id,
        evidence_kinds=frozenset(
            same_object_merge.require_strings("default_evidence")
        ),
    )
    role_assignment = _bw_assign_roles_and_purge(
        video=video,
        frames=frames,
        transition=transition,
        all_records=all_records,
        all_arrays=all_arrays,
        bind_minimum_coverage=cue_role_binding.require_number(
            "minimum_coverage"
        ),
        wall_position_iou=role_binding.require_number("wall_position_iou"),
        static_pair_iou=role_binding.require_number("static_pair_iou"),
    )

    debug_segment_videos: dict[str, str] = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1":
        debug_segment_videos = _bw_write_segment_debug_videos(
            video=video, frames=frames, transition=transition, output=output
        )

    tracking = {
        "mode": "tracking.gdino_two_segment_point_probe",
        "scenario": scenario_name,
        "two_segment": transition,
        "gdino": gdino_summaries,
        "window_box_filter_dropped": window_drops,
        "probe_dropped": {
            seg: [
                {k: v for k, v in item.items() if not k.startswith("_")}
                for item in dropped
            ]
            for seg, dropped in dropped_by_segment.items()
        },
        "same_object_merge": merge_summary,
        "role_binding": role_assignment,
        "debug_segment_videos": debug_segment_videos,
    }
    return payload_prompts, [], tracking


def run_sam3_video_tracks(
    *,
    video: Path,
    object_plan: Path,
    output: Path,
    checkpoint: str | None,
    sam3_version: str,
    mode: str,
    bench: str | None,
    generic_prompt: str,
    prompt_frame_index: int,
    compile_model: bool,
    max_num_objects: int,
    async_loading_frames: bool,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    model: Any | None = None,
    use_fa3: bool | None = None,
) -> None:
    task_start = time.perf_counter()
    metadata = _video_metadata(video)
    object_plan_payload = json.loads(object_plan.read_text(encoding="utf-8"))
    physion_scenario_name = _physion_pp_scenario_name(
        object_plan_payload=object_plan_payload,
    )
    is_physion_pp = _is_physion_pp_scenario_name(physion_scenario_name)
    if is_physion_pp:
        prompts = _physion_main_prompts({})
        resolved_mode = "physion-multi-round"
    else:
        prompts, resolved_mode = _select_prompts(
            object_plan_payload=object_plan_payload,
            mode=mode,
            bench=bench,
            generic_prompt=generic_prompt,
        )
    tracking_family_record = _resolve_tracking_family_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    temporal_partition_record = _resolve_temporal_partition_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    cue_role_binding_record = _resolve_cue_role_binding_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    static_role_assignment_record = _resolve_static_role_assignment_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    cross_segment_identity_record = _resolve_cross_segment_identity_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    tracking_family_route = (
        tracking_family_record.get("route")
        if tracking_family_record is not None
        else None
    )
    temporal_partition_route = (
        temporal_partition_record.get("route")
        if temporal_partition_record is not None
        else None
    )
    tracking_dispatch = _tracking_dispatch_name(
        is_physion_pp=is_physion_pp,
        scenario_name=physion_scenario_name,
        tracking_family_route=tracking_family_route,
        temporal_partition_route=temporal_partition_route,
    )
    tracking_module_profile = None
    if tracking_dispatch in {
        "friction_platform_tracking",
        "bouncy_platform_tracking",
        "bouncy_wall_tracking",
        "mass_collision_tracking",
    }:
        tracking_module_profile = default_module_profile_policy().resolve_tracking(
            str(tracking_family_route),
            scenario=str(physion_scenario_name),
        )
    if not prompts:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "tool": "sam3_video_tracks",
                    "status": "skipped",
                    "reason": "no_bench_or_vlm_concept_prompts",
                    "video": str(video),
                    "object_plan": str(object_plan),
                    "bench": bench,
                    "requested_mode": mode,
                    "resolved_mode": resolved_mode,
                    "physion_scenario": physion_scenario_name,
                    "tracking_family_route": tracking_family_record,
                    "temporal_partition_route": temporal_partition_record,
                    "cue_role_binding_route": cue_role_binding_record,
                    "static_role_assignment_route": static_role_assignment_record,
                    "cross_segment_identity_route": cross_segment_identity_record,
                    "prompts": [],
                    "track_record_count": 0,
                    "tracked_frame_count": 0,
                    "track_count_by_object": {},
                    "tracks": [],
                    "tracks_by_object": {},
                    "note": "No bench-level concepts or VLM-provided SAM3 video tracking concepts were available.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    if metadata["frame_count"] <= 0:
        raise ValueError(f"Could not read frame count from video: {video}")
    if not (0 <= prompt_frame_index < metadata["frame_count"]):
        raise ValueError(
            f"prompt_frame_index={prompt_frame_index} is outside valid range [0, {metadata['frame_count'] - 1}]"
        )
    if max_num_objects < len(prompts):
        raise ValueError(f"max_num_objects={max_num_objects} is smaller than prompt count {len(prompts)}")

    if model is None:
        model, use_fa3 = _load_model(
            checkpoint=checkpoint,
            sam3_version=sam3_version,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
            async_loading_frames=async_loading_frames,
        )
    all_records: list[dict[str, Any]] = []
    all_arrays: dict[str, np.ndarray] = {}
    physion_tracking: dict[str, Any] | None = None
    if tracking_dispatch == "mass_collision_tracking":
        if tracking_module_profile is None:
            raise ValueError("mass-collision tracking has no module profile")
        payload_prompts, payload_prompt_responses, physion_tracking = _run_physion_pp_mass_collision_tracking(
            model=model,
            video=video,
            scenario_name=str(physion_scenario_name),
            output=output,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=tracking_module_profile,
        )
    elif tracking_dispatch == "friction_collision_tracking":
        payload_prompts, payload_prompt_responses, physion_tracking = _run_physion_pp_two_segment_text_tracking(
            model=model,
            video=video,
            scenario_name=str(physion_scenario_name),
            output=output,
            all_records=all_records,
            all_arrays=all_arrays,
        )
    elif tracking_dispatch == "bouncy_wall_tracking":
        if tracking_module_profile is None:
            raise ValueError("bouncy-wall tracking has no module profile")
        payload_prompts, payload_prompt_responses, physion_tracking = _run_physion_pp_two_segment_tracking(
            model=model,
            video=video,
            scenario_name=str(physion_scenario_name),
            output=output,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=tracking_module_profile,
        )
    elif tracking_dispatch == "bouncy_platform_tracking":
        if tracking_module_profile is None:
            raise ValueError("bouncy-platform tracking has no module profile")
        payload_prompts, payload_prompt_responses, physion_tracking = _run_physion_pp_bouncy_platform_tracking(
            model=model,
            video=video,
            scenario_name=str(physion_scenario_name),
            max_frame_num_to_track=max_frame_num_to_track,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=tracking_module_profile,
        )
    elif tracking_dispatch == "friction_platform_tracking":
        if tracking_module_profile is None:
            raise ValueError("friction-platform tracking has no module profile")
        payload_prompts, payload_prompt_responses, physion_tracking = _run_physion_pp_friction_platform_tracking(
            model=model,
            video=video,
            scenario_name=str(physion_scenario_name),
            prompt_frame_index=prompt_frame_index,
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            all_records=all_records,
            all_arrays=all_arrays,
            module_profile=tracking_module_profile,
        )
    elif tracking_dispatch == "joint_prompt_tracking":
        prompt_responses = _run_joint_prompt_tracking(
            model=model,
            video=video,
            prompts=prompts,
            resolved_mode=resolved_mode,
            prompt_frame_index=prompt_frame_index,
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            all_records=all_records,
            all_arrays=all_arrays,
        )
        payload_prompts = list(prompts)
        payload_prompt_responses = list(prompt_responses)
    else:
        raise ValueError(f"unsupported tracking dispatch: {tracking_dispatch!r}")

    if cue_role_binding_record is not None:
        _record_cue_role_binding_result(
            physion_tracking,
            cue_role_binding_record,
        )
    if static_role_assignment_record is not None:
        _record_static_role_assignment_result(
            physion_tracking,
            static_role_assignment_record,
        )
    if cross_segment_identity_record is not None:
        _record_cross_segment_identity_result(
            physion_tracking,
            cross_segment_identity_record,
        )

    sidecar_path = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if all_arrays:
        np.savez_compressed(sidecar_path, **all_arrays)

    records_by_object: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        records_by_object.setdefault(str(record["object_id"]), []).append(record)

    debug_artifacts: dict[str, Any] = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1" and all_arrays:
        debug_videos = write_sam3_video_track_overlay_videos(
            video=video,
            records_by_object=records_by_object,
            mask_sidecar=sidecar_path,
            output_dir=output.parent / "debug" / "videos",
            fps=float(metadata.get("fps") or 0.0),
        )
        debug_artifacts["track_overlay_videos"] = debug_videos

    payload = {
        "tool": "sam3_video_tracks",
        "status": "ok",
        "version": sam3_version,
        "mode": mode,
        "video": str(video),
        "object_plan": str(object_plan),
        "checkpoint": checkpoint,
        "bench": bench,
        "requested_mode": mode,
        "resolved_mode": resolved_mode,
        "physion_scenario": physion_scenario_name,
        "tracking_family_route": tracking_family_record,
        "temporal_partition_route": temporal_partition_record,
        "cue_role_binding_route": cue_role_binding_record,
        "static_role_assignment_route": static_role_assignment_record,
        "cross_segment_identity_route": cross_segment_identity_record,
        "tracking_mode": (
            "tracking.gdino_two_segment_boxes"
            if is_physion_pp
            and tracking_family_route == MASS_COLLISION_TRACKING_FAMILY_ROUTE
            else "tracking.sam3_two_segment_text"
            if is_physion_pp and _is_physion_pp_two_segment_text_scenario(physion_scenario_name)
            else "physion_pp_two_segment_gdino_tracks"
            if is_physion_pp
            and tracking_family_route == BOUNCY_WALL_TRACKING_FAMILY_ROUTE
            else "friction_platform_role_specific_box_point_tracks"
            if is_physion_pp
            and tracking_family_route == FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE
            else "physion_multi_round_merged_tracks"
            if is_physion_pp
            else "tracking.sam3_broad_text"
        ),
        "prompt_frame_index": int(prompt_frame_index),
        "propagation_direction": propagation_direction,
        "max_frame_num_to_track": max_frame_num_to_track,
        "compile": bool(compile_model),
        "async_loading_frames": bool(async_loading_frames),
        "max_num_objects": int(max_num_objects),
        "use_fa3": bool(use_fa3),
        "video_metadata": metadata,
        "target_object_count": len(payload_prompts),
        "generic_prompt": generic_prompt if resolved_mode == "generic-movable" else None,
        "prompts": payload_prompts,
        "prompt_responses": payload_prompt_responses,
        "physion_tracking": physion_tracking,
        "track_record_count": len(all_records),
        "tracked_frame_count": len({record["frame_index"] for record in all_records}),
        "tracked_frame_indices": sorted({record["frame_index"] for record in all_records}),
        "track_count_by_object": {object_id: len(records) for object_id, records in records_by_object.items()},
        "mask_sidecar": str(sidecar_path) if all_arrays else None,
        "tracks": all_records,
        "tracks_by_object": records_by_object,
        "debug_artifacts": debug_artifacts,
        "elapsed_sec": time.perf_counter() - task_start,
        "note": "SAM3 video tracking output for Object Segmentation and Event Detection.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sam3-version", choices=["sam3", "sam3.1"], default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "bench-concepts", "object-prompts", "generic-movable", "vlm-concept-prompts"],
        default="auto",
    )
    parser.add_argument("--bench", default=None)
    parser.add_argument(
        "--generic-prompt",
        default="all movable objects on the floor, including balls, cylinders, and cubes",
    )
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=8)
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument(
        "--propagation-direction",
        choices=["both", "forward", "backward"],
        default="forward",
    )
    parser.add_argument("--max-frame-num-to-track", type=int, default=None)
    args = parser.parse_args()
    run_sam3_video_tracks(
        video=Path(args.video),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        checkpoint=args.checkpoint,
        sam3_version=args.sam3_version or (
            "sam3" if args.mode in {"auto", "bench-concepts", "generic-movable", "vlm-concept-prompts"} else "sam3.1"
        ),
        mode=args.mode,
        bench=args.bench,
        generic_prompt=args.generic_prompt,
        prompt_frame_index=args.prompt_frame_index,
        compile_model=args.compile,
        max_num_objects=args.max_num_objects,
        async_loading_frames=args.async_loading_frames,
        propagation_direction=args.propagation_direction,
        max_frame_num_to_track=args.max_frame_num_to_track,
    )


if __name__ == "__main__":
    main()
