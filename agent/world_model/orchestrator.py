from __future__ import annotations

from copy import deepcopy
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Dict, Optional, Sequence, Set, Union

import numpy as np
from tqdm import tqdm

from agent.world_model.answer import (
    SUCCESS_ANSWER_BACKEND_DECISION_ID,
    SUCCESS_ANSWER_BACKEND_ROUTES,
    WORLD_MODEL_ANSWER_FALLBACK_DECISION_ID,
    WORLD_MODEL_ANSWER_FALLBACK_ROUTE,
    extracted_final_answer,
    run_direct_answer_fallback,
    run_final_answer,
)
from agent.world_model.artifacts import ArtifactManager
from agent.world_model.foundationpose_worker import FoundationPoseWorkerClient
from agent.world_model.geocalib_worker import GeoCalibWorkerClient
from agent.world_model.moge2_worker import MoGe2WorkerClient
from agent.world_model.planner import (
    SCENE_VLM_HORIZONTAL_ROLL_SUPPORT_BUNDLE_ROUTE,
    build_dry_run_scene_object_plan,
    build_track_bootstrap_scene_object_plan,
    build_vlm_scene_assessment,
)
from agent.world_model.sam3d_worker import SAM3DWorkerClient
from agent.world_model.sam3_video_tracks_worker import SAM3VideoTracksWorkerClient
from agent.world_model.schemas import ObjectPlan, TargetObject, ToolResult, WorldModelQuestionResult
from agent.world_model.simulation import QueryConditionedPhysicalRolloutAdapter
from agent.world_model.tools import (
    OBJECT_INVENTORY_SOURCE_DECISION_ID,
    TRACK_BOOTSTRAP_THEN_TRACK_DERIVED_INVENTORY_ROUTE,
    build_default_adapters,
)
from agent.world_model.video_metric_depth_worker import VideoMetricDepthWorkerClient
from benchmark.clevrer import ClevrerQuestion, ClevrerScene, load_validation_scenes
from benchmark.metrics import (
    compute_metrics,
    compute_physion_pp_metrics,
    normalize_descriptive_answer,
)
from utils.config import ModelConfig
from utils.run import ensure_run_dir, write_json
from utils.terminal import terminal_print


OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE = "object-segmentation-and-event-detection"
WORLD_MODEL_VIDEO_SOURCE_DECISION_ID = "INP-001.world_model_video_source"
ANSWER_FALLBACK_VIDEO_SOURCE_DECISION_ID = "INP-002.answer_fallback_video_source"
TEMPORAL_PARTITION_DECISION_ID = "INP-003.temporal_partition"
SPLIT_TEMPORAL_PARTITION_ROUTE = "temporal.cue_two_segment"
TEMPORAL_PARTITION_ROUTES = frozenset(
    {
        "temporal.single_video",
        SPLIT_TEMPORAL_PARTITION_ROUTE,
    }
)
SCENE_ASSESSMENT_BUNDLE_DECISION_ID = "SCN-001.scene_assessment_bundle"
TRACKING_FAMILY_DECISION_ID = "TRK-001.tracking_family"
TRACKING_FAMILY_ROUTES = frozenset(
    {
        "tracking.sam3_broad_text",
        "tracking.gdino_role_box_point",
        "tracking.gdino_two_segment_point_probe",
        "tracking.gdino_per_part_seeds",
        "tracking.sam3_two_segment_text",
        "tracking.gdino_two_segment_boxes",
    }
)
CUE_ROLE_BINDING_DECISION_ID = "TRK-002.cue_role_binding"
CUE_ROLE_BINDING_ROUTE = "role_binding.cue_rounds"
STATIC_ROLE_ASSIGNMENT_DECISION_ID = "TRK-003.static_role_assignment"
STATIC_ROLE_ASSIGNMENT_ROUTES = frozenset(
    {
        "static_roles.platform_displacement",
        "static_roles.wall_patient_two_segment",
        "static_roles.bounce_platform_displacement",
        "static_roles.collision_structure",
        "static_roles.mass_structure",
    }
)
CROSS_SEGMENT_IDENTITY_DECISION_ID = "TRK-004.cross_segment_identity"
CROSS_SEGMENT_IDENTITY_SCENARIOS = frozenset(
    {
        "bouncy_wall_pp",
        "friction_collision_pp",
        "mass_collision_pp",
    }
)
CROSS_SEGMENT_IDENTITY_ROUTES = frozenset(
    {"identity.vlm_ab"}
)
MASS_EXTRA_PATIENT_LINK_DECISION_ID = "TRK-005.mass_extra_patient_link"
MASS_EXTRA_PATIENT_LINK_ROUTE = "role_link.vlm_identity"
TRACK_VLM_LABELING_DECISION_ID = "LBL-001.track_vlm_labeling"
CLEVRER_TRACK_VLM_LABELING_ROUTE = (
    "label.vlm_geometry_basic"
)
TRACK_VLM_LABELING_ROUTE_BY_BENCHMARK = {
    "clevrer": CLEVRER_TRACK_VLM_LABELING_ROUTE,
    "physion_pp": PHYSION_PP_TRACK_VLM_LABELING_ROUTE,
}
FALSE_POSITIVE_ADVICE_EFFECT_DECISION_ID = "LBL-002.false_positive_advice_effect"
FALSE_POSITIVE_ADVICE_EFFECT_ROUTE = (
    "label.false_positive_advice_only"
)
INTRINSICS_BACKEND_DECISION_ID = "DEP-001.intrinsics_backend"
INTRINSICS_BACKEND_ROUTE = "intrinsics.moge2_fixed_camera"
DEPTH_PARTITION_DECISION_ID = "DEP-002.depth_partition"
SPLIT_DEPTH_PARTITION_ROUTE = "depth.vda_two_segment_affine"
DEPTH_PARTITION_ROUTES = frozenset(
    {
        "depth.vda_full_video",
        SPLIT_DEPTH_PARTITION_ROUTE,
    }
)
SEGMENT_DEPTH_ALIGNMENT_DECISION_ID = "DEP-003.segment_depth_alignment"
SEGMENT_DEPTH_ALIGNMENT_ROUTE = (
    "depth.affine_background_curtain"
)
SAM3D_OBSERVATION_SELECTION_DECISION_ID = "GEO-001.sam3d_observation_selection"
SAM3D_OBSERVATION_SELECTION_ROUTES = frozenset(
    {
        "geometry.static_temporal_composite",
        "geometry.selected_observation",
    }
)
CROSS_SEGMENT_MESH_REUSE_DECISION_ID = "GEO-002.cross_segment_mesh_reuse"
CROSS_SEGMENT_MESH_REUSE_ROUTES = frozenset(
    {
        "mesh_reuse.same_role_identity",
        "mesh_reuse.ball_agent_patient_conditional",
    }
)
MESH_CONDITIONING_DECISION_ID = "GEO-004.mesh_conditioning"
CLEVRER_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.aabb_no_guard"
)


POSE_INPUT_AND_FOUNDATIONPOSE_DECISION_ID = (
    "POS-001.pose_input_and_foundationpose"
)
POSE_INPUT_AND_FOUNDATIONPOSE_ROUTE = (
    "pose.mask_select_validate_track"
)
GROUND_MOTION_GATE_DECISION_ID = "POS-002.ground_motion_gate"
GROUND_MOTION_GATE_ROUTES = frozenset(
    {
        "ground.vlm_horizontal_motion",
        "ground.agent_airborne_fixture_ground",
        "ground.collision_forced_true",
    }
)
GRAVITY_ESTIMATOR_DECISION_ID = "POS-003.gravity_estimator"
GRAVITY_ESTIMATOR_ROUTES = frozenset(
    {
        "gravity.overlap_aabb_post_snap",
        "gravity.overlap_obb_pre_snap",
        "gravity.geocalib_8frame_z_negative",
        "gravity.geocalib_8frame",
    }
)
GRAVITY_CONSTRAINTS_DECISION_ID = "POS-004.gravity_roll_and_z_constraints"
GRAVITY_CONSTRAINT_ROUTES = frozenset(
    {
        "gravity_constraints.overlap_roll",
        "gravity_constraints.roll_zero_z_negative",
        "gravity_constraints.roll_zero",
    }
)
ROTATION_POLICY_DECISION_ID = "POS-005.rotation_policy"
ROTATION_POLICY_ROUTE = "rotation.geometry_conditioned"
SUPPORT_SNAP_DECISION_ID = "POS-006.support_snap"
SUPPORT_SNAP_ROUTES = frozenset(
    {
        "support.ray_slide_ground",
        "support.platform_sphere_agent",
        "support.wall_sphere_agent",
        "support.agent_exempt",
        "support.moving_patient_exempt",
        "support.patient_exempt",
    }
)
STATIC_FIXTURE_FLUSH_DECISION_ID = "POS-007.static_fixture_flush"
STATIC_FIXTURE_FLUSH_ROUTES = frozenset(
    {
        "flush.static_fixture",
        "flush.segment1_patient",
        "flush.linked_extra",
    }
)
LINE_LAYOUT_DECISION_ID = "POS-008.line_layout"
LINE_LAYOUT_ROUTES = frozenset(
    {
        "layout.line_static_orientation",
        "layout.not_applicable",
    }
)
FOUNDATIONPOSE_AGENT_GEOMETRY_DECISION_ID = (
    "GEO-003.foundationpose_agent_geometry"
)
FOUNDATIONPOSE_AGENT_GEOMETRY_ROUTES = frozenset(
    {
        "agent_geometry.sphere_no_foundationpose",
        "agent_geometry.native_foundationpose",
    }
)
FRICTION_PLATFORM_AGENT_TRAJECTORY_DECISION_ID = (
    "POS-009.friction_platform_agent_trajectory"
)
BOUNCY_PLATFORM_AGENT_TRAJECTORY_DECISION_ID = (
    "POS-010.bouncy_platform_agent_trajectory"
)
BOUNCY_WALL_AGENT_TRAJECTORY_DECISION_ID = (
    "POS-011.bouncy_wall_agent_trajectory"
)
AGENT_TRAJECTORY_DECISION_BY_SCENARIO = {
    "friction_platform_pp": FRICTION_PLATFORM_AGENT_TRAJECTORY_DECISION_ID,
    "bouncy_platform_pp": BOUNCY_PLATFORM_AGENT_TRAJECTORY_DECISION_ID,
    "bouncy_wall_pp": BOUNCY_WALL_AGENT_TRAJECTORY_DECISION_ID,
}
AGENT_TRAJECTORY_ROUTES = frozenset(
    {
        "trajectory.platform_sphere",
        "trajectory.native_ray",
        "trajectory.bounce_mask",
        "trajectory.wall_mixed_geometry",
        "trajectory.wall_native",
    }
)
COLLISION_PATIENT_MOTION_GATE_DECISION_ID = (
    "POS-012.collision_patient_motion_gate"
)
COLLISION_PATIENT_MOTION_GATE_ROUTE = (
    "motion.mask_centroid_5px"
)
COLLISION_PATIENT_DROP_DECISION_ID = "POS-013.collision_patient_drop"
COLLISION_PATIENT_DROP_ROUTE = "trajectory.patient_vertical_drop"
MASS_EXTRA_MESH_ADOPT_DECISION_ID = (
    "POS-014.mass_extra_flush_and_mesh_adopt"
)
MASS_EXTRA_MESH_ADOPT_ROUTE = (
    "mesh_adopt.linked_extra_to_patient"
)
MASS_AGENT_FLUSH_DECISION_ID = "POS-015.mass_agent_flush"
MASS_AGENT_FLUSH_ROUTE = "flush.mass_agent"
MASS_BALL_TRAJECTORY_DECISION_ID = (
    "POS-016.mass_ball_precontact_trajectory"
)
MASS_BALL_TRAJECTORY_ROUTE = (
    "trajectory.ball_precontact_mask_line"
)
COLLISION_PATIENT_SCENARIOS = frozenset(
    {"friction_collision_pp", "mass_collision_pp"}
)
SWR_FIT_BACKEND_DECISION_ID = "SWR-001.fit_backend"
SWR_FIT_BACKEND_ROUTES = frozenset(
    {
        "swr_backend.impulse_analytic",
        "swr_backend.surface_friction_sphere",
        "swr_backend.wall_bounce_sphere",
        "swr_backend.platform_bounce_sphere",
        "swr_backend.collision_friction_spheres",
        "swr_backend.collision_mass_spheres",
    }
)
SWR_FIT_STRATEGY_DECISION_ID = "SWR-002.fit_strategy"
SWR_FIT_STRATEGY_ROUTES = frozenset(
    {
        "swr_fit.bounded_plane_dynamic_sphere",
        "swr_fit.two_segment_shared_speed",
        "swr_fit.bounded_planes_full_trajectory",
        "swr_fit.two_segment_shared_speed_radii",
        "swr_fit.ball_agent_impulse_mass",
    }
)
SWR_FIT_GEOMETRY_SOURCE_DECISION_ID = "SWR-003.fit_geometry_source"
SWR_FIT_GEOMETRY_SOURCE_ROUTE = "geometry.corrected_mesh"
SWR_VISUAL_POSE_PRESERVATION_DECISION_ID = (
    "SWR-004.visual_pose_preservation"
)
SWR_VISUAL_POSE_PRESERVATION_ROUTES = frozenset(
    {
        "visual_pose.position_only",
        "visual_pose.corrected_rotation",
    }
)
QUESTION_ROLLOUT_BACKEND_DECISION_ID = "ROL-001.question_rollout_backend"
QUESTION_ROLLOUT_BACKEND_ROUTES = frozenset(
    {
        "rollout.tool_physics",
        "rollout.surface_friction_analytic",
        "rollout.wall_bounce_analytic",
        "rollout.platform_bounce_analytic",
        "rollout.collision_friction_analytic",
        "rollout.collision_mass_analytic",
    }
)
CLEVRER_TOOL_PLANNING_DECISION_ID = "ROL-002.clevrer_tool_planning"
CLEVRER_TOOL_PLANNING_ROUTE = "planning.two_round_tools"
CHOICE_LETTER_PATTERN = re.compile(r"[A-Z]")
STOP_AFTER_STAGE_TO_TOOL = {
    "object-planning": "object_plan",
    OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE: "sam3_video_track_labels",
    "metric-mesh-reconstruction": "mesh_conditioning",
    "pose-tracking": "pose_correction",
    "simulatable-world-reconstruction": "simulatable_world_reconstruction",
    "query-conditioned-physical-rollout": "query_conditioned_physical_rollout",
    "answering": "final_answer",
    "evaluation": None,
}
STAGE_ORDER = tuple(STOP_AFTER_STAGE_TO_TOOL)
TOOL_TO_DISPLAY_STAGE = {
    "object_plan": "object-planning",
    "sam3_video_tracks": OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE,
    "sam3_video_track_labels": OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE,
    "moge2_intrinsics": "metric-mesh-reconstruction",
    "sam3d_objects": "metric-mesh-reconstruction",
    "mesh_conditioning": "metric-mesh-reconstruction",
    "video_metric_depth": "metric-mesh-reconstruction",
    "pose_frames": "pose-tracking",
    "pose_sam3_masks": "pose-tracking",
    "pose_sam3_mask_selection": "pose-tracking",
    "pose_frame_boundary_validation": "pose-tracking",
    "foundationpose": "pose-tracking",
    "pose_correction": "pose-tracking",
    "simulatable_world_reconstruction": "simulatable-world-reconstruction",
    "query_conditioned_physical_rollout": "query-conditioned-physical-rollout",
    "final_answer": "artifact-based-answering",
}


















def _resolve_tracking_family(
    benchmark: str,
    *,
    scenario: str | None = None,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        TRACKING_FAMILY_DECISION_ID,
        RouteContext(benchmark=benchmark, scenario=scenario),
    )
    route = resolved_route["route"]
    if route not in TRACKING_FAMILY_ROUTES:
        raise RoutePolicyValidationError(
            f"unsupported tracking-family route: {route!r}"
        )
    return resolved_route
















def _resolve_mass_extra_patient_link(
    benchmark: str,
    *,
    scenario: str,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        MASS_EXTRA_PATIENT_LINK_DECISION_ID,
        RouteContext(benchmark=benchmark, scenario=scenario),
    )
    route = resolved_route["route"]
    if route != MASS_EXTRA_PATIENT_LINK_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported mass-extra-patient-link route: {route!r}"
        )
    return resolved_route








def _resolve_false_positive_advice_effect(
    benchmark: str,
    *,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        FALSE_POSITIVE_ADVICE_EFFECT_DECISION_ID,
        RouteContext(benchmark=benchmark),
    )
    route = resolved_route["route"]
    if route != FALSE_POSITIVE_ADVICE_EFFECT_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported false-positive-advice-effect route: {route!r}"
        )
    return resolved_route












def _resolve_segment_depth_alignment(
    benchmark: str,
    *,
    scenario: str,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        SEGMENT_DEPTH_ALIGNMENT_DECISION_ID,
        RouteContext(benchmark=benchmark, scenario=scenario),
    )
    route = resolved_route["route"]
    if route != SEGMENT_DEPTH_ALIGNMENT_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported segment-depth-alignment route: {route!r}"
        )
    return resolved_route




def _resolve_sam3d_observation_selection(
    benchmark: str,
    *,
    scenario: str | None = None,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        SAM3D_OBSERVATION_SELECTION_DECISION_ID,
        RouteContext(benchmark=benchmark, scenario=scenario),
    )
    route = resolved_route["route"]
    if route not in SAM3D_OBSERVATION_SELECTION_ROUTES:
        raise RoutePolicyValidationError(
            f"unsupported SAM3D-observation-selection route: {route!r}"
        )
    return resolved_route












def _resolve_pose_input_and_foundationpose(
    benchmark: str,
    *,
    scenario: str | None = None,
    policy: RoutePolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        POSE_INPUT_AND_FOUNDATIONPOSE_DECISION_ID,
        RouteContext(benchmark=benchmark, scenario=scenario),
    )
    route = resolved_route["route"]
    if route != POSE_INPUT_AND_FOUNDATIONPOSE_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported pose-input-and-FoundationPose route: {route!r}"
        )
    return resolved_route




































































def _runs_through_stage(stop_after_stage: str, required_stage: str) -> bool:
    return STAGE_ORDER.index(stop_after_stage) >= STAGE_ORDER.index(required_stage)


def _display_stage(tool_name: str) -> str:
    return TOOL_TO_DISPLAY_STAGE.get(tool_name, tool_name.replace("_", "-"))


def _normalize_log_message(message: str) -> str:
    replacements = {
        "physics_alignment_blender_render": "debug_render",
        "physics_alignment command": "world_reconstruction_fit command",
        "physics_alignment_manifest": "world_reconstruction_manifest",
        "trajectory_informed_physics_alignment": "trajectory_informed_world_reconstruction",
    }
    value = message
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _log(stage: str, message: str) -> None:
    message = _normalize_log_message(message)
    terminal_print(f"[{stage}] {message}", flush=True)


def _log_tool(tool_name: str, message: str) -> None:
    _log(_display_stage(tool_name), f"tool={tool_name} {message}")


def _short_text(text: Optional[str], limit: int = 1200) -> str:
    if not text:
        return ""
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _stage_summary(stage: str, payload: Dict[str, Any]) -> str:
    if stage == "object_plan":
        objects = payload.get("target_objects") or []
        ids = [item.get("object_id") for item in objects if isinstance(item, dict)]
        scene_objects = payload.get("scene_objects") if isinstance(payload.get("scene_objects"), dict) else {}
        static_objects = scene_objects.get("static_objects") if isinstance(scene_objects, dict) else []
        special_scene = payload.get("special_scene") if isinstance(payload.get("special_scene"), dict) else {}
        horizontal = (
            special_scene.get("horizontal_plane_motion")
            if isinstance(special_scene.get("horizontal_plane_motion"), dict)
            else {}
        )
        if payload.get("dynamic_objects_pending"):
            return (
                f"dynamic_objects=pending source={payload.get('dynamic_objects_source')} "
                f"static_objects={len(static_objects or [])} "
                f"horizontal_plane_motion={horizontal.get('applies')}"
            )
        return (
            f"track_derived_objects={len(objects)} static_objects={len(static_objects or [])} "
            f"horizontal_plane_motion={horizontal.get('applies')} "
            f"object_ids={ids}"
        )
    if stage == "pose_correction":
        rotation = (
            payload.get("rotation_correction")
            if isinstance(payload.get("rotation_correction"), dict)
            else {}
        )
        position = (
            payload.get("support_plane_position_correction")
            if isinstance(payload.get("support_plane_position_correction"), dict)
            else {}
        )
        return (
            f"stage={payload.get('pose_correction_stage')} "
            f"sampled_frame_indices={payload.get('sampled_frame_indices')} "
            f"gravity_direction_camera={payload.get('gravity_direction_camera')} "
            f"rotation_applied={rotation.get('applied')} position_applied={position.get('applied')} "
        )
    if stage == "simulatable_world_reconstruction":
        trajectories = payload.get("corrected_trajectories") if isinstance(payload.get("corrected_trajectories"), dict) else {}
        return (
            f"stage={payload.get('simulatable_world_reconstruction_stage')} "
            f"trajectory_source={trajectories.get('source')} "
            f"objects={len(trajectories.get('objects') or [])}"
        )
    if stage == "sam3_video_tracks":
        return (
            f"mode={payload.get('mode')} prompts={len(payload.get('prompts') or [])} "
            f"tracks={payload.get('track_record_count')} frames={payload.get('tracked_frame_count')} "
            f"track_count_by_object={payload.get('track_count_by_object')}"
        )
    if stage == "sam3_video_track_labels":
        labels = payload.get("track_labels") or []
        object_keyframes = payload.get("object_keyframes") or []
        objects = [item for item in labels if isinstance(item, dict) and item.get("object_id")]
        suggested_fp = [
            item for item in labels if isinstance(item, dict) and item.get("vlm_false_positive_suggestion") is True
        ]
        return (
            f"labels={len(labels)} objects={len(objects)} "
            f"object_keyframes={len(object_keyframes)} "
            f"vlm_false_positive_suggestions={len(suggested_fp)} "
            f"image_count={len(payload.get('image_paths') or [])}"
        )
    if stage == "query_conditioned_physical_rollout":
        trajectories = payload.get("simulated_trajectories") or payload.get("trajectories") or []
        if not trajectories and isinstance(payload.get("predictions"), list):
            trajectories = payload["predictions"]
        backend = payload.get("backend") or payload.get("tool")
        tool_calls = (payload.get("tool_call_request") or {}).get("tool_calls") or []
        edited_rollouts = payload.get("edited_rollouts") or []
        tool_planning = payload.get("tool_planning") or {}
        return (
            f"backend={backend} trajectories={len(trajectories)} "
            f"tool_calls={len(tool_calls)} edited_rollouts={len(edited_rollouts)} "
            f"planner={tool_planning.get('status')} fit_error={payload.get('fit_error')}"
        )
    if stage == "final_answer":
        error = _short_text(str(payload.get("error_message") or ""), limit=200)
        suffix = f" error={error}" if error else ""
        return f"status={payload.get('status')} extracted_answer={payload.get('extracted_answer')}{suffix}"
    return f"keys={sorted(payload.keys())[:12]}"


def _stage_result_ref(result: ToolResult) -> Dict[str, object]:
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "artifact_path": result.artifact_path,
        "message": result.message,
        "elapsed_sec": result.elapsed_sec,
    }


def _record_timing(result: ToolResult, elapsed: float) -> ToolResult:
    result.elapsed_sec = elapsed
    return result








def _elapsed_text(result: ToolResult) -> str:
    elapsed = result.elapsed_sec or 0.0
    return f"elapsed={elapsed:.1f}s"


def _allowed_questions(question: ClevrerQuestion, allowed_question_types: Set[str]) -> bool:
    return not allowed_question_types or question.question_type.lower() in allowed_question_types


def _filter_scenes_by_ids(
    scenes: Sequence[ClevrerScene],
    scene_ids: Optional[Sequence[int]],
    *,
    benchmark_name: str = "CLEVRER",
) -> list[ClevrerScene]:
    if scene_ids is None:
        return list(scenes)
    requested = [int(scene_id) for scene_id in scene_ids]
    requested_set = set(requested)
    filtered = [scene for scene in scenes if scene.scene_index in requested_set]
    found = {scene.scene_index for scene in filtered}
    missing = [scene_id for scene_id in requested if scene_id not in found]
    if missing:
        raise ValueError(f"{benchmark_name} scene_ids not found: {missing}")
    return filtered


def _derive_final_status(stage_results: Sequence[ToolResult], dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    if any(result.status == "tool_error" for result in stage_results):
        return "tool_error"
    if any(result.status == "tool_not_configured" for result in stage_results):
        return "tool_not_configured"
    if any(result.status == "parse_error" for result in stage_results):
        return "parse_error"
    return "ok"


def _pipeline_failure_from_stage_results(stage_results: Sequence[ToolResult]) -> dict[str, Any]:
    for result in stage_results:
        if result.status not in {"tool_error", "tool_not_configured", "parse_error", "agent_error"}:
            continue
        error_message = result.message or result.payload.get("error_message")
        failure = {
            "failed_stage": _display_stage(result.tool_name),
            "failed_tool": result.tool_name,
            "status": result.status,
            "error_message": error_message,
            "artifact_path": result.artifact_path,
        }
        if result.payload.get("raw_response") is not None:
            failure["raw_response"] = result.payload["raw_response"]
        return failure
    return {
        "failed_stage": "pipeline",
        "failed_tool": None,
        "status": "agent_error",
        "error_message": "Pipeline failed without a failed ToolResult.",
        "artifact_path": None,
    }


def _stage_succeeded(result: ToolResult) -> bool:
    return result.status in {"ok", "loaded", "dry_run"}


def _question_status(*, request_error: bool, parse_error: bool, is_correct: bool) -> str:
    if request_error:
        return "request_error"
    if parse_error:
        return "parse_error"
    if is_correct:
        return "correct"
    return "wrong_answer"


def _extract_choice_letters(answer: Optional[str]) -> Set[str]:
    if answer is None:
        return set()
    normalized = answer.strip().upper()
    if normalized == "NONE":
        return set()
    letters = set()
    for part in re.split(r"[\s,]+", normalized):
        token = part.strip()
        if len(token) == 1 and CHOICE_LETTER_PATTERN.fullmatch(token):
            letters.add(token)
    return letters


def _metric_question_from_result(question: ClevrerQuestion, result: WorldModelQuestionResult) -> Dict[str, object]:
    request_error = result.status in {"tool_error", "tool_not_configured", "agent_error"}
    parse_error = result.status == "parse_error" or (not request_error and result.final_answer is None)
    if question.question_type == "descriptive":
        expected = normalize_descriptive_answer(question.answer or "")
        normalized_prediction = normalize_descriptive_answer(result.final_answer or "")
        is_correct = (not request_error) and (not parse_error) and normalized_prediction == expected
        return {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "prediction": result.final_answer,
            "usage": None,
            "extracted_answer": result.final_answer,
            "normalized_prediction": normalized_prediction,
            "expected_answer": question.answer,
            "normalized_expected_answer": expected,
            "request_error": request_error,
            "parse_error": parse_error,
            "error_message": result.error_message,
            "is_correct": is_correct,
            "status": _question_status(request_error=request_error, parse_error=parse_error, is_correct=is_correct),
        }

    predicted_letters = _extract_choice_letters(result.final_answer)
    choice_predictions = []
    for index, choice in enumerate(question.choices):
        label = chr(ord("A") + index)
        normalized_prediction = "correct" if label in predicted_letters else "wrong"
        expected = "correct" if (choice.answer or "").strip().lower() == "correct" else "wrong"
        choice_predictions.append(
            {
                "choice_id": choice.choice_id,
                "choice_letter": label,
                "choice": choice.choice,
                "prediction": normalized_prediction,
                "normalized_prediction": normalized_prediction,
                "expected_answer": choice.answer,
                "normalized_expected_answer": expected,
                "is_correct": (not request_error) and (not parse_error) and normalized_prediction == expected,
            }
        )
    is_correct = (not request_error) and (not parse_error) and all(choice["is_correct"] for choice in choice_predictions)
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "raw_prediction": result.final_answer,
        "usage": None,
        "predicted_letters": sorted(predicted_letters),
        "request_error": request_error,
        "parse_error": parse_error,
        "error_message": result.error_message,
        "is_correct": is_correct,
        "status": _question_status(request_error=request_error, parse_error=parse_error, is_correct=is_correct),
        "choices": choice_predictions,
    }


def _physion_pp_metric_question_from_result(
    question: PhysionPPQuestion,
    result: WorldModelQuestionResult,
) -> Dict[str, object]:
    request_error = result.status in {"tool_error", "tool_not_configured", "agent_error"}
    normalized_prediction = str(result.final_answer or "").strip().lower()
    parse_error = (
        result.status == "parse_error"
        or (not request_error and normalized_prediction not in {"yes", "no"})
    )
    expected = str(question.answer).strip().lower()
    is_correct = (
        not request_error
        and not parse_error
        and normalized_prediction == expected
    )
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "prediction": result.final_answer,
        "raw_prediction": result.final_answer,
        "usage": None,
        "extracted_answer": result.final_answer,
        "normalized_prediction": normalized_prediction,
        "expected_answer": question.answer,
        "normalized_expected_answer": expected,
        "ground_truth_outcome": question.ground_truth_outcome,
        "request_error": request_error,
        "parse_error": parse_error,
        "error_message": result.error_message,
        "is_correct": is_correct,
        "status": _question_status(
            request_error=request_error,
            parse_error=parse_error,
            is_correct=is_correct,
        ),
    }








class _StopAfterStage(Exception):
    pass


class WorldModelAgent:
    def __init__(
        self,
        *,
        config: ModelConfig,
        run_dir: Path,
        dry_run: bool,
        stop_after_stage: str,
        debug_artifacts: bool = False,
        persistent_workers: bool = True,
        answer_enabled: bool = True,
        route_policy: RoutePolicy | None = None,
        enabled_route_options: Sequence[str] = (),
    ):
        self.config = config
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.stop_after_stage = stop_after_stage
        self.stop_after_tool = STOP_AFTER_STAGE_TO_TOOL[stop_after_stage]
        self.artifacts = ArtifactManager(
            run_dir,
            debug_artifacts=debug_artifacts,
        )
        self.answer_enabled = answer_enabled
        self.persistent_workers = persistent_workers
        self.route_policy = route_policy or load_route_policy()
        self.enabled_route_options = tuple(
            option.option_id
            for option in self.route_policy.enabled_optional_routes(
                enabled_route_options
            )
        )
        self.video_metric_depth_worker: Optional[VideoMetricDepthWorkerClient] = None
        self.foundationpose_worker: Optional[FoundationPoseWorkerClient] = None
        self.sam3d_worker: Optional[SAM3DWorkerClient] = None
        self.sam3_video_tracks_worker: Optional[SAM3VideoTracksWorkerClient] = None
        self.geocalib_worker: Optional[GeoCalibWorkerClient] = None
        self.moge2_worker: Optional[MoGe2WorkerClient] = None
        self.model_startup_times: Dict[str, Any] = {}
        self.scene_object_plans: Dict[int, tuple[ObjectPlan, Path]] = {}
        self.scene_stage_results: Dict[int, list[ToolResult]] = {}
        self.scene_world_dirs: Dict[int, Path] = {}
        self.scene_world_modeling_failures: Dict[int, Dict[str, Any]] = {}
        self.scene_reconstruction_elapsed_sec: Dict[int, float] = {}
        self.scene_timings: list[Dict[str, Any]] = []
        self._stop_after_scene_index: Optional[int] = None
        should_start_workers = self.persistent_workers and not self.dry_run
        if should_start_workers and _runs_through_stage(
            self.stop_after_stage,
            OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE,
        ):
            self.sam3_video_tracks_worker = SAM3VideoTracksWorkerClient(run_dir)
            _log("run", "worker_start tool=sam3_video_tracks mode=persistent")
            startup_info = self.sam3_video_tracks_worker.start()
            self.model_startup_times["sam3_video_tracks"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=sam3_video_tracks startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
        if should_start_workers and _runs_through_stage(self.stop_after_stage, "metric-mesh-reconstruction"):
            self.video_metric_depth_worker = VideoMetricDepthWorkerClient(run_dir)
            _log("run", "worker_start tool=video_metric_depth mode=persistent")
            startup_info = self.video_metric_depth_worker.start()
            self.model_startup_times["video_metric_depth"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=video_metric_depth startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
            self.sam3d_worker = SAM3DWorkerClient(run_dir)
            _log("run", "worker_start tool=sam3d mode=persistent")
            startup_info = self.sam3d_worker.start()
            self.model_startup_times["sam3d"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=sam3d startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
            self.moge2_worker = MoGe2WorkerClient(run_dir)
            _log("run", "worker_start tool=moge2 mode=persistent")
            startup_info = self.moge2_worker.start()
            self.model_startup_times["moge2"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=moge2 startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
        if should_start_workers and _runs_through_stage(self.stop_after_stage, "pose-tracking"):
            self.foundationpose_worker = FoundationPoseWorkerClient(run_dir)
            _log("run", "worker_start tool=foundationpose mode=persistent")
            startup_info = self.foundationpose_worker.start()
            self.model_startup_times["foundationpose"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=foundationpose startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
            self.geocalib_worker = GeoCalibWorkerClient(run_dir)
            _log("run", "worker_start tool=geocalib mode=persistent")
            startup_info = self.geocalib_worker.start()
            self.model_startup_times["geocalib"] = startup_info
            self.artifacts.write(run_dir / "model_startup_times.json", self.model_startup_times)
            _log(
                "run",
                f"worker_ready tool=geocalib startup_elapsed={startup_info.get('startup_elapsed_sec', 0.0):.1f}s",
            )
        self.adapters = build_default_adapters(
            self.artifacts,
            config=config,
            dry_run=dry_run,
            video_metric_depth_worker=self.video_metric_depth_worker,
            foundationpose_worker=self.foundationpose_worker,
            sam3d_worker=self.sam3d_worker,
            sam3_video_tracks_worker=self.sam3_video_tracks_worker,
            geocalib_worker=self.geocalib_worker,
            moge2_worker=self.moge2_worker,
        )
        self.simulator = QueryConditionedPhysicalRolloutAdapter(self.artifacts, config=config, dry_run=dry_run)

    def _scene_object_plan_path(self, scene: ClevrerScene) -> Path:
        return (
            self.run_dir
            / "artifacts"
            / f"scene_{scene.scene_index}"
            / "world-modeling"
            / "object-identification-and-planning"
            / "object_plan"
            / "object_plan.json"
        )

    def _get_scene_object_plan(self, scene: ClevrerScene) -> tuple[ObjectPlan, Path, float, bool]:
        stored = self.scene_object_plans.get(scene.scene_index)
        if stored is not None:
            object_plan, object_plan_path = self._refresh_scene_object_plan_cache(scene=scene, fallback=stored[0])
            return object_plan, object_plan_path, 0.0, True

        stage_start = time.perf_counter()
        _log("object-planning", f"start scene={scene.scene_index} scope=scene")
        scene_assessment = None
        object_inventory_route = None
        temporal_partition_route = None
        tracking_family_route = None
        cue_role_binding_route = None
        static_role_assignment_route = None
        cross_segment_identity_route = None
        mass_extra_patient_link_route = None
        track_vlm_labeling_route = None
        false_positive_advice_effect_route = None
        intrinsics_backend_route = None
        depth_partition_route = None
        segment_depth_alignment_route = None
        sam3d_observation_selection_route = None
        cross_segment_mesh_reuse_route = None
        mesh_conditioning_route = None
        pose_input_and_foundationpose_route = None
        ground_motion_gate_route = None
        gravity_estimator_route = None
        gravity_constraints_route = None
        rotation_policy_route = None
        support_snap_route = None
        static_fixture_flush_route = None
        line_layout_route = None
        foundationpose_agent_geometry_route = None
        agent_trajectory_route = None
        collision_patient_motion_gate_route = None
        collision_patient_drop_route = None
        mass_extra_mesh_adopt_route = None
        mass_agent_flush_route = None
        mass_ball_trajectory_route = None
        swr_fit_backend_route = None
        swr_fit_strategy_route = None
        swr_fit_geometry_source_route = None
        swr_visual_pose_preservation_route = None
        if not self.dry_run:
            policy_benchmark = _pipeline_policy_benchmark(scene)
            scene_assessment_route = None
            if policy_benchmark is not None:
                object_inventory_route = _resolve_object_inventory_source(
                    policy_benchmark
                )
                track_vlm_labeling_route = _resolve_track_vlm_labeling(
                    policy_benchmark
                )
                false_positive_advice_effect_route = (
                    _resolve_false_positive_advice_effect(policy_benchmark)
                )
                intrinsics_backend_route = _resolve_intrinsics_backend(
                    policy_benchmark
                )
                scenario = str(getattr(scene, "scenario", "") or "").strip() or None
                temporal_partition_route = _resolve_temporal_partition(
                    policy_benchmark,
                    scenario=scenario,
                )
                if temporal_partition_route["route"] == SPLIT_TEMPORAL_PARTITION_ROUTE:
                    cross_segment_mesh_reuse_route = (
                        _resolve_cross_segment_mesh_reuse(
                            policy_benchmark,
                            scenario=str(scenario or ""),
                        )
                    )
                depth_partition_route = _resolve_depth_partition(
                    policy_benchmark,
                    scenario=scenario,
                )
                sam3d_observation_selection_route = (
                    _resolve_sam3d_observation_selection(
                        policy_benchmark,
                        scenario=scenario,
                    )
                )
                mesh_conditioning_route = _resolve_mesh_conditioning(
                    policy_benchmark,
                    scenario=scenario,
                )
                pose_input_and_foundationpose_route = (
                    _resolve_pose_input_and_foundationpose(
                        policy_benchmark,
                        scenario=scenario,
                    )
                )
                ground_motion_gate_route = _resolve_ground_motion_gate(
                    policy_benchmark,
                    scenario=scenario,
                )
                gravity_estimator_route = _resolve_gravity_estimator(
                    policy_benchmark,
                    scenario=scenario,
                )
                gravity_constraints_route = _resolve_gravity_constraints(
                    policy_benchmark,
                    scenario=scenario,
                )
                rotation_policy_route = _resolve_rotation_policy(
                    policy_benchmark,
                    scenario=scenario,
                )
                support_snap_route = _resolve_support_snap(
                    policy_benchmark,
                    scenario=scenario,
                )
                swr_fit_backend_route = _resolve_swr_fit_backend(
                    policy_benchmark,
                    scenario=scenario,
                    policy=self.route_policy,
                )
                if policy_benchmark == "physion_pp":
                    swr_fit_strategy_route = _resolve_swr_fit_strategy(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                        policy=self.route_policy,
                    )
                    swr_fit_geometry_source_route = (
                        _resolve_swr_fit_geometry_source(
                            policy_benchmark,
                            scenario=str(scenario or ""),
                            policy=self.route_policy,
                        )
                    )
                    normalized_scenario = str(scenario or "").strip().lower()
                    if normalized_scenario in COLLISION_PATIENT_SCENARIOS:
                        swr_visual_pose_preservation_route = (
                            _resolve_swr_visual_pose_preservation(
                                policy_benchmark,
                                scenario=normalized_scenario,
                                enabled_route_options=self.enabled_route_options,
                                policy=self.route_policy,
                            )
                        )
                if depth_partition_route["route"] == SPLIT_DEPTH_PARTITION_ROUTE:
                    segment_depth_alignment_route = (
                        _resolve_segment_depth_alignment(
                            policy_benchmark,
                            scenario=str(scenario or ""),
                        )
                    )
                tracking_family_route = _resolve_tracking_family(
                    policy_benchmark,
                    scenario=scenario,
                )
                if policy_benchmark == "physion_pp":
                    foundationpose_agent_geometry_route = (
                        _resolve_foundationpose_agent_geometry(
                            policy_benchmark,
                            scenario=str(scenario or ""),
                            enabled_route_options=self.enabled_route_options,
                            policy=self.route_policy,
                        )
                    )
                    agent_trajectory_route = _resolve_agent_trajectory(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                        enabled_route_options=self.enabled_route_options,
                        policy=self.route_policy,
                    )
                    static_fixture_flush_route = _resolve_static_fixture_flush(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                    )
                    line_layout_route = _resolve_line_layout(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                    )
                    cue_role_binding_route = _resolve_cue_role_binding(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                    )
                    static_role_assignment_route = _resolve_static_role_assignment(
                        policy_benchmark,
                        scenario=str(scenario or ""),
                    )
                    normalized_scenario = str(scenario or "").strip().lower()
                    if normalized_scenario in COLLISION_PATIENT_SCENARIOS:
                        collision_patient_motion_gate_route = (
                            _resolve_collision_patient_motion_gate(
                                policy_benchmark,
                                scenario=normalized_scenario,
                                policy=self.route_policy,
                            )
                        )
                        collision_patient_drop_route = (
                            _resolve_collision_patient_drop(
                                policy_benchmark,
                                scenario=normalized_scenario,
                                policy=self.route_policy,
                            )
                        )
                    if normalized_scenario in CROSS_SEGMENT_IDENTITY_SCENARIOS:
                        cross_segment_identity_route = _resolve_cross_segment_identity(
                            policy_benchmark,
                            scenario=normalized_scenario,
                        )
                    if normalized_scenario == "mass_collision_pp":
                        mass_extra_patient_link_route = _resolve_mass_extra_patient_link(
                            policy_benchmark,
                            scenario=normalized_scenario,
                        )
                        mass_extra_mesh_adopt_route = _resolve_mass_extra_mesh_adopt(
                            policy_benchmark,
                            scenario=normalized_scenario,
                            policy=self.route_policy,
                        )
                        mass_agent_flush_route = _resolve_mass_agent_flush(
                            policy_benchmark,
                            scenario=normalized_scenario,
                            policy=self.route_policy,
                        )
                        mass_ball_trajectory_route = _resolve_mass_ball_trajectory(
                            policy_benchmark,
                            scenario=normalized_scenario,
                            policy=self.route_policy,
                        )
                scene_assessment_route = _resolve_scene_assessment_bundle(
                    policy_benchmark
                )
            scene_assessment = build_vlm_scene_assessment(
                self.config,
                scene,
                route=(
                    scene_assessment_route["route"]
                    if scene_assessment_route is not None
                    else SCENE_VLM_HORIZONTAL_ROLL_SUPPORT_BUNDLE_ROUTE
                ),
            )
            if scene_assessment_route is not None:
                scene_assessment["resolved_route"] = scene_assessment_route
            _log(
                "object-planning",
                f"scene_assessment scene={scene.scene_index} status={scene_assessment.get('status')} "
                f"horizontal_plane_motion={scene_assessment.get('special_scene', {}).get('horizontal_plane_motion', {}).get('applies')} "
                f"roll_stabilization={scene_assessment.get('special_scene', {}).get('roll_stabilization', {}).get('applies')}",
            )
        object_plan = (
            build_dry_run_scene_object_plan(scene)
            if self.dry_run
            else (
                _build_routed_track_bootstrap_object_plan(
                    scene,
                    scene_assessment=scene_assessment,
                    resolved_route=object_inventory_route,
                )
                if object_inventory_route is not None
                else build_track_bootstrap_scene_object_plan(
                    scene,
                    scene_assessment=scene_assessment,
                )
            )
        )
        if tracking_family_route is not None:
            _record_tracking_family_route(object_plan, tracking_family_route)
        if temporal_partition_route is not None:
            _record_temporal_partition_route(
                object_plan,
                temporal_partition_route,
            )
        if cross_segment_mesh_reuse_route is not None:
            _record_cross_segment_mesh_reuse_route(
                object_plan,
                cross_segment_mesh_reuse_route,
            )
        if mesh_conditioning_route is not None:
            _record_mesh_conditioning_route(
                object_plan,
                mesh_conditioning_route,
            )
        if pose_input_and_foundationpose_route is not None:
            _record_pose_input_and_foundationpose_route(
                object_plan,
                pose_input_and_foundationpose_route,
            )
        if ground_motion_gate_route is not None:
            _record_ground_motion_gate_route(
                object_plan,
                ground_motion_gate_route,
            )
        if gravity_estimator_route is not None:
            _record_gravity_route(object_plan, gravity_estimator_route)
        if gravity_constraints_route is not None:
            _record_gravity_route(object_plan, gravity_constraints_route)
        if rotation_policy_route is not None:
            _record_rotation_policy_route(object_plan, rotation_policy_route)
        if support_snap_route is not None:
            _record_pose_route(
                object_plan,
                support_snap_route,
                decision_id=SUPPORT_SNAP_DECISION_ID,
                supported_routes=SUPPORT_SNAP_ROUTES,
                key="support_snap_route",
            )
        if static_fixture_flush_route is not None:
            _record_pose_route(
                object_plan,
                static_fixture_flush_route,
                decision_id=STATIC_FIXTURE_FLUSH_DECISION_ID,
                supported_routes=STATIC_FIXTURE_FLUSH_ROUTES,
                key="static_fixture_flush_route",
            )
        if line_layout_route is not None:
            _record_pose_route(
                object_plan,
                line_layout_route,
                decision_id=LINE_LAYOUT_DECISION_ID,
                supported_routes=LINE_LAYOUT_ROUTES,
                key="line_layout_route",
            )
        for route_record, decision_id, route, key in (
            (
                collision_patient_motion_gate_route,
                COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
                COLLISION_PATIENT_MOTION_GATE_ROUTE,
                "collision_patient_motion_gate_route",
            ),
            (
                collision_patient_drop_route,
                COLLISION_PATIENT_DROP_DECISION_ID,
                COLLISION_PATIENT_DROP_ROUTE,
                "collision_patient_drop_route",
            ),
            (
                mass_extra_mesh_adopt_route,
                MASS_EXTRA_MESH_ADOPT_DECISION_ID,
                MASS_EXTRA_MESH_ADOPT_ROUTE,
                "mass_extra_flush_and_mesh_adopt_route",
            ),
            (
                mass_agent_flush_route,
                MASS_AGENT_FLUSH_DECISION_ID,
                MASS_AGENT_FLUSH_ROUTE,
                "mass_agent_flush_route",
            ),
            (
                mass_ball_trajectory_route,
                MASS_BALL_TRAJECTORY_DECISION_ID,
                MASS_BALL_TRAJECTORY_ROUTE,
                "mass_ball_precontact_trajectory_route",
            ),
        ):
            if route_record is not None:
                _record_pose_route(
                    object_plan,
                    route_record,
                    decision_id=decision_id,
                    supported_routes=frozenset({route}),
                    key=key,
                )
        if foundationpose_agent_geometry_route is not None:
            _record_agent_geometry_and_trajectory_routes(
                object_plan,
                geometry_route=foundationpose_agent_geometry_route,
                trajectory_route=agent_trajectory_route,
            )
        if swr_fit_backend_route is not None:
            _record_swr_fit_backend_route(object_plan, swr_fit_backend_route)
        if swr_fit_strategy_route is not None:
            _record_swr_fit_strategy_route(object_plan, swr_fit_strategy_route)
        if swr_fit_geometry_source_route is not None:
            _record_swr_fit_geometry_source_route(
                object_plan,
                swr_fit_geometry_source_route,
            )
        if swr_visual_pose_preservation_route is not None:
            _record_swr_visual_pose_preservation_route(
                object_plan,
                swr_visual_pose_preservation_route,
            )
        if cue_role_binding_route is not None:
            _record_cue_role_binding_route(object_plan, cue_role_binding_route)
        if static_role_assignment_route is not None:
            _record_static_role_assignment_route(
                object_plan,
                static_role_assignment_route,
            )
        if cross_segment_identity_route is not None:
            _record_cross_segment_identity_route(
                object_plan,
                cross_segment_identity_route,
            )
        if mass_extra_patient_link_route is not None:
            _record_mass_extra_patient_link_route(
                object_plan,
                mass_extra_patient_link_route,
            )
        if track_vlm_labeling_route is not None:
            _record_track_vlm_labeling_route(
                object_plan,
                track_vlm_labeling_route,
            )
        if false_positive_advice_effect_route is not None:
            _record_false_positive_advice_effect_route(
                object_plan,
                false_positive_advice_effect_route,
            )
        if intrinsics_backend_route is not None:
            _record_intrinsics_backend_route(
                object_plan,
                intrinsics_backend_route,
            )
        if depth_partition_route is not None:
            _record_depth_partition_route(
                object_plan,
                depth_partition_route,
            )
        if segment_depth_alignment_route is not None:
            _record_segment_depth_alignment_route(
                object_plan,
                segment_depth_alignment_route,
            )
        if sam3d_observation_selection_route is not None:
            _record_sam3d_observation_selection_route(
                object_plan,
                sam3d_observation_selection_route,
            )
        object_plan_path = self._scene_object_plan_path(scene)
        payload = object_plan.to_dict()
        payload["scope"] = "scene"
        self.artifacts.write(object_plan_path, payload)
        object_plan_elapsed = time.perf_counter() - stage_start
        self.scene_object_plans[scene.scene_index] = (object_plan, object_plan_path)
        self.scene_reconstruction_elapsed_sec[scene.scene_index] = (
            self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0) + object_plan_elapsed
        )
        _log(
            "object-planning",
            f"end scene={scene.scene_index} scope=scene status={object_plan.status} "
            f"elapsed={object_plan_elapsed:.1f}s artifact={object_plan_path} "
            f"{_stage_summary('object_plan', payload)}",
        )
        self.artifacts.append_trace(
            {
                "event": "scene_object_plan_complete",
                "scene_index": scene.scene_index,
                "status": object_plan.status,
                "artifact_path": str(object_plan_path),
                "elapsed_sec": object_plan_elapsed,
            }
        )
        return object_plan, object_plan_path, object_plan_elapsed, False

    def _object_plan_from_payload(self, payload: Dict[str, Any], fallback: ObjectPlan) -> ObjectPlan:
        target_objects = []
        for item in payload.get("target_objects") or []:
            if not isinstance(item, dict):
                continue
            target_objects.append(
                TargetObject(
                    object_id=str(item.get("object_id") or f"obj_{len(target_objects) + 1}"),
                    description=str(item.get("description") or ""),
                    role=str(item.get("role") or ""),
                    geometry_type=str(item.get("geometry_type") or "irregular"),
                    geometry_confidence=item.get("geometry_confidence"),
                    appearance=deepcopy(item.get("appearance") if isinstance(item.get("appearance"), dict) else {}),
                    source_track_id=(
                        str(item.get("source_track_id"))
                        if item.get("source_track_id") is not None
                        else None
                    ),
                )
            )
        if not target_objects:
            return fallback
        return ObjectPlan(
            scene_index=int(payload.get("scene_index", fallback.scene_index)),
            question_id=int(payload.get("question_id", fallback.question_id)),
            question_type=str(payload.get("question_type") or fallback.question_type),
            question=str(payload.get("question") or fallback.question),
            choices=deepcopy(payload.get("choices") if isinstance(payload.get("choices"), list) else fallback.choices),
            target_objects=target_objects,
            reasoning=str(payload.get("reasoning") or fallback.reasoning),
            scene_objects=deepcopy(
                payload.get("scene_objects") if isinstance(payload.get("scene_objects"), dict) else fallback.scene_objects
            ),
            special_scene=deepcopy(
                payload.get("special_scene") if isinstance(payload.get("special_scene"), dict) else fallback.special_scene
            ),
            status=str(payload.get("status") or fallback.status),
            error_message=payload.get("error_message"),
        )

    def _refresh_scene_object_plan_cache(
        self,
        *,
        scene: ClevrerScene,
        fallback: ObjectPlan,
    ) -> tuple[ObjectPlan, Path]:
        object_plan_path = self._scene_object_plan_path(scene)
        payload = self.artifacts.read_optional(object_plan_path) or {}
        object_plan = self._object_plan_from_payload(payload, fallback) if payload else fallback
        self.scene_object_plans[scene.scene_index] = (object_plan, object_plan_path)
        return object_plan, object_plan_path

    def _question_object_plan_from_scene_plan(
        self,
        *,
        scene_plan: ObjectPlan,
        question: ClevrerQuestion,
    ) -> ObjectPlan:
        return ObjectPlan(
            scene_index=scene_plan.scene_index,
            question_id=question.question_id,
            question_type=question.question_type,
            question="",
            choices=[],
            target_objects=deepcopy(scene_plan.target_objects),
            reasoning=scene_plan.reasoning,
            scene_objects=deepcopy(scene_plan.scene_objects),
            special_scene=deepcopy(scene_plan.special_scene),
            status=scene_plan.status,
            error_message=scene_plan.error_message,
        )

    def _scene_world_dir(self, scene: ClevrerScene) -> Path:
        path = self.run_dir / "artifacts" / f"scene_{scene.scene_index}" / "world-modeling"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_or_reuse_scene_stages(
        self,
        *,
        scene: ClevrerScene,
        question: ClevrerQuestion,
        object_plan: ObjectPlan,
        stage_results: list[ToolResult],
    ) -> None:
        stored_results = self.scene_stage_results.get(scene.scene_index)
        if stored_results is not None:
            scene_world_dir = self.scene_world_dirs[scene.scene_index]
            _log(
                "world-modeling",
                f"reuse scene={scene.scene_index} question={question.question_id} "
                f"source={scene_world_dir}",
            )
            for result in deepcopy(stored_results):
                result.elapsed_sec = 0.0
                result.payload["scene_reuse"] = {
                    "reused": True,
                    "source_world_model_dir": str(scene_world_dir),
                }
                stage_results.append(result)
                self._maybe_stop_after(result.tool_name, scene, question)
                if not _stage_succeeded(result):
                    self._record_world_modeling_failure(scene=scene, result=result)
                    break
            return

        scene_world_dir = self._scene_world_dir(scene)
        self.scene_world_dirs[scene.scene_index] = scene_world_dir
        for adapter in self.adapters:
            stage_start = time.perf_counter()
            _log_tool(adapter.tool_name, f"start scene={scene.scene_index} scope=scene")
            result = adapter.run(scene=scene, question_dir=scene_world_dir, object_plan=object_plan)
            stage_elapsed = time.perf_counter() - stage_start
            result = _record_timing(result, stage_elapsed)
            stage_results.append(result)
            self.scene_stage_results[scene.scene_index] = deepcopy(stage_results)
            self.scene_reconstruction_elapsed_sec[scene.scene_index] = (
                self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0) + stage_elapsed
            )
            message = _short_text(result.message)
            _log_tool(
                result.tool_name,
                f"end scene={scene.scene_index} scope=scene "
                f"status={result.status} {_elapsed_text(result)} "
                f"artifact={result.artifact_path} {_stage_summary(result.tool_name, result.payload)}",
            )
            if message:
                _log_tool(result.tool_name, f"message scene={scene.scene_index} scope=scene {message}")
            self._refresh_scene_object_plan_cache(scene=scene, fallback=object_plan)
            self.artifacts.append_trace(
                {
                    "event": "scene_stage_complete",
                    "stage": result.tool_name,
                    "scene_index": scene.scene_index,
                    "status": result.status,
                    "artifact_path": result.artifact_path,
                    "elapsed_sec": result.elapsed_sec,
                }
            )
            self._maybe_stop_after(result.tool_name, scene, question)
            if not _stage_succeeded(result):
                self._record_world_modeling_failure(scene=scene, result=result)
                _log(
                    "world-modeling",
                    f"stop scene={scene.scene_index} failed_stage={result.tool_name} "
                    f"status={result.status}",
                )
                break

    def _record_world_modeling_failure(self, *, scene: ClevrerScene, result: ToolResult) -> None:
        if scene.scene_index in self.scene_world_modeling_failures:
            return
        payload_error = result.payload.get("error_message") if isinstance(result.payload, dict) else None
        self.scene_world_modeling_failures[scene.scene_index] = {
            "status": "error",
            "failed_stage": result.tool_name,
            "failed_status": result.status,
            "failed_artifact_path": result.artifact_path,
            "error_message": result.message or payload_error,
        }

    def _world_modeling_failure_result(self, *, scene: ClevrerScene) -> ToolResult | None:
        failure = self.scene_world_modeling_failures.get(scene.scene_index)
        if not failure:
            return None
        return ToolResult(
            "world_modeling",
            "tool_error",
            failure.get("failed_artifact_path"),
            message=failure.get("error_message"),
            payload={
                "tool": "world_modeling",
                "status": "error",
                **failure,
            },
        )

    def close(self) -> None:
        if self.video_metric_depth_worker is not None:
            self.video_metric_depth_worker.close()
            self.video_metric_depth_worker = None
        if self.foundationpose_worker is not None:
            self.foundationpose_worker.close()
            self.foundationpose_worker = None
        if self.sam3_video_tracks_worker is not None:
            self.sam3_video_tracks_worker.close()
            self.sam3_video_tracks_worker = None
        if self.sam3d_worker is not None:
            self.sam3d_worker.close()
            self.sam3d_worker = None
        if self.geocalib_worker is not None:
            self.geocalib_worker.close()
            self.geocalib_worker = None
        if self.moge2_worker is not None:
            self.moge2_worker.close()
            self.moge2_worker = None

    def _maybe_stop_after(self, tool_name: str, scene: ClevrerScene, question: ClevrerQuestion) -> None:
        if self.stop_after_tool == tool_name:
            self._stop_after_scene_index = scene.scene_index
            _log(
                _display_stage(tool_name),
                f"stop scene={scene.scene_index} question={question.question_id} "
                f"after_tool={tool_name} requested={self.stop_after_stage}",
            )
            raise _StopAfterStage

    def stop_after_reached_for_scene(self, scene: ClevrerScene) -> bool:
        return self._stop_after_scene_index == scene.scene_index

    def _skip_final_answer(self, scene: ClevrerScene, question: ClevrerQuestion, question_dir: Path) -> ToolResult:
        output_path = self.artifacts.artifact_path(question_dir, "final_answer", "final_answer.json")
        payload = {
            "tool": "final_answer",
            "status": "skipped",
            "tool_status": "skipped",
            "reason": "answer_not_implemented_for_benchmark",
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "extracted_answer": None,
            "note": "World modeling and query-conditioned rollout ran, but final benchmark answering is not implemented for this benchmark yet.",
        }
        self.artifacts.write(output_path, payload)
        return ToolResult("final_answer", "skipped", str(output_path), payload=payload)

    def _cleanup_question_like_dir(self, question_dir: Path) -> None:
        if not question_dir.is_dir():
            return
        pose_sam3_dir = self.artifacts.tool_dir(question_dir, "pose_sam3_masks")
        _remove_tree(pose_sam3_dir / "frames")
        _remove_tree(pose_sam3_dir / "candidates")
        _remove_tree(self.artifacts.tool_dir(question_dir, "sam3d_objects") / "meshes")
        mesh_conditioning_dir = self.artifacts.tool_dir(question_dir, "mesh_conditioning")
        _remove_tree(mesh_conditioning_dir / "debug_projection")
        _remove_tree(self.artifacts.tool_dir(question_dir, "moge2_intrinsics") / "sampled_frames")
        video_metric_depth_dir = self.artifacts.tool_dir(question_dir, "video_metric_depth")
        _remove_tree(video_metric_depth_dir / "video_metric_frames")
        _trim_video_metric_depth_sidecar(video_metric_depth_dir / "video_metric_depth.json")
        _remove_tree(self.artifacts.tool_dir(question_dir, "simulatable_world_reconstruction") / "debug")

    def cleanup_scene(self, scene: ClevrerScene) -> None:
        scene_dir = self.run_dir / "artifacts" / f"scene_{scene.scene_index}"
        if not scene_dir.exists():
            return
        for question_dir in scene_dir.glob("question_*"):
            self._cleanup_question_like_dir(question_dir)
            _remove_empty_dirs(question_dir)
        self._cleanup_question_like_dir(scene_dir / "world-modeling")
        _remove_empty_dirs(scene_dir / "world-modeling")
        _log("run", f"cleanup scene={scene.scene_index} removed temporary artifacts")

    def _run_pipeline_direct_answer_fallback(
        self,
        *,
        scene: ClevrerScene | PhysionPPScene,
        question: ClevrerQuestion | PhysionPPQuestion,
        question_dir: Path,
        stage_results: list[ToolResult],
        pipeline_failure: dict[str, Any],
    ) -> ToolResult:
        stage_start = time.perf_counter()
        _log(
            "artifact-based-answering",
            f"direct_answer_fallback scene={scene.scene_index} question={question.question_id} "
            f"failed_stage={pipeline_failure.get('failed_stage')}",
        )
        fallback_benchmark = _pipeline_policy_benchmark(scene)
        if fallback_benchmark is None:
            raise RoutePolicyValidationError(
                f"unsupported answer-fallback scene type: {type(scene).__name__}"
            )
        fallback_video_route = _resolve_answer_fallback_video_source(
            fallback_benchmark
        )
        fallback_route = _resolve_world_model_answer_fallback(
            fallback_benchmark,
            scenario=(
                str(getattr(scene, "scenario", "") or "").strip()
                or None
            ),
            question_type=(
                str(question.question_type or "").strip() or None
            ),
            policy=self.route_policy,
        )
        result = run_direct_answer_fallback(
            config=self.config,
            scene=scene,
            question=question,
            question_dir=question_dir,
            artifacts=self.artifacts,
            dry_run=self.dry_run,
            pipeline_failure=pipeline_failure,
            video_source_route=fallback_video_route["route"],
            world_model_answer_fallback_route=fallback_route,
        )
        result = _record_timing(result, time.perf_counter() - stage_start)
        stage_results.append(result)
        message = _short_text(result.message)
        _log_tool(
            result.tool_name,
            f"fallback end scene={scene.scene_index} question={question.question_id} "
            f"status={result.status} {_elapsed_text(result)} "
            f"artifact={result.artifact_path} "
            f"{_stage_summary(result.tool_name, result.payload)}",
        )
        if message:
            _log_tool(
                result.tool_name,
                f"message scene={scene.scene_index} question={question.question_id} {message}",
            )
        return result

    def process_question(self, scene: ClevrerScene, question: ClevrerQuestion) -> WorldModelQuestionResult:
        question_dir = self.artifacts.question_dir(scene.scene_index, question.question_id)
        question_start = time.perf_counter()
        scene_reconstruction_before = self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0)
        stage_results: list[ToolResult] = []
        answer_result = None
        fallback_attempted = False
        fallback_used = False
        active_stage = "object-planning"
        _log(
            "question",
            f"start scene={scene.scene_index} question={question.question_id} "
            f"type={question.question_type} artifact_dir={question_dir}",
        )
        self.artifacts.append_trace(
            {
                "event": "question_start",
                "scene_index": scene.scene_index,
                "question_id": question.question_id,
                "question_type": question.question_type,
            }
        )
        try:
            scene_object_plan, scene_object_plan_path, object_plan_elapsed, object_plan_scene_reuse = (
                self._get_scene_object_plan(scene)
            )
            object_plan = self._question_object_plan_from_scene_plan(
                scene_plan=scene_object_plan,
                question=question,
            )
            _log(
                "object-planning",
                f"end scene={scene.scene_index} question={question.question_id} "
                f"status={object_plan.status} elapsed={object_plan_elapsed:.1f}s "
                f"scene_reuse={object_plan_scene_reuse} artifact={scene_object_plan_path} "
                f"{_stage_summary('object_plan', object_plan.to_dict())}",
            )

            self._maybe_stop_after("object_plan", scene, question)
            active_stage = "world-modeling"
            self._run_or_reuse_scene_stages(
                scene=scene,
                question=question,
                object_plan=scene_object_plan,
                stage_results=stage_results,
            )
            scene_object_plan, _ = self._refresh_scene_object_plan_cache(
                scene=scene,
                fallback=scene_object_plan,
            )
            object_plan = self._question_object_plan_from_scene_plan(
                scene_plan=scene_object_plan,
                question=question,
            )
            world_modeling_failure = self.scene_world_modeling_failures.get(scene.scene_index)
            if world_modeling_failure is not None:
                failure_result = self._world_modeling_failure_result(scene=scene)
                if failure_result is not None and not any(
                    item.tool_name == "world_modeling" for item in stage_results
                ):
                    stage_results.append(failure_result)
                fallback_attempted = True
                fallback_used = True
                answer_result = self._run_pipeline_direct_answer_fallback(
                    scene=scene,
                    question=question,
                    question_dir=question_dir,
                    stage_results=stage_results,
                    pipeline_failure=world_modeling_failure,
                )
                self._maybe_stop_after(answer_result.tool_name, scene, question)
                status = "dry_run" if self.dry_run else answer_result.status
                question_wall_elapsed = time.perf_counter() - question_start
                scene_reconstruction_total = self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0)
                question_scene_reconstruction_elapsed = max(
                    0.0,
                    scene_reconstruction_total - scene_reconstruction_before,
                )
                question_without_scene_reconstruction_elapsed = max(
                    0.0,
                    question_wall_elapsed - question_scene_reconstruction_elapsed,
                )
                result = WorldModelQuestionResult(
                    scene_index=scene.scene_index,
                    video_filename=scene.video_filename,
                    question_id=question.question_id,
                    question_type=question.question_type,
                    question=question.question,
                    status=status,
                    final_answer=extracted_final_answer(answer_result),
                    error_message=None if status in {"ok", "dry_run"} else answer_result.message,
                    artifact_dir=str(question_dir),
                    stages=[_stage_result_ref(item) for item in stage_results],
                    elapsed_with_scene_reconstruction_sec=(
                        question_without_scene_reconstruction_elapsed + scene_reconstruction_total
                    ),
                    elapsed_without_scene_reconstruction_sec=question_without_scene_reconstruction_elapsed,
                )
                result_path = question_dir / "artifact-based-answering" / "result.json"
                self.artifacts.write(result_path, result.to_dict())
                self.artifacts.append_trace(
                    {
                        "event": "question_complete",
                        "scene_index": scene.scene_index,
                        "question_id": question.question_id,
                        "status": result.status,
                        "fallback_used": True,
                        "world_modeling_failed": True,
                        "world_modeling_failed_stage": world_modeling_failure.get("failed_stage"),
                        "elapsed_with_scene_reconstruction_sec": result.elapsed_with_scene_reconstruction_sec,
                        "elapsed_without_scene_reconstruction_sec": result.elapsed_without_scene_reconstruction_sec,
                    }
                )
                elapsed_with_scene = result.elapsed_with_scene_reconstruction_sec or 0.0
                elapsed_without_scene = result.elapsed_without_scene_reconstruction_sec or 0.0
                _log(
                    "question",
                    f"end scene={scene.scene_index} question={question.question_id} "
                    f"status={result.status} final_answer={result.final_answer} "
                    f"world_modeling_failed_stage={world_modeling_failure.get('failed_stage')} "
                    f"elapsed_with_scene_reconstruction={elapsed_with_scene:.1f}s "
                    f"elapsed_without_scene_reconstruction={elapsed_without_scene:.1f}s "
                    f"artifact={result_path}",
                )
                return result

            active_stage = "query-conditioned-physical-rollout"
            stage_start = time.perf_counter()
            _log(
                "query-conditioned-physical-rollout",
                f"start scene={scene.scene_index} question={question.question_id}",
            )
            question_rollout_backend_route = None
            clevrer_tool_planning_route = None
            if not self.dry_run:
                policy_benchmark = _pipeline_policy_benchmark(scene)
                if policy_benchmark is not None:
                    question_rollout_backend_route = (
                        _resolve_question_rollout_backend(
                            policy_benchmark,
                            scenario=(
                                str(getattr(scene, "scenario", "") or "").strip()
                                or None
                            ),
                            question_type=(
                                str(question.question_type or "").strip() or None
                            ),
                            policy=self.route_policy,
                        )
                    )
                    if policy_benchmark == "clevrer":
                        clevrer_tool_planning_route = (
                            _resolve_clevrer_tool_planning(
                                policy_benchmark,
                                question_type=(
                                    str(question.question_type or "").strip()
                                    or None
                                ),
                                policy=self.route_policy,
                            )
                        )
            simulation_result = self.simulator.run(
                scene=scene,
                question=question,
                question_dir=question_dir,
                object_plan=object_plan,
                stage_results=stage_results,
                world_model_dir=self.scene_world_dirs.get(scene.scene_index),
                question_rollout_backend_route=(
                    question_rollout_backend_route
                ),
                clevrer_tool_planning_route=(
                    clevrer_tool_planning_route
                ),
            )
            simulation_result = _record_timing(simulation_result, time.perf_counter() - stage_start)
            stage_results.append(simulation_result)
            message = _short_text(simulation_result.message)
            _log_tool(
                simulation_result.tool_name,
                f"end scene={scene.scene_index} question={question.question_id} "
                f"status={simulation_result.status} {_elapsed_text(simulation_result)} "
                f"artifact={simulation_result.artifact_path} "
                f"{_stage_summary(simulation_result.tool_name, simulation_result.payload)}",
            )
            if message:
                _log_tool(
                    simulation_result.tool_name,
                    f"message scene={scene.scene_index} question={question.question_id} {message}",
                )
            self._maybe_stop_after(simulation_result.tool_name, scene, question)
            pre_answer_status = _derive_final_status(stage_results, dry_run=self.dry_run)
            if pre_answer_status in {"ok", "dry_run"}:
                active_stage = "artifact-based-answering"
                stage_start = time.perf_counter()
                _log(
                    "artifact-based-answering",
                    f"start scene={scene.scene_index} question={question.question_id}",
                )
                if self.answer_enabled:
                    success_answer_backend_route = None
                    if not self.dry_run:
                        answer_benchmark = _pipeline_policy_benchmark(scene)
                        if answer_benchmark is not None:
                            success_answer_backend_route = (
                                _resolve_success_answer_backend(
                                    answer_benchmark,
                                    scenario=(
                                        str(getattr(scene, "scenario", "") or "").strip()
                                        or None
                                    ),
                                    question_type=(
                                        str(question.question_type or "").strip()
                                        or None
                                    ),
                                    policy=self.route_policy,
                                )
                            )
                    answer_result = run_final_answer(
                        config=self.config,
                        scene=scene,
                        question=question,
                        question_dir=question_dir,
                        artifacts=self.artifacts,
                        dry_run=self.dry_run,
                        success_answer_backend_route=(
                            success_answer_backend_route
                        ),
                    )
                else:
                    answer_result = self._skip_final_answer(scene=scene, question=question, question_dir=question_dir)
                answer_result = _record_timing(answer_result, time.perf_counter() - stage_start)
                stage_results.append(answer_result)
                message = _short_text(answer_result.message)
                _log_tool(
                    answer_result.tool_name,
                    f"end scene={scene.scene_index} question={question.question_id} "
                    f"status={answer_result.status} {_elapsed_text(answer_result)} "
                    f"artifact={answer_result.artifact_path} "
                    f"{_stage_summary(answer_result.tool_name, answer_result.payload)}",
                )
                if message:
                    _log_tool(
                        answer_result.tool_name,
                        f"message scene={scene.scene_index} question={question.question_id} {message}",
                    )
                if self.answer_enabled and not _stage_succeeded(answer_result):
                    fallback_attempted = True
                    fallback_used = True
                    answer_result = self._run_pipeline_direct_answer_fallback(
                        scene=scene,
                        question=question,
                        question_dir=question_dir,
                        stage_results=stage_results,
                        pipeline_failure=_pipeline_failure_from_stage_results(stage_results),
                    )
                self._maybe_stop_after(answer_result.tool_name, scene, question)
            elif self.answer_enabled:
                fallback_attempted = True
                fallback_used = True
                answer_result = self._run_pipeline_direct_answer_fallback(
                    scene=scene,
                    question=question,
                    question_dir=question_dir,
                    stage_results=stage_results,
                    pipeline_failure=_pipeline_failure_from_stage_results(stage_results),
                )
                self._maybe_stop_after(answer_result.tool_name, scene, question)
            else:
                _log(
                    "artifact-based-answering",
                    f"skip scene={scene.scene_index} question={question.question_id} "
                    f"reason=pre_answer_status:{pre_answer_status}",
                )

            if self.dry_run:
                status = "dry_run"
            elif fallback_used and answer_result is not None:
                status = answer_result.status
            else:
                status = _derive_final_status(stage_results, dry_run=False)
            question_wall_elapsed = time.perf_counter() - question_start
            scene_reconstruction_total = self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0)
            question_scene_reconstruction_elapsed = max(
                0.0,
                scene_reconstruction_total - scene_reconstruction_before,
            )
            question_without_scene_reconstruction_elapsed = max(
                0.0,
                question_wall_elapsed - question_scene_reconstruction_elapsed,
            )
            result = WorldModelQuestionResult(
                scene_index=scene.scene_index,
                video_filename=scene.video_filename,
                question_id=question.question_id,
                question_type=question.question_type,
                question=question.question,
                status=status,
                final_answer=extracted_final_answer(answer_result),
                error_message=None if status in {"ok", "dry_run"} else "One or more tools are not configured or failed.",
                artifact_dir=str(question_dir),
                stages=[_stage_result_ref(item) for item in stage_results],
                elapsed_with_scene_reconstruction_sec=(
                    question_without_scene_reconstruction_elapsed + scene_reconstruction_total
                ),
                elapsed_without_scene_reconstruction_sec=question_without_scene_reconstruction_elapsed,
            )
        except _StopAfterStage:
            status = _derive_final_status(stage_results, dry_run=self.dry_run)
            question_wall_elapsed = time.perf_counter() - question_start
            scene_reconstruction_total = self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0)
            question_scene_reconstruction_elapsed = max(
                0.0,
                scene_reconstruction_total - scene_reconstruction_before,
            )
            question_without_scene_reconstruction_elapsed = max(
                0.0,
                question_wall_elapsed - question_scene_reconstruction_elapsed,
            )
            result = WorldModelQuestionResult(
                scene_index=scene.scene_index,
                video_filename=scene.video_filename,
                question_id=question.question_id,
                question_type=question.question_type,
                question=question.question,
                status=status,
                final_answer=extracted_final_answer(answer_result),
                error_message=None,
                artifact_dir=str(question_dir),
                stages=[_stage_result_ref(item) for item in stage_results],
                elapsed_with_scene_reconstruction_sec=(
                    question_without_scene_reconstruction_elapsed + scene_reconstruction_total
                ),
                elapsed_without_scene_reconstruction_sec=question_without_scene_reconstruction_elapsed,
            )
        except Exception as exc:
            pipeline_failure = {
                "failed_stage": active_stage,
                "failed_tool": None,
                "status": "agent_error",
                "error_message": str(exc),
                "artifact_path": None,
            }
            fallback_error = None
            if self.answer_enabled and not fallback_attempted:
                fallback_attempted = True
                try:
                    answer_result = self._run_pipeline_direct_answer_fallback(
                        scene=scene,
                        question=question,
                        question_dir=question_dir,
                        stage_results=stage_results,
                        pipeline_failure=pipeline_failure,
                    )
                    fallback_used = True
                except Exception as fallback_exc:
                    fallback_error = str(fallback_exc)
            question_wall_elapsed = time.perf_counter() - question_start
            scene_reconstruction_total = self.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0)
            question_scene_reconstruction_elapsed = max(
                0.0,
                scene_reconstruction_total - scene_reconstruction_before,
            )
            question_without_scene_reconstruction_elapsed = max(
                0.0,
                question_wall_elapsed - question_scene_reconstruction_elapsed,
            )
            result = WorldModelQuestionResult(
                scene_index=scene.scene_index,
                video_filename=scene.video_filename,
                question_id=question.question_id,
                question_type=question.question_type,
                question=question.question,
                status=(
                    answer_result.status
                    if fallback_used and answer_result is not None
                    else "agent_error"
                ),
                final_answer=extracted_final_answer(answer_result),
                error_message=(
                    None
                    if fallback_used and answer_result is not None and answer_result.status == "ok"
                    else (
                        f"{exc}; direct-answer fallback failed: {fallback_error}"
                        if fallback_error
                        else str(exc)
                    )
                ),
                artifact_dir=str(question_dir),
                stages=[_stage_result_ref(item) for item in stage_results],
                elapsed_with_scene_reconstruction_sec=(
                    question_without_scene_reconstruction_elapsed + scene_reconstruction_total
                ),
                elapsed_without_scene_reconstruction_sec=question_without_scene_reconstruction_elapsed,
            )
            _log(
                "question",
                f"error scene={scene.scene_index} question={question.question_id} "
                f"elapsed={question_wall_elapsed:.1f}s error={_short_text(str(exc))} "
                f"fallback_used={fallback_used}",
            )
        result_path = question_dir / "artifact-based-answering" / "result.json"
        self.artifacts.write(result_path, result.to_dict())
        self.artifacts.append_trace(
            {
                "event": "question_complete",
                "scene_index": scene.scene_index,
                "question_id": question.question_id,
                "status": result.status,
                "fallback_used": fallback_used,
                "elapsed_with_scene_reconstruction_sec": result.elapsed_with_scene_reconstruction_sec,
                "elapsed_without_scene_reconstruction_sec": result.elapsed_without_scene_reconstruction_sec,
            }
        )
        elapsed_with_scene = result.elapsed_with_scene_reconstruction_sec or 0.0
        elapsed_without_scene = result.elapsed_without_scene_reconstruction_sec or 0.0
        _log(
            "question",
            f"end scene={scene.scene_index} question={question.question_id} "
            f"status={result.status} final_answer={result.final_answer} "
            f"elapsed_with_scene_reconstruction={elapsed_with_scene:.1f}s "
            f"elapsed_without_scene_reconstruction={elapsed_without_scene:.1f}s "
            f"artifact={result_path}",
        )
        return result


def _compute_status_metrics(results: Sequence[WorldModelQuestionResult]) -> Dict[str, object]:
    status_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        type_counts[result.question_type] = type_counts.get(result.question_type, 0) + 1
    return {
        "total_questions": len(results),
        "status_counts": status_counts,
        "question_type_counts": type_counts,
    }


def _write_run_outputs(
    *,
    run_dir: Path,
    run_config: Dict[str, object],
    predictions: list[Dict[str, object]],
    metric_predictions: list[Dict[str, object]],
    metrics: Dict[str, object],
    scene_timings: list[Dict[str, object]],
) -> None:
    bundle = {
        "schema_version": 1,
        "config": run_config,
        "predictions": predictions,
        "metric_predictions": metric_predictions,
        "metrics": metrics,
        "scene_timings": scene_timings,
        "summary": {
            "mode": run_config.get("mode"),
            "benchmark": run_config.get("benchmark"),
            "total_questions": metrics.get("world_model_agent", {}).get("total_questions", len(predictions)),
            "status_counts": metrics.get("world_model_agent", {}).get("status_counts", {}),
        },
    }
    write_json(run_dir / "run.json", bundle)
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "predictions.json", predictions)
    write_json(run_dir / "metric_predictions.json", metric_predictions)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "scene_timings.json", scene_timings)


def _world_model_result_from_dict(payload: Dict[str, Any]) -> WorldModelQuestionResult:
    return WorldModelQuestionResult(
        scene_index=int(payload["scene_index"]),
        video_filename=str(payload["video_filename"]),
        question_id=int(payload["question_id"]),
        question_type=str(payload["question_type"]),
        question=str(payload["question"]),
        status=str(payload["status"]),
        final_answer=payload.get("final_answer"),
        error_message=payload.get("error_message"),
        artifact_dir=str(payload["artifact_dir"]),
        stages=list(payload.get("stages") or []),
        elapsed_with_scene_reconstruction_sec=payload.get("elapsed_with_scene_reconstruction_sec"),
        elapsed_without_scene_reconstruction_sec=payload.get("elapsed_without_scene_reconstruction_sec"),
    )


def _load_resume_results(run_dir: Path) -> Dict[tuple[int, int], WorldModelQuestionResult]:
    results: Dict[tuple[int, int], WorldModelQuestionResult] = {}
    predictions_path = run_dir / "predictions.json"
    if predictions_path.exists():
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"resume predictions must be a list: {predictions_path}")
        for item in payload:
            if not isinstance(item, dict):
                continue
            result = _world_model_result_from_dict(item)
            results[(result.scene_index, result.question_id)] = result

    result_paths = sorted(
        (run_dir / "artifacts").glob(
            "scene_*/question_*/artifact-based-answering/result.json"
        )
    )
    for result_path in result_paths:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        result = _world_model_result_from_dict(payload)
        key = (result.scene_index, result.question_id)
        existing = results.get(key)
        if existing is None or result.status in {"ok", "dry_run"}:
            results[key] = result
    return results


def _resume_scene_is_complete(
    scene: ClevrerScene,
    allowed_question_types: Set[str],
    resume_results: Dict[tuple[int, int], WorldModelQuestionResult],
) -> bool:
    required_questions = [
        question
        for question in scene.questions
        if _allowed_questions(question, allowed_question_types)
    ]
    if not required_questions:
        return False
    for question in required_questions:
        result = resume_results.get((scene.scene_index, question.question_id))
        if result is None or result.status not in {"ok", "dry_run"}:
            return False
    return True


def _build_metric_predictions(
    scenes: Sequence[ClevrerScene],
    results: Sequence[WorldModelQuestionResult],
    allowed_question_types: Set[str],
) -> list[Dict[str, object]]:
    result_by_key = {
        (result.scene_index, result.question_id): result
        for result in results
    }
    predictions = []
    for scene in scenes:
        scene_questions = []
        for question in scene.questions:
            if not _allowed_questions(question, allowed_question_types):
                continue
            result = result_by_key.get((scene.scene_index, question.question_id))
            if result is None:
                continue
            scene_questions.append(_metric_question_from_result(question, result))
        predictions.append(
            {
                "scene_index": scene.scene_index,
                "video_filename": scene.video_filename,
                "video_path": scene.video_path,
                "questions": scene_questions,
            }
        )
    return predictions


def _build_physion_pp_metric_predictions(
    scenes: Sequence[PhysionPPScene],
    results: Sequence[WorldModelQuestionResult],
    allowed_question_types: Set[str],
) -> list[Dict[str, object]]:
    result_by_key = {
        (result.scene_index, result.question_id): result
        for result in results
    }
    predictions = []
    for scene in scenes:
        scene_questions = []
        for question in scene.questions:
            if not _allowed_questions(question, allowed_question_types):
                continue
            result = result_by_key.get((scene.scene_index, question.question_id))
            if result is None:
                continue
            scene_questions.append(_physion_pp_metric_question_from_result(question, result))
        predictions.append(
            {
                "scene_index": scene.scene_index,
                "property_name": scene.property_name,
                "scenario": scene.scenario,
                "stimulus_id": scene.stimulus_id,
                "pair_id": scene.pair_id,
                "video_filename": scene.video_filename,
                "video_path": str(scene.video_path),
                "questions": scene_questions,
            }
        )
    return predictions


































def run_clevrer_validation_world_model_agent(
    *,
    config: ModelConfig,
    dataset_root: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    scene_ids: Optional[Sequence[int]] = None,
    question_types: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    stop_after_stage: str = "evaluation",
    debug_artifacts: bool = False,
    persistent_workers: bool = True,
    resume_run_dir: Optional[Union[str, Path]] = None,
) -> Path:
    input_video_route, trim_cue_flash = _resolve_world_model_video_source("clevrer")
    answer_fallback_video_route = _resolve_answer_fallback_video_source("clevrer")
    if trim_cue_flash:
        raise RoutePolicyValidationError(
            "CLEVRER has no cue-flash staging input for a trimmed-video route"
        )
    if resume_run_dir is None:
        run_dir = ensure_run_dir(provider=config.provider, model=config.model, mode="world-model-agent")
        resume_results: Dict[tuple[int, int], WorldModelQuestionResult] = {}
    else:
        run_dir = Path(resume_run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise ValueError(f"resume_run_dir does not exist: {run_dir}")
        resume_results = _load_resume_results(run_dir)
    scenes = load_validation_scenes(dataset_root=dataset_root)
    scenes = _filter_scenes_by_ids(scenes, scene_ids)
    if limit is not None:
        scenes = scenes[:limit]
    allowed_question_types = {value.strip().lower() for value in (question_types or []) if value.strip()}
    skipped_scene_ids = {
        scene.scene_index
        for scene in scenes
        if _resume_scene_is_complete(scene, allowed_question_types, resume_results)
    }
    agent = WorldModelAgent(
        config=config,
        run_dir=run_dir,
        dry_run=dry_run,
        stop_after_stage=stop_after_stage,
        debug_artifacts=debug_artifacts,
        persistent_workers=persistent_workers,
    )
    total_questions = sum(
        1
        for scene in scenes
        for question in scene.questions
        if _allowed_questions(question, allowed_question_types)
    )
    _log(
        "run",
        f"start run_dir={run_dir} scenes={len(scenes)} questions={total_questions} "
        f"dry_run={dry_run} question_types={sorted(allowed_question_types)} "
        f"stop_after_stage={stop_after_stage} debug_artifacts={debug_artifacts} "
        f"persistent_workers={persistent_workers} resume={resume_run_dir is not None} "
        f"resume_skipped_scenes={sorted(skipped_scene_ids)}",
    )

    results = []
    try:
        for scene in tqdm(scenes, desc="CLEVRER world-model-agent"):
            scene_start = time.perf_counter()
            scene_question_count_before = len(results)
            allowed_scene_questions = [
                question
                for question in scene.questions
                if _allowed_questions(question, allowed_question_types)
            ]
            if scene.scene_index in skipped_scene_ids:
                for question in allowed_scene_questions:
                    results.append(resume_results[(scene.scene_index, question.question_id)])
                scene_elapsed = 0.0
                scene_timing = {
                    "scene_index": scene.scene_index,
                    "video_filename": scene.video_filename,
                    "question_count": len(allowed_scene_questions),
                    "scene_elapsed_sec": scene_elapsed,
                    "scene_reconstruction_elapsed_sec": 0.0,
                    "resume_skipped": True,
                }
                agent.scene_timings.append(scene_timing)
                _log(
                    "run",
                    f"resume_skip scene={scene.scene_index} questions={len(allowed_scene_questions)}",
                )
                continue
            for question in allowed_scene_questions:
                results.append(agent.process_question(scene, question))
                if agent.stop_after_reached_for_scene(scene):
                    _log(
                        "run",
                        f"stop_after scene={scene.scene_index} break_remaining_questions "
                        f"requested={stop_after_stage}",
                    )
                    break
            agent.cleanup_scene(scene)
            scene_elapsed = time.perf_counter() - scene_start
            scene_question_count = len(results) - scene_question_count_before
            scene_timing = {
                "scene_index": scene.scene_index,
                "video_filename": scene.video_filename,
                "question_count": scene_question_count,
                "scene_elapsed_sec": scene_elapsed,
                "scene_reconstruction_elapsed_sec": agent.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0),
            }
            agent.scene_timings.append(scene_timing)
            _log(
                "run",
                f"scene_complete scene={scene.scene_index} questions={scene_question_count} "
                f"scene_elapsed={scene_elapsed:.1f}s "
                f"scene_reconstruction_elapsed={scene_timing['scene_reconstruction_elapsed_sec']:.1f}s",
            )
    finally:
        agent.close()

    predictions = [result.to_dict() for result in results]
    metric_predictions = _build_metric_predictions(scenes, results, allowed_question_types)
    metrics = compute_metrics(metric_predictions)
    metrics["world_model_agent"] = _compute_status_metrics(results)
    run_config = config.to_safe_dict()
    run_config.update(
        {
            "dataset_root": dataset_root,
            "limit": limit,
            "scene_ids": list(scene_ids) if scene_ids is not None else None,
            "question_types": sorted(allowed_question_types),
            "mode": "world-model-agent",
            "benchmark": "clevrer-validation",
            "resolved_routes": {
                WORLD_MODEL_VIDEO_SOURCE_DECISION_ID: input_video_route,
                ANSWER_FALLBACK_VIDEO_SOURCE_DECISION_ID: answer_fallback_video_route,
            },
            "dry_run": dry_run,
            "stop_after_stage": stop_after_stage,
            "debug_artifacts": debug_artifacts,
            "persistent_workers": persistent_workers,
            "resume_run_dir": str(run_dir) if resume_run_dir is not None else None,
            "resume_skipped_scene_ids": sorted(skipped_scene_ids),
            "model_startup_times": agent.model_startup_times,
            "scene_timings": agent.scene_timings,
        }
    )
    _write_run_outputs(
        run_dir=run_dir,
        run_config=run_config,
        predictions=predictions,
        metric_predictions=metric_predictions,
        metrics=metrics,
        scene_timings=agent.scene_timings,
    )
    _log(
        "run",
        f"end run_dir={run_dir} questions={len(results)} "
        f"status_counts={metrics['world_model_agent'].get('status_counts', {})}",
    )
    return run_dir




def run_physion_pp_test_world_model_agent(
    *,
    config: ModelConfig,
    dataset_root: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    scene_ids: Optional[Sequence[int]] = None,
    question_types: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    stop_after_stage: str = "evaluation",
    debug_artifacts: bool = False,
    persistent_workers: bool = True,
    resume_run_dir: Optional[Union[str, Path]] = None,
    properties: Optional[Sequence[str]] = None,
    enabled_route_options: Sequence[str] = (),
) -> Path:
    route_policy = load_route_policy()
    normalized_route_options = tuple(
        option.option_id
        for option in route_policy.enabled_optional_routes(enabled_route_options)
    )
    input_video_route, trim_cue_flash = _resolve_world_model_video_source(
        "physion_pp",
        policy=route_policy,
    )
    answer_fallback_video_route = _resolve_answer_fallback_video_source(
        "physion_pp",
        policy=route_policy,
    )
    if resume_run_dir is None:
        run_dir = ensure_run_dir(provider=config.provider, model=config.model, mode="world-model-agent")
    else:
        run_dir = Path(resume_run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise ValueError(f"resume_run_dir does not exist: {run_dir}")
    # The loader preserves official scene indices before property filtering.
    scenes = load_physion_pp_test_scenes(dataset_root=dataset_root)
    scenes = [scene for scene in scenes if scene.scenario in RIGID_SCENARIOS]
    if properties:
        wanted_properties = {value for value in properties}
        scenes = [scene for scene in scenes if scene.property_name in wanted_properties]
    scenes = _filter_scenes_by_ids(scenes, scene_ids, benchmark_name="Physion++")
    if limit is not None:
        scenes = scenes[:limit]
    scenes = resolve_cue_clip_scenes(scenes, dataset_root=dataset_root)
    # World-model mode consumes the tail-trimmed clip (scene.video_path points at it);
    # the original cue clip and the flash-region npz stay alongside in the run dir for
    # the track-role binding stage. Direct-answer mode keeps the untrimmed clip.
    scenes = stage_cue_clips_into_run_dir(
        scenes,
        run_dir=run_dir,
        trim_cue_flash=trim_cue_flash,
    )
    allowed_question_types = {value.strip().lower() for value in (question_types or []) if value.strip()}
    answer_enabled = _runs_through_stage(stop_after_stage, "answering")
    agent = WorldModelAgent(
        config=config,
        run_dir=run_dir,
        dry_run=dry_run,
        stop_after_stage=stop_after_stage,
        debug_artifacts=debug_artifacts,
        persistent_workers=persistent_workers,
        answer_enabled=answer_enabled,
        route_policy=route_policy,
        enabled_route_options=normalized_route_options,
    )
    total_questions = sum(
        1
        for scene in scenes
        for question in scene.questions
        if _allowed_questions(question, allowed_question_types)
    )
    _log(
        "run",
        f"start run_dir={run_dir} scenes={len(scenes)} questions={total_questions} "
        f"dry_run={dry_run} question_types={sorted(allowed_question_types)} "
        f"stop_after_stage={stop_after_stage} debug_artifacts={debug_artifacts} "
        f"persistent_workers={persistent_workers} answer_enabled={answer_enabled} "
        f"benchmark=physion_pp enabled_route_options={list(normalized_route_options)}",
    )

    results = []
    try:
        for scene in tqdm(scenes, desc="Physion++ world-model-agent"):
            scene_start = time.perf_counter()
            scene_question_count_before = len(results)
            for question in scene.questions:
                if _allowed_questions(question, allowed_question_types):
                    results.append(agent.process_question(scene, question))
                    if agent.stop_after_reached_for_scene(scene):
                        _log(
                            "run",
                            f"stop_after scene={scene.scene_index} break_remaining_questions "
                            f"requested={stop_after_stage}",
                        )
                        break
            agent.cleanup_scene(scene)
            scene_elapsed = time.perf_counter() - scene_start
            scene_question_count = len(results) - scene_question_count_before
            scene_timing = {
                "scene_index": scene.scene_index,
                "scenario": scene.scenario,
                "video_filename": scene.video_filename,
                "question_count": scene_question_count,
                "scene_elapsed_sec": scene_elapsed,
                "scene_reconstruction_elapsed_sec": agent.scene_reconstruction_elapsed_sec.get(scene.scene_index, 0.0),
            }
            agent.scene_timings.append(scene_timing)
            _log(
                "run",
                f"scene_complete scene={scene.scene_index} scenario={scene.scenario} "
                f"questions={scene_question_count} scene_elapsed={scene_elapsed:.1f}s "
                f"scene_reconstruction_elapsed={scene_timing['scene_reconstruction_elapsed_sec']:.1f}s",
            )
    finally:
        agent.close()

    predictions = [result.to_dict() for result in results]
    if answer_enabled:
        metric_predictions = _build_physion_pp_metric_predictions(
            scenes,
            results,
            allowed_question_types,
        )
        metrics = compute_physion_pp_metrics(metric_predictions)
        metrics["world_model_agent"] = {
            **_compute_status_metrics(results),
            "answering": "enabled",
        }
    else:
        metric_predictions = []
        metrics = {
            "world_model_agent": {
                **_compute_status_metrics(results),
                "answering": "skipped",
                "answering_note": "The requested stop stage precedes artifact-based answering.",
            }
        }
    run_config = config.to_safe_dict()
    run_config.update(
        {
            "dataset_root": dataset_root,
            "limit": limit,
            "scene_ids": list(scene_ids) if scene_ids is not None else None,
            "question_types": sorted(allowed_question_types),
            "properties": list(properties) if properties else None,
            "supported_scenarios": list(RIGID_SCENARIOS),
            "mode": "world-model-agent",
            "benchmark": "physion_pp",
            "world_model_benchmark_treatment": "physion_pp",
            "route_policy_id": route_policy.policy_id,
            "enabled_route_options": list(normalized_route_options),
            "resolved_routes": {
                WORLD_MODEL_VIDEO_SOURCE_DECISION_ID: input_video_route,
                ANSWER_FALLBACK_VIDEO_SOURCE_DECISION_ID: answer_fallback_video_route,
            },
            "physion_pp_cue_video_dir": str(run_dir / PHYSION_PP_RUN_CUE_VIDEO_DIRNAME),
            "dry_run": dry_run,
            "stop_after_stage": stop_after_stage,
            "debug_artifacts": debug_artifacts,
            "persistent_workers": persistent_workers,
            "answer_enabled": answer_enabled,
            "model_startup_times": agent.model_startup_times,
            "scene_timings": agent.scene_timings,
        }
    )
    _write_run_outputs(
        run_dir=run_dir,
        run_config=run_config,
        predictions=predictions,
        metric_predictions=metric_predictions,
        metrics=metrics,
        scene_timings=agent.scene_timings,
    )
    _log(
        "run",
        f"end run_dir={run_dir} questions={len(results)} "
        f"status_counts={metrics['world_model_agent'].get('status_counts', {})} "
        f"answering={'enabled' if answer_enabled else 'skipped'}",
    )
    return run_dir
