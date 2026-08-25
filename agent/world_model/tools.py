from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from agent.query import (
    answer_with_image_files,
    extract_answer_tag,
)
from agent.world_model.artifacts import ArtifactManager
from agent.world_model.foundationpose_worker import FoundationPoseWorkerClient
from agent.world_model.geocalib_worker import GeoCalibWorkerClient
from agent.world_model.moge2_worker import MoGe2WorkerClient
from agent.world_model.sam3d_worker import SAM3DWorkerClient
from agent.world_model.sam3_video_tracks_worker import SAM3VideoTracksWorkerClient
from agent.world_model.schemas import ObjectPlan, TargetObject, ToolResult
from agent.world_model.video_metric_depth_worker import VideoMetricDepthWorkerClient
from benchmark.clevrer import ClevrerScene
from benchmark.specs import SAM3_VIDEO_TRACK_PROMPTS_BY_BENCHMARK
from scripts.world_model.mesh_projection import (
    cuda_mask_rasterizer_stats,
    mask_occluded_mesh_mask,
    render_mask_occluded_mesh,
    render_mesh_mask_cuda,
    render_mesh_depth,
    reset_cuda_mask_rasterizer_stats,
)
from utils.config import ModelConfig
from utils.terminal import terminal_print


OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE = "object-segmentation-and-event-detection"
OBJECT_INVENTORY_SOURCE_DECISION_ID = "SCN-002.object_inventory_source"
TRACK_BOOTSTRAP_THEN_TRACK_DERIVED_INVENTORY_ROUTE = (
    "inventory.from_tracks"
)
MASS_EXTRA_PATIENT_LINK_DECISION_ID = "TRK-005.mass_extra_patient_link"
MASS_EXTRA_PATIENT_LINK_ROUTE = "role_link.vlm_identity"
TEMPORAL_PARTITION_DECISION_ID = "INP-003.temporal_partition"
SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE = "temporal.single_video"
SPLIT_TEMPORAL_PARTITION_ROUTE = "temporal.cue_two_segment"
CROSS_SEGMENT_MESH_REUSE_DECISION_ID = "GEO-002.cross_segment_mesh_reuse"
SAME_ROLE_CROSS_SEGMENT_MESH_REUSE_ROUTE = (
    "mesh_reuse.same_role_identity"
)
MASS_COLLISION_CROSS_SEGMENT_MESH_REUSE_ROUTE = (
    "mesh_reuse.ball_agent_patient_conditional"
)
CROSS_SEGMENT_MESH_REUSE_ROUTES = frozenset(
    {
        SAME_ROLE_CROSS_SEGMENT_MESH_REUSE_ROUTE,
        MASS_COLLISION_CROSS_SEGMENT_MESH_REUSE_ROUTE,
    }
)
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
SCENE_HORIZONTAL_GROUND_MOTION_ROUTE = (
    "ground.vlm_horizontal_motion"
)
MASS_COLLISION_FORCED_GROUND_MOTION_ROUTE = (
    "ground.collision_forced_true"
)
GROUND_MOTION_GATE_ROUTES = frozenset(
    {
        SCENE_HORIZONTAL_GROUND_MOTION_ROUTE,
        PHYSION_PP_AGENT_FREE_GROUND_MOTION_ROUTE,
        MASS_COLLISION_FORCED_GROUND_MOTION_ROUTE,
    }
)
GRAVITY_ESTIMATOR_DECISION_ID = "POS-003.gravity_estimator"
CLEVRER_OVERLAP_GRAVITY_ESTIMATOR_ROUTE = (
    "gravity.overlap_aabb_post_snap"
)
FRICTION_COLLISION_GEOCALIB_GRAVITY_ROUTE = (
    "gravity.geocalib_8frame_z_negative"
)
MASS_COLLISION_GEOCALIB_GRAVITY_ROUTE = (
    "gravity.geocalib_8frame"
)
GRAVITY_ESTIMATOR_ROUTES = frozenset(
    {
        CLEVRER_OVERLAP_GRAVITY_ESTIMATOR_ROUTE,
        PHYSION_PP_OVERLAP_GRAVITY_ESTIMATOR_ROUTE,
        FRICTION_COLLISION_GEOCALIB_GRAVITY_ROUTE,
        MASS_COLLISION_GEOCALIB_GRAVITY_ROUTE,
    }
)
GRAVITY_CONSTRAINTS_DECISION_ID = "POS-004.gravity_roll_and_z_constraints"
OVERLAP_GRAVITY_CONSTRAINTS_ROUTE = "gravity_constraints.overlap_roll"
FRICTION_COLLISION_GRAVITY_CONSTRAINTS_ROUTE = (
    "gravity_constraints.roll_zero_z_negative"
)
MASS_COLLISION_GRAVITY_CONSTRAINTS_ROUTE = "gravity_constraints.roll_zero"
GRAVITY_CONSTRAINT_ROUTES = frozenset(
    {
        OVERLAP_GRAVITY_CONSTRAINTS_ROUTE,
        FRICTION_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
        MASS_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
    }
)
GRAVITY_ROUTE_PAIRS = frozenset(
    {
        (
            CLEVRER_OVERLAP_GRAVITY_ESTIMATOR_ROUTE,
            OVERLAP_GRAVITY_CONSTRAINTS_ROUTE,
        ),
        (
            PHYSION_PP_OVERLAP_GRAVITY_ESTIMATOR_ROUTE,
            OVERLAP_GRAVITY_CONSTRAINTS_ROUTE,
        ),
        (
            FRICTION_COLLISION_GEOCALIB_GRAVITY_ROUTE,
            FRICTION_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
        ),
        (
            MASS_COLLISION_GEOCALIB_GRAVITY_ROUTE,
            MASS_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
        ),
    }
)
ROTATION_POLICY_DECISION_ID = "POS-005.rotation_policy"
ROTATION_POLICY_ROUTE = "rotation.geometry_conditioned"
SUPPORT_SNAP_DECISION_ID = "POS-006.support_snap"
CLEVRER_SUPPORT_SNAP_ROUTE = "support.ray_slide_ground"
FRICTION_PLATFORM_SUPPORT_SNAP_ROUTE = (
    "support.platform_sphere_agent"
)
BOUNCY_WALL_SUPPORT_SNAP_ROUTE = (
    "support.wall_sphere_agent"
)
BOUNCY_PLATFORM_SUPPORT_SNAP_ROUTE = "support.agent_exempt"
FRICTION_COLLISION_SUPPORT_SNAP_ROUTE = (
    "support.moving_patient_exempt"
)
MASS_COLLISION_SUPPORT_SNAP_ROUTE = "support.patient_exempt"
SUPPORT_SNAP_ROUTES = frozenset(
    {
        CLEVRER_SUPPORT_SNAP_ROUTE,
        FRICTION_PLATFORM_SUPPORT_SNAP_ROUTE,
        BOUNCY_WALL_SUPPORT_SNAP_ROUTE,
        BOUNCY_PLATFORM_SUPPORT_SNAP_ROUTE,
        FRICTION_COLLISION_SUPPORT_SNAP_ROUTE,
        MASS_COLLISION_SUPPORT_SNAP_ROUTE,
    }
)
STATIC_FIXTURE_FLUSH_DECISION_ID = "POS-007.static_fixture_flush"
STATIC_FIXTURE_FLUSH_ROUTE = "flush.static_fixture"
SEG1_PATIENT_FLUSH_ROUTE = "flush.segment1_patient"
LINKED_EXTRA_FLUSH_ROUTE = "flush.linked_extra"
STATIC_FIXTURE_FLUSH_ROUTES = frozenset(
    {
        STATIC_FIXTURE_FLUSH_ROUTE,
        SEG1_PATIENT_FLUSH_ROUTE,
        LINKED_EXTRA_FLUSH_ROUTE,
    }
)
LINE_LAYOUT_DECISION_ID = "POS-008.line_layout"
ACTIVE_LINE_LAYOUT_ROUTE = "layout.line_static_orientation"
SKIP_LINE_LAYOUT_ROUTE = "layout.not_applicable"
LINE_LAYOUT_ROUTES = frozenset(
    {
        ACTIVE_LINE_LAYOUT_ROUTE,
        SKIP_LINE_LAYOUT_ROUTE,
    }
)
FOUNDATIONPOSE_AGENT_GEOMETRY_DECISION_ID = (
    "GEO-003.foundationpose_agent_geometry"
)
FOUNDATIONPOSE_AGENT_GEOMETRY_ROUTES = frozenset(
    {"agent_geometry.sphere_no_foundationpose", "agent_geometry.native_foundationpose"}
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
AGENT_TRAJECTORY_ROUTES_BY_DECISION = {
    FRICTION_PLATFORM_AGENT_TRAJECTORY_DECISION_ID: frozenset(
        {
            "trajectory.platform_sphere",
            "trajectory.native_ray",
        }
    ),
    BOUNCY_PLATFORM_AGENT_TRAJECTORY_DECISION_ID: frozenset(
        {"trajectory.bounce_mask"}
    ),
    BOUNCY_WALL_AGENT_TRAJECTORY_DECISION_ID: frozenset(
        {
            "trajectory.wall_mixed_geometry",
            "trajectory.wall_native",
        }
    ),
}
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
FRICTION_PLATFORM_BOUNDED_FIT_STRATEGY = (
    "swr_fit.bounded_plane_dynamic_sphere"
)
FRICTION_PLATFORM_SINGLE_PLANE_FALLBACK_STRATEGY = (
    "physionpp_friction_platform_single_dynamic_3d_sphere"
)
BOUNCY_WALL_FIT_STRATEGY = (
    "swr_fit.two_segment_shared_speed"
)
BOUNCY_WALL_INTERNAL_FIT_STRATEGY = (
    "physionpp_bouncy_wall_joint_two_segment_dynamic_3d_sphere"
)
SWR_FIT_STRATEGY_BY_SCENARIO = {
    "friction_platform_pp": FRICTION_PLATFORM_BOUNDED_FIT_STRATEGY,
    "bouncy_wall_pp": BOUNCY_WALL_FIT_STRATEGY,
    "bouncy_platform_pp": (
        "swr_fit.bounded_planes_full_trajectory"
    ),
    "friction_collision_pp": (
        "swr_fit.two_segment_shared_speed_radii"
    ),
    "mass_collision_pp": (
        "swr_fit.ball_agent_impulse_mass"
    ),
}
SWR_FIT_STRATEGY_ROUTES = frozenset(SWR_FIT_STRATEGY_BY_SCENARIO.values())
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
SWR_FIT_STRATEGY_BY_BACKEND = {
    "swr_backend.surface_friction_sphere": FRICTION_PLATFORM_BOUNDED_FIT_STRATEGY,
    "swr_backend.wall_bounce_sphere": BOUNCY_WALL_FIT_STRATEGY,
    "swr_backend.platform_bounce_sphere": SWR_FIT_STRATEGY_BY_SCENARIO[
        "bouncy_platform_pp"
    ],
    "swr_backend.collision_friction_spheres": SWR_FIT_STRATEGY_BY_SCENARIO[
        "friction_collision_pp"
    ],
    "swr_backend.collision_mass_spheres": SWR_FIT_STRATEGY_BY_SCENARIO[
        "mass_collision_pp"
    ],
}
TOOL_TO_DISPLAY_STAGE = {
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
}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
GEOCALIB_GRAVITY_METHOD = "geocalib_uniform_8_frame_robust_average"
DEFAULT_VIDEO_METRIC_DEPTH_CMD = "python -m scripts.world_model.run_video_metric_depth"
DEFAULT_GEOCALIB_CMD = "python -m scripts.world_model.run_geocalib_gravity"
DEFAULT_SAM3_VIDEO_TRACKS_CMD = "python -m scripts.world_model.run_sam3_video_tracks"
DEFAULT_SAM3_VIDEO_TRACK_LABELS_CMD = (
    "python -m scripts.world_model.label_sam3_video_tracks"
)
DEFAULT_SAM3D_OBJECTS_CMD = "python -m scripts.world_model.run_sam3d_objects"
DEFAULT_MOGE2_INTRINSICS_CMD = "python -m scripts.world_model.run_moge2_intrinsics"
DEFAULT_MESH_CONDITIONING_CMD = "python -m scripts.world_model.run_mesh_conditioning"
DEFAULT_FOUNDATIONPOSE_CMD = "python -m scripts.world_model.run_foundationpose"
DEFAULT_IMPULSE_ANALYTIC_SYSID_CMD = (
    "python -m scripts.world_model.run_impulse_analytic_sysid"
)
DEFAULT_PHYSIONPP_FRICTION_SPHERE_SYSID_CMD = (
    "python -m scripts.world_model.run_physionpp_friction_sphere_sysid"
)
DEFAULT_PHYSIONPP_BOUNCY_WALL_SPHERE_SYSID_CMD = (
    "python -m scripts.world_model.run_physionpp_bouncy_wall_sphere_sysid"
)
DEFAULT_PHYSIONPP_BOUNCY_PLATFORM_SPHERE_SYSID_CMD = (
    "python -m scripts.world_model.run_physionpp_bouncy_platform_sphere_sysid"
)
DEFAULT_PHYSIONPP_FRICTION_COLLISION_SPHERE_SYSID_CMD = (
    "python -m scripts.world_model.run_physionpp_friction_collision_sphere_sysid"
)
DEFAULT_PHYSIONPP_MASS_COLLISION_SPHERE_SYSID_CMD = (
    "python -m scripts.world_model.run_physionpp_mass_collision_sphere_sysid"
)
DEFAULT_SAM3_VIDEO_TRACK_LABEL_INPUTS_CMD = (
    "python scripts/world_model/export_sam3_video_track_label_inputs.py"
)
DEFAULT_BLENDER_CMD = "third_party/blender/blender"
CLEVRER_LATEST_DEBUG_RENDER_PROFILE = "clevrer_refined_v10"








def _record_mass_extra_patient_link_result(
    link_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != MASS_EXTRA_PATIENT_LINK_DECISION_ID:
        raise ValueError(
            "mass-extra-patient-link route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != MASS_EXTRA_PATIENT_LINK_ROUTE:
        raise ValueError(f"unsupported mass-extra-patient-link route: {route!r}")
    link_payload["resolved_route"] = deepcopy(route_record)








def _route_policy_benchmark_for_scene(scene: Any) -> str | None:
    if isinstance(scene, ClevrerScene):
        return "clevrer"
    return None




def _record_false_positive_advice_effect_result(
    labels_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != FALSE_POSITIVE_ADVICE_EFFECT_DECISION_ID:
        raise ValueError(
            "false-positive-advice-effect route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != FALSE_POSITIVE_ADVICE_EFFECT_ROUTE:
        raise ValueError(
            f"unsupported false-positive-advice-effect route: {route!r}"
        )
    labels = labels_payload.get("track_labels")
    label_items = labels if isinstance(labels, list) else []
    suggested_items = [
        item
        for item in label_items
        if isinstance(item, dict)
        and item.get("vlm_false_positive_suggestion") is True
    ]
    suggested_track_ids = sorted(
        str(item.get("track_id"))
        for item in suggested_items
        if item.get("track_id")
    )
    labels_payload["false_positive_advice_effect_route"] = deepcopy(route_record)
    labels_payload["false_positive_advice_effect"] = {
        "policy": FALSE_POSITIVE_ADVICE_EFFECT_ROUTE,
        "suggestion_count": len(suggested_items),
        "suggested_track_ids": suggested_track_ids,
        "automatic_deletion_count": 0,
        "effect": "advice_recorded_without_automatic_inventory_deletion",
        "resolved_route": deepcopy(route_record),
    }












def _record_segment_depth_alignment_result(
    depth_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != SEGMENT_DEPTH_ALIGNMENT_DECISION_ID:
        raise ValueError(
            "segment-depth-alignment route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != SEGMENT_DEPTH_ALIGNMENT_ROUTE:
        raise ValueError(
            f"unsupported segment-depth-alignment route: {route!r}"
        )
    actual_partition = depth_payload.get("two_segment_depth_inference")
    if isinstance(actual_partition, dict):
        if actual_partition.get("applied") is not True:
            raise ValueError(
                "segment-depth-alignment route requires an applied split-depth artifact"
            )
        affine_align = actual_partition.get("affine_align")
        if not isinstance(affine_align, dict) or affine_align.get("applied") is not True:
            raise ValueError(
                "segment-depth-alignment route requires applied affine alignment"
            )
        curtain_gap = actual_partition.get("curtain_gap")
        expected_gap_policy = (
            "per_pixel_linear_interpolation_between_aligned_segment_endpoints"
        )
        if (
            not isinstance(curtain_gap, dict)
            or curtain_gap.get("policy") != expected_gap_policy
        ):
            raise ValueError(
                "segment-depth-alignment route requires the reference curtain-gap "
                "interpolation policy"
            )
    depth_payload["segment_depth_alignment_route"] = deepcopy(route_record)






def _record_sam3d_observation_selection_result(
    sam3d_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if (
        route_record.get("decision_id")
        != SAM3D_OBSERVATION_SELECTION_DECISION_ID
    ):
        raise ValueError(
            "SAM3D-observation-selection route record has an unexpected "
            f"decision_id: {route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in SAM3D_OBSERVATION_SELECTION_ROUTES:
        raise ValueError(
            f"unsupported SAM3D-observation-selection route: {route!r}"
        )
    objects = sam3d_payload.get("objects")
    object_items = objects if isinstance(objects, list) else []
    composite_items = [
        item
        for item in object_items
        if isinstance(item, dict)
        and (
            "physion_pp_static_temporal_composite" in item
            or item.get("selected_frame_image_source")
            == "physion_pp_static_temporal_composite"
        )
    ]
    if route == "geometry.selected_observation" and composite_items:
        raise ValueError(
            "selected-observation route produced a static temporal composite"
        )
    sam3d_payload["sam3d_observation_selection_route"] = deepcopy(route_record)








def _record_pose_input_and_foundationpose_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if (
        route_record.get("decision_id")
        != POSE_INPUT_AND_FOUNDATIONPOSE_DECISION_ID
    ):
        raise ValueError(
            "pose-input route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != POSE_INPUT_AND_FOUNDATIONPOSE_ROUTE:
        raise ValueError(
            f"unsupported pose-input-and-FoundationPose route: {route!r}"
        )
    payload["pose_input_and_foundationpose_route"] = deepcopy(route_record)




















































def _display_stage(tool_name: str) -> str:
    return TOOL_TO_DISPLAY_STAGE.get(tool_name, tool_name.replace("_", "-"))


def _verbose_tool_io() -> bool:
    return (os.getenv("PHYSMIND_LOG_TOOL_IO") or "").strip().lower() in {"1", "true", "yes", "on"}












def _normalize_tool_log_message(message: str) -> str:
    replacements = {
        "physics_alignment_blender_render": "debug_render",
        "physics_alignment command": "world_reconstruction_fit command",
        "physics_alignment stdout": "world_reconstruction_fit stdout",
        "physics_alignment stderr": "world_reconstruction_fit stderr",
        "physics_alignment_manifest": "world_reconstruction_manifest",
        "trajectory_informed_physics_alignment": "trajectory_informed_world_reconstruction",
    }
    value = message
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _is_tool_io_message(message: str) -> bool:
    return (
        " stdout " in f" {message} "
        or " stderr " in f" {message} "
        or message.startswith("stdout ")
        or message.startswith("stderr ")
    )


def _log_tool(tool_name: str, message: str) -> None:
    message = _normalize_tool_log_message(message)
    if _is_tool_io_message(message) and not _verbose_tool_io():
        return
    terminal_print(f"[{_display_stage(tool_name)}] tool={tool_name} {message}", flush=True)


def _scene_object_plan_path(question_dir: Path) -> Path:
    scene_world_dir = question_dir if question_dir.name == "world-modeling" else question_dir.parent / "world-modeling"
    return (
        scene_world_dir
        / "object-identification-and-planning"
        / "object_plan"
        / "object_plan.json"
    )


def _short_text(text: str, limit: int = 1200) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _boundary_extension_values(
    *,
    winner: float,
    initial_start: float,
    initial_stop: float,
    step: float,
    expanded_start: float,
    expanded_stop: float,
    boundary_steps: int,
) -> tuple[list[float], Optional[str]]:
    """Return new coarse-grid values when the winner enters a boundary band."""
    tolerance = max(abs(step) * 1e-6, 1e-12)
    if winner <= initial_start + boundary_steps * step + tolerance:
        count = max(0, int(round((initial_start - expanded_start) / step)))
        return [initial_start - step * index for index in range(1, count + 1)], "lower"
    if winner >= initial_stop - boundary_steps * step - tolerance:
        count = max(0, int(round((expanded_stop - initial_stop) / step)))
        return [initial_stop + step * index for index in range(1, count + 1)], "upper"
    return [], None


def _bounded_grid_boundary_extension_values(
    *,
    winner: float,
    initial_values: Sequence[float],
    hard_limits: tuple[float, float],
    boundary_steps: int,
) -> tuple[list[float], Optional[str]]:
    """Extend a coarse grid at one boundary without exceeding its hard limits.

    ``_boundary_extension_values`` assumes the hard limit aligns with the coarse
    step. Some flush scale grids do not (for example 0.70 -> 0.05 in 0.075
    increments), so this wrapper removes an overshooting step and explicitly adds
    the configured endpoint.
    """
    values = [float(value) for value in initial_values]
    if len(values) < 2:
        return [], None
    ordered = sorted(set(values))
    positive_steps = sorted(
        abs(right - left)
        for left, right in zip(ordered[:-1], ordered[1:])
        if abs(right - left) > 1e-12
    )
    if not positive_steps:
        return [], None
    step = positive_steps[0]
    initial_start, initial_stop = ordered[0], ordered[-1]
    hard_lo, hard_hi = hard_limits
    tolerance = max(step * 1e-6, 1e-12)
    near_lower = winner <= initial_start + boundary_steps * step + tolerance
    near_upper = winner >= initial_stop - boundary_steps * step - tolerance
    # A short symmetric grid can make a wide boundary band overlap at its center
    # (the standard fixture shift grid has seven points and a three-step band).
    # A winner in both bands carries no outward direction, so keep the initial grid.
    if near_lower and near_upper:
        return [], None
    proposed, direction = _boundary_extension_values(
        winner=float(winner),
        initial_start=initial_start,
        initial_stop=initial_stop,
        step=step,
        expanded_start=float(hard_lo),
        expanded_stop=float(hard_hi),
        boundary_steps=boundary_steps,
    )
    proposed = [
        float(value)
        for value in proposed
        if hard_lo - tolerance <= float(value) <= hard_hi + tolerance
    ]
    if direction == "lower" and hard_lo < initial_start - tolerance:
        if not proposed or min(proposed) > hard_lo + tolerance:
            proposed.append(float(hard_lo))
    elif direction == "upper" and hard_hi > initial_stop + tolerance:
        if not proposed or max(proposed) < hard_hi - tolerance:
            proposed.append(float(hard_hi))
    return proposed, direction


def _stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"invalid mesh vertices: {mesh_path}")
    return vertices


def _plane_basis_for_up_axis(up_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, up_axis))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(up_axis, helper)
    u_axis /= np.linalg.norm(u_axis)
    return u_axis, np.cross(up_axis, u_axis)


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) * cos_a + cross * sin_a + np.outer(axis, axis) * (1.0 - cos_a)


# Flush-refinement candidate scoring runs in a process pool: the mesh rasterizer is a
# pure-Python per-triangle loop, so threads gain nothing under the GIL. Workers receive
# the (picklable, numpy-only) score state once via the pool initializer and each job is
# just (pose, scale_u, scale_v).
FLUSH_REFINE_POOL_MAX_WORKERS = 12
_FLUSH_REFINE_SCORE_STATE: Dict[str, Any] = {}


def _bool_mask_iou(rendered_mask: np.ndarray, target_mask: np.ndarray) -> float | None:
    rendered = np.asarray(rendered_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    if rendered.shape != target.shape:
        return None
    union_count = int(np.logical_or(rendered, target).sum())
    if union_count <= 0:
        return None
    return float(int(np.logical_and(rendered, target).sum())) / float(union_count)





















































# In-plane refinement grids for static ground fixtures: translation offsets along the
# support plane (meters), yaw about the up axis (degrees), and in-plane extent scale.
# The wide coarse scale range compensates for OBB extents fitted from SAM3D meshes,
# which systematically underestimate the mat footprint.
STATIC_FIXTURE_REFINE_ROUNDS: tuple[Dict[str, Any], ...] = (
    {
        "shift": tuple(np.arange(-0.15, 0.151, 0.05)),
        "yaw_deg": tuple(np.arange(-10.0, 10.01, 2.5)),
        "scale": tuple(np.arange(0.7, 1.451, 0.075)),
        "scale_mode": "absolute",
    },
    {
        "shift": tuple(np.arange(-0.04, 0.041, 0.01)),
        "yaw_deg": tuple(np.arange(-2.0, 2.01, 0.5)),
        "scale": tuple(np.arange(-0.06, 0.061, 0.015) + 1.0),
        "scale_mode": "relative",
    },
)
STATIC_FIXTURE_REFINE_MAX_EVAL_FRAMES = 9
STATIC_FIXTURE_REFINE_BOUNDARY_STEPS = 2
STATIC_FIXTURE_REFINE_EXPANDED_LIMITS: Dict[str, tuple[float, float]] = {
    "shift": (-0.60, 0.60),
    "yaw_deg": (-45.0, 45.0),
    "scale": (0.05, 2.00),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_video_frames(
    *,
    video_path: str,
    output_dir: Path,
    frame_indices: Sequence[int],
    error_label: str,
    extension: str = ".jpg",
) -> Dict[int, str]:
    import cv2

    unique_indices = sorted({int(item) for item in frame_indices})
    if not unique_indices:
        return {}
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported exported frame extension for {error_label}: {extension}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for {error_label} frame export: {video_path}")

    artifact_by_frame: Dict[int, str] = {}
    try:
        for raw_frame_index in unique_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Unable to read {error_label} frame {raw_frame_index}: {video_path}")
            image_path = output_dir / f"frame_{raw_frame_index:05d}{extension.lower()}"
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension.lower() == ".png" else []
            cv2.imwrite(str(image_path), frame, params)
            artifact_by_frame[raw_frame_index] = str(image_path)
    finally:
        capture.release()
    return artifact_by_frame


def _mask_boundary_diagnostic(mask: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(mask).astype(bool)
    if value.ndim > 2:
        value = np.squeeze(value)
    if value.ndim != 2 or value.size == 0:
        return {"touching_boundary": False, "boundary_sides": [], "mask_shape": list(value.shape)}
    sides = []
    if bool(value[0, :].any()):
        sides.append("top")
    if bool(value[-1, :].any()):
        sides.append("bottom")
    if bool(value[:, 0].any()):
        sides.append("left")
    if bool(value[:, -1].any()):
        sides.append("right")
    return {
        "touching_boundary": bool(sides),
        "boundary_sides": sides,
        "mask_shape": [int(value.shape[0]), int(value.shape[1])],
    }


def _masks_touch(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    import cv2

    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape or a.ndim != 2:
        return False
    if not a.any() or not b.any():
        return False
    dilated = cv2.dilate(a.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return bool(np.logical_and(dilated, b).any())


def _load_mask_array(sidecar: str | None, mask_key: str | None) -> Optional[np.ndarray]:
    if not sidecar or not mask_key:
        return None
    if not Path(str(sidecar)).exists():
        return None
    arrays = np.load(sidecar)
    if str(mask_key) not in arrays:
        return None
    value = arrays[str(mask_key)].astype(bool)
    if value.ndim > 2:
        value = np.squeeze(value)
    return value if value.ndim == 2 else None


def _records_by_track_from_sam3_tracks(tracks_payload: Dict[str, Any]) -> dict[str, list[Dict[str, Any]]]:
    records_by_track: dict[str, list[Dict[str, Any]]] = {}
    grouped = tracks_payload.get("tracks_by_object")
    if isinstance(grouped, dict):
        for track_id, records in grouped.items():
            if isinstance(records, list):
                records_by_track[str(track_id)] = [item for item in records if isinstance(item, dict)]
    if records_by_track:
        return records_by_track
    for record in tracks_payload.get("tracks", []):
        if not isinstance(record, dict):
            continue
        track_id = str(record.get("object_id") or record.get("track_id") or "")
        if track_id:
            records_by_track.setdefault(track_id, []).append(record)
    return records_by_track


def _accepted_sam3_track_ids_by_object(labels_payload: Dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in labels_payload.get("object_keyframes", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "").strip()
        track_id = str(item.get("source_track_id") or item.get("track_id") or "").strip()
        if object_id and track_id:
            output[object_id] = track_id
    for item in labels_payload.get("track_labels", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "").strip()
        track_id = str(item.get("track_id") or item.get("source_track_id") or "").strip()
        if object_id and track_id and object_id not in output:
            output[object_id] = track_id
    return output


def _track_interval_from_records(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    frame_indices = []
    areas = []
    for record in records:
        try:
            frame_indices.append(int(record.get("frame_index")))
        except (TypeError, ValueError):
            pass
        try:
            areas.append(float(record.get("area")))
        except (TypeError, ValueError):
            pass
    frame_indices = sorted(set(frame_indices))
    return {
        "first_frame_index": frame_indices[0] if frame_indices else None,
        "last_frame_index": frame_indices[-1] if frame_indices else None,
        "frame_count": len(frame_indices),
        "max_area": max(areas) if areas else None,
    }


def _accepted_sam3_video_records_from_labels(
    *,
    labels_payload: Dict[str, Any],
    tracks_payload: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], Optional[str]]:
    sidecar = tracks_payload.get("mask_sidecar") or labels_payload.get("mask_sidecar")
    object_by_track = {track_id: object_id for object_id, track_id in _accepted_sam3_track_ids_by_object(labels_payload).items()}
    records_by_track = _records_by_track_from_sam3_tracks(tracks_payload)
    output: list[Dict[str, Any]] = []
    for track_id, object_id in sorted(object_by_track.items()):
        records = sorted(
            records_by_track.get(track_id, []),
            key=lambda item: int(item.get("frame_index", 0) or 0),
        )
        interval = _track_interval_from_records(records)
        for record in records:
            mask_key = record.get("mask_key")
            if not mask_key:
                continue
            output.append(
                {
                    "object_id": object_id,
                    "frame_index": record.get("frame_index"),
                    "sam3_prompt": record.get("prompt") or record.get("text_prompt"),
                    "track_id": track_id,
                    "source_track_id": track_id,
                    "selected_mask_key": str(mask_key),
                    "mask_key": str(mask_key),
                    "selected_area": record.get("area"),
                    "area": record.get("area"),
                    "selected_bbox_xyxy": record.get("bbox_xyxy"),
                    "bbox_xyxy": record.get("bbox_xyxy"),
                    "selected_centroid_xy": record.get("centroid_xy"),
                    "centroid_xy": record.get("centroid_xy"),
                    "selected_score": record.get("score"),
                    "selected_candidate_id": f"{track_id}__frame_{int(record.get('frame_index', 0) or 0):05d}",
                    "selected_candidate_overlay_path": None,
                    "source_mask_key": str(mask_key),
                    "track_interval": interval,
                    "status": "accepted_track_record",
                }
            )
    return output, str(sidecar) if sidecar else None


def _annotate_mask_contacts(records: list[Dict[str, Any]], sidecar: str | None) -> list[Dict[str, Any]]:
    masks_by_index = []
    for index, record in enumerate(records):
        mask = _load_mask_array(sidecar, record.get("selected_mask_key"))
        boundary = _mask_boundary_diagnostic(mask) if mask is not None else {"touching_boundary": False, "boundary_sides": []}
        annotated = dict(record)
        annotated["touching_boundary"] = boundary.get("touching_boundary", False)
        annotated["boundary_sides"] = boundary.get("boundary_sides", [])
        annotated["touching_other_mask"] = False
        annotated["touching_other_object_ids"] = []
        masks_by_index.append((index, annotated, mask))

    for index, record, mask in masks_by_index:
        if mask is None:
            continue
        touching_ids = []
        for other_index, other, other_mask in masks_by_index:
            if other_index == index or other_mask is None:
                continue
            if int(other.get("frame_index", -1)) != int(record.get("frame_index", -2)):
                continue
            if str(other.get("object_id")) == str(record.get("object_id")):
                continue
            if _masks_touch(mask, other_mask):
                touching_ids.append(str(other.get("object_id")))
        record["touching_other_mask"] = bool(touching_ids)
        record["touching_other_object_ids"] = sorted(set(touching_ids))
    return [record for _, record, _ in masks_by_index]


def _selected_mask_boundary_diagnostics(pose_sam3_payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    selected_masks = pose_sam3_payload.get("selected_masks") or []
    sidecar = pose_sam3_payload.get("mask_sidecar")
    if not selected_masks or not sidecar:
        return []
    arrays = np.load(sidecar)
    diagnostics = []
    for item in selected_masks:
        if not isinstance(item, dict):
            continue
        mask_key = item.get("selected_mask_key")
        record = {
            "object_id": item.get("object_id"),
            "selected_candidate_id": item.get("selected_candidate_id"),
            "selected_mask_key": mask_key,
            "selected_bbox_xyxy": item.get("selected_bbox_xyxy"),
            "selected_area": item.get("selected_area"),
        }
        if not mask_key or str(mask_key) not in arrays:
            record.update(
                {
                    "touching_boundary": False,
                    "boundary_sides": [],
                    "status": "missing_mask_array",
                }
            )
            diagnostics.append(record)
            continue
        record.update(_mask_boundary_diagnostic(arrays[str(mask_key)]))
        record["status"] = "touching_boundary" if record["touching_boundary"] else "ok"
        diagnostics.append(record)
    return diagnostics


def _external_tool_env(question_dir: Path, debug_artifacts: bool) -> Dict[str, str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    env["PHYSMIND_QUESTION_DIR"] = str(question_dir)
    env["PHYSMIND_DEBUG_ARTIFACTS"] = "1" if debug_artifacts else "0"
    return env


def _parse_json_response(text: str) -> Dict[str, Any]:
    answer = extract_answer_tag(text) or text
    match = JSON_OBJECT_PATTERN.search(answer)
    if not match:
        raise ValueError("No JSON object found in VLM response.")
    return json.loads(match.group(0))


def _payload_summary(tool_name: str, payload: Dict[str, Any]) -> str:
    if tool_name == "sam3_video_tracks":
        return (
            f"mode={payload.get('mode')} prompts={len(payload.get('prompts') or [])} "
            f"tracks={payload.get('track_record_count')} frames={payload.get('tracked_frame_count')} "
            f"track_count_by_object={payload.get('track_count_by_object')} "
            f"mask_sidecar={payload.get('mask_sidecar')}"
        )
    if tool_name == "sam3_video_track_labels":
        labels = payload.get("track_labels") or []
        objects = [item for item in labels if isinstance(item, dict) and item.get("object_id")]
        suggested_fp = [
            item for item in labels if isinstance(item, dict) and item.get("vlm_false_positive_suggestion") is True
        ]
        return (
            f"labels={len(labels)} objects={len(objects)} "
            f"vlm_false_positive_suggestions={len(suggested_fp)} "
            f"image_count={len(payload.get('image_paths') or [])}"
        )
    if tool_name == "pose_sam3_mask_selection":
        selections = payload.get("selections") or []
        selected = [
            f"{item.get('object_id')}:{item.get('selected_candidate_id')}({item.get('status')})"
            for item in selections
            if isinstance(item, dict)
        ]
        return f"selections={selected}"
    if tool_name == "pose_frame_boundary_validation":
        touching = [
            f"{item.get('object_id')}:{','.join(item.get('boundary_sides') or [])}"
            for item in payload.get("diagnostics", [])
            if isinstance(item, dict) and item.get("touching_boundary")
        ]
        return (
            f"boundary_status={payload.get('boundary_status')} touching={touching} "
            f"retry_status={payload.get('retry_status')}"
        )
    if tool_name == "moge2_intrinsics":
        camera_intrinsics = payload.get("camera_intrinsics") or {}
        return (
            f"sampled_frame_indices={payload.get('sampled_frame_indices')} "
            f"K_fixed={camera_intrinsics.get('K_fixed')}"
        )
    if tool_name == "video_metric_depth":
        frames = payload.get("frames") or []
        metadata = payload.get("video_metadata") or {}
        return (
            f"frames={len(frames) or metadata.get('frame_count')} "
            f"tensor_sidecar={payload.get('tensor_sidecar')} "
            f"metric_depth_source={payload.get('metric_depth_source')}"
        )
    if tool_name == "sam3d_objects":
        meshes = payload.get("meshes") or payload.get("objects") or []
        mesh_paths = [
            item.get("mesh_path") or item.get("glb_path") or item.get("gaussian_ply_path")
            for item in meshes[:5]
            if isinstance(item, dict)
        ]
        return f"meshes={len(meshes)} sample_paths={mesh_paths}"
    if tool_name == "mesh_conditioning":
        objects = payload.get("objects") or []
        actions = [
            f"{item.get('object_id')}:{item.get('conditioning_action')}"
            for item in objects
            if isinstance(item, dict)
        ]
        ok_count = sum(1 for item in objects if isinstance(item, dict) and item.get("status") == "ok")
        projection_summaries = []
        for item in objects[:5]:
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            diagnostics = item.get("projection_diagnostics") or {}
            action = item.get("conditioning_action")
            area_ratio = diagnostics.get("estimated_area_scale_if_applied")
            depth_ratio = diagnostics.get("estimated_depth_scale_if_applied")
            if area_ratio is None or depth_ratio is None:
                continue
            projection_summaries.append(
                f"{item.get('object_id')}:mesh={action},depth_diag={depth_ratio:.3f},area_diag={area_ratio:.3f}"
            )
        return (
            f"objects={len(objects)} ok={ok_count} failed={len(objects) - ok_count} actions={actions} "
            f"projection_diagnostics={projection_summaries}"
        )
    if tool_name == "foundationpose":
        trajectories = payload.get("trajectories") or payload.get("objects") or []
        pose_count = 0
        failed = []
        intervals = []
        for item in trajectories:
            if not isinstance(item, dict):
                continue
            poses = item.get("poses") or item.get("trajectory") or []
            pose_count += len(poses)
            if item.get("tracking_start_frame_index") is not None:
                intervals.append(
                    f"{item.get('object_id')}:{item.get('tracking_start_frame_index')}->{item.get('tracking_end_frame_index')}"
                )
            if item.get("status") not in {None, "ok"}:
                failed.append(item.get("object_id"))
        return (
            f"objects={len(trajectories)} poses={pose_count} intervals={intervals} "
            f"failed_objects={failed}"
        )
    if tool_name == "pose_correction":
        gravity = payload.get("gravity_direction_camera")
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
            f"gravity_direction_camera={gravity} "
            f"rotation_applied={rotation.get('applied')} position_applied={position.get('applied')} "
        )
    if tool_name == "simulatable_world_reconstruction":
        trajectories = payload.get("corrected_trajectories") if isinstance(payload.get("corrected_trajectories"), dict) else {}
        return (
            f"stage={payload.get('simulatable_world_reconstruction_stage')} "
            f"trajectory_source={trajectories.get('source')} "
            f"objects={len(trajectories.get('objects') or [])}"
        )
    if tool_name == "pose_frames":
        object_frames = [
            (
                f"{item.get('object_id')}:"
                f"{item.get('first_full_visible_frame_index')}"
                f"({item.get('first_frame_selection_rule')})"
                f"->{item.get('last_full_visible_frame_index')}"
                f"({item.get('last_frame_selection_rule')})"
            )
            for item in payload.get("object_pose_frames", [])
            if isinstance(item, dict)
        ]
        return f"selected_frame_indices={payload.get('selected_frame_indices')} object_frames={object_frames}"
    if tool_name == "pose_sam3_masks":
        masks = payload.get("masks") or []
        objects = sorted({str(item.get("object_id")) for item in masks if item.get("object_id") is not None})
        return f"masks={len(masks)} objects={objects} mask_sidecar={payload.get('mask_sidecar')}"
    return f"keys={sorted(payload.keys())[:12]}"


class ExternalToolAdapter:
    tool_name = "external"
    env_var = ""
    default_command: Optional[str] = None
    artifact_name = "artifact.json"

    def __init__(
        self,
        artifacts: ArtifactManager,
        config: ModelConfig,
        dry_run: bool = False,
        video_metric_depth_worker: Optional[VideoMetricDepthWorkerClient] = None,
        foundationpose_worker: Optional[FoundationPoseWorkerClient] = None,
        sam3d_worker: Optional[SAM3DWorkerClient] = None,
        sam3_video_tracks_worker: Optional[SAM3VideoTracksWorkerClient] = None,
    ):
        self.artifacts = artifacts
        self.config = config
        self.dry_run = dry_run
        self.video_metric_depth_worker = video_metric_depth_worker
        self.foundationpose_worker = foundationpose_worker
        self.sam3d_worker = sam3d_worker
        self.sam3_video_tracks_worker = sam3_video_tracks_worker

    def _configured_command(self) -> Optional[str]:
        if not self.env_var:
            return self.default_command
        return os.getenv(self.env_var) or self.default_command

    def _load_existing(self, artifact_path: Path) -> Optional[ToolResult]:
        payload = self.artifacts.read_optional(artifact_path)
        if payload is None:
            return None
        _log_tool(self.tool_name, f"loaded artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(
            tool_name=self.tool_name,
            status="loaded",
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def _write_placeholder(self, artifact_path: Path, payload: Dict[str, Any], status: str) -> ToolResult:
        self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"{status} artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(
            tool_name=self.tool_name,
            status=status,
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def _activation_from_poses(self, poses: list[Dict[str, Any]]) -> Dict[str, Any]:
        frame_indices = sorted(
            int(pose["frame_index"])
            for pose in poses
            if isinstance(pose, dict)
            and pose.get("frame_index") is not None
            and pose.get("corrected_pose_4x4") is not None
        )
        if not frame_indices:
            return {
                "state_model": "active_interval_absent_outside",
                "active": False,
                "first_active_frame": None,
                "last_active_frame": None,
                "active_frame_count": 0,
                "inactive_policy": "absent",
                "appearance_policy": "appear_at_first_active_frame_disappear_after_last_active_frame",
                "continuous_active_interval": False,
            }
        first_frame = int(frame_indices[0])
        last_frame = int(frame_indices[-1])
        return {
            "state_model": "active_interval_absent_outside",
            "active": True,
            "first_active_frame": first_frame,
            "last_active_frame": last_frame,
            "active_frame_count": len(frame_indices),
            "inactive_policy": "absent",
            "appearance_policy": "appear_at_first_active_frame_disappear_after_last_active_frame",
            "continuous_active_interval": len(frame_indices) == last_frame - first_frame + 1,
        }

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing_payload = self.artifacts.read_optional(artifact_path)
        if existing_payload is not None and existing_payload.get("physics_rollout") is not None:
            _log_tool(self.tool_name, f"loaded artifact={artifact_path} {_payload_summary(self.tool_name, existing_payload)}")
            return ToolResult(
                tool_name=self.tool_name,
                status="loaded",
                artifact_path=str(artifact_path),
                payload=existing_payload,
            )
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="dry_run",
            )
        command = self._configured_command()
        if not command:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="tool_not_configured",
            )
        args = shlex.split(command) + [
            "--video",
            str(scene.video_path),
            "--object-plan",
            str(_scene_object_plan_path(question_dir)),
            "--output",
            str(artifact_path),
        ]
        env = _external_tool_env(question_dir, self.artifacts.debug_artifacts)
        start = time.perf_counter()
        _log_tool(
            self.tool_name,
            f"command start command={command} video={scene.video_path} output={artifact_path}",
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"stderr {stderr}")
        if completed.returncode != 0:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=(completed.stderr or completed.stdout or "").strip(),
            )
        payload = self.artifacts.read_optional(artifact_path) or {}
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(
            tool_name=self.tool_name,
            status="ok",
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "target_objects": [item.__dict__ for item in object_plan.target_objects],
            "note": "Placeholder artifact. Configure the tool command env var to run the real adapter.",
        }


class VideoMetricDepthAdapter(ExternalToolAdapter):
    tool_name = "video_metric_depth"
    env_var = "PHYSMIND_VIDEO_METRIC_DEPTH_CMD"
    default_command = DEFAULT_VIDEO_METRIC_DEPTH_CMD
    artifact_name = "video_metric_depth.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        depth_partition_route = None
        segment_depth_alignment_route = None
        if not self.dry_run:
            policy_benchmark = _route_policy_benchmark_for_scene(scene)
            scenario = str(getattr(scene, "scenario", "") or "").strip() or None
            depth_partition_route = _require_depth_partition_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario,
                object_plan=object_plan,
            )
            segment_depth_alignment_route = (
                _require_segment_depth_alignment_route(
                    policy_benchmark=policy_benchmark,
                    scenario=scenario,
                    object_plan=object_plan,
                    depth_partition_route=depth_partition_route,
                )
            )
        existing = self._load_existing(artifact_path)
        if existing:
            if depth_partition_route is not None and isinstance(existing.payload, dict):
                _record_video_metric_depth_route_results(
                    existing.payload,
                    depth_partition_route=depth_partition_route,
                    segment_depth_alignment_route=segment_depth_alignment_route,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="dry_run",
            )

        if self.video_metric_depth_worker is not None:
            _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            response = self.video_metric_depth_worker.request(
                {
                    "task_type": self.tool_name,
                    "video": str(scene.video_path),
                    "object_plan": str(_scene_object_plan_path(question_dir)),
                    "output": str(artifact_path),
                    "question_dir": str(question_dir),
                    "debug_artifacts": self.artifacts.debug_artifacts,
                }
            )
            elapsed = time.perf_counter() - start
            _log_tool(self.tool_name, f"worker request end status={response.get('status')} elapsed={elapsed:.1f}s")
            if response.get("status") != "ok":
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(response.get("message") or response),
                )
            payload = response.get("payload") or self.artifacts.read_optional(artifact_path) or {}
            if depth_partition_route is not None:
                _record_video_metric_depth_route_results(
                    payload,
                    depth_partition_route=depth_partition_route,
                    segment_depth_alignment_route=segment_depth_alignment_route,
                )
                self.artifacts.write(artifact_path, payload)
            _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
            return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)

        command = self._configured_command()
        if not command:
            result = self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="tool_not_configured",
            )
            if depth_partition_route is not None and isinstance(result.payload, dict):
                _record_video_metric_depth_route_results(
                    result.payload,
                    depth_partition_route=depth_partition_route,
                    segment_depth_alignment_route=segment_depth_alignment_route,
                )
                self.artifacts.write(artifact_path, result.payload)
            return result
        args = shlex.split(command) + [
            "--video",
            str(scene.video_path),
            "--object-plan",
            str(_scene_object_plan_path(question_dir)),
            "--output",
            str(artifact_path),
        ]
        env = _external_tool_env(question_dir, self.artifacts.debug_artifacts)
        start = time.perf_counter()
        _log_tool(
            self.tool_name,
            f"command start command={command} video={scene.video_path} output={artifact_path}",
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"stderr {stderr}")
        if completed.returncode != 0:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=(completed.stderr or completed.stdout or "").strip(),
            )

        payload = self.artifacts.read_optional(artifact_path) or {}
        if depth_partition_route is not None:
            _record_video_metric_depth_route_results(
                payload,
                depth_partition_route=depth_partition_route,
                segment_depth_alignment_route=segment_depth_alignment_route,
            )
            self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)


class PoseCorrectionAdapter(ExternalToolAdapter):
    tool_name = "pose_correction"
    env_var = "PHYSMIND_GEOCALIB_CMD"
    default_command = DEFAULT_GEOCALIB_CMD
    artifact_name = "pose_correction.json"

    def __init__(
        self,
        *,
        artifacts: ArtifactManager,
        config: ModelConfig,
        dry_run: bool,
        geocalib_worker: Optional[GeoCalibWorkerClient] = None,
    ):
        super().__init__(artifacts=artifacts, config=config, dry_run=dry_run)
        self.geocalib_worker = geocalib_worker
        # Physion++-only lossless memo for pose-correction inputs (depth sidecar,
        # SAM3 mask records/arrays, foundationpose mesh paths, mesh geometry). Entries are
        # keyed by source-file signatures (path, mtime_ns, size) so in-place rewrites
        # invalidate naturally, and every hit returns defensive copies so callers can
        # never mutate cached state. Cleared at each run() entry.
        self._question_cache: Dict[Any, Any] = {}
        self._question_cache_enabled = False

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        self._question_cache.clear()
        scenario = self._object_plan_scenario(object_plan).strip().lower()
        self._question_cache_enabled = scenario.endswith("_pp")
        ground_motion_route = None
        gravity_estimator_route = None
        gravity_constraints_route = None
        rotation_policy_route = None
        support_snap_route = None
        static_fixture_flush_route = None
        line_layout_route = None
        agent_geometry_route_record = None
        agent_trajectory_route_record = None
        collision_patient_motion_gate_route = None
        collision_patient_drop_route = None
        mass_extra_mesh_adopt_route = None
        mass_agent_flush_route = None
        mass_ball_trajectory_route = None
        if not self.dry_run:
            policy_benchmark = _route_policy_benchmark_for_scene(scene)
            ground_motion_route = _require_ground_motion_gate_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            gravity_estimator_route, gravity_constraints_route = (
                _require_gravity_route_pair(
                    policy_benchmark=policy_benchmark,
                    scenario=scenario or None,
                    object_plan=object_plan,
                )
            )
            rotation_policy_route = _require_rotation_policy_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            support_snap_route = _require_support_snap_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            static_fixture_flush_route = _require_static_fixture_flush_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            line_layout_route = _require_line_layout_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            collision_patient_motion_gate_route = (
                _require_collision_patient_motion_gate_route(
                    policy_benchmark=policy_benchmark,
                    scenario=scenario or None,
                    object_plan=object_plan,
                )
            )
            collision_patient_drop_route = _require_collision_patient_drop_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            mass_extra_mesh_adopt_route = _require_mass_extra_mesh_adopt_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            mass_agent_flush_route = _require_mass_agent_flush_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            mass_ball_trajectory_route = _require_mass_ball_trajectory_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario or None,
                object_plan=object_plan,
            )
            agent_geometry_route_record = (
                self._foundationpose_agent_geometry_route_record(object_plan)
            )
            agent_trajectory_route_record = self._agent_trajectory_route_record(
                object_plan
            )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if ground_motion_route is not None:
                _record_ground_motion_gate_result(
                    existing.payload,
                    ground_motion_route,
                )
            if (
                gravity_estimator_route is not None
                and gravity_constraints_route is not None
            ):
                _record_gravity_route_results(
                    existing.payload,
                    estimator_route=gravity_estimator_route,
                    constraints_route=gravity_constraints_route,
                )
            if rotation_policy_route is not None:
                _record_rotation_policy_result(
                    existing.payload,
                    rotation_policy_route,
                )
            for route_record, key, decision_id, supported_routes in (
                (
                    support_snap_route,
                    "support_snap_route",
                    SUPPORT_SNAP_DECISION_ID,
                    SUPPORT_SNAP_ROUTES,
                ),
                (
                    static_fixture_flush_route,
                    "static_fixture_flush_route",
                    STATIC_FIXTURE_FLUSH_DECISION_ID,
                    STATIC_FIXTURE_FLUSH_ROUTES,
                ),
                (
                    line_layout_route,
                    "line_layout_route",
                    LINE_LAYOUT_DECISION_ID,
                    LINE_LAYOUT_ROUTES,
                ),
                (
                    collision_patient_motion_gate_route,
                    "collision_patient_motion_gate_route",
                    COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
                    frozenset({COLLISION_PATIENT_MOTION_GATE_ROUTE}),
                ),
                (
                    collision_patient_drop_route,
                    "collision_patient_drop_route",
                    COLLISION_PATIENT_DROP_DECISION_ID,
                    frozenset({COLLISION_PATIENT_DROP_ROUTE}),
                ),
                (
                    mass_extra_mesh_adopt_route,
                    "mass_extra_flush_and_mesh_adopt_route",
                    MASS_EXTRA_MESH_ADOPT_DECISION_ID,
                    frozenset({MASS_EXTRA_MESH_ADOPT_ROUTE}),
                ),
                (
                    mass_agent_flush_route,
                    "mass_agent_flush_route",
                    MASS_AGENT_FLUSH_DECISION_ID,
                    frozenset({MASS_AGENT_FLUSH_ROUTE}),
                ),
                (
                    mass_ball_trajectory_route,
                    "mass_ball_precontact_trajectory_route",
                    MASS_BALL_TRAJECTORY_DECISION_ID,
                    frozenset({MASS_BALL_TRAJECTORY_ROUTE}),
                ),
            ):
                if route_record is not None:
                    _record_pose_route_result(
                        existing.payload,
                        route_record,
                        key=key,
                        decision_id=decision_id,
                        supported_routes=supported_routes,
                    )
            policy_mismatch = (
                existing.payload.get("physion_pp_agent_geometry_policy") != expected_policy
                or existing.payload.get("foundationpose_agent_geometry_route")
                != agent_geometry_route_record
                or existing.payload.get("agent_trajectory_route")
                != agent_trajectory_route_record
            )
            if (
                "trajectory_correction" not in existing.payload
                or "rotation_correction" not in existing.payload
                or "support_plane_position_correction" not in existing.payload
                or "pose_overlap_optimization" not in existing.payload
                or policy_mismatch
            ):
                existing_ground_contact = existing.payload.get("ground_contact_classification")
                if (
                    self._per_object_ground_contact_enabled(object_plan)
                    and not self.dry_run
                    and not (
                        isinstance(existing_ground_contact, dict)
                        and existing_ground_contact.get("applied") is True
                    )
                ):
                    existing_ground_contact = self._classify_ground_contact_per_object(
                        question_dir=question_dir,
                        object_plan=object_plan,
                    )
                self._append_trajectory_correction(
                    payload=existing.payload,
                    question_dir=question_dir,
                    object_plan=object_plan,
                    ground_contact=(
                        existing_ground_contact if isinstance(existing_ground_contact, dict) else None
                    ),
                )
                self.artifacts.write(artifact_path, existing.payload)
            elif ground_motion_route is not None:
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                {
                    **self.placeholder_payload(scene=scene, object_plan=object_plan),
                    "question_id": object_plan.question_id,
                    "pose_correction_stage": "gravity_direction_estimation",
                    "gravity_direction_camera": None,
                    "gravity_direction_coordinate_frame": "opencv_camera",
                    "trajectory_correction": self._trajectory_correction_placeholder(object_plan),
                },
                status="dry_run",
            )

        per_object_ground = self._per_object_ground_contact_enabled(object_plan)
        ground_contact: Optional[Dict[str, Any]] = None
        if per_object_ground:
            ground_contact = self._classify_ground_contact_per_object(
                question_dir=question_dir,
                object_plan=object_plan,
            )
        overlap_gate_open = (
            bool(ground_contact and ground_contact.get("on_ground_object_ids"))
            if per_object_ground
            else self._horizontal_plane_motion(object_plan).get("applies") is True
        )
        if self._geocalib_gravity_enabled(object_plan):
            overlap_gate_open = False
        if overlap_gate_open:
            payload = {
                "tool": self.tool_name,
                "status": "ok",
                "scene_index": scene.scene_index,
                "question_id": object_plan.question_id,
                "pose_correction_stage": "gravity_rotation_position_correction",
                "gravity_estimation_method": "sam3_projection_overlap_plane_normal_optimization",
                "gravity_direction_camera": None,
                "gravity_direction_coordinate_frame": "opencv_camera",
                "gravity_direction_convention": (
                    "unit up-vector in OpenCV camera coordinates selected by SAM3 projection overlap grid search"
                ),
                "horizontal_plane_motion": self._horizontal_plane_motion(object_plan),
                "roll_stabilization": self._roll_stabilization(object_plan),
            }
            self._append_trajectory_correction(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                ground_contact=ground_contact,
            )
            overlap = payload.get("pose_overlap_optimization")
            if isinstance(overlap, dict) and overlap.get("applied") is True:
                self.artifacts.write(artifact_path, payload)
                _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
                return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)
            if not per_object_ground:
                payload["status"] = "tool_error"
                payload["error_message"] = (
                    overlap.get("reason") if isinstance(overlap, dict) else "SAM3 projection overlap grid search failed"
                )
                self.artifacts.write(artifact_path, payload)
                _log_tool(
                    self.tool_name,
                    f"tool_error artifact={artifact_path} error={_short_text(str(payload['error_message']), 500)}",
                )
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(payload["error_message"]),
                    payload=payload,
                )
            overlap_reason = overlap.get("reason") if isinstance(overlap, dict) else "SAM3 projection overlap grid search failed"
            _log_tool(
                self.tool_name,
                "per_object_ground_gate overlap path failed "
                f"({_short_text(str(overlap_reason), 300)}); falling back to geocalib gravity estimation",
            )

        if self.geocalib_worker is not None:
            _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            request = {
                "task_type": self.tool_name,
                "video": str(scene.video_path),
                "output": str(artifact_path),
                "question_dir": str(question_dir),
                "num_frames": 8,
            }
            if frame_range is not None:
                request["frame_start"], request["frame_end"] = frame_range
            gravity_z_constraint = self._geocalib_gravity_z_constraint(object_plan)
            if gravity_z_constraint is not None:
                request["gravity_z_constraint"] = gravity_z_constraint
            response = self.geocalib_worker.request(request)
            elapsed = time.perf_counter() - start
            _log_tool(self.tool_name, f"worker request end status={response.get('status')} elapsed={elapsed:.1f}s")
            if response.get("status") != "ok":
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(response.get("message") or response),
                )
            payload = response.get("payload") or self.artifacts.read_optional(artifact_path) or {}
            result = self._write_pose_correction_payload(
                scene=scene,
                object_plan=object_plan,
                question_dir=question_dir,
                artifact_path=artifact_path,
                payload=payload,
                ground_contact=ground_contact,
            )
            _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, result.payload or {})}")
            return result

        command = self._configured_command()
        if not command:
            placeholder = self.placeholder_payload(scene=scene, object_plan=object_plan)
            if ground_motion_route is not None:
                _record_ground_motion_gate_result(
                    placeholder,
                    ground_motion_route,
                )
            if (
                gravity_estimator_route is not None
                and gravity_constraints_route is not None
            ):
                _record_gravity_route_results(
                    placeholder,
                    estimator_route=gravity_estimator_route,
                    constraints_route=gravity_constraints_route,
                )
            if rotation_policy_route is not None:
                _record_rotation_policy_result(
                    placeholder,
                    rotation_policy_route,
                )
            return self._write_placeholder(
                artifact_path,
                placeholder,
                status="tool_not_configured",
            )
        args = shlex.split(command) + [
            "--video",
            str(scene.video_path),
            "--output",
            str(artifact_path),
            "--num-frames",
            "8",
        ]
        if frame_range is not None:
            args.extend(["--frame-start", str(frame_range[0]), "--frame-end", str(frame_range[1])])
        gravity_z_constraint = self._geocalib_gravity_z_constraint(object_plan)
        if gravity_z_constraint is not None:
            args.extend(["--gravity-z-constraint", gravity_z_constraint])
        env = _external_tool_env(question_dir, self.artifacts.debug_artifacts)
        start = time.perf_counter()
        _log_tool(self.tool_name, f"command start command={command} video={scene.video_path} output={artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"stderr {stderr}")
        if completed.returncode != 0:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=(completed.stderr or completed.stdout or "").strip(),
            )
        payload = self.artifacts.read_optional(artifact_path) or {}
        result = self._write_pose_correction_payload(
            scene=scene,
            object_plan=object_plan,
            question_dir=question_dir,
            artifact_path=artifact_path,
            payload=payload,
            ground_contact=ground_contact,
        )
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, result.payload or {})}")
        return result

    def _write_pose_correction_payload(
        self,
        *,
        scene: ClevrerScene,
        object_plan: ObjectPlan,
        question_dir: Path,
        artifact_path: Path,
        payload: Dict[str, Any],
        ground_contact: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        payload["scene_index"] = scene.scene_index
        payload["question_id"] = object_plan.question_id
        payload["pose_correction_stage"] = "gravity_rotation_position_correction"
        if self._gravity_roll_zero_enabled(object_plan):
            self._zero_gravity_roll_component(payload)
        self._append_trajectory_correction(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
            ground_contact=ground_contact,
        )
        self.artifacts.write(artifact_path, payload)
        return ToolResult(
            tool_name=self.tool_name,
            status="ok",
            artifact_path=str(artifact_path),
            payload=payload,
        )







    def _standard_corrected_trajectory_source(self, *, source: Dict[str, Any], source_name: str) -> Dict[str, Any]:
        objects = []
        for item in source.get("objects", []):
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            poses = []
            missing_pose_count = 0
            for pose in item.get("poses", []):
                if not isinstance(pose, dict):
                    continue
                corrected_pose = pose.get("corrected_pose_4x4") or pose.get("pose_4x4")
                if corrected_pose is None:
                    missing_pose_count += 1
                    continue
                corrected_translation = pose.get("corrected_translation_camera")
                if corrected_translation is None:
                    try:
                        matrix = np.asarray(corrected_pose, dtype=np.float64).reshape(4, 4)
                        corrected_translation = matrix[:3, 3].tolist()
                    except (TypeError, ValueError):
                        missing_pose_count += 1
                        continue
                poses.append(
                    {
                        **pose,
                        "corrected_pose_4x4": corrected_pose,
                        "corrected_translation_camera": corrected_translation,
                    }
                )
            objects.append(
                {
                    **item,
                    "status": "ok" if poses and missing_pose_count == 0 else "partial",
                    "source": source_name,
                    "render_pose_source": source_name,
                    "pose_count": len(poses),
                    "missing_pose_count": missing_pose_count,
                    "activation": self._activation_from_poses(poses),
                    "poses": poses,
                }
            )
        return {
            "applied": bool(objects),
            "source": source_name,
            "pose_field": "corrected_pose_4x4",
            "translation_field": "corrected_translation_camera",
            "objects": objects,
        }

    def _trajectory_correction_placeholder(self, object_plan: ObjectPlan) -> Dict[str, Any]:
        horizontal = self._horizontal_plane_motion(object_plan)
        return {
            "support_plane_position_correction": {
                "applied": False,
                "reason": "dry_run",
                "horizontal_plane_motion": horizontal,
                "objects": [],
            }
        }

    def _horizontal_plane_motion(self, object_plan: ObjectPlan) -> Dict[str, Any]:
        special_scene = object_plan.special_scene if isinstance(object_plan.special_scene, dict) else {}
        horizontal = special_scene.get("horizontal_plane_motion")
        horizontal = horizontal if isinstance(horizontal, dict) else {}
        # The resolved POS-002 route is the sole authority for applying the
        # mass-collision ground-motion override.
        if (
            self._ground_motion_route(object_plan)
            == MASS_COLLISION_FORCED_GROUND_MOTION_ROUTE
            and horizontal.get("applies") is not True
        ):
            return {
                "applies": True,
                "forced_by": "physion_pp_mass_collision_scenario_prior",
                "vlm_original": horizontal or None,
            }
        return horizontal

    def _roll_stabilization(self, object_plan: ObjectPlan) -> Dict[str, Any]:
        special_scene = object_plan.special_scene if isinstance(object_plan.special_scene, dict) else {}
        roll_stabilization = special_scene.get("roll_stabilization")
        return roll_stabilization if isinstance(roll_stabilization, dict) else {}

    def _object_plan_scenario(self, object_plan: ObjectPlan) -> str:
        special_scene = object_plan.special_scene if isinstance(object_plan.special_scene, dict) else {}
        scene_metadata = (
            special_scene.get("scene_metadata")
            if isinstance(special_scene.get("scene_metadata"), dict)
            else {}
        )
        for value in (scene_metadata.get("scenario"), special_scene.get("scenario")):
            if value:
                return str(value).strip()
        return ""

    def _ground_motion_route(self, object_plan: ObjectPlan) -> Optional[str]:
        special_scene = (
            object_plan.special_scene
            if isinstance(object_plan.special_scene, dict)
            else {}
        )
        route_record = special_scene.get("ground_motion_gate_route")
        if not isinstance(route_record, dict):
            return None
        route = route_record.get("route")
        if route not in GROUND_MOTION_GATE_ROUTES:
            raise ValueError(f"unsupported ground-motion-gate route: {route!r}")
        return str(route)



    def _foundationpose_agent_geometry_route_record(
        self,
        object_plan: ObjectPlan,
    ) -> Optional[Dict[str, Any]]:
        return foundationpose_agent_geometry_route(
            object_plan,
            require_for_main_scenario=not self.dry_run,
        )

    def _agent_trajectory_route_record(
        self,
        object_plan: ObjectPlan,
    ) -> Optional[Dict[str, Any]]:
        scenario = self._object_plan_scenario(object_plan).strip().lower()
        expected_decision_id = AGENT_TRAJECTORY_DECISION_BY_SCENARIO.get(scenario)
        special_scene = (
            object_plan.special_scene
            if isinstance(object_plan.special_scene, dict)
            else {}
        )
        route_record = special_scene.get("agent_trajectory_route")
        if not isinstance(route_record, dict):
            if expected_decision_id is not None and not self.dry_run:
                raise ValueError(
                    f"{scenario!r} is missing its agent-trajectory route record"
                )
            return None
        if expected_decision_id is None:
            raise ValueError(
                f"agent-trajectory route is not applicable to scenario {scenario!r}"
            )
        if route_record.get("decision_id") != expected_decision_id:
            raise ValueError(
                "agent-trajectory route has an unexpected decision_id: "
                f"{route_record.get('decision_id')!r}, expected {expected_decision_id!r}"
            )
        route = route_record.get("route")
        if route not in AGENT_TRAJECTORY_ROUTES_BY_DECISION[expected_decision_id]:
            raise ValueError(
                f"unsupported agent-trajectory route for {expected_decision_id}: {route!r}"
            )
        context = route_record.get("context")
        if not isinstance(context, dict):
            raise ValueError("agent-trajectory route is missing its context")
        expected_context = {
            "benchmark": "physion_pp",
            "scenario": scenario,
            "role": "agent",
        }
        for key, expected in expected_context.items():
            if context.get(key) != expected:
                raise ValueError(
                    "agent-trajectory route context mismatch: "
                    f"{key}={context.get(key)!r}, expected {expected!r}"
                )
        return route_record

    def _per_object_ground_contact_enabled(self, object_plan: ObjectPlan) -> bool:
        return (
            self._ground_motion_route(object_plan)
            == PHYSION_PP_AGENT_FREE_GROUND_MOTION_ROUTE
        )

    def _gravity_route_value(self, object_plan: ObjectPlan, key: str) -> Optional[str]:
        special_scene = (
            object_plan.special_scene
            if isinstance(object_plan.special_scene, dict)
            else {}
        )
        route_record = special_scene.get(key)
        if not isinstance(route_record, dict):
            return None
        route = route_record.get("route")
        supported = (
            GRAVITY_ESTIMATOR_ROUTES
            if key == "gravity_estimator_route"
            else GRAVITY_CONSTRAINT_ROUTES
        )
        if route not in supported:
            raise ValueError(f"unsupported route in {key}: {route!r}")
        return str(route)

    def _geocalib_gravity_enabled(self, object_plan: ObjectPlan) -> bool:
        return self._gravity_route_value(
            object_plan,
            "gravity_estimator_route",
        ) in {
            FRICTION_COLLISION_GEOCALIB_GRAVITY_ROUTE,
            MASS_COLLISION_GEOCALIB_GRAVITY_ROUTE,
        }

    def _gravity_roll_zero_enabled(self, object_plan: ObjectPlan) -> bool:
        return self._gravity_route_value(
            object_plan,
            "gravity_constraints_route",
        ) in {
            FRICTION_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
            MASS_COLLISION_GRAVITY_CONSTRAINTS_ROUTE,
        }


    def _geocalib_gravity_z_constraint(self, object_plan: ObjectPlan) -> Optional[str]:
        if (
            self._gravity_route_value(object_plan, "gravity_constraints_route")
            == FRICTION_COLLISION_GRAVITY_CONSTRAINTS_ROUTE
        ):
            return "negative"
        return None

    def _ground_motion_gate(self, *, payload: Dict[str, Any], object_plan: ObjectPlan) -> Dict[str, Any]:
        """Single gate for the ground-plane machinery.

        The resolved POS-002 route chooses either the Physion++ agent/free-vs-fixture
        object rule or the scene-level horizontal-plane assessment. Airborne collision
        patients remain exempt from support snapping through the independent motion test.
        """
        route_record = (
            object_plan.special_scene.get("ground_motion_gate_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        if self._per_object_ground_contact_enabled(object_plan):
            classification = payload.get("ground_contact_classification")
            classification = classification if isinstance(classification, dict) else {}
            on_ground = {str(value) for value in classification.get("on_ground_object_ids") or []}
            airborne = {str(value) for value in classification.get("airborne_object_ids") or []}
            return {
                "mode": "per_object_ground_contact",
                "applies": bool(on_ground),
                "reason": (
                    None
                    if on_ground
                    else "per-object ground-contact classification found no on-ground objects"
                ),
                "on_ground_object_ids": on_ground,
                "airborne_object_ids": airborne,
                "resolved_route": deepcopy(route_record),
            }
        horizontal = self._horizontal_plane_motion(object_plan)
        # friction_collision / mass_collision: the motion-tested moving seg2 patient is
        # airborne (knocked off the ledge / dropping in from above), so VDA depth on it
        # is untrustworthy -- exempt it from the snap, drop it from the ground-height
        # fit, and keep it out of the gravity candidate scoring. Everything else keeps
        # the scene-level gate semantics.
        pp_exempt: set[str] = set()
        for block_key in ("physion_pp_friction_collision", "physion_pp_mass_collision"):
            block = payload.get(block_key)
            if isinstance(block, dict):
                pp_exempt |= {str(value) for value in block.get("ground_snap_exempt_object_ids") or []}
        return {
            "mode": "scene_horizontal_plane_motion",
            "applies": horizontal.get("applies") is True,
            "reason": "object_plan.special_scene.horizontal_plane_motion.applies is not true",
            "on_ground_object_ids": None,
            "airborne_object_ids": pp_exempt,
            "resolved_route": deepcopy(route_record),
        }





    def _classify_ground_contact_per_object(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> Dict[str, Any]:
        scenario = self._object_plan_scenario(object_plan)
        route = self._ground_motion_route(object_plan)
        result: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_agent_free_rule",
            "scenario": scenario,
            "supersedes_scene_horizontal_plane_motion": True,
            "decisions_by_object_id": {},
            "on_ground_object_ids": [],
            "airborne_object_ids": [],
            "resolved_route": deepcopy(
                object_plan.special_scene.get("ground_motion_gate_route")
            ),
        }
        targets = list(object_plan.target_objects or [])
        if not targets:
            result["reason"] = "object_plan.target_objects is empty"
            return result

        decisions: Dict[str, Dict[str, Any]] = {}
        dynamic_targets: list[TargetObject] = []
        for item in targets:
            object_id = str(item.object_id)
            if _is_physion_static_ground_fixture_track(item.source_track_id):
                decisions[object_id] = {
                    "on_ground": True,
                    "answer": "STATIC_FIXTURE",
                    "classified_by": "static_ground_fixture_track_prefix",
                    "source_track_id": item.source_track_id,
                }
            else:
                dynamic_targets.append(item)

        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(
                question_dir,
                "sam3_video_tracks.json",
            )
        ) or {}
        role_binding = (
            (tracks_payload.get("physion_tracking") or {}).get("role_binding")
            or {}
        )
        # Two-segment Bouncy Wall has one agent per segment; both are free-moving.
        assignments = (
            role_binding.get("assignments")
            if isinstance(role_binding.get("assignments"), dict)
            else {}
        )
        agent_tracks = {
            str(assignments[segment]["agent"])
            for segment in ("seg1", "seg2")
            if isinstance(assignments.get(segment), dict)
            and assignments[segment].get("agent")
        }
        agent_track = str(role_binding.get("agent_track") or "")
        if agent_track:
            agent_tracks.add(agent_track)
        if not agent_tracks:
            raise ValueError(
                "Physion++ agent/free ground route requires an agent track in "
                "physion_tracking.role_binding"
            )
        for item in dynamic_targets:
            object_id = str(item.object_id)
            is_agent = str(item.source_track_id) in agent_tracks
            decisions[object_id] = {
                "on_ground": not is_agent,
                "answer": "RULE_AGENT_FREE" if is_agent else "RULE_ON_GROUND",
                "classified_by": "physion_pp_agent_free_rule",
                "source_track_id": item.source_track_id,
                "agent_tracks": sorted(agent_tracks),
            }

        result["decisions_by_object_id"] = decisions
        result["on_ground_object_ids"] = sorted(
            object_id for object_id, decision in decisions.items() if decision.get("on_ground") is True
        )
        result["airborne_object_ids"] = sorted(
            object_id for object_id, decision in decisions.items() if decision.get("on_ground") is False
        )
        result["applied"] = True
        _log_tool(
            self.tool_name,
            f"ground_contact scenario={scenario} rule=physion_pp_agent_free "
            f"on_ground={result['on_ground_object_ids']} airborne={result['airborne_object_ids']}",
        )
        return result

    def _append_trajectory_correction(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        ground_contact: Optional[Dict[str, Any]] = None,
    ) -> None:
        ground_motion_route = (
            object_plan.special_scene.get("ground_motion_gate_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        if isinstance(ground_motion_route, dict):
            _record_ground_motion_gate_result(payload, ground_motion_route)
        gravity_estimator_route = (
            object_plan.special_scene.get("gravity_estimator_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        gravity_constraints_route = (
            object_plan.special_scene.get("gravity_constraints_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        if (
            isinstance(gravity_estimator_route, dict)
            and isinstance(gravity_constraints_route, dict)
        ):
            _record_gravity_route_results(
                payload,
                estimator_route=gravity_estimator_route,
                constraints_route=gravity_constraints_route,
            )
        rotation_policy_route = (
            object_plan.special_scene.get("rotation_policy_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        if isinstance(rotation_policy_route, dict):
            _record_rotation_policy_result(payload, rotation_policy_route)
        support_snap_route = self._pose_route_record(
            object_plan,
            key="support_snap_route",
            decision_id=SUPPORT_SNAP_DECISION_ID,
            supported_routes=SUPPORT_SNAP_ROUTES,
        )
        static_fixture_flush_route = self._pose_route_record(
            object_plan,
            key="static_fixture_flush_route",
            decision_id=STATIC_FIXTURE_FLUSH_DECISION_ID,
            supported_routes=STATIC_FIXTURE_FLUSH_ROUTES,
        )
        line_layout_route = self._pose_route_record(
            object_plan,
            key="line_layout_route",
            decision_id=LINE_LAYOUT_DECISION_ID,
            supported_routes=LINE_LAYOUT_ROUTES,
        )
        collision_patient_motion_gate_route = self._pose_route_record(
            object_plan,
            key="collision_patient_motion_gate_route",
            decision_id=COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
            supported_routes=frozenset({COLLISION_PATIENT_MOTION_GATE_ROUTE}),
        )
        collision_patient_drop_route = self._pose_route_record(
            object_plan,
            key="collision_patient_drop_route",
            decision_id=COLLISION_PATIENT_DROP_DECISION_ID,
            supported_routes=frozenset({COLLISION_PATIENT_DROP_ROUTE}),
        )
        mass_extra_mesh_adopt_route = self._pose_route_record(
            object_plan,
            key="mass_extra_flush_and_mesh_adopt_route",
            decision_id=MASS_EXTRA_MESH_ADOPT_DECISION_ID,
            supported_routes=frozenset({MASS_EXTRA_MESH_ADOPT_ROUTE}),
        )
        mass_agent_flush_route = self._pose_route_record(
            object_plan,
            key="mass_agent_flush_route",
            decision_id=MASS_AGENT_FLUSH_DECISION_ID,
            supported_routes=frozenset({MASS_AGENT_FLUSH_ROUTE}),
        )
        mass_ball_trajectory_route = self._pose_route_record(
            object_plan,
            key="mass_ball_precontact_trajectory_route",
            decision_id=MASS_BALL_TRAJECTORY_DECISION_ID,
            supported_routes=frozenset({MASS_BALL_TRAJECTORY_ROUTE}),
        )
        for route_record, key, decision_id, supported_routes in (
            (
                support_snap_route,
                "support_snap_route",
                SUPPORT_SNAP_DECISION_ID,
                SUPPORT_SNAP_ROUTES,
            ),
            (
                static_fixture_flush_route,
                "static_fixture_flush_route",
                STATIC_FIXTURE_FLUSH_DECISION_ID,
                STATIC_FIXTURE_FLUSH_ROUTES,
            ),
            (
                line_layout_route,
                "line_layout_route",
                LINE_LAYOUT_DECISION_ID,
                LINE_LAYOUT_ROUTES,
            ),
            (
                collision_patient_motion_gate_route,
                "collision_patient_motion_gate_route",
                COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
                frozenset({COLLISION_PATIENT_MOTION_GATE_ROUTE}),
            ),
            (
                collision_patient_drop_route,
                "collision_patient_drop_route",
                COLLISION_PATIENT_DROP_DECISION_ID,
                frozenset({COLLISION_PATIENT_DROP_ROUTE}),
            ),
            (
                mass_extra_mesh_adopt_route,
                "mass_extra_flush_and_mesh_adopt_route",
                MASS_EXTRA_MESH_ADOPT_DECISION_ID,
                frozenset({MASS_EXTRA_MESH_ADOPT_ROUTE}),
            ),
            (
                mass_agent_flush_route,
                "mass_agent_flush_route",
                MASS_AGENT_FLUSH_DECISION_ID,
                frozenset({MASS_AGENT_FLUSH_ROUTE}),
            ),
            (
                mass_ball_trajectory_route,
                "mass_ball_precontact_trajectory_route",
                MASS_BALL_TRAJECTORY_DECISION_ID,
                frozenset({MASS_BALL_TRAJECTORY_ROUTE}),
            ),
        ):
            if route_record is not None:
                _record_pose_route_result(
                    payload,
                    route_record,
                    key=key,
                    decision_id=decision_id,
                    supported_routes=supported_routes,
                )
        payload["physion_pp_agent_geometry_policy"] = geometry_policy
        geometry_route_record = self._foundationpose_agent_geometry_route_record(
            object_plan
        )
        if geometry_route_record is not None:
            payload["foundationpose_agent_geometry_route"] = deepcopy(
                geometry_route_record
            )
        trajectory_route_record = self._agent_trajectory_route_record(object_plan)
        if trajectory_route_record is not None:
            payload["agent_trajectory_route"] = deepcopy(trajectory_route_record)
        if ground_contact is not None:
            payload["ground_contact_classification"] = ground_contact
        if fc_context is not None:
            payload["physion_pp_friction_collision"] = fc_context
        if mc_context is not None:
            payload["physion_pp_mass_collision"] = mc_context
        foundationpose_trajectories = self._foundationpose_trajectory_source(question_dir)
        if self._geocalib_gravity_enabled(object_plan):
            overlap_optimization = {
                "applied": False,
                "reason": "gravity comes from geocalib with roll zeroed; overlap grid search disabled",
                "horizontal_plane_motion": self._horizontal_plane_motion(object_plan),
            }
        else:
            overlap_optimization = self._overlap_optimized_pose_correction(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                foundationpose_trajectories=foundationpose_trajectories,
            )
        overlap_support = overlap_optimization.get(
            "support_plane_position_correction"
        )
        if isinstance(overlap_support, dict) and support_snap_route is not None:
            overlap_support["resolved_route"] = deepcopy(support_snap_route)
        payload["pose_overlap_optimization"] = overlap_optimization
        if overlap_optimization.get("applied") is True:
            payload["gravity_direction_camera"] = overlap_optimization["best_normal_camera"]
            payload["gravity_direction_coordinate_frame"] = "opencv_camera"
            payload["gravity_direction_convention"] = "up_direction_camera"
            payload["gravity_estimation_method"] = "sam3_projection_overlap_plane_normal_optimization"
            payload["rotation_correction"] = overlap_optimization["rotation_correction"]
            payload["support_plane_position_correction"] = overlap_optimization["support_plane_position_correction"]
        else:
            payload["rotation_correction"] = self._rotation_correction(
                payload=payload,
                question_dir=question_dir,
                trajectory_correction={"corrected_trajectories": foundationpose_trajectories},
                object_plan=object_plan,
            )
            payload["support_plane_position_correction"] = self._support_plane_position_correction(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                source_trajectories=payload["rotation_correction"],
                source_name="rotation_correction.corrected_pose_4x4",
            )
            if support_snap_route is not None:
                payload["support_plane_position_correction"]["resolved_route"] = (
                    deepcopy(support_snap_route)
                )
        final_source, final_source_name = self._first_applied_trajectory_source(
            (
                (
                    payload["support_plane_position_correction"],
                    "support_plane_position_correction.corrected_pose_4x4",
                ),
                (payload["rotation_correction"], "rotation_correction.corrected_pose_4x4"),
                (foundationpose_trajectories, "foundationpose.pose_4x4"),
            )
        )
        execution_order = [payload.get("gravity_estimation_method") or "gravity_direction_estimation"]
        if overlap_optimization.get("applied") is True:
            execution_order.append("sam3_projection_overlap_plane_normal_optimization")
        execution_order.extend(
            [
                "rotation_correction",
                "support_plane_position_correction",
                "static_fixture_flush_refinement",
                "active_interval_simulation_state",
            ]
        )
        payload["trajectory_correction"] = {
            "execution_order": execution_order,
            "pose_overlap_optimization": overlap_optimization,
            "support_plane_position_correction": payload["support_plane_position_correction"],
            "corrected_trajectories": self._corrected_trajectories(
                source_trajectories=final_source,
                source_name=final_source_name,
            ),
        }
        if fc_context is not None:
            payload["trajectory_correction"]["execution_order"] = [
                *payload["trajectory_correction"]["execution_order"],
                "physion_pp_friction_collision_seg1_patient_flush",
                "physion_pp_friction_collision_drop_refinement",
            ]
        if mc_context is not None:
            payload["trajectory_correction"]["execution_order"] = [
                *payload["trajectory_correction"]["execution_order"],
                "physion_pp_mass_collision_extra_flush",
                "physion_pp_mass_collision_agent_flush_refinement",
                "physion_pp_mass_collision_ball_trajectory_refinement",
                "physion_pp_mass_collision_drop_refinement",
            ]

    def _overlap_optimized_pose_correction(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        foundationpose_trajectories: Dict[str, Any],
    ) -> Dict[str, Any]:
        overlap_profile = _overlap_gravity_route_profile(object_plan)
        estimator_route = self._gravity_route_value(
            object_plan,
            "gravity_estimator_route",
        )
        constraints_route = self._gravity_route_value(
            object_plan,
            "gravity_constraints_route",
        )
        if (
            estimator_route is not None
            and estimator_route != overlap_profile["route"]
        ):
            raise ValueError(
                "overlap gravity optimization is not authorized by route: "
                f"{estimator_route!r}"
            )
        if (
            constraints_route is not None
            and constraints_route != OVERLAP_GRAVITY_CONSTRAINTS_ROUTE
        ):
            raise ValueError(
                "overlap gravity constraints are not authorized by route: "
                f"{constraints_route!r}"
            )
        horizontal = self._horizontal_plane_motion(object_plan)
        gate = self._ground_motion_gate(payload=payload, object_plan=object_plan)
        if gate["applies"] is not True:
            return {
                "applied": False,
                "reason": gate["reason"],
                "ground_motion_gate_mode": gate["mode"],
                "horizontal_plane_motion": horizontal,
            }
        if foundationpose_trajectories.get("applied") is not True:
            return {
                "applied": False,
                "reason": "missing foundationpose trajectory source",
                "horizontal_plane_motion": horizontal,
            }
        try:
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            mesh_geometries = {
                object_id: self._load_mesh_geometry(Path(mesh_path))
                for object_id, mesh_path in mesh_paths.items()
            }
            proxy_mesh_geometries = self._overlap_proxy_mesh_geometries(
                mesh_geometries=mesh_geometries,
                object_plan=object_plan,
                use_obb=bool(overlap_profile["use_obb_proxy"]),
            )
        except Exception as exc:
            return {
                "applied": False,
                "reason": f"failed to load overlap optimization inputs: {exc}",
                "horizontal_plane_motion": horizontal,
            }

        roll_stabilized = self._roll_stabilization(object_plan).get("applies") is True
        candidates = self._overlap_candidate_up_axes(roll_stabilized=roll_stabilized)
        if not candidates:
            return {
                "applied": False,
                "reason": "no valid candidate plane normals",
                "horizontal_plane_motion": horizontal,
            }

        # Physion++ scores base-aligned gravity candidates before ground snapping;
        # the silhouette-preserving similarity snap is applied only to the winner.
        # Other profiles may score post-snap candidates.
        pp_profile = bool(overlap_profile["pp_base_alignment"])
        pp_rotation_only_scoring = bool(overlap_profile["pre_snap_scoring"])

        # friction_collision / mass_collision: gravity evidence is seg1-only (fc: the
        # seg1 patient alone; mc: the kept extras plus the agent's pre-impact window).
        # seg2 objects never vote (their depth lives in the affine-remapped seg2 world
        # and the per-clip gravity is inherited from seg1 by construction), and raw-FP
        # rotations are candidate-independent, i.e. pure contrast dilution. Restricting
        # the scored object set also restricts the sampled frames to seg1; the optional
        # per-object frame caps additionally cut the mc agent to pre-contact frames.
        pp_scoring_block = next(
            (
                payload.get(block_key)
                for block_key in ("physion_pp_friction_collision", "physion_pp_mass_collision")
                if isinstance(payload.get(block_key), dict)
            ),
            None,
        )
        pp_scoring_ids = (
            {str(value) for value in pp_scoring_block.get("gravity_scoring_object_ids") or []}
            if pp_scoring_block
            else set()
        )
        pp_scoring_frame_caps = (
            {
                str(key): int(value)
                for key, value in (pp_scoring_block.get("gravity_scoring_frame_caps") or {}).items()
                if value is not None
            }
            if pp_scoring_block
            else {}
        )

        def evaluate_candidate(candidate: tuple[int, np.ndarray]) -> Dict[str, Any]:
            index, candidate_axis = candidate
            candidate_payload = dict(payload)
            candidate_payload["gravity_direction_camera"] = candidate_axis.tolist()
            candidate_payload["gravity_estimation_method"] = "sam3_projection_overlap_candidate"
            rotation = self._rotation_correction(
                payload=candidate_payload,
                question_dir=question_dir,
                trajectory_correction={"corrected_trajectories": foundationpose_trajectories},
                object_plan=object_plan,
                # Scoring-only: no abstention allowed (see PHYSION_PP_SCORING_BASE_ALIGN_CAP_DEG).
                # Passed for EVERY pp scenario (including mass_collision, whose scoring
                # world is post-snap): it doubles as the in-band scoring marker that
                # turns on the mc agent's scoring-only straightening.
                pp_base_align_cap_deg=(
                    PHYSION_PP_SCORING_BASE_ALIGN_CAP_DEG if pp_profile else None
                ),
            )
            if pp_rotation_only_scoring:
                support = None
                scoring_source = rotation
            else:
                support = self._support_plane_position_correction(
                    payload=candidate_payload,
                    question_dir=question_dir,
                    object_plan=object_plan,
                    source_trajectories=rotation,
                    source_name="rotation_correction.corrected_pose_4x4",
                )
                scoring_source = support
            airborne_ids = gate.get("airborne_object_ids") or set()
            if airborne_ids:
                scored_objects = [
                    item
                    for item in scoring_source.get("objects", [])
                    if isinstance(item, dict) and str(item.get("object_id")) not in airborne_ids
                ]
                if scored_objects:
                    scoring_source = {**scoring_source, "objects": scored_objects}
            if pp_scoring_ids:
                scored_objects = []
                for item in scoring_source.get("objects", []):
                    if not isinstance(item, dict) or str(item.get("object_id")) not in pp_scoring_ids:
                        continue
                    frame_cap = pp_scoring_frame_caps.get(str(item.get("object_id")))
                    if frame_cap is not None:
                        capped_poses = [
                            pose
                            for pose in item.get("poses", [])
                            if isinstance(pose, dict)
                            and self._pose_frame_index(pose) is not None
                            and self._pose_frame_index(pose) <= frame_cap
                        ]
                        if not capped_poses:
                            continue
                        item = {**item, "poses": capped_poses}
                    scored_objects.append(item)
                if scored_objects:
                    scoring_source = {**scoring_source, "objects": scored_objects}
            score = self._pose_overlap_score(
                source_trajectories=scoring_source,
                mesh_geometries=proxy_mesh_geometries,
                mask_records=mask_records,
                mask_arrays=mask_arrays,
                intrinsics=intrinsics,
                image_shape=image_shape,
                static_render_reuse=bool(
                    overlap_profile["static_render_reuse"]
                ),
            )
            evaluation = {
                "candidate_index": index,
                "normal_camera": candidate_axis.tolist(),
                "mean_iou": score.get("mean_iou"),
                "sample_count": score.get("sample_count"),
                "frame_indices": score.get("frame_indices"),
            }
            mean_iou = score.get("mean_iou")
            return {
                "evaluation": evaluation,
                "best": None
                if mean_iou is None
                else {
                    "candidate_index": index,
                    "best_normal_camera": candidate_axis.tolist(),
                    "mean_iou": float(mean_iou),
                    "score": score,
                    "rotation_correction": rotation,
                    "support_plane_position_correction": support,
                },
            }

        def evaluate_candidates(candidate_entries: list[tuple[int, np.ndarray]]) -> list[Dict[str, Any]]:
            return [evaluate_candidate(candidate) for candidate in candidate_entries]

        candidate_entries = list(enumerate(candidates))
        search_strategy: Dict[str, Any] = {
            "name": "full_grid",
            "candidate_grid_count": len(candidate_entries),
        }
        if roll_stabilized:
            coarse_entries = [
                (index, candidate_axis)
                for index, candidate_axis in candidate_entries
                if index % 2 == 0
            ]
            coarse_results = evaluate_candidates(coarse_entries)
            coarse_ranked = [
                result["best"]
                for result in coarse_results
                if result.get("best") is not None
            ]
            coarse_ranked.sort(key=lambda item: float(item["mean_iou"]), reverse=True)
            if coarse_ranked:
                coarse_best_index = int(coarse_ranked[0]["candidate_index"])
                coarse_evaluated_indices = {
                    int(result["evaluation"]["candidate_index"])
                    for result in coarse_results
                    if isinstance(result.get("evaluation"), dict)
                }
                refinement_entries = [
                    (index, candidate_axis)
                    for index, candidate_axis in candidate_entries
                    if abs(index - coarse_best_index) <= 1 and index not in coarse_evaluated_indices
                ]
                refinement_results = evaluate_candidates(refinement_entries)
                candidate_results = coarse_results + refinement_results
                search_strategy = {
                    "name": "coarse_to_fine_roll_stabilized",
                    "coarse_step_degrees": 10,
                    "refinement_step_degrees": 5,
                    "candidate_grid_count": len(candidate_entries),
                    "coarse_candidate_count": len(coarse_entries),
                    "refinement_candidate_count": len(refinement_entries),
                    "coarse_best_candidate_index": coarse_best_index,
                    "coarse_best_angle_degrees": float(coarse_best_index * 5),
                }
                if overlap_profile["pp_fine_refinement"]:
                    # Physion++: statics are base-aligned to each candidate up inside the
                    # scoring loop, so the objective is sharp enough to be worth resolving
                    # below the 5-degree grid (2 degrees of residual tilt is ~35cm of lift
                    # at the far end of a 10m ramp).
                    ranked_now = sorted(
                        (result["best"] for result in candidate_results if result.get("best") is not None),
                        key=lambda entry: -float(entry["mean_iou"]),
                    )
                    if ranked_now:
                        best_axis = np.asarray(ranked_now[0]["best_normal_camera"], dtype=np.float64)
                        best_angle = float(np.degrees(np.arctan2(abs(best_axis[2]), abs(best_axis[1]))))
                        evaluated_angles = set()
                        for result in candidate_results:
                            normal = np.asarray(result["evaluation"]["normal_camera"], dtype=np.float64)
                            evaluated_angles.add(round(float(np.degrees(np.arctan2(abs(normal[2]), abs(normal[1])))), 3))
                        fine_entries = []
                        for offset in (-2.5, -1.25, 1.25, 2.5):
                            angle = best_angle + offset
                            if not 0.0 <= angle <= 90.0 or round(angle, 3) in evaluated_angles:
                                continue
                            theta = float(np.deg2rad(angle))
                            fine_entries.append(
                                (
                                    100 + len(fine_entries),
                                    np.array([0.0, -np.cos(theta), -np.sin(theta)], dtype=np.float64),
                                )
                            )
                        if fine_entries:
                            candidate_results = candidate_results + evaluate_candidates(fine_entries)
                            search_strategy["pp_fine_refinement"] = {
                                "center_angle_degrees": best_angle,
                                "offsets_degrees": [-2.5, -1.25, 1.25, 2.5],
                                "evaluated_count": len(fine_entries),
                            }
            else:
                candidate_results = coarse_results
                search_strategy = {
                    "name": "coarse_to_fine_roll_stabilized",
                    "coarse_step_degrees": 10,
                    "refinement_step_degrees": 5,
                    "candidate_grid_count": len(candidate_entries),
                    "coarse_candidate_count": len(coarse_entries),
                    "refinement_candidate_count": 0,
                    "reason": "no coarse candidate produced valid overlap samples",
                }
        else:
            candidate_results = evaluate_candidates(candidate_entries)

        evaluations = []
        ranked_results: list[Dict[str, Any]] = []
        for result in candidate_results:
            evaluations.append(result["evaluation"])
            candidate_best = result["best"]
            if candidate_best is None:
                continue
            ranked_results.append(candidate_best)

        if not ranked_results:
            return {
                "applied": False,
                "reason": "no candidate produced valid overlap samples",
                "horizontal_plane_motion": horizontal,
                "candidate_count": len(candidates),
                "evaluations": evaluations,
            }
        ranked_results.sort(key=lambda item: float(item["mean_iou"]), reverse=True)
        best = ranked_results[0]
        if overlap_profile["winner_post_snap"]:
            # The exam is over. The scored world used UNCAPPED alignment (no
            # abstention) and, for mass_collision, scoring-only agent straightening;
            # the DELIVERED world recomputes the rotation with the protective cap and
            # raw-FP agents, then takes the snap once, at the winning gravity only.
            winner_payload = dict(payload)
            winner_payload["gravity_direction_camera"] = list(best["best_normal_camera"])
            winner_payload["gravity_estimation_method"] = "sam3_projection_overlap_candidate"
            best["rotation_correction"] = self._rotation_correction(
                payload=winner_payload,
                question_dir=question_dir,
                trajectory_correction={"corrected_trajectories": foundationpose_trajectories},
                object_plan=object_plan,
            )
            best["support_plane_position_correction"] = self._support_plane_position_correction(
                payload=winner_payload,
                question_dir=question_dir,
                object_plan=object_plan,
                source_trajectories=best["rotation_correction"],
                source_name="rotation_correction.corrected_pose_4x4",
            )
        return {
            "applied": True,
            "method": "grid_search_plane_normal_by_rendered_visible_mask_sam3_iou",
            "objective": "maximize mean IoU between z-buffer visible rendered proxy mesh masks and SAM3 masks",
            "scoring_world": (
                "rotation_only_no_snap" if pp_rotation_only_scoring else "post_ground_snap"
            ),
            "scoring_base_align_cap_deg": (
                PHYSION_PP_SCORING_BASE_ALIGN_CAP_DEG
                if pp_profile
                else PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG
            ),
            "horizontal_plane_motion": horizontal,
            "roll_stabilization": self._roll_stabilization(object_plan),
            "search_strategy": search_strategy,
            "candidate_count": len(evaluations),
            "candidate_grid_count": len(candidates),
            "best_candidate_index": best["candidate_index"],
            "best_normal_camera": best["best_normal_camera"],
            "best_mean_iou": best["mean_iou"],
            "score": best["score"],
            "scoring_mesh": "geometry_proxy",
            "pp_gravity_scoring_object_ids": sorted(pp_scoring_ids) if pp_scoring_ids else None,
            "pp_gravity_scoring_frame_caps": pp_scoring_frame_caps or None,
            "evaluations": evaluations,
            "rotation_correction": best["rotation_correction"],
            "support_plane_position_correction": best["support_plane_position_correction"],
        }

    def _overlap_candidate_up_axes(self, *, roll_stabilized: bool) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []

        def add(axis: np.ndarray) -> None:
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm <= 1e-12:
                return
            axis = axis / norm
            if float(np.dot(axis, np.array([0.0, -1.0, 0.0], dtype=np.float64))) < 0.0:
                axis = -axis
            key = tuple(np.round(axis, 6).tolist())
            if any(tuple(np.round(existing, 6).tolist()) == key for existing in candidates):
                return
            candidates.append(axis)

        if roll_stabilized:
            for theta in np.deg2rad(np.linspace(0.0, 90.0, 19)):
                add(np.array([0.0, -np.cos(theta), -np.sin(theta)], dtype=np.float64))
        else:
            for x_component in np.linspace(-0.65, 0.65, 5):
                for z_component in np.linspace(-0.65, 0.65, 5):
                    add(np.array([x_component, -1.0, z_component], dtype=np.float64))
        return candidates

    def _pose_overlap_score(
        self,
        *,
        source_trajectories: Dict[str, Any],
        mesh_geometries: Dict[str, tuple[np.ndarray, np.ndarray]],
        mask_records: Dict[str, Dict[int, Dict[str, Any]]],
        mask_arrays: Dict[str, np.ndarray],
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
        static_render_reuse: bool = False,
    ) -> Dict[str, Any]:
        if source_trajectories.get("applied") is not True:
            return {"mean_iou": None, "sample_count": 0, "reason": "source trajectories not applied"}
        pose_by_object_frame: dict[str, dict[int, np.ndarray]] = {}
        available_frames: set[int] = set()
        for item in source_trajectories.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "")
            if object_id not in mesh_geometries:
                continue
            by_frame = pose_by_object_frame.setdefault(object_id, {})
            for pose in item.get("poses", []):
                if not isinstance(pose, dict) or pose.get("status") not in {None, "ok"}:
                    continue
                frame_index = self._pose_frame_index(pose)
                matrix = pose.get("corrected_pose_4x4")
                if frame_index is None or matrix is None:
                    continue
                if frame_index not in mask_records.get(object_id, {}):
                    continue
                by_frame[frame_index] = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
                available_frames.add(frame_index)
        max_eval_frames = 8
        frame_indices = self._sample_overlap_frames(sorted(available_frames), max_frames=max_eval_frames)
        ious = []
        per_frame = []
        # Physion++ static fixtures keep one pose for the whole clip and the
        # camera intrinsics are constant, so the expensive software rasterization is
        # identical across eval frames; only the per-frame depth occlusion differs.
        # Cache (rendered_mask, rendered_depth) keyed by the exact pose/intrinsic bytes
        # and recompute just the occlusion — bitwise-identical to rendering per frame.
        render_cache: Dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for frame_index in frame_indices:
            frame_objects = {
                object_id: by_frame[frame_index]
                for object_id, by_frame in pose_by_object_frame.items()
                if frame_index in by_frame
            }
            if not frame_objects:
                continue
            K = self._intrinsic_for_frame(intrinsics, frame_index)
            frame_union = self._object_mask_union(mask_records, mask_arrays, frame_index, image_shape)
            rendered_masks = {}
            target_masks = {}
            for object_id, matrix in frame_objects.items():
                mask_record = mask_records.get(object_id, {}).get(frame_index)
                if not mask_record:
                    continue
                target_mask = mask_arrays.get(str(mask_record.get("mask_key") or ""))
                target_mask = self._resize_mask_to_shape(target_mask, image_shape)
                if target_mask is None:
                    continue
                vertices, faces = mesh_geometries[object_id]
                if static_render_reuse:
                    cache_key = (
                        object_id,
                        np.ascontiguousarray(matrix).tobytes(),
                        np.ascontiguousarray(K).tobytes(),
                    )
                    cached_render = render_cache.get(cache_key)
                    if cached_render is None:
                        vertices_camera = vertices @ matrix[:3, :3].T + matrix[:3, 3].reshape(1, 3)
                        cached_render = render_mesh_depth(
                            vertices_camera=vertices_camera,
                            faces=faces,
                            intrinsic=K,
                            image_shape=image_shape,
                        )
                        render_cache[cache_key] = cached_render
                    rendered_mask, rendered_depth = cached_render
                    rendered_masks[object_id] = mask_occluded_mesh_mask(
                        rendered_mask=rendered_mask,
                        occluder_mask=frame_union,
                        self_mask=target_mask,
                    )
                else:
                    vertices_camera = vertices @ matrix[:3, :3].T + matrix[:3, 3].reshape(1, 3)
                    rendered = render_mask_occluded_mesh(
                        vertices_camera=vertices_camera,
                        faces=faces,
                        intrinsic=K,
                        image_shape=image_shape,
                        occluder_mask=frame_union,
                        self_mask=target_mask,
                    )
                    rendered_masks[object_id] = rendered["visible_mask"]
                target_masks[object_id] = target_mask
            frame_ious = []
            for object_id, rendered_mask in rendered_masks.items():
                target_mask = target_masks.get(object_id)
                if target_mask is None:
                    continue
                iou = self._mask_iou(rendered_mask, target_mask)
                if iou is None:
                    continue
                ious.append(float(iou))
                frame_ious.append({"object_id": object_id, "iou": float(iou)})
            if frame_ious:
                per_frame.append({"frame_index": frame_index, "objects": frame_ious})
        return {
            "mean_iou": float(np.mean(ious)) if ious else None,
            "sample_count": len(ious),
            "frame_indices": frame_indices,
            "per_frame": per_frame,
            "max_eval_frames": max_eval_frames,
        }

    def _sample_overlap_frames(self, frame_indices: list[int], *, max_frames: int) -> list[int]:
        if len(frame_indices) <= max_frames:
            return frame_indices
        indices = np.linspace(0, len(frame_indices) - 1, max_frames)
        return [frame_indices[int(round(index))] for index in indices]

    def _foundationpose_trajectory_source(self, question_dir: Path) -> Dict[str, Any]:
        artifact_path = self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
        foundationpose = self.artifacts.read_optional(artifact_path)
        if not foundationpose:
            return {
                "applied": False,
                "reason": "missing foundationpose_poses.json",
                "source": "foundationpose.pose_4x4",
                "objects": [],
            }

        objects = []
        for item in foundationpose.get("objects", []):
            if not isinstance(item, dict):
                continue
            poses = []
            missing_pose_count = 0
            for pose in item.get("poses", []):
                if not isinstance(pose, dict):
                    continue
                pose_4x4 = pose.get("pose_4x4")
                if pose_4x4 is None:
                    missing_pose_count += 1
                    continue
                matrix = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
                if not np.isfinite(matrix).all():
                    missing_pose_count += 1
                    continue
                translation = matrix[:3, 3].tolist()
                poses.append(
                    {
                        "frame_index": pose.get("frame_index"),
                        "source_translation_camera": translation,
                        "corrected_translation_camera": translation,
                        "pose_4x4": matrix.tolist(),
                        "corrected_pose_4x4": matrix.tolist(),
                    }
                )
            objects.append(
                {
                    "object_id": item.get("object_id"),
                    "status": "ok" if poses and missing_pose_count == 0 else "partial",
                    "source": "foundationpose.pose_4x4",
                    "pose_count": len(poses),
                    "missing_pose_count": missing_pose_count,
                    "activation": self._activation_from_poses(poses),
                    "poses": poses,
                }
            )

        return {
            "applied": bool(objects),
            "source": "foundationpose.pose_4x4",
            "pose_field": "corrected_pose_4x4",
            "translation_field": "corrected_translation_camera",
            "objects": objects,
        }

    def _first_applied_trajectory_source(
        self,
        candidates: Sequence[tuple[Dict[str, Any], str]],
    ) -> tuple[Dict[str, Any], str]:
        for payload, source_name in candidates:
            if isinstance(payload, dict) and payload.get("applied") is True:
                return payload, source_name
        return candidates[-1]

    @staticmethod
    def _support_item_ground_height(support_item: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(support_item, dict):
            return None
        for pose in support_item.get("poses", []):
            if isinstance(pose, dict) and pose.get("support_plane_height_along_up_axis") is not None:
                return float(pose["support_plane_height_along_up_axis"])
        return None

    @staticmethod
    def _support_bottom_offset(
        vertices: np.ndarray,
        support_axis_local: np.ndarray,
        *,
        base_percentile: Optional[float] = None,
    ) -> float:
        heights = vertices @ support_axis_local
        if base_percentile is not None:
            return float(np.percentile(heights, base_percentile))
        return float(np.min(heights))




    def _support_plane_position_correction(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        source_trajectories: Dict[str, Any],
        source_name: str,
    ) -> Dict[str, Any]:
        support_route_record = self._pose_route_record(
            object_plan,
            key="support_snap_route",
            decision_id=SUPPORT_SNAP_DECISION_ID,
            supported_routes=SUPPORT_SNAP_ROUTES,
        )
        support_route = (
            str(support_route_record["route"])
            if support_route_record is not None
            else None
        )
        support_module = None
        if support_route_record is not None:
            context = support_route_record.get("context")
            if not isinstance(context, dict):
                raise ValueError("support-snap route is missing its context")
            support_profile = default_module_profile_policy().resolve_route(
                SUPPORT_SNAP_DECISION_ID,
                str(support_route_record["route"]),
                benchmark=str(context.get("benchmark") or "").strip().lower(),
                scenario=str(context.get("scenario") or "").strip().lower(),
            )
            support_module = support_profile.module("support_snap")
            if (
                support_module.implementation
                != "mesh_centroid_ray_slide_to_support_plane"
            ):
                raise ValueError(
                    "unsupported support-snap module implementation: "
                    f"{support_module.implementation!r}"
                )
        horizontal = self._horizontal_plane_motion(object_plan)
        gate = self._ground_motion_gate(payload=payload, object_plan=object_plan)
        if gate["applies"] is not True:
            return {
                "applied": False,
                "reason": gate["reason"],
                "ground_motion_gate_mode": gate["mode"],
                "horizontal_plane_motion": horizontal,
                "source": source_name,
                "objects": [],
            }
        if source_trajectories.get("applied") is not True:
            return {
                "applied": False,
                "reason": f"missing {source_name}",
                "horizontal_plane_motion": horizontal,
                "source": source_name,
                "objects": [],
            }
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            return {
                "applied": False,
                "reason": "missing or invalid gravity_direction_camera",
                "horizontal_plane_motion": horizontal,
                "source": source_name,
                "objects": [],
            }
        # A low vertex percentile gives irregular statics a robust resting base while
        # matching the minimum on clean primitive meshes with dense flat bottoms.
        base_percentile = (
            PHYSION_PP_STATIC_BASE_PERCENTILE
            if (
                support_module.require_boolean("robust_static_base_percentile")
                if support_module is not None
                else self._object_plan_scenario(object_plan).lower().endswith("_pp")
            )
            else None
        )

        mesh_paths = self._foundationpose_mesh_paths(question_dir)
        geometry_policy = payload.get("physion_pp_agent_geometry_policy") or {}
        detected_sphere_agent_ids = {
            str(object_id)
            for object_id in geometry_policy.get("agent_object_ids", [])
            if geometry_policy.get("scenario_gated") is True
        }
        sphere_agent_ids = (
            detected_sphere_agent_ids
            if (
                support_module.require_boolean("exclude_sphere_agents")
                if support_module is not None
                else True
            )
            else set()
        )
        object_items = [
            item
            for item in source_trajectories.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("object_id") or "") not in sphere_agent_ids
        ]
        missing_meshes = [
            str(item.get("object_id") or "")
            for item in object_items
            if str(item.get("object_id") or "") not in mesh_paths
        ]
        if missing_meshes:
            return {
                "applied": False,
                "reason": "missing required mesh_path in foundationpose_poses.json",
                "source": f"{source_name} + foundationpose_poses.mesh_path",
                "missing_object_ids": missing_meshes,
                "objects": [],
            }

        try:
            mesh_vertices = {
                object_id: self._cached_mesh_vertices(Path(mesh_path))
                for object_id, mesh_path in mesh_paths.items()
                if any(str(item.get("object_id") or "") == object_id for item in object_items)
            }
            mesh_centroids = {
                object_id: np.mean(vertices, axis=0)
                for object_id, vertices in mesh_vertices.items()
            }
            intrinsics, _ = self._video_metric_intrinsics_and_shape(question_dir)
        except Exception as exc:
            return {
                "applied": False,
                "reason": f"failed to load required support-plane inputs: {exc}",
                "source": (
                    f"{source_name} + foundationpose_poses.mesh_path + "
                    "mesh_centroid + video_metric_depth.intrinsics"
                ),
                "objects": [],
            }

        samples = []
        for item in object_items:
            object_id = str(item.get("object_id") or "")
            vertices = mesh_vertices[object_id]
            for pose in item.get("poses", []):
                if not isinstance(pose, dict) or pose.get("corrected_pose_4x4") is None:
                    continue
                matrix = np.asarray(pose["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                rotation = matrix[:3, :3]
                translation = matrix[:3, 3]
                support_axis_local = rotation.T @ up_axis
                local_bottom_offset = self._support_bottom_offset(
                    vertices, support_axis_local, base_percentile=base_percentile
                )
                bottom_height = float(np.dot(translation, up_axis) + local_bottom_offset)
                samples.append(
                    {
                        "object_id": object_id,
                        "frame_index": pose.get("frame_index"),
                        "bottom_height_along_up_axis": bottom_height,
                    }
                )
        if not samples:
            return {
                "applied": False,
                "reason": "no valid pose samples for support plane fitting",
                "source": f"{source_name} + foundationpose_poses.mesh_path",
                "objects": [],
            }
        # Airborne classification is independent of the selected agent geometry.
        # Default FP/BW sphere agents were already removed from object_items above;
        # native-agent options still need the same no-ground-snap semantics so their
        # trajectories remain available to the scenario-specific refinement.
        detected_exempt_object_ids = {
            str(value) for value in (gate.get("airborne_object_ids") or set())
        }
        exempt_object_ids = (
            detected_exempt_object_ids
            if (
                support_module.require_boolean("honor_airborne_exemptions")
                if support_module is not None
                else True
            )
            else set()
        )
        fit_samples = [
            sample for sample in samples if str(sample["object_id"]) not in exempt_object_ids
        ] or samples
        ground_height = float(np.median([sample["bottom_height_along_up_axis"] for sample in fit_samples]))

        objects = []
        for item in object_items:
            object_id = str(item.get("object_id") or "")
            is_exempt = object_id in exempt_object_ids
            vertices = mesh_vertices[object_id]
            local_centroid = mesh_centroids[object_id]
            corrected_poses = []
            skipped_poses = []
            adjustments = []
            for pose in item.get("poses", []):
                if not isinstance(pose, dict):
                    continue
                frame_index = self._pose_frame_index(pose)
                source_pose = pose.get("corrected_pose_4x4")
                pose_payload = {
                    "frame_index": pose.get("frame_index"),
                    "source_pose_4x4": source_pose,
                    "corrected_pose_4x4": source_pose,
                }
                if frame_index is None or source_pose is None:
                    pose_payload["status"] = "skipped"
                    pose_payload["reason"] = "missing frame_index or corrected_pose_4x4"
                    skipped_poses.append(pose_payload)
                    corrected_poses.append(pose_payload)
                    continue

                matrix = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
                rotation = matrix[:3, :3]
                translation = matrix[:3, 3]
                if is_exempt:
                    support_axis_local = rotation.T @ up_axis
                    local_bottom_offset = self._support_bottom_offset(
                        vertices, support_axis_local, base_percentile=base_percentile
                    )
                    bottom_height = float(np.dot(translation, up_axis) + local_bottom_offset)
                    pose_payload.update(
                        {
                            "status": "ok",
                            "ground_snap_exempt": True,
                            "corrected_pose_4x4": matrix.tolist(),
                            "source_translation_camera": translation.tolist(),
                            "corrected_translation_camera": translation.tolist(),
                            "source_bottom_height_along_up_axis": bottom_height,
                            "corrected_bottom_height_along_up_axis": bottom_height,
                            "support_plane_height_along_up_axis": ground_height,
                            "translation_adjustment_camera": [0.0, 0.0, 0.0],
                            "translation_adjustment_norm": 0.0,
                        }
                    )
                    corrected_poses.append(pose_payload)
                    continue
                if object_id in similarity_snap_ids:
                    support_axis_local = rotation.T @ up_axis
                    local_bottom_offset = self._support_bottom_offset(
                        vertices, support_axis_local, base_percentile=base_percentile
                    )
                    bottom_height = float(np.dot(translation, up_axis) + local_bottom_offset)
                    scale = ground_height / bottom_height if abs(bottom_height) > 1e-9 else float("nan")
                    pose_payload["similarity_snap_fallback"] = {
                        "reason": "similarity scale out of gate; using rigid ray slide",
                        "scale": float(scale) if np.isfinite(scale) else None,
                        "scale_gate": list(PHYSION_PP_SIMILARITY_SNAP_SCALE_GATE),
                    }
                K = self._intrinsic_for_frame(intrinsics, frame_index)
                rotated_centroid = rotation @ local_centroid
                source_anchor_camera = rotated_centroid + translation
                anchor_depth = float(source_anchor_camera[2])
                if not np.isfinite(anchor_depth) or anchor_depth <= 1e-9:
                    pose_payload["status"] = "skipped"
                    pose_payload["reason"] = "mesh centroid is behind or too close to the camera"
                    skipped_poses.append(pose_payload)
                    corrected_poses.append(pose_payload)
                    continue

                anchor_h = K @ (source_anchor_camera / anchor_depth)
                anchor_projection = [float(anchor_h[0]), float(anchor_h[1])]
                ray = source_anchor_camera
                ray_norm = float(np.linalg.norm(ray))
                if not np.isfinite(ray_norm) or ray_norm <= 1e-12:
                    pose_payload["status"] = "skipped"
                    pose_payload["reason"] = "invalid mesh centroid camera ray"
                    skipped_poses.append(pose_payload)
                    corrected_poses.append(pose_payload)
                    continue
                support_axis_local = rotation.T @ up_axis
                local_bottom_offset = self._support_bottom_offset(
                    vertices, support_axis_local, base_percentile=base_percentile
                )
                denominator = float(np.dot(ray, up_axis))
                if not np.isfinite(denominator) or abs(denominator) <= 1e-9:
                    pose_payload["status"] = "skipped"
                    pose_payload["reason"] = "camera ray is parallel to support plane"
                    skipped_poses.append(pose_payload)
                    corrected_poses.append(pose_payload)
                    continue
                ray_scale = (ground_height - local_bottom_offset + float(np.dot(rotated_centroid, up_axis))) / denominator
                if not np.isfinite(ray_scale) or ray_scale <= 0:
                    pose_payload["status"] = "skipped"
                    pose_payload["reason"] = "invalid support-plane ray intersection"
                    pose_payload["ray_scale"] = float(ray_scale) if np.isfinite(ray_scale) else None
                    skipped_poses.append(pose_payload)
                    corrected_poses.append(pose_payload)
                    continue

                corrected_anchor_camera = ray * ray_scale
                corrected_translation = corrected_anchor_camera - rotated_centroid
                corrected_matrix = matrix.copy()
                corrected_matrix[:3, 3] = corrected_translation
                source_bottom_height = float(np.dot(translation, up_axis) + local_bottom_offset)
                corrected_bottom_height = float(np.dot(corrected_translation, up_axis) + local_bottom_offset)
                adjustment = corrected_translation - translation
                adjustments.append(float(np.linalg.norm(adjustment)))
                pose_payload.update(
                    {
                        "status": "ok",
                        "corrected_pose_4x4": corrected_matrix.tolist(),
                        "source_translation_camera": translation.tolist(),
                        "corrected_translation_camera": corrected_translation.tolist(),
                        "anchor_projection_xy": anchor_projection,
                        "anchor_source": "pose_corrected_mesh_centroid",
                        "local_mesh_centroid": local_centroid.tolist(),
                        "source_anchor_camera": source_anchor_camera.tolist(),
                        "corrected_anchor_camera": corrected_anchor_camera.tolist(),
                        "camera_ray": ray.tolist(),
                        "ray_scale": float(ray_scale),
                        "source_bottom_height_along_up_axis": source_bottom_height,
                        "corrected_bottom_height_along_up_axis": corrected_bottom_height,
                        "support_plane_height_along_up_axis": ground_height,
                        "translation_adjustment_camera": adjustment.tolist(),
                        "translation_adjustment_norm": float(np.linalg.norm(adjustment)),
                    }
                )
                corrected_poses.append(pose_payload)

            valid_count = sum(1 for pose in corrected_poses if pose.get("status") == "ok")
            similarity_scales = [
                float(pose["similarity_scale"])
                for pose in corrected_poses
                if isinstance(pose.get("similarity_scale"), float) and pose.get("similarity_folded")
            ]
            objects.append(
                {
                    "object_id": item.get("object_id"),
                    "status": "ok" if valid_count == len(corrected_poses) and corrected_poses else "partial",
                    "ground_snap_exempt": is_exempt,
                    "mesh_path": mesh_paths[object_id],
                    "similarity_scale": float(np.median(similarity_scales)) if similarity_scales else None,
                    "pose_count": len(corrected_poses),
                    "corrected_pose_count": valid_count,
                    "skipped_pose_count": len(skipped_poses),
                    "activation": self._activation_from_poses(corrected_poses),
                    "max_translation_adjustment_norm": float(max(adjustments) if adjustments else 0.0),
                    "mean_translation_adjustment_norm": float(np.mean(adjustments) if adjustments else 0.0),
                    "poses": corrected_poses,
                }
            )

        return {
            "applied": any(item.get("corrected_pose_count", 0) > 0 for item in objects),
            "source": (
                f"{source_name} + foundationpose_poses.mesh_path + "
                "pose_corrected_mesh_centroid + video_metric_depth.intrinsics"
            ),
            "ground_motion_gate_mode": gate["mode"],
            "ground_snap_exempt_object_ids": sorted(
                str(item.get("object_id")) for item in objects if item.get("ground_snap_exempt")
            ),
            "sphere_agent_excluded_object_ids": sorted(sphere_agent_ids),
            "ground_height_fit_sample_count": len(fit_samples),
            "horizontal_plane_motion": horizontal,
            "normal_camera": up_axis.tolist(),
            "normal_semantics": f"{payload.get('gravity_estimation_method') or 'gravity_direction_estimation'}_up_direction_camera",
            "ground_height_along_normal": ground_height,
            "ground_height_estimator": "median_min_height_along_up_axis_across_pose_corrected_meshes",
            "sample_count": len(samples),
            "position_rule": (
                "translation is placed so the pose-corrected mesh centroid remains on its source camera ray, "
                "with the transformed local bottom support height equal to the fitted support plane"
            ),
            "pose_field": "corrected_pose_4x4",
            "objects": objects,
        }

    def _corrected_trajectories(
        self,
        *,
        source_trajectories: Dict[str, Any],
        source_name: str,
    ) -> Dict[str, Any]:
        if source_trajectories.get("applied") is True:
            objects = []
            for item in source_trajectories.get("objects", []):
                if not isinstance(item, dict) or item.get("status") != "ok":
                    continue
                final_poses = []
                missing_pose_count = 0
                for pose in item.get("poses", []):
                    if not isinstance(pose, dict):
                        continue
                    corrected_pose = pose.get("corrected_pose_4x4")
                    if corrected_pose is None:
                        missing_pose_count += 1
                    final_poses.append(
                        {
                            "frame_index": pose.get("frame_index"),
                            "source_translation_camera": pose.get("source_translation_camera"),
                            "corrected_translation_camera": pose.get("corrected_translation_camera"),
                            "translation_adjustment_norm": pose.get("translation_adjustment_norm"),
                            "anchor_projection_xy": pose.get("anchor_projection_xy"),
                            "anchor_source": pose.get("anchor_source"),
                            "source_anchor_camera": pose.get("source_anchor_camera"),
                            "corrected_anchor_camera": pose.get("corrected_anchor_camera"),
                            "corrected_pose_4x4": corrected_pose,
                        }
                    )
                objects.append(
                    {
                        "object_id": item.get("object_id"),
                        "status": "ok" if final_poses and missing_pose_count == 0 else "partial",
                        "source": source_name,
                        "pose_count": len(final_poses),
                        "missing_pose_count": missing_pose_count,
                        "activation": self._activation_from_poses(final_poses),
                        "max_translation_adjustment_norm": item.get("max_translation_adjustment_norm"),
                        "mean_translation_adjustment_norm": item.get("mean_translation_adjustment_norm"),
                        "poses": final_poses,
                    }
                )
            return {
                "applied": bool(objects),
                "source": source_name,
                "pose_field": "corrected_pose_4x4",
                "translation_field": "corrected_translation_camera",
                "objects": objects,
            }

        return {
            "applied": False,
            "reason": "no corrected trajectory source available",
            "source": None,
            "pose_field": "corrected_pose_4x4",
            "translation_field": "corrected_translation_camera",
            "objects": [],
        }
























    def _rotation_correction(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        trajectory_correction: Dict[str, Any],
        object_plan: ObjectPlan,
        pp_base_align_cap_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        rotation_policy_route = (
            object_plan.special_scene.get("rotation_policy_route")
            if isinstance(object_plan.special_scene, dict)
            else None
        )
        if isinstance(rotation_policy_route, dict):
            route = rotation_policy_route.get("route")
            if route != ROTATION_POLICY_ROUTE:
                raise ValueError(f"unsupported rotation-policy route: {route!r}")
        corrected_trajectories = trajectory_correction.get("corrected_trajectories")
        if not isinstance(corrected_trajectories, dict) or corrected_trajectories.get("applied") is not True:
            return {
                "applied": False,
                "reason": "missing trajectory_correction.corrected_trajectories",
                "source": "trajectory_correction.corrected_trajectories",
                "objects": [],
            }
        source_name = str(corrected_trajectories.get("source") or "trajectory_correction.corrected_trajectories")

        geometry_by_object_id = {
            str(item.object_id): str(item.geometry_type or "").strip().lower()
            for item in object_plan.target_objects
        }
        mesh_axes_by_object_id = self._foundationpose_local_axes_by_object(question_dir)
        horizontal = self._horizontal_plane_motion(object_plan)
        horizontal_motion = horizontal.get("applies") is True
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        scenario_is_pp = self._object_plan_scenario(object_plan).lower().endswith("_pp")
        # friction_collision: every track is dynamic, but the yellow-cue patients (both
        # segments) never really rotate -- they sit still or fall upright -- so they take
        # the CLEVRER-style closest-axis straightening regardless of geometry type, while
        # the red-cue agents keep the raw FP rotation via the unchanged _pp guard below.
        fc_block = payload.get("physion_pp_friction_collision")
        fc_patient_ids = (
            {str(value) for value in fc_block.get("patient_object_ids") or []}
            if isinstance(fc_block, dict)
            else set()
        )
        # mass_collision: the seg2 patient and the seg1 kept extras take the same
        # always-straighten treatment; the seg1 agent joins them ONLY inside gravity
        # scoring calls (pp_base_align_cap_deg is the in-band scoring marker: only the
        # candidate evaluations pass it) -- pre-impact the route treats it as flat on the
        # floor, so per-candidate straightening turns it into a discriminative scorer,
        # while the DELIVERED agent rotation stays raw FP. The balls are exempted from
        # the raw-FP _pp guard so the sphere identity-rotation path applies to them.
        mc_block = payload.get("physion_pp_mass_collision")
        mc_straighten_ids: set[str] = set()
        mc_rotation_exception_ids: set[str] = set()
        if isinstance(mc_block, dict):
            mc_straighten_ids = {str(value) for value in mc_block.get("straighten_object_ids") or []}
            if pp_base_align_cap_deg is not None:
                mc_straighten_ids |= {
                    str(value) for value in mc_block.get("scoring_straighten_object_ids") or []
                }
            mc_rotation_exception_ids = mc_straighten_ids | {
                str(value) for value in mc_block.get("ball_object_ids") or []
            }
        objects = []
        for item in corrected_trajectories.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "")
            geometry_type = geometry_by_object_id.get(object_id, "")
            sphere_rotation = geometry_type == "sphere"
            is_pp_straighten = object_id in fc_patient_ids or object_id in mc_straighten_ids
            is_pp_rotation_exception = is_pp_straighten or object_id in mc_rotation_exception_ids
            if scenario_is_pp and object_id not in static_fixture_object_ids and not is_pp_rotation_exception:
                # Physion++: every non-static track is the free-moving red-cue agent
                # (slides/tips on platforms); neither sphere identity-rotation nor axis
                # alignment may touch its visually tracked rotation. friction_collision
                # patients and the mass_collision patient/extras/balls are the exception
                # (dynamic tracks that must stay straight / sphere identity).
                sphere_rotation = False
            is_pp_static = scenario_is_pp and object_id in static_fixture_object_ids
            physical_rotation_constraint = None
            canonical_local_axes = None
            if (
                # Physion++ statics rest on the ground by construction, so their axis
                # alignment must not depend on the planner's scene-level horizontal flag.
                # friction_collision patients (and the mass_collision patient/extras)
                # are straightened for EVERY geometry type (scenario prior: they rest or
                # fall upright, never roll).
                (horizontal_motion or is_pp_static or is_pp_straighten)
                and (geometry_type in {"box", "cube", "cylinder"} or (is_pp_straighten and not sphere_rotation))
                and not (scenario_is_pp and object_id not in static_fixture_object_ids and not is_pp_straighten)
            ):
                canonical_local_axes = mesh_axes_by_object_id.get(object_id)
                physical_rotation_constraint = self._physical_rotation_constraint(
                    geometry_type=geometry_type,
                    up_axis=up_axis,
                    canonical_local_axes=canonical_local_axes,
                )
            # Physion++ irregular statics have no primitive axes to align; fit the resting
            # base plane instead and rotate it onto the up axis (the irregular-geometry
            # analogue of closest-axis-to-up). Running inside every overlap candidate
            # evaluation turns the gravity search into a hypothesis test: aligning the
            # base to the TRUE up preserves the silhouette, aligning to a wrong up breaks
            # it, so the rendered-mask IoU discriminates candidates.
            pp_base_plane = None
            corrected_poses = []
            missing_pose_count = 0
            for pose in item.get("poses", []):
                if not isinstance(pose, dict):
                    continue
                source_pose = pose.get("corrected_pose_4x4")
                pose_payload = {
                    "frame_index": pose.get("frame_index"),
                    "source_pose_4x4": source_pose,
                    "corrected_pose_4x4": source_pose,
                    "rotation_policy": "preserved",
                }
                if source_pose is None:
                    missing_pose_count += 1
                    corrected_poses.append(pose_payload)
                    continue
                if sphere_rotation:
                    matrix = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
                    corrected_matrix = matrix.copy()
                    corrected_matrix[:3, :3] = np.eye(3, dtype=np.float64)
                    pose_payload["corrected_pose_4x4"] = corrected_matrix.tolist()
                    pose_payload["rotation_policy"] = "ignored_identity_rotation"
                    pose_payload["corrected_translation_camera"] = corrected_matrix[:3, 3].tolist()
                elif pp_base_plane is not None:
                    matrix = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
                    corrected_matrix, alignment = self._apply_base_plane_alignment(
                        matrix=matrix,
                        up_axis=up_axis,
                        base_plane=pp_base_plane,
                        cap_deg=(
                            pp_base_align_cap_deg
                            if pp_base_align_cap_deg is not None
                            else PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG
                        ),
                    )
                    pose_payload["rotation_alignment"] = alignment
                    if alignment.get("applied") is True:
                        pose_payload["corrected_pose_4x4"] = corrected_matrix.tolist()
                        pose_payload["rotation_policy"] = "pp_static_base_plane_aligned_to_up"
                        pose_payload["corrected_translation_camera"] = corrected_matrix[:3, 3].tolist()
                    else:
                        pose_payload["rotation_policy"] = "preserved_visual_rotation"
                elif physical_rotation_constraint is not None and canonical_local_axes:
                    matrix = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
                    corrected_matrix = matrix.copy()
                    if up_axis is not None:
                        aligned_rotation, alignment = self._align_rotation_closest_axis_to_up(
                            matrix[:3, :3],
                            up_axis=up_axis,
                            local_axes=canonical_local_axes,
                            canonicalize_twist=geometry_type == "cylinder",
                        )
                        pose_payload["rotation_alignment"] = alignment
                        corrected_matrix[:3, :3] = aligned_rotation
                        pose_payload["corrected_pose_4x4"] = corrected_matrix.tolist()
                        pose_payload["rotation_policy"] = "closest_local_axis_aligned_to_up_axis"
                        pose_payload["corrected_translation_camera"] = corrected_matrix[:3, 3].tolist()
                    else:
                        pose_payload["rotation_policy"] = "preserved_visual_rotation_missing_up_axis"
                    pose_payload["physical_rotation_constraint"] = physical_rotation_constraint
                elif physical_rotation_constraint is not None:
                    pose_payload["rotation_policy"] = "preserved_visual_rotation_missing_foundationpose_mesh_axes"
                    pose_payload["physical_rotation_constraint"] = physical_rotation_constraint
                else:
                    pose_payload["rotation_policy"] = "preserved_visual_rotation"
                corrected_poses.append(pose_payload)

            if sphere_rotation:
                rotation_policy = "ignored_identity_rotation"
            elif pp_base_plane is not None:
                rotation_policy = "pp_static_base_plane_aligned_to_up"
            elif (
                physical_rotation_constraint is not None
                and canonical_local_axes
            ):
                rotation_policy = "closest_local_axis_aligned_to_up_axis"
            elif physical_rotation_constraint is not None:
                rotation_policy = "preserved_visual_rotation_missing_canonical_local_axes"
            else:
                rotation_policy = "preserved"
            object_payload = {
                "object_id": item.get("object_id"),
                "status": "ok" if corrected_poses and missing_pose_count == 0 else "partial",
                "geometry_type": geometry_type,
                "source": source_name,
                "pose_count": len(corrected_poses),
                "missing_pose_count": missing_pose_count,
                "activation": self._activation_from_poses(corrected_poses),
                "rotation_policy": rotation_policy,
                "physical_rotation_constraint": physical_rotation_constraint,
                "poses": corrected_poses,
            }
            objects.append(object_payload)

        result = {
            "applied": bool(objects),
            "source": source_name,
            "policy": (
                "ignore sphere rotation; align the selected support axis of horizontal-plane objects to the "
                "estimated up axis; canonicalize cylinder twist around that axis; preserve other object rotations"
            ),
            "horizontal_plane_motion": horizontal,
            "up_axis_camera": up_axis.tolist() if up_axis is not None else None,
            "pose_field": "corrected_pose_4x4",
            "objects": objects,
            "resolved_route": deepcopy(rotation_policy_route),
        }
        return result


    @staticmethod
    def _apply_base_plane_alignment(
        *,
        matrix: np.ndarray,
        up_axis: np.ndarray,
        base_plane: Dict[str, Any],
        cap_deg: float = PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Rotate a pose about its base centroid so the fitted base normal matches up."""
        normal_local = np.asarray(base_plane["normal_local"], dtype=np.float64)
        centroid_local = np.asarray(base_plane["centroid_local"], dtype=np.float64)
        rotation = matrix[:3, :3]
        normal_camera = rotation @ normal_local
        cos_angle = float(np.clip(normal_camera @ up_axis, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        info: Dict[str, Any] = {
            "method": "pp_static_base_plane_to_up",
            "base_tilt_deg": angle_deg,
            "cap_deg": cap_deg,
            "applied": False,
        }
        if angle_deg <= 1e-3:
            info["reason"] = "base already aligned"
            return matrix, info
        if angle_deg > cap_deg:
            info["reason"] = "base tilt exceeds alignment cap"
            return matrix, info
        rotation_axis = np.cross(normal_camera, up_axis)
        axis_norm = float(np.linalg.norm(rotation_axis))
        if axis_norm <= 1e-9:
            info["reason"] = "degenerate rotation axis"
            return matrix, info
        align_rotation = _rotation_about_axis(rotation_axis / axis_norm, float(np.arccos(cos_angle)))
        base_centroid_camera = rotation @ centroid_local + matrix[:3, 3]
        corrected = matrix.copy()
        corrected[:3, :3] = align_rotation @ rotation
        corrected[:3, 3] = base_centroid_camera - corrected[:3, :3] @ centroid_local
        info["applied"] = True
        return corrected, info

    def _physical_rotation_constraint(
        self,
        *,
        geometry_type: str,
        up_axis: np.ndarray | None,
        canonical_local_axes: list[np.ndarray] | None,
    ) -> Dict[str, Any]:
        constraint: Dict[str, Any] = {
            "policy": "align_selected_support_axis_to_up_axis",
            "reason": "horizontal-plane motion requires a stable support axis aligned with the ground normal",
            "horizontal_plane_motion": True,
        }
        if geometry_type == "cylinder":
            constraint["ignored_degrees_of_freedom"] = ["rotation_about_cylinder_axis"]
        if geometry_type in {"box", "cube", "cylinder"}:
            constraint["desired_support_relation"] = "bottom_face_parallel_to_horizontal_plane"
        if up_axis is not None:
            constraint["up_axis_camera"] = up_axis.tolist()
        constraint["axis_selection"] = "per_frame_closest_signed_geometry_axis_to_up_axis"
        constraint["geometry_axis_count"] = len(canonical_local_axes or [])
        constraint["geometry_axis_source"] = {
            "source": "mesh_conditioning.foundationpose_local_axes",
            "status": "ok" if canonical_local_axes else "missing",
            "coordinate_frame": "foundationpose_canonical_metric_mesh_frame",
            "reason": "Pose correction uses geometry axes expressed in the same canonical metric mesh frame consumed by FoundationPose.",
        }
        return constraint

    def _foundationpose_local_axes_by_object(self, question_dir: Path) -> dict[str, list[np.ndarray]]:
        mesh_conditioning = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "mesh_conditioning.json")
        )
        if not isinstance(mesh_conditioning, dict):
            return {}
        axes_by_object: dict[str, list[np.ndarray]] = {}
        for item in mesh_conditioning.get("objects", []):
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            object_id = str(item.get("object_id") or "")
            axes_payload = item.get("foundationpose_local_axes")
            axes = axes_payload.get("axes") if isinstance(axes_payload, dict) else None
            if not object_id or not isinstance(axes, list):
                continue
            parsed_axes = []
            for axis in axes:
                array = np.asarray(axis, dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(array))
                if array.shape == (3,) and np.isfinite(norm) and norm > 1e-12:
                    parsed_axes.append(array / norm)
            if parsed_axes:
                axes_by_object[object_id] = parsed_axes
        return axes_by_object

    def _align_rotation_closest_axis_to_up(
        self,
        rotation: np.ndarray,
        *,
        up_axis: np.ndarray,
        local_axes: list[np.ndarray],
        canonicalize_twist: bool = False,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        target_axis = up_axis / max(float(np.linalg.norm(up_axis)), 1e-12)
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for axis_index, local_axis in enumerate(local_axes):
            local_axis = np.asarray(local_axis, dtype=np.float64).reshape(3)
            local_norm = float(np.linalg.norm(local_axis))
            if not np.isfinite(local_norm) or local_norm <= 1e-12:
                continue
            axis = rotation @ (local_axis / local_norm)
            axis_norm = float(np.linalg.norm(axis))
            if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
                continue
            axis = axis / axis_norm
            for sign in (-1, 1):
                signed_axis = float(sign) * axis
                score = float(np.dot(signed_axis, target_axis))
                candidates.append((score, axis_index, sign, signed_axis))
        if not candidates:
            return rotation, {
                "status": "skipped",
                "reason": "no_valid_local_axis",
            }
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, axis_index, sign, source_axis = candidates[0]
        cross = np.cross(source_axis, target_axis)
        cross_norm = float(np.linalg.norm(cross))
        dot = max(-1.0, min(1.0, score))
        angle = float(np.arctan2(cross_norm, dot))
        angle_degrees = float(np.degrees(angle))
        if cross_norm <= 1e-12:
            if dot >= 0.0:
                aligned = rotation
            else:
                aligned = self._axis_angle_rotation(self._orthogonal_unit_axis(source_axis), np.pi) @ rotation
        else:
            aligned = self._axis_angle_rotation(cross / cross_norm, angle) @ rotation
        if canonicalize_twist:
            aligned = self._canonical_rotation_for_axis(
                selected_local_axis=np.asarray(local_axes[int(axis_index)], dtype=np.float64).reshape(3) * float(sign),
                target_axis=target_axis,
            )
        return aligned, {
            "status": "ok",
            "selected_axis_index": int(axis_index),
            "selected_axis_sign": int(sign),
            "angle_degrees": angle_degrees,
            "selected_axis_score": float(score),
            "axis_selection_policy": "best_axis",
            "axis_source": "mesh_conditioning.foundationpose_local_axes",
            "source": "per_frame_closest_signed_geometry_axis_to_up_axis",
            "twist_policy": "canonicalized_around_selected_axis" if canonicalize_twist else "preserved",
        }

    def _canonical_rotation_for_axis(self, *, selected_local_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
        local_z = selected_local_axis / max(float(np.linalg.norm(selected_local_axis)), 1e-12)
        world_z = target_axis / max(float(np.linalg.norm(target_axis)), 1e-12)
        local_x = self._stable_perpendicular_axis(local_z, prefer_world_x=False)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / max(float(np.linalg.norm(local_y)), 1e-12)
        local_basis = np.stack([local_x, local_y, local_z], axis=1)

        world_x = self._stable_perpendicular_axis(world_z, prefer_world_x=True)
        world_y = np.cross(world_z, world_x)
        world_y = world_y / max(float(np.linalg.norm(world_y)), 1e-12)
        world_basis = np.stack([world_x, world_y, world_z], axis=1)
        return world_basis @ local_basis.T

    def _stable_perpendicular_axis(self, axis: np.ndarray, *, prefer_world_x: bool) -> np.ndarray:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        seeds = (
            [np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)]
            if prefer_world_x
            else [np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)]
        )
        for seed in seeds:
            candidate = seed - float(np.dot(seed, axis)) * axis
            norm = float(np.linalg.norm(candidate))
            if np.isfinite(norm) and norm > 1e-9:
                return candidate / norm
        return self._orthogonal_unit_axis(axis)

    def _axis_angle_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float64,
        )

    def _orthogonal_unit_axis(self, axis: np.ndarray) -> np.ndarray:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, seed))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        orthogonal = seed - float(np.dot(seed, axis)) * axis
        return orthogonal / max(float(np.linalg.norm(orthogonal)), 1e-12)

    def _zero_gravity_roll_component(self, payload: Dict[str, Any]) -> None:
        up = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up is None:
            return
        roll_free = np.array([0.0, float(up[1]), float(up[2])], dtype=np.float64)
        norm = float(np.linalg.norm(roll_free))
        if norm <= 1e-6:
            return
        roll_free /= norm
        removed_roll_deg = float(
            np.degrees(np.arccos(np.clip(float(np.dot(up, roll_free)), -1.0, 1.0)))
        )
        payload["gravity_roll_zeroing"] = {
            "applied": True,
            "constraint": "up_direction_camera[0] = 0",
            "input_gravity_direction_camera": up.tolist(),
            "removed_roll_angle_deg": removed_roll_deg,
        }
        payload["gravity_direction_camera"] = roll_free.tolist()
        payload["gravity_estimation_method"] = (
            f"{payload.get('gravity_estimation_method') or GEOCALIB_GRAVITY_METHOD}_roll_zeroed"
        )

    def _unit_gravity_direction(self, gravity_direction_camera: Any) -> np.ndarray | None:
        if not isinstance(gravity_direction_camera, list) or len(gravity_direction_camera) != 3:
            return None
        normal = np.asarray(gravity_direction_camera, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0:
            return None
        return normal / norm

    def _unit_up_direction(self, gravity_direction_camera: Any) -> np.ndarray | None:
        # GeoCalib names this vector "gravity", but its perspective-field code uses it
        # directly as the projected up-vector direction.
        return self._unit_gravity_direction(gravity_direction_camera)

    def _question_cache_signature(self, path: Path) -> tuple | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def _foundationpose_mesh_paths(self, question_dir: Path) -> Dict[str, str]:
        artifact_path = self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
        cache_key = None
        if self._question_cache_enabled:
            signature = self._question_cache_signature(artifact_path)
            if signature is not None:
                cache_key = ("foundationpose_mesh_paths", signature)
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    return dict(cached)
        foundationpose = self.artifacts.read_optional(artifact_path)
        if not foundationpose:
            return {}
        mesh_paths = {}
        for item in foundationpose.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "")
            mesh_path = item.get("mesh_path")
            if object_id and mesh_path and Path(str(mesh_path)).exists():
                mesh_paths[object_id] = str(mesh_path)
        if cache_key is not None:
            self._question_cache[cache_key] = dict(mesh_paths)
        return mesh_paths

    def _sam3_video_mask_records_by_object_frame(self, question_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
        labels_path = self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
        tracks_path = self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        cache_key = None
        if self._question_cache_enabled:
            labels_signature = self._question_cache_signature(labels_path)
            tracks_signature = self._question_cache_signature(tracks_path)
            if labels_signature is not None and tracks_signature is not None:
                cache_key = ("sam3_video_mask_records", labels_signature, tracks_signature)
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    return deepcopy(cached)
        labels_payload = self.artifacts.read_optional(labels_path)
        tracks_payload = self.artifacts.read_optional(tracks_path)
        if not labels_payload:
            raise ValueError("missing sam3_video_track_labels.json")
        if not tracks_payload:
            raise ValueError("missing sam3_video_tracks.json")
        selected_records, sidecar = _accepted_sam3_video_records_from_labels(
            labels_payload=labels_payload,
            tracks_payload=tracks_payload,
        )
        if not sidecar or not Path(str(sidecar)).exists():
            raise ValueError("missing sam3_video_track_labels/tracks mask_sidecar")
        records: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for record in selected_records:
            if not isinstance(record, dict):
                continue
            object_id = str(record.get("object_id") or "")
            mask_key = str(record.get("selected_mask_key") or record.get("mask_key") or "")
            if not object_id or not mask_key:
                continue
            try:
                frame_index = int(record.get("frame_index"))
            except (TypeError, ValueError):
                continue
            by_frame = records.setdefault(object_id, {})
            previous = by_frame.get(frame_index)
            if previous is not None and float(previous.get("area") or 0.0) >= float(record.get("area") or 0.0):
                continue
            by_frame[frame_index] = {
                "object_id": object_id,
                "frame_index": frame_index,
                "mask_key": mask_key,
                "sidecar": str(sidecar),
                "area": record.get("selected_area") or record.get("area"),
                "centroid_xy": record.get("selected_centroid_xy") or record.get("centroid_xy"),
            }
        if cache_key is not None:
            self._question_cache[cache_key] = deepcopy(records)
        return records

    def _sam3_video_mask_arrays(self, records_by_object_frame: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, np.ndarray]:
        keys_by_sidecar: Dict[str, set[str]] = {}
        for by_frame in records_by_object_frame.values():
            for record in by_frame.values():
                sidecar = str(record.get("sidecar") or "")
                mask_key = str(record.get("mask_key") or "")
                if sidecar and mask_key:
                    keys_by_sidecar.setdefault(sidecar, set()).add(mask_key)
        cache_key = None
        if self._question_cache_enabled and keys_by_sidecar:
            signature_entries = []
            for sidecar in sorted(keys_by_sidecar):
                signature = self._question_cache_signature(Path(sidecar))
                if signature is None:
                    signature_entries = None
                    break
                signature_entries.append((signature, tuple(sorted(keys_by_sidecar[sidecar]))))
            if signature_entries is not None:
                cache_key = ("sam3_video_mask_arrays", tuple(signature_entries))
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    return {key: value.copy() for key, value in cached.items()}
        arrays_by_key: Dict[str, np.ndarray] = {}
        for sidecar, keys in keys_by_sidecar.items():
            with np.load(sidecar) as arrays:
                for key in keys:
                    if key in arrays:
                        arrays_by_key[key] = np.asarray(arrays[key]).astype(np.uint8)
        if cache_key is not None:
            self._question_cache[cache_key] = {key: value.copy() for key, value in arrays_by_key.items()}
        return arrays_by_key

    def _video_metric_intrinsics_and_shape(self, question_dir: Path) -> tuple[np.ndarray, tuple[int, int]]:
        intrinsics, _metric_depth, shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
        return intrinsics, shape

    def _video_metric_intrinsics_depth_and_shape(self, question_dir: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        payload = self.artifacts.read_optional(self.artifacts.artifact_path_by_name(question_dir, "video_metric_depth.json"))
        if not payload:
            raise ValueError("missing video_metric_depth.json")
        sidecar = payload.get("tensor_sidecar")
        if not sidecar or not Path(str(sidecar)).exists():
            raise ValueError("missing video_metric_depth tensor_sidecar")
        cache_key = None
        if self._question_cache_enabled:
            signature = self._question_cache_signature(Path(str(sidecar)))
            if signature is not None:
                cache_key = ("video_metric_depth_sidecar", signature)
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    intrinsics, metric_depth, processed_shape = cached
                    shape = (
                        processed_shape
                        if processed_shape is not None
                        else self._video_metric_processed_shape_from_payload(payload)
                    )
                    return intrinsics.copy(), metric_depth.copy(), shape
        with np.load(str(sidecar)) as arrays:
            intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64)
            metric_depth = np.asarray(arrays["metric_depth"], dtype=np.float32)
            processed_images = np.asarray(arrays["processed_images"]) if "processed_images" in arrays else None
        if processed_images is not None:
            if processed_images.ndim < 3:
                raise ValueError(f"invalid processed_images shape: {processed_images.shape}")
            shape = (int(processed_images.shape[1]), int(processed_images.shape[2]))
        else:
            shape = self._video_metric_processed_shape_from_payload(payload)
        if cache_key is not None:
            self._question_cache[cache_key] = (
                intrinsics.copy(),
                metric_depth.copy(),
                shape if processed_images is not None else None,
            )
        return intrinsics, metric_depth, shape

    def _depth_for_frame(self, depths: np.ndarray, frame_index: int) -> np.ndarray:
        if depths.ndim == 4 and depths.shape[-1] == 1:
            index = max(0, min(int(frame_index), depths.shape[0] - 1))
            return np.asarray(depths[index, ..., 0], dtype=np.float32)
        if depths.ndim == 3:
            index = max(0, min(int(frame_index), depths.shape[0] - 1))
            return np.asarray(depths[index], dtype=np.float32)
        raise ValueError(f"unsupported metric_depth shape: {depths.shape}")

    def _video_metric_processed_shape_from_payload(self, payload: Dict[str, Any]) -> tuple[int, int]:
        frames = payload.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                preprocess = frame.get("preprocess_geometry")
                if not isinstance(preprocess, dict):
                    continue
                size = preprocess.get("final_processed_size_wh")
                if isinstance(size, list) and len(size) == 2:
                    width, height = int(size[0]), int(size[1])
                    if width > 0 and height > 0:
                        return height, width
        video_metadata = payload.get("video_metadata")
        if isinstance(video_metadata, dict):
            width = int(video_metadata.get("width") or 0)
            height = int(video_metadata.get("height") or 0)
            if width > 0 and height > 0:
                return height, width
        raise ValueError("video_metric_depth sidecar is missing processed_images and payload has no processed shape")

    def _intrinsic_for_frame(self, intrinsics: np.ndarray, frame_index: int) -> np.ndarray:
        if intrinsics.ndim == 3:
            index = max(0, min(int(frame_index), intrinsics.shape[0] - 1))
            return np.asarray(intrinsics[index], dtype=np.float64).reshape(3, 3)
        if intrinsics.ndim == 2:
            return np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        raise ValueError(f"unsupported intrinsics shape: {intrinsics.shape}")

    def _overlap_proxy_mesh_geometries(
        self,
        *,
        mesh_geometries: Dict[str, tuple[np.ndarray, np.ndarray]],
        object_plan: ObjectPlan,
        use_obb: bool,
    ) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        geometry_by_object_id = {
            str(item.object_id): str(item.geometry_type or "").strip().lower()
            for item in object_plan.target_objects
        }
        return {
            object_id: self._overlap_proxy_mesh_geometry(
                mesh_geometry=mesh_geometry,
                geometry_type=geometry_by_object_id.get(object_id, ""),
                use_obb=use_obb,
            )
            for object_id, mesh_geometry in mesh_geometries.items()
        }

    def _overlap_proxy_mesh_geometry(
        self,
        *,
        mesh_geometry: tuple[np.ndarray, np.ndarray],
        geometry_type: str,
        use_obb: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices, faces = mesh_geometry
        geometry_type = str(geometry_type or "").strip().lower()
        if geometry_type in {"box", "cube"}:
            return self._box_proxy_mesh_geometry(vertices, use_obb=use_obb)
        if geometry_type == "sphere":
            return self._primitive_proxy_mesh_geometry(vertices, primitive="sphere")
        if geometry_type == "cylinder":
            return self._primitive_proxy_mesh_geometry(vertices, primitive="cylinder")
        return mesh_geometry

    _BOX_PROXY_FACES = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )

    def _box_proxy_mesh_geometry(
        self, source_vertices: np.ndarray, *, use_obb: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if use_obb:
            oriented = self._oriented_box_proxy_mesh_geometry(source_vertices)
            if oriented is not None:
                return oriented
        bounds_min = np.min(source_vertices, axis=0)
        bounds_max = np.max(source_vertices, axis=0)
        x0, y0, z0 = bounds_min.tolist()
        x1, y1, z1 = bounds_max.tolist()
        vertices = np.asarray(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        return vertices, self._BOX_PROXY_FACES.copy()

    def _oriented_box_proxy_mesh_geometry(
        self, source_vertices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        import trimesh

        try:
            to_origin, extents = trimesh.bounds.oriented_bounds(
                np.asarray(source_vertices, dtype=np.float64)
            )
        except Exception:
            return None
        transform = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
        rotation = transform[:3, :3]
        center = transform[:3, 3]
        half = np.asarray(extents, dtype=np.float64).reshape(3) * 0.5
        unit = np.asarray(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        vertices = (unit * half.reshape(1, 3)) @ rotation.T + center.reshape(1, 3)
        return vertices, self._BOX_PROXY_FACES.copy()

    def _primitive_proxy_mesh_geometry(
        self,
        source_vertices: np.ndarray,
        *,
        primitive: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        if primitive == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        elif primitive == "cylinder":
            mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=16)
        else:
            raise ValueError(f"unsupported primitive proxy: {primitive}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        source_min = np.min(source_vertices, axis=0)
        source_max = np.max(source_vertices, axis=0)
        source_center = 0.5 * (source_min + source_max)
        source_extents = np.maximum(source_max - source_min, 1e-9)
        proxy_min = np.min(vertices, axis=0)
        proxy_max = np.max(vertices, axis=0)
        proxy_center = 0.5 * (proxy_min + proxy_max)
        proxy_extents = np.maximum(proxy_max - proxy_min, 1e-9)
        vertices = (vertices - proxy_center.reshape(1, 3)) * (source_extents / proxy_extents).reshape(1, 3)
        vertices = vertices + source_center.reshape(1, 3)
        return vertices, faces

    def _load_mesh_geometry(self, mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        cache_key = None
        if self._question_cache_enabled:
            signature = self._question_cache_signature(mesh_path)
            if signature is not None:
                cache_key = ("mesh_geometry", signature)
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    return cached[0].copy(), cached[1].copy()
        mesh = trimesh.load(mesh_path, force="mesh")
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"invalid mesh vertices: {mesh_path}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"invalid triangular mesh faces: {mesh_path}")
        if cache_key is not None:
            self._question_cache[cache_key] = (vertices.copy(), faces.copy())
        return vertices, faces

    def _cached_mesh_vertices(self, mesh_path: Path) -> np.ndarray:
        cache_key = None
        if self._question_cache_enabled:
            signature = self._question_cache_signature(mesh_path)
            if signature is not None:
                cache_key = ("mesh_vertices", signature)
                cached = self._question_cache.get(cache_key)
                if cached is not None:
                    return cached.copy()
        vertices = _load_mesh_vertices(mesh_path)
        if cache_key is not None:
            self._question_cache[cache_key] = vertices.copy()
        return vertices

    def _export_mesh_with_source_colors(
        self,
        *,
        vertices: np.ndarray,
        faces: np.ndarray,
        source_mesh_path: Path,
        output_path: Path,
    ) -> None:
        """Export a derived mesh, carrying the source's per-vertex colors over.

        Derived meshes (flush-flattened statics, ray-rescaled agent) only transform
        vertices and never reorder them, so the source COLOR_0 maps 1:1. Without it the
        Blender debug render falls back to the debug palette and the object loses its
        video appearance. Colors are cosmetic: any failure is swallowed."""
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        try:
            source = trimesh.load(source_mesh_path, force="mesh")
            if hasattr(source, "geometry"):
                source = trimesh.util.concatenate(tuple(source.geometry.values()))
            colors = np.asarray(source.visual.vertex_colors)
            if colors.ndim == 2 and len(colors) == len(mesh.vertices):
                mesh.visual.vertex_colors = colors
        except Exception:
            pass
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)

    def _resize_mask_to_shape(self, mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
        import cv2

        if mask is None:
            return None
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None
        target_h, target_w = shape
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    def _mask_iou(self, rendered_mask: np.ndarray, target_mask: np.ndarray) -> float | None:
        rendered = np.asarray(rendered_mask).astype(bool)
        target = np.asarray(target_mask).astype(bool)
        if rendered.shape != target.shape:
            return None
        union = np.logical_or(rendered, target)
        union_count = int(union.sum())
        if union_count <= 0:
            return None
        intersection_count = int(np.logical_and(rendered, target).sum())
        return float(intersection_count) / float(union_count)

    def _object_mask_union(
        self,
        records_all: Dict[str, Dict[int, Dict[str, Any]]],
        mask_arrays: Dict[str, np.ndarray],
        frame_index: int,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Union of EVERY recognized object's mask at one frame, at image_shape.

        Fed as the ``occluder_mask`` for depth-free occlusion: whichever object owns a
        pixel in the observed image is the front-most one there, so a rendered pixel of
        the current object that lands on any other object's mask is treated as occluded.
        ``mask_occluded_mesh_mask`` subtracts the object's own mask internally, so passing
        the full union (own included) yields exactly "the other objects".
        """
        union: np.ndarray | None = None
        fi = int(frame_index)
        for by_frame in records_all.values():
            record = by_frame.get(fi) or {}
            mask = self._resize_mask_to_shape(
                mask_arrays.get(str(record.get("mask_key") or "")), image_shape
            )
            if mask is None:
                continue
            mask_bool = np.asarray(mask).astype(bool)
            union = mask_bool if union is None else (union | mask_bool)
        return union

    def _object_mask_union_by_frame(
        self,
        question_dir: Path,
        frames: Any,
        image_shape: tuple[int, int],
    ) -> Dict[int, np.ndarray | None]:
        """{frame_index: all-object mask union} loaded from the SAM3 video masks.

        Used where only the single object's mask records are in scope (the flush
        member/static-fixture refiners); the loaded records/arrays are memoized per
        question so repeated calls do not re-read the sidecar.
        """
        cache = getattr(self, "_object_union_records_cache", None)
        if cache is None:
            cache = self._object_union_records_cache = {}
        key = str(question_dir)
        entry = cache.get(key)
        if entry is None:
            records_all = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(records_all)
            entry = cache[key] = (records_all, mask_arrays)
        records_all, mask_arrays = entry
        return {
            int(f): self._object_mask_union(records_all, mask_arrays, int(f), image_shape)
            for f in frames
        }

    def _pose_frame_index(self, pose: Dict[str, Any]) -> int | None:
        try:
            return int(pose.get("frame_index"))
        except (TypeError, ValueError):
            return None


class SimulatableWorldReconstructionAdapter(ExternalToolAdapter):
    tool_name = "simulatable_world_reconstruction"
    env_var = ""
    artifact_name = "simulatable_world_reconstruction.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        swr_fit_backend_route = None
        swr_fit_strategy_route = None
        swr_fit_geometry_source_route = None
        swr_visual_pose_preservation_route = None
        if not self.dry_run:
            special_scene = (
                object_plan.special_scene
                if isinstance(object_plan.special_scene, dict)
                else {}
            )
            scene_metadata = (
                special_scene.get("scene_metadata")
                if isinstance(special_scene.get("scene_metadata"), dict)
                else {}
            )
            scenario = str(
                scene_metadata.get("scenario")
                or special_scene.get("scenario")
                or getattr(scene, "scenario", "")
                or ""
            ).strip().lower() or None
            policy_benchmark = _route_policy_benchmark_for_scene(scene)
            swr_fit_backend_route = _require_swr_fit_backend_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario,
                object_plan=object_plan,
            )
            swr_fit_strategy_route = _require_swr_fit_strategy_route(
                policy_benchmark=policy_benchmark,
                scenario=scenario,
                object_plan=object_plan,
            )
            swr_fit_geometry_source_route = (
                _require_swr_fit_geometry_source_route(
                    policy_benchmark=policy_benchmark,
                    scenario=scenario,
                    object_plan=object_plan,
                )
            )
            swr_visual_pose_preservation_route = (
                _require_swr_visual_pose_preservation_route(
                    policy_benchmark=policy_benchmark,
                    scenario=scenario,
                    object_plan=object_plan,
                )
            )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if swr_fit_backend_route is not None:
                expected_backend = str(swr_fit_backend_route["route"])
                existing_fit = existing.payload.get("world_reconstruction_fit")
                existing_manifest = existing.payload.get(
                    "world_reconstruction_fit_manifest"
                )
                observed_backends = []
                if isinstance(existing_fit, dict) and existing_fit.get("backend"):
                    observed_backends.append(str(existing_fit["backend"]))
                if isinstance(existing_manifest, dict):
                    rollout = existing_manifest.get("rollout")
                    if isinstance(rollout, dict) and rollout.get("backend"):
                        observed_backends.append(str(rollout["backend"]))
                if any(value != expected_backend for value in observed_backends):
                    raise ValueError(
                        "cached SWR backend conflicts with the resolved route: "
                        f"observed={observed_backends!r} expected={expected_backend!r}"
                    )
                _record_swr_fit_backend_result(
                    existing.payload,
                    swr_fit_backend_route,
                )
                if swr_fit_strategy_route is not None:
                    strategy_resolution = self._validate_fit_strategy_result(
                        existing.payload,
                        backend=expected_backend,
                        swr_fit_strategy_route=swr_fit_strategy_route,
                    )
                    _record_swr_fit_strategy_result(
                        existing.payload,
                        swr_fit_strategy_route,
                    )
                    existing.payload["fit_strategy_resolution"] = (
                        strategy_resolution
                    )
                if swr_fit_geometry_source_route is not None:
                    existing_manifest_route = (
                        existing_manifest.get("swr_fit_geometry_source_route")
                        if isinstance(existing_manifest, dict)
                        else None
                    )
                    if not isinstance(existing_manifest_route, dict):
                        raise ValueError(
                            "cached Physion++ SWR manifest is missing its "
                            "SWR fit-geometry-source route; rerun SWR instead "
                            "of relabeling the cached result"
                        )
                    _record_swr_fit_geometry_source_result(
                        {},
                        existing_manifest_route,
                    )
                    if (
                        existing_manifest_route.get("route")
                        != swr_fit_geometry_source_route.get("route")
                    ):
                        raise ValueError(
                            "cached SWR geometry source conflicts with the "
                            "resolved route"
                        )
                    _record_swr_fit_geometry_source_result(
                        existing.payload,
                        swr_fit_geometry_source_route,
                    )
                if swr_visual_pose_preservation_route is not None:
                    existing_visual_route = existing.payload.get(
                        "swr_visual_pose_preservation_route"
                    )
                    if not isinstance(existing_visual_route, dict):
                        raise ValueError(
                            "cached collision SWR artifact is missing its "
                            "visual-pose-preservation route; rerun SWR instead "
                            "of relabeling the cached result"
                        )
                    if existing_visual_route != swr_visual_pose_preservation_route:
                        raise ValueError(
                            "cached SWR visual-pose-preservation route mismatch"
                        )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                {
                    **self.placeholder_payload(scene=scene, object_plan=object_plan),
                    "question_id": object_plan.question_id,
                    "simulatable_world_reconstruction_stage": "world_reconstruction_fit_manifest",
                    "pose_correction_artifact": None,
                    "trajectory_correction": None,
                    "target_trajectories": None,
                    "world_reconstruction_fit_manifest": None,
                },
                status="dry_run",
            )

        pose_correction_path = self.artifacts.artifact_path_by_name(question_dir, "pose_correction.json")
        pose_correction = self.artifacts.read_optional(pose_correction_path)
        if not pose_correction:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=f"missing required pose correction artifact: {pose_correction_path}",
            )

        trajectory_correction = pose_correction.get("trajectory_correction")
        corrected_trajectories = (
            trajectory_correction.get("corrected_trajectories")
            if isinstance(trajectory_correction, dict)
            else None
        )
        video_metric_depth = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "video_metric_depth.json")
        )
        video_metadata = video_metric_depth.get("video_metadata") if isinstance(video_metric_depth, dict) else {}
        if not isinstance(video_metadata, dict):
            video_metadata = {}
        target_trajectories = self._target_trajectories(
            corrected_trajectories=corrected_trajectories,
            object_plan=object_plan,
            pose_correction=pose_correction,
            question_dir=question_dir,
            swr_fit_geometry_source_route=swr_fit_geometry_source_route,
        )
        world_reconstruction_fit_manifest = self._physics_alignment_manifest(
            scene=scene,
            object_plan=object_plan,
            video_metadata=video_metadata,
            pose_correction_path=pose_correction_path,
            target_trajectories=target_trajectories,
            pose_correction=pose_correction,
            swr_fit_backend_route=swr_fit_backend_route,
            swr_fit_strategy_route=swr_fit_strategy_route,
            swr_fit_geometry_source_route=swr_fit_geometry_source_route,
            swr_visual_pose_preservation_route=(
                swr_visual_pose_preservation_route
            ),
        )
        payload = {
            **self.placeholder_payload(scene=scene, object_plan=object_plan),
            "question_id": object_plan.question_id,
            "simulatable_world_reconstruction_stage": "world_reconstruction_fit_manifest",
            "source_artifact": str(pose_correction_path),
            "gravity_direction_camera": pose_correction.get("gravity_direction_camera"),
            "gravity_direction_coordinate_frame": pose_correction.get("gravity_direction_coordinate_frame"),
            "rotation_correction": pose_correction.get("rotation_correction"),
            "support_plane_position_correction": pose_correction.get("support_plane_position_correction"),
            "trajectory_correction": trajectory_correction,
            "corrected_trajectories": corrected_trajectories,
            "target_trajectories": target_trajectories,
            "world_reconstruction_fit_manifest": world_reconstruction_fit_manifest,
            "active_interval_simulation_state": self._active_interval_state(corrected_trajectories),
        }
        if swr_fit_backend_route is not None:
            _record_swr_fit_backend_result(payload, swr_fit_backend_route)
        if swr_fit_strategy_route is not None:
            _record_swr_fit_strategy_result(payload, swr_fit_strategy_route)
        if swr_fit_geometry_source_route is not None:
            _record_swr_fit_geometry_source_result(
                payload,
                swr_fit_geometry_source_route,
            )
        if swr_visual_pose_preservation_route is not None:
            _record_swr_visual_pose_preservation_result(
                payload,
                swr_visual_pose_preservation_route,
            )
        physics_result = self._run_physics_alignment(
            question_dir=question_dir,
            physics_alignment_manifest=world_reconstruction_fit_manifest,
            swr_fit_backend_route=swr_fit_backend_route,
            swr_fit_strategy_route=swr_fit_strategy_route,
            swr_fit_geometry_source_route=swr_fit_geometry_source_route,
            swr_visual_pose_preservation_route=(
                swr_visual_pose_preservation_route
            ),
        )
        payload["physics_rollout"] = physics_result.get("physics_rollout")
        payload["alignment_optimization"] = physics_result.get("alignment_optimization")
        payload["fit_error"] = physics_result.get("fit_error")
        payload["world_reconstruction_fit"] = {
            "status": physics_result.get("status"),
            "artifact": physics_result.get("artifact"),
            "message": physics_result.get("message"),
            "backend": physics_result.get("backend"),
        }
        if swr_fit_backend_route is not None:
            payload["world_reconstruction_fit"]["resolved_route"] = deepcopy(
                swr_fit_backend_route
            )
        if swr_fit_strategy_route is not None:
            fit_strategy_resolution = physics_result.get(
                "fit_strategy_resolution"
            )
            payload["fit_strategy_resolution"] = fit_strategy_resolution
            payload["world_reconstruction_fit"]["strategy"] = (
                (fit_strategy_resolution or {}).get("effective_strategy")
            )
            payload["world_reconstruction_fit"]["resolved_strategy_route"] = (
                deepcopy(swr_fit_strategy_route)
            )
        if swr_fit_geometry_source_route is not None:
            payload["world_reconstruction_fit"]["resolved_geometry_source_route"] = (
                deepcopy(swr_fit_geometry_source_route)
            )
        if swr_visual_pose_preservation_route is not None:
            payload["world_reconstruction_fit"][
                "resolved_visual_pose_preservation_route"
            ] = deepcopy(swr_visual_pose_preservation_route)
        self.artifacts.write(artifact_path, payload)
        return ToolResult(
            tool_name=self.tool_name,
            status="ok" if physics_result.get("status") in {"ok", "tool_not_configured"} else "tool_error",
            artifact_path=str(artifact_path),
            message=physics_result.get("message"),
            payload=payload,
        )

    def _active_interval_state(self, corrected_trajectories: Any) -> Dict[str, Any]:
        if not isinstance(corrected_trajectories, dict):
            return {
                "applied": False,
                "reason": "missing corrected trajectories",
                "objects": [],
            }
        objects = []
        for item in corrected_trajectories.get("objects", []):
            if not isinstance(item, dict):
                continue
            objects.append(
                {
                    "object_id": item.get("object_id"),
                    "activation": item.get("activation"),
                    "inactive_policy": "absent",
                    "appearance_policy": "appear_at_first_active_frame_disappear_after_last_active_frame",
                }
            )
        return {
            "applied": bool(objects),
            "source": "pose_correction.trajectory_correction.corrected_trajectories",
            "objects": objects,
        }

    def _target_trajectories(
        self,
        *,
        corrected_trajectories: Any,
        object_plan: ObjectPlan,
        pose_correction: Dict[str, Any],
        question_dir: Path,
        swr_fit_geometry_source_route: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if swr_fit_geometry_source_route is not None:
            _record_swr_fit_geometry_source_result(
                {},
                swr_fit_geometry_source_route,
            )
        if not isinstance(corrected_trajectories, dict) or corrected_trajectories.get("applied") is not True:
            return {
                "applied": False,
                "reason": "missing corrected trajectories",
                "source": "pose_correction.trajectory_correction.corrected_trajectories",
                "objects": [],
            }
        target_by_id = {str(item.object_id): item for item in object_plan.target_objects}
        mesh_by_id = self._support_mesh_by_object_id(pose_correction)
        local_axes_by_id = self._foundationpose_local_axes_payload_by_object(question_dir)
        objects = []
        for item in corrected_trajectories.get("objects", []):
            if not isinstance(item, dict) or item.get("status") not in {None, "ok", "partial"}:
                continue
            object_id = str(item.get("object_id") or "")
            if not object_id:
                continue
            target = target_by_id.get(object_id)
            poses = []
            for pose in item.get("poses", []):
                if not isinstance(pose, dict):
                    continue
                matrix = pose.get("corrected_pose_4x4")
                if matrix is None:
                    continue
                try:
                    arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(arr).all() or pose.get("frame_index") is None:
                    continue
                poses.append(
                    {
                        "frame_index": int(pose["frame_index"]),
                        "corrected_pose_4x4": arr.tolist(),
                        "position_camera": arr[:3, 3].astype(float).tolist(),
                    }
                )
            poses.sort(key=lambda record: int(record["frame_index"]))
            mesh_path = mesh_by_id.get(object_id)
            if not mesh_path:
                raise ValueError(
                    "missing pose-correction final effective mesh_path for "
                    f"SWR target object {object_id}"
                )
            mesh_file = Path(mesh_path)
            if not mesh_file.is_file():
                raise ValueError(
                    "pose-correction final effective mesh does not exist for "
                    f"SWR target object {object_id}: {mesh_file}"
                )
            try:
                dimensions = self._mesh_dimensions(mesh_file)
            except Exception as exc:
                raise ValueError(
                    "failed to read pose-correction final effective mesh for "
                    f"SWR target object {object_id}: {mesh_file}"
                ) from exc
            effective_geometry = str(
                item.get("effective_geometry_type")
                or (target.geometry_type if target else "unknown")
            )
            source_geometry = str(
                item.get("source_geometry_type")
                or (target.geometry_type if target else "unknown")
            )
            target_record = {
                "object_id": object_id,
                "status": "ok" if poses else "missing_poses",
                "description": target.description if target else "",
                "geometry_type": effective_geometry,
                "source_geometry_type": source_geometry,
                "appearance": target.appearance if target else {},
                "mesh_path": mesh_path,
                "dimensions": dimensions,
                "dimensions_source": "mesh_aabb_extent",
                "foundationpose_local_axes": local_axes_by_id.get(object_id),
                "activation": item.get("activation") if isinstance(item.get("activation"), dict) else self._activation_from_poses(poses),
                "pose_count": len(poses),
                "poses": poses,
            }
            if swr_fit_geometry_source_route is not None:
                target_record["mesh_source"] = SWR_FIT_GEOMETRY_SOURCE_ROUTE
            objects.append(target_record)
        payload = {
            "applied": bool(objects),
            "source": corrected_trajectories.get("source") or "pose_correction.trajectory_correction.corrected_trajectories",
            "pose_field": "corrected_pose_4x4",
            "translation_field": "position_camera",
            "objects": objects,
        }
        if swr_fit_geometry_source_route is not None:
            payload["fit_geometry_source"] = SWR_FIT_GEOMETRY_SOURCE_ROUTE
            payload["swr_fit_geometry_source_route"] = deepcopy(
                swr_fit_geometry_source_route
            )
        return payload

    def _support_mesh_by_object_id(self, pose_correction: Dict[str, Any]) -> Dict[str, str]:
        support = pose_correction.get("support_plane_position_correction")
        if not isinstance(support, dict):
            return {}
        mesh_by_id = {}
        for item in support.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "")
            mesh_path = item.get("mesh_path")
            if object_id and mesh_path:
                mesh_by_id[object_id] = str(mesh_path)
        return mesh_by_id

    def _mesh_dimensions(self, mesh_path: Path) -> List[float]:
        vertices = _load_mesh_vertices(mesh_path)
        min_corner = np.min(vertices, axis=0)
        max_corner = np.max(vertices, axis=0)
        dimensions = max_corner - min_corner
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all():
            raise ValueError(f"invalid mesh dimensions: {mesh_path}")
        return [max(float(value), 0.02) for value in dimensions.tolist()]

    def _foundationpose_local_axes_payload_by_object(self, question_dir: Path) -> Dict[str, Dict[str, Any]]:
        mesh_conditioning = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "mesh_conditioning.json")
        )
        if not isinstance(mesh_conditioning, dict):
            return {}
        axes_by_id = {}
        for item in mesh_conditioning.get("objects", []):
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            object_id = str(item.get("object_id") or "")
            axes = item.get("foundationpose_local_axes")
            if object_id and isinstance(axes, dict):
                axes_by_id[object_id] = axes
        return axes_by_id

    def _physics_alignment_fit_strategy(
        self,
        *,
        backend: str,
        swr_fit_strategy_route: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        expected_strategy = SWR_FIT_STRATEGY_BY_BACKEND.get(backend)
        if expected_strategy is None:
            if swr_fit_strategy_route is not None:
                raise ValueError(
                    "SWR fit-strategy route is not applicable to backend "
                    f"{backend!r}"
                )
            return None
        if swr_fit_strategy_route is None:
            raise ValueError(
                f"missing SWR fit-strategy route for backend {backend!r}"
            )
        if (
            swr_fit_strategy_route.get("decision_id")
            != SWR_FIT_STRATEGY_DECISION_ID
        ):
            raise ValueError(
                "SWR fit-strategy route has an unexpected decision_id: "
                f"{swr_fit_strategy_route.get('decision_id')!r}"
            )
        strategy = str(swr_fit_strategy_route.get("route") or "")
        if strategy not in SWR_FIT_STRATEGY_ROUTES:
            raise ValueError(f"unsupported SWR fit strategy route: {strategy!r}")
        if strategy != expected_strategy:
            raise ValueError(
                "SWR fit strategy does not match the selected backend: "
                f"{strategy!r} != {expected_strategy!r} for {backend!r}"
            )
        return strategy

    def _observed_fit_strategy(
        self,
        result: Dict[str, Any],
        *,
        backend: str,
    ) -> tuple[Optional[str], list[str]]:
        resolution = result.get("fit_strategy_resolution")
        if isinstance(resolution, dict):
            effective = str(resolution.get("effective_strategy") or "") or None
            internal = resolution.get("backend_internal_strategies")
            internal_strategies = (
                [str(value) for value in internal]
                if isinstance(internal, list)
                else []
            )
            if effective is not None:
                return effective, internal_strategies

        fit_summary = result.get("world_reconstruction_fit")
        if isinstance(fit_summary, dict) and fit_summary.get("strategy"):
            return str(fit_summary["strategy"]), []

        strategy_container = (
            result.get("joint_alignment_optimization")
            if backend == "swr_backend.wall_bounce_sphere"
            else result.get("alignment_optimization")
        )
        observed = (
            str(strategy_container.get("strategy"))
            if isinstance(strategy_container, dict)
            and strategy_container.get("strategy")
            else None
        )
        internal_strategies = []
        if backend == "swr_backend.wall_bounce_sphere":
            segments = result.get("segments")
            if isinstance(segments, list):
                for segment in segments:
                    alignment = (
                        segment.get("alignment_optimization")
                        if isinstance(segment, dict)
                        else None
                    )
                    if isinstance(alignment, dict) and alignment.get("strategy"):
                        internal_strategies.append(str(alignment["strategy"]))
        return observed, internal_strategies

    def _validate_fit_strategy_result(
        self,
        result: Dict[str, Any],
        *,
        backend: str,
        swr_fit_strategy_route: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        requested = self._physics_alignment_fit_strategy(
            backend=backend,
            swr_fit_strategy_route=swr_fit_strategy_route,
        )
        if requested is None:
            return None
        observed, internal_strategies = self._observed_fit_strategy(
            result,
            backend=backend,
        )
        fallback_applied = (
            backend == "swr_backend.surface_friction_sphere"
            and observed == FRICTION_PLATFORM_SINGLE_PLANE_FALLBACK_STRATEGY
        )
        if observed != requested and not fallback_applied:
            raise ValueError(
                "world reconstruction fit strategy conflicts with the resolved "
                f"route: {observed!r} != {requested!r}"
            )
        if backend == "swr_backend.wall_bounce_sphere":
            if internal_strategies != [
                BOUNCY_WALL_INTERNAL_FIT_STRATEGY,
                BOUNCY_WALL_INTERNAL_FIT_STRATEGY,
            ]:
                raise ValueError(
                    "bouncy-wall backend-internal fit strategy contract mismatch: "
                    f"{internal_strategies!r}"
                )
        return {
            "requested_strategy": requested,
            "effective_strategy": observed,
            "fallback_applied": fallback_applied,
            "fallback_kind": (
                "runtime_optimization_failure_fallback"
                if fallback_applied
                else None
            ),
            "backend_internal_strategies": internal_strategies,
        }

    def _physics_alignment_manifest(
        self,
        *,
        scene: ClevrerScene,
        object_plan: ObjectPlan,
        video_metadata: Dict[str, Any],
        pose_correction_path: Path,
        target_trajectories: Dict[str, Any],
        pose_correction: Dict[str, Any],
        swr_fit_backend_route: Optional[Dict[str, Any]],
        swr_fit_strategy_route: Optional[Dict[str, Any]],
        swr_fit_geometry_source_route: Optional[Dict[str, Any]],
        swr_visual_pose_preservation_route: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        backend = self._physics_alignment_backend(swr_fit_backend_route)
        self._physics_alignment_fit_strategy(
            backend=backend,
            swr_fit_strategy_route=swr_fit_strategy_route,
        )
        if backend in PHYSION_PP_SWR_FIT_BACKEND_ROUTES:
            if swr_fit_geometry_source_route is None:
                raise ValueError(
                    "missing SWR fit-geometry-source route for Physion++ backend "
                    f"{backend!r}"
                )
            _record_swr_fit_geometry_source_result(
                {},
                swr_fit_geometry_source_route,
            )
        elif swr_fit_geometry_source_route is not None:
            raise ValueError(
                "SWR fit-geometry-source route is not applicable to backend "
                f"{backend!r}"
            )
        if backend in {
            "swr_backend.collision_friction_spheres",
            "swr_backend.collision_mass_spheres",
        }:
            if swr_visual_pose_preservation_route is None:
                raise ValueError(
                    "missing SWR visual-pose-preservation route for collision "
                    f"backend {backend!r}"
                )
            _record_swr_visual_pose_preservation_result(
                {},
                swr_visual_pose_preservation_route,
            )
        elif swr_visual_pose_preservation_route is not None:
            raise ValueError(
                "SWR visual-pose-preservation route is not applicable to "
                f"backend {backend!r}"
            )
        manifest = {
            "stage": "simulatable_world_reconstruction",
            "mode": "trajectory_informed_physics_alignment",
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "question_id": object_plan.question_id,
            "pose_correction_artifact": str(pose_correction_path),
            "video_metadata": video_metadata,
            "gravity_direction_camera": pose_correction.get("gravity_direction_camera"),
            "gravity_direction_coordinate_frame": pose_correction.get("gravity_direction_coordinate_frame"),
            "gravity_direction_convention": pose_correction.get("gravity_direction_convention"),
            "object_plan": object_plan.to_dict(),
            "static_scene_objects": self._static_scene_objects(object_plan),
            "target_trajectories": target_trajectories,
            "support_plane_position_correction": pose_correction.get("support_plane_position_correction"),
            "activation_policy": {
                "state_model": "active_interval_absent_outside",
                "inactive_policy": "absent",
                "appearance_policy": "appear_at_first_active_frame_disappear_after_last_active_frame",
            },
            "rollout": {
                "backend": backend,
                "target_pose_field": "corrected_pose_4x4",
                "target_translation_field": "position_camera",
                "rotation_loss": "disabled",
            },
        }
        if swr_fit_backend_route is not None:
            manifest["swr_fit_backend_route"] = deepcopy(swr_fit_backend_route)
        if swr_fit_strategy_route is not None:
            manifest["swr_fit_strategy_route"] = deepcopy(
                swr_fit_strategy_route
            )
        if swr_fit_geometry_source_route is not None:
            manifest["swr_fit_geometry_source_route"] = deepcopy(
                swr_fit_geometry_source_route
            )
        if swr_visual_pose_preservation_route is not None:
            manifest["swr_visual_pose_preservation_route"] = deepcopy(
                swr_visual_pose_preservation_route
            )
        return manifest

    def _physics_alignment_backend(
        self,
        swr_fit_backend_route: Optional[Dict[str, Any]],
    ) -> str:
        if swr_fit_backend_route is None:
            return "swr_backend.impulse_analytic"
        if (
            swr_fit_backend_route.get("decision_id")
            != SWR_FIT_BACKEND_DECISION_ID
        ):
            raise ValueError(
                "SWR fit-backend route has an unexpected decision_id: "
                f"{swr_fit_backend_route.get('decision_id')!r}"
            )
        backend = str(swr_fit_backend_route.get("route") or "")
        if backend not in SWR_FIT_BACKEND_ROUTES:
            raise ValueError(f"unsupported SWR fit backend route: {backend!r}")
        return backend

    def _corrected_rotation_maps_blender_world(
        self,
        physics_alignment_manifest: Dict[str, Any],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        target = physics_alignment_manifest.get("target_trajectories")
        objects = target.get("objects") if isinstance(target, dict) else None
        if not isinstance(objects, list):
            raise ValueError(
                "SWR visual pose preservation requires formal "
                "target_trajectories.objects"
            )
        opencv_to_blender = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pose_maps: Dict[str, Dict[int, np.ndarray]] = {}
        for item in objects:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            frame_map: Dict[int, np.ndarray] = {}
            for pose in item.get("poses") or []:
                if not isinstance(pose, dict):
                    continue
                frame_index = pose.get("frame_index")
                pose_camera = pose.get("corrected_pose_4x4")
                if frame_index is None or pose_camera is None:
                    continue
                try:
                    camera_matrix = np.asarray(
                        pose_camera,
                        dtype=np.float64,
                    ).reshape(4, 4)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(camera_matrix).all():
                    continue
                frame_map[int(frame_index)] = (
                    opencv_to_blender @ camera_matrix
                )[:3, :3]
            if frame_map:
                pose_maps[object_id] = frame_map
        if not pose_maps:
            raise ValueError(
                "SWR visual pose preservation found no corrected rotations"
            )
        return pose_maps

    def _decorate_swr_visual_pose_records(
        self,
        *,
        records_by_object: Any,
        pose_maps: Dict[str, Dict[int, np.ndarray]],
        attachment_counts: Dict[str, int],
    ) -> int:
        if not isinstance(records_by_object, dict):
            return 0
        decorated_count = 0
        for raw_object_id, records in records_by_object.items():
            object_id = str(raw_object_id)
            if not isinstance(records, list) or not records:
                continue
            frame_map = pose_maps.get(object_id)
            if not frame_map:
                raise ValueError(
                    "SWR visual pose preservation is missing corrected "
                    f"rotations for object {object_id}"
                )
            first_frame = min(frame_map)
            for record in records:
                if not isinstance(record, dict):
                    continue
                try:
                    frame_index = int(record["frame_index"])
                    position = np.asarray(
                        record["position_blender_world_m"],
                        dtype=np.float64,
                    ).reshape(3)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "SWR visual pose preservation found an invalid fitted "
                        f"record for object {object_id}"
                    ) from exc
                if frame_index in frame_map:
                    rotation = frame_map[frame_index]
                    policy = "exact_corrected_rotation"
                else:
                    earlier = [
                        value for value in frame_map if value <= frame_index
                    ]
                    if earlier:
                        rotation = frame_map[max(earlier)]
                        policy = "held_last_corrected_rotation"
                    else:
                        rotation = frame_map[first_frame]
                        policy = "held_first_corrected_rotation"
                visual_pose = np.eye(4, dtype=np.float64)
                visual_pose[:3, :3] = rotation
                visual_pose[:3, 3] = position
                record["rotation_blender_world_3x3"] = (
                    rotation.astype(float).tolist()
                )
                record["pose_blender_world_4x4"] = (
                    visual_pose.astype(float).tolist()
                )
                record["visual_pose_only"] = True
                record["visual_rotation_policy"] = policy
                attachment_counts[policy] = (
                    attachment_counts.get(policy, 0) + 1
                )
                decorated_count += 1
        return decorated_count

    def _apply_swr_visual_pose_preservation(
        self,
        *,
        result: Dict[str, Any],
        physics_alignment_manifest: Dict[str, Any],
        route_record: Dict[str, Any],
    ) -> None:
        _record_swr_visual_pose_preservation_result({}, route_record)
        route = str(route_record["route"])
        if route == "visual_pose.position_only":
            result["visual_pose_preservation"] = {
                "enabled": False,
                "position_source": "SWR fitted trajectory",
                "rotation_source": None,
                "affects_physics": False,
                "affects_contact_answer": False,
                "attached_fit_record_count": 0,
                "attachment_counts": {},
            }
            return
        pose_maps = self._corrected_rotation_maps_blender_world(
            physics_alignment_manifest
        )
        attachment_counts: Dict[str, int] = {}
        decorated_count = 0
        segments = result.get("segments")
        if not isinstance(segments, list):
            raise ValueError(
                "SWR visual pose preservation requires collision fit segments"
            )
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            decorated_count += self._decorate_swr_visual_pose_records(
                records_by_object=segment.get("target_trajectories"),
                pose_maps=pose_maps,
                attachment_counts=attachment_counts,
            )
            physics_rollout = segment.get("physics_rollout")
            decorated_count += self._decorate_swr_visual_pose_records(
                records_by_object=(
                    physics_rollout.get("simulated_trajectories")
                    if isinstance(physics_rollout, dict)
                    else None
                ),
                pose_maps=pose_maps,
                attachment_counts=attachment_counts,
            )
        if decorated_count == 0:
            raise ValueError(
                "SWR visual pose preservation found no fitted records to decorate"
            )
        result["visual_pose_preservation"] = {
            "enabled": True,
            "position_source": "SWR fitted trajectory",
            "rotation_source": "pose_correction corrected_pose_4x4",
            "missing_frame_policy": (
                "exact else hold last corrected rotation, or first before onset"
            ),
            "affects_physics": False,
            "affects_contact_answer": False,
            "attached_fit_record_count": decorated_count,
            "attachment_counts": attachment_counts,
        }

    def _run_physics_alignment(
        self,
        *,
        question_dir: Path,
        physics_alignment_manifest: Dict[str, Any],
        swr_fit_backend_route: Optional[Dict[str, Any]],
        swr_fit_strategy_route: Optional[Dict[str, Any]],
        swr_fit_geometry_source_route: Optional[Dict[str, Any]],
        swr_visual_pose_preservation_route: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            backend = self._physics_alignment_backend(swr_fit_backend_route)
            requested_strategy = self._physics_alignment_fit_strategy(
                backend=backend,
                swr_fit_strategy_route=swr_fit_strategy_route,
            )
        except ValueError as exc:
            return {
                "status": "tool_error",
                "artifact": str(self.artifacts.artifact_path(question_dir, self.tool_name, "world_reconstruction_fit.json")),
                "message": str(exc),
                "backend": None,
            }
        if backend == "swr_backend.surface_friction_sphere":
            command = os.getenv("PHYSMIND_PHYSIONPP_FRICTION_SPHERE_SYSID_CMD") or DEFAULT_PHYSIONPP_FRICTION_SPHERE_SYSID_CMD
            script_name = "run_physionpp_friction_sphere_sysid.py"
            missing_message = "PHYSMIND_PHYSIONPP_FRICTION_SPHERE_SYSID_CMD is not set"
        elif backend == "swr_backend.wall_bounce_sphere":
            command = (
                os.getenv("PHYSMIND_PHYSIONPP_BOUNCY_WALL_SPHERE_SYSID_CMD")
                or DEFAULT_PHYSIONPP_BOUNCY_WALL_SPHERE_SYSID_CMD
            )
            script_name = "run_physionpp_bouncy_wall_sphere_sysid.py"
            missing_message = "PHYSMIND_PHYSIONPP_BOUNCY_WALL_SPHERE_SYSID_CMD is not set"
        elif backend == "swr_backend.platform_bounce_sphere":
            command = (
                os.getenv("PHYSMIND_PHYSIONPP_BOUNCY_PLATFORM_SPHERE_SYSID_CMD")
                or DEFAULT_PHYSIONPP_BOUNCY_PLATFORM_SPHERE_SYSID_CMD
            )
            script_name = "run_physionpp_bouncy_platform_sphere_sysid.py"
            missing_message = "PHYSMIND_PHYSIONPP_BOUNCY_PLATFORM_SPHERE_SYSID_CMD is not set"
        elif backend == "swr_backend.collision_friction_spheres":
            command = (
                os.getenv("PHYSMIND_PHYSIONPP_FRICTION_COLLISION_SPHERE_SYSID_CMD")
                or DEFAULT_PHYSIONPP_FRICTION_COLLISION_SPHERE_SYSID_CMD
            )
            script_name = "run_physionpp_friction_collision_sphere_sysid.py"
            missing_message = "PHYSMIND_PHYSIONPP_FRICTION_COLLISION_SPHERE_SYSID_CMD is not set"
        elif backend == "swr_backend.collision_mass_spheres":
            command = (
                os.getenv("PHYSMIND_PHYSIONPP_MASS_COLLISION_SPHERE_SYSID_CMD")
                or DEFAULT_PHYSIONPP_MASS_COLLISION_SPHERE_SYSID_CMD
            )
            script_name = "run_physionpp_mass_collision_sphere_sysid.py"
            missing_message = "PHYSMIND_PHYSIONPP_MASS_COLLISION_SPHERE_SYSID_CMD is not set"
        else:
            command = os.getenv("PHYSMIND_IMPULSE_ANALYTIC_SYSID_CMD") or DEFAULT_IMPULSE_ANALYTIC_SYSID_CMD
            script_name = "run_impulse_analytic_sysid.py"
            missing_message = "PHYSMIND_IMPULSE_ANALYTIC_SYSID_CMD is not set"
        result_path = self.artifacts.artifact_path(question_dir, self.tool_name, "world_reconstruction_fit.json")
        manifest_path = self.artifacts.artifact_path(question_dir, self.tool_name, "world_reconstruction_fit_manifest.json")
        physics_alignment_manifest = json.loads(json.dumps(physics_alignment_manifest))
        rollout = physics_alignment_manifest.setdefault("rollout", {})
        if isinstance(rollout, dict):
            rollout["backend"] = backend
        self.artifacts.write(manifest_path, physics_alignment_manifest)
        if not command:
            return {
                "status": "tool_not_configured",
                "artifact": str(result_path),
                "message": missing_message,
                "backend": backend,
            }
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "world_model" / script_name
        if backend in {
            "swr_backend.wall_bounce_sphere",
            "swr_backend.platform_bounce_sphere",
            "swr_backend.collision_friction_spheres",
            "swr_backend.collision_mass_spheres",
        }:
            args = self._physionpp_world_modeling_alignment_command(
                command=command,
                script_path=script_path,
                world_modeling_dir=question_dir,
                result_path=result_path,
            )
            if backend in {
                "swr_backend.collision_friction_spheres",
                "swr_backend.collision_mass_spheres",
            } and self.artifacts.debug_artifacts:
                args.append("--render-video")
        else:
            args = self._physics_alignment_command(
                command,
                script_path,
                manifest_path,
                result_path,
                render_video=self.artifacts.debug_artifacts,
            )
        if (
            backend == "swr_backend.surface_friction_sphere"
            and requested_strategy == FRICTION_PLATFORM_BOUNDED_FIT_STRATEGY
        ):
            args.extend(["--rollout-mode", "bounded_planes"])
        start = time.perf_counter()
        _log_tool(
            self.tool_name,
            f"physics_alignment command start backend={backend} command={command} manifest={manifest_path} output={result_path}",
        )
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(result_path.parent),
            env=_external_tool_env(question_dir, self.artifacts.debug_artifacts),
        )
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"physics_alignment command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"physics_alignment stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"physics_alignment stderr {stderr}")
        if completed.returncode != 0:
            return {
                "status": "tool_error",
                "artifact": str(result_path),
                "message": (completed.stderr or completed.stdout or "").strip(),
                "backend": backend,
            }
        result = self.artifacts.read_optional(result_path)
        if not isinstance(result, dict) or result.get("status") != "ok":
            return {
                "status": "tool_error",
                "artifact": str(result_path),
                "message": (completed.stderr or completed.stdout or "world reconstruction fit did not produce an ok artifact").strip(),
                "backend": result.get("backend") if isinstance(result, dict) else None,
            }
        if result.get("backend") != backend:
            return {
                "status": "tool_error",
                "artifact": str(result_path),
                "message": (
                    "world reconstruction fit backend conflicts with the resolved "
                    f"route: {result.get('backend')!r} != {backend!r}"
                ),
                "backend": result.get("backend"),
            }
        if swr_visual_pose_preservation_route is not None:
            try:
                self._apply_swr_visual_pose_preservation(
                    result=result,
                    physics_alignment_manifest=physics_alignment_manifest,
                    route_record=swr_visual_pose_preservation_route,
                )
            except ValueError as exc:
                return {
                    "status": "tool_error",
                    "artifact": str(result_path),
                    "message": str(exc),
                    "backend": result.get("backend"),
                }
        try:
            fit_strategy_resolution = self._validate_fit_strategy_result(
                result,
                backend=backend,
                swr_fit_strategy_route=swr_fit_strategy_route,
            )
        except ValueError as exc:
            return {
                "status": "tool_error",
                "artifact": str(result_path),
                "message": str(exc),
                "backend": result.get("backend"),
            }
        if swr_fit_backend_route is not None:
            _record_swr_fit_backend_result(result, swr_fit_backend_route)
        if swr_fit_strategy_route is not None:
            _record_swr_fit_strategy_result(result, swr_fit_strategy_route)
        if swr_fit_geometry_source_route is not None:
            _record_swr_fit_geometry_source_result(
                result,
                swr_fit_geometry_source_route,
            )
        if swr_visual_pose_preservation_route is not None:
            _record_swr_visual_pose_preservation_result(
                result,
                swr_visual_pose_preservation_route,
            )
        if fit_strategy_resolution is not None:
            result["fit_strategy_resolution"] = fit_strategy_resolution
        if (
            swr_fit_backend_route is not None
            or swr_fit_strategy_route is not None
            or swr_fit_geometry_source_route is not None
            or swr_visual_pose_preservation_route is not None
        ):
            self.artifacts.write(result_path, result)
        rollout = result.setdefault("physics_rollout", {})
        if not isinstance(rollout, dict):
            result["physics_rollout"] = {}
            rollout = result["physics_rollout"]
        if self.artifacts.debug_artifacts:
            render_command = os.getenv("PHYSMIND_BLENDER_CMD") or DEFAULT_BLENDER_CMD
            if render_command:
                self._render_physics_alignment_blender_debug(
                    question_dir=question_dir,
                    result_path=result_path,
                    result=result,
                    physics_alignment_manifest=physics_alignment_manifest,
                    command=render_command,
                )
            else:
                rollout["blender_debug_render"] = {
                    "status": "tool_not_configured",
                    "message": "PHYSMIND_BLENDER_CMD is not set",
                }
                self.artifacts.write(result_path, result)
        else:
            rollout = result.setdefault("physics_rollout", {})
            if isinstance(rollout, dict):
                rollout["blender_debug_render"] = {
                    "status": "skipped",
                    "message": "debug_artifacts is disabled",
                }
                self.artifacts.write(result_path, result)
        return {
            **result,
            "status": "ok",
            "artifact": str(result_path),
            "message": result.get("message"),
            "backend": result.get("backend"),
        }

    def _physics_alignment_command(
        self,
        command: str,
        script_path: Path,
        manifest_path: Path,
        result_path: Path,
        *,
        render_video: bool,
    ) -> list[str]:
        tokens = shlex.split(command)
        runner_args = [
            "--manifest",
            str(manifest_path.resolve()),
            "--output",
            str(result_path.resolve()),
        ]
        if render_video:
            runner_args.append("--render-video")
        script_name = script_path.name
        module_name = f"scripts.world_model.{script_path.stem}"
        normalized_tokens = [
            str(script_path) if Path(token).name == script_name else token
            for token in tokens
        ]
        if any(Path(token).name == script_name or token == module_name for token in normalized_tokens):
            return normalized_tokens + runner_args
        return tokens + [str(script_path), *runner_args]

    def _physionpp_world_modeling_alignment_command(
        self,
        *,
        command: str,
        script_path: Path,
        world_modeling_dir: Path,
        result_path: Path,
    ) -> list[str]:
        tokens = shlex.split(command)
        runner_args = [
            "--world-modeling-dir",
            str(world_modeling_dir.resolve()),
            "--output",
            str(result_path.resolve()),
            "--output-dir",
            str(result_path.parent.resolve()),
        ]
        script_name = script_path.name
        module_name = f"scripts.world_model.{script_path.stem}"
        normalized_tokens = [
            str(script_path) if Path(token).name == script_name else token
            for token in tokens
        ]
        if any(Path(token).name == script_name or token == module_name for token in normalized_tokens):
            return normalized_tokens + runner_args
        return tokens + [str(script_path), *runner_args]



    def _blender_world_pose_to_opencv_camera_pose(self, pose_blender_world: np.ndarray) -> np.ndarray:
        opencv_to_blender = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.linalg.inv(opencv_to_blender) @ pose_blender_world

    def _static_scene_objects(self, object_plan: ObjectPlan) -> list[Dict[str, Any]]:
        scene_objects = object_plan.scene_objects if isinstance(object_plan.scene_objects, dict) else {}
        static_objects = scene_objects.get("static_objects")
        return [item for item in static_objects if isinstance(item, dict)] if isinstance(static_objects, list) else []

class SAM3VideoTracksAdapter(ExternalToolAdapter):
    tool_name = "sam3_video_tracks"
    env_var = "PHYSMIND_SAM3_VIDEO_TRACKS_CMD"
    default_command = DEFAULT_SAM3_VIDEO_TRACKS_CMD
    artifact_name = "sam3_video_tracks.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            return existing
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="dry_run",
            )
        command = self._configured_command()
        if not command and self.sam3_video_tracks_worker is None:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="skipped",
            )
        object_plan_path = _scene_object_plan_path(question_dir)
        bench = str(getattr(scene, "benchmark", "clevrer"))
        if self.sam3_video_tracks_worker is not None:
            _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
            start = time.perf_counter()
            response = self.sam3_video_tracks_worker.request(
                {
                    "task_type": self.tool_name,
                    "video": str(scene.video_path),
                    "object_plan": str(object_plan_path),
                    "output": str(artifact_path),
                    "mode": "auto",
                    "bench": bench,
                    "debug_artifacts": self.artifacts.debug_artifacts,
                }
            )
            elapsed = time.perf_counter() - start
            _log_tool(self.tool_name, f"worker request end status={response.get('status')} elapsed={elapsed:.1f}s")
            if response.get("status") != "ok":
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(response.get("message") or response),
                    payload={
                        "tool": self.tool_name,
                        "status": "tool_error",
                        "error_message": str(response.get("message") or response),
                    },
                )
            payload = response.get("payload") or self.artifacts.read_optional(artifact_path) or {}
            _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
            return ToolResult(
                tool_name=self.tool_name,
                status="skipped" if payload.get("status") == "skipped" else "ok",
                artifact_path=str(artifact_path),
                payload=payload,
            )

        args = shlex.split(command) + [
            "--mode",
            "auto",
            "--bench",
            bench,
            "--video",
            str(scene.video_path),
            "--object-plan",
            str(object_plan_path),
            "--output",
            str(artifact_path),
        ]
        env = _external_tool_env(question_dir, self.artifacts.debug_artifacts)
        start = time.perf_counter()
        _log_tool(
            self.tool_name,
            f"command start command={command} video={scene.video_path} output={artifact_path}",
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"stderr {stderr}")
        if completed.returncode != 0:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=(completed.stderr or completed.stdout or "").strip(),
                payload={
                    "tool": self.tool_name,
                    "status": "tool_error",
                    "error_message": (completed.stderr or completed.stdout or "").strip(),
                },
            )
        payload = self.artifacts.read_optional(artifact_path) or {}
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(
            tool_name=self.tool_name,
            status="skipped" if payload.get("status") == "skipped" else "ok",
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        bench = str(getattr(scene, "benchmark", "clevrer"))
        bench_prompts = SAM3_VIDEO_TRACK_PROMPTS_BY_BENCHMARK.get(
            bench,
            SAM3_VIDEO_TRACK_PROMPTS_BY_BENCHMARK["clevrer"],
        )
        special_scene = object_plan.special_scene if isinstance(object_plan.special_scene, dict) else {}
        vlm_prompts = special_scene.get("sam3_video_tracking_prompts")
        if not isinstance(vlm_prompts, list):
            vlm_prompts = []
        return {
            "tool": self.tool_name,
            "status": "placeholder",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "video_filename": scene.video_filename,
            "mode": "auto",
            "bench": bench,
            "prompts": bench_prompts,
            "vlm_prompt_count": len(vlm_prompts),
            "target_object_count": len(object_plan.target_objects),
            "note": (
                "Optional SAM3 video tracking artifact. Configure PHYSMIND_SAM3_VIDEO_TRACKS_CMD "
                "to run scripts/world_model/run_sam3_video_tracks.py. Prompt priority is bench-level concepts, then VLM concepts."
            ),
        }


class SAM3VideoTrackLabelsAdapter(ExternalToolAdapter):
    tool_name = "sam3_video_track_labels"
    env_var = "PHYSMIND_SAM3_VIDEO_TRACK_LABELS_CMD"
    default_command = DEFAULT_SAM3_VIDEO_TRACK_LABELS_CMD
    artifact_name = "sam3_video_track_labels.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        track_vlm_labeling_route = None
        false_positive_advice_effect_route = None
        if not self.dry_run:
            policy_benchmark = _route_policy_benchmark_for_scene(scene)
            track_vlm_labeling_route = _require_track_vlm_labeling_route(
                policy_benchmark=policy_benchmark,
                object_plan=object_plan,
            )
            false_positive_advice_effect_route = (
                _require_false_positive_advice_effect_route(
                    policy_benchmark=policy_benchmark,
                    object_plan=object_plan,
                )
            )
        existing = self._load_existing(artifact_path)
        if existing:
            payload_status = existing.payload.get("status") if isinstance(existing.payload, dict) else None
            if payload_status == "ok" and existing.payload and self._is_current_label_payload(existing.payload):
                if track_vlm_labeling_route is not None:
                    _record_track_vlm_labeling_result(
                        existing.payload,
                        track_vlm_labeling_route,
                    )
                if false_positive_advice_effect_route is not None:
                    _record_false_positive_advice_effect_result(
                        existing.payload,
                        false_positive_advice_effect_route,
                    )
                self._apply_track_derived_object_plan(
                    scene=scene,
                    question_dir=question_dir,
                    object_plan=object_plan,
                    labels_payload=existing.payload,
                )
                self.artifacts.write(artifact_path, existing.payload)
                return existing
            if payload_status != "ok":
                if track_vlm_labeling_route is not None and isinstance(existing.payload, dict):
                    _record_track_vlm_labeling_result(
                        existing.payload,
                        track_vlm_labeling_route,
                    )
                    if false_positive_advice_effect_route is not None:
                        _record_false_positive_advice_effect_result(
                            existing.payload,
                            false_positive_advice_effect_route,
                        )
                    self.artifacts.write(artifact_path, existing.payload)
                return existing
            _log_tool(self.tool_name, f"ignoring stale track label schema artifact={artifact_path}")

        tracks_path = self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        tracks_payload = self.artifacts.read_optional(tracks_path)
        if not tracks_payload or tracks_payload.get("status") in {"placeholder", "skipped"}:
            placeholder = self.placeholder_payload(
                scene=scene,
                object_plan=object_plan,
                reason="missing_sam3_video_tracks",
            )
            if track_vlm_labeling_route is not None:
                _record_track_vlm_labeling_result(
                    placeholder,
                    track_vlm_labeling_route,
                )
            if false_positive_advice_effect_route is not None:
                _record_false_positive_advice_effect_result(
                    placeholder,
                    false_positive_advice_effect_route,
                )
            return self._write_placeholder(
                artifact_path,
                placeholder,
                status="skipped",
            )
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan, reason="dry_run"),
                status="dry_run",
            )

        command = self._configured_command()
        if not command:
            placeholder = self.placeholder_payload(
                scene=scene,
                object_plan=object_plan,
                reason="command_not_configured",
            )
            if track_vlm_labeling_route is not None:
                _record_track_vlm_labeling_result(
                    placeholder,
                    track_vlm_labeling_route,
                )
            if false_positive_advice_effect_route is not None:
                _record_false_positive_advice_effect_result(
                    placeholder,
                    false_positive_advice_effect_route,
                )
            return self._write_placeholder(
                artifact_path,
                placeholder,
                status="skipped",
            )

        label_inputs_path = artifact_path.parent / "sam3_video_track_label_inputs.json"
        label_images_dir = artifact_path.parent / "label_inputs"
        export_args = shlex.split(DEFAULT_SAM3_VIDEO_TRACK_LABEL_INPUTS_CMD) + [
            "--sam3-video-tracks",
            str(tracks_path),
            "--video",
            str(scene.video_path),
            "--output",
            str(label_inputs_path),
            "--image-dir",
            str(label_images_dir),
        ]
        export_result = self._run_command(
            args=export_args,
            artifact_path=label_inputs_path,
            tool_label="track_label_input_export",
        )
        if export_result.status != "ok":
            return ToolResult(
                tool_name=self.tool_name,
                status=export_result.status,
                artifact_path=str(artifact_path),
                message=export_result.message,
                payload=export_result.payload,
            )

        args = shlex.split(command) + [
            "--label-inputs",
            str(label_inputs_path),
            "--object-plan",
            str(_scene_object_plan_path(question_dir)),
            "--output",
            str(artifact_path),
            "--bench",
            str(getattr(scene, "benchmark", "clevrer")),
        ]
        result = self._run_command(args=args, artifact_path=artifact_path, tool_label=self.tool_name)
        if result.status == "ok" and result.payload:
            if track_vlm_labeling_route is not None:
                _record_track_vlm_labeling_result(
                    result.payload,
                    track_vlm_labeling_route,
                )
            if false_positive_advice_effect_route is not None:
                _record_false_positive_advice_effect_result(
                    result.payload,
                    false_positive_advice_effect_route,
                )
            self._apply_track_derived_object_plan(
                scene=scene,
                question_dir=question_dir,
                object_plan=object_plan,
                labels_payload=result.payload,
            )
            self.artifacts.write(artifact_path, result.payload)
        return result

    def _run_command(self, *, args: list[str], artifact_path: Path, tool_label: str) -> ToolResult:
        env = _external_tool_env(artifact_path.parent, self.artifacts.debug_artifacts)
        start = time.perf_counter()
        _log_tool(self.tool_name, f"{tool_label} command start output={artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
        elapsed = time.perf_counter() - start
        stdout = _short_text(completed.stdout or "")
        stderr = _short_text(completed.stderr or "")
        _log_tool(self.tool_name, f"{tool_label} command end returncode={completed.returncode} elapsed={elapsed:.1f}s")
        if stdout:
            _log_tool(self.tool_name, f"{tool_label} stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"{tool_label} stderr {stderr}")
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=message,
                payload={"tool": self.tool_name, "status": "tool_error", "error_message": message},
            )
        payload = self.artifacts.read_optional(artifact_path) or {}
        payload["_command_elapsed_sec"] = elapsed
        _log_tool(self.tool_name, f"{tool_label} artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)

    def _is_current_label_payload(self, payload: Dict[str, Any]) -> bool:
        if payload.get("labeling_mode") != "track_attribute_descriptions":
            return False
        if payload.get("label_input_mode") != "single_representative_red_contour_image":
            return False
        labels = payload.get("track_labels")
        if not isinstance(labels, list):
            return False
        for item in labels:
            if not isinstance(item, dict):
                continue
            if "geometry_type" not in item or not isinstance(item.get("appearance"), dict):
                return False
        return True



    def _apply_track_derived_object_plan(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
        labels_payload: Dict[str, Any],
    ) -> None:
        if not self.dry_run:
            false_positive_advice_effect_route = (
                _require_false_positive_advice_effect_route(
                    policy_benchmark=_route_policy_benchmark_for_scene(scene),
                    object_plan=object_plan,
                )
            )
            if false_positive_advice_effect_route is not None:
                _record_false_positive_advice_effect_result(
                    labels_payload,
                    false_positive_advice_effect_route,
                )
        tracks_path = self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        tracks_payload = self.artifacts.read_optional(tracks_path) or {}
        mass_extra_patient_link_route = _require_mass_extra_patient_link_route(
            object_plan=object_plan,
            tracks_payload=tracks_payload,
        )
        cross_segment_mesh_reuse_route = _require_cross_segment_mesh_reuse_route(
            object_plan=object_plan,
            tracks_payload=tracks_payload,
        )
        records_by_track = self._records_by_track(tracks_payload)
        labels_by_track = {
            str(item.get("track_id")): item
            for item in labels_payload.get("track_labels", [])
            if isinstance(item, dict) and item.get("track_id")
        }
        label_inputs_path = self.artifacts.tool_dir(question_dir, self.tool_name) / "sam3_video_track_label_inputs.json"
        label_inputs_payload = self.artifacts.read_optional(label_inputs_path) or {}
        representative_by_track = {
            str(item.get("track_id")): item
            for item in label_inputs_payload.get("tracks", [])
            if isinstance(item, dict) and item.get("track_id")
        }
        raw_track_count = len(records_by_track)
        # Role-bound agent/patient tracks bypass the minimum-area gate. Frame-count
        # and duplicate gates still apply.
        physion_tracking = tracks_payload.get("physion_tracking")
        role_binding = physion_tracking.get("role_binding") if isinstance(physion_tracking, dict) else None
        role_bound_track_ids = set()
        track_infos = []
        for track_id, records in records_by_track.items():
            label = labels_by_track.get(track_id, {})
            stats = self._track_stats(records)
            if stats["frame_count"] < 8:
                continue
            if stats["max_area"] < 80 and str(track_id) not in role_bound_track_ids:
                continue
            track_infos.append({"track_id": track_id, "label": label, "records": records, "stats": stats})
        track_infos.sort(
            key=lambda item: (
                int(item["stats"].get("first_frame_index") or 0),
                str(item["track_id"]),
            )
        )

        accepted_infos = []
        for info in track_infos:
            if any(self._is_duplicate_track(info, existing) for existing in accepted_infos):
                continue
            accepted_infos.append(info)

        target_objects = []
        for index, info in enumerate(accepted_infos, start=1):
            label = info["label"]
            object_id = f"obj_{index}"
            appearance = label.get("appearance") if isinstance(label.get("appearance"), dict) else {}
            geometry_type = self._geometry_type(label.get("geometry_type"))
            if forced_box:
                geometry_type = "box"
            target_objects.append(
                TargetObject(
                    object_id=object_id,
                    description=self._description(label=label, geometry_type=geometry_type),
                    role=(
                        "static ground fixture detected by SAM3 full-video tracking"
                        if is_static_fixture
                        else "dynamic object detected by SAM3 full-video tracking"
                    ),
                    geometry_type=geometry_type,
                    geometry_confidence=self._confidence(label) if not forced_box else 1.0,
                    appearance={
                        "color": self._color(appearance.get("color")),
                        "material": self._material(appearance.get("material")),
                        "material_confidence": self._confidence(appearance),
                        "reasoning": str(appearance.get("reasoning") or label.get("reason") or ""),
                    },
                    source_track_id=str(info["track_id"]),
                )
            )


        object_plan.target_objects = target_objects
        object_plan.reasoning = (
            "Dynamic object inventory derived from accepted SAM3 full-video tracks. "
            "VLM labels are used only for per-track appearance and geometry metadata."
        )
        object_plan.scene_objects = self._scene_objects(target_objects)
        object_plan.special_scene = self._special_scene_with_inventory(
            existing=object_plan.special_scene,
            accepted_infos=accepted_infos,
            rejected_count=raw_track_count - len(accepted_infos),
        )
        object_plan.status = "track_derived"
        plan_payload = object_plan.to_dict()
        plan_payload["scope"] = "scene"
        plan_payload["object_inventory_source"] = "sam3_full_video_tracks"
        if mesh_reuse:
            plan_payload["two_segment_mesh_reuse"] = mesh_reuse
        self.artifacts.write(_scene_object_plan_path(question_dir), plan_payload)
        labels_payload["track_derived_object_plan"] = {
            "status": "ok",
            "object_plan": str(_scene_object_plan_path(question_dir)),
            "target_object_count": len(target_objects),
            "accepted_track_ids": [str(info["track_id"]) for info in accepted_infos],
            "rejected_track_count": raw_track_count - len(accepted_infos),
        }
        labels_payload["mask_sidecar"] = label_inputs_payload.get("mask_sidecar")
        labels_payload["object_keyframe_source"] = "sam3_video_track_label_representatives"
        object_keyframes = []
        for target, info in zip(target_objects, accepted_infos):
            track_id = str(info["track_id"])
            label = labels_by_track.get(track_id)
            if isinstance(label, dict):
                label["object_id"] = target.object_id
            representative_info = representative_by_track.get(track_id, {})
            representative = (
                representative_info.get("representative_frame")
                if isinstance(representative_info.get("representative_frame"), dict)
                else {}
            )
            object_keyframes.append(
                {
                    "object_id": target.object_id,
                    "track_id": track_id,
                    "source_track_id": track_id,
                    "frame_index": representative.get("frame_index"),
                    "mask_key": representative.get("mask_key"),
                    "area": representative.get("area"),
                    "bbox_xyxy": representative.get("bbox_xyxy"),
                    "centroid_xy": representative.get("centroid_xy"),
                    "representative_image_path": representative_info.get("representative_image_path"),
                    "representative_overlay_path": representative_info.get("representative_overlay_path"),
                    "selected_frame_image": representative.get("selected_frame_image")
                    or representative_info.get("selected_frame_image"),
                    "selected_frame_image_format": representative.get("selected_frame_image_format")
                    or representative_info.get("selected_frame_image_format"),
                    "selected_frame_image_fingerprint": representative.get("selected_frame_image_fingerprint")
                    or representative_info.get("selected_frame_image_fingerprint"),
                    "source_mask_sidecar": label_inputs_payload.get("mask_sidecar"),
                    "selection_rule": (representative.get("selection") or {}).get("selection_rule"),
                    "selection_criteria": representative.get("selection"),
                    "status": "ok" if representative.get("frame_index") is not None else "missing_representative_frame",
                }
            )
        labels_payload["object_keyframes"] = object_keyframes

    def _records_by_track(self, tracks_payload: Dict[str, Any]) -> dict[str, list[Dict[str, Any]]]:
        records_by_track: dict[str, list[Dict[str, Any]]] = {}
        grouped = tracks_payload.get("tracks_by_object")
        if isinstance(grouped, dict):
            for track_id, records in grouped.items():
                if isinstance(records, list):
                    records_by_track[str(track_id)] = [item for item in records if isinstance(item, dict)]
        if records_by_track:
            return records_by_track
        for record in tracks_payload.get("tracks", []):
            if not isinstance(record, dict):
                continue
            track_id = str(record.get("object_id") or "")
            if track_id:
                records_by_track.setdefault(track_id, []).append(record)
        return records_by_track

    def _track_stats(self, records: list[Dict[str, Any]]) -> Dict[str, Any]:
        frame_indices = []
        areas = []
        for record in records:
            try:
                frame_indices.append(int(record.get("frame_index")))
            except (TypeError, ValueError):
                pass
            try:
                areas.append(float(record.get("area")))
            except (TypeError, ValueError):
                pass
        frame_indices = sorted(set(frame_indices))
        return {
            "frame_count": len(frame_indices),
            "first_frame_index": frame_indices[0] if frame_indices else None,
            "last_frame_index": frame_indices[-1] if frame_indices else None,
            "max_area": max(areas) if areas else 0.0,
            "median_area": float(np.median(np.asarray(areas, dtype=np.float64))) if areas else 0.0,
        }

    def _is_duplicate_track(self, candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
        candidate_by_frame = self._record_by_frame(candidate.get("records") or [])
        existing_by_frame = self._record_by_frame(existing.get("records") or [])
        common_frames = sorted(set(candidate_by_frame) & set(existing_by_frame))
        if len(common_frames) < 5:
            return False
        if len(common_frames) > 20:
            step = max(1, len(common_frames) // 20)
            common_frames = common_frames[::step][:20]
        ious = [
            self._bbox_iou(candidate_by_frame[frame].get("bbox_xyxy"), existing_by_frame[frame].get("bbox_xyxy"))
            for frame in common_frames
        ]
        valid_ious = [value for value in ious if value is not None]
        return bool(valid_ious) and float(np.mean(valid_ious)) >= 0.75

    def _record_by_frame(self, records: list[Dict[str, Any]]) -> dict[int, Dict[str, Any]]:
        output = {}
        for record in records:
            try:
                frame_index = int(record.get("frame_index"))
            except (TypeError, ValueError):
                continue
            output[frame_index] = record
        return output

    def _bbox_iou(self, first: Any, second: Any) -> Optional[float]:
        if not (isinstance(first, list) and isinstance(second, list) and len(first) == 4 and len(second) == 4):
            return None
        try:
            ax1, ay1, ax2, ay2 = [float(value) for value in first]
            bx1, by1, bx2, by2 = [float(value) for value in second]
        except (TypeError, ValueError):
            return None
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return None if union <= 0.0 else inter / union

    def _geometry_type(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"cube", "cuboid"}:
            return "box"
        return text if text in {"sphere", "box", "cylinder", "irregular"} else "irregular"

    def _material(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in {"metal", "rubber"} else "rubber"

    def _color(self, value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split()) or "unknown"

    def _confidence(self, item: Dict[str, Any]) -> Optional[float]:
        try:
            return max(0.0, min(1.0, float(item.get("confidence"))))
        except (TypeError, ValueError):
            return None

    def _description(self, *, label: Dict[str, Any], geometry_type: str) -> str:
        description = " ".join(str(label.get("description") or "").split())
        if description:
            return description
        appearance = label.get("appearance") if isinstance(label.get("appearance"), dict) else {}
        color = self._color(appearance.get("color"))
        material = self._material(appearance.get("material"))
        return " ".join(value for value in [color, material, geometry_type] if value and value != "unknown")

    def _scene_objects(self, target_objects: list[TargetObject]) -> Dict[str, Any]:
        return {
            "dynamic_objects": [item.object_id for item in target_objects],
            "static_objects": [
                {
                    "object_id": "ground_plane",
                    "description": "horizontal floor or support plane",
                    "role": "static collision support",
                    "geometry_type": "plane",
                    "appearance": {
                        "color": "gray",
                        "material": "rubber",
                        "material_confidence": 0.5,
                        "reasoning": "default static support appearance",
                    },
                }
            ],
        }

    def _special_scene_with_inventory(
        self,
        *,
        existing: Dict[str, Any],
        accepted_infos: list[Dict[str, Any]],
        rejected_count: int,
    ) -> Dict[str, Any]:
        special_scene = dict(existing) if isinstance(existing, dict) else {}
        special_scene.setdefault(
            "horizontal_plane_motion",
            {
                "applies": "unknown",
                "confidence": None,
                "reason": "No scene-level VLM assessment was available.",
            },
        )
        special_scene.setdefault(
            "roll_stabilization",
            {
                "applies": "unknown",
                "confidence": None,
                "reason": "No scene-level VLM assessment was available.",
            },
        )
        special_scene["sam3_video_tracking_prompts"] = []
        special_scene["track_derived_inventory"] = {
            "source": "sam3_video_tracks + sam3_video_track_labels",
            "accepted_track_ids": [str(info["track_id"]) for info in accepted_infos],
            "accepted_track_count": len(accepted_infos),
            "rejected_track_count": int(rejected_count),
        }
        return special_scene

    def placeholder_payload(
        self,
        scene: ClevrerScene,
        object_plan: ObjectPlan,
        reason: str = "placeholder",
    ) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "status": "placeholder",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "video_filename": scene.video_filename,
            "reason": reason,
            "track_labels": [],
            "target_object_count": len(object_plan.target_objects),
            "note": "Optional VLM appearance and geometry labels for SAM3 video tracks. Object instances are derived from accepted tracks.",
        }


class MaskSelectionAdapterBase(ExternalToolAdapter):
    source_artifact_name = ""
    source_artifact_label = ""
    selection_note = ""
    missing_note = ""
    source_records_key = ""
    source_sidecar_key = "mask_sidecar"

    def _source_artifact_path(self, question_dir: Path) -> Path:
        return self.artifacts.artifact_path_by_name(question_dir, self.source_artifact_name)

    def _source_is_valid(self, payload: Optional[Dict[str, Any]]) -> bool:
        if not payload:
            return False
        if not payload.get(self.source_sidecar_key):
            return False
        records = payload.get(self.source_records_key) if self.source_records_key else None
        return isinstance(records, list) and len(records) > 0

    def _selection_payload_from_source(
        self,
        *,
        scene: ClevrerScene,
        object_plan: ObjectPlan,
        source_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(source_payload.get("mask_selection") or {})
        payload.update(
            {
                "tool": self.tool_name,
                "status": str(source_payload.get("mask_selection_status") or payload.get("status") or "loaded"),
                "scene_index": scene.scene_index,
                "question_id": object_plan.question_id,
                "video_filename": scene.video_filename,
                self.source_artifact_label: str(Path(self.source_artifact_name)),
                "selections": source_payload.get("selected_masks") or [],
                "note": self.selection_note,
            }
        )
        return payload

    def _merge_selection_payload(self, source_payload: Dict[str, Any], payload: Dict[str, Any], status: str) -> None:
        source_payload["selected_masks"] = payload.get("selections", [])
        source_payload["mask_selection_status"] = status
        source_payload["mask_selection"] = {
            key: value
            for key, value in payload.items()
            if key not in {"selections"}
        }

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self._source_artifact_path(question_dir)
        source_payload = self.artifacts.read_optional(artifact_path)
        if source_payload and source_payload.get("selected_masks") and self._source_is_valid(source_payload):
            payload = self._selection_payload_from_source(
                scene=scene,
                object_plan=object_plan,
                source_payload=source_payload,
            )
            _log_tool(self.tool_name, f"loaded artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
            return ToolResult(
                tool_name=self.tool_name,
                status=str(source_payload.get("mask_selection_status") or "loaded"),
                artifact_path=str(artifact_path),
                payload=payload,
            )

        placeholder = self.placeholder_payload(scene=scene, object_plan=object_plan)
        if not self._source_is_valid(source_payload):
            status = "dry_run" if self.dry_run else "placeholder"
            _log_tool(
                self.tool_name,
                f"{status} source_missing_or_invalid={artifact_path} {_payload_summary(self.tool_name, placeholder)}",
            )
            return ToolResult(
                tool_name=self.tool_name,
                status=status,
                artifact_path=str(artifact_path),
                payload=placeholder,
            )
        if self.dry_run:
            _log_tool(self.tool_name, f"dry_run artifact={artifact_path} {_payload_summary(self.tool_name, placeholder)}")
            return ToolResult(
                tool_name=self.tool_name,
                status="dry_run",
                artifact_path=str(artifact_path),
                payload=placeholder,
            )

        try:
            payload = self._select_masks(
                scene=scene,
                question_dir=question_dir,
                object_plan=object_plan,
                sam3_payload=source_payload or {},
            )
        except Exception as exc:
            payload = {
                "tool": self.tool_name,
                "status": "tool_error",
                "scene_index": scene.scene_index,
                "question_id": object_plan.question_id,
                "video_filename": scene.video_filename,
                "error_message": str(exc),
            }
            _log_tool(self.tool_name, f"tool_error artifact={artifact_path} error={_short_text(str(exc), 500)}")
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=str(exc),
                payload=payload,
            )

        status = "parse_error" if any(item.get("status") == "parse_error" for item in payload["selections"]) else "ok"
        payload["status"] = status
        self._merge_selection_payload(source_payload or {}, payload, status=status)
        self.artifacts.write(artifact_path, source_payload)
        _log_tool(self.tool_name, f"{status} artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(
            tool_name=self.tool_name,
            status=status,
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def _select_masks(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
        sam3_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        track_labels = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
        ) or {}
        frame_image_by_object = {
            str(item.get("object_id")): item.get("selected_frame_image")
            for item in track_labels.get("object_keyframes", [])
            if isinstance(item, dict)
        }
        frame_image_by_object_frame = {}
        for frame_info in sam3_payload.get("single_frame_inputs", []):
            if not isinstance(frame_info, dict):
                continue
            frame_dir = frame_info.get("frame_dir")
            frame_index = frame_info.get("source_frame_index")
            if frame_dir is None or frame_index is None:
                continue
            object_id = Path(str(frame_dir)).parent.name
            image_path = Path(str(frame_dir)) / "00000.jpg"
            frame_image_by_object_frame[(object_id, int(frame_index))] = str(image_path)

        candidates_by_object_frame: dict[tuple[str, int], list[Dict[str, Any]]] = {}
        for record in sam3_payload.get("tracks", []):
            if not isinstance(record, dict) or not record.get("mask_key"):
                continue
            object_id = str(record.get("object_id"))
            try:
                frame_index = int(record.get("frame_index"))
            except (TypeError, ValueError):
                continue
            candidates_by_object_frame.setdefault((object_id, frame_index), []).append(
                {
                    "candidate_id": record.get("candidate_id"),
                    "candidate_index": record.get("candidate_index"),
                    "frame_index": frame_index,
                    "sam_object_id": record.get("sam_object_id"),
                    "mask_key": record.get("mask_key"),
                    "text_prompt": record.get("text_prompt"),
                    "area": record.get("area"),
                    "bbox_xyxy": record.get("bbox_xyxy"),
                    "centroid_xy": record.get("centroid_xy"),
                    "score": record.get("score"),
                    "candidate_overlay_path": record.get("candidate_overlay_path"),
                }
            )
        selections = []
        for target in object_plan.target_objects:
            object_id = target.object_id
            object_frame_items = [
                (frame_index, candidates)
                for (candidate_object_id, frame_index), candidates in candidates_by_object_frame.items()
                if candidate_object_id == object_id
            ]
            object_frame_items.sort(key=lambda item: item[0])
            if not object_frame_items:
                selections.append(
                    {
                        "object_id": object_id,
                        "status": "missing_candidates",
                        "candidate_count": 0,
                        "selected_candidate_id": None,
                        "selected_mask_key": None,
                    }
                )
                continue
            for frame_index, candidates in object_frame_items:
                candidates = [
                    item
                    for item in candidates
                    if isinstance(item, dict) and item.get("mask_key")
                ]
                candidates.sort(key=lambda item: int(item.get("candidate_index", 0)))
                if not candidates:
                    continue
                if len(candidates) == 1:
                    candidate = candidates[0]
                    selections.append(
                        self._selection_record(
                            object_id,
                            candidate,
                            "auto_single_candidate",
                            "Only one SAM3 candidate was returned for this object/frame.",
                            candidate_count=len(candidates),
                        )
                    )
                    continue

                selection = self._select_with_vlm(
                    scene=scene,
                    object_id=object_id,
                    description=target.description,
                    question_id=object_plan.question_id,
                    frame_image=(
                        frame_image_by_object_frame.get((object_id, frame_index))
                        or frame_image_by_object.get(object_id)
                    ),
                    candidates=candidates,
                )
                selections.append(selection)

        return {
            "tool": self.tool_name,
            "status": "ok",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "video_filename": scene.video_filename,
            self.source_artifact_label: str(Path(self.source_artifact_name)),
            "selections": selections,
            "note": self.selection_note,
        }

    def _select_with_vlm(
        self,
        *,
        scene: ClevrerScene,
        object_id: str,
        description: str,
        question_id: int,
        frame_image: str | None,
        candidates: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        image_paths = []
        labels = []
        if frame_image and Path(frame_image).exists():
            image_paths.append(frame_image)
            labels.append("original")
        candidate_overlay_count = 0
        for candidate in candidates:
            overlay_path = candidate.get("candidate_overlay_path")
            if overlay_path and Path(overlay_path).exists():
                image_paths.append(overlay_path)
                labels.append(str(candidate["candidate_id"]))
                candidate_overlay_count += 1
        if candidate_overlay_count == 0:
            return {
                "object_id": object_id,
                "status": "parse_error",
                "candidate_count": len(candidates),
                "selected_candidate_id": None,
                "selected_mask_key": None,
                "error_message": "No candidate overlay images were available for VLM mask selection.",
            }

        candidate_lines = [
            (
                f"- {candidate.get('candidate_id')}: area={candidate.get('area')}, "
                f"bbox={candidate.get('bbox_xyxy')}, centroid={candidate.get('centroid_xy')}, "
                f"score={candidate.get('score')}"
            )
            for candidate in candidates
        ]
        prompt = (
            "You are selecting the correct segmentation mask for one target object.\n"
            "The first image, if present, is the original selected frame. Each candidate image is the same frame with one SAM3 mask overlay.\n"
            "Choose the candidate that best covers the target object and avoids other objects/background.\n\n"
            f"Target object_id: {object_id}\n"
            f"Target description: {description}\n"
            f"Candidate IDs:\n" + "\n".join(candidate_lines) + "\n\n"
            "Return compact JSON inside <answer> </answer> with this shape:\n"
            "{\n"
            '  "selected_candidate_id": "candidate_000",\n'
            '  "reason": "brief visual reason"\n'
            "}\n"
            "The selected_candidate_id must be one of the listed candidate IDs."
        )
        response = answer_with_image_files(
            config=self.config,
            prompt=prompt,
            image_paths=image_paths,
            image_labels=labels,
            request_context={
                "scene_index": scene.scene_index,
                "question_id": question_id,
                "stage": self.tool_name,
            },
        )
        try:
            payload = _parse_json_response(response.text)
            selected_candidate_id = str(payload.get("selected_candidate_id") or "")
            candidate = next(
                item for item in candidates if str(item.get("candidate_id")) == selected_candidate_id
            )
            return self._selection_record(
                object_id,
                candidate,
                "vlm_selected",
                str(payload.get("reason", "")),
                candidate_count=len(candidates),
                raw_response=response.text,
            )
        except Exception as exc:
            return {
                "object_id": object_id,
                "status": "parse_error",
                "candidate_count": len(candidates),
                "selected_candidate_id": None,
                "selected_mask_key": None,
                "error_message": str(exc),
                "raw_response": response.text,
            }

    def _selection_record(
        self,
        object_id: str,
        candidate: Dict[str, Any],
        status: str,
        reason: str,
        candidate_count: int,
        raw_response: str | None = None,
    ) -> Dict[str, Any]:
        record = {
            "object_id": object_id,
            "status": status,
            "candidate_count": candidate_count,
            "selected_candidate_id": candidate.get("candidate_id"),
            "selected_candidate_index": candidate.get("candidate_index"),
            "frame_index": candidate.get("frame_index"),
            "sam3_prompt": candidate.get("text_prompt"),
            "selected_sam_object_id": candidate.get("sam_object_id"),
            "selected_mask_key": candidate.get("mask_key"),
            "selected_area": candidate.get("area"),
            "selected_bbox_xyxy": candidate.get("bbox_xyxy"),
            "selected_centroid_xy": candidate.get("centroid_xy"),
            "selected_score": candidate.get("score"),
            "selected_candidate_overlay_path": candidate.get("candidate_overlay_path"),
            "reason": reason,
        }
        if raw_response is not None:
            record["raw_response"] = raw_response
        return record

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "video_filename": scene.video_filename,
            self.source_artifact_label: str(Path(self.source_artifact_name)),
            "selections": [
                {
                    "object_id": item.object_id,
                    "status": "placeholder",
                    "selected_candidate_id": None,
                    "selected_mask_key": None,
                }
                for item in object_plan.target_objects
            ],
            "note": self.missing_note,
        }


class SAM3DObjectsAdapter(ExternalToolAdapter):
    tool_name = "sam3d_objects"
    env_var = "PHYSMIND_SAM3D_OBJECTS_CMD"
    default_command = DEFAULT_SAM3D_OBJECTS_CMD
    artifact_name = "sam3d_meshes.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        observation_selection_route = None
        if not self.dry_run:
            observation_selection_route = (
                _require_sam3d_observation_selection_route(
                    policy_benchmark=_route_policy_benchmark_for_scene(scene),
                    scenario=str(getattr(scene, "scenario", "") or "").strip() or None,
                    object_plan=object_plan,
                )
            )
        if self.sam3d_worker is None or self.dry_run:
            result = super().run(
                scene=scene,
                question_dir=question_dir,
                object_plan=object_plan,
            )
            if (
                observation_selection_route is not None
                and isinstance(result.payload, dict)
                and result.status in {"ok", "loaded", "tool_not_configured"}
            ):
                _record_sam3d_observation_selection_result(
                    result.payload,
                    observation_selection_route,
                )
                self.artifacts.write(Path(result.artifact_path), result.payload)
            return result
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if observation_selection_route is not None and isinstance(existing.payload, dict):
                _record_sam3d_observation_selection_result(
                    existing.payload,
                    observation_selection_route,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        start = time.perf_counter()
        _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
        try:
            response = self.sam3d_worker.request(
                {
                    "type": "task",
                    "task_type": self.tool_name,
                    "video": str(scene.video_path),
                    "object_plan": str(_scene_object_plan_path(question_dir)),
                    "output": str(artifact_path),
                    "question_dir": str(question_dir),
                    "debug_artifacts": self.artifacts.debug_artifacts,
                }
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=str(exc),
                payload={"tool": self.tool_name, "status": "tool_error", "error_message": str(exc)},
            )
        elapsed = time.perf_counter() - start
        _log_tool(self.tool_name, f"worker request end status={response.get('status')} elapsed={elapsed:.1f}s")
        if response.get("status") != "ok":
            message = str(response.get("message") or response)
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=message,
                payload={"tool": self.tool_name, "status": "tool_error", "error_message": message},
            )
        payload = self.artifacts.read_optional(artifact_path) or response.get("payload") or {}
        if observation_selection_route is not None:
            _record_sam3d_observation_selection_result(
                payload,
                observation_selection_route,
            )
            self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)


class MoGe2IntrinsicsAdapter(ExternalToolAdapter):
    tool_name = "moge2_intrinsics"
    env_var = "PHYSMIND_MOGE2_INTRINSICS_CMD"
    default_command = DEFAULT_MOGE2_INTRINSICS_CMD
    artifact_name = "moge2_intrinsics.json"

    def __init__(
        self,
        *,
        artifacts: ArtifactManager,
        config: ModelConfig,
        dry_run: bool,
        moge2_worker: Optional[MoGe2WorkerClient] = None,
    ):
        super().__init__(artifacts=artifacts, config=config, dry_run=dry_run)
        self.moge2_worker = moge2_worker

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        intrinsics_backend_route = None
        if not self.dry_run:
            intrinsics_backend_route = _require_intrinsics_backend_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        existing = self._load_existing(artifact_path)
        if existing:
            if intrinsics_backend_route is not None and isinstance(existing.payload, dict):
                _record_intrinsics_backend_result(
                    existing.payload,
                    intrinsics_backend_route,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if self.dry_run:
            return self._write_placeholder(
                artifact_path,
                self.placeholder_payload(scene=scene, object_plan=object_plan),
                status="dry_run",
            )

        start = time.perf_counter()
        if self.moge2_worker is None:
            result = super().run(scene=scene, question_dir=question_dir, object_plan=object_plan)
        else:
            _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
            try:
                response = self.moge2_worker.request(
                    {
                        "type": "task",
                        "task_type": self.tool_name,
                        "video": str(scene.video_path),
                        "output": str(artifact_path),
                    }
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(exc),
                    payload={"tool": self.tool_name, "status": "tool_error", "error_message": str(exc)},
                )
            _log_tool(
                self.tool_name,
                f"worker request end status={response.get('status')} elapsed={time.perf_counter() - start:.1f}s",
            )
            if response.get("status") != "ok":
                message = str(response.get("message") or response)
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=message,
                    payload={"tool": self.tool_name, "status": "tool_error", "error_message": message},
                )
            payload = self.artifacts.read_optional(artifact_path) or response.get("payload") or {}
            result = ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)
        if (
            intrinsics_backend_route is not None
            and result.status in {"ok", "skipped"}
            and isinstance(result.payload, dict)
        ):
            _record_intrinsics_backend_result(
                result.payload,
                intrinsics_backend_route,
            )
            self.artifacts.write(artifact_path, result.payload)
        elapsed = time.perf_counter() - start
        _log_tool(self.tool_name, f"artifact={artifact_path} elapsed={elapsed:.1f}s {_payload_summary(self.tool_name, result.payload or {})}")
        return result


class MeshConditioningAdapter(ExternalToolAdapter):
    tool_name = "mesh_conditioning"
    env_var = "PHYSMIND_MESH_CONDITIONING_CMD"
    default_command = DEFAULT_MESH_CONDITIONING_CMD
    artifact_name = "mesh_conditioning.json"

    def run(
        self,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_mesh_conditioning_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                scenario=str(getattr(scene, "scenario", "") or "") or None,
                object_plan=object_plan,
            )
        result = super().run(
            scene=scene,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        if route_record is None or not result.payload:
            return result
        _record_mesh_conditioning_result(result.payload, route_record)
        if result.artifact_path:
            self.artifacts.write(Path(result.artifact_path), result.payload)
        return result

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "objects": [
                {
                    "object_id": item.object_id,
                    "geometry_type": item.geometry_type,
                    "conditioning_action": None,
                    "conditioned_mesh_path": None,
                    "status": "placeholder",
                }
                for item in object_plan.target_objects
            ],
            "note": "Configure PHYSMIND_MESH_CONDITIONING_CMD to fit regular primitives or decimate irregular meshes.",
        }


class FoundationPoseAdapter(ExternalToolAdapter):
    tool_name = "foundationpose"
    env_var = "PHYSMIND_FOUNDATIONPOSE_CMD"
    default_command = DEFAULT_FOUNDATIONPOSE_CMD
    artifact_name = "foundationpose_poses.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_pose_input_and_foundationpose_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        scenario = PoseCorrectionAdapter._object_plan_scenario(self, object_plan).strip().lower()
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        candidate_agent_ids = sphere_agent_object_ids(
            scenario=scenario,
            tracks_payload=tracks_payload,
            target_objects=object_plan.target_objects,
        )
        agent_geometry_route = foundationpose_agent_geometry_route(
            object_plan,
            require_for_main_scenario=not self.dry_run,
        )
        expected_policy = route_agent_geometry_policy_payload(
            route_record=agent_geometry_route,
            scenario=scenario,
            agent_object_ids=candidate_agent_ids,
        )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing_payload = self.artifacts.read_optional(artifact_path)
        if (
            existing_payload is not None
            and (
                existing_payload.get("physion_pp_agent_geometry_policy") != expected_policy
                or existing_payload.get("foundationpose_agent_geometry_route")
                != agent_geometry_route
            )
        ):
            mismatch_payload = {
                "tool": self.tool_name,
                "status": "tool_error",
                "expected_physion_pp_agent_geometry_policy": expected_policy,
                "found_physion_pp_agent_geometry_policy": existing_payload.get(
                    "physion_pp_agent_geometry_policy"
                ),
                "expected_foundationpose_agent_geometry_route": agent_geometry_route,
                "found_foundationpose_agent_geometry_route": existing_payload.get(
                    "foundationpose_agent_geometry_route"
                ),
            }
            if route_record is not None:
                _record_pose_input_and_foundationpose_result(
                    mismatch_payload,
                    route_record,
                )
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=(
                    "FoundationPose artifact geometry-policy mismatch; use a fresh run directory "
                    "instead of mixing sphere-agent and native-agent pose artifacts."
                ),
                payload=mismatch_payload,
            )
        if self.foundationpose_worker is None or self.dry_run:
            result = super().run(scene=scene, question_dir=question_dir, object_plan=object_plan)
            if agent_geometry_route is not None and isinstance(result.payload, dict):
                result.payload["foundationpose_agent_geometry_route"] = deepcopy(
                    agent_geometry_route
                )
            if route_record is not None and isinstance(result.payload, dict):
                _record_pose_input_and_foundationpose_result(
                    result.payload,
                    route_record,
                )
                if result.artifact_path:
                    self.artifacts.write(Path(result.artifact_path), result.payload)
            return result
        existing = self._load_existing(artifact_path)
        if existing:
            if agent_geometry_route is not None and isinstance(existing.payload, dict):
                existing.payload["foundationpose_agent_geometry_route"] = deepcopy(
                    agent_geometry_route
                )
            if route_record is not None and isinstance(existing.payload, dict):
                _record_pose_input_and_foundationpose_result(
                    existing.payload,
                    route_record,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        start = time.perf_counter()
        _log_tool(self.tool_name, f"worker request start video={scene.video_path} output={artifact_path}")
        try:
            response = self.foundationpose_worker.request(
                {
                    "type": "task",
                    "task_type": self.tool_name,
                    "video": str(scene.video_path),
                    "object_plan": str(_scene_object_plan_path(question_dir)),
                    "output": str(artifact_path),
                    "question_dir": str(question_dir),
                    "debug": 0,
                    "debug_artifacts": 1 if self.artifacts.debug_artifacts else 0,
                }
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=str(exc),
                payload={"tool": self.tool_name, "status": "tool_error", "error_message": str(exc)},
            )
        elapsed = time.perf_counter() - start
        _log_tool(self.tool_name, f"worker request end status={response.get('status')} elapsed={elapsed:.1f}s")
        if response.get("status") != "ok":
            message = str(response.get("message") or response)
            return ToolResult(
                tool_name=self.tool_name,
                status="tool_error",
                artifact_path=str(artifact_path),
                message=message,
                payload={"tool": self.tool_name, "status": "tool_error", "error_message": message},
            )
        payload = self.artifacts.read_optional(artifact_path) or response.get("payload") or {}
        if agent_geometry_route is not None:
            payload["foundationpose_agent_geometry_route"] = deepcopy(
                agent_geometry_route
            )
        if route_record is not None:
            _record_pose_input_and_foundationpose_result(payload, route_record)
            self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "pose_frames": str(Path("pose_frames.json")),
            "pose_sam3_masks": str(Path("pose_sam3_masks.json")),
            "video_metric_depth": str(Path("video_metric_depth.json")),
            "objects": [
                {
                    "object_id": item.object_id,
                    "status": "placeholder",
                    "tracking_start_frame_index": None,
                    "tracking_end_frame_index": None,
                    "initial_mask_artifact": str(Path("pose_sam3_masks.json")),
                    "initial_mask_sidecar": str(Path("pose_sam3_masks.npz")),
                    "poses": [],
                }
                for item in object_plan.target_objects
            ],
            "note": "FoundationPose uses pose_sam3_masks first-frame masks for register(), then tracks through each object's pose-frame interval with video metric depth and MoGe-2 fixed intrinsics.",
        }


class PoseSAM3MaskSelectionAdapter(MaskSelectionAdapterBase):
    tool_name = "pose_sam3_mask_selection"
    env_var = ""
    artifact_name = "pose_sam3_masks.json"
    source_artifact_name = "pose_sam3_masks.json"
    source_records_key = "masks"
    source_artifact_label = "source_pose_sam3_artifact"
    selection_note = "For each object, choose exactly one SAM3 first-frame mask for FoundationPose initialization. Multi-candidate objects are selected by VLM from candidate overlay images."
    missing_note = "Select one pose SAM3 candidate mask per target object before FoundationPose."

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_pose_input_and_foundationpose_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        result = super().run(
            scene=scene,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        if route_record is not None and isinstance(result.payload, dict):
            _record_pose_input_and_foundationpose_result(
                result.payload,
                route_record,
            )
        return result


class PoseFramesAdapter(ExternalToolAdapter):
    tool_name = "pose_frames"
    env_var = ""
    artifact_name = "pose_frames.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_pose_input_and_foundationpose_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if route_record is not None and isinstance(existing.payload, dict):
                _record_pose_input_and_foundationpose_result(
                    existing.payload,
                    route_record,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if not self.dry_run:
            try:
                payload = self._build_rule_based_pose_frames(
                    scene=scene,
                    question_dir=question_dir,
                    object_plan=object_plan,
                )
                if self.artifacts.debug_artifacts:
                    self._write_pose_frame_images(scene=scene, question_dir=question_dir, payload=payload)
                else:
                    payload["context_frame_artifacts"] = []
                    payload["selected_frame_artifacts"] = []
                if route_record is not None:
                    _record_pose_input_and_foundationpose_result(
                        payload,
                        route_record,
                    )
            except Exception as exc:
                payload = {
                    "tool": self.tool_name,
                    "status": "tool_error",
                    "scene_index": scene.scene_index,
                    "question_id": object_plan.question_id,
                    "video_filename": scene.video_filename,
                    "error_message": str(exc),
                }
                if route_record is not None:
                    _record_pose_input_and_foundationpose_result(
                        payload,
                        route_record,
                    )
                self.artifacts.write(artifact_path, payload)
                _log_tool(self.tool_name, f"tool_error artifact={artifact_path} error={_short_text(str(exc), 500)}")
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=str(exc),
                    payload=payload,
                )
            self.artifacts.write(artifact_path, payload)
            _log_tool(self.tool_name, f"ok artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
            return ToolResult(
                tool_name=self.tool_name,
                status="ok",
                artifact_path=str(artifact_path),
                payload=payload,
            )
        return self._write_placeholder(
            artifact_path,
            self._with_pose_frame_images(
                scene=scene,
                question_dir=question_dir,
                payload=self.placeholder_payload(scene=scene, object_plan=object_plan),
            )
            if self.artifacts.debug_artifacts
            else {
                **self.placeholder_payload(scene=scene, object_plan=object_plan),
                "context_frame_artifacts": [],
                "selected_frame_artifacts": [],
            },
            status="dry_run",
        )

    def _build_rule_based_pose_frames(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> Dict[str, Any]:
        labels_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
        ) or {}
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        keyframes_by_object = {
            str(item.get("object_id")): item
            for item in labels_payload.get("object_keyframes", [])
            if item.get("object_id") is not None and item.get("status") == "ok"
        }
        selected_records, sidecar = _accepted_sam3_video_records_from_labels(
            labels_payload=labels_payload,
            tracks_payload=tracks_payload,
        )
        records = _annotate_mask_contacts(selected_records, sidecar)
        records_by_object: dict[str, list[Dict[str, Any]]] = {}
        for record in records:
            object_id = str(record.get("object_id"))
            if not object_id or record.get("selected_mask_key") is None:
                continue
            records_by_object.setdefault(object_id, []).append(record)

        object_pose_frames = []
        for target in object_plan.target_objects:
            keyframe = keyframes_by_object.get(target.object_id)
            candidates = sorted(
                records_by_object.get(target.object_id, []),
                key=lambda item: (int(item.get("frame_index", 0)), -int(item.get("selected_area") or 0)),
            )
            if keyframe is None or not keyframe.get("mask_key"):
                object_pose_frames.append(
                    {
                        "object_id": target.object_id,
                        "first_full_visible_frame_index": None,
                        "last_full_visible_frame_index": None,
                        "sam3_prompt": target.description,
                        "registration_frame_selection_rule": "missing_selected_frame_keyframe",
                        "first_frame_selection_rule": "missing_selected_frame_keyframe",
                        "last_frame_selection_rule": "missing_selected_frame_keyframe",
                        "selection_criteria": {
                            "registration": {
                                "source": "sam3_video_track_labels.object_keyframes",
                                "candidate_count": len(labels_payload.get("object_keyframes", [])),
                                "fallback_used": False,
                            },
                            "first": {
                                "candidate_count": len(candidates),
                                "eligible_candidate_count": 0,
                                "fallback_used": False,
                            },
                            "last": {"candidate_count": len(candidates), "rule": "missing_selected_frame_keyframe"},
                        },
                        "first_frame_candidate_count": len(candidates),
                        "first_frame_eligible_count": 0,
                        "first_frame_fallback_used": False,
                        "first_frame_reason": "No SAM3 track-label representative keyframe was available for pose registration.",
                        "last_frame_reason": "No SAM3 track-label representative keyframe was available for pose registration.",
                        "status": "missing_selected_frame_keyframe",
                    }
                )
                continue

            keyframe_mask_key = str(keyframe.get("mask_key"))
            selected = next(
                (item for item in candidates if str(item.get("selected_mask_key")) == keyframe_mask_key),
                None,
            )
            if selected is None:
                selected = next(
                    (
                        item
                        for item in candidates
                        if str(item.get("source_track_id") or item.get("track_id"))
                        == str(keyframe.get("source_track_id") or keyframe.get("track_id"))
                    ),
                    candidates[0] if candidates else {},
                )
            full_visible_candidates = [item for item in candidates if item.get("touching_boundary") is not True]
            if not full_visible_candidates:
                object_pose_frames.append(
                    {
                        "object_id": target.object_id,
                        "first_full_visible_frame_index": None,
                        "last_full_visible_frame_index": None,
                        "sam3_prompt": keyframe.get("sam3_prompt") or selected.get("sam3_prompt") or target.description,
                        "registration_frame_selection_rule": "metric_mesh_selected_keyframe",
                        "first_frame_selection_rule": "no_full_visible_sam3_video_mask",
                        "last_frame_selection_rule": "no_full_visible_sam3_video_mask",
                        "selection_criteria": {
                            "registration": {
                                "source": "sam3_video_track_labels.object_keyframes",
                                "rule": keyframe.get("selection_rule"),
                                "mask_key": keyframe_mask_key,
                                "frame_index": int(keyframe["frame_index"]),
                                "fallback_used": False,
                            },
                            "first": {
                                "source": "full-video SAM3 masks",
                                "candidate_count": len(candidates),
                                "eligible_candidate_count": 0,
                                "rule": "earliest mask that does not touch the image boundary",
                                "fallback_used": False,
                            },
                            "last": {
                                "source": "full-video SAM3 masks",
                                "candidate_count": len(candidates),
                                "eligible_candidate_count": 0,
                                "rule": "latest mask that does not touch the image boundary",
                                "fallback_used": False,
                            },
                        },
                        "first_frame_candidate_count": len(candidates),
                        "first_frame_eligible_count": 0,
                        "first_frame_fallback_used": False,
                        "last_frame_candidate_count": len(candidates),
                        "last_frame_eligible_count": 0,
                        "last_frame_fallback_used": False,
                        "first_frame_reason": "No SAM3 video mask is fully inside the image boundary.",
                        "last_frame_reason": "No SAM3 video mask is fully inside the image boundary.",
                        "initial_mask_key": keyframe_mask_key,
                        "initial_mask_frame_index": int(keyframe["frame_index"]),
                        "status": "no_full_visible_frame",
                    }
                )
                continue

            interval = selected.get("track_interval") if isinstance(selected.get("track_interval"), dict) else {}
            first_full_visible_record = min(
                full_visible_candidates,
                key=lambda item: (int(item.get("frame_index", keyframe["frame_index"])), -int(item.get("selected_area") or 0)),
            )
            last_full_visible_record = max(
                full_visible_candidates,
                key=lambda item: (int(item.get("frame_index", keyframe["frame_index"])), int(item.get("selected_area") or 0)),
            )
            first_interval_record = first_full_visible_record
            last_interval_record = last_full_visible_record
            boundary_area_continuation: Optional[Dict[str, Any]] = None
            first_frame_index = int(first_interval_record.get("frame_index", keyframe["frame_index"]))
            last_frame_index = int(last_interval_record.get("frame_index", keyframe["frame_index"]))
            registration_frame_index = int(keyframe["frame_index"])
            first_boundary_extended = first_frame_index < int(
                first_full_visible_record.get("frame_index", keyframe["frame_index"])
            )
            last_boundary_extended = last_frame_index > int(
                last_full_visible_record.get("frame_index", keyframe["frame_index"])
            )
            object_pose_frames.append(
                {
                    "object_id": target.object_id,
                    "first_full_visible_frame_index": int(first_frame_index),
                    "last_full_visible_frame_index": int(last_frame_index),
                    "sam3_prompt": keyframe.get("sam3_prompt") or selected.get("sam3_prompt") or target.description,
                    "first_frame_reason": (
                        "Physion++ pose interval extends before the earliest non-boundary mask while each "
                        "continuous boundary mask retains at least 50% of the first-touch anchor area; "
                        "registration still reuses the non-boundary metric-mesh keyframe."
                        if first_boundary_extended
                        else "Pose interval starts at the earliest SAM3 video mask that does not touch the image boundary; "
                        "registration reuses the metric-mesh selected keyframe mask."
                    ),
                    "last_frame_reason": (
                        "Physion++ pose interval extends after the latest non-boundary mask while each continuous "
                        "boundary mask retains at least 50% of the first-touch anchor area."
                        if last_boundary_extended
                        else "Pose interval ends at the latest SAM3 video mask that does not touch the image boundary."
                    ),
                    "registration_frame_reason": (
                        "FoundationPose registration frame is reused from SAM3 video track-label representatives."
                    ),
                    "confidence": selected.get("label_confidence") or keyframe.get("label_confidence"),
                    "description": target.description,
                    "role": target.role,
                    "registration_frame_selection_rule": "metric_mesh_selected_keyframe",
                    "first_frame_selection_rule": (
                        "physion_pp_boundary_area_continuation_earliest_track_frame"
                        if first_boundary_extended
                        else "earliest_full_visible_video_track_frame"
                    ),
                    "last_frame_selection_rule": (
                        "physion_pp_boundary_area_continuation_latest_track_frame"
                        if last_boundary_extended
                        else "latest_full_visible_video_track_frame"
                    ),
                    "selection_criteria": {
                        "registration": {
                            "source": "sam3_video_track_labels.object_keyframes",
                            "rule": keyframe.get("selection_rule"),
                            "mask_key": keyframe_mask_key,
                            "frame_index": registration_frame_index,
                            "fallback_used": False,
                        },
                        "first": {
                            "source": "full-video SAM3 masks",
                            "candidate_count": len(candidates),
                            "eligible_candidate_count": len(full_visible_candidates),
                            "rule": (
                                "earliest continuous boundary mask above half first-touch area"
                                if first_boundary_extended
                                else "earliest mask that does not touch the image boundary"
                            ),
                            "fallback_used": False,
                            "boundary_area_continuation": (
                                boundary_area_continuation.get("left")
                                if boundary_area_continuation is not None
                                else None
                            ),
                        },
                        "last": {
                            "source": "full-video SAM3 masks",
                            "rule": (
                                "latest continuous boundary mask above half first-touch area"
                                if last_boundary_extended
                                else "latest mask that does not touch the image boundary"
                            ),
                            "candidate_count": len(candidates),
                            "eligible_candidate_count": len(full_visible_candidates),
                            "fallback_used": False,
                            "boundary_area_continuation": (
                                boundary_area_continuation.get("right")
                                if boundary_area_continuation is not None
                                else None
                            ),
                        },
                    },
                    "first_frame_candidate_count": len(candidates),
                    "first_frame_eligible_count": len(full_visible_candidates),
                    "first_frame_fallback_used": False,
                    "first_frame_boundary_area_continuation_used": first_boundary_extended,
                    "last_frame_candidate_count": len(candidates),
                    "last_frame_eligible_count": len(full_visible_candidates),
                    "last_frame_fallback_used": False,
                    "last_frame_boundary_area_continuation_used": last_boundary_extended,
                    "initial_mask_key": keyframe_mask_key,
                    "initial_mask_frame_index": registration_frame_index,
                    "initial_mask_area": keyframe.get("area"),
                    "initial_mask_bbox_xyxy": keyframe.get("bbox_xyxy"),
                    "initial_selected_candidate_id": keyframe.get("selected_candidate_id"),
                    "initial_selected_candidate_overlay_path": keyframe.get("selected_candidate_overlay_path"),
                    "initial_touching_boundary": keyframe.get("touching_boundary"),
                    "initial_boundary_sides": keyframe.get("boundary_sides"),
                    "initial_touching_other_mask": keyframe.get("touching_other_mask"),
                    "initial_touching_other_object_ids": keyframe.get("touching_other_object_ids"),
                    "last_mask_key": last_interval_record.get("selected_mask_key"),
                    "last_mask_area": last_interval_record.get("selected_area"),
                    "last_mask_bbox_xyxy": last_interval_record.get("selected_bbox_xyxy"),
                    "metric_mesh_keyframe": {
                        "frame_index": registration_frame_index,
                        "mask_key": keyframe_mask_key,
                        "selected_frame_image": keyframe.get("selected_frame_image"),
                        "selected_frame_image_format": keyframe.get("selected_frame_image_format"),
                        "selected_frame_image_fingerprint": keyframe.get("selected_frame_image_fingerprint"),
                    },
                    "track_interval": interval,
                    "physion_pp_boundary_area_continuation": boundary_area_continuation,
                    "status": "ok",
                }
            )

        selected_frame_indices = sorted(
            {
                int(frame)
                for item in object_pose_frames
                if isinstance(item, dict)
                for frame in (
                    item.get("initial_mask_frame_index"),
                    item.get("first_full_visible_frame_index"),
                    item.get("last_full_visible_frame_index"),
                )
                if frame is not None
            }
        )
        return {
            "tool": self.tool_name,
            "status": "ok" if selected_frame_indices else "missing_candidate_mask",
            "selection_method": "metric_mesh_keyframe_registration_with_sam3_video_track_interval",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "question_type": object_plan.question_type,
            "video_filename": scene.video_filename,
            "video_metadata": tracks_payload.get("video_metadata"),
            "context_frame_indices": [],
            "selected_frame_indices": selected_frame_indices,
            "object_pose_frames": object_pose_frames,
            "source_sam3_tracks_artifact": str(self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")),
            "source_sam3_track_labels_artifact": str(self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")),
            "rule": {
                "registration": "reuse SAM3 track-label representative keyframe mask",
                "first": "full-video SAM3 track interval start",
                "last": "full-video SAM3 track last frame",
            },
            "note": (
                "FoundationPose registration strictly reuses SAM3 track-label representative keyframes and their masks. "
                "Tracking output is restricted to the fully visible SAM3 interval."
            ),
        }

    def _with_pose_frame_images(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._write_pose_frame_images(scene=scene, question_dir=question_dir, payload=payload)
        return payload

    def _write_pose_frame_images(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        payload: Dict[str, Any],
    ) -> None:
        context_frame_indices = payload.get("context_frame_indices") or []
        selected_frame_indices = payload.get("selected_frame_indices") or []
        if not context_frame_indices and not selected_frame_indices:
            payload["context_frame_artifacts"] = []
            payload["selected_frame_artifacts"] = []
            return

        tool_dir = self.artifacts.tool_dir(question_dir, self.tool_name)
        context_artifacts = _export_video_frames(
            video_path=scene.video_path,
            output_dir=tool_dir / "context_frames",
            frame_indices=context_frame_indices,
            error_label="pose context",
        )
        selected_artifacts = _export_video_frames(
            video_path=scene.video_path,
            output_dir=tool_dir / "frames",
            frame_indices=selected_frame_indices,
            error_label="pose selected",
        )

        object_ids_by_frame: Dict[int, list[str]] = {}
        for item in payload.get("object_pose_frames", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id"))
            first_frame = item.get("first_full_visible_frame_index")
            registration_frame = item.get("initial_mask_frame_index")
            last_frame = item.get("last_full_visible_frame_index")
            if registration_frame is not None:
                registration_frame = int(registration_frame)
                if registration_frame in selected_artifacts:
                    item["registration_frame_image"] = selected_artifacts[registration_frame]
                    item["initial_mask_frame_image"] = selected_artifacts[registration_frame]
                    object_ids_by_frame.setdefault(registration_frame, []).append(object_id)
            if first_frame is not None:
                first_frame = int(first_frame)
                if first_frame in selected_artifacts:
                    item["first_full_visible_frame_image"] = selected_artifacts[first_frame]
                    object_ids_by_frame.setdefault(first_frame, []).append(object_id)
            if last_frame is not None:
                last_frame = int(last_frame)
                if last_frame in selected_artifacts:
                    item["last_full_visible_frame_image"] = selected_artifacts[last_frame]
                    object_ids_by_frame.setdefault(last_frame, []).append(object_id)

        payload["context_frame_artifacts"] = [
            {
                "frame_index": frame_index,
                "image_path": image_path,
            }
            for frame_index, image_path in sorted(context_artifacts.items())
        ]
        payload["selected_frame_artifacts"] = [
            {
                "frame_index": frame_index,
                "image_path": image_path,
                "object_ids": sorted(set(object_ids_by_frame.get(frame_index, []))),
            }
            for frame_index, image_path in sorted(selected_artifacts.items())
        ]

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "selected_frame_indices": [0],
            "object_pose_frames": [
                {
                    "object_id": item.object_id,
                    "first_full_visible_frame_index": 0,
                    "last_full_visible_frame_index": 0,
                    "sam3_prompt": item.description,
                    "first_frame_selection_rule": "dry_run_default_frame",
                    "last_frame_selection_rule": "dry_run_default_frame",
                    "selection_criteria": {
                        "first": {
                            "candidate_count": 1,
                            "eligible_candidate_count": 1,
                            "fallback_used": False,
                        },
                        "last": {
                            "candidate_count": 1,
                            "rule": "dry_run_default_frame",
                            "fallback_used": False,
                        },
                    },
                    "first_frame_reason": "Dry-run placeholder first complete unobstructed frame.",
                    "last_frame_reason": "Dry-run placeholder last visible frame.",
                }
                for item in object_plan.target_objects
            ],
            "note": "Pose frames identify FoundationPose initialization and tracking endpoint metadata.",
        }


class PoseSAM3Adapter(ExternalToolAdapter):
    tool_name = "pose_sam3_masks"
    env_var = ""
    artifact_name = "pose_sam3_masks.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_pose_input_and_foundationpose_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        if self.dry_run:
            return super().run(scene=scene, question_dir=question_dir, object_plan=object_plan)
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if route_record is not None and isinstance(existing.payload, dict):
                _record_pose_input_and_foundationpose_result(
                    existing.payload,
                    route_record,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        synthesized = self._build_from_video_mask_selection(scene=scene, question_dir=question_dir, object_plan=object_plan)
        if synthesized is not None:
            sidecar_path = artifact_path.with_suffix(".npz")
            arrays = synthesized.pop("_arrays")
            if arrays:
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(sidecar_path, **arrays)
            synthesized["mask_sidecar"] = str(sidecar_path)
            if route_record is not None:
                _record_pose_input_and_foundationpose_result(
                    synthesized,
                    route_record,
                )
            self.artifacts.write(artifact_path, synthesized)
            _log_tool(self.tool_name, f"ok artifact={artifact_path} {_payload_summary(self.tool_name, synthesized)}")
            return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=synthesized)
        message = (
            "Pose SAM3 masks must be reused from Object Segmentation and Event Detection, "
            "but no SAM3 track-label representative mask was available."
        )
        payload = {
            "tool": self.tool_name,
            "status": "missing_candidate_mask",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "video_filename": scene.video_filename,
            "pose_frames": str(self.artifacts.artifact_path_by_name(question_dir, "pose_frames.json")),
            "source_sam3_track_labels_artifact": str(
                self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
            ),
            "target_object_count": len(object_plan.target_objects),
            "mask_record_count": 0,
            "masks": [],
            "selected_masks": [],
            "message": message,
        }
        if route_record is not None:
            _record_pose_input_and_foundationpose_result(payload, route_record)
        self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"missing_candidate_mask artifact={artifact_path} message={message}")
        return ToolResult(
            tool_name=self.tool_name,
            status="tool_error",
            artifact_path=str(artifact_path),
            message=message,
            payload=payload,
        )

    def _build_from_video_mask_selection(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> Optional[Dict[str, Any]]:
        pose_frames = self.artifacts.read_optional(self.artifacts.artifact_path_by_name(question_dir, "pose_frames.json")) or {}
        labels_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
        ) or {}
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        _selected_records, sidecar = _accepted_sam3_video_records_from_labels(
            labels_payload=labels_payload,
            tracks_payload=tracks_payload,
        )
        if not sidecar or not Path(sidecar).exists():
            return None
        source_arrays = np.load(sidecar)
        masks = []
        arrays = {}
        for item in pose_frames.get("object_pose_frames", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id"))
            mask_key = item.get("initial_mask_key")
            mask_frame = item.get("initial_mask_frame_index")
            if mask_frame is None:
                mask_frame = item.get("first_full_visible_frame_index")
            if not mask_key or str(mask_key) not in source_arrays or mask_frame is None:
                continue
            pose_mask_key = f"{object_id}__pose_initial__frame_{int(mask_frame):05d}"
            arrays[pose_mask_key] = source_arrays[str(mask_key)].astype(np.uint8)
            masks.append(
                {
                    "object_id": object_id,
                    "candidate_id": item.get("initial_selected_candidate_id") or "candidate_000",
                    "candidate_index": 0,
                    "frame_index": int(mask_frame),
                    "sam_session_frame_index": 0,
                    "sam_object_id": item.get("initial_selected_candidate_id"),
                    "mask_key": pose_mask_key,
                    "text_prompt": item.get("sam3_prompt"),
                    "purpose": "foundationpose_initial_mask",
                    "first_full_visible_frame_index": item.get("first_full_visible_frame_index"),
                    "last_full_visible_frame_index": item.get("last_full_visible_frame_index"),
                    "tracking_start_frame_index": item.get("first_full_visible_frame_index"),
                    "tracking_end_frame_index": item.get("last_full_visible_frame_index"),
                    "area": item.get("initial_mask_area"),
                    "bbox_xyxy": item.get("initial_mask_bbox_xyxy"),
                    "candidate_overlay_path": item.get("initial_selected_candidate_overlay_path"),
                    "source_sam3_artifact": str(
                        self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
                    ),
                    "source_sam3_mask_key": str(mask_key),
                }
            )
        if not masks:
            return None
        selected_masks = [
            {
                "object_id": item["object_id"],
                "status": "reused_sam3_video_track_mask",
                "candidate_count": 1,
                "selected_candidate_id": item.get("candidate_id"),
                "selected_candidate_index": item.get("candidate_index"),
                "frame_index": item.get("frame_index"),
                "sam3_prompt": item.get("text_prompt"),
                "selected_sam_object_id": item.get("sam_object_id"),
                "selected_mask_key": item.get("mask_key"),
                "selected_area": item.get("area"),
                "selected_bbox_xyxy": item.get("bbox_xyxy"),
                "selected_candidate_overlay_path": item.get("candidate_overlay_path"),
                "reason": "Reused the selected full-video SAM3 track mask for FoundationPose initialization.",
            }
            for item in masks
        ]
        return {
            "tool": self.tool_name,
            "status": "ok",
            "version": "sam3",
            "execution_mode": "reused_sam3_video_track_masks",
            "video": str(scene.video_path),
            "object_plan": str(_scene_object_plan_path(question_dir)),
            "pose_frames": str(self.artifacts.artifact_path_by_name(question_dir, "pose_frames.json")),
            "source_sam3_tracks_artifact": str(self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")),
            "source_sam3_track_labels_artifact": str(
                self.artifacts.artifact_path_by_name(question_dir, "sam3_video_track_labels.json")
            ),
            "source_sam3_mask_sidecar": sidecar,
            "mask_sidecar": str(Path(self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)).with_suffix(".npz")),
            "target_object_count": len(object_plan.target_objects),
            "mask_record_count": len(masks),
            "masks": masks,
            "selected_masks": selected_masks,
            "mask_selection_status": "ok",
            "mask_selection": {
                "tool": "pose_sam3_mask_selection",
                "status": "ok",
                "selection_method": "reused_sam3_video_track_masks",
            },
            "_arrays": arrays,
            "note": "Pose SAM3 masks are reused from Object Segmentation and Event Detection; SAM3 is not run a second time.",
        }

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "scene_index": scene.scene_index,
            "video_filename": scene.video_filename,
            "pose_frames": str(Path("pose_frames.json")),
            "mask_sidecar": str(Path("pose_sam3_masks.npz")),
            "target_object_count": len(object_plan.target_objects),
            "mask_record_count": 0,
            "masks": [],
            "note": "Dry-run placeholder. Real pose SAM3 masks are reused from Object Segmentation and Event Detection.",
        }


class PoseFrameBoundaryValidationAdapter(ExternalToolAdapter):
    tool_name = "pose_frame_boundary_validation"
    env_var = ""
    artifact_name = "pose_frame_boundary_validation.json"

    def run(self, scene: ClevrerScene, question_dir: Path, object_plan: ObjectPlan) -> ToolResult:
        route_record = None
        if not self.dry_run:
            route_record = _require_pose_input_and_foundationpose_route(
                policy_benchmark=_route_policy_benchmark_for_scene(scene),
                object_plan=object_plan,
            )
        artifact_path = self.artifacts.artifact_path(question_dir, self.tool_name, self.artifact_name)
        existing = self._load_existing(artifact_path)
        if existing:
            if route_record is not None and isinstance(existing.payload, dict):
                _record_pose_input_and_foundationpose_result(
                    existing.payload,
                    route_record,
                )
                self.artifacts.write(artifact_path, existing.payload)
            return existing
        if self.dry_run:
            payload = self.placeholder_payload(scene=scene, object_plan=object_plan)
            return self._write_placeholder(artifact_path, payload, status="dry_run")

        pose_frames_path = self.artifacts.artifact_path_by_name(question_dir, "pose_frames.json")
        pose_sam3_path = self.artifacts.artifact_path_by_name(question_dir, "pose_sam3_masks.json")
        pose_frames = self.artifacts.read_optional(pose_frames_path) or {}
        pose_sam3_payload = self.artifacts.read_optional(pose_sam3_path) or {}
        if not pose_sam3_payload.get("selected_masks"):
            selection_result = PoseSAM3MaskSelectionAdapter(
                artifacts=self.artifacts,
                config=self.config,
                dry_run=self.dry_run,
            ).run(scene=scene, question_dir=question_dir, object_plan=object_plan)
            pose_sam3_payload = self.artifacts.read_optional(pose_sam3_path) or {}
            if selection_result.status not in {"ok", "loaded"}:
                payload = {
                    "tool": self.tool_name,
                    "status": "tool_error",
                    "boundary_status": "selection_failed",
                    "scene_index": scene.scene_index,
                    "question_id": object_plan.question_id,
                    "pose_sam3_masks": str(pose_sam3_path),
                    "message": selection_result.message,
                }
                if route_record is not None:
                    _record_pose_input_and_foundationpose_result(
                        payload,
                        route_record,
                    )
                self.artifacts.write(artifact_path, payload)
                return ToolResult(
                    tool_name=self.tool_name,
                    status="tool_error",
                    artifact_path=str(artifact_path),
                    message=selection_result.message,
                    payload=payload,
                )

        diagnostics = self._diagnostics_with_frame_context(pose_frames, pose_sam3_payload)
        touching = [item for item in diagnostics if item.get("touching_boundary")]
        if not touching:
            payload = {
                "tool": self.tool_name,
                "status": "ok",
                "boundary_status": "ok",
                "retry_status": "not_needed",
                "scene_index": scene.scene_index,
                "question_id": object_plan.question_id,
                "pose_frames": str(pose_frames_path),
                "pose_sam3_masks": str(pose_sam3_path),
                "diagnostics": diagnostics,
                "note": "Selected FoundationPose initialization masks do not touch image boundaries.",
            }
            if route_record is not None:
                _record_pose_input_and_foundationpose_result(
                    payload,
                    route_record,
                )
            self.artifacts.write(artifact_path, payload)
            _log_tool(self.tool_name, f"ok artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
            return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)

        payload = {
            "tool": self.tool_name,
            "status": "ok",
            "boundary_status": "registration_keyframe_touches_boundary",
            "retry_status": "not_applicable_metric_mesh_keyframe_reuse",
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "pose_frames": str(pose_frames_path),
            "pose_sam3_masks": str(pose_sam3_path),
            "diagnostics": diagnostics,
            "note": (
                "FoundationPose registration intentionally reuses the metric-mesh keyframe and its mask. "
                "The selected registration mask touches the image boundary; this diagnostic is recorded "
                "without replacing the metric-mesh registration anchor."
            ),
        }
        if route_record is not None:
            _record_pose_input_and_foundationpose_result(payload, route_record)
        self.artifacts.write(artifact_path, payload)
        _log_tool(self.tool_name, f"ok artifact={artifact_path} {_payload_summary(self.tool_name, payload)}")
        return ToolResult(tool_name=self.tool_name, status="ok", artifact_path=str(artifact_path), payload=payload)

    def _diagnostics_with_frame_context(
        self,
        pose_frames: Dict[str, Any],
        pose_sam3_payload: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        first_frame_by_object = {
            str(item.get("object_id")): item.get("first_full_visible_frame_index")
            for item in pose_frames.get("object_pose_frames", [])
            if isinstance(item, dict)
        }
        diagnostics = _selected_mask_boundary_diagnostics(pose_sam3_payload)
        for item in diagnostics:
            object_id = str(item.get("object_id"))
            item["first_full_visible_frame_index"] = first_frame_by_object.get(object_id)
        return diagnostics

    def placeholder_payload(self, scene: ClevrerScene, object_plan: ObjectPlan) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "status": "dry_run" if self.dry_run else "placeholder",
            "boundary_status": None,
            "retry_status": None,
            "scene_index": scene.scene_index,
            "question_id": object_plan.question_id,
            "diagnostics": [],
            "note": "Validate that FoundationPose initialization masks do not touch image boundaries.",
        }


def build_default_adapters(
    artifacts: ArtifactManager,
    config: ModelConfig,
    dry_run: bool,
    video_metric_depth_worker: Optional[VideoMetricDepthWorkerClient] = None,
    foundationpose_worker: Optional[FoundationPoseWorkerClient] = None,
    sam3d_worker: Optional[SAM3DWorkerClient] = None,
    sam3_video_tracks_worker: Optional[SAM3VideoTracksWorkerClient] = None,
    geocalib_worker: Optional[GeoCalibWorkerClient] = None,
    moge2_worker: Optional[MoGe2WorkerClient] = None,
) -> list[ExternalToolAdapter]:
    return [
        SAM3VideoTracksAdapter(
            artifacts=artifacts,
            config=config,
            dry_run=dry_run,
            sam3_video_tracks_worker=sam3_video_tracks_worker,
        ),
        SAM3VideoTrackLabelsAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
        MoGe2IntrinsicsAdapter(artifacts=artifacts, config=config, dry_run=dry_run, moge2_worker=moge2_worker),
        VideoMetricDepthAdapter(
            artifacts=artifacts,
            config=config,
            dry_run=dry_run,
            video_metric_depth_worker=video_metric_depth_worker,
        ),
        SAM3DObjectsAdapter(artifacts=artifacts, config=config, dry_run=dry_run, sam3d_worker=sam3d_worker),
        MeshConditioningAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
        PoseFramesAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
        PoseSAM3Adapter(artifacts=artifacts, config=config, dry_run=dry_run),
        PoseSAM3MaskSelectionAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
        PoseFrameBoundaryValidationAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
        FoundationPoseAdapter(
            artifacts=artifacts,
            config=config,
            dry_run=dry_run,
            foundationpose_worker=foundationpose_worker,
        ),
        PoseCorrectionAdapter(
            artifacts=artifacts,
            config=config,
            dry_run=dry_run,
            geocalib_worker=geocalib_worker,
        ),
        SimulatableWorldReconstructionAdapter(artifacts=artifacts, config=config, dry_run=dry_run),
    ]
