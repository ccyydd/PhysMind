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
from agent.world_model.cross_segment_identity import decide_same_object_yes_no
from agent.world_model.debug_artifacts import run_world_reconstruction_debug_render
from agent.world_model.foundationpose_worker import FoundationPoseWorkerClient
from agent.world_model.geocalib_worker import GeoCalibWorkerClient
from agent.world_model.moge2_worker import MoGe2WorkerClient
from agent.world_model.module_profiles import default_module_profile_policy
from agent.world_model.physion_pp_agent_geometry import (
    foundationpose_agent_geometry_route,
    route_agent_geometry_policy_payload,
    sphere_agent_object_ids,
)
from agent.world_model.sam3d_worker import SAM3DWorkerClient
from agent.world_model.sam3_video_tracks_worker import SAM3VideoTracksWorkerClient
from agent.world_model.schemas import ObjectPlan, TargetObject, ToolResult
from agent.world_model.video_metric_depth_worker import VideoMetricDepthWorkerClient
from benchmark.clevrer import ClevrerScene
from benchmark.physion_pp import PhysionPPScene
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
PHYSION_PP_TRACK_VLM_LABELING_ROUTE = (
    "label.vlm_geometry_concave"
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
PHYSION_PP_WARN_ONLY_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.obb_warn"
)
PHYSION_PP_ROLLBACK_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.obb_rollback"
)
PHYSION_PP_WARN_ONLY_MESH_CONDITIONING_SCENARIOS = frozenset(
    {
        "friction_platform_pp",
        "bouncy_platform_pp",
        "friction_collision_pp",
        "mass_collision_pp",
    }
)


def _expected_mesh_conditioning_route(
    benchmark: str,
    scenario: str | None,
) -> str | None:
    if benchmark == "clevrer":
        return CLEVRER_MESH_CONDITIONING_ROUTE
    normalized_scenario = str(scenario or "").strip().lower()
    if benchmark == "physion_pp" and normalized_scenario == "bouncy_wall_pp":
        return PHYSION_PP_ROLLBACK_MESH_CONDITIONING_ROUTE
    if (
        benchmark == "physion_pp"
        and normalized_scenario in PHYSION_PP_WARN_ONLY_MESH_CONDITIONING_SCENARIOS
    ):
        return PHYSION_PP_WARN_ONLY_MESH_CONDITIONING_ROUTE
    return None
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
PHYSION_PP_AGENT_FREE_GROUND_MOTION_ROUTE = (
    "ground.agent_airborne_fixture_ground"
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
PHYSION_PP_OVERLAP_GRAVITY_ESTIMATOR_ROUTE = (
    "gravity.overlap_obb_pre_snap"
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
def _overlap_gravity_route_profile(object_plan: ObjectPlan) -> Dict[str, Any]:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("gravity_estimator_route")
    if not isinstance(route_record, dict):
        raise ValueError("overlap gravity optimization is missing its estimator route")
    if route_record.get("decision_id") != GRAVITY_ESTIMATOR_DECISION_ID:
        raise ValueError(
            "gravity estimator route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    policy = default_module_profile_policy()
    profile = policy.route_profiles.get(route)
    if profile is None or profile.decision_id != GRAVITY_ESTIMATOR_DECISION_ID:
        raise ValueError(
            f"gravity estimator route is not an overlap profile: {route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("gravity estimator route is missing its context")
    benchmark = str(context.get("benchmark") or "").strip().lower()
    scenario = str(context.get("scenario") or "").strip().lower()
    if benchmark != profile.benchmark:
        raise ValueError(
            "overlap gravity route benchmark context mismatch: "
            f"{benchmark!r} != {profile.benchmark!r}"
        )
    allowed_scenarios = frozenset(profile.scenarios)
    if allowed_scenarios and scenario not in allowed_scenarios:
        raise ValueError(
            "overlap gravity route scenario context mismatch: "
            f"{scenario!r} not in {sorted(allowed_scenarios)!r}"
        )
    if not allowed_scenarios and scenario:
        raise ValueError(
            "CLEVRER overlap gravity route must not record a Physion++ scenario: "
            f"{scenario!r}"
        )
    resolved_profile = policy.resolve_route(
        GRAVITY_ESTIMATOR_DECISION_ID,
        route,
        benchmark=benchmark,
        scenario=scenario,
    )
    module = resolved_profile.module("overlap_gravity")
    if module.implementation != "sam3_projection_overlap_plane_normal_optimization":
        raise ValueError(
            "unsupported overlap-gravity module implementation: "
            f"{module.implementation!r}"
        )
    proxy = module.require_string("proxy")
    if proxy not in {"aabb", "obb"}:
        raise ValueError(f"unsupported overlap-gravity proxy: {proxy!r}")
    return {
        "route": route,
        "benchmark": resolved_profile.benchmark,
        "scenarios": allowed_scenarios,
        "pre_snap_scoring": module.require_boolean("pre_snap_scoring"),
        "use_obb_proxy": proxy == "obb",
        "static_render_reuse": module.require_boolean("static_render_reuse"),
        "pp_fine_refinement": module.require_boolean("pp_fine_refinement"),
        "pp_base_alignment": module.require_boolean("pp_base_alignment"),
        "winner_post_snap": module.require_boolean("winner_post_snap"),
        "scenario": scenario,
    }
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
PHYSION_PP_SWR_FIT_BACKEND_ROUTES = SWR_FIT_BACKEND_ROUTES - {
    "swr_backend.impulse_analytic"
}
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
PHYSIONPP_LATEST_DEBUG_RENDER_PROFILE = "physionpp_refined_6f3213_camera_only"
PHYSIONPP_SWR_BACKENDS = frozenset(
    {
        "swr_backend.surface_friction_sphere",
        "swr_backend.wall_bounce_sphere",
        "swr_backend.platform_bounce_sphere",
        "swr_backend.collision_friction_spheres",
        "swr_backend.collision_mass_spheres",
    }
)
PHYSION_YELLOW_PATCH_TRACK_PREFIX = "physion_yellow_patch_"
PHYSION_RAMP_TRACK_PREFIX = "physion_ramp_"
PHYSION_PP_STATIC_TRACK_PREFIX = "physion_pp_static_"
# Two-segment mesh reuse is dispatched by the resolved GEO-002 route record. Its
# scenarios and route choice live in configs/pipeline_route_policy.json instead of a
# second Python allowlist.
# Physion++ friction_collision: two-segment text-tracked scenario reconstructed the
# CLEVRER way. Every track (agent + patient in both segments) stays a dynamic
# FoundationPose object -- there are no physion_pp_static_ fixtures, so the similarity
# snap and the static flush machinery are inert by construction and every object takes
# the per-frame rigid ray slide onto the fitted ground plane (the scenario camera is
# steep enough that the slide is well-conditioned). Scenario-specific treatment on top:
# - patients (both segments) are straightened per frame (closest signed local axis to
#   up) regardless of geometry type; agents keep the raw FP rotation.
# - gravity comes from eight uniform GeoCalib frames in seg1, with the camera-x roll
#   component zeroed and renormalized; the SAM3 overlap gravity search is disabled.
# - the seg1 patient gets a flush-style in-plane search (translate + yaw + in-plane
#   stretch, height frozen, NO ground pin -- the ray slide already grounded it) and its
#   post-flush mesh is adopted by the seg2 patient (same physical object, same size).
# - the seg2 patient is motion-tested on mask centroid displacement; when it moves
#   (knocked off the ledge before the cut) its early frames are airborne, VDA depth on
#   it is untrustworthy, so it is ground-snap exempt and its trajectory is re-derived
#   by the drop refinement: first-frame centroid-ray anchor + (yaw, ray-depth) IoU
#   search on the straightened mesh, then a per-frame height-only IoU descent with the
#   ground-plane projection and yaw frozen (the fall is vertical by scenario prior).
PHYSION_PP_FRICTION_COLLISION_SCENARIOS = {"friction_collision_pp"}
PHYSION_PP_FC_DROP_MIN_MASK_PX = 20  # frames with smaller masks are left unrefined
PHYSION_PP_FC_DROP_DEPTH_COARSE = (0.60, 1.40, 17)  # first-frame ray-depth search: lo/hi factor, steps
PHYSION_PP_FC_DROP_DEPTH_EXPANDED = (0.05, 2.00)
PHYSION_PP_FC_DROP_DEPTH_BOUNDARY_STEPS = 2
PHYSION_PP_FC_DROP_DEPTH_FINE = (0.94, 1.06, 13)
PHYSION_PP_FC_DROP_YAW_COARSE_DEG = 15.0  # first-frame yaw grid over the full circle
PHYSION_PP_FC_DROP_YAW_FINE_DEG = (10.0, 5.0, 2.0)  # local yaw refinement offsets (+/- each)
PHYSION_PP_FC_DROP_HEIGHT_COARSE_M = (0.90, 0.12, 0.03)  # per-frame height window: down, up, step
PHYSION_PP_FC_DROP_HEIGHT_FINE_M = (0.03, 0.005)  # +/- span, step around the coarse best
PHYSION_PP_FC_DROP_DECIMATE_FACES = 400  # decimated mesh for the search; full mesh for reporting
# Mass-collision reconstruction uses per-frame ray slides for dynamic tracks. GeoCalib
# supplies segment-one gravity, segment-two agent/ball tracks reuse segment-one meshes,
# and a linked extra may provide the patient mesh. Patient positions come from mask rays
# intersecting the vertical motion plane; adopted meshes fit yaw and independent meshes
# fit scale. Degenerate motion planes are reported without a geometric fallback.
PHYSION_PP_MASS_COLLISION_SCENARIOS = {"mass_collision_pp"}
PHYSION_PP_MC_CONTACT_DILATE_PX = 2  # ball-mask dilation for the ball/agent contact test
PHYSION_PP_MC_CONTACT_MARGIN_FRAMES = 3  # agent scoring window ends this many frames before contact
PHYSION_PP_MC_PLANE_MIN_SPAN_M = 0.1  # ball-start<->agent ground-projection distance guard
PHYSION_PP_MC_DROP_MIN_MASK_PX = 20  # frames with smaller patient masks are left unrefined
PHYSION_PP_MC_DROP_MIN_FRAMES = 5  # minimum solvable frames for the drop refinement (seg2 is ~16)
PHYSION_PP_MC_DROP_MIN_RAY_PLANE_DOT = 0.15  # skip rays nearly parallel to the prior plane
PHYSION_PP_MC_DROP_YAW_COARSE_DEG = 15.0  # branch A: global yaw grid over the full circle
PHYSION_PP_MC_DROP_YAW_FINE_DEG = (10.0, 5.0, 2.0)  # branch A: local yaw refinement offsets (+/- each)
PHYSION_PP_MC_DROP_SCALE_COARSE = (0.60, 1.40, 17)  # branch B: global scale grid about the mesh centroid
PHYSION_PP_MC_DROP_SCALE_EXPANDED = (0.05, 2.00)
PHYSION_PP_MC_DROP_SCALE_BOUNDARY_STEPS = 2
PHYSION_PP_MC_DROP_SCALE_FINE = (0.94, 1.06, 13)
PHYSION_PP_MC_DROP_DECIMATE_FACES = 400  # decimated mesh for the search; full mesh for reporting
# Mass-collision ball trajectories are derived from masks. The pre-contact agent center is
# fixed; each segment searches its own ground-plane ball start, while both segments share
# ONE uniform ball scale. Those anchors define the directed ball->agent line, and per-frame
# centers come from the centroid-ray/line closest approach. Search scores use STRICTLY
# pre-contact frames; the contact frame and later frames are reserved for the deviation
# cut below. After the
# ball's mask clearly touches the agent's mask AND the centroid starts deviating from
# the fitted line (post-impact bounce/deflection), the track is CUT and the remaining
# poses removed -- the ball deliberately disappears rather than being tracked through
# an unmodeled deflection.
PHYSION_PP_MC_BALL_MIN_MASK_PX = 20  # frames with smaller ball masks are skipped
PHYSION_PP_MC_BALL_MIN_FIT_FRAMES = 3  # minimum pre-contact frames to fit the line
PHYSION_PP_MC_BALL_DEVIATION_PX = 4.0  # centroid-vs-line pixel residual that counts as deviation
PHYSION_PP_MC_BALL_DEVIATION_FRAMES = 2  # consecutive deviating frames before the cut
PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M = (0.20, 0.40, 0.80)
PHYSION_PP_MC_BALL_START_COARSE_STEP_M = 0.05
PHYSION_PP_MC_BALL_START_BOUNDARY_STEPS = 2
PHYSION_PP_MC_BALL_START_FINE_SPAN_M = 0.02
PHYSION_PP_MC_BALL_START_FINE_STEP_M = 0.01
PHYSION_PP_MC_BALL_SCALE_COARSE = (0.60, 1.40, 17)
PHYSION_PP_MC_BALL_SCALE_EXPANDED = (0.05, 2.00)
PHYSION_PP_MC_BALL_SCALE_BOUNDARY_STEPS = 2
PHYSION_PP_MC_BALL_SCALE_FINE = (0.94, 1.06, 13)
PHYSION_PP_MC_BALL_SCALE_TOPK = 3
PHYSION_PP_MC_BALL_SEARCH_MAX_WORKERS = 32
# Skip poorly conditioned centroid-ray/line intersections and interpolate them from
# neighboring solved frames.
PHYSION_PP_MC_BALL_MIN_RAY_LINE_SIN2 = 0.02
# mass_collision agent per-frame flush: the ray slide fixes the agent's depth via the
# ground constraint, but in-plane residuals from FP remain (the agent gets shoved and
# slides). Each frame independently searches ONLY the ground-plane translation (height
# and the raw FP rotation stay untouched) for the best rendered-mask IoU against that
# frame's SAM3 mask.
PHYSION_PP_MC_AGENT_FLUSH_MIN_MASK_PX = 20
PHYSION_PP_MC_AGENT_FLUSH_COARSE_M = (0.20, 0.05)  # +/- span, step per ground axis
PHYSION_PP_MC_AGENT_FLUSH_FINE_M = (0.02, 0.01)
PHYSION_PP_MC_AGENT_FLUSH_DECIMATE_FACES = 400
# Fit one in-plane agent stretch jointly over both segments, bake it about the mesh
# centroid, and rerun the per-frame translation flush on the stretched mesh.
PHYSION_PP_MC_AGENT_STRETCH_COARSE = (0.75, 1.25, 0.05)  # lo, hi, step per axis
PHYSION_PP_MC_AGENT_STRETCH_FINE = (0.04, 0.01)  # +/- span, step around the coarse best
PHYSION_PP_MC_AGENT_STRETCH_MAX_FRAMES = 48  # joint scoring frame cap across segments
PHYSION_PP_MC_AGENT_STRETCH_MIN_GAIN = 0.005  # bake only if the fit improves at least this
# Base plane for Physion++ static fixtures: the height below which this fraction of
# mesh vertices lie is taken as the resting base; vertices below it are shaved flat.
PHYSION_PP_STATIC_BASE_PERCENTILE = 2.0
# Base-plane alignment for Physion++ statics: plane fitted to the lowest vertex band,
# pose rotated so the base normal matches the up axis; larger tilts are left alone.
PHYSION_PP_STATIC_BASE_BAND_PERCENTILE = 10.0
PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG = 20.0
# Gravity-search scoring permits the full base-alignment angle range. The delivered
# pose is recomputed with PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG.
PHYSION_PP_SCORING_BASE_ALIGN_CAP_DEG = 180.0
# Physion++ statics may resolve ground-contact residuals through a camera-centered
# similarity rescale, which preserves the projected silhouette. Values outside the
# scale gate use the rigid ray slide.
PHYSION_PP_SIMILARITY_SNAP_SCALE_GATE = (0.5, 2.0)
# Physion++ agent ray/manifold trajectory refinement derives the trajectory from image
# evidence and the corrected static world:
# support surface rasterized to a height map, the ground-projected motion LINE fitted
# by direct 2-DOF search on all-frame bottom-pixel reprojection error, per-frame
# positions read off the line-on-surface curve, and the agent metric scale re-derived
# once from the contact-depth ratio.
PHYSION_PP_AGENT_RAY_CELL_M = 0.06
PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX = 8.0
PHYSION_PP_AGENT_RAY_TRIM_FRACTION = 0.7
PHYSION_PP_AGENT_RAY_MIN_MATCHED = 10
PHYSION_PP_AGENT_RAY_STATIC_DEPTH_TOL_M = 0.05
PHYSION_PP_AGENT_SPHERE_MIN_MASK_PX = 20
PHYSION_PP_AGENT_SPHERE_MIN_FRAMES = 5
PHYSION_PP_AGENT_SPHERE_FIT_MAX_FRAMES = 24
PHYSION_PP_AGENT_SPHERE_RADIUS_COARSE_STEPS = 13
PHYSION_PP_AGENT_SPHERE_RADIUS_FINE_STEPS = 9
PHYSION_PP_AGENT_SPHERE_RADIUS_BOUNDARY_STEPS = 2
PHYSION_PP_AGENT_SPHERE_RADIUS_MAX_EXPANSIONS = 4
PHYSION_PP_FP_SPHERE_PATH_LOCAL_SPAN_M = 0.30
PHYSION_PP_BOUNDARY_AREA_CONTINUATION_RATIO = 0.5


def _require_track_derived_inventory_route(
    object_plan: ObjectPlan,
) -> Dict[str, Any]:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("object_inventory_route")
    if not isinstance(route_record, dict):
        raise ValueError("object plan is missing its object-inventory route record")
    if route_record.get("decision_id") != OBJECT_INVENTORY_SOURCE_DECISION_ID:
        raise ValueError(
            "object-inventory route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != TRACK_BOOTSTRAP_THEN_TRACK_DERIVED_INVENTORY_ROUTE:
        raise ValueError(f"unsupported object-inventory route: {route!r}")
    return route_record


def _record_track_derived_inventory_route(
    *,
    object_plan: ObjectPlan,
    labels_payload: Dict[str, Any],
) -> Dict[str, Any]:
    route_record = _require_track_derived_inventory_route(object_plan)
    labels_payload["object_inventory_route"] = deepcopy(route_record)
    return route_record


def _require_mass_extra_patient_link_route(
    *,
    object_plan: ObjectPlan,
    tracks_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("mass_extra_patient_link_route")
    normalized_scenario = str(
        tracks_payload.get("physion_scenario") or ""
    ).strip().lower()
    is_target_scope = normalized_scenario == "mass_collision_pp"
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "Mass Collision object plan is missing its "
                "mass-extra-patient-link route record"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "mass-extra-patient-link route record is not applicable to this context: "
            f"scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != MASS_EXTRA_PATIENT_LINK_DECISION_ID:
        raise ValueError(
            "mass-extra-patient-link route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != MASS_EXTRA_PATIENT_LINK_ROUTE:
        raise ValueError(f"unsupported mass-extra-patient-link route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("mass-extra-patient-link route record is missing its context")
    if context.get("benchmark") != "physion_pp":
        raise ValueError(
            "mass-extra-patient-link route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != 'physion_pp'"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "mass-extra-patient-link route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


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


def _require_cross_segment_mesh_reuse_route(
    *,
    object_plan: ObjectPlan,
    tracks_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    temporal_record = special_scene.get("temporal_partition_route")
    route_record = special_scene.get("cross_segment_mesh_reuse_route")
    if not isinstance(temporal_record, dict):
        if isinstance(route_record, dict):
            raise ValueError(
                "cross-segment-mesh-reuse route record is missing its "
                "temporal-partition dependency"
            )
        return None
    if temporal_record.get("decision_id") != TEMPORAL_PARTITION_DECISION_ID:
        raise ValueError(
            "temporal-partition route record has an unexpected decision_id: "
            f"{temporal_record.get('decision_id')!r}"
        )
    temporal_route = temporal_record.get("route")
    if temporal_route not in {
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE,
        SPLIT_TEMPORAL_PARTITION_ROUTE,
    }:
        raise ValueError(f"unsupported temporal-partition route: {temporal_route!r}")
    temporal_context = temporal_record.get("context")
    if not isinstance(temporal_context, dict):
        raise ValueError("temporal-partition route record is missing its context")
    normalized_scenario = str(
        tracks_payload.get("physion_scenario") or ""
    ).strip().lower()
    recorded_temporal_scenario = str(
        temporal_context.get("scenario") or ""
    ).strip().lower()
    if recorded_temporal_scenario != normalized_scenario:
        raise ValueError(
            "temporal-partition route scenario context mismatch: "
            f"{recorded_temporal_scenario!r} != {normalized_scenario!r}"
        )
    is_target_scope = temporal_route == SPLIT_TEMPORAL_PARTITION_ROUTE
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "split-video object plan is missing its "
                "cross-segment-mesh-reuse route record"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "cross-segment-mesh-reuse route record is not applicable to "
            f"temporal route {temporal_route!r}"
        )
    if route_record.get("decision_id") != CROSS_SEGMENT_MESH_REUSE_DECISION_ID:
        raise ValueError(
            "cross-segment-mesh-reuse route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in CROSS_SEGMENT_MESH_REUSE_ROUTES:
        raise ValueError(f"unsupported cross-segment-mesh-reuse route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("cross-segment-mesh-reuse route record is missing its context")
    if context.get("benchmark") != "physion_pp":
        raise ValueError(
            "cross-segment-mesh-reuse route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != 'physion_pp'"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "cross-segment-mesh-reuse route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


def _require_track_vlm_labeling_route(
    *,
    policy_benchmark: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("track_vlm_labeling_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target labeling scope is missing its track-VLM-labeling route record: "
                f"benchmark={normalized_benchmark!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "track-VLM-labeling route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r}"
        )
    if route_record.get("decision_id") != TRACK_VLM_LABELING_DECISION_ID:
        raise ValueError(
            "track-VLM-labeling route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    expected_route = TRACK_VLM_LABELING_ROUTE_BY_BENCHMARK[
        normalized_benchmark
    ]
    if route != expected_route:
        raise ValueError(
            "track-VLM-labeling route does not match its benchmark: "
            f"{route!r} != {expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("track-VLM-labeling route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "track-VLM-labeling route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    return route_record


def _record_track_vlm_labeling_result(
    labels_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != TRACK_VLM_LABELING_DECISION_ID:
        raise ValueError(
            "track-VLM-labeling route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    context = route_record.get("context")
    benchmark = (
        str(context.get("benchmark") or "").strip().lower()
        if isinstance(context, dict)
        else ""
    )
    expected_route = TRACK_VLM_LABELING_ROUTE_BY_BENCHMARK.get(benchmark)
    if route != expected_route:
        raise ValueError(
            "track-VLM-labeling route does not match its recorded benchmark: "
            f"{route!r} != {expected_route!r}"
        )
    labels_payload["track_vlm_labeling_route"] = deepcopy(route_record)


def _route_policy_benchmark_for_scene(scene: Any) -> str | None:
    if isinstance(scene, PhysionPPScene):
        return "physion_pp"
    if isinstance(scene, ClevrerScene):
        return "clevrer"
    return None


def _require_false_positive_advice_effect_route(
    *,
    policy_benchmark: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("false_positive_advice_effect_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target labeling scope is missing its "
                "false-positive-advice-effect route record: "
                f"benchmark={normalized_benchmark!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "false-positive-advice-effect route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r}"
        )
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
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(
            "false-positive-advice-effect route record is missing its context"
        )
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "false-positive-advice-effect route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    return route_record


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


def _require_intrinsics_backend_route(
    *,
    policy_benchmark: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("intrinsics_backend_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target intrinsics scope is missing its intrinsics-backend route record: "
                f"benchmark={normalized_benchmark!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "intrinsics-backend route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r}"
        )
    if route_record.get("decision_id") != INTRINSICS_BACKEND_DECISION_ID:
        raise ValueError(
            "intrinsics-backend route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != INTRINSICS_BACKEND_ROUTE:
        raise ValueError(f"unsupported intrinsics-backend route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("intrinsics-backend route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "intrinsics-backend route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    return route_record


def _record_intrinsics_backend_result(
    intrinsics_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != INTRINSICS_BACKEND_DECISION_ID:
        raise ValueError(
            "intrinsics-backend route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != INTRINSICS_BACKEND_ROUTE:
        raise ValueError(f"unsupported intrinsics-backend route: {route!r}")
    intrinsics_payload["intrinsics_backend_route"] = deepcopy(route_record)


def _require_depth_partition_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("depth_partition_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target metric-depth scope is missing its depth-partition route record: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "depth-partition route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != DEPTH_PARTITION_DECISION_ID:
        raise ValueError(
            "depth-partition route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in DEPTH_PARTITION_ROUTES:
        raise ValueError(f"unsupported depth-partition route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("depth-partition route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "depth-partition route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "depth-partition route scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_depth_partition_result(
    depth_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != DEPTH_PARTITION_DECISION_ID:
        raise ValueError(
            "depth-partition route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in DEPTH_PARTITION_ROUTES:
        raise ValueError(f"unsupported depth-partition route: {route!r}")
    actual_partition = depth_payload.get("two_segment_depth_inference")
    if isinstance(actual_partition, dict) and isinstance(
        actual_partition.get("applied"),
        bool,
    ):
        expected_split = route == SPLIT_DEPTH_PARTITION_ROUTE
        if actual_partition["applied"] is not expected_split:
            raise ValueError(
                "video metric-depth artifact does not match its resolved "
                "depth-partition route: "
                f"route={route!r} applied={actual_partition['applied']!r}"
            )
    depth_payload["depth_partition_route"] = deepcopy(route_record)


def _require_segment_depth_alignment_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
    depth_partition_route: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("segment_depth_alignment_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    requires_alignment = (
        isinstance(depth_partition_route, dict)
        and depth_partition_route.get("route") == SPLIT_DEPTH_PARTITION_ROUTE
    )
    if not isinstance(route_record, dict):
        if requires_alignment:
            raise ValueError(
                "split metric-depth scope is missing its segment-depth-alignment "
                "route record: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not requires_alignment:
        raise ValueError(
            "segment-depth-alignment route record is not applicable to this "
            "depth-partition route"
        )
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
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(
            "segment-depth-alignment route record is missing its context"
        )
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "segment-depth-alignment route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "segment-depth-alignment route scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


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


def _record_video_metric_depth_route_results(
    depth_payload: Dict[str, Any],
    *,
    depth_partition_route: Dict[str, Any] | None,
    segment_depth_alignment_route: Dict[str, Any] | None,
) -> None:
    if depth_partition_route is not None:
        _record_depth_partition_result(depth_payload, depth_partition_route)
    if segment_depth_alignment_route is not None:
        _record_segment_depth_alignment_result(
            depth_payload,
            segment_depth_alignment_route,
        )


def _require_sam3d_observation_selection_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("sam3d_observation_selection_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target SAM3D scope is missing its observation-selection route "
                "record: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "SAM3D-observation-selection route record is not applicable to this "
            "context"
        )
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
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(
            "SAM3D-observation-selection route record is missing its context"
        )
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "SAM3D-observation-selection route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "SAM3D-observation-selection route scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


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


def _require_mesh_conditioning_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("mesh_conditioning_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target mesh-conditioning scope is missing its route record: "
                f"benchmark={normalized_benchmark!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "mesh-conditioning route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r}"
        )
    if route_record.get("decision_id") != MESH_CONDITIONING_DECISION_ID:
        raise ValueError(
            "mesh-conditioning route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    expected_route = _expected_mesh_conditioning_route(
        normalized_benchmark,
        scenario,
    )
    if route != expected_route:
        raise ValueError(
            "mesh-conditioning route does not match its benchmark/scenario: "
            f"{route!r} != {expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("mesh-conditioning route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "mesh-conditioning route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    recorded_scenario = str(context.get("scenario") or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower()
    if recorded_scenario != normalized_scenario:
        raise ValueError(
            "mesh-conditioning route scenario context mismatch: "
            f"{recorded_scenario!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_mesh_conditioning_result(
    mesh_payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != MESH_CONDITIONING_DECISION_ID:
        raise ValueError(
            "mesh-conditioning route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    context = route_record.get("context")
    benchmark = (
        str(context.get("benchmark") or "").strip().lower()
        if isinstance(context, dict)
        else ""
    )
    scenario = (
        str(context.get("scenario") or "").strip().lower()
        if isinstance(context, dict)
        else ""
    )
    expected_route = _expected_mesh_conditioning_route(benchmark, scenario)
    if route != expected_route:
        raise ValueError(
            "mesh-conditioning route does not match its recorded context: "
            f"{route!r} != {expected_route!r}"
        )
    mesh_payload["mesh_conditioning_route"] = deepcopy(route_record)


def _require_pose_input_and_foundationpose_route(
    *,
    policy_benchmark: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("pose_input_and_foundationpose_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target pose-input scope is missing its route record: "
                f"benchmark={normalized_benchmark!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "pose-input route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r}"
        )
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
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("pose-input route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "pose-input route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    return route_record


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


def _require_ground_motion_gate_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("ground_motion_gate_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target ground-motion scope is missing its route record: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "ground-motion route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != GROUND_MOTION_GATE_DECISION_ID:
        raise ValueError(
            "ground-motion route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in GROUND_MOTION_GATE_ROUTES:
        raise ValueError(f"unsupported ground-motion-gate route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("ground-motion route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "ground-motion route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "ground-motion route scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_ground_motion_gate_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != GROUND_MOTION_GATE_DECISION_ID:
        raise ValueError(
            "ground-motion route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in GROUND_MOTION_GATE_ROUTES:
        raise ValueError(f"unsupported ground-motion-gate route: {route!r}")
    payload["ground_motion_gate_route"] = deepcopy(route_record)


def _require_gravity_route_record(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
    key: str,
    decision_id: str,
    supported_routes: frozenset[str],
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get(key)
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                f"target gravity scope is missing its {key}: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            f"{key} is not applicable to this context: "
            f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != decision_id:
        raise ValueError(
            f"{key} has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in supported_routes:
        raise ValueError(f"unsupported route in {key}: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(f"{key} is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            f"{key} benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            f"{key} scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


def _require_gravity_route_pair(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    estimator = _require_gravity_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="gravity_estimator_route",
        decision_id=GRAVITY_ESTIMATOR_DECISION_ID,
        supported_routes=GRAVITY_ESTIMATOR_ROUTES,
    )
    constraints = _require_gravity_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="gravity_constraints_route",
        decision_id=GRAVITY_CONSTRAINTS_DECISION_ID,
        supported_routes=GRAVITY_CONSTRAINT_ROUTES,
    )
    if estimator is None or constraints is None:
        if estimator is not None or constraints is not None:
            raise ValueError("gravity estimator and constraints routes must be paired")
        return None, None
    pair = (str(estimator.get("route")), str(constraints.get("route")))
    if pair not in GRAVITY_ROUTE_PAIRS:
        raise ValueError(f"incompatible gravity route pair: {pair!r}")
    return estimator, constraints


def _record_gravity_route_results(
    payload: Dict[str, Any],
    *,
    estimator_route: Dict[str, Any],
    constraints_route: Dict[str, Any],
) -> None:
    pair = (str(estimator_route.get("route")), str(constraints_route.get("route")))
    if pair not in GRAVITY_ROUTE_PAIRS:
        raise ValueError(f"incompatible gravity route pair: {pair!r}")
    payload["gravity_estimator_route"] = deepcopy(estimator_route)
    payload["gravity_constraints_route"] = deepcopy(constraints_route)


def _require_rotation_policy_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("rotation_policy_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = normalized_benchmark in {"clevrer", "physion_pp"}
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target rotation-policy scope is missing its route record: "
                f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "rotation-policy route record is not applicable to this context: "
            f"benchmark={normalized_benchmark!r} scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != ROTATION_POLICY_DECISION_ID:
        raise ValueError(
            "rotation-policy route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != ROTATION_POLICY_ROUTE:
        raise ValueError(f"unsupported rotation-policy route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("rotation-policy route record is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "rotation-policy route benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "rotation-policy route scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    return route_record


def _record_rotation_policy_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    if route_record.get("decision_id") != ROTATION_POLICY_DECISION_ID:
        raise ValueError(
            "rotation-policy route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    if route_record.get("route") != ROTATION_POLICY_ROUTE:
        raise ValueError(
            f"unsupported rotation-policy route: {route_record.get('route')!r}"
        )
    payload["rotation_policy_route"] = deepcopy(route_record)


def _require_pose_route_record(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
    key: str,
    decision_id: str,
    supported_routes: frozenset[str],
    benchmark_default_routes: Dict[str, str] | None = None,
    physion_pp_routes_by_scenario: Dict[str, str] | None = None,
    role: str | None = None,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get(key)
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    default_routes = benchmark_default_routes or {}
    scenario_routes = physion_pp_routes_by_scenario or {}
    is_target_scope = (
        normalized_benchmark in default_routes
        or (
            normalized_benchmark == "physion_pp"
            and normalized_scenario in scenario_routes
        )
    )
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                f"target {decision_id} scope is missing its {key}: "
                f"benchmark={normalized_benchmark!r} "
                f"scenario={normalized_scenario!r}"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            f"{key} is not applicable to this context: "
            f"benchmark={normalized_benchmark!r} "
            f"scenario={normalized_scenario!r}"
        )
    if route_record.get("decision_id") != decision_id:
        raise ValueError(
            f"{key} has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in supported_routes:
        raise ValueError(f"unsupported route in {key}: {route!r}")
    expected_route = default_routes.get(normalized_benchmark)
    if expected_route is None and normalized_benchmark == "physion_pp":
        expected_route = scenario_routes.get(normalized_scenario or "")
    if route != expected_route:
        raise ValueError(
            f"{key} route/context mismatch: {route!r} != {expected_route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(f"{key} is missing its context")
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            f"{key} benchmark context mismatch: "
            f"{context.get('benchmark')!r} != {normalized_benchmark!r}"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            f"{key} scenario context mismatch: "
            f"{context.get('scenario')!r} != {normalized_scenario!r}"
        )
    if context.get("role") != role:
        raise ValueError(
            f"{key} role context mismatch: "
            f"{context.get('role')!r} != {role!r}"
        )
    return route_record


def _require_support_snap_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="support_snap_route",
        decision_id=SUPPORT_SNAP_DECISION_ID,
        supported_routes=SUPPORT_SNAP_ROUTES,
        benchmark_default_routes={"clevrer": CLEVRER_SUPPORT_SNAP_ROUTE},
        physion_pp_routes_by_scenario={
            "friction_platform_pp": FRICTION_PLATFORM_SUPPORT_SNAP_ROUTE,
            "bouncy_wall_pp": BOUNCY_WALL_SUPPORT_SNAP_ROUTE,
            "bouncy_platform_pp": BOUNCY_PLATFORM_SUPPORT_SNAP_ROUTE,
            "friction_collision_pp": FRICTION_COLLISION_SUPPORT_SNAP_ROUTE,
            "mass_collision_pp": MASS_COLLISION_SUPPORT_SNAP_ROUTE,
        },
    )


def _require_static_fixture_flush_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="static_fixture_flush_route",
        decision_id=STATIC_FIXTURE_FLUSH_DECISION_ID,
        supported_routes=STATIC_FIXTURE_FLUSH_ROUTES,
        physion_pp_routes_by_scenario={
            "friction_platform_pp": STATIC_FIXTURE_FLUSH_ROUTE,
            "bouncy_wall_pp": STATIC_FIXTURE_FLUSH_ROUTE,
            "bouncy_platform_pp": STATIC_FIXTURE_FLUSH_ROUTE,
            "friction_collision_pp": SEG1_PATIENT_FLUSH_ROUTE,
            "mass_collision_pp": LINKED_EXTRA_FLUSH_ROUTE,
        },
    )


def _require_line_layout_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="line_layout_route",
        decision_id=LINE_LAYOUT_DECISION_ID,
        supported_routes=LINE_LAYOUT_ROUTES,
        physion_pp_routes_by_scenario={
            "friction_platform_pp": ACTIVE_LINE_LAYOUT_ROUTE,
            "bouncy_platform_pp": ACTIVE_LINE_LAYOUT_ROUTE,
            "bouncy_wall_pp": SKIP_LINE_LAYOUT_ROUTE,
            "friction_collision_pp": SKIP_LINE_LAYOUT_ROUTE,
            "mass_collision_pp": SKIP_LINE_LAYOUT_ROUTE,
        },
    )


def _require_collision_patient_motion_gate_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="collision_patient_motion_gate_route",
        decision_id=COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
        supported_routes=frozenset({COLLISION_PATIENT_MOTION_GATE_ROUTE}),
        physion_pp_routes_by_scenario={
            item: COLLISION_PATIENT_MOTION_GATE_ROUTE
            for item in COLLISION_PATIENT_SCENARIOS
        },
        role="patient",
    )


def _require_collision_patient_drop_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="collision_patient_drop_route",
        decision_id=COLLISION_PATIENT_DROP_DECISION_ID,
        supported_routes=frozenset({COLLISION_PATIENT_DROP_ROUTE}),
        physion_pp_routes_by_scenario={
            item: COLLISION_PATIENT_DROP_ROUTE
            for item in COLLISION_PATIENT_SCENARIOS
        },
        role="patient",
    )


def _require_mass_extra_mesh_adopt_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="mass_extra_flush_and_mesh_adopt_route",
        decision_id=MASS_EXTRA_MESH_ADOPT_DECISION_ID,
        supported_routes=frozenset({MASS_EXTRA_MESH_ADOPT_ROUTE}),
        physion_pp_routes_by_scenario={
            "mass_collision_pp": MASS_EXTRA_MESH_ADOPT_ROUTE,
        },
        role="patient",
    )


def _require_mass_agent_flush_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="mass_agent_flush_route",
        decision_id=MASS_AGENT_FLUSH_DECISION_ID,
        supported_routes=frozenset({MASS_AGENT_FLUSH_ROUTE}),
        physion_pp_routes_by_scenario={
            "mass_collision_pp": MASS_AGENT_FLUSH_ROUTE,
        },
        role="agent",
    )


def _require_mass_ball_trajectory_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="mass_ball_precontact_trajectory_route",
        decision_id=MASS_BALL_TRAJECTORY_DECISION_ID,
        supported_routes=frozenset({MASS_BALL_TRAJECTORY_ROUTE}),
        physion_pp_routes_by_scenario={
            "mass_collision_pp": MASS_BALL_TRAJECTORY_ROUTE,
        },
        role="ball",
    )


def _require_swr_fit_backend_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="swr_fit_backend_route",
        decision_id=SWR_FIT_BACKEND_DECISION_ID,
        supported_routes=SWR_FIT_BACKEND_ROUTES,
        benchmark_default_routes={
            "clevrer": "swr_backend.impulse_analytic",
        },
        physion_pp_routes_by_scenario={
            "friction_platform_pp": "swr_backend.surface_friction_sphere",
            "bouncy_wall_pp": "swr_backend.wall_bounce_sphere",
            "bouncy_platform_pp": "swr_backend.platform_bounce_sphere",
            "friction_collision_pp": "swr_backend.collision_friction_spheres",
            "mass_collision_pp": "swr_backend.collision_mass_spheres",
        },
    )


def _record_swr_fit_backend_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    _record_pose_route_result(
        payload,
        route_record,
        key="swr_fit_backend_route",
        decision_id=SWR_FIT_BACKEND_DECISION_ID,
        supported_routes=SWR_FIT_BACKEND_ROUTES,
    )


def _require_swr_fit_strategy_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="swr_fit_strategy_route",
        decision_id=SWR_FIT_STRATEGY_DECISION_ID,
        supported_routes=SWR_FIT_STRATEGY_ROUTES,
        physion_pp_routes_by_scenario=SWR_FIT_STRATEGY_BY_SCENARIO,
    )


def _record_swr_fit_strategy_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    _record_pose_route_result(
        payload,
        route_record,
        key="swr_fit_strategy_route",
        decision_id=SWR_FIT_STRATEGY_DECISION_ID,
        supported_routes=SWR_FIT_STRATEGY_ROUTES,
    )


def _require_swr_fit_geometry_source_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    return _require_pose_route_record(
        policy_benchmark=policy_benchmark,
        scenario=scenario,
        object_plan=object_plan,
        key="swr_fit_geometry_source_route",
        decision_id=SWR_FIT_GEOMETRY_SOURCE_DECISION_ID,
        supported_routes=frozenset({SWR_FIT_GEOMETRY_SOURCE_ROUTE}),
        physion_pp_routes_by_scenario={
            scenario_name: SWR_FIT_GEOMETRY_SOURCE_ROUTE
            for scenario_name in SWR_FIT_STRATEGY_BY_SCENARIO
        },
    )


def _record_swr_fit_geometry_source_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    _record_pose_route_result(
        payload,
        route_record,
        key="swr_fit_geometry_source_route",
        decision_id=SWR_FIT_GEOMETRY_SOURCE_DECISION_ID,
        supported_routes=frozenset({SWR_FIT_GEOMETRY_SOURCE_ROUTE}),
    )


def _require_swr_visual_pose_preservation_route(
    *,
    policy_benchmark: str | None,
    scenario: str | None,
    object_plan: ObjectPlan,
) -> Dict[str, Any] | None:
    special_scene = (
        object_plan.special_scene
        if isinstance(object_plan.special_scene, dict)
        else {}
    )
    route_record = special_scene.get("swr_visual_pose_preservation_route")
    normalized_benchmark = str(policy_benchmark or "").strip().lower()
    normalized_scenario = str(scenario or "").strip().lower() or None
    is_target_scope = (
        normalized_benchmark == "physion_pp"
        and normalized_scenario in COLLISION_PATIENT_SCENARIOS
    )
    if not isinstance(route_record, dict):
        if is_target_scope:
            raise ValueError(
                "target SWR-004.visual_pose_preservation scope is missing "
                "its swr_visual_pose_preservation_route"
            )
        return None
    if not is_target_scope:
        raise ValueError(
            "swr_visual_pose_preservation_route is not applicable to "
            f"benchmark={normalized_benchmark!r} "
            f"scenario={normalized_scenario!r}"
        )
    if (
        route_record.get("decision_id")
        != SWR_VISUAL_POSE_PRESERVATION_DECISION_ID
    ):
        raise ValueError(
            "swr_visual_pose_preservation_route has an unexpected "
            f"decision_id: {route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in SWR_VISUAL_POSE_PRESERVATION_ROUTES:
        raise ValueError(
            f"unsupported SWR visual-pose-preservation route: {route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(
            "swr_visual_pose_preservation_route is missing its context"
        )
    if context.get("benchmark") != normalized_benchmark:
        raise ValueError(
            "SWR visual-pose-preservation benchmark context mismatch"
        )
    if context.get("scenario") != normalized_scenario:
        raise ValueError(
            "SWR visual-pose-preservation scenario context mismatch"
        )
    if route == "visual_pose.corrected_rotation":
        if (
            route_record.get("option_id")
            != "option.visual_pose.corrected_rotation"
        ):
            raise ValueError(
                "corrected-rotation visual pose preservation requires the "
                "approved ALT-SWR-001 option record"
            )
    return route_record


def _record_swr_visual_pose_preservation_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
) -> None:
    _record_pose_route_result(
        payload,
        route_record,
        key="swr_visual_pose_preservation_route",
        decision_id=SWR_VISUAL_POSE_PRESERVATION_DECISION_ID,
        supported_routes=SWR_VISUAL_POSE_PRESERVATION_ROUTES,
    )


def _record_pose_route_result(
    payload: Dict[str, Any],
    route_record: Dict[str, Any],
    *,
    key: str,
    decision_id: str,
    supported_routes: frozenset[str],
) -> None:
    if route_record.get("decision_id") != decision_id:
        raise ValueError(
            f"{key} has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in supported_routes:
        raise ValueError(f"unsupported route in {key}: {route!r}")
    payload[key] = deepcopy(route_record)


def _display_stage(tool_name: str) -> str:
    return TOOL_TO_DISPLAY_STAGE.get(tool_name, tool_name.replace("_", "-"))


def _verbose_tool_io() -> bool:
    return (os.getenv("PHYSMIND_LOG_TOOL_IO") or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_physion_edge_exempt_track(track_id: Any) -> bool:
    # Large static physics fixtures (yellow patient mat, Roll pink/purple ramp) may sit partly
    # off-frame; they are exempt from the "all masks touch the boundary" drop, unlike the
    # dynamic objects. Carried as a track-id prefix so every downstream site that only has the
    # track-id string (e.g. the keyframe boundary fallback) can recognise it.
    track_id = str(track_id or "")
    return track_id.startswith(PHYSION_YELLOW_PATCH_TRACK_PREFIX) or track_id.startswith(
        PHYSION_RAMP_TRACK_PREFIX
    )


def _is_physion_pp_scenario(tracks_payload: Any) -> bool:
    # Physion++ fixtures (ramps, mats, wedge/house structures) routinely sit partly off-frame
    # for the entire clip and carry no exemption prefix (tracks come from GDINO visual prompts,
    # treated uniformly regardless of prompt provenance). The edge-of-frame exemption is
    # therefore always on for *_pp scenarios, keyed off the payload's scenario metadata.
    if not isinstance(tracks_payload, dict):
        return False
    scenario = str(tracks_payload.get("physion_scenario") or "").strip().lower()
    return scenario.endswith("_pp")


def _physion_pp_boundary_area_continuation(
    candidates: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extend a partially border-touching track out from its fully visible span.

    Each side uses its own first border-touching mask as the area anchor and walks
    outwards without crossing frame gaps or an area below 50% of that anchor.
    Callers must gate this helper to Physion++ and to tracks with at least one
    non-boundary candidate.  All-boundary tracks retain their existing fallback.
    """

    def _frame(record: Dict[str, Any]) -> int:
        return int(record.get("frame_index", 0))

    def _area(record: Dict[str, Any]) -> float:
        try:
            return float(record.get("selected_area") or record.get("area") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    full_visible_indices = [
        index
        for index, record in enumerate(candidates)
        if record.get("touching_boundary") is not True
    ]
    if not candidates or not full_visible_indices:
        raise ValueError(
            "Physion++ boundary-area continuation requires a non-empty track "
            "with at least one non-boundary mask"
        )

    first_full_index = min(full_visible_indices)
    last_full_index = max(full_visible_indices)
    first_index = first_full_index
    last_index = last_full_index

    def _scan_side(*, direction: int, start_index: int) -> tuple[int, Dict[str, Any]]:
        anchor_index = start_index + direction
        side = "left" if direction < 0 else "right"
        diagnostics: Dict[str, Any] = {
            "side": side,
            "area_ratio_threshold": PHYSION_PP_BOUNDARY_AREA_CONTINUATION_RATIO,
            "anchor_frame_index": None,
            "anchor_area": None,
            "minimum_area": None,
            "accepted_frame_count": 0,
            "stop_reason": "track_boundary",
        }
        if anchor_index < 0 or anchor_index >= len(candidates):
            return start_index, diagnostics

        anchor = candidates[anchor_index]
        anchor_area = _area(anchor)
        diagnostics.update(
            {
                "anchor_frame_index": _frame(anchor),
                "anchor_area": anchor_area,
                "minimum_area": (
                    anchor_area * PHYSION_PP_BOUNDARY_AREA_CONTINUATION_RATIO
                ),
            }
        )
        if anchor.get("touching_boundary") is not True:
            diagnostics["stop_reason"] = "anchor_not_boundary_touching"
            return start_index, diagnostics
        if anchor_area <= 0.0:
            diagnostics["stop_reason"] = "invalid_anchor_area"
            return start_index, diagnostics

        minimum_area = anchor_area * PHYSION_PP_BOUNDARY_AREA_CONTINUATION_RATIO
        accepted_index = start_index
        previous_frame = _frame(candidates[start_index])
        index = anchor_index
        while 0 <= index < len(candidates):
            record = candidates[index]
            frame_index = _frame(record)
            expected_frame = previous_frame + direction
            if frame_index != expected_frame:
                diagnostics.update(
                    {
                        "stop_reason": "frame_gap",
                        "stop_frame_index": frame_index,
                        "expected_frame_index": expected_frame,
                    }
                )
                break
            if record.get("touching_boundary") is not True:
                diagnostics.update(
                    {
                        "stop_reason": "non_boundary_mask",
                        "stop_frame_index": frame_index,
                    }
                )
                break
            area = _area(record)
            if area <= minimum_area:
                diagnostics.update(
                    {
                        "stop_reason": "area_at_or_below_threshold",
                        "stop_frame_index": frame_index,
                        "stop_frame_area": area,
                    }
                )
                break
            accepted_index = index
            previous_frame = frame_index
            diagnostics["accepted_frame_count"] += 1
            index += direction
        else:
            diagnostics["stop_reason"] = "track_boundary"

        diagnostics["extended_to_frame_index"] = _frame(candidates[accepted_index])
        return accepted_index, diagnostics

    first_index, left = _scan_side(direction=-1, start_index=first_full_index)
    last_index, right = _scan_side(direction=1, start_index=last_full_index)
    first_full_frame = _frame(candidates[first_full_index])
    last_full_frame = _frame(candidates[last_full_index])
    first_frame = _frame(candidates[first_index])
    last_frame = _frame(candidates[last_index])
    return {
        "policy": "physion_pp_contiguous_boundary_masks_above_half_first_touch_area",
        "area_ratio_threshold": PHYSION_PP_BOUNDARY_AREA_CONTINUATION_RATIO,
        "applied": first_index != first_full_index or last_index != last_full_index,
        "full_visible_frame_range": [first_full_frame, last_full_frame],
        "extended_frame_range": [first_frame, last_frame],
        "added_frame_count": (
            int(left["accepted_frame_count"]) + int(right["accepted_frame_count"])
        ),
        "first_record": candidates[first_index],
        "last_record": candidates[last_index],
        "left": left,
        "right": right,
    }


def _is_physion_static_ground_fixture_track(track_id: Any) -> bool:
    # The Physion++ yellow patient mat never moves and lies flush on the floor. It gets a
    # dedicated static-fixture path (forced box geometry, single cross-frame-verified
    # registration, flush-to-ground correction) instead of per-frame FoundationPose
    # tracking, whose thin/low-texture failure mode makes the mat drift along the view ray.
    # Physion++ non-agent tracks (physion_pp_static_) join the same static treatment but
    # keep their own SAM3D geometry: only the yellow mat forces box (see
    # _is_physion_forced_box_fixture_track).
    track_id = str(track_id or "")
    return track_id.startswith(PHYSION_YELLOW_PATCH_TRACK_PREFIX) or track_id.startswith(
        PHYSION_PP_STATIC_TRACK_PREFIX
    )


def _is_physion_forced_box_fixture_track(track_id: Any) -> bool:
    # The thin yellow mat uses a box proxy; other static fixtures keep SAM3D geometry.
    return str(track_id or "").startswith(PHYSION_YELLOW_PATCH_TRACK_PREFIX)


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


def _analytic_sphere_mask(
    *,
    center_camera: np.ndarray,
    radius: float,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Exact pinhole silhouette of a camera-space sphere, cropped to its projected cube."""
    center = np.asarray(center_camera, dtype=np.float64).reshape(3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    radius = float(radius)
    height, width = int(image_shape[0]), int(image_shape[1])
    output = np.zeros((height, width), dtype=bool)
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0.0:
        return output
    if center[2] <= radius + 1e-6:
        return output
    corners = center + np.asarray(
        [
            [sx * radius, sy * radius, sz * radius]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    projected = corners @ intrinsic.T
    if np.any(projected[:, 2] <= 1e-6):
        return output
    uv = projected[:, :2] / projected[:, 2:3]
    x0 = max(0, int(np.floor(float(uv[:, 0].min()))) - 1)
    x1 = min(width, int(np.ceil(float(uv[:, 0].max()))) + 2)
    y0 = max(0, int(np.floor(float(uv[:, 1].min()))) - 1)
    y1 = min(height, int(np.ceil(float(uv[:, 1].max()))) + 2)
    if x0 >= x1 or y0 >= y1:
        return output
    ys, xs = np.ogrid[y0:y1, x0:x1]
    inverse_intrinsic = np.linalg.inv(intrinsic)
    dx = inverse_intrinsic[0, 0] * xs + inverse_intrinsic[0, 1] * ys + inverse_intrinsic[0, 2]
    dy = inverse_intrinsic[1, 0] * xs + inverse_intrinsic[1, 1] * ys + inverse_intrinsic[1, 2]
    dz = inverse_intrinsic[2, 0] * xs + inverse_intrinsic[2, 1] * ys + inverse_intrinsic[2, 2]
    ray_dot_center = dx * center[0] + dy * center[1] + dz * center[2]
    ray_norm_sq = dx * dx + dy * dy + dz * dz
    discriminant = (
        ray_dot_center * ray_dot_center
        - ray_norm_sq * float(center @ center)
        + radius * radius * ray_norm_sq
    )
    output[y0:y1, x0:x1] = (discriminant >= 0.0) & (ray_dot_center > 0.0)
    return output


def _sphere_radius_seed(
    *,
    centers: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
) -> tuple[float, float, Dict[str, Any]]:
    estimates = []
    one_pixel = []
    distances = []
    for center, mask, intrinsic in zip(centers, masks, intrinsics):
        center = np.asarray(center, dtype=np.float64).reshape(3)
        area = int(np.asarray(mask, dtype=bool).sum())
        K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
        distance = float(np.linalg.norm(center))
        focal = float(np.sqrt(abs(K[0, 0] * K[1, 1])))
        if area <= 0 or distance <= 0.0 or focal <= 0.0:
            continue
        pixel_radius = float(np.sqrt(area / np.pi))
        estimates.append(distance * float(np.sin(np.arctan(pixel_radius / focal))))
        one_pixel.append(distance * float(np.sin(np.arctan(1.0 / focal))))
        distances.append(distance)
    finite = [value for value in estimates if np.isfinite(value) and value > 0.0]
    if not finite:
        raise ValueError("sphere radius seed has no finite mask/center estimates")
    seed = float(np.median(finite))
    pixel_floor = max(1e-4, float(np.median(one_pixel)) if one_pixel else 1e-4)
    camera_cap = 0.8 * float(min(distances))
    return seed, pixel_floor, {
        "method": "median_mask_equivalent_angular_radius",
        "sample_count": len(finite),
        "estimate_p10_p90_m": [
            round(float(np.percentile(finite, 10)), 6),
            round(float(np.percentile(finite, 90)), 6),
        ],
    }


def _adaptive_sphere_radius_search(
    *,
    seed_radius: float,
    pixel_floor: float,
    camera_cap: float,
    objective,
) -> tuple[float, float, Dict[str, Any]]:
    seed = float(seed_radius)
    hard_lo = max(float(pixel_floor), 0.2 * seed)
    hard_hi = min(float(camera_cap), 3.0 * seed)
    if not np.isfinite(hard_lo) or not np.isfinite(hard_hi) or hard_hi <= hard_lo:
        raise ValueError(f"invalid adaptive sphere radius bounds [{hard_lo}, {hard_hi}]")
    lo = max(hard_lo, 0.6 * seed)
    hi = min(hard_hi, 1.4 * seed)
    if hi <= lo:
        lo, hi = hard_lo, hard_hi
    score_cache: Dict[float, float] = {}

    def score(radius: float) -> float:
        key = round(float(radius), 9)
        if key not in score_cache:
            value = float(objective(float(radius)))
            score_cache[key] = value if np.isfinite(value) else float("-inf")
        return score_cache[key]

    initial_range = [float(lo), float(hi)]
    expansions: list[str] = []
    winner = seed
    winner_score = float("-inf")
    for _ in range(PHYSION_PP_AGENT_SPHERE_RADIUS_MAX_EXPANSIONS + 1):
        values = np.linspace(lo, hi, PHYSION_PP_AGENT_SPHERE_RADIUS_COARSE_STEPS)
        scored = [(float(value), score(float(value))) for value in values]
        winner_index = int(np.argmax([entry[1] for entry in scored]))
        winner, winner_score = scored[winner_index]
        lower_boundary = winner_index < PHYSION_PP_AGENT_SPHERE_RADIUS_BOUNDARY_STEPS
        upper_boundary = winner_index >= (
            len(scored) - PHYSION_PP_AGENT_SPHERE_RADIUS_BOUNDARY_STEPS
        )
        changed = False
        if lower_boundary and lo > hard_lo + 1e-12:
            new_lo = max(hard_lo, lo / 1.5)
            if new_lo < lo - 1e-12:
                lo, changed = new_lo, True
                expansions.append("lower")
        if upper_boundary and hi < hard_hi - 1e-12:
            new_hi = min(hard_hi, hi * 1.5)
            if new_hi > hi + 1e-12:
                hi, changed = new_hi, True
                expansions.append("upper")
        if not changed:
            break
    coarse_step = (hi - lo) / max(PHYSION_PP_AGENT_SPHERE_RADIUS_COARSE_STEPS - 1, 1)
    fine_values = np.linspace(
        max(hard_lo, winner - coarse_step),
        min(hard_hi, winner + coarse_step),
        PHYSION_PP_AGENT_SPHERE_RADIUS_FINE_STEPS,
    )
    fine_scored = [(float(value), score(float(value))) for value in fine_values]
    winner, winner_score = max(fine_scored, key=lambda entry: entry[1])
    saturated = winner <= hard_lo + 1e-9 or winner >= hard_hi - 1e-9
    return float(winner), float(winner_score), {
        "seed_radius_m": round(seed, 6),
        "initial_range_m": [round(value, 6) for value in initial_range],
        "hard_range_m": [round(hard_lo, 6), round(hard_hi, 6)],
        "evaluated_range_m": [round(lo, 6), round(hi, 6)],
        "expansion_directions": expansions,
        "final_radius_m": round(float(winner), 6),
        "final_fit_iou": round(float(winner_score), 6),
        "search_saturated": bool(saturated),
        "candidate_count": len(score_cache),
    }


def _flush_refine_score(state: Dict[str, Any], pose: np.ndarray, scale_u: float, scale_v: float) -> float:
    """Mean rendered-vs-SAM3-mask IoU of one flush-refinement candidate.

    A static-fixture candidate has ONE pose for every eval frame, so with a constant
    intrinsic the triangle rasterization (the expensive part) runs once and only the
    cheap vectorized occlusion comparison repeats per frame; genuinely per-frame
    intrinsics fall back to rendering inside the frame loop."""
    scaling = (
        scale_u * state["outer_u"] + scale_v * state["outer_v"] + state["outer_support"]
    )
    candidate_vertices = (state["vertices"] - state["obb_center"]) @ scaling.T + state["obb_center"]
    vertices_camera = candidate_vertices @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)
    constant_intrinsic = state["constant_intrinsic"]
    rendered_mask = rendered_depth = None
    if constant_intrinsic is not None:
        rendered_mask, rendered_depth = render_mesh_depth(
            vertices_camera=vertices_camera,
            faces=state["faces"],
            intrinsic=constant_intrinsic,
            image_shape=state["image_shape"],
        )
    ious = []
    for frame_index, target_mask in state["target_masks"].items():
        if constant_intrinsic is None:
            rendered_mask, rendered_depth = render_mesh_depth(
                vertices_camera=vertices_camera,
                faces=state["faces"],
                intrinsic=state["intrinsic_by_frame"][frame_index],
                image_shape=state["image_shape"],
            )
        visible_mask = mask_occluded_mesh_mask(
            rendered_mask=rendered_mask,
            occluder_mask=state["occluder_by_frame"].get(frame_index),
            self_mask=target_mask,
        )
        iou = _bool_mask_iou(visible_mask, target_mask)
        if iou is not None:
            ious.append(float(iou))
    return float(np.mean(ious)) if ious else float("-inf")


def _visible_against_static_depth(
    *,
    points_camera: np.ndarray,
    projected_uv: np.ndarray,
    valid: np.ndarray,
    static_depth: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject 3D samples hidden behind the reconstructed static scene.

    Pixels without rendered static geometry remain eligible. A sample on the visible
    support surface is retained within ``tolerance_m``; a sample farther from the
    camera than the front-most static mesh at its projected pixel is occluded.
    """
    points = np.asarray(points_camera, dtype=np.float64)
    uv = np.asarray(projected_uv, dtype=np.float64)
    eligible = np.asarray(valid, dtype=bool).copy()
    depth = np.asarray(static_depth, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera must have shape (N, 3), got {points.shape}")
    if uv.shape != (len(points), 2):
        raise ValueError(f"projected_uv must have shape ({len(points)}, 2), got {uv.shape}")
    if eligible.shape != (len(points),):
        raise ValueError(f"valid must have shape ({len(points)},), got {eligible.shape}")
    if depth.ndim != 2:
        raise ValueError(f"static_depth must be 2D, got {depth.shape}")

    eligible &= np.isfinite(points[:, 2]) & np.isfinite(uv).all(axis=1)
    sample_indices = np.nonzero(eligible)[0]
    occluded = np.zeros(len(points), dtype=bool)
    if not len(sample_indices):
        return eligible, occluded

    xs = np.rint(uv[sample_indices, 0]).astype(np.int64)
    ys = np.rint(uv[sample_indices, 1]).astype(np.int64)
    height, width = depth.shape
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    bounded_indices = sample_indices[in_bounds]
    if not len(bounded_indices):
        return eligible, occluded

    static_z = depth[ys[in_bounds], xs[in_bounds]]
    has_static = np.isfinite(static_z) & (static_z > 0.0)
    hidden = has_static & (
        points[bounded_indices, 2] > static_z + float(tolerance_m)
    )
    occluded[bounded_indices[hidden]] = True
    return eligible & ~occluded, occluded


def _flush_refine_score_cuda(
    state: Dict[str, Any], pose: np.ndarray, scale_u: float, scale_v: float
) -> float:
    """CUDA-mask equivalent of ``_flush_refine_score`` for serial GPU searches."""
    scaling = (
        scale_u * state["outer_u"] + scale_v * state["outer_v"] + state["outer_support"]
    )
    candidate_vertices = (
        (state["vertices"] - state["obb_center"]) @ scaling.T + state["obb_center"]
    )
    vertices_camera = candidate_vertices @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)
    constant_intrinsic = state["constant_intrinsic"]
    rendered_mask = None
    if constant_intrinsic is not None:
        rendered_mask = render_mesh_mask_cuda(
            vertices_camera=vertices_camera,
            faces=state["faces"],
            intrinsic=constant_intrinsic,
            image_shape=state["image_shape"],
        )
    ious = []
    for frame_index, target_mask in state["target_masks"].items():
        if constant_intrinsic is None:
            rendered_mask = render_mesh_mask_cuda(
                vertices_camera=vertices_camera,
                faces=state["faces"],
                intrinsic=state["intrinsic_by_frame"][frame_index],
                image_shape=state["image_shape"],
            )
        visible_mask = mask_occluded_mesh_mask(
            rendered_mask=rendered_mask,
            occluder_mask=state["occluder_by_frame"].get(frame_index),
            self_mask=target_mask,
        )
        iou = _bool_mask_iou(visible_mask, target_mask)
        if iou is not None:
            ious.append(float(iou))
    return float(np.mean(ious)) if ious else float("-inf")


def _flush_refine_pool_init(state: Dict[str, Any]) -> None:
    _FLUSH_REFINE_SCORE_STATE.clear()
    _FLUSH_REFINE_SCORE_STATE.update(state)


def _flush_refine_pool_score(job: tuple) -> float:
    pose, scale_u, scale_v = job
    return _flush_refine_score(_FLUSH_REFINE_SCORE_STATE, pose, float(scale_u), float(scale_v))


# mass_collision agent flush search helpers. Per-frame translation candidates are
# mutually independent; production evaluates them in deterministic serial order while
# each expensive mask rasterization runs on the persistent nvdiffrast CUDA context.
_MC_AGENT_FLUSH_POOL_STATE: Dict[str, Any] = {}


def _mc_agent_flush_pool_init(state: Dict[str, Any]) -> None:
    _MC_AGENT_FLUSH_POOL_STATE.clear()
    _MC_AGENT_FLUSH_POOL_STATE.update(state)


def _mc_agent_pose_iou(
    state: Dict[str, Any],
    seg: str,
    frame_index: int,
    rotation: np.ndarray,
    translation: np.ndarray,
    verts: np.ndarray | None = None,
) -> float | None:
    seg_state = state["segs"][seg]
    vertices = seg_state["verts"] if verts is None else verts
    rendered = render_mesh_mask_cuda(
        vertices_camera=vertices @ rotation.T + translation,
        faces=seg_state["faces"],
        intrinsic=seg_state["intrinsics"][frame_index],
        image_shape=state["image_shape"],
    )
    rendered = np.asarray(rendered).astype(bool)
    target = np.asarray(seg_state["masks"][frame_index]).astype(bool)
    if rendered.shape != target.shape:
        return None
    union_count = int(np.logical_or(rendered, target).sum())
    if union_count <= 0:
        return None
    return float(int(np.logical_and(rendered, target).sum())) / float(union_count)


def _mc_agent_offset_search(objective, coarse: tuple, fine: tuple) -> tuple:
    base_iou = objective(0.0, 0.0)
    if base_iou is None:
        return None, None
    best = {"iou": float(base_iou), "du": 0.0, "dv": 0.0}
    span, step = coarse
    offsets = np.arange(-span, span + 1e-9, step)
    for du in offsets:
        for dv in offsets:
            if du == 0.0 and dv == 0.0:
                continue
            iou = objective(float(du), float(dv))
            if iou is not None and iou > best["iou"]:
                best = {"iou": float(iou), "du": float(du), "dv": float(dv)}
    span, step = fine
    # Freeze the fine window around the COARSE best before iterating: re-evaluating
    # the range from the mutating best would let the window creep beyond the
    # documented search radius.
    fine_du = np.arange(best["du"] - span, best["du"] + span + 1e-9, step)
    fine_dv = np.arange(best["dv"] - span, best["dv"] + span + 1e-9, step)
    for du in fine_du:
        for dv in fine_dv:
            iou = objective(float(du), float(dv))
            if iou is not None and iou > best["iou"]:
                best = {"iou": float(iou), "du": float(du), "dv": float(dv)}
    return best, float(base_iou)


def _mc_agent_frame_task(state: Dict[str, Any], seg: str, frame_index: int) -> tuple:
    seg_state = state["segs"][seg]
    matrix = seg_state["poses"][frame_index]
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    gu, gv = state["gu"], state["gv"]

    def objective(du: float, dv: float) -> float | None:
        return _mc_agent_pose_iou(state, seg, frame_index, rotation, translation + du * gu + dv * gv)

    best, base_iou = _mc_agent_offset_search(objective, state["coarse"], state["fine"])
    if base_iou is None:
        return (seg, frame_index, None, None, 0.0, 0.0)
    return (seg, frame_index, base_iou, best["iou"], best["du"], best["dv"])


def _mc_agent_flush_pool_frame(job: tuple) -> tuple:
    seg, frame_index = job
    return _mc_agent_frame_task(_MC_AGENT_FLUSH_POOL_STATE, str(seg), int(frame_index))


def _mc_agent_static_offset_task(state: Dict[str, Any], du: float, dv: float) -> float | None:
    static = state["static"]
    shifted = static["base_translation"] + float(du) * state["gu"] + float(dv) * state["gv"]
    ious = [
        value
        for value in (
            _mc_agent_pose_iou(
                state,
                "seg",
                frame_index,
                static["rotation"],
                shifted,
            )
            for frame_index in static["frames"]
        )
        if value is not None
    ]
    return float(np.mean(ious)) if ious else None


def _mc_agent_flush_pool_static_offset(job: tuple) -> float | None:
    du, dv = job
    return _mc_agent_static_offset_task(
        _MC_AGENT_FLUSH_POOL_STATE,
        float(du),
        float(dv),
    )


def _mc_agent_stretch_task(state: Dict[str, Any], su: float, sv: float) -> float | None:
    stretch = state["stretch"]
    du_local, dv_local = stretch["du_local"], stretch["dv_local"]
    ious = []
    for seg, frame_index in stretch["samples"]:
        seg_state = state["segs"][seg]
        verts = seg_state["verts"]
        centroid = seg_state["centroid"]
        rel = verts - centroid
        rel = (
            rel
            + (su - 1.0) * np.outer(rel @ du_local, du_local)
            + (sv - 1.0) * np.outer(rel @ dv_local, dv_local)
        )
        matrix = seg_state["poses"][frame_index]
        iou = _mc_agent_pose_iou(
            state, seg, frame_index, matrix[:3, :3], matrix[:3, 3], verts=centroid + rel
        )
        if iou is not None:
            ious.append(iou)
    return float(np.mean(ious)) if ious else None


def _mc_agent_flush_pool_stretch(job: tuple) -> float | None:
    su, sv = job
    return _mc_agent_stretch_task(_MC_AGENT_FLUSH_POOL_STATE, float(su), float(sv))


# mass_collision ball joint search: candidate starts are independent once the shared
# scale grid is fixed. A fork pool keeps the search state read-only and evaluates one
# (segment, start offset) against every requested scale, avoiding repeated ray-line
# solves and process scheduling per scale.
_MC_BALL_JOINT_POOL_STATE: Dict[str, Any] = {}


def _mc_ball_joint_pool_init(state: Dict[str, Any]) -> None:
    _MC_BALL_JOINT_POOL_STATE.clear()
    _MC_BALL_JOINT_POOL_STATE.update(state)


def _mc_ball_line_centers(
    state: Dict[str, Any],
    seg: str,
    start_center: np.ndarray,
    direction: np.ndarray,
    frames: Sequence[int],
) -> Dict[int, np.ndarray]:
    seg_state = state["segments"][seg]
    centers: Dict[int, np.ndarray] = {}
    start_frame = int(seg_state["start_frame"])
    for frame_index in frames:
        frame_index = int(frame_index)
        if frame_index == start_frame:
            centers[frame_index] = start_center
            continue
        ray = seg_state["rays"].get(frame_index)
        if ray is None:
            continue
        ray_dot_dir = float(ray @ direction)
        denom = 1.0 - ray_dot_dir * ray_dot_dir
        if denom < PHYSION_PP_MC_BALL_MIN_RAY_LINE_SIN2:
            continue
        t = (float(ray @ start_center) - float(direction @ start_center) * ray_dot_dir) / denom
        if t <= 1e-3:
            continue
        along = t * ray_dot_dir - float(direction @ start_center)
        centers[frame_index] = start_center + along * direction
    return centers


def _mc_ball_joint_candidate_scores(
    state: Dict[str, Any],
    seg: str,
    du: float,
    dv: float,
    scales: Sequence[float],
) -> tuple[str, float, float, list[tuple[float, Optional[float], int]]]:
    seg_state = state["segments"][seg]
    gu = state["gu"]
    gv = state["gv"]
    up_axis = state["up_axis"]
    start_center = (
        np.asarray(seg_state["base_start_center"], dtype=np.float64)
        + float(du) * gu
        + float(dv) * gv
    )
    agent_center = np.asarray(seg_state["agent_center"], dtype=np.float64)
    direction = agent_center - start_center
    direction = direction - float(direction @ up_axis) * up_axis
    span = float(np.linalg.norm(direction))
    if span < PHYSION_PP_MC_PLANE_MIN_SPAN_M:
        return str(seg), float(du), float(dv), [
            (float(scale), None, 0) for scale in scales
        ]
    direction = direction / span
    # fit_frames is constructed only after contact is found and contains frame < contact.
    fit_frames = seg_state["fit_frames"]
    centers = _mc_ball_line_centers(state, seg, start_center, direction, fit_frames)
    if len(centers) < PHYSION_PP_MC_BALL_MIN_FIT_FRAMES:
        return str(seg), float(du), float(dv), [
            (float(scale), None, len(centers)) for scale in scales
        ]

    target_masks = seg_state["target_masks"]
    intrinsics = seg_state["intrinsics"]
    image_height, image_width = state["image_shape"]
    scale_values = np.asarray(tuple(float(scale) for scale in scales), dtype=np.float64)
    radii = scale_values * float(state["base_radius"])
    max_radius = float(radii.max())
    iou_sums = np.zeros(len(scale_values), dtype=np.float64)
    valid_frames = 0

    # The conditioned mass-collision ball is an exact sphere (the canonical mesh has
    # constant vertex radius). Render its pinhole silhouette analytically: a pixel ray
    # belongs to the mask iff its ray/sphere quadratic has a non-negative discriminant.
    # This is the same full-resolution binary-mask IoU objective as triangle
    # rasterization, without looping over hundreds of sphere triangles for every scale.
    for frame_index, center in centers.items():
        center = np.asarray(center, dtype=np.float64)
        intrinsic = np.asarray(intrinsics[frame_index], dtype=np.float64)
        if center[2] <= max_radius + 1e-6:
            continue

        # Project a conservative cube around the largest searched sphere to bound the
        # exact conic. No image pixels outside this box can intersect the sphere.
        corners = center + np.asarray(
            [
                [sx * max_radius, sy * max_radius, sz * max_radius]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        projected = corners @ intrinsic.T
        uv = projected[:, :2] / projected[:, 2:3]
        x0 = max(0, int(np.floor(float(uv[:, 0].min()))) - 1)
        x1 = min(int(image_width), int(np.ceil(float(uv[:, 0].max()))) + 2)
        y0 = max(0, int(np.floor(float(uv[:, 1].min()))) - 1)
        y1 = min(int(image_height), int(np.ceil(float(uv[:, 1].max()))) + 2)
        if x0 >= x1 or y0 >= y1:
            continue

        ys, xs = np.ogrid[y0:y1, x0:x1]
        inverse_intrinsic = np.linalg.inv(intrinsic)
        dx = inverse_intrinsic[0, 0] * xs + inverse_intrinsic[0, 1] * ys + inverse_intrinsic[0, 2]
        dy = inverse_intrinsic[1, 0] * xs + inverse_intrinsic[1, 1] * ys + inverse_intrinsic[1, 2]
        dz = inverse_intrinsic[2, 0] * xs + inverse_intrinsic[2, 1] * ys + inverse_intrinsic[2, 2]
        ray_dot_center = dx * center[0] + dy * center[1] + dz * center[2]
        ray_norm_sq = dx * dx + dy * dy + dz * dz
        base_discriminant = ray_dot_center * ray_dot_center - ray_norm_sq * float(center @ center)
        discriminants = (
            base_discriminant[None, :, :]
            + radii[:, None, None] * radii[:, None, None] * ray_norm_sq[None, :, :]
        )
        rendered = (discriminants >= 0.0) & (ray_dot_center[None, :, :] > 0.0)
        target = np.asarray(target_masks[frame_index], dtype=bool)
        target_crop = target[y0:y1, x0:x1]
        rendered_area = rendered.sum(axis=(1, 2), dtype=np.int64)
        intersection = np.logical_and(rendered, target_crop[None, :, :]).sum(
            axis=(1, 2), dtype=np.int64
        )
        target_area = int(target.sum())
        union = target_area + rendered_area - intersection
        valid = union > 0
        iou_sums[valid] += intersection[valid] / union[valid]
        if bool(valid.all()):
            valid_frames += 1

    return str(seg), float(du), float(dv), [
        (
            float(scale),
            float(iou_sums[index] / valid_frames) if valid_frames else None,
            valid_frames,
        )
        for index, scale in enumerate(scale_values)
    ]


def _mc_ball_joint_pool_candidate(job: tuple) -> tuple:
    seg, du, dv, scales = job
    return _mc_ball_joint_candidate_scores(
        _MC_BALL_JOINT_POOL_STATE,
        str(seg),
        float(du),
        float(dv),
        tuple(float(value) for value in scales),
    )


def _mc_ball_offset_grid(span: float, step: float) -> list[tuple[float, float]]:
    values = np.arange(-float(span), float(span) + 1e-9, float(step))
    return [(round(float(du), 8), round(float(dv), 8)) for du in values for dv in values]


def _mc_ball_start_near_boundary(
    *, du: float, dv: float, span: float, step: float, boundary_steps: int
) -> bool:
    band = float(step) * int(boundary_steps)
    tolerance = max(abs(float(step)) * 1e-6, 1e-12)
    return (
        abs(float(du)) >= float(span) - band - tolerance
        or abs(float(dv)) >= float(span) - band - tolerance
    )


# Friction-platform fixtures and agent motion share one ground line. The static layout
# is re-parameterized as a
# shared ground line (angle theta, lateral offset c) + per-object along-line slide
# s_i + per-object stretch k_i PERPENDICULAR to the line (releases the deadlock where
# a wrong mesh width plus the center-on-line constraint makes every slide misfit the
# mask). Yaw locks each object's nearest in-plane OBB axis exactly parallel to the
# line. The joint objective is the CURVATURE-WEIGHTED IoU sum: w_i is the measured
# IoU curvature along the object's view-ray-parallel flat direction, so objects whose
# masks pin them hard, while flatter object objectives receive less weight. Heights stay
# flush-pinned; the along-line scale keeps the flush result. GATED to scenarios whose
# statics are collinear on the ground: friction_platform_pp (verified) and
# bouncy_platform_pp (static structure + mats share one ground line).
# Agent-ray trajectory refinement re-derives the agent position on a line-on-support-
# surface curve. This prior is enabled only for friction_platform_pp.
PHYSION_PP_AGENT_RAY_SCENARIOS = {"friction_platform_pp"}
# The bouncy-platform airborne agent's ground projection follows the static collinear line.
# statics (line-layout theta,c). Lifting that ground line to a VERTICAL plane P removes
# the single depth DOF the mask centroid leaves free, so each frame's agent-mask
# centroid ray has a UNIQUE intersection with P -> the 3D ball center, with NO depth
# input. The tiny agent is refit as a SPHERE (rotation-free) with one global radius that
# maximizes the rendered-mask IoU. FoundationPose is NEVER used as a reference (no FP-IoU
# gate): the collinear prior is trusted and the refinement always applies when a line is
# available. The line is taken from line-layout in full mode (theta,c); in two_static mode
# line-layout leaves the fixtures un-lined (c=None, only their per-object elongation theta),
# so the agent line is fit HERE the SAME way full mode does it: the PCA/connecting line
# THROUGH the static centers (exact for 2 points -- both lie on it), NOT the elongation theta
# (which need not pass through them, and is noisy for near-square fixtures).
PHYSION_PP_BOUNCE_MIN_MATCHED = 8
PHYSION_PP_BOUNCE_MIN_RAY_PLANE_DOT = 0.15  # skip near-degenerate rays (parallel to P)
PHYSION_PP_BOUNCE_FIT_MIN_AREA_PX = 30  # reliable frames for the global-radius fit
PHYSION_PP_BOUNCE_RADIUS_COARSE_M = (0.03, 0.30, 0.01)  # (start, stop, step)
PHYSION_PP_BOUNCE_RADIUS_EXPANDED_M = (0.01, 0.50)
PHYSION_PP_BOUNCE_RADIUS_BOUNDARY_STEPS = 2
PHYSION_PP_BOUNCE_RADIUS_FINE_STEP_M = 0.0025
# Physion++ bouncy_wall dynamic-object trajectory (mask-only position + orientation policy,
# zero depth for position). Position: the object ground path is perpendicular to the wall
# along its centerline, so lifting that line to a vertical plane P makes each frame's mask
# centroid ray meet P at a unique 3D center. Orientation always uses the FoundationPose
# register+track rotation. The seg1 mesh gets ONE global uniform scale fitted across
# prior-plane centers with FP rotations; seg2 reuses that fitted mesh exactly.
PHYSION_PP_BOUNCE_WALL_SCENARIOS = {"bouncy_wall_pp"}
PHYSION_PP_BW_TRAJ_MIN_MATCHED = 5
PHYSION_PP_BW_TRAJ_MIN_RAY_PLANE_DOT = 0.15
PHYSION_PP_BW_TRAJ_WALL_UP_DOT_MAX = 0.6  # wall thin-axis normal is ~horizontal (|n.up| below this)
PHYSION_PP_BW_TRAJ_DECIMATE_FACES = 400  # decimate mesh for internal IoU scoring only
PHYSION_PP_BW_TRAJ_SCALE_COARSE = (0.60, 1.40, 17)
PHYSION_PP_BW_TRAJ_SCALE_EXPANDED = (0.05, 2.00)
PHYSION_PP_BW_TRAJ_SCALE_BOUNDARY_STEPS = 2
PHYSION_PP_BW_TRAJ_SCALE_FINE = (0.94, 1.06, 13)
PHYSION_PP_BW_TRAJ_SCALE_MAX_FRAMES = 24
PHYSION_PP_LINE_THETA_SPAN_DEG = 8.0
PHYSION_PP_LINE_THETA_STEP_DEG = 2.0
PHYSION_PP_LINE_C_SPAN_M = 0.4
PHYSION_PP_LINE_C_STEP_M = 0.1
PHYSION_PP_LINE_S_SPAN_M = 1.2
PHYSION_PP_LINE_S_STEP_M = 0.15
PHYSION_PP_LINE_S_FINE_STEP_M = 0.035
PHYSION_PP_LINE_K_GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
PHYSION_PP_LINE_K_FINE = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
PHYSION_PP_LINE_K_CLIP = (0.6, 1.5)
PHYSION_PP_LINE_TOPK_FULL = 3
PHYSION_PP_LINE_CURV_DELTA_M = 0.3
PHYSION_PP_LINE_W_MIN = 0.02
PHYSION_PP_LINE_FAST_FRAME_COUNT = 3
# With exactly two statics, positional collinearity is underconstrained. Apply only the
# orientation prior: yaws snap mutually parallel to the
# yaw-curvature-weighted circular mean of the two axis angles (weighted so elongated,
# mask-sharp objects dominate the direction vote), centers never move, and the
# perpendicular stretch k stays (a sharp 1-D per-object fit at the fixed pose).
# Tiered guard per object: snap+stretch -> snap only -> full revert on IoU drop.
PHYSION_PP_LINE_TWO_STATIC_YAW_DELTA_DEG = 5.0
PHYSION_PP_LINE_TWO_STATIC_GUARD_EPS = 0.02

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
# Physion++ statics use a wider in-plane search after base alignment and ground pinning.
PHYSION_PP_STATIC_REFINE_ROUNDS: tuple[Dict[str, Any], ...] = (
    {
        "shift": tuple(np.arange(-0.6, 0.601, 0.15)),
        "yaw_deg": tuple(np.arange(-16.0, 16.01, 4.0)),
        "scale": tuple(np.arange(0.75, 1.301, 0.05)),
        "scale_mode": "absolute",
    },
    {
        "shift": tuple(np.arange(-0.15, 0.151, 0.05)),
        "yaw_deg": tuple(np.arange(-4.0, 4.01, 1.0)),
        "scale": tuple(np.arange(-0.04, 0.041, 0.01) + 1.0),
        "scale_mode": "relative",
    },
    {
        "shift": tuple(np.arange(-0.04, 0.041, 0.02)),
        "yaw_deg": tuple(np.arange(-1.0, 1.01, 0.5)),
        "scale": tuple(np.arange(-0.015, 0.0151, 0.005) + 1.0),
        "scale_mode": "relative",
    },
)
PHYSION_PP_STATIC_REFINE_BOUNDARY_STEPS = 2
PHYSION_PP_STATIC_REFINE_EXPANDED_LIMITS: Dict[str, tuple[float, float]] = {
    "shift": (-1.20, 1.20),
    "yaw_deg": (-48.0, 48.0),
    "scale": (0.05, 2.00),
}


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
            expected_policy, _agent_ids = self._physion_pp_sphere_agent_context(
                question_dir=question_dir,
                object_plan=object_plan,
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
            self._maybe_render_world_reconstruction_step_debug(
                world_reconstruction_path=artifact_path,
                payload=existing.payload,
                question_dir=question_dir,
            )
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
                self._maybe_render_world_reconstruction_step_debug(
                    world_reconstruction_path=artifact_path,
                    payload=payload,
                    question_dir=question_dir,
                )
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
            frame_range = self._geocalib_seg1_frame_range(
                question_dir=question_dir,
                object_plan=object_plan,
            )
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
        frame_range = self._geocalib_seg1_frame_range(
            question_dir=question_dir,
            object_plan=object_plan,
        )
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
        self._maybe_render_world_reconstruction_step_debug(
            world_reconstruction_path=artifact_path,
            payload=payload,
            question_dir=question_dir,
        )
        return ToolResult(
            tool_name=self.tool_name,
            status="ok",
            artifact_path=str(artifact_path),
            payload=payload,
        )

    def _write_pp_step_rgb_overlay(
        self,
        *,
        step_name: str,
        step_payload: Dict[str, Any],
        question_dir: Path,
        output_path: Path,
    ) -> None:
        """Physion++ debug-only: reproject the step's object states onto the original RGB
        at four sampled frames (first / one-third / two-thirds / last), tiled 2x2 into one
        image, so the moving agent's trajectory fit is reviewable, not just frame 0.
        Green contour = SAM3 mask; per-object tint/contour = rendered pose.
        Never raises: any failure is logged and swallowed (debug artifact only)."""
        try:
            import cv2

            tracks_payload = self.artifacts.read_optional(
                self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
            ) or {}
            if not _is_physion_pp_scenario(tracks_payload):
                return
            video_path = tracks_payload.get("video")
            if not video_path:
                return
            capture = cv2.VideoCapture(str(video_path))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                capture.release()
                return
            panel_frame_indices = sorted({
                0, total_frames // 3, (2 * total_frames) // 3, total_frames - 1
            })
            frames_bgr: Dict[int, np.ndarray] = {}
            for frame_index in panel_frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if ok:
                    frames_bgr[frame_index] = frame
            capture.release()
            if not frames_bgr:
                return

            if step_name == "pose_corrected":
                corrected = (step_payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
                objects = [item for item in corrected.get("objects", []) if isinstance(item, dict)]
                pose_keys = ("corrected_pose_4x4",)
                support_by_id = {
                    str(item.get("object_id")): item
                    for item in ((step_payload.get("support_plane_position_correction") or {}).get("objects") or [])
                    if isinstance(item, dict)
                }
            else:
                fp_payload = self.artifacts.read_optional(
                    self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
                ) or {}
                objects = [item for item in fp_payload.get("objects", []) if isinstance(item, dict)]
                pose_keys = ("raw_pose_4x4", "pose_4x4")
                support_by_id = {}

            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)

            palette = [(0, 0, 255), (255, 120, 0), (0, 220, 255), (255, 0, 255), (0, 255, 0)]
            geometry_cache: Dict[str, tuple] = {}
            panels = []
            for frame_index in panel_frame_indices:
                frame = frames_bgr.get(frame_index)
                if frame is None:
                    continue
                if frame.shape[:2] != image_shape:
                    frame = cv2.resize(frame, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
                overlay = frame.copy()
                for index, item in enumerate(objects):
                    object_id = str(item.get("object_id") or "")
                    pose_entry = next(
                        (
                            p for p in item.get("poses", [])
                            if isinstance(p, dict) and self._pose_frame_index(p) == frame_index
                        ),
                        None,
                    )
                    if pose_entry is None:
                        continue
                    pose = None
                    for key in pose_keys:
                        if pose_entry.get(key) is not None:
                            pose = np.asarray(pose_entry[key], dtype=np.float64).reshape(4, 4)
                            break
                    if pose is None:
                        continue
                    support_item = support_by_id.get(object_id)
                    mesh_path = (
                        support_item.get("mesh_path")
                        if isinstance(support_item, dict) and support_item.get("mesh_path")
                        else mesh_paths.get(object_id)
                    )
                    if not mesh_path:
                        continue
                    if mesh_path not in geometry_cache:
                        geometry_cache[mesh_path] = self._load_mesh_geometry(Path(mesh_path))
                    vertices, faces = geometry_cache[mesh_path]
                    own_record = (mask_records.get(object_id) or {}).get(frame_index) or {}
                    own_mask = self._resize_mask_to_shape(
                        mask_arrays.get(str(own_record.get("mask_key") or "")), image_shape
                    )
                    if own_mask is None:
                        own_mask = np.zeros(image_shape, dtype=bool)
                    rendered = render_mask_occluded_mesh(
                        vertices_camera=vertices @ pose[:3, :3].T + pose[:3, 3],
                        faces=faces,
                        intrinsic=self._intrinsic_for_frame(intrinsics, frame_index),
                        image_shape=image_shape,
                        occluder_mask=self._object_mask_union(
                            mask_records, mask_arrays, frame_index, image_shape
                        ),
                        self_mask=own_mask.astype(np.uint8),
                    )
                    visible = rendered["visible_mask"].astype(bool)
                    color = np.array(palette[index % len(palette)], dtype=np.float32)
                    overlay[visible] = (0.6 * overlay[visible].astype(np.float32) + 0.4 * color).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in color), 1)
                    sam_contours, _ = cv2.findContours(
                        own_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, sam_contours, -1, (0, 255, 0), 1)
                cv2.putText(
                    overlay, f"{step_name} f{frame_index}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                )
                panels.append(overlay)
            if not panels:
                return
            while len(panels) < 4:
                panels.append(np.zeros_like(panels[0]))
            grid = np.concatenate(
                [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:4], axis=1)], axis=0
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), grid)
            _log_tool(self.tool_name, f"debug_rgb_overlay step={step_name} artifact={output_path}")
        except Exception as exc:
            _log_tool(
                self.tool_name,
                f"debug_rgb_overlay failed step={step_name} error={_short_text(str(exc), 200)}",
            )

    def _write_pose_step_silhouette_grid(
        self,
        *,
        world_reconstruction_path: Path,
        payload: Dict[str, Any],
        question_dir: Path,
    ) -> None:
        """Pose-correction debug grid: rows = objects, columns = pipeline steps
        (fp_raw / rotation / support / flush). Each cell reprojects that step's mesh+pose
        at the object's best-visible frame over its SAM3 mask (red), drawing the
        occlusion-aware visible silhouette (green) and the full raw silhouette (yellow),
        with both IoUs (occ = what the pipeline scores, fair to occluded objects; raw =
        full projection, exposes over-enlargement the occlusion mask hides). Lets a
        reviewer localize which step/object degraded a fit. When a step has no pose for an
        object (e.g. a dynamic object skips the static-fixture steps) the cell falls back
        to the most recent available step and is tagged. Software-rasterized (no Blender),
        so it renders even when Blender is unavailable; never raises (debug artifact only)."""
        try:
            import cv2

            # Only Physion++ scenes populate the rotation/support/flush fixture steps.
            rotation = payload.get("rotation_correction") or {}
            support = payload.get("support_plane_position_correction") or {}
            flush = payload.get("static_fixture_flush_refinement") or {}
            if not (rotation.get("applied") or support.get("applied") or flush.get("applied")):
                return

            foundationpose = self.artifacts.read_optional(
                self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
            ) or {}
            fp_objects = {
                str(o.get("object_id")): o
                for o in foundationpose.get("objects", []) if isinstance(o, dict)
            }
            if not fp_objects:
                return
            rot_by_id = {str(o.get("object_id")): o for o in rotation.get("objects", []) if isinstance(o, dict)}
            sup_by_id = {str(o.get("object_id")): o for o in support.get("objects", []) if isinstance(o, dict)}
            flush_by_id = {str(o.get("object_id")): o for o in flush.get("objects", []) if isinstance(o, dict)}

            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            plan_payload = self.artifacts.read_optional(_scene_object_plan_path(question_dir)) or {}
            geometry_by_id = {
                str(o.get("object_id")): str(o.get("geometry_type") or "")
                for o in plan_payload.get("target_objects", []) if isinstance(o, dict)
            }

            steps = ["fp_raw", "rotation", "support", "flush"]
            # friction_collision / mass_collision: objects walk DIFFERENT steps (the
            # flush column only covers the seg1 patient / kept extras; the moving seg2
            # patient is rewritten afterwards by the drop refinement), so a fifth
            # "final" column shows the delivered corrected_trajectories pose and each
            # final cell is tagged with the action that actually produced it
            # (drop / flush / exempt / snap).
            pp_final_block = next(
                (
                    payload.get(block_key)
                    for block_key in ("physion_pp_friction_collision", "physion_pp_mass_collision")
                    if isinstance(payload.get(block_key), dict)
                    and payload[block_key].get("applies") is True
                ),
                None,
            )
            corrected_by_id: Dict[str, Dict[str, Any]] = {}
            final_tag_by_id: Dict[str, str] = {}
            if pp_final_block is not None:
                steps.append("final")
                corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
                corrected_by_id = {
                    str(o.get("object_id")): o
                    for o in corrected.get("objects", [])
                    if isinstance(o, dict)
                }
                flush_ok_ids = {
                    str(o.get("object_id"))
                    for o in (payload.get("static_fixture_flush_refinement") or {}).get("objects", [])
                    if isinstance(o, dict) and o.get("status") == "ok"
                }
                exempt_ids = {
                    str(v) for v in pp_final_block.get("ground_snap_exempt_object_ids") or []
                }
                for oid, obj in corrected_by_id.items():
                    if any(
                        isinstance(p, dict) and (p.get("fc_drop_refined") or p.get("mc_drop_refined"))
                        for p in obj.get("poses", [])
                    ):
                        final_tag_by_id[oid] = "drop"
                    elif oid in flush_ok_ids:
                        final_tag_by_id[oid] = "flush"
                    elif oid in exempt_ids:
                        final_tag_by_id[oid] = "exempt"
                    else:
                        final_tag_by_id[oid] = "snap"

            def pose_at(oid: str, step: str, frame: int) -> Optional[np.ndarray]:
                if step == "fp_raw":
                    for p in (fp_objects.get(oid) or {}).get("poses", []):
                        if isinstance(p, dict) and self._pose_frame_index(p) == frame:
                            for key in ("raw_pose_4x4", "pose_4x4"):
                                if p.get(key) is not None:
                                    return np.asarray(p[key], dtype=np.float64).reshape(4, 4)
                    return None
                if step in ("rotation", "support"):
                    item = (rot_by_id if step == "rotation" else sup_by_id).get(oid) or {}
                    for p in item.get("poses", []):
                        if (
                            isinstance(p, dict)
                            and self._pose_frame_index(p) == frame
                            and p.get("corrected_pose_4x4") is not None
                        ):
                            return np.asarray(p["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    return None
                if step == "flush":
                    item = flush_by_id.get(oid) or {}
                    if item.get("refined_pose_4x4") is not None:
                        return np.asarray(item["refined_pose_4x4"], dtype=np.float64).reshape(4, 4)
                if step == "final":
                    item = corrected_by_id.get(oid) or {}
                    for p in item.get("poses", []):
                        if (
                            isinstance(p, dict)
                            and self._pose_frame_index(p) == frame
                            and p.get("corrected_pose_4x4") is not None
                        ):
                            return np.asarray(p["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                return None

            def mesh_path_at(oid: str, step: str) -> Optional[str]:
                if step == "final" and (corrected_by_id.get(oid) or {}).get("mesh_path"):
                    return str(corrected_by_id[oid]["mesh_path"])
                if step in ("flush", "final") and (flush_by_id.get(oid) or {}).get("refined_mesh_path"):
                    return str(flush_by_id[oid]["refined_mesh_path"])
                if step in ("support", "final") and (sup_by_id.get(oid) or {}).get("mesh_path"):
                    return str(sup_by_id[oid]["mesh_path"])
                return mesh_paths.get(oid)

            fp_posed_frames: Dict[str, set] = {}
            for oid, item in fp_objects.items():
                posed = set()
                for p in item.get("poses", []):
                    if isinstance(p, dict) and (p.get("raw_pose_4x4") is not None or p.get("pose_4x4") is not None):
                        fi = self._pose_frame_index(p)
                        if fi is not None:
                            posed.add(fi)
                fp_posed_frames[oid] = posed

            # One representative frame per object: the largest-mask frame that also carries a
            # foundationpose pose (so a dynamic object whose mask extends a frame past its last
            # tracked pose still renders), falling back to the largest mask overall.
            rep_frame: Dict[str, int] = {}
            for oid in fp_objects:
                frames = mask_records.get(oid) or {}
                if not frames:
                    continue
                posed = fp_posed_frames.get(oid) or set()
                candidates = [f for f in frames if f in posed] or list(frames)
                rep_frame[oid] = max(candidates, key=lambda f: float((frames[f] or {}).get("area") or 0.0))
            ordered_ids = [oid for oid in sorted(fp_objects) if oid in rep_frame]
            if not ordered_ids:
                return

            tracks_payload = self.artifacts.read_optional(
                self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
            ) or {}
            video_path = tracks_payload.get("video")
            rgb_by_frame: Dict[int, np.ndarray] = {}
            if video_path:
                capture = cv2.VideoCapture(str(video_path))
                for fr in sorted({rep_frame[oid] for oid in ordered_ids}):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
                    ok, frame = capture.read()
                    if ok:
                        if frame.shape[:2] != image_shape:
                            frame = cv2.resize(frame, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
                        rgb_by_frame[int(fr)] = frame
                capture.release()

            cell_h, cell_w, label_w = 176, 176, 110
            kernel = np.ones((2, 2), np.uint8)
            geometry_cache: Dict[str, tuple] = {}

            def geom(mesh_path: str) -> tuple:
                if mesh_path not in geometry_cache:
                    geometry_cache[mesh_path] = self._load_mesh_geometry(Path(mesh_path))
                return geometry_cache[mesh_path]

            rows = []
            for oid in ordered_ids:
                frame = rep_frame[oid]
                own_record = (mask_records.get(oid) or {}).get(frame) or {}
                target = self._resize_mask_to_shape(mask_arrays.get(str(own_record.get("mask_key") or "")), image_shape)
                if target is None:
                    continue
                base = rgb_by_frame.get(int(frame))
                if base is None:
                    base = np.full((*image_shape, 3), 70, dtype=np.uint8)
                intrinsic = self._intrinsic_for_frame(intrinsics, frame)
                depth = self._depth_for_frame(metric_depth, frame)

                label = np.full((cell_h, label_w, 3), 40, dtype=np.uint8)
                cv2.putText(label, oid, (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                geo = geometry_by_id.get(oid, "")
                if geo:
                    cv2.putText(label, geo[:14], (5, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
                cv2.putText(label, f"f{int(frame)}", (5, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
                cells = [label]

                for step in steps:
                    canvas = (0.5 * base).astype(np.uint8)
                    canvas[target] = (
                        0.4 * canvas[target].astype(np.float32) + np.array([0, 0, 150], np.float32)
                    ).astype(np.uint8)
                    used = step
                    pose = pose_at(oid, step, frame)
                    if pose is None:  # fall back to the most recent available step
                        for prev in steps[: steps.index(step)][::-1]:
                            pose = pose_at(oid, prev, frame)
                            if pose is not None:
                                used = prev
                                break
                    mocc_txt, raw_txt = "-", "-"
                    if pose is not None:
                        vertices, faces = geom(mesh_path_at(oid, used))
                        rendered = render_mask_occluded_mesh(
                            vertices_camera=vertices @ pose[:3, :3].T + pose[:3, 3],
                            faces=faces,
                            intrinsic=intrinsic,
                            image_shape=image_shape,
                            occluder_mask=self._object_mask_union(
                                mask_records, mask_arrays, frame, image_shape
                            ),
                            self_mask=target.astype(np.uint8),
                        )
                        visible = rendered["visible_mask"].astype(np.uint8)
                        raw = rendered["rendered_mask"].astype(np.uint8)
                        mocc = self._mask_iou(visible.astype(bool), target)
                        raw_iou = self._mask_iou(raw.astype(bool), target)
                        mocc_txt = f"{mocc:.2f}" if mocc is not None else "-"
                        raw_txt = f"{raw_iou:.2f}" if raw_iou is not None else "-"
                        canvas[(raw - cv2.erode(raw, kernel)) > 0] = (0, 255, 255)  # raw silhouette (yellow)
                        canvas[(visible - cv2.erode(visible, kernel)) > 0] = (0, 255, 0)  # visible (green)
                    canvas = cv2.resize(canvas, (cell_w, cell_h))
                    cv2.putText(canvas, step, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    if step == "final" and final_tag_by_id.get(oid):
                        cv2.putText(
                            canvas, final_tag_by_id[oid], (4, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 180), 1,
                        )
                    cv2.putText(
                        canvas, f"mocc {mocc_txt} raw {raw_txt}", (4, 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                    )
                    if used != step:
                        cv2.putText(
                            canvas, f"(fb:{used})", (108, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1,
                        )
                    cells.append(canvas)
                rows.append(cv2.hconcat(cells))

            if not rows:
                return
            grid = cv2.vconcat(rows)
            debug_root = world_reconstruction_path.parent / "debug_render_steps"
            debug_root.mkdir(parents=True, exist_ok=True)
            output_path = debug_root / "per_object_step_silhouette_grid.png"
            cv2.imwrite(str(output_path), grid)
            _log_tool(self.tool_name, f"pose_step_silhouette_grid artifact={output_path}")
        except Exception as exc:
            _log_tool(
                self.tool_name,
                f"pose_step_silhouette_grid failed error={_short_text(str(exc), 200)}",
            )

    def _maybe_render_world_reconstruction_step_debug(
        self,
        *,
        world_reconstruction_path: Path,
        payload: Dict[str, Any],
        question_dir: Path,
    ) -> None:
        if not self.artifacts.debug_artifacts:
            return
        self._write_pose_step_silhouette_grid(
            world_reconstruction_path=world_reconstruction_path,
            payload=payload,
            question_dir=question_dir,
        )
        command = os.getenv("PHYSMIND_BLENDER_CMD") or DEFAULT_BLENDER_CMD
        debug_artifacts = dict(payload.get("debug_artifacts") or {})
        step_debug = dict(debug_artifacts.get("world_reconstruction_step_debug_renders") or {})
        if not command:
            step_debug["status"] = "tool_not_configured"
            step_debug["reason"] = "PHYSMIND_BLENDER_CMD is not set"
            debug_artifacts["world_reconstruction_step_debug_renders"] = step_debug
            payload["debug_artifacts"] = debug_artifacts
            self.artifacts.write(world_reconstruction_path, payload)
            return

        debug_root = world_reconstruction_path.parent / "debug_render_steps"
        step_results: Dict[str, Any] = {}
        # The per-step Blender renders are independent subprocesses writing to disjoint
        # step directories, so pending steps are prepared serially (skip/reuse checks,
        # pp overlay, step-input write) and then rendered concurrently. Each render's
        # inputs, command, and outputs are unchanged — only the wall-clock overlaps.
        pending_renders: list[Dict[str, Any]] = []
        step_order: list[str] = []
        for step_name, source in self._world_reconstruction_step_debug_sources(payload=payload, question_dir=question_dir):
            step_order.append(step_name)
            step_dir = debug_root / step_name
            step_input = step_dir / "world_reconstruction_step_input.json"
            output_video = step_dir / "world_reconstruction_debug.mp4"
            source_camera_video = step_dir / "world_reconstruction_debug_camera.mp4"
            output_json = step_dir / "world_reconstruction_debug.json"
            if source.get("applied") is not True:
                step_results[step_name] = {
                    "status": "skipped",
                    "reason": source.get("reason") or "source step is not applied",
                    "step_input": str(step_input),
                }
                continue

            step_payload = self._world_reconstruction_step_input_payload(
                payload=payload,
                question_dir=question_dir,
                step_name=step_name,
                source=source,
            )
            # Physion++ human-review artifact: the step's world state reprojected onto the
            # original RGB first frame (own rasterizer, independent of the Blender render).
            self._write_pp_step_rgb_overlay(
                step_name=step_name,
                step_payload=step_payload,
                question_dir=question_dir,
                output_path=step_dir / "world_reconstruction_rgb_overlay_frames.png",
            )
            step_hash = _stable_json_hash(step_payload)
            existing_payload = self.artifacts.read_optional(output_json) if output_json.exists() else None
            if (
                source_camera_video.exists()
                and output_json.exists()
                and isinstance(existing_payload, dict)
                and existing_payload.get("status") == "ok"
                and (existing_payload.get("world_reconstruction_step") or {}).get("step_input_hash") == step_hash
            ):
                step_results[step_name] = {
                    "status": "existing",
                    "source_camera_video": str(source_camera_video),
                    "output_json": str(output_json),
                    "step_input": str(step_input),
                }
                continue

            step_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts.write(step_input, step_payload)
            pending_renders.append(
                {
                    "step_name": step_name,
                    "step_hash": step_hash,
                    "step_input": step_input,
                    "output_video": output_video,
                    "source_camera_video": source_camera_video,
                    "output_json": output_json,
                    "pose_field": source.get("pose_field"),
                }
            )

        render_results: Dict[str, Dict[str, Any]] = {}
        if pending_renders:
            env = _external_tool_env(question_dir, True)
            for pending in pending_renders:
                _log_tool(
                    self.tool_name,
                    f"debug_step_render start step={pending['step_name']} output={pending['output_video']}",
                )
            with ThreadPoolExecutor(max_workers=len(pending_renders)) as pool:
                futures = {
                    pending["step_name"]: pool.submit(
                        run_world_reconstruction_debug_render,
                        command=command,
                        render_input_path=pending["step_input"],
                        output_video=pending["output_video"],
                        output_json=pending["output_json"],
                        env=env,
                    )
                    for pending in pending_renders
                }
                for pending in pending_renders:
                    render_results[pending["step_name"]] = futures[pending["step_name"]].result()

        for pending in pending_renders:
            step_name = pending["step_name"]
            step_hash = pending["step_hash"]
            step_input = pending["step_input"]
            source_camera_video = pending["source_camera_video"]
            output_json = pending["output_json"]
            render_result = render_results[step_name]
            elapsed = float(render_result.get("elapsed_sec") or 0.0)
            _log_tool(
                self.tool_name,
                f"debug_step_render end step={step_name} returncode={render_result.get('returncode')} elapsed={elapsed:.1f}s",
            )
            stdout = _short_text(str(render_result.get("stdout") or ""))
            stderr = _short_text(str(render_result.get("stderr") or ""))
            if stdout:
                _log_tool(self.tool_name, f"debug_step_render stdout step={step_name} {stdout}")
            if stderr:
                _log_tool(self.tool_name, f"debug_step_render stderr step={step_name} {stderr}")

            if render_result.get("status") == "ok" and source_camera_video.exists() and output_json.exists():
                render_payload = self.artifacts.read_optional(output_json) or {}
                render_payload["world_reconstruction_step"] = {
                    "step_name": step_name,
                    "step_input": str(step_input),
                    "step_input_hash": step_hash,
                    "source": self._world_reconstruction_step_pose_source_name(step_name),
                    "pose_field": pending["pose_field"],
                }
                self.artifacts.write(output_json, render_payload)
                step_results[step_name] = {
                    "status": "ok",
                    "source_camera_video": str(source_camera_video),
                    "output_json": str(output_json),
                    "step_input": str(step_input),
                    "elapsed_sec": elapsed,
                }
            else:
                step_results[step_name] = {
                    "status": "tool_error",
                    "source_camera_video": str(source_camera_video),
                    "output_json": str(output_json),
                    "step_input": str(step_input),
                    "elapsed_sec": elapsed,
                    "message": str(render_result.get("message") or render_result.get("stderr") or render_result.get("stdout") or "").strip(),
                }

        # Rebuild in source-step order: skipped/existing entries land in step_results
        # during the prepare pass and rendered ones during the collect pass, so a mixed
        # outcome would otherwise reorder the serialized steps mapping.
        step_results = {
            step_name: step_results[step_name] for step_name in step_order if step_name in step_results
        }
        debug_artifacts["world_reconstruction_step_debug_renders"] = {
            "status": "ok" if all(item.get("status") in {"ok", "existing", "skipped"} for item in step_results.values()) else "partial",
            "debug_root": str(debug_root),
            "steps": step_results,
        }
        payload["debug_artifacts"] = debug_artifacts
        self.artifacts.write(world_reconstruction_path, payload)

    def _world_reconstruction_step_debug_sources(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
    ) -> list[tuple[str, Dict[str, Any]]]:
        return [
            ("foundationpose_raw", self._foundationpose_trajectory_source(question_dir)),
            (
                "pose_corrected",
                (
                    payload.get("trajectory_correction", {}).get("corrected_trajectories")
                    if isinstance(payload.get("trajectory_correction"), dict)
                    else {}
                ),
            ),
        ]

    def _world_reconstruction_step_input_payload(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        step_name: str,
        source: Dict[str, Any],
    ) -> Dict[str, Any]:
        step_payload = dict(payload)
        video_metric_depth = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "video_metric_depth.json")
        )
        if isinstance(video_metric_depth, dict) and isinstance(video_metric_depth.get("video_metadata"), dict):
            step_payload["video_metadata"] = dict(video_metric_depth["video_metadata"])
        step_payload["question_dir"] = str(question_dir)
        try:
            step_payload["debug_mesh_paths"] = dict(self._foundationpose_mesh_paths(question_dir))
        except Exception:
            step_payload["debug_mesh_paths"] = {}
        step_payload["world_reconstruction_step_debug"] = {
            "enabled": True,
            "step_name": step_name,
            "source": self._world_reconstruction_step_pose_source_name(step_name),
            "pose_field": source.get("pose_field"),
        }
        step_payload["trajectory_correction"] = {
            **(payload.get("trajectory_correction") if isinstance(payload.get("trajectory_correction"), dict) else {}),
            "corrected_trajectories": self._standard_corrected_trajectory_source(
                source=source,
                source_name=self._world_reconstruction_step_pose_source_name(step_name),
            ),
        }
        return step_payload

    def _world_reconstruction_step_pose_source_name(self, step_name: str) -> str:
        if step_name == "foundationpose_raw":
            return "foundationpose.pose_4x4"
        if step_name == "pose_corrected":
            return "support_plane_position_correction.corrected_pose_4x4"
        return step_name

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

    def _pose_route_record(
        self,
        object_plan: ObjectPlan,
        *,
        key: str,
        decision_id: str,
        supported_routes: frozenset[str],
    ) -> Optional[Dict[str, Any]]:
        special_scene = (
            object_plan.special_scene
            if isinstance(object_plan.special_scene, dict)
            else {}
        )
        route_record = special_scene.get(key)
        if not isinstance(route_record, dict):
            return None
        if route_record.get("decision_id") != decision_id:
            raise ValueError(
                f"{key} has an unexpected decision_id: "
                f"{route_record.get('decision_id')!r}"
            )
        route = route_record.get("route")
        if route not in supported_routes:
            raise ValueError(f"unsupported route in {key}: {route!r}")
        return route_record

    def _module_for_route_record(
        self,
        resolved_route: Dict[str, Any],
        *,
        decision_id: str,
        module_name: str,
    ):
        if resolved_route.get("decision_id") != decision_id:
            raise ValueError(
                f"{module_name} route has an unexpected decision_id: "
                f"{resolved_route.get('decision_id')!r}"
            )
        context = resolved_route.get("context")
        if not isinstance(context, dict):
            raise ValueError(f"{module_name} route is missing its context")
        profile = default_module_profile_policy().resolve_route(
            decision_id,
            str(resolved_route.get("route") or ""),
            benchmark=str(context.get("benchmark") or "").strip().lower(),
            scenario=str(context.get("scenario") or "").strip().lower(),
        )
        return profile.module(module_name)

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

    def _geocalib_seg1_frame_range(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> Optional[tuple[int, int]]:
        if not self._geocalib_gravity_enabled(object_plan):
            return None
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        physion_tracking = tracks_payload.get("physion_tracking") or {}
        two_segment = physion_tracking.get("two_segment") or {}
        seg1 = two_segment.get("seg1") if isinstance(two_segment, dict) else None
        if not isinstance(seg1, (list, tuple)) or len(seg1) != 2:
            raise ValueError("collision GeoCalib gravity requires two_segment.seg1 bounds")
        frame_start, frame_end = int(seg1[0]), int(seg1[1])
        if frame_start < 0 or frame_start > frame_end:
            raise ValueError(f"invalid collision seg1 bounds: [{frame_start}, {frame_end}]")
        return frame_start, frame_end

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

    def _collision_patient_motion_parameters(
        self,
        resolved_route: Dict[str, Any],
    ) -> tuple[float, int]:
        if (
            resolved_route.get("decision_id")
            != COLLISION_PATIENT_MOTION_GATE_DECISION_ID
            or resolved_route.get("route") != COLLISION_PATIENT_MOTION_GATE_ROUTE
        ):
            raise ValueError(
                "collision patient motion measurement is not authorized by its "
                f"resolved route: {resolved_route!r}"
            )
        module = self._module_for_route_record(
            resolved_route,
            decision_id=COLLISION_PATIENT_MOTION_GATE_DECISION_ID,
            module_name="patient_motion_gate",
        )
        if module.implementation != "segment_edge_mask_centroid_displacement":
            raise ValueError(
                "unsupported patient-motion-gate module implementation: "
                f"{module.implementation!r}"
            )
        return (
            module.require_number("minimum_displacement_px"),
            module.require_integer("edge_frames"),
        )

    def _physion_pp_collision_patient_motion(
        self,
        *,
        patient_object_id: str,
        two_segment: Dict[str, Any],
        mask_records: Dict[str, Dict[int, Dict[str, Any]]],
        mask_arrays: Dict[str, np.ndarray],
        resolved_route: Dict[str, Any],
    ) -> Dict[str, Any]:
        minimum_displacement_px, edge_frames = (
            self._collision_patient_motion_parameters(resolved_route)
        )
        motion: Dict[str, Any] = {
            "moving": False,
            "threshold_px": minimum_displacement_px,
            "displacement_px": None,
            "resolved_route": deepcopy(resolved_route),
        }
        frames = sorted(mask_records.get(patient_object_id) or {})
        seg2_range = two_segment.get("seg2")
        if isinstance(seg2_range, (list, tuple)) and len(seg2_range) == 2:
            lo, hi = int(seg2_range[0]), int(seg2_range[1])
            frames = [frame for frame in frames if lo <= frame <= hi]

        def _edge_centroid(edge_frames: list[int]) -> Optional[np.ndarray]:
            points = []
            for frame in edge_frames:
                record = (mask_records.get(patient_object_id) or {}).get(frame) or {}
                mask = mask_arrays.get(str(record.get("mask_key") or ""))
                if mask is None or not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                points.append((float(xs.mean()), float(ys.mean())))
            if not points:
                return None
            return np.median(np.asarray(points, dtype=np.float64), axis=0)

        edge = edge_frames
        first = _edge_centroid(frames[:edge])
        last = _edge_centroid(frames[-edge:])
        if first is not None and last is not None and len(frames) >= 2:
            displacement = float(np.linalg.norm(last - first))
            motion["displacement_px"] = round(displacement, 2)
            motion["moving"] = displacement >= minimum_displacement_px
            motion["frame_count"] = len(frames)
        else:
            motion["reason"] = "not enough seg2 patient masks for the motion test"
        return motion

    def _physion_pp_friction_collision_context(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
        motion_gate_route: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """friction_collision context: role-to-object ids, segment ranges, and the
        seg2-patient motion test (mask-centroid net displacement). Computed once per
        pose-correction run and stashed on the payload, so the gravity search's
        candidate payload copies, the ground gate, the rotation correction, and the fc
        flush/drop refinements all read one consistent decision."""
        if self._object_plan_scenario(object_plan).lower() not in PHYSION_PP_FRICTION_COLLISION_SCENARIOS:
            return None
        if not isinstance(motion_gate_route, dict):
            raise ValueError(
                "friction_collision is missing its collision patient motion-gate route"
            )
        motion_threshold_px, _motion_edge_frames = (
            self._collision_patient_motion_parameters(motion_gate_route)
        )
        context: Dict[str, Any] = {
            "applies": True,
            "scenario": self._object_plan_scenario(object_plan),
            "source": "physion_pp_friction_collision_context",
            "motion_gate_route": deepcopy(motion_gate_route),
        }
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        physion_tracking = tracks_payload.get("physion_tracking") or {}
        two_segment = (
            physion_tracking.get("two_segment")
            if isinstance(physion_tracking.get("two_segment"), dict)
            else {}
        )
        role_binding = physion_tracking.get("role_binding") or {}
        assignments = (
            role_binding.get("assignments") if isinstance(role_binding.get("assignments"), dict) else {}
        )
        object_id_by_track = {
            str(item.source_track_id): str(item.object_id)
            for item in object_plan.target_objects
            if item.source_track_id
        }
        roles: Dict[str, Optional[str]] = {}
        for seg in ("seg1", "seg2"):
            seg_roles = assignments.get(seg) if isinstance(assignments.get(seg), dict) else {}
            for role in ("agent", "patient"):
                track = seg_roles.get(role)
                roles[f"{seg}_{role}"] = object_id_by_track.get(str(track)) if track else None
        context["role_object_ids"] = roles
        context["patient_object_ids"] = sorted(
            {oid for key, oid in roles.items() if oid and key.endswith("_patient")}
        )
        context["agent_object_ids"] = sorted(
            {oid for key, oid in roles.items() if oid and key.endswith("_agent")}
        )
        context["two_segment"] = {
            key: two_segment.get(key) for key in ("a", "b", "seg1", "seg2") if key in two_segment
        }
        context["gravity_scoring_object_ids"] = (
            [roles["seg1_patient"]] if roles.get("seg1_patient") else []
        )

        # seg2 patient motion test on mask centroids: static vs knocked-and-falling is
        # bimodal (fc64 audit: <1px vs 30-48px net displacement), so 5px is a safe cut.
        seg2_patient = roles.get("seg2_patient")
        if seg2_patient:
            try:
                mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
                mask_arrays = self._sam3_video_mask_arrays(mask_records)
                motion = self._physion_pp_collision_patient_motion(
                    patient_object_id=seg2_patient,
                    two_segment=two_segment,
                    mask_records=mask_records,
                    mask_arrays=mask_arrays,
                    resolved_route=motion_gate_route,
                )
            except Exception as exc:
                motion = {
                    "moving": False,
                    "threshold_px": motion_threshold_px,
                    "displacement_px": None,
                    "reason": f"motion test failed: {_short_text(str(exc), 160)}",
                    "resolved_route": deepcopy(motion_gate_route),
                }
        else:
            motion = {
                "moving": False,
                "threshold_px": motion_threshold_px,
                "displacement_px": None,
                "reason": "no seg2 patient object",
                "resolved_route": deepcopy(motion_gate_route),
            }
        context["seg2_patient_motion"] = motion
        context["ground_snap_exempt_object_ids"] = (
            [seg2_patient] if seg2_patient and motion.get("moving") else []
        )
        _log_tool(
            self.tool_name,
            f"friction_collision context roles={roles} "
            f"seg2_patient_moving={motion.get('moving')} disp_px={motion.get('displacement_px')}",
        )
        return context

    def _physion_pp_mass_collision_context(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
        motion_gate_route: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """mass_collision context: role/ball/extra object ids, segment ranges, the
        seg2-patient motion test, and the agent's pre-impact gravity-scoring window.
        Computed once per pose-correction run and stashed on the payload so the gravity
        search's candidate copies, the ground gate, the rotation correction, and the mc
        flush/drop refinements all read one consistent decision.

        Gravity evidence is seg1-only: the kept extras (parked near-ground objects)
        plus the agent restricted to its pre-impact window -- before the ball's mask
        first touches the agent's mask the route treats the agent as stationary on the floor, so
        straightening it per candidate inside the scoring loop is physically sound. The
        DELIVERED agent rotation stays raw FP (the agent is NOT a static object; only
        its pre-impact stillness is borrowed), and the ball (sphere,
        candidate-independent silhouette) never votes."""
        if self._object_plan_scenario(object_plan).lower() not in PHYSION_PP_MASS_COLLISION_SCENARIOS:
            return None
        if not isinstance(motion_gate_route, dict):
            raise ValueError(
                "mass_collision is missing its collision patient motion-gate route"
            )
        motion_threshold_px, _motion_edge_frames = (
            self._collision_patient_motion_parameters(motion_gate_route)
        )
        context: Dict[str, Any] = {
            "applies": True,
            "scenario": self._object_plan_scenario(object_plan),
            "source": "physion_pp_mass_collision_context",
            "motion_gate_route": deepcopy(motion_gate_route),
        }
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        physion_tracking = tracks_payload.get("physion_tracking") or {}
        two_segment = (
            physion_tracking.get("two_segment")
            if isinstance(physion_tracking.get("two_segment"), dict)
            else {}
        )
        role_binding = physion_tracking.get("role_binding") or {}
        assignments = (
            role_binding.get("assignments") if isinstance(role_binding.get("assignments"), dict) else {}
        )
        ball = role_binding.get("ball") if isinstance(role_binding.get("ball"), dict) else {}
        object_id_by_track = {
            str(item.source_track_id): str(item.object_id)
            for item in object_plan.target_objects
            if item.source_track_id
        }
        roles: Dict[str, Optional[str]] = {}
        for seg in ("seg1", "seg2"):
            seg_roles = assignments.get(seg) if isinstance(assignments.get(seg), dict) else {}
            for role in ("agent", "patient"):
                track = seg_roles.get(role)
                roles[f"{seg}_{role}"] = object_id_by_track.get(str(track)) if track else None
            ball_track = ball.get(seg)
            roles[f"{seg}_ball"] = object_id_by_track.get(str(ball_track)) if ball_track else None
        context["role_object_ids"] = roles
        extras: list[Dict[str, Any]] = []
        for extra in role_binding.get("kept_extra_tracks") or []:
            if not isinstance(extra, dict):
                continue
            track = str(extra.get("track") or "")
            object_id = object_id_by_track.get(track)
            if object_id:
                extras.append(
                    {"track": track, "object_id": object_id, "settled_frame": extra.get("settled_frame")}
                )
        context["extras"] = extras
        extra_ids = [str(entry["object_id"]) for entry in extras]
        context["extra_object_ids"] = extra_ids
        seg2_patient = roles.get("seg2_patient")
        seg1_agent = roles.get("seg1_agent")
        seg1_ball = roles.get("seg1_ball")
        context["patient_object_ids"] = [seg2_patient] if seg2_patient else []
        context["ball_object_ids"] = sorted(
            {oid for key, oid in roles.items() if oid and key.endswith("_ball")}
        )
        context["two_segment"] = {
            key: two_segment.get(key) for key in ("a", "b", "seg1", "seg2") if key in two_segment
        }
        # The extra<->patient link is decided ONCE at plan time (focused VLM identity inside the
        # two-segment mesh reuse tagging); read the recorded decision so pose correction
        # and the plan stage can never disagree.
        plan_payload = self.artifacts.read_optional(_scene_object_plan_path(question_dir)) or {}
        mesh_reuse = plan_payload.get("two_segment_mesh_reuse") or {}
        link = mesh_reuse.get("mc_extra_patient_link") or {}
        linked_track = str(link.get("extra_track") or "") if link.get("linked") is True else ""
        context["linked_extra_object_id"] = object_id_by_track.get(linked_track) if linked_track else None
        context["extra_patient_link"] = link or None
        # Size-lock provenance: when the plan-stage reuse actually redirected the seg2
        # patient onto the extra's mesh, the patient size is locked even if the extra
        # flush later fails -- the drop refinement must then search yaw, never scale.
        context["patient_mesh_reused"] = any(
            isinstance(entry, dict) and entry.get("role") == "patient" and entry.get("seg1_track") == linked_track
            for entry in mesh_reuse.get("reused") or []
        ) if linked_track else False
        # Rotation carve-outs: patients + extras are straightened per frame in the
        # DELIVERED world; the agent joins them ONLY inside gravity-scoring candidate
        # evaluations; balls keep the sphere identity-rotation path.
        context["straighten_object_ids"] = sorted({oid for oid in [seg2_patient, *extra_ids] if oid})

        mask_records: Dict[str, Dict[int, Dict[str, Any]]] = {}
        mask_arrays: Dict[str, np.ndarray] = {}
        try:
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
        except Exception as exc:
            context["mask_load_error"] = _short_text(str(exc), 160)

        # Pre-impact window: first seg1 frame where the (dilated) ball mask touches the
        # agent mask, minus a safety margin against SAM3 mask bleed at near-contact.
        contact: Dict[str, Any] = {"contact_frame": None, "window_end_frame": None}
        frame_caps: Dict[str, int] = {}
        if seg1_agent and seg1_ball and mask_records:
            try:
                import cv2

                kernel = np.ones(
                    (PHYSION_PP_MC_CONTACT_DILATE_PX * 2 + 1,) * 2, np.uint8
                )
                agent_frames = mask_records.get(seg1_agent) or {}
                ball_frames = mask_records.get(seg1_ball) or {}
                common = sorted(set(agent_frames) & set(ball_frames))
                seg1_range = two_segment.get("seg1")
                if isinstance(seg1_range, (list, tuple)) and len(seg1_range) == 2:
                    lo, hi = int(seg1_range[0]), int(seg1_range[1])
                    common = [frame for frame in common if lo <= frame <= hi]
                for frame in common:
                    agent_mask = mask_arrays.get(str((agent_frames.get(frame) or {}).get("mask_key") or ""))
                    ball_mask = mask_arrays.get(str((ball_frames.get(frame) or {}).get("mask_key") or ""))
                    if agent_mask is None or ball_mask is None or agent_mask.shape != ball_mask.shape:
                        continue
                    dilated = cv2.dilate(ball_mask.astype(np.uint8), kernel) > 0
                    if bool(np.logical_and(dilated, agent_mask > 0).any()):
                        contact["contact_frame"] = int(frame)
                        break
                if contact["contact_frame"] is not None:
                    window_end = int(contact["contact_frame"]) - PHYSION_PP_MC_CONTACT_MARGIN_FRAMES
                    contact["window_end_frame"] = window_end
                    if window_end >= 0:
                        frame_caps[str(seg1_agent)] = window_end
            except Exception as exc:
                contact["reason"] = f"contact test failed: {_short_text(str(exc), 160)}"
        context["agent_pre_impact"] = contact
        context["gravity_scoring_frame_caps"] = frame_caps
        # The agent votes ONLY when a positive-length pre-impact window was established
        # under the route's pre-impact stationary assumption. Contact too early, a
        # failed contact test, or NO detected contact (a mask gap is not evidence the
        # ball never arrived) all drop the agent from the scoring set; the extras carry
        # the gravity evidence alone. With no extras either, the empty set falls back
        # to the generic unfiltered pp scoring (pre-existing semantics).
        agent_votes = bool(seg1_agent) and str(seg1_agent) in frame_caps
        if not seg1_agent:
            context["agent_gravity_scoring"] = "excluded_no_agent"
        elif agent_votes:
            context["agent_gravity_scoring"] = "included_pre_impact_window"
        elif contact.get("contact_frame") is not None:
            context["agent_gravity_scoring"] = "excluded_empty_window"
        else:
            context["agent_gravity_scoring"] = "excluded_contact_unknown"
        context["scoring_straighten_object_ids"] = [seg1_agent] if agent_votes else []
        context["gravity_scoring_object_ids"] = sorted(
            {oid for oid in [*extra_ids, seg1_agent if agent_votes else None] if oid}
        )

        # seg2 patient motion test on mask centroids (same rule and constants as
        # friction_collision: static vs falling is bimodal, 5px is a safe cut).
        if seg2_patient and mask_records:
            try:
                motion = self._physion_pp_collision_patient_motion(
                    patient_object_id=seg2_patient,
                    two_segment=two_segment,
                    mask_records=mask_records,
                    mask_arrays=mask_arrays,
                    resolved_route=motion_gate_route,
                )
            except Exception as exc:
                motion = {
                    "moving": False,
                    "threshold_px": motion_threshold_px,
                    "displacement_px": None,
                    "reason": f"motion test failed: {_short_text(str(exc), 160)}",
                    "resolved_route": deepcopy(motion_gate_route),
                }
        else:
            motion = {
                "moving": False,
                "threshold_px": motion_threshold_px,
                "displacement_px": None,
                "reason": "no seg2 patient object",
                "resolved_route": deepcopy(motion_gate_route),
            }
        context["seg2_patient_motion"] = motion
        context["ground_snap_exempt_object_ids"] = (
            [seg2_patient] if seg2_patient and motion.get("moving") else []
        )
        _log_tool(
            self.tool_name,
            f"mass_collision context roles={roles} extras={extra_ids} "
            f"linked_extra={context['linked_extra_object_id']} "
            f"contact_frame={contact.get('contact_frame')} "
            f"seg2_patient_moving={motion.get('moving')} disp_px={motion.get('displacement_px')}",
        )
        return context

    def _classify_ground_contact_per_object(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> Dict[str, Any]:
        scenario = self._object_plan_scenario(object_plan)
        route = self._ground_motion_route(object_plan)
        if route != PHYSION_PP_AGENT_FREE_GROUND_MOTION_ROUTE:
            raise ValueError(
                "per-object ground classification requires the Physion++ "
                f"agent/free fixture route, got {route!r}"
            )
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
        geometry_policy, _sphere_agent_ids = self._physion_pp_sphere_agent_context(
            question_dir=question_dir,
            object_plan=object_plan,
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
        fc_context = self._physion_pp_friction_collision_context(
            question_dir=question_dir,
            object_plan=object_plan,
            motion_gate_route=collision_patient_motion_gate_route,
        )
        if fc_context is not None:
            payload["physion_pp_friction_collision"] = fc_context
        mc_context = self._physion_pp_mass_collision_context(
            question_dir=question_dir,
            object_plan=object_plan,
            motion_gate_route=collision_patient_motion_gate_route,
        )
        if mc_context is not None:
            payload["physion_pp_mass_collision"] = mc_context
        if self._physion_pp_similarity_snap_ids(
            object_plan,
            support_route=(
                str(support_snap_route["route"])
                if support_snap_route is not None
                else None
            ),
        ):
            self._restore_physion_pp_similarity_meshes(question_dir)
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
        self._bake_physion_pp_similarity_snap(
            question_dir=question_dir,
            support=payload.get("support_plane_position_correction"),
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
        self._apply_static_fixture_flush_for_route(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
            resolved_route=static_fixture_flush_route,
        )
        # MUST run before the agent ray refinement: the agent trajectory is re-laid on
        # the support manifold rasterized from the statics, so any static layout change
        # after it would leave a stale, inconsistent agent trajectory.
        self._apply_physion_pp_line_layout_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
            resolved_route=line_layout_route,
        )
        # friction_platform_pp only: re-lay the agent trajectory on the fitted
        # line-on-support-surface manifold and derive its metric scale once.
        self._apply_physion_pp_agent_ray_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # bouncy_platform_pp airborne agent: pin the ball trajectory from mask + the
        # shared collinear line (vertical-plane intersection), no depth. Runs after
        # line-layout (needs its final theta,c); the friction-platform-only agent-ray
        # call above is a no-op for this scenario.
        self._apply_physion_pp_bounce_trajectory_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # bouncy_wall_pp dynamic object(s): pin per-frame position from mask + the
        # wall-centerline vertical plane (no depth), and set orientation from the FP
        # register+track rotation. Runs after the static-fixture flush (needs the wall
        # poses); the
        # friction-platform-only agent-ray call above is a no-op for this scenario.
        self._apply_physion_pp_bouncy_wall_trajectory_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # friction_collision moving seg2 patient: re-derive the vertical-drop trajectory
        # from masks (first-frame ray anchor + per-frame height-only descent). Runs last
        # so it overwrites the snap-exempt FP poses with the mask-pinned fall, using the
        # seg1 post-flush mesh handed over above.
        self._apply_physion_pp_friction_collision_drop_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # mass_collision agent: per-frame ground-plane flush against the SAM3 mask
        # (height + raw FP rotation untouched). Runs before the ball/drop refinements
        # so the drop plane's agent anchor reads the polished position.
        self._apply_physion_pp_mass_collision_agent_flush_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # mass_collision ball: FP track_one freezes on the small ball; pin the per-frame
        # position onto a mask-fitted straight ground-plane line through the CURRENT
        # start pose (kept bit-identical, so the drop refinement's ball anchor below is
        # unaffected), and CUT the track after agent contact + line deviation.
        self._apply_physion_pp_mass_collision_ball_trajectory_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        # mass_collision moving seg2 patient: vertical-drop trajectory pinned by the
        # ball-agent line prior (per-frame mask-centroid ray ∩ vertical plane, zero
        # depth dependence). Runs last so it overwrites the snap-exempt FP poses.
        self._apply_physion_pp_mass_collision_drop_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
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

    def _physion_pp_similarity_snap_ids(
        self,
        object_plan: ObjectPlan,
        *,
        support_route: Optional[str] = None,
    ) -> set[str]:
        """Objects that take the similarity ground snap: Physion++ static fixtures only."""
        if support_route is None:
            similarity_snap_static_fixtures = self._object_plan_scenario(
                object_plan
            ).lower().endswith("_pp")
        else:
            benchmark = (
                "clevrer"
                if support_route == CLEVRER_SUPPORT_SNAP_ROUTE
                else "physion_pp"
            )
            scenario = (
                "" if benchmark == "clevrer" else self._object_plan_scenario(object_plan)
            )
            profile = default_module_profile_policy().resolve_route(
                SUPPORT_SNAP_DECISION_ID,
                support_route,
                benchmark=benchmark,
                scenario=scenario,
            )
            module = profile.module("support_snap")
            if (
                module.implementation
                != "mesh_centroid_ray_slide_to_support_plane"
            ):
                raise ValueError(
                    "unsupported support-snap module implementation: "
                    f"{module.implementation!r}"
                )
            similarity_snap_static_fixtures = module.require_boolean(
                "similarity_snap_static_fixtures"
            )
        if not similarity_snap_static_fixtures:
            return set()
        return {
            str(target.object_id)
            for target in object_plan.target_objects
            if str(target.source_track_id or "").startswith(PHYSION_PP_STATIC_TRACK_PREFIX)
        }

    def _restore_physion_pp_similarity_meshes(self, question_dir: Path) -> None:
        """Idempotency guard: reset foundationpose mesh paths to their pre-similarity
        sources so a re-run never compounds scales on already-scaled meshes."""
        fp_path = self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
        payload = self.artifacts.read_optional(fp_path)
        if not isinstance(payload, dict):
            return
        changed = False
        for record in payload.get("objects", []):
            if isinstance(record, dict) and record.get("similarity_source_mesh_path"):
                record["mesh_path"] = record.pop("similarity_source_mesh_path")
                record.pop("similarity_scale", None)
                changed = True
        if changed:
            self.artifacts.write(fp_path, payload)

    def _bake_physion_pp_similarity_snap(self, *, question_dir: Path, support: Any) -> None:
        """Un-fold the similarity-snapped poses back to orthonormal rotations and bake
        the scales into derived meshes. The scaled mesh is propagated through BOTH
        foundationpose_poses.json (flush refinement and the debug renders read it) and
        the support entries (line-layout, agent-ray, and SWR read those), so every
        downstream consumer sees one consistent scaled world."""
        if not isinstance(support, dict):
            return
        pending = []
        for item in support.get("objects", []):
            if not isinstance(item, dict):
                continue
            folded = [
                pose
                for pose in item.get("poses", [])
                if isinstance(pose, dict) and pose.get("similarity_folded") is True
            ]
            if folded:
                pending.append((item, folded))
        if not pending:
            return
        fp_path = self.artifacts.artifact_path_by_name(question_dir, "foundationpose_poses.json")
        fp_payload = self.artifacts.read_optional(fp_path) or {}
        fp_by_id = {
            str(record.get("object_id")): record
            for record in fp_payload.get("objects", [])
            if isinstance(record, dict)
        }
        mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "similarity_scaled_meshes"
        rewritten = False
        for item, folded in pending:
            object_id = str(item.get("object_id") or "")
            for pose in folded:
                pose_scale = float(pose.get("similarity_scale") or 1.0)
                matrix = np.asarray(pose["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                matrix[:3, :3] = matrix[:3, :3] / pose_scale
                pose["corrected_pose_4x4"] = matrix.tolist()
                pose["similarity_folded"] = False
            scale = float(item.get("similarity_scale") or 1.0)
            if abs(scale - 1.0) <= 1e-9:
                continue
            source_mesh = Path(str(item.get("mesh_path")))
            vertices, faces = self._load_mesh_geometry(source_mesh)
            scaled_path = mesh_dir / f"{object_id}_similarity_local.glb"
            self._export_mesh_with_source_colors(
                vertices=vertices * scale,
                faces=faces,
                source_mesh_path=source_mesh,
                output_path=scaled_path,
            )
            item["pre_similarity_mesh_path"] = str(source_mesh)
            item["mesh_path"] = str(scaled_path)
            record = fp_by_id.get(object_id)
            if record is not None:
                record["similarity_source_mesh_path"] = record.get("mesh_path")
                record["mesh_path"] = str(scaled_path)
                record["similarity_scale"] = scale
                rewritten = True
        if rewritten:
            self.artifacts.write(fp_path, fp_payload)

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
        similarity_snap_ids = self._physion_pp_similarity_snap_ids(
            object_plan,
            support_route=support_route,
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
                    if (
                        np.isfinite(scale)
                        and PHYSION_PP_SIMILARITY_SNAP_SCALE_GATE[0] <= scale <= PHYSION_PP_SIMILARITY_SNAP_SCALE_GATE[1]
                    ):
                        # Folded similarity pose: [[s*R, s*t]] renders the scaled world
                        # through the generic linear transform during the overlap gravity
                        # search; _bake_physion_pp_similarity_snap un-folds the winner back
                        # to an orthonormal rotation and bakes s into a derived mesh.
                        folded = matrix.copy()
                        folded[:3, :3] = scale * rotation
                        folded[:3, 3] = scale * translation
                        pose_payload.update(
                            {
                                "status": "ok",
                                "corrected_pose_4x4": folded.tolist(),
                                "source_translation_camera": translation.tolist(),
                                "corrected_translation_camera": (scale * translation).tolist(),
                                "similarity_scale": float(scale),
                                "similarity_folded": True,
                                "source_bottom_height_along_up_axis": bottom_height,
                                "corrected_bottom_height_along_up_axis": float(scale * bottom_height),
                                "support_plane_height_along_up_axis": ground_height,
                                "translation_adjustment_camera": (scale * translation - translation).tolist(),
                                "translation_adjustment_norm": float(np.linalg.norm(scale * translation - translation)),
                            }
                        )
                        adjustments.append(float(np.linalg.norm(scale * translation - translation)))
                        corrected_poses.append(pose_payload)
                        continue
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

    def _static_ground_fixture_object_ids(self, object_plan: ObjectPlan) -> set[str]:
        return {
            str(item.object_id)
            for item in object_plan.target_objects
            if _is_physion_static_ground_fixture_track(item.source_track_id)
        }

    def _apply_static_fixture_flush_for_route(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        resolved_route: Optional[Dict[str, Any]],
    ) -> None:
        if resolved_route is None:
            # Without a resolved route, each refinement applies its own eligibility checks.
            self._apply_static_fixture_flush_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
            )
            self._apply_physion_pp_friction_collision_flush_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
            )
            self._apply_physion_pp_mass_collision_flush_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
            )
            return
        if resolved_route.get("decision_id") != STATIC_FIXTURE_FLUSH_DECISION_ID:
            raise ValueError(
                "static-fixture-flush route has an unexpected decision_id: "
                f"{resolved_route.get('decision_id')!r}"
            )
        context = resolved_route.get("context")
        if not isinstance(context, dict):
            raise ValueError("static-fixture-flush route is missing its context")
        route = str(resolved_route.get("route") or "")
        profile = default_module_profile_policy().resolve_route(
            STATIC_FIXTURE_FLUSH_DECISION_ID,
            route,
            benchmark=str(context.get("benchmark") or "").strip().lower(),
            scenario=str(context.get("scenario") or "").strip().lower(),
        )
        module = profile.module("static_fixture_flush")
        dispatch = {
            "static_ground_fixture_coordinate_descent": (
                self._apply_static_fixture_flush_refinement
            ),
            "segment1_patient_axis_flush_and_mesh_handoff": (
                self._apply_physion_pp_friction_collision_flush_refinement
            ),
            "kept_extra_coordinate_descent_and_linked_mesh_handoff": (
                self._apply_physion_pp_mass_collision_flush_refinement
            ),
        }
        implementation = dispatch.get(module.implementation)
        if implementation is None:
            raise ValueError(
                "unsupported static-fixture-flush module implementation: "
                f"{module.implementation!r}"
            )
        implementation(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
        )
        result = payload.get("static_fixture_flush_refinement")
        if isinstance(result, dict):
            result["resolved_route"] = deepcopy(resolved_route)

    def _apply_static_fixture_flush_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        static_ids = self._static_ground_fixture_object_ids(object_plan)
        if not static_ids:
            return
        summary: Dict[str, Any] = {
            "applied": False,
            "policy": (
                "coordinate descent over in-plane translation, yaw about the up axis, and "
                "in-plane extent scale; every move stays inside the support plane so the "
                "flush-to-ground constraint is preserved by construction"
            ),
            "objective": "maximize mean IoU between rendered visible box masks and SAM3 masks",
            "objects": [],
        }
        payload["static_fixture_flush_refinement"] = summary
        trajectory_correction = payload.get("trajectory_correction")
        corrected = (
            trajectory_correction.get("corrected_trajectories")
            if isinstance(trajectory_correction, dict)
            else None
        )
        if not isinstance(corrected, dict) or corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing or invalid gravity_direction_camera"
            return
        try:
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
        except Exception as exc:
            summary["reason"] = f"failed to load refinement inputs: {exc}"
            return

        support = payload.get("support_plane_position_correction")
        support_objects = {
            str(item.get("object_id")): item
            for item in (support.get("objects", []) if isinstance(support, dict) else [])
            if isinstance(item, dict)
        }
        track_by_object_id = {
            str(item.object_id): str(item.source_track_id or "")
            for item in object_plan.target_objects
        }
        # Two-segment (bouncy_wall) reuse statics are the SAME physical object as their seg1
        # source, so they must be the SAME size in both segments. Flush the seg1 (non-reuse)
        # statics first and record each one's FINAL (post-flush) mesh -- that mesh already
        # carries the source's similarity-snap scale and flush scale. Then a seg2 reuse static
        # does NOT flush its own conditioned mesh: it adopts the seg1 source's final mesh and
        # only re-solves position (in-plane translate + yaw + ground pin), with the scale search
        # pinned to identity. So the reused object ends up the exact size of its seg1 source,
        # placed by flush in seg2. The reuse static keeps its own similarity snap purely for a
        # ground-contacting START pose (its rescaled mesh is discarded here); we deliberately do
        # NOT route it through the rigid ray-slide fallback, whose grazing-angle lever the snap
        # exists to avoid. Non-reuse statics keep the full free scale search.
        reuse_source_of = {
            str(item.object_id): str(item.mesh_reuse_source_object_id)
            for item in object_plan.target_objects
            if getattr(item, "mesh_reuse_source_object_id", None)
        }
        static_items = [
            item
            for item in corrected.get("objects", [])
            if isinstance(item, dict) and str(item.get("object_id") or "") in static_ids
        ]
        # non-reuse (seg1) first, so their final mesh is recorded before the reuse (seg2) pass
        static_items.sort(key=lambda it: str(it.get("object_id") or "") in reuse_source_of)
        source_final_mesh: Dict[str, str] = {}
        for item in static_items:
            object_id = str(item.get("object_id") or "")
            source_id = reuse_source_of.get(object_id)
            reuse_mesh = source_final_mesh.get(source_id) if source_id else None
            entry = self._refine_static_fixture_object(
                object_id=object_id,
                trajectory_item=item,
                support_item=support_objects.get(object_id),
                mesh_path=reuse_mesh or mesh_paths.get(object_id),
                mask_records=mask_records.get(object_id) or {},
                mask_arrays=mask_arrays,
                intrinsics=intrinsics,
                image_shape=image_shape,
                up_axis=up_axis,
                question_dir=question_dir,
                flatten_base=track_by_object_id.get(object_id, "").startswith(
                    PHYSION_PP_STATIC_TRACK_PREFIX
                ),
                forced_scale=(1.0, 1.0) if reuse_mesh else None,
            )
            summary["objects"].append(entry)
            if source_id is None and entry.get("status") == "ok":
                final_mesh = entry.get("refined_mesh_path") or mesh_paths.get(object_id)
                if final_mesh:
                    source_final_mesh[object_id] = str(final_mesh)
        summary["applied"] = any(entry.get("status") == "ok" for entry in summary["objects"])

        # When the routed fixture set contains seg1/seg2 mesh-reuse pairs, run one more
        # joint pass after the ordinary per-object flush. The two clips
        # show the SAME physical mat/wall, so this pass shares one anisotropic in-plane
        # scale while retaining independent translation and yaw per segment. FP/BP have
        # no such reuse pairs and therefore retain the ordinary single-object result.
        scenario = self._object_plan_scenario(object_plan).lower()
        if not reuse_source_of:
            return
        second_started = time.monotonic()
        second_summary: Dict[str, Any] = {
            "applied": False,
            "scenario": scenario,
            "gate": "bouncy_wall_pp scenario allowlist",
            "policy": (
                "second flush after the per-object pass; each seg1/seg2 physical fixture "
                "pair shares one in-plane anisotropic scale while retaining independent "
                "in-plane translation and yaw"
            ),
            "objective": "maximize summed mean visible-mask IoU over both segments",
            "groups": [],
        }
        summary["second_joint_refinement"] = second_summary
        first_pass_objects = list(summary["objects"])
        first_pass_by_id = {
            str(entry.get("object_id")): entry
            for entry in first_pass_objects
            if isinstance(entry, dict) and entry.get("object_id")
        }
        static_by_id = {
            str(item.get("object_id")): item
            for item in static_items
            if isinstance(item, dict) and item.get("object_id")
        }
        joint_pairs = sorted(
            (source_id, reuse_id)
            for reuse_id, source_id in reuse_source_of.items()
            if source_id in static_ids and reuse_id in static_ids
        )
        if len(joint_pairs) != 2:
            second_summary["reason"] = (
                f"expected two complete static reuse pairs, got {joint_pairs}"
            )
            return

        def _current_static_mesh(object_id: str) -> Optional[str]:
            support_item = support_objects.get(object_id)
            if isinstance(support_item, dict) and support_item.get("mesh_path"):
                return str(support_item["mesh_path"])
            first_entry = first_pass_by_id.get(object_id) or {}
            return first_entry.get("refined_mesh_path") or mesh_paths.get(object_id)

        incomplete_pairs = [
            [source_id, reuse_id]
            for source_id, reuse_id in joint_pairs
            if any(
                object_id not in static_by_id
                or (first_pass_by_id.get(object_id) or {}).get("status") != "ok"
                or not _current_static_mesh(object_id)
                for object_id in (source_id, reuse_id)
            )
        ]
        if incomplete_pairs:
            second_summary["reason"] = (
                f"first flush did not produce complete pairs for {incomplete_pairs}"
            )
            return

        import trimesh

        joint_objects: list[Dict[str, Any]] = []
        for source_id, reuse_id in joint_pairs:
            group_started = time.monotonic()
            group_results = self._refine_static_fixture_group(
                [
                    {
                        "object_id": object_id,
                        "trajectory_item": static_by_id[object_id],
                        "support_item": support_objects.get(object_id),
                        "mesh_path": _current_static_mesh(object_id),
                        "mask_records": mask_records.get(object_id) or {},
                        "mask_arrays": mask_arrays,
                        "intrinsics": intrinsics,
                        "image_shape": image_shape,
                        "up_axis": up_axis,
                        "question_dir": question_dir,
                        "flatten_base": track_by_object_id.get(object_id, "").startswith(
                            PHYSION_PP_STATIC_TRACK_PREFIX
                        ),
                    }
                    for object_id in (source_id, reuse_id)
                ]
            )
            group_by_id = {
                str(entry.get("object_id")): entry
                for entry in group_results
                if isinstance(entry, dict) and entry.get("object_id")
            }
            if any(
                (group_by_id.get(object_id) or {}).get("status") != "ok"
                for object_id in (source_id, reuse_id)
            ):
                raise RuntimeError(
                    f"bouncy-wall joint flush failed for {[source_id, reuse_id]}: "
                    f"{group_results}"
                )
            source_scale = np.asarray(
                [
                    group_by_id[source_id]["scale_u"],
                    group_by_id[source_id]["scale_v"],
                ],
                dtype=np.float64,
            )
            reuse_scale = np.asarray(
                [
                    group_by_id[reuse_id]["scale_u"],
                    group_by_id[reuse_id]["scale_v"],
                ],
                dtype=np.float64,
            )
            if not np.array_equal(source_scale, reuse_scale):
                raise RuntimeError(
                    f"bouncy-wall joint flush produced different shared scales for "
                    f"{[source_id, reuse_id]}: {source_scale.tolist()} vs "
                    f"{reuse_scale.tolist()}"
                )
            final_extents: Dict[str, np.ndarray] = {}
            for object_id in (source_id, reuse_id):
                final_mesh_path = _current_static_mesh(object_id)
                if not final_mesh_path:
                    raise RuntimeError(
                        f"bouncy-wall joint flush missing final mesh for {object_id}"
                    )
                final_vertices, _ = self._load_mesh_geometry(Path(final_mesh_path))
                final_extents[object_id] = np.sort(
                    np.asarray(
                        trimesh.bounds.oriented_bounds(final_vertices)[1],
                        dtype=np.float64,
                    )
                )
            extent_delta = np.abs(
                final_extents[source_id] - final_extents[reuse_id]
            )
            relative_extent_delta = extent_delta / np.maximum(
                final_extents[source_id], 1e-9
            )
            max_relative_extent_delta = float(relative_extent_delta.max())
            size_tolerance = 1e-5
            if max_relative_extent_delta > size_tolerance:
                raise RuntimeError(
                    f"bouncy-wall joint flush final mesh size mismatch for "
                    f"{[source_id, reuse_id]}: max_relative_extent_delta="
                    f"{max_relative_extent_delta:.8f} > {size_tolerance}"
                )
            for object_id in (source_id, reuse_id):
                result = group_by_id[object_id]
                result["first_pass_refined_mean_iou"] = (
                    first_pass_by_id[object_id].get("refined_mean_iou")
                )
                joint_objects.append(result)
            start_sum = sum(
                float(group_by_id[object_id]["start_mean_iou"])
                for object_id in (source_id, reuse_id)
            )
            refined_sum = sum(
                float(group_by_id[object_id]["refined_mean_iou"])
                for object_id in (source_id, reuse_id)
            )
            second_summary["groups"].append(
                {
                    "source_object_id": source_id,
                    "reuse_object_id": reuse_id,
                    "shared_scale_u": float(group_by_id[source_id]["scale_u"]),
                    "shared_scale_v": float(group_by_id[source_id]["scale_v"]),
                    "start_sum_iou": start_sum,
                    "refined_sum_iou": refined_sum,
                    "delta_sum_iou": refined_sum - start_sum,
                    "search_elapsed_sec": round(time.monotonic() - group_started, 2),
                    "shared_physical_size_verification": {
                        "verified": True,
                        "obb_extents_source": final_extents[source_id].tolist(),
                        "obb_extents_reuse": final_extents[reuse_id].tolist(),
                        "max_relative_extent_delta": max_relative_extent_delta,
                        "relative_tolerance": size_tolerance,
                    },
                }
            )

        # All downstream consumers read summary.objects, support mesh_path, and the
        # corrected trajectory poses.  Publish only the joint-pass entries there so
        # bouncy-wall trajectory fitting and SWR consume the final mesh/pose rather than
        # the stale first-pass result; retain the first pass separately for diagnostics.
        joint_by_id = {str(entry["object_id"]): entry for entry in joint_objects}
        summary["first_pass_objects"] = first_pass_objects
        summary["objects"] = [
            joint_by_id[str(entry["object_id"])] for entry in first_pass_objects
        ]
        summary["policy"] = (
            f"{summary['policy']}; bouncy_wall_pp then runs a second joint shared-scale "
            "pass over each seg1/seg2 static pair"
        )
        second_summary["applied"] = True
        second_summary["search_elapsed_sec"] = round(time.monotonic() - second_started, 2)
        summary["applied"] = bool(joint_objects)

    def _apply_physion_pp_friction_collision_flush_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """friction_collision seg1-patient flush search + size hand-off to seg2.

        The seg1 patient is physically static, but it is reconstructed as a dynamic
        FoundationPose track (the whole scenario runs the CLEVRER way), so the standard
        static-fixture flush never sees it. This pass reuses the flush machinery on the
        patient's post-snap pose with flatten_base=False: the coordinate descent walks
        in-plane translation + yaw + in-plane stretch only, the height stays frozen at
        the ray-slide result (no base alignment, no ground pin -- the slide already
        grounded it), and the single refined pose is broadcast across the seg1 interval
        (the object never moves there). The post-flush mesh is the size authority for
        the SAME physical object in seg2: the seg2 patient adopts it on its support
        entry + trajectory item (SWR, the debug renders, and the drop refinement all
        read those); seg2 gets NO flush of its own.

        Writes the standard static_fixture_flush_refinement payload key (unset for this
        scenario otherwise) so the pose-step silhouette grid's flush column picks the
        result up unchanged."""
        fc = payload.get("physion_pp_friction_collision")
        if not isinstance(fc, dict) or fc.get("applies") is not True:
            return
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_friction_collision_seg1_patient_flush",
            "policy": (
                "coordinate descent over in-plane translation, yaw about the up axis, and "
                "in-plane extent scale, seeded from the ray-slide pose with the height frozen "
                "(no base alignment, no ground pin); refined pose broadcast across seg1; the "
                "post-flush mesh is adopted by the seg2 patient (same physical object)"
            ),
            "objective": "maximize mean IoU between rendered visible mesh masks and SAM3 masks",
            "objects": [],
        }
        payload["static_fixture_flush_refinement"] = summary
        roles = fc.get("role_object_ids") or {}
        seg1_patient = str(roles.get("seg1_patient") or "")
        seg2_patient = str(roles.get("seg2_patient") or "")
        if not seg1_patient:
            summary["reason"] = "no seg1 patient object id"
            return
        trajectory_correction = payload.get("trajectory_correction")
        corrected = (
            trajectory_correction.get("corrected_trajectories")
            if isinstance(trajectory_correction, dict)
            else None
        )
        if not isinstance(corrected, dict) or corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing or invalid gravity_direction_camera"
            return
        try:
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
        except Exception as exc:
            summary["reason"] = f"failed to load refinement inputs: {exc}"
            return
        support = payload.get("support_plane_position_correction")
        support_objects = {
            str(item.get("object_id")): item
            for item in (support.get("objects", []) if isinstance(support, dict) else [])
            if isinstance(item, dict)
        }
        trajectory_items = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        trajectory_item = trajectory_items.get(seg1_patient)
        if trajectory_item is None:
            summary["reason"] = "seg1 patient missing from corrected trajectories"
            return
        entry = self._refine_static_fixture_object(
            object_id=seg1_patient,
            trajectory_item=trajectory_item,
            support_item=support_objects.get(seg1_patient),
            mesh_path=mesh_paths.get(seg1_patient),
            mask_records=mask_records.get(seg1_patient) or {},
            mask_arrays=mask_arrays,
            intrinsics=intrinsics,
            image_shape=image_shape,
            up_axis=up_axis,
            question_dir=question_dir,
            flatten_base=False,
        )
        entry["role"] = "seg1_patient"
        summary["objects"].append(entry)
        summary["applied"] = entry.get("status") == "ok"
        refined_mesh = entry.get("refined_mesh_path")
        fc["flush.segment1_patient"] = {
            "status": entry.get("status"),
            "start_mean_iou": entry.get("start_mean_iou"),
            "refined_mean_iou": entry.get("refined_mean_iou"),
            "scale_u": entry.get("scale_u"),
            "scale_v": entry.get("scale_v"),
            "refined_mesh_path": refined_mesh,
        }
        if entry.get("status") != "ok":
            return
        # _refine_static_fixture_object already broadcast the refined pose across the
        # trajectory item and re-pointed the seg1 support entry at the rescaled mesh;
        # carry the mesh on the trajectory item too so the debug renders prefer it over
        # the foundationpose fallback (bounce-refinement convention).
        if refined_mesh:
            trajectory_item["mesh_path"] = str(refined_mesh)
            trajectory_item["mesh_source"] = "physion_pp_friction_collision_seg1_patient_flush"
            # Size hand-off: the seg2 patient is the same physical object, so it must
            # render and simulate at exactly the seg1 post-flush size. In-plane stretch
            # leaves the support-axis extent untouched, so the seg2 ground contact from
            # the ray slide survives the swap unchanged.
            if seg2_patient:
                seg2_support = support_objects.get(seg2_patient)
                if isinstance(seg2_support, dict):
                    seg2_support["source_mesh_path"] = seg2_support.get("mesh_path")
                    seg2_support["mesh_path"] = str(refined_mesh)
                    seg2_support["mesh_source"] = "physion_pp_friction_collision_seg1_patient_flush"
                seg2_item = trajectory_items.get(seg2_patient)
                if isinstance(seg2_item, dict):
                    seg2_item["mesh_path"] = str(refined_mesh)
                    seg2_item["mesh_source"] = "physion_pp_friction_collision_seg1_patient_flush"
                fc["seg2_patient_mesh_adopted_from_seg1_flush"] = True
        _log_tool(
            self.tool_name,
            f"friction_collision seg1_patient_flush status={entry.get('status')} "
            f"iou {entry.get('start_mean_iou')} -> {entry.get('refined_mean_iou')} "
            f"scale=({entry.get('scale_u')}, {entry.get('scale_v')}) mesh={refined_mesh}",
        )

    def _apply_physion_pp_mass_collision_flush_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """mass_collision seg1 kept-extra flush search + mesh hand-off to the seg2 patient.

        The kept extras are physically parked objects reconstructed as dynamic
        FoundationPose tracks (the whole scenario runs the CLEVRER way), so the standard
        static flush never sees them. This pass reuses the flush machinery on each
        extra's post-snap pose with flatten_base=False: in-plane translation + yaw +
        in-plane stretch only, height frozen at the ray-slide result (no base alignment,
        no ground pin -- the slide already grounded it), refined pose broadcast across
        seg1. The VLM-linked extra is the SAME physical object as the seg2 patient:
        its post-flush mesh is adopted on the patient's support entry + trajectory item,
        locking the patient size before the drop refinement (branch A: yaw-only search).
        Unlinked extras are still flushed (they are scene content either way).

        Writes the standard static_fixture_flush_refinement payload key (unset for this
        scenario otherwise) so the pose-step silhouette grid's flush column picks the
        result up unchanged."""
        mc = payload.get("physion_pp_mass_collision")
        if not isinstance(mc, dict) or mc.get("applies") is not True:
            return
        mesh_adopt_route = _require_mass_extra_mesh_adopt_route(
            policy_benchmark="physion_pp",
            scenario=self._object_plan_scenario(object_plan),
            object_plan=object_plan,
        )
        if not isinstance(mesh_adopt_route, dict):
            raise ValueError("mass_collision mesh adopt route was not resolved")
        mesh_adoption_module = self._module_for_route_record(
            mesh_adopt_route,
            decision_id=MASS_EXTRA_MESH_ADOPT_DECISION_ID,
            module_name="mesh_adoption",
        )
        if (
            mesh_adoption_module.implementation
            != "adopt_linked_extra_final_mesh_for_patient"
        ):
            raise ValueError(
                "unsupported mass mesh-adoption module implementation: "
                f"{mesh_adoption_module.implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_mass_collision_extra_flush",
            "policy": (
                "coordinate descent over in-plane translation, yaw about the up axis, and "
                "in-plane extent scale per kept extra, seeded from the ray-slide pose with "
                "the height frozen (no base alignment, no ground pin); refined pose "
                "broadcast across seg1; the VLM-linked extra's post-flush mesh is adopted "
                "by the seg2 patient (same physical object)"
            ),
            "objective": "maximize mean IoU between rendered visible mesh masks and SAM3 masks",
            "objects": [],
        }
        payload["static_fixture_flush_refinement"] = summary
        extras = [
            entry
            for entry in mc.get("extras") or []
            if isinstance(entry, dict) and entry.get("object_id")
        ]
        if not extras:
            summary["reason"] = "no kept extra objects"
            return
        trajectory_correction = payload.get("trajectory_correction")
        corrected = (
            trajectory_correction.get("corrected_trajectories")
            if isinstance(trajectory_correction, dict)
            else None
        )
        if not isinstance(corrected, dict) or corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing or invalid gravity_direction_camera"
            return
        try:
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
        except Exception as exc:
            summary["reason"] = f"failed to load refinement inputs: {exc}"
            return
        support = payload.get("support_plane_position_correction")
        support_objects = {
            str(item.get("object_id")): item
            for item in (support.get("objects", []) if isinstance(support, dict) else [])
            if isinstance(item, dict)
        }
        trajectory_items = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        linked_extra = str(mc.get("linked_extra_object_id") or "")
        seg2_patient = str((mc.get("role_object_ids") or {}).get("seg2_patient") or "")
        adopt_summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_mass_collision_linked_extra_mesh_adopt",
            "linked_extra_object_id": linked_extra or None,
            "seg2_patient_object_id": seg2_patient or None,
            "resolved_route": deepcopy(mesh_adopt_route),
        }
        if not linked_extra:
            adopt_summary["reason"] = "TRK-005 did not link an extra to the seg2 patient"
        elif not seg2_patient:
            adopt_summary["reason"] = "missing seg2 patient object id"
        mc["extra_mesh_adopt"] = adopt_summary
        flush_by_object: Dict[str, Any] = {}
        for extra in extras:
            object_id = str(extra["object_id"])
            trajectory_item = trajectory_items.get(object_id)
            if trajectory_item is None:
                summary["objects"].append(
                    {"object_id": object_id, "status": "skipped", "reason": "missing from corrected trajectories"}
                )
                continue
            entry = self._refine_static_fixture_object(
                object_id=object_id,
                trajectory_item=trajectory_item,
                support_item=support_objects.get(object_id),
                mesh_path=mesh_paths.get(object_id),
                mask_records=mask_records.get(object_id) or {},
                mask_arrays=mask_arrays,
                intrinsics=intrinsics,
                image_shape=image_shape,
                up_axis=up_axis,
                question_dir=question_dir,
                flatten_base=False,
            )
            entry["role"] = "seg1_extra"
            summary["objects"].append(entry)
            flush_by_object[object_id] = {
                "status": entry.get("status"),
                "start_mean_iou": entry.get("start_mean_iou"),
                "refined_mean_iou": entry.get("refined_mean_iou"),
                "scale_u": entry.get("scale_u"),
                "scale_v": entry.get("scale_v"),
                "refined_mesh_path": entry.get("refined_mesh_path"),
            }
            refined_mesh = entry.get("refined_mesh_path")
            if entry.get("status") != "ok" or not refined_mesh:
                continue
            # _refine_static_fixture_object already broadcast the refined pose and
            # re-pointed the support entry; carry the mesh on the trajectory item too so
            # the debug renders prefer it (bounce-refinement convention).
            trajectory_item["mesh_path"] = str(refined_mesh)
            trajectory_item["mesh_source"] = "physion_pp_mass_collision_extra_flush"
            if (
                mesh_adopt_route.get("route") == MASS_EXTRA_MESH_ADOPT_ROUTE
                and object_id == linked_extra
                and seg2_patient
            ):
                # Size hand-off: the seg2 patient is the same physical object, so it
                # must render and simulate at exactly the post-flush size. In-plane
                # stretch leaves the support-axis extent untouched, so the drop
                # refinement's ground clearance is unaffected by the swap.
                patient_support = support_objects.get(seg2_patient)
                if isinstance(patient_support, dict):
                    patient_support["source_mesh_path"] = patient_support.get("mesh_path")
                    patient_support["mesh_path"] = str(refined_mesh)
                    patient_support["mesh_source"] = "physion_pp_mass_collision_extra_flush"
                patient_item = trajectory_items.get(seg2_patient)
                if isinstance(patient_item, dict):
                    patient_item["mesh_path"] = str(refined_mesh)
                    patient_item["mesh_source"] = "physion_pp_mass_collision_extra_flush"
                mc["seg2_patient_mesh_adopted_from_seg1_flush"] = True
                adopt_summary.update(
                    {
                        "applied": True,
                        "refined_mesh_path": str(refined_mesh),
                    }
                )
                adopt_summary.pop("reason", None)
        summary["applied"] = any(
            isinstance(entry, dict) and entry.get("status") == "ok" for entry in summary["objects"]
        )
        mc["extra_flush"] = flush_by_object
        if linked_extra and seg2_patient and adopt_summary.get("applied") is not True:
            adopt_summary["reason"] = (
                "linked extra flush did not export a replacement mesh; the existing "
                "GEO-002 reused patient mesh remains in effect"
            )
        _log_tool(
            self.tool_name,
            f"mass_collision extra_flush objects={list(flush_by_object)} "
            f"adopted_by_patient={mc.get('seg2_patient_mesh_adopted_from_seg1_flush') is True}",
        )

    def _collision_patient_drop_implementation(
        self,
        *,
        object_plan: ObjectPlan,
        resolved_route: Dict[str, Any],
    ) -> str:
        module = self._module_for_route_record(
            resolved_route,
            decision_id=COLLISION_PATIENT_DROP_DECISION_ID,
            module_name="patient_drop",
        )
        if module.implementation != "scenario_selected_vertical_drop":
            raise ValueError(
                "unsupported patient-drop module implementation: "
                f"{module.implementation!r}"
            )
        scenario = self._object_plan_scenario(object_plan).strip().lower()
        return module.require_string(scenario)

    def _apply_physion_pp_friction_collision_drop_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """friction_collision moving seg2 patient: vertical-drop trajectory from masks.

        When the agent knocks the patient off the ledge before the cut, seg2 opens with
        the patient in mid-air, where VDA metric depth (and therefore the FP register
        depth) is untrustworthy. The trajectory is re-derived from image evidence only:
        - anchor frame (first solvable seg2 mask frame): the mask centroid ray pins the
          object to one line of sight; the mesh is straightened (support axis // up,
          taken from the rotation-corrected pose) and a (yaw, ray-depth) grid maximizes
          the rendered-mask IoU -> the initial pose.
        - subsequent frames: the ground-plane projection and the yaw stay FROZEN
          (scenario prior: the fall is vertical); each frame searches the height along
          the up axis only, seeded by velocity extrapolation, scored by mask IoU.
        No landing anchor and no ground clamp: seg2 never observes the landing.
        Static seg2 patients keep the standard FP + straighten + snap trajectory."""
        fc = payload.get("physion_pp_friction_collision")
        if not isinstance(fc, dict) or fc.get("applies") is not True:
            return
        drop_route = _require_collision_patient_drop_route(
            policy_benchmark="physion_pp",
            scenario=self._object_plan_scenario(object_plan),
            object_plan=object_plan,
        )
        if not isinstance(drop_route, dict):
            raise ValueError("friction_collision patient-drop route was not resolved")
        drop_implementation = self._collision_patient_drop_implementation(
            object_plan=object_plan,
            resolved_route=drop_route,
        )
        if drop_implementation != "mask_anchor_vertical_height_search":
            raise ValueError(
                "unsupported friction-collision patient-drop scenario variant: "
                f"{drop_implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_friction_collision_drop_refinement",
            "resolved_route": deepcopy(drop_route),
        }
        fc["drop_refinement"] = summary
        motion = fc.get("seg2_patient_motion") or {}
        if motion.get("moving") is not True:
            summary["reason"] = (
                "seg2 patient is static (centroid displacement below threshold); "
                "standard FP + straighten + snap trajectory kept"
            )
            return
        roles = fc.get("role_object_ids") or {}
        seg2_patient = str(roles.get("seg2_patient") or "")
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        item = next(
            (
                candidate
                for candidate in corrected.get("objects", [])
                if isinstance(candidate, dict) and str(candidate.get("object_id")) == seg2_patient
            ),
            None,
        )
        if item is None:
            summary["reason"] = "seg2 patient missing from corrected trajectories"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        try:
            import trimesh

            support = payload.get("support_plane_position_correction") or {}
            support_item = next(
                (
                    candidate
                    for candidate in support.get("objects", [])
                    if isinstance(candidate, dict) and str(candidate.get("object_id")) == seg2_patient
                ),
                None,
            )
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mesh_path = (
                item.get("mesh_path")
                or (support_item or {}).get("mesh_path")
                or mesh_paths.get(seg2_patient)
            )
            if not mesh_path:
                summary["reason"] = "missing seg2 patient mesh"
                return
            verts_full, faces_full = self._load_mesh_geometry(Path(mesh_path))
            mesh_centroid = verts_full.mean(axis=0)
            verts_search, faces_search = verts_full, faces_full
            if len(faces_full) > PHYSION_PP_FC_DROP_DECIMATE_FACES:
                try:
                    simplified = trimesh.Trimesh(
                        vertices=verts_full, faces=faces_full, process=False
                    ).simplify_quadric_decimation(PHYSION_PP_FC_DROP_DECIMATE_FACES)
                    if len(simplified.faces) >= 8:
                        verts_search = np.asarray(simplified.vertices, dtype=np.float64)
                        faces_search = np.asarray(simplified.faces)
                except Exception:
                    pass

            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)

            target_masks: Dict[int, np.ndarray] = {}
            for frame_index, record in (mask_records.get(seg2_patient) or {}).items():
                mask = self._resize_mask_to_shape(
                    mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                )
                if mask is not None and int(mask.sum()) >= PHYSION_PP_FC_DROP_MIN_MASK_PX:
                    target_masks[int(frame_index)] = mask
            frames = sorted(target_masks)
            if len(frames) < 2:
                summary["reason"] = f"too few solvable seg2 patient mask frames ({len(frames)})"
                return

            poses_by_frame: Dict[int, Dict[str, Any]] = {}
            for pose_entry in item.get("poses", []):
                frame_index = self._pose_frame_index(pose_entry)
                if frame_index is not None:
                    poses_by_frame[frame_index] = pose_entry

            anchor_frame = frames[0]
            # Straightened base rotation: the rotation correction already aligned the
            # patient's closest local axis to up; reuse the earliest corrected rotation
            # and let the anchor grid search only the yaw about the up axis.
            base_rotation = None
            for frame_index in frames:
                pose_entry = poses_by_frame.get(frame_index)
                if pose_entry is not None and pose_entry.get("corrected_pose_4x4") is not None:
                    base_rotation = np.asarray(
                        pose_entry["corrected_pose_4x4"], dtype=np.float64
                    ).reshape(4, 4)[:3, :3]
                    break
            if base_rotation is None:
                summary["reason"] = "no corrected pose to seed the straightened rotation"
                return

            def _iou_at(
                center: np.ndarray, rotation: np.ndarray, frame_index: int, verts: np.ndarray, faces: np.ndarray
            ) -> Optional[float]:
                translation = center - rotation @ mesh_centroid
                intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                rendered, _ = render_mesh_depth(
                    vertices_camera=verts @ rotation.T + translation,
                    faces=faces,
                    intrinsic=intrinsic,
                    image_shape=image_shape,
                )
                return self._mask_iou(rendered, target_masks[frame_index])

            anchor_mask = target_masks[anchor_frame]
            ys, xs = np.nonzero(anchor_mask)
            anchor_intrinsic = np.asarray(
                self._intrinsic_for_frame(intrinsics, anchor_frame), dtype=np.float64
            )
            anchor_ray = np.linalg.inv(anchor_intrinsic) @ np.array(
                [float(xs.mean()), float(ys.mean()), 1.0]
            )
            anchor_ray = anchor_ray / max(float(np.linalg.norm(anchor_ray)), 1e-12)

            # Depth seed: median VDA depth inside the anchor mask. Mid-air it is biased,
            # but it brackets the coarse-to-fine ray-depth search well; fall back to the
            # corrected pose's centroid depth when the depth map is unusable.
            depth_seed = None
            depth_map = self._depth_for_frame(metric_depth, anchor_frame)
            if depth_map is not None and depth_map.shape == anchor_mask.shape:
                depth_values = np.asarray(depth_map)[anchor_mask]
                depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 1e-3)]
                if depth_values.size:
                    depth_seed = float(np.median(depth_values))
            if depth_seed is None:
                pose_entry = poses_by_frame.get(anchor_frame)
                if pose_entry is not None and pose_entry.get("corrected_pose_4x4") is not None:
                    matrix = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    centroid_camera = matrix[:3, :3] @ mesh_centroid + matrix[:3, 3]
                    projected = float(centroid_camera @ anchor_ray)
                    if np.isfinite(projected) and projected > 1e-3:
                        depth_seed = projected
            if depth_seed is None:
                summary["reason"] = "no usable anchor depth seed"
                return

            depth_lo, depth_hi, depth_steps = PHYSION_PP_FC_DROP_DEPTH_COARSE
            depth_step = (depth_hi - depth_lo) / max(int(depth_steps) - 1, 1)
            best = {"iou": -1.0, "depth": None, "depth_factor": None, "yaw": 0.0}
            yaw_values = np.deg2rad(np.arange(0.0, 360.0, PHYSION_PP_FC_DROP_YAW_COARSE_DEG))
            for depth_factor in np.linspace(depth_lo, depth_hi, int(depth_steps)):
                depth = depth_seed * float(depth_factor)
                center = anchor_ray * float(depth)
                for yaw in yaw_values:
                    rotation = self._axis_angle_rotation(up_axis, float(yaw)) @ base_rotation
                    iou = _iou_at(center, rotation, anchor_frame, verts_search, faces_search)
                    if iou is not None and iou > best["iou"]:
                        best = {
                            "iou": float(iou),
                            "depth": float(depth),
                            "depth_factor": float(depth_factor),
                            "yaw": float(yaw),
                        }
            if best["depth"] is None:
                summary["reason"] = "anchor (yaw, depth) search produced no valid IoU"
                return
            initial_winner_factor = float(best["depth_factor"])
            expanded_factors, expanded_direction = _boundary_extension_values(
                winner=initial_winner_factor,
                initial_start=depth_lo,
                initial_stop=depth_hi,
                step=depth_step,
                expanded_start=PHYSION_PP_FC_DROP_DEPTH_EXPANDED[0],
                expanded_stop=PHYSION_PP_FC_DROP_DEPTH_EXPANDED[1],
                boundary_steps=PHYSION_PP_FC_DROP_DEPTH_BOUNDARY_STEPS,
            )
            for depth_factor in expanded_factors:
                depth = depth_seed * float(depth_factor)
                center = anchor_ray * float(depth)
                for yaw in yaw_values:
                    rotation = self._axis_angle_rotation(up_axis, float(yaw)) @ base_rotation
                    iou = _iou_at(center, rotation, anchor_frame, verts_search, faces_search)
                    if iou is not None and iou > best["iou"]:
                        best = {
                            "iou": float(iou),
                            "depth": float(depth),
                            "depth_factor": float(depth_factor),
                            "yaw": float(yaw),
                        }
            coarse_winner_factor = float(best["depth_factor"])
            fine_lo, fine_hi, fine_steps = PHYSION_PP_FC_DROP_DEPTH_FINE
            for yaw_span in PHYSION_PP_FC_DROP_YAW_FINE_DEG:
                yaw_offsets = np.deg2rad([-yaw_span, 0.0, yaw_span])
                for depth in np.linspace(best["depth"] * fine_lo, best["depth"] * fine_hi, int(fine_steps)):
                    center = anchor_ray * float(depth)
                    for yaw_offset in yaw_offsets:
                        yaw = best["yaw"] + float(yaw_offset)
                        rotation = self._axis_angle_rotation(up_axis, yaw) @ base_rotation
                        iou = _iou_at(center, rotation, anchor_frame, verts_search, faces_search)
                        if iou is not None and iou > best["iou"]:
                            best = {
                                "iou": float(iou),
                                "depth": float(depth),
                                "depth_factor": float(depth / depth_seed),
                                "yaw": yaw,
                            }

            drop_rotation = self._axis_angle_rotation(up_axis, best["yaw"]) @ base_rotation
            anchor_center = anchor_ray * best["depth"]
            anchor_height = float(anchor_center @ up_axis)
            horizontal_component = anchor_center - anchor_height * up_axis

            down_span, up_span, coarse_step = PHYSION_PP_FC_DROP_HEIGHT_COARSE_M
            fine_span, fine_step = PHYSION_PP_FC_DROP_HEIGHT_FINE_M
            centers: Dict[int, np.ndarray] = {anchor_frame: anchor_center}
            heights: Dict[int, float] = {anchor_frame: anchor_height}
            solved_frames = [anchor_frame]
            per_frame_iou: Dict[int, float] = {anchor_frame: best["iou"]}
            for frame_index in frames[1:]:
                last_frame = solved_frames[-1]
                predicted = heights[last_frame]
                if len(solved_frames) >= 2:
                    prev_frame = solved_frames[-2]
                    gap = max(last_frame - prev_frame, 1)
                    velocity = (heights[last_frame] - heights[prev_frame]) / gap
                    predicted = heights[last_frame] + velocity * (frame_index - last_frame)
                best_height, best_iou = None, -1.0
                for height in np.arange(predicted - down_span, predicted + up_span + 1e-9, coarse_step):
                    center = horizontal_component + float(height) * up_axis
                    if float(center[2]) <= 0.05:
                        continue
                    iou = _iou_at(center, drop_rotation, frame_index, verts_search, faces_search)
                    if iou is not None and iou > best_iou:
                        best_iou, best_height = float(iou), float(height)
                if best_height is None:
                    continue  # frame left unrefined
                for height in np.arange(
                    best_height - fine_span, best_height + fine_span + 1e-9, fine_step
                ):
                    center = horizontal_component + float(height) * up_axis
                    if float(center[2]) <= 0.05:
                        continue
                    iou = _iou_at(center, drop_rotation, frame_index, verts_search, faces_search)
                    if iou is not None and iou > best_iou:
                        best_iou, best_height = float(iou), float(height)
                heights[frame_index] = best_height
                centers[frame_index] = horizontal_component + best_height * up_axis
                per_frame_iou[frame_index] = best_iou
                solved_frames.append(frame_index)

            refined_entries = 0
            before_ious: list[float] = []
            after_ious: list[float] = []
            for pose_entry in item.get("poses", []):
                frame_index = self._pose_frame_index(pose_entry)
                if frame_index is None:
                    continue
                if frame_index not in centers:
                    pose_entry["fc_drop_refined"] = False
                    continue
                if pose_entry.get("corrected_pose_4x4") is not None:
                    old = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    old_center = old[:3, :3] @ mesh_centroid + old[:3, 3]
                    old_iou = _iou_at(old_center, old[:3, :3], frame_index, verts_full, faces_full)
                    if old_iou is not None:
                        before_ious.append(float(old_iou))
                translation = centers[frame_index] - drop_rotation @ mesh_centroid
                pose = np.eye(4)
                pose[:3, :3] = drop_rotation
                pose[:3, 3] = translation
                pose_entry["corrected_pose_4x4"] = pose.tolist()
                pose_entry["corrected_translation_camera"] = translation.tolist()
                pose_entry["fc_drop_refined"] = True
                refined_entries += 1
                new_iou = _iou_at(centers[frame_index], drop_rotation, frame_index, verts_full, faces_full)
                if new_iou is not None:
                    after_ious.append(float(new_iou))

            summary.update(
                {
                    "applied": refined_entries > 0,
                    "object_id": seg2_patient,
                    "policy": (
                        "first-frame centroid-ray (yaw, ray-depth) IoU anchor on the straightened "
                        "mesh; ground-plane projection and yaw frozen; per-frame height-only IoU "
                        "descent with velocity extrapolation; no landing anchor, no ground clamp"
                    ),
                    "mesh_path": str(mesh_path),
                    "anchor_frame": int(anchor_frame),
                    "anchor_depth_seed_m": round(depth_seed, 4),
                    "anchor_depth_m": round(float(best["depth"]), 4),
                    "anchor_depth_search": {
                        "initial_range_factor": [depth_lo, depth_hi],
                        "boundary_band_steps": PHYSION_PP_FC_DROP_DEPTH_BOUNDARY_STEPS,
                        "initial_winner_factor": round(initial_winner_factor, 4),
                        "expanded": bool(expanded_factors),
                        "expanded_direction": expanded_direction,
                        "evaluated_coarse_range_factor": [
                            round(min([depth_lo, *expanded_factors]), 4),
                            round(max([depth_hi, *expanded_factors]), 4),
                        ],
                        "coarse_winner_factor": round(coarse_winner_factor, 4),
                        "final_winner_factor": round(float(best["depth_factor"]), 4),
                    },
                    "anchor_yaw_deg": round(float(np.degrees(best["yaw"])) % 360.0, 2),
                    "anchor_iou": round(float(best["iou"]), 4),
                    "mask_frames": len(frames),
                    "solved_frames": len(centers),
                    "refined_pose_entries": refined_entries,
                    "before_iou_mean": round(float(np.mean(before_ious)), 4) if before_ious else None,
                    "after_iou_mean": round(float(np.mean(after_ious)), 4) if after_ious else None,
                    "drop_height_total_m": round(float(heights[solved_frames[0]] - heights[solved_frames[-1]]), 4),
                }
            )
            _log_tool(
                self.tool_name,
                f"friction_collision drop_refinement {seg2_patient} frames={len(centers)}/{len(frames)} "
                f"anchor(depth={summary['anchor_depth_m']} yaw={summary['anchor_yaw_deg']} iou={summary['anchor_iou']}) "
                f"iou {summary['before_iou_mean']} -> {summary['after_iou_mean']}",
            )
        except Exception as exc:
            summary["reason"] = f"drop refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(
                self.tool_name,
                f"friction_collision drop_refinement error={_short_text(str(exc), 300)}",
            )

    def _apply_physion_pp_mass_collision_drop_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """mass_collision moving seg2 patient: vertical-drop trajectory from masks and
        the ball-agent line prior.

        The patient drops in from above, so VDA metric depth on it is untrustworthy and
        the ray slide is exempted. Scenario prior: the fall is strictly vertical (yaw
        and ground projection fixed) and the landing point lies on the line between the
        ball's seg2 start and the agent. Both anchors rest on the floor at the seg2
        start, so their corrected (rotation + ray-slide) centers are depth-trustworthy:
        the vertical plane through them pins each mask-centroid ray to a 3D center with
        ZERO depth dependence. The ground projection is then frozen at the per-frame
        median along-line coordinate and each height re-solves as the closest approach
        of the centroid ray to that fixed vertical line.
        - branch A (mesh adopted from the linked extra's flush): size locked, ONE global
          yaw searched by whole-trajectory mask IoU.
        - branch B (own seg2 reconstruction): yaw frozen at the straightened anchor
          rotation, ONE global scale (about the mesh centroid) searched by
          whole-trajectory mask IoU; the scaled mesh is exported and adopted on the
          support entry + trajectory item.
        A degenerate plane (ball start and agent nearly coincident on the ground)
        signals an upstream error: deliberately NO fallback -- the patient stays
        unrefined and the summary flags plane_degenerate for human review."""
        mc = payload.get("physion_pp_mass_collision")
        if not isinstance(mc, dict) or mc.get("applies") is not True:
            return
        drop_route = _require_collision_patient_drop_route(
            policy_benchmark="physion_pp",
            scenario=self._object_plan_scenario(object_plan),
            object_plan=object_plan,
        )
        if not isinstance(drop_route, dict):
            raise ValueError("mass_collision patient-drop route was not resolved")
        drop_implementation = self._collision_patient_drop_implementation(
            object_plan=object_plan,
            resolved_route=drop_route,
        )
        if drop_implementation != "ball_agent_plane_vertical_raycast":
            raise ValueError(
                "unsupported mass-collision patient-drop scenario variant: "
                f"{drop_implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_mass_collision_drop_refinement",
            "resolved_route": deepcopy(drop_route),
        }
        mc["drop_refinement"] = summary
        motion = mc.get("seg2_patient_motion") or {}
        if motion.get("moving") is not True:
            summary["reason"] = (
                "seg2 patient is static (centroid displacement below threshold); "
                "standard FP + straighten + snap trajectory kept"
            )
            return
        roles = mc.get("role_object_ids") or {}
        seg2_patient = str(roles.get("seg2_patient") or "")
        seg2_agent = str(roles.get("seg2_agent") or "")
        seg2_ball = str(roles.get("seg2_ball") or "")
        if not seg2_patient or not seg2_agent or not seg2_ball:
            summary["reason"] = "missing seg2 patient/agent/ball object ids"
            return
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        objects_by_id = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        item = objects_by_id.get(seg2_patient)
        if item is None:
            summary["reason"] = "seg2 patient missing from corrected trajectories"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        try:
            import trimesh

            support = payload.get("support_plane_position_correction") or {}
            support_by_id = {
                str(entry.get("object_id")): entry
                for entry in support.get("objects", [])
                if isinstance(entry, dict)
            }
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mesh_path = (
                item.get("mesh_path")
                or (support_by_id.get(seg2_patient) or {}).get("mesh_path")
                or mesh_paths.get(seg2_patient)
            )
            if not mesh_path:
                summary["reason"] = "missing seg2 patient mesh"
                return

            # --- prior plane anchors: earliest corrected seg2 centers of ball and agent
            #     (both post-snap, i.e. ray-slid onto the fitted ground -> depth-clean) ---
            seg2_range = (mc.get("two_segment") or {}).get("seg2")

            def _earliest_corrected_center(object_id: str) -> Optional[np.ndarray]:
                entry = objects_by_id.get(object_id)
                if entry is None:
                    return None
                anchor_mesh = (
                    (support_by_id.get(object_id) or {}).get("mesh_path")
                    or entry.get("mesh_path")
                    or mesh_paths.get(object_id)
                )
                if not anchor_mesh:
                    return None
                vertices, _faces = self._load_mesh_geometry(Path(anchor_mesh))
                centroid = vertices.mean(axis=0)
                poses = sorted(
                    (
                        pose
                        for pose in entry.get("poses", [])
                        if isinstance(pose, dict)
                        and pose.get("corrected_pose_4x4") is not None
                        and self._pose_frame_index(pose) is not None
                    ),
                    key=lambda pose: self._pose_frame_index(pose),
                )
                if isinstance(seg2_range, (list, tuple)) and len(seg2_range) == 2:
                    lo, hi = int(seg2_range[0]), int(seg2_range[1])
                    poses = [pose for pose in poses if lo <= self._pose_frame_index(pose) <= hi]
                if not poses:
                    return None
                matrix = np.asarray(poses[0]["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                return matrix[:3, :3] @ centroid + matrix[:3, 3]

            ball_center = _earliest_corrected_center(seg2_ball)
            agent_center = _earliest_corrected_center(seg2_agent)
            if ball_center is None or agent_center is None:
                summary["reason"] = "missing corrected seg2 ball/agent anchor centers"
                return
            gu, gv = _plane_basis_for_up_axis(up_axis)
            ball_2d = np.array([float(ball_center @ gu), float(ball_center @ gv)])
            agent_2d = np.array([float(agent_center @ gu), float(agent_center @ gv)])
            span = float(np.linalg.norm(agent_2d - ball_2d))
            summary["plane_anchor_span_m"] = round(span, 4)
            if span < PHYSION_PP_MC_PLANE_MIN_SPAN_M:
                # Coincident anchors mean something upstream is wrong (the ball is fired
                # AT the agent from a distance by construction). Deliberately NO
                # fallback: flag for human review and keep the patient unrefined.
                summary["plane_degenerate"] = True
                summary["reason"] = f"ball/agent ground projections nearly coincide ({span:.3f} m)"
                return
            direction_2d = (agent_2d - ball_2d) / span
            line_dir = float(direction_2d[0]) * gu + float(direction_2d[1]) * gv
            plane_normal = float(-direction_2d[1]) * gu + float(direction_2d[0]) * gv
            c_offset = float(
                np.mean([float(ball_center @ plane_normal), float(agent_center @ plane_normal)])
            )

            # --- per-frame mask-centroid rays ∩ plane -> along-line/height coordinates ---
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            target_masks: Dict[int, np.ndarray] = {}
            rays: Dict[int, np.ndarray] = {}
            s_values: Dict[int, float] = {}
            for frame_index, record in (mask_records.get(seg2_patient) or {}).items():
                mask = self._resize_mask_to_shape(
                    mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                )
                if mask is None or int(mask.sum()) < PHYSION_PP_MC_DROP_MIN_MASK_PX:
                    continue
                ys, xs = np.nonzero(mask)
                intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                ray = np.linalg.inv(intrinsic) @ np.array([float(xs.mean()), float(ys.mean()), 1.0])
                ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
                denom = float(ray @ plane_normal)
                if abs(denom) < PHYSION_PP_MC_DROP_MIN_RAY_PLANE_DOT:
                    continue
                depth_scale = c_offset / denom
                if depth_scale <= 0:
                    continue
                target_masks[int(frame_index)] = mask
                rays[int(frame_index)] = ray
                s_values[int(frame_index)] = float((ray * depth_scale) @ line_dir)
            if len(target_masks) < PHYSION_PP_MC_DROP_MIN_FRAMES:
                summary["reason"] = f"too few solvable seg2 patient mask frames ({len(target_masks)})"
                return

            # Freeze the ground projection at the median along-line coordinate; each
            # height re-solves as the closest approach of the centroid ray to the fixed
            # vertical line q(h) = base_point + h*up.
            s_star = float(np.median(list(s_values.values())))
            base_point = c_offset * plane_normal + s_star * line_dir
            centers: Dict[int, np.ndarray] = {}
            for frame_index, ray in rays.items():
                ray_dot_up = float(ray @ up_axis)
                denom2 = 1.0 - ray_dot_up * ray_dot_up
                if denom2 < 1e-6:
                    continue
                t = (float(ray @ base_point) - float(up_axis @ base_point) * ray_dot_up) / denom2
                if t <= 1e-3:
                    continue
                height = t * ray_dot_up - float(up_axis @ base_point)
                centers[frame_index] = base_point + height * up_axis
            frames = sorted(centers)
            if len(frames) < PHYSION_PP_MC_DROP_MIN_FRAMES:
                summary["reason"] = f"too few ray-line solvable frames ({len(frames)})"
                return
            s_series = [s_values[frame] for frame in frames]
            summary["s_star_m"] = round(s_star, 4)
            summary["s_spread_m"] = round(float(np.max(s_series) - np.min(s_series)), 4)

            # --- mesh + straightened base rotation ---
            verts_full, faces_full = self._load_mesh_geometry(Path(mesh_path))
            mesh_centroid = verts_full.mean(axis=0)
            verts_search, faces_search = verts_full, faces_full
            if len(faces_full) > PHYSION_PP_MC_DROP_DECIMATE_FACES:
                try:
                    simplified = trimesh.Trimesh(
                        vertices=verts_full, faces=faces_full, process=False
                    ).simplify_quadric_decimation(PHYSION_PP_MC_DROP_DECIMATE_FACES)
                    if len(simplified.faces) >= 8:
                        verts_search = np.asarray(simplified.vertices, dtype=np.float64)
                        faces_search = np.asarray(simplified.faces)
                except Exception:
                    pass
            poses_by_frame: Dict[int, Dict[str, Any]] = {}
            for pose_entry in item.get("poses", []):
                frame_index = self._pose_frame_index(pose_entry)
                if frame_index is not None:
                    poses_by_frame[frame_index] = pose_entry
            base_rotation = None
            for frame_index in frames:
                pose_entry = poses_by_frame.get(frame_index)
                if pose_entry is not None and pose_entry.get("corrected_pose_4x4") is not None:
                    base_rotation = np.asarray(
                        pose_entry["corrected_pose_4x4"], dtype=np.float64
                    ).reshape(4, 4)[:3, :3]
                    break
            if base_rotation is None:
                for pose_entry in item.get("poses", []):
                    if isinstance(pose_entry, dict) and pose_entry.get("corrected_pose_4x4") is not None:
                        base_rotation = np.asarray(
                            pose_entry["corrected_pose_4x4"], dtype=np.float64
                        ).reshape(4, 4)[:3, :3]
                        break
            if base_rotation is None:
                summary["reason"] = "no corrected pose to seed the straightened rotation"
                return

            def _iou_at(
                center: np.ndarray, rotation: np.ndarray, frame_index: int, verts: np.ndarray, faces: np.ndarray
            ) -> Optional[float]:
                translation = center - rotation @ mesh_centroid
                intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                rendered, _ = render_mesh_depth(
                    vertices_camera=verts @ rotation.T + translation,
                    faces=faces,
                    intrinsic=intrinsic,
                    image_shape=image_shape,
                )
                return self._mask_iou(rendered, target_masks[frame_index])

            def _mean_iou(rotation: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> Optional[float]:
                ious = [
                    _iou_at(centers[frame], rotation, frame, verts, faces) for frame in frames
                ]
                ious = [value for value in ious if value is not None]
                return float(np.mean(ious)) if ious else None

            # Branch A whenever the patient's size is locked to the extra -- either the
            # flush hand-off succeeded OR the plan-stage reuse already redirected the
            # patient onto the extra's mesh (flush may fail without unlocking the size).
            adopted = (
                mc.get("seg2_patient_mesh_adopted_from_seg1_flush") is True
                or mc.get("patient_mesh_reused") is True
            )
            final_scale = 1.0
            scaled_mesh_path: Optional[Path] = None
            if adopted:
                # --- branch A: size locked by the seg1 flush hand-off; ONE global yaw ---
                branch = "A_adopted_mesh_search_yaw"
                best = {"iou": -1.0, "yaw": 0.0}
                for yaw in np.deg2rad(np.arange(0.0, 360.0, PHYSION_PP_MC_DROP_YAW_COARSE_DEG)):
                    rotation = self._axis_angle_rotation(up_axis, float(yaw)) @ base_rotation
                    mean_iou = _mean_iou(rotation, verts_search, faces_search)
                    if mean_iou is not None and mean_iou > best["iou"]:
                        best = {"iou": mean_iou, "yaw": float(yaw)}
                if best["iou"] < 0:
                    summary["reason"] = "global yaw search produced no valid IoU"
                    return
                for yaw_span in PHYSION_PP_MC_DROP_YAW_FINE_DEG:
                    for yaw_offset in np.deg2rad([-yaw_span, yaw_span]):
                        yaw = best["yaw"] + float(yaw_offset)
                        rotation = self._axis_angle_rotation(up_axis, yaw) @ base_rotation
                        mean_iou = _mean_iou(rotation, verts_search, faces_search)
                        if mean_iou is not None and mean_iou > best["iou"]:
                            best = {"iou": mean_iou, "yaw": yaw}
                drop_rotation = self._axis_angle_rotation(up_axis, best["yaw"]) @ base_rotation
                summary["yaw_deg"] = round(float(np.degrees(best["yaw"])) % 360.0, 2)
                final_verts_full, final_faces_full = verts_full, faces_full
            else:
                # --- branch B: yaw frozen at the straightened anchor rotation; ONE
                #     global scale about the mesh centroid (silhouette-preserving pivot:
                #     the trajectory centers stay valid for every candidate scale) ---
                branch = "B_own_mesh_search_scale"
                drop_rotation = base_rotation

                def _scaled(verts: np.ndarray, scale: float) -> np.ndarray:
                    return (verts - mesh_centroid) * float(scale) + mesh_centroid

                lo, hi, steps = PHYSION_PP_MC_DROP_SCALE_COARSE
                scale_step = (hi - lo) / max(int(steps) - 1, 1)
                best = {"iou": -1.0, "scale": None}
                for scale in np.linspace(lo, hi, int(steps)):
                    mean_iou = _mean_iou(drop_rotation, _scaled(verts_search, scale), faces_search)
                    if mean_iou is not None and mean_iou > best["iou"]:
                        best = {"iou": mean_iou, "scale": float(scale)}
                if best["scale"] is None:
                    summary["reason"] = "global scale search produced no valid IoU"
                    return
                initial_winner_scale = float(best["scale"])
                expanded_scales, expanded_direction = _boundary_extension_values(
                    winner=initial_winner_scale,
                    initial_start=lo,
                    initial_stop=hi,
                    step=scale_step,
                    expanded_start=PHYSION_PP_MC_DROP_SCALE_EXPANDED[0],
                    expanded_stop=PHYSION_PP_MC_DROP_SCALE_EXPANDED[1],
                    boundary_steps=PHYSION_PP_MC_DROP_SCALE_BOUNDARY_STEPS,
                )
                for scale in expanded_scales:
                    mean_iou = _mean_iou(
                        drop_rotation, _scaled(verts_search, scale), faces_search
                    )
                    if mean_iou is not None and mean_iou > best["iou"]:
                        best = {"iou": mean_iou, "scale": float(scale)}
                coarse_winner_scale = float(best["scale"])
                fine_lo, fine_hi, fine_steps = PHYSION_PP_MC_DROP_SCALE_FINE
                for scale in np.linspace(best["scale"] * fine_lo, best["scale"] * fine_hi, int(fine_steps)):
                    mean_iou = _mean_iou(drop_rotation, _scaled(verts_search, scale), faces_search)
                    if mean_iou is not None and mean_iou > best["iou"]:
                        best = {"iou": mean_iou, "scale": float(scale)}
                final_scale = float(best["scale"])
                summary["scale"] = round(final_scale, 4)
                summary["scale_search"] = {
                    "initial_range": [lo, hi],
                    "boundary_band_steps": PHYSION_PP_MC_DROP_SCALE_BOUNDARY_STEPS,
                    "initial_winner": round(initial_winner_scale, 4),
                    "expanded": bool(expanded_scales),
                    "expanded_direction": expanded_direction,
                    "evaluated_coarse_range": [
                        round(min([lo, *expanded_scales]), 4),
                        round(max([hi, *expanded_scales]), 4),
                    ],
                    "coarse_winner": round(coarse_winner_scale, 4),
                    "final_winner": round(final_scale, 4),
                }
                final_verts_full = _scaled(verts_full, final_scale)
                final_faces_full = faces_full
                # Export the scaled mesh and adopt it on the support entry + trajectory
                # item (BOTH places -- downstream renders/manifests read either).
                mesh_dir = (
                    self.artifacts.tool_dir(question_dir, self.tool_name)
                    / "mass_collision_drop_refinement"
                )
                mesh_dir.mkdir(parents=True, exist_ok=True)
                scaled_mesh_path = mesh_dir / f"{seg2_patient}_drop_scaled.glb"
                scaled_mesh = trimesh.load(Path(mesh_path), force="mesh")
                if hasattr(scaled_mesh, "geometry"):
                    scaled_mesh = trimesh.util.concatenate(tuple(scaled_mesh.geometry.values()))
                scaled_mesh.vertices = (
                    (np.asarray(scaled_mesh.vertices, dtype=np.float64) - mesh_centroid) * final_scale
                    + mesh_centroid
                )
                scaled_mesh.export(scaled_mesh_path)
                patient_support = support_by_id.get(seg2_patient)
                if isinstance(patient_support, dict):
                    patient_support["source_mesh_path"] = patient_support.get("mesh_path")
                    patient_support["mesh_path"] = str(scaled_mesh_path)
                    patient_support["mesh_source"] = "physion_pp_mass_collision_drop_refinement"
                item["mesh_path"] = str(scaled_mesh_path)
                item["mesh_source"] = "physion_pp_mass_collision_drop_refinement"

            # --- write-back: rewrite existing patient poses on the prior trajectory and
            #     append poses for masked frames with no FP entry (the patient can be
            #     visible before FP registration picks it up) ---
            refined_entries = 0
            before_ious: list[float] = []
            after_ious: list[float] = []
            existing_frames: set[int] = set()
            for pose_entry in item.get("poses", []):
                frame_index = self._pose_frame_index(pose_entry)
                if frame_index is None:
                    continue
                existing_frames.add(frame_index)
                if frame_index not in centers:
                    pose_entry["mc_drop_refined"] = False
                    continue
                if pose_entry.get("corrected_pose_4x4") is not None:
                    old = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    old_center = old[:3, :3] @ mesh_centroid + old[:3, 3]
                    old_iou = _iou_at(old_center, old[:3, :3], frame_index, verts_full, faces_full)
                    if old_iou is not None:
                        before_ious.append(float(old_iou))
                translation = centers[frame_index] - drop_rotation @ mesh_centroid
                pose = np.eye(4)
                pose[:3, :3] = drop_rotation
                pose[:3, 3] = translation
                pose_entry["corrected_pose_4x4"] = pose.tolist()
                pose_entry["corrected_translation_camera"] = translation.tolist()
                pose_entry["mc_drop_refined"] = True
                refined_entries += 1
                new_iou = _iou_at(
                    centers[frame_index], drop_rotation, frame_index, final_verts_full, final_faces_full
                )
                if new_iou is not None:
                    after_ious.append(float(new_iou))
            appended = 0
            for frame_index in frames:
                if frame_index in existing_frames:
                    continue
                translation = centers[frame_index] - drop_rotation @ mesh_centroid
                pose = np.eye(4)
                pose[:3, :3] = drop_rotation
                pose[:3, 3] = translation
                item.setdefault("poses", []).append(
                    {
                        "frame_index": int(frame_index),
                        "corrected_pose_4x4": pose.tolist(),
                        "corrected_translation_camera": translation.tolist(),
                        "mc_drop_refined": True,
                    }
                )
                refined_entries += 1
                appended += 1
                new_iou = _iou_at(
                    centers[frame_index], drop_rotation, frame_index, final_verts_full, final_faces_full
                )
                if new_iou is not None:
                    after_ious.append(float(new_iou))
            item["poses"].sort(key=lambda pose: self._pose_frame_index(pose) or 0)

            heights = {frame: float(centers[frame] @ up_axis) for frame in frames}
            summary.update(
                {
                    "applied": refined_entries > 0,
                    "object_id": seg2_patient,
                    "branch": branch,
                    "policy": (
                        "vertical plane through the corrected seg2 ball-start and agent "
                        "centers; per-frame mask-centroid ray ∩ plane; ground projection "
                        "frozen at the median along-line coordinate, heights re-solved as "
                        "ray/vertical-line closest approach; branch A searches one global "
                        "yaw on the adopted mesh, branch B one global scale on the own mesh"
                    ),
                    "mesh_path": str(scaled_mesh_path or mesh_path),
                    "plane_theta_deg": round(
                        float(np.degrees(np.arctan2(float(line_dir @ gv), float(line_dir @ gu))) % 180.0),
                        2,
                    ),
                    "plane_offset_m": round(c_offset, 4),
                    "search_iou": round(float(best["iou"]), 4),
                    "mask_frames": len(target_masks),
                    "solved_frames": len(frames),
                    "refined_pose_entries": refined_entries,
                    "appended_frames": appended,
                    "before_iou_mean": round(float(np.mean(before_ious)), 4) if before_ious else None,
                    "after_iou_mean": round(float(np.mean(after_ious)), 4) if after_ious else None,
                    "drop_height_total_m": round(float(heights[frames[0]] - heights[frames[-1]]), 4),
                }
            )
            _log_tool(
                self.tool_name,
                f"mass_collision drop_refinement {seg2_patient} branch={branch} "
                f"frames={len(frames)}/{len(target_masks)} span={span:.3f} "
                f"{'yaw=' + str(summary.get('yaw_deg')) if adopted else 'scale=' + str(summary.get('scale'))} "
                f"iou {summary['before_iou_mean']} -> {summary['after_iou_mean']}",
            )
        except Exception as exc:
            summary["reason"] = f"drop refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(
                self.tool_name,
                f"mass_collision drop_refinement error={_short_text(str(exc), 300)}",
            )

    def _apply_physion_pp_mass_collision_agent_flush_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """mass_collision agent: ground-plane flush + ONE global in-plane stretch.

        The route treats the agent as stationary before the ball's mask first
        touches its mask. The seg1 pre-contact window identifies one mesh-local upright
        axis, shared by both segments because they reuse the same physical mesh. Each
        PRE-CONTACT window aligns that axis to gravity without consulting the VLM, then
        solves ONE pose (window-mean IoU over a single in-plane offset, median
        translation seed) and broadcasts it -- per-frame refinement there would only
        re-inject FP jitter into a stationary interval. From the contact frame on,
        each frame independently grid-searches ONLY the ground-plane translation
        (height and the raw FP rotation stay untouched) for the best rendered-mask
        IoU; the per-frame searches are mutually independent and keep deterministic
        serial candidate order while mask rasterization runs on GPU. Pass 2:
        A single global (su, sv) stretch along the ground axes is folded into the mesh local frame at
        the earliest parked rotation -- is fitted jointly over BOTH segments' frames:
        the agent is the same physical object in both segments, so the cross-segment
        size lock survives by construction. Pass 3: the stretched mesh is baked,
        adopted on both segments' support entries + trajectory items, and the
        per-frame translation flush re-runs on it. Applied to both segments'
        agents."""
        mc = payload.get("physion_pp_mass_collision")
        if not isinstance(mc, dict) or mc.get("applies") is not True:
            return
        agent_flush_route = _require_mass_agent_flush_route(
            policy_benchmark="physion_pp",
            scenario=self._object_plan_scenario(object_plan),
            object_plan=object_plan,
        )
        if not isinstance(agent_flush_route, dict):
            raise ValueError("mass_collision agent-flush route was not resolved")
        agent_flush_module = self._module_for_route_record(
            agent_flush_route,
            decision_id=MASS_AGENT_FLUSH_DECISION_ID,
            module_name="agent_flush",
        )
        if (
            agent_flush_module.implementation
            != "precontact_static_then_per_frame_ground_translation"
        ):
            raise ValueError(
                "unsupported mass agent-flush module implementation: "
                f"{agent_flush_module.implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_mass_collision_agent_flush_refinement",
            "segments": {},
            "rasterizer_backend": "nvdiffrast_cuda",
            "resolved_route": deepcopy(agent_flush_route),
        }
        mc["agent_flush"] = summary
        reset_cuda_mask_rasterizer_stats()
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        roles = mc.get("role_object_ids") or {}
        objects_by_id = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        try:
            import cv2
            import trimesh

            support = payload.get("support_plane_position_correction") or {}
            support_by_id = {
                str(entry.get("object_id")): entry
                for entry in support.get("objects", [])
                if isinstance(entry, dict)
            }
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            canonical_axes_by_id = self._foundationpose_local_axes_by_object(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            gu, gv = _plane_basis_for_up_axis(up_axis)

            decimate_faces_setting = os.environ.get(
                "PHYSMIND_MC_AGENT_FLUSH_DECIMATE_FACES",
                str(PHYSION_PP_MC_AGENT_FLUSH_DECIMATE_FACES),
            )
            try:
                agent_search_face_target = int(decimate_faces_setting)
            except ValueError as exc:
                raise ValueError(
                    "PHYSMIND_MC_AGENT_FLUSH_DECIMATE_FACES must be an integer"
                ) from exc
            if agent_search_face_target < 8:
                raise ValueError(
                    "PHYSMIND_MC_AGENT_FLUSH_DECIMATE_FACES must be at least 8"
                )
            summary["search_face_target"] = agent_search_face_target

            contexts: Dict[str, Dict[str, Any]] = {}
            for seg in ("seg1", "seg2"):
                agent_id = str(roles.get(f"{seg}_agent") or "")
                item = objects_by_id.get(agent_id)
                if not agent_id or item is None:
                    summary["segments"][seg] = {"applied": False, "reason": "no agent in corrected trajectories"}
                    continue
                mesh_path = (
                    item.get("mesh_path")
                    or (support_by_id.get(agent_id) or {}).get("mesh_path")
                    or mesh_paths.get(agent_id)
                )
                if not mesh_path:
                    summary["segments"][seg] = {"applied": False, "reason": "missing agent mesh"}
                    continue
                verts_full, faces_full = self._load_mesh_geometry(Path(mesh_path))
                verts_search, faces_search = verts_full, faces_full
                if len(faces_full) > agent_search_face_target:
                    try:
                        simplified = trimesh.Trimesh(
                            vertices=verts_full, faces=faces_full, process=False
                        ).simplify_quadric_decimation(
                            face_count=agent_search_face_target
                        )
                        if len(simplified.faces) >= 8:
                            verts_search = np.asarray(simplified.vertices, dtype=np.float64)
                            faces_search = np.asarray(simplified.faces)
                    except Exception as exc:
                        raise RuntimeError(
                            f"agent search mesh decimation failed for {agent_id}: {exc}"
                        ) from exc
                masks: Dict[int, np.ndarray] = {}
                for frame_index, record in (mask_records.get(agent_id) or {}).items():
                    mask = self._resize_mask_to_shape(
                        mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                    )
                    if mask is not None and int(mask.sum()) >= PHYSION_PP_MC_AGENT_FLUSH_MIN_MASK_PX:
                        masks[int(frame_index)] = mask
                # ball-agent contact frame: the static-window prior splits here
                ball_id = str(roles.get(f"{seg}_ball") or "")
                contact_frame = None
                ball_frames = mask_records.get(ball_id) or {}
                agent_frames_raw = mask_records.get(agent_id) or {}
                if ball_frames:
                    kernel = np.ones((PHYSION_PP_MC_CONTACT_DILATE_PX * 2 + 1,) * 2, np.uint8)
                    for frame_index in sorted(set(ball_frames) & set(agent_frames_raw)):
                        ball_mask = mask_arrays.get(str((ball_frames.get(frame_index) or {}).get("mask_key") or ""))
                        agent_mask = mask_arrays.get(str((agent_frames_raw.get(frame_index) or {}).get("mask_key") or ""))
                        if ball_mask is None or agent_mask is None or ball_mask.shape != agent_mask.shape:
                            continue
                        dilated = cv2.dilate(ball_mask.astype(np.uint8), kernel) > 0
                        if bool(np.logical_and(dilated, agent_mask > 0).any()):
                            contact_frame = int(frame_index)
                            break
                contexts[seg] = {
                    "agent_id": agent_id,
                    "item": item,
                    "mesh_path": str(mesh_path),
                    "verts_search": verts_search,
                    "faces_search": faces_search,
                    "search_centroid": verts_search.mean(axis=0),
                    "masks": masks,
                    "contact_frame": contact_frame,
                    "source_mesh_faces": int(len(faces_full)),
                    "search_mesh_faces": int(len(faces_search)),
                    "sam3d_canonical_axes": canonical_axes_by_id.get(agent_id),
                }
            if not contexts:
                summary["reason"] = "no agents to refine"
                return

            # Both trials use the same physical agent mesh. Determine its SAM3D
            # canonical upright axis once from the seg1 parked, pre-contact poses, then
            # reuse that exact axis in seg2. This resolves FoundationPose's 90-degree
            # symmetry ambiguity without deriving axes from irregular mesh geometry.
            upright_source_seg = "seg1" if "seg1" in contexts else next(iter(contexts))
            upright_source = contexts[upright_source_seg]
            upright_contact = upright_source.get("contact_frame")
            upright_rotations = []
            for pose in upright_source["item"].get("poses", []):
                frame = self._pose_frame_index(pose)
                if (
                    frame is None
                    or pose.get("corrected_pose_4x4") is None
                    or upright_contact is None
                    or frame >= upright_contact
                ):
                    continue
                upright_rotations.append(
                    np.asarray(pose["corrected_pose_4x4"], dtype=np.float64)
                    .reshape(4, 4)[:3, :3]
                )
            canonical_axes = upright_source.get("sam3d_canonical_axes")
            if upright_rotations and canonical_axes and len(canonical_axes) == 3:
                canonical_axis_names = ("x", "y", "z")
                axis_scores = [
                    float(
                        np.median(
                            [
                                abs(
                                    float(
                                        np.dot(
                                            rotation @ canonical_axes[axis_index],
                                            up_axis,
                                        )
                                    )
                                )
                                for rotation in upright_rotations
                            ]
                        )
                    )
                    for axis_index in range(len(canonical_axes))
                ]
                upright_axis_index = int(np.argmax(axis_scores))
                upright_axis_local = np.asarray(
                    canonical_axes[upright_axis_index], dtype=np.float64
                )
                signed_scores = [
                    float(np.dot(rotation @ upright_axis_local, up_axis))
                    for rotation in upright_rotations
                ]
                if float(np.median(signed_scores)) < 0.0:
                    upright_axis_local = -upright_axis_local
                for ctx in contexts.values():
                    ctx["upright_axis_local"] = upright_axis_local.copy()
                summary["pre_contact_upright"] = {
                    "applied": True,
                    "policy": "seg1_pre_contact_sam3d_canonical_axis_shared_across_segments",
                    "source_segment": upright_source_seg,
                    "source_frames": len(upright_rotations),
                    "canonical_axis_index": upright_axis_index,
                    "canonical_axis_name": canonical_axis_names[upright_axis_index],
                    "axis_local": upright_axis_local.tolist(),
                    "canonical_axis_span_m": float(
                        np.ptp(
                            np.asarray(upright_source["verts_search"], dtype=np.float64)
                            @ upright_axis_local
                        )
                    ),
                    "canonical_axis_alignment_scores": {
                        name: score
                        for name, score in zip(canonical_axis_names, axis_scores)
                    },
                    "axis_alignment_score": axis_scores[upright_axis_index],
                    "vlm_gate": "ignored",
                }
            else:
                summary["pre_contact_upright"] = {
                    "applied": False,
                    "reason": (
                        "no seg1 pre-contact poses available to identify upright axis"
                        if not upright_rotations
                        else "missing or incomplete SAM3D canonical axes for seg1 agent"
                    ),
                    "vlm_gate": "ignored",
                }

            # CUDA contexts must stay in the parent process. Candidate evaluation
            # remains serial at the Python level while every raster call runs on GPU.
            worker_count = 1
            summary["configured_workers"] = worker_count

            def _make_state(ctx: Dict[str, Any], verts_search: np.ndarray, frames: list[int]) -> Dict[str, Any]:
                poses_by_frame: Dict[int, np.ndarray] = {}
                for pose in ctx["item"].get("poses", []):
                    frame = self._pose_frame_index(pose)
                    if frame is not None and pose.get("corrected_pose_4x4") is not None:
                        poses_by_frame[frame] = np.asarray(
                            pose["corrected_pose_4x4"], dtype=np.float64
                        ).reshape(4, 4)
                return {
                    "segs": {
                        "seg": {
                            "verts": verts_search,
                            "faces": np.asarray(ctx["faces_search"], dtype=np.int32),
                            "masks": {frame: ctx["masks"][frame] for frame in frames if frame in ctx["masks"]},
                            "poses": {frame: poses_by_frame[frame] for frame in frames if frame in poses_by_frame},
                            "intrinsics": {
                                frame: np.asarray(
                                    self._intrinsic_for_frame(intrinsics, frame), dtype=np.float64
                                )
                                for frame in frames
                            },
                            "centroid": verts_search.mean(axis=0),
                        }
                    },
                    "gu": gu,
                    "gv": gv,
                    "image_shape": image_shape,
                    "coarse": PHYSION_PP_MC_AGENT_FLUSH_COARSE_M,
                    "fine": PHYSION_PP_MC_AGENT_FLUSH_FINE_M,
                }

            def translation_pass(ctx: Dict[str, Any], verts_search: np.ndarray) -> Dict[str, Any]:
                search_started = time.monotonic()
                contact_frame = ctx["contact_frame"]
                pose_entries = sorted(
                    (
                        (self._pose_frame_index(pose), pose)
                        for pose in ctx["item"].get("poses", [])
                        if isinstance(pose, dict)
                        and pose.get("corrected_pose_4x4") is not None
                        and self._pose_frame_index(pose) is not None
                    ),
                    key=lambda pair: pair[0],
                )
                static_entries = [
                    (frame, pose) for frame, pose in pose_entries
                    if contact_frame is not None and frame < contact_frame
                ]
                dynamic_entries = [
                    (frame, pose) for frame, pose in pose_entries
                    if contact_frame is None or frame >= contact_frame
                ]
                all_frames = [frame for frame, _pose in pose_entries]
                state = _make_state(ctx, verts_search, all_frames)
                before_ious: list[float] = []
                after_ious: list[float] = []
                deltas: list[float] = []
                refined = 0
                static_stats: Dict[str, Any] = {"applied": False, "frames": len(static_entries)}
                if contact_frame is None:
                    static_stats["reason"] = "contact unavailable; per-frame fallback"

                # --- pre-contact static window: ONE pose, window-mean IoU, broadcast ---
                masked_static = [(frame, pose) for frame, pose in static_entries if frame in ctx["masks"]]
                if masked_static:
                    source_rotation = np.asarray(
                        masked_static[len(masked_static) // 2][1]["corrected_pose_4x4"],
                        dtype=np.float64,
                    ).reshape(4, 4)[:3, :3]
                    static_rotation = source_rotation
                    upright_alignment: Dict[str, Any] = {
                        "applied": False,
                        "reason": "upright axis unavailable",
                    }
                    upright_axis_local = ctx.get("upright_axis_local")
                    if upright_axis_local is not None:
                        static_rotation, alignment = self._align_rotation_closest_axis_to_up(
                            source_rotation,
                            up_axis=up_axis,
                            local_axes=[upright_axis_local],
                            canonicalize_twist=False,
                        )
                        upright_alignment = {
                            "applied": alignment.get("status") == "ok",
                            "policy": "fixed_seg1_local_axis_aligned_to_gravity",
                            "alignment_angle_deg": alignment.get("angle_degrees"),
                            "axis_local": np.asarray(
                                upright_axis_local, dtype=np.float64
                            ).tolist(),
                            "vlm_gate": "ignored",
                        }
                    base_translation = np.median(
                        np.stack(
                            [
                                np.asarray(pose["corrected_pose_4x4"], dtype=np.float64)
                                .reshape(4, 4)[:3, 3]
                                for _frame, pose in masked_static
                            ]
                        ),
                        axis=0,
                    )
                    # Rotate around the mesh centroid, then pin the new bottom back to
                    # the already fitted support plane before any in-plane search.
                    mesh_centroid = verts_search.mean(axis=0)
                    centroid_camera = source_rotation @ mesh_centroid + base_translation
                    base_translation = centroid_camera - static_rotation @ mesh_centroid
                    ground_height = self._support_item_ground_height(
                        support_by_id.get(ctx["agent_id"])
                    )
                    if ground_height is not None:
                        support_axis_local = static_rotation.T @ up_axis
                        bottom_local = float(
                            np.percentile(
                                verts_search @ support_axis_local,
                                PHYSION_PP_STATIC_BASE_PERCENTILE,
                            )
                        )
                        bottom_camera = bottom_local + float(np.dot(base_translation, up_axis))
                        ground_shift = float(ground_height) - bottom_camera
                        base_translation = base_translation + ground_shift * up_axis
                        upright_alignment["ground_pin_shift_m"] = ground_shift
                        upright_alignment["support_plane_height"] = float(ground_height)
                    state["static"] = {
                        "frames": [frame for frame, _pose in masked_static],
                        "rotation": static_rotation,
                        "base_translation": base_translation,
                    }
                    static_stats["upright_alignment"] = upright_alignment

                dynamic_jobs = [
                    ("seg", frame) for frame, _pose in dynamic_entries if frame in ctx["masks"]
                ]
                pool = None
                pool_used = False
                pool_failed = False
                if worker_count > 1 and (masked_static or len(dynamic_jobs) > 4):
                    try:
                        pool = ProcessPoolExecutor(
                            max_workers=worker_count,
                            mp_context=multiprocessing.get_context("fork"),
                            initializer=_mc_agent_flush_pool_init,
                            initargs=(state,),
                        )
                        pool_used = True
                    except Exception as exc:
                        _log_tool(
                            self.tool_name,
                            f"mc agent_flush pool unavailable, serial: {_short_text(str(exc), 160)}",
                        )

                def _pool_scores(fn, jobs: list, chunksize: int) -> list:
                    nonlocal pool, pool_failed
                    if pool is not None:
                        try:
                            return list(pool.map(fn, jobs, chunksize=chunksize))
                        except Exception as exc:
                            pool_failed = True
                            _log_tool(
                                self.tool_name,
                                f"mc agent_flush pool failed, serial fallback: {_short_text(str(exc), 160)}",
                            )
                            pool.shutdown(wait=False, cancel_futures=True)
                            pool = None
                    return []

                if masked_static:
                    mid_rotation = state["static"]["rotation"]
                    base_translation = state["static"]["base_translation"]
                    base_iou = _mc_agent_static_offset_task(state, 0.0, 0.0)
                    best = None
                    if base_iou is not None:
                        best = {"iou": float(base_iou), "du": 0.0, "dv": 0.0}
                        span, step = PHYSION_PP_MC_AGENT_FLUSH_COARSE_M
                        offsets = np.arange(-span, span + 1e-9, step)
                        coarse_jobs = [
                            (float(du), float(dv))
                            for du in offsets
                            for dv in offsets
                            if du != 0.0 or dv != 0.0
                        ]
                        coarse_scores = _pool_scores(
                            _mc_agent_flush_pool_static_offset,
                            coarse_jobs,
                            chunksize=2,
                        )
                        if not coarse_scores:
                            coarse_scores = [
                                _mc_agent_static_offset_task(state, du, dv)
                                for du, dv in coarse_jobs
                            ]
                        for (du, dv), value in zip(coarse_jobs, coarse_scores):
                            if value is not None and value > best["iou"]:
                                best = {"iou": float(value), "du": du, "dv": dv}

                        span, step = PHYSION_PP_MC_AGENT_FLUSH_FINE_M
                        fine_jobs = [
                            (float(du), float(dv))
                            for du in np.arange(best["du"] - span, best["du"] + span + 1e-9, step)
                            for dv in np.arange(best["dv"] - span, best["dv"] + span + 1e-9, step)
                        ]
                        fine_scores = _pool_scores(
                            _mc_agent_flush_pool_static_offset,
                            fine_jobs,
                            chunksize=2,
                        )
                        if not fine_scores:
                            fine_scores = [
                                _mc_agent_static_offset_task(state, du, dv)
                                for du, dv in fine_jobs
                            ]
                        for (du, dv), value in zip(fine_jobs, fine_scores):
                            if value is not None and value > best["iou"]:
                                best = {"iou": float(value), "du": du, "dv": dv}
                    if base_iou is not None:
                        static_translation = base_translation + best["du"] * gu + best["dv"] * gv
                        static_pose = np.eye(4)
                        static_pose[:3, :3] = mid_rotation
                        static_pose[:3, 3] = static_translation
                        for frame, pose in masked_static:
                            old = np.asarray(pose["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                            old_iou = _mc_agent_pose_iou(state, "seg", frame, old[:3, :3], old[:3, 3])
                            new_iou = _mc_agent_pose_iou(state, "seg", frame, mid_rotation, static_translation)
                            if old_iou is not None:
                                before_ious.append(float(old_iou))
                            if new_iou is not None:
                                after_ious.append(float(new_iou))
                        for _frame, pose in static_entries:
                            pose["corrected_pose_4x4"] = static_pose.tolist()
                            pose["corrected_translation_camera"] = static_translation.tolist()
                            pose["mc_agent_flush_refined"] = True
                            pose["mc_agent_static_window"] = True
                            refined += 1
                        deltas.append(float(np.hypot(best["du"], best["dv"])))
                        static_stats.update(
                            {
                                "applied": True,
                                "window_mean_iou": round(best["iou"], 4),
                                "shift_m": round(float(np.hypot(best["du"], best["dv"])), 4),
                            }
                        )
                    else:
                        static_stats["reason"] = "no renderable static frame"
                elif static_entries:
                    static_stats["reason"] = "no masked static frames"

                # --- post-contact: per-frame search, frame-parallel (independent tasks) ---
                results: list[tuple] = []
                if dynamic_jobs:
                    results = _pool_scores(
                        _mc_agent_flush_pool_frame,
                        dynamic_jobs,
                        chunksize=4,
                    )
                    if not results:
                        results = [
                            _mc_agent_frame_task(state, seg_key, frame) for seg_key, frame in dynamic_jobs
                        ]
                if pool is not None:
                    pool.shutdown(wait=True)
                entries_by_frame = {frame: pose for frame, pose in dynamic_entries}
                for _seg_key, frame_index, base_iou, best_iou, best_du, best_dv in results:
                    if base_iou is None:
                        continue
                    pose_entry = entries_by_frame[frame_index]
                    before_ious.append(float(base_iou))
                    after_ious.append(float(best_iou))
                    delta = float(np.hypot(best_du, best_dv))
                    if delta > 1e-9:
                        matrix = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                        shifted = matrix[:3, 3] + best_du * gu + best_dv * gv
                        matrix = matrix.copy()
                        matrix[:3, 3] = shifted
                        pose_entry["corrected_pose_4x4"] = matrix.tolist()
                        pose_entry["corrected_translation_camera"] = shifted.tolist()
                        pose_entry["mc_agent_flush_refined"] = True
                        deltas.append(delta)
                        refined += 1
                return {
                    "applied": refined > 0,
                    "agent_object_id": ctx["agent_id"],
                    "contact_frame": contact_frame,
                    "static_window": static_stats,
                    "dynamic_frames": len(dynamic_entries),
                    "evaluated_frames": len(before_ious),
                    "refined_pose_entries": refined,
                    "before_iou_mean": round(float(np.mean(before_ious)), 4) if before_ious else None,
                    "after_iou_mean": round(float(np.mean(after_ious)), 4) if after_ious else None,
                    "mean_shift_m": round(float(np.mean(deltas)), 4) if deltas else 0.0,
                    "max_shift_m": round(float(np.max(deltas)), 4) if deltas else 0.0,
                    "source_mesh_faces": ctx["source_mesh_faces"],
                    "search_mesh_faces": ctx["search_mesh_faces"],
                    "parallel_workers": worker_count if pool_used and not pool_failed else 0,
                    "pool_fallback": pool_failed,
                    "search_elapsed_sec": round(time.monotonic() - search_started, 2),
                }

            # --- Pass 1: per-frame translation on the original mesh ---
            for seg, ctx in contexts.items():
                summary["segments"][seg] = translation_pass(ctx, ctx["verts_search"])

            # --- Pass 2: ONE global (su, sv) fitted jointly over both segments ---
            stretch: Dict[str, Any] = {"applied": False, "su": 1.0, "sv": 1.0}
            summary["global_stretch"] = stretch
            ref_rotation = None
            for seg in ("seg1", "seg2"):
                ctx = contexts.get(seg)
                if ctx is None:
                    continue
                poses = sorted(
                    (
                        pose
                        for pose in ctx["item"].get("poses", [])
                        if isinstance(pose, dict)
                        and pose.get("corrected_pose_4x4") is not None
                        and self._pose_frame_index(pose) is not None
                    ),
                    key=lambda pose: self._pose_frame_index(pose),
                )
                if poses:
                    ref_rotation = np.asarray(
                        poses[0]["corrected_pose_4x4"], dtype=np.float64
                    ).reshape(4, 4)[:3, :3]
                    break
            if ref_rotation is None:
                stretch["reason"] = "no reference rotation"
            else:
                du_local = ref_rotation.T @ gu
                dv_local = ref_rotation.T @ gv

                def _stretched(verts: np.ndarray, centroid: np.ndarray, su: float, sv: float) -> np.ndarray:
                    rel = verts - centroid
                    rel = (
                        rel
                        + (su - 1.0) * np.outer(rel @ du_local, du_local)
                        + (sv - 1.0) * np.outer(rel @ dv_local, dv_local)
                    )
                    return centroid + rel

                # Sample collection preserves the v3 serial order exactly (item pose
                # order per segment, then the same stride subsample), so the fitted
                # (su, sv) is bitwise-identical to the serial implementation.
                sample_pairs: list[tuple[str, int]] = []
                stretch_segs: Dict[str, Dict[str, Any]] = {}
                for seg, ctx in contexts.items():
                    frames_with_pose: Dict[int, np.ndarray] = {}
                    for pose_entry in ctx["item"].get("poses", []):
                        frame_index = self._pose_frame_index(pose_entry)
                        if (
                            frame_index is None
                            or pose_entry.get("corrected_pose_4x4") is None
                            or frame_index not in ctx["masks"]
                        ):
                            continue
                        if frame_index not in frames_with_pose:
                            sample_pairs.append((seg, frame_index))
                        frames_with_pose[frame_index] = np.asarray(
                            pose_entry["corrected_pose_4x4"], dtype=np.float64
                        ).reshape(4, 4)
                    stretch_segs[seg] = {
                        "verts": ctx["verts_search"],
                        "faces": np.asarray(ctx["faces_search"], dtype=np.int32),
                        "centroid": ctx["search_centroid"],
                        "masks": {frame: ctx["masks"][frame] for frame in frames_with_pose},
                        "poses": frames_with_pose,
                        "intrinsics": {
                            frame: np.asarray(
                                self._intrinsic_for_frame(intrinsics, frame), dtype=np.float64
                            )
                            for frame in frames_with_pose
                        },
                    }
                if len(sample_pairs) > PHYSION_PP_MC_AGENT_STRETCH_MAX_FRAMES:
                    stride = int(np.ceil(len(sample_pairs) / PHYSION_PP_MC_AGENT_STRETCH_MAX_FRAMES))
                    sample_pairs = sample_pairs[::stride]
                samples = sample_pairs
                stretch_state = {
                    "segs": stretch_segs,
                    "image_shape": image_shape,
                    "stretch": {
                        "du_local": du_local,
                        "dv_local": dv_local,
                        "samples": sample_pairs,
                    },
                }

                stretch_started = time.monotonic()
                stretch_pool = None
                stretch_pool_used = False
                stretch_pool_failed = False
                if worker_count > 1:
                    try:
                        stretch_pool = ProcessPoolExecutor(
                            max_workers=worker_count,
                            mp_context=multiprocessing.get_context("fork"),
                            initializer=_mc_agent_flush_pool_init,
                            initargs=(stretch_state,),
                        )
                        stretch_pool_used = True
                    except Exception as exc:
                        _log_tool(
                            self.tool_name,
                            f"mc agent_stretch pool unavailable, serial: {_short_text(str(exc), 160)}",
                        )

                def _stretch_scores(jobs: list) -> list:
                    nonlocal stretch_pool, stretch_pool_failed
                    if stretch_pool is not None:
                        try:
                            return list(
                                stretch_pool.map(
                                    _mc_agent_flush_pool_stretch,
                                    jobs,
                                    chunksize=4,
                                )
                            )
                        except Exception as exc:
                            stretch_pool_failed = True
                            _log_tool(
                                self.tool_name,
                                f"mc agent_stretch pool failed, serial fallback: {_short_text(str(exc), 160)}",
                            )
                            stretch_pool.shutdown(wait=False, cancel_futures=True)
                            stretch_pool = None
                    return [
                        _mc_agent_stretch_task(stretch_state, float(su), float(sv))
                        for su, sv in jobs
                    ]

                base_fit = _mc_agent_stretch_task(stretch_state, 1.0, 1.0)
                best = {"iou": base_fit if base_fit is not None else -1.0, "su": 1.0, "sv": 1.0}
                lo, hi, step = PHYSION_PP_MC_AGENT_STRETCH_COARSE
                coarse_jobs = [
                    (float(su), float(sv))
                    for su in np.arange(lo, hi + 1e-9, step)
                    for sv in np.arange(lo, hi + 1e-9, step)
                ]
                for (su, sv), value in zip(coarse_jobs, _stretch_scores(coarse_jobs)):
                    if value is not None and value > best["iou"]:
                        best = {"iou": float(value), "su": float(su), "sv": float(sv)}
                span, step = PHYSION_PP_MC_AGENT_STRETCH_FINE
                fine_jobs = [
                    (float(su), float(sv))
                    for su in np.arange(best["su"] - span, best["su"] + span + 1e-9, step)
                    for sv in np.arange(best["sv"] - span, best["sv"] + span + 1e-9, step)
                ]
                for (su, sv), value in zip(fine_jobs, _stretch_scores(fine_jobs)):
                    if value is not None and value > best["iou"]:
                        best = {"iou": float(value), "su": float(su), "sv": float(sv)}
                if stretch_pool is not None:
                    stretch_pool.shutdown(wait=True)
                gain = (best["iou"] - base_fit) if base_fit is not None else 0.0
                stretch.update(
                    {
                        "su": round(best["su"], 4),
                        "sv": round(best["sv"], 4),
                        "fit_iou_base": round(base_fit, 4) if base_fit is not None else None,
                        "fit_iou": round(best["iou"], 4),
                        "gain": round(gain, 4),
                        "sample_frames": len(samples),
                        "reference": "earliest_corrected_agent_rotation",
                        "parallel_workers": (
                            worker_count
                            if stretch_pool_used and not stretch_pool_failed
                            else 0
                        ),
                        "pool_fallback": stretch_pool_failed,
                        "search_elapsed_sec": round(time.monotonic() - stretch_started, 2),
                    }
                )

                # --- Pass 3: bake, adopt on BOTH segments, re-run translation ---
                if (abs(best["su"] - 1.0) > 1e-6 or abs(best["sv"] - 1.0) > 1e-6) and gain >= PHYSION_PP_MC_AGENT_STRETCH_MIN_GAIN:
                    source_ctx = contexts.get("seg1") or contexts.get("seg2")
                    mesh_obj = trimesh.load(Path(source_ctx["mesh_path"]), force="mesh")
                    if hasattr(mesh_obj, "geometry"):
                        mesh_obj = trimesh.util.concatenate(tuple(mesh_obj.geometry.values()))
                    full_verts = np.asarray(mesh_obj.vertices, dtype=np.float64)
                    mesh_obj.vertices = _stretched(full_verts, full_verts.mean(axis=0), best["su"], best["sv"])
                    mesh_dir = (
                        self.artifacts.tool_dir(question_dir, self.tool_name)
                        / "mass_collision_agent_flush"
                    )
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    stretched_path = mesh_dir / "agent_inplane_stretched.glb"
                    mesh_obj.export(stretched_path)
                    stretch["mesh_path"] = str(stretched_path)
                    stretch["applied"] = True
                    for seg, ctx in contexts.items():
                        agent_id = ctx["agent_id"]
                        agent_support = support_by_id.get(agent_id)
                        if isinstance(agent_support, dict):
                            agent_support["source_mesh_path"] = agent_support.get("mesh_path")
                            agent_support["mesh_path"] = str(stretched_path)
                            agent_support["mesh_source"] = "physion_pp_mass_collision_agent_flush"
                        ctx["item"]["mesh_path"] = str(stretched_path)
                        ctx["item"]["mesh_source"] = "physion_pp_mass_collision_agent_flush"
                        pre_stretch = summary["segments"][seg]
                        stretched_search = _stretched(
                            ctx["verts_search"], ctx["search_centroid"], best["su"], best["sv"]
                        )
                        result = translation_pass(ctx, stretched_search)
                        result["pre_stretch"] = pre_stretch
                        summary["segments"][seg] = result

            summary["applied"] = any(
                isinstance(entry, dict) and entry.get("applied") is True
                for entry in summary["segments"].values()
            )
            summary["gpu_rasterizer"] = cuda_mask_rasterizer_stats()
            for seg, entry in summary["segments"].items():
                _log_tool(
                    self.tool_name,
                    f"mass_collision agent_flush {seg} {entry.get('agent_object_id')} "
                    f"frames={entry.get('evaluated_frames')} refined={entry.get('refined_pose_entries')} "
                    f"iou {entry.get('before_iou_mean')} -> {entry.get('after_iou_mean')} "
                    f"shift mean={entry.get('mean_shift_m')} max={entry.get('max_shift_m')}",
                )
            _log_tool(
                self.tool_name,
                f"mass_collision agent_stretch su={stretch.get('su')} sv={stretch.get('sv')} "
                f"fit {stretch.get('fit_iou_base')} -> {stretch.get('fit_iou')} applied={stretch.get('applied')}",
            )
        except Exception as exc:
            summary["gpu_rasterizer"] = cuda_mask_rasterizer_stats()
            summary["reason"] = f"agent flush failed: {_short_text(str(exc), 300)}"
            _log_tool(
                self.tool_name,
                f"mass_collision agent_flush error={_short_text(str(exc), 300)}",
            )
    def _apply_physion_pp_mass_collision_ball_trajectory_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """mass_collision ball: joint two-segment start/scale search + trajectory cut.

        The polished agent centers are fixed. Seg1 and seg2 search independent ball
        starts, while one uniform ball scale is shared. Each candidate start defines a
        directed ground line to that segment's agent; mask-centroid rays recover the
        per-frame centers. Search IoU is evaluated STRICTLY on frames before the first
        ball/agent mask contact. Contact and later frames are used only to find the
        post-impact deviation cut and never influence start or scale selection."""
        mc = payload.get("physion_pp_mass_collision")
        if not isinstance(mc, dict) or mc.get("applies") is not True:
            return
        ball_route = _require_mass_ball_trajectory_route(
            policy_benchmark="physion_pp",
            scenario=self._object_plan_scenario(object_plan),
            object_plan=object_plan,
        )
        if not isinstance(ball_route, dict):
            raise ValueError("mass_collision ball-trajectory route was not resolved")
        ball_trajectory_module = self._module_for_route_record(
            ball_route,
            decision_id=MASS_BALL_TRAJECTORY_DECISION_ID,
            module_name="ball_trajectory",
        )
        if (
            ball_trajectory_module.implementation
            != "joint_segment_mask_line_scale_then_postcontact_cut"
        ):
            raise ValueError(
                "unsupported mass ball-trajectory module implementation: "
                f"{ball_trajectory_module.implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_mass_collision_ball_trajectory_refinement",
            "segments": {},
            "resolved_route": deepcopy(ball_route),
        }
        mc["ball_trajectory_refinement"] = summary
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        roles = mc.get("role_object_ids") or {}
        objects_by_id = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        try:
            import cv2
            import trimesh

            support = payload.get("support_plane_position_correction") or {}
            support_by_id = {
                str(entry.get("object_id")): entry
                for entry in support.get("objects", [])
                if isinstance(entry, dict)
            }
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            gu, gv = _plane_basis_for_up_axis(up_axis)
            contact_kernel = np.ones((PHYSION_PP_MC_CONTACT_DILATE_PX * 2 + 1,) * 2, np.uint8)

            # Phase A: build both immutable segment contexts before searching. The
            # agent center is the median polished pre-contact center (agent flush has
            # already broadcast one static pose over this window). A missing contact
            # invalidates the segment: using all remaining frames would leak post-impact
            # observations into the search objective.
            contexts: Dict[str, Dict[str, Any]] = {}
            for seg in ("seg1", "seg2"):
                seg_summary: Dict[str, Any] = {"applied": False, "search_frames": "strictly_pre_contact"}
                summary["segments"][seg] = seg_summary
                ball_id = str(roles.get(f"{seg}_ball") or "")
                agent_id = str(roles.get(f"{seg}_agent") or "")
                item = objects_by_id.get(ball_id)
                agent_item = objects_by_id.get(agent_id)
                if not ball_id or item is None or not agent_id or agent_item is None:
                    seg_summary["reason"] = "missing ball or agent in corrected trajectories"
                    continue
                seg_range = (mc.get("two_segment") or {}).get(seg)
                lo, hi = (
                    (int(seg_range[0]), int(seg_range[1]))
                    if isinstance(seg_range, (list, tuple)) and len(seg_range) == 2
                    else (None, None)
                )

                def _in_segment(frame: int) -> bool:
                    return lo is None or lo <= frame <= hi

                mesh_path = (
                    item.get("mesh_path")
                    or (support_by_id.get(ball_id) or {}).get("mesh_path")
                    or mesh_paths.get(ball_id)
                )
                if not mesh_path:
                    seg_summary["reason"] = "missing ball mesh"
                    continue
                verts_full, faces_full = self._load_mesh_geometry(Path(mesh_path))
                mesh_centroid = verts_full.mean(axis=0)
                poses = sorted(
                    (
                        pose
                        for pose in item.get("poses", [])
                        if isinstance(pose, dict)
                        and pose.get("corrected_pose_4x4") is not None
                        and self._pose_frame_index(pose) is not None
                        and _in_segment(self._pose_frame_index(pose))
                    ),
                    key=lambda pose: self._pose_frame_index(pose),
                )
                if not poses:
                    seg_summary["reason"] = "no corrected ball poses in segment"
                    continue
                start_matrix = np.asarray(poses[0]["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                base_rotation = start_matrix[:3, :3]
                base_start_center = base_rotation @ mesh_centroid + start_matrix[:3, 3]
                start_frame = self._pose_frame_index(poses[0])

                # per-frame masks + centroid rays
                ball_frames = mask_records.get(ball_id) or {}
                agent_frames = mask_records.get(agent_id) or {} if agent_id else {}
                target_masks: Dict[int, np.ndarray] = {}
                rays: Dict[int, np.ndarray] = {}
                centroids_px: Dict[int, np.ndarray] = {}
                for frame_index, record in ball_frames.items():
                    frame_index = int(frame_index)
                    if not _in_segment(frame_index):
                        continue
                    mask = self._resize_mask_to_shape(
                        mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                    )
                    if mask is None or int(mask.sum()) < PHYSION_PP_MC_BALL_MIN_MASK_PX:
                        continue
                    ys, xs = np.nonzero(mask)
                    centroid = np.array([float(xs.mean()), float(ys.mean())])
                    intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                    ray = np.linalg.inv(intrinsic) @ np.array([centroid[0], centroid[1], 1.0])
                    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
                    target_masks[frame_index] = mask
                    rays[frame_index] = ray
                    centroids_px[frame_index] = centroid
                frames = sorted(target_masks)
                if not frames:
                    seg_summary["reason"] = "no usable ball masks in segment"
                    continue

                # contact frame: dilated ball mask touches the agent mask
                contact_frame = None
                for frame_index in frames:
                    agent_record = agent_frames.get(frame_index) or {}
                    agent_mask = mask_arrays.get(str(agent_record.get("mask_key") or ""))
                    ball_mask = mask_arrays.get(str((ball_frames.get(frame_index) or {}).get("mask_key") or ""))
                    if agent_mask is None or ball_mask is None or agent_mask.shape != ball_mask.shape:
                        continue
                    dilated = cv2.dilate(ball_mask.astype(np.uint8), contact_kernel) > 0
                    if bool(np.logical_and(dilated, agent_mask > 0).any()):
                        contact_frame = frame_index
                        break
                seg_summary["contact_frame"] = contact_frame
                if contact_frame is None:
                    seg_summary["reason"] = "ball/agent contact unavailable; refusing post-contact search leakage"
                    continue
                fit_frames = [f for f in frames if f < contact_frame]
                if len(fit_frames) < PHYSION_PP_MC_BALL_MIN_FIT_FRAMES:
                    seg_summary["reason"] = f"too few pre-contact fit frames ({len(fit_frames)})"
                    continue

                # Fixed agent anchor: median of all polished pre-contact pose centers.
                agent_mesh_path = (
                    (support_by_id.get(agent_id) or {}).get("mesh_path")
                    or agent_item.get("mesh_path")
                    or mesh_paths.get(agent_id)
                )
                if not agent_mesh_path:
                    seg_summary["reason"] = "missing polished agent mesh"
                    continue
                agent_verts, _agent_faces = self._load_mesh_geometry(Path(agent_mesh_path))
                agent_centroid = agent_verts.mean(axis=0)
                agent_centers = []
                for pose in agent_item.get("poses", []):
                    frame_index = self._pose_frame_index(pose)
                    if (
                        frame_index is None
                        or not _in_segment(frame_index)
                        or frame_index >= contact_frame
                        or pose.get("corrected_pose_4x4") is None
                    ):
                        continue
                    matrix = np.asarray(pose["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    agent_centers.append(matrix[:3, :3] @ agent_centroid + matrix[:3, 3])
                if not agent_centers:
                    seg_summary["reason"] = "no polished pre-contact agent centers"
                    continue
                agent_center = np.median(np.stack(agent_centers), axis=0)

                contexts[seg] = {
                    "seg": seg,
                    "ball_id": ball_id,
                    "agent_id": agent_id,
                    "item": item,
                    "poses": poses,
                    "lo": lo,
                    "hi": hi,
                    "mesh_path": str(mesh_path),
                    "verts_full": verts_full,
                    "faces_full": faces_full,
                    "mesh_centroid": mesh_centroid,
                    "base_rotation": base_rotation,
                    "base_start_center": base_start_center,
                    "start_frame": int(start_frame),
                    "agent_center": agent_center,
                    "target_masks": target_masks,
                    "rays": rays,
                    "centroids_px": centroids_px,
                    "intrinsics": {
                        frame: np.asarray(self._intrinsic_for_frame(intrinsics, frame), dtype=np.float64)
                        for frame in frames
                    },
                    "frames": frames,
                    "fit_frames": fit_frames,
                    "contact_frame": int(contact_frame),
                }
                seg_summary["fit_frames"] = len(fit_frames)
                seg_summary["fit_frame_indices"] = [int(frame) for frame in fit_frames]
                seg_summary["last_search_frame"] = int(fit_frames[-1])
                seg_summary["contact_frame_excluded_from_search"] = True

            if set(contexts) != {"seg1", "seg2"}:
                summary["reason"] = "joint ball search requires valid pre-contact contexts for both segments"
                return

            # Both trials show the same physical sphere. Search and final write-back use
            # one canonical seg1 mesh, while each segment retains its own rotation and
            # start/agent anchors.
            canonical = contexts["seg1"]
            canonical_radii = np.linalg.norm(
                canonical["verts_full"] - canonical["mesh_centroid"], axis=1
            )
            base_radius = float(np.median(canonical_radii))
            radius_relative_spread = float(
                np.max(np.abs(canonical_radii - base_radius)) / max(base_radius, 1e-12)
            )
            if base_radius <= 0.0 or radius_relative_spread > 0.01:
                summary["reason"] = (
                    "canonical mass-collision ball mesh is not spherical enough for "
                    f"analytic IoU search (relative radius spread={radius_relative_spread:.4f})"
                )
                return
            for context in contexts.values():
                context["mesh_centroid"] = canonical["mesh_centroid"]

            search_state = {
                "segments": contexts,
                "gu": gu,
                "gv": gv,
                "up_axis": up_axis,
                "image_shape": image_shape,
                "base_radius": base_radius,
            }
            score_cache: Dict[tuple[str, float, float, float], tuple[Optional[float], int]] = {}
            render_candidate_batches = 0
            pool_used = False
            pool_failed = False
            configured_workers = PHYSION_PP_MC_BALL_SEARCH_MAX_WORKERS
            worker_setting = os.environ.get("PHYSMIND_MC_BALL_SEARCH_WORKERS")
            if worker_setting:
                try:
                    configured_workers = max(1, int(worker_setting))
                except ValueError:
                    _log_tool(
                        self.tool_name,
                        f"invalid PHYSMIND_MC_BALL_SEARCH_WORKERS={worker_setting!r}; "
                        f"using {PHYSION_PP_MC_BALL_SEARCH_MAX_WORKERS}",
                    )
            worker_count = max(1, min(configured_workers, os.cpu_count() or 1))
            summary["configured_workers"] = worker_count
            pool = None
            if worker_count > 1:
                try:
                    pool = ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=multiprocessing.get_context("fork"),
                        initializer=_mc_ball_joint_pool_init,
                        initargs=(search_state,),
                    )
                    pool_used = True
                except Exception as exc:
                    _log_tool(
                        self.tool_name,
                        f"mc ball joint pool unavailable, serial: {_short_text(str(exc), 160)}",
                    )

            def _key(seg: str, du: float, dv: float, scale: float) -> tuple[str, float, float, float]:
                return (str(seg), round(float(du), 8), round(float(dv), 8), round(float(scale), 8))

            def _evaluate(
                offsets_by_seg: Dict[str, Sequence[tuple[float, float]]],
                scales: Sequence[float],
            ) -> None:
                nonlocal pool, pool_failed, render_candidate_batches
                scales = tuple(sorted({round(float(value), 8) for value in scales}))
                jobs = []
                for seg, offsets in offsets_by_seg.items():
                    for du, dv in offsets:
                        missing = tuple(
                            scale for scale in scales
                            if _key(seg, du, dv, scale) not in score_cache
                        )
                        if missing:
                            jobs.append((seg, float(du), float(dv), missing))
                if not jobs:
                    return
                render_candidate_batches += len(jobs)
                results = None
                if pool is not None:
                    try:
                        results = list(pool.map(_mc_ball_joint_pool_candidate, jobs, chunksize=1))
                    except Exception as exc:
                        pool_failed = True
                        _log_tool(
                            self.tool_name,
                            f"mc ball joint pool failed, serial fallback: {_short_text(str(exc), 160)}",
                        )
                        pool.shutdown(wait=False, cancel_futures=True)
                        pool = None
                if results is None:
                    results = [
                        _mc_ball_joint_candidate_scores(
                            search_state, str(seg), float(du), float(dv), tuple(scales_for_job)
                        )
                        for seg, du, dv, scales_for_job in jobs
                    ]
                for seg, du, dv, values in results:
                    for scale, score, valid_count in values:
                        score_cache[_key(seg, du, dv, scale)] = (score, int(valid_count))

            offsets_by_seg: Dict[str, set[tuple[float, float]]] = {
                seg: set(_mc_ball_offset_grid(PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M[0], PHYSION_PP_MC_BALL_START_COARSE_STEP_M))
                for seg in contexts
            }
            span_index = {seg: 0 for seg in contexts}
            scale_lo, scale_hi, scale_steps = PHYSION_PP_MC_BALL_SCALE_COARSE
            scale_step = (scale_hi - scale_lo) / max(int(scale_steps) - 1, 1)
            scales = {
                round(float(value), 8)
                for value in np.linspace(scale_lo, scale_hi, int(scale_steps))
            }
            search_started = time.monotonic()

            def _best_for(seg: str, scale: float) -> Optional[Dict[str, float]]:
                best = None
                for du, dv in sorted(offsets_by_seg[seg]):
                    score, valid_count = score_cache.get(_key(seg, du, dv, scale), (None, 0))
                    if score is None:
                        continue
                    candidate = {
                        "score": float(score),
                        "du": float(du),
                        "dv": float(dv),
                        "valid_frames": int(valid_count),
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate
                return best

            def _joint(scale: float) -> Optional[Dict[str, Any]]:
                by_seg = {seg: _best_for(seg, scale) for seg in contexts}
                if any(value is None for value in by_seg.values()):
                    return None
                return {
                    "scale": float(scale),
                    "score": float(np.mean([value["score"] for value in by_seg.values()])),
                    "segments": by_seg,
                }

            def _ranked_scales() -> list[Dict[str, Any]]:
                ranked = [value for value in (_joint(scale) for scale in sorted(scales)) if value is not None]
                return sorted(ranked, key=lambda value: (-value["score"], value["scale"]))

            def _expand_start_boundaries() -> bool:
                """Expand complete 2-D grids for top scale basins until winners leave the band."""
                expanded_any_stage = False
                while True:
                    ranked = _ranked_scales()
                    if not ranked:
                        return expanded_any_stage
                    top = ranked[:PHYSION_PP_MC_BALL_SCALE_TOPK]
                    expanded_this_round = False
                    for seg in contexts:
                        index = span_index[seg]
                        if index + 1 >= len(PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M):
                            continue
                        current_span = PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M[index]
                        if not any(
                            _mc_ball_start_near_boundary(
                                du=value["segments"][seg]["du"],
                                dv=value["segments"][seg]["dv"],
                                span=current_span,
                                step=PHYSION_PP_MC_BALL_START_COARSE_STEP_M,
                                boundary_steps=PHYSION_PP_MC_BALL_START_BOUNDARY_STEPS,
                            )
                            for value in top
                        ):
                            continue
                        span_index[seg] += 1
                        new_span = PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M[span_index[seg]]
                        new_grid = set(
                            _mc_ball_offset_grid(
                                new_span,
                                PHYSION_PP_MC_BALL_START_COARSE_STEP_M,
                            )
                        )
                        added = new_grid - offsets_by_seg[seg]
                        offsets_by_seg[seg].update(added)
                        _evaluate({seg: sorted(added)}, sorted(scales))
                        expanded_this_round = True
                        expanded_any_stage = True
                    if not expanded_this_round:
                        return expanded_any_stage

            try:
                _evaluate({seg: sorted(values) for seg, values in offsets_by_seg.items()}, sorted(scales))

                # Iteratively expand a segment's complete square grid when any top scale
                # winner enters the configured boundary band. New rings are evaluated for
                # every current scale, so expansion never leaves missing corner regions.
                _expand_start_boundaries()
                if not _ranked_scales():
                    summary["reason"] = "joint start/scale search produced no valid IoU"
                    return

                # Scale expansion uses the existing boundary-band helper. Every new
                # scale is evaluated over the complete current start grids.
                ranked = _ranked_scales()
                initial_scale_winner = float(ranked[0]["scale"])
                expanded_scales, scale_expanded_direction = _boundary_extension_values(
                    winner=initial_scale_winner,
                    initial_start=scale_lo,
                    initial_stop=scale_hi,
                    step=scale_step,
                    expanded_start=PHYSION_PP_MC_BALL_SCALE_EXPANDED[0],
                    expanded_stop=PHYSION_PP_MC_BALL_SCALE_EXPANDED[1],
                    boundary_steps=PHYSION_PP_MC_BALL_SCALE_BOUNDARY_STEPS,
                )
                new_scales = {round(float(value), 8) for value in expanded_scales} - scales
                if new_scales:
                    scales.update(new_scales)
                    _evaluate(
                        {seg: sorted(values) for seg, values in offsets_by_seg.items()},
                        sorted(new_scales),
                    )
                    _expand_start_boundaries()

                # Fine scale candidates are generated around the top coarse basins and
                # first scored over the full coarse/expanded start grids.
                ranked = _ranked_scales()
                coarse_top = ranked[:PHYSION_PP_MC_BALL_SCALE_TOPK]
                fine_scale_lo, fine_scale_hi, fine_scale_steps = PHYSION_PP_MC_BALL_SCALE_FINE
                fine_scales = {
                    round(float(value), 8)
                    for entry in coarse_top
                    for value in np.linspace(
                        max(PHYSION_PP_MC_BALL_SCALE_EXPANDED[0], entry["scale"] * fine_scale_lo),
                        min(PHYSION_PP_MC_BALL_SCALE_EXPANDED[1], entry["scale"] * fine_scale_hi),
                        int(fine_scale_steps),
                    )
                }
                unseen_fine_scales = fine_scales - scales
                scales.update(fine_scales)
                if unseen_fine_scales:
                    _evaluate(
                        {seg: sorted(values) for seg, values in offsets_by_seg.items()},
                        sorted(unseen_fine_scales),
                    )
                    _expand_start_boundaries()

                # Fine start windows for the top joint scale candidates. If the winner
                # enters a one-step boundary band, grow the window by another fine span;
                # never exceed the configured coarse/expanded global range.
                fine_top = _ranked_scales()[:PHYSION_PP_MC_BALL_SCALE_TOPK]
                for entry in fine_top:
                    scale = float(entry["scale"])
                    for seg in contexts:
                        coarse_best = _best_for(seg, scale)
                        if coarse_best is None:
                            continue
                        hard_span = PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M[span_index[seg]]
                        u_lo = max(-hard_span, coarse_best["du"] - PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                        u_hi = min(hard_span, coarse_best["du"] + PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                        v_lo = max(-hard_span, coarse_best["dv"] - PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                        v_hi = min(hard_span, coarse_best["dv"] + PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                        while True:
                            u_values = np.arange(u_lo, u_hi + 1e-9, PHYSION_PP_MC_BALL_START_FINE_STEP_M)
                            v_values = np.arange(v_lo, v_hi + 1e-9, PHYSION_PP_MC_BALL_START_FINE_STEP_M)
                            local = {
                                (round(float(du), 8), round(float(dv), 8))
                                for du in u_values for dv in v_values
                            }
                            added = local - offsets_by_seg[seg]
                            offsets_by_seg[seg].update(added)
                            _evaluate({seg: sorted(added)}, [scale])
                            best = _best_for(seg, scale)
                            if best is None:
                                break
                            step = PHYSION_PP_MC_BALL_START_FINE_STEP_M
                            near_u_lo = best["du"] <= u_lo + step + 1e-9 and u_lo > -hard_span
                            near_u_hi = best["du"] >= u_hi - step - 1e-9 and u_hi < hard_span
                            near_v_lo = best["dv"] <= v_lo + step + 1e-9 and v_lo > -hard_span
                            near_v_hi = best["dv"] >= v_hi - step - 1e-9 and v_hi < hard_span
                            if not any((near_u_lo, near_u_hi, near_v_lo, near_v_hi)):
                                break
                            if near_u_lo:
                                u_lo = max(-hard_span, u_lo - PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                            if near_u_hi:
                                u_hi = min(hard_span, u_hi + PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                            if near_v_lo:
                                v_lo = max(-hard_span, v_lo - PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
                            if near_v_hi:
                                v_hi = min(hard_span, v_hi + PHYSION_PP_MC_BALL_START_FINE_SPAN_M)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)

            ranked = _ranked_scales()
            if not ranked:
                summary["reason"] = "joint start/scale search produced no final candidate"
                return
            winner = ranked[0]
            final_scale = float(winner["scale"])
            final_by_seg = winner["segments"]
            scale_hard_lo, scale_hard_hi = PHYSION_PP_MC_BALL_SCALE_EXPANDED
            scale_saturated = (
                final_scale <= scale_hard_lo + scale_step + 1e-9
                or final_scale >= scale_hard_hi - scale_step - 1e-9
            )

            # Export ONE uniformly scaled canonical ball mesh and adopt it on both
            # segments. Scaling about the mesh centroid preserves every solved center.
            canonical_mesh = trimesh.load(Path(canonical["mesh_path"]), force="mesh")
            if hasattr(canonical_mesh, "geometry"):
                canonical_mesh = trimesh.util.concatenate(tuple(canonical_mesh.geometry.values()))
            canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float64)
            canonical_centroid = canonical_vertices.mean(axis=0)
            canonical_mesh.vertices = (
                canonical_vertices - canonical_centroid
            ) * final_scale + canonical_centroid
            mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "mass_collision_ball_joint"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            scaled_mesh_path = mesh_dir / "ball_joint_scaled.glb"
            canonical_mesh.export(scaled_mesh_path)
            final_mesh_centroid = np.asarray(canonical_mesh.vertices, dtype=np.float64).mean(axis=0)
            for context in contexts.values():
                item = context["item"]
                item["source_mesh_path"] = item.get("mesh_path")
                item["mesh_path"] = str(scaled_mesh_path)
                item["mesh_source"] = "physion_pp_mass_collision_ball_joint_search"
                support_item = support_by_id.get(context["ball_id"])
                if isinstance(support_item, dict):
                    support_item["source_mesh_path"] = support_item.get("mesh_path")
                    support_item["mesh_path"] = str(scaled_mesh_path)
                    support_item["mesh_source"] = "physion_pp_mass_collision_ball_joint_search"

            summary["joint_search"] = {
                "shared_scale": round(final_scale, 6),
                "joint_mean_iou": round(float(winner["score"]), 4),
                "objective": "full_resolution_analytic_sphere_mask_iou",
                "search_frame_policy": "frame_index < first_ball_agent_contact_frame",
                "post_contact_frames_used_for_search": False,
                "canonical_radius_m": round(base_radius, 6),
                "canonical_radius_relative_spread": round(radius_relative_spread, 8),
                "scale_initial_range": [scale_lo, scale_hi],
                "scale_evaluated_range": [round(min(scales), 6), round(max(scales), 6)],
                "scale_initial_winner": round(initial_scale_winner, 6),
                "scale_boundary_band_steps": PHYSION_PP_MC_BALL_SCALE_BOUNDARY_STEPS,
                "scale_boundary_expanded": bool(expanded_scales),
                "scale_expanded_direction": scale_expanded_direction,
                "scale_search_saturated": scale_saturated,
                "candidate_batches": render_candidate_batches,
                "cached_scores": len(score_cache),
                "parallel_workers": worker_count if pool_used and not pool_failed else 0,
                "pool_fallback": pool_failed,
                "search_elapsed_sec": round(time.monotonic() - search_started, 2),
            }

            # Phase C: write both trajectories from the selected starts/scale. All-frame
            # centers are computed only now; post-contact entries participate solely in
            # the deviation cut below, never in winner selection.
            for seg, context in contexts.items():
                seg_summary = summary["segments"][seg]
                selected = final_by_seg[seg]
                selected_span = PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M[span_index[seg]]
                start_search_saturated = _mc_ball_start_near_boundary(
                    du=selected["du"],
                    dv=selected["dv"],
                    span=selected_span,
                    step=PHYSION_PP_MC_BALL_START_FINE_STEP_M,
                    boundary_steps=1,
                ) and span_index[seg] == len(PHYSION_PP_MC_BALL_START_SEARCH_SPANS_M) - 1
                start_center = (
                    context["base_start_center"]
                    + selected["du"] * gu
                    + selected["dv"] * gv
                )
                line_dir = context["agent_center"] - start_center
                line_dir = line_dir - float(line_dir @ up_axis) * up_axis
                line_span = float(np.linalg.norm(line_dir))
                if line_span < PHYSION_PP_MC_PLANE_MIN_SPAN_M:
                    seg_summary["reason"] = f"selected ball/agent line degenerate ({line_span:.3f} m)"
                    continue
                line_dir = line_dir / line_span
                centers = _mc_ball_line_centers(
                    search_state, seg, start_center, line_dir, context["frames"]
                )
                contact_frame = context["contact_frame"]
                cut_frame = None
                run = 0
                run_start = None
                for frame_index in context["frames"]:
                    # The first touching frame is the last physically required ball
                    # pose. Deviation detection begins strictly after it, so the
                    # write-back can never delete the collision pose itself.
                    if frame_index <= contact_frame:
                        continue
                    center = centers.get(frame_index)
                    if center is None:
                        residual = float("inf")
                    else:
                        intrinsic = context["intrinsics"][frame_index]
                        uvw = intrinsic @ center
                        projected = np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])
                        residual = float(np.linalg.norm(projected - context["centroids_px"][frame_index]))
                    if residual >= PHYSION_PP_MC_BALL_DEVIATION_PX:
                        run += 1
                        if run_start is None:
                            run_start = frame_index
                        if run >= PHYSION_PP_MC_BALL_DEVIATION_FRAMES:
                            cut_frame = run_start
                            break
                    else:
                        run = 0
                        run_start = None
                seg_summary["cut_frame"] = cut_frame

                # write-back: rewrite pre-cut poses onto the line, append missing masked
                # frames, interpolate pre-cut pose frames without a solvable mask, and
                # REMOVE everything from the cut frame on (the ball disappears).
                keep_limit = cut_frame if cut_frame is not None else None
                solved = {
                    frame: center
                    for frame, center in centers.items()
                    if keep_limit is None or frame < keep_limit
                }
                solved_frames = sorted(solved)
                s_by_frame = {
                    frame: float((solved[frame] - start_center) @ line_dir) for frame in solved_frames
                }
                refined = 0
                removed = 0
                appended = 0
                kept_entries = []
                existing_frames: set[int] = set()
                item = context["item"]
                base_rotation = context["base_rotation"]
                for pose_entry in item.get("poses", []):
                    frame_index = self._pose_frame_index(pose_entry)
                    if (
                        frame_index is None
                        or (context["lo"] is not None and frame_index < context["lo"])
                        or (context["hi"] is not None and frame_index > context["hi"])
                    ):
                        kept_entries.append(pose_entry)
                        continue
                    existing_frames.add(frame_index)
                    if keep_limit is not None and frame_index >= keep_limit:
                        removed += 1
                        continue
                    center = solved.get(frame_index)
                    if center is None and solved_frames:
                        # no solvable mask this frame: interpolate along the line
                        s_value = float(
                            np.interp(frame_index, solved_frames, [s_by_frame[f] for f in solved_frames])
                        )
                        center = start_center + s_value * line_dir
                    if center is None:
                        kept_entries.append(pose_entry)
                        continue
                    translation = center - base_rotation @ final_mesh_centroid
                    pose = np.eye(4)
                    pose[:3, :3] = base_rotation
                    pose[:3, 3] = translation
                    pose_entry["corrected_pose_4x4"] = pose.tolist()
                    pose_entry["corrected_translation_camera"] = translation.tolist()
                    pose_entry["mc_ball_line_refined"] = True
                    kept_entries.append(pose_entry)
                    refined += 1
                for frame_index in solved_frames:
                    if frame_index in existing_frames:
                        continue
                    translation = solved[frame_index] - base_rotation @ final_mesh_centroid
                    pose = np.eye(4)
                    pose[:3, :3] = base_rotation
                    pose[:3, 3] = translation
                    kept_entries.append(
                        {
                            "frame_index": int(frame_index),
                            "corrected_pose_4x4": pose.tolist(),
                            "corrected_translation_camera": translation.tolist(),
                            "mc_ball_line_refined": True,
                        }
                    )
                    appended += 1
                kept_entries.sort(key=lambda pose: self._pose_frame_index(pose) or 0)
                item["poses"] = kept_entries
                seg_summary.update(
                    {
                        "applied": refined + appended > 0,
                        "ball_object_id": context["ball_id"],
                        "agent_object_id": context["agent_id"],
                        "start_frame": int(context["start_frame"]),
                        "start_du_m": round(float(selected["du"]), 4),
                        "start_dv_m": round(float(selected["dv"]), 4),
                        "start_shift_m": round(float(np.hypot(selected["du"], selected["dv"])), 4),
                        "start_search_span_m": selected_span,
                        "start_boundary_expanded": span_index[seg] > 0,
                        "start_search_saturated": start_search_saturated,
                        "line_direction_ground": line_dir.tolist(),
                        "line_theta_deg": round(
                            float(np.degrees(np.arctan2(line_dir @ gv, line_dir @ gu))) % 360.0,
                            2,
                        ),
                        "line_anchor_span_m": round(line_span, 4),
                        "fit_iou": round(float(selected["score"]), 4),
                        "fit_frames": int(selected["valid_frames"]),
                        "refined_pose_entries": refined,
                        "appended_frames": appended,
                        "removed_pose_entries": removed,
                    }
                )
                _log_tool(
                    self.tool_name,
                    f"mass_collision ball_joint {seg} {context['ball_id']} "
                    f"scale={final_scale:.4f} start=({selected['du']:.3f},{selected['dv']:.3f}) "
                    f"theta={seg_summary.get('line_theta_deg')} fit_iou={seg_summary.get('fit_iou')} "
                    f"pre_contact_frames={seg_summary.get('fit_frames')} contact={contact_frame} "
                    f"cut={cut_frame} refined={refined} appended={appended} removed={removed}",
                )
            summary["applied"] = any(
                entry.get("applied") is True for entry in summary["segments"].values()
            )
        except Exception as exc:
            summary["reason"] = f"ball trajectory refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(
                self.tool_name,
                f"mass_collision ball_line error={_short_text(str(exc), 300)}",
            )

    def _apply_physion_pp_line_layout_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        resolved_route: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Joint collinear re-layout of the friction_platform_pp statics (see the
        constants block for the prior and objective). Consumes the flush-refined
        poses/meshes as initialization, rewrites the corrected trajectories and the
        support items' mesh paths in place (the debug render and every downstream
        consumer read meshes from the support items), and must run BEFORE the agent
        ray refinement. Any failure leaves the flush results untouched."""
        summary: Dict[str, Any] = {"applied": False, "source": "physion_pp_line_layout_refinement"}
        payload["physion_pp_line_layout_refinement"] = summary
        scenario = self._object_plan_scenario(object_plan).strip().lower()
        if resolved_route is not None:
            if resolved_route.get("decision_id") != LINE_LAYOUT_DECISION_ID:
                raise ValueError(
                    "line-layout route has an unexpected decision_id: "
                    f"{resolved_route.get('decision_id')!r}"
                )
            route = resolved_route.get("route")
            if route not in LINE_LAYOUT_ROUTES:
                raise ValueError(f"unsupported line-layout route: {route!r}")
            summary["resolved_route"] = deepcopy(resolved_route)
            route_enabled = route == ACTIVE_LINE_LAYOUT_ROUTE
        else:
            route_enabled = False
        if not route_enabled:
            summary["reason"] = f"scenario {scenario or 'unknown'} not gated for line layout"
            return
        flush = payload.get("static_fixture_flush_refinement") or {}
        entries = {
            str(e.get("object_id")): e
            for e in (flush.get("objects") or [])
            if isinstance(e, dict) and e.get("status") == "ok" and e.get("eval_frame_indices")
        }
        if len(entries) < 2:
            summary["reason"] = f"only {len(entries)} refined statics"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing or invalid gravity_direction_camera"
            return
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        trajectory_by_id = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        support = payload.get("support_plane_position_correction") or {}
        support_by_id = {
            str(item.get("object_id")): item
            for item in (support.get("objects") or [])
            if isinstance(item, dict)
        }
        try:
            import multiprocessing

            import trimesh

            mask_records_all = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records_all)
            intrinsics, metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            gu, gv = _plane_basis_for_up_axis(up_axis)

            def to2(p3: np.ndarray) -> np.ndarray:
                return np.array([float(p3 @ gu), float(p3 @ gv)])

            def to3(p2: np.ndarray) -> np.ndarray:
                return p2[0] * gu + p2[1] * gv

            def build_state(oid: str, frames: list, V: np.ndarray, F: np.ndarray,
                            obb_center: np.ndarray, perp_l: np.ndarray, along_l: np.ndarray,
                            support_l: np.ndarray) -> Optional[Dict[str, Any]]:
                target_masks = {}
                for f in frames:
                    rec = (mask_records_all.get(oid) or {}).get(int(f)) or {}
                    tm = self._resize_mask_to_shape(
                        mask_arrays.get(str(rec.get("mask_key") or "")), image_shape
                    )
                    if tm is not None:
                        target_masks[int(f)] = tm
                if not target_masks:
                    return None
                K = self._intrinsic_for_frame(intrinsics, frames[0])
                return {
                    "vertices": V, "faces": F, "obb_center": obb_center,
                    "outer_u": np.outer(perp_l, perp_l), "outer_v": np.outer(along_l, along_l),
                    "outer_support": np.outer(support_l, support_l),
                    "image_shape": image_shape, "constant_intrinsic": K,
                    "intrinsic_by_frame": {f: K for f in target_masks},
                    "target_masks": target_masks,
                    "occluder_by_frame": {
                        f: self._object_mask_union(mask_records_all, mask_arrays, f, image_shape)
                        for f in target_masks
                    },
                }

            objs: Dict[str, Dict[str, Any]] = {}
            for oid, entry in entries.items():
                mesh_path = entry.get("refined_mesh_path") or mesh_paths.get(oid)
                if not mesh_path:
                    continue
                vertices, faces = self._load_mesh_geometry(Path(mesh_path))
                pose = np.asarray(entry["refined_pose_4x4"], dtype=np.float64).reshape(4, 4)
                to_origin, _extents = trimesh.bounds.oriented_bounds(vertices)
                obb_transform = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
                axes_local = obb_transform[:3, :3]
                obb_center = obb_transform[:3, 3]
                support_col = int(np.argmax(np.abs(axes_local.T @ (pose[:3, :3].T @ up_axis))))
                inplane_cols = [c for c in range(3) if c != support_col]
                eval_frames = [int(f) for f in entry["eval_frame_indices"]]
                sub = (
                    eval_frames
                    if len(eval_frames) <= PHYSION_PP_LINE_FAST_FRAME_COUNT
                    else [eval_frames[0], eval_frames[len(eval_frames) // 2], eval_frames[-1]]
                )
                objs[oid] = {
                    "V": vertices, "F": faces, "pose": pose, "obb_center": obb_center,
                    "axes_local_inplane": [axes_local[:, c] for c in inplane_cols],
                    "axis_local_support": axes_local[:, support_col],
                    "axes_cam": [pose[:3, :3] @ axes_local[:, c] for c in inplane_cols],
                    "eval_frames": eval_frames, "sub_frames": sub,
                    "iou_baseline": float(entry.get("refined_mean_iou") or 0.0),
                    "mesh_path": str(mesh_path),
                }
            if len(objs) < 2:
                summary["reason"] = f"only {len(objs)} statics with usable meshes"
                return

            oids = sorted(objs)
            centers2 = {
                oid: to2((o["pose"][:3, :3] @ o["obb_center"]) + o["pose"][:3, 3])
                for oid, o in objs.items()
            }
            stacked = np.stack([centers2[oid] for oid in oids])
            _, _, vt = np.linalg.svd(stacked - stacked.mean(0), full_matrices=False)
            theta0 = float(np.degrees(np.arctan2(vt[0][1], vt[0][0]))) % 180.0

            def dir2(theta_deg: float) -> np.ndarray:
                th = np.deg2rad(theta_deg)
                return np.array([np.cos(th), np.sin(th)])

            two_static_mode = len(oids) == 2
            if not two_static_mode:
                reset_cuda_mask_rasterizer_stats()

            def line_score(
                state: Dict[str, Any], pose: np.ndarray, scale_u: float, scale_v: float
            ) -> float:
                if two_static_mode:
                    return _flush_refine_score(state, pose, scale_u, scale_v)
                return _flush_refine_score_cuda(state, pose, scale_u, scale_v)

            for oid, o in objs.items():
                best_idx, best_axis, best_align = 0, None, -1.0
                for idx, axis_cam in enumerate(o["axes_cam"]):
                    a2 = to2(axis_cam)
                    norm = float(np.linalg.norm(a2))
                    if norm < 1e-6:
                        continue
                    a2 = a2 / norm
                    # 2-static mode has no trustworthy line to align to (the 2-center
                    # PCA direction inherits the position errors), so the dominant axis
                    # is the ELONGATION direction -- it carries the yaw evidence.
                    align = norm if two_static_mode else abs(float(a2 @ dir2(theta0)))
                    if align > best_align:
                        best_align, best_axis, best_idx = align, a2, idx
                if best_axis is None:
                    summary["reason"] = f"{oid} has no usable in-plane axis"
                    return
                o["phi_deg"] = float(np.degrees(np.arctan2(best_axis[1], best_axis[0]))) % 180.0
                along_local = o["axes_local_inplane"][best_idx]
                perp_local = o["axes_local_inplane"][1 - best_idx]
                o["perp_local"], o["along_local"] = perp_local, along_local
                o["state_fast"] = build_state(
                    oid, o["sub_frames"], o["V"], o["F"], o["obb_center"],
                    perp_local, along_local, o["axis_local_support"],
                )
                o["state_full"] = build_state(
                    oid, o["eval_frames"], o["V"], o["F"], o["obb_center"],
                    perp_local, along_local, o["axis_local_support"],
                )
                if o["state_fast"] is None or o["state_full"] is None:
                    summary["reason"] = f"{oid} has no usable target masks"
                    return
                center_cam = (o["pose"][:3, :3] @ o["obb_center"]) + o["pose"][:3, 3]
                c_ground = center_cam - np.dot(center_cam, up_axis) * up_axis
                flat_dir = c_ground / max(1e-9, float(np.linalg.norm(c_ground)))
                curvature_scores = []
                for t in (-PHYSION_PP_LINE_CURV_DELTA_M, 0.0, PHYSION_PP_LINE_CURV_DELTA_M):
                    probe_pose = o["pose"].copy()
                    probe_pose[:3, 3] = probe_pose[:3, 3] + flat_dir * t
                    curvature_scores.append(
                        line_score(o["state_full"], probe_pose, 1.0, 1.0)
                    )
                im, i0, ip = curvature_scores
                o["w"] = max(
                    PHYSION_PP_LINE_W_MIN,
                    float(-(im + ip - 2 * i0) / (PHYSION_PP_LINE_CURV_DELTA_M ** 2)),
                )
                if two_static_mode:
                    yaw_scores = []
                    for delta_deg in (
                        -PHYSION_PP_LINE_TWO_STATIC_YAW_DELTA_DEG,
                        0.0,
                        PHYSION_PP_LINE_TWO_STATIC_YAW_DELTA_DEG,
                    ):
                        rotation = _rotation_about_axis(up_axis, float(np.deg2rad(delta_deg)))
                        centroid_cam = o["pose"][:3, :3] @ o["obb_center"] + o["pose"][:3, 3]
                        yaw_pose = o["pose"].copy()
                        yaw_pose[:3, :3] = rotation @ o["pose"][:3, :3]
                        yaw_pose[:3, 3] = centroid_cam - yaw_pose[:3, :3] @ o["obb_center"]
                        yaw_scores.append(line_score(o["state_full"], yaw_pose, 1.0, 1.0))
                    ym, y0, yp = yaw_scores
                    o["w_yaw"] = max(
                        1e-4,
                        float(-(ym + yp - 2 * y0) / (PHYSION_PP_LINE_TWO_STATIC_YAW_DELTA_DEG ** 2)),
                    )

            def snapped_pose(o: Dict[str, Any], theta_deg: float) -> np.ndarray:
                delta = (theta_deg - o["phi_deg"] + 90.0) % 180.0 - 90.0
                rotation = _rotation_about_axis(up_axis, float(np.deg2rad(delta)))
                centroid_cam = o["pose"][:3, :3] @ o["obb_center"] + o["pose"][:3, 3]
                out = o["pose"].copy()
                out[:3, :3] = rotation @ o["pose"][:3, :3]
                out[:3, 3] = centroid_cam - out[:3, :3] @ o["obb_center"]
                return out

            def pose_at(o: Dict[str, Any], snap: np.ndarray, target2: np.ndarray) -> np.ndarray:
                current2 = to2(snap[:3, :3] @ o["obb_center"] + snap[:3, 3])
                out = snap.copy()
                out[:3, 3] = out[:3, 3] + to3(target2 - current2)
                return out

            # Exactly-2-statics: orientation-only degradation (see constants block).
            two_static_final = None
            two_static_tiers: Dict[str, str] = {}
            if two_static_mode:
                mean_vec = np.zeros(2)
                for o in objs.values():
                    doubled = np.deg2rad(2.0 * o["phi_deg"])
                    mean_vec += o["w_yaw"] * np.array([np.cos(doubled), np.sin(doubled)])
                theta_star = float(np.degrees(np.arctan2(mean_vec[1], mean_vec[0])) / 2.0) % 180.0
                two_poses, two_ious, two_ks = {}, {}, {}
                for oid in oids:
                    o = objs[oid]
                    snap = snapped_pose(o, theta_star)
                    iou_snap = float(line_score(o["state_full"], snap, 1.0, 1.0))
                    best_k, best_iou = 1.0, iou_snap
                    for k in PHYSION_PP_LINE_K_GRID:
                        score = float(line_score(o["state_full"], snap, float(k), 1.0))
                        if score > best_iou:
                            best_iou, best_k = score, float(k)
                    for dk in PHYSION_PP_LINE_K_FINE:
                        k = float(np.clip(best_k + dk, *PHYSION_PP_LINE_K_CLIP))
                        score = float(line_score(o["state_full"], snap, k, 1.0))
                        if score > best_iou:
                            best_iou, best_k = score, k
                    tier, pose_f, k_f, iou_f = "snap_and_stretch", snap, best_k, best_iou
                    if iou_f < o["iou_baseline"] - PHYSION_PP_LINE_TWO_STATIC_GUARD_EPS:
                        tier, pose_f, k_f, iou_f = "snap_only", snap, 1.0, iou_snap
                    if iou_f < o["iou_baseline"] - PHYSION_PP_LINE_TWO_STATIC_GUARD_EPS:
                        tier, pose_f, k_f, iou_f = "reverted", o["pose"], 1.0, o["iou_baseline"]
                    two_static_tiers[oid] = tier
                    two_poses[oid], two_ious[oid], two_ks[oid] = pose_f, iou_f, k_f
                two_static_final = {
                    "total": sum(two_ious.values()), "theta": theta_star, "c": None,
                    "poses": two_poses, "ious": two_ious, "ks": two_ks,
                }

            pool_states: Dict[Any, Any] = {}
            for oid in oids:
                pool_states[(oid, "fast")] = objs[oid]["state_fast"]
                pool_states[(oid, "full")] = objs[oid]["state_full"]
            worker_count = 1
            search_started = time.monotonic()
            s_grid = np.arange(
                -PHYSION_PP_LINE_S_SPAN_M,
                PHYSION_PP_LINE_S_SPAN_M + 1e-9,
                PHYSION_PP_LINE_S_STEP_M,
            )
            candidates = []
            theta_values = (
                []
                if two_static_final is not None
                else np.arange(
                    theta0 - PHYSION_PP_LINE_THETA_SPAN_DEG,
                    theta0 + PHYSION_PP_LINE_THETA_SPAN_DEG + 1e-9,
                    PHYSION_PP_LINE_THETA_STEP_DEG,
                )
            )
            for theta in theta_values:
                d = dir2(float(theta))
                n = np.array([-d[1], d[0]])
                snaps = {oid: snapped_pose(objs[oid], float(theta)) for oid in oids}
                s_init = {oid: float(centers2[oid] @ d) for oid in oids}
                c_center = float(np.mean([centers2[oid] @ n for oid in oids]))
                c_values = list(
                    np.arange(
                        c_center - PHYSION_PP_LINE_C_SPAN_M,
                        c_center + PHYSION_PP_LINE_C_SPAN_M + 1e-9,
                        PHYSION_PP_LINE_C_STEP_M,
                    )
                )

                def batch(jobs_meta):
                    return [
                        (
                            meta,
                            _flush_refine_score_cuda(
                                pool_states[state_key],
                                np.asarray(pose_flat, dtype=np.float64).reshape(4, 4),
                                float(scale_u),
                                1.0,
                            ),
                        )
                        for (state_key, pose_flat, scale_u), meta in jobs_meta
                    ]

                jm = []
                for c in c_values:
                    for oid in oids:
                        for ds in s_grid:
                            s = s_init[oid] + float(ds)
                            pose = pose_at(objs[oid], snaps[oid], n * c + d * s)
                            jm.append(
                                (
                                    ((oid, "fast"), pose.ravel().tolist(), 1.0),
                                    (c, oid, s),
                                )
                            )
                stage_a: Dict[float, Dict[str, tuple]] = {}
                for (c, oid, s), score in batch(jm):
                    best = stage_a.setdefault(c, {}).setdefault(
                        oid, (float("-inf"), None)
                    )
                    if score > best[0]:
                        stage_a[c][oid] = (score, s)
                jm = []
                for c in c_values:
                    for oid in oids:
                        s = stage_a[c][oid][1]
                        pose = pose_at(objs[oid], snaps[oid], n * c + d * s)
                        for k in PHYSION_PP_LINE_K_GRID:
                            jm.append(
                                (((oid, "fast"), pose.ravel().tolist(), k), (c, oid, k))
                            )
                stage_b: Dict[float, Dict[str, tuple]] = {}
                for (c, oid, k), score in batch(jm):
                    best = stage_b.setdefault(c, {}).setdefault(
                        oid, (float("-inf"), 1.0)
                    )
                    if score > best[0]:
                        stage_b[c][oid] = (score, k)
                jm = []
                for c in c_values:
                    for oid in oids:
                        k = stage_b[c][oid][1]
                        for ds in s_grid:
                            s = s_init[oid] + float(ds)
                            pose = pose_at(objs[oid], snaps[oid], n * c + d * s)
                            jm.append(
                                (
                                    ((oid, "fast"), pose.ravel().tolist(), k),
                                    (c, oid, s, k),
                                )
                            )
                stage_c: Dict[float, Dict[str, tuple]] = {}
                for (c, oid, s, k), score in batch(jm):
                    best = stage_c.setdefault(c, {}).setdefault(
                        oid, (float("-inf"), None, 1.0)
                    )
                    if score > best[0]:
                        stage_c[c][oid] = (score, s, k)
                for c in c_values:
                    total = sum(objs[oid]["w"] * stage_c[c][oid][0] for oid in oids)
                    candidates.append(
                        {
                            "theta": float(theta), "c": float(c),
                            "s": {oid: stage_c[c][oid][1] for oid in oids},
                            "k": {oid: stage_c[c][oid][2] for oid in oids},
                            "total_fast": total,
                        }
                    )
            candidates.sort(key=lambda item: -item["total_fast"])

            best_final = two_static_final
            for cand in candidates[:PHYSION_PP_LINE_TOPK_FULL]:
                d = dir2(cand["theta"])
                n = np.array([-d[1], d[0]])
                snaps = {oid: snapped_pose(objs[oid], cand["theta"]) for oid in oids}
                jobs, meta = [], []
                for oid in oids:
                    s_fine = np.arange(
                        cand["s"][oid] - 2 * PHYSION_PP_LINE_S_STEP_M,
                        cand["s"][oid] + 2 * PHYSION_PP_LINE_S_STEP_M + 1e-9,
                        PHYSION_PP_LINE_S_FINE_STEP_M,
                    )
                    for s in s_fine:
                        for dk in PHYSION_PP_LINE_K_FINE:
                            k = float(
                                np.clip(
                                    cand["k"][oid] + dk, *PHYSION_PP_LINE_K_CLIP
                                )
                            )
                            jobs.append(
                                (
                                    (oid, "full"),
                                    pose_at(objs[oid], snaps[oid], n * cand["c"] + d * float(s)).ravel().tolist(),
                                    k,
                                )
                            )
                            meta.append((oid, float(s), k))
                scores = [
                    _flush_refine_score_cuda(
                        pool_states[state_key],
                        np.asarray(pose_flat, dtype=np.float64).reshape(4, 4),
                        float(scale_u),
                        1.0,
                    )
                    for state_key, pose_flat, scale_u in jobs
                ]
                bests: Dict[str, tuple] = {}
                for (oid, s, k), score in zip(meta, scores):
                    if oid not in bests or score > bests[oid][0]:
                        bests[oid] = (score, s, k)
                total = sum(objs[oid]["w"] * bests[oid][0] for oid in oids)
                if best_final is None or total > best_final["total"]:
                    best_final = {
                        "total": total, "theta": cand["theta"], "c": cand["c"],
                        "poses": {
                            oid: pose_at(
                                objs[oid],
                                snaps[oid],
                                n * cand["c"] + d * bests[oid][1],
                            )
                            for oid in oids
                        },
                        "ious": {oid: float(bests[oid][0]) for oid in oids},
                        "ks": {oid: float(bests[oid][2]) for oid in oids},
                    }
            if best_final is None:
                summary["reason"] = "search produced no candidate"
                return

            mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "line_layout_meshes"
            rows = []
            for oid in oids:
                o = objs[oid]
                pose_new = best_final["poses"][oid]
                k = best_final["ks"][oid]
                mesh_out = o["mesh_path"]
                if abs(k - 1.0) > 1e-3:
                    scaling = (
                        k * np.outer(o["perp_local"], o["perp_local"])
                        + np.outer(o["along_local"], o["along_local"])
                        + np.outer(o["axis_local_support"], o["axis_local_support"])
                    )
                    scaled = (o["V"] - o["obb_center"]) @ scaling.T + o["obb_center"]
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    mesh_out = str(mesh_dir / f"{oid}_line_layout_local.glb")
                    self._export_mesh_with_source_colors(
                        vertices=scaled, faces=o["F"],
                        source_mesh_path=Path(o["mesh_path"]), output_path=Path(mesh_out),
                    )
                translation = pose_new[:3, 3].tolist()
                trajectory_item = trajectory_by_id.get(oid)
                if trajectory_item is not None:
                    for pose_entry in trajectory_item.get("poses", []):
                        if isinstance(pose_entry, dict):
                            pose_entry["corrected_pose_4x4"] = pose_new.tolist()
                            pose_entry["corrected_translation_camera"] = translation
                    trajectory_item["physion_pp_line_layout_applied"] = True
                support_item = support_by_id.get(oid)
                if support_item is not None and mesh_out != o["mesh_path"]:
                    support_item["pre_line_layout_mesh_path"] = support_item.get("mesh_path")
                    support_item["mesh_path"] = mesh_out
                row = {
                    "object_id": oid, "w_curvature": round(o["w"], 4),
                    "k_perpendicular": round(k, 3),
                    "iou_flush": round(o["iou_baseline"], 4),
                    "iou_line_layout": round(best_final["ious"][oid], 4),
                    "move_m": round(float(np.linalg.norm(pose_new[:3, 3] - o["pose"][:3, 3])), 3),
                    "yaw_snap_deg": round((best_final["theta"] - o["phi_deg"] + 90.0) % 180.0 - 90.0, 1),
                    "refined_mesh_path": mesh_out,
                }
                if two_static_final is not None:
                    row["tier"] = two_static_tiers.get(oid)
                    row["w_yaw"] = round(o.get("w_yaw", 0.0), 5)
                rows.append(row)
            summary.update({
                "applied": True,
                "mode": (
                    "two_static_orientation_only"
                    if two_static_final is not None
                    else "full_line_layout"
                ),
                "line_theta_deg": round(best_final["theta"], 2),
                "line_offset_m": (
                    None if best_final.get("c") is None else round(best_final["c"], 4)
                ),
                "objects": rows,
                "search_elapsed_sec": round(time.monotonic() - search_started, 2),
                "parallel_workers": worker_count,
            })
            if not two_static_mode:
                summary["rasterizer_backend"] = "nvdiffrast_cuda"
                summary["gpu_rasterizer_stats"] = cuda_mask_rasterizer_stats()
            order = (payload.get("trajectory_correction") or {}).get("execution_order")
            if isinstance(order, list) and "physion_pp_line_layout_refinement" not in order:
                try:
                    order.insert(order.index("active_interval_simulation_state"), "physion_pp_line_layout_refinement")
                except ValueError:
                    order.append("physion_pp_line_layout_refinement")
            _log_tool(
                self.tool_name,
                "line_layout mode={} theta={:.1f} c={} objects={}".format(
                    summary["mode"], best_final["theta"],
                    "n/a" if best_final.get("c") is None else f"{best_final['c']:.3f}",
                    [
                        (r["object_id"], r.get("tier") or "joint", r["k_perpendicular"], r["move_m"])
                        for r in rows
                    ],
                ),
            )
        except Exception as exc:
            # The payload mutations above are transactional-in-spirit but not literally:
            # if we reached summary["applied"]=True the corrections ARE in effect and
            # only the trailing bookkeeping failed -- say so instead of "keeping flush".
            summary["reason"] = f"line layout error: {_short_text(str(exc), 200)}"
            if summary.get("applied"):
                _log_tool(self.tool_name, f"line_layout post-apply error (corrections kept): {_short_text(str(exc), 200)}")
            else:
                _log_tool(self.tool_name, f"line_layout failed, keeping flush results: {_short_text(str(exc), 200)}")

    def _physion_pp_sphere_agent_context(
        self,
        *,
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> tuple[Dict[str, Any], set[str]]:
        scenario = self._object_plan_scenario(object_plan).strip().lower()
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        candidate_agent_ids = sphere_agent_object_ids(
            scenario=scenario,
            tracks_payload=tracks_payload,
            target_objects=object_plan.target_objects,
        )
        route_record = self._foundationpose_agent_geometry_route_record(object_plan)
        policy = route_agent_geometry_policy_payload(
            route_record=route_record,
            scenario=scenario,
            agent_object_ids=candidate_agent_ids,
        )
        agent_ids = set(policy["agent_object_ids"])
        return policy, agent_ids

    def _adopt_physion_pp_agent_sphere(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        object_ids: Sequence[str],
        radius: float,
        centers_by_object_frame: Dict[str, Dict[int, np.ndarray]],
        source: str,
        output_subdir: str,
    ) -> str:
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=float(radius))
        mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / output_subdir
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / "agent_sphere_local.glb"
        sphere.export(mesh_path)
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        corrected_objects = corrected.setdefault("objects", [])
        support = payload.get("support_plane_position_correction") or {}
        support_objects = support.setdefault("objects", [])
        target_by_id = {str(target.object_id): target for target in object_plan.target_objects}
        for object_id in object_ids:
            frames = centers_by_object_frame.get(str(object_id)) or {}
            if not frames:
                continue
            item = next(
                (
                    value
                    for value in corrected_objects
                    if isinstance(value, dict) and str(value.get("object_id")) == str(object_id)
                ),
                None,
            )
            if item is None:
                item = {"object_id": str(object_id)}
                corrected_objects.append(item)
            poses = []
            for frame_index, center in sorted(frames.items()):
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
                poses.append(
                    {
                        "frame_index": int(frame_index),
                        "corrected_pose_4x4": pose.tolist(),
                        "corrected_translation_camera": pose[:3, 3].tolist(),
                        "rotation_policy": "ignored_identity_rotation_sphere_agent",
                        "sphere_agent_refined": True,
                    }
                )
            target = target_by_id.get(str(object_id))
            source_geometry = str(getattr(target, "geometry_type", "unknown") or "unknown")
            item.update(
                {
                    "status": "ok",
                    "source": source,
                    "pose_count": len(poses),
                    "missing_pose_count": 0,
                    "activation": self._activation_from_poses(poses),
                    "source_geometry_type": source_geometry,
                    "effective_geometry_type": "sphere",
                    "mesh_path": str(mesh_path),
                    "mesh_source": source,
                    "sphere_radius_m": float(radius),
                    "poses": poses,
                }
            )
            support_item = next(
                (
                    value
                    for value in support_objects
                    if isinstance(value, dict) and str(value.get("object_id")) == str(object_id)
                ),
                None,
            )
            if support_item is None:
                support_item = {"object_id": str(object_id), "status": "ok", "poses": []}
                support_objects.append(support_item)
            support_item.update(
                {
                    "source_mesh_path": support_item.get("mesh_path"),
                    "mesh_path": str(mesh_path),
                    "mesh_source": source,
                    "source_geometry_type": source_geometry,
                    "effective_geometry_type": "sphere",
                    "sphere_radius_m": float(radius),
                }
            )
        corrected["applied"] = bool(corrected_objects)
        return str(mesh_path)

    def _apply_physion_pp_agent_ray_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        route_record = self._agent_trajectory_route_record(object_plan)
        if route_record is None:
            return
        if (
            route_record.get("decision_id")
            != FRICTION_PLATFORM_AGENT_TRAJECTORY_DECISION_ID
        ):
            return
        policy, agent_ids = self._physion_pp_sphere_agent_context(
            question_dir=question_dir,
            object_plan=object_plan,
        )
        payload["physion_pp_agent_geometry_policy"] = policy
        payload["agent_trajectory_route"] = deepcopy(route_record)
        module = self._module_for_route_record(
            route_record,
            decision_id=FRICTION_PLATFORM_AGENT_TRAJECTORY_DECISION_ID,
            module_name="agent_trajectory",
        )
        if module.implementation == "support_surface_mask_sphere_trajectory":
            self._apply_physion_pp_friction_platform_sphere_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                agent_ids=agent_ids,
            )
        elif (
            module.implementation
            == "support_surface_native_mesh_ray_trajectory"
        ):
            self._apply_physion_pp_agent_ray_native_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
            )
        else:
            raise ValueError(
                "unsupported friction-platform agent-trajectory module implementation: "
                f"{module.implementation!r}"
            )
        summary = payload.get("physion_pp_agent_trajectory_refinement")
        if isinstance(summary, dict):
            summary["resolved_route"] = deepcopy(route_record)

    def _apply_physion_pp_friction_platform_sphere_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        agent_ids: set[str],
    ) -> None:
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "trajectory.platform_sphere",
            "geometry": "sphere",
            "foundationpose_agent_policy": "skipped",
        }
        payload["physion_pp_agent_trajectory_refinement"] = summary
        if len(agent_ids) != 1:
            summary["reason"] = f"expected exactly one role-bound agent, got {sorted(agent_ids)}"
            return
        agent_id = next(iter(agent_ids))
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected static trajectories not applied"
            return
        support = payload.get("support_plane_position_correction") or {}
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        objects = {
            str(item.get("object_id")): item
            for item in corrected.get("objects", [])
            if isinstance(item, dict)
        }
        support_by_id = {
            str(item.get("object_id")): item
            for item in support.get("objects", [])
            if isinstance(item, dict)
        }
        line_layout_by_id = {
            str(item.get("object_id")): item
            for item in (payload.get("physion_pp_line_layout_refinement") or {}).get("objects", [])
            if isinstance(item, dict)
        }
        static_ids = [
            str(target.object_id)
            for target in object_plan.target_objects
            if _is_physion_static_ground_fixture_track(target.source_track_id)
        ]
        try:
            import trimesh

            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            gu, gv = _plane_basis_for_up_axis(up_axis)
            surface_points = []
            used_static_ids = []
            for object_id in static_ids:
                item = objects.get(object_id) or {}
                pose_entry = next(
                    (
                        pose
                        for pose in item.get("poses", [])
                        if isinstance(pose, dict) and pose.get("corrected_pose_4x4") is not None
                    ),
                    None,
                )
                mesh_path = (
                    (line_layout_by_id.get(object_id) or {}).get("refined_mesh_path")
                    or (support_by_id.get(object_id) or {}).get("mesh_path")
                )
                if pose_entry is None or not mesh_path:
                    continue
                pose = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                vertices, faces = self._load_mesh_geometry(Path(mesh_path))
                dense = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).subdivide_to_size(
                    max_edge=PHYSION_PP_AGENT_RAY_CELL_M, max_iter=16
                )
                posed = np.asarray(dense.vertices, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
                if len(posed) > 2_000_000:
                    posed = posed[np.linspace(0, len(posed) - 1, 2_000_000, dtype=np.int64)]
                surface_points.append(posed)
                used_static_ids.append(object_id)
            if not surface_points:
                summary["reason"] = "no final static support surface"
                return
            surface = np.vstack(surface_points)
            surface_gp = np.stack([surface @ gu, surface @ gv], axis=1)
            surface_h = surface @ up_axis
            # Sphere mode skips FoundationPose for the agent. Derive the fill height
            # from the final, pose-corrected static meshes
            # that also define the collision/clearance raster below.
            ground = float(np.min(surface_h))
            cell = PHYSION_PP_AGENT_RAY_CELL_M
            grid_min = surface_gp.min(axis=0) - 0.5
            grid_max = surface_gp.max(axis=0) + 0.5
            nu = int(np.ceil((grid_max[0] - grid_min[0]) / cell)) + 1
            nv = int(np.ceil((grid_max[1] - grid_min[1]) / cell)) + 1
            raster = np.full((nu, nv), ground, dtype=np.float64)
            iu = np.clip(((surface_gp[:, 0] - grid_min[0]) / cell).astype(int), 0, nu - 1)
            iv = np.clip(((surface_gp[:, 1] - grid_min[1]) / cell).astype(int), 0, nv - 1)
            np.maximum.at(raster, (iu, iv), surface_h)

            target_masks: Dict[int, np.ndarray] = {}
            observations: Dict[int, np.ndarray] = {}
            for frame_index, record in (mask_records.get(agent_id) or {}).items():
                mask = self._resize_mask_to_shape(
                    mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                )
                if mask is None or int(mask.sum()) < PHYSION_PP_AGENT_SPHERE_MIN_MASK_PX:
                    continue
                ys, xs = np.nonzero(mask)
                y_bottom = int(ys.max())
                x_bottom = float(xs[ys >= y_bottom - 1].mean())
                frame_index = int(frame_index)
                target_masks[frame_index] = mask
                observations[frame_index] = np.array([x_bottom, y_bottom + 0.5], dtype=np.float64)
            if len(observations) < PHYSION_PP_AGENT_RAY_MIN_MATCHED:
                summary["reason"] = f"too few agent mask observations ({len(observations)})"
                return

            center_gp = 0.5 * (grid_min + grid_max)
            line_half_span = 0.75 * float(np.linalg.norm(grid_max - grid_min)) + 0.5
            sample_step = max(0.03, 0.5 * cell)
            xs_grid = np.arange(-line_half_span, line_half_span + 1e-9, sample_step)
            observed = np.stack([observations[frame] for frame in sorted(observations)])

            def curve_for(theta: float, offset: float):
                direction2 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
                normal2 = np.array([-direction2[1], direction2[0]], dtype=np.float64)
                base = center_gp + offset * normal2
                ground_points = base[None, :] + xs_grid[:, None] * direction2[None, :]
                inside = (
                    (ground_points[:, 0] >= grid_min[0])
                    & (ground_points[:, 0] <= grid_max[0])
                    & (ground_points[:, 1] >= grid_min[1])
                    & (ground_points[:, 1] <= grid_max[1])
                )
                ii = np.clip(((ground_points[:, 0] - grid_min[0]) / cell).astype(int), 0, nu - 1)
                jj = np.clip(((ground_points[:, 1] - grid_min[1]) / cell).astype(int), 0, nv - 1)
                heights = np.maximum(ground, raster[ii, jj])
                points = ground_points[:, 0:1] * gu[None, :] + ground_points[:, 1:2] * gv[None, :]
                points = points + (heights - points @ up_axis)[:, None] * up_axis[None, :]
                valid = inside & (points[:, 2] > 0.5)
                K = np.asarray(self._intrinsic_for_frame(intrinsics, 0), dtype=np.float64)
                uv = (K @ (points.T / np.maximum(points[:, 2], 1e-6))).T[:, :2]
                valid &= np.isfinite(uv).all(axis=1)
                return points, ground_points, uv, valid, direction2

            def line_loss(theta: float, offset: float) -> float:
                _points, _gp, uv, valid, _direction = curve_for(theta, offset)
                visible_uv = uv[valid]
                if len(visible_uv) < 10:
                    return float("inf")
                distances = np.sqrt(((observed[:, None, :] - visible_uv[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
                keep = max(1, int(len(distances) * PHYSION_PP_AGENT_RAY_TRIM_FRACTION))
                return float(np.sort(distances)[:keep].mean())

            best = (float("inf"), 0.0, 0.0)
            for theta in np.deg2rad(np.arange(0.0, 180.0, 3.0)):
                direction2 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
                normal2 = np.array([-direction2[1], direction2[0]], dtype=np.float64)
                corner_offsets = [
                    float((np.array([x, y]) - center_gp) @ normal2)
                    for x in (grid_min[0], grid_max[0])
                    for y in (grid_min[1], grid_max[1])
                ]
                for offset in np.arange(min(corner_offsets), max(corner_offsets) + 1e-9, 0.15):
                    loss = line_loss(theta, float(offset))
                    if loss < best[0]:
                        best = (loss, float(theta), float(offset))
            for dtheta in np.deg2rad(np.arange(-3.0, 3.01, 0.5)):
                for doffset in np.arange(-0.15, 0.151, 0.03):
                    loss = line_loss(best[1] + float(dtheta), best[2] + float(doffset))
                    if loss < best[0]:
                        best = (loss, best[1] + float(dtheta), best[2] + float(doffset))
            line_loss_px, theta, offset = best
            points, ground_points, uv_curve, valid, direction2 = curve_for(theta, offset)
            valid_indices = np.nonzero(valid)[0]
            contact_index: Dict[int, int] = {}
            reprojection: Dict[int, float] = {}
            for frame_index, pixel in observations.items():
                distances = np.linalg.norm(uv_curve[valid] - pixel, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX:
                    contact_index[frame_index] = int(valid_indices[index])
                    reprojection[frame_index] = float(distances[index])
            if len(contact_index) < PHYSION_PP_AGENT_RAY_MIN_MATCHED:
                summary["reason"] = f"too few surface-matched frames ({len(contact_index)})"
                return

            first_contact = min(contact_index)
            airborne_bottom: Dict[int, np.ndarray] = {}
            prefix = [frame for frame in observations if frame < first_contact and frame not in contact_index]
            if prefix:
                impact = points[contact_index[first_contact]]
                heights = np.arange(0.0, 6.001, 0.02)
                vertical_points = impact[None, :] + heights[:, None] * up_axis[None, :]
                K = np.asarray(self._intrinsic_for_frame(intrinsics, 0), dtype=np.float64)
                vertical_uv = (K @ (vertical_points.T / np.maximum(vertical_points[:, 2], 1e-6))).T[:, :2]
                valid_vertical = vertical_points[:, 2] > 0.5
                for frame_index in prefix:
                    distances = np.linalg.norm(vertical_uv[valid_vertical] - observations[frame_index], axis=1)
                    index = int(np.argmin(distances))
                    if distances[index] <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX:
                        airborne_bottom[frame_index] = vertical_points[np.nonzero(valid_vertical)[0][index]]

            def clearance_height(gp: np.ndarray, radius: float) -> float:
                ci = int(np.clip(round((gp[0] - grid_min[0]) / cell), 0, nu - 1))
                cj = int(np.clip(round((gp[1] - grid_min[1]) / cell), 0, nv - 1))
                reach = int(np.ceil(float(radius) / cell))
                i0, i1 = max(0, ci - reach), min(nu, ci + reach + 1)
                j0, j1 = max(0, cj - reach), min(nv, cj + reach + 1)
                grid_i, grid_j = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
                sample_gp_u = grid_min[0] + grid_i * cell
                sample_gp_v = grid_min[1] + grid_j * cell
                distance2 = (sample_gp_u - gp[0]) ** 2 + (sample_gp_v - gp[1]) ** 2
                within = distance2 <= float(radius) ** 2 + 1e-12
                caps = raster[i0:i1, j0:j1] + np.sqrt(
                    np.maximum(0.0, float(radius) ** 2 - distance2)
                )
                return max(ground + float(radius), float(np.max(caps[within])))

            def clearance_center(curve_index: int, radius: float) -> np.ndarray:
                gp = ground_points[int(curve_index)]
                center_height = clearance_height(gp, radius)
                center = gp[0] * gu + gp[1] * gv
                return center + (center_height - float(center @ up_axis)) * up_axis

            def clearance_margin(center: np.ndarray, radius: float) -> float:
                gp = np.array([float(center @ gu), float(center @ gv)], dtype=np.float64)
                return float(center @ up_axis) - clearance_height(gp, radius)

            local_radius = max(1, int(round(PHYSION_PP_FP_SPHERE_PATH_LOCAL_SPAN_M / sample_step)))
            fit_frames_all = sorted([*contact_index, *airborne_bottom])
            if len(fit_frames_all) > PHYSION_PP_AGENT_SPHERE_FIT_MAX_FRAMES:
                selected = np.linspace(
                    0, len(fit_frames_all) - 1, PHYSION_PP_AGENT_SPHERE_FIT_MAX_FRAMES, dtype=int
                )
                fit_frames = [fit_frames_all[index] for index in sorted(set(selected.tolist()))]
            else:
                fit_frames = fit_frames_all

            def best_frame_center(radius: float, frame_index: int) -> tuple[np.ndarray, float] | None:
                target = target_masks.get(frame_index)
                if target is None:
                    return None
                K = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                if frame_index in airborne_bottom:
                    impact_index = contact_index[first_contact]
                    impact = points[impact_index]
                    height_above_impact = max(
                        0.0,
                        float((airborne_bottom[frame_index] - impact) @ up_axis),
                    )
                    center = clearance_center(impact_index, radius) + height_above_impact * up_axis
                    value = _bool_mask_iou(
                        _analytic_sphere_mask(
                            center_camera=center,
                            radius=radius,
                            intrinsic=K,
                            image_shape=image_shape,
                        ),
                        target,
                    )
                    return (center, float(value)) if value is not None else None
                base_index = contact_index.get(frame_index)
                if base_index is None:
                    return None
                candidates = [
                    index
                    for index in range(
                        max(0, base_index - local_radius),
                        min(len(points), base_index + local_radius + 1),
                    )
                    if valid[index]
                ]
                best_candidate = None
                for index in candidates:
                    center = clearance_center(index, radius)
                    value = _bool_mask_iou(
                        _analytic_sphere_mask(
                            center_camera=center,
                            radius=radius,
                            intrinsic=K,
                            image_shape=image_shape,
                        ),
                        target,
                    )
                    if value is not None and (best_candidate is None or value > best_candidate[1]):
                        best_candidate = (center, float(value))
                return best_candidate

            score_cache: Dict[float, tuple[float, Dict[int, np.ndarray]]] = {}

            def trajectory_score(radius: float) -> float:
                key = round(float(radius), 9)
                if key not in score_cache:
                    centers = {}
                    values = []
                    for frame_index in fit_frames:
                        result = best_frame_center(float(radius), frame_index)
                        if result is None:
                            continue
                        centers[frame_index], value = result
                        values.append(value)
                    score_cache[key] = (
                        float(np.mean(values)) if values else float("-inf"),
                        centers,
                    )
                return score_cache[key][0]

            seed_centers = []
            seed_masks = []
            seed_intrinsics = []
            for frame_index in fit_frames:
                if frame_index in contact_index:
                    center = points[contact_index[frame_index]] + 0.1 * up_axis
                else:
                    center = airborne_bottom[frame_index] + 0.1 * up_axis
                seed_centers.append(center)
                seed_masks.append(target_masks[frame_index])
                seed_intrinsics.append(self._intrinsic_for_frame(intrinsics, frame_index))
            seed, pixel_floor, seed_summary = _sphere_radius_seed(
                centers=seed_centers,
                masks=seed_masks,
                intrinsics=seed_intrinsics,
            )
            camera_cap = 0.8 * min(float(np.linalg.norm(center)) for center in seed_centers)
            radius, fit_iou, radius_search = _adaptive_sphere_radius_search(
                seed_radius=seed,
                pixel_floor=pixel_floor,
                camera_cap=camera_cap,
                objective=trajectory_score,
            )
            final_centers = {}
            final_ious = []
            for frame_index in fit_frames_all:
                result = best_frame_center(radius, frame_index)
                if result is None:
                    continue
                final_centers[frame_index], value = result
                final_ious.append(value)
            if len(final_centers) < PHYSION_PP_AGENT_SPHERE_MIN_FRAMES:
                summary["reason"] = f"too few final sphere poses ({len(final_centers)})"
                return
            mesh_path = self._adopt_physion_pp_agent_sphere(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                object_ids=[agent_id],
                radius=radius,
                centers_by_object_frame={agent_id: final_centers},
                source=summary["source"],
                output_subdir="friction_platform_sphere_refinement",
            )
            sequence = [float(final_centers[frame] @ (direction2[0] * gu + direction2[1] * gv)) for frame in sorted(final_centers)]
            clearance_margins = [
                clearance_margin(center, radius) for center in final_centers.values()
            ]
            summary.update(
                {
                    "applied": True,
                    "agent_object_id": agent_id,
                    "static_object_ids": used_static_ids,
                    "agent_path_source": "independent_fp_free_line_on_final_support",
                    "static_line_layout_reused_for_agent": False,
                    "line_theta_deg": round(float(np.degrees(theta) % 180.0), 3),
                    "line_offset_from_support_center_m": round(float(offset), 4),
                    "line_trimmed_bottom_reprojection_px": round(float(line_loss_px), 3),
                    "radius_m": round(float(radius), 6),
                    "radius_seed": seed_summary,
                    "radius_search": radius_search,
                    "fit_iou": round(float(fit_iou), 6),
                    "all_frame_iou": round(float(np.mean(final_ious)), 6) if final_ious else None,
                    "mask_observation_frames": len(observations),
                    "surface_frames": len(contact_index),
                    "airborne_frames": sorted(airborne_bottom),
                    "final_pose_frames": len(final_centers),
                    "clearance_policy": "full_lower_hemisphere_height_field_non_penetration",
                    "minimum_clearance_margin_m": round(float(min(clearance_margins)), 6),
                    "max_along_path_jump_m": (
                        round(float(np.max(np.abs(np.diff(sequence)))), 4) if len(sequence) > 1 else 0.0
                    ),
                    "sphere_mesh_path": mesh_path,
                }
            )
            _log_tool(
                self.tool_name,
                f"friction_platform sphere agent={agent_id} frames={len(final_centers)} "
                f"radius={radius:.4f} iou={summary['all_frame_iou']}",
            )
        except Exception as exc:
            summary["reason"] = f"friction-platform sphere refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(self.tool_name, f"friction-platform sphere error={_short_text(str(exc), 300)}")

    def _apply_physion_pp_agent_ray_native_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """Refine a Physion++ agent trajectory on the support manifold.

        The agent position is re-derived per frame as the point on a "line-on-support-
        surface" curve whose reprojection matches the observed SAM3 bottom pixel; the
        FoundationPose depth never enters. Rotations are preserved. Applied AFTER the
        static flush refinement so the support manifold is the final corrected world.
        Failure of any step leaves the trajectory untouched (reason recorded)."""
        summary: Dict[str, Any] = {"applied": False, "source": "physion_pp_agent_ray_manifold_refinement"}
        payload["physion_pp_agent_trajectory_refinement"] = summary
        scenario = self._object_plan_scenario(object_plan).lower()
        if not scenario.endswith("_pp"):
            summary["reason"] = "not a physion_pp scenario"
            return
        if scenario not in PHYSION_PP_AGENT_RAY_SCENARIOS:
            summary["reason"] = "agent ray refinement is restricted to friction_platform_pp"
            return
        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        role_binding = (tracks_payload.get("physion_tracking") or {}).get("role_binding") or {}
        agent_track = str(role_binding.get("agent_track") or "")
        if not agent_track:
            summary["reason"] = "no agent track in role binding"
            return
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        objects = {str(i.get("object_id")): i for i in corrected.get("objects", []) if isinstance(i, dict)}
        agent_id = next(
            (str(t.object_id) for t in object_plan.target_objects if str(t.source_track_id) == agent_track),
            "",
        )
        static_ids = [
            str(t.object_id)
            for t in object_plan.target_objects
            if _is_physion_static_ground_fixture_track(t.source_track_id)
        ]
        if not agent_id or agent_id not in objects or not static_ids:
            summary["reason"] = "agent or static objects missing from corrected trajectories"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        support = payload.get("support_plane_position_correction") or {}
        ground = support.get("ground_height_along_normal")
        if up_axis is None or ground is None:
            summary["reason"] = "missing up axis or ground height"
            return
        ground = float(ground)
        support_by_id = {str(i.get("object_id")): i for i in support.get("objects", []) if isinstance(i, dict)}
        try:
            import trimesh

            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)

            u_axis, v_axis = _plane_basis_for_up_axis(up_axis)
            K0 = self._intrinsic_for_frame(intrinsics, 0)
            # --- support surface height raster over the FINAL corrected statics ---
            points = []
            static_depth = np.full(image_shape, np.inf, dtype=np.float32)
            static_depth_object_ids = []
            for oid in static_ids:
                item = objects.get(oid)
                pose_entry = next(
                    (p for p in (item or {}).get("poses", []) if isinstance(p, dict) and p.get("corrected_pose_4x4") is not None),
                    None,
                )
                mesh_path = (support_by_id.get(oid) or {}).get("mesh_path") or mesh_paths.get(oid)
                if pose_entry is None or not mesh_path:
                    continue
                pose = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                vertices, faces = self._load_mesh_geometry(Path(mesh_path))
                vertices_camera = vertices @ pose[:3, :3].T + pose[:3, 3]
                rendered_mask, rendered_depth = render_mesh_depth(
                    vertices_camera=vertices_camera,
                    faces=faces,
                    intrinsic=K0,
                    image_shape=image_shape,
                )
                rendered_valid = (
                    np.asarray(rendered_mask, dtype=bool)
                    & np.isfinite(rendered_depth)
                    & (rendered_depth > 0.0)
                )
                if rendered_valid.any():
                    static_depth[rendered_valid] = np.minimum(
                        static_depth[rendered_valid], rendered_depth[rendered_valid]
                    )
                    static_depth_object_ids.append(oid)
                # Vertex-only sampling underestimates decimated surfaces between vertices
                # (~16 cm vertex spacing vs 6 cm cells) and buried the agent 0.1-0.25 m
                # into ramp ridges. Subdividing to the cell size makes the vertex set a
                # faithful surface sampling; with the holes gone, the 1-cell max-dilation
                # that used to bridge them only added 3-6 cm of float and is dropped.
                dense = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).subdivide_to_size(
                    max_edge=PHYSION_PP_AGENT_RAY_CELL_M, max_iter=16
                )
                posed = np.asarray(dense.vertices, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
                # Pathological-mesh memory fuse only -- never a tuning knob. Thinning below
                # ~1 vertex per cell reopens raster holes (which now fall straight to
                # ground), so this must stay far above any real fixture (largest observed
                # friction ramp: 360k vertices).
                if len(posed) > 2_000_000:
                    posed = posed[np.linspace(0, len(posed) - 1, 2_000_000, dtype=np.int64)]
                points.append(posed)
            if not points:
                summary["reason"] = "no static surface points"
                return
            static_depth_valid = np.isfinite(static_depth)
            if not static_depth_valid.any():
                summary["reason"] = "final static meshes rendered no camera depth"
                return
            static_depth[~static_depth_valid] = 0.0
            summary["static_depth_visibility"] = {
                "applied": True,
                "source": "final_corrected_static_meshes",
                "object_ids": static_depth_object_ids,
                "covered_pixels": int(static_depth_valid.sum()),
                "image_pixels": int(static_depth_valid.size),
                "depth_tolerance_m": PHYSION_PP_AGENT_RAY_STATIC_DEPTH_TOL_M,
            }
            surface = np.vstack(points)
            surface_gp = np.stack([surface @ u_axis, surface @ v_axis], 1)
            surface_h = surface @ up_axis
            cell = PHYSION_PP_AGENT_RAY_CELL_M
            grid_min = surface_gp.min(0) - 0.5
            grid_max = surface_gp.max(0) + 0.5
            nu = int(np.ceil((grid_max[0] - grid_min[0]) / cell)) + 1
            nv = int(np.ceil((grid_max[1] - grid_min[1]) / cell)) + 1
            raster = np.full((nu, nv), ground)
            iu = np.clip(((surface_gp[:, 0] - grid_min[0]) / cell).astype(int), 0, nu - 1)
            iv = np.clip(((surface_gp[:, 1] - grid_min[1]) / cell).astype(int), 0, nv - 1)
            np.maximum.at(raster, (iu, iv), surface_h)

            def support_heights(ground_points: np.ndarray) -> np.ndarray:
                ii = np.clip(((ground_points[:, 0] - grid_min[0]) / cell).astype(int), 0, nu - 1)
                jj = np.clip(((ground_points[:, 1] - grid_min[1]) / cell).astype(int), 0, nv - 1)
                return np.maximum(ground, raster[ii, jj])

            # --- per-frame observation: bottom pixel of the agent mask + camera ray ---
            agent_masks = mask_records.get(agent_id) or {}
            K0_inv = np.linalg.inv(np.asarray(K0, dtype=np.float64))
            agent_vertices, agent_faces = self._load_mesh_geometry(Path(mesh_paths[agent_id]))
            agent_centroid = agent_vertices.mean(axis=0)
            agent_item = objects[agent_id]
            pose_by_frame: Dict[int, np.ndarray] = {}
            for p in agent_item.get("poses", []):
                frame_index = self._pose_frame_index(p)
                if frame_index is None or p.get("corrected_pose_4x4") is None:
                    continue
                pose_by_frame[frame_index] = np.asarray(p["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
            observations: Dict[int, np.ndarray] = {}
            rays: Dict[int, np.ndarray] = {}
            old_depth: Dict[int, float] = {}
            for frame_index, pose in pose_by_frame.items():
                record = agent_masks.get(frame_index) or {}
                mask = self._resize_mask_to_shape(mask_arrays.get(str(record.get("mask_key") or "")), image_shape)
                if mask is None or not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                y_bottom = int(ys.max())
                x_bottom = float(xs[ys >= y_bottom - 1].mean())
                observations[frame_index] = np.array([x_bottom, y_bottom + 0.5])
                ray = K0_inv @ np.array([x_bottom, y_bottom + 0.5, 1.0])
                rays[frame_index] = ray / np.linalg.norm(ray)
                old_depth[frame_index] = float((agent_centroid @ pose[:3, :3].T + pose[:3, 3])[2])
            if len(observations) < PHYSION_PP_AGENT_RAY_MIN_MATCHED:
                summary["reason"] = "too few agent mask observations"
                return

            # --- search center: closest-approach contact anchors (fallback: FP projections) ---
            anchor_gp = []
            for frame_index, ray in rays.items():
                lam0 = old_depth[frame_index] / ray[2]
                lams = np.linspace(max(1.0, lam0 - 7.0), lam0 + 7.0, 141)
                pts3 = lams[:, None] * ray[None, :]
                gps = np.stack([pts3 @ u_axis, pts3 @ v_axis], 1)
                g = pts3 @ up_axis - support_heights(gps)
                i = int(np.argmin(np.abs(g)))
                if abs(float(g[i])) <= 0.10:
                    anchor_gp.append(gps[i])
            if len(anchor_gp) >= 5:
                center0 = np.asarray(anchor_gp, dtype=np.float64).mean(0)
            else:
                fp_gp = np.array(
                    [[(old_depth[f] * rays[f]) @ u_axis, (old_depth[f] * rays[f]) @ v_axis] for f in rays]
                )
                center0 = np.median(fp_gp, axis=0)

            # --- direct 2-DOF line search on all-frame reprojection error ---
            observed = np.stack([observations[f] for f in sorted(observations)])
            xs_grid = np.linspace(-9.0, 9.0, 601)

            def curve_for(theta: float, offset: float):
                direction = np.array([np.cos(theta), np.sin(theta)])
                normal = np.array([-direction[1], direction[0]])
                base = center0 + normal * offset
                ground_points = base[None, :] + xs_grid[:, None] * direction[None, :]
                heights = support_heights(ground_points)
                pts3 = ground_points[:, 0:1] * u_axis[None, :] + ground_points[:, 1:2] * v_axis[None, :]
                pts3 = pts3 + (heights - pts3 @ up_axis)[:, None] * up_axis[None, :]
                depth_valid = pts3[:, 2] > 0.5
                uv = (np.asarray(K0, dtype=np.float64) @ (pts3.T / np.maximum(pts3[:, 2], 1e-6))).T[:, :2]
                visible, occluded = _visible_against_static_depth(
                    points_camera=pts3,
                    projected_uv=uv,
                    valid=depth_valid,
                    static_depth=static_depth,
                    tolerance_m=PHYSION_PP_AGENT_RAY_STATIC_DEPTH_TOL_M,
                )
                return pts3, uv, visible, occluded

            def line_loss(theta: float, offset: float) -> float:
                _, uv, valid, _occluded = curve_for(theta, offset)
                uv = uv[valid]
                if len(uv) < 10:
                    return 1e9
                dmin = np.sqrt(((observed[:, None, :] - uv[None, :, :]) ** 2).sum(-1)).min(1)
                k = max(1, int(len(dmin) * PHYSION_PP_AGENT_RAY_TRIM_FRACTION))
                return float(np.sort(dmin)[:k].mean())

            best = (1e9, 0.0, 0.0)
            for theta in np.deg2rad(np.arange(0.0, 180.0, 3.0)):
                for offset in np.arange(-2.0, 2.01, 0.15):
                    loss = line_loss(theta, offset)
                    if loss < best[0]:
                        best = (loss, theta, offset)
            for d_theta in np.deg2rad(np.arange(-3.0, 3.01, 0.5)):
                for d_offset in np.arange(-0.15, 0.151, 0.03):
                    loss = line_loss(best[1] + d_theta, best[2] + d_offset)
                    if loss < best[0]:
                        best = (loss, best[1] + d_theta, best[2] + d_offset)
            loss_px, theta, offset = best
            pts3, uv_curve, valid, curve_occluded = curve_for(theta, offset)
            valid_indices = np.nonzero(valid)[0]

            # --- per-frame matching -> contact trajectory ---
            contact: Dict[int, np.ndarray] = {}
            x_along: Dict[int, float] = {}
            reproj: Dict[int, float] = {}
            occluded_near_observations = 0
            for frame_index, pix in observations.items():
                all_dists = np.linalg.norm(uv_curve - pix, axis=1)
                occluded_near_observations += int(
                    np.count_nonzero(
                        curve_occluded
                        & (all_dists <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX)
                    )
                )
                dists = np.linalg.norm(uv_curve[valid] - pix, axis=1)
                i = int(np.argmin(dists))
                if dists[i] <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX:
                    gi = int(valid_indices[i])
                    contact[frame_index] = pts3[gi]
                    x_along[frame_index] = float(xs_grid[gi])
                    reproj[frame_index] = float(dists[i])
            if len(contact) < PHYSION_PP_AGENT_RAY_MIN_MATCHED:
                summary["reason"] = f"too few matched frames ({len(contact)})"
                return

            # --- falling-phase extension: frames BEFORE the first surface match are
            # airborne; their ground projection is pinned to the impact point (the first
            # matched frame's ground position, on the same motion line), so the object's
            # bottom travels the VERTICAL line through it. Same reprojection matching,
            # no explicit phase classifier: manifold reachability does the split.
            airborne: Dict[int, np.ndarray] = {}
            airborne_reproj: Dict[int, float] = {}
            airborne_occluded_candidates = 0
            first_contact_frame = min(contact)
            prefix_frames = [f for f in observations if f < first_contact_frame and f not in contact]
            if prefix_frames:
                impact_point = contact[first_contact_frame]
                t_grid = np.arange(0.0, 6.001, 0.02)
                vertical_pts = impact_point[None, :] + t_grid[:, None] * up_axis[None, :]
                v_depth_valid = vertical_pts[:, 2] > 0.5
                vertical_uv = (
                    np.asarray(K0, dtype=np.float64) @ (vertical_pts.T / np.maximum(vertical_pts[:, 2], 1e-6))
                ).T[:, :2]
                v_valid, v_occluded = _visible_against_static_depth(
                    points_camera=vertical_pts,
                    projected_uv=vertical_uv,
                    valid=v_depth_valid,
                    static_depth=static_depth,
                    tolerance_m=PHYSION_PP_AGENT_RAY_STATIC_DEPTH_TOL_M,
                )
                v_indices = np.nonzero(v_valid)[0]
                for frame_index in prefix_frames:
                    pix = observations[frame_index]
                    all_dists = np.linalg.norm(vertical_uv - pix, axis=1)
                    airborne_occluded_candidates += int(
                        np.count_nonzero(
                            v_occluded
                            & (all_dists <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX)
                        )
                    )
                    if not len(v_indices):
                        continue
                    dists = np.linalg.norm(vertical_uv[v_valid] - pix, axis=1)
                    i = int(np.argmin(dists))
                    if dists[i] <= PHYSION_PP_AGENT_RAY_REPROJ_TOL_PX:
                        airborne[frame_index] = vertical_pts[int(v_indices[i])]
                        airborne_reproj[frame_index] = float(dists[i])

            # --- one-shot scale from contact-depth ratio ---
            ratios = [contact[f][2] / old_depth[f] for f in contact if old_depth.get(f, 0.0) > 0.0]
            scale = float(np.median(ratios))
            scaled_vertices = agent_vertices * scale
            mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "agent_ray_refinement"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            scaled_mesh_path = mesh_dir / f"{agent_id}_ray_scaled_local.glb"
            self._export_mesh_with_source_colors(
                vertices=scaled_vertices,
                faces=agent_faces,
                source_mesh_path=Path(mesh_paths[agent_id]),
                output_path=scaled_mesh_path,
            )
            support_agent = support_by_id.get(agent_id)
            if isinstance(support_agent, dict):
                support_agent["source_mesh_path"] = support_agent.get("mesh_path")
                support_agent["mesh_path"] = str(scaled_mesh_path)
                support_agent["mesh_source"] = "physion_pp_agent_ray_refinement"

            # --- rewrite the agent trajectory (rotation preserved, position from the
            # matched bottom point: surface contact or airborne vertical line) ---
            unmatched = []
            for p in agent_item.get("poses", []):
                frame_index = self._pose_frame_index(p)
                if frame_index is None or p.get("corrected_pose_4x4") is None:
                    continue
                if frame_index in contact:
                    bottom_point = contact[frame_index]
                    phase = "surface"
                    err_px = reproj[frame_index]
                elif frame_index in airborne:
                    bottom_point = airborne[frame_index]
                    phase = "airborne_vertical"
                    err_px = airborne_reproj[frame_index]
                else:
                    unmatched.append(frame_index)
                    p["agent_ray_refined"] = False
                    continue
                pose = np.asarray(p["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                axis_local = pose[:3, :3].T @ up_axis
                half_height = -float(np.percentile((scaled_vertices - scaled_vertices.mean(0)) @ axis_local, 2.0))
                center = bottom_point + up_axis * half_height
                pose[:3, 3] = center - pose[:3, :3] @ scaled_vertices.mean(0)
                p["corrected_pose_4x4"] = pose.tolist()
                p["corrected_translation_camera"] = pose[:3, 3].tolist()
                p["agent_ray_refined"] = True
                p["agent_ray_phase"] = phase
                p["agent_ray_reprojection_px"] = round(err_px, 2)
                if frame_index in x_along:
                    p["agent_ray_x_along_line"] = round(x_along[frame_index], 3)

            frames_sorted = sorted(contact)
            xs_seq = np.array([x_along[f] for f in frames_sorted])
            sign = 1.0 if xs_seq[-1] >= xs_seq[0] else -1.0
            direction = np.array([np.cos(theta), np.sin(theta)])
            summary["static_depth_visibility"].update(
                {
                    "selected_curve_samples": int(len(curve_occluded)),
                    "selected_curve_occluded_samples": int(curve_occluded.sum()),
                    "occluded_candidates_near_observations": int(
                        occluded_near_observations + airborne_occluded_candidates
                    ),
                }
            )
            summary.update(
                {
                    "applied": True,
                    "agent_object_id": agent_id,
                    "line_direction_ground": (sign * direction).tolist(),
                    "line_offset_m": float(offset),
                    "line_trimmed_loss_px": round(float(loss_px), 2),
                    "matched_frames": len(contact),
                    "airborne_frames": sorted(airborne),
                    "observation_frames": len(observations),
                    "unmatched_frames": unmatched,
                    "monotonic_fraction": round(float(np.mean(sign * np.diff(xs_seq) >= -0.03)), 3),
                    "reprojection_p90_px": round(float(np.percentile(list(reproj.values()), 90)), 2),
                    "scale": round(scale, 4),
                    "scale_ratio_p10_p90": [
                        round(float(np.percentile(ratios, 10)), 4),
                        round(float(np.percentile(ratios, 90)), 4),
                    ],
                    "scaled_mesh_path": str(scaled_mesh_path),
                    "slide_range_m": [round(float(xs_seq[0]), 3), round(float(xs_seq[-1]), 3)],
                }
            )
            _log_tool(
                self.tool_name,
                f"agent_ray_refinement agent={agent_id} matched={len(contact)}/{len(observations)} "
                f"loss={loss_px:.2f}px scale={scale:.3f} mono={summary['monotonic_fraction']}",
            )
        except Exception as exc:
            summary["reason"] = f"agent ray refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(self.tool_name, f"agent_ray_refinement error={_short_text(str(exc), 300)}")

    def _apply_physion_pp_bounce_trajectory_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        """Physion++ bouncy_platform airborne-agent trajectory (mask-only, zero depth).

        The agent ground projection is constrained to the shared collinear line
        (line-layout theta,c); lifting that line to a vertical plane P removes the single
        depth DOF the mask centroid leaves free, so each frame's centroid ray meets P at
        a unique 3D ball center. The agent is refit as a sphere (rotation-free) whose one
        global radius maximizes rendered-mask IoU. Adopted only if it tracks the mask at
        least as well as the current FoundationPose box; else the FP trajectory is kept.
        """
        route_record = self._agent_trajectory_route_record(object_plan)
        if route_record is None:
            return
        if (
            route_record.get("decision_id")
            != BOUNCY_PLATFORM_AGENT_TRAJECTORY_DECISION_ID
        ):
            return
        module = self._module_for_route_record(
            route_record,
            decision_id=BOUNCY_PLATFORM_AGENT_TRAJECTORY_DECISION_ID,
            module_name="agent_trajectory",
        )
        if module.implementation != "line_vertical_plane_mask_sphere_trajectory":
            raise ValueError(
                "unsupported bouncy-platform agent-trajectory module implementation: "
                f"{module.implementation!r}"
            )
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "trajectory.bounce_mask",
            "resolved_route": deepcopy(route_record),
        }
        payload["physion_pp_agent_trajectory_refinement"] = summary
        payload["agent_trajectory_route"] = deepcopy(route_record)

        line_layout = payload.get("physion_pp_line_layout_refinement") or {}
        if line_layout.get("applied") is not True or line_layout.get("line_theta_deg") is None:
            summary["reason"] = "line layout not applied (no line orientation)"
            return
        theta = float(np.deg2rad(float(line_layout["line_theta_deg"])))
        # full mode gives the offset c; two_static mode gives only orientation (c=None) and
        # c is derived below from the static centers once the meshes are loaded.
        c_offset = line_layout.get("line_offset_m")
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        support = payload.get("support_plane_position_correction") or {}
        ground = support.get("ground_height_along_normal")
        if up_axis is None or ground is None:
            summary["reason"] = "missing up axis or ground height"
            return
        ground = float(ground)
        gu, gv = _plane_basis_for_up_axis(up_axis)
        plane_normal = -np.sin(theta) * gu + np.cos(theta) * gv  # in-ground normal of the line
        line_dir = np.cos(theta) * gu + np.sin(theta) * gv       # along-line direction

        tracks_payload = self.artifacts.read_optional(
            self.artifacts.artifact_path_by_name(question_dir, "sam3_video_tracks.json")
        ) or {}
        role_binding = (tracks_payload.get("physion_tracking") or {}).get("role_binding") or {}
        agent_track = str(role_binding.get("agent_track") or "")
        if not agent_track:
            summary["reason"] = "no agent track in role binding"
            return
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        objects = {str(i.get("object_id")): i for i in corrected.get("objects", []) if isinstance(i, dict)}
        agent_id = next(
            (str(t.object_id) for t in object_plan.target_objects if str(t.source_track_id) == agent_track),
            "",
        )
        if not agent_id or agent_id not in objects:
            summary["reason"] = "agent missing from corrected trajectories"
            return
        support_by_id = {str(i.get("object_id")): i for i in support.get("objects", []) if isinstance(i, dict)}
        agent_item = objects[agent_id]
        try:
            import trimesh

            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            agent_masks = mask_records.get(agent_id) or {}

            # --- two_static gate: line-layout only snapped the fixtures' elongation theta
            #     and did NOT move them onto a common line (c=None). Fit the agent line the
            #     SAME way full mode does it -- the PCA/connecting line THROUGH the static
            #     OBB centers. For 2 points this is exact (both lie on it), unlike the
            #     elongation theta, which need not pass through them. Overrides the theta-
            #     based plane_normal/line_dir set above. ---
            two_static_derived = c_offset is None
            if two_static_derived:
                static_ids = [
                    str(t.object_id)
                    for t in object_plan.target_objects
                    if _is_physion_static_ground_fixture_track(t.source_track_id)
                ]
                static_centers = []
                for oid in static_ids:
                    item = objects.get(oid)
                    pose_entry = next(
                        (p for p in (item or {}).get("poses", [])
                         if isinstance(p, dict) and p.get("corrected_pose_4x4") is not None),
                        None,
                    )
                    mesh_path = (support_by_id.get(oid) or {}).get("mesh_path") or mesh_paths.get(oid)
                    if pose_entry is None or not mesh_path:
                        continue
                    pose = np.asarray(pose_entry["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    vertices, _f = self._load_mesh_geometry(Path(mesh_path))
                    to_origin, _ext = trimesh.bounds.oriented_bounds(vertices)
                    obb_center = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))[:3, 3]
                    static_centers.append(pose[:3, :3] @ obb_center + pose[:3, 3])
                if len(static_centers) < 2:
                    summary["reason"] = "two_static line needs >= 2 static centers"
                    return
                centers2d = np.stack([[float(c @ gu), float(c @ gv)] for c in static_centers])
                spread = float(np.linalg.norm(centers2d.max(0) - centers2d.min(0)))
                if spread < 0.1:
                    summary["reason"] = f"static centers too close to define a line ({spread:.3f} m)"
                    return
                _u, _s, vt = np.linalg.svd(centers2d - centers2d.mean(0), full_matrices=False)
                d2 = vt[0] / np.linalg.norm(vt[0])
                line_dir = float(d2[0]) * gu + float(d2[1]) * gv
                plane_normal = float(-d2[1]) * gu + float(d2[0]) * gv
                c_offset = float(np.mean([float(c @ plane_normal) for c in static_centers]))
            else:
                c_offset = float(c_offset)

            # --- per-frame mask centroid -> camera ray -> intersection with plane P ---
            centers: Dict[int, np.ndarray] = {}
            target_masks: Dict[int, np.ndarray] = {}
            areas: Dict[int, int] = {}
            for frame_index, record in agent_masks.items():
                mask = self._resize_mask_to_shape(
                    mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                )
                if mask is None or not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                if xs.size < 3:
                    continue
                intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                ray = np.linalg.inv(intrinsic) @ np.array([float(xs.mean()), float(ys.mean()), 1.0])
                ray = ray / np.linalg.norm(ray)
                denom = float(ray @ plane_normal)
                if abs(denom) < PHYSION_PP_BOUNCE_MIN_RAY_PLANE_DOT:
                    continue
                depth_scale = c_offset / denom
                if depth_scale <= 0:
                    continue
                centers[int(frame_index)] = ray * depth_scale
                target_masks[int(frame_index)] = mask
                areas[int(frame_index)] = int(xs.size)
            if len(centers) < PHYSION_PP_BOUNCE_MIN_MATCHED:
                summary["reason"] = f"too few solvable frames ({len(centers)})"
                return

            sphere_geometry_cache: Dict[float, tuple[np.ndarray, np.ndarray]] = {}

            def sphere_geometry(radius: float) -> tuple[np.ndarray, np.ndarray]:
                key = float(radius)
                cached = sphere_geometry_cache.get(key)
                if cached is None:
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=key)
                    cached = (
                        np.asarray(sphere.vertices, dtype=np.float64),
                        np.asarray(sphere.faces),
                    )
                    sphere_geometry_cache[key] = cached
                return cached

            def sphere_iou_at(radius: float, frame_index: int) -> float | None:
                verts, faces = sphere_geometry(radius)
                rendered_mask = render_mesh_mask_cuda(
                    vertices_camera=verts + centers[frame_index],
                    faces=faces,
                    intrinsic=np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64),
                    image_shape=image_shape,
                )
                return self._mask_iou(rendered_mask, target_masks[frame_index])

            def sphere_mean_iou(radius: float, frames: list) -> float:
                ious = [sphere_iou_at(radius, f) for f in frames]
                ious = [v for v in ious if v is not None]
                return float(np.mean(ious)) if ious else float("-inf")

            # --- fit ONE global radius (search on a frame subsample; radius is global) ---
            fit_frames = [f for f in sorted(centers) if areas[f] >= PHYSION_PP_BOUNCE_FIT_MIN_AREA_PX] or sorted(centers)
            fit_sample = fit_frames[:: max(1, len(fit_frames) // 20)]
            reset_cuda_mask_rasterizer_stats()
            radius_search_started = time.monotonic()
            start, stop, step = PHYSION_PP_BOUNCE_RADIUS_COARSE_M
            coarse = [(float(r), sphere_mean_iou(float(r), fit_sample)) for r in np.arange(start, stop + 1e-9, step)]
            best_radius = max(coarse, key=lambda item: item[1])[0]
            initial_winner_radius = float(best_radius)
            expanded_radii, expanded_direction = _boundary_extension_values(
                winner=initial_winner_radius,
                initial_start=start,
                initial_stop=stop,
                step=step,
                expanded_start=PHYSION_PP_BOUNCE_RADIUS_EXPANDED_M[0],
                expanded_stop=PHYSION_PP_BOUNCE_RADIUS_EXPANDED_M[1],
                boundary_steps=PHYSION_PP_BOUNCE_RADIUS_BOUNDARY_STEPS,
            )
            coarse.extend(
                (float(radius), sphere_mean_iou(float(radius), fit_sample))
                for radius in expanded_radii
            )
            best_radius = max(coarse, key=lambda item: item[1])[0]
            coarse_winner_radius = float(best_radius)
            fine_radii = np.arange(
                max(PHYSION_PP_BOUNCE_RADIUS_EXPANDED_M[0], best_radius - 0.015),
                min(PHYSION_PP_BOUNCE_RADIUS_EXPANDED_M[1], best_radius + 0.015) + 1e-9,
                PHYSION_PP_BOUNCE_RADIUS_FINE_STEP_M,
            )
            best_radius, best_radius_iou = max(
                [(float(r), sphere_mean_iou(float(r), fit_sample)) for r in fine_radii],
                key=lambda item: item[1],
            )

            # Sphere-vs-mask mean IoU over all solved frames (fit-quality diagnostic only;
            # FoundationPose is NEVER rendered or compared, per directive). The refinement
            # always applies -- the collinear prior is trusted, no adopt/reject gate.
            sphere_mean = sphere_mean_iou(best_radius, sorted(centers))

            # --- export the sphere mesh (uniform source-mean color); repoint the agent to
            #     it via the support entry (same convention as agent-ray) ---
            sphere_export_verts, sphere_export_faces = sphere_geometry(best_radius)
            mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "bounce_trajectory_refinement"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            sphere_mesh_path = mesh_dir / f"{agent_id}_bounce_sphere_local.glb"
            sphere_mesh = trimesh.Trimesh(vertices=sphere_export_verts, faces=sphere_export_faces, process=False)
            try:
                source_mesh = trimesh.load(Path(mesh_paths[agent_id]), force="mesh")
                if hasattr(source_mesh, "geometry"):
                    source_mesh = trimesh.util.concatenate(tuple(source_mesh.geometry.values()))
                source_colors = np.asarray(source_mesh.visual.vertex_colors)
                if source_colors.ndim == 2 and len(source_colors) > 0:
                    sphere_mesh.visual.vertex_colors = np.tile(
                        source_colors.mean(axis=0).astype(np.uint8), (len(sphere_export_verts), 1)
                    )
            except Exception:
                pass
            sphere_mesh.export(sphere_mesh_path)
            support_agent = support_by_id.get(agent_id)
            if isinstance(support_agent, dict):
                support_agent["source_mesh_path"] = support_agent.get("mesh_path")
                support_agent["mesh_path"] = str(sphere_mesh_path)
                support_agent["mesh_source"] = "physion_pp_bounce_trajectory_refinement"
            # Carry the sphere on the corrected-trajectory item too: the debug render /
            # downstream manifest reads the mesh from the trajectory object (preferred over
            # the foundationpose debug_mesh_paths fallback, which is still the box).
            agent_item["mesh_path"] = str(sphere_mesh_path)
            agent_item["mesh_source"] = "physion_pp_bounce_trajectory_refinement"

            # --- rewrite existing agent poses to the solved centers (rotation-free sphere:
            #     pose = identity R + translation); append poses for masked frames that had
            #     no FoundationPose entry (the ball is visible before FP registration) ---
            sphere_centroid = sphere_export_verts.mean(axis=0)

            def bounce_pose(center: np.ndarray) -> np.ndarray:
                pose = np.eye(4)
                pose[:3, 3] = center - sphere_centroid
                return pose

            heights: Dict[int, float] = {}
            existing_frames = set()
            for pose_entry in agent_item.get("poses", []):
                frame_index = self._pose_frame_index(pose_entry)
                if frame_index is None:
                    continue
                existing_frames.add(frame_index)
                if frame_index not in centers:
                    pose_entry["bounce_refined"] = False
                    continue
                pose = bounce_pose(centers[frame_index])
                pose_entry["corrected_pose_4x4"] = pose.tolist()
                pose_entry["corrected_translation_camera"] = pose[:3, 3].tolist()
                pose_entry["bounce_refined"] = True
                heights[frame_index] = float(centers[frame_index] @ up_axis - ground)
                pose_entry["bounce_height_m"] = round(heights[frame_index], 4)
            appended = 0
            for frame_index in sorted(centers):
                if frame_index in existing_frames:
                    continue
                pose = bounce_pose(centers[frame_index])
                heights[frame_index] = float(centers[frame_index] @ up_axis - ground)
                agent_item.setdefault("poses", []).append({
                    "frame_index": int(frame_index),
                    "corrected_pose_4x4": pose.tolist(),
                    "corrected_translation_camera": pose[:3, 3].tolist(),
                    "bounce_refined": True,
                    "bounce_height_m": round(heights[frame_index], 4),
                })
                appended += 1
            agent_item["poses"].sort(key=lambda p: self._pose_frame_index(p) or 0)

            frames_sorted = sorted(centers)
            s_seq = np.array([float(centers[f] @ line_dir) for f in frames_sorted])
            sign = 1.0 if s_seq[-1] >= s_seq[0] else -1.0
            summary.update({
                "applied": True,
                "agent_object_id": agent_id,
                "geometry": "sphere",
                "fitted_radius_m": round(float(best_radius), 4),
                "radius_search": {
                    "initial_range_m": [start, stop],
                    "boundary_band_steps": PHYSION_PP_BOUNCE_RADIUS_BOUNDARY_STEPS,
                    "initial_winner_m": round(initial_winner_radius, 4),
                    "expanded": bool(expanded_radii),
                    "expanded_direction": expanded_direction,
                    "evaluated_coarse_range_m": [
                        round(min([start, *expanded_radii]), 4),
                        round(max([stop, *expanded_radii]), 4),
                    ],
                    "coarse_winner_m": round(coarse_winner_radius, 4),
                    "final_winner_m": round(float(best_radius), 4),
                },
                "radius_search_elapsed_sec": round(
                    time.monotonic() - radius_search_started, 2
                ),
                "rasterizer_backend": "nvdiffrast_cuda",
                "gpu_rasterizer_stats": cuda_mask_rasterizer_stats(),
                "line_theta_deg": round(
                    float(np.degrees(np.arctan2(float(line_dir @ gv), float(line_dir @ gu))) % 180.0), 2
                ),
                "line_offset_m": round(float(c_offset), 4),
                "line_position_source": (
                    "two_static_pca_through_static_centers" if two_static_derived else "line_layout_full"
                ),
                "solved_frames": len(centers),
                "appended_frames": appended,
                "iou_sphere_mean": round(sphere_mean, 4),
                "iou_sphere_fit_mean": round(float(best_radius_iou), 4),
                "height_range_m": [
                    round(float(min(heights.values())), 3),
                    round(float(max(heights.values())), 3),
                ],
                "along_line_monotonic_fraction": (
                    round(float(np.mean(sign * np.diff(s_seq) >= -0.03)), 3) if len(s_seq) > 1 else 1.0
                ),
                "sphere_mesh_path": str(sphere_mesh_path),
            })
            _log_tool(
                self.tool_name,
                f"bounce_trajectory agent={agent_id} solved={len(centers)} (+{appended}) "
                f"radius={best_radius:.3f} iou_sphere={sphere_mean:.3f} "
                f"line={'two_static' if two_static_derived else 'full'}",
            )
        except Exception as exc:
            summary["reason"] = f"bounce trajectory refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(self.tool_name, f"bounce_trajectory error={_short_text(str(exc), 300)}")

    def _apply_physion_pp_bouncy_wall_trajectory_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
    ) -> None:
        route_record = self._agent_trajectory_route_record(object_plan)
        if route_record is None:
            return
        if (
            route_record.get("decision_id")
            != BOUNCY_WALL_AGENT_TRAJECTORY_DECISION_ID
        ):
            return
        policy, agent_ids = self._physion_pp_sphere_agent_context(
            question_dir=question_dir,
            object_plan=object_plan,
        )
        payload["physion_pp_agent_geometry_policy"] = policy
        payload["agent_trajectory_route"] = deepcopy(route_record)
        module = self._module_for_route_record(
            route_record,
            decision_id=BOUNCY_WALL_AGENT_TRAJECTORY_DECISION_ID,
            module_name="agent_trajectory",
        )
        if module.implementation == "wall_plane_native_mesh_fp_rotation":
            self._apply_physion_pp_bouncy_wall_native_trajectory_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                excluded_object_ids=set(),
            )
            summary = payload.get("physion_pp_agent_trajectory_refinement")
            if isinstance(summary, dict):
                summary["resolved_route"] = deepcopy(route_record)
            return
        if (
            module.implementation
            != "wall_plane_sphere_agents_native_non_agents"
        ):
            raise ValueError(
                "unsupported bouncy-wall agent-trajectory module implementation: "
                f"{module.implementation!r}"
            )
        self._apply_physion_pp_bouncy_wall_sphere_refinement(
            payload=payload,
            question_dir=question_dir,
            object_plan=object_plan,
            agent_ids=agent_ids,
        )
        sphere_summary = payload.get("physion_pp_agent_trajectory_refinement") or {}
        non_agent_summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_bouncy_wall_native_non_agent_refinement",
            "objects": [],
        }
        if any(
            "dynamic" in str(getattr(target, "role", "") or "").lower()
            and str(target.object_id) not in agent_ids
            for target in object_plan.target_objects
        ):
            self._apply_physion_pp_bouncy_wall_native_trajectory_refinement(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                excluded_object_ids=agent_ids,
            )
            non_agent_summary = payload.get("physion_pp_agent_trajectory_refinement") or non_agent_summary
        payload["physion_pp_agent_trajectory_refinement"] = {
            "applied": bool(sphere_summary.get("applied") or non_agent_summary.get("applied")),
            "source": "trajectory.wall_mixed_geometry",
            "resolved_route": deepcopy(route_record),
            "sphere_agents": sphere_summary,
            "native_non_agents": non_agent_summary,
        }

    def _apply_physion_pp_bouncy_wall_sphere_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        agent_ids: set[str],
    ) -> None:
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_bouncy_wall_sphere_refinement",
            "geometry": "sphere",
            "foundationpose_agent_policy": "skipped",
            "objects": [],
        }
        payload["physion_pp_agent_trajectory_refinement"] = summary
        if not agent_ids:
            summary["reason"] = "no role-bound bouncy-wall agents"
            return
        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if up_axis is None or corrected.get("applied") is not True:
            summary["reason"] = "missing up axis or corrected static trajectories"
            return
        try:
            statics = {
                str(item.get("object_id")): item
                for item in (payload.get("static_fixture_flush_refinement") or {}).get("objects", [])
                if isinstance(item, dict)
                and item.get("refined_pose_4x4") is not None
                and item.get("refined_mesh_path")
            }

            def wall_plane(object_id: str):
                item = statics.get(object_id)
                if item is None:
                    return None
                pose = np.asarray(item["refined_pose_4x4"], dtype=np.float64).reshape(4, 4)
                vertices, _faces = self._load_mesh_geometry(Path(item["refined_mesh_path"]))
                thin_axis = int(np.argmin(vertices.max(axis=0) - vertices.min(axis=0)))
                wall_normal = pose[:3, :3][:, thin_axis]
                wall_normal = wall_normal / max(float(np.linalg.norm(wall_normal)), 1e-12)
                if abs(float(wall_normal @ up_axis)) >= PHYSION_PP_BW_TRAJ_WALL_UP_DOT_MAX:
                    return None
                center = (vertices @ pose[:3, :3].T + pose[:3, 3]).mean(axis=0)
                plane_normal = np.cross(
                    wall_normal - float(wall_normal @ up_axis) * up_axis,
                    up_axis,
                )
                plane_normal = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-12)
                return center, plane_normal

            walls = [object_id for object_id in sorted(statics) if wall_plane(object_id) is not None]
            if not walls:
                summary["reason"] = "no flush-refined wall plane"
                return
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)
            target_by_id = {str(target.object_id): target for target in object_plan.target_objects}
            centers_by_object: Dict[str, Dict[int, np.ndarray]] = {}
            masks_by_object: Dict[str, Dict[int, np.ndarray]] = {}
            segment_by_object: Dict[str, str] = {}
            for object_id in sorted(agent_ids):
                target = target_by_id.get(object_id)
                if target is None:
                    summary["objects"].append(
                        {"object_id": object_id, "applied": False, "reason": "missing object-plan target"}
                    )
                    continue
                source_track = str(target.source_track_id or "")
                segment = "seg1" if source_track.startswith("seg1") else "seg2"
                plane = wall_plane(walls[0] if segment == "seg1" else walls[-1])
                if plane is None:
                    summary["objects"].append(
                        {"object_id": object_id, "segment": segment, "applied": False, "reason": "missing segment wall plane"}
                    )
                    continue
                plane_center, plane_normal = plane
                centers: Dict[int, np.ndarray] = {}
                masks: Dict[int, np.ndarray] = {}
                for frame_index, record in (mask_records.get(object_id) or {}).items():
                    mask = self._resize_mask_to_shape(
                        mask_arrays.get(str(record.get("mask_key") or "")), image_shape
                    )
                    if mask is None or int(mask.sum()) < PHYSION_PP_AGENT_SPHERE_MIN_MASK_PX:
                        continue
                    ys, xs = np.nonzero(mask)
                    K = np.asarray(self._intrinsic_for_frame(intrinsics, int(frame_index)), dtype=np.float64)
                    ray = np.linalg.inv(K) @ np.array([float(xs.mean()), float(ys.mean()), 1.0])
                    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
                    denominator = float(ray @ plane_normal)
                    if abs(denominator) < PHYSION_PP_BW_TRAJ_MIN_RAY_PLANE_DOT:
                        continue
                    depth = float((plane_center @ plane_normal) / denominator)
                    if depth <= 0.0:
                        continue
                    centers[int(frame_index)] = ray * depth
                    masks[int(frame_index)] = mask
                if len(centers) < PHYSION_PP_BW_TRAJ_MIN_MATCHED:
                    summary["objects"].append(
                        {
                            "object_id": object_id,
                            "segment": segment,
                            "applied": False,
                            "reason": f"too few solvable mask-plane frames ({len(centers)})",
                        }
                    )
                    continue
                centers_by_object[object_id] = centers
                masks_by_object[object_id] = masks
                segment_by_object[object_id] = segment
            if not centers_by_object:
                summary["reason"] = "no bouncy-wall agent has a solvable FP-free trajectory"
                return

            fit_frames_by_object: Dict[str, list[int]] = {}
            for object_id, centers in centers_by_object.items():
                frames = sorted(centers)
                if len(frames) > PHYSION_PP_AGENT_SPHERE_FIT_MAX_FRAMES:
                    indices = np.linspace(
                        0, len(frames) - 1, PHYSION_PP_AGENT_SPHERE_FIT_MAX_FRAMES, dtype=int
                    )
                    frames = [frames[index] for index in sorted(set(indices.tolist()))]
                fit_frames_by_object[object_id] = frames

            seed_centers = []
            seed_masks = []
            seed_intrinsics = []
            for object_id, frames in fit_frames_by_object.items():
                for frame_index in frames:
                    seed_centers.append(centers_by_object[object_id][frame_index])
                    seed_masks.append(masks_by_object[object_id][frame_index])
                    seed_intrinsics.append(self._intrinsic_for_frame(intrinsics, frame_index))
            seed, pixel_floor, seed_summary = _sphere_radius_seed(
                centers=seed_centers,
                masks=seed_masks,
                intrinsics=seed_intrinsics,
            )
            camera_cap = 0.8 * min(float(np.linalg.norm(center)) for center in seed_centers)

            def objective(radius: float) -> float:
                segment_values: Dict[str, list[float]] = {"seg1": [], "seg2": []}
                for object_id, frames in fit_frames_by_object.items():
                    for frame_index in frames:
                        rendered = _analytic_sphere_mask(
                            center_camera=centers_by_object[object_id][frame_index],
                            radius=radius,
                            intrinsic=self._intrinsic_for_frame(intrinsics, frame_index),
                            image_shape=image_shape,
                        )
                        value = _bool_mask_iou(rendered, masks_by_object[object_id][frame_index])
                        if value is not None:
                            segment_values[segment_by_object[object_id]].append(float(value))
                means = [float(np.mean(values)) for values in segment_values.values() if values]
                return float(np.mean(means)) if means else float("-inf")

            radius, fit_iou, radius_search = _adaptive_sphere_radius_search(
                seed_radius=seed,
                pixel_floor=pixel_floor,
                camera_cap=camera_cap,
                objective=objective,
            )
            mesh_path = self._adopt_physion_pp_agent_sphere(
                payload=payload,
                question_dir=question_dir,
                object_plan=object_plan,
                object_ids=sorted(centers_by_object),
                radius=radius,
                centers_by_object_frame=centers_by_object,
                source=summary["source"],
                output_subdir="bouncy_wall_sphere_refinement",
            )
            all_ious = []
            for object_id, centers in centers_by_object.items():
                values = []
                for frame_index, center in centers.items():
                    value = _bool_mask_iou(
                        _analytic_sphere_mask(
                            center_camera=center,
                            radius=radius,
                            intrinsic=self._intrinsic_for_frame(intrinsics, frame_index),
                            image_shape=image_shape,
                        ),
                        masks_by_object[object_id][frame_index],
                    )
                    if value is not None:
                        values.append(float(value))
                all_ious.extend(values)
                summary["objects"].append(
                    {
                        "object_id": object_id,
                        "segment": segment_by_object[object_id],
                        "applied": True,
                        "solved_frames": len(centers),
                        "all_frame_iou": round(float(np.mean(values)), 6) if values else None,
                    }
                )
            summary.update(
                {
                    "applied": True,
                    "agent_object_ids": sorted(centers_by_object),
                    "radius_shared_across_segments": True,
                    "radius_m": round(float(radius), 6),
                    "radius_seed": seed_summary,
                    "radius_search": radius_search,
                    "fit_iou": round(float(fit_iou), 6),
                    "all_frame_iou": round(float(np.mean(all_ious)), 6) if all_ious else None,
                    "sphere_mesh_path": mesh_path,
                    "position_source": "mask_centroid_ray_intersection_with_segment_wall_vertical_plane",
                    "orientation_source": "identity_sphere",
                }
            )
            _log_tool(
                self.tool_name,
                f"bouncy_wall sphere agents={sorted(centers_by_object)} radius={radius:.4f} "
                f"iou={summary['all_frame_iou']}",
            )
        except Exception as exc:
            summary["reason"] = f"bouncy-wall sphere refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(self.tool_name, f"bouncy-wall sphere error={_short_text(str(exc), 300)}")

    def _apply_physion_pp_bouncy_wall_native_trajectory_refinement(
        self,
        *,
        payload: Dict[str, Any],
        question_dir: Path,
        object_plan: ObjectPlan,
        excluded_object_ids: set[str],
    ) -> None:
        """Physion++ bouncy_wall dynamic-object trajectory (mask-only position + orientation
        policy, zero depth for position).

        Position: each dynamic object's ground path is perpendicular to the wall along its
        centerline; lifting that line to a vertical plane P removes the single depth DOF the
        mask centroid leaves free, so each frame's centroid ray meets P at a unique 3D center
        (FoundationPose depth on the small object is unreliable and its track drifts).
        Orientation: use the FoundationPose orientation (smooth, physical). The FP raw
        trajectory (its drifting depth/position) is NEVER used -- only its rotation. The policy
        is applied on the seg1 track and mirrored onto its seg2 sibling (same physical object via
        mesh_reuse_source_object_id) so both segments are reconstructed the same way; each segment
        uses its own (possibly slightly shifted) wall plane and seg2 is not re-registered. Before
        rewriting the trajectory, seg1 fits ONE uniform mesh scale over a frame subsample with
        the prior-plane centers and FP rotations fixed. A boundary winner expands the coarse
        range; the fitted mesh is then reused by seg2 exactly. See the constants block."""
        scenario = self._object_plan_scenario(object_plan).lower()
        if scenario not in PHYSION_PP_BOUNCE_WALL_SCENARIOS:
            return
        summary: Dict[str, Any] = {
            "applied": False,
            "source": "physion_pp_bouncy_wall_mask_trajectory_refinement",
            "objects": [],
        }
        payload["physion_pp_agent_trajectory_refinement"] = summary

        up_axis = self._unit_up_direction(payload.get("gravity_direction_camera"))
        if up_axis is None:
            summary["reason"] = "missing up axis"
            return
        corrected = (payload.get("trajectory_correction") or {}).get("corrected_trajectories") or {}
        if corrected.get("applied") is not True:
            summary["reason"] = "corrected trajectories not applied"
            return
        objects = {str(i.get("object_id")): i for i in corrected.get("objects", []) if isinstance(i, dict)}
        support = payload.get("support_plane_position_correction") or {}
        support_by_id = {
            str(item.get("object_id")): item
            for item in support.get("objects", [])
            if isinstance(item, dict)
        }
        dyns = [
            target
            for target in object_plan.target_objects
            if "dynamic" in str(getattr(target, "role", "") or "").lower()
            and str(target.object_id) not in excluded_object_ids
        ]
        if not dyns:
            summary["reason"] = "no dynamic objects"
            return

        try:
            import trimesh

            statics = {
                str(o.get("object_id")): o
                for o in (payload.get("static_fixture_flush_refinement") or {}).get("objects", [])
                if isinstance(o, dict) and o.get("refined_pose_4x4") is not None and o.get("refined_mesh_path")
            }

            def _wall_plane(wall_id: str):
                o = statics.get(wall_id)
                if o is None:
                    return None
                pose = np.asarray(o["refined_pose_4x4"], dtype=np.float64).reshape(4, 4)
                rot, trans = pose[:3, :3], pose[:3, 3]
                verts, _ = self._load_mesh_geometry(Path(o["refined_mesh_path"]))
                thin = int(np.argmin(verts.max(0) - verts.min(0)))
                normal = rot[:, thin]
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                if abs(float(normal @ up_axis)) >= PHYSION_PP_BW_TRAJ_WALL_UP_DOT_MAX:
                    return None  # thin axis is vertical (a mat), not the wall panel
                center = (verts @ rot.T + trans).mean(0)
                plane_normal = np.cross(normal - (normal @ up_axis) * up_axis, up_axis)
                plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
                return center, plane_normal

            walls = [oid for oid in sorted(statics) if _wall_plane(oid) is not None]
            if not walls:
                summary["reason"] = "no wall plane"
                return

            mesh_paths = self._foundationpose_mesh_paths(question_dir)
            mask_records = self._sam3_video_mask_records_by_object_frame(question_dir)
            mask_arrays = self._sam3_video_mask_arrays(mask_records)
            intrinsics, _metric_depth, image_shape = self._video_metric_intrinsics_depth_and_shape(question_dir)

            def _decimate(verts: np.ndarray, faces: np.ndarray):
                if len(faces) <= PHYSION_PP_BW_TRAJ_DECIMATE_FACES:
                    return verts, faces
                try:
                    simplified = trimesh.Trimesh(vertices=verts, faces=faces, process=False).simplify_quadric_decimation(
                        PHYSION_PP_BW_TRAJ_DECIMATE_FACES
                    )
                    if len(simplified.faces) >= 8:
                        return np.asarray(simplified.vertices, dtype=np.float64), np.asarray(simplified.faces)
                except Exception:
                    pass
                return verts, faces

            def _geodesic_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
                cos_t = (float(np.trace(rot_a.T @ rot_b)) - 1.0) / 2.0
                return float(np.degrees(np.arccos(float(np.clip(cos_t, -1.0, 1.0)))))

            def _build_context(target):
                object_id = str(target.object_id)
                track = str(getattr(target, "source_track_id", "") or "")
                record: Dict[str, Any] = {"object_id": object_id, "track": track}
                summary["objects"].append(record)
                if object_id not in objects or object_id not in mesh_paths:
                    record["reason"] = "missing from corrected trajectories or mesh"
                    return None
                segment = "seg1" if track.startswith("seg1") else "seg2"
                # each segment uses its own wall plane (the wall pose can shift between segments)
                wall_plane = _wall_plane(walls[0] if segment == "seg1" else walls[-1])
                if wall_plane is None:
                    record["reason"] = "no wall plane for segment"
                    return None
                plane_center, plane_normal = wall_plane
                item = objects[object_id]
                verts_full, faces_full = self._load_mesh_geometry(Path(mesh_paths[object_id]))
                mesh_centroid = verts_full.mean(0)
                verts_search, faces_search = _decimate(verts_full, faces_full)

                fp_rotation: Dict[int, np.ndarray] = {}
                for pose_entry in item.get("poses", []):
                    frame_index = self._pose_frame_index(pose_entry)
                    if frame_index is None or pose_entry.get("corrected_pose_4x4") is None:
                        continue
                    fp_rotation[frame_index] = np.asarray(
                        pose_entry["corrected_pose_4x4"], dtype=np.float64
                    ).reshape(4, 4)[:3, :3]

                centers: Dict[int, np.ndarray] = {}
                target_masks: Dict[int, np.ndarray] = {}
                for frame_index, mrec in (mask_records.get(object_id) or {}).items():
                    frame_index = int(frame_index)
                    if frame_index not in fp_rotation:
                        continue
                    mask = self._resize_mask_to_shape(mask_arrays.get(str(mrec.get("mask_key") or "")), image_shape)
                    if mask is None or not mask.any():
                        continue
                    ys, xs = np.nonzero(mask)
                    if xs.size < 3:
                        continue
                    intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                    ray = np.linalg.inv(intrinsic) @ np.array([float(xs.mean()), float(ys.mean()), 1.0])
                    ray = ray / np.linalg.norm(ray)
                    denom = float(ray @ plane_normal)
                    if abs(denom) < PHYSION_PP_BW_TRAJ_MIN_RAY_PLANE_DOT:
                        continue
                    depth_scale = float((plane_center @ plane_normal) / denom)
                    if depth_scale <= 0:
                        continue
                    centers[frame_index] = ray * depth_scale
                    target_masks[frame_index] = mask
                if len(centers) < PHYSION_PP_BW_TRAJ_MIN_MATCHED:
                    record["reason"] = f"too few solvable frames ({len(centers)})"
                    return None
                return {
                    "object_id": object_id, "track": track, "segment": segment, "record": record,
                    "item": item, "verts_full": verts_full, "faces_full": faces_full,
                    "verts_search": verts_search, "faces_search": faces_search,
                    "mesh_centroid": mesh_centroid, "centers": centers,
                    "target_masks": target_masks, "frames": sorted(centers), "fp_rotation": fp_rotation,
                    "mesh_path": str(mesh_paths[object_id]),
                }

            def _iou_at(ctx, rotation, frame_index, verts, faces):
                # position is ALWAYS the mask-only plane intersection; FP contributes rotation only
                intrinsic = np.asarray(self._intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                translation = ctx["centers"][frame_index] - rotation @ ctx["mesh_centroid"]
                rendered, _ = render_mesh_depth(
                    vertices_camera=verts @ rotation.T + translation,
                    faces=faces, intrinsic=intrinsic, image_shape=image_shape,
                )
                return self._mask_iou(rendered, ctx["target_masks"][frame_index])

            def _uniform_frame_sample(frames: list[int], limit: int) -> list[int]:
                if len(frames) <= limit:
                    return list(frames)
                indices = np.linspace(0, len(frames) - 1, limit, dtype=int)
                return [frames[index] for index in sorted(set(indices.tolist()))]

            def _scaled_about_centroid(ctx, verts: np.ndarray, scale: float) -> np.ndarray:
                centroid = ctx["mesh_centroid"]
                return centroid + (verts - centroid) * float(scale)

            def _scale_mean_iou(
                ctx,
                scale: float,
                frames: list[int],
                *,
                full_mesh: bool = False,
            ) -> float:
                source_verts = ctx["verts_full"] if full_mesh else ctx["verts_search"]
                faces = ctx["faces_full"] if full_mesh else ctx["faces_search"]
                vertices = _scaled_about_centroid(ctx, source_verts, scale)
                values = []
                for frame_index in frames:
                    rotation = ctx["fp_rotation"][frame_index]
                    translation = (
                        ctx["centers"][frame_index]
                        - rotation @ ctx["mesh_centroid"]
                    )
                    rendered = render_mesh_mask_cuda(
                        vertices_camera=vertices @ rotation.T + translation,
                        faces=faces,
                        intrinsic=np.asarray(
                            self._intrinsic_for_frame(intrinsics, frame_index),
                            dtype=np.float64,
                        ),
                        image_shape=image_shape,
                    )
                    values.append(
                        self._mask_iou(rendered, ctx["target_masks"][frame_index])
                    )
                values = [float(value) for value in values if value is not None]
                return float(np.mean(values)) if values else float("-inf")

            def _fit_seg1_scale(ctx) -> Dict[str, Any]:
                fit_frames = _uniform_frame_sample(
                    ctx["frames"], PHYSION_PP_BW_TRAJ_SCALE_MAX_FRAMES
                )
                lo, hi, steps = PHYSION_PP_BW_TRAJ_SCALE_COARSE
                step = (hi - lo) / max(int(steps) - 1, 1)
                search_started = time.monotonic()
                reset_cuda_mask_rasterizer_stats()
                coarse = [
                    (float(scale), _scale_mean_iou(ctx, float(scale), fit_frames))
                    for scale in np.linspace(lo, hi, int(steps))
                ]
                finite = [entry for entry in coarse if np.isfinite(entry[1])]
                if not finite:
                    return {
                        "applied": False,
                        "reason": "global scale search produced no valid IoU",
                        "fit_frames": len(fit_frames),
                    }
                initial_winner, _initial_iou = max(finite, key=lambda entry: entry[1])
                expanded_scales, expanded_direction = _boundary_extension_values(
                    winner=initial_winner,
                    initial_start=lo,
                    initial_stop=hi,
                    step=step,
                    expanded_start=PHYSION_PP_BW_TRAJ_SCALE_EXPANDED[0],
                    expanded_stop=PHYSION_PP_BW_TRAJ_SCALE_EXPANDED[1],
                    boundary_steps=PHYSION_PP_BW_TRAJ_SCALE_BOUNDARY_STEPS,
                )
                coarse.extend(
                    (
                        float(scale),
                        _scale_mean_iou(ctx, float(scale), fit_frames),
                    )
                    for scale in expanded_scales
                )
                finite = [entry for entry in coarse if np.isfinite(entry[1])]
                coarse_winner, _coarse_iou = max(finite, key=lambda entry: entry[1])
                fine_lo, fine_hi, fine_steps = PHYSION_PP_BW_TRAJ_SCALE_FINE
                hard_lo, hard_hi = PHYSION_PP_BW_TRAJ_SCALE_EXPANDED
                fine_scales = np.linspace(
                    max(hard_lo, coarse_winner * fine_lo),
                    min(hard_hi, coarse_winner * fine_hi),
                    int(fine_steps),
                )
                fine = [
                    (float(scale), _scale_mean_iou(ctx, float(scale), fit_frames))
                    for scale in fine_scales
                ]
                finite_fine = [entry for entry in fine if np.isfinite(entry[1])]
                final_scale, fit_iou = max(
                    [*finite, *finite_fine], key=lambda entry: entry[1]
                )
                fit_iou_before = _scale_mean_iou(ctx, 1.0, fit_frames)
                all_iou_before = _scale_mean_iou(
                    ctx, 1.0, ctx["frames"], full_mesh=True
                )
                all_iou_after = _scale_mean_iou(
                    ctx, final_scale, ctx["frames"], full_mesh=True
                )

                source_mesh_path = Path(ctx["mesh_path"])
                scaled_full = _scaled_about_centroid(
                    ctx, ctx["verts_full"], final_scale
                )
                scaled_search = _scaled_about_centroid(
                    ctx, ctx["verts_search"], final_scale
                )
                mesh_path = (
                    self.artifacts.tool_dir(question_dir, self.tool_name)
                    / "bouncy_wall_trajectory_refinement"
                    / f"{ctx['object_id']}_prior_scale_local.glb"
                )
                self._export_mesh_with_source_colors(
                    vertices=scaled_full,
                    faces=ctx["faces_full"],
                    source_mesh_path=source_mesh_path,
                    output_path=mesh_path,
                )
                ctx["verts_full"] = scaled_full
                ctx["verts_search"] = scaled_search
                ctx["mesh_path"] = str(mesh_path)
                saturated = (
                    final_scale <= hard_lo + 1e-9
                    or final_scale >= hard_hi - 1e-9
                )
                return {
                    "applied": True,
                    "policy": (
                        "one uniform seg1 mesh scale maximizing multi-frame SAM3 mask IoU "
                        "at bouncy-wall prior-plane centers with FP rotations fixed; seg2 "
                        "inherits the baked mesh"
                    ),
                    "initial_range": [lo, hi],
                    "boundary_band_steps": PHYSION_PP_BW_TRAJ_SCALE_BOUNDARY_STEPS,
                    "initial_winner": round(float(initial_winner), 6),
                    "expanded": bool(expanded_scales),
                    "expanded_direction": expanded_direction,
                    "evaluated_coarse_range": [
                        round(min([lo, *expanded_scales]), 6),
                        round(max([hi, *expanded_scales]), 6),
                    ],
                    "coarse_winner": round(float(coarse_winner), 6),
                    "final_scale": round(float(final_scale), 6),
                    "search_saturated": bool(saturated),
                    "fit_frames": len(fit_frames),
                    "fit_iou_before": round(float(fit_iou_before), 4),
                    "fit_iou_after": round(float(fit_iou), 4),
                    "all_frame_iou_before": round(float(all_iou_before), 4),
                    "all_frame_iou_after": round(float(all_iou_after), 4),
                    "mesh_path": str(mesh_path),
                    "rasterizer": cuda_mask_rasterizer_stats(),
                    "elapsed_sec": round(time.monotonic() - search_started, 2),
                }

            def _adopt_fitted_mesh(
                *,
                object_id: str,
                mesh_path: str,
                source_object_id: str,
                scale: float,
            ) -> None:
                previous_mesh = mesh_paths.get(object_id)
                mesh_paths[object_id] = mesh_path
                item = objects.get(object_id)
                if isinstance(item, dict):
                    item["pre_bouncy_wall_scale_mesh_path"] = (
                        item.get("mesh_path") or previous_mesh
                    )
                    item["mesh_path"] = mesh_path
                    item["mesh_source"] = "physion_pp_bouncy_wall_prior_scale_search"
                    item["bouncy_wall_scale"] = float(scale)
                    item["bouncy_wall_scale_source_object_id"] = source_object_id
                support_item = support_by_id.get(object_id)
                if isinstance(support_item, dict):
                    support_item["pre_bouncy_wall_scale_mesh_path"] = (
                        support_item.get("mesh_path") or previous_mesh
                    )
                    support_item["mesh_path"] = mesh_path
                    support_item["mesh_source"] = (
                        "physion_pp_bouncy_wall_prior_scale_search"
                    )
                    support_item["bouncy_wall_scale"] = float(scale)
                    support_item["bouncy_wall_scale_source_object_id"] = source_object_id

            def _fp_orientation_quality(ctx):
                ious: list[float] = []
                jitters: list[float] = []
                previous = None
                for frame_index in ctx["frames"]:
                    value = _iou_at(ctx, ctx["fp_rotation"][frame_index], frame_index, ctx["verts_search"], ctx["faces_search"])
                    if value is not None:
                        ious.append(float(value))
                    if previous is not None:
                        jitters.append(_geodesic_deg(previous, ctx["fp_rotation"][frame_index]))
                    previous = ctx["fp_rotation"][frame_index]
                fp_median = float(np.median(ious)) if ious else 0.0
                reg_frame_iou = float(np.percentile(ious, 90)) if ious else 0.0
                jitter_median = float(np.median(jitters)) if jitters else 0.0
                return fp_median, reg_frame_iou, jitter_median

            def _fp_orientation(ctx):
                return {frame_index: ctx["fp_rotation"][frame_index] for frame_index in ctx["frames"]}

            def _mean_iou_full(ctx, chosen):
                ious: list[float] = []
                for frame_index in ctx["frames"]:
                    value = _iou_at(ctx, chosen[frame_index], frame_index, ctx["verts_full"], ctx["faces_full"])
                    if value is not None:
                        ious.append(float(value))
                return float(np.mean(ious)) if ious else 0.0

            def _write_back(ctx, chosen, method, metrics):
                record = ctx["record"]
                centers = ctx["centers"]
                mesh_centroid = ctx["mesh_centroid"]
                record.update({"segment": ctx["segment"], "orientation_method": method})
                record.update(metrics)
                for pose_entry in ctx["item"].get("poses", []):
                    frame_index = self._pose_frame_index(pose_entry)
                    if frame_index is None:
                        continue
                    if frame_index not in centers:
                        pose_entry["bouncy_wall_refined"] = False
                        continue
                    rotation = chosen[frame_index]
                    translation = centers[frame_index] - rotation @ mesh_centroid
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = translation
                    pose_entry["corrected_pose_4x4"] = pose.tolist()
                    pose_entry["corrected_translation_camera"] = translation.tolist()
                    pose_entry["bouncy_wall_refined"] = True
                record["applied"] = True
                _log_tool(
                    self.tool_name,
                    f"bouncy_wall_trajectory {ctx['object_id']} {ctx['segment']} method={method} "
                    f"frames={len(ctx['frames'])} new_iou={metrics.get('new_iou_mean')} "
                    f"fp_orient_iou={metrics.get('fp_orientation_iou_mean')}",
                )

            def _fp_track_solution(ctx):
                # Position always comes from the mask-plane intersection; orientation always
                # comes from the FoundationPose track.
                fp_median, reg_frame_iou, jitter_median = _fp_orientation_quality(ctx)
                chosen = _fp_orientation(ctx)
                fp_mean = _mean_iou_full(ctx, chosen)
                metrics = {
                    "fp_track_iou_median": round(fp_median, 4),
                    "registration_frame_iou": round(reg_frame_iou, 4),
                    "fp_track_jitter_median_deg": round(jitter_median, 1),
                    "solved_frames": len(ctx["frames"]),
                    "new_iou_mean": round(fp_mean, 4),
                    "fp_orientation_iou_mean": round(fp_mean, 4),
                }
                return chosen, metrics

            def _apply_forced_fp_track(ctx):
                # seg2 mirrors the seg1 FP-track orientation policy.
                fp_median, reg_frame_iou, jitter_median = _fp_orientation_quality(ctx)
                chosen = _fp_orientation(ctx)
                fp_mean = _mean_iou_full(ctx, chosen)
                metrics = {
                    "fp_track_iou_median": round(fp_median, 4),
                    "registration_frame_iou": round(reg_frame_iou, 4),
                    "fp_track_jitter_median_deg": round(jitter_median, 1),
                    "solved_frames": len(ctx["frames"]),
                    "new_iou_mean": round(fp_mean, 4),
                    "fp_orientation_iou_mean": round(fp_mean, 4),
                    "forced_by_seg1_method": "fp_track",
                }
                _write_back(ctx, chosen, "fp_track", metrics)

            # Pass 1 applies FP-track orientation to each seg1 object. Pass 2 applies the
            # same policy to its seg2 sibling and adopts the seg1 prior-fitted mesh so both
            # segments are reconstructed with exactly the same metric size.
            method_by_source: Dict[str, str] = {}
            scale_by_source: Dict[str, Dict[str, Any]] = {}
            seg1_dyns = [t for t in dyns if str(getattr(t, "source_track_id", "") or "").startswith("seg1")]
            seg2_dyns = [t for t in dyns if not str(getattr(t, "source_track_id", "") or "").startswith("seg1")]
            for target in seg1_dyns:
                ctx = _build_context(target)
                if ctx is None:
                    continue
                scale_search = _fit_seg1_scale(ctx)
                ctx["record"]["scale_search"] = scale_search
                if scale_search.get("applied") is True:
                    scale_info = {
                        "scale": float(scale_search["final_scale"]),
                        "mesh_path": str(scale_search["mesh_path"]),
                    }
                    scale_by_source[ctx["object_id"]] = scale_info
                    _adopt_fitted_mesh(
                        object_id=ctx["object_id"],
                        mesh_path=scale_info["mesh_path"],
                        source_object_id=ctx["object_id"],
                        scale=scale_info["scale"],
                    )
                chosen, metrics = _fp_track_solution(ctx)
                method_by_source[ctx["object_id"]] = "fp_track"
                _write_back(ctx, chosen, "fp_track", metrics)
            for target in seg2_dyns:
                source = str(getattr(target, "mesh_reuse_source_object_id", "") or "")
                scale_info = scale_by_source.get(source)
                if scale_info is not None:
                    _adopt_fitted_mesh(
                        object_id=str(target.object_id),
                        mesh_path=scale_info["mesh_path"],
                        source_object_id=source,
                        scale=scale_info["scale"],
                    )
                ctx = _build_context(target)
                if ctx is None:
                    continue
                if scale_info is not None:
                    ctx["record"]["scale_inherited_from_seg1"] = {
                        "source_object_id": source,
                        "scale": scale_info["scale"],
                        "mesh_path": scale_info["mesh_path"],
                    }
                forced = method_by_source.get(source)
                if forced is None:
                    chosen, metrics = _fp_track_solution(ctx)
                    method_by_source[ctx["object_id"]] = "fp_track"
                    _write_back(ctx, chosen, "fp_track", metrics)
                else:
                    _apply_forced_fp_track(ctx)
            summary["applied"] = any(r.get("applied") for r in summary["objects"])
        except Exception as exc:
            summary["reason"] = f"bouncy_wall trajectory refinement failed: {_short_text(str(exc), 300)}"
            _log_tool(self.tool_name, f"bouncy_wall_trajectory error={_short_text(str(exc), 300)}")

    def _flush_prepare_member(
        self,
        *,
        object_id: str,
        trajectory_item: Dict[str, Any],
        support_item: Optional[Dict[str, Any]],
        mesh_path: Optional[str],
        mask_records: Dict[int, Dict[str, Any]],
        mask_arrays: Dict[str, np.ndarray],
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
        up_axis: np.ndarray,
        question_dir: Path,
        flatten_base: bool = False,
    ) -> Dict[str, Any]:
        """Per-object flush setup, factored out so the single-object and joint-group
        refiners share identical geometry/pose/scorer construction. Returns
        {"ok": False, "entry": <skip entry>} on any setup failure, else a member context
        with the mesh, base-aligned+flush-pinned base pose, OBB plane axes, and the
        per-object score / yaw / scale closures. Mirrors the setup block of
        _refine_static_fixture_object exactly."""
        import trimesh

        entry: Dict[str, Any] = {"object_id": object_id, "status": "skipped"}
        if not mesh_path:
            entry["reason"] = "missing foundationpose mesh_path"
            return {"ok": False, "entry": entry}
        poses = [
            pose
            for pose in trajectory_item.get("poses", [])
            if isinstance(pose, dict)
            and pose.get("corrected_pose_4x4") is not None
            and self._pose_frame_index(pose) is not None
        ]
        if not poses:
            entry["reason"] = "no valid corrected poses"
            return {"ok": False, "entry": entry}
        pose_frames = {self._pose_frame_index(pose) for pose in poses}
        eval_frames = self._sample_overlap_frames(
            sorted(frame for frame in mask_records if frame in pose_frames),
            max_frames=STATIC_FIXTURE_REFINE_MAX_EVAL_FRAMES,
        )
        if not eval_frames:
            entry["reason"] = "no frames with both a corrected pose and a SAM3 mask"
            return {"ok": False, "entry": entry}
        try:
            vertices, faces = self._load_mesh_geometry(Path(mesh_path))
        except Exception as exc:
            entry["reason"] = f"failed to load mesh geometry: {exc}"
            return {"ok": False, "entry": entry}
        base_pose = np.asarray(poses[0]["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)

        base_alignment: Optional[Dict[str, Any]] = None
        if flatten_base:
            base_alignment = {"applied": False}
            axis_local = base_pose[:3, :3].T @ up_axis
            local_heights = vertices @ axis_local
            band = local_heights <= np.percentile(local_heights, PHYSION_PP_STATIC_BASE_BAND_PERCENTILE)
            band_vertices = vertices[band]
            if len(band_vertices) >= 16:
                keep = np.ones(len(band_vertices), dtype=bool)
                normal_local = axis_local
                for _ in range(3):
                    center = band_vertices[keep].mean(axis=0)
                    _, _, vt = np.linalg.svd(band_vertices[keep] - center, full_matrices=False)
                    normal_local = vt[-1]
                    if float(normal_local @ axis_local) < 0.0:
                        normal_local = -normal_local
                    distance = np.abs((band_vertices - center) @ normal_local)
                    keep = distance < np.percentile(distance[keep], 80)
                normal_camera = base_pose[:3, :3] @ normal_local
                cos_angle = float(np.clip(normal_camera @ up_axis, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_angle)))
                base_alignment.update({"base_tilt_deg": angle_deg, "band_vertex_count": int(band.sum())})
                if 1e-3 < angle_deg <= PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG:
                    rotation_axis = np.cross(normal_camera, up_axis)
                    axis_norm = float(np.linalg.norm(rotation_axis))
                    if axis_norm > 1e-9:
                        rotation_axis = rotation_axis / axis_norm
                        align_rotation = _rotation_about_axis(rotation_axis, float(np.arccos(cos_angle)))
                        base_centroid_local = band_vertices.mean(axis=0)
                        base_centroid_camera = base_pose[:3, :3] @ base_centroid_local + base_pose[:3, 3]
                        aligned = base_pose.copy()
                        aligned[:3, :3] = align_rotation @ base_pose[:3, :3]
                        aligned[:3, 3] = base_centroid_camera - aligned[:3, :3] @ base_centroid_local
                        base_pose = aligned
                        base_alignment["applied"] = True
                elif angle_deg > PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG:
                    base_alignment["reason"] = "base tilt exceeds alignment cap"
                    base_alignment["cap_deg"] = PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG
            else:
                base_alignment["reason"] = "too few base-band vertices for plane fit"
            pinned_ground = self._support_item_ground_height(support_item)
            if pinned_ground is not None:
                axis_local_pin = base_pose[:3, :3].T @ up_axis
                bottom_pin = float(
                    np.percentile(vertices @ axis_local_pin, PHYSION_PP_STATIC_BASE_PERCENTILE)
                )
                base_cam_pin = bottom_pin + float(base_pose[:3, 3] @ up_axis)
                base_pose = base_pose.copy()
                base_pose[:3, 3] = base_pose[:3, 3] - up_axis * (base_cam_pin - float(pinned_ground))
                base_alignment["pre_search_flush_pin"] = True

        to_origin, _extents = trimesh.bounds.oriented_bounds(vertices)
        obb_transform = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
        obb_axes = obb_transform[:3, :3]
        obb_center = obb_transform[:3, 3]
        support_axis_local = base_pose[:3, :3].T @ up_axis
        axis_alignment = np.abs(obb_axes.T @ support_axis_local)
        support_column = int(np.argmax(axis_alignment))
        plane_columns = [column for column in range(3) if column != support_column]
        plane_axis_u = obb_axes[:, plane_columns[0]]
        plane_axis_v = obb_axes[:, plane_columns[1]]
        support_axis_box = obb_axes[:, support_column]

        def _scaled_vertices(scale_u: float, scale_v: float) -> np.ndarray:
            scaling = (
                scale_u * np.outer(plane_axis_u, plane_axis_u)
                + scale_v * np.outer(plane_axis_v, plane_axis_v)
                + np.outer(support_axis_box, support_axis_box)
            )
            return (vertices - obb_center) @ scaling.T + obb_center

        target_masks: Dict[int, np.ndarray] = {}
        for frame_index in eval_frames:
            mask_record = mask_records.get(frame_index) or {}
            target_mask = self._resize_mask_to_shape(
                mask_arrays.get(str(mask_record.get("mask_key") or "")), image_shape
            )
            if target_mask is not None:
                target_masks[frame_index] = target_mask

        eval_frame_order = list(target_masks)
        eval_intrinsics = [self._intrinsic_for_frame(intrinsics, f) for f in eval_frame_order]
        constant_intrinsic = (
            eval_intrinsics[0]
            if eval_intrinsics
            and all(np.array_equal(eval_intrinsics[0], K) for K in eval_intrinsics[1:])
            else None
        )
        score_state: Dict[str, Any] = {
            "vertices": vertices,
            "faces": faces,
            "obb_center": obb_center,
            "outer_u": np.outer(plane_axis_u, plane_axis_u),
            "outer_v": np.outer(plane_axis_v, plane_axis_v),
            "outer_support": np.outer(support_axis_box, support_axis_box),
            "image_shape": image_shape,
            "constant_intrinsic": constant_intrinsic,
            "intrinsic_by_frame": dict(zip(eval_frame_order, eval_intrinsics)),
            "target_masks": target_masks,
            "occluder_by_frame": self._object_mask_union_by_frame(
                question_dir, eval_frame_order, image_shape
            ),
        }

        def _score(pose: np.ndarray, scale_u: float, scale_v: float) -> float:
            return _flush_refine_score(score_state, pose, scale_u, scale_v)

        u_camera, v_camera = _plane_basis_for_up_axis(up_axis)

        def _yawed(pose: np.ndarray, angle_rad: float) -> np.ndarray:
            rotation = _rotation_about_axis(up_axis, angle_rad)
            centroid_camera = pose[:3, :3] @ obb_center + pose[:3, 3]
            out = pose.copy()
            out[:3, :3] = rotation @ pose[:3, :3]
            out[:3, 3] = centroid_camera - out[:3, :3] @ obb_center
            return out

        return {
            "ok": True,
            "entry": entry,
            "object_id": object_id,
            "trajectory_item": trajectory_item,
            "support_item": support_item,
            "mesh_path": mesh_path,
            "question_dir": question_dir,
            "flatten_base": flatten_base,
            "up_axis": up_axis,
            "vertices": vertices,
            "faces": faces,
            "obb_center": obb_center,
            "base_pose": base_pose,
            "base_alignment": base_alignment,
            "eval_frames": eval_frames,
            "score_state": score_state,
            "constant_intrinsic": constant_intrinsic,
            "u_camera": u_camera,
            "v_camera": v_camera,
            "start_score": _score(base_pose, 1.0, 1.0),
            "_scaled_vertices": _scaled_vertices,
            "_score": _score,
            "_yawed": _yawed,
        }

    def _flush_finalize_member(
        self,
        *,
        mem: Dict[str, Any],
        refined_pose: np.ndarray,
        scale_u: float,
        scale_v: float,
        best_score: float,
        flush_search_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per-object flush finalize (base-flatten + ground-flush, mesh export, support /
        trajectory writeback, entry) shared by the single-object and joint-group refiners.
        Mirrors the finalize block of _refine_static_fixture_object."""
        object_id = mem["object_id"]
        vertices = mem["vertices"]
        faces = mem["faces"]
        up_axis = mem["up_axis"]
        support_item = mem["support_item"]
        base_pose = mem["base_pose"]
        entry = mem["entry"]

        rescaled = abs(scale_u - 1.0) > 1e-3 or abs(scale_v - 1.0) > 1e-3
        final_vertices = mem["_scaled_vertices"](scale_u, scale_v)
        base_flattening: Optional[Dict[str, Any]] = None
        if mem["flatten_base"]:
            axis_local = refined_pose[:3, :3].T @ up_axis
            heights = final_vertices @ axis_local
            min_height = float(heights.min())
            bottom = float(np.percentile(heights, PHYSION_PP_STATIC_BASE_PERCENTILE))
            below = heights < bottom
            if bool(below.any()):
                final_vertices = final_vertices + np.outer(
                    np.clip(bottom - heights, 0.0, None), axis_local
                )
            ground_height = self._support_item_ground_height(support_item)
            refined_pose = refined_pose.copy()
            if ground_height is not None:
                base_height_camera = bottom + float(np.dot(refined_pose[:3, 3], up_axis))
                sink = base_height_camera - float(ground_height)
            else:
                sink = bottom - min_height
            refined_pose[:3, 3] = refined_pose[:3, 3] - up_axis * sink
            base_flattening = {
                "applied": True,
                "base_percentile": PHYSION_PP_STATIC_BASE_PERCENTILE,
                "base_height_local": bottom,
                "min_height_local": min_height,
                "shaved_vertex_count": int(below.sum()),
                "flush_sink_distance": float(sink),
                "flush_anchor": "support_plane_height" if ground_height is not None else "old_min_vertex",
            }
        translation_shift = refined_pose[:3, 3] - base_pose[:3, 3]
        refined_mesh_path: Optional[str] = None
        if rescaled or (base_flattening or {}).get("shaved_vertex_count"):
            mesh_dir = self.artifacts.tool_dir(mem["question_dir"], self.tool_name) / "static_fixture_meshes"
            refined_mesh_file = mesh_dir / f"{object_id}_flush_box_local.glb"
            self._export_mesh_with_source_colors(
                vertices=final_vertices,
                faces=faces,
                source_mesh_path=Path(mem["mesh_path"]),
                output_path=refined_mesh_file,
            )
            refined_mesh_path = str(refined_mesh_file)
            if isinstance(support_item, dict):
                support_item["source_mesh_path"] = support_item.get("mesh_path")
                support_item["mesh_path"] = refined_mesh_path
                support_item["mesh_source"] = "static_fixture_flush_refinement"

        for pose in mem["trajectory_item"].get("poses", []):
            if not isinstance(pose, dict) or pose.get("corrected_pose_4x4") is None:
                continue
            pose["corrected_pose_4x4"] = refined_pose.tolist()
            pose["corrected_translation_camera"] = refined_pose[:3, 3].tolist()
        mem["trajectory_item"]["static_fixture_flush_refinement_applied"] = True

        start_score = mem["start_score"]
        entry.update(
            {
                "status": "ok",
                "eval_frame_indices": [int(frame) for frame in mem["eval_frames"]],
                "start_mean_iou": None if not np.isfinite(start_score) else float(start_score),
                "refined_mean_iou": None if not np.isfinite(best_score) else float(best_score),
                "translation_shift_camera": translation_shift.tolist(),
                "translation_shift_norm": float(np.linalg.norm(translation_shift)),
                "scale_u": float(scale_u),
                "scale_v": float(scale_v),
                "refined_mesh_path": refined_mesh_path,
                "refined_pose_4x4": refined_pose.tolist(),
                "base_alignment": mem["base_alignment"],
                "base_flattening": base_flattening,
                "flush_search": flush_search_summary,
            }
        )
        return entry

    def _refine_static_fixture_group(self, setups: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Nested joint flush for one bouncy-wall seg1/seg2 static-fixture pair.

        Shared scale is the outer variable.  Every scale candidate independently
        re-optimizes each segment's two in-plane translations and yaw before the pair's
        summed IoU is compared.  This reaches scale+translation basins that alternating
        coordinate descent cannot cross while keeping the physical size shared exactly.
        """
        members: list[Dict[str, Any]] = []
        skipped_entries: list[Dict[str, Any]] = []
        for setup in setups:
            mem = self._flush_prepare_member(**setup)
            if mem.get("ok"):
                mem["_state_pose"] = mem["base_pose"].copy()
                mem["_best_score"] = float(mem["start_score"])
                mem["_eval"] = {"evaluated": 1, "skipped_noop": 0}
                mem["_search_started"] = time.monotonic()
                members.append(mem)
            else:
                skipped_entries.append(mem["entry"])
        if skipped_entries or len(members) != len(setups):
            return skipped_entries
        if len(members) != 2:
            raise ValueError(
                f"nested joint flush requires one complete pair, got {len(members)} members"
            )

        scale = {"scale_u": 1.0, "scale_v": 1.0}
        round_records: list[Dict[str, Any]] = []

        def _optimize_member_pose(
            mem: Dict[str, Any],
            *,
            reference_pose: np.ndarray,
            trial_scale: Dict[str, float],
            round_spec: Dict[str, Any],
        ) -> tuple[np.ndarray, float]:
            pose = reference_pose.copy()
            score = float(
                mem["_score"](pose, trial_scale["scale_u"], trial_scale["scale_v"])
            )
            mem["_eval"]["evaluated"] += 1
            for axis_camera in (mem["u_camera"], mem["v_camera"]):
                axis_origin = pose.copy()
                best_delta = 0.0
                best_axis_score = score
                for delta in round_spec["shift"]:
                    if float(delta) == 0.0:
                        mem["_eval"]["skipped_noop"] += 1
                        continue
                    candidate = axis_origin.copy()
                    candidate[:3, 3] = candidate[:3, 3] + axis_camera * float(delta)
                    candidate_score = float(
                        mem["_score"](
                            candidate,
                            trial_scale["scale_u"],
                            trial_scale["scale_v"],
                        )
                    )
                    mem["_eval"]["evaluated"] += 1
                    if candidate_score > best_axis_score:
                        best_axis_score = candidate_score
                        best_delta = float(delta)
                pose = axis_origin.copy()
                pose[:3, 3] = pose[:3, 3] + axis_camera * best_delta
                score = best_axis_score

            yaw_origin = pose.copy()
            best_angle = 0.0
            best_yaw_score = score
            for angle_deg in round_spec["yaw_deg"]:
                if float(angle_deg) == 0.0:
                    mem["_eval"]["skipped_noop"] += 1
                    continue
                candidate = mem["_yawed"](
                    yaw_origin, float(np.deg2rad(angle_deg))
                )
                candidate_score = float(
                    mem["_score"](
                        candidate,
                        trial_scale["scale_u"],
                        trial_scale["scale_v"],
                    )
                )
                mem["_eval"]["evaluated"] += 1
                if candidate_score > best_yaw_score:
                    best_yaw_score = candidate_score
                    best_angle = float(angle_deg)
            return (
                mem["_yawed"](yaw_origin, float(np.deg2rad(best_angle))),
                best_yaw_score,
            )

        for round_index, round_spec in enumerate(PHYSION_PP_STATIC_REFINE_ROUNDS):
            round_record: Dict[str, Any] = {
                "round_index": round_index,
                "scale_mode": round_spec.get("scale_mode"),
                "dimensions": [],
            }
            for scale_key in ("scale_u", "scale_v"):
                reference_poses = [mem["_state_pose"].copy() for mem in members]
                start_group_score = sum(float(mem["_best_score"]) for mem in members)
                best_group_score = start_group_score
                best_value = float(scale[scale_key])
                best_poses = [pose.copy() for pose in reference_poses]
                best_member_scores = [float(mem["_best_score"]) for mem in members]
                candidate_values = [
                    (
                        float(scale[scale_key]) * float(factor)
                        if round_spec.get("scale_mode") == "relative"
                        else float(factor)
                    )
                    for factor in round_spec["scale"]
                ]
                for trial_value in candidate_values:
                    trial_scale = {**scale, scale_key: float(trial_value)}
                    trial_poses: list[np.ndarray] = []
                    trial_scores: list[float] = []
                    for mem, reference_pose in zip(members, reference_poses):
                        pose, score = _optimize_member_pose(
                            mem,
                            reference_pose=reference_pose,
                            trial_scale=trial_scale,
                            round_spec=round_spec,
                        )
                        trial_poses.append(pose)
                        trial_scores.append(score)
                    group_score = sum(trial_scores)
                    if group_score > best_group_score:
                        best_group_score = group_score
                        best_value = float(trial_value)
                        best_poses = trial_poses
                        best_member_scores = trial_scores

                scale_before = float(scale[scale_key])
                scale[scale_key] = best_value
                for mem, pose, score in zip(
                    members, best_poses, best_member_scores
                ):
                    mem["_state_pose"] = pose
                    mem["_best_score"] = score
                round_record["dimensions"].append(
                    {
                        "scale_key": scale_key,
                        "scale_before": scale_before,
                        "scale_after": best_value,
                        "candidate_count": len(candidate_values),
                        "start_sum_iou": start_group_score,
                        "refined_sum_iou": best_group_score,
                        "delta_sum_iou": best_group_score - start_group_score,
                        "winner_at_candidate_boundary": best_value
                        in (min(candidate_values), max(candidate_values)),
                    }
                )
            round_records.append(round_record)

        results: list[Dict[str, Any]] = []
        for mem in members:
            flush_search_summary = {
                "candidate_evaluations": mem["_eval"]["evaluated"],
                "skipped_noop_candidates": mem["_eval"]["skipped_noop"],
                "rasterize_once": mem["constant_intrinsic"] is not None,
                "parallel_workers": 0,
                "search_elapsed_sec": round(time.monotonic() - mem["_search_started"], 2),
                "joint_scale_group": sorted(m["object_id"] for m in members),
                "search_policy": (
                    "nested shared-scale candidates with independent per-segment "
                    "translation/yaw re-optimization"
                ),
                "nested_rounds": round_records,
            }
            results.append(
                self._flush_finalize_member(
                    mem=mem,
                    refined_pose=mem["_state_pose"],
                    scale_u=scale["scale_u"],
                    scale_v=scale["scale_v"],
                    best_score=mem["_best_score"],
                    flush_search_summary=flush_search_summary,
                )
            )
        return results

    def _refine_static_fixture_object(
        self,
        *,
        object_id: str,
        trajectory_item: Dict[str, Any],
        support_item: Optional[Dict[str, Any]],
        mesh_path: Optional[str],
        mask_records: Dict[int, Dict[str, Any]],
        mask_arrays: Dict[str, np.ndarray],
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
        up_axis: np.ndarray,
        question_dir: Path,
        flatten_base: bool = False,
        preserve_scale: bool = False,
        forced_scale: tuple[float, float] | None = None,
    ) -> Dict[str, Any]:
        import trimesh

        entry: Dict[str, Any] = {"object_id": object_id, "status": "skipped"}
        if not mesh_path:
            entry["reason"] = "missing foundationpose mesh_path"
            return entry
        poses = [
            pose
            for pose in trajectory_item.get("poses", [])
            if isinstance(pose, dict)
            and pose.get("corrected_pose_4x4") is not None
            and self._pose_frame_index(pose) is not None
        ]
        if not poses:
            entry["reason"] = "no valid corrected poses"
            return entry
        pose_frames = {self._pose_frame_index(pose) for pose in poses}
        eval_frames = self._sample_overlap_frames(
            sorted(frame for frame in mask_records if frame in pose_frames),
            max_frames=STATIC_FIXTURE_REFINE_MAX_EVAL_FRAMES,
        )
        if not eval_frames:
            entry["reason"] = "no frames with both a corrected pose and a SAM3 mask"
            return entry

        try:
            vertices, faces = self._load_mesh_geometry(Path(mesh_path))
        except Exception as exc:
            entry["reason"] = f"failed to load mesh geometry: {exc}"
            return entry
        base_pose = np.asarray(poses[0]["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)

        base_alignment: Optional[Dict[str, Any]] = None
        if flatten_base:
            # Fit the resting base plane of irregular static geometry and rotate
            # the pose about the base centroid so the base normal matches the up axis --
            # the direct analogue of the primitives' closest-axis-to-up alignment.
            base_alignment = {"applied": False}
            axis_local = base_pose[:3, :3].T @ up_axis
            local_heights = vertices @ axis_local
            band = local_heights <= np.percentile(local_heights, PHYSION_PP_STATIC_BASE_BAND_PERCENTILE)
            band_vertices = vertices[band]
            if len(band_vertices) >= 16:
                keep = np.ones(len(band_vertices), dtype=bool)
                normal_local = axis_local
                for _ in range(3):
                    center = band_vertices[keep].mean(axis=0)
                    _, _, vt = np.linalg.svd(band_vertices[keep] - center, full_matrices=False)
                    normal_local = vt[-1]
                    if float(normal_local @ axis_local) < 0.0:
                        normal_local = -normal_local
                    distance = np.abs((band_vertices - center) @ normal_local)
                    keep = distance < np.percentile(distance[keep], 80)
                normal_camera = base_pose[:3, :3] @ normal_local
                cos_angle = float(np.clip(normal_camera @ up_axis, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_angle)))
                base_alignment.update({"base_tilt_deg": angle_deg, "band_vertex_count": int(band.sum())})
                if 1e-3 < angle_deg <= PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG:
                    rotation_axis = np.cross(normal_camera, up_axis)
                    axis_norm = float(np.linalg.norm(rotation_axis))
                    if axis_norm > 1e-9:
                        rotation_axis = rotation_axis / axis_norm
                        align_rotation = _rotation_about_axis(rotation_axis, float(np.arccos(cos_angle)))
                        base_centroid_local = band_vertices.mean(axis=0)
                        base_centroid_camera = base_pose[:3, :3] @ base_centroid_local + base_pose[:3, 3]
                        aligned = base_pose.copy()
                        aligned[:3, :3] = align_rotation @ base_pose[:3, :3]
                        aligned[:3, 3] = base_centroid_camera - aligned[:3, :3] @ base_centroid_local
                        base_pose = aligned
                        base_alignment["applied"] = True
                elif angle_deg > PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG:
                    base_alignment["reason"] = "base tilt exceeds alignment cap"
                    base_alignment["cap_deg"] = PHYSION_PP_STATIC_BASE_ALIGN_MAX_DEG
            else:
                base_alignment["reason"] = "too few base-band vertices for plane fit"
            # Pin the percentile base onto the support plane BEFORE the in-plane search:
            # shift/yaw moves preserve height, so flushness holds by construction while
            # the wide grids below re-optimize the silhouette match.
            pinned_ground = self._support_item_ground_height(support_item)
            if pinned_ground is not None:
                axis_local_pin = base_pose[:3, :3].T @ up_axis
                bottom_pin = float(
                    np.percentile(vertices @ axis_local_pin, PHYSION_PP_STATIC_BASE_PERCENTILE)
                )
                base_cam_pin = bottom_pin + float(base_pose[:3, 3] @ up_axis)
                base_pose = base_pose.copy()
                base_pose[:3, 3] = base_pose[:3, 3] - up_axis * (base_cam_pin - float(pinned_ground))
                base_alignment["pre_search_flush_pin"] = True

        to_origin, _extents = trimesh.bounds.oriented_bounds(vertices)
        obb_transform = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
        obb_axes = obb_transform[:3, :3]
        obb_center = obb_transform[:3, 3]
        support_axis_local = base_pose[:3, :3].T @ up_axis
        axis_alignment = np.abs(obb_axes.T @ support_axis_local)
        support_column = int(np.argmax(axis_alignment))
        plane_columns = [column for column in range(3) if column != support_column]
        plane_axis_u = obb_axes[:, plane_columns[0]]
        plane_axis_v = obb_axes[:, plane_columns[1]]
        support_axis_box = obb_axes[:, support_column]

        def _scaled_vertices(scale_u: float, scale_v: float) -> np.ndarray:
            scaling = (
                scale_u * np.outer(plane_axis_u, plane_axis_u)
                + scale_v * np.outer(plane_axis_v, plane_axis_v)
                + np.outer(support_axis_box, support_axis_box)
            )
            return (vertices - obb_center) @ scaling.T + obb_center

        target_masks: Dict[int, np.ndarray] = {}
        for frame_index in eval_frames:
            mask_record = mask_records.get(frame_index) or {}
            target_mask = self._resize_mask_to_shape(
                mask_arrays.get(str(mask_record.get("mask_key") or "")), image_shape
            )
            if target_mask is not None:
                target_masks[frame_index] = target_mask

        # Everything the candidate scorer needs, numpy-only so it pickles into pool
        # workers. The intrinsic is hoisted out of the frame loop when it is constant
        # across the eval frames (the MoGe-2 K_fixed pipeline guarantees this); the
        # per-frame fallback keeps behavior identical for per-frame intrinsics.
        eval_frame_order = list(target_masks)
        eval_intrinsics = [self._intrinsic_for_frame(intrinsics, f) for f in eval_frame_order]
        constant_intrinsic = (
            eval_intrinsics[0]
            if eval_intrinsics
            and all(np.array_equal(eval_intrinsics[0], K) for K in eval_intrinsics[1:])
            else None
        )
        score_state: Dict[str, Any] = {
            "vertices": vertices,
            "faces": faces,
            "obb_center": obb_center,
            "outer_u": np.outer(plane_axis_u, plane_axis_u),
            "outer_v": np.outer(plane_axis_v, plane_axis_v),
            "outer_support": np.outer(support_axis_box, support_axis_box),
            "image_shape": image_shape,
            "constant_intrinsic": constant_intrinsic,
            "intrinsic_by_frame": dict(zip(eval_frame_order, eval_intrinsics)),
            "target_masks": target_masks,
            "occluder_by_frame": self._object_mask_union_by_frame(
                question_dir, eval_frame_order, image_shape
            ),
        }

        def _score(pose: np.ndarray, scale_u: float, scale_v: float) -> float:
            return _flush_refine_score(score_state, pose, scale_u, scale_v)

        search_started = time.monotonic()
        eval_counter = {"evaluated": 0, "skipped_noop": 0}
        pool_ctx: Dict[str, Any] = {"pool": None, "workers": 0}

        def _score_many(jobs: list) -> list:
            eval_counter["evaluated"] += len(jobs)
            pool = pool_ctx["pool"]
            if pool is not None:
                try:
                    return list(pool.map(_flush_refine_pool_score, jobs))
                except Exception as exc:
                    _log_tool(
                        self.tool_name,
                        f"flush_refine pool scoring failed, serial fallback: {_short_text(str(exc), 160)}",
                    )
                    try:
                        pool.shutdown(wait=False)
                    except Exception:
                        pass
                    pool_ctx["pool"], pool_ctx["workers"] = None, 0
            return [_flush_refine_score(score_state, pose, su, sv) for pose, su, sv in jobs]

        u_camera, v_camera = _plane_basis_for_up_axis(up_axis)
        # forced_scale (two-segment reuse): mesh_path passed in is already the seg1 source's
        # FINAL (post-flush) mesh, so pin the scale to identity and skip the scale search --
        # flush only re-solves this object's position (translate + yaw + ground pin).
        if forced_scale is not None:
            preserve_scale = True
        init_su, init_sv = forced_scale if forced_scale is not None else (1.0, 1.0)
        state = {"pose": base_pose.copy(), "scale_u": float(init_su), "scale_v": float(init_sv)}
        start_score = _score(state["pose"], state["scale_u"], state["scale_v"])
        eval_counter["evaluated"] += 1
        best_score = start_score

        def _yawed(pose: np.ndarray, angle_rad: float) -> np.ndarray:
            rotation = _rotation_about_axis(up_axis, angle_rad)
            centroid_camera = pose[:3, :3] @ obb_center + pose[:3, 3]
            out = pose.copy()
            out[:3, :3] = rotation @ pose[:3, :3]
            out[:3, 3] = centroid_camera - out[:3, :3] @ obb_center
            return out

        # Coordinate descent, batched: each sweep's candidates are scored together
        # (in the pool when available) and applied with the sequential rule -- strict
        # improvement over the running best, earliest maximum wins -- so results are
        # identical to per-candidate evaluation. Candidates bitwise-identical to the
        # incumbent state can never pass the strict >, so they are skipped outright.
        try:
            worker_count = max(1, min(FLUSH_REFINE_POOL_MAX_WORKERS, os.cpu_count() or 1))
            if worker_count > 1:
                pool_ctx["pool"] = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                    initializer=_flush_refine_pool_init,
                    initargs=(score_state,),
                )
                pool_ctx["workers"] = worker_count
        except Exception as exc:
            pool_ctx["pool"], pool_ctx["workers"] = None, 0
            _log_tool(
                self.tool_name,
                f"flush_refine pool unavailable, serial scoring: {_short_text(str(exc), 160)}",
            )
        if flatten_base:
            refine_rounds = PHYSION_PP_STATIC_REFINE_ROUNDS
            boundary_steps = PHYSION_PP_STATIC_REFINE_BOUNDARY_STEPS
            expanded_limits = PHYSION_PP_STATIC_REFINE_EXPANDED_LIMITS
        else:
            refine_rounds = STATIC_FIXTURE_REFINE_ROUNDS
            boundary_steps = STATIC_FIXTURE_REFINE_BOUNDARY_STEPS
            expanded_limits = STATIC_FIXTURE_REFINE_EXPANDED_LIMITS
        boundary_dimensions: list[Dict[str, Any]] = []

        def _boundary_record(
            *,
            dof: str,
            initial_values: Sequence[float],
            initial_winner: float,
            extension_values: Sequence[float],
            direction: Optional[str],
            final_winner: float,
            hard_limits: tuple[float, float],
        ) -> None:
            initial = [float(value) for value in initial_values]
            extension = [float(value) for value in extension_values]
            evaluated = [*initial, *extension]
            hard_lo, hard_hi = hard_limits
            tolerance = max(
                min(
                    (
                        abs(right - left)
                        for left, right in zip(sorted(set(initial))[:-1], sorted(set(initial))[1:])
                        if abs(right - left) > 1e-12
                    ),
                    default=1.0,
                )
                * 1e-6,
                1e-12,
            )
            boundary_dimensions.append(
                {
                    "dof": dof,
                    "enabled": True,
                    "initial_range": [min(initial), max(initial)],
                    "boundary_band_steps": int(boundary_steps),
                    "initial_winner": float(initial_winner),
                    "expanded": bool(extension),
                    "expanded_direction": direction,
                    "evaluated_coarse_range": [min(evaluated), max(evaluated)],
                    "final_coarse_winner": float(final_winner),
                    "hard_limits": [float(hard_lo), float(hard_hi)],
                    "search_saturated": bool(
                        final_winner <= hard_lo + tolerance
                        or final_winner >= hard_hi - tolerance
                    ),
                }
            )

        try:
            for round_index, round_spec in enumerate(refine_rounds):
                for axis_name, axis_camera in zip(
                    ("translation_u", "translation_v"), (u_camera, v_camera)
                ):
                    deltas, jobs = [], []
                    for delta in round_spec["shift"]:
                        candidate = state["pose"].copy()
                        candidate[:3, 3] = candidate[:3, 3] + axis_camera * float(delta)
                        if np.array_equal(candidate, state["pose"]):
                            eval_counter["skipped_noop"] += 1
                            continue
                        deltas.append(float(delta))
                        jobs.append((candidate, state["scale_u"], state["scale_v"]))
                    best_delta = 0.0
                    for delta, score in zip(deltas, _score_many(jobs)):
                        if score > best_score:
                            best_score, best_delta = score, delta
                    initial_best_delta = best_delta
                    extension_deltas: list[float] = []
                    extension_direction: Optional[str] = None
                    if round_index == 0:
                        extension_deltas, extension_direction = (
                            _bounded_grid_boundary_extension_values(
                                winner=initial_best_delta,
                                initial_values=round_spec["shift"],
                                hard_limits=expanded_limits["shift"],
                                boundary_steps=boundary_steps,
                            )
                        )
                        extension_jobs = []
                        for delta in extension_deltas:
                            candidate = state["pose"].copy()
                            candidate[:3, 3] = candidate[:3, 3] + axis_camera * float(delta)
                            extension_jobs.append(
                                (candidate, state["scale_u"], state["scale_v"])
                            )
                        for delta, score in zip(extension_deltas, _score_many(extension_jobs)):
                            if score > best_score:
                                best_score, best_delta = score, float(delta)
                        _boundary_record(
                            dof=axis_name,
                            initial_values=round_spec["shift"],
                            initial_winner=initial_best_delta,
                            extension_values=extension_deltas,
                            direction=extension_direction,
                            final_winner=best_delta,
                            hard_limits=expanded_limits["shift"],
                        )
                    state["pose"][:3, 3] = state["pose"][:3, 3] + axis_camera * best_delta
                angles, jobs = [], []
                for angle_deg in round_spec["yaw_deg"]:
                    candidate = _yawed(state["pose"], float(np.deg2rad(angle_deg)))
                    if np.array_equal(candidate, state["pose"]):
                        eval_counter["skipped_noop"] += 1
                        continue
                    angles.append(float(angle_deg))
                    jobs.append((candidate, state["scale_u"], state["scale_v"]))
                best_angle = 0.0
                for angle_deg, score in zip(angles, _score_many(jobs)):
                    if score > best_score:
                        best_score, best_angle = score, angle_deg
                initial_best_angle = best_angle
                extension_angles: list[float] = []
                extension_direction = None
                if round_index == 0:
                    extension_angles, extension_direction = (
                        _bounded_grid_boundary_extension_values(
                            winner=initial_best_angle,
                            initial_values=round_spec["yaw_deg"],
                            hard_limits=expanded_limits["yaw_deg"],
                            boundary_steps=boundary_steps,
                        )
                    )
                    extension_jobs = [
                        (
                            _yawed(state["pose"], float(np.deg2rad(angle_deg))),
                            state["scale_u"],
                            state["scale_v"],
                        )
                        for angle_deg in extension_angles
                    ]
                    for angle_deg, score in zip(extension_angles, _score_many(extension_jobs)):
                        if score > best_score:
                            best_score, best_angle = score, float(angle_deg)
                    _boundary_record(
                        dof="yaw",
                        initial_values=round_spec["yaw_deg"],
                        initial_winner=initial_best_angle,
                        extension_values=extension_angles,
                        direction=extension_direction,
                        final_winner=best_angle,
                        hard_limits=expanded_limits["yaw_deg"],
                    )
                state["pose"] = _yawed(state["pose"], float(np.deg2rad(best_angle)))
                # Reused two-segment statics must keep seg1's exact size, so their flush
                # never touches scale (it only searches in-plane translation and yaw).
                if not preserve_scale:
                    for scale_key in ("scale_u", "scale_v"):
                        values, jobs = [], []
                        for factor in round_spec["scale"]:
                            trial_value = (
                                state[scale_key] * float(factor)
                                if round_spec.get("scale_mode") == "relative"
                                else float(factor)
                            )
                            if trial_value == state[scale_key]:
                                eval_counter["skipped_noop"] += 1
                                continue
                            values.append(trial_value)
                            trial = {**state, scale_key: trial_value}
                            jobs.append((state["pose"], trial["scale_u"], trial["scale_v"]))
                        best_value = state[scale_key]
                        for trial_value, score in zip(values, _score_many(jobs)):
                            if score > best_score:
                                best_score, best_value = score, trial_value
                        initial_best_value = best_value
                        extension_values: list[float] = []
                        extension_direction = None
                        if round_index == 0:
                            extension_values, extension_direction = (
                                _bounded_grid_boundary_extension_values(
                                    winner=initial_best_value,
                                    initial_values=round_spec["scale"],
                                    hard_limits=expanded_limits["scale"],
                                    boundary_steps=boundary_steps,
                                )
                            )
                            extension_jobs = []
                            for trial_value in extension_values:
                                trial = {**state, scale_key: float(trial_value)}
                                extension_jobs.append(
                                    (state["pose"], trial["scale_u"], trial["scale_v"])
                                )
                            for trial_value, score in zip(
                                extension_values, _score_many(extension_jobs)
                            ):
                                if score > best_score:
                                    best_score, best_value = score, float(trial_value)
                            _boundary_record(
                                dof=scale_key,
                                initial_values=round_spec["scale"],
                                initial_winner=initial_best_value,
                                extension_values=extension_values,
                                direction=extension_direction,
                                final_winner=best_value,
                                hard_limits=expanded_limits["scale"],
                            )
                        state[scale_key] = best_value
        finally:
            if pool_ctx["pool"] is not None:
                pool_ctx["pool"].shutdown(wait=False)
                pool_ctx["pool"] = None
        flush_search_summary = {
            "candidate_evaluations": eval_counter["evaluated"],
            "skipped_noop_candidates": eval_counter["skipped_noop"],
            "rasterize_once": constant_intrinsic is not None,
            "parallel_workers": pool_ctx["workers"],
            "search_elapsed_sec": round(time.monotonic() - search_started, 2),
            "boundary_adaptation": {
                "policy": "expand an enabled coarse DOF when its winner is within the boundary band",
                "active_dofs": [
                    "translation_u",
                    "translation_v",
                    "yaw",
                    *([] if preserve_scale else ["scale_u", "scale_v"]),
                ],
                "dimensions": boundary_dimensions,
                "disabled_dofs": (
                    [
                        {
                            "dof": "scale_u",
                            "reason": "scale locked by preserve_scale/forced_scale",
                        },
                        {
                            "dof": "scale_v",
                            "reason": "scale locked by preserve_scale/forced_scale",
                        },
                    ]
                    if preserve_scale
                    else []
                ),
            },
        }

        refined_pose = state["pose"]
        rescaled = abs(state["scale_u"] - 1.0) > 1e-3 or abs(state["scale_v"] - 1.0) > 1e-3
        final_vertices = _scaled_vertices(state["scale_u"], state["scale_v"])
        base_flattening: Optional[Dict[str, Any]] = None
        if flatten_base:
            # Physion++ statics rest on the floor: take the vertex-count low percentile
            # along the up axis as the base plane, shave every vertex below it flat, and
            # re-anchor the pose so the shaved base sits exactly on the fitted support
            # plane. Percentile of POINTS, not of the height extent: a dense flat bottom
            # face keeps the plane at the face instead of pushing the body down.
            axis_local = refined_pose[:3, :3].T @ up_axis
            heights = final_vertices @ axis_local
            min_height = float(heights.min())
            bottom = float(np.percentile(heights, PHYSION_PP_STATIC_BASE_PERCENTILE))
            below = heights < bottom
            if bool(below.any()):
                final_vertices = final_vertices + np.outer(
                    np.clip(bottom - heights, 0.0, None), axis_local
                )
            ground_height = self._support_item_ground_height(support_item)
            refined_pose = refined_pose.copy()
            if ground_height is not None:
                base_height_camera = bottom + float(np.dot(refined_pose[:3, 3], up_axis))
                sink = base_height_camera - float(ground_height)
            else:
                sink = bottom - min_height
            refined_pose[:3, 3] = refined_pose[:3, 3] - up_axis * sink
            base_flattening = {
                "applied": True,
                "base_percentile": PHYSION_PP_STATIC_BASE_PERCENTILE,
                "base_height_local": bottom,
                "min_height_local": min_height,
                "shaved_vertex_count": int(below.sum()),
                "flush_sink_distance": float(sink),
                "flush_anchor": "support_plane_height" if ground_height is not None else "old_min_vertex",
            }
        translation_shift = refined_pose[:3, 3] - base_pose[:3, 3]
        refined_mesh_path: Optional[str] = None
        # forced_scale (reuse) always exports: mesh_path is the seg1 source's mesh, so this
        # object's own refined_mesh_path / support_item mesh must be re-pointed at that shape
        # even when the identity scale search leaves the vertices untouched.
        if rescaled or (base_flattening or {}).get("shaved_vertex_count") or forced_scale is not None:
            mesh_dir = self.artifacts.tool_dir(question_dir, self.tool_name) / "static_fixture_meshes"
            refined_mesh_file = mesh_dir / f"{object_id}_flush_box_local.glb"
            self._export_mesh_with_source_colors(
                vertices=final_vertices,
                faces=faces,
                source_mesh_path=Path(mesh_path),
                output_path=refined_mesh_file,
            )
            refined_mesh_path = str(refined_mesh_file)
            if isinstance(support_item, dict):
                support_item["source_mesh_path"] = support_item.get("mesh_path")
                support_item["mesh_path"] = refined_mesh_path
                support_item["mesh_source"] = "static_fixture_flush_refinement"

        for pose in trajectory_item.get("poses", []):
            if not isinstance(pose, dict) or pose.get("corrected_pose_4x4") is None:
                continue
            pose["corrected_pose_4x4"] = refined_pose.tolist()
            pose["corrected_translation_camera"] = refined_pose[:3, 3].tolist()
        trajectory_item["static_fixture_flush_refinement_applied"] = True

        entry.update(
            {
                "status": "ok",
                "eval_frame_indices": [int(frame) for frame in eval_frames],
                "start_mean_iou": None if not np.isfinite(start_score) else float(start_score),
                "refined_mean_iou": None if not np.isfinite(best_score) else float(best_score),
                "translation_shift_camera": translation_shift.tolist(),
                "translation_shift_norm": float(np.linalg.norm(translation_shift)),
                "scale_u": float(state["scale_u"]),
                "scale_v": float(state["scale_v"]),
                "refined_mesh_path": refined_mesh_path,
                "refined_pose_4x4": refined_pose.tolist(),
                "base_alignment": base_alignment,
                "base_flattening": base_flattening,
                "flush_search": flush_search_summary,
            }
        )
        return entry

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
        static_fixture_object_ids = self._static_ground_fixture_object_ids(object_plan)
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
            if (
                is_pp_static
                and geometry_type not in {"box", "cube", "cylinder", "sphere"}
                and up_axis is not None
            ):
                pp_base_plane = self._pp_static_base_plane(
                    question_dir=question_dir,
                    object_id=object_id,
                    item=item,
                    up_axis=up_axis,
                )
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

    def _pp_static_base_plane(
        self,
        *,
        question_dir: Path,
        object_id: str,
        item: Dict[str, Any],
        up_axis: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Fit the resting base plane (lowest vertex band, robustly trimmed) of a
        Physion++ irregular static in the mesh-local frame. Returns normal + centroid
        for _apply_base_plane_alignment, or None when unavailable."""
        try:
            mesh_path = self._foundationpose_mesh_paths(question_dir).get(object_id)
            if not mesh_path:
                return None
            cache = getattr(self, "_pp_mesh_vertices_cache", None)
            if cache is None:
                cache = {}
                self._pp_mesh_vertices_cache = cache
            vertices = cache.get(mesh_path)
            if vertices is None:
                vertices = _load_mesh_vertices(Path(mesh_path))
                cache[mesh_path] = vertices
            first_pose = next(
                (
                    np.asarray(pose["corrected_pose_4x4"], dtype=np.float64).reshape(4, 4)
                    for pose in item.get("poses", [])
                    if isinstance(pose, dict) and pose.get("corrected_pose_4x4") is not None
                ),
                None,
            )
            if first_pose is None:
                return None
            axis_local = first_pose[:3, :3].T @ up_axis
            heights = vertices @ axis_local
            band = heights <= np.percentile(heights, PHYSION_PP_STATIC_BASE_BAND_PERCENTILE)
            band_vertices = vertices[band]
            if len(band_vertices) < 16:
                return None
            keep = np.ones(len(band_vertices), dtype=bool)
            normal_local = axis_local
            for _ in range(3):
                center = band_vertices[keep].mean(axis=0)
                _, _, vt = np.linalg.svd(band_vertices[keep] - center, full_matrices=False)
                normal_local = vt[-1]
                if float(normal_local @ axis_local) < 0.0:
                    normal_local = -normal_local
                distance = np.abs((band_vertices - center) @ normal_local)
                keep = distance < np.percentile(distance[keep], 80)
            return {
                "normal_local": normal_local,
                "centroid_local": band_vertices.mean(axis=0),
                "band_vertex_count": int(band.sum()),
            }
        except Exception:
            return None

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

    def _render_physics_alignment_blender_debug(
        self,
        *,
        question_dir: Path,
        result_path: Path,
        result: Dict[str, Any],
        physics_alignment_manifest: Dict[str, Any],
        command: str,
    ) -> None:
        render_dir = result_path.parent.parent / "debug"
        render_input_path = render_dir / "world_reconstruction_step_input.json"
        output_video = render_dir / "world_reconstruction_debug.mp4"
        output_json = render_dir / "world_reconstruction_debug.json"
        rollout = result.setdefault("physics_rollout", {})
        if not isinstance(rollout, dict):
            result["physics_rollout"] = {}
            rollout = result["physics_rollout"]

        try:
            render_payload = self._physics_alignment_blender_render_payload(
                question_dir=question_dir,
                result=result,
                physics_alignment_manifest=physics_alignment_manifest,
            )
            render_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts.write(render_input_path, render_payload)
        except Exception as exc:
            rollout["blender_debug_render"] = {
                "status": "tool_error",
                "message": f"failed to build render input: {exc}",
                "input": str(render_input_path),
            }
            self.artifacts.write(result_path, result)
            return

        _log_tool(
            self.tool_name,
            f"physics_alignment_blender_render command start input={render_input_path} output={output_video}",
        )
        render_result = run_world_reconstruction_debug_render(
            command=command,
            render_input_path=render_input_path,
            output_video=output_video,
            output_json=output_json,
        )
        elapsed = float(render_result.get("elapsed_sec") or 0.0)
        stdout = _short_text(str(render_result.get("stdout") or ""))
        stderr = _short_text(str(render_result.get("stderr") or ""))
        _log_tool(
            self.tool_name,
            f"physics_alignment_blender_render command end returncode={render_result.get('returncode')} elapsed={elapsed:.1f}s",
        )
        if stdout:
            _log_tool(self.tool_name, f"physics_alignment_blender_render stdout {stdout}")
        if stderr:
            _log_tool(self.tool_name, f"physics_alignment_blender_render stderr {stderr}")

        if render_result.get("status") != "ok":
            _log_tool(
                self.tool_name,
                f"debug_render error returncode={render_result.get('returncode')} "
                f"message={_short_text(str(render_result.get('message') or render_result.get('stderr') or render_result.get('stdout') or ''))}",
            )
            rollout["blender_debug_render"] = {
                "status": "tool_error",
                "message": str(render_result.get("message") or render_result.get("stderr") or render_result.get("stdout") or "").strip(),
                "input": str(render_input_path),
                "output_json": str(output_json),
            }
            self.artifacts.write(result_path, result)
            return

        render_result = self.artifacts.read_optional(output_json)
        if not isinstance(render_result, dict):
            rollout["blender_debug_render"] = {
                "status": "tool_error",
                "message": "latest debug renderer did not produce a JSON object",
                "input": str(render_input_path),
                "output_json": str(output_json),
            }
            self.artifacts.write(result_path, result)
            return
        backend = str(result.get("backend") or "")
        expected_profile = (
            CLEVRER_LATEST_DEBUG_RENDER_PROFILE
            if backend == "swr_backend.impulse_analytic"
            else PHYSIONPP_LATEST_DEBUG_RENDER_PROFILE
            if backend in PHYSIONPP_SWR_BACKENDS
            else None
        )
        actual_profile = str(render_result.get("render_profile") or "")
        if expected_profile is None or actual_profile != expected_profile:
            rollout["blender_debug_render"] = {
                "status": "tool_error",
                "message": (
                    "latest debug render profile conflicts with the SWR backend: "
                    f"backend={backend!r} expected={expected_profile!r} "
                    f"actual={actual_profile!r}"
                ),
                "input": str(render_input_path),
                "output_json": str(output_json),
            }
            self.artifacts.write(result_path, result)
            return
        source_camera_video = (
            render_result.get("source_camera_video_path")
            or render_result.get("video_path")
        )
        if not source_camera_video or not Path(str(source_camera_video)).is_file():
            rollout["blender_debug_render"] = {
                "status": "tool_error",
                "message": "latest debug renderer did not produce its source-camera video",
                "input": str(render_input_path),
                "output_json": str(output_json),
            }
            self.artifacts.write(result_path, result)
            return
        rollout["blender_debug_render"] = {
            "status": "ok",
            "render_profile": actual_profile,
            "input": str(render_input_path),
            "output_json": str(output_json),
            "source_camera_video_path": str(source_camera_video),
            "blend_path": str(output_json.with_suffix(".blend")),
            "render_result": render_result,
        }
        self.artifacts.write(result_path, result)

    def _physics_alignment_blender_render_payload(
        self,
        *,
        question_dir: Path,
        result: Dict[str, Any],
        physics_alignment_manifest: Dict[str, Any],
        simulated_override: Optional[Dict[str, list[dict[str, Any]]]] = None,
        source: str = "simulatable_world_reconstruction.world_reconstruction_fit",
    ) -> Dict[str, Any]:
        rollout = result.get("physics_rollout") if isinstance(result.get("physics_rollout"), dict) else {}
        simulated = simulated_override if simulated_override is not None else (
            rollout.get("simulated_trajectories") if isinstance(rollout.get("simulated_trajectories"), dict) else {}
        )
        if not simulated:
            raise ValueError("missing physics_rollout.simulated_trajectories")

        target_trajectories = physics_alignment_manifest.get("target_trajectories")
        target_objects = target_trajectories.get("objects", []) if isinstance(target_trajectories, dict) else []
        target_by_object_id = {
            str(item.get("object_id")): item
            for item in target_objects
            if isinstance(item, dict) and item.get("object_id")
        }
        render_objects = []
        for object_id, records in sorted(simulated.items()):
            if not isinstance(records, list):
                continue
            target = target_by_object_id.get(str(object_id), {})
            mesh_path = target.get("mesh_path")
            if not mesh_path:
                continue
            poses = []
            for record in records:
                if not isinstance(record, dict) or record.get("frame_index") is None:
                    continue
                matrix = record.get("pose_4x4")
                if matrix is None:
                    continue
                try:
                    pose_blender = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(pose_blender).all():
                    continue
                pose_camera = self._blender_world_pose_to_opencv_camera_pose(pose_blender)
                poses.append(
                    {
                        "frame_index": int(record["frame_index"]),
                        "corrected_pose_4x4": pose_camera.tolist(),
                        "corrected_translation_camera": pose_camera[:3, 3].astype(float).tolist(),
                        "source_pose_4x4_blender_world": pose_blender.tolist(),
                    }
                )
            poses.sort(key=lambda item: int(item["frame_index"]))
            if not poses:
                continue
            render_objects.append(
                {
                    "object_id": str(object_id),
                    "status": "ok",
                    "mesh_path": str(mesh_path),
                    "activation": target.get("activation") if isinstance(target.get("activation"), dict) else self._activation_from_poses(poses),
                    "render_pose_source": "world_reconstruction_fit.physics_rollout.simulated_trajectories.pose_4x4",
                    "mesh_source": "pose_correction.support_plane_position_correction",
                    "pose_count": len(poses),
                    "poses": poses,
                }
            )
        if not render_objects:
            raise ValueError("no renderable simulated object trajectories")

        return {
            "tool": "simulatable_world_reconstruction_debug_render_input",
            "status": "ok",
            "question_dir": str(question_dir),
            "source": source,
            "video_metadata": physics_alignment_manifest.get("video_metadata") if isinstance(physics_alignment_manifest.get("video_metadata"), dict) else {},
            "gravity_direction_camera": physics_alignment_manifest.get("gravity_direction_camera"),
            "gravity_direction_coordinate_frame": physics_alignment_manifest.get("gravity_direction_coordinate_frame"),
            "static_scene_objects": physics_alignment_manifest.get("static_scene_objects", []),
            "support_plane_position_correction": physics_alignment_manifest.get("support_plane_position_correction"),
            "vrdp_support_plane": result.get("vrdp_support_plane") if isinstance(result.get("vrdp_support_plane"), dict) else None,
            "analytic_support_plane": result.get("analytic_support_plane") if isinstance(result.get("analytic_support_plane"), dict) else None,
            "trajectory_correction": {
                "applied": True,
                "source": source,
                "corrected_trajectories": {
                    "applied": True,
                    "source": source,
                    "pose_field": "corrected_pose_4x4",
                    "translation_field": "corrected_translation_camera",
                    "objects": render_objects,
                },
            },
            "world_reconstruction_fit": {
                "backend": result.get("backend"),
                "mode": result.get("mode"),
                "fit_error": result.get("fit_error"),
                "alignment_optimization": result.get("alignment_optimization"),
            },
        }

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
        if self.artifacts.debug_artifacts:
            export_args.append("--write-selected-frame")
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

    def _mc_extra_patient_link(
        self,
        tracks_payload: Dict[str, Any],
        *,
        resolved_route: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Link kept seg1 extras to the seg2 patient with focused VLM YES/NO decisions."""
        link: Dict[str, Any] = {
            "linked": False,
            "method": "vlm_yes_no_same_object",
        }
        _record_mass_extra_patient_link_result(link, resolved_route)
        role_binding = (tracks_payload.get("physion_tracking") or {}).get("role_binding") or {}
        extras = [
            entry
            for entry in role_binding.get("kept_extra_tracks") or []
            if isinstance(entry, dict) and entry.get("track")
        ]
        patient_track = str(role_binding.get("patient_track") or "")
        if not extras:
            link["method"] = "no_candidate"
            link["reason"] = "no kept extra tracks"
            return link
        if not patient_track:
            raise ValueError("TRK-005 VLM identity is missing the seg2 patient track")
        video_path = tracks_payload.get("video")
        mask_sidecar = tracks_payload.get("mask_sidecar")
        if not video_path or not mask_sidecar:
            raise ValueError("TRK-005 VLM identity is missing its video or mask sidecar")
        capture = None
        try:
            import cv2

            records_by_track = self._records_by_track(tracks_payload)
            capture = cv2.VideoCapture(str(video_path))
            with np.load(str(mask_sidecar)) as arrays:

                def _frame_and_mask(
                    track_id: str,
                    frame_index: Optional[int],
                ) -> tuple[np.ndarray, np.ndarray, int]:
                    records = [
                        record
                        for record in records_by_track.get(str(track_id)) or []
                        if isinstance(record, dict) and record.get("mask_key")
                    ]
                    if not records:
                        raise ValueError(
                            f"TRK-005 VLM identity has no mask records for {track_id}"
                        )
                    record = None
                    if frame_index is not None:
                        record = next(
                            (
                                r
                                for r in records
                                if int(r.get("frame_index", -1)) == int(frame_index)
                            ),
                            None,
                        )
                    if record is None:
                        record = max(records, key=lambda r: float(r.get("area") or 0.0))
                    mask_key = str(record.get("mask_key"))
                    if mask_key not in arrays.files:
                        raise ValueError(
                            f"TRK-005 VLM identity mask sidecar is missing {mask_key}"
                        )
                    mask = arrays[mask_key] > 0
                    selected_frame = int(record.get("frame_index") or 0)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(
                            f"TRK-005 VLM identity could not decode frame {selected_frame}"
                        )
                    if frame.shape[:2] != mask.shape:
                        frame = cv2.resize(
                            frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR
                        )
                    if not mask.any():
                        raise ValueError(
                            f"TRK-005 VLM identity mask is empty for {track_id}"
                        )
                    return frame, mask, selected_frame

                patient_frame, patient_mask, patient_frame_index = _frame_and_mask(
                    patient_track,
                    None,
                )
                decisions = []
                for entry in extras:
                    extra_track = str(entry["track"])
                    extra_frame, extra_mask, extra_frame_index = _frame_and_mask(
                        extra_track,
                        entry.get("settled_frame"),
                    )
                    decision = decide_same_object_yes_no(
                        config=self.config,
                        reference_image=patient_frame,
                        reference_mask=patient_mask,
                        candidate_image=extra_frame,
                        candidate_mask=extra_mask,
                        reference_id=patient_track,
                        candidate_id=extra_track,
                        request_context={
                            "stage": "mass_extra_patient_link",
                            "decision_id": MASS_EXTRA_PATIENT_LINK_DECISION_ID,
                        },
                    )
                    decision["reference_frame_index"] = patient_frame_index
                    decision["candidate_frame_index"] = extra_frame_index
                    decisions.append(decision)
                linked_decisions = [
                    decision for decision in decisions if decision["linked"]
                ]
                if len(linked_decisions) > 1:
                    raise ValueError(
                        "TRK-005 VLM identity linked multiple extras to one patient: "
                        f"{[decision['candidate_track'] for decision in linked_decisions]}"
                    )
                link["vlm_decisions"] = decisions
                link["patient_track"] = patient_track
                if linked_decisions:
                    link["linked"] = True
                    link["extra_track"] = linked_decisions[0]["candidate_track"]
                else:
                    link["reason"] = "VLM rejected all kept extras"
        finally:
            if capture is not None:
                capture.release()
        return link

    def _apply_two_segment_mesh_reuse(
        self,
        *,
        tracks_payload: Dict[str, Any],
        target_objects: list[TargetObject],
        cross_segment_mesh_reuse_route: Dict[str, Any] | None,
        mass_extra_patient_link_route: Dict[str, Any] | None,
    ) -> Dict[str, Any] | None:
        """Physion++ two-segment scenarios: tag each seg2 object to reuse its same-role
        seg1 mesh.

        The clip splices two independent trials that share the same physical objects; only
        the layout and the mover's trajectory differ. Reconstructing every object
        twice wastes work and, worse, yields two slightly different meshes for the same
        object. Instead the seg2 object reuses seg1's mesh (geometry + metric scale) and
        only its pose is re-estimated per segment. Role-to-track pairing comes from the
        two-segment detector's role_binding.assignments; mass_collision additionally
        pairs the ball (stored under role_binding.ball) and the seg2 patient with the
        VLM-linked seg1 kept extra (the same physical object, parked in seg1). Gated
        to two-segment scenarios; no tag is set for any other scenario, so the whole
        reuse path stays inert there.
        """
        if cross_segment_mesh_reuse_route is None:
            return None
        route = cross_segment_mesh_reuse_route.get("route")
        if route not in CROSS_SEGMENT_MESH_REUSE_ROUTES:
            raise ValueError(f"unsupported cross-segment-mesh-reuse route: {route!r}")
        physion_scenario = str(tracks_payload.get("physion_scenario") or "").strip().lower()
        role_binding = (tracks_payload.get("physion_tracking") or {}).get("role_binding") or {}
        assignments = role_binding.get("assignments")
        if not isinstance(assignments, dict):
            return None
        seg1 = assignments.get("seg1") if isinstance(assignments.get("seg1"), dict) else {}
        seg2 = assignments.get("seg2") if isinstance(assignments.get("seg2"), dict) else {}
        object_id_by_track = {
            str(obj.source_track_id): obj.object_id
            for obj in target_objects
            if obj.source_track_id
        }
        object_by_id = {obj.object_id: obj for obj in target_objects}
        reused: list[Dict[str, Any]] = []
        skipped: list[Dict[str, Any]] = []
        pairs: list[tuple[str, Any, Any]] = [
            (role, seg1.get(role), seg2.get(role)) for role in ("agent", "patient", "wall")
        ]
        mc_link: Optional[Dict[str, Any]] = None
        if route == MASS_COLLISION_CROSS_SEGMENT_MESH_REUSE_ROUTE:
            if mass_extra_patient_link_route is None:
                raise ValueError(
                    "Mass Collision mesh reuse is missing its "
                    "mass-extra-patient-link route record"
                )
            ball = role_binding.get("ball") if isinstance(role_binding.get("ball"), dict) else {}
            pairs.append(("ball", ball.get("seg1"), ball.get("seg2")))
            mc_link = self._mc_extra_patient_link(
                tracks_payload,
                resolved_route=mass_extra_patient_link_route,
            )
            if mc_link.get("linked") is True:
                pairs.append(("patient", mc_link.get("extra_track"), role_binding.get("patient_track")))
        for role, source_track, target_track in pairs:
            if not source_track or not target_track:
                continue
            source_object = object_id_by_track.get(str(source_track))
            target_object = object_id_by_track.get(str(target_track))
            if not source_object or not target_object or source_object == target_object:
                skipped.append(
                    {
                        "role": role,
                        "seg1_track": str(source_track),
                        "seg2_track": str(target_track),
                        "seg1_object": source_object,
                        "seg2_object": target_object,
                        "reason": "missing_accepted_object_for_one_or_both_segments",
                    }
                )
                continue
            object_by_id[target_object].mesh_reuse_source_object_id = source_object
            reused.append(
                {
                    "role": role,
                    "seg2_object": target_object,
                    "seg1_source_object": source_object,
                    "seg2_track": str(target_track),
                    "seg1_track": str(source_track),
                }
            )
        if not reused and not skipped and mc_link is None:
            return None
        result = {
            "scenario": physion_scenario,
            "policy": "seg2 objects reuse seg1 same-role conditioned mesh; pose re-estimated per segment",
            "resolved_route": deepcopy(cross_segment_mesh_reuse_route),
            "reused": reused,
            "skipped": skipped,
        }
        if mc_link is not None:
            result["mc_extra_patient_link"] = mc_link
        return result

    def _apply_track_derived_object_plan(
        self,
        *,
        scene: ClevrerScene,
        question_dir: Path,
        object_plan: ObjectPlan,
        labels_payload: Dict[str, Any],
    ) -> None:
        if isinstance(scene, (ClevrerScene, PhysionPPScene)):
            _record_track_derived_inventory_route(
                object_plan=object_plan,
                labels_payload=labels_payload,
            )
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
        is_physion_pp = _is_physion_pp_scenario(tracks_payload)
        raw_track_count = len(records_by_track)
        # Role-bound agent/patient tracks bypass the minimum-area gate. Frame-count
        # and duplicate gates still apply.
        physion_tracking = tracks_payload.get("physion_tracking")
        role_binding = physion_tracking.get("role_binding") if isinstance(physion_tracking, dict) else None
        role_bound_track_ids = set()
        if is_physion_pp and isinstance(role_binding, dict):
            for _role_key in ("agent_track", "patient_track"):
                _bound_id = str(role_binding.get(_role_key) or "").strip()
                if _bound_id:
                    role_bound_track_ids.add(_bound_id)
            # mass_collision: the recognition cleanup already dropped every unbound track,
            # so every survivor is role-relevant -- the per-segment agents, the seg1/seg2
            # ball (stored under role_binding.ball, not agent_track/patient_track), and the
            # kept extras (the parked seg2 patient observed in seg1). Exempt the whole
            # roster, not just the two seg2 role tracks.
            if str(tracks_payload.get("physion_scenario") or "").strip().lower() == "mass_collision_pp":
                _assignments = role_binding.get("assignments")
                if isinstance(_assignments, dict):
                    for _seg_roles in _assignments.values():
                        if isinstance(_seg_roles, dict):
                            for _role_track in _seg_roles.values():
                                _bound_id = str(_role_track or "").strip()
                                if _bound_id:
                                    role_bound_track_ids.add(_bound_id)
                _ball = role_binding.get("ball")
                if isinstance(_ball, dict):
                    for _seg_key in ("seg1", "seg2"):
                        _bound_id = str(_ball.get(_seg_key) or "").strip()
                        if _bound_id:
                            role_bound_track_ids.add(_bound_id)
                for _extra in role_binding.get("kept_extra_tracks") or []:
                    if isinstance(_extra, dict):
                        _bound_id = str(_extra.get("track") or "").strip()
                        if _bound_id:
                            role_bound_track_ids.add(_bound_id)
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
            is_static_fixture = is_physion_pp and _is_physion_static_ground_fixture_track(info["track_id"])
            forced_box = is_physion_pp and _is_physion_forced_box_fixture_track(info["track_id"])
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

        mesh_reuse = self._apply_two_segment_mesh_reuse(
            tracks_payload=tracks_payload,
            target_objects=target_objects,
            cross_segment_mesh_reuse_route=cross_segment_mesh_reuse_route,
            mass_extra_patient_link_route=mass_extra_patient_link_route,
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
        is_physion_pp_bench = _is_physion_pp_scenario(tracks_payload)
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
                is_physion_edge_exempt = (
                    is_physion_pp_bench
                    and (
                        _is_physion_edge_exempt_track(target.source_track_id)
                        or _is_physion_pp_scenario(tracks_payload)
                    )
                    and candidates
                )
                if is_physion_edge_exempt:
                    interval = selected.get("track_interval") if isinstance(selected.get("track_interval"), dict) else {}
                    first_record = min(
                        candidates,
                        key=lambda item: (int(item.get("frame_index", keyframe["frame_index"])), -int(item.get("selected_area") or 0)),
                    )
                    last_record = max(
                        candidates,
                        key=lambda item: (int(item.get("frame_index", keyframe["frame_index"])), int(item.get("selected_area") or 0)),
                    )
                    first_frame_index = int(first_record.get("frame_index", keyframe["frame_index"]))
                    last_frame_index = int(last_record.get("frame_index", keyframe["frame_index"]))
                    registration_frame_index = int(keyframe["frame_index"])
                    object_pose_frames.append(
                        {
                            "object_id": target.object_id,
                            "first_full_visible_frame_index": first_frame_index,
                            "last_full_visible_frame_index": last_frame_index,
                            "sam3_prompt": keyframe.get("sam3_prompt") or selected.get("sam3_prompt") or target.description,
                            "first_frame_reason": (
                                "Physion++ edge-exempt fixture (yellow mat or ramp) has no mask fully inside the "
                                "image boundary; pose interval falls back to the earliest available SAM3 video mask."
                            ),
                            "last_frame_reason": (
                                "Physion++ edge-exempt fixture (yellow mat or ramp) has no mask fully inside the "
                                "image boundary; pose interval falls back to the latest available SAM3 video mask."
                            ),
                            "registration_frame_reason": (
                                "FoundationPose registration frame is reused from SAM3 track-label representatives."
                            ),
                            "confidence": selected.get("label_confidence") or keyframe.get("label_confidence"),
                            "description": target.description,
                            "role": target.role,
                            "registration_frame_selection_rule": "metric_mesh_selected_keyframe",
                            "first_frame_selection_rule": "physion_edge_exempt_boundary_fallback_earliest_track_frame",
                            "last_frame_selection_rule": "physion_edge_exempt_boundary_fallback_latest_track_frame",
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
                                    "eligible_candidate_count": 0,
                                    "rule": "earliest available mask for Physion++ edge-exempt fixture boundary fallback",
                                    "fallback_used": True,
                                },
                                "last": {
                                    "source": "full-video SAM3 masks",
                                    "candidate_count": len(candidates),
                                    "eligible_candidate_count": 0,
                                    "rule": "latest available mask for Physion++ edge-exempt fixture boundary fallback",
                                    "fallback_used": True,
                                },
                            },
                            "first_frame_candidate_count": len(candidates),
                            "first_frame_eligible_count": 0,
                            "first_frame_fallback_used": True,
                            "last_frame_candidate_count": len(candidates),
                            "last_frame_eligible_count": 0,
                            "last_frame_fallback_used": True,
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
                            "last_mask_key": last_record.get("selected_mask_key"),
                            "last_mask_area": last_record.get("selected_area"),
                            "last_mask_bbox_xyxy": last_record.get("selected_bbox_xyxy"),
                            "metric_mesh_keyframe": {
                                "frame_index": registration_frame_index,
                                "mask_key": keyframe_mask_key,
                                "selected_frame_image": keyframe.get("selected_frame_image"),
                                "selected_frame_image_format": keyframe.get("selected_frame_image_format"),
                                "selected_frame_image_fingerprint": keyframe.get("selected_frame_image_fingerprint"),
                            },
                            "track_interval": interval,
                            "status": "ok",
                            "physion_edge_exempt_boundary_fallback": True,
                        }
                    )
                    continue
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
            if is_physion_pp_bench and len(full_visible_candidates) < len(candidates):
                boundary_area_continuation = _physion_pp_boundary_area_continuation(candidates)
                first_interval_record = boundary_area_continuation.pop("first_record")
                last_interval_record = boundary_area_continuation.pop("last_record")
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
