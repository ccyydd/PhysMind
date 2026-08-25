from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.world_model import analytic_swr_common  # noqa: E402
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr  # noqa: E402
from scripts.world_model import swr_sysid_common as sysid_common  # noqa: E402


GRAVITY_M_PER_S2 = friction_swr.GRAVITY_M_PER_S2
PHYSIONPP_PHYSICS_DT_SEC = friction_swr.PHYSIONPP_PHYSICS_DT_SEC
DEFAULT_LEARNING_RATES = friction_swr.DEFAULT_LEARNING_RATES
DEFAULT_STEPS = 400
DEFAULT_PATIENCE = 100
MIN_FRICTION = friction_swr.MIN_FRICTION
MAX_FRICTION = friction_swr.MAX_FRICTION
MIN_RESTITUTION = 0.0
MAX_RESTITUTION = 1.0
RESTITUTION_PRIOR_WEIGHT = 0.002
RADIUS_PRIOR_WEIGHT = friction_swr.RADIUS_PRIOR_WEIGHT
GRAVITY_SCALE_PRIOR_WEIGHT = friction_swr.GRAVITY_SCALE_PRIOR_WEIGHT
MIN_GRAVITY_SCALE = friction_swr.MIN_GRAVITY_SCALE
MAX_GRAVITY_SCALE = friction_swr.MAX_GRAVITY_SCALE
GLOBAL_GROUND_OBJECT_ID = "global_ground"
GLOBAL_GROUND_PLANE_ID = "global_ground_plane"
GLOBAL_GROUND_MIN_HALF_EXTENT_M = 5.0


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _object_id_for_source_track(object_plan: dict[str, Any], source_track_id: str | None) -> str | None:
    if not source_track_id:
        return None
    for item in object_plan.get("target_objects") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_track_id") or "") == str(source_track_id):
            object_id = str(item.get("object_id") or "")
            return object_id or None
    return None


def _role_assignments(world_modeling_dir: Path, object_plan: dict[str, Any]) -> list[dict[str, str]]:
    tracks_path = world_modeling_dir / "object-segmentation-and-event-detection" / "sam3_video_tracks" / "sam3_video_tracks.json"
    if not tracks_path.exists():
        raise ValueError(f"missing sam3_video_tracks: {tracks_path}")
    tracks = _load_json(tracks_path)
    if not isinstance(tracks, dict):
        raise ValueError(f"invalid sam3_video_tracks: {tracks_path}")
    role_binding = ((tracks.get("physion_tracking") or {}).get("role_binding") or {})
    assignments = role_binding.get("assignments") if isinstance(role_binding, dict) else None
    if not isinstance(assignments, dict):
        raise ValueError("bouncy_wall requires physion_tracking.role_binding.assignments")
    output: list[dict[str, str]] = []
    for segment_name in sorted(assignments):
        roles = assignments.get(segment_name)
        if not isinstance(roles, dict):
            continue
        agent_id = _object_id_for_source_track(object_plan, roles.get("agent"))
        wall_id = _object_id_for_source_track(object_plan, roles.get("wall"))
        patient_id = _object_id_for_source_track(object_plan, roles.get("patient"))
        if not agent_id or not wall_id:
            continue
        item = {
            "segment": str(segment_name),
            "agent_object_id": agent_id,
            "wall_object_id": wall_id,
        }
        if patient_id:
            item["patient_object_id"] = patient_id
        output.append(item)
    if not output:
        raise ValueError("no usable bouncy_wall segment assignments")
    return output


def _raw_from_unit_interval(value: float, lower: float, upper: float) -> float:
    return friction_swr._raw_from_unit_interval(value, lower, upper)


def _segment_records(
    *,
    target: dict[str, list[dict[str, Any]]],
    object_id: str,
) -> list[dict[str, Any]]:
    records = list(target.get(object_id) or [])
    return sorted(records, key=lambda record: int(record["frame_index"]))


def _global_ground_plane_from_pose_correction(
    *,
    manifest: dict[str, Any],
    target_all: dict[str, list[dict[str, Any]]],
    segments: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = manifest.get("support_plane_position_correction")
    if not isinstance(support, dict) or support.get("applied") is not True:
        raise ValueError("bouncy_wall requires an applied pose-correction support plane")
    normal_camera = support.get("normal_camera")
    ground_height = support.get("ground_height_along_normal")
    if not isinstance(normal_camera, list) or len(normal_camera) != 3:
        raise ValueError("pose-correction support plane has no normal_camera")
    if not isinstance(ground_height, (int, float)) or not math.isfinite(float(ground_height)):
        raise ValueError("pose-correction support plane has no finite ground_height_along_normal")

    normal_blender, point_blender = sysid_common._support_plane_in_blender(manifest)
    up = np.asarray(normal_blender, dtype=np.float64).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    surface_point = np.asarray(point_blender, dtype=np.float64).reshape(3)
    ground_offset = float(np.dot(surface_point, up))
    agent_ids = list(dict.fromkeys(str(segment["agent_object_id"]) for segment in segments))
    all_positions: list[np.ndarray] = []
    for object_id in agent_ids:
        records = _segment_records(target=target_all, object_id=object_id)
        if not records:
            continue
        positions = np.asarray([record["position"] for record in records], dtype=np.float64).reshape(-1, 3)
        all_positions.append(positions)
    if not all_positions:
        raise ValueError("cannot size global ground without agent trajectories")

    positions_concat = np.concatenate(all_positions, axis=0)
    center = np.mean(positions_concat, axis=0)
    center = center + (ground_offset - float(np.dot(center, up))) * up
    reference_axis = np.zeros(3, dtype=np.float64)
    reference_axis[int(np.argmin(np.abs(up)))] = 1.0
    axis_u = np.cross(up, reference_axis)
    axis_u = axis_u / max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(up, axis_u)
    axis_v = axis_v / max(float(np.linalg.norm(axis_v)), 1e-12)
    relative = positions_concat - center.reshape(1, 3)
    tangent_extent = float(
        max(
            np.max(np.abs(relative @ axis_u)),
            np.max(np.abs(relative @ axis_v)),
        )
    )
    half_extent = max(GLOBAL_GROUND_MIN_HALF_EXTENT_M, tangent_extent + 2.0)
    corners = np.asarray(
        [
            center - half_extent * axis_u - half_extent * axis_v,
            center + half_extent * axis_u - half_extent * axis_v,
            center + half_extent * axis_u + half_extent * axis_v,
            center - half_extent * axis_u + half_extent * axis_v,
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]],
        dtype=np.float64,
    )
    plane = {
        "plane_id": GLOBAL_GROUND_PLANE_ID,
        "object_id": GLOBAL_GROUND_OBJECT_ID,
        "normal_np": up,
        "offset": ground_offset,
        "surface_point_np": up * ground_offset,
        "triangles_np": triangles,
        "area": float((2.0 * half_extent) ** 2),
    }
    plane.update(
        friction_swr._plane_boundary_runtime(
            normal=up,
            offset=ground_offset,
            triangles=triangles,
        )
    )
    diagnostics = {
        "plane_id": GLOBAL_GROUND_PLANE_ID,
        "object_id": GLOBAL_GROUND_OBJECT_ID,
        "source": "pose_correction.support_plane_position_correction",
        "normal_blender_world": up.astype(float).tolist(),
        "surface_point_blender_world_m": surface_point.astype(float).tolist(),
        "offset_m": ground_offset,
        "ground_height_along_normal_camera_m": float(ground_height),
        "half_extent_m": float(half_extent),
    }
    return plane, diagnostics


def _serializable_runtime_plane(plane: dict[str, Any]) -> dict[str, Any]:
    return {
        "plane_id": str(plane.get("plane_id") or ""),
        "object_id": str(plane.get("object_id") or ""),
        "normal": np.asarray(plane["normal_np"], dtype=np.float64).reshape(3).astype(float).tolist(),
        "offset": float(plane["offset"]),
        "surface_point": (
            np.asarray(plane["surface_point_np"], dtype=np.float64).reshape(3).astype(float).tolist()
        ),
        "triangles": (
            np.asarray(plane["triangles_np"], dtype=np.float64).reshape(-1, 3, 3).astype(float).tolist()
        ),
        "area": float(plane.get("area") or 0.0),
    }


def _wall_event_initialization(
    *,
    records: list[dict[str, Any]],
    radius: float,
    wall_planes: list[dict[str, Any]],
) -> dict[str, Any]:
    positions = np.asarray([record["position"] for record in records], dtype=np.float64).reshape(-1, 3)
    frames = np.asarray([int(record["frame_index"]) for record in records], dtype=np.int64)
    contact_index = None
    contact = None
    for index, point in enumerate(positions):
        candidate = friction_swr._select_bounded_plane_contact(
            point=point,
            radius=float(radius),
            planes=wall_planes,
            contact_margin=friction_swr.BOUNDED_PLANE_CONTACT_MARGIN_M,
        )
        if candidate is not None:
            contact_index = int(index)
            contact = candidate
            break
    if contact_index is None:
        initialization_count = min(max(len(records), 3), 10)
        velocity = friction_swr._fit_initial_velocity(
            records[:initialization_count],
            PHYSIONPP_PHYSICS_DT_SEC,
        )
        return {
            "velocity": velocity,
            "restitution": 0.8,
            "source": "leading_trajectory_window_no_observed_wall_contact",
            "trajectory_frame_count": int(initialization_count),
            "wall_contact_frame_index": None,
            "wall_plane_id": None,
        }

    initialization_count = max(contact_index, 3)
    velocity = friction_swr._fit_initial_velocity(
        records[:initialization_count],
        PHYSIONPP_PHYSICS_DT_SEC,
    )
    restitution = 0.8
    if contact is not None and len(positions) >= 2:
        delta_t = np.diff(frames).astype(np.float64) * float(PHYSIONPP_PHYSICS_DT_SEC)
        interval_velocity = np.diff(positions, axis=0) / np.maximum(delta_t[:, None], 1e-12)
        normal = np.asarray(contact["normal_np"], dtype=np.float64).reshape(3)
        normal_speed = interval_velocity @ normal
        pre_values = normal_speed[max(0, contact_index - 3) : contact_index]
        incoming = pre_values[pre_values < 0.0]
        post_values = normal_speed[contact_index : min(len(normal_speed), contact_index + 20)]
        outgoing = post_values[post_values > 0.0]
        if len(incoming) and len(outgoing):
            restitution = float(
                np.clip(
                    np.median(outgoing[:3]) / max(-float(np.median(incoming)), 1e-12),
                    MIN_RESTITUTION,
                    MAX_RESTITUTION,
                )
            )
    return {
        "velocity": velocity,
        "restitution": restitution,
        "source": "pre_wall_contact_trajectory",
        "trajectory_frame_count": int(initialization_count),
        "wall_contact_frame_index": int(frames[contact_index]),
        "wall_plane_id": str(contact["plane"].get("plane_id")) if contact is not None else None,
    }


def _wall_plane_ids(planes: list[dict[str, Any]], wall_object_id: str) -> set[str]:
    return {str(plane.get("plane_id")) for plane in planes if str(plane.get("object_id")) == str(wall_object_id)}


def _support_friction_plane_ids(
    *,
    target_positions: np.ndarray,
    radius: float,
    planes: list[dict[str, Any]],
    wall_plane_ids: set[str],
) -> set[str] | None:
    for point in np.asarray(target_positions, dtype=np.float64).reshape(-1, 3):
        contact = friction_swr._select_bounded_plane_contact(
            point=point,
            radius=float(radius),
            planes=[plane for plane in planes if str(plane.get("plane_id")) not in wall_plane_ids],
            contact_margin=friction_swr.BOUNDED_PLANE_CONTACT_MARGIN_M,
        )
        if contact is not None:
            return {str(contact["plane"].get("plane_id"))}
    return None


def _free_flight(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    gravity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x + v * dt + 0.5 * gravity * dt * dt, v + gravity * dt


def _first_contact(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    radius: torch.Tensor,
    gravity: torch.Tensor,
    planes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return friction_swr._first_bounded_plane_contact(
        x=x,
        v=v,
        dt=dt,
        radius=radius,
        gravity=gravity,
        planes=planes,
    )


def _reflect_from_wall(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    radius: torch.Tensor,
    contact: dict[str, Any],
    restitution: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    plane = friction_swr._torch_plane_from_runtime(contact, device=device, dtype=dtype)
    normal = plane["normal"] / torch.clamp(torch.linalg.norm(plane["normal"]), min=x.new_tensor(1e-12))
    x = friction_swr._project_to_contact(
        x=x,
        dynamic_radius=radius,
        plane_point=plane["surface_point"],
        normal=normal,
    )
    vn = torch.sum(v * normal)
    incoming = torch.minimum(vn, vn.new_tensor(0.0))
    v = v - (1.0 + restitution) * incoming * normal
    return x, v


def _rollout_bouncy_wall_bounded_planes(
    *,
    frames: list[int],
    physics_dt_sec: float,
    start_position: torch.Tensor,
    velocity: torch.Tensor,
    support_friction: torch.Tensor,
    wall_restitution: torch.Tensor,
    dynamic_radius: torch.Tensor,
    bounded_planes: list[dict[str, Any]],
    wall_plane_ids: set[str],
    support_friction_plane_ids: set[str] | None,
    gravity_direction: torch.Tensor,
    gravity_magnitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    gravity = gravity_direction * gravity_magnitude
    device = start_position.device
    dtype = start_position.dtype
    wall_planes = [plane for plane in bounded_planes if str(plane.get("plane_id")) in wall_plane_ids]
    zero_friction = start_position.new_tensor(0.0)

    def friction_for_support(contact: dict[str, Any]) -> torch.Tensor:
        if support_friction_plane_ids is None:
            return support_friction
        return support_friction if str(contact["plane"].get("plane_id")) in support_friction_plane_ids else zero_friction

    def contact_for_current(point: torch.Tensor, contact: dict[str, Any] | None) -> dict[str, Any] | None:
        if contact is None:
            return None
        return friction_swr._bounded_plane_contact_candidate(
            point=point.detach().cpu().numpy().astype(np.float64),
            radius=float(dynamic_radius.detach().cpu()),
            plane=contact["plane"],
        )

    x = start_position.clone()
    v = velocity.clone()
    active_support: dict[str, Any] | None = None
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    events: list[dict[str, Any]] = []
    for frame_offset, frame_index in enumerate(frames):
        positions.append(x.clone())
        velocities.append(v.clone())
        if frame_offset + 1 >= len(frames):
            break
        dt = start_position.new_tensor(
            (int(frames[frame_offset + 1]) - int(frames[frame_offset])) * float(physics_dt_sec)
        )
        if active_support is not None:
            wall_contact = _first_contact(
                x=x,
                v=v,
                dt=dt,
                radius=dynamic_radius,
                gravity=gravity,
                planes=wall_planes,
            )
            if wall_contact is not None:
                contact_time = wall_contact["contact_time"]
                support_plane = friction_swr._torch_plane_from_runtime(active_support, device=device, dtype=dtype)
                support_normal = support_plane["normal"] / torch.clamp(
                    torch.linalg.norm(support_plane["normal"]), min=start_position.new_tensor(1e-12)
                )
                x, v = analytic_swr_common.advance_on_plane_torch(
                    x=x,
                    v=v,
                    dt=contact_time,
                    radius=dynamic_radius,
                    plane_point=support_plane["surface_point"],
                    normal=support_normal,
                    gravity=gravity,
                    friction=friction_for_support(active_support),
                )
                x, v = _reflect_from_wall(
                    x=x,
                    v=v,
                    radius=dynamic_radius,
                    contact=wall_contact,
                    restitution=wall_restitution,
                    device=device,
                    dtype=dtype,
                )
                remaining = torch.clamp(dt - contact_time, min=0.0)
                x, v = analytic_swr_common.advance_on_plane_torch(
                    x=x,
                    v=v,
                    dt=remaining,
                    radius=dynamic_radius,
                    plane_point=support_plane["surface_point"],
                    normal=support_normal,
                    gravity=gravity,
                    friction=friction_for_support(active_support),
                )
                events.append(
                    {
                        "event_type": "bouncy_wall_contact",
                        "frame_index": int(frames[frame_offset + 1]),
                        "phase": "support_motion_to_wall_bounce",
                        "plane_id": str(wall_contact["plane"].get("plane_id")),
                        "support_object_id": str(wall_contact["plane"].get("object_id")),
                    }
                )
                continue
            support_plane = friction_swr._torch_plane_from_runtime(active_support, device=device, dtype=dtype)
            support_normal = support_plane["normal"] / torch.clamp(
                torch.linalg.norm(support_plane["normal"]), min=start_position.new_tensor(1e-12)
            )
            x_next, v_next = analytic_swr_common.advance_on_plane_torch(
                x=x,
                v=v,
                dt=dt,
                radius=dynamic_radius,
                plane_point=support_plane["surface_point"],
                normal=support_normal,
                gravity=gravity,
                friction=friction_for_support(active_support),
            )
            refreshed = contact_for_current(x_next, active_support)
            if refreshed is not None:
                active_support = refreshed
                x, v = x_next, v_next
            else:
                active_support = None
                x, v = _free_flight(x=x_next, v=v_next, dt=dt.new_tensor(0.0), gravity=gravity)
                events.append(
                    {
                        "event_type": "bounded_plane_contact_transition",
                        "frame_index": int(frames[frame_offset + 1]),
                        "phase": "support_to_free_flight",
                    }
                )
            continue

        contact = _first_contact(
            x=x,
            v=v,
            dt=dt,
            radius=dynamic_radius,
            gravity=gravity,
            planes=bounded_planes,
        )
        if contact is None:
            x, v = _free_flight(x=x, v=v, dt=dt, gravity=gravity)
            continue
        contact_time = contact["contact_time"]
        x_hit = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
        v_hit = v + gravity * contact_time
        plane_id = str(contact["plane"].get("plane_id"))
        if plane_id in wall_plane_ids:
            x_hit, v_hit = _reflect_from_wall(
                x=x_hit,
                v=v_hit,
                radius=dynamic_radius,
                contact=contact,
                restitution=wall_restitution,
                device=device,
                dtype=dtype,
            )
            remaining = torch.clamp(dt - contact_time, min=0.0)
            x, v = _free_flight(x=x_hit, v=v_hit, dt=remaining, gravity=gravity)
            events.append(
                {
                    "event_type": "bouncy_wall_contact",
                    "frame_index": int(frames[frame_offset + 1]),
                    "phase": "free_flight_to_wall_bounce",
                    "plane_id": plane_id,
                    "support_object_id": str(contact["plane"].get("object_id")),
                }
            )
            continue
        plane = friction_swr._torch_plane_from_runtime(contact, device=device, dtype=dtype)
        normal = plane["normal"] / torch.clamp(torch.linalg.norm(plane["normal"]), min=start_position.new_tensor(1e-12))
        x, v, contact_started, _hit = analytic_swr_common.advance_sphere_plane_interval_torch(
            x=x,
            v=v,
            dt=dt,
            radius=dynamic_radius,
            plane_point=plane["surface_point"],
            normal=normal,
            gravity=gravity,
            friction=friction_for_support(contact),
            restitution=0.0,
            contact_started=False,
        )
        active_support = contact if contact_started else None
        if contact_started:
            events.append(
                {
                    "event_type": "bounded_plane_contact_transition",
                    "frame_index": int(frames[frame_offset + 1]),
                    "phase": "free_flight_to_support_contact",
                    "plane_id": plane_id,
                    "support_object_id": str(contact["plane"].get("object_id")),
                }
            )
    return torch.stack(positions, dim=0), torch.stack(velocities, dim=0), events


def _segment_result_trajectories(
    *,
    target: dict[str, list[dict[str, Any]]],
    agent_id: str,
    static_ids: list[str],
    frames: list[int],
    predicted: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    return friction_swr._result_trajectories(
        target=target,
        agent_id=agent_id,
        static_ids=static_ids,
        frames=frames,
        predicted=predicted,
    )


def _prepare_one_segment(
    *,
    manifest: dict[str, Any],
    target_all: dict[str, list[dict[str, Any]]],
    segment: dict[str, str],
    global_ground_plane: dict[str, Any],
    global_ground_diagnostics: dict[str, Any],
    output_dir: Path,
    fix_radius: bool,
) -> dict[str, Any] | None:
    agent_id = segment["agent_object_id"]
    wall_id = segment["wall_object_id"]
    patient_id = segment.get("patient_object_id")
    static_ids = [value for value in [patient_id, wall_id] if value and value in target_all]
    if agent_id not in target_all or wall_id not in target_all:
        return None
    target = {object_id: target_all[object_id] for object_id in [agent_id, *static_ids] if object_id in target_all}
    records = _segment_records(target=target, object_id=agent_id)
    if len(records) < 3:
        return {
            "segment": segment["segment"],
            "status": "skipped",
            "reason": f"too few agent trajectory frames ({len(records)})",
            "agent_object_id": agent_id,
            "wall_object_id": wall_id,
            "patient_object_id": patient_id,
        }
    frames = [int(record["frame_index"]) for record in records]
    if any(next_frame <= frame for frame, next_frame in zip(frames, frames[1:])):
        raise ValueError(f"{segment['segment']} has non-increasing frames")
    target_positions_np = np.asarray([record["position"] for record in records], dtype=np.float64)
    radius_init = friction_swr._initial_radius_by_object(
        manifest=manifest,
        object_ids=[agent_id],
    )[agent_id]
    static_meshes = friction_swr._static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
    gravity_direction_np = friction_swr._gravity_direction_blender_world(manifest)
    bounded_planes_payload = friction_swr._decompose_static_meshes_into_bounded_planes(
        static_meshes=static_meshes,
        gravity_direction=gravity_direction_np,
    )
    bounded_planes_debug_path = output_dir / f"{segment['segment']}_bounded_static_planes_debug.json"
    _write_json(bounded_planes_debug_path, bounded_planes_payload)
    runtime_payload = dict(bounded_planes_payload)
    runtime_payload["_static_meshes_runtime"] = static_meshes
    all_planes = friction_swr._bounded_planes_runtime(runtime_payload)
    _filtered_planes, plane_filter = friction_swr._filter_bounded_planes_by_target_swept_path(
        planes=all_planes,
        target_positions=target_positions_np,
        radius_init=radius_init,
        fix_radius=fix_radius,
    )
    # Bouncy-wall scenes need both support and wall contacts. The swept-path filter can
    # reject the support plane when the recovered sphere center is slightly outside the
    # bounded polygon, which turns the rollout into free flight. Keep all decomposed
    # planes for contact detection; use the filter only as diagnostics.
    planes = [*all_planes, global_ground_plane]
    plane_filter["used_for_contact_detection"] = False
    plane_filter["contact_detection_plane_count"] = int(len(planes))
    plane_filter["contact_detection_policy"] = "all_bounded_planes"
    wall_ids = _wall_plane_ids(planes, wall_id)
    if not wall_ids:
        wall_ids = _wall_plane_ids(all_planes, wall_id)
        planes = all_planes
    if not wall_ids:
        raise ValueError(f"{segment['segment']} has no wall bounded planes for {wall_id}")
    support_friction_ids = _support_friction_plane_ids(
        target_positions=target_positions_np,
        radius=radius_init,
        planes=planes,
        wall_plane_ids=wall_ids,
    )
    wall_initialization = _wall_event_initialization(
        records=records,
        radius=radius_init,
        wall_planes=[plane for plane in planes if str(plane.get("plane_id")) in wall_ids],
    )
    v0_init = np.asarray(wall_initialization["velocity"], dtype=np.float64)
    restitution_init = float(wall_initialization["restitution"])
    friction_init = friction_swr._fit_initial_friction(records, PHYSIONPP_PHYSICS_DT_SEC)
    return {
        "segment": segment["segment"],
        "agent_object_id": agent_id,
        "wall_object_id": wall_id,
        "patient_object_id": patient_id,
        "static_object_ids": static_ids,
        "frames": frames,
        "target": target,
        "target_positions_np": target_positions_np,
        "radius_init": float(radius_init),
        "bounded_planes": planes,
        "bounded_planes_payload": bounded_planes_payload,
        "bounded_planes_debug_path": str(bounded_planes_debug_path),
        "plane_filter": plane_filter,
        "wall_plane_ids": wall_ids,
        "support_friction_plane_ids": support_friction_ids,
        "v0_init": v0_init,
        "friction_init": float(friction_init),
        "restitution_init": restitution_init,
        "wall_initialization": wall_initialization,
        "global_ground_diagnostics": global_ground_diagnostics,
    }


def _optimize_segments_joint(
    *,
    prepared_segments: list[dict[str, Any]],
    gravity_direction_np: np.ndarray,
    lr: float,
    curriculum_prefix_step_frames: int,
    steps: int,
    patience: int,
    fix_radius: bool,
    optimize_gravity: bool,
) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float64
    longest_segment = max(prepared_segments, key=lambda item: len(item["frames"]))
    radius_init = float(np.median([float(item["radius_init"]) for item in prepared_segments]))
    friction_init = float(longest_segment["friction_init"])
    restitution_candidates = [
        float(item["restitution_init"])
        for item in prepared_segments
        if item["wall_initialization"]["wall_contact_frame_index"] is not None
    ]
    restitution_init = float(np.median(restitution_candidates)) if restitution_candidates else 0.8
    radius_init_tensor = torch.tensor(radius_init, dtype=dtype, device=device)
    targets = [torch.tensor(item["target_positions_np"], dtype=dtype, device=device) for item in prepared_segments]
    reference_segment = next(
        (item for item in prepared_segments if str(item.get("segment")) == "seg1"),
        prepared_segments[0],
    )
    shared_speed_init = max(float(np.linalg.norm(reference_segment["v0_init"])), 1e-4)
    shared_speed_raw = torch.tensor(
        math.log(math.expm1(shared_speed_init)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    direction_angles = []
    for item in prepared_segments:
        initial_direction = np.asarray(item["v0_init"], dtype=np.float64)
        if float(np.linalg.norm(initial_direction)) < 1e-6:
            initial_direction = item["target_positions_np"][-1] - item["target_positions_np"][0]
        if float(np.linalg.norm(initial_direction)) < 1e-6:
            initial_direction = np.asarray(reference_segment["v0_init"], dtype=np.float64)
        direction_angles.append(
            torch.tensor(
                [
                    math.atan2(float(initial_direction[1]), float(initial_direction[0])),
                    math.atan2(
                        float(initial_direction[2]),
                        math.hypot(float(initial_direction[0]), float(initial_direction[1])),
                    ),
                ],
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
    friction_raw = torch.tensor(
        _raw_from_unit_interval(friction_init, MIN_FRICTION, MAX_FRICTION),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    restitution_raw = torch.tensor(
        _raw_from_unit_interval(restitution_init, MIN_RESTITUTION, MAX_RESTITUTION),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    radius_raw = None if fix_radius else torch.zeros((), dtype=dtype, device=device, requires_grad=True)
    gravity_scale_raw = (
        torch.tensor(
            _raw_from_unit_interval(1.0, MIN_GRAVITY_SCALE, MAX_GRAVITY_SCALE),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        if optimize_gravity
        else None
    )
    parameters = [shared_speed_raw, *direction_angles, friction_raw, restitution_raw]
    if radius_raw is not None:
        parameters.append(radius_raw)
    if gravity_scale_raw is not None:
        parameters.append(gravity_scale_raw)
    gravity_direction = torch.tensor(gravity_direction_np, dtype=dtype, device=device)

    def rollout():
        shared_speed = torch.nn.functional.softplus(shared_speed_raw)
        velocity_vectors = []
        for angles in direction_angles:
            azimuth, elevation = angles.unbind()
            velocity_vectors.append(
                shared_speed
                * torch.stack(
                    [
                        torch.cos(elevation) * torch.cos(azimuth),
                        torch.cos(elevation) * torch.sin(azimuth),
                        torch.sin(elevation),
                    ]
                )
            )
        friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        restitution = MIN_RESTITUTION + (MAX_RESTITUTION - MIN_RESTITUTION) * torch.sigmoid(restitution_raw)
        radius, radius_log_scale = friction_swr._radius_from_raw(
            radius_init_tensor=radius_init_tensor,
            radius_raw=radius_raw,
        )
        gravity_scale = (
            radius_init_tensor.new_tensor(1.0)
            if gravity_scale_raw is None
            else MIN_GRAVITY_SCALE
            + (MAX_GRAVITY_SCALE - MIN_GRAVITY_SCALE) * torch.sigmoid(gravity_scale_raw)
        )
        gravity_magnitude = radius_init_tensor.new_tensor(GRAVITY_M_PER_S2) * gravity_scale
        segment_rollouts = []
        for item, target_positions, velocity in zip(prepared_segments, targets, velocity_vectors):
            predicted, predicted_velocities, events = _rollout_bouncy_wall_bounded_planes(
                frames=item["frames"],
                physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                start_position=target_positions[0],
                velocity=velocity,
                support_friction=friction,
                wall_restitution=restitution,
                dynamic_radius=radius,
                bounded_planes=item["bounded_planes"],
                wall_plane_ids=item["wall_plane_ids"],
                support_friction_plane_ids=item["support_friction_plane_ids"],
                gravity_direction=gravity_direction,
                gravity_magnitude=gravity_magnitude,
            )
            segment_rollouts.append(
                {
                    "predicted": predicted,
                    "predicted_velocities": predicted_velocities,
                    "events": events,
                }
            )
        return (
            segment_rollouts,
            friction,
            restitution,
            radius,
            radius_log_scale,
            gravity_scale,
            gravity_magnitude,
            shared_speed,
            velocity_vectors,
        )

    observed_frames = max(len(item["frames"]) for item in prepared_segments)

    def segment_prefix_count(segment_frames: int, joint_end: int) -> int:
        if joint_end >= observed_frames:
            return int(segment_frames)
        return min(int(segment_frames), max(2, int(math.ceil(segment_frames * joint_end / observed_frames))))

    def evaluate(train_end: int, validation_end: int) -> dict[str, Any]:
        (
            rollouts,
            _friction,
            restitution,
            _radius,
            radius_log_scale,
            gravity_scale,
            _gravity_magnitude,
            _shared_speed,
            _velocity_vectors,
        ) = rollout()
        prefix_values = []
        validation_values = []
        for target_positions, segment_rollout in zip(targets, rollouts):
            train_count = segment_prefix_count(len(target_positions), train_end)
            validation_count = segment_prefix_count(len(target_positions), validation_end)
            predicted = segment_rollout["predicted"]
            prefix_values.append(
                torch.sqrt(torch.mean(torch.sum((predicted[:train_count] - target_positions[:train_count]) ** 2, dim=1)))
            )
            validation_values.append(
                torch.sqrt(
                    torch.mean(
                        torch.sum(
                            (predicted[:validation_count] - target_positions[:validation_count]) ** 2,
                            dim=1,
                        )
                    )
                )
            )
        prefix_rmse = torch.mean(torch.stack(prefix_values))
        validation_rmse = torch.mean(torch.stack(validation_values))
        loss = (
            prefix_rmse
            + targets[0].new_tensor(RADIUS_PRIOR_WEIGHT) * radius_log_scale * radius_log_scale
            + targets[0].new_tensor(GRAVITY_SCALE_PRIOR_WEIGHT) * (gravity_scale - 1.0) ** 2
            + targets[0].new_tensor(RESTITUTION_PRIOR_WEIGHT) * (restitution - 0.8) ** 2
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("joint bouncy-wall optimization produced a non-finite loss")
        return {
            "loss": loss,
            "selection_metric": float(validation_rmse.detach().cpu()),
            "prefix_rmse": float(prefix_rmse.detach().cpu()),
        }

    schedule = analytic_swr_common.optimize_adam_prefix_curriculum(
        parameters=parameters,
        evaluate=evaluate,
        observed_frames=observed_frames,
        prefix_step_frames=int(curriculum_prefix_step_frames),
        steps_per_prefix=int(steps),
        patience=int(patience),
        lr=float(lr),
    )
    (
        rollouts,
        friction,
        restitution,
        radius,
        radius_log_scale,
        gravity_scale,
        gravity_magnitude,
        shared_speed,
        velocity_vectors,
    ) = rollout()
    segment_results = []
    for item, target_positions, velocity, segment_rollout in zip(
        prepared_segments,
        targets,
        velocity_vectors,
        rollouts,
    ):
        rmse = torch.sqrt(torch.mean(torch.sum((segment_rollout["predicted"] - target_positions) ** 2, dim=1)))
        segment_results.append(
            {
                "segment": item["segment"],
                "rmse": float(rmse.detach().cpu()),
                "velocity": velocity.detach().cpu().numpy().astype(float),
                "predicted": segment_rollout["predicted"].detach().cpu().numpy().astype(float),
                "terminal_velocity": (
                    segment_rollout["predicted_velocities"][-1].detach().cpu().numpy().astype(float)
                ),
                "events": segment_rollout["events"],
            }
        )
    return {
        "loss": float(np.mean([item["rmse"] for item in segment_results])),
        "support_friction": float(friction.detach().cpu()),
        "wall_restitution": float(restitution.detach().cpu()),
        "radius": float(radius.detach().cpu()),
        "radius_init": radius_init,
        "radius_log_scale": float(radius_log_scale.detach().cpu()),
        "radius_fixed": bool(fix_radius),
        "gravity_scale": float(gravity_scale.detach().cpu()),
        "gravity_m_per_s2": float(gravity_magnitude.detach().cpu()),
        "gravity_fixed": not bool(optimize_gravity),
        "shared_initial_speed_m_per_s": float(shared_speed.detach().cpu()),
        "curriculum_schedule": schedule,
        "segments": segment_results,
        "initialization": {
            "support_sliding_friction": friction_init,
            "wall_restitution": restitution_init,
            "radius_init_m": radius_init,
            "shared_initial_speed_m_per_s": shared_speed_init,
            "shared_initial_speed_source_segment": str(reference_segment["segment"]),
        },
    }


def _joint_segment_result(
    *,
    prepared: dict[str, Any],
    optimized: dict[str, Any],
    shared: dict[str, Any],
    lr: float,
    curriculum_prefix_step_frames: int,
    steps: int,
    patience: int,
    optimize_gravity: bool,
) -> dict[str, Any]:
    agent_id = prepared["agent_object_id"]
    predicted = optimized["predicted"]
    simulated = _segment_result_trajectories(
        target=prepared["target"],
        agent_id=agent_id,
        static_ids=prepared["static_object_ids"],
        frames=prepared["frames"],
        predicted=predicted,
    )
    wall_initialization = prepared["wall_initialization"]
    return {
        "segment": prepared["segment"],
        "status": "ok",
        "agent_object_id": agent_id,
        "wall_object_id": prepared["wall_object_id"],
        "patient_object_id": prepared["patient_object_id"],
        "static_object_ids": prepared["static_object_ids"],
        "frame_range": [int(prepared["frames"][0]), int(prepared["frames"][-1])],
        "frame_count": int(len(prepared["frames"])),
        "dynamic_object_rmse_m": float(optimized["rmse"]),
        "target_trajectories": prepared["target"],
        "physics_rollout": {
            "simulator": "swr_backend.wall_bounce_sphere",
            "contact_proxy_policy": "dynamic_shared_3d_sphere_bounded_static_planes_with_wall_restitution",
            "simulated_trajectories": simulated,
            "sphere_radius_3d_by_object": {agent_id: float(shared["radius"])},
            "sphere_radius_init_3d_by_object": {agent_id: float(prepared["radius_init"])},
            "bounded_static_planes": {
                "status": prepared["bounded_planes_payload"].get("status"),
                "debug_path": prepared["bounded_planes_debug_path"],
                "method": prepared["bounded_planes_payload"].get("method"),
                "plane_count": prepared["bounded_planes_payload"].get("plane_count"),
                "candidate_filter": prepared["plane_filter"],
            },
            "global_ground_plane": prepared["global_ground_diagnostics"],
            "contact_plane_geometry": [
                _serializable_runtime_plane(plane) for plane in prepared["bounded_planes"]
            ],
            "wall_plane_ids": sorted(prepared["wall_plane_ids"]),
            "support_friction_plane_ids": (
                sorted(prepared["support_friction_plane_ids"])
                if prepared["support_friction_plane_ids"]
                else None
            ),
            "contact_events": optimized["events"],
            "terminal_state": {
                "frame_index": int(prepared["frames"][-1]),
                "position_blender_world_m": predicted[-1].astype(float).tolist(),
                "velocity_blender_world_m_per_s": optimized["terminal_velocity"].astype(float).tolist(),
            },
        },
        "alignment_optimization": {
            "strategy": "physionpp_bouncy_wall_joint_two_segment_dynamic_3d_sphere",
            "parameter_scope": {
                "shared": [
                    "initial_speed_magnitude_m_per_s",
                    "support_sliding_friction",
                    "wall_restitution",
                    "optimized_radius_m",
                    "gravity_scale",
                ],
                "per_segment": ["initial_velocity_direction_blender_world"],
            },
            "optimizer": {
                "method": (
                    "torch_adam_joint_prefix_curriculum"
                    if int(curriculum_prefix_step_frames) > 0
                    else "torch_adam_joint_full_trajectory"
                ),
                "lr": float(lr),
                "curriculum_prefix_step_frames": int(curriculum_prefix_step_frames),
                "steps": int(steps),
                "early_stop_patience": int(patience),
                "fix_radius": bool(shared["radius_fixed"]),
                "optimize_gravity": bool(optimize_gravity),
                "physics_dt_sec": PHYSIONPP_PHYSICS_DT_SEC,
                "curriculum_schedule": shared["curriculum_schedule"],
            },
            "best_parameters": {
                agent_id: {
                    "initial_velocity_blender_world_m_per_s": optimized["velocity"].tolist(),
                    "initial_speed_magnitude_m_per_s": float(shared["shared_initial_speed_m_per_s"]),
                    "initial_velocity_direction_blender_world": (
                        optimized["velocity"] / max(float(np.linalg.norm(optimized["velocity"])), 1e-12)
                    ).tolist(),
                    "support_sliding_friction": float(shared["support_friction"]),
                    "wall_restitution": float(shared["wall_restitution"]),
                    "optimized_radius_m": float(shared["radius"]),
                    "radius_init_m": float(shared["radius_init"]),
                    "radius_scale": float(shared["radius"] / max(shared["radius_init"], 1e-12)),
                    "radius_fixed": bool(shared["radius_fixed"]),
                    "gravity_scale": float(shared["gravity_scale"]),
                    "gravity_m_per_s2": float(shared["gravity_m_per_s2"]),
                    "gravity_fixed": bool(shared["gravity_fixed"]),
                }
            },
            "initialization": {
                "initial_velocity_blender_world_m_per_s": prepared["v0_init"].astype(float).tolist(),
                "initial_speed_magnitude_m_per_s": float(shared["initialization"]["shared_initial_speed_m_per_s"]),
                "initial_velocity_direction_blender_world": (
                    prepared["v0_init"] / max(float(np.linalg.norm(prepared["v0_init"])), 1e-12)
                ).astype(float).tolist(),
                "support_sliding_friction": float(shared["initialization"]["support_sliding_friction"]),
                "wall_restitution": float(shared["initialization"]["wall_restitution"]),
                "radius_init_m": float(shared["initialization"]["radius_init_m"]),
                "velocity_source": wall_initialization["source"],
                "velocity_trajectory_frame_count": wall_initialization["trajectory_frame_count"],
                "observed_wall_contact_frame_index": wall_initialization["wall_contact_frame_index"],
                "observed_wall_plane_id": wall_initialization["wall_plane_id"],
            },
        },
    }


def run_physionpp_bouncy_wall_sphere_sysid(
    *,
    world_modeling_dir: Path,
    output_path: Path,
    output_dir: Path,
    lr: float,
    curriculum_prefix_step_frames: int,
    steps: int,
    patience: int,
    fix_radius: bool,
    optimize_gravity: bool,
) -> dict[str, Any]:
    manifest = sysid_common.build_manifest_from_world_modeling(world_modeling_dir)
    object_plan = manifest["object_plan"]
    target_all = sysid_common._target_records_from_swr(manifest)
    segments = _role_assignments(world_modeling_dir, object_plan)
    gravity_direction_np = friction_swr._gravity_direction_blender_world(manifest)
    global_ground_plane, global_ground_diagnostics = _global_ground_plane_from_pose_correction(
        manifest=manifest,
        target_all=target_all,
        segments=segments,
    )
    prepared_segments = []
    segment_results = []
    for segment in segments:
        prepared = _prepare_one_segment(
            manifest=manifest,
            target_all=target_all,
            segment=segment,
            global_ground_plane=global_ground_plane,
            global_ground_diagnostics=global_ground_diagnostics,
            output_dir=output_dir,
            fix_radius=fix_radius,
        )
        if prepared is None:
            continue
        if prepared.get("status") == "skipped":
            segment_results.append(prepared)
            continue
        prepared_segments.append(prepared)
    if not prepared_segments:
        raise ValueError("no valid bouncy_wall segments were prepared")
    joint_result = _optimize_segments_joint(
        prepared_segments=prepared_segments,
        gravity_direction_np=gravity_direction_np,
        lr=lr,
        curriculum_prefix_step_frames=curriculum_prefix_step_frames,
        steps=steps,
        patience=patience,
        fix_radius=fix_radius,
        optimize_gravity=optimize_gravity,
    )
    optimized_by_segment = {
        str(item["segment"]): item for item in joint_result["segments"]
    }
    for prepared in prepared_segments:
        optimized = optimized_by_segment[str(prepared["segment"])]
        segment_results.append(
            _joint_segment_result(
                prepared=prepared,
                optimized=optimized,
                shared=joint_result,
                lr=lr,
                curriculum_prefix_step_frames=curriculum_prefix_step_frames,
                steps=steps,
                patience=patience,
                optimize_gravity=optimize_gravity,
            )
        )
    segment_order = {str(segment["segment"]): index for index, segment in enumerate(segments)}
    segment_results.sort(key=lambda item: segment_order.get(str(item.get("segment")), len(segment_order)))
    ok_segments = [item for item in segment_results if item.get("status") == "ok"]
    if not ok_segments:
        raise ValueError("no bouncy_wall segments were optimized")
    rmse_values = [float(item["dynamic_object_rmse_m"]) for item in ok_segments]
    payload: dict[str, Any] = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.wall_bounce_sphere",
        "mode": "trajectory_informed_physics_alignment",
        "message": "Physion++ bouncy wall joint two-segment sphere SWR completed",
        "source_world_modeling_dir": str(world_modeling_dir),
        "segment_count": int(len(segment_results)),
        "optimized_segment_count": int(len(ok_segments)),
        "global_ground_plane": global_ground_diagnostics,
        "dynamic_object_rmse_m": float(np.mean(rmse_values)),
        "dynamic_object_rmse_max_m": float(np.max(rmse_values)),
        "joint_alignment_optimization": {
            "strategy": "swr_fit.two_segment_shared_speed",
            "loss": float(joint_result["loss"]),
            "parameter_scope": {
                "shared": [
                    "initial_speed_magnitude_m_per_s",
                    "support_sliding_friction",
                    "wall_restitution",
                    "optimized_radius_m",
                    "gravity_scale",
                ],
                "per_segment": ["initial_velocity_direction_blender_world"],
            },
            "best_shared_parameters": {
                "initial_speed_magnitude_m_per_s": float(joint_result["shared_initial_speed_m_per_s"]),
                "support_sliding_friction": float(joint_result["support_friction"]),
                "wall_restitution": float(joint_result["wall_restitution"]),
                "optimized_radius_m": float(joint_result["radius"]),
                "radius_init_m": float(joint_result["radius_init"]),
                "radius_fixed": bool(joint_result["radius_fixed"]),
                "gravity_scale": float(joint_result["gravity_scale"]),
                "gravity_m_per_s2": float(joint_result["gravity_m_per_s2"]),
                "gravity_fixed": bool(joint_result["gravity_fixed"]),
            },
        },
        "segments": segment_results,
    }
    _write_json(output_path, payload)
    _write_json(
        output_dir / "summary.json",
        {
            "status": "ok",
            "backend": payload["backend"],
            "source_world_modeling_dir": str(world_modeling_dir),
            "dynamic_object_rmse_m": payload["dynamic_object_rmse_m"],
            "dynamic_object_rmse_max_m": payload["dynamic_object_rmse_max_m"],
            "segments": [
                {
                    "segment": item.get("segment"),
                    "status": item.get("status"),
                    "agent_object_id": item.get("agent_object_id"),
                    "wall_object_id": item.get("wall_object_id"),
                    "patient_object_id": item.get("patient_object_id"),
                    "frame_range": item.get("frame_range"),
                    "dynamic_object_rmse_m": item.get("dynamic_object_rmse_m"),
                    "parameters": (
                        (item.get("alignment_optimization") or {}).get("best_parameters") or {}
                    ).get(str(item.get("agent_object_id") or ""), {}),
                }
                for item in segment_results
            ],
        },
    )
    return payload


def _run_learning_rate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(max(int(payload["worker_threads"]), 1))
    output_dir = Path(payload["output_dir"])
    output_path = output_dir / "physics_alignment.json"
    result = run_physionpp_bouncy_wall_sphere_sysid(
        world_modeling_dir=Path(payload["world_modeling_dir"]),
        output_path=output_path,
        output_dir=output_dir,
        lr=float(payload["lr"]),
        curriculum_prefix_step_frames=int(payload["curriculum_prefix_step_frames"]),
        steps=int(payload["steps"]),
        patience=int(payload["patience"]),
        fix_radius=bool(payload["fix_radius"]),
        optimize_gravity=bool(payload["optimize_gravity"]),
    )
    return {
        "lr": float(payload["lr"]),
        "dynamic_object_rmse_m": float(result["dynamic_object_rmse_m"]),
        "dynamic_object_rmse_max_m": float(result["dynamic_object_rmse_max_m"]),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physion++ bouncy_wall analytic sphere SWR.")
    parser.add_argument("--world-modeling-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lrs", default=",".join(str(value) for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--workers", type=int, default=len(DEFAULT_LEARNING_RATES))
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument(
        "--curriculum-prefix-step-frames",
        type=int,
        default=0,
        help="Cumulative trajectory prefix interval; use 0 for full-trajectory optimization only.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--fix-radius", action="store_true")
    parser.add_argument("--fix-gravity", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_dir = Path(args.output_dir) if args.output_dir else output_path.parent
    learning_rates = [float(value.strip()) for value in str(args.lrs).split(",") if value.strip()]
    if not learning_rates:
        raise ValueError("--lrs must contain at least one learning rate")
    trial_root = output_dir / "learning_rate_trials"
    payloads = [
        {
            "world_modeling_dir": str(Path(args.world_modeling_dir)),
            "output_dir": str(trial_root / f"lr_{lr:g}"),
            "lr": lr,
            "curriculum_prefix_step_frames": int(args.curriculum_prefix_step_frames),
            "steps": int(args.steps),
            "patience": int(args.patience),
            "worker_threads": int(args.worker_threads),
            "fix_radius": bool(args.fix_radius),
            "optimize_gravity": not bool(args.fix_gravity),
        }
        for lr in learning_rates
    ]
    trials, worker_count = analytic_swr_common.run_parallel_trials(
        payloads=payloads,
        worker=_run_learning_rate_trial,
        max_workers=int(args.workers),
    )
    selected = min(trials, key=lambda item: float(item["dynamic_object_rmse_m"]))
    result = _load_json(Path(selected["output_path"]))
    if not isinstance(result, dict):
        raise ValueError(f"invalid selected result: {selected['output_path']}")
    result["optimizer_multi_learning_rate"] = {
        "learning_rates": learning_rates,
        "worker_count": int(worker_count),
        "selected_learning_rate": float(selected["lr"]),
        "trials": sorted(trials, key=lambda item: float(item["lr"])),
    }
    _write_json(output_path, result)
    _write_json(
        output_dir / "summary.json",
        {
            "status": result.get("status"),
            "backend": result.get("backend"),
            "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
            "dynamic_object_rmse_max_m": result.get("dynamic_object_rmse_max_m"),
            "optimizer_multi_learning_rate": result.get("optimizer_multi_learning_rate"),
            "segments": [
                {
                    "segment": item.get("segment"),
                    "status": item.get("status"),
                    "agent_object_id": item.get("agent_object_id"),
                    "wall_object_id": item.get("wall_object_id"),
                    "patient_object_id": item.get("patient_object_id"),
                    "frame_range": item.get("frame_range"),
                    "dynamic_object_rmse_m": item.get("dynamic_object_rmse_m"),
                }
                for item in result.get("segments", [])
                if isinstance(item, dict)
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "backend": result.get("backend"),
                "output": str(output_path),
                "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
                "dynamic_object_rmse_max_m": result.get("dynamic_object_rmse_max_m"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
