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

from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr  # noqa: E402


DEFAULT_MAX_FUTURE_FRAMES = 300
DEFAULT_STATIONARY_SPEED_M_PER_S = 0.01
DEFAULT_STATIONARY_WINDOW_FRAMES = 10


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _segment2(fit: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in fit.get("segments") or []
        if isinstance(item, dict) and item.get("status") == "ok" and str(item.get("segment")) == "seg2"
    ]
    if len(matches) != 1:
        raise ValueError(f"friction-collision future rollout requires one valid seg2 fit, found {len(matches)}")
    return matches[0]


def _segment_initial_position(segment: dict[str, Any], object_id: str) -> np.ndarray:
    frame_range = segment.get("frame_range")
    if not isinstance(frame_range, list) or len(frame_range) != 2:
        raise ValueError("seg2 has no valid frame_range")
    first_frame = int(frame_range[0])
    records = (segment.get("target_trajectories") or {}).get(object_id) or []
    record = next(
        (
            item
            for item in records
            if isinstance(item, dict) and int(item.get("frame_index", -1)) == first_frame
        ),
        None,
    )
    values = record.get("position_blender_world_m") if isinstance(record, dict) else None
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"seg2 has no initial target position for {object_id} at frame {first_frame}")
    return np.asarray(values, dtype=np.float64).reshape(3)


def _segment_initial_velocity(segment: dict[str, Any], object_id: str) -> np.ndarray:
    values = (segment.get("optimized_initial_velocity_blender_world_m_per_s") or {}).get(object_id)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"seg2 has no optimized initial velocity for {object_id}")
    return np.asarray(values, dtype=np.float64).reshape(3)


def _boundary_pose(segment: dict[str, Any], object_id: str) -> np.ndarray:
    state = (segment.get("observed_boundary_state") or {}).get(object_id) or {}
    values = state.get("pose_blender_world_4x4")
    matrix = np.asarray(values, dtype=np.float64) if values is not None else np.empty((0, 0))
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"seg2 observed boundary has no valid pose_blender_world_4x4 for {object_id}")
    return matrix


def _fixed_collision_obb(
    *,
    fit: dict[str, Any],
    segment: dict[str, Any],
    object_id: str,
) -> dict[str, np.ndarray]:
    geometry = (fit.get("collision_geometry_by_object") or {}).get(object_id) or {}
    if geometry.get("type") != "oriented_box":
        raise ValueError(f"fit has no oriented-box collision geometry for {object_id}")
    local_center = np.asarray(geometry.get("local_center_m"), dtype=np.float64)
    half_extents = np.asarray(geometry.get("half_extents_m"), dtype=np.float64)
    if local_center.shape != (3,) or half_extents.shape != (3,):
        raise ValueError(f"invalid oriented-box dimensions for {object_id}")
    if not np.all(np.isfinite(local_center)) or not np.all(np.isfinite(half_extents)):
        raise ValueError(f"non-finite oriented-box dimensions for {object_id}")
    if np.any(half_extents <= 0.0):
        raise ValueError(f"non-positive oriented-box half extents for {object_id}")

    rotation = _boundary_pose(segment, object_id)[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if orthogonality_error > 1e-3 or float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"invalid terminal rotation for {object_id}: error={orthogonality_error}")
    left, _singular_values, right = np.linalg.svd(rotation)
    rotation = left @ right
    return {
        "half_extents": half_extents,
        "rotation": rotation,
        "world_center_offset": rotation @ local_center,
    }


def _support_plane_point(support: dict[str, Any]) -> np.ndarray:
    values = support.get("surface_point_blender_world_m")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("fit support plane has no fixed surface point")
    return np.asarray(values, dtype=np.float64).reshape(3)


def _rollout_role(
    *,
    frames: list[int],
    start_position: np.ndarray,
    initial_velocity: np.ndarray,
    friction: float,
    radius: float,
    plane_point: np.ndarray,
    up: np.ndarray,
    gravity_direction: np.ndarray,
    gravity_magnitude: float,
    physics_dt_sec: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predicted, events = friction_swr._rollout_plane_analytic(
        frames=frames,
        physics_dt_sec=physics_dt_sec,
        start_position=torch.tensor(start_position, dtype=torch.float64),
        velocity=torch.tensor(initial_velocity, dtype=torch.float64),
        friction=torch.tensor(float(friction), dtype=torch.float64),
        dynamic_radius=torch.tensor(float(radius), dtype=torch.float64),
        plane_point=torch.tensor(plane_point, dtype=torch.float64),
        plane_normal=torch.tensor(up, dtype=torch.float64),
        gravity_direction=torch.tensor(gravity_direction, dtype=torch.float64),
        gravity_magnitude=torch.tensor(float(gravity_magnitude), dtype=torch.float64),
    )
    return predicted.detach().cpu().numpy().astype(float), events


def _frame_speeds(positions: np.ndarray, physics_dt_sec: float) -> np.ndarray:
    speeds = np.full((len(positions),), np.nan, dtype=np.float64)
    if len(positions) > 1:
        speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) / max(float(physics_dt_sec), 1e-12)
    return speeds


def _both_stationary_offset(
    *,
    agent_speeds: np.ndarray,
    patient_speeds: np.ndarray,
    observed_end_offset: int,
    threshold_m_per_s: float,
    window_frames: int,
) -> int | None:
    if window_frames <= 0:
        return None
    consecutive = 0
    for offset in range(max(int(observed_end_offset) + 1, 1), len(agent_speeds)):
        stationary = (
            math.isfinite(float(agent_speeds[offset]))
            and math.isfinite(float(patient_speeds[offset]))
            and float(agent_speeds[offset]) <= float(threshold_m_per_s)
            and float(patient_speeds[offset]) <= float(threshold_m_per_s)
        )
        if stationary:
            consecutive += 1
            if consecutive >= int(window_frames):
                return offset
        else:
            consecutive = 0
    return None


def _obb_sat_axes(agent_rotation: np.ndarray, patient_rotation: np.ndarray) -> list[np.ndarray]:
    axes = [agent_rotation[:, index] for index in range(3)]
    axes.extend(patient_rotation[:, index] for index in range(3))
    for agent_index in range(3):
        for patient_index in range(3):
            axis = np.cross(agent_rotation[:, agent_index], patient_rotation[:, patient_index])
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axes.append(axis / norm)
    return axes


def _obb_projection_limits(
    *,
    axes: list[np.ndarray],
    agent_rotation: np.ndarray,
    agent_half_extents: np.ndarray,
    patient_rotation: np.ndarray,
    patient_half_extents: np.ndarray,
) -> np.ndarray:
    limits = []
    for axis in axes:
        agent_radius = float(np.dot(agent_half_extents, np.abs(agent_rotation.T @ axis)))
        patient_radius = float(np.dot(patient_half_extents, np.abs(patient_rotation.T @ axis)))
        limits.append(agent_radius + patient_radius)
    return np.asarray(limits, dtype=np.float64)


def _linear_projection_interval(constant: float, slope: float, limit: float) -> tuple[float, float] | None:
    if abs(slope) <= 1e-15:
        return (-math.inf, math.inf) if abs(constant) <= limit else None
    first = (-limit - constant) / slope
    second = (limit - constant) / slope
    return (min(first, second), max(first, second))


def _relative_segment_first_obb_contact(
    *,
    relative_start: np.ndarray,
    relative_end: np.ndarray,
    axes: list[np.ndarray],
    projection_limits: np.ndarray,
) -> float | None:
    start = np.asarray(relative_start, dtype=np.float64)
    delta = np.asarray(relative_end, dtype=np.float64) - start
    lower = 0.0
    upper = 1.0
    for axis, limit in zip(axes, projection_limits):
        interval = _linear_projection_interval(
            float(np.dot(start, axis)),
            float(np.dot(delta, axis)),
            float(limit),
        )
        if interval is None:
            return None
        lower = max(lower, interval[0])
        upper = min(upper, interval[1])
        if lower > upper:
            return None
    return float(np.clip(lower, 0.0, 1.0))


def _relative_segment_closest_obb(
    *,
    relative_start: np.ndarray,
    relative_end: np.ndarray,
    axes: list[np.ndarray],
    projection_limits: np.ndarray,
) -> tuple[float, float, float]:
    start = np.asarray(relative_start, dtype=np.float64)
    delta = np.asarray(relative_end, dtype=np.float64) - start
    intercepts: list[float] = []
    slopes: list[float] = []
    for axis, limit in zip(axes, projection_limits):
        projection_start = float(np.dot(start, axis))
        projection_delta = float(np.dot(delta, axis))
        intercepts.extend([projection_start - float(limit), -projection_start - float(limit)])
        slopes.extend([projection_delta, -projection_delta])

    candidates = {0.0, 1.0}
    for first in range(len(intercepts)):
        for second in range(first + 1, len(intercepts)):
            slope_delta = slopes[first] - slopes[second]
            if abs(slope_delta) <= 1e-15:
                continue
            fraction = (intercepts[second] - intercepts[first]) / slope_delta
            if 0.0 <= fraction <= 1.0:
                candidates.add(float(fraction))

    def separation(fraction: float) -> float:
        return max(
            intercept + slope * fraction
            for intercept, slope in zip(intercepts, slopes)
        )

    best_fraction = min(candidates, key=separation)
    relative = start + best_fraction * delta
    return float(separation(best_fraction)), float(np.linalg.norm(relative)), float(best_fraction)


def _contact_scan(
    *,
    agent_positions: np.ndarray,
    patient_positions: np.ndarray,
    frames: list[int],
    observed_end_offset: int,
    final_offset: int,
    agent_obb: dict[str, np.ndarray],
    patient_obb: dict[str, np.ndarray],
    physics_dt_sec: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    closest = {
        "surface_distance_m": None,
        "distance_metric": "maximum_signed_separating_axis_gap",
        "center_distance_m": None,
        "frame_index": None,
        "time_after_observation_sec": None,
        "interval_fraction": None,
    }
    first_contact = None
    axes = _obb_sat_axes(agent_obb["rotation"], patient_obb["rotation"])
    projection_limits = _obb_projection_limits(
        axes=axes,
        agent_rotation=agent_obb["rotation"],
        agent_half_extents=agent_obb["half_extents"],
        patient_rotation=patient_obb["rotation"],
        patient_half_extents=patient_obb["half_extents"],
    )
    agent_centers = agent_positions + agent_obb["world_center_offset"]
    patient_centers = patient_positions + patient_obb["world_center_offset"]
    for end_offset in range(max(int(observed_end_offset) + 1, 1), int(final_offset) + 1):
        start_offset = end_offset - 1
        relative_start = agent_centers[start_offset] - patient_centers[start_offset]
        relative_end = agent_centers[end_offset] - patient_centers[end_offset]
        surface_distance, distance, closest_fraction = _relative_segment_closest_obb(
            relative_start=relative_start,
            relative_end=relative_end,
            axes=axes,
            projection_limits=projection_limits,
        )
        frame_value = float(frames[start_offset]) + closest_fraction * float(
            int(frames[end_offset]) - int(frames[start_offset])
        )
        if closest["surface_distance_m"] is None or surface_distance < float(closest["surface_distance_m"]):
            closest = {
                "surface_distance_m": float(surface_distance),
                "distance_metric": "maximum_signed_separating_axis_gap",
                "center_distance_m": float(distance),
                "frame_index": frame_value,
                "time_after_observation_sec": float(
                    (frame_value - float(frames[observed_end_offset])) * float(physics_dt_sec)
                ),
                "interval_fraction": float(closest_fraction),
            }
        contact_fraction = _relative_segment_first_obb_contact(
            relative_start=relative_start,
            relative_end=relative_end,
            axes=axes,
            projection_limits=projection_limits,
        )
        if contact_fraction is None:
            continue
        contact_frame = float(frames[start_offset]) + contact_fraction * float(
            int(frames[end_offset]) - int(frames[start_offset])
        )
        first_contact = {
            "frame_index": contact_frame,
            "time_after_observation_sec": float(
                (contact_frame - float(frames[observed_end_offset])) * float(physics_dt_sec)
            ),
            "interval_start_frame": int(frames[start_offset]),
            "interval_end_frame": int(frames[end_offset]),
            "interval_fraction": float(contact_fraction),
            "source": "continuous_3d_fixed_orientation_obb_sat",
        }
        break
    return first_contact, closest


def run_physionpp_friction_collision_future_rollout(
    *,
    fit: dict[str, Any],
    max_future_frames: int = DEFAULT_MAX_FUTURE_FRAMES,
    stationary_speed_m_per_s: float = DEFAULT_STATIONARY_SPEED_M_PER_S,
    stationary_window_frames: int = DEFAULT_STATIONARY_WINDOW_FRAMES,
) -> dict[str, Any]:
    segment = _segment2(fit)
    agent_id = str(segment["agent_object_id"])
    patient_id = str(segment["patient_object_id"])
    observed_first_frame = int(segment["frame_range"][0])
    observed_last_frame = int(segment["frame_range"][1])
    frame_start = observed_first_frame
    frames = list(range(observed_first_frame, observed_last_frame + int(max_future_frames) + 1))
    observed_end_offset = observed_last_frame - observed_first_frame

    support = fit.get("support_plane") or {}
    up = np.asarray(support["up_direction_blender_world"], dtype=np.float64).reshape(3)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    gravity_direction = np.asarray(support["gravity_direction_blender_world"], dtype=np.float64).reshape(3)
    gravity_direction /= max(float(np.linalg.norm(gravity_direction)), 1e-12)
    plane_point = _support_plane_point(support)
    radii = fit.get("shared_role_radii_m") or {}
    agent_radius = float(radii["agent"])
    patient_radius = float(radii["patient"])
    params = ((fit.get("alignment_optimization") or {}).get("best_parameters") or {})
    agent_friction = float(params["agent_ground_friction"])
    patient_friction = float(params["patient_ground_friction"])
    gravity_magnitude = float(params.get("gravity_m_per_s2") or friction_swr.GRAVITY_M_PER_S2)
    physics_dt_sec = float(
        ((fit.get("alignment_optimization") or {}).get("optimizer") or {}).get("physics_dt_sec")
        or friction_swr.PHYSIONPP_PHYSICS_DT_SEC
    )
    agent_initial_position = _segment_initial_position(segment, agent_id)
    patient_initial_position = _segment_initial_position(segment, patient_id)
    agent_initial_velocity = _segment_initial_velocity(segment, agent_id)
    patient_initial_velocity = _segment_initial_velocity(segment, patient_id)

    agent_positions, agent_plane_events = _rollout_role(
        frames=frames,
        start_position=agent_initial_position,
        initial_velocity=agent_initial_velocity,
        friction=agent_friction,
        radius=agent_radius,
        plane_point=plane_point,
        up=up,
        gravity_direction=gravity_direction,
        gravity_magnitude=gravity_magnitude,
        physics_dt_sec=physics_dt_sec,
    )
    patient_positions, patient_plane_events = _rollout_role(
        frames=frames,
        start_position=patient_initial_position,
        initial_velocity=patient_initial_velocity,
        friction=patient_friction,
        radius=patient_radius,
        plane_point=plane_point,
        up=up,
        gravity_direction=gravity_direction,
        gravity_magnitude=gravity_magnitude,
        physics_dt_sec=physics_dt_sec,
    )
    agent_speeds = _frame_speeds(agent_positions, physics_dt_sec)
    patient_speeds = _frame_speeds(patient_positions, physics_dt_sec)
    agent_obb = _fixed_collision_obb(fit=fit, segment=segment, object_id=agent_id)
    patient_obb = _fixed_collision_obb(fit=fit, segment=segment, object_id=patient_id)
    stationary_offset = _both_stationary_offset(
        agent_speeds=agent_speeds,
        patient_speeds=patient_speeds,
        observed_end_offset=observed_end_offset,
        threshold_m_per_s=stationary_speed_m_per_s,
        window_frames=stationary_window_frames,
    )
    scan_end_offset = len(frames) - 1 if stationary_offset is None else int(stationary_offset)
    first_contact, closest = _contact_scan(
        agent_positions=agent_positions,
        patient_positions=patient_positions,
        frames=frames,
        observed_end_offset=observed_end_offset,
        final_offset=scan_end_offset,
        agent_obb=agent_obb,
        patient_obb=patient_obb,
        physics_dt_sec=physics_dt_sec,
    )
    if first_contact is not None:
        final_offset = min(scan_end_offset, int(math.ceil(float(first_contact["frame_index"]))) - frame_start)
        stop_reason = "agent_patient_contact"
    elif stationary_offset is not None:
        final_offset = int(stationary_offset)
        stop_reason = "both_dynamic_objects_stationary"
    else:
        final_offset = scan_end_offset
        stop_reason = "max_future_frames"

    future_trajectories = {
        agent_id: [
            {
                "frame_index": int(frames[offset]),
                "position_blender_world_m": agent_positions[offset].astype(float).tolist(),
                "speed_m_per_s": float(agent_speeds[offset]),
            }
            for offset in range(observed_end_offset + 1, final_offset + 1)
        ],
        patient_id: [
            {
                "frame_index": int(frames[offset]),
                "position_blender_world_m": patient_positions[offset].astype(float).tolist(),
                "speed_m_per_s": float(patient_speeds[offset]),
            }
            for offset in range(observed_end_offset + 1, final_offset + 1)
        ],
    }
    return {
        "tool": "physionpp_friction_collision_future_rollout",
        "status": "ok",
        "backend": "rollout.collision_friction_analytic",
        "simulator": "swr_backend.collision_friction_spheres",
        "source_segment": "seg2",
        "rollout_state_policy": "continuous_from_optimized_seg2_initial_state",
        "agent_object_id": agent_id,
        "patient_object_id": patient_id,
        "patient_motion_mode": segment.get("patient_motion_mode"),
        "physical_parameters": {
            "agent_segment_initial_position_blender_world_m": agent_initial_position.astype(float).tolist(),
            "patient_segment_initial_position_blender_world_m": patient_initial_position.astype(float).tolist(),
            "agent_segment_initial_velocity_blender_world_m_per_s": agent_initial_velocity.astype(float).tolist(),
            "patient_segment_initial_velocity_blender_world_m_per_s": patient_initial_velocity.astype(float).tolist(),
            "agent_radius_m": agent_radius,
            "patient_radius_m": patient_radius,
            "agent_ground_friction": agent_friction,
            "patient_ground_friction": patient_friction,
            "agent_collision_obb_half_extents_m": agent_obb["half_extents"].astype(float).tolist(),
            "patient_collision_obb_half_extents_m": patient_obb["half_extents"].astype(float).tolist(),
            "gravity_direction_blender_world": gravity_direction.astype(float).tolist(),
            "gravity_m_per_s2": gravity_magnitude,
        },
        "horizon": {
            "max_future_frames": int(max_future_frames),
            "observed_first_frame": int(observed_first_frame),
            "observed_last_frame": int(observed_last_frame),
            "rollout_last_frame": int(frames[final_offset]),
            "future_frame_count": int(max(final_offset - observed_end_offset, 0)),
            "physics_dt_sec": physics_dt_sec,
            "stop_reason": stop_reason,
            "stationary_speed_m_per_s": float(stationary_speed_m_per_s),
            "stationary_window_frames": int(stationary_window_frames),
        },
        "patient_contact": {
            "will_contact": first_contact is not None,
            "first_contact": first_contact,
            "closest_approach": closest,
            "contact_geometry": "continuous_3d_fixed_orientation_obb_sat",
        },
        "future_trajectories": future_trajectories,
        "plane_contact_events": {
            "agent": agent_plane_events,
            "patient": patient_plane_events,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a fitted Physion++ friction-collision test segment.")
    parser.add_argument("--fit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-future-frames", type=int, default=DEFAULT_MAX_FUTURE_FRAMES)
    parser.add_argument("--stationary-speed", type=float, default=DEFAULT_STATIONARY_SPEED_M_PER_S)
    parser.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW_FRAMES)
    args = parser.parse_args()
    payload = run_physionpp_friction_collision_future_rollout(
        fit=_load_json(Path(args.fit)),
        max_future_frames=args.max_future_frames,
        stationary_speed_m_per_s=args.stationary_speed,
        stationary_window_frames=args.stationary_window,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
