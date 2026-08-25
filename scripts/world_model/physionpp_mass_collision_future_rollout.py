from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.world_model import physionpp_friction_collision_future_rollout as collision_future  # noqa: E402
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr  # noqa: E402


DEFAULT_MAX_FUTURE_FRAMES = 300
DEFAULT_STATIONARY_SPEED_M_PER_S = 0.01
DEFAULT_STATIONARY_WINDOW_FRAMES = 10


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _terminal_state(segment: dict[str, Any], object_id: str) -> tuple[int, np.ndarray, np.ndarray]:
    state = (segment.get("terminal_state") or {}).get(object_id)
    if not isinstance(state, dict):
        raise ValueError(f"seg2 has no terminal state for {object_id}")
    frame = state.get("frame_index")
    position = np.asarray(state.get("position_blender_world_m"), dtype=np.float64)
    velocity = np.asarray(state.get("velocity_blender_world_m_per_s"), dtype=np.float64)
    if not isinstance(frame, int) or position.shape != (3,) or velocity.shape != (3,):
        raise ValueError(f"seg2 has an invalid terminal state for {object_id}")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError(f"seg2 has a non-finite terminal state for {object_id}")
    return frame, position, velocity


def _trajectory_records(
    *,
    frames: list[int],
    positions: np.ndarray,
    speeds: np.ndarray,
    first_future_offset: int,
    final_offset: int,
) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(frames[offset]),
            "position_blender_world_m": positions[offset].astype(float).tolist(),
            "speed_m_per_s": float(speeds[offset]),
        }
        for offset in range(first_future_offset, final_offset + 1)
    ]


def run_physionpp_mass_collision_future_rollout(
    *,
    fit: dict[str, Any],
    max_future_frames: int = DEFAULT_MAX_FUTURE_FRAMES,
    stationary_speed_m_per_s: float = DEFAULT_STATIONARY_SPEED_M_PER_S,
    stationary_window_frames: int = DEFAULT_STATIONARY_WINDOW_FRAMES,
) -> dict[str, Any]:
    segment = collision_future._segment2(fit)
    agent_id = str(segment["agent_object_id"])
    patient_id = str(segment["patient_object_id"])
    agent_frame, agent_position, agent_velocity = _terminal_state(segment, agent_id)
    patient_frame, patient_position, patient_velocity = _terminal_state(segment, patient_id)
    if agent_frame != patient_frame:
        raise ValueError(
            "mass-collision future rollout requires agent and patient terminal states at the same frame: "
            f"{agent_frame} != {patient_frame}"
        )
    observed_last_frame = agent_frame
    frames = list(range(observed_last_frame, observed_last_frame + int(max_future_frames) + 1))
    observed_end_offset = 0

    support = fit.get("support_plane") or {}
    up = np.asarray(support.get("up_direction_blender_world"), dtype=np.float64).reshape(3)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    gravity_direction = np.asarray(support.get("gravity_direction_blender_world"), dtype=np.float64).reshape(3)
    gravity_direction /= max(float(np.linalg.norm(gravity_direction)), 1e-12)
    plane_point = collision_future._support_plane_point(support)
    radii = fit.get("shared_role_radii_m") or {}
    agent_radius = float(radii["agent"])
    patient_radius = float(radii["patient"])
    optimization = fit.get("alignment_optimization") or {}
    params = optimization.get("best_parameters") or {}
    agent_friction = float(params["agent_ground_friction"])
    patient_friction = float(params["patient_ground_friction"])
    gravity_magnitude = float(params.get("gravity_m_per_s2") or friction_swr.GRAVITY_M_PER_S2)
    physics_dt_sec = float(
        (optimization.get("optimizer") or {}).get("physics_dt_sec")
        or friction_swr.PHYSIONPP_PHYSICS_DT_SEC
    )

    agent_positions, agent_plane_events = collision_future._rollout_role(
        frames=frames,
        start_position=agent_position,
        initial_velocity=agent_velocity,
        friction=agent_friction,
        radius=agent_radius,
        plane_point=plane_point,
        up=up,
        gravity_direction=gravity_direction,
        gravity_magnitude=gravity_magnitude,
        physics_dt_sec=physics_dt_sec,
    )
    patient_positions, patient_plane_events = collision_future._rollout_role(
        frames=frames,
        start_position=patient_position,
        initial_velocity=patient_velocity,
        friction=patient_friction,
        radius=patient_radius,
        plane_point=plane_point,
        up=up,
        gravity_direction=gravity_direction,
        gravity_magnitude=gravity_magnitude,
        physics_dt_sec=physics_dt_sec,
    )
    agent_speeds = collision_future._frame_speeds(agent_positions, physics_dt_sec)
    patient_speeds = collision_future._frame_speeds(patient_positions, physics_dt_sec)
    agent_obb = collision_future._fixed_collision_obb(fit=fit, segment=segment, object_id=agent_id)
    patient_obb = collision_future._fixed_collision_obb(fit=fit, segment=segment, object_id=patient_id)
    stationary_offset = collision_future._both_stationary_offset(
        agent_speeds=agent_speeds,
        patient_speeds=patient_speeds,
        observed_end_offset=observed_end_offset,
        threshold_m_per_s=stationary_speed_m_per_s,
        window_frames=stationary_window_frames,
    )
    scan_end_offset = len(frames) - 1 if stationary_offset is None else int(stationary_offset)
    first_contact, closest = collision_future._contact_scan(
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
        final_offset = min(
            scan_end_offset,
            int(math.ceil(float(first_contact["frame_index"]))) - observed_last_frame,
        )
        stop_reason = "agent_patient_contact"
    elif stationary_offset is not None:
        final_offset = int(stationary_offset)
        stop_reason = "both_dynamic_objects_stationary"
    else:
        final_offset = scan_end_offset
        stop_reason = "max_future_frames"

    return {
        "tool": "physionpp_mass_collision_future_rollout",
        "status": "ok",
        "backend": "rollout.collision_mass_analytic",
        "simulator": "swr_backend.collision_mass_spheres",
        "source_segment": "seg2",
        "rollout_state_policy": "continue_from_fitted_seg2_terminal_state",
        "agent_object_id": agent_id,
        "patient_object_id": patient_id,
        "physical_parameters": {
            "agent_terminal_position_blender_world_m": agent_position.astype(float).tolist(),
            "patient_terminal_position_blender_world_m": patient_position.astype(float).tolist(),
            "agent_terminal_velocity_blender_world_m_per_s": agent_velocity.astype(float).tolist(),
            "patient_terminal_velocity_blender_world_m_per_s": patient_velocity.astype(float).tolist(),
            "agent_mass_over_ball_mass": float(params["agent_mass_over_ball_mass"]),
            "ball_agent_restitution": float(params["ball_agent_restitution"]),
            "agent_ground_friction": agent_friction,
            "patient_ground_friction": patient_friction,
            "agent_radius_m": agent_radius,
            "patient_radius_m": patient_radius,
            "agent_collision_obb_half_extents_m": agent_obb["half_extents"].astype(float).tolist(),
            "patient_collision_obb_half_extents_m": patient_obb["half_extents"].astype(float).tolist(),
            "gravity_direction_blender_world": gravity_direction.astype(float).tolist(),
            "gravity_m_per_s2": gravity_magnitude,
        },
        "horizon": {
            "max_future_frames": int(max_future_frames),
            "observed_last_frame": int(observed_last_frame),
            "rollout_last_frame": int(frames[final_offset]),
            "future_frame_count": int(final_offset),
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
        "future_trajectories": {
            agent_id: _trajectory_records(
                frames=frames,
                positions=agent_positions,
                speeds=agent_speeds,
                first_future_offset=1,
                final_offset=final_offset,
            ),
            patient_id: _trajectory_records(
                frames=frames,
                positions=patient_positions,
                speeds=patient_speeds,
                first_future_offset=1,
                final_offset=final_offset,
            ),
        },
        "plane_contact_events": {
            "agent": agent_plane_events,
            "patient": patient_plane_events,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a fitted Physion++ mass-collision test segment.")
    parser.add_argument("--fit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-future-frames", type=int, default=DEFAULT_MAX_FUTURE_FRAMES)
    parser.add_argument("--stationary-speed", type=float, default=DEFAULT_STATIONARY_SPEED_M_PER_S)
    parser.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW_FRAMES)
    args = parser.parse_args()
    payload = run_physionpp_mass_collision_future_rollout(
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
