from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.world_model import physionpp_friction_future_rollout as friction_future  # noqa: E402
from scripts.world_model import physionpp_future_rollout_common as future_common  # noqa: E402
from scripts.world_model import run_physionpp_bouncy_wall_sphere_sysid as bouncy_swr  # noqa: E402
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr  # noqa: E402


DEFAULT_MAX_FUTURE_FRAMES = 300
DEFAULT_STATIONARY_SPEED_M_PER_S = 0.01
DEFAULT_STATIONARY_WINDOW_FRAMES = 10


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment2_fit(fit: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in fit.get("segments") or []
        if isinstance(item, dict) and item.get("status") == "ok" and str(item.get("segment")) == "seg2"
    ]
    if len(matches) != 1:
        raise ValueError(f"bouncy-wall future rollout requires exactly one valid seg2 fit, found {len(matches)}")
    return matches[0]


def _best_parameters(segment: dict[str, Any], agent_id: str) -> dict[str, Any]:
    alignment = segment.get("alignment_optimization") or {}
    best = alignment.get("best_parameters") or {}
    params = best.get(agent_id)
    if not isinstance(params, dict):
        raise ValueError(f"seg2 fit has no best parameters for {agent_id}")
    return params


def _first_patient_contact(
    *,
    positions: np.ndarray,
    frames: list[int],
    events: list[dict[str, Any]],
    patient_id: str,
    patient_planes: list[dict[str, Any]],
    radius: float,
    observed_last_frame: int,
    final_offset: int,
    physics_dt_sec: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    closest, proximity_samples = future_common.scan_patient_proximity(
        positions=positions,
        frames=frames,
        patient_planes=patient_planes,
        radius=radius,
        final_offset=final_offset,
    )
    first_contact = None
    event_frames = {
        int(event["frame_index"])
        for event in events
        if str(event.get("support_object_id") or "") == patient_id
        and event.get("frame_index") is not None
    }
    for sample in proximity_samples:
        frame = int(sample["frame_index"])
        point = sample["point_np"]
        plane_id = sample["plane_id"]
        boundary_distance = sample["boundary_distance_m"]
        contact = (
            friction_swr._select_bounded_plane_contact(
                point=point,
                radius=radius,
                planes=patient_planes,
            )
            if patient_planes
            else None
        )
        if first_contact is None and (frame in event_frames or contact is not None):
            first_contact = {
                "frame_index": frame,
                "time_sec": float((frame - observed_last_frame) * physics_dt_sec),
                "plane_id": (
                    str((contact.get("plane") or {}).get("plane_id")) if contact is not None else plane_id
                ),
                "support_object_id": patient_id,
                "signed_distance_m": None if contact is None else float(contact.get("signed")),
                "boundary_distance_m": (
                    boundary_distance if contact is None else float(contact.get("boundary_distance"))
                ),
            }
    return first_contact, closest


def run_physionpp_bouncy_wall_future_rollout(
    *,
    fit: dict[str, Any],
    max_future_frames: int = DEFAULT_MAX_FUTURE_FRAMES,
    stationary_speed_m_per_s: float = DEFAULT_STATIONARY_SPEED_M_PER_S,
    stationary_window_frames: int = DEFAULT_STATIONARY_WINDOW_FRAMES,
) -> dict[str, Any]:
    segment = _segment2_fit(fit)
    agent_id = str(segment["agent_object_id"])
    patient_id = str(segment["patient_object_id"])
    wall_id = str(segment["wall_object_id"])
    physics = segment.get("physics_rollout") or {}
    terminal = physics.get("terminal_state") or {}
    observed_last_frame = int(terminal["frame_index"])
    start_position = np.asarray(terminal["position_blender_world_m"], dtype=np.float64).reshape(3)
    start_velocity = np.asarray(terminal["velocity_blender_world_m_per_s"], dtype=np.float64).reshape(3)
    params = _best_parameters(segment, agent_id)
    radius = float(params["optimized_radius_m"])
    support_friction = float(params["support_sliding_friction"])
    wall_restitution = float(params["wall_restitution"])
    gravity_magnitude = float(params["gravity_m_per_s2"])
    planes = [
        future_common.runtime_plane(
            item,
            offset_source="payload_offset",
        )
        for item in physics.get("contact_plane_geometry") or []
    ]
    if not planes:
        raise ValueError("seg2 fit has no self-contained contact plane geometry")
    wall_plane_ids = {str(value) for value in physics.get("wall_plane_ids") or []}
    raw_friction_ids = physics.get("support_friction_plane_ids")
    friction_plane_ids = (
        {str(value) for value in raw_friction_ids}
        if isinstance(raw_friction_ids, list)
        else None
    )
    global_ground = fit.get("global_ground_plane") or {}
    up = np.asarray(global_ground["normal_blender_world"], dtype=np.float64).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    gravity_direction = -up
    physics_dt_sec = float(bouncy_swr.PHYSIONPP_PHYSICS_DT_SEC)
    frames = list(range(observed_last_frame, observed_last_frame + int(max_future_frames) + 1))
    predicted, predicted_velocities, events = bouncy_swr._rollout_bouncy_wall_bounded_planes(
        frames=frames,
        physics_dt_sec=physics_dt_sec,
        start_position=torch.tensor(start_position, dtype=torch.float64),
        velocity=torch.tensor(start_velocity, dtype=torch.float64),
        support_friction=torch.tensor(support_friction, dtype=torch.float64),
        wall_restitution=torch.tensor(wall_restitution, dtype=torch.float64),
        dynamic_radius=torch.tensor(radius, dtype=torch.float64),
        bounded_planes=planes,
        wall_plane_ids=wall_plane_ids,
        support_friction_plane_ids=friction_plane_ids,
        gravity_direction=torch.tensor(gravity_direction, dtype=torch.float64),
        gravity_magnitude=torch.tensor(gravity_magnitude, dtype=torch.float64),
    )
    positions = predicted.detach().cpu().numpy().astype(float)
    velocities = predicted_velocities.detach().cpu().numpy().astype(float)
    speeds = [float(np.linalg.norm(velocity)) for velocity in velocities]
    stationary_offset = friction_future._stationary_stop_offset(
        speeds=speeds,
        observed_count=1,
        threshold_m_per_s=stationary_speed_m_per_s,
        window_frames=stationary_window_frames,
    )
    final_offset = len(frames) - 1 if stationary_offset is None else int(stationary_offset)
    patient_planes = [plane for plane in planes if str(plane.get("object_id") or "") == patient_id]
    first_contact, closest = _first_patient_contact(
        positions=positions,
        frames=frames,
        events=events,
        patient_id=patient_id,
        patient_planes=patient_planes,
        radius=radius,
        observed_last_frame=observed_last_frame,
        final_offset=final_offset,
        physics_dt_sec=physics_dt_sec,
    )
    future_records = [
        {
            "frame_index": int(frames[offset]),
            "position": positions[offset].tolist(),
            "velocity_m_per_s": velocities[offset].tolist(),
            "speed_m_per_s": speeds[offset],
        }
        for offset in range(1, final_offset + 1)
    ]
    return {
        "tool": "physionpp_bouncy_wall_future_rollout",
        "status": "ok",
        "backend": "rollout.wall_bounce_analytic",
        "simulator": "swr_backend.wall_bounce_sphere",
        "source_segment": "seg2",
        "agent_object_id": agent_id,
        "patient_object_id": patient_id,
        "wall_object_id": wall_id,
        "horizon": {
            "max_future_frames": int(max_future_frames),
            "observed_last_frame": observed_last_frame,
            "rollout_last_frame": int(frames[final_offset]),
            "future_frame_count": int(final_offset),
            "physics_dt_sec": physics_dt_sec,
            "stop_reason": "stationary" if stationary_offset is not None else "max_future_frames",
            "stationary_speed_m_per_s": float(stationary_speed_m_per_s),
            "stationary_window_frames": int(stationary_window_frames),
        },
        "patient_contact": {
            "will_contact": first_contact is not None,
            "first_contact": first_contact,
            "closest_approach": closest,
            "patient_plane_count": int(len(patient_planes)),
        },
        "future_trajectory": future_records,
        "contact_events": [
            event
            for event in events
            if int(event.get("frame_index", -1)) > observed_last_frame
            and int(event.get("frame_index", 10**9)) <= int(frames[final_offset])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a fitted Physion++ bouncy-wall test segment.")
    parser.add_argument("--fit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-future-frames", type=int, default=DEFAULT_MAX_FUTURE_FRAMES)
    parser.add_argument("--stationary-speed", type=float, default=DEFAULT_STATIONARY_SPEED_M_PER_S)
    parser.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW_FRAMES)
    args = parser.parse_args()
    payload = run_physionpp_bouncy_wall_future_rollout(
        fit=_load_json(Path(args.fit)),
        max_future_frames=args.max_future_frames,
        stationary_speed_m_per_s=args.stationary_speed,
        stationary_window_frames=args.stationary_window,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "will_contact": payload["patient_contact"]["will_contact"],
                "rollout_last_frame": payload["horizon"]["rollout_last_frame"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
