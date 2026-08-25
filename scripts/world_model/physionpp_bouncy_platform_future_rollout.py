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
from scripts.world_model import run_physionpp_bouncy_platform_sphere_sysid as bouncy_platform_swr  # noqa: E402
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr  # noqa: E402


DEFAULT_MAX_FUTURE_FRAMES = 300
DEFAULT_STATIONARY_SPEED_M_PER_S = 0.01
DEFAULT_STATIONARY_WINDOW_FRAMES = 10


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _best_parameters(fit: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    alignment = fit.get("alignment_optimization") or {}
    target = alignment.get("optimization_target") or {}
    best = alignment.get("best_parameters") or {}
    agent_id = str(target.get("object_id") or "")
    params = best.get(agent_id) if agent_id else None
    if agent_id and isinstance(params, dict):
        return agent_id, params
    raise ValueError("bouncy-platform future rollout requires fitted agent parameters")


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
    exact_sample_contact = None
    for sample in proximity_samples:
        frame = int(sample["frame_index"])
        point = sample["point_np"]
        if exact_sample_contact is None and patient_planes:
            contact = friction_swr._select_bounded_plane_contact(
                point=point,
                radius=radius,
                planes=patient_planes,
                contact_margin=1e-9,
            )
            if contact is not None:
                exact_sample_contact = {
                    "frame_index": frame,
                    "time_sec": float((frame - observed_last_frame) * physics_dt_sec),
                    "plane_id": str((contact.get("plane") or {}).get("plane_id") or ""),
                    "support_object_id": patient_id,
                    "source": "exact_sampled_geometry_contact",
                }

    event_contacts = sorted(
        (
            event
            for event in events
            if str(event.get("support_object_id") or "") == patient_id
            and observed_last_frame < int(event.get("frame_index", -1)) <= int(frames[final_offset])
        ),
        key=lambda event: int(event["frame_index"]),
    )
    if event_contacts:
        event = event_contacts[0]
        frame = int(event["frame_index"])
        first_contact = {
            "frame_index": frame,
            "time_sec": float((frame - observed_last_frame) * physics_dt_sec),
            "plane_id": str(event.get("plane_id") or ""),
            "support_object_id": patient_id,
            "source": "continuous_analytic_contact_event",
        }
    else:
        first_contact = exact_sample_contact
    return first_contact, closest


def run_physionpp_bouncy_platform_future_rollout(
    *,
    fit: dict[str, Any],
    manifest: dict[str, Any],
    object_plan: dict[str, Any],
    sam3_tracks: dict[str, Any],
    max_future_frames: int = DEFAULT_MAX_FUTURE_FRAMES,
    stationary_speed_m_per_s: float = DEFAULT_STATIONARY_SPEED_M_PER_S,
    stationary_window_frames: int = DEFAULT_STATIONARY_WINDOW_FRAMES,
) -> dict[str, Any]:
    agent_id, params = _best_parameters(fit)
    physics = fit.get("physics_rollout") or {}
    terminal = physics.get("terminal_state") or {}
    observed_last_frame = int(terminal["frame_index"])
    start_position = np.asarray(terminal["position_blender_world_m"], dtype=np.float64).reshape(3)
    start_velocity = np.asarray(terminal["velocity_blender_world_m_per_s"], dtype=np.float64).reshape(3)
    radius = float(params["optimized_radius_m"])
    friction = float(params["sliding_friction"])
    restitution = float(params["restitution"])
    gravity_direction = friction_swr._gravity_direction_blender_world(manifest)
    planes = [
        future_common.runtime_plane(
            item,
            offset_source="surface_point_dot_normal",
        )
        for item in physics.get("contact_plane_geometry") or []
    ]
    if not planes:
        raise ValueError("bouncy-platform fit has no self-contained contact plane geometry")

    role_binding = friction_future._physion_role_binding(sam3_tracks)
    agent_track = str(role_binding.get("agent_track") or "")
    patient_track = str(role_binding.get("patient_track") or "")
    role_agent_id = friction_future._object_id_for_source_track(object_plan, agent_track)
    patient_id = friction_future._object_id_for_source_track(object_plan, patient_track)
    if not patient_id:
        raise ValueError("bouncy-platform future rollout could not resolve the yellow-cued patient object")

    physics_dt_sec = float(
        (fit.get("trajectory_physics_initialization") or {}).get("physics_dt_sec")
        or friction_swr.PHYSIONPP_PHYSICS_DT_SEC
    )
    frames = list(range(observed_last_frame, observed_last_frame + int(max_future_frames) + 1))
    predicted, predicted_velocities, events = bouncy_platform_swr._rollout_bouncy_platform(
        frames=frames,
        start_position=torch.tensor(start_position, dtype=torch.float64),
        initial_velocity=torch.tensor(start_velocity, dtype=torch.float64),
        friction=torch.tensor(friction, dtype=torch.float64),
        restitution=torch.tensor(restitution, dtype=torch.float64),
        radius=torch.tensor(radius, dtype=torch.float64),
        planes=planes,
        gravity_direction=torch.tensor(gravity_direction, dtype=torch.float64),
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
        "tool": "physionpp_bouncy_platform_future_rollout",
        "status": "ok",
        "backend": "rollout.platform_bounce_analytic",
        "simulator": "swr_backend.platform_bounce_sphere",
        "source_segment": "single_sequence",
        "agent_object_id": agent_id,
        "role_bound_agent_object_id": role_agent_id,
        "patient_object_id": patient_id,
        "role_binding": {
            "agent_track": agent_track,
            "patient_track": patient_track,
            "status": role_binding.get("status"),
        },
        "physical_parameters": {
            "initial_position_blender_world_m": start_position.astype(float).tolist(),
            "initial_velocity_blender_world_m_per_s": start_velocity.astype(float).tolist(),
            "sphere_radius_m": radius,
            "sliding_friction": friction,
            "restitution": restitution,
            "gravity_m_per_s2": friction_swr.GRAVITY_M_PER_S2,
        },
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
            if observed_last_frame < int(event.get("frame_index", -1)) <= int(frames[final_offset])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a fitted Physion++ bouncy-platform scene.")
    parser.add_argument("--fit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--sam3-tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-future-frames", type=int, default=DEFAULT_MAX_FUTURE_FRAMES)
    parser.add_argument("--stationary-speed", type=float, default=DEFAULT_STATIONARY_SPEED_M_PER_S)
    parser.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW_FRAMES)
    args = parser.parse_args()
    payload = run_physionpp_bouncy_platform_future_rollout(
        fit=_load_json(Path(args.fit)),
        manifest=_load_json(Path(args.manifest)),
        object_plan=_load_json(Path(args.object_plan)),
        sam3_tracks=_load_json(Path(args.sam3_tracks)),
        max_future_frames=args.max_future_frames,
        stationary_speed_m_per_s=args.stationary_speed,
        stationary_window_frames=args.stationary_window,
    )
    output = Path(args.output)
    _write_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_write_payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "will_contact": payload["patient_contact"]["will_contact"],
                "rollout_last_frame": payload["horizon"]["rollout_last_frame"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
