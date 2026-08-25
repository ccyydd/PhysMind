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

from scripts.world_model import run_physionpp_friction_sphere_sysid as swr
from scripts.world_model import swr_sysid_common as sysid_common


DEFAULT_MAX_FUTURE_FRAMES = 300
DEFAULT_STATIONARY_SPEED_M_PER_S = 0.01
DEFAULT_STATIONARY_WINDOW_FRAMES = 10


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_object_items(object_plan: dict[str, Any]) -> list[dict[str, Any]]:
    items = object_plan.get("target_objects")
    return items if isinstance(items, list) else []


def _object_id_for_source_track(object_plan: dict[str, Any], source_track_id: str | None) -> str | None:
    if not source_track_id:
        return None
    for item in _target_object_items(object_plan):
        if isinstance(item, dict) and str(item.get("source_track_id") or "") == str(source_track_id):
            object_id = str(item.get("object_id") or "")
            if object_id:
                return object_id
    return None


def _physion_role_binding(sam3_tracks: dict[str, Any]) -> dict[str, Any]:
    tracking = sam3_tracks.get("physion_tracking") if isinstance(sam3_tracks, dict) else {}
    if not isinstance(tracking, dict):
        return {}
    role_binding = tracking.get("role_binding")
    return role_binding if isinstance(role_binding, dict) else {}


def _best_parameters_for_agent(fit: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    alignment = fit.get("alignment_optimization") if isinstance(fit.get("alignment_optimization"), dict) else {}
    target = alignment.get("optimization_target") if isinstance(alignment.get("optimization_target"), dict) else {}
    best = alignment.get("best_parameters") if isinstance(alignment.get("best_parameters"), dict) else {}
    agent_id = str(target.get("object_id") or "")
    if agent_id and isinstance(best.get(agent_id), dict):
        return agent_id, best[agent_id]
    for object_id, params in best.items():
        if isinstance(params, dict):
            return str(object_id), params
    raise ValueError("Physion++ future rollout requires alignment_optimization.best_parameters")


def _runtime_bounded_planes(
    *,
    fit: dict[str, Any],
    manifest: dict[str, Any],
    target: dict[str, list[dict[str, Any]]],
    static_ids: list[str],
) -> list[dict[str, Any]]:
    physics = fit.get("physics_rollout") if isinstance(fit.get("physics_rollout"), dict) else {}
    bounded = physics.get("bounded_static_planes") if isinstance(physics.get("bounded_static_planes"), dict) else {}
    debug_path = bounded.get("debug_path")
    if debug_path and Path(str(debug_path)).exists():
        plane_payload = _load_json(Path(str(debug_path)))
    else:
        static_meshes = swr._static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
        plane_payload = swr._decompose_static_meshes_into_bounded_planes(
            static_meshes=static_meshes,
            gravity_direction=swr._gravity_direction_blender_world(manifest),
        )
    static_meshes = swr._static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
    runtime_payload = dict(plane_payload)
    runtime_payload["_static_meshes_runtime"] = static_meshes
    planes = swr._bounded_planes_runtime(runtime_payload)
    if not planes:
        raise ValueError("Physion++ future rollout has no bounded static planes")
    return planes


def _friction_plane_ids_from_fit(*, fit: dict[str, Any], params: dict[str, Any]) -> set[str] | None:
    raw_ids = params.get("friction_plane_ids")
    if isinstance(raw_ids, list) and raw_ids:
        return {str(value) for value in raw_ids if str(value)}
    physics = fit.get("physics_rollout") if isinstance(fit.get("physics_rollout"), dict) else {}
    for event in physics.get("contact_events") or []:
        if not isinstance(event, dict):
            continue
        plane_id = str(event.get("plane_id") or "")
        if plane_id:
            return {plane_id}
    return None


def _rollout_positions(
    *,
    fit: dict[str, Any],
    manifest: dict[str, Any],
    max_future_frames: int,
) -> tuple[str, np.ndarray, list[int], list[dict[str, Any]], list[dict[str, Any]], float, float, set[str] | None]:
    target = sysid_common._target_records_from_swr(manifest)
    agent_id, static_ids = swr._select_agent_and_statics(manifest, target)
    agent_id_from_fit, params = _best_parameters_for_agent(fit)
    if agent_id_from_fit != agent_id:
        agent_id = agent_id_from_fit
    target = {object_id: target[object_id] for object_id in [agent_id, *static_ids] if object_id in target}
    frames_observed = [int(record["frame_index"]) for record in target[agent_id]]
    if len(frames_observed) < 2:
        raise ValueError("Physion++ future rollout requires at least two observed agent frames")
    observed_last_frame = int(frames_observed[-1])
    frames = list(range(int(frames_observed[0]), observed_last_frame + int(max_future_frames) + 1))
    start_position = np.asarray(target[agent_id][0]["position"], dtype=np.float64).reshape(3)
    velocity = np.asarray(params["initial_velocity_blender_world_m_per_s"], dtype=np.float64).reshape(3)
    friction = float(params.get("sliding_friction", 0.0))
    radius = float(params["optimized_radius_m"])
    gravity_magnitude = float(params["gravity_m_per_s2"])
    physics_dt_sec = float((fit.get("trajectory_physics_initialization") or {}).get("physics_dt_sec") or swr.PHYSIONPP_PHYSICS_DT_SEC)
    planes = _runtime_bounded_planes(fit=fit, manifest=manifest, target=target, static_ids=static_ids)
    friction_plane_ids = _friction_plane_ids_from_fit(fit=fit, params=params)
    support = (fit.get("physics_rollout") or {}).get("support_plane") or {}
    gravity_direction = np.asarray(
        support.get("gravity_direction") or swr._gravity_direction_blender_world(manifest),
        dtype=np.float64,
    ).reshape(3)
    gravity_norm = max(float(np.linalg.norm(gravity_direction)), 1e-12)
    gravity_direction = gravity_direction / gravity_norm
    predicted, events = swr._rollout_bounded_planes_analytic(
        frames=frames,
        physics_dt_sec=physics_dt_sec,
        start_position=torch.tensor(start_position, dtype=torch.float64),
        velocity=torch.tensor(velocity, dtype=torch.float64),
        friction=torch.tensor(friction, dtype=torch.float64),
        dynamic_radius=torch.tensor(radius, dtype=torch.float64),
        bounded_planes=planes,
        gravity_direction=torch.tensor(gravity_direction, dtype=torch.float64),
        gravity_magnitude=torch.tensor(gravity_magnitude, dtype=torch.float64),
        friction_plane_ids=friction_plane_ids,
    )
    return agent_id, predicted.detach().cpu().numpy(), frames, events, planes, radius, physics_dt_sec, friction_plane_ids


def _frame_speeds(positions: np.ndarray, frames: list[int], physics_dt_sec: float) -> list[float | None]:
    speeds: list[float | None] = [None]
    for prev, curr, prev_frame, curr_frame in zip(positions[:-1], positions[1:], frames[:-1], frames[1:]):
        dt = max(float(int(curr_frame) - int(prev_frame)) * float(physics_dt_sec), 1e-12)
        speeds.append(float(np.linalg.norm(np.asarray(curr) - np.asarray(prev)) / dt))
    return speeds


def _stationary_stop_offset(
    *,
    speeds: list[float | None],
    observed_count: int,
    threshold_m_per_s: float,
    window_frames: int,
) -> int | None:
    if window_frames <= 0:
        return None
    consecutive = 0
    for index in range(max(observed_count, 1), len(speeds)):
        speed = speeds[index]
        if speed is not None and speed <= float(threshold_m_per_s):
            consecutive += 1
            if consecutive >= int(window_frames):
                return index
        else:
            consecutive = 0
    return None


def _closest_patient_distance(
    *,
    point: np.ndarray,
    radius: float,
    patient_planes: list[dict[str, Any]],
) -> tuple[float | None, str | None, float | None, float | None]:
    best: tuple[float, str | None, float | None, float | None] | None = None
    for plane in patient_planes:
        closest = swr._bounded_plane_surface_closest_point(point=point, plane=plane)
        if closest is None:
            continue
        surface_gap = float(closest["distance_m"]) - float(radius)
        distance = max(surface_gap, 0.0)
        if best is None or distance < best[0]:
            best = (
                distance,
                str(plane.get("plane_id")),
                surface_gap,
                float(closest["boundary_distance_m"]),
            )
    if best is None:
        return None, None, None, None
    return best


def run_physionpp_friction_future_rollout(
    *,
    fit: dict[str, Any],
    manifest: dict[str, Any],
    object_plan: dict[str, Any],
    sam3_tracks: dict[str, Any],
    max_future_frames: int = DEFAULT_MAX_FUTURE_FRAMES,
    stationary_speed_m_per_s: float = DEFAULT_STATIONARY_SPEED_M_PER_S,
    stationary_window_frames: int = DEFAULT_STATIONARY_WINDOW_FRAMES,
) -> dict[str, Any]:
    agent_id, positions, frames, events, planes, radius, physics_dt_sec, friction_plane_ids = _rollout_positions(
        fit=fit,
        manifest=manifest,
        max_future_frames=max_future_frames,
    )
    observed_last_frame = max(
        int(record["frame_index"])
        for record in ((fit.get("target_trajectories") or {}).get(agent_id) or [])
        if isinstance(record, dict) and record.get("frame_index") is not None
    )
    observed_count = sum(1 for frame in frames if int(frame) <= observed_last_frame)
    speeds = _frame_speeds(positions, frames, physics_dt_sec)
    stationary_offset = _stationary_stop_offset(
        speeds=speeds,
        observed_count=observed_count,
        threshold_m_per_s=stationary_speed_m_per_s,
        window_frames=stationary_window_frames,
    )
    final_offset = len(frames) - 1 if stationary_offset is None else int(stationary_offset)
    role_binding = _physion_role_binding(sam3_tracks)
    patient_track = str(role_binding.get("patient_track") or "")
    agent_track = str(role_binding.get("agent_track") or "")
    patient_id = _object_id_for_source_track(object_plan, patient_track)
    role_agent_id = _object_id_for_source_track(object_plan, agent_track)
    patient_planes = [plane for plane in planes if patient_id and str(plane.get("object_id")) == patient_id]
    event_contacts = sorted(
        (
            event
            for event in events
            if str(event.get("support_object_id") or "") == str(patient_id or "")
            and float(event.get("continuous_frame_index", event.get("frame_index", -1)))
            > float(observed_last_frame)
            and float(event.get("continuous_frame_index", event.get("frame_index", 10**9)))
            <= float(frames[final_offset])
        ),
        key=lambda event: float(event.get("continuous_frame_index", event.get("frame_index", 10**9))),
    )
    first_contact: dict[str, Any] | None = None
    if event_contacts:
        event = event_contacts[0]
        contact_frame = float(event.get("continuous_frame_index", event["frame_index"]))
        first_contact = {
            "frame_index": contact_frame,
            "time_sec": float((contact_frame - observed_last_frame) * physics_dt_sec),
            "plane_id": str(event.get("plane_id") or ""),
            "support_object_id": patient_id,
            "signed_distance_m": event.get("signed_distance_m"),
            "boundary_distance_m": event.get("boundary_distance_m"),
            "contact_kind": str(event.get("contact_kind") or "face"),
            "source": "continuous_analytic_contact_event",
        }
    closest = {
        "distance_m": None,
        "frame_index": None,
        "plane_id": None,
        "normal_gap_m": None,
        "boundary_distance_m": None,
    }
    for offset, frame in enumerate(frames):
        if offset < observed_count or offset > final_offset:
            continue
        point = np.asarray(positions[offset], dtype=np.float64).reshape(3)
        distance, plane_id, normal_gap, boundary_distance = _closest_patient_distance(
            point=point,
            radius=radius,
            patient_planes=patient_planes,
        )
        if distance is not None and (closest["distance_m"] is None or distance < float(closest["distance_m"])):
            closest = {
                "distance_m": float(distance),
                "frame_index": int(frame),
                "plane_id": plane_id,
                "normal_gap_m": normal_gap,
                "boundary_distance_m": boundary_distance,
            }
        contact = (
            swr._select_bounded_plane_contact(
                point=point,
                radius=radius,
                planes=patient_planes,
                contact_margin=1e-9,
            )
            if patient_planes
            else None
        )
        if first_contact is None and contact is not None:
            first_contact = {
                "frame_index": int(frame),
                "time_sec": float((int(frame) - observed_last_frame) * physics_dt_sec),
                "plane_id": str((contact.get("plane") or {}).get("plane_id")),
                "support_object_id": patient_id,
                "signed_distance_m": float(contact.get("signed")),
                "boundary_distance_m": float(contact.get("boundary_distance")),
                "contact_kind": str(contact.get("contact_kind") or "face"),
                "source": "exact_sampled_contact_at_frame",
            }
    future_records = []
    for offset, frame in enumerate(frames[: final_offset + 1]):
        if int(frame) <= observed_last_frame:
            continue
        future_records.append(
            {
                "frame_index": int(frame),
                "position": [float(value) for value in positions[offset].tolist()],
                "speed_m_per_s": None if speeds[offset] is None else float(speeds[offset]),
            }
        )
    return {
        "tool": "physionpp_friction_future_rollout",
        "status": "ok",
        "backend": "physionpp_friction_future_analytic_rollout",
        "simulator": "swr_backend.surface_friction_sphere",
        "agent_object_id": agent_id,
        "role_bound_agent_object_id": role_agent_id,
        "patient_object_id": patient_id,
        "role_binding": {
            "agent_track": agent_track,
            "patient_track": patient_track,
            "status": role_binding.get("status"),
        },
        "friction_model": {
            "friction_plane_ids": sorted(friction_plane_ids) if friction_plane_ids is not None else None,
            "other_plane_sliding_friction": 0.0 if friction_plane_ids is not None else None,
        },
        "horizon": {
            "max_future_frames": int(max_future_frames),
            "observed_last_frame": int(observed_last_frame),
            "rollout_last_frame": int(frames[final_offset]),
            "future_frame_count": int(len(future_records)),
            "physics_dt_sec": float(physics_dt_sec),
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
            if int(event.get("frame_index", -1)) > int(observed_last_frame)
            and int(event.get("frame_index", 10**9)) <= int(frames[final_offset])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out Physion++ friction-platform future OCP evidence.")
    parser.add_argument("--fit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--sam3-tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-future-frames", type=int, default=DEFAULT_MAX_FUTURE_FRAMES)
    parser.add_argument("--stationary-speed", type=float, default=DEFAULT_STATIONARY_SPEED_M_PER_S)
    parser.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW_FRAMES)
    args = parser.parse_args()
    payload = run_physionpp_friction_future_rollout(
        fit=_load_json(Path(args.fit)),
        manifest=_load_json(Path(args.manifest)),
        object_plan=_load_json(Path(args.object_plan)),
        sam3_tracks=_load_json(Path(args.sam3_tracks)),
        max_future_frames=args.max_future_frames,
        stationary_speed_m_per_s=args.stationary_speed,
        stationary_window_frames=args.stationary_window,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
