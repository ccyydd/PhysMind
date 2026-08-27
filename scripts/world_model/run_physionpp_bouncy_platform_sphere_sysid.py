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

from scripts.world_model import analytic_swr_common
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr
from scripts.world_model import swr_sysid_common as sysid_common


DEFAULT_LEARNING_RATES = friction_swr.DEFAULT_LEARNING_RATES
DEFAULT_STEPS = 400
DEFAULT_PATIENCE = 100
MIN_RESTITUTION = 0.0
MAX_RESTITUTION = 1.0
INITIAL_RESTITUTION = 0.8
INITIAL_FRICTION = 0.1
INITIAL_VELOCITY_WINDOW_FRAMES = 10
BOUNCE_SPEED_EPS_M_PER_S = 1e-6


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reflect_from_plane(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    radius: torch.Tensor,
    contact: dict[str, Any],
    restitution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plane = friction_swr._torch_plane_from_runtime(contact, device=x.device, dtype=x.dtype)
    normal = plane["normal"] / torch.clamp(torch.linalg.norm(plane["normal"]), min=x.new_tensor(1e-12))
    x = friction_swr._project_to_contact(
        x=x,
        dynamic_radius=radius,
        plane_point=plane["surface_point"],
        normal=normal,
    )
    incoming_speed = torch.minimum(torch.sum(v * normal), v.new_tensor(0.0))
    return x, v - (1.0 + restitution) * incoming_speed * normal, normal


def _rollout_bouncy_platform(
    *,
    frames: list[int],
    start_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
    radius: torch.Tensor,
    planes: list[dict[str, Any]],
    gravity_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if not planes:
        raise ValueError("bouncy-platform rollout requires at least one bounded plane")
    gravity = gravity_direction * start_position.new_tensor(friction_swr.GRAVITY_M_PER_S2)
    x = start_position.clone()
    v = initial_velocity.clone()
    active_support: dict[str, Any] | None = None
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    events: list[dict[str, Any]] = []

    def refresh_contact(point: torch.Tensor, contact: dict[str, Any] | None) -> dict[str, Any] | None:
        if contact is None:
            return None
        return friction_swr._bounded_plane_contact_candidate(
            point=point.detach().cpu().numpy().astype(np.float64),
            radius=float(radius.detach().cpu()),
            plane=contact["plane"],
        )

    def select_contact(point: torch.Tensor) -> dict[str, Any] | None:
        return friction_swr._select_bounded_plane_contact(
            point=point.detach().cpu().numpy().astype(np.float64),
            radius=float(radius.detach().cpu()),
            planes=planes,
            contact_margin=1e-9,
        )

    def event(frame_index: int, phase: str, contact: dict[str, Any]) -> dict[str, Any]:
        plane = contact["plane"]
        return {
            "event_type": "bouncy_platform_plane_contact",
            "frame_index": int(frame_index),
            "phase": phase,
            "plane_id": str(plane.get("plane_id") or ""),
            "support_object_id": str(plane.get("object_id") or ""),
        }

    for frame_offset, frame_index in enumerate(frames):
        positions.append(x.clone())
        velocities.append(v.clone())
        if frame_offset + 1 >= len(frames):
            break
        next_frame = int(frames[frame_offset + 1])
        dt = x.new_tensor((next_frame - int(frame_index)) * friction_swr.PHYSIONPP_PHYSICS_DT_SEC)

        if active_support is not None:
            active_plane_id = str(active_support["plane"].get("plane_id") or "")
            other_planes = [plane for plane in planes if str(plane.get("plane_id") or "") != active_plane_id]
            collision = friction_swr._first_bounded_plane_contact(
                x=x,
                v=v,
                dt=dt,
                radius=radius,
                gravity=gravity,
                planes=other_planes,
            )
            support_plane = friction_swr._torch_plane_from_runtime(
                active_support,
                device=x.device,
                dtype=x.dtype,
            )
            support_normal = support_plane["normal"] / torch.clamp(
                torch.linalg.norm(support_plane["normal"]),
                min=x.new_tensor(1e-12),
            )
            if collision is not None:
                contact_time = collision["contact_time"]
                x, v = analytic_swr_common.advance_on_plane_torch(
                    x=x,
                    v=v,
                    dt=contact_time,
                    radius=radius,
                    plane_point=support_plane["surface_point"],
                    normal=support_normal,
                    gravity=gravity,
                    friction=friction,
                )
                x, v, _normal = _reflect_from_plane(
                    x=x,
                    v=v,
                    radius=radius,
                    contact=collision,
                    restitution=restitution,
                )
                remaining = torch.clamp(dt - contact_time, min=0.0)
                x, v = analytic_swr_common.advance_on_plane_torch(
                    x=x,
                    v=v,
                    dt=remaining,
                    radius=radius,
                    plane_point=support_plane["surface_point"],
                    normal=support_normal,
                    gravity=gravity,
                    friction=friction,
                )
                events.append(event(next_frame, "support_motion_to_plane_bounce", collision))
                continue

            x_next, v_next = analytic_swr_common.advance_on_plane_torch(
                x=x,
                v=v,
                dt=dt,
                radius=radius,
                plane_point=support_plane["surface_point"],
                normal=support_normal,
                gravity=gravity,
                friction=friction,
            )
            refreshed = refresh_contact(x_next, active_support)
            if refreshed is not None:
                x, v = x_next, v_next
                active_support = refreshed
                continue
            next_support = select_contact(x_next)
            if next_support is not None:
                next_plane = friction_swr._torch_plane_from_runtime(
                    next_support,
                    device=x.device,
                    dtype=x.dtype,
                )
                next_normal = next_plane["normal"] / torch.clamp(
                    torch.linalg.norm(next_plane["normal"]),
                    min=x.new_tensor(1e-12),
                )
                x = friction_swr._project_to_contact(
                    x=x_next,
                    dynamic_radius=radius,
                    plane_point=next_plane["surface_point"],
                    normal=next_normal,
                )
                v = v_next - torch.sum(v_next * next_normal) * next_normal
                active_support = next_support
                events.append(event(next_frame, "bounded_plane_contact_switch", next_support))
            else:
                x, v = x_next, v_next
                active_support = None
                events.append(
                    {
                        "event_type": "bouncy_platform_plane_contact",
                        "frame_index": next_frame,
                        "phase": "support_to_free_flight",
                    }
                )
            continue

        contact = friction_swr._first_bounded_plane_contact(
            x=x,
            v=v,
            dt=dt,
            radius=radius,
            gravity=gravity,
            planes=planes,
        )
        if contact is None:
            x = x + v * dt + 0.5 * gravity * dt * dt
            v = v + gravity * dt
            continue

        contact_time = contact["contact_time"]
        x_hit = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
        v_hit = v + gravity * contact_time
        x_hit, v_hit, normal = _reflect_from_plane(
            x=x_hit,
            v=v_hit,
            radius=radius,
            contact=contact,
            restitution=restitution,
        )
        remaining = torch.clamp(dt - contact_time, min=0.0)
        outgoing_normal_speed = torch.sum(v_hit * normal)
        if bool((outgoing_normal_speed > BOUNCE_SPEED_EPS_M_PER_S).detach().cpu()):
            x = x_hit + v_hit * remaining + 0.5 * gravity * remaining * remaining
            v = v_hit + gravity * remaining
            events.append(event(next_frame, "free_flight_to_plane_bounce", contact))
        else:
            plane = friction_swr._torch_plane_from_runtime(contact, device=x.device, dtype=x.dtype)
            x, v = analytic_swr_common.advance_on_plane_torch(
                x=x_hit,
                v=v_hit,
                dt=remaining,
                radius=radius,
                plane_point=plane["surface_point"],
                normal=normal,
                gravity=gravity,
                friction=friction,
            )
            active_support = contact
            events.append(event(next_frame, "free_flight_to_support_contact", contact))

    return torch.stack(positions), torch.stack(velocities), events


def _optimize(
    *,
    frames: list[int],
    target_positions_np: np.ndarray,
    v0_init: np.ndarray,
    radius: float,
    planes: list[dict[str, Any]],
    gravity_direction_np: np.ndarray,
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    dtype = torch.float64
    target = torch.tensor(target_positions_np, dtype=dtype)
    start_position = target[0]
    radius_tensor = target.new_tensor(radius)
    gravity_direction = torch.tensor(gravity_direction_np, dtype=dtype)
    velocity = torch.tensor(v0_init, dtype=dtype, requires_grad=True)
    friction_raw = torch.tensor(
        friction_swr._raw_from_unit_interval(INITIAL_FRICTION, friction_swr.MIN_FRICTION, friction_swr.MAX_FRICTION),
        dtype=dtype,
        requires_grad=True,
    )
    restitution_raw = torch.tensor(
        friction_swr._raw_from_unit_interval(INITIAL_RESTITUTION, MIN_RESTITUTION, MAX_RESTITUTION),
        dtype=dtype,
        requires_grad=True,
    )
    parameters = [velocity, friction_raw, restitution_raw]

    def rollout() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], torch.Tensor, torch.Tensor]:
        friction = friction_swr.MIN_FRICTION + (
            friction_swr.MAX_FRICTION - friction_swr.MIN_FRICTION
        ) * torch.sigmoid(friction_raw)
        restitution = torch.sigmoid(restitution_raw)
        predicted, predicted_velocities, events = _rollout_bouncy_platform(
            frames=frames,
            start_position=start_position,
            initial_velocity=velocity,
            friction=friction,
            restitution=restitution,
            radius=radius_tensor,
            planes=planes,
            gravity_direction=gravity_direction,
        )
        return predicted, predicted_velocities, events, friction, restitution

    def evaluate(train_end: int, validation_end: int) -> dict[str, Any]:
        predicted, _velocities, _events, _friction, _restitution = rollout()
        prefix_rmse = torch.sqrt(torch.mean(torch.sum((predicted[:train_end] - target[:train_end]) ** 2, dim=1)))
        validation_rmse = torch.sqrt(
            torch.mean(torch.sum((predicted[:validation_end] - target[:validation_end]) ** 2, dim=1))
        )
        if not bool(torch.isfinite(prefix_rmse).detach().cpu()):
            raise RuntimeError("bouncy-platform optimization produced a non-finite loss")
        return {
            "loss": prefix_rmse,
            "selection_metric": float(validation_rmse.detach().cpu()),
            "prefix_rmse": float(prefix_rmse.detach().cpu()),
        }

    schedule = analytic_swr_common.optimize_adam_prefix_curriculum(
        parameters=parameters,
        evaluate=evaluate,
        observed_frames=len(frames),
        prefix_step_frames=0,
        steps_per_prefix=steps,
        patience=patience,
        lr=lr,
    )
    predicted, predicted_velocities, events, friction, restitution = rollout()
    rmse = torch.sqrt(torch.mean(torch.sum((predicted - target) ** 2, dim=1)))
    return {
        "rmse": float(rmse.detach().cpu()),
        "initial_velocity": velocity.detach().cpu().numpy().astype(float),
        "friction": float(friction.detach().cpu()),
        "restitution": float(restitution.detach().cpu()),
        "predicted": predicted.detach().cpu().numpy().astype(float),
        "terminal_velocity": predicted_velocities[-1].detach().cpu().numpy().astype(float),
        "events": events,
        "optimizer_schedule": schedule,
    }


def _serializable_plane(plane: dict[str, Any]) -> dict[str, Any]:
    return {
        "plane_id": str(plane.get("plane_id") or ""),
        "object_id": str(plane.get("object_id") or ""),
        "normal": np.asarray(plane["normal_np"], dtype=np.float64).astype(float).tolist(),
        "surface_point": np.asarray(plane["surface_point_np"], dtype=np.float64).astype(float).tolist(),
        "area": float(plane.get("area") or 0.0),
        "triangles": np.asarray(plane["triangles_np"], dtype=np.float64).astype(float).tolist(),
    }


def run_physionpp_bouncy_platform_sphere_sysid(
    *,
    world_modeling_dir: Path,
    output_path: Path,
    output_dir: Path,
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    manifest = sysid_common.build_manifest_from_world_modeling(world_modeling_dir)
    manifest["mode"] = "physionpp_bouncy_platform_offline_sysid"
    manifest["rollout"] = {
        "backend": "swr_backend.platform_bounce_sphere",
        "target_pose_field": "corrected_pose_4x4",
        "target_translation_field": "position_camera",
        "rotation_loss": "disabled",
    }
    manifest_path = output_dir / "physics_alignment_manifest.json"
    _write_json(manifest_path, manifest)
    target_all = sysid_common._target_records_from_swr(manifest)
    agent_id, static_ids = friction_swr._select_agent_and_statics(manifest, target_all)
    target = {object_id: target_all[object_id] for object_id in [agent_id, *static_ids]}
    records = sorted(target[agent_id], key=lambda record: int(record["frame_index"]))
    frames = [int(record["frame_index"]) for record in records]
    if len(frames) < 3:
        raise ValueError("bouncy-platform SWR requires at least three dynamic trajectory frames")
    if any(next_frame <= frame for frame, next_frame in zip(frames, frames[1:])):
        raise ValueError("bouncy-platform trajectory frames must be strictly increasing")
    target_positions_np = np.asarray([record["position"] for record in records], dtype=np.float64)
    radius = friction_swr._initial_radius_by_object(
        manifest=manifest,
        object_ids=[agent_id],
    )[agent_id]
    gravity_direction_np = friction_swr._gravity_direction_blender_world(manifest)
    static_meshes = friction_swr._static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
    bounded_payload = friction_swr._decompose_static_meshes_into_bounded_planes(
        static_meshes=static_meshes,
        gravity_direction=gravity_direction_np,
    )
    bounded_debug_path = output_dir / "bounded_static_planes_debug.json"
    _write_json(bounded_debug_path, bounded_payload)
    runtime_payload = dict(bounded_payload)
    runtime_payload["_static_meshes_runtime"] = static_meshes
    planes = friction_swr._bounded_planes_runtime(runtime_payload)
    if not planes:
        raise ValueError("bouncy-platform static meshes produced no bounded planes")
    initialization_records = records[: min(len(records), INITIAL_VELOCITY_WINDOW_FRAMES)]
    v0_init = friction_swr._fit_initial_velocity(
        initialization_records,
        friction_swr.PHYSIONPP_PHYSICS_DT_SEC,
    )
    optimized = _optimize(
        frames=frames,
        target_positions_np=target_positions_np,
        v0_init=v0_init,
        radius=float(radius),
        planes=planes,
        gravity_direction_np=gravity_direction_np,
        lr=lr,
        steps=steps,
        patience=patience,
    )
    simulated = friction_swr._result_trajectories(
        target=target,
        agent_id=agent_id,
        static_ids=static_ids,
        frames=frames,
        predicted=optimized["predicted"],
    )
    event_payload = []
    for item in optimized["events"]:
        payload = dict(item)
        payload["dynamic_object_id"] = agent_id
        support_id = str(payload.get("support_object_id") or "")
        payload["object_ids"] = [agent_id, support_id] if support_id else [agent_id]
        event_payload.append(payload)
    result = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.platform_bounce_sphere",
        "mode": "trajectory_informed_physics_alignment",
        "message": "Physion++ bouncy platform analytic bounded-plane sphere SWR completed",
        "source_world_modeling_dir": str(world_modeling_dir),
        "target_trajectories": target,
        "trajectory_physics_initialization": {
            "applied": True,
            "source": "leading_trajectory_constant_acceleration_fit",
            "physics_dt_sec": friction_swr.PHYSIONPP_PHYSICS_DT_SEC,
            "frame_count": len(initialization_records),
            "objects": [
                {
                    "object_id": agent_id,
                    "initial_velocity_blender_world_m_per_s": v0_init.astype(float).tolist(),
                    "initial_sliding_friction": INITIAL_FRICTION,
                    "initial_restitution": INITIAL_RESTITUTION,
                }
            ],
        },
        "physics_rollout": {
            "simulator": "swr_backend.platform_bounce_sphere",
            "contact_proxy_policy": "fixed_dynamic_3d_sphere_all_bounded_static_planes",
            "simulated_trajectories": simulated,
            "proxy_dimensions_by_object": {agent_id: [2.0 * float(radius)] * 3},
            "sphere_radius_3d_by_object": {agent_id: float(radius)},
            "sphere_radius_init_3d_by_object": {agent_id: float(radius)},
            "static_mesh_contact_objects": [
                {
                    "object_id": item["object_id"],
                    "mesh_path": item["mesh_path"],
                    "triangle_count": int(item["triangle_count"]),
                }
                for item in static_meshes
            ],
            "bounded_static_planes": {
                "status": bounded_payload.get("status"),
                "method": bounded_payload.get("method"),
                "plane_count": int(len(planes)),
                "per_object_plane_count": {
                    str(item.get("object_id")): int(item.get("plane_count") or 0)
                    for item in bounded_payload.get("objects", [])
                    if isinstance(item, dict)
                },
                "debug_path": str(bounded_debug_path),
                "contact_detection_policy": "all_bounded_planes",
            },
            "contact_plane_geometry": [_serializable_plane(plane) for plane in planes],
            "contact_events": event_payload,
            "terminal_state": {
                "frame_index": frames[-1],
                "position_blender_world_m": optimized["predicted"][-1].astype(float).tolist(),
                "velocity_blender_world_m_per_s": optimized["terminal_velocity"].astype(float).tolist(),
            },
        },
        "alignment_optimization": {
            "strategy": "swr_fit.bounded_planes_full_trajectory",
            "optimizer": {
                "method": "torch_adam_full_trajectory",
                "lr": float(lr),
                "steps": int(steps),
                "early_stop_patience": int(patience),
                "curriculum_enabled": False,
                "schedule": optimized["optimizer_schedule"],
                "physics_dt_sec": friction_swr.PHYSIONPP_PHYSICS_DT_SEC,
            },
            "optimization_target": {
                "object_id": agent_id,
                "loss_name": "dynamic_object_rmse_m",
                "loss_space": "world_3d",
                "frame_count": len(frames),
            },
            "best_parameters": {
                agent_id: {
                    "initial_velocity_blender_world_m_per_s": optimized["initial_velocity"].astype(float).tolist(),
                    "sliding_friction": float(optimized["friction"]),
                    "restitution": float(optimized["restitution"]),
                    "optimized_radius_m": float(radius),
                    "radius_fixed": True,
                    "gravity_m_per_s2": friction_swr.GRAVITY_M_PER_S2,
                    "gravity_fixed": True,
                }
            },
            "dynamic_object_rmse_m": float(optimized["rmse"]),
        },
        "dynamic_object_rmse_m": float(optimized["rmse"]),
    }
    _write_json(output_path, result)
    _write_json(
        output_dir / "summary.json",
        {
            "status": "ok",
            "backend": result["backend"],
            "agent_object_id": agent_id,
            "static_object_ids": static_ids,
            "dynamic_object_rmse_m": float(optimized["rmse"]),
            "parameters": result["alignment_optimization"]["best_parameters"],
            "bounded_static_planes": result["physics_rollout"]["bounded_static_planes"],
        },
    )
    return result


def _run_learning_rate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(max(int(payload["worker_threads"]), 1))
    output_dir = Path(payload["output_dir"])
    output_path = output_dir / "physics_alignment.json"
    result = run_physionpp_bouncy_platform_sphere_sysid(
        world_modeling_dir=Path(payload["world_modeling_dir"]),
        output_path=output_path,
        output_dir=output_dir,
        lr=float(payload["lr"]),
        steps=int(payload["steps"]),
        patience=int(payload["patience"]),
    )
    return {
        "lr": float(payload["lr"]),
        "dynamic_object_rmse_m": float(result["dynamic_object_rmse_m"]),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physion++ bouncy-platform analytic sphere SWR.")
    parser.add_argument("--world-modeling-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lrs", default=",".join(str(value) for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--workers", type=int, default=len(DEFAULT_LEARNING_RATES))
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--render-video", action="store_true")
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
            "steps": int(args.steps),
            "patience": int(args.patience),
            "worker_threads": int(args.worker_threads),
        }
        for lr in learning_rates
    ]
    trials, worker_count = analytic_swr_common.run_parallel_trials(
        payloads=payloads,
        worker=_run_learning_rate_trial,
        max_workers=int(args.workers),
    )
    trials.sort(key=lambda item: float(item["lr"]))
    selected = min(trials, key=lambda item: float(item["dynamic_object_rmse_m"]))
    result = _load_json(Path(selected["output_path"]))
    result["alignment_optimization"]["optimizer"]["multi_learning_rate"] = {
        "learning_rates": learning_rates,
        "worker_count": int(worker_count),
        "selected_learning_rate": float(selected["lr"]),
        "trials": trials,
    }
    _write_json(output_path, result)
    _write_json(
        output_dir / "summary.json",
        {
            "status": result.get("status"),
            "backend": result.get("backend"),
            "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
            "parameters": (result.get("alignment_optimization") or {}).get("best_parameters"),
            "multi_learning_rate": result["alignment_optimization"]["optimizer"]["multi_learning_rate"],
        },
    )
    debug_video = None
    if args.render_video:
        manifest = _load_json(Path(selected["output_path"]).parent / "physics_alignment_manifest.json")
        debug_video = friction_swr._render_bounded_plane_camera_debug(
            manifest=manifest,
            result=result,
            output_dir=output_dir,
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "backend": result.get("backend"),
                "output": str(output_path),
                "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
                "debug_video": str(debug_video) if debug_video is not None else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
