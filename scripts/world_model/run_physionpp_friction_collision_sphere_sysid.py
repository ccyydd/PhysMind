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


PHYSIONPP_PHYSICS_DT_SEC = friction_swr.PHYSIONPP_PHYSICS_DT_SEC
DEFAULT_LEARNING_RATES = friction_swr.DEFAULT_LEARNING_RATES
DEFAULT_STEPS = 400
DEFAULT_PATIENCE = 100
MIN_FRICTION = friction_swr.MIN_FRICTION
MAX_FRICTION = friction_swr.MAX_FRICTION
DEFAULT_PATIENT_FRICTION = 0.1
DEFAULT_MASS_RATIO = 1.0
DEFAULT_RESTITUTION = 0.5
MAX_INITIAL_SPEED_M_PER_S = 20.0
RADIUS_PRIOR_WEIGHT = friction_swr.RADIUS_PRIOR_WEIGHT


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pose_context(
    world_modeling_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    context, support, roles, split = sysid_common.load_collision_pose_context(
        world_modeling_dir,
        context_key="physion_pp_friction_collision",
        context_required_error=(
            "friction_collision SWR requires "
            "physion_pp_friction_collision pose context"
        ),
        support_required_error=(
            "friction_collision SWR requires the corrected support-plane direction"
        ),
        bindings_required_error=(
            "friction_collision pose context has no role bindings or segment split"
        ),
    )

    segments: list[dict[str, Any]] = []
    for segment_name in ("seg1", "seg2"):
        frame_range = split.get(segment_name)
        agent_id = roles.get(f"{segment_name}_agent")
        patient_id = roles.get(f"{segment_name}_patient")
        if not isinstance(frame_range, list) or len(frame_range) != 2 or not agent_id or not patient_id:
            raise ValueError(f"incomplete friction_collision role binding for {segment_name}")
        segments.append(
            {
                "segment": segment_name,
                "agent_object_id": str(agent_id),
                "patient_object_id": str(patient_id),
                "frame_range": [int(frame_range[0]), int(frame_range[1])],
            }
        )
    return segments, context, support


def _role_radius_profile(
    segments: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    role_object_ids = {
        role: [str(segment[f"{role}_object_id"]) for segment in segments]
        for role in ("agent", "patient")
    }
    object_role_assignments = [
        (str(segment[f"{role}_object_id"]), role)
        for segment in segments
        for role in ("agent", "patient")
    ]
    return role_object_ids, object_role_assignments


def _collision_geometry_object_ids(
    segments: list[dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(segment[key])
            for segment in segments
            for key in ("agent_object_id", "patient_object_id")
        }
    )


def _prepare_segments(
    *,
    target: dict[str, list[dict[str, Any]]],
    segments: list[dict[str, Any]],
    radius_by_object: dict[str, float],
    support: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    moving_seg2_patient = bool((context.get("seg2_patient_motion") or {}).get("moving"))
    for segment in segments:
        agent_id = str(segment["agent_object_id"])
        patient_id = str(segment["patient_object_id"])
        agent_records = sysid_common.records_in_frame_range(
            target, agent_id, segment["frame_range"]
        )
        patient_records = sysid_common.records_in_frame_range(
            target, patient_id, segment["frame_range"]
        )
        agent_by_frame = {int(record["frame_index"]): record for record in agent_records}
        patient_by_frame = {int(record["frame_index"]): record for record in patient_records}
        frames = sorted(set(agent_by_frame) & set(patient_by_frame))
        if len(frames) < 2:
            raise ValueError(f"{segment['segment']} has fewer than two common trajectory frames")
        terminal_frame = int(frames[-1])
        target_world = np.stack(
            [
                np.stack(
                    [
                        np.asarray(agent_by_frame[frame]["position"], dtype=np.float64),
                        np.asarray(patient_by_frame[frame]["position"], dtype=np.float64),
                    ],
                    axis=0,
                )
                for frame in frames
            ],
            axis=0,
        )
        patient_moves = segment["segment"] == "seg2" and moving_seg2_patient
        center_distance = np.linalg.norm(target_world[:, 0] - target_world[:, 1], axis=1)
        contact_residual = center_distance - (
            float(radius_by_object[agent_id]) + float(radius_by_object[patient_id])
        )
        prepared.append(
            {
                **segment,
                "frames": frames,
                "target_world_np": target_world,
                "terminal_pose_blender_world_4x4": {
                    agent_id: np.asarray(
                        sysid_common._opencv_pose_to_blender_world(
                            agent_by_frame[terminal_frame]["pose_4x4"]
                        ),
                        dtype=np.float64,
                    ),
                    patient_id: np.asarray(
                        sysid_common._opencv_pose_to_blender_world(
                            patient_by_frame[terminal_frame]["pose_4x4"]
                        ),
                        dtype=np.float64,
                    ),
                },
                "agent_v0_init": sysid_common.fit_initial_velocity(
                    frames,
                    target_world[:, 0],
                    physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                ),
                "patient_v0_init": (
                    sysid_common.fit_initial_velocity(
                        frames,
                        target_world[:, 1],
                        physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                    )
                    if patient_moves
                    else np.zeros(3, dtype=np.float64)
                ),
                "agent_friction_init": sysid_common.fit_tangential_friction(
                    frames=frames,
                    positions=target_world[:, 0],
                    up=support["up"],
                    physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                    gravity_magnitude=friction_swr.GRAVITY_M_PER_S2,
                    min_friction=MIN_FRICTION,
                    max_friction=MAX_FRICTION,
                ),
                "patient_moves": patient_moves,
                "patient_displacement_m": float(
                    np.max(np.linalg.norm(target_world[:, 1] - target_world[0, 1], axis=1))
                ),
                "minimum_target_contact_residual_m": float(np.min(contact_residual)),
                "observed_collision": any(
                    float(contact_residual[index - 1]) > 0.0 and float(contact_residual[index]) <= 0.0
                    for index in range(1, len(contact_residual))
                ),
            }
        )
    return prepared


def _rollout_sphere(
    *,
    frames: list[int],
    start_position: torch.Tensor,
    velocity: torch.Tensor,
    friction: torch.Tensor,
    radius: torch.Tensor,
    plane_point: torch.Tensor,
    support: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return friction_swr._rollout_plane_analytic(
        frames=frames,
        physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
        start_position=start_position,
        velocity=velocity,
        friction=friction,
        dynamic_radius=radius,
        plane_point=plane_point,
        plane_normal=torch.tensor(support["up"], dtype=torch.float64),
        gravity_direction=torch.tensor(support["gravity_direction"], dtype=torch.float64),
        gravity_magnitude=torch.tensor(friction_swr.GRAVITY_M_PER_S2, dtype=torch.float64),
    )


def _run_rollouts(
    *,
    prepared: list[dict[str, Any]],
    support: dict[str, Any],
    shared_radii: dict[str, float],
    agent_shared_speed_raw: torch.Tensor,
    agent_direction_angles: list[torch.Tensor],
    patient_velocity_raw: list[torch.Tensor | None],
    agent_friction_raw: torch.Tensor,
    agent_radius_raw: torch.Tensor,
    patient_radius_raw: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, torch.Tensor]]:
    agent_friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(agent_friction_raw)
    agent_shared_speed = torch.nn.functional.softplus(agent_shared_speed_raw)
    patient_friction = agent_friction.new_tensor(DEFAULT_PATIENT_FRICTION)
    agent_radius, agent_radius_log_scale = friction_swr._radius_from_raw(
        radius_init_tensor=agent_friction.new_tensor(shared_radii["agent"]),
        radius_raw=agent_radius_raw,
    )
    patient_radius, patient_radius_log_scale = friction_swr._radius_from_raw(
        radius_init_tensor=agent_friction.new_tensor(shared_radii["patient"]),
        radius_raw=patient_radius_raw,
    )
    plane_point = torch.tensor(support["plane_point"], dtype=torch.float64)
    segment_results: list[dict[str, Any]] = []
    segment_losses: list[torch.Tensor] = []
    for segment_index, item in enumerate(prepared):
        target_world = torch.tensor(item["target_world_np"], dtype=torch.float64)
        azimuth, elevation = agent_direction_angles[segment_index].unbind()
        agent_velocity = agent_shared_speed * torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ]
        )
        patient_raw = patient_velocity_raw[segment_index]
        patient_velocity = (
            MAX_INITIAL_SPEED_M_PER_S * torch.tanh(patient_raw)
            if patient_raw is not None
            else torch.zeros_like(agent_velocity)
        )
        predicted_agent, agent_plane_events = _rollout_sphere(
            frames=item["frames"],
            start_position=target_world[0, 0],
            velocity=agent_velocity,
            friction=agent_friction,
            radius=agent_radius,
            plane_point=plane_point,
            support=support,
        )
        predicted_patient, patient_plane_events = _rollout_sphere(
            frames=item["frames"],
            start_position=target_world[0, 1],
            velocity=patient_velocity,
            friction=patient_friction,
            radius=patient_radius,
            plane_point=plane_point,
            support=support,
        )
        predicted_world = torch.stack([predicted_agent, predicted_patient], dim=1)
        squared_distance = torch.sum((predicted_world - target_world) ** 2, dim=-1)
        segment_loss = torch.mean(squared_distance)
        segment_losses.append(segment_loss)
        segment_results.append(
            {
                "predicted_world": predicted_world,
                "squared_distance": squared_distance,
                "agent_velocity": agent_velocity,
                "patient_velocity": patient_velocity,
                "agent_plane_events": agent_plane_events,
                "patient_plane_events": patient_plane_events,
            }
        )
    trajectory_loss = torch.mean(torch.stack(segment_losses))
    loss = trajectory_loss + agent_friction.new_tensor(RADIUS_PRIOR_WEIGHT) * (
        agent_radius_log_scale.square() + patient_radius_log_scale.square()
    )
    return loss, segment_results, {
        "agent_friction": agent_friction,
        "patient_friction": patient_friction,
        "agent_shared_speed": agent_shared_speed,
        "agent_radius": agent_radius,
        "patient_radius": patient_radius,
        "agent_radius_log_scale": agent_radius_log_scale,
        "patient_radius_log_scale": patient_radius_log_scale,
        "trajectory_loss": trajectory_loss,
    }


def _optimize(
    *,
    prepared: list[dict[str, Any]],
    support: dict[str, Any],
    shared_radii: dict[str, float],
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    agent_speed_candidates = [float(np.linalg.norm(item["agent_v0_init"])) for item in prepared]
    agent_shared_speed_init = max(float(np.median(agent_speed_candidates)), 1e-4)
    agent_shared_speed_raw = torch.tensor(
        math.log(math.expm1(agent_shared_speed_init)),
        dtype=torch.float64,
        requires_grad=True,
    )
    fallback_direction = np.asarray(
        max(prepared, key=lambda item: float(np.linalg.norm(item["agent_v0_init"])))["agent_v0_init"],
        dtype=np.float64,
    )
    agent_direction_angles = [
        sysid_common.direction_angles_from_value(
            item["agent_v0_init"],
            fallback_direction,
        )
        for item in prepared
    ]
    patient_velocity_raw = [
        sysid_common.velocity_raw_from_value(
            item["patient_v0_init"],
            maximum_speed_m_per_s=MAX_INITIAL_SPEED_M_PER_S,
        )
        if item["patient_moves"]
        else None
        for item in prepared
    ]
    agent_friction_init = float(np.median([item["agent_friction_init"] for item in prepared]))
    agent_friction_raw = torch.tensor(
        sysid_common.raw_from_interval(
            agent_friction_init,
            MIN_FRICTION,
            MAX_FRICTION,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    agent_radius_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    patient_radius_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    parameters: list[torch.Tensor] = [
        agent_shared_speed_raw,
        *agent_direction_angles,
        *[value for value in patient_velocity_raw if value is not None],
        agent_friction_raw,
        agent_radius_raw,
        patient_radius_raw,
    ]
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    best_loss = float("inf")
    best_state: list[torch.Tensor] | None = None
    best_step = -1
    stale_steps = 0
    executed_steps = 0
    history: list[dict[str, float | int]] = []
    for step in range(max(int(steps), 1)):
        executed_steps = step + 1
        optimizer.zero_grad(set_to_none=True)
        loss, _segments, _physics = _run_rollouts(
            prepared=prepared,
            support=support,
            shared_radii=shared_radii,
            agent_shared_speed_raw=agent_shared_speed_raw,
            agent_direction_angles=agent_direction_angles,
            patient_velocity_raw=patient_velocity_raw,
            agent_friction_raw=agent_friction_raw,
            agent_radius_raw=agent_radius_raw,
            patient_radius_raw=patient_radius_raw,
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            break
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if step == 0 or (step + 1) % 25 == 0:
            history.append({"step": int(step + 1), "loss": loss_value})
        if loss_value + 1e-12 < best_loss:
            best_loss = loss_value
            best_state = [parameter.detach().clone() for parameter in parameters]
            best_step = step + 1
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= max(int(patience), 1):
            break
    if best_state is None:
        raise RuntimeError("friction_collision optimization produced no finite state")
    with torch.no_grad():
        for parameter, best_value in zip(parameters, best_state):
            parameter.copy_(best_value)
    loss, segment_results, physics = _run_rollouts(
        prepared=prepared,
        support=support,
        shared_radii=shared_radii,
        agent_shared_speed_raw=agent_shared_speed_raw,
        agent_direction_angles=agent_direction_angles,
        patient_velocity_raw=patient_velocity_raw,
        agent_friction_raw=agent_friction_raw,
        agent_radius_raw=agent_radius_raw,
        patient_radius_raw=patient_radius_raw,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "trajectory_loss": float(physics["trajectory_loss"].detach().cpu()),
        "best_step": int(best_step),
        "executed_steps": int(executed_steps),
        "history": history,
        "segments": segment_results,
        "physics": physics,
        "initialization": {
            "agent_friction": agent_friction_init,
            "patient_friction": DEFAULT_PATIENT_FRICTION,
            "agent_shared_initial_speed_m_per_s": agent_shared_speed_init,
            "agent_segment_initial_speed_candidates_m_per_s": agent_speed_candidates,
            "agent_radius_m": float(shared_radii["agent"]),
            "patient_radius_m": float(shared_radii["patient"]),
        },
    }


def _simulated_trajectories(
    *,
    item: dict[str, Any],
    predicted_world: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for object_index, object_id in enumerate((item["agent_object_id"], item["patient_object_id"])):
        output[str(object_id)] = [
            {
                "frame_index": int(frame_index),
                "position_blender_world_m": position.astype(float).tolist(),
                "position_opencv_camera_m": sysid_common._blender_vector_to_opencv_camera(
                    position.astype(float).tolist()
                ),
            }
            for frame_index, position in zip(item["frames"], predicted_world[:, object_index])
        ]
    return output


def _target_trajectories(item: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for object_index, object_id in enumerate((item["agent_object_id"], item["patient_object_id"])):
        output[str(object_id)] = [
            {
                "frame_index": int(frame_index),
                "position_blender_world_m": position.astype(float).tolist(),
            }
            for frame_index, position in zip(item["frames"], item["target_world_np"][:, object_index])
        ]
    return output


def _plane_events_with_role(events: list[dict[str, Any]], role: str, object_id: str) -> list[dict[str, Any]]:
    return [
        {
            **event,
            "dynamic_role": role,
            "dynamic_object_id": object_id,
            "support_role": f"{role}_proxy_support_plane",
        }
        for event in events
    ]


def run_physionpp_friction_collision_sphere_sysid(
    *,
    world_modeling_dir: Path,
    output_path: Path,
    output_dir: Path,
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    manifest = sysid_common.build_manifest_from_world_modeling(world_modeling_dir)
    manifest["mode"] = "physionpp_friction_collision_offline_sysid"
    manifest.setdefault("rollout", {})["backend"] = "swr_backend.collision_friction_spheres"
    target = sysid_common._target_records_from_swr(manifest)
    segments, context, support_context = _pose_context(world_modeling_dir)
    role_object_ids, object_role_assignments = _role_radius_profile(segments)
    fallback_radius_by_object = friction_swr._initial_radius_by_object(
        manifest=manifest,
        object_ids=[object_id for object_id, _role in object_role_assignments],
    )
    shared_radii, radius_by_object = sysid_common.shared_role_radii_from_mesh_envelopes(
        role_object_ids=role_object_ids,
        object_role_assignments=object_role_assignments,
        fallback_radius_by_object=fallback_radius_by_object,
        object_specs=friction_swr._object_specs(manifest),
        mesh_loader=friction_swr._mesh_from_path,
        envelope_scale=sysid_common.COLLISION_INITIAL_RADIUS_ENVELOPE_SCALE,
    )
    collision_geometry = sysid_common.collision_oriented_box_geometry_by_object(
        object_specs=friction_swr._object_specs(manifest),
        object_ids=_collision_geometry_object_ids(segments),
        mesh_loader=friction_swr._mesh_from_path,
    )
    support = sysid_common.collision_support_geometry(
        manifest=manifest,
        segments=segments,
        support_context=support_context,
    )
    prepared = _prepare_segments(
        target=target,
        segments=segments,
        radius_by_object=radius_by_object,
        support=support,
        context=context,
    )
    optimized = _optimize(
        prepared=prepared,
        support=support,
        shared_radii=shared_radii,
        lr=lr,
        steps=steps,
        patience=patience,
    )
    physics = optimized["physics"]
    optimized_radii = {
        "agent": float(physics["agent_radius"].detach().cpu()),
        "patient": float(physics["patient_radius"].detach().cpu()),
    }
    segment_payloads: list[dict[str, Any]] = []
    all_squared_distances: list[np.ndarray] = []
    for item, segment_result in zip(prepared, optimized["segments"]):
        predicted_world = segment_result["predicted_world"].detach().cpu().numpy()
        squared_distance = segment_result["squared_distance"].detach().cpu().numpy()
        all_squared_distances.append(squared_distance.reshape(-1))
        object_rmse = {
            str(item["agent_object_id"]): float(np.sqrt(np.mean(squared_distance[:, 0]))),
            str(item["patient_object_id"]): float(np.sqrt(np.mean(squared_distance[:, 1]))),
        }
        simulated = _simulated_trajectories(item=item, predicted_world=predicted_world)
        agent_id = str(item["agent_object_id"])
        patient_id = str(item["patient_object_id"])
        predicted_residual = np.linalg.norm(predicted_world[:, 0] - predicted_world[:, 1], axis=1) - (
            optimized_radii["agent"] + optimized_radii["patient"]
        )
        segment_payloads.append(
            {
                "segment": item["segment"],
                "status": "ok",
                "agent_object_id": agent_id,
                "patient_object_id": patient_id,
                "frame_range": [int(item["frames"][0]), int(item["frames"][-1])],
                "frame_count": int(len(item["frames"])),
                "dynamic_object_rmse_m": float(np.sqrt(np.mean(squared_distance))),
                "object_rmse_m": object_rmse,
                "patient_motion_mode": "gravity_driven" if item["patient_moves"] else "stationary",
                "observed_collision": bool(item["observed_collision"]),
                "minimum_target_contact_residual_m": float(item["minimum_target_contact_residual_m"]),
                "minimum_simulated_contact_residual_m": float(np.min(predicted_residual)),
                "patient_target_displacement_m": float(item["patient_displacement_m"]),
                "target_trajectories": _target_trajectories(item),
                "observed_boundary_state": {
                    agent_id: {
                        "frame_index": int(item["frames"][-1]),
                        "position_blender_world_m": item["target_world_np"][-1, 0].astype(float).tolist(),
                        "pose_blender_world_4x4": item["terminal_pose_blender_world_4x4"][agent_id]
                        .astype(float)
                        .tolist(),
                    },
                    patient_id: {
                        "frame_index": int(item["frames"][-1]),
                        "position_blender_world_m": item["target_world_np"][-1, 1].astype(float).tolist(),
                        "pose_blender_world_4x4": item["terminal_pose_blender_world_4x4"][patient_id]
                        .astype(float)
                        .tolist(),
                    },
                },
                "physics_rollout": {
                    "simulator": "swr_backend.collision_friction_spheres",
                    "frame_indices": [int(frame) for frame in item["frames"]],
                    "contact_proxy_policy": "two_independent_3d_spheres_with_role_support_planes",
                    "simulated_trajectories": simulated,
                    "sphere_radius_m_by_object": {
                        agent_id: optimized_radii["agent"],
                        patient_id: optimized_radii["patient"],
                    },
                    "plane_contact_events": [
                        *_plane_events_with_role(segment_result["agent_plane_events"], "agent", agent_id),
                        *_plane_events_with_role(segment_result["patient_plane_events"], "patient", patient_id),
                    ],
                },
                "initialization": {
                    "agent_initial_velocity_blender_world_m_per_s": item["agent_v0_init"].astype(float).tolist(),
                    "patient_initial_velocity_blender_world_m_per_s": item["patient_v0_init"].astype(float).tolist(),
                    "agent_friction": float(item["agent_friction_init"]),
                },
                "optimized_initial_velocity_blender_world_m_per_s": {
                    agent_id: segment_result["agent_velocity"].detach().cpu().tolist(),
                    patient_id: segment_result["patient_velocity"].detach().cpu().tolist(),
                },
                "terminal_state": {
                    agent_id: {
                        "frame_index": int(item["frames"][-1]),
                        "position_blender_world_m": predicted_world[-1, 0].astype(float).tolist(),
                    },
                    patient_id: {
                        "frame_index": int(item["frames"][-1]),
                        "position_blender_world_m": predicted_world[-1, 1].astype(float).tolist(),
                    },
                },
            }
        )
    all_squared = np.concatenate(all_squared_distances)
    payload: dict[str, Any] = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.collision_friction_spheres",
        "mode": "trajectory_informed_physics_alignment",
        "message": "Physion++ friction-collision joint two-segment 3D analytic sphere SWR completed",
        "source_world_modeling_dir": str(world_modeling_dir),
        "dynamic_object_rmse_m": float(np.sqrt(np.mean(all_squared))),
        "segment_equal_weight_loss_m2": float(optimized["trajectory_loss"]),
        "regularized_optimization_loss": float(optimized["loss"]),
        "support_plane": {
            "source": support["source"],
            "up_direction_blender_world": support["up"].astype(float).tolist(),
            "gravity_direction_blender_world": support["gravity_direction"].astype(float).tolist(),
            "tangent_1_blender_world": support["tangent_1"].astype(float).tolist(),
            "tangent_2_blender_world": support["tangent_2"].astype(float).tolist(),
            "surface_offset_blender_world_m": float(support["plane_offset"]),
            "surface_point_blender_world_m": support["plane_point"].astype(float).tolist(),
            "shared_by_segments": support["segment_names"],
            "shared_by_roles": ["agent", "patient"],
            "pose_correction_ground_height_camera_m": support["pose_correction_ground_height_camera_m"],
        },
        "shared_role_radii_m": optimized_radii,
        "initial_shared_role_radii_m": shared_radii,
        "initial_shared_role_radius_source": "scaled_median_mesh_local_origin_to_farthest_vertex",
        "initial_radius_envelope_scale": sysid_common.COLLISION_INITIAL_RADIUS_ENVELOPE_SCALE,
        "collision_geometry_by_object": collision_geometry,
        "alignment_optimization": {
            "strategy": "swr_fit.two_segment_shared_speed_radii",
            "optimizer": {
                "method": "torch_adam_full_trajectory",
                "learning_rate": float(lr),
                "steps": int(steps),
                "early_stop_patience": int(patience),
                "best_step": int(optimized["best_step"]),
                "executed_steps": int(optimized["executed_steps"]),
                "physics_dt_sec": float(PHYSIONPP_PHYSICS_DT_SEC),
                "radius_parameterization": "radius_init_times_exp_bounded_log_scale",
                "radius_scale_range": [
                    float(math.exp(-friction_swr.MAX_RADIUS_LOG_SCALE)),
                    float(math.exp(friction_swr.MAX_RADIUS_LOG_SCALE)),
                ],
            },
            "parameter_scope": {
                "shared": [
                    "agent_ground_friction",
                    "agent_initial_speed_magnitude_m_per_s",
                    "agent_radius_m",
                    "patient_radius_m",
                ],
                "per_segment": ["agent_initial_velocity_direction"],
                "conditional": ["seg2_patient_initial_velocity_blender_world_m_per_s_if_observed_moving"],
                "fixed": [
                    "patient_ground_friction",
                    "gravity_magnitude",
                    "support_plane_direction",
                ],
            },
            "initialization": optimized["initialization"],
            "best_parameters": {
                "agent_ground_friction": float(physics["agent_friction"].detach().cpu()),
                "patient_ground_friction": float(physics["patient_friction"].detach().cpu()),
                "agent_initial_speed_magnitude_m_per_s": float(
                    physics["agent_shared_speed"].detach().cpu()
                ),
                "agent_radius_m": optimized_radii["agent"],
                "patient_radius_m": optimized_radii["patient"],
                "agent_radius_scale": optimized_radii["agent"] / shared_radii["agent"],
                "patient_radius_scale": optimized_radii["patient"] / shared_radii["patient"],
                "gravity_m_per_s2": float(friction_swr.GRAVITY_M_PER_S2),
            },
            "unidentified_collision_priors": {
                "agent_mass_over_patient_mass": DEFAULT_MASS_RATIO,
                "agent_patient_restitution": DEFAULT_RESTITUTION,
                "reason": "the observed clips end before the queried agent-patient contact",
            },
            "history": optimized["history"],
            "loss_space": "world_3d",
        },
        "physion_pp_friction_collision": context,
        "segments": segment_payloads,
    }
    _write_json(output_path, payload)
    _write_json(
        output_dir / "summary.json",
        {
            "status": "ok",
            "backend": payload["backend"],
            "dynamic_object_rmse_m": payload["dynamic_object_rmse_m"],
            "segment_equal_weight_loss_m2": payload["segment_equal_weight_loss_m2"],
            "best_parameters": payload["alignment_optimization"]["best_parameters"],
            "segments": [
                {
                    "segment": item["segment"],
                    "frame_range": item["frame_range"],
                    "dynamic_object_rmse_m": item["dynamic_object_rmse_m"],
                    "object_rmse_m": item["object_rmse_m"],
                    "patient_motion_mode": item["patient_motion_mode"],
                }
                for item in segment_payloads
            ],
        },
    )
    return payload


def _render_debug_video(result: dict[str, Any], output_path: Path) -> None:
    import cv2

    segments = [item for item in result.get("segments", []) if isinstance(item, dict) and item.get("status") == "ok"]
    if not segments:
        return
    support = result["support_plane"]
    up = np.asarray(support["up_direction_blender_world"], dtype=np.float64)
    axis_u = np.asarray(support["tangent_1_blender_world"], dtype=np.float64)
    axis_v = np.asarray(support["tangent_2_blender_world"], dtype=np.float64)
    all_positions = []
    for item in segments:
        for source in (item["target_trajectories"], item["physics_rollout"]["simulated_trajectories"]):
            for records in source.values():
                all_positions.extend(np.asarray([record["position_blender_world_m"] for record in records], dtype=np.float64))
    points = np.asarray(all_positions, dtype=np.float64).reshape(-1, 3)
    top = np.stack([points @ axis_u, points @ axis_v], axis=1)
    side = np.stack([points @ axis_u, points @ up], axis=1)

    width, height = 1120, 520
    panel_width = width // 2
    margin = 55

    def transform(values: np.ndarray, panel_index: int):
        lower = np.min(values, axis=0)
        upper = np.max(values, axis=0)
        center = 0.5 * (lower + upper)
        span = max(float(np.max(upper - lower)) * 1.2, 0.5)

        def to_pixel(point: np.ndarray) -> tuple[int, int]:
            normalized = (np.asarray(point, dtype=np.float64) - center) / span
            return (
                int(round(panel_index * panel_width + panel_width / 2 + normalized[0] * (panel_width - 2 * margin))),
                int(round(height / 2 - normalized[1] * (height - 2 * margin))),
            )

        return to_pixel, span

    top_pixel, top_span = transform(top, 0)
    side_pixel, side_span = transform(side, 1)
    colors = {
        "target_agent": (0, 150, 245),
        "target_patient": (40, 80, 220),
        "sim_agent": (65, 180, 65),
        "sim_patient": (220, 120, 40),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for item in segments:
        agent_id = str(item["agent_object_id"])
        patient_id = str(item["patient_object_id"])
        target = item["target_trajectories"]
        simulated = item["physics_rollout"]["simulated_trajectories"]
        frames = item["physics_rollout"]["frame_indices"]
        world_arrays = {
            "target_agent": np.asarray([record["position_blender_world_m"] for record in target[agent_id]]),
            "target_patient": np.asarray([record["position_blender_world_m"] for record in target[patient_id]]),
            "sim_agent": np.asarray([record["position_blender_world_m"] for record in simulated[agent_id]]),
            "sim_patient": np.asarray([record["position_blender_world_m"] for record in simulated[patient_id]]),
        }
        radii = item["physics_rollout"]["sphere_radius_m_by_object"]
        for index, frame_index in enumerate(frames):
            image = np.full((height, width, 3), 248, dtype=np.uint8)
            cv2.line(image, (panel_width, 0), (panel_width, height), (180, 180, 180), 1)
            cv2.putText(image, "top view", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
            cv2.putText(image, "side view", (panel_width + 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
            for key, world in world_arrays.items():
                projections = (
                    np.stack([world @ axis_u, world @ axis_v], axis=1),
                    np.stack([world @ axis_u, world @ up], axis=1),
                )
                object_id = agent_id if key.endswith("agent") else patient_id
                thickness = -1 if key.startswith("sim") else 2
                for panel_index, (values, pixel, span) in enumerate(
                    ((projections[0], top_pixel, top_span), (projections[1], side_pixel, side_span))
                ):
                    history = np.asarray([pixel(value) for value in values[: index + 1]], dtype=np.int32)
                    if len(history) >= 2:
                        cv2.polylines(image, [history], False, colors[key], 2, cv2.LINE_AA)
                    radius_px = max(
                        4,
                        int(round(float(radii[object_id]) / span * (panel_width - 2 * margin))),
                    )
                    cv2.circle(image, pixel(values[index]), radius_px, colors[key], thickness, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{item['segment']} frame {frame_index} | outline=target filled=3D analytic rollout",
                (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    writer.release()


def _run_learning_rate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(max(int(payload["worker_threads"]), 1))
    output_dir = Path(payload["output_dir"])
    output_path = output_dir / "physics_alignment.json"
    result = run_physionpp_friction_collision_sphere_sysid(
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
        "segment_equal_weight_loss_m2": float(result["segment_equal_weight_loss_m2"]),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physion++ friction_collision 3D analytic sphere SWR.")
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
    trial_payloads = [
        {
            "world_modeling_dir": str(Path(args.world_modeling_dir)),
            "output_dir": str(trial_root / f"lr_{lr:g}"),
            "lr": float(lr),
            "steps": int(args.steps),
            "patience": int(args.patience),
            "worker_threads": int(args.worker_threads),
        }
        for lr in learning_rates
    ]
    trials, worker_count = analytic_swr_common.run_parallel_trials(
        payloads=trial_payloads,
        worker=_run_learning_rate_trial,
        max_workers=int(args.workers),
    )
    selected = min(trials, key=lambda item: float(item["segment_equal_weight_loss_m2"]))
    result = _load_json(Path(selected["output_path"]))
    if not isinstance(result, dict):
        raise ValueError(f"invalid selected result: {selected['output_path']}")
    result["optimizer_multi_learning_rate"] = {
        "learning_rates": learning_rates,
        "worker_count": int(worker_count),
        "selected_learning_rate": float(selected["lr"]),
        "selection_metric": "segment_equal_weight_loss_m2",
        "trials": sorted(trials, key=lambda item: float(item["lr"])),
    }
    _write_json(output_path, result)
    _write_json(
        output_dir / "summary.json",
        {
            "status": result.get("status"),
            "backend": result.get("backend"),
            "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
            "segment_equal_weight_loss_m2": result.get("segment_equal_weight_loss_m2"),
            "optimizer_multi_learning_rate": result.get("optimizer_multi_learning_rate"),
            "segments": [
                {
                    "segment": item.get("segment"),
                    "frame_range": item.get("frame_range"),
                    "dynamic_object_rmse_m": item.get("dynamic_object_rmse_m"),
                    "object_rmse_m": item.get("object_rmse_m"),
                    "patient_motion_mode": item.get("patient_motion_mode"),
                }
                for item in result.get("segments", [])
                if isinstance(item, dict)
            ],
        },
    )
    if args.render_video:
        _render_debug_video(result, output_dir / "debug" / "target_vs_rollout.mp4")
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "backend": result.get("backend"),
                "dynamic_object_rmse_m": result.get("dynamic_object_rmse_m"),
                "selected_learning_rate": selected["lr"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
