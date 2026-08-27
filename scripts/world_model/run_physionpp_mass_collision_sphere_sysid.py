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
MIN_MASS_RATIO = 0.1
MAX_MASS_RATIO = 10.0
MIN_RESTITUTION = 0.0
MAX_RESTITUTION = 1.0
DEFAULT_MASS_RATIO = 1.0
DEFAULT_RESTITUTION = 0.0
DEFAULT_PATIENT_FRICTION = 0.1
MAX_INITIAL_SPEED_M_PER_S = 20.0
RADIUS_PRIOR_WEIGHT = friction_swr.RADIUS_PRIOR_WEIGHT
RESTITUTION_PRIOR_WEIGHT = 1e-3
CONTACT_NEWTON_ITERS = 8


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _contact_frame(context: dict[str, Any], segment_name: str) -> int:
    sources = (
        ((context.get("agent_flush") or {}).get("segments") or {}).get(segment_name),
        ((context.get("ball_trajectory_refinement") or {}).get("segments") or {}).get(segment_name),
    )
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("contact_frame"), (int, float)):
            return int(source["contact_frame"])
    if segment_name == "seg1":
        source = context.get("agent_pre_impact")
        if isinstance(source, dict) and isinstance(source.get("contact_frame"), (int, float)):
            return int(source["contact_frame"])
    raise ValueError(f"mass_collision pose context has no {segment_name} ball-agent contact frame")


def _pose_context(
    world_modeling_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    context, support, roles, split = sysid_common.load_collision_pose_context(
        world_modeling_dir,
        context_key="physion_pp_mass_collision",
        context_required_error=(
            "mass_collision SWR requires physion_pp_mass_collision pose context"
        ),
        support_required_error=(
            "mass_collision SWR requires the corrected support plane"
        ),
        bindings_required_error=(
            "mass_collision pose context has no role bindings or segment split"
        ),
    )

    segments: list[dict[str, Any]] = []
    for segment_name in ("seg1", "seg2"):
        frame_range = split.get(segment_name)
        ball_id = roles.get(f"{segment_name}_ball")
        agent_id = roles.get(f"{segment_name}_agent")
        patient_id = roles.get(f"{segment_name}_patient")
        if not isinstance(frame_range, list) or len(frame_range) != 2 or not ball_id or not agent_id:
            raise ValueError(f"incomplete mass_collision role binding for {segment_name}")
        if segment_name == "seg2" and not patient_id:
            raise ValueError("mass_collision seg2 has no patient role")
        segments.append(
            {
                "segment": segment_name,
                "ball_object_id": str(ball_id),
                "agent_object_id": str(agent_id),
                "patient_object_id": str(patient_id) if patient_id else None,
                "frame_range": [int(frame_range[0]), int(frame_range[1])],
                "ball_agent_contact_frame": _contact_frame(context, segment_name),
            }
        )
    return segments, context, support


def _role_object_ids(segments: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"ball": [], "agent": [], "patient": []}
    for segment in segments:
        for role in result:
            object_id = segment.get(f"{role}_object_id")
            if object_id and str(object_id) not in result[role]:
                result[role].append(str(object_id))
    return result


def _fit_linear_velocity(frames: list[int], positions: np.ndarray) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(3, dtype=np.float64)
    times = (np.asarray(frames, dtype=np.float64) - float(frames[0])) * PHYSIONPP_PHYSICS_DT_SEC
    design = np.stack([np.ones_like(times), times], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(positions, dtype=np.float64), rcond=None)
    return np.asarray(coefficients[1], dtype=np.float64)


def _window_velocity(
    frames: list[int],
    positions: np.ndarray,
    *,
    contact_frame: int,
    before: bool,
    window: int = 6,
) -> np.ndarray | None:
    indices = [
        index
        for index, frame in enumerate(frames)
        if (frame < contact_frame if before else frame > contact_frame)
    ]
    indices = indices[-window:] if before else indices[:window]
    if len(indices) < 2:
        return None
    return _fit_linear_velocity(
        [frames[index] for index in indices],
        np.asarray([positions[index] for index in indices], dtype=np.float64),
    )


def _collision_initialization(prepared: list[dict[str, Any]]) -> tuple[float, float, list[dict[str, Any]]]:
    mass_ratios: list[float] = []
    restitutions: list[float] = []
    observations: list[dict[str, Any]] = []
    for item in prepared:
        contact_frame = int(item["ball_agent_contact_frame"])
        ball_frames = item["ball_frames"]
        agent_frames = item["agent_frames"]
        ball_target = item["target_ball_np"]
        agent_target = item["target_agent_np"]
        ball_pre = _window_velocity(ball_frames, ball_target, contact_frame=contact_frame, before=True)
        agent_pre = _window_velocity(agent_frames, agent_target, contact_frame=contact_frame, before=True)
        ball_post = _window_velocity(ball_frames, ball_target, contact_frame=contact_frame, before=False)
        agent_post = _window_velocity(agent_frames, agent_target, contact_frame=contact_frame, before=False)
        if ball_pre is None or agent_pre is None or agent_post is None:
            continue
        ball_contact_index = min(range(len(ball_frames)), key=lambda index: abs(ball_frames[index] - contact_frame))
        agent_contact_index = min(range(len(agent_frames)), key=lambda index: abs(agent_frames[index] - contact_frame))
        normal = ball_target[ball_contact_index] - agent_target[agent_contact_index]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            continue
        normal = normal / normal_norm
        agent_delta = float(np.dot(agent_post - agent_pre, normal))
        pre_relative = float(np.dot(ball_pre - agent_pre, normal))
        if ball_post is not None:
            ball_delta = float(np.dot(ball_post - ball_pre, normal))
            post_relative = float(np.dot(ball_post - agent_post, normal))
            ratio = abs(ball_delta) / max(abs(agent_delta), 1e-8)
            restitution = -post_relative / min(pre_relative, -1e-8)
            source = "two_sided_velocity_change"
        else:
            incoming_speed = max(-pre_relative, 1e-8)
            transferred_speed = max(abs(agent_delta), 1e-8)
            restitution = DEFAULT_RESTITUTION
            ratio = (1.0 + restitution) * incoming_speed / transferred_speed - 1.0
            source = "ball_pre_and_agent_post_with_default_restitution"
        if math.isfinite(ratio):
            mass_ratios.append(float(np.clip(ratio, MIN_MASS_RATIO, MAX_MASS_RATIO)))
        if math.isfinite(restitution):
            restitutions.append(float(np.clip(restitution, MIN_RESTITUTION, MAX_RESTITUTION)))
        observations.append(
            {
                "agent_mass_over_ball_mass": float(np.clip(ratio, MIN_MASS_RATIO, MAX_MASS_RATIO)),
                "ball_agent_restitution": float(np.clip(restitution, MIN_RESTITUTION, MAX_RESTITUTION)),
                "source": source,
            }
        )
    mass_ratio = float(np.median(mass_ratios)) if mass_ratios else DEFAULT_MASS_RATIO
    restitution = float(np.median(restitutions)) if restitutions else DEFAULT_RESTITUTION
    return mass_ratio, restitution, observations


def _terminal_pose(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(sysid_common._opencv_pose_to_blender_world(record["pose_4x4"]), dtype=np.float64)


def _prepare_segments(
    *,
    target: dict[str, list[dict[str, Any]]],
    segments: list[dict[str, Any]],
    support: dict[str, Any],
    collision_geometry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for segment in segments:
        ball_id = str(segment["ball_object_id"])
        agent_id = str(segment["agent_object_id"])
        ball_records = sysid_common.records_in_frame_range(
            target, ball_id, segment["frame_range"]
        )
        agent_records = sysid_common.records_in_frame_range(
            target, agent_id, segment["frame_range"]
        )
        ball_by_frame = {int(record["frame_index"]): record for record in ball_records}
        agent_by_frame = {int(record["frame_index"]): record for record in agent_records}
        ball_frames = sorted(ball_by_frame)
        agent_frames = sorted(agent_by_frame)
        frames = sorted(set(ball_frames) | set(agent_frames))
        contact_frame = int(segment["ball_agent_contact_frame"])
        if (
            len(ball_frames) < 2
            or len(agent_frames) < 4
            or ball_frames[0] != agent_frames[0]
            or not (frames[0] < contact_frame <= frames[-1])
        ):
            raise ValueError(
                f"{segment['segment']} has insufficient ball-agent trajectory around contact frame {contact_frame}"
            )
        target_ball = np.asarray([ball_by_frame[frame]["position"] for frame in ball_frames], dtype=np.float64)
        target_agent = np.asarray([agent_by_frame[frame]["position"] for frame in agent_frames], dtype=np.float64)
        pre_ball_indices = [index for index, frame in enumerate(ball_frames) if frame < contact_frame]
        pre_agent_indices = [index for index, frame in enumerate(agent_frames) if frame < contact_frame]
        post_agent_indices = [index for index, frame in enumerate(agent_frames) if frame > contact_frame]
        ball_v0 = _fit_linear_velocity(
            [ball_frames[index] for index in pre_ball_indices],
            target_ball[pre_ball_indices],
        )
        agent_v0 = _fit_linear_velocity(
            [agent_frames[index] for index in pre_agent_indices],
            target_agent[pre_agent_indices],
        )
        agent_friction_records = post_agent_indices if len(post_agent_indices) >= 5 else list(range(len(agent_frames)))
        ball_contact_record = ball_by_frame[ball_frames[min(range(len(ball_frames)), key=lambda index: abs(ball_frames[index] - contact_frame))]]
        agent_contact_record = agent_by_frame[agent_frames[min(range(len(agent_frames)), key=lambda index: abs(agent_frames[index] - contact_frame))]]
        ball_geometry = collision_geometry[ball_id]
        agent_geometry = collision_geometry[agent_id]
        frame_to_index = {frame: index for index, frame in enumerate(frames)}
        item: dict[str, Any] = {
            **segment,
            "frames": frames,
            "ball_frames": ball_frames,
            "agent_frames": agent_frames,
            "ball_rollout_indices": [frame_to_index[frame] for frame in ball_frames],
            "agent_rollout_indices": [frame_to_index[frame] for frame in agent_frames],
            "target_ball_np": target_ball,
            "target_agent_np": target_agent,
            "ball_v0_init": ball_v0,
            "agent_v0_init": agent_v0,
            "ball_obb_half_extents_np": np.asarray(ball_geometry["half_extents_m"], dtype=np.float64),
            "agent_obb_half_extents_np": np.asarray(agent_geometry["half_extents_m"], dtype=np.float64),
            "ball_obb_rotation_np": _terminal_pose(ball_contact_record)[:3, :3],
            "agent_obb_rotation_np": _terminal_pose(agent_contact_record)[:3, :3],
            "agent_friction_init": sysid_common.fit_tangential_friction(
                frames=[agent_frames[index] for index in agent_friction_records],
                positions=target_agent[agent_friction_records],
                up=support["up"],
                physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                gravity_magnitude=friction_swr.GRAVITY_M_PER_S2,
                min_friction=MIN_FRICTION,
                max_friction=MAX_FRICTION,
            ),
            "terminal_pose_blender_world_4x4": {
                ball_id: _terminal_pose(ball_by_frame[ball_frames[-1]]),
                agent_id: _terminal_pose(agent_by_frame[agent_frames[-1]]),
            },
        }
        patient_id = segment.get("patient_object_id")
        if patient_id:
            patient_records = sysid_common.records_in_frame_range(
                target, str(patient_id), segment["frame_range"]
            )
            if len(patient_records) < 2:
                raise ValueError(f"{segment['segment']} patient has fewer than two trajectory frames")
            patient_frames = [int(record["frame_index"]) for record in patient_records]
            patient_target = np.asarray([record["position"] for record in patient_records], dtype=np.float64)
            item.update(
                {
                    "patient_frames": patient_frames,
                    "target_patient_np": patient_target,
                    "patient_v0_init": sysid_common.fit_initial_velocity(
                        patient_frames,
                        patient_target,
                        physics_dt_sec=PHYSIONPP_PHYSICS_DT_SEC,
                    ),
                }
            )
            item["terminal_pose_blender_world_4x4"][str(patient_id)] = _terminal_pose(patient_records[-1])
        prepared.append(item)
    return prepared


def _ballistic_motion(
    *,
    start_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    gravity: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = times.reshape(-1, 1)
    positions = start_position + initial_velocity * dt + 0.5 * gravity * dt * dt
    velocities = initial_velocity + gravity * dt
    return positions, velocities


def _agent_plane_motion(
    *,
    start_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    friction: torch.Tensor,
    up: torch.Tensor,
    times: torch.Tensor,
    start_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent_velocity = initial_velocity - torch.sum(initial_velocity * up) * up
    positions, velocities = analytic_swr_common.closed_form_constant_deceleration(
        x0=start_position.reshape(1, 3),
        v0=tangent_velocity.reshape(1, 3),
        deceleration=friction.reshape(1) * friction.new_tensor(friction_swr.GRAVITY_M_PER_S2),
        t=times,
        t0=start_time.reshape(1),
    )
    return positions[:, 0], velocities[:, 0]


def _obb_support_radius(
    *,
    direction: torch.Tensor,
    rotation: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    unit_direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-12)
    local_direction = rotation.transpose(0, 1) @ unit_direction
    return torch.sum(torch.abs(local_direction) * half_extents)


def _obb_pair_support_distance(
    *,
    relative_position: torch.Tensor,
    ball_rotation: torch.Tensor,
    ball_half_extents: torch.Tensor,
    agent_rotation: torch.Tensor,
    agent_half_extents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = relative_position / torch.clamp(torch.linalg.vector_norm(relative_position), min=1e-12)
    support_distance = _obb_support_radius(
        direction=normal,
        rotation=ball_rotation,
        half_extents=ball_half_extents,
    ) + _obb_support_radius(
        direction=-normal,
        rotation=agent_rotation,
        half_extents=agent_half_extents,
    )
    return support_distance, normal


def _solve_contact_time(
    *,
    ball_start_position: torch.Tensor,
    ball_initial_velocity: torch.Tensor,
    agent_start_position: torch.Tensor,
    agent_initial_velocity: torch.Tensor,
    agent_friction: torch.Tensor,
    gravity: torch.Tensor,
    up: torch.Tensor,
    ball_rotation: torch.Tensor,
    ball_half_extents: torch.Tensor,
    agent_rotation: torch.Tensor,
    agent_half_extents: torch.Tensor,
    event_center: torch.Tensor,
    window_seconds: torch.Tensor,
    newton_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.clamp(event_center - window_seconds, min=0.0)
    upper = event_center + window_seconds
    t = event_center.clone()
    for _ in range(max(int(newton_iters), 0)):
        ball_position, ball_velocity = _ballistic_motion(
            start_position=ball_start_position,
            initial_velocity=ball_initial_velocity,
            gravity=gravity,
            times=t.reshape(1),
        )
        agent_position, agent_velocity = _agent_plane_motion(
            start_position=agent_start_position,
            initial_velocity=agent_initial_velocity,
            friction=agent_friction,
            up=up,
            times=t.reshape(1),
            start_time=t.new_tensor(0.0),
        )
        relative_position = ball_position[0] - agent_position[0]
        distance = torch.linalg.vector_norm(relative_position)
        support_distance, normal = _obb_pair_support_distance(
            relative_position=relative_position,
            ball_rotation=ball_rotation,
            ball_half_extents=ball_half_extents,
            agent_rotation=agent_rotation,
            agent_half_extents=agent_half_extents,
        )
        residual = distance - support_distance
        derivative = torch.sum(normal * (ball_velocity[0] - agent_velocity[0]))
        valid = torch.abs(derivative) > 1e-8
        safe_derivative = torch.where(valid, derivative, torch.ones_like(derivative))
        step = torch.where(valid, residual / safe_derivative, torch.zeros_like(residual))
        t = torch.minimum(torch.maximum(t - step, lower), upper)
    ball_position, _ = _ballistic_motion(
        start_position=ball_start_position,
        initial_velocity=ball_initial_velocity,
        gravity=gravity,
        times=t.reshape(1),
    )
    agent_position, _ = _agent_plane_motion(
        start_position=agent_start_position,
        initial_velocity=agent_initial_velocity,
        friction=agent_friction,
        up=up,
        times=t.reshape(1),
        start_time=t.new_tensor(0.0),
    )
    residual, _normal = _obb_pair_support_distance(
        relative_position=ball_position[0] - agent_position[0],
        ball_rotation=ball_rotation,
        ball_half_extents=ball_half_extents,
        agent_rotation=agent_rotation,
        agent_half_extents=agent_half_extents,
    )
    residual = torch.linalg.vector_norm(ball_position[0] - agent_position[0]) - residual
    return t, residual


def _rollout_ball_agent_segment(
    *,
    frames: list[int],
    contact_frame: int,
    start_positions: torch.Tensor,
    initial_velocities: torch.Tensor,
    agent_friction: torch.Tensor,
    ball_obb_rotation: torch.Tensor,
    ball_obb_half_extents: torch.Tensor,
    agent_obb_rotation: torch.Tensor,
    agent_obb_half_extents: torch.Tensor,
    agent_mass_over_ball_mass: torch.Tensor,
    restitution: torch.Tensor,
    support: dict[str, Any],
    newton_iters: int = CONTACT_NEWTON_ITERS,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    times = start_positions.new_tensor(
        [(float(frame) - float(frames[0])) * PHYSIONPP_PHYSICS_DT_SEC for frame in frames]
    )
    up = start_positions.new_tensor(support["up"])
    up = up / torch.clamp(torch.linalg.vector_norm(up), min=1e-12)
    gravity = start_positions.new_tensor(support["gravity_direction"]) * start_positions.new_tensor(
        friction_swr.GRAVITY_M_PER_S2
    )
    event_center = times.new_tensor((float(contact_frame) - float(frames[0])) * PHYSIONPP_PHYSICS_DT_SEC)
    event_time, contact_residual = _solve_contact_time(
        ball_start_position=start_positions[0],
        ball_initial_velocity=initial_velocities[0],
        agent_start_position=start_positions[1],
        agent_initial_velocity=initial_velocities[1],
        agent_friction=agent_friction,
        gravity=gravity,
        up=up,
        ball_rotation=ball_obb_rotation,
        ball_half_extents=ball_obb_half_extents,
        agent_rotation=agent_obb_rotation,
        agent_half_extents=agent_obb_half_extents,
        event_center=event_center,
        window_seconds=event_center.new_tensor(PHYSIONPP_PHYSICS_DT_SEC),
        newton_iters=newton_iters,
    )
    ball_pre_positions, ball_pre_velocities = _ballistic_motion(
        start_position=start_positions[0],
        initial_velocity=initial_velocities[0],
        gravity=gravity,
        times=times,
    )
    agent_pre_positions, agent_pre_velocities = _agent_plane_motion(
        start_position=start_positions[1],
        initial_velocity=initial_velocities[1],
        friction=agent_friction,
        up=up,
        times=times,
        start_time=times.new_tensor(0.0),
    )
    ball_hit_position, ball_hit_velocity = _ballistic_motion(
        start_position=start_positions[0],
        initial_velocity=initial_velocities[0],
        gravity=gravity,
        times=event_time.reshape(1),
    )
    agent_hit_position, agent_hit_velocity = _agent_plane_motion(
        start_position=start_positions[1],
        initial_velocity=initial_velocities[1],
        friction=agent_friction,
        up=up,
        times=event_time.reshape(1),
        start_time=event_time.new_tensor(0.0),
    )
    relative_position = ball_hit_position[0] - agent_hit_position[0]
    _support_distance, center_normal = _obb_pair_support_distance(
        relative_position=relative_position,
        ball_rotation=ball_obb_rotation,
        ball_half_extents=ball_obb_half_extents,
        agent_rotation=agent_obb_rotation,
        agent_half_extents=agent_obb_half_extents,
    )
    collision_normal = relative_position - torch.sum(relative_position * up) * up
    if bool((torch.linalg.vector_norm(collision_normal) <= 1e-8).detach().cpu()):
        collision_normal = center_normal
    pre_velocity = torch.stack([ball_hit_velocity[0], agent_hit_velocity[0]])
    post_velocity, impulse, relative_normal_velocity = analytic_swr_common.sphere_sphere_impulse_update_torch(
        velocities=pre_velocity,
        i=0,
        j=1,
        normal_from_j_to_i=collision_normal,
        mass_i=agent_mass_over_ball_mass.new_tensor(1.0),
        mass_j=agent_mass_over_ball_mass,
        restitution=restitution,
    )
    ball_post_positions, ball_post_velocities = _ballistic_motion(
        start_position=ball_hit_position[0],
        initial_velocity=post_velocity[0],
        gravity=gravity,
        times=torch.clamp(times - event_time, min=0.0),
    )
    agent_post_positions, agent_post_velocities = _agent_plane_motion(
        start_position=agent_hit_position[0],
        initial_velocity=post_velocity[1],
        friction=agent_friction,
        up=up,
        times=times,
        start_time=event_time,
    )
    after = (times >= event_time).reshape(-1, 1)
    ball_positions = torch.where(after, ball_post_positions, ball_pre_positions)
    ball_velocities = torch.where(after, ball_post_velocities, ball_pre_velocities)
    agent_positions = torch.where(after, agent_post_positions, agent_pre_positions)
    agent_velocities = torch.where(after, agent_post_velocities, agent_pre_velocities)
    event = {
        "event_type": "ball_agent_sphere_collision",
        "frame_index": event_time / event_time.new_tensor(PHYSIONPP_PHYSICS_DT_SEC)
        + event_time.new_tensor(float(frames[0])),
        "event_time_from_segment_start_s": event_time,
        "contact_residual_m": contact_residual,
        "normal_ball_from_agent": collision_normal
        / torch.clamp(torch.linalg.vector_norm(collision_normal), min=1e-12),
        "pre_velocity": pre_velocity,
        "post_velocity": post_velocity,
        "relative_normal_velocity_m_per_s": relative_normal_velocity,
        "impulse_ball": impulse,
    }
    return (
        torch.stack([ball_positions, agent_positions], dim=1),
        torch.stack([ball_velocities, agent_velocities], dim=1),
        [event],
        [],
    )


def _rollout_patient(
    *,
    item: dict[str, Any],
    velocity: torch.Tensor,
    support: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    frames = item["patient_frames"]
    times = velocity.new_tensor(
        [(float(frame) - float(frames[0])) * PHYSIONPP_PHYSICS_DT_SEC for frame in frames]
    )
    gravity = velocity.new_tensor(support["gravity_direction"]) * velocity.new_tensor(
        friction_swr.GRAVITY_M_PER_S2
    )
    positions, velocities = _ballistic_motion(
        start_position=velocity.new_tensor(item["target_patient_np"][0]),
        initial_velocity=velocity,
        gravity=gravity,
        times=times,
    )
    return positions, velocities, []


def _run_rollouts(
    *,
    prepared: list[dict[str, Any]],
    support: dict[str, Any],
    shared_radii: dict[str, float],
    ball_shared_speed_raw: torch.Tensor,
    ball_direction_angles: list[torch.Tensor],
    agent_velocity_raw: list[torch.Tensor],
    patient_velocity_raw: torch.Tensor,
    agent_friction_raw: torch.Tensor,
    ball_radius_raw: torch.Tensor,
    agent_radius_raw: torch.Tensor,
    mass_ratio_raw: torch.Tensor,
    restitution_raw: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, torch.Tensor]]:
    agent_friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(agent_friction_raw)
    ball_friction = agent_friction.new_tensor(0.0)
    mass_ratio = MIN_MASS_RATIO + (MAX_MASS_RATIO - MIN_MASS_RATIO) * torch.sigmoid(mass_ratio_raw)
    restitution = MIN_RESTITUTION + (MAX_RESTITUTION - MIN_RESTITUTION) * torch.sigmoid(restitution_raw)
    ball_shared_speed = torch.nn.functional.softplus(ball_shared_speed_raw)
    ball_radius, ball_radius_log_scale = friction_swr._radius_from_raw(
        radius_init_tensor=agent_friction.new_tensor(shared_radii["ball"]),
        radius_raw=ball_radius_raw,
    )
    agent_radius, agent_radius_log_scale = friction_swr._radius_from_raw(
        radius_init_tensor=agent_friction.new_tensor(shared_radii["agent"]),
        radius_raw=agent_radius_raw,
    )
    patient_radius = agent_friction.new_tensor(shared_radii["patient"])
    ball_radius_scale = ball_radius / agent_friction.new_tensor(shared_radii["ball"])
    agent_radius_scale = agent_radius / agent_friction.new_tensor(shared_radii["agent"])
    segment_results: list[dict[str, Any]] = []
    segment_losses: list[torch.Tensor] = []
    for index, item in enumerate(prepared):
        azimuth, elevation = ball_direction_angles[index].unbind()
        ball_velocity = ball_shared_speed * torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ]
        )
        agent_velocity = MAX_INITIAL_SPEED_M_PER_S * torch.tanh(agent_velocity_raw[index])
        start_positions = torch.stack(
            [
                ball_velocity.new_tensor(item["target_ball_np"][0]),
                ball_velocity.new_tensor(item["target_agent_np"][0]),
            ]
        )
        predicted, predicted_velocities, collision_events, plane_events = _rollout_ball_agent_segment(
            frames=item["frames"],
            contact_frame=int(item["ball_agent_contact_frame"]),
            start_positions=start_positions,
            initial_velocities=torch.stack([ball_velocity, agent_velocity]),
            agent_friction=agent_friction,
            ball_obb_rotation=ball_velocity.new_tensor(item["ball_obb_rotation_np"]),
            ball_obb_half_extents=ball_velocity.new_tensor(item["ball_obb_half_extents_np"]) * ball_radius_scale,
            agent_obb_rotation=ball_velocity.new_tensor(item["agent_obb_rotation_np"]),
            agent_obb_half_extents=ball_velocity.new_tensor(item["agent_obb_half_extents_np"]) * agent_radius_scale,
            agent_mass_over_ball_mass=mass_ratio,
            restitution=restitution,
            support=support,
        )
        ball_target = ball_velocity.new_tensor(item["target_ball_np"])
        agent_target = ball_velocity.new_tensor(item["target_agent_np"])
        ball_error = torch.sum((predicted[item["ball_rollout_indices"], 0] - ball_target) ** 2, dim=-1)
        agent_error = torch.sum((predicted[item["agent_rollout_indices"], 1] - agent_target) ** 2, dim=-1)
        role_losses = [torch.mean(ball_error), torch.mean(agent_error)]
        result: dict[str, Any] = {
            "predicted_dynamics": predicted,
            "predicted_dynamics_velocities": predicted_velocities,
            "ball_velocity": ball_velocity,
            "agent_velocity": agent_velocity,
            "collision_events": collision_events,
            "plane_events": plane_events,
            "role_squared_errors": [ball_error, agent_error],
        }
        if item.get("patient_object_id"):
            patient_velocity = MAX_INITIAL_SPEED_M_PER_S * torch.tanh(patient_velocity_raw)
            predicted_patient, predicted_patient_velocities, patient_plane_events = _rollout_patient(
                item=item,
                velocity=patient_velocity,
                support=support,
            )
            target_patient = patient_velocity.new_tensor(item["target_patient_np"])
            patient_squared_error = torch.sum((predicted_patient - target_patient) ** 2, dim=-1)
            role_losses.append(torch.mean(patient_squared_error))
            result.update(
                {
                    "patient_velocity": patient_velocity,
                    "predicted_patient": predicted_patient,
                    "predicted_patient_velocities": predicted_patient_velocities,
                    "patient_squared_error": patient_squared_error,
                    "patient_plane_events": patient_plane_events,
                }
            )
        segment_loss = torch.mean(torch.stack(role_losses))
        result["segment_loss"] = segment_loss
        segment_losses.append(segment_loss)
        segment_results.append(result)
    trajectory_loss = torch.mean(torch.stack(segment_losses))
    radius_regularization = ball_radius_log_scale.square() + agent_radius_log_scale.square()
    restitution_regularization = restitution.square()
    loss = (
        trajectory_loss
        + trajectory_loss.new_tensor(RADIUS_PRIOR_WEIGHT) * radius_regularization
        + trajectory_loss.new_tensor(RESTITUTION_PRIOR_WEIGHT) * restitution_regularization
    )
    return loss, segment_results, {
        "trajectory_loss": trajectory_loss,
        "ball_friction": ball_friction,
        "agent_friction": agent_friction,
        "patient_friction": ball_friction.new_tensor(DEFAULT_PATIENT_FRICTION),
        "agent_mass_over_ball_mass": mass_ratio,
        "ball_agent_restitution": restitution,
        "ball_shared_speed": ball_shared_speed,
        "ball_radius": ball_radius,
        "agent_radius": agent_radius,
        "patient_radius": patient_radius,
        "ball_radius_log_scale": ball_radius_log_scale,
        "agent_radius_log_scale": agent_radius_log_scale,
    }


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-8)
    return value + math.log(-math.expm1(-value))


def _optimize(
    *,
    prepared: list[dict[str, Any]],
    support: dict[str, Any],
    shared_radii: dict[str, float],
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    speed_candidates = [float(np.linalg.norm(item["ball_v0_init"])) for item in prepared]
    shared_speed_init = max(float(np.median(speed_candidates)), 1e-4)
    ball_shared_speed_raw = torch.tensor(_softplus_inverse(shared_speed_init), dtype=torch.float64, requires_grad=True)
    fallback_direction = np.asarray(max(prepared, key=lambda item: np.linalg.norm(item["ball_v0_init"]))["ball_v0_init"])
    ball_direction_angles = [
        sysid_common.direction_angles_from_value(
            item["ball_v0_init"],
            fallback_direction,
        )
        for item in prepared
    ]
    agent_velocity_raw = [
        sysid_common.velocity_raw_from_value(
            item["agent_v0_init"],
            maximum_speed_m_per_s=MAX_INITIAL_SPEED_M_PER_S,
        )
        for item in prepared
    ]
    seg2 = next(item for item in prepared if item["segment"] == "seg2")
    patient_velocity_raw = sysid_common.velocity_raw_from_value(
        seg2["patient_v0_init"],
        maximum_speed_m_per_s=MAX_INITIAL_SPEED_M_PER_S,
    )
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
    mass_ratio_init, restitution_init, impulse_initialization = _collision_initialization(prepared)
    mass_ratio_raw = torch.tensor(
        sysid_common.raw_from_interval(
            mass_ratio_init,
            MIN_MASS_RATIO,
            MAX_MASS_RATIO,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    restitution_raw = torch.tensor(
        sysid_common.raw_from_interval(
            restitution_init,
            MIN_RESTITUTION,
            MAX_RESTITUTION,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    ball_radius_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    agent_radius_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    parameters = [
        ball_shared_speed_raw,
        *ball_direction_angles,
        *agent_velocity_raw,
        patient_velocity_raw,
        agent_friction_raw,
        ball_radius_raw,
        agent_radius_raw,
        mass_ratio_raw,
        restitution_raw,
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
            ball_shared_speed_raw=ball_shared_speed_raw,
            ball_direction_angles=ball_direction_angles,
            agent_velocity_raw=agent_velocity_raw,
            patient_velocity_raw=patient_velocity_raw,
            agent_friction_raw=agent_friction_raw,
            ball_radius_raw=ball_radius_raw,
            agent_radius_raw=agent_radius_raw,
            mass_ratio_raw=mass_ratio_raw,
            restitution_raw=restitution_raw,
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
        raise RuntimeError("mass_collision optimization produced no finite state")
    with torch.no_grad():
        for parameter, state in zip(parameters, best_state):
            parameter.copy_(state)
    loss, segment_results, physics = _run_rollouts(
        prepared=prepared,
        support=support,
        shared_radii=shared_radii,
        ball_shared_speed_raw=ball_shared_speed_raw,
        ball_direction_angles=ball_direction_angles,
        agent_velocity_raw=agent_velocity_raw,
        patient_velocity_raw=patient_velocity_raw,
        agent_friction_raw=agent_friction_raw,
        ball_radius_raw=ball_radius_raw,
        agent_radius_raw=agent_radius_raw,
        mass_ratio_raw=mass_ratio_raw,
        restitution_raw=restitution_raw,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "trajectory_loss": float(physics["trajectory_loss"].detach().cpu()),
        "best_step": best_step,
        "executed_steps": executed_steps,
        "history": history,
        "segments": segment_results,
        "physics": physics,
        "initialization": {
            "ball_shared_initial_speed_m_per_s": shared_speed_init,
            "ball_segment_initial_speed_candidates_m_per_s": speed_candidates,
            "ball_ground_friction": 0.0,
            "agent_ground_friction": agent_friction_init,
            "agent_mass_over_ball_mass": mass_ratio_init,
            "ball_agent_restitution": restitution_init,
            "impulse_observations": impulse_initialization,
            "ball_radius_m": shared_radii["ball"],
            "agent_radius_m": shared_radii["agent"],
            "patient_radius_m": shared_radii["patient"],
        },
    }


def _trajectory_records(frames: list[int], positions: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(frame),
            "position_blender_world_m": position.astype(float).tolist(),
            "position_opencv_camera_m": sysid_common._blender_vector_to_opencv_camera(position.astype(float).tolist()),
        }
        for frame, position in zip(frames, positions)
    ]


def _event_to_json(event: dict[str, Any], ball_id: str, agent_id: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "event_type": str(event["event_type"]),
        "object_ids": [ball_id, agent_id],
    }
    for key, value in event.items():
        if key == "event_type":
            continue
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            output[key] = float(array) if array.ndim == 0 else array.astype(float).tolist()
        else:
            output[key] = value
    return output


def run_physionpp_mass_collision_sphere_sysid(
    *,
    world_modeling_dir: Path,
    output_path: Path,
    output_dir: Path,
    lr: float,
    steps: int,
    patience: int,
) -> dict[str, Any]:
    manifest = sysid_common.build_manifest_from_world_modeling(world_modeling_dir)
    manifest["mode"] = "physionpp_mass_collision_offline_sysid"
    manifest.setdefault("rollout", {})["backend"] = "swr_backend.collision_mass_spheres"
    target = sysid_common._target_records_from_swr(manifest)
    segments, context, support_context = _pose_context(world_modeling_dir)
    role_object_ids = _role_object_ids(segments)
    object_role_assignments = [
        (object_id, role)
        for role, object_ids in role_object_ids.items()
        for object_id in object_ids
    ]
    fallback_radius_by_object = friction_swr._initial_radius_by_object(
        manifest=manifest,
        object_ids=[object_id for object_id, _role in object_role_assignments],
    )
    shared_radii, _radius_by_object = sysid_common.shared_role_radii_from_mesh_envelopes(
        role_object_ids=role_object_ids,
        object_role_assignments=object_role_assignments,
        fallback_radius_by_object=fallback_radius_by_object,
        object_specs=friction_swr._object_specs(manifest),
        mesh_loader=friction_swr._mesh_from_path,
        envelope_scale=sysid_common.COLLISION_INITIAL_RADIUS_ENVELOPE_SCALE,
    )
    collision_geometry = sysid_common.collision_oriented_box_geometry_by_object(
        object_specs=friction_swr._object_specs(manifest),
        object_ids={
            object_id
            for object_ids in role_object_ids.values()
            for object_id in object_ids
        },
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
        support=support,
        collision_geometry=collision_geometry,
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
        "ball": float(physics["ball_radius"].detach().cpu()),
        "agent": float(physics["agent_radius"].detach().cpu()),
        "patient": float(physics["patient_radius"].detach().cpu()),
    }
    segment_payloads: list[dict[str, Any]] = []
    all_squared_errors: list[np.ndarray] = []
    for item, result in zip(prepared, optimized["segments"]):
        ball_id = str(item["ball_object_id"])
        agent_id = str(item["agent_object_id"])
        predicted = result["predicted_dynamics"].detach().cpu().numpy()
        predicted_velocities = result["predicted_dynamics_velocities"].detach().cpu().numpy()
        ball_error = result["role_squared_errors"][0].detach().cpu().numpy()
        agent_error = result["role_squared_errors"][1].detach().cpu().numpy()
        all_squared_errors.extend([ball_error, agent_error])
        target_trajectories = {
            ball_id: _trajectory_records(item["ball_frames"], item["target_ball_np"]),
            agent_id: _trajectory_records(item["agent_frames"], item["target_agent_np"]),
        }
        simulated_trajectories = {
            ball_id: _trajectory_records(item["frames"], predicted[:, 0]),
            agent_id: _trajectory_records(item["frames"], predicted[:, 1]),
        }
        object_rmse = {
            ball_id: float(np.sqrt(np.mean(ball_error))),
            agent_id: float(np.sqrt(np.mean(agent_error))),
        }
        terminal_state = {
            ball_id: {
                "frame_index": int(item["frames"][-1]),
                "position_blender_world_m": predicted[-1, 0].astype(float).tolist(),
                "velocity_blender_world_m_per_s": predicted_velocities[-1, 0].astype(float).tolist(),
            },
            agent_id: {
                "frame_index": int(item["frames"][-1]),
                "position_blender_world_m": predicted[-1, 1].astype(float).tolist(),
                "velocity_blender_world_m_per_s": predicted_velocities[-1, 1].astype(float).tolist(),
            },
        }
        optimized_velocities = {
            ball_id: result["ball_velocity"].detach().cpu().tolist(),
            agent_id: result["agent_velocity"].detach().cpu().tolist(),
        }
        if item.get("patient_object_id"):
            patient_id = str(item["patient_object_id"])
            predicted_patient = result["predicted_patient"].detach().cpu().numpy()
            patient_error = result["patient_squared_error"].detach().cpu().numpy()
            all_squared_errors.append(patient_error)
            target_trajectories[patient_id] = _trajectory_records(item["patient_frames"], item["target_patient_np"])
            simulated_trajectories[patient_id] = _trajectory_records(item["patient_frames"], predicted_patient)
            object_rmse[patient_id] = float(np.sqrt(np.mean(patient_error)))
            terminal_state[patient_id] = {
                "frame_index": int(item["patient_frames"][-1]),
                "position_blender_world_m": predicted_patient[-1].astype(float).tolist(),
                "velocity_blender_world_m_per_s": result["predicted_patient_velocities"][-1]
                .detach()
                .cpu()
                .tolist(),
            }
            optimized_velocities[patient_id] = result["patient_velocity"].detach().cpu().tolist()
        observed_boundary_state = {
            object_id: {
                "frame_index": int(records[-1]["frame_index"]),
                "position_blender_world_m": records[-1]["position_blender_world_m"],
                "pose_blender_world_4x4": item["terminal_pose_blender_world_4x4"][object_id].astype(float).tolist(),
            }
            for object_id, records in target_trajectories.items()
        }
        segment_payloads.append(
            {
                "segment": item["segment"],
                "status": "ok",
                "ball_object_id": ball_id,
                "agent_object_id": agent_id,
                "patient_object_id": item.get("patient_object_id"),
                "frame_range": [int(item["frame_range"][0]), int(item["frame_range"][1])],
                "fitted_frame_range": [int(item["frames"][0]), int(item["frames"][-1])],
                "frame_count": len(item["frames"]),
                "observed_frame_count_by_role": {
                    "ball": len(item["ball_frames"]),
                    "agent": len(item["agent_frames"]),
                    "patient": len(item.get("patient_frames", [])),
                },
                "dynamic_object_rmse_m": float(math.sqrt(float(result["segment_loss"].detach().cpu()))),
                "object_rmse_m": object_rmse,
                "target_trajectories": target_trajectories,
                "observed_boundary_state": observed_boundary_state,
                "physics_rollout": {
                    "simulator": "swr_backend.collision_mass_spheres",
                    "contact_proxy_policy": "3d_sphere_ball_agent_impulse_and_fixed_patient_obb",
                    "simulated_trajectories": simulated_trajectories,
                    "sphere_radius_m_by_object": {
                        ball_id: optimized_radii["ball"],
                        agent_id: optimized_radii["agent"],
                        **(
                            {str(item["patient_object_id"]): optimized_radii["patient"]}
                            if item.get("patient_object_id")
                            else {}
                        ),
                    },
                    "ball_agent_collision_events": [
                        _event_to_json(event, ball_id, agent_id) for event in result["collision_events"]
                    ],
                    "plane_contact_events": result["plane_events"],
                },
                "initialization": {
                    "ball_initial_velocity_blender_world_m_per_s": item["ball_v0_init"].astype(float).tolist(),
                    "agent_initial_velocity_blender_world_m_per_s": item["agent_v0_init"].astype(float).tolist(),
                    "patient_initial_velocity_blender_world_m_per_s": (
                        item["patient_v0_init"].astype(float).tolist() if item.get("patient_object_id") else None
                    ),
                    "ball_agent_contact_frame": int(item["ball_agent_contact_frame"]),
                },
                "optimized_initial_velocity_blender_world_m_per_s": optimized_velocities,
                "terminal_state": terminal_state,
            }
        )
    all_squared = np.concatenate([values.reshape(-1) for values in all_squared_errors])
    payload: dict[str, Any] = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.collision_mass_spheres",
        "mode": "trajectory_informed_physics_alignment",
        "message": "Physion++ mass-collision joint two-segment 3D analytic sphere SWR completed",
        "source_world_modeling_dir": str(world_modeling_dir),
        "dynamic_object_rmse_m": float(np.sqrt(np.mean(all_squared))),
        "segment_equal_weight_loss_m2": float(optimized["trajectory_loss"]),
        "regularized_optimization_loss": float(optimized["loss"]),
        "support_plane": {
            "source": support["source"],
            "up_direction_blender_world": support["up"].astype(float).tolist(),
            "gravity_direction_blender_world": support["gravity_direction"].astype(float).tolist(),
            "surface_offset_blender_world_m": float(support["plane_offset"]),
            "surface_point_blender_world_m": support["plane_point"].astype(float).tolist(),
            "pose_correction_ground_height_camera_m": support["pose_correction_ground_height_camera_m"],
        },
        "shared_role_radii_m": optimized_radii,
        "initial_shared_role_radii_m": shared_radii,
        "initial_shared_role_radius_source": "scaled_median_mesh_local_origin_to_farthest_vertex",
        "initial_radius_envelope_scale": sysid_common.COLLISION_INITIAL_RADIUS_ENVELOPE_SCALE,
        "collision_geometry_by_object": collision_geometry,
        "alignment_optimization": {
            "strategy": "swr_fit.ball_agent_impulse_mass",
            "optimizer": {
                "method": "torch_adam_full_trajectory",
                "learning_rate": float(lr),
                "steps": int(steps),
                "early_stop_patience": int(patience),
                "best_step": int(optimized["best_step"]),
                "executed_steps": int(optimized["executed_steps"]),
                "physics_dt_sec": PHYSIONPP_PHYSICS_DT_SEC,
                "contact_newton_iterations": CONTACT_NEWTON_ITERS,
                "radius_scale_range": [
                    float(math.exp(-friction_swr.MAX_RADIUS_LOG_SCALE)),
                    float(math.exp(friction_swr.MAX_RADIUS_LOG_SCALE)),
                ],
            },
            "parameter_scope": {
                "shared": [
                    "agent_mass_over_ball_mass",
                    "ball_agent_restitution",
                    "agent_ground_friction",
                    "ball_initial_speed_magnitude_m_per_s",
                    "ball_radius_m",
                    "agent_radius_m",
                ],
                "per_segment": [
                    "ball_initial_velocity_direction",
                    "agent_initial_velocity_blender_world_m_per_s",
                ],
                "seg2_only": ["patient_initial_velocity_blender_world_m_per_s"],
                "fixed": [
                    "patient_ground_friction",
                    "ball_ground_friction",
                    "patient_radius_m",
                    "gravity_magnitude",
                    "support_plane",
                    "ball_reference_mass",
                ],
            },
            "initialization": optimized["initialization"],
            "best_parameters": {
                "agent_mass_over_ball_mass": float(physics["agent_mass_over_ball_mass"].detach().cpu()),
                "ball_reference_mass": 1.0,
                "ball_agent_restitution": float(physics["ball_agent_restitution"].detach().cpu()),
                "ball_ground_friction": float(physics["ball_friction"].detach().cpu()),
                "agent_ground_friction": float(physics["agent_friction"].detach().cpu()),
                "patient_ground_friction": DEFAULT_PATIENT_FRICTION,
                "ball_initial_speed_magnitude_m_per_s": float(physics["ball_shared_speed"].detach().cpu()),
                "ball_radius_m": optimized_radii["ball"],
                "agent_radius_m": optimized_radii["agent"],
                "patient_radius_m": optimized_radii["patient"],
                "gravity_m_per_s2": friction_swr.GRAVITY_M_PER_S2,
            },
            "history": optimized["history"],
            "loss_space": "world_3d_equal_segment_equal_role",
        },
        "physion_pp_mass_collision": context,
        "segments": segment_payloads,
    }
    _write_json(output_path, payload)
    _write_json(
        output_dir / "summary.json",
        {
            "status": payload["status"],
            "backend": payload["backend"],
            "dynamic_object_rmse_m": payload["dynamic_object_rmse_m"],
            "segment_equal_weight_loss_m2": payload["segment_equal_weight_loss_m2"],
            "best_parameters": payload["alignment_optimization"]["best_parameters"],
            "segments": [
                {
                    "segment": segment["segment"],
                    "dynamic_object_rmse_m": segment["dynamic_object_rmse_m"],
                    "object_rmse_m": segment["object_rmse_m"],
                }
                for segment in segment_payloads
            ],
        },
    )
    return payload


def _render_debug_video(result: dict[str, Any], output_path: Path) -> None:
    import cv2

    segments = [item for item in result.get("segments", []) if item.get("status") == "ok"]
    if not segments:
        raise ValueError("mass_collision result has no valid segment to render")
    all_points: list[np.ndarray] = []
    for segment in segments:
        for group in (segment["target_trajectories"], segment["physics_rollout"]["simulated_trajectories"]):
            for records in group.values():
                all_points.extend(np.asarray(record["position_blender_world_m"], dtype=np.float64) for record in records)
    points = np.asarray(all_points, dtype=np.float64)
    axes = np.argsort(np.var(points, axis=0))[-2:]
    minimum = np.min(points[:, axes], axis=0)
    maximum = np.max(points[:, axes], axis=0)
    center = 0.5 * (minimum + maximum)
    extent = max(float(np.max(maximum - minimum)) * 1.2, 1e-3)
    width, height = 960, 540
    scale = 0.82 * min(width, height) / extent

    def pixel(position: list[float]) -> tuple[int, int]:
        values = np.asarray(position, dtype=np.float64)[axes]
        return (
            int(round(width * 0.5 + (values[0] - center[0]) * scale)),
            int(round(height * 0.5 - (values[1] - center[1]) * scale)),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    colors = [(50, 170, 255), (60, 70, 230), (80, 210, 80)]
    for segment in segments:
        targets = segment["target_trajectories"]
        simulated = segment["physics_rollout"]["simulated_trajectories"]
        object_ids = [segment["ball_object_id"], segment["agent_object_id"]]
        if segment.get("patient_object_id"):
            object_ids.append(segment["patient_object_id"])
        frame_maps = {
            object_id: {
                int(record["frame_index"]): record for record in records
            }
            for object_id, records in simulated.items()
        }
        target_maps = {
            object_id: {
                int(record["frame_index"]): record for record in records
            }
            for object_id, records in targets.items()
        }
        frame_indices = sorted({frame for mapping in frame_maps.values() for frame in mapping})
        for frame_index in frame_indices:
            image = np.full((height, width, 3), 245, dtype=np.uint8)
            for role_index, object_id in enumerate(object_ids):
                color = colors[role_index]
                target_record = target_maps.get(object_id, {}).get(frame_index)
                simulated_record = frame_maps.get(object_id, {}).get(frame_index)
                if target_record:
                    cv2.circle(image, pixel(target_record["position_blender_world_m"]), 10, color, 2, cv2.LINE_AA)
                if simulated_record:
                    cv2.circle(image, pixel(simulated_record["position_blender_world_m"]), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{segment['segment']} frame {frame_index} | outline=target filled=analytic rollout",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    writer.release()


def _run_learning_rate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    result = run_physionpp_mass_collision_sphere_sysid(
        world_modeling_dir=Path(payload["world_modeling_dir"]),
        output_path=Path(payload["output_dir"]) / "world_reconstruction_fit.json",
        output_dir=Path(payload["output_dir"]),
        lr=float(payload["lr"]),
        steps=int(payload["steps"]),
        patience=int(payload["patience"]),
    )
    return {
        "lr": float(payload["lr"]),
        "output_path": str(Path(payload["output_dir"]) / "world_reconstruction_fit.json"),
        "segment_equal_weight_loss_m2": float(result["segment_equal_weight_loss_m2"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physion++ mass_collision 3D analytic sphere SWR.")
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
    trials, worker_count = analytic_swr_common.run_parallel_trials(
        payloads=[
            {
                "world_modeling_dir": str(Path(args.world_modeling_dir)),
                "output_dir": str(trial_root / f"lr_{lr:g}"),
                "lr": lr,
                "steps": int(args.steps),
                "patience": int(args.patience),
                "worker_threads": int(args.worker_threads),
            }
            for lr in learning_rates
        ],
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
