from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


def sphere_sphere_impulse_update_torch(
    *,
    velocities: torch.Tensor,
    i: int,
    j: int,
    normal_from_j_to_i: torch.Tensor,
    mass_i: torch.Tensor,
    mass_j: torch.Tensor,
    restitution: torch.Tensor,
    require_approaching: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a frictionless normal impulse to two translational bodies."""
    normal = normal_from_j_to_i / torch.clamp(
        torch.linalg.vector_norm(normal_from_j_to_i),
        min=normal_from_j_to_i.new_tensor(1e-12),
    )
    relative_normal_velocity = torch.sum((velocities[i] - velocities[j]) * normal)
    if require_approaching and not bool((relative_normal_velocity < 0.0).detach().cpu()):
        return velocities, torch.zeros_like(normal), relative_normal_velocity
    impulse_scalar = -(1.0 + restitution) * relative_normal_velocity
    impulse_scalar = impulse_scalar / (1.0 / mass_i + 1.0 / mass_j)
    impulse = impulse_scalar * normal
    updated = velocities.clone()
    updated[i] = updated[i] + impulse / mass_i
    updated[j] = updated[j] - impulse / mass_j
    return updated, impulse, relative_normal_velocity


def _first_contact_time_values(*, signed0: float, vn0: float, an: float, dt: float) -> list[float]:
    candidates: list[float] = []
    if abs(an) > 1e-12:
        discriminant = vn0 * vn0 - 2.0 * an * signed0
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            candidates.extend([(-vn0 - root) / an, (-vn0 + root) / an])
    elif abs(vn0) > 1e-12:
        candidates.append(-signed0 / vn0)
    return [float(value) for value in candidates if -1e-9 <= value <= dt + 1e-9]


def first_sphere_plane_contact_time_np(
    *,
    x: np.ndarray,
    v: np.ndarray,
    radius: float,
    plane_point: np.ndarray,
    normal: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> float | None:
    signed0 = float(np.dot(x - plane_point, normal) - radius)
    candidates = _first_contact_time_values(
        signed0=signed0,
        vn0=float(np.dot(v, normal)),
        an=float(np.dot(gravity, normal)),
        dt=float(dt),
    )
    return min(max(0.0, value) for value in candidates) if candidates else None


def first_sphere_plane_contact_time_torch(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    radius: torch.Tensor,
    plane_point: torch.Tensor,
    normal: torch.Tensor,
    gravity: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor | None:
    signed0 = torch.sum((x - plane_point) * normal) - radius
    vn0 = torch.sum(v * normal)
    an = torch.sum(gravity * normal)
    dt_value = float(dt.detach().cpu())
    candidates: list[torch.Tensor] = []
    if abs(float(an.detach().cpu())) > 1e-12:
        discriminant = vn0 * vn0 - 2.0 * an * signed0
        if bool((discriminant >= 0.0).detach().cpu()):
            root = torch.sqrt(torch.clamp(discriminant, min=0.0))
            candidates.extend([(-vn0 - root) / an, (-vn0 + root) / an])
    elif abs(float(vn0.detach().cpu())) > 1e-12:
        candidates.append(-signed0 / vn0)
    valid = [
        value
        for value in candidates
        if -1e-9 <= float(value.detach().cpu()) <= dt_value + 1e-9
    ]
    if not valid:
        return None
    valid.sort(key=lambda value: float(value.detach().cpu()))
    return torch.clamp(valid[0], min=0.0, max=dt_value)


def advance_on_plane_np(
    *,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    radius: float,
    plane_point: np.ndarray,
    normal: np.ndarray,
    gravity: np.ndarray,
    friction: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed = float(np.dot(x - plane_point, normal) - radius)
    x = x - signed * normal
    v = v - float(np.dot(v, normal)) * normal
    gravity_tangent = gravity - float(np.dot(gravity, normal)) * normal
    normal_acceleration = abs(float(np.dot(gravity, normal)))
    speed = float(np.linalg.norm(v))
    tangent_gravity = float(np.linalg.norm(gravity_tangent))
    if speed <= 1e-12 and friction * normal_acceleration >= tangent_gravity:
        return x, np.zeros_like(v)
    direction = (
        v / speed
        if speed > 1e-12
        else gravity_tangent / max(tangent_gravity, 1e-12)
    )
    acceleration = gravity_tangent - friction * normal_acceleration * direction
    v_next = v + acceleration * dt
    if speed > 1e-12 and float(np.dot(v_next, v)) < 0.0:
        acceleration_sq = float(np.dot(acceleration, acceleration))
        stop_t = float(np.clip(-float(np.dot(v, acceleration)) / max(acceleration_sq, 1e-12), 0.0, dt))
        stopped_velocity = v + acceleration * stop_t
        if float(np.linalg.norm(stopped_velocity)) <= 1e-7:
            x = x + v * stop_t + 0.5 * acceleration * stop_t * stop_t
            v_next = np.zeros_like(v)
        else:
            x = x + v * dt + 0.5 * acceleration * dt * dt
    else:
        x = x + v * dt + 0.5 * acceleration * dt * dt
    signed = float(np.dot(x - plane_point, normal) - radius)
    x = x - signed * normal
    return x, v_next - float(np.dot(v_next, normal)) * normal


def advance_on_plane_torch(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    radius: torch.Tensor,
    plane_point: torch.Tensor,
    normal: torch.Tensor,
    gravity: torch.Tensor,
    friction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    signed = torch.sum((x - plane_point) * normal) - radius
    x = x - signed * normal
    v = v - torch.sum(v * normal) * normal
    gravity_tangent = gravity - torch.sum(gravity * normal) * normal
    normal_acceleration = torch.abs(torch.sum(gravity * normal))
    speed = torch.linalg.vector_norm(v)
    tangent_gravity = torch.linalg.vector_norm(gravity_tangent)
    moving = bool((speed > 1e-6).detach().cpu())
    if not moving and bool((friction * normal_acceleration >= tangent_gravity).detach().cpu()):
        return x, torch.zeros_like(v)
    direction = v / torch.clamp(speed, min=1e-12) if moving else gravity_tangent / torch.clamp(tangent_gravity, min=1e-12)
    acceleration = gravity_tangent - friction * normal_acceleration * direction
    v_next = v + acceleration * dt
    if moving and bool((torch.sum(v_next * v) < 0.0).detach().cpu()):
        acceleration_sq = torch.sum(acceleration * acceleration)
        stop_t = torch.clamp(-torch.sum(v * acceleration) / torch.clamp(acceleration_sq, min=1e-12), min=0.0, max=float(dt.detach().cpu()))
        stopped_velocity = v + acceleration * stop_t
        if bool((torch.linalg.vector_norm(stopped_velocity) <= 1e-7).detach().cpu()):
            x = x + v * stop_t + 0.5 * acceleration * stop_t * stop_t
            v_next = torch.zeros_like(v)
        else:
            x = x + v * dt + 0.5 * acceleration * dt * dt
    else:
        x = x + v * dt + 0.5 * acceleration * dt * dt
    signed = torch.sum((x - plane_point) * normal) - radius
    x = x - signed * normal
    return x, v_next - torch.sum(v_next * normal) * normal


def advance_sphere_plane_interval_np(
    *,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    radius: float,
    plane_point: np.ndarray,
    normal: np.ndarray,
    gravity: np.ndarray,
    friction: float,
    restitution: float,
    contact_started: bool,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    signed0 = float(np.dot(x - plane_point, normal) - radius)
    if signed0 <= 1e-9 or contact_started:
        x_next, v_next = advance_on_plane_np(
            x=x, v=v, dt=dt, radius=radius, plane_point=plane_point,
            normal=normal, gravity=gravity, friction=friction,
        )
        return x_next, v_next, True, False
    x_free = x + v * dt + 0.5 * gravity * dt * dt
    v_free = v + gravity * dt
    if float(np.dot(x_free - plane_point, normal) - radius) > 0.0:
        return x_free, v_free, False, False
    contact_time = first_sphere_plane_contact_time_np(
        x=x, v=v, radius=radius, plane_point=plane_point, normal=normal,
        gravity=gravity, dt=dt,
    )
    if contact_time is None:
        return x_free, v_free, False, False
    x_hit = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
    v_hit = v + gravity * contact_time
    x_hit = x_hit - float(np.dot(x_hit - plane_point, normal) - radius) * normal
    normal_velocity = float(np.dot(v_hit, normal))
    v_hit = v_hit - normal_velocity * normal - restitution * min(normal_velocity, 0.0) * normal
    x_next, v_next = advance_on_plane_np(
        x=x_hit, v=v_hit, dt=dt - contact_time, radius=radius,
        plane_point=plane_point, normal=normal, gravity=gravity, friction=friction,
    )
    return x_next, v_next, True, True


def advance_sphere_plane_interval_torch(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    radius: torch.Tensor,
    plane_point: torch.Tensor,
    normal: torch.Tensor,
    gravity: torch.Tensor,
    friction: torch.Tensor,
    restitution: float,
    contact_started: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
    signed0 = torch.sum((x - plane_point) * normal) - radius
    if bool((signed0 <= 1e-9).detach().cpu()) or contact_started:
        x_next, v_next = advance_on_plane_torch(
            x=x, v=v, dt=dt, radius=radius, plane_point=plane_point,
            normal=normal, gravity=gravity, friction=friction,
        )
        return x_next, v_next, True, False
    x_free = x + v * dt + 0.5 * gravity * dt * dt
    v_free = v + gravity * dt
    if bool((torch.sum((x_free - plane_point) * normal) - radius > 0.0).detach().cpu()):
        return x_free, v_free, False, False
    contact_time = first_sphere_plane_contact_time_torch(
        x=x, v=v, radius=radius, plane_point=plane_point, normal=normal,
        gravity=gravity, dt=dt,
    )
    if contact_time is None:
        return x_free, v_free, False, False
    x_hit = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
    v_hit = v + gravity * contact_time
    x_hit = x_hit - (torch.sum((x_hit - plane_point) * normal) - radius) * normal
    normal_velocity = torch.sum(v_hit * normal)
    v_hit = v_hit - normal_velocity * normal - restitution * torch.minimum(normal_velocity, normal_velocity.new_tensor(0.0)) * normal
    x_next, v_next = advance_on_plane_torch(
        x=x_hit, v=v_hit, dt=dt - contact_time, radius=radius,
        plane_point=plane_point, normal=normal, gravity=gravity, friction=friction,
    )
    return x_next, v_next, True, True


def closed_form_constant_deceleration(
    *,
    x0: torch.Tensor,
    v0: torch.Tensor,
    deceleration: torch.Tensor,
    t: torch.Tensor,
    t0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate straight-line Coulomb deceleration with exact stopping."""
    if t.ndim == 1 and t0.ndim == 1 and v0.ndim >= 2 and int(t0.numel()) == int(v0.shape[-2]):
        dt = t.reshape(-1, 1, 1) - t0.reshape(1, -1, 1)
    else:
        dt = t - t0
        while dt.ndim < v0.ndim:
            dt = dt.unsqueeze(-1)
    dt = torch.clamp(dt, min=0.0)
    speed0 = torch.sqrt(torch.sum(v0 * v0, dim=-1, keepdim=True) + 1e-12)
    direction = v0 / speed0
    acceleration = deceleration.reshape(-1, 1)
    moving_dt = torch.minimum(dt, speed0 / torch.clamp(acceleration, min=1e-12))
    distance = speed0 * moving_dt - 0.5 * acceleration * moving_dt * moving_dt
    speed = torch.clamp(speed0 - acceleration * dt, min=0.0)
    return x0 + direction * distance, direction * speed


def constant_deceleration_stop_time(
    *,
    start_speed: float,
    start_time: float,
    deceleration: float,
) -> float:
    if start_speed <= 1e-10 or deceleration <= 1e-12:
        return float(start_time)
    return float(start_time) + float(start_speed) / float(deceleration)


def constant_deceleration_quadratic_coefficients(
    *,
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_time: float,
    deceleration: float,
    segment_start: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.asarray(start_position, dtype=np.float64)
    v0 = np.asarray(start_velocity, dtype=np.float64)
    speed = float(np.linalg.norm(v0))
    if speed <= 1e-10 or deceleration <= 1e-12:
        return x0, v0 * 0.0, v0 * 0.0
    direction = v0 / speed
    stop_time = constant_deceleration_stop_time(
        start_speed=speed,
        start_time=start_time,
        deceleration=deceleration,
    )
    if segment_start >= stop_time - 1e-10:
        stop_distance = 0.5 * speed * speed / deceleration
        return x0 + direction * stop_distance, v0 * 0.0, v0 * 0.0
    c2 = -0.5 * deceleration
    c1 = speed + deceleration * start_time
    c0 = -speed * start_time - 0.5 * deceleration * start_time * start_time
    return x0 + direction * c0, direction * c1, direction * c2


def curriculum_prefix_values(*, observed_frames: int, prefix_step_frames: int) -> list[int]:
    if int(prefix_step_frames) <= 0:
        return [0]
    step = max(int(prefix_step_frames), 1)
    values = list(range(step, max(int(observed_frames), 0), step))
    values.append(0)
    return values


@dataclass
class CurriculumStageBest:
    metric_value: float = float("inf")
    loss: float = float("inf")
    step: int = 0
    prefix_rmse: float = float("nan")
    state: Any = None
    steps_since_best: int = 0

    def consider(
        self,
        *,
        metric_value: float,
        loss: float,
        step: int,
        prefix_rmse: float,
        state: Any,
    ) -> bool:
        if not math.isfinite(float(metric_value)) or float(metric_value) >= self.metric_value:
            self.steps_since_best += 1
            return False
        self.metric_value = float(metric_value)
        self.loss = float(loss)
        self.step = int(step)
        self.prefix_rmse = float(prefix_rmse)
        self.state = state
        self.steps_since_best = 0
        return True

    def should_stop(self, patience: int) -> bool:
        return int(patience) > 0 and self.steps_since_best >= int(patience)


def clone_parameter_state(parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def restore_parameter_state(parameters: list[torch.Tensor], state: list[torch.Tensor] | None) -> None:
    if state is None:
        return
    with torch.no_grad():
        for parameter, value in zip(parameters, state):
            parameter.copy_(value)


def optimize_adam_prefix_curriculum(
    *,
    parameters: list[torch.Tensor],
    evaluate: Callable[[int, int], dict[str, Any]],
    observed_frames: int,
    prefix_step_frames: int,
    steps_per_prefix: int,
    patience: int,
    lr: float,
) -> list[dict[str, Any]]:
    """Optimize cumulative trajectory prefixes and restore each prefix's best state."""
    optimizer = torch.optim.Adam(parameters, lr=float(lr), weight_decay=0.0)
    schedule: list[dict[str, Any]] = []
    global_step = 0
    prefixes = curriculum_prefix_values(
        observed_frames=observed_frames,
        prefix_step_frames=prefix_step_frames,
    )
    for prefix_frames in prefixes:
        train_end = int(observed_frames) if prefix_frames == 0 else min(int(prefix_frames), int(observed_frames))
        validation_end = min(train_end + max(int(prefix_step_frames), 1), int(observed_frames))
        stage_best = CurriculumStageBest()
        steps_run = 0
        early_stopped = False
        for _local_step in range(max(int(steps_per_prefix), 1)):
            optimizer.zero_grad()
            evaluation = evaluate(train_end, validation_end)
            loss = evaluation["loss"]
            loss_value = float(loss.detach().cpu())
            stage_best.consider(
                metric_value=float(evaluation["selection_metric"]),
                loss=loss_value,
                step=global_step + 1,
                prefix_rmse=float(evaluation["prefix_rmse"]),
                state=clone_parameter_state(parameters),
            )
            loss.backward()
            optimizer.step()
            global_step += 1
            steps_run += 1
            if stage_best.should_stop(patience):
                early_stopped = True
                break
        restore_parameter_state(parameters, stage_best.state)
        schedule.append(
            {
                "prefix_frames": None if prefix_frames == 0 else int(prefix_frames),
                "training_frame_count": train_end,
                "validation_frame_count": validation_end,
                "steps_run": steps_run,
                "best_step": stage_best.step,
                "best_loss": stage_best.loss,
                "best_metric_value": stage_best.metric_value,
                "best_prefix_rmse_m": stage_best.prefix_rmse,
                "early_stop_patience": int(patience),
                "early_stopped": early_stopped,
            }
        )
    return schedule


def run_parallel_trials(
    *,
    payloads: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    worker_count = max(1, min(int(max_workers), len(payloads)))
    if worker_count == 1:
        return [worker(payload) for payload in payloads], worker_count
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, payload) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results, worker_count
