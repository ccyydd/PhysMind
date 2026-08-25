from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from scripts.world_model import analytic_swr_common


GRAVITY_M_PER_S2 = 9.81
MIN_GRAVITY_SCALE = 0.5
MAX_GRAVITY_SCALE = 1.5
GRAVITY_SCALE_PRIOR_WEIGHT = 0.01
MIN_FRICTION = 1e-5
MAX_FRICTION = 1.0
RADIUS_PRIOR_WEIGHT = 0.02
MAX_RADIUS_LOG_SCALE = math.log(3.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot normalize vector: {vector}")
    return vector / norm


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _raw_from_unit_interval(value: float, lower: float, upper: float) -> float:
    clipped = float(np.clip((float(value) - lower) / max(upper - lower, 1e-12), 1e-6, 1.0 - 1e-6))
    return float(math.log(clipped / (1.0 - clipped)))


def _slope_frame(slope_degrees: float) -> dict[str, np.ndarray]:
    downslope = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    cross_slope = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    angle = math.radians(float(slope_degrees))
    # World z is up; gravity is negative z. A positive slope angle means +x is downhill.
    normal = _unit(np.asarray([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float64))
    return {
        "normal": normal,
        "downslope": downslope,
        "cross_slope": cross_slope,
        "gravity_direction": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    }


def _rollout_sphere_on_slope(
    *,
    fps: float,
    duration: float,
    radius: float,
    slope_degrees: float,
    friction: float,
    restitution: float,
    start_height_above_surface: float,
    initial_tangent_speed: float,
    gravity_magnitude: float,
) -> list[dict[str, Any]]:
    frame = _slope_frame(slope_degrees)
    normal = frame["normal"]
    downslope = frame["downslope"]
    gravity = frame["gravity_direction"] * float(gravity_magnitude)
    plane_point = np.zeros(3, dtype=np.float64)
    x = plane_point + normal * (float(radius) + float(start_height_above_surface)) - downslope * 2.0
    v = downslope * float(initial_tangent_speed)
    dt = 1.0 / float(fps)
    frame_count = int(round(float(duration) * float(fps))) + 1
    records: list[dict[str, Any]] = []
    contact_started = False
    contact_frame = None
    for frame_index in range(frame_count):
        on_plane = bool(abs(float(np.dot(x - plane_point, normal)) - float(radius)) < 1e-5)
        records.append(
            {
                "frame_index": frame_index,
                "time_sec": frame_index / float(fps),
                "position": x.astype(float).tolist(),
                "velocity": v.astype(float).tolist(),
                "speed_m_per_s": float(np.linalg.norm(v)),
                "on_plane": on_plane,
            }
        )
        if frame_index + 1 >= frame_count:
            break
        x, v, contact_started, new_contact = analytic_swr_common.advance_sphere_plane_interval_np(
            x=x,
            v=v,
            dt=dt,
            radius=float(radius),
            normal=normal,
            gravity=gravity,
            friction=float(friction),
            restitution=float(restitution),
            plane_point=plane_point,
            contact_started=contact_started,
        )
        if new_contact and contact_frame is None:
            contact_frame = frame_index
    for record in records:
        record["first_contact_frame"] = contact_frame
    return records


def _rollout_sphere_on_slope_torch(
    *,
    frame_count: int,
    fps: float,
    start_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    radius: torch.Tensor,
    slope_degrees: float,
    friction: torch.Tensor,
    restitution: float,
    gravity_magnitude: torch.Tensor,
) -> torch.Tensor:
    frame = _slope_frame(slope_degrees)
    normal = torch.tensor(frame["normal"], dtype=start_position.dtype, device=start_position.device)
    gravity = (
        torch.tensor(frame["gravity_direction"], dtype=start_position.dtype, device=start_position.device)
        * gravity_magnitude
    )
    plane_point = torch.zeros(3, dtype=start_position.dtype, device=start_position.device)
    dt = start_position.new_tensor(1.0 / float(fps))
    x = start_position.clone()
    v = initial_velocity.clone()
    positions = []
    contact_started = False
    for frame_index in range(int(frame_count)):
        positions.append(x.clone())
        if frame_index + 1 >= int(frame_count):
            break
        x, v, contact_started, _new_contact = analytic_swr_common.advance_sphere_plane_interval_torch(
            x=x,
            v=v,
            dt=dt,
            radius=radius,
            normal=normal,
            gravity=gravity,
            friction=friction,
            restitution=restitution,
            plane_point=plane_point,
            contact_started=contact_started,
        )
    return torch.stack(positions, dim=0)


def _fit_synthetic_rollout(
    *,
    target_records: list[dict[str, Any]],
    fps: float,
    slope_degrees: float,
    restitution: float,
    radius_init: float,
    friction_init: float,
    fix_radius: bool,
    optimize_gravity: bool,
    prefix_step_frames: int,
    steps_per_prefix: int,
    early_stop_patience: int,
    lr: float,
) -> dict[str, Any]:
    dtype = torch.float64
    device = torch.device("cpu")
    target_positions_np = np.asarray([record["position"] for record in target_records], dtype=np.float64)
    target_positions = torch.tensor(target_positions_np, dtype=dtype, device=device)
    start_position = target_positions[0].detach().clone()
    v0_np = np.zeros(3, dtype=np.float64)
    velocity = torch.tensor(v0_np, dtype=dtype, device=device, requires_grad=True)
    friction_raw = torch.tensor(
        _raw_from_unit_interval(friction_init, MIN_FRICTION, MAX_FRICTION),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    radius_raw = None if fix_radius else torch.zeros((), dtype=dtype, device=device, requires_grad=True)
    radius_init_tensor = torch.tensor(float(radius_init), dtype=dtype, device=device)
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
    parameters = [velocity, friction_raw]
    if radius_raw is not None:
        parameters.append(radius_raw)
    if gravity_scale_raw is not None:
        parameters.append(gravity_scale_raw)

    def evaluate(train_end: int, validation_end: int) -> dict[str, Any]:
        friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        if radius_raw is None:
            radius_log_scale = radius_init_tensor.new_tensor(0.0)
            radius = radius_init_tensor
        else:
            radius_log_scale = torch.clamp(radius_raw, -MAX_RADIUS_LOG_SCALE, MAX_RADIUS_LOG_SCALE)
            radius = radius_init_tensor * torch.exp(radius_log_scale)
        gravity_scale = (
            radius_init_tensor.new_tensor(1.0)
            if gravity_scale_raw is None
            else MIN_GRAVITY_SCALE
            + (MAX_GRAVITY_SCALE - MIN_GRAVITY_SCALE) * torch.sigmoid(gravity_scale_raw)
        )
        gravity_magnitude = radius_init_tensor.new_tensor(GRAVITY_M_PER_S2) * gravity_scale
        predicted = _rollout_sphere_on_slope_torch(
            frame_count=len(target_records),
            fps=fps,
            start_position=start_position,
            initial_velocity=velocity,
            radius=radius,
            slope_degrees=slope_degrees,
            friction=friction,
            restitution=restitution,
            gravity_magnitude=gravity_magnitude,
        )
        prefix_rmse = torch.sqrt(torch.mean(torch.sum((predicted[:train_end] - target_positions[:train_end]) ** 2, dim=1)))
        validation_rmse = torch.sqrt(
            torch.mean(torch.sum((predicted[:validation_end] - target_positions[:validation_end]) ** 2, dim=1))
        )
        loss = (
            prefix_rmse
            + target_positions.new_tensor(RADIUS_PRIOR_WEIGHT) * radius_log_scale * radius_log_scale
            + target_positions.new_tensor(GRAVITY_SCALE_PRIOR_WEIGHT) * (gravity_scale - 1.0) ** 2
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("synthetic slope curriculum produced a non-finite loss")
        return {
            "loss": loss,
            "selection_metric": float(validation_rmse.detach().cpu()),
            "prefix_rmse": float(prefix_rmse.detach().cpu()),
            "predicted": predicted,
            "friction": friction,
            "radius": radius,
            "gravity_scale": gravity_scale,
            "gravity_magnitude": gravity_magnitude,
        }

    curriculum_schedule = analytic_swr_common.optimize_adam_prefix_curriculum(
        parameters=parameters,
        evaluate=evaluate,
        observed_frames=len(target_records),
        prefix_step_frames=prefix_step_frames,
        steps_per_prefix=steps_per_prefix,
        patience=early_stop_patience,
        lr=lr,
    )
    final_evaluation = evaluate(len(target_records), len(target_records))
    predicted = final_evaluation["predicted"].detach().cpu().numpy().astype(float)
    final_rmse = float(final_evaluation["prefix_rmse"])
    final_friction = float(final_evaluation["friction"].detach().cpu())
    final_radius = float(final_evaluation["radius"].detach().cpu())
    final_gravity_scale = float(final_evaluation["gravity_scale"].detach().cpu())
    final_gravity_magnitude = float(final_evaluation["gravity_magnitude"].detach().cpu())
    predicted_records = []
    for source, position in zip(target_records, predicted):
        predicted_records.append(
            {
                "frame_index": int(source["frame_index"]),
                "time_sec": float(source["time_sec"]),
                "position": position.astype(float).tolist(),
                "velocity": [],
                "speed_m_per_s": 0.0,
                "on_plane": False,
                "first_contact_frame": source.get("first_contact_frame"),
            }
        )
    return {
        "status": "ok",
        "optimizer": {
            "method": "torch_adam_prefix_curriculum",
            "lr": float(lr),
            "prefix_step_frames": int(prefix_step_frames),
            "steps_per_prefix": int(steps_per_prefix),
            "early_stop_patience": int(early_stop_patience),
            "curriculum_schedule": curriculum_schedule,
            "parameterization": "sigmoid_friction_log_radius_scale_sigmoid_gravity_scale",
            "radius_prior_weight": RADIUS_PRIOR_WEIGHT,
            "gravity_scale_prior_weight": GRAVITY_SCALE_PRIOR_WEIGHT,
            "rollout": "analytic_plane_contact",
        },
        "fit_rmse_m": final_rmse,
        "parameters": {
            "initial_velocity_m_per_s": velocity.detach().cpu().numpy().astype(float).tolist(),
            "friction": final_friction,
            "radius_m": final_radius,
            "radius_fixed": bool(fix_radius),
            "gravity_scale": final_gravity_scale,
            "gravity_m_per_s2": final_gravity_magnitude,
            "gravity_fixed": not bool(optimize_gravity),
        },
        "predicted_trajectory": predicted_records,
    }


def _fit_synthetic_lr_trial(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(max(int(payload.get("worker_threads", 1)), 1))
    result = _fit_synthetic_rollout(
        target_records=payload["target_records"],
        fps=float(payload["fps"]),
        slope_degrees=float(payload["slope_degrees"]),
        restitution=float(payload["restitution"]),
        radius_init=float(payload["radius_init"]),
        friction_init=float(payload["friction_init"]),
        fix_radius=bool(payload["fix_radius"]),
        optimize_gravity=bool(payload["optimize_gravity"]),
        prefix_step_frames=int(payload["prefix_step_frames"]),
        steps_per_prefix=int(payload["steps_per_prefix"]),
        early_stop_patience=int(payload["early_stop_patience"]),
        lr=float(payload["lr"]),
    )
    return {
        "lr": float(payload["lr"]),
        "fit_rmse_m": float(result["fit_rmse_m"]),
        "result": result,
    }


def _look_at_camera(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _unit(target - position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _unit(np.cross(forward, world_up))
    up = _unit(np.cross(right, forward))
    return right, up, forward


def _project_perspective(
    point: np.ndarray,
    *,
    camera_position: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    focal_px: float,
    cx: float,
    cy: float,
) -> tuple[int, int] | None:
    rel = point - camera_position
    depth = float(np.dot(rel, forward))
    if depth <= 1e-6:
        return None
    u = cx + focal_px * float(np.dot(rel, right)) / depth
    v = cy - focal_px * float(np.dot(rel, up)) / depth
    return int(round(u)), int(round(v))


def _draw_text(image: np.ndarray, text: str, y: int) -> None:
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)


def _draw_ball(image: np.ndarray, center: tuple[int, int], radius_px: int, color: tuple[int, int, int]) -> None:
    cv2.circle(image, center, radius_px, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(image, center, max(1, radius_px - 2), color, -1, cv2.LINE_AA)
    highlight = (center[0] - radius_px // 3, center[1] - radius_px // 3)
    cv2.circle(image, highlight, max(2, radius_px // 5), (255, 190, 190), -1, cv2.LINE_AA)


def _render_physion_like(
    *,
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    width = int(payload["render"]["width"])
    height = int(payload["render"]["height"])
    fps = float(payload["config"]["fps"])
    radius = float(payload["config"]["radius_m"])
    trajectory = payload["trajectory"]
    positions = np.asarray([record["position"] for record in trajectory], dtype=np.float64)
    center = np.mean(positions, axis=0)
    extent = np.maximum(np.ptp(positions, axis=0), 1e-3)
    scene_scale = max(float(extent[0]), float(extent[2]), 1.0)
    camera_position = center + np.asarray([-0.35 * scene_scale, -1.35 * scene_scale, 0.70 * scene_scale], dtype=np.float64)
    right, up, forward = _look_at_camera(camera_position, center)
    focal_px = min(width, height) * 0.72
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    plane_x = np.linspace(float(positions[:, 0].min()) - 1.0, float(positions[:, 0].max()) + 1.0, 8)
    plane_y = np.linspace(-1.2, 1.2, 4)
    slope_angle = math.radians(float(payload["config"]["slope_degrees"]))
    plane_points = [np.asarray([x, y, -math.tan(slope_angle) * x], dtype=np.float64) for x in plane_x for y in plane_y]
    for record in trajectory:
        image = np.full((height, width, 3), (236, 238, 240), dtype=np.uint8)
        projected = [
            _project_perspective(
                point,
                camera_position=camera_position,
                right=right,
                up=up,
                forward=forward,
                focal_px=focal_px,
                cx=width / 2.0,
                cy=height / 2.0,
            )
            for point in plane_points
        ]
        projected = [point for point in projected if point is not None]
        if projected:
            hull = cv2.convexHull(np.asarray(projected, dtype=np.int32))
            cv2.fillConvexPoly(image, hull, (185, 184, 174), cv2.LINE_AA)
            cv2.polylines(image, [hull], True, (120, 120, 112), 2, cv2.LINE_AA)
        for past in trajectory[: int(record["frame_index"]) + 1]:
            p = _project_perspective(
                np.asarray(past["position"], dtype=np.float64),
                camera_position=camera_position,
                right=right,
                up=up,
                forward=forward,
                focal_px=focal_px,
                cx=width / 2.0,
                cy=height / 2.0,
            )
            if p is not None:
                cv2.circle(image, p, 2, (220, 90, 60), -1, cv2.LINE_AA)
        ball_center = _project_perspective(
            np.asarray(record["position"], dtype=np.float64),
            camera_position=camera_position,
            right=right,
            up=up,
            forward=forward,
            focal_px=focal_px,
            cx=width / 2.0,
            cy=height / 2.0,
        )
        if ball_center is not None:
            rel = np.asarray(record["position"], dtype=np.float64) - camera_position
            depth = max(float(np.dot(rel, forward)), 1e-6)
            _draw_ball(image, ball_center, max(6, int(round(focal_px * radius / depth))), (45, 85, 230))
        _draw_text(image, "Physion++-like synthetic view", 24)
        _draw_text(image, f"frame={record['frame_index']} t={record['time_sec']:.2f}s speed={record['speed_m_per_s']:.2f} m/s", 48)
        writer.write(image)
    writer.release()


def _render_side_view(
    *,
    payload: dict[str, Any],
    output_path: Path,
    fitted: list[dict[str, Any]] | None = None,
    fitted_radius: float | None = None,
) -> None:
    width = int(payload["render"]["width"])
    height = int(payload["render"]["height"])
    fps = float(payload["config"]["fps"])
    radius = float(payload["config"]["radius_m"])
    trajectory = payload["trajectory"]
    all_records = trajectory + (fitted or [])
    positions = np.asarray([record["position"] for record in all_records], dtype=np.float64)
    x_min, x_max = float(positions[:, 0].min()) - 0.6, float(positions[:, 0].max()) + 0.6
    z_min, z_max = float(positions[:, 2].min()) - 0.4, float(positions[:, 2].max()) + 0.6
    slope_angle = math.radians(float(payload["config"]["slope_degrees"]))
    scale = 0.88 * min(width / max(x_max - x_min, 1e-6), height / max(z_max - z_min, 1e-6))
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)

    def project(point: np.ndarray) -> tuple[int, int]:
        u = width / 2.0 + (float(point[0]) - x_center) * scale
        v = height / 2.0 - (float(point[2]) - z_center) * scale
        return int(round(u)), int(round(v))

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    plane_a = np.asarray([x_min - 1.0, 0.0, -math.tan(slope_angle) * (x_min - 1.0)], dtype=np.float64)
    plane_b = np.asarray([x_max + 1.0, 0.0, -math.tan(slope_angle) * (x_max + 1.0)], dtype=np.float64)
    for record in trajectory:
        image = np.full((height, width, 3), (246, 247, 248), dtype=np.uint8)
        pa, pb = project(plane_a), project(plane_b)
        ground_poly = np.asarray([pa, pb, (width + 60, height + 80), (-60, height + 80)], dtype=np.int32)
        cv2.fillPoly(image, [ground_poly], (184, 184, 174), cv2.LINE_AA)
        cv2.line(image, pa, pb, (90, 90, 86), 3, cv2.LINE_AA)
        pts = [project(np.asarray(past["position"], dtype=np.float64)) for past in trajectory[: int(record["frame_index"]) + 1]]
        if len(pts) >= 2:
            cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (70, 120, 230), 2, cv2.LINE_AA)
        if fitted:
            fit_radius = float(fitted_radius if fitted_radius is not None else radius)
            fit_pts = [
                project(np.asarray(past["position"], dtype=np.float64))
                for past in fitted[: int(record["frame_index"]) + 1]
            ]
            if len(fit_pts) >= 2:
                cv2.polylines(image, [np.asarray(fit_pts, dtype=np.int32)], False, (60, 180, 70), 2, cv2.LINE_AA)
            fit_center = project(np.asarray(fitted[int(record["frame_index"])]["position"], dtype=np.float64))
            cv2.circle(image, fit_center, max(4, int(round(fit_radius * scale))), (60, 180, 70), -1, cv2.LINE_AA)
        center = project(np.asarray(record["position"], dtype=np.float64))
        _draw_ball(image, center, max(5, int(round(radius * scale))), (45, 85, 230))
        _draw_text(image, "Slope side view", 24)
        label = f"frame={record['frame_index']} contact={record['first_contact_frame']} on_plane={record['on_plane']}"
        if fitted:
            label += " target=blue fit=green"
        _draw_text(image, label, 48)
        writer.write(image)
    writer.release()


def _render_videos(payload: dict[str, Any], output_dir: Path, fit_result: dict[str, Any] | None = None) -> list[str]:
    physion_like = output_dir / "synthetic_slope_physion_like.mp4"
    side = output_dir / "synthetic_slope_side_view.mp4"
    _render_physion_like(payload=payload, output_path=physion_like)
    _render_side_view(payload=payload, output_path=side)
    videos = [str(physion_like.resolve()), str(side.resolve())]
    if fit_result:
        overlay = output_dir / "synthetic_slope_fit_overlay_side_view.mp4"
        _render_side_view(
            payload=payload,
            output_path=overlay,
            fitted=fit_result["predicted_trajectory"],
            fitted_radius=float(fit_result.get("parameters", {}).get("radius_m", payload["config"]["radius_m"])),
        )
        videos.append(str(overlay.resolve()))
    return videos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/tmp_physionpp_slope_synthetic")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--radius", type=float, default=0.18)
    parser.add_argument("--slope-degrees", type=float, default=18.0)
    parser.add_argument("--friction", type=float, default=0.35)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--start-height", type=float, default=1.5)
    parser.add_argument("--initial-tangent-speed", type=float, default=0.15)
    parser.add_argument("--gravity-m-per-s2", type=float, default=GRAVITY_M_PER_S2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--fit-lrs", default="0.01,0.025,0.05")
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--fit-worker-threads", type=int, default=1)
    parser.add_argument("--curriculum-prefix-step-frames", type=int, default=10)
    parser.add_argument("--curriculum-steps-per-prefix", type=int, default=400)
    parser.add_argument("--curriculum-early-stop-patience", type=int, default=100)
    parser.add_argument("--fit-friction-init", type=float, default=0.1)
    parser.add_argument("--fit-radius-init-scale", type=float, default=1.0)
    parser.add_argument("--fit-optimize-radius", action="store_true")
    parser.add_argument("--fit-fix-gravity", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _slope_frame(args.slope_degrees)
    trajectory = _rollout_sphere_on_slope(
        fps=args.fps,
        duration=args.duration,
        radius=args.radius,
        slope_degrees=args.slope_degrees,
        friction=args.friction,
        restitution=args.restitution,
        start_height_above_surface=args.start_height,
        initial_tangent_speed=args.initial_tangent_speed,
        gravity_magnitude=args.gravity_m_per_s2,
    )
    payload = {
        "config": {
            "fps": float(args.fps),
            "duration_sec": float(args.duration),
            "rollout": "analytic_plane_contact",
            "radius_m": float(args.radius),
            "slope_degrees": float(args.slope_degrees),
            "friction": float(args.friction),
            "restitution": float(args.restitution),
            "start_height_above_surface_m": float(args.start_height),
            "initial_tangent_speed_m_per_s": float(args.initial_tangent_speed),
            "gravity_m_per_s2": float(args.gravity_m_per_s2),
        },
        "slope": {
            "normal": frame["normal"].astype(float).tolist(),
            "downslope": frame["downslope"].astype(float).tolist(),
            "cross_slope": frame["cross_slope"].astype(float).tolist(),
            "gravity_direction": frame["gravity_direction"].astype(float).tolist(),
        },
        "trajectory": trajectory,
        "render": {
            "width": int(args.width),
            "height": int(args.height),
        },
    }
    payload_path = output_dir / "synthetic_slope_rollout.json"
    _write_json(payload_path, payload)
    _write_json(output_dir / "config.json", payload["config"])
    fit_result = None
    if args.fit:
        candidate_lrs = [float(item.strip()) for item in str(args.fit_lrs).split(",") if item.strip()]
        if not candidate_lrs:
            raise ValueError("--fit-lrs must contain at least one learning rate")
        trial_payloads = [
            {
                "target_records": trajectory,
                "fps": float(args.fps),
                "slope_degrees": float(args.slope_degrees),
                "restitution": float(args.restitution),
                "radius_init": float(args.radius) * float(args.fit_radius_init_scale),
                "friction_init": float(args.fit_friction_init),
                "fix_radius": not bool(args.fit_optimize_radius),
                "optimize_gravity": not bool(args.fit_fix_gravity),
                "prefix_step_frames": int(args.curriculum_prefix_step_frames),
                "steps_per_prefix": int(args.curriculum_steps_per_prefix),
                "early_stop_patience": int(args.curriculum_early_stop_patience),
                "lr": lr,
                "worker_threads": int(args.fit_worker_threads),
            }
            for lr in candidate_lrs
        ]
        trials, worker_count = analytic_swr_common.run_parallel_trials(
            payloads=trial_payloads,
            worker=_fit_synthetic_lr_trial,
            max_workers=int(args.fit_workers),
        )
        trials.sort(key=lambda item: candidate_lrs.index(float(item["lr"])))
        selected = min(trials, key=lambda item: float(item["fit_rmse_m"]))
        fit_result = selected["result"]
        fit_result["optimizer"]["lr_multistart"] = {
            "candidate_lrs": candidate_lrs,
            "workers": int(worker_count),
            "selected_lr": float(selected["lr"]),
            "selection_metric": "full_trajectory_rmse_m",
            "trials": [
                {"lr": float(item["lr"]), "fit_rmse_m": float(item["fit_rmse_m"])}
                for item in trials
            ],
        }
        _write_json(output_dir / "fit_result.json", fit_result)
    videos = [] if args.skip_render else _render_videos(payload, output_dir, fit_result=fit_result)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "trajectory": str(payload_path.resolve()),
                "fit_result": str((output_dir / "fit_result.json").resolve()) if fit_result else None,
                "fit_rmse_m": fit_result.get("fit_rmse_m") if fit_result else None,
                "videos": videos,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
