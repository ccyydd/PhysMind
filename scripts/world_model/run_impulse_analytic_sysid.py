from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares

from scripts.world_model import analytic_swr_common
from scripts.world_model import swr_sysid_common as sysid_common


GRAVITY = 9.806
CONTACT_ASSUMPTIONS = ("known_position_time", "known_time", "unknown_contact")
MODE_IMPULSE_FORCED_FRAME = "impulse-forced-frame"
MODE_IMPULSE_FORCED_WINDOW = "impulse-forced-window"
MODE_NORMAL_COLLISION = "normal-collision"
IMPULSE_STAGE_MODES = (MODE_IMPULSE_FORCED_FRAME, MODE_IMPULSE_FORCED_WINDOW)
SHAPE_COLLISION_LINE_INSIDE_FRACTION_THRESHOLD = 2.0 / 3.0
SHAPE_COLLISION_LINE_SAMPLE_COUNT = 65
SHAPE_COLLISION_CLUSTER_GAP_MISSING_FRAMES = 2
MIN_MASS = 0.05
MAX_MASS = 20.0
MIN_FRICTION = 1e-5
MAX_FRICTION = 1.0
MIN_PAIR_MASS_RATIO = 0.05
MAX_PAIR_MASS_RATIO = 20.0
OBJECT_SOLVE_RANK_TOLERANCE = 1e-10
OBJECT_SOLVE_NULLSPACE_PRIOR_WEIGHT = 1.0
OBJECT_SOLVE_RESTITUTION_EQUALITY_PRIOR_WEIGHT = 1.0
EVENT_SOURCE_DETECTED_WINDOW = "detected_window"
EVENT_SOURCE_FREE_ROLLOUT = "free_rollout"
EVENT_SOURCES = (EVENT_SOURCE_DETECTED_WINDOW, EVENT_SOURCE_FREE_ROLLOUT)
CURRICULUM_BEST_PREFIX_LOSS = "prefix_loss"
CURRICULUM_BEST_FULL_ROLLOUT_RMSE = "full_rollout_rmse"
CURRICULUM_BEST_NEXT_WINDOW_RMSE = "next_window_rmse"
CURRICULUM_BEST_FUTURE_ROLLOUT_RMSE = "future_rollout_rmse"
CURRICULUM_BEST_METRICS = (
    CURRICULUM_BEST_PREFIX_LOSS,
    CURRICULUM_BEST_FULL_ROLLOUT_RMSE,
    CURRICULUM_BEST_NEXT_WINDOW_RMSE,
    CURRICULUM_BEST_FUTURE_ROLLOUT_RMSE,
)
PIPELINE_STAGE = "stage"
PIPELINE_FULL = "full"
PIPELINE_THREE_STAGE = "three_stage"
PIPELINES = (PIPELINE_STAGE, PIPELINE_FULL, PIPELINE_THREE_STAGE)
IMPULSE_OPTIMIZER_ADAM = "adam"
IMPULSE_OPTIMIZER_LEAST_SQUARES = "least_squares"
IMPULSE_OPTIMIZER_LBFGS = "lbfgs"
IMPULSE_OPTIMIZERS = (IMPULSE_OPTIMIZER_ADAM, IMPULSE_OPTIMIZER_LEAST_SQUARES, IMPULSE_OPTIMIZER_LBFGS)
ANALYTIC_OPTIMIZER_ADAM = "adam"
ANALYTIC_OPTIMIZER_LEAST_SQUARES = "least_squares"
ANALYTIC_OPTIMIZER_LBFGS = "lbfgs"
ANALYTIC_OPTIMIZERS = (ANALYTIC_OPTIMIZER_ADAM, ANALYTIC_OPTIMIZER_LEAST_SQUARES, ANALYTIC_OPTIMIZER_LBFGS)
IMPULSE_VELOCITY_BOUND_M_PER_S = 20.0
IMPULSE_DELTA_V_BOUND_M_PER_S = 20.0
ANALYTIC_VELOCITY_BOUND_M_PER_S = 20.0
ANALYTIC_RAW_PARAMETER_BOUND = 20.0
SHAPE_CIRCLE = 0
SHAPE_BOX = 1
DEBUG_ARTIFACTS_ENV = "PHYSMIND_DEBUG_ARTIFACTS"


def _impulse_stage_name(mode: str) -> str:
    if mode == MODE_IMPULSE_FORCED_FRAME:
        return "impulse_forced_frame"
    if mode == MODE_IMPULSE_FORCED_WINDOW:
        return "impulse_forced_window"
    if mode == MODE_NORMAL_COLLISION:
        return "normal_collision"
    raise ValueError(f"unsupported impulse stage mode: {mode}")


def _render_video_enabled(render_video: bool | None) -> bool:
    if render_video is not None:
        return bool(render_video)
    return os.getenv(DEBUG_ARTIFACTS_ENV) == "1"


def _unit_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _support_plane_basis(manifest: dict[str, Any]) -> dict[str, Any]:
    normal_blender, point_blender = sysid_common._support_plane_in_blender(manifest)
    normal = _unit_np(np.asarray(normal_blender, dtype=np.float64).reshape(3))
    basis = sysid_common._velocity_tangent_basis(manifest)
    if basis is None:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_1 = _unit_np(np.cross(normal, reference))
        tangent_2 = _unit_np(np.cross(normal, tangent_1))
    else:
        tangent_1 = _unit_np(np.asarray(basis[0], dtype=np.float64).reshape(3))
        tangent_2 = _unit_np(np.asarray(basis[1], dtype=np.float64).reshape(3))
    return {
        "normal": normal,
        "origin": np.asarray(point_blender, dtype=np.float64).reshape(3),
        "tangent_1": tangent_1,
        "tangent_2": tangent_2,
    }


def _to_plane_coords(position: list[float], plane: dict[str, Any]) -> tuple[np.ndarray, float]:
    delta = np.asarray(position, dtype=np.float64).reshape(3) - plane["origin"]
    xy = np.asarray(
        [float(np.dot(delta, plane["tangent_1"])), float(np.dot(delta, plane["tangent_2"]))],
        dtype=np.float64,
    )
    height = float(np.dot(delta, plane["normal"]))
    return xy, height


def _from_plane_coords(xy: np.ndarray, height: float, plane: dict[str, Any]) -> np.ndarray:
    return (
        plane["origin"]
        + float(xy[0]) * plane["tangent_1"]
        + float(xy[1]) * plane["tangent_2"]
        + float(height) * plane["normal"]
    )


def _shape_radius(spec: dict[str, Any], dimensions: list[float]) -> float:
    geometry_type = str(spec.get("geometry_type") or "").lower()
    dims = [max(float(value), 0.02) for value in dimensions] or [0.2, 0.2, 0.2]
    if geometry_type == "sphere":
        return max(dims) * 0.5
    if geometry_type == "cylinder":
        return max(dims[0], dims[1]) * 0.5
    if geometry_type in {"cube", "box"}:
        return math.sqrt(dims[0] * dims[0] + dims[1] * dims[1]) * 0.5
    return max(dims) * 0.5


def _half_extents_2d(spec: dict[str, Any], dimensions: list[float]) -> list[float]:
    dims = [max(float(value), 0.02) for value in dimensions] or [0.2, 0.2, 0.2]
    geometry_type = str(spec.get("geometry_type") or "").lower()
    if geometry_type in {"cube", "box"}:
        return [0.5 * dims[0], 0.5 * dims[1]]
    radius = _shape_radius(spec, dimensions)
    return [radius, radius]


def _inertia_coefficient_2d(spec: dict[str, Any], dimensions: list[float]) -> float:
    geometry_type = str(spec.get("geometry_type") or "").lower()
    if geometry_type in {"cube", "box"}:
        half_x, half_y = _half_extents_2d(spec, dimensions)
        width = 2.0 * half_x
        height = 2.0 * half_y
        return max((width * width + height * height) / 12.0, 1e-8)
    radius = _shape_radius(spec, dimensions)
    return max(0.5 * radius * radius, 1e-8)


def _pose_yaw_angle(pose_4x4: Any, plane: dict[str, Any]) -> float:
    tangent_1 = np.asarray(plane["tangent_1"], dtype=np.float64)
    tangent_2 = np.asarray(plane["tangent_2"], dtype=np.float64)
    pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
    blender_pose = np.asarray(sysid_common._opencv_pose_to_blender_world(pose.tolist()), dtype=np.float64)
    axis = blender_pose[:3, 0]
    return math.atan2(float(np.dot(axis, tangent_2)), float(np.dot(axis, tangent_1)))


def _simulated_trajectories(
    *,
    object_ids: list[str],
    target: dict[str, list[dict[str, Any]]],
    target_plane: dict[str, list[dict[str, Any]]],
    simulated_2d: dict[str, dict[str, np.ndarray]],
    plane: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for object_id in object_ids:
        records = []
        simulated_payload = simulated_2d.get(object_id, {})
        xy_values = np.asarray(simulated_payload.get("positions", np.empty((0, 2), dtype=np.float64)), dtype=np.float64)
        frame_values = [
            int(value)
            for value in np.asarray(simulated_payload.get("frame_indices", np.empty((0,), dtype=np.int64))).tolist()
        ]
        xy_by_frame = {
            frame_index: xy_values[index]
            for index, frame_index in enumerate(frame_values)
            if index < len(xy_values)
        }
        height = float(target_plane[object_id][0]["plane_height"]) if target_plane.get(object_id) else 0.0
        first_pose = target[object_id][0]["pose_4x4"]
        first_pose_blender = sysid_common._opencv_pose_to_blender_world(first_pose)
        for target_record in target[object_id]:
            frame_index = int(target_record["frame_index"])
            if frame_index not in xy_by_frame:
                continue
            position = _from_plane_coords(np.asarray(xy_by_frame[frame_index], dtype=np.float64), height, plane)
            pose = [[float(value) for value in row] for row in first_pose_blender]
            pose[0][3] = float(position[0])
            pose[1][3] = float(position[1])
            pose[2][3] = float(position[2])
            records.append(
                {
                    "frame_index": frame_index,
                    "position": position.astype(float).tolist(),
                    "position_opencv_camera": sysid_common._blender_vector_to_opencv_camera(
                        position.astype(float).tolist()
                    ),
                    "pose_4x4": pose,
                }
            )
        output[object_id] = records
    return output


def _project_target_to_plane(fit: dict[str, Any]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    plane = fit["analytic_support_plane"]
    origin = np.asarray(plane["origin"], dtype=np.float64)
    tangent_1 = np.asarray(plane["tangent_1"], dtype=np.float64)
    tangent_2 = np.asarray(plane["tangent_2"], dtype=np.float64)
    target_plane: dict[str, list[dict[str, Any]]] = {}
    for object_id, records in fit["target_trajectories"].items():
        converted = []
        for record in records:
            position = np.asarray(record["position"], dtype=np.float64)
            delta = position - origin
            converted.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "plane_xy": [float(np.dot(delta, tangent_1)), float(np.dot(delta, tangent_2))],
                }
            )
        target_plane[str(object_id)] = converted
    return sorted(target_plane), target_plane


def _object_scales(fit: dict[str, Any], object_ids: list[str], target_plane: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    half_extents = fit.get("physics_rollout", {}).get("shape_half_extents_2d_by_object", {})
    output = {}
    for object_id in object_ids:
        value = half_extents.get(object_id)
        if isinstance(value, list) and len(value) >= 2:
            output[object_id] = 2.0 * max(float(value[0]), float(value[1]))
            continue
        xy = np.asarray([record["plane_xy"] for record in target_plane.get(object_id, [])], dtype=np.float64)
        if xy.size:
            span = np.max(xy, axis=0) - np.min(xy, axis=0)
            output[object_id] = max(float(np.max(span)), 1e-12)
        else:
            output[object_id] = 1.0
    return output


def _target_tensor(object_ids: list[str], target_plane: dict[str, list[dict[str, Any]]]) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    frames = sorted({int(record["frame_index"]) for records in target_plane.values() for record in records})
    frame_to_offset = {frame: index for index, frame in enumerate(frames)}
    target = torch.zeros((len(frames), len(object_ids), 2), dtype=torch.float64)
    mask = torch.zeros((len(frames), len(object_ids), 1), dtype=torch.float64)
    for object_index, object_id in enumerate(object_ids):
        for record in target_plane[object_id]:
            offset = frame_to_offset[int(record["frame_index"])]
            target[offset, object_index] = torch.tensor(record["plane_xy"], dtype=torch.float64)
            mask[offset, object_index, 0] = 1.0
    return frames, target, mask


def _active_metadata(
    object_ids: list[str],
    target_plane: dict[str, list[dict[str, Any]]],
    frames: list[int],
    target: torch.Tensor,
    fps: float,
) -> dict[str, Any]:
    frame_to_offset = {int(frame): index for index, frame in enumerate(frames)}
    frame0 = int(frames[0]) if frames else 0
    first_frames: list[int | None] = []
    last_frames: list[int | None] = []
    first_offsets: list[int] = []
    last_offsets: list[int] = []
    first_xy: list[torch.Tensor] = []
    active_mask = torch.zeros((len(frames), len(object_ids), 1), dtype=target.dtype, device=target.device)
    for object_index, object_id in enumerate(object_ids):
        records = sorted(target_plane.get(object_id, []), key=lambda item: int(item["frame_index"]))
        if not records:
            first_frames.append(None)
            last_frames.append(None)
            first_offsets.append(0)
            last_offsets.append(-1)
            first_xy.append(torch.zeros((2,), dtype=target.dtype, device=target.device))
            continue
        first_frame = int(records[0]["frame_index"])
        last_frame = int(records[-1]["frame_index"])
        first_offset = int(frame_to_offset[first_frame])
        last_offset = int(frame_to_offset[last_frame])
        first_frames.append(first_frame)
        last_frames.append(last_frame)
        first_offsets.append(first_offset)
        last_offsets.append(last_offset)
        first_xy.append(target[first_offset, object_index].clone())
        for frame_offset, frame in enumerate(frames):
            if first_frame <= int(frame) < last_frame:
                active_mask[frame_offset, object_index, 0] = 1.0
    start_times = torch.tensor(
        [
            0.0 if first_frame is None else (float(first_frame) - float(frame0)) / max(float(fps), 1e-12)
            for first_frame in first_frames
        ],
        dtype=target.dtype,
        device=target.device,
    )
    end_times = torch.tensor(
        [
            0.0 if last_frame is None else (float(last_frame) - float(frame0)) / max(float(fps), 1e-12)
            for last_frame in last_frames
        ],
        dtype=target.dtype,
        device=target.device,
    )
    return {
        "first_frames": first_frames,
        "last_frames": last_frames,
        "first_offsets": torch.tensor(first_offsets, dtype=torch.int64, device=target.device),
        "last_offsets": torch.tensor(last_offsets, dtype=torch.int64, device=target.device),
        "first_xy": torch.stack(first_xy) if first_xy else torch.empty((0, 2), dtype=target.dtype, device=target.device),
        "active_mask": active_mask,
        "active_start_times": start_times,
        "active_end_times": end_times,
        "frame0": frame0,
    }


def _active_interval_summary(object_ids: list[str], active_metadata: dict[str, Any]) -> dict[str, dict[str, int | None]]:
    first_frames = active_metadata["first_frames"]
    last_frames = active_metadata["last_frames"]
    return {
        object_id: {
            "first_active_frame": None if first_frames[object_index] is None else int(first_frames[object_index]),
            "last_active_frame": None if last_frames[object_index] is None else int(last_frames[object_index]),
        }
        for object_index, object_id in enumerate(object_ids)
    }


def _xy_by_object_frame(
    object_ids: list[str],
    target_plane: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[int, np.ndarray]]:
    xy_by_object: dict[str, dict[int, np.ndarray]] = {object_id: {} for object_id in object_ids}
    for object_id in object_ids:
        for record in target_plane.get(object_id, []):
            try:
                frame_index = int(record["frame_index"])
                xy = np.asarray(record["plane_xy"], dtype=np.float64).reshape(2)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(xy).all():
                xy_by_object[object_id][frame_index] = xy
    return xy_by_object


def _yaw_by_object_frame(fit: dict[str, Any], object_ids: list[str]) -> dict[str, dict[int, float]]:
    plane = fit.get("analytic_support_plane", {})
    trajectories = fit.get("target_trajectories", {})
    yaw_by_object: dict[str, dict[int, float]] = {object_id: {} for object_id in object_ids}
    if not plane or not isinstance(trajectories, dict):
        return yaw_by_object
    for object_id in object_ids:
        records = trajectories.get(object_id) or []
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("frame_index") is None:
                continue
            pose = record.get("corrected_pose_4x4") or record.get("pose_4x4")
            if pose is None:
                continue
            try:
                yaw = float(_pose_yaw_angle(pose, plane))
            except (TypeError, ValueError):
                continue
            if math.isfinite(yaw):
                yaw_by_object[object_id][int(record["frame_index"])] = yaw
    return yaw_by_object


def _box_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    cos_value = math.cos(float(yaw))
    sin_value = math.sin(float(yaw))
    return np.asarray([cos_value, sin_value], dtype=np.float64), np.asarray([-sin_value, cos_value], dtype=np.float64)


def _point_local_to_box(
    *,
    point_xy: np.ndarray,
    box_center_xy: np.ndarray,
    box_yaw: float,
) -> np.ndarray:
    axis_x, axis_y = _box_axes(box_yaw)
    delta = point_xy - box_center_xy
    return np.asarray([float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y))], dtype=np.float64)


def _point_inside_shape(
    *,
    point_xy: np.ndarray,
    center_xy: np.ndarray,
    shape_id: int,
    half_extents_xy: np.ndarray,
    radius: float,
    yaw: float | None,
) -> bool:
    if int(shape_id) == SHAPE_CIRCLE:
        return bool(float(np.linalg.norm(point_xy - center_xy)) <= float(radius) + 1e-12)
    if int(shape_id) == SHAPE_BOX:
        if yaw is None:
            return False
        local = _point_local_to_box(point_xy=point_xy, box_center_xy=center_xy, box_yaw=float(yaw))
        return bool(np.all(np.abs(local) <= half_extents_xy + 1e-12))
    return False


def _point_box_signed_distance(
    *,
    point_xy: np.ndarray,
    box_center_xy: np.ndarray,
    box_half_extents_xy: np.ndarray,
    box_yaw: float,
) -> float:
    local = _point_local_to_box(point_xy=point_xy, box_center_xy=box_center_xy, box_yaw=box_yaw)
    q = np.abs(local) - box_half_extents_xy
    outside = np.maximum(q, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    inside_distance = min(float(max(q[0], q[1])), 0.0)
    return outside_distance + inside_distance


def _box_box_signed_residual(
    *,
    center_a_xy: np.ndarray,
    half_extents_a_xy: np.ndarray,
    yaw_a: float,
    center_b_xy: np.ndarray,
    half_extents_b_xy: np.ndarray,
    yaw_b: float,
) -> float:
    axes = [*_box_axes(yaw_a), *_box_axes(yaw_b)]
    center_delta = center_b_xy - center_a_xy
    separations: list[float] = []
    overlaps: list[float] = []
    axes_a = _box_axes(yaw_a)
    axes_b = _box_axes(yaw_b)
    for axis in axes:
        projection_a = sum(float(half_extents_a_xy[index]) * abs(float(np.dot(axes_a[index], axis))) for index in range(2))
        projection_b = sum(float(half_extents_b_xy[index]) * abs(float(np.dot(axes_b[index], axis))) for index in range(2))
        distance = abs(float(np.dot(center_delta, axis)))
        separation = distance - projection_a - projection_b
        separations.append(separation)
        overlaps.append(-separation)
    max_separation = max(separations)
    if max_separation > 0.0:
        return float(max_separation)
    return float(-min(overlaps))


def _shape_contact_residual(
    *,
    center_a_xy: np.ndarray,
    shape_a: int,
    half_extents_a_xy: np.ndarray,
    radius_a: float,
    yaw_a: float | None,
    center_b_xy: np.ndarray,
    shape_b: int,
    half_extents_b_xy: np.ndarray,
    radius_b: float,
    yaw_b: float | None,
) -> tuple[float, float, str] | None:
    center_distance = float(np.linalg.norm(center_a_xy - center_b_xy))
    if not math.isfinite(center_distance):
        return None
    if shape_a == SHAPE_CIRCLE and shape_b == SHAPE_CIRCLE:
        return center_distance - float(radius_a) - float(radius_b), center_distance, "circle_circle"
    if shape_a == SHAPE_BOX and shape_b == SHAPE_CIRCLE:
        if yaw_a is None:
            return None
        residual = _point_box_signed_distance(
            point_xy=center_b_xy,
            box_center_xy=center_a_xy,
            box_half_extents_xy=half_extents_a_xy,
            box_yaw=float(yaw_a),
        ) - float(radius_b)
        return float(residual), center_distance, "box_circle"
    if shape_a == SHAPE_CIRCLE and shape_b == SHAPE_BOX:
        if yaw_b is None:
            return None
        residual = _point_box_signed_distance(
            point_xy=center_a_xy,
            box_center_xy=center_b_xy,
            box_half_extents_xy=half_extents_b_xy,
            box_yaw=float(yaw_b),
        ) - float(radius_a)
        return float(residual), center_distance, "box_circle"
    if shape_a == SHAPE_BOX and shape_b == SHAPE_BOX:
        if yaw_a is None or yaw_b is None:
            return None
        residual = _box_box_signed_residual(
            center_a_xy=center_a_xy,
            half_extents_a_xy=half_extents_a_xy,
            yaw_a=float(yaw_a),
            center_b_xy=center_b_xy,
            half_extents_b_xy=half_extents_b_xy,
            yaw_b=float(yaw_b),
        )
        return float(residual), center_distance, "box_box"
    return None


def _center_line_inside_fraction(
    *,
    center_a_xy: np.ndarray,
    shape_a: int,
    half_extents_a_xy: np.ndarray,
    radius_a: float,
    yaw_a: float | None,
    center_b_xy: np.ndarray,
    shape_b: int,
    half_extents_b_xy: np.ndarray,
    radius_b: float,
    yaw_b: float | None,
) -> float | None:
    if not np.isfinite(center_a_xy).all() or not np.isfinite(center_b_xy).all():
        return None
    sample_points = np.asarray(
        [
            (1.0 - alpha) * center_a_xy + alpha * center_b_xy
            for alpha in np.linspace(0.0, 1.0, SHAPE_COLLISION_LINE_SAMPLE_COUNT)
        ],
        dtype=np.float64,
    )
    inside_count = 0
    for point in sample_points:
        inside_a = _point_inside_shape(
            point_xy=point,
            center_xy=center_a_xy,
            shape_id=shape_a,
            half_extents_xy=half_extents_a_xy,
            radius=radius_a,
            yaw=yaw_a,
        )
        inside_b = _point_inside_shape(
            point_xy=point,
            center_xy=center_b_xy,
            shape_id=shape_b,
            half_extents_xy=half_extents_b_xy,
            radius=radius_b,
            yaw=yaw_b,
        )
        if inside_a or inside_b:
            inside_count += 1
    return float(inside_count) / float(SHAPE_COLLISION_LINE_SAMPLE_COUNT)


def _cluster_contact_frame_records(
    frame_records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for record in sorted(frame_records, key=lambda item: int(item["frame_index"])):
        frame_index = int(record["frame_index"])
        if not clusters:
            clusters.append([record])
            continue
        previous_frame = int(clusters[-1][-1]["frame_index"])
        missing_count = int(frame_index) - previous_frame - 1
        if missing_count <= SHAPE_COLLISION_CLUSTER_GAP_MISSING_FRAMES:
            clusters[-1].append(record)
        else:
            clusters.append([record])
    return clusters


def _shape_pose_collision_events(
    *,
    object_ids: list[str],
    target_plane: dict[str, list[dict[str, Any]]],
    yaw_by_object: dict[str, dict[int, float]],
    shape_ids_by_object: dict[str, int],
    half_extents_by_object: dict[str, np.ndarray],
    contact_radii_by_object: dict[str, float],
    shape_name_by_object: dict[str, str],
) -> list[dict[str, Any]]:
    xy_by_object = _xy_by_object_frame(object_ids, target_plane)
    events = []
    for left_index, left_id in enumerate(object_ids):
        for right_id in object_ids[left_index + 1 :]:
            if (
                left_id not in contact_radii_by_object
                or right_id not in contact_radii_by_object
                or left_id not in shape_ids_by_object
                or right_id not in shape_ids_by_object
                or left_id not in half_extents_by_object
                or right_id not in half_extents_by_object
            ):
                continue
            common_frames = sorted(set(xy_by_object[left_id]).intersection(xy_by_object[right_id]))
            if len(common_frames) < 3:
                continue
            frame_records: list[dict[str, Any]] = []
            records_by_frame: dict[int, dict[str, Any]] = {}
            for frame_index in common_frames:
                yaw_left = yaw_by_object.get(left_id, {}).get(int(frame_index))
                yaw_right = yaw_by_object.get(right_id, {}).get(int(frame_index))
                line_inside_fraction = _center_line_inside_fraction(
                    center_a_xy=xy_by_object[left_id][frame_index],
                    shape_a=int(shape_ids_by_object[left_id]),
                    half_extents_a_xy=half_extents_by_object[left_id],
                    radius_a=float(contact_radii_by_object[left_id]),
                    yaw_a=yaw_left,
                    center_b_xy=xy_by_object[right_id][frame_index],
                    shape_b=int(shape_ids_by_object[right_id]),
                    half_extents_b_xy=half_extents_by_object[right_id],
                    radius_b=float(contact_radii_by_object[right_id]),
                    yaw_b=yaw_right,
                )
                if line_inside_fraction is None:
                    continue
                residual_result = _shape_contact_residual(
                    center_a_xy=xy_by_object[left_id][frame_index],
                    shape_a=int(shape_ids_by_object[left_id]),
                    half_extents_a_xy=half_extents_by_object[left_id],
                    radius_a=float(contact_radii_by_object[left_id]),
                    yaw_a=yaw_left,
                    center_b_xy=xy_by_object[right_id][frame_index],
                    shape_b=int(shape_ids_by_object[right_id]),
                    half_extents_b_xy=half_extents_by_object[right_id],
                    radius_b=float(contact_radii_by_object[right_id]),
                    yaw_b=yaw_right,
                )
                if residual_result is None:
                    continue
                residual, center_distance, contact_model = residual_result
                if not math.isfinite(residual):
                    continue
                record = {
                    "frame_index": int(frame_index),
                    "center_distance": float(center_distance),
                    "line_inside_fraction": float(line_inside_fraction),
                    "contact_residual_m": float(residual),
                    "abs_contact_residual_m": abs(float(residual)),
                    "contact_model": str(contact_model),
                    "yaw_rad_by_object": {
                        left_id: None if yaw_left is None else float(yaw_left),
                        right_id: None if yaw_right is None else float(yaw_right),
                    },
                }
                records_by_frame[int(frame_index)] = record
                if line_inside_fraction >= SHAPE_COLLISION_LINE_INSIDE_FRACTION_THRESHOLD:
                    frame_records.append(record)
            for cluster in _cluster_contact_frame_records(frame_records):
                min_distance_record = min(cluster, key=lambda item: float(item["center_distance"]))
                first_frame = int(cluster[0]["frame_index"])
                last_frame = int(cluster[-1]["frame_index"])
                contact_frame = int(min_distance_record["frame_index"])
                common_index = common_frames.index(contact_frame)
                if common_index <= 0 or common_index >= len(common_frames) - 1:
                    continue
                previous_frame = int(common_frames[common_index - 1])
                next_frame = int(common_frames[common_index + 1])
                if previous_frame not in records_by_frame or next_frame not in records_by_frame:
                    continue
                previous_distance = float(records_by_frame[previous_frame]["center_distance"])
                next_distance = float(records_by_frame[next_frame]["center_distance"])
                contact_distance = float(min_distance_record["center_distance"])
                if previous_distance < contact_distance - 1e-9 or next_distance < contact_distance - 1e-9:
                    continue
                event = {
                    "event_index": len(events),
                    "event_type": "pair_collision_candidate",
                    "frame": contact_frame,
                    "contact_frame": contact_frame,
                    "object_ids": [left_id, right_id],
                    "start_frame": first_frame,
                    "end_frame": last_frame,
                    "peak_frame": contact_frame,
                    "contact_model": str(min_distance_record["contact_model"]),
                    "line_inside_fraction_threshold": float(SHAPE_COLLISION_LINE_INSIDE_FRACTION_THRESHOLD),
                    "line_inside_fraction": float(min_distance_record["line_inside_fraction"]),
                    "contact_residual_m": float(min_distance_record["contact_residual_m"]),
                    "abs_contact_residual_m": float(min_distance_record["abs_contact_residual_m"]),
                    "contact_center_distance": contact_distance,
                    "event_score": float(min_distance_record["line_inside_fraction"]),
                    "verified": True,
                    "verification_source": "2d_shape_pose_center_line_inside_fraction_cluster_argmin",
                    "verification_status": "ok",
                    "source": "swr_2d_shape_pose_center_line_inside_fraction_argmin",
                    "shape_by_object": {
                        left_id: shape_name_by_object.get(left_id),
                        right_id: shape_name_by_object.get(right_id),
                    },
                    "line_inside_fraction_by_frame": [
                        {
                            "frame_index": int(record["frame_index"]),
                            "line_inside_fraction": float(record["line_inside_fraction"]),
                            "center_distance": float(record["center_distance"]),
                            "contact_residual_m": float(record["contact_residual_m"]),
                            "contact_model": str(record["contact_model"]),
                            "yaw_rad_by_object": record["yaw_rad_by_object"],
                        }
                        for record in cluster
                    ],
                    "contact_residual_by_frame": [
                        {
                            "frame_index": int(record["frame_index"]),
                            "contact_residual_m": float(record["contact_residual_m"]),
                            "abs_contact_residual_m": float(record["abs_contact_residual_m"]),
                            "line_inside_fraction": float(record["line_inside_fraction"]),
                            "center_distance": float(record["center_distance"]),
                            "contact_model": str(record["contact_model"]),
                            "yaw_rad_by_object": record["yaw_rad_by_object"],
                        }
                        for record in cluster
                    ],
                    "center_distance_by_frame": [
                        {
                            "frame_index": int(frame_index),
                            "center_distance": float(records_by_frame[int(frame_index)]["center_distance"]),
                        }
                        for frame_index in common_frames
                        if first_frame <= int(frame_index) <= last_frame and int(frame_index) in records_by_frame
                    ],
                }
                events.append(event)
    events.sort(key=lambda item: (int(item["frame"]), tuple(item["object_ids"])))
    for event_index, event in enumerate(events):
        event["event_index"] = int(event_index)
    return events


def _detected_collision_events(
    *,
    fit: dict[str, Any],
    object_ids: list[str],
    target_plane: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    object_scales_by_id = _object_scales(fit, object_ids, target_plane)
    shape_ids, half_extents, _angles, contact_radii, shape_names, _contact_proxy_names = _shape_metadata(
        fit=fit,
        object_ids=object_ids,
        object_scales_by_id=object_scales_by_id,
    )
    shape_ids_by_object = {
        object_id: int(shape_ids[object_index])
        for object_index, object_id in enumerate(object_ids)
    }
    half_extents_by_object = {
        object_id: np.asarray(half_extents.detach().cpu()[object_index].tolist(), dtype=np.float64)
        for object_index, object_id in enumerate(object_ids)
    }
    contact_radii_by_object = {
        object_id: float(contact_radii.detach().cpu()[object_index])
        for object_index, object_id in enumerate(object_ids)
    }
    shape_name_by_object = {
        object_id: str(shape_names[object_index])
        for object_index, object_id in enumerate(object_ids)
    }
    return _shape_pose_collision_events(
        object_ids=object_ids,
        target_plane=target_plane,
        yaw_by_object=_yaw_by_object_frame(fit, object_ids),
        shape_ids_by_object=shape_ids_by_object,
        half_extents_by_object=half_extents_by_object,
        contact_radii_by_object=contact_radii_by_object,
        shape_name_by_object=shape_name_by_object,
    )


def _event_normals(event_specs: list[dict[str, Any]], frames: list[int], target: torch.Tensor) -> torch.Tensor:
    frame_to_offset = {int(frame): index for index, frame in enumerate(frames)}
    normals = []
    for event in event_specs:
        object_indices = [int(index) for index in event["object_indices"]]
        if len(object_indices) != 2:
            normals.append([1.0, 0.0])
            continue
        offset = frame_to_offset.get(int(event["frame"]), 0)
        rel = target[offset, object_indices[0]] - target[offset, object_indices[1]]
        norm = torch.sqrt(torch.sum(rel * rel) + 1e-12)
        normals.append((rel / norm).detach().cpu().tolist())
    return torch.tensor(normals, dtype=torch.float64, device=target.device)


def _object_pairs(object_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(object_count) for j in range(i + 1, object_count)]


def _impulse_collision_update(
    *,
    v: torch.Tensor,
    i: int,
    j: int,
    normal: torch.Tensor,
    mass: torch.Tensor,
    restitution: torch.Tensor,
    require_approaching: bool,
) -> torch.Tensor:
    updated, _impulse, _relative_normal_velocity = analytic_swr_common.sphere_sphere_impulse_update_torch(
        velocities=v,
        i=i,
        j=j,
        normal_from_j_to_i=normal,
        mass_i=mass[i],
        mass_j=mass[j],
        restitution=restitution,
        require_approaching=require_approaching,
    )
    return updated


def _free_contact_sliding_step(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    friction: torch.Tensor,
    dt: torch.Tensor,
    gravity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    speed = torch.sqrt(torch.sum(v * v, dim=1, keepdim=True) + 1e-12)
    candidate = v - friction.reshape(-1, 1) * gravity * dt * (v / speed)
    zero = torch.zeros((), dtype=torch.float64, device=x.device)
    v_next = torch.where(
        v > 0.0,
        torch.maximum(candidate, zero),
        torch.where(v < 0.0, torch.minimum(candidate, zero), zero),
    )
    return x + 0.5 * dt * (v + v_next), v_next


def _first_circle_circle_toi(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    object_pairs: list[tuple[int, int]],
    object_scales: torch.Tensor,
    dt: torch.Tensor,
    active_pair_indices: set[int] | None = None,
) -> tuple[int, int, int, torch.Tensor] | None:
    zero = torch.zeros((), dtype=torch.float64, device=x.device)
    eps = torch.tensor(1e-12, dtype=torch.float64, device=x.device)
    best: tuple[int, int, int, torch.Tensor] | None = None
    best_tau = float("inf")
    for pair_index, (i, j) in enumerate(object_pairs):
        if active_pair_indices is not None and int(pair_index) not in active_pair_indices:
            continue
        rel_x = x[i] - x[j]
        rel_v = v[i] - v[j]
        contact_distance = 0.5 * (object_scales[i] + object_scales[j])
        distance = torch.sqrt(torch.sum(rel_x * rel_x) + eps)
        projected_now = torch.sum(rel_v * (rel_x / distance))
        tau: torch.Tensor | None = None
        if bool(((distance <= contact_distance) & (projected_now < 0.0)).detach().cpu()):
            tau = zero
        else:
            a = torch.sum(rel_v * rel_v)
            if bool((a <= eps).detach().cpu()):
                continue
            b = 2.0 * torch.sum(rel_x * rel_v)
            c = torch.sum(rel_x * rel_x) - contact_distance * contact_distance
            discriminant = b * b - 4.0 * a * c
            if bool((discriminant < 0.0).detach().cpu()):
                continue
            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
            denominator = 2.0 * a
            for root in ((-b - sqrt_discriminant) / denominator, (-b + sqrt_discriminant) / denominator):
                if not bool(((root >= -1e-10) & (root <= dt + 1e-10)).detach().cpu()):
                    continue
                root_clamped = torch.clamp(root, min=zero, max=dt)
                contact_rel = rel_x + root_clamped * rel_v
                contact_normal = contact_rel / torch.sqrt(torch.sum(contact_rel * contact_rel) + eps)
                projected_at_contact = torch.sum(rel_v * contact_normal)
                if not bool((projected_at_contact < 0.0).detach().cpu()):
                    continue
                if tau is None or float(root_clamped.detach().cpu()) < float(tau.detach().cpu()):
                    tau = root_clamped
        if tau is None:
            continue
        tau_value = float(tau.detach().cpu())
        if tau_value < best_tau:
            best_tau = tau_value
            best = (int(pair_index), int(i), int(j), tau)
    return best


def _rollout_event_assumption(
    *,
    target: torch.Tensor,
    frames: list[int],
    event_specs: list[dict[str, Any]],
    pair_index_by_key: dict[tuple[int, int], int],
    v0: torch.Tensor,
    friction: torch.Tensor,
    impulses: torch.Tensor,
    pair_delta_v: torch.Tensor | None = None,
    pair_log_alpha: torch.Tensor | None = None,
    mass: torch.Tensor | None = None,
    pair_restitution: torch.Tensor | None = None,
    event_normals: torch.Tensor | None = None,
    mode: str = MODE_IMPULSE_FORCED_FRAME,
    normal_source: str = "target",
    fps: float,
    active_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.tensor(1.0 / max(float(fps), 1e-12), dtype=torch.float64, device=target.device)
    gravity = torch.tensor(float(GRAVITY), dtype=torch.float64, device=target.device)
    if active_metadata is None:
        x = target[0].clone()
        v = v0
        active_mask = torch.ones((len(frames), int(target.shape[1]), 1), dtype=target.dtype, device=target.device)
        first_offsets = torch.zeros((int(target.shape[1]),), dtype=torch.int64, device=target.device)
        first_xy = target[0].clone()
    else:
        x = active_metadata["first_xy"].clone()
        v = torch.zeros_like(v0)
        active_mask = active_metadata["active_mask"]
        first_offsets = active_metadata["first_offsets"]
        first_xy = active_metadata["first_xy"]
    positions = []
    velocities = []
    events_by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for event_index, event in enumerate(event_specs):
        events_by_frame.setdefault(int(event["frame"]), []).append((event_index, event))
    for frame_offset, frame in enumerate(frames):
        first_now = first_offsets == int(frame_offset)
        if bool(torch.any(first_now).detach().cpu()):
            x = x.clone()
            v = v.clone()
            x[first_now] = first_xy[first_now]
            v[first_now] = v0[first_now]
        frame_active = active_mask[frame_offset]
        for event_index, event in events_by_frame.get(int(frame), []):
            impulse = impulses[event_index]
            object_indices = [int(index) for index in event["object_indices"]]
            if any(float(frame_active[index, 0].detach().cpu()) <= 0.0 for index in object_indices):
                continue
            if len(object_indices) == 1:
                v = v.clone()
                v[object_indices[0]] = v[object_indices[0]] + impulse
            elif len(object_indices) == 2:
                v = v.clone()
                if mode == MODE_NORMAL_COLLISION:
                    if mass is None or pair_restitution is None or event_normals is None:
                        raise ValueError("normal-collision mode requires mass, pair_restitution, and event_normals")
                    i, j = object_indices
                    if normal_source == "rollout":
                        rel = x[i] - x[j]
                        normal = rel / torch.sqrt(torch.sum(rel * rel) + 1e-12)
                    else:
                        normal = event_normals[event_index]
                    pair_index = pair_index_by_key[tuple(sorted((i, j)))]
                    v = _impulse_collision_update(
                        v=v,
                        i=i,
                        j=j,
                        normal=normal,
                        mass=mass,
                        restitution=pair_restitution[pair_index],
                        require_approaching=False,
                    )
                else:
                    if pair_delta_v is None or pair_log_alpha is None:
                        raise ValueError("impulse-forced-frame pair events require pair_delta_v and pair_log_alpha")
                    i, j = object_indices
                    delta_v = pair_delta_v[event_index]
                    alpha = torch.exp(torch.clamp(pair_log_alpha[event_index], -5.0, 5.0))
                    v[i] = v[i] + delta_v
                    v[j] = v[j] - alpha * delta_v
        positions.append(x)
        velocities.append(v)
        if frame_offset + 1 >= len(frames):
            break
        speed = torch.sqrt(torch.sum(v * v, dim=1, keepdim=True) + 1e-12)
        candidate = v - friction.reshape(-1, 1) * gravity * dt * (v / speed)
        zero = torch.zeros((), dtype=torch.float64, device=target.device)
        v_next = torch.where(
            v > 0.0,
            torch.maximum(candidate, zero),
            torch.where(v < 0.0, torch.minimum(candidate, zero), zero),
        )
        x_next = x + 0.5 * dt * (v + v_next)
        x = frame_active * x_next + (1.0 - frame_active) * x
        v = frame_active * v_next + (1.0 - frame_active) * v
    return torch.stack(positions), torch.stack(velocities)


def _rollout_impulse_analytic(
    *,
    target: torch.Tensor,
    frames: list[int],
    event_specs: list[dict[str, Any]],
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
    v0: torch.Tensor,
    friction: torch.Tensor,
    impulses: torch.Tensor,
    pair_delta_v: torch.Tensor,
    pair_log_alpha: torch.Tensor,
    fps: float,
    event_window_frames: int,
    newton_iters: int,
    active_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    frame0 = int(frames[0])
    frame_times = torch.tensor(
        [(float(frame) - float(frame0)) / max(float(fps), 1e-12) for frame in frames],
        dtype=torch.float64,
        device=target.device,
    )
    object_count = int(target.shape[1])
    if active_metadata is None:
        start_time = torch.full((object_count,), float(frame_times[0].detach().cpu()), dtype=target.dtype, device=target.device)
        active_end_times = torch.full((object_count,), float(frame_times[-1].detach().cpu()), dtype=target.dtype, device=target.device)
        start_x = target[0].clone()
    else:
        start_time = active_metadata["active_start_times"].clone()
        active_end_times = active_metadata["active_end_times"].clone()
        start_x = active_metadata["first_xy"].clone()
    start_v = v0
    predicted, predicted_velocity = _closed_form_motion(
        start_x,
        start_v,
        friction,
        frame_times,
        start_time,
    )
    event_records: list[dict[str, Any]] = []
    window_seconds = target.new_tensor(float(event_window_frames) / max(float(fps), 1e-12))
    sorted_events = sorted(enumerate(event_specs), key=lambda item: (int(item[1]["frame"]), int(item[0])))
    for event_index, event in sorted_events:
        object_indices = [int(index) for index in event["object_indices"]]
        if len(object_indices) == 1:
            i = object_indices[0]
            event_time = target.new_tensor((float(event["frame"]) - float(frame0)) / max(float(fps), 1e-12))
            lower_bound = start_time[i]
            upper_bound = active_end_times[i]
            if not bool((upper_bound > lower_bound + target.new_tensor(1e-6)).detach().cpu()):
                continue
            event_time = torch.clamp(
                event_time,
                min=float(lower_bound.detach().cpu()),
                max=float(upper_bound.detach().cpu()),
            )
            event_positions, pre_event_velocities = _closed_form_motion(start_x, start_v, friction, event_time, start_time)
            post_event_velocities = pre_event_velocities.clone()
            post_event_velocities[i] = post_event_velocities[i] + impulses[event_index]
            next_start_time = start_time.clone()
            next_start_x = start_x.clone()
            next_start_v = start_v.clone()
            next_start_time[i] = event_time
            next_start_x[i] = event_positions[i]
            next_start_v[i] = post_event_velocities[i]
            post_positions, post_velocities = _closed_form_motion(
                next_start_x,
                next_start_v,
                friction,
                frame_times,
                next_start_time,
            )
            object_mask = torch.zeros((object_count,), dtype=torch.bool, device=target.device)
            object_mask[i] = True
            after_mask = (frame_times >= event_time).reshape(-1, 1, 1) & object_mask.reshape(1, -1, 1)
            predicted = torch.where(after_mask, post_positions, predicted)
            predicted_velocity = torch.where(after_mask, post_velocities, predicted_velocity)
            event_records.append(
                {
                    "event_index": torch.tensor(int(event_index), dtype=torch.int64, device=target.device),
                    "event_type": str(event.get("event_type", "unary_impulse")),
                    "event_time_s": event_time,
                    "event_frame_float": event_time * target.new_tensor(float(fps)) + target.new_tensor(float(frame0)),
                    "object_indices": [int(i)],
                    "contact_residual_m": target.new_tensor(0.0),
                    "normal": target.new_tensor([1.0, 0.0]),
                    "pre_velocity": pre_event_velocities,
                    "post_velocity": post_event_velocities,
                    "relative_velocity": target.new_zeros((2,)),
                    "relative_normal_velocity": target.new_tensor(0.0),
                }
            )
            start_time = next_start_time
            start_x = next_start_x
            start_v = next_start_v
            continue
        if len(object_indices) != 2 or str(event.get("event_type")) != "pair_collision_candidate":
            raise ValueError("analytic impulse rollout supports pair_collision_candidate events only")
        i, j = object_indices
        lower_bound = torch.maximum(start_time[i], start_time[j])
        upper_bound = torch.minimum(active_end_times[i], active_end_times[j])
        if not bool((upper_bound > lower_bound + target.new_tensor(1e-6)).detach().cpu()):
            continue
        event_center = target.new_tensor((float(event["frame"]) - float(frame0)) / max(float(fps), 1e-12))
        if int(event_window_frames) <= 0:
            event_time = torch.clamp(
                event_center,
                min=float(lower_bound.detach().cpu()),
                max=float(upper_bound.detach().cpu()),
            )
        else:
            event_time = _solve_event_time(
                start_x=start_x,
                start_v=start_v,
                start_time=start_time,
                friction=friction,
                event_time_center=event_center,
                window_seconds=window_seconds,
                i=i,
                j=j,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                newton_iters=newton_iters,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        contact_residual, normal, event_positions, pre_event_velocities, relative_velocity, _contact_model = _pair_contact_value(
            start_x=start_x,
            start_v=start_v,
            start_time=start_time,
            friction=friction,
            t=event_time,
            i=i,
            j=j,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
        )
        post_event_velocities = pre_event_velocities.clone()
        delta_v = pair_delta_v[event_index]
        alpha = torch.exp(torch.clamp(pair_log_alpha[event_index], -5.0, 5.0))
        post_event_velocities[i] = post_event_velocities[i] + delta_v
        post_event_velocities[j] = post_event_velocities[j] - alpha * delta_v
        next_start_time = start_time.clone()
        next_start_x = start_x.clone()
        next_start_v = start_v.clone()
        next_start_time[i] = event_time
        next_start_time[j] = event_time
        next_start_x[i] = event_positions[i]
        next_start_x[j] = event_positions[j]
        next_start_v[i] = post_event_velocities[i]
        next_start_v[j] = post_event_velocities[j]
        post_positions, post_velocities = _closed_form_motion(
            next_start_x,
            next_start_v,
            friction,
            frame_times,
            next_start_time,
        )
        object_mask = torch.zeros((object_count,), dtype=torch.bool, device=target.device)
        object_mask[i] = True
        object_mask[j] = True
        after_mask = (frame_times >= event_time).reshape(-1, 1, 1) & object_mask.reshape(1, -1, 1)
        predicted = torch.where(after_mask, post_positions, predicted)
        predicted_velocity = torch.where(after_mask, post_velocities, predicted_velocity)
        event_records.append(
            {
                "event_index": torch.tensor(int(event_index), dtype=torch.int64, device=target.device),
                "event_type": "pair_collision_candidate",
                "event_time_s": event_time,
                "event_frame_float": event_time * target.new_tensor(float(fps)) + target.new_tensor(float(frame0)),
                "object_indices": [int(i), int(j)],
                "contact_residual_m": contact_residual,
                "normal": normal,
                "pre_velocity": pre_event_velocities,
                "post_velocity": post_event_velocities,
                "relative_velocity": relative_velocity,
                "relative_normal_velocity": torch.sum(normal * relative_velocity),
            }
        )
        start_time = next_start_time
        start_x = next_start_x
        start_v = next_start_v
    return predicted, predicted_velocity, event_records


def _rollout_free_contact(
    *,
    target: torch.Tensor,
    frames: list[int],
    object_pairs: list[tuple[int, int]],
    object_scales: torch.Tensor,
    v0: torch.Tensor,
    friction: torch.Tensor,
    mass: torch.Tensor,
    pair_restitution: torch.Tensor,
    fps: float,
    use_toi: bool = False,
    contacted_pair_indices: set[int] | None = None,
    active_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.tensor(1.0 / max(float(fps), 1e-12), dtype=torch.float64, device=target.device)
    gravity = torch.tensor(float(GRAVITY), dtype=torch.float64, device=target.device)
    if active_metadata is None:
        x = target[0].clone()
        v = v0
        active_mask = torch.ones((len(frames), int(target.shape[1]), 1), dtype=target.dtype, device=target.device)
        first_offsets = torch.zeros((int(target.shape[1]),), dtype=torch.int64, device=target.device)
        first_xy = target[0].clone()
    else:
        x = active_metadata["first_xy"].clone()
        v = torch.zeros_like(v0)
        active_mask = active_metadata["active_mask"]
        first_offsets = active_metadata["first_offsets"]
        first_xy = active_metadata["first_xy"]
    positions = []
    velocities = []
    for frame_offset, _frame in enumerate(frames):
        first_now = first_offsets == int(frame_offset)
        if bool(torch.any(first_now).detach().cpu()):
            x = x.clone()
            v = v.clone()
            x[first_now] = first_xy[first_now]
            v[first_now] = v0[first_now]
        frame_active = active_mask[frame_offset]
        active_pair_indices = {
            int(pair_index)
            for pair_index, (i, j) in enumerate(object_pairs)
            if float(frame_active[i, 0].detach().cpu()) > 0.0 and float(frame_active[j, 0].detach().cpu()) > 0.0
        }
        if use_toi:
            positions.append(x)
            velocities.append(v)
            if frame_offset + 1 >= len(frames):
                break
            event = _first_circle_circle_toi(
                x=x,
                v=v,
                object_pairs=object_pairs,
                object_scales=object_scales,
                dt=dt,
                active_pair_indices=active_pair_indices,
            )
            if event is None:
                x_next, v_next = _free_contact_sliding_step(x=x, v=v, friction=friction, dt=dt, gravity=gravity)
                x = frame_active * x_next + (1.0 - frame_active) * x
                v = frame_active * v_next + (1.0 - frame_active) * v
                continue
            pair_index, i, j, tau = event
            if contacted_pair_indices is not None:
                contacted_pair_indices.add(int(pair_index))
            x_next, v_next = _free_contact_sliding_step(x=x, v=v, friction=friction, dt=tau, gravity=gravity)
            x = frame_active * x_next + (1.0 - frame_active) * x
            v = frame_active * v_next + (1.0 - frame_active) * v
            rel = x[i] - x[j]
            normal = rel / torch.sqrt(torch.sum(rel * rel) + 1e-12)
            v = _impulse_collision_update(
                v=v,
                i=i,
                j=j,
                normal=normal,
                mass=mass,
                restitution=pair_restitution[pair_index],
                require_approaching=True,
            )
            remaining_dt = torch.clamp(dt - tau, min=0.0)
            x_next, v_next = _free_contact_sliding_step(x=x, v=v, friction=friction, dt=remaining_dt, gravity=gravity)
            x = frame_active * x_next + (1.0 - frame_active) * x
            v = frame_active * v_next + (1.0 - frame_active) * v
            continue
        for pair_index, (i, j) in enumerate(object_pairs):
            if int(pair_index) not in active_pair_indices:
                continue
            rel = x[i] - x[j]
            distance = torch.sqrt(torch.sum(rel * rel) + 1e-12)
            contact_distance = 0.5 * (object_scales[i] + object_scales[j])
            if bool((distance <= contact_distance).detach().cpu()):
                if contacted_pair_indices is not None:
                    contacted_pair_indices.add(int(pair_index))
                normal = rel / distance
                v = _impulse_collision_update(
                    v=v,
                    i=i,
                    j=j,
                    normal=normal,
                    mass=mass,
                    restitution=pair_restitution[pair_index],
                    require_approaching=True,
                )
        positions.append(x)
        velocities.append(v)
        if frame_offset + 1 >= len(frames):
            break
        x_next, v_next = _free_contact_sliding_step(x=x, v=v, friction=friction, dt=dt, gravity=gravity)
        x = frame_active * x_next + (1.0 - frame_active) * x
        v = frame_active * v_next + (1.0 - frame_active) * v
    return torch.stack(positions), torch.stack(velocities)


def _plot(
    output_png: Path,
    object_ids: list[str],
    target: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def display_xy(values: np.ndarray) -> np.ndarray:
        return np.stack([-values[:, 1], values[:, 0]], axis=1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(object_ids), 1)))
    for object_index, object_id in enumerate(object_ids):
        color = colors[object_index]
        valid = np.ones((target.shape[0],), dtype=bool)
        if mask is not None:
            valid = np.asarray(mask[:, object_index, 0] > 0.0, dtype=bool)
        if not np.any(valid):
            continue
        target_xy = target[valid, object_index]
        predicted_xy = predicted[valid, object_index]
        target_display = display_xy(target_xy)
        predicted_display = display_xy(predicted_xy)
        ax.plot(target_display[:, 0], target_display[:, 1], "-", color=color, label=f"{object_id} true")
        ax.plot(predicted_display[:, 0], predicted_display[:, 1], "--", color=color, label=f"{object_id} recovered")
        ax.scatter(target_display[0, 0], target_display[0, 1], marker="o", color=color, s=18)
        ax.scatter(target_display[-1, 0], target_display[-1, 1], marker="s", color=color, s=18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("display x = -plane y (m)")
    ax.set_ylabel("display y = plane x (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def _display_xy(values: np.ndarray) -> np.ndarray:
    return np.stack([-values[..., 1], values[..., 0]], axis=-1)


def _video_colors(count: int) -> list[tuple[int, int, int]]:
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    return [palette[index % len(palette)] for index in range(max(int(count), 1))]


def _draw_video_panel(
    canvas: np.ndarray,
    *,
    origin_x: int,
    width: int,
    height: int,
    title: str,
    frame_label: str,
    object_ids: list[str],
    positions: np.ndarray,
    mask_row: np.ndarray,
    transform_scale: float,
    transform_min: np.ndarray,
    colors: list[tuple[int, int, int]],
    contact_radii: np.ndarray,
) -> None:
    import cv2

    cv2.rectangle(canvas, (origin_x, 0), (origin_x + width - 1, height - 1), (255, 255, 255), -1)
    cv2.rectangle(canvas, (origin_x, 0), (origin_x + width - 1, height - 1), (210, 210, 210), 1)
    cv2.putText(canvas, title, (origin_x + 24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, frame_label, (origin_x + 24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    for object_index, object_id in enumerate(object_ids):
        if float(mask_row[object_index, 0]) <= 0.0:
            continue
        display = _display_xy(positions[object_index])
        point = (display - transform_min) * transform_scale
        point[1] = float(height) - point[1]
        center = np.asarray([origin_x + point[0], point[1]], dtype=np.float64)
        radius_px = max(5, int(round(float(contact_radii[object_index]) * transform_scale)))
        color_rgb = colors[object_index]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        center_i = (int(round(center[0])), int(round(center[1])))
        cv2.circle(canvas, center_i, radius_px, color_bgr, 2, cv2.LINE_AA)
        cv2.circle(canvas, center_i, 3, color_bgr, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            object_id,
            (center_i[0] + radius_px + 4, center_i[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bgr,
            1,
            cv2.LINE_AA,
        )


def _render_comparison_video(
    output_mp4: Path,
    *,
    object_ids: list[str],
    frames: list[int],
    target: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    target_mask: np.ndarray | None = None,
    predicted_mask: np.ndarray | None = None,
    contact_radii: np.ndarray,
    source_fps: float,
    width: int = 1280,
    height: int = 720,
    fps_override: float | None = None,
    target_title: str = "GT",
    predicted_title: str = "Recovered",
) -> None:
    import cv2

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    video_fps = float(fps_override) if fps_override is not None else max(1.0, min(float(source_fps), 25.0))
    panel_width = int(width) // 2
    display_target = _display_xy(target)
    display_predicted = _display_xy(predicted)
    target_panel_mask = np.asarray(target_mask if target_mask is not None else mask, dtype=np.float64)
    predicted_panel_mask = np.asarray(predicted_mask if predicted_mask is not None else mask, dtype=np.float64)
    active_target = np.asarray(target_panel_mask[..., 0] > 0.0, dtype=bool)
    active_predicted = np.asarray(predicted_panel_mask[..., 0] > 0.0, dtype=bool)
    all_points = np.concatenate([display_target[active_target], display_predicted[active_predicted]], axis=0)
    if all_points.size == 0:
        raise ValueError("no active points to render")
    margin = 52
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((panel_width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    center = 0.5 * (min_xy + max_xy)
    view_span = np.asarray([(panel_width - 2 * margin) / scale, (height - 2 * margin) / scale], dtype=np.float64)
    transform_min = center - 0.5 * view_span
    transform_min[1] -= margin / scale
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_mp4}")
    colors = _video_colors(len(object_ids))
    try:
        for frame_offset, frame_index in enumerate(frames):
            canvas = np.full((int(height), int(width), 3), 255, dtype=np.uint8)
            label = f"frame {int(frame_index)}"
            _draw_video_panel(
                canvas,
                origin_x=0,
                width=panel_width,
                height=int(height),
                title=target_title,
                frame_label=label,
                object_ids=object_ids,
                positions=target[frame_offset],
                mask_row=target_panel_mask[frame_offset],
                transform_scale=scale,
                transform_min=transform_min,
                colors=colors,
                contact_radii=contact_radii,
            )
            _draw_video_panel(
                canvas,
                origin_x=panel_width,
                width=int(width) - panel_width,
                height=int(height),
                title=predicted_title,
                frame_label=label,
                object_ids=object_ids,
                positions=predicted[frame_offset],
                mask_row=predicted_panel_mask[frame_offset],
                transform_scale=scale,
                transform_min=transform_min,
                colors=colors,
                contact_radii=contact_radii,
            )
            cv2.line(canvas, (panel_width, 0), (panel_width, int(height)), (190, 190, 190), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def _rmse_summary(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    object_ids: list[str],
) -> tuple[torch.Tensor, dict[str, float | None]]:
    diff = (predicted - target) * mask
    overall_rmse = torch.sqrt(torch.sum(diff * diff) / torch.clamp(torch.sum(mask), min=1.0))
    per_object = {}
    for object_index, object_id in enumerate(object_ids):
        object_diff = diff[:, object_index]
        object_mask = mask[:, object_index]
        denominator = torch.sum(object_mask)
        per_object[object_id] = (
            float(torch.sqrt(torch.sum(object_diff * object_diff) / denominator).cpu())
            if float(denominator.detach().cpu()) > 0.0
            else None
        )
    return overall_rmse, per_object


def _position_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = predicted - target
    weight = mask.expand_as(diff)
    return torch.sum(diff * diff * weight) / torch.clamp(torch.sum(mask), min=1.0)


def _active_prefix_mask(mask: torch.Tensor, prefix_frames: int) -> torch.Tensor:
    if int(prefix_frames) <= 0:
        return mask
    observed_index = torch.cumsum(mask[:, :, 0], dim=0)
    prefix = (mask[:, :, 0] > 0.0) & (observed_index <= float(prefix_frames))
    return mask * prefix.to(dtype=mask.dtype).unsqueeze(2)


def _active_future_mask(mask: torch.Tensor, prefix_frames: int) -> torch.Tensor:
    if int(prefix_frames) <= 0:
        return mask
    observed_index = torch.cumsum(mask[:, :, 0], dim=0)
    future = (mask[:, :, 0] > 0.0) & (observed_index > float(prefix_frames))
    future_mask = mask * future.to(dtype=mask.dtype).unsqueeze(2)
    if bool(torch.any(future_mask > 0.0).detach().cpu()):
        return future_mask
    return mask


def _frame_count_for_mask(mask: torch.Tensor) -> int:
    if mask.numel() == 0:
        return 0
    active_frames = torch.any(mask[:, :, 0] > 0.0, dim=1)
    offsets = torch.nonzero(active_frames, as_tuple=False).flatten()
    if int(offsets.numel()) <= 0:
        return 1
    return int(offsets[-1].detach().cpu()) + 1


def _active_prefix_fade_mask(
    mask: torch.Tensor,
    previous_prefix_frames: int,
    prefix_frames: int,
    fade_alpha: float,
) -> torch.Tensor:
    if int(previous_prefix_frames) <= 0:
        return _active_prefix_mask(mask, prefix_frames)
    old_mask = _active_prefix_mask(mask, previous_prefix_frames)
    new_prefix_mask = _active_prefix_mask(mask, prefix_frames)
    new_only_mask = torch.clamp(new_prefix_mask - old_mask, min=0.0)
    return old_mask + float(fade_alpha) * new_only_mask


def _active_prefix_end_times(
    *,
    mask: torch.Tensor,
    frames: list[int],
    fps: float,
    frame0: int,
    prefix_frames: int,
    active_metadata: dict[str, Any],
) -> torch.Tensor | None:
    if int(prefix_frames) <= 0:
        return None
    frame_times = torch.tensor(
        [(float(frame) - float(frame0)) / max(float(fps), 1e-12) for frame in frames],
        dtype=mask.dtype,
        device=mask.device,
    )
    prefix_mask = _active_prefix_mask(mask, prefix_frames)[:, :, 0] > 0.0
    end_times = active_metadata["active_start_times"].clone()
    for object_index in range(mask.shape[1]):
        offsets = torch.nonzero(prefix_mask[:, object_index], as_tuple=False).flatten()
        if int(offsets.numel()) > 0:
            end_times[object_index] = frame_times[int(offsets[-1].detach().cpu())]
    return end_times


def _mass_from_free(mass_free: torch.Tensor, object_count: int) -> torch.Tensor:
    fixed = torch.ones((1,), dtype=torch.float64, device=mass_free.device)
    if object_count <= 1:
        return fixed[:object_count]
    return torch.cat([fixed, torch.clamp(mass_free, 0.05, 20.0)])


def _run_impulse_stage(
    input_fit: Path,
    output_dir: Path,
    steps: int,
    lr: float,
    impulse_optimizer: str,
    mode: str,
    contact_assumption: str,
    event_window_frames: int,
    newton_iters: int,
    render_video: bool | None = None,
) -> dict[str, Any]:
    if contact_assumption not in CONTACT_ASSUMPTIONS:
        raise ValueError(f"unsupported contact_assumption: {contact_assumption}")
    if impulse_optimizer not in IMPULSE_OPTIMIZERS:
        raise ValueError(f"unsupported impulse optimizer: {impulse_optimizer}")
    if mode not in (*IMPULSE_STAGE_MODES, MODE_NORMAL_COLLISION):
        raise ValueError(f"unsupported mode: {mode}")
    if mode in IMPULSE_STAGE_MODES and contact_assumption == "unknown_contact":
        raise ValueError(f"{mode} mode requires explicit event frames")
    render_video_flag = _render_video_enabled(render_video)
    normal_source = {
        "known_position_time": "target",
        "known_time": "rollout",
        "unknown_contact": "free_contact",
    }[contact_assumption]
    fit = _load_json(input_fit)
    object_ids, target_plane = _project_target_to_plane(fit)
    fps = float(fit["trajectory_physics_initialization"]["fps"])
    scales = _object_scales(fit, object_ids, target_plane)
    shape_ids, _half_extents, _object_angles, contact_radii, _shape_names, _contact_proxy_names = _shape_metadata(
        fit=fit,
        object_ids=object_ids,
        object_scales_by_id=scales,
    )
    events = _detected_collision_events(
        fit=fit,
        object_ids=object_ids,
        target_plane=target_plane,
    )
    object_index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    event_specs = [
        {
            **event,
            "object_indices": [object_index_by_id[object_id] for object_id in event["object_ids"]],
        }
        for event in events
    ]
    frames, target, mask = _target_tensor(object_ids, target_plane)
    device = torch.device("cpu")
    target = target.to(device)
    mask = mask.to(device)
    active_metadata = _active_metadata(object_ids, target_plane, frames, target, fps)
    object_scales_tensor = torch.tensor([float(scales[object_id]) for object_id in object_ids], dtype=torch.float64, device=device)
    event_normals = _event_normals(event_specs, frames, target)
    unary_event_indices = [index for index, event in enumerate(event_specs) if len(event["object_indices"]) == 1]
    pair_event_indices = [index for index, event in enumerate(event_specs) if len(event["object_indices"]) == 2]
    object_pairs = _object_pairs(len(object_ids))
    pair_index_by_key = {tuple(pair): index for index, pair in enumerate(object_pairs)}
    v0 = torch.zeros((len(object_ids), 2), dtype=torch.float64, device=device, requires_grad=True)
    friction = torch.zeros((len(object_ids),), dtype=torch.float64, device=device, requires_grad=True)
    impulses = torch.zeros((len(event_specs), 2), dtype=torch.float64, device=device, requires_grad=True)
    pair_delta_v = torch.zeros((len(event_specs), 2), dtype=torch.float64, device=device, requires_grad=True)
    pair_log_alpha = torch.zeros((len(event_specs),), dtype=torch.float64, device=device, requires_grad=True)
    mass_free = torch.ones((max(len(object_ids) - 1, 0),), dtype=torch.float64, device=device, requires_grad=True)
    restitution_raw = torch.zeros((len(object_pairs),), dtype=torch.float64, device=device, requires_grad=True)
    if mode == MODE_NORMAL_COLLISION:
        variables = [v0, friction, mass_free, restitution_raw]
        if contact_assumption != "unknown_contact" and unary_event_indices:
            variables.append(impulses)
    elif mode in IMPULSE_STAGE_MODES:
        variables = [v0, friction, impulses, pair_delta_v, pair_log_alpha]
    else:
        raise ValueError(f"unsupported mode: {mode}")
    trace = []
    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    optimizer_details: dict[str, Any] = {}

    def compute_loss_and_prediction() -> tuple[torch.Tensor, torch.Tensor]:
        pair_restitution = torch.sigmoid(restitution_raw) if mode == MODE_NORMAL_COLLISION else torch.zeros(
            (len(object_pairs),), dtype=torch.float64, device=device
        )
        mass = _mass_from_free(mass_free, len(object_ids))
        if contact_assumption == "unknown_contact":
            predicted, _ = _rollout_free_contact(
                target=target,
                frames=frames,
                object_pairs=object_pairs,
                object_scales=object_scales_tensor,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                mass=mass,
                pair_restitution=pair_restitution,
                fps=fps,
                active_metadata=active_metadata,
            )
        elif mode in IMPULSE_STAGE_MODES:
            predicted, _velocities, _event_records = _rollout_impulse_analytic(
                target=target,
                frames=frames,
                event_specs=event_specs,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                impulses=impulses,
                pair_delta_v=pair_delta_v,
                pair_log_alpha=pair_log_alpha,
                fps=fps,
                event_window_frames=0 if mode == MODE_IMPULSE_FORCED_FRAME else int(event_window_frames),
                newton_iters=newton_iters,
                active_metadata=active_metadata,
            )
        else:
            predicted, _ = _rollout_event_assumption(
                target=target,
                frames=frames,
                event_specs=event_specs,
                pair_index_by_key=pair_index_by_key,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                impulses=impulses,
                pair_delta_v=pair_delta_v,
                pair_log_alpha=pair_log_alpha,
                mass=mass,
                pair_restitution=pair_restitution,
                event_normals=event_normals,
                mode=mode,
                normal_source=normal_source,
                fps=fps,
                active_metadata=active_metadata,
            )
        position_loss = _position_mse(predicted, target, mask)
        return position_loss, predicted

    def update_best(current_loss: float) -> None:
        nonlocal best_loss, best_state
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = (
                v0.detach().clone(),
                friction.detach().clone(),
                impulses.detach().clone(),
                pair_delta_v.detach().clone(),
                pair_log_alpha.detach().clone(),
                mass_free.detach().clone(),
                restitution_raw.detach().clone(),
            )

    def pack_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = [
            v0.detach().cpu().numpy().reshape(-1),
            friction.detach().cpu().numpy().reshape(-1),
        ]
        lower_parts = [
            np.full((v0.numel(),), -IMPULSE_VELOCITY_BOUND_M_PER_S, dtype=np.float64),
            np.zeros((friction.numel(),), dtype=np.float64),
        ]
        upper_parts = [
            np.full((v0.numel(),), IMPULSE_VELOCITY_BOUND_M_PER_S, dtype=np.float64),
            np.ones((friction.numel(),), dtype=np.float64),
        ]
        if unary_event_indices:
            parts.append(impulses.detach().cpu().numpy()[unary_event_indices].reshape(-1))
            lower_parts.append(np.full((2 * len(unary_event_indices),), -IMPULSE_DELTA_V_BOUND_M_PER_S, dtype=np.float64))
            upper_parts.append(np.full((2 * len(unary_event_indices),), IMPULSE_DELTA_V_BOUND_M_PER_S, dtype=np.float64))
        if pair_event_indices:
            parts.append(pair_delta_v.detach().cpu().numpy()[pair_event_indices].reshape(-1))
            lower_parts.append(np.full((2 * len(pair_event_indices),), -IMPULSE_DELTA_V_BOUND_M_PER_S, dtype=np.float64))
            upper_parts.append(np.full((2 * len(pair_event_indices),), IMPULSE_DELTA_V_BOUND_M_PER_S, dtype=np.float64))
            parts.append(pair_log_alpha.detach().cpu().numpy()[pair_event_indices].reshape(-1))
            lower_parts.append(np.full((len(pair_event_indices),), -5.0, dtype=np.float64))
            upper_parts.append(np.full((len(pair_event_indices),), 5.0, dtype=np.float64))
        return (
            np.concatenate(parts).astype(np.float64, copy=False),
            np.concatenate(lower_parts),
            np.concatenate(upper_parts),
        )

    def assign_state(values: np.ndarray) -> None:
        cursor = 0
        with torch.no_grad():
            next_cursor = cursor + v0.numel()
            v0.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor].reshape(tuple(v0.shape)),
                    dtype=torch.float64,
                    device=device,
                )
            )
            cursor = next_cursor
            next_cursor = cursor + friction.numel()
            friction.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor],
                    dtype=torch.float64,
                    device=device,
                )
            )
            cursor = next_cursor
            if unary_event_indices:
                next_cursor = cursor + 2 * len(unary_event_indices)
                impulses[unary_event_indices] = torch.as_tensor(
                    values[cursor:next_cursor].reshape((len(unary_event_indices), 2)),
                    dtype=torch.float64,
                    device=device,
                )
                cursor = next_cursor
            if pair_event_indices:
                next_cursor = cursor + 2 * len(pair_event_indices)
                pair_delta_v[pair_event_indices] = torch.as_tensor(
                    values[cursor:next_cursor].reshape((len(pair_event_indices), 2)),
                    dtype=torch.float64,
                    device=device,
                )
                cursor = next_cursor
                next_cursor = cursor + len(pair_event_indices)
                pair_log_alpha[pair_event_indices] = torch.as_tensor(
                    values[cursor:next_cursor],
                    dtype=torch.float64,
                    device=device,
                )

    def masked_position_residual(values: np.ndarray) -> np.ndarray:
        assign_state(values)
        with torch.no_grad():
            _loss, predicted = compute_loss_and_prediction()
            diff = predicted - target
            active = mask.expand_as(diff) > 0.0
            return diff[active].detach().cpu().numpy().astype(np.float64, copy=False)

    if impulse_optimizer == IMPULSE_OPTIMIZER_ADAM:
        optimizer = torch.optim.Adam(variables, lr=float(lr), weight_decay=0.0)
        for step in range(max(int(steps), 0)):
            optimizer.zero_grad()
            loss, predicted = compute_loss_and_prediction()
            position_loss = loss
            current_loss = float(loss.detach().cpu())
            if current_loss < best_loss:
                update_best(current_loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                friction.clamp_(0.0, 1.0)
                pair_log_alpha.clamp_(-5.0, 5.0)
                mass_free.clamp_(0.05, 20.0)
            with torch.no_grad():
                if step < 20 or (step + 1) % 100 == 0:
                    rmse = torch.sqrt(_position_mse(predicted, target, mask)).detach().cpu()
                    trace.append(
                        {
                            "step": int(step + 1),
                            "loss": current_loss,
                            "position_loss": float(position_loss.detach().cpu()),
                            "rmse_m": float(rmse),
                        }
                    )
    elif impulse_optimizer == IMPULSE_OPTIMIZER_LBFGS:
        optimizer = torch.optim.LBFGS(
            variables,
            lr=1.0,
            max_iter=max(int(steps), 1),
            max_eval=max(2 * int(steps), 1),
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad()
            loss, predicted = compute_loss_and_prediction()
            position_loss = loss
            current_loss = float(loss.detach().cpu())
            if current_loss < best_loss:
                update_best(current_loss)
            loss.backward()
            closure_calls += 1
            if closure_calls <= 20 or closure_calls % 100 == 0:
                with torch.no_grad():
                    rmse = torch.sqrt(_position_mse(predicted, target, mask)).detach().cpu()
                    trace.append(
                        {
                            "step": int(closure_calls),
                            "loss": current_loss,
                            "position_loss": float(position_loss.detach().cpu()),
                            "rmse_m": float(rmse),
                        }
                    )
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            friction.clamp_(0.0, 1.0)
            pair_log_alpha.clamp_(-5.0, 5.0)
            mass_free.clamp_(0.05, 20.0)
        optimizer_details = {
            "method": "torch_lbfgs",
            "line_search_fn": "strong_wolfe",
            "lr": 1.0,
            "max_iter": int(max(int(steps), 1)),
            "closure_calls": int(closure_calls),
        }
    else:
        x0, lower_bounds, upper_bounds = pack_state()
        with torch.no_grad():
            initial_loss, initial_predicted = compute_loss_and_prediction()
            trace.append(
                {
                    "step": 0,
                    "loss": float(initial_loss.detach().cpu()),
                    "position_loss": float(initial_loss.detach().cpu()),
                    "rmse_m": float(torch.sqrt(_position_mse(initial_predicted, target, mask)).detach().cpu()),
                }
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"scipy\.optimize\._lsq\..*",
            )
            result = least_squares(
                masked_position_residual,
                x0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                x_scale=1.0,
                max_nfev=max(int(steps), 1),
                ftol=None,
                xtol=None,
                gtol=1e-10,
            )
        assign_state(result.x)
        with torch.no_grad():
            final_loss, final_predicted = compute_loss_and_prediction()
            trace.append(
                {
                    "step": int(result.nfev),
                    "loss": float(final_loss.detach().cpu()),
                    "position_loss": float(final_loss.detach().cpu()),
                    "rmse_m": float(torch.sqrt(_position_mse(final_predicted, target, mask)).detach().cpu()),
                    "least_squares_cost": float(result.cost),
                    "least_squares_optimality": float(result.optimality),
                    "least_squares_status": int(result.status),
                    "least_squares_message": str(result.message),
                }
            )
        optimizer_details = {
            "method": "trf",
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "status": int(result.status),
            "message": str(result.message),
            "success": bool(result.success),
        }
    if best_state is not None:
        with torch.no_grad():
            v0.copy_(best_state[0])
            friction.copy_(best_state[1])
            impulses.copy_(best_state[2])
            pair_delta_v.copy_(best_state[3])
            pair_log_alpha.copy_(best_state[4])
            mass_free.copy_(best_state[5])
            restitution_raw.copy_(best_state[6])
    with torch.no_grad():
        pair_restitution = torch.sigmoid(restitution_raw) if mode == MODE_NORMAL_COLLISION else torch.zeros(
            (len(object_pairs),), dtype=torch.float64, device=device
        )
        mass = _mass_from_free(mass_free, len(object_ids))
        if contact_assumption == "unknown_contact":
            optimization_predicted, optimization_velocities = _rollout_free_contact(
                target=target,
                frames=frames,
                object_pairs=object_pairs,
                object_scales=object_scales_tensor,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                mass=mass,
                pair_restitution=pair_restitution,
                fps=fps,
                active_metadata=active_metadata,
            )
            impulse_event_records: list[dict[str, Any]] = []
        elif mode in IMPULSE_STAGE_MODES:
            optimization_predicted, optimization_velocities, impulse_event_records = _rollout_impulse_analytic(
                target=target,
                frames=frames,
                event_specs=event_specs,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                impulses=impulses,
                pair_delta_v=pair_delta_v,
                pair_log_alpha=pair_log_alpha,
                fps=fps,
                event_window_frames=0 if mode == MODE_IMPULSE_FORCED_FRAME else int(event_window_frames),
                newton_iters=newton_iters,
                active_metadata=active_metadata,
            )
        else:
            optimization_predicted, optimization_velocities = _rollout_event_assumption(
                target=target,
                frames=frames,
                event_specs=event_specs,
                pair_index_by_key=pair_index_by_key,
                v0=v0,
                friction=torch.clamp(friction, 0.0, 1.0),
                impulses=impulses,
                pair_delta_v=pair_delta_v,
                pair_log_alpha=pair_log_alpha,
                mass=mass,
                pair_restitution=pair_restitution,
                event_normals=event_normals,
                mode=mode,
                normal_source=normal_source,
                fps=fps,
                active_metadata=active_metadata,
            )
            impulse_event_records = []
        free_predicted, _ = _rollout_free_contact(
            target=target,
            frames=frames,
            object_pairs=object_pairs,
            object_scales=object_scales_tensor,
            v0=v0,
            friction=torch.clamp(friction, 0.0, 1.0),
            mass=mass,
            pair_restitution=pair_restitution,
            fps=fps,
            active_metadata=active_metadata,
        )
        optimization_rmse, optimization_per_object = _rmse_summary(optimization_predicted, target, mask, object_ids)
        free_rollout_rmse, free_rollout_per_object = _rmse_summary(free_predicted, target, mask, object_ids)
        frame_to_offset = {int(frame): index for index, frame in enumerate(frames)}
        event_velocity_summary = []
        event_used_normals = []
        impulse_event_records_by_index = {
            int(record["event_index"].detach().cpu()): record
            for record in impulse_event_records
        }
        for event_index, event in enumerate(events):
            object_indices = [int(index) for index in event_specs[event_index]["object_indices"]]
            offset = frame_to_offset.get(int(event["frame"]), 0)
            before_offset = max(offset - 1, 0)
            after_offset = min(offset, int(optimization_velocities.shape[0]) - 1)
            target_normal = event_normals[event_index]
            impulse_record = impulse_event_records_by_index.get(int(event_index))
            if impulse_record is not None:
                used_normal = impulse_record["normal"]
            elif mode == MODE_NORMAL_COLLISION and normal_source == "rollout" and len(object_indices) == 2:
                rel = optimization_predicted[offset, object_indices[0]] - optimization_predicted[offset, object_indices[1]]
                used_normal = rel / torch.sqrt(torch.sum(rel * rel) + 1e-12)
            else:
                used_normal = target_normal
            event_used_normals.append(used_normal.detach().clone())
            pair_restitution_value = None
            if mode == MODE_NORMAL_COLLISION and len(object_indices) == 2:
                pair_index = pair_index_by_key[tuple(sorted(object_indices))]
                pair_restitution_value = float(pair_restitution.detach().cpu()[pair_index])
            summary = {
                "event_index": int(event_index),
                "event_type": event["event_type"],
                "frame": int(event["frame"]),
                "object_ids": event["object_ids"],
                "normal_source": normal_source if mode == MODE_NORMAL_COLLISION else None,
                "event_time_s": (
                    float(impulse_record["event_time_s"].detach().cpu())
                    if impulse_record is not None
                    else (float(event["frame"]) - float(frames[0])) / max(float(fps), 1e-12)
                ),
                "event_frame_float": (
                    float(impulse_record["event_frame_float"].detach().cpu())
                    if impulse_record is not None
                    else float(event["frame"])
                ),
                "contact_residual_m": (
                    float(impulse_record["contact_residual_m"].detach().cpu())
                    if impulse_record is not None
                    else None
                ),
                "normal_plane": [float(value) for value in used_normal.detach().cpu().tolist()],
                "target_normal_plane": [float(value) for value in target_normal.detach().cpu().tolist()],
                "restitution": pair_restitution_value,
                "pre_event_velocity_plane_m_per_s": {
                    object_ids[index]: [
                        float(value)
                        for value in (
                            impulse_record["pre_velocity"].detach().cpu()[index]
                            if impulse_record is not None
                            else optimization_velocities.detach().cpu()[before_offset, index]
                        ).tolist()
                    ]
                    for index in object_indices
                },
                "post_event_velocity_plane_m_per_s": {
                    object_ids[index]: [
                        float(value)
                        for value in (
                            impulse_record["post_velocity"].detach().cpu()[index]
                            if impulse_record is not None
                            else optimization_velocities.detach().cpu()[after_offset, index]
                        ).tolist()
                    ]
                    for index in object_indices
                },
            }
            event_velocity_summary.append(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_plot in (
        "optimization_true_vs_recovered.png",
        "free_rollout_true_vs_recovered.png",
        "free_impulse_true_vs_recovered.png",
        "forced_frame_impulse_true_vs_recovered.png",
        "impulse_forced_frame_true_vs_recovered.png",
        "impulse_forced_frame_free_rollout_true_vs_recovered.png",
        "impulse_forced_window_true_vs_recovered.png",
        "impulse_forced_window_free_rollout_true_vs_recovered.png",
    ):
        (output_dir / obsolete_plot).unlink(missing_ok=True)
    stage_name = _impulse_stage_name(mode)
    stage_plot_path = output_dir / f"{stage_name}.png"
    _plot(
        stage_plot_path,
        object_ids,
        target.detach().cpu().numpy(),
        optimization_predicted.detach().cpu().numpy(),
        mask.detach().cpu().numpy(),
    )
    stage_video_path = stage_plot_path.with_suffix(".mp4") if render_video_flag else None
    if stage_video_path is not None:
        _render_comparison_video(
            stage_video_path,
            object_ids=object_ids,
            frames=frames,
            target=target.detach().cpu().numpy(),
            predicted=optimization_predicted.detach().cpu().numpy(),
            mask=mask.detach().cpu().numpy(),
            contact_radii=contact_radii.detach().cpu().numpy(),
            source_fps=fps,
        )
    free_rollout_plot_path = output_dir / f"{stage_name}_free_rollout.png"
    _plot(
        free_rollout_plot_path,
        object_ids,
        target.detach().cpu().numpy(),
        free_predicted.detach().cpu().numpy(),
        mask.detach().cpu().numpy(),
    )
    free_rollout_video_path = free_rollout_plot_path.with_suffix(".mp4") if render_video_flag else None
    if free_rollout_video_path is not None:
        _render_comparison_video(
            free_rollout_video_path,
            object_ids=object_ids,
            frames=frames,
            target=target.detach().cpu().numpy(),
            predicted=free_predicted.detach().cpu().numpy(),
            mask=mask.detach().cpu().numpy(),
            contact_radii=contact_radii.detach().cpu().numpy(),
            source_fps=fps,
        )
    payload = {
        "input_fit": str(input_fit),
        "steps": int(steps),
        "lr": float(lr),
        "fps": float(fps),
        "model": mode,
        "optimizer": impulse_optimizer,
        "optimizer_details": optimizer_details,
        "contact_assumption": contact_assumption,
        "stage_name": stage_name,
        "render_video": render_video_flag,
        "event_window_frames": 0 if mode == MODE_IMPULSE_FORCED_FRAME else int(event_window_frames),
        "newton_iters": int(newton_iters),
        "normal_source": normal_source if mode == MODE_NORMAL_COLLISION else None,
        "event_semantics": (
            "known_position_time uses target event normals; known_time uses event frames with rollout normals; "
            "unknown_contact uses only free geometry-triggered contacts"
            if mode == MODE_NORMAL_COLLISION
            else (
                "closed_form_sliding; event time is fixed to detected frame; "
                "pair events use free delta_v_i=v and delta_v_j=-alpha*v"
                if mode == MODE_IMPULSE_FORCED_FRAME
                else (
                    "closed_form_sliding; event time is solved within the detected frame window; "
                    "pair events use free delta_v_i=v and delta_v_j=-alpha*v"
                )
            )
        ),
        "free_rollout_semantics": "final metrics use no event frames; contacts are triggered only by proxy geometry overlap",
        "collision_detection_policy": "2d_true_shape_pose_center_line_inside_fraction",
        "object_scales_m": scales,
        "active_intervals": _active_interval_summary(object_ids, active_metadata),
        "events": events,
        "stage_rmse_m": float(optimization_rmse.cpu()),
        "optimization_rmse_m": float(optimization_rmse.cpu()),
        "optimization_per_object_rmse_m": optimization_per_object,
        "free_rollout_rmse_m": float(free_rollout_rmse.cpu()),
        "free_rollout_per_object_rmse_m": free_rollout_per_object,
        "overall_rmse_m": float(free_rollout_rmse.cpu()),
        "per_object_rmse_m": free_rollout_per_object,
        "parameters": {
            object_id: {
                "initial_velocity_plane_m_per_s": [float(value) for value in v0.detach().cpu()[object_index].tolist()],
                "ground_friction": float(torch.clamp(friction.detach(), 0.0, 1.0).cpu()[object_index]),
                "mass": float(mass.detach().cpu()[object_index]),
            }
            for object_index, object_id in enumerate(object_ids)
        },
        "pair_restitution": [
            {
                "object_ids": [object_ids[i], object_ids[j]],
                "restitution": float(pair_restitution.detach().cpu()[pair_index]),
            }
            for pair_index, (i, j) in enumerate(object_pairs)
        ],
        "event_impulses": [
            {
                "event_index": int(index),
                "event_type": event["event_type"],
                "frame": int(event["frame"]),
                "event_time_s": (
                    float(impulse_event_records_by_index[index]["event_time_s"].detach().cpu())
                    if index in impulse_event_records_by_index
                    else None
                ),
                "event_frame_float": (
                    float(impulse_event_records_by_index[index]["event_frame_float"].detach().cpu())
                    if index in impulse_event_records_by_index
                    else None
                ),
                "contact_residual_m": (
                    float(impulse_event_records_by_index[index]["contact_residual_m"].detach().cpu())
                    if index in impulse_event_records_by_index
                    else None
                ),
                "object_ids": event["object_ids"],
                "impulse_plane": (
                    [float(value) for value in pair_delta_v.detach().cpu()[index].tolist()]
                    if mode in IMPULSE_STAGE_MODES and len(event["object_ids"]) == 2
                    else [float(value) for value in impulses.detach().cpu()[index].tolist()]
                ),
                "impulse_semantics": (
                    "field stores pair delta_v_i; not a physical impulse"
                    if mode in IMPULSE_STAGE_MODES and len(event["object_ids"]) == 2
                    else (
                        "pair_physical_impulse; delta_v_i=P/m_i and delta_v_j=-P/m_j"
                        if len(event["object_ids"]) == 2
                        else "unary_delta_v"
                    )
                ),
                "pair_delta_v_alpha": (
                    {
                        "alpha_delta_v_j_over_delta_v_i": float(
                            torch.exp(torch.clamp(pair_log_alpha.detach(), -5.0, 5.0)).cpu()[index]
                        ),
                        "mass_ratio_i_over_j_convention": float(
                            torch.exp(torch.clamp(pair_log_alpha.detach(), -5.0, 5.0)).cpu()[index]
                        ),
                        "delta_v_plane_by_object": {
                            event["object_ids"][0]: [
                                float(value) for value in pair_delta_v.detach().cpu()[index].tolist()
                            ],
                            event["object_ids"][1]: [
                                float(value)
                                for value in (
                                    -torch.exp(torch.clamp(pair_log_alpha.detach(), -5.0, 5.0))[index]
                                    * pair_delta_v.detach()
                                )
                                .cpu()[index]
                                .tolist()
                            ],
                        },
                    }
                    if mode in IMPULSE_STAGE_MODES and len(event["object_ids"]) == 2
                    else None
                ),
                "normal_source": normal_source if mode == MODE_NORMAL_COLLISION else None,
                "normal_plane": [float(value) for value in event_used_normals[index].detach().cpu().tolist()],
                "target_normal_plane": [float(value) for value in event_normals.detach().cpu()[index].tolist()],
                "restitution": (
                    float(pair_restitution.detach().cpu()[pair_index_by_key[tuple(sorted(event_specs[index]["object_indices"]))]])
                    if mode == MODE_NORMAL_COLLISION and len(event["object_ids"]) == 2
                    else None
                ),
            }
            for index, event in enumerate(events)
        ],
        "event_velocity_summary": event_velocity_summary,
        "trace": trace,
        "stage_plot_path": str(stage_plot_path),
        "stage_video_path": str(stage_video_path) if stage_video_path is not None else None,
        "impulse_forced_frame_plot_path": str(stage_plot_path) if mode == MODE_IMPULSE_FORCED_FRAME else None,
        "impulse_forced_frame_video_path": (
            str(stage_video_path) if mode == MODE_IMPULSE_FORCED_FRAME and stage_video_path is not None else None
        ),
        "impulse_forced_frame_free_rollout_plot_path": (
            str(free_rollout_plot_path) if mode == MODE_IMPULSE_FORCED_FRAME else None
        ),
        "impulse_forced_frame_free_rollout_video_path": (
            str(free_rollout_video_path)
            if mode == MODE_IMPULSE_FORCED_FRAME and free_rollout_video_path is not None
            else None
        ),
        "impulse_forced_window_plot_path": str(stage_plot_path) if mode == MODE_IMPULSE_FORCED_WINDOW else None,
        "impulse_forced_window_video_path": (
            str(stage_video_path) if mode == MODE_IMPULSE_FORCED_WINDOW and stage_video_path is not None else None
        ),
        "impulse_forced_window_free_rollout_plot_path": (
            str(free_rollout_plot_path) if mode == MODE_IMPULSE_FORCED_WINDOW else None
        ),
        "impulse_forced_window_free_rollout_video_path": (
            str(free_rollout_video_path)
            if mode == MODE_IMPULSE_FORCED_WINDOW and free_rollout_video_path is not None
            else None
        ),
        "normal_collision_plot_path": str(stage_plot_path) if mode == MODE_NORMAL_COLLISION else None,
        "free_rollout_plot_path": str(free_rollout_plot_path),
        "free_rollout_video_path": str(free_rollout_video_path) if free_rollout_video_path is not None else None,
        "plot_path": str(stage_plot_path),
        "video_path": str(stage_video_path) if stage_video_path is not None else None,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _pair_mass_ratio_raw_from_value(value: float) -> float:
    normalized = (float(value) - MIN_PAIR_MASS_RATIO) / (MAX_PAIR_MASS_RATIO - MIN_PAIR_MASS_RATIO)
    return _logit(normalized)


def _friction_raw_from_value(value: float) -> float:
    normalized = (float(value) - MIN_FRICTION) / (MAX_FRICTION - MIN_FRICTION)
    return _logit(normalized)


def _restitution_raw_from_value(value: float) -> float:
    return _logit(float(np.clip(value, 1e-6, 1.0 - 1e-6)))


def _median_or_default(values: list[float], default: float) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float(default)
    return float(np.median(np.asarray(finite_values, dtype=np.float64)))


def _validate_supported_shapes(fit: dict[str, Any], object_ids: list[str]) -> None:
    specs = {
        str(item.get("object_id")): str(item.get("geometry_type") or "").lower()
        for item in fit.get("trajectory_physics_initialization", {}).get("objects", [])
        if isinstance(item, dict)
    }
    unsupported = []
    for object_id in object_ids:
        geometry_type = specs.get(object_id, "")
        if geometry_type and geometry_type not in {"sphere", "cylinder", "box", "cube"}:
            unsupported.append((object_id, geometry_type))
    if unsupported:
        values = ", ".join(f"{object_id}:{geometry_type}" for object_id, geometry_type in unsupported)
        raise ValueError(f"analytic-event-collision-v1 does not support object shapes: {values}")


def _shape_metadata(
    *,
    fit: dict[str, Any],
    object_ids: list[str],
    object_scales_by_id: dict[str, float],
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    specs = {
        str(item.get("object_id")): str(item.get("geometry_type") or "").lower()
        for item in fit.get("trajectory_physics_initialization", {}).get("objects", [])
        if isinstance(item, dict)
    }
    half_extents_by_object = fit.get("physics_rollout", {}).get("shape_half_extents_2d_by_object", {})
    plane = fit.get("analytic_support_plane", {})
    target_trajectories = fit.get("target_trajectories", {})
    shape_ids: list[int] = []
    shape_names: list[str] = []
    half_extents: list[list[float]] = []
    contact_radii: list[float] = []
    contact_proxy_names: list[str] = []
    angles: list[float] = []
    for object_id in object_ids:
        geometry_type = specs.get(object_id, "")
        is_box = geometry_type in {"box", "cube"}
        shape_ids.append(SHAPE_BOX if is_box else SHAPE_CIRCLE)
        shape_names.append("box" if is_box else "circle")
        raw_half_extents = half_extents_by_object.get(object_id)
        if isinstance(raw_half_extents, list) and len(raw_half_extents) >= 2:
            half_extent_x = max(float(raw_half_extents[0]), 1e-12)
            half_extent_y = max(float(raw_half_extents[1]), 1e-12)
        else:
            radius = 0.5 * max(float(object_scales_by_id.get(object_id, 0.0)), 1e-12)
            half_extent_x = radius
            half_extent_y = radius
        half_extents.append([half_extent_x, half_extent_y])
        if is_box:
            contact_radii.append(float(math.sqrt(4.0 * half_extent_x * half_extent_y / math.pi)))
            contact_proxy_names.append("equal_area_circle")
        else:
            contact_radii.append(0.5 * max(float(object_scales_by_id.get(object_id, 0.0)), 1e-12))
            contact_proxy_names.append("native_circle")
        records = target_trajectories.get(object_id) or []
        if records and isinstance(records[0], dict) and records[0].get("pose_4x4") is not None and plane:
            angles.append(float(_pose_yaw_angle(records[0]["pose_4x4"], plane)))
        else:
            angles.append(0.0)
    return (
        tuple(shape_ids),
        torch.tensor(half_extents, dtype=torch.float64),
        torch.tensor(angles, dtype=torch.float64),
        torch.tensor(contact_radii, dtype=torch.float64),
        shape_names,
        contact_proxy_names,
    )


def _pair_initialization_from_impulse_forced_frame_result(
    *,
    impulse_forced_frame_result: dict[str, Any],
    object_index_by_id: dict[str, int],
    object_pairs: list[tuple[int, int]],
    pair_index_by_key: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    ratio_observations: list[list[float]] = [[] for _ in object_pairs]
    restitution_observations: list[list[float]] = [[] for _ in object_pairs]

    def pair_index_and_order(object_ids: list[str]) -> tuple[int, int, int] | None:
        if len(object_ids) != 2:
            return None
        i = object_index_by_id.get(str(object_ids[0]))
        j = object_index_by_id.get(str(object_ids[1]))
        if i is None or j is None:
            return None
        pair_key = tuple(sorted((i, j)))
        pair_index = pair_index_by_key.get(pair_key)
        if pair_index is None:
            return None
        return pair_index, i, j

    def add_ratio(object_ids: list[str], event_ratio_i_over_j: float) -> None:
        resolved = pair_index_and_order(object_ids)
        if resolved is None:
            return
        pair_index, i, j = resolved
        if not math.isfinite(float(event_ratio_i_over_j)) or float(event_ratio_i_over_j) <= 0.0:
            return
        pair_key = object_pairs[pair_index]
        pair_ratio = float(event_ratio_i_over_j) if (i, j) == pair_key else 1.0 / float(event_ratio_i_over_j)
        ratio_observations[pair_index].append(float(np.clip(pair_ratio, MIN_PAIR_MASS_RATIO, MAX_PAIR_MASS_RATIO)))

    for item in impulse_forced_frame_result.get("event_impulses", []):
        object_ids = [str(object_id) for object_id in item.get("object_ids", [])]
        pair_delta = item.get("pair_delta_v_alpha")
        if not isinstance(pair_delta, dict):
            continue
        ratio = pair_delta.get("mass_ratio_i_over_j_convention", pair_delta.get("alpha_delta_v_j_over_delta_v_i"))
        if ratio is not None:
            add_ratio(object_ids, float(ratio))

    for summary in impulse_forced_frame_result.get("event_velocity_summary", []):
        object_ids = [str(object_id) for object_id in summary.get("object_ids", [])]
        resolved = pair_index_and_order(object_ids)
        if resolved is None:
            continue
        pair_index, i, j = resolved
        normal_values = summary.get("normal_plane") or summary.get("target_normal_plane")
        pre_by_object = summary.get("pre_event_velocity_plane_m_per_s") or {}
        post_by_object = summary.get("post_event_velocity_plane_m_per_s") or {}
        if normal_values is None or object_ids[0] not in pre_by_object or object_ids[1] not in pre_by_object:
            continue
        if object_ids[0] not in post_by_object or object_ids[1] not in post_by_object:
            continue
        normal = np.asarray(normal_values, dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-12 or not math.isfinite(normal_norm):
            continue
        normal = normal / normal_norm
        pre_i = np.asarray(pre_by_object[object_ids[0]], dtype=np.float64)
        pre_j = np.asarray(pre_by_object[object_ids[1]], dtype=np.float64)
        post_i = np.asarray(post_by_object[object_ids[0]], dtype=np.float64)
        post_j = np.asarray(post_by_object[object_ids[1]], dtype=np.float64)
        pre_relative_normal = float(np.dot(pre_i - pre_j, normal))
        post_relative_normal = float(np.dot(post_i - post_j, normal))
        if abs(pre_relative_normal) > 1e-12:
            restitution = -post_relative_normal / pre_relative_normal
            if math.isfinite(restitution):
                restitution_observations[pair_index].append(float(np.clip(restitution, 1e-6, 1.0 - 1e-6)))
        delta_i = post_i - pre_i
        delta_j = post_j - pre_j
        delta_i_normal = float(np.dot(delta_i, normal))
        delta_j_normal = float(np.dot(delta_j, normal))
        if abs(delta_i_normal) > 1e-12:
            event_ratio_i_over_j = -delta_j_normal / delta_i_normal
            add_ratio(object_ids, event_ratio_i_over_j)

    pair_mass_ratio = np.asarray(
        [_median_or_default(values, 1.0) for values in ratio_observations],
        dtype=np.float64,
    )
    pair_restitution = np.asarray(
        [_median_or_default(values, 0.5) for values in restitution_observations],
        dtype=np.float64,
    )
    return pair_mass_ratio, pair_restitution


def _pair_initialization_from_analytic_result(
    *,
    analytic_result: dict[str, Any],
    object_index_by_id: dict[str, int],
    object_pairs: list[tuple[int, int]],
    pair_index_by_key: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    pair_mass_ratio = np.full((len(object_pairs),), 1.0, dtype=np.float64)
    pair_restitution = np.full((len(object_pairs),), 0.5, dtype=np.float64)

    for item in analytic_result.get("pair_collision_parameters", []):
        object_ids = [str(object_id) for object_id in item.get("object_ids", [])]
        if len(object_ids) != 2:
            continue
        i = object_index_by_id.get(object_ids[0])
        j = object_index_by_id.get(object_ids[1])
        if i is None or j is None:
            continue
        pair_key = tuple(sorted((i, j)))
        pair_index = pair_index_by_key.get(pair_key)
        if pair_index is None:
            continue
        direct_pair = bool(item.get("direct_pair_parameter_available", False))
        raw_ratio = (
            item.get("optimized_pair_mass_ratio_i_over_j")
            if direct_pair and item.get("optimized_pair_mass_ratio_i_over_j") is not None
            else item.get("mass_ratio_i_over_j")
        )
        raw_restitution = (
            item.get("optimized_pair_restitution")
            if direct_pair and item.get("optimized_pair_restitution") is not None
            else item.get("restitution")
        )
        if raw_ratio is not None and math.isfinite(float(raw_ratio)) and float(raw_ratio) > 0.0:
            ratio = float(raw_ratio)
            if (i, j) != object_pairs[pair_index]:
                ratio = 1.0 / max(ratio, 1e-12)
            pair_mass_ratio[pair_index] = float(np.clip(ratio, MIN_PAIR_MASS_RATIO, MAX_PAIR_MASS_RATIO))
        if raw_restitution is not None and math.isfinite(float(raw_restitution)):
            pair_restitution[pair_index] = float(np.clip(float(raw_restitution), 1e-6, 1.0 - 1e-6))
    return pair_mass_ratio, pair_restitution


def _closed_form_motion(
    x0: torch.Tensor,
    v0: torch.Tensor,
    friction: torch.Tensor,
    t: torch.Tensor,
    t0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return analytic_swr_common.closed_form_constant_deceleration(
        x0=x0,
        v0=v0,
        deceleration=friction * x0.new_tensor(GRAVITY),
        t=t,
        t0=t0,
    )


def _pair_contact_from_motion(
    *,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    i: int,
    j: int,
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    shape_i = int(shape_ids[i])
    shape_j = int(shape_ids[j])
    relative_velocity = velocities[..., i, :] - velocities[..., j, :]
    rel = positions[..., i, :] - positions[..., j, :]
    distance = torch.sqrt(torch.sum(rel * rel, dim=-1) + 1e-12)
    normal = rel / distance.unsqueeze(-1)
    contact_distance = contact_radii[i] + contact_radii[j]
    contact_model = (
        "circle_circle"
        if shape_i == SHAPE_CIRCLE and shape_j == SHAPE_CIRCLE
        else "circle_circle_proxy"
    )
    return distance - contact_distance, normal, relative_velocity, contact_model


def _pair_contact_value(
    *,
    start_x: torch.Tensor,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    friction: torch.Tensor,
    t: torch.Tensor,
    i: int,
    j: int,
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    positions, velocities = _closed_form_motion(start_x, start_v, friction, t, start_time)
    residual, normal, relative_velocity, contact_model = _pair_contact_from_motion(
        positions=positions,
        velocities=velocities,
        i=i,
        j=j,
        contact_radii=contact_radii,
        shape_ids=shape_ids,
    )
    return residual, normal, positions, velocities, relative_velocity, contact_model


def _solve_event_time(
    *,
    start_x: torch.Tensor,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    friction: torch.Tensor,
    event_time_center: torch.Tensor,
    window_seconds: torch.Tensor,
    i: int,
    j: int,
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
    newton_iters: int,
    lower_bound: torch.Tensor | None = None,
    upper_bound: torch.Tensor | None = None,
) -> torch.Tensor:
    pair_start_time = torch.maximum(start_time[i], start_time[j]) if start_time.ndim > 0 else start_time
    lower = torch.maximum(pair_start_time, event_time_center - window_seconds)
    if lower_bound is not None:
        lower = torch.maximum(lower, lower_bound)
    upper = event_time_center + window_seconds
    if upper_bound is not None:
        upper = torch.minimum(upper, upper_bound)
    upper = torch.maximum(lower + event_time_center.new_tensor(1e-6), upper)
    t = torch.clamp(event_time_center, min=float(lower.detach().cpu()), max=float(upper.detach().cpu()))
    for _ in range(max(int(newton_iters), 0)):
        residual, normal, _positions, _velocities, relative_velocity, _contact_model = _pair_contact_value(
            start_x=start_x,
            start_v=start_v,
            start_time=start_time,
            friction=friction,
            t=t,
            i=i,
            j=j,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
        )
        derivative = torch.sum(normal * relative_velocity)
        valid_derivative = torch.abs(derivative) > 1e-8
        safe_derivative = torch.where(valid_derivative, derivative, torch.ones_like(derivative))
        step = torch.where(valid_derivative, residual / safe_derivative, torch.zeros_like(residual))
        t = torch.minimum(torch.maximum(t - step, lower), upper)
    return t


def _object_motion_quadratic_coefficients_np(
    *,
    start_x: torch.Tensor,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    friction: torch.Tensor,
    object_index: int,
    segment_start: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.asarray(start_x.detach().cpu()[object_index].tolist(), dtype=np.float64)
    v0 = np.asarray(start_v.detach().cpu()[object_index].tolist(), dtype=np.float64)
    t0 = float(start_time.detach().cpu()[object_index])
    acceleration = float(friction.detach().cpu()[object_index]) * GRAVITY
    return analytic_swr_common.constant_deceleration_quadratic_coefficients(
        start_position=x0,
        start_velocity=v0,
        start_time=t0,
        deceleration=acceleration,
        segment_start=segment_start,
    )


def _object_stop_time_np(
    *,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    friction: torch.Tensor,
    object_index: int,
) -> float:
    v0 = np.asarray(start_v.detach().cpu()[object_index].tolist(), dtype=np.float64)
    speed = float(np.linalg.norm(v0))
    acceleration = float(friction.detach().cpu()[object_index]) * GRAVITY
    return analytic_swr_common.constant_deceleration_stop_time(
        start_speed=speed,
        start_time=float(start_time.detach().cpu()[object_index]),
        deceleration=acceleration,
    )


def _pair_contact_polynomial_roots_np(
    *,
    start_x: torch.Tensor,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    friction: torch.Tensor,
    i: int,
    j: int,
    radius_sum: float,
    segment_start: float,
    segment_end: float,
) -> list[float]:
    coeff_i = _object_motion_quadratic_coefficients_np(
        start_x=start_x,
        start_v=start_v,
        start_time=start_time,
        friction=friction,
        object_index=i,
        segment_start=segment_start,
    )
    coeff_j = _object_motion_quadratic_coefficients_np(
        start_x=start_x,
        start_v=start_v,
        start_time=start_time,
        friction=friction,
        object_index=j,
        segment_start=segment_start,
    )
    c0 = coeff_i[0] - coeff_j[0]
    c1 = coeff_i[1] - coeff_j[1]
    c2 = coeff_i[2] - coeff_j[2]
    polynomial = np.asarray(
        [
            float(np.dot(c2, c2)),
            float(2.0 * np.dot(c1, c2)),
            float(np.dot(c1, c1) + 2.0 * np.dot(c0, c2)),
            float(2.0 * np.dot(c0, c1)),
            float(np.dot(c0, c0) - float(radius_sum) * float(radius_sum)),
        ],
        dtype=np.float64,
    )
    first_nonzero = 0
    while first_nonzero < len(polynomial) - 1 and abs(float(polynomial[first_nonzero])) < 1e-12:
        first_nonzero += 1
    trimmed = polynomial[first_nonzero:]
    if len(trimmed) <= 1:
        return []
    roots = np.roots(trimmed)
    output = []
    lower = float(segment_start) + 1e-8
    upper = float(segment_end) + 1e-9
    for root in roots:
        if abs(float(np.imag(root))) > 1e-7:
            continue
        value = float(np.real(root))
        if lower <= value <= upper and math.isfinite(value):
            rel_position = c0 + c1 * value + c2 * value * value
            distance = float(np.linalg.norm(rel_position))
            if distance <= 1e-12:
                continue
            rel_velocity = c1 + 2.0 * c2 * value
            relative_normal_velocity = float(np.dot(rel_position / distance, rel_velocity))
            if relative_normal_velocity < -1e-7:
                output.append(value)
    return sorted(set(round(value, 12) for value in output))


def _find_free_rollout_pair_candidate(
    *,
    start_x: torch.Tensor,
    start_v: torch.Tensor,
    start_time: torch.Tensor,
    frame_times: torch.Tensor,
    friction: torch.Tensor,
    object_pairs: list[tuple[int, int]],
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
    active_end_times: torch.Tensor,
    event_search_end_times: torch.Tensor | None,
) -> dict[str, Any] | None:
    final_time = frame_times[-1]
    final_value = float(final_time.detach().cpu())
    best: dict[str, Any] | None = None
    for pair_index, (i, j) in enumerate(object_pairs):
        pair_start = torch.maximum(start_time[i], start_time[j])
        pair_end = torch.minimum(final_time, torch.minimum(active_end_times[i], active_end_times[j]))
        if event_search_end_times is not None:
            pair_end = torch.minimum(pair_end, torch.minimum(event_search_end_times[i], event_search_end_times[j]))
        start_value = float(pair_start.detach().cpu())
        pair_end_value = float(pair_end.detach().cpu())
        if pair_end_value <= start_value + 1e-6 or final_value <= start_value + 1e-6:
            continue
        stop_times = [
            _object_stop_time_np(start_v=start_v, start_time=start_time, friction=friction, object_index=i),
            _object_stop_time_np(start_v=start_v, start_time=start_time, friction=friction, object_index=j),
        ]
        split_values = [start_value + 1e-8, pair_end_value]
        split_values.extend(value for value in stop_times if start_value + 1e-8 < value < pair_end_value)
        split_values = sorted(set(round(float(value), 12) for value in split_values))
        radius_sum = float((contact_radii[i] + contact_radii[j]).detach().cpu())
        contact_model = (
            "circle_circle"
            if int(shape_ids[i]) == SHAPE_CIRCLE and int(shape_ids[j]) == SHAPE_CIRCLE
            else "circle_circle_proxy"
        )
        roots: list[tuple[float, float, float]] = []
        for segment_start, segment_end in zip(split_values[:-1], split_values[1:]):
            if segment_end <= segment_start + 1e-8:
                continue
            segment_roots = _pair_contact_polynomial_roots_np(
                start_x=start_x,
                start_v=start_v,
                start_time=start_time,
                friction=friction,
                i=i,
                j=j,
                radius_sum=radius_sum,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            roots.extend((root, segment_start, segment_end) for root in segment_roots)
        if not roots:
            continue
        event_time_value, segment_start, segment_end = min(roots, key=lambda item: item[0])
        candidate_time = start_x.new_tensor(event_time_value)
        candidate_time_value = float(candidate_time.detach().cpu())
        candidate = {
            "pair_index": int(pair_index),
            "object_indices": [int(i), int(j)],
            "event_center": candidate_time,
            "event_lower_bound": start_x.new_tensor(segment_start),
            "event_upper_bound": start_x.new_tensor(segment_end),
            "contact_model": contact_model,
            "coarse_abs_contact_residual_m": None,
            "coarse_relative_normal_velocity_m_per_s": None,
        }
        if best is None:
            best = candidate
            continue
        best_time_value = float(best["event_center"].detach().cpu())
        if candidate_time_value < best_time_value:
            best = candidate
    return best


def _collision_update(
    *,
    velocities: torch.Tensor,
    i: int,
    j: int,
    normal: torch.Tensor,
    mass_ratio_i_over_j: torch.Tensor,
    restitution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return analytic_swr_common.sphere_sphere_impulse_update_torch(
        velocities=velocities,
        i=i,
        j=j,
        normal_from_j_to_i=normal,
        mass_i=mass_ratio_i_over_j,
        mass_j=torch.ones_like(mass_ratio_i_over_j),
        restitution=restitution,
    )


def _analytic_rollout(
    *,
    target: torch.Tensor,
    frames: list[int],
    event_specs: list[dict[str, Any]],
    pair_index_by_key: dict[tuple[int, int], int],
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
    v0: torch.Tensor,
    friction: torch.Tensor,
    pair_mass_ratio: torch.Tensor,
    pair_restitution: torch.Tensor,
    fps: float,
    event_source: str,
    event_window_frames: int,
    newton_iters: int,
    object_pairs: list[tuple[int, int]] | None = None,
    max_free_rollout_events: int = 0,
    active_metadata: dict[str, Any] | None = None,
    event_search_end_times: torch.Tensor | None = None,
    rollout_frame_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    frame0 = int(frames[0])
    effective_frame_count = len(frames)
    if rollout_frame_count is not None:
        effective_frame_count = max(min(int(rollout_frame_count), len(frames)), 1)
    rollout_frames = frames[:effective_frame_count]
    frame_times = torch.tensor(
        [(float(frame) - float(frame0)) / max(float(fps), 1e-12) for frame in rollout_frames],
        dtype=torch.float64,
        device=target.device,
    )
    object_count = int(target.shape[1])
    if active_metadata is None:
        start_time = torch.full((object_count,), float(frame_times[0].detach().cpu()), dtype=target.dtype, device=target.device)
        active_end_times = torch.full((object_count,), float(frame_times[-1].detach().cpu()), dtype=target.dtype, device=target.device)
        start_x = target[0].clone()
    else:
        start_time = active_metadata["active_start_times"].clone()
        active_end_times = active_metadata["active_end_times"].clone()
        start_x = active_metadata["first_xy"].clone()
    start_v = v0
    predicted, predicted_velocity = _closed_form_motion(
        start_x,
        start_v,
        friction,
        frame_times,
        start_time,
    )
    event_records: list[dict[str, Any]] = []

    def apply_pair_event(
        *,
        event_index: int,
        object_indices: list[int],
        event_center: torch.Tensor,
        window_seconds: torch.Tensor,
        source: str,
        source_frame: int | None,
        event_lower_bound: torch.Tensor | None = None,
        event_upper_bound: torch.Tensor | None = None,
        coarse_abs_contact_residual_m: torch.Tensor | None = None,
        coarse_relative_normal_velocity_m_per_s: torch.Tensor | None = None,
    ) -> None:
        nonlocal predicted, predicted_velocity, start_time, start_x, start_v
        i, j = object_indices
        lower_bound = torch.maximum(start_time[i], start_time[j])
        upper_bound = torch.minimum(active_end_times[i], active_end_times[j])
        if not bool((upper_bound > lower_bound + target.new_tensor(1e-6)).detach().cpu()):
            return
        event_time = _solve_event_time(
            start_x=start_x,
            start_v=start_v,
            start_time=start_time,
            friction=friction,
            event_time_center=event_center,
            window_seconds=window_seconds,
            i=i,
            j=j,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
            newton_iters=newton_iters,
            lower_bound=event_lower_bound if event_lower_bound is not None else lower_bound,
            upper_bound=event_upper_bound if event_upper_bound is not None else upper_bound,
        )
        contact_residual, normal, event_positions, pre_event_velocities, relative_velocity, contact_model = _pair_contact_value(
            start_x=start_x,
            start_v=start_v,
            start_time=start_time,
            friction=friction,
            t=event_time,
            i=i,
            j=j,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
        )
        pair_key = tuple(sorted((i, j)))
        pair_index = pair_index_by_key[pair_key]
        event_mass_ratio = pair_mass_ratio[pair_index] if (i, j) == pair_key else 1.0 / pair_mass_ratio[pair_index]
        post_event_velocities, impulse_vector, relative_normal_velocity = _collision_update(
            velocities=pre_event_velocities,
            i=i,
            j=j,
            normal=normal,
            mass_ratio_i_over_j=event_mass_ratio,
            restitution=pair_restitution[pair_index],
        )
        next_start_time = start_time.clone()
        next_start_x = start_x.clone()
        next_start_v = start_v.clone()
        next_start_time[i] = event_time
        next_start_time[j] = event_time
        next_start_x[i] = event_positions[i]
        next_start_x[j] = event_positions[j]
        next_start_v[i] = post_event_velocities[i]
        next_start_v[j] = post_event_velocities[j]
        post_positions, post_velocities = _closed_form_motion(
            next_start_x,
            next_start_v,
            friction,
            frame_times,
            next_start_time,
        )
        object_mask = torch.zeros((object_count,), dtype=torch.bool, device=target.device)
        object_mask[i] = True
        object_mask[j] = True
        after_mask = (frame_times >= event_time).reshape(-1, 1, 1) & object_mask.reshape(1, -1, 1)
        predicted = torch.where(after_mask, post_positions, predicted)
        predicted_velocity = torch.where(after_mask, post_velocities, predicted_velocity)
        event_records.append(
            {
                "event_index": torch.tensor(int(event_index), dtype=torch.int64, device=target.device),
                "event_type": "pair_collision_candidate",
                "event_source": source,
                "source_frame": source_frame,
                "contact_model": contact_model,
                "object_indices": [int(i), int(j)],
                "event_time_s": event_time,
                "event_frame_float": event_time * target.new_tensor(float(fps)) + target.new_tensor(float(frame0)),
                "contact_residual_m": contact_residual,
                "normal": normal,
                "pre_velocity": pre_event_velocities,
                "post_velocity": post_event_velocities,
                "relative_velocity": relative_velocity,
                "relative_normal_velocity": relative_normal_velocity,
                "impulse": impulse_vector,
                "coarse_abs_contact_residual_m": coarse_abs_contact_residual_m,
                "coarse_relative_normal_velocity_m_per_s": coarse_relative_normal_velocity_m_per_s,
            }
        )
        start_time = next_start_time
        start_x = next_start_x
        start_v = next_start_v

    if event_source == EVENT_SOURCE_DETECTED_WINDOW:
        window_seconds = target.new_tensor(float(event_window_frames) / max(float(fps), 1e-12))
        sorted_events = sorted(enumerate(event_specs), key=lambda item: (int(item[1]["frame"]), int(item[0])))
        for event_index, event in sorted_events:
            object_indices = [int(index) for index in event["object_indices"]]
            if len(object_indices) != 2 or str(event.get("event_type")) != "pair_collision_candidate":
                raise ValueError("analytic-event-collision-v1 supports pair_collision_candidate events only")
            event_center = target.new_tensor((float(event["frame"]) - float(frame0)) / max(float(fps), 1e-12))
            if bool((event_center > frame_times[-1] + target.new_tensor(1e-9)).detach().cpu()):
                continue
            apply_pair_event(
                event_index=event_index,
                object_indices=object_indices,
                event_center=event_center,
                window_seconds=window_seconds,
                source=EVENT_SOURCE_DETECTED_WINDOW,
                source_frame=int(event["frame"]),
            )
    elif event_source == EVENT_SOURCE_FREE_ROLLOUT:
        if object_pairs is None:
            raise ValueError(f"{event_source} requires object_pairs")
        event_cap = int(max_free_rollout_events) if int(max_free_rollout_events) > 0 else len(object_pairs)
        for event_index in range(max(event_cap, 0)):
            candidate = _find_free_rollout_pair_candidate(
                start_x=start_x,
                start_v=start_v,
                start_time=start_time,
                frame_times=frame_times,
                friction=friction,
                object_pairs=object_pairs,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                active_end_times=active_end_times,
                event_search_end_times=event_search_end_times,
            )
            if candidate is None:
                break
            candidate_window_seconds = torch.maximum(
                candidate["event_center"] - candidate["event_lower_bound"],
                candidate["event_upper_bound"] - candidate["event_center"],
            ) + target.new_tensor(1e-9)
            apply_pair_event(
                event_index=event_index,
                object_indices=candidate["object_indices"],
                event_center=candidate["event_center"],
                window_seconds=candidate_window_seconds,
                source=event_source,
                source_frame=None,
                event_lower_bound=candidate.get("event_lower_bound"),
                event_upper_bound=candidate.get("event_upper_bound"),
                coarse_abs_contact_residual_m=candidate["coarse_abs_contact_residual_m"],
                coarse_relative_normal_velocity_m_per_s=candidate["coarse_relative_normal_velocity_m_per_s"],
            )
    else:
        raise ValueError(f"unsupported event_source: {event_source}")
    return predicted, predicted_velocity, event_records


def _pair_mass_ratio_from_raw(pair_mass_ratio_raw: torch.Tensor) -> torch.Tensor:
    return MIN_PAIR_MASS_RATIO + (MAX_PAIR_MASS_RATIO - MIN_PAIR_MASS_RATIO) * torch.sigmoid(pair_mass_ratio_raw)


def _weighted_matrix_and_vector(
    matrix: np.ndarray,
    vector: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_weights = np.where(np.isfinite(weights), weights, 0.0)
    sqrt_weights = np.sqrt(np.clip(finite_weights, 0.0, np.inf))
    return sqrt_weights.reshape(-1, 1) * matrix, sqrt_weights * vector


def _rank_and_nullspace(matrix: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    variable_count = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    if variable_count <= 0:
        return 0, np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if matrix.shape[0] == 0:
        return 0, np.eye(variable_count, dtype=np.float64), np.zeros((0,), dtype=np.float64)
    _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size:
        threshold = OBJECT_SOLVE_RANK_TOLERANCE * max(matrix.shape) * float(singular_values[0])
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0
    return rank, vt[rank:].T, singular_values


def _bounded_weighted_lstsq(
    *,
    matrix: np.ndarray,
    vector: np.ndarray,
    weights: np.ndarray,
    lower_bound: float | None,
    upper_bound: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    weighted_matrix, weighted_vector = _weighted_matrix_and_vector(matrix, vector, weights)
    variable_count = int(matrix.shape[1])
    bounds_enabled = lower_bound is not None or upper_bound is not None
    if not bounds_enabled:
        solution, *_ = np.linalg.lstsq(
            weighted_matrix,
            weighted_vector,
            rcond=OBJECT_SOLVE_RANK_TOLERANCE,
        )
        return solution, {
            "solver": "numpy_lstsq",
            "bounded": False,
        }

    lower = -np.inf if lower_bound is None else float(lower_bound)
    upper = np.inf if upper_bound is None else float(upper_bound)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"A NumPy version .* is required for this version of SciPy.*",
                category=UserWarning,
            )
            from scipy.optimize import lsq_linear  # type: ignore

        result = lsq_linear(
            weighted_matrix,
            weighted_vector,
            bounds=(
                np.full((variable_count,), lower, dtype=np.float64),
                np.full((variable_count,), upper, dtype=np.float64),
            ),
            tol=1e-12,
            lsmr_tol="auto",
            max_iter=1000,
        )
        if np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=np.float64), {
                "solver": "scipy_lsq_linear",
                "bounded": True,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "iteration_count": int(result.nit),
            }
    except Exception as exc:  # pragma: no cover - used only when SciPy is unavailable.
        fallback_error = repr(exc)
    else:
        fallback_error = "scipy_lsq_linear_returned_non_finite_solution"

    solution, *_ = np.linalg.lstsq(
        weighted_matrix,
        weighted_vector,
        rcond=OBJECT_SOLVE_RANK_TOLERANCE,
    )
    clipped = np.clip(solution, lower, upper)
    return clipped, {
        "solver": "numpy_lstsq_clipped_fallback",
        "bounded": True,
        "fallback_error": fallback_error,
    }


def _weighted_linear_solution_with_prior(
    *,
    rows: list[np.ndarray],
    rhs: list[float],
    weights: list[float],
    object_count: int,
    prior: np.ndarray | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    nullspace_prior_weight: float = OBJECT_SOLVE_NULLSPACE_PRIOR_WEIGHT,
    explicit_prior_rows: list[np.ndarray] | None = None,
    explicit_prior_rhs: list[float] | None = None,
    explicit_prior_weights: list[float] | None = None,
    explicit_prior_types: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if object_count <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "variable_count": 0,
            "equation_count": 0,
            "pair_equation_count": 0,
            "prior_equation_count": 0,
            "explicit_prior_equation_count": 0,
            "nullspace_prior_equation_count": 0,
            "rank": 0,
            "post_explicit_prior_rank": 0,
            "post_explicit_prior_nullity": 0,
            "augmented_rank": 0,
            "nullity": 0,
            "weighted_residual_norm": 0.0,
            "weighted_residual_sse": 0.0,
            "pair_weighted_residual_norm": 0.0,
            "pair_weighted_residual_sse": 0.0,
            "solve_strategy": "empty",
        }
    prior_vector = (
        np.zeros((object_count,), dtype=np.float64)
        if prior is None
        else np.asarray(prior, dtype=np.float64).reshape(object_count)
    )
    if rows:
        matrix = np.vstack(rows).astype(np.float64)
    else:
        matrix = np.zeros((0, object_count), dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64)
    weighted_matrix, weighted_vector = _weighted_matrix_and_vector(matrix, vector, weight_vector)
    rank, _null_basis, singular_values = _rank_and_nullspace(weighted_matrix)
    nullity = max(int(object_count) - rank, 0)
    prior_rows: list[np.ndarray] = []
    prior_rhs: list[float] = []
    prior_weights: list[float] = []
    prior_equations: list[dict[str, Any]] = []
    for prior_index, row in enumerate(explicit_prior_rows or []):
        typed_row = np.asarray(row, dtype=np.float64).reshape(object_count)
        prior_rows.append(typed_row)
        target = 0.0 if explicit_prior_rhs is None else float(explicit_prior_rhs[prior_index])
        weight = 1.0 if explicit_prior_weights is None else float(explicit_prior_weights[prior_index])
        prior_rhs.append(target)
        prior_weights.append(weight)
        prior_type = "explicit_prior"
        if explicit_prior_types is not None and prior_index < len(explicit_prior_types):
            prior_type = str(explicit_prior_types[prior_index])
        prior_equations.append(
            {
                "type": prior_type,
                "coefficients": [float(value) for value in typed_row.tolist()],
                "rhs": target,
                "weight": weight,
            }
        )

    if prior_rows:
        rank_context_matrix = np.vstack([matrix, np.vstack(prior_rows).astype(np.float64)])
        rank_context_vector = np.concatenate([vector, np.asarray(prior_rhs, dtype=np.float64)])
        rank_context_weights = np.concatenate([weight_vector, np.asarray(prior_weights, dtype=np.float64)])
    else:
        rank_context_matrix = matrix
        rank_context_vector = vector
        rank_context_weights = weight_vector
    weighted_rank_context_matrix, _weighted_rank_context_vector = _weighted_matrix_and_vector(
        rank_context_matrix,
        rank_context_vector,
        rank_context_weights,
    )
    post_explicit_prior_rank, post_explicit_null_basis, _post_explicit_singular_values = _rank_and_nullspace(
        weighted_rank_context_matrix
    )
    post_explicit_prior_nullity = max(int(object_count) - post_explicit_prior_rank, 0)
    explicit_prior_count = len(prior_rows)
    if post_explicit_prior_nullity > 0:
        for basis_index in range(post_explicit_prior_nullity):
            nullspace_prior_index = basis_index + explicit_prior_count
            row = np.asarray(post_explicit_null_basis[:, basis_index], dtype=np.float64)
            row_norm = float(np.linalg.norm(row))
            if row_norm <= 1e-12 or not math.isfinite(row_norm):
                continue
            row = row / row_norm
            target = float(np.dot(row, prior_vector))
            prior_rows.append(row)
            prior_rhs.append(target)
            prior_weights.append(float(nullspace_prior_weight))
            prior_equations.append(
                {
                    "type": "nullspace_prior",
                    "index": int(nullspace_prior_index),
                    "coefficients": [float(value) for value in row.tolist()],
                    "rhs": target,
                    "weight": float(nullspace_prior_weight),
                }
            )
    if prior_rows:
        augmented_matrix = np.vstack([matrix, np.vstack(prior_rows).astype(np.float64)])
        augmented_vector = np.concatenate([vector, np.asarray(prior_rhs, dtype=np.float64)])
        augmented_weights = np.concatenate([weight_vector, np.asarray(prior_weights, dtype=np.float64)])
    else:
        augmented_matrix = matrix
        augmented_vector = vector
        augmented_weights = weight_vector
    solution, solver_diagnostics = _bounded_weighted_lstsq(
        matrix=augmented_matrix,
        vector=augmented_vector,
        weights=augmented_weights,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    augmented_weighted_matrix, augmented_weighted_vector = _weighted_matrix_and_vector(
        augmented_matrix,
        augmented_vector,
        augmented_weights,
    )
    augmented_rank, _augmented_null_basis, augmented_singular_values = _rank_and_nullspace(augmented_weighted_matrix)
    pair_residual = weighted_matrix @ solution - weighted_vector
    augmented_residual = augmented_weighted_matrix @ solution - augmented_weighted_vector
    pair_residual_sse = float(np.sum(pair_residual * pair_residual))
    augmented_residual_sse = float(np.sum(augmented_residual * augmented_residual))
    lower_active_indices: list[int] = []
    upper_active_indices: list[int] = []
    bound_tolerance = 1e-9
    if lower_bound is not None:
        lower_active_indices = [
            int(index)
            for index, value in enumerate(solution.tolist())
            if float(value) <= float(lower_bound) + bound_tolerance
        ]
    if upper_bound is not None:
        upper_active_indices = [
            int(index)
            for index, value in enumerate(solution.tolist())
            if float(value) >= float(upper_bound) - bound_tolerance
        ]
    return solution, {
        "variable_count": int(object_count),
        "equation_count": int(augmented_matrix.shape[0]),
        "pair_equation_count": int(matrix.shape[0]),
        "prior_equation_count": int(len(prior_rows)),
        "explicit_prior_equation_count": int(explicit_prior_count),
        "nullspace_prior_equation_count": int(len(prior_rows) - explicit_prior_count),
        "rank": rank,
        "post_explicit_prior_rank": int(post_explicit_prior_rank),
        "post_explicit_prior_nullity": int(post_explicit_prior_nullity),
        "augmented_rank": int(augmented_rank),
        "nullity": int(nullity),
        "weighted_residual_norm": float(np.linalg.norm(augmented_residual)),
        "weighted_residual_sse": augmented_residual_sse,
        "pair_weighted_residual_norm": float(np.linalg.norm(pair_residual)),
        "pair_weighted_residual_sse": pair_residual_sse,
        "has_pair_residual": bool(pair_residual_sse > 1e-12),
        "has_augmented_residual": bool(augmented_residual_sse > 1e-12),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "augmented_singular_values": [float(value) for value in augmented_singular_values.tolist()],
        "is_underdetermined": bool(nullity > 0),
        "is_overdetermined": bool(matrix.shape[0] > rank),
        "prior_applied": bool(prior_rows),
        "nullspace_prior_applied": bool(len(prior_rows) - explicit_prior_count > 0),
        "nullspace_prior_weight": float(nullspace_prior_weight),
        "prior_equations": prior_equations,
        "lower_bound": None if lower_bound is None else float(lower_bound),
        "upper_bound": None if upper_bound is None else float(upper_bound),
        "lower_bound_active_indices": lower_active_indices,
        "upper_bound_active_indices": upper_active_indices,
        "solver": solver_diagnostics,
        "solve_strategy": (
            "bounded_weighted_lstsq_with_nullspace_priors"
            if solver_diagnostics.get("bounded")
            else "weighted_lstsq_with_nullspace_priors"
        ),
    }


def _rank_with_prior_rows(
    *,
    rows: list[np.ndarray],
    weights: list[float],
    object_count: int,
    prior_rows: list[np.ndarray],
    prior_weights: list[float],
) -> int:
    if rows:
        matrix = np.vstack(rows).astype(np.float64)
    else:
        matrix = np.zeros((0, object_count), dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64)
    if prior_rows:
        matrix = np.vstack([matrix, np.vstack(prior_rows).astype(np.float64)])
        weight_vector = np.concatenate([weight_vector, np.asarray(prior_weights, dtype=np.float64)])
    weighted_matrix, _weighted_vector = _weighted_matrix_and_vector(
        matrix,
        np.zeros((matrix.shape[0],), dtype=np.float64),
        weight_vector,
    )
    rank, _null_basis, _singular_values = _rank_and_nullspace(weighted_matrix)
    return int(rank)


def _restitution_equality_prior_candidates(
    object_count: int,
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed_candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for observation in observations:
        i = int(observation["i"])
        j = int(observation["j"])
        if i == j or i < 0 or j < 0 or i >= object_count or j >= object_count:
            continue
        pair = tuple(sorted((i, j)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        row = np.zeros((object_count,), dtype=np.float64)
        row[i] = 1.0
        row[j] = -1.0
        observed_candidates.append(
            {
                "row": row,
                "type": f"restitution_equal_log_objects_{i}_{j}_observed_pair",
                "object_indices": [int(i), int(j)],
                "source": "observed_pair",
            }
        )

    fallback_candidates: list[dict[str, Any]] = []
    for i in range(object_count):
        for j in range(i + 1, object_count):
            pair = (i, j)
            if pair in seen_pairs:
                continue
            row = np.zeros((object_count,), dtype=np.float64)
            row[i] = 1.0
            row[j] = -1.0
            fallback_candidates.append(
                {
                    "row": row,
                    "type": f"restitution_equal_log_objects_{i}_{j}_fallback_pair",
                    "object_indices": [int(i), int(j)],
                    "source": "fallback_pair",
                }
            )
    return observed_candidates, fallback_candidates


def _select_restitution_equality_priors(
    *,
    rows: list[np.ndarray],
    rhs: list[float],
    weights: list[float],
    object_count: int,
    observations: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[float], list[float], list[str], dict[str, Any]]:
    base_rank = _rank_with_prior_rows(
        rows=rows,
        weights=weights,
        object_count=object_count,
        prior_rows=[],
        prior_weights=[],
    )
    base_nullity = max(int(object_count) - int(base_rank), 0)
    observed_candidates, fallback_candidates = _restitution_equality_prior_candidates(object_count, observations)
    selected: list[dict[str, Any]] = []
    chosen_steps: list[dict[str, Any]] = []

    def selected_rows() -> list[np.ndarray]:
        return [np.asarray(item["row"], dtype=np.float64) for item in selected]

    def selected_weights() -> list[float]:
        return [OBJECT_SOLVE_RESTITUTION_EQUALITY_PRIOR_WEIGHT for _item in selected]

    def selected_types() -> list[str]:
        return [str(item["type"]) for item in selected]

    def try_candidate_pool(candidates: list[dict[str, Any]]) -> None:
        remaining = list(candidates)
        while len(selected) < base_nullity:
            current_rank = _rank_with_prior_rows(
                rows=rows,
                weights=weights,
                object_count=object_count,
                prior_rows=selected_rows(),
                prior_weights=selected_weights(),
            )
            if current_rank >= object_count:
                break
            best_index: int | None = None
            best_score: tuple[float, float, int] | None = None
            best_diagnostics: dict[str, Any] | None = None
            for candidate_index, candidate in enumerate(remaining):
                test_rows = [*selected_rows(), np.asarray(candidate["row"], dtype=np.float64)]
                test_weights = [*selected_weights(), OBJECT_SOLVE_RESTITUTION_EQUALITY_PRIOR_WEIGHT]
                new_rank = _rank_with_prior_rows(
                    rows=rows,
                    weights=weights,
                    object_count=object_count,
                    prior_rows=test_rows,
                    prior_weights=test_weights,
                )
                if new_rank <= current_rank:
                    continue
                _solution, diagnostics = _weighted_linear_solution_with_prior(
                    rows=rows,
                    rhs=rhs,
                    weights=weights,
                    object_count=object_count,
                    lower_bound=float(np.log(1e-6)),
                    upper_bound=0.0,
                    explicit_prior_rows=test_rows,
                    explicit_prior_rhs=[0.0 for _row in test_rows],
                    explicit_prior_weights=test_weights,
                    explicit_prior_types=[*selected_types(), str(candidate["type"])],
                )
                score = (
                    float(diagnostics["pair_weighted_residual_sse"]),
                    float(diagnostics["weighted_residual_sse"]),
                    int(diagnostics["post_explicit_prior_nullity"]),
                )
                if best_score is None or score < best_score:
                    best_index = candidate_index
                    best_score = score
                    best_diagnostics = diagnostics
            if best_index is None:
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            chosen_steps.append(
                {
                    "type": str(chosen["type"]),
                    "object_indices": [int(value) for value in chosen["object_indices"]],
                    "source": str(chosen["source"]),
                    "pair_weighted_residual_sse_after_selection": (
                        None
                        if best_diagnostics is None
                        else float(best_diagnostics["pair_weighted_residual_sse"])
                    ),
                    "post_explicit_prior_nullity_after_selection": (
                        None
                        if best_diagnostics is None
                        else int(best_diagnostics["post_explicit_prior_nullity"])
                    ),
                }
            )

    try_candidate_pool(observed_candidates)
    if len(selected) < base_nullity:
        try_candidate_pool(fallback_candidates)

    final_rank = _rank_with_prior_rows(
        rows=rows,
        weights=weights,
        object_count=object_count,
        prior_rows=selected_rows(),
        prior_weights=selected_weights(),
    )
    diagnostics = {
        "strategy": "rank_increasing_equal_restitution_priors_by_pair_residual",
        "base_rank": int(base_rank),
        "base_nullity": int(base_nullity),
        "observed_candidate_count": int(len(observed_candidates)),
        "fallback_candidate_count": int(len(fallback_candidates)),
        "selected_prior_count": int(len(selected)),
        "selected_observed_prior_count": int(
            sum(1 for item in selected if str(item["source"]) == "observed_pair")
        ),
        "selected_fallback_prior_count": int(
            sum(1 for item in selected if str(item["source"]) == "fallback_pair")
        ),
        "final_rank_with_selected_priors": int(final_rank),
        "remaining_nullity_after_selected_priors": int(max(object_count - final_rank, 0)),
        "selected_priors": chosen_steps,
    }
    return (
        selected_rows(),
        [0.0 for _item in selected],
        selected_weights(),
        selected_types(),
        diagnostics,
    )


def _solve_log_masses(
    object_count: int,
    observations: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if object_count <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "variable_count": 0,
            "equation_count": 0,
            "rank": 0,
            "nullity": 0,
            "solve_strategy": "empty",
        }
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    weights: list[float] = []
    for observation in observations:
        i = int(observation["i"])
        j = int(observation["j"])
        ratio = float(np.clip(observation["mass_ratio_i_over_j"], MIN_PAIR_MASS_RATIO, MAX_PAIR_MASS_RATIO))
        row = np.zeros((object_count,), dtype=np.float64)
        row[i] = 1.0
        row[j] = -1.0
        rows.append(row)
        rhs.append(float(np.log(ratio)))
        weights.append(float(observation.get("weight", 1.0)))
    mass_gauge = np.zeros((object_count,), dtype=np.float64)
    mass_gauge[0] = 1.0
    return _weighted_linear_solution_with_prior(
        rows=rows,
        rhs=rhs,
        weights=weights,
        object_count=object_count,
        lower_bound=float(np.log(MIN_MASS)),
        upper_bound=float(np.log(MAX_MASS)),
        explicit_prior_rows=[mass_gauge],
        explicit_prior_rhs=[0.0],
        explicit_prior_weights=[1.0],
        explicit_prior_types=["mass_gauge_log_object_0_to_zero"],
    )


def _solve_log_restitutions(
    object_count: int,
    observations: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if object_count <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "variable_count": 0,
            "equation_count": 0,
            "rank": 0,
            "nullity": 0,
            "solve_strategy": "empty",
        }
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    weights: list[float] = []
    for observation in observations:
        i = int(observation["i"])
        j = int(observation["j"])
        restitution = float(np.clip(observation["pair_restitution"], 1e-6, 0.999999))
        row = np.zeros((object_count,), dtype=np.float64)
        row[i] = 1.0
        row[j] = 1.0
        rows.append(row)
        rhs.append(float(np.log(restitution)))
        weights.append(float(observation.get("weight", 1.0)))
    prior_rows, prior_rhs, prior_weights, prior_types, prior_selection_diagnostics = _select_restitution_equality_priors(
        rows=rows,
        rhs=rhs,
        weights=weights,
        object_count=object_count,
        observations=observations,
    )
    solution, diagnostics = _weighted_linear_solution_with_prior(
        rows=rows,
        rhs=rhs,
        weights=weights,
        object_count=object_count,
        lower_bound=float(np.log(1e-6)),
        upper_bound=0.0,
        explicit_prior_rows=prior_rows,
        explicit_prior_rhs=prior_rhs,
        explicit_prior_weights=prior_weights,
        explicit_prior_types=prior_types,
    )
    diagnostics["equality_prior_selection"] = prior_selection_diagnostics
    return solution, diagnostics


def _event_record_loss(
    event_records: list[dict[str, Any]],
    contact_residual_weight: float,
    approach_penalty_weight: float,
) -> torch.Tensor:
    if not event_records:
        return torch.tensor(0.0, dtype=torch.float64)
    device = event_records[0]["contact_residual_m"].device
    contact_terms = [record["contact_residual_m"] * record["contact_residual_m"] for record in event_records]
    approach_terms = [torch.relu(record["relative_normal_velocity"]) ** 2 for record in event_records]
    return (
        torch.tensor(float(contact_residual_weight), dtype=torch.float64, device=device) * torch.stack(contact_terms).mean()
        + torch.tensor(float(approach_penalty_weight), dtype=torch.float64, device=device) * torch.stack(approach_terms).mean()
    )


def _soft_contact_loss(
    *,
    start_x: torch.Tensor,
    start_time: torch.Tensor,
    active_end_times: torch.Tensor,
    v0: torch.Tensor,
    friction: torch.Tensor,
    frame_times: torch.Tensor,
    object_pairs: list[tuple[int, int]],
    contact_radii: torch.Tensor,
    shape_ids: tuple[int, ...],
    weight: float,
    temperature: float,
    sample_count: int,
) -> torch.Tensor:
    if float(weight) <= 0.0 or not object_pairs:
        return torch.zeros((), dtype=torch.float64, device=start_x.device)
    final_time = frame_times[-1]
    if float(final_time.detach().cpu()) <= 0.0:
        return torch.zeros((), dtype=torch.float64, device=start_x.device)
    losses = []
    temp = max(float(temperature), 1e-6)
    for i, j in object_pairs:
        pair_start = torch.maximum(start_time[i], start_time[j])
        pair_end = torch.minimum(final_time, torch.minimum(active_end_times[i], active_end_times[j]))
        start_value = float(pair_start.detach().cpu())
        end_value = float(pair_end.detach().cpu())
        if end_value <= start_value + 1e-6:
            continue
        sample_times = torch.linspace(
            start_value,
            end_value,
            max(int(sample_count), 2),
            dtype=torch.float64,
            device=start_x.device,
        )
        positions, velocities = _closed_form_motion(
            start_x,
            v0,
            friction,
            sample_times,
            start_time,
        )
        residual, normal, relative_velocity, _contact_model = _pair_contact_from_motion(
            positions=positions,
            velocities=velocities,
            i=i,
            j=j,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
        )
        relative_normal_velocity = torch.sum(normal * relative_velocity, dim=1)
        weights = torch.softmax(-torch.abs(residual) / temp, dim=0)
        soft_residual = torch.sum(weights * residual)
        soft_approach = torch.sum(weights * relative_normal_velocity)
        losses.append(soft_residual * soft_residual + torch.relu(soft_approach) ** 2)
    if not losses:
        return torch.zeros((), dtype=torch.float64, device=start_x.device)
    return torch.tensor(float(weight), dtype=torch.float64, device=start_x.device) * torch.stack(losses).mean()


def _build_event_specs(
    events: list[dict[str, Any]],
    object_index_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            **event,
            "object_indices": [object_index_by_id[object_id] for object_id in event["object_ids"]],
        }
        for event in events
    ]


def run(
    input_fit: Path,
    output_dir: Path,
    init_from_impulse_forced_frame_result: Path | None,
    init_from_analytic_result: Path | None,
    steps: int,
    lr: float,
    analytic_optimizer: str,
    event_source: str,
    warmup_steps: int | None,
    warmup_prefix_frames: int | None,
    curriculum_prefix_step_frames: int,
    curriculum_steps_per_prefix: int,
    curriculum_best_metric: str,
    curriculum_fade_in: bool,
    curriculum_early_stop_patience: int,
    event_window_frames: int,
    newton_iters: int,
    contact_residual_weight: float,
    approach_penalty_weight: float,
    soft_contact_weight: float,
    soft_contact_temperature: float,
    soft_contact_samples: int,
    max_free_rollout_events: int,
    render_video: bool | None = None,
) -> dict[str, Any]:
    if event_source not in EVENT_SOURCES:
        raise ValueError(f"unsupported event_source: {event_source}")
    if analytic_optimizer not in ANALYTIC_OPTIMIZERS:
        raise ValueError(f"unsupported analytic optimizer: {analytic_optimizer}")
    if analytic_optimizer in {ANALYTIC_OPTIMIZER_LEAST_SQUARES, ANALYTIC_OPTIMIZER_LBFGS} and int(warmup_steps or 0) > 0:
        raise ValueError(f"{analytic_optimizer} analytic optimizer does not support warmup")
    if analytic_optimizer == ANALYTIC_OPTIMIZER_LEAST_SQUARES and float(soft_contact_weight) > 0.0:
        raise ValueError("least_squares analytic optimizer does not support soft_contact_weight")
    if curriculum_best_metric not in CURRICULUM_BEST_METRICS:
        raise ValueError(f"unsupported curriculum_best_metric: {curriculum_best_metric}")
    curriculum_enabled = int(curriculum_prefix_step_frames) > 0
    effective_curriculum_fade_in = bool(curriculum_fade_in)
    effective_curriculum_early_stop_patience = max(int(curriculum_early_stop_patience), 0)
    effective_curriculum_validation_frames = (
        max(int(curriculum_prefix_step_frames), 0)
        if curriculum_enabled and curriculum_best_metric == CURRICULUM_BEST_NEXT_WINDOW_RMSE
        else 0
    )
    if curriculum_enabled and event_source != EVENT_SOURCE_FREE_ROLLOUT:
        raise ValueError("prefix curriculum is supported only for free_rollout")
    if curriculum_enabled and analytic_optimizer not in {ANALYTIC_OPTIMIZER_ADAM, ANALYTIC_OPTIMIZER_LBFGS}:
        raise ValueError("prefix curriculum is supported only for Adam and L-BFGS")
    if curriculum_enabled and int(warmup_steps or 0) > 0:
        raise ValueError("prefix curriculum does not use optimizer warmup_steps")
    if init_from_impulse_forced_frame_result is not None and init_from_analytic_result is not None:
        raise ValueError("use only one of init_from_impulse_forced_frame_result or init_from_analytic_result")
    render_video_flag = _render_video_enabled(render_video)
    impulse_forced_frame_result: dict[str, Any] | None = None
    analytic_init_result: dict[str, Any] | None = None
    if init_from_impulse_forced_frame_result is not None:
        impulse_forced_frame_result = _load_json(init_from_impulse_forced_frame_result)
        if str(impulse_forced_frame_result.get("model")) not in IMPULSE_STAGE_MODES:
            raise ValueError("init_from_impulse_forced_frame_result must point to an impulse stage result.json")
    if init_from_analytic_result is not None:
        analytic_init_result = _load_json(init_from_analytic_result)
        if str(analytic_init_result.get("model")) != "analytic-event-collision-v1":
            raise ValueError("init_from_analytic_result must point to an analytic-event-collision-v1 result.json")
    fit = _load_json(input_fit)
    object_ids, target_plane = _project_target_to_plane(fit)
    _validate_supported_shapes(fit, object_ids)
    fps = float(fit["trajectory_physics_initialization"]["fps"])
    scales = _object_scales(fit, object_ids, target_plane)
    shape_ids, half_extents, object_angles, contact_radii, shape_names, contact_proxy_names = _shape_metadata(
        fit=fit,
        object_ids=object_ids,
        object_scales_by_id=scales,
    )
    events = _detected_collision_events(
        fit=fit,
        object_ids=object_ids,
        target_plane=target_plane,
    )
    if impulse_forced_frame_result is not None:
        events = impulse_forced_frame_result.get("events", [])
    if event_source == EVENT_SOURCE_DETECTED_WINDOW:
        for event in events:
            if str(event.get("event_type")) != "pair_collision_candidate" or len(event.get("object_ids", [])) != 2:
                raise ValueError("analytic-event-collision-v1 supports pair_collision_candidate events only")
    object_index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    event_specs = _build_event_specs(events, object_index_by_id)
    object_pairs = _object_pairs(len(object_ids))
    pair_index_by_key = {tuple(pair): index for index, pair in enumerate(object_pairs)}
    frames, target, mask = _target_tensor(object_ids, target_plane)
    device = torch.device("cpu")
    target = target.to(device)
    mask = mask.to(device)
    active_metadata = _active_metadata(object_ids, target_plane, frames, target, fps)
    half_extents = half_extents.to(device)
    object_angles = object_angles.to(device)
    contact_radii = contact_radii.to(device)
    if analytic_init_result is not None:
        v0 = torch.tensor(
            [
                analytic_init_result["parameters"][object_id]["initial_velocity_plane_m_per_s"]
                for object_id in object_ids
            ],
            dtype=torch.float64,
            device=device,
        )
    elif impulse_forced_frame_result is not None:
        v0 = torch.tensor(
            [
                impulse_forced_frame_result["parameters"][object_id]["initial_velocity_plane_m_per_s"]
                for object_id in object_ids
            ],
            dtype=torch.float64,
            device=device,
        )
    else:
        v0 = torch.zeros((len(object_ids), 2), dtype=torch.float64, device=device)
    v0.requires_grad_(True)
    if analytic_init_result is not None:
        friction_raw = torch.tensor(
            [
                _friction_raw_from_value(
                    min(max(float(analytic_init_result["parameters"][object_id]["ground_friction"]), MIN_FRICTION), MAX_FRICTION)
                )
                for object_id in object_ids
            ],
            dtype=torch.float64,
            device=device,
        )
    elif impulse_forced_frame_result is not None:
        friction_raw = torch.tensor(
            [
                _friction_raw_from_value(
                    min(
                        max(
                            float(impulse_forced_frame_result["parameters"][object_id]["ground_friction"]),
                            MIN_FRICTION,
                        ),
                        MAX_FRICTION,
                    )
                )
                for object_id in object_ids
            ],
            dtype=torch.float64,
            device=device,
        )
    else:
        friction_raw = torch.full(
            (len(object_ids),),
            _friction_raw_from_value(MIN_FRICTION),
            dtype=torch.float64,
            device=device,
        )
    friction_raw.requires_grad_(True)
    if analytic_init_result is not None:
        initial_pair_mass_ratio_np, initial_pair_restitution_np = _pair_initialization_from_analytic_result(
            analytic_result=analytic_init_result,
            object_index_by_id=object_index_by_id,
            object_pairs=object_pairs,
            pair_index_by_key=pair_index_by_key,
        )
        pair_mass_ratio_raw = torch.tensor(
            [_pair_mass_ratio_raw_from_value(value) for value in initial_pair_mass_ratio_np.tolist()],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        restitution_raw = torch.tensor(
            [_restitution_raw_from_value(value) for value in initial_pair_restitution_np.tolist()],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
    elif impulse_forced_frame_result is not None:
        initial_pair_mass_ratio_np, initial_pair_restitution_np = _pair_initialization_from_impulse_forced_frame_result(
            impulse_forced_frame_result=impulse_forced_frame_result,
            object_index_by_id=object_index_by_id,
            object_pairs=object_pairs,
            pair_index_by_key=pair_index_by_key,
        )
        pair_mass_ratio_raw = torch.tensor(
            [_pair_mass_ratio_raw_from_value(value) for value in initial_pair_mass_ratio_np.tolist()],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        restitution_raw = torch.tensor(
            [_restitution_raw_from_value(value) for value in initial_pair_restitution_np.tolist()],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
    else:
        pair_mass_ratio_raw = torch.full(
            (len(object_pairs),),
            _pair_mass_ratio_raw_from_value(1.0),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        restitution_raw = torch.zeros((len(object_pairs),), dtype=torch.float64, device=device, requires_grad=True)
    variables = [v0, friction_raw, pair_mass_ratio_raw, restitution_raw]
    trace = []
    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    optimizer_details: dict[str, Any] = {}
    effective_warmup_steps = (
        int(warmup_steps)
        if warmup_steps is not None
        else 0
    )
    effective_warmup_prefix_frames = (
        int(warmup_prefix_frames)
        if warmup_prefix_frames is not None
        else 0
    )
    frame0 = int(frames[0])
    frame_times = torch.tensor(
        [(float(frame) - float(frame0)) / max(float(fps), 1e-12) for frame in frames],
        dtype=torch.float64,
        device=device,
    )
    current_loss_mask = mask
    current_event_search_end_times: torch.Tensor | None = None
    current_prefix_frames = 0
    current_previous_prefix_frames: int | None = None
    current_fade_alpha = 1.0
    current_fade_in_enabled = False
    current_validation_prefix_frames = 0
    current_validation_loss_mask = mask
    current_validation_event_search_end_times: torch.Tensor | None = None
    current_rollout_frame_count = len(frames)
    current_rollout_metric_mask = mask
    current_rollout_event_search_end_times: torch.Tensor | None = None
    curriculum_schedule: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    if event_source == EVENT_SOURCE_FREE_ROLLOUT:
        plot_prefix = "analytic_free_rollout"
        stage_name = "analytic_free_rollout"
    elif event_source == EVENT_SOURCE_DETECTED_WINDOW:
        plot_prefix = "analytic_forced_window"
        stage_name = "analytic_forced_window"
    else:
        raise ValueError(f"unsupported event_source: {event_source}")
    if curriculum_enabled:
        plot_prefix = "analytic_free_rollout_curriculum"
        stage_name = "analytic_free_rollout_curriculum"
    for obsolete_plot in (
        "source_impulse_forced_frame.png",
        "source_initialization_stage.png",
        "analytic_forced_window_initial_before_refinement.png",
        "analytic_free_rollout_initial_before_refinement.png",
    ):
        (output_dir / obsolete_plot).unlink(missing_ok=True)

    with torch.no_grad():
        initial_friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        initial_pair_mass_ratio = _pair_mass_ratio_from_raw(pair_mass_ratio_raw)
        initial_pair_restitution = torch.sigmoid(restitution_raw)
        initial_predicted, _initial_velocities, _initial_event_records = _analytic_rollout(
            target=target,
            frames=frames,
            event_specs=event_specs,
            pair_index_by_key=pair_index_by_key,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
            v0=v0,
            friction=initial_friction,
            pair_mass_ratio=initial_pair_mass_ratio,
            pair_restitution=initial_pair_restitution,
            fps=fps,
            event_source=event_source,
            event_window_frames=event_window_frames,
            newton_iters=newton_iters,
            object_pairs=object_pairs,
            max_free_rollout_events=max_free_rollout_events,
            active_metadata=active_metadata,
            event_search_end_times=current_event_search_end_times,
        )
        initial_analytic_rmse, initial_analytic_per_object_rmse = _rmse_summary(
            initial_predicted,
            target,
            mask,
            object_ids,
        )

    def event_records_for_current_loss(event_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if current_event_search_end_times is None:
            return event_records
        loss_event_records = []
        for record in event_records:
            object_indices = [int(index) for index in record.get("object_indices", [])]
            if len(object_indices) != 2:
                loss_event_records.append(record)
                continue
            i, j = object_indices
            pair_end = torch.minimum(current_event_search_end_times[i], current_event_search_end_times[j])
            if bool((record["event_time_s"] <= pair_end + target.new_tensor(1e-9)).detach().cpu()):
                loss_event_records.append(record)
        return loss_event_records

    def compute_loss_and_prediction(step_index: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        pair_mass_ratio = _pair_mass_ratio_from_raw(pair_mass_ratio_raw)
        pair_restitution = torch.sigmoid(restitution_raw)
        warmup_active = step_index is not None and step_index < max(effective_warmup_steps, 0)
        if warmup_active:
            predicted, velocities = _closed_form_motion(
                active_metadata["first_xy"],
                v0,
                friction,
                frame_times[:current_rollout_frame_count],
                active_metadata["active_start_times"],
            )
            event_records: list[dict[str, Any]] = []
        else:
            predicted, velocities, event_records = _analytic_rollout(
                target=target,
                frames=frames,
                event_specs=event_specs,
                pair_index_by_key=pair_index_by_key,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                v0=v0,
                friction=friction,
                pair_mass_ratio=pair_mass_ratio,
                pair_restitution=pair_restitution,
                fps=fps,
                event_source=event_source,
                event_window_frames=event_window_frames,
                newton_iters=newton_iters,
                object_pairs=object_pairs,
                max_free_rollout_events=max_free_rollout_events,
                active_metadata=active_metadata,
                event_search_end_times=current_rollout_event_search_end_times,
                rollout_frame_count=current_rollout_frame_count,
            )
        predicted_frame_count = int(predicted.shape[0])
        if warmup_active and effective_warmup_prefix_frames > 0:
            warmup_mask = _active_prefix_mask(mask, effective_warmup_prefix_frames)[:predicted_frame_count]
            position_loss = _position_mse(predicted, target[:predicted_frame_count], warmup_mask)
        else:
            position_loss = _position_mse(
                predicted,
                target[:predicted_frame_count],
                current_loss_mask[:predicted_frame_count],
            )
        loss_event_records = event_records_for_current_loss(event_records)
        event_loss = _event_record_loss(loss_event_records, contact_residual_weight, approach_penalty_weight).to(device)
        soft_contact_loss = _soft_contact_loss(
            start_x=active_metadata["first_xy"],
            start_time=active_metadata["active_start_times"],
            active_end_times=active_metadata["active_end_times"],
            v0=v0,
            friction=friction,
            frame_times=frame_times[: _frame_count_for_mask(current_loss_mask)],
            object_pairs=object_pairs,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
            weight=soft_contact_weight,
            temperature=soft_contact_temperature,
            sample_count=soft_contact_samples,
        )
        return position_loss + event_loss + soft_contact_loss, predicted, velocities, event_records

    def update_best(current_loss: float) -> None:
        nonlocal best_loss, best_state
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = (
                v0.detach().clone(),
                friction_raw.detach().clone(),
                pair_mass_ratio_raw.detach().clone(),
                restitution_raw.detach().clone(),
            )

    def pack_analytic_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = [
            v0.detach().cpu().numpy().reshape(-1),
            friction_raw.detach().cpu().numpy().reshape(-1),
            pair_mass_ratio_raw.detach().cpu().numpy().reshape(-1),
            restitution_raw.detach().cpu().numpy().reshape(-1),
        ]
        lower_parts = [
            np.full((v0.numel(),), -ANALYTIC_VELOCITY_BOUND_M_PER_S, dtype=np.float64),
            np.full((friction_raw.numel(),), -ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
            np.full((pair_mass_ratio_raw.numel(),), -ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
            np.full((restitution_raw.numel(),), -ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
        ]
        upper_parts = [
            np.full((v0.numel(),), ANALYTIC_VELOCITY_BOUND_M_PER_S, dtype=np.float64),
            np.full((friction_raw.numel(),), ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
            np.full((pair_mass_ratio_raw.numel(),), ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
            np.full((restitution_raw.numel(),), ANALYTIC_RAW_PARAMETER_BOUND, dtype=np.float64),
        ]
        return (
            np.concatenate(parts).astype(np.float64, copy=False),
            np.concatenate(lower_parts),
            np.concatenate(upper_parts),
        )

    def assign_analytic_state(values: np.ndarray) -> None:
        cursor = 0
        with torch.no_grad():
            next_cursor = cursor + v0.numel()
            v0.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor].reshape(tuple(v0.shape)),
                    dtype=torch.float64,
                    device=device,
                )
            )
            cursor = next_cursor
            next_cursor = cursor + friction_raw.numel()
            friction_raw.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor],
                    dtype=torch.float64,
                    device=device,
                )
            )
            cursor = next_cursor
            next_cursor = cursor + pair_mass_ratio_raw.numel()
            pair_mass_ratio_raw.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor],
                    dtype=torch.float64,
                    device=device,
                )
            )
            cursor = next_cursor
            next_cursor = cursor + restitution_raw.numel()
            restitution_raw.copy_(
                torch.as_tensor(
                    values[cursor:next_cursor],
                    dtype=torch.float64,
                    device=device,
                )
            )

    def analytic_residual(values: np.ndarray) -> np.ndarray:
        assign_analytic_state(values)
        with torch.no_grad():
            _loss, predicted, _velocities, event_records = compute_loss_and_prediction(None)
            predicted_frame_count = int(predicted.shape[0])
            residual_mask = current_loss_mask[:predicted_frame_count]
            active = residual_mask.expand_as(predicted) > 0.0
            position_denominator = math.sqrt(max(float(torch.sum(residual_mask).detach().cpu()), 1.0))
            residual_parts = [
                ((predicted - target[:predicted_frame_count])[active] / position_denominator)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            ]
            loss_event_records = event_records_for_current_loss(event_records)
            if loss_event_records:
                event_count_scale = math.sqrt(max(float(len(loss_event_records)), 1.0))
                contact_scale = math.sqrt(max(float(contact_residual_weight), 0.0)) / event_count_scale
                approach_scale = math.sqrt(max(float(approach_penalty_weight), 0.0)) / event_count_scale
                if contact_scale > 0.0:
                    residual_parts.append(
                        np.asarray(
                            [
                                contact_scale * float(record["contact_residual_m"].detach().cpu())
                                for record in loss_event_records
                            ],
                            dtype=np.float64,
                        )
                    )
                if approach_scale > 0.0:
                    residual_parts.append(
                        np.asarray(
                            [
                                approach_scale
                                * max(float(record["relative_normal_velocity"].detach().cpu()), 0.0)
                                for record in loss_event_records
                            ],
                            dtype=np.float64,
                        )
                    )
            return np.concatenate(residual_parts)

    def curriculum_prefix_values() -> list[int]:
        max_observed_frames = int(torch.max(torch.sum(mask[:, :, 0], dim=0)).detach().cpu()) if mask.numel() else 0
        return analytic_swr_common.curriculum_prefix_values(
            observed_frames=max_observed_frames,
            prefix_step_frames=curriculum_prefix_step_frames,
        )

    def set_curriculum_prefix(
        prefix_frames: int,
        previous_prefix_frames: int | None = None,
        fade_alpha: float = 1.0,
        fade_in_enabled: bool = False,
    ) -> None:
        nonlocal current_prefix_frames, current_loss_mask, current_event_search_end_times
        nonlocal current_previous_prefix_frames, current_fade_alpha, current_fade_in_enabled
        nonlocal current_validation_prefix_frames, current_validation_loss_mask, current_validation_event_search_end_times
        nonlocal current_rollout_frame_count, current_rollout_metric_mask, current_rollout_event_search_end_times
        current_prefix_frames = int(prefix_frames)
        current_previous_prefix_frames = None if previous_prefix_frames is None else int(previous_prefix_frames)
        current_fade_alpha = float(fade_alpha)
        current_fade_in_enabled = bool(fade_in_enabled and current_previous_prefix_frames is not None)
        if current_prefix_frames > 0:
            if current_fade_in_enabled:
                current_loss_mask = _active_prefix_fade_mask(
                    mask,
                    previous_prefix_frames=current_previous_prefix_frames,
                    prefix_frames=current_prefix_frames,
                    fade_alpha=current_fade_alpha,
                )
            else:
                current_loss_mask = _active_prefix_mask(mask, current_prefix_frames)
            current_event_search_end_times = _active_prefix_end_times(
                mask=mask,
                frames=frames,
                fps=fps,
                frame0=frame0,
                prefix_frames=current_prefix_frames,
                active_metadata=active_metadata,
            )
        else:
            if current_fade_in_enabled:
                current_loss_mask = _active_prefix_fade_mask(
                    mask,
                    previous_prefix_frames=current_previous_prefix_frames,
                    prefix_frames=0,
                    fade_alpha=current_fade_alpha,
                )
            else:
                current_loss_mask = mask
            current_event_search_end_times = None
        if current_prefix_frames > 0:
            current_validation_prefix_frames = current_prefix_frames + effective_curriculum_validation_frames
            current_validation_loss_mask = _active_prefix_mask(mask, current_validation_prefix_frames)
            current_validation_event_search_end_times = _active_prefix_end_times(
                mask=mask,
                frames=frames,
                fps=fps,
                frame0=frame0,
                prefix_frames=current_validation_prefix_frames,
                active_metadata=active_metadata,
            )
        else:
            current_validation_prefix_frames = 0
            current_validation_loss_mask = mask
            current_validation_event_search_end_times = None
        if curriculum_best_metric == CURRICULUM_BEST_FULL_ROLLOUT_RMSE:
            current_rollout_metric_mask = mask
            current_rollout_event_search_end_times = None
        elif curriculum_best_metric == CURRICULUM_BEST_FUTURE_ROLLOUT_RMSE:
            current_rollout_metric_mask = _active_future_mask(mask, current_prefix_frames)
            current_rollout_event_search_end_times = None
        elif curriculum_best_metric == CURRICULUM_BEST_NEXT_WINDOW_RMSE:
            current_rollout_metric_mask = current_validation_loss_mask
            current_rollout_event_search_end_times = current_validation_event_search_end_times
        else:
            current_rollout_metric_mask = current_loss_mask
            current_rollout_event_search_end_times = current_event_search_end_times
        current_rollout_frame_count = _frame_count_for_mask(current_rollout_metric_mask)

    def curriculum_stage_uses_fade_in(previous_prefix_frames: int | None) -> bool:
        return bool(effective_curriculum_fade_in and previous_prefix_frames is not None)

    def curriculum_fade_alpha_for_step(
        local_step: int,
        step_count: int,
        previous_prefix_frames: int | None,
    ) -> float:
        if not curriculum_stage_uses_fade_in(previous_prefix_frames):
            return 1.0
        if int(step_count) <= 1:
            return 1.0
        return float(min(max(int(local_step), 0), int(step_count) - 1)) / float(int(step_count) - 1)

    def curriculum_fade_bounds(previous_prefix_frames: int | None, step_count: int) -> tuple[float, float]:
        if not curriculum_stage_uses_fade_in(previous_prefix_frames):
            return 1.0, 1.0
        if int(step_count) <= 1:
            return 1.0, 1.0
        return 0.0, 1.0

    def restore_curriculum_state(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if state is None:
            return
        with torch.no_grad():
            v0.copy_(state[0])
            friction_raw.copy_(state[1])
            pair_mass_ratio_raw.copy_(state[2])
            restitution_raw.copy_(state[3])

    def clone_curriculum_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            v0.detach().clone(),
            friction_raw.detach().clone(),
            pair_mass_ratio_raw.detach().clone(),
            restitution_raw.detach().clone(),
        )

    def append_curriculum_schedule(
        *,
        stage_trace_start: int,
        stage_best_step: int,
        stage_best_loss: float,
        stage_best_metric_value: float,
        stage_best_prefix_rmse: float,
        final_prefix_rmse: float,
        fade_in_enabled: bool,
        fade_alpha_start: float,
        fade_alpha_end: float,
        max_steps: int,
        steps_run: int,
        early_stop_patience: int,
        early_stopped: bool,
    ) -> None:
        with torch.no_grad():
            friction_eval = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
            pair_mass_ratio_eval = _pair_mass_ratio_from_raw(pair_mass_ratio_raw)
            pair_restitution_eval = torch.sigmoid(restitution_raw)
            full_predicted, _full_velocities, _full_events = _analytic_rollout(
                target=target,
                frames=frames,
                event_specs=event_specs,
                pair_index_by_key=pair_index_by_key,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                v0=v0,
                friction=friction_eval,
                pair_mass_ratio=pair_mass_ratio_eval,
                pair_restitution=pair_restitution_eval,
                fps=fps,
                event_source=EVENT_SOURCE_FREE_ROLLOUT,
                event_window_frames=event_window_frames,
                newton_iters=newton_iters,
                object_pairs=object_pairs,
                max_free_rollout_events=max_free_rollout_events,
                active_metadata=active_metadata,
            )
            full_rmse, _full_per_object = _rmse_summary(full_predicted, target, mask, object_ids)
            validation_rmse = (
                float(stage_best_metric_value)
                if curriculum_best_metric == CURRICULUM_BEST_NEXT_WINDOW_RMSE
                else None
            )
            curriculum_schedule.append(
                {
                    "prefix_frames": None if current_prefix_frames <= 0 else int(current_prefix_frames),
                    "previous_prefix_frames": (
                        None
                        if current_previous_prefix_frames is None or current_previous_prefix_frames <= 0
                        else int(current_previous_prefix_frames)
                    ),
                    "fade_in": bool(fade_in_enabled),
                    "fade_alpha_start": float(fade_alpha_start),
                    "fade_alpha_end": float(fade_alpha_end),
                    "validation_prefix_frames": (
                        None if current_validation_prefix_frames <= 0 else int(current_validation_prefix_frames)
                    ),
                    "validation_frames": int(effective_curriculum_validation_frames),
                    "metric_frame_count": int(_frame_count_for_mask(current_rollout_metric_mask)),
                    "metric_active_observations": float(torch.sum(current_rollout_metric_mask).detach().cpu()),
                    "steps": int(steps_run),
                    "max_steps": int(max_steps),
                    "steps_run": int(steps_run),
                    "early_stop_patience": int(early_stop_patience),
                    "early_stopped": bool(early_stopped),
                    "trace_start": int(stage_trace_start),
                    "trace_end": int(len(trace)),
                    "best_step": int(stage_best_step),
                    "best_loss": float(stage_best_loss),
                    "best_metric": str(curriculum_best_metric),
                    "best_metric_value": float(stage_best_metric_value),
                    "validation_rmse_m": None if validation_rmse is None else float(validation_rmse),
                    "best_prefix_rmse_m": float(stage_best_prefix_rmse),
                    "final_prefix_rmse_m": float(final_prefix_rmse),
                    "prefix_rmse_m": float(stage_best_prefix_rmse),
                    "free_rollout_rmse_m": float(full_rmse.cpu()),
                }
            )

    def current_curriculum_metric(current_loss: float, predicted: torch.Tensor) -> float:
        predicted_frame_count = int(predicted.shape[0])
        if curriculum_best_metric == CURRICULUM_BEST_FULL_ROLLOUT_RMSE:
            return float(
                torch.sqrt(_position_mse(predicted, target[:predicted_frame_count], mask[:predicted_frame_count]))
                .detach()
                .cpu()
            )
        if curriculum_best_metric == CURRICULUM_BEST_NEXT_WINDOW_RMSE:
            return float(
                torch.sqrt(
                    _position_mse(
                        predicted,
                        target[:predicted_frame_count],
                        current_validation_loss_mask[:predicted_frame_count],
                    )
                )
                .detach()
                .cpu()
            )
        if curriculum_best_metric == CURRICULUM_BEST_FUTURE_ROLLOUT_RMSE:
            return float(
                torch.sqrt(
                    _position_mse(
                        predicted,
                        target[:predicted_frame_count],
                        current_rollout_metric_mask[:predicted_frame_count],
                    )
                )
                .detach()
                .cpu()
            )
        return float(current_loss)

    if analytic_optimizer == ANALYTIC_OPTIMIZER_ADAM:
        optimizer = torch.optim.Adam(variables, lr=float(lr), weight_decay=0.0)
        if curriculum_enabled:
            prefix_values = curriculum_prefix_values()
            per_prefix_steps = max(int(curriculum_steps_per_prefix), 1)
            global_step = 0
            previous_prefix_frames: int | None = None
            for prefix_frames in prefix_values:
                stage_fade_in = curriculum_stage_uses_fade_in(previous_prefix_frames)
                fade_alpha_start, fade_alpha_end = curriculum_fade_bounds(previous_prefix_frames, per_prefix_steps)
                stage_trace_start = len(trace)
                stage_best = analytic_swr_common.CurriculumStageBest()
                final_prefix_rmse = float("nan")
                steps_run = 0
                early_stopped = False
                for local_step in range(per_prefix_steps):
                    fade_alpha = curriculum_fade_alpha_for_step(
                        local_step,
                        per_prefix_steps,
                        previous_prefix_frames,
                    )
                    set_curriculum_prefix(
                        prefix_frames,
                        previous_prefix_frames=previous_prefix_frames,
                        fade_alpha=fade_alpha,
                        fade_in_enabled=stage_fade_in,
                    )
                    optimizer.zero_grad()
                    loss, predicted, _velocities, _event_records = compute_loss_and_prediction(None)
                    current_loss = float(loss.detach().cpu())
                    with torch.no_grad():
                        predicted_frame_count = int(predicted.shape[0])
                        current_prefix_rmse = torch.sqrt(
                            _position_mse(
                                predicted,
                                target[:predicted_frame_count],
                                current_loss_mask[:predicted_frame_count],
                            )
                        ).detach().cpu()
                        final_prefix_rmse = float(current_prefix_rmse)
                    current_metric_value = current_curriculum_metric(current_loss, predicted)
                    stage_best.consider(
                        metric_value=current_metric_value,
                        loss=current_loss,
                        step=global_step + 1,
                        prefix_rmse=final_prefix_rmse,
                        state=clone_curriculum_state(),
                    )
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    steps_run = int(local_step + 1)
                    if local_step < 5 or (local_step + 1) % 100 == 0 or local_step + 1 == per_prefix_steps:
                        with torch.no_grad():
                            predicted_frame_count = int(predicted.shape[0])
                            rollout_rmse = torch.sqrt(
                                _position_mse(
                                    predicted,
                                    target[:predicted_frame_count],
                                    current_rollout_metric_mask[:predicted_frame_count],
                                )
                            ).detach().cpu()
                            trace.append(
                                {
                                    "step": int(global_step),
                                    "prefix_frames": None if current_prefix_frames <= 0 else int(current_prefix_frames),
                                    "previous_prefix_frames": (
                                        None
                                        if current_previous_prefix_frames is None or current_previous_prefix_frames <= 0
                                        else int(current_previous_prefix_frames)
                                    ),
                                    "fade_in": bool(current_fade_in_enabled),
                                    "fade_alpha": float(current_fade_alpha),
                                    "rollout_frame_count": int(predicted_frame_count),
                                    "loss": current_loss,
                                    "prefix_rmse_m": final_prefix_rmse,
                                    "rmse_m": float(rollout_rmse),
                                }
                            )
                    if stage_best.should_stop(effective_curriculum_early_stop_patience):
                        early_stopped = True
                        break
                restore_curriculum_state(stage_best.state)
                set_curriculum_prefix(
                    prefix_frames,
                    previous_prefix_frames=previous_prefix_frames,
                    fade_alpha=fade_alpha_end,
                    fade_in_enabled=stage_fade_in,
                )
                if current_prefix_frames == 0 and current_fade_alpha >= 1.0:
                    update_best(stage_best.loss)
                append_curriculum_schedule(
                    stage_trace_start=stage_trace_start,
                    stage_best_step=stage_best.step,
                    stage_best_loss=stage_best.loss,
                    stage_best_metric_value=stage_best.metric_value,
                    stage_best_prefix_rmse=stage_best.prefix_rmse,
                    final_prefix_rmse=final_prefix_rmse,
                    fade_in_enabled=stage_fade_in,
                    fade_alpha_start=fade_alpha_start,
                    fade_alpha_end=fade_alpha_end,
                    max_steps=per_prefix_steps,
                    steps_run=steps_run,
                    early_stop_patience=effective_curriculum_early_stop_patience,
                    early_stopped=early_stopped,
                )
                previous_prefix_frames = int(prefix_frames)
            current_prefix_frames = 0
            current_loss_mask = mask
            current_event_search_end_times = None
            current_rollout_frame_count = len(frames)
            current_rollout_metric_mask = mask
            current_rollout_event_search_end_times = None
        else:
            for step in range(max(int(steps), 0)):
                optimizer.zero_grad()
                loss, predicted, _velocities, _event_records = compute_loss_and_prediction(step)
                current_loss = float(loss.detach().cpu())
                if step >= max(effective_warmup_steps, 0):
                    update_best(current_loss)
                loss.backward()
                optimizer.step()
                if step < 20 or (step + 1) % 100 == 0:
                    with torch.no_grad():
                        rmse = torch.sqrt(_position_mse(predicted, target, mask)).detach().cpu()
                        trace.append(
                            {
                                "step": int(step + 1),
                                "loss": current_loss,
                                "rmse_m": float(rmse),
                            }
                        )
    elif analytic_optimizer == ANALYTIC_OPTIMIZER_LBFGS:
        if curriculum_enabled:
            prefix_values = curriculum_prefix_values()
            per_prefix_steps = max(int(curriculum_steps_per_prefix), 1)
            global_step = 0
            previous_prefix_frames: int | None = None
            for prefix_frames in prefix_values:
                stage_fade_in = curriculum_stage_uses_fade_in(previous_prefix_frames)
                fade_alpha_start, fade_alpha_end = curriculum_fade_bounds(previous_prefix_frames, per_prefix_steps)
                set_curriculum_prefix(
                    prefix_frames,
                    previous_prefix_frames=previous_prefix_frames,
                    fade_alpha=fade_alpha_start,
                    fade_in_enabled=stage_fade_in,
                )
                stage_trace_start = len(trace)
                stage_best_loss = float("inf")
                stage_best_metric_value = float("inf")
                stage_best_step = 0
                stage_best_prefix_rmse = float("nan")
                stage_best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                final_prefix_rmse = float("nan")
                local_closure_calls = 0
                optimizer = torch.optim.LBFGS(
                    variables,
                    lr=float(lr),
                    max_iter=per_prefix_steps,
                    max_eval=max(2 * per_prefix_steps, 1),
                    tolerance_grad=1e-10,
                    tolerance_change=1e-12,
                    history_size=50,
                    line_search_fn="strong_wolfe",
                )

                def curriculum_closure() -> torch.Tensor:
                    nonlocal global_step, local_closure_calls, stage_best_loss, stage_best_step, stage_best_state
                    nonlocal stage_best_metric_value, stage_best_prefix_rmse, final_prefix_rmse
                    fade_alpha = curriculum_fade_alpha_for_step(
                        local_closure_calls,
                        per_prefix_steps,
                        previous_prefix_frames,
                    )
                    set_curriculum_prefix(
                        prefix_frames,
                        previous_prefix_frames=previous_prefix_frames,
                        fade_alpha=fade_alpha,
                        fade_in_enabled=stage_fade_in,
                    )
                    optimizer.zero_grad()
                    loss, predicted, _velocities, _event_records = compute_loss_and_prediction(None)
                    current_loss = float(loss.detach().cpu())
                    with torch.no_grad():
                        predicted_frame_count = int(predicted.shape[0])
                        current_prefix_rmse = torch.sqrt(
                            _position_mse(
                                predicted,
                                target[:predicted_frame_count],
                                current_loss_mask[:predicted_frame_count],
                            )
                        ).detach().cpu()
                        final_prefix_rmse = float(current_prefix_rmse)
                        rollout_rmse = torch.sqrt(
                            _position_mse(
                                predicted,
                                target[:predicted_frame_count],
                                current_rollout_metric_mask[:predicted_frame_count],
                            )
                        ).detach().cpu()
                    global_step += 1
                    current_metric_value = current_curriculum_metric(current_loss, predicted)
                    if current_metric_value < stage_best_metric_value:
                        stage_best_metric_value = float(current_metric_value)
                        stage_best_loss = current_loss
                        stage_best_step = int(global_step)
                        stage_best_prefix_rmse = float(final_prefix_rmse)
                        stage_best_state = (
                            v0.detach().clone(),
                            friction_raw.detach().clone(),
                            pair_mass_ratio_raw.detach().clone(),
                            restitution_raw.detach().clone(),
                        )
                    loss.backward()
                    local_closure_calls += 1
                    if local_closure_calls <= 5 or local_closure_calls % 100 == 0:
                        trace.append(
                            {
                                "step": int(global_step),
                                "prefix_frames": None if current_prefix_frames <= 0 else int(current_prefix_frames),
                                "previous_prefix_frames": (
                                    None
                                    if current_previous_prefix_frames is None or current_previous_prefix_frames <= 0
                                    else int(current_previous_prefix_frames)
                                ),
                                "fade_in": bool(current_fade_in_enabled),
                                "fade_alpha": float(current_fade_alpha),
                                "rollout_frame_count": int(predicted_frame_count),
                                "loss": current_loss,
                                "prefix_rmse_m": final_prefix_rmse,
                                "rmse_m": float(rollout_rmse),
                            }
                        )
                    return loss

                optimizer.step(curriculum_closure)
                restore_curriculum_state(stage_best_state)
                set_curriculum_prefix(
                    prefix_frames,
                    previous_prefix_frames=previous_prefix_frames,
                    fade_alpha=fade_alpha_end,
                    fade_in_enabled=stage_fade_in,
                )
                if current_prefix_frames == 0 and current_fade_alpha >= 1.0:
                    update_best(stage_best_loss)
                append_curriculum_schedule(
                    stage_trace_start=stage_trace_start,
                    stage_best_step=stage_best_step,
                    stage_best_loss=stage_best_loss,
                    stage_best_metric_value=stage_best_metric_value,
                    stage_best_prefix_rmse=stage_best_prefix_rmse,
                    final_prefix_rmse=final_prefix_rmse,
                    fade_in_enabled=stage_fade_in,
                    fade_alpha_start=fade_alpha_start,
                    fade_alpha_end=fade_alpha_end,
                    max_steps=per_prefix_steps,
                    steps_run=local_closure_calls,
                    early_stop_patience=0,
                    early_stopped=False,
                )
                previous_prefix_frames = int(prefix_frames)
            current_prefix_frames = 0
            current_loss_mask = mask
            current_event_search_end_times = None
            current_rollout_frame_count = len(frames)
            current_rollout_metric_mask = mask
            current_rollout_event_search_end_times = None
            optimizer_details = {
                "method": "torch_lbfgs_curriculum",
                "line_search_fn": "strong_wolfe",
                "lr": float(lr),
                "max_iter_per_prefix": int(per_prefix_steps),
                "closure_calls": int(global_step),
            }
        else:
            optimizer = torch.optim.LBFGS(
                variables,
                lr=1.0,
                max_iter=max(int(steps), 1),
                max_eval=max(2 * int(steps), 1),
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            closure_calls = 0

            def closure() -> torch.Tensor:
                nonlocal closure_calls
                optimizer.zero_grad()
                loss, predicted, _velocities, _event_records = compute_loss_and_prediction(None)
                current_loss = float(loss.detach().cpu())
                update_best(current_loss)
                loss.backward()
                closure_calls += 1
                if closure_calls <= 20 or closure_calls % 100 == 0:
                    with torch.no_grad():
                        rmse = torch.sqrt(_position_mse(predicted, target, mask)).detach().cpu()
                        trace.append(
                            {
                                "step": int(closure_calls),
                                "loss": current_loss,
                                "rmse_m": float(rmse),
                            }
                        )
                return loss

            optimizer.step(closure)
            optimizer_details = {
                "method": "torch_lbfgs",
                "line_search_fn": "strong_wolfe",
                "lr": 1.0,
                "max_iter": int(max(int(steps), 1)),
                "closure_calls": int(closure_calls),
            }
    else:
        x0, lower_bounds, upper_bounds = pack_analytic_state()
        with torch.no_grad():
            initial_loss, initial_predicted, _initial_velocities, _initial_event_records = compute_loss_and_prediction(None)
            trace.append(
                {
                    "step": 0,
                    "loss": float(initial_loss.detach().cpu()),
                    "rmse_m": float(torch.sqrt(_position_mse(initial_predicted, target, mask)).detach().cpu()),
                }
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"scipy\.optimize\._lsq\..*",
            )
            result = least_squares(
                analytic_residual,
                x0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                x_scale=1.0,
                max_nfev=max(int(steps), 1),
                ftol=None,
                xtol=None,
                gtol=1e-10,
            )
        assign_analytic_state(result.x)
        with torch.no_grad():
            final_loss, final_predicted, _final_velocities, _final_event_records = compute_loss_and_prediction(None)
            trace.append(
                {
                    "step": int(result.nfev),
                    "loss": float(final_loss.detach().cpu()),
                    "rmse_m": float(torch.sqrt(_position_mse(final_predicted, target, mask)).detach().cpu()),
                    "least_squares_cost": float(result.cost),
                    "least_squares_optimality": float(result.optimality),
                    "least_squares_status": int(result.status),
                    "least_squares_message": str(result.message),
                }
            )
        optimizer_details = {
            "method": "trf",
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "status": int(result.status),
            "message": str(result.message),
            "success": bool(result.success),
        }
    if best_state is not None:
        with torch.no_grad():
            v0.copy_(best_state[0])
            friction_raw.copy_(best_state[1])
            pair_mass_ratio_raw.copy_(best_state[2])
            restitution_raw.copy_(best_state[3])
    with torch.no_grad():
        friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        pair_mass_ratio = _pair_mass_ratio_from_raw(pair_mass_ratio_raw)
        pair_restitution = torch.sigmoid(restitution_raw)
        predicted, velocities, event_records = _analytic_rollout(
            target=target,
            frames=frames,
            event_specs=event_specs,
            pair_index_by_key=pair_index_by_key,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
            v0=v0,
            friction=friction,
            pair_mass_ratio=pair_mass_ratio,
            pair_restitution=pair_restitution,
            fps=fps,
            event_source=event_source,
            event_window_frames=event_window_frames,
            newton_iters=newton_iters,
            object_pairs=object_pairs,
            max_free_rollout_events=max_free_rollout_events,
            active_metadata=active_metadata,
        )
        overall_rmse, per_object_rmse = _rmse_summary(predicted, target, mask, object_ids)
        if event_source == EVENT_SOURCE_FREE_ROLLOUT:
            free_rollout_predicted = predicted
            free_rollout_event_records = event_records
            free_rollout_rmse = overall_rmse
            free_rollout_per_object_rmse = per_object_rmse
        else:
            free_rollout_predicted, _free_rollout_velocities, free_rollout_event_records = _analytic_rollout(
                target=target,
                frames=frames,
                event_specs=event_specs,
                pair_index_by_key=pair_index_by_key,
                contact_radii=contact_radii,
                shape_ids=shape_ids,
                v0=v0,
                friction=friction,
                pair_mass_ratio=pair_mass_ratio,
                pair_restitution=pair_restitution,
                fps=fps,
                event_source=EVENT_SOURCE_FREE_ROLLOUT,
                event_window_frames=event_window_frames,
                newton_iters=newton_iters,
                object_pairs=object_pairs,
                max_free_rollout_events=max_free_rollout_events,
                active_metadata=active_metadata,
            )
            free_rollout_rmse, free_rollout_per_object_rmse = _rmse_summary(
                free_rollout_predicted,
                target,
                mask,
                object_ids,
            )

    plot_path = output_dir / f"{plot_prefix}.png" if render_video_flag else None
    if plot_path is not None:
        _plot(
            plot_path,
            object_ids,
            target.detach().cpu().numpy(),
            predicted.detach().cpu().numpy(),
            mask.detach().cpu().numpy(),
        )
    video_path = plot_path.with_suffix(".mp4") if plot_path is not None else None
    if video_path is not None:
        _render_comparison_video(
            video_path,
            object_ids=object_ids,
            frames=frames,
            target=target.detach().cpu().numpy(),
            predicted=predicted.detach().cpu().numpy(),
            mask=mask.detach().cpu().numpy(),
            contact_radii=contact_radii.detach().cpu().numpy(),
            source_fps=fps,
        )
    if event_source == EVENT_SOURCE_FREE_ROLLOUT:
        free_rollout_plot_path = plot_path
        free_rollout_video_path = video_path
    else:
        free_rollout_plot_path = output_dir / f"{plot_prefix}_free_rollout.png" if render_video_flag else None
        if free_rollout_plot_path is not None:
            _plot(
                free_rollout_plot_path,
                object_ids,
                target.detach().cpu().numpy(),
                free_rollout_predicted.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )
        free_rollout_video_path = free_rollout_plot_path.with_suffix(".mp4") if free_rollout_plot_path is not None else None
        if free_rollout_video_path is not None:
            _render_comparison_video(
                free_rollout_video_path,
                object_ids=object_ids,
                frames=frames,
                target=target.detach().cpu().numpy(),
                predicted=free_rollout_predicted.detach().cpu().numpy(),
                mask=mask.detach().cpu().numpy(),
                contact_radii=contact_radii.detach().cpu().numpy(),
                source_fps=fps,
            )
    model_events = []
    for record in event_records:
        event_index = int(record["event_index"].detach().cpu())
        object_indices = [int(index) for index in record["object_indices"]]
        source_frame = record.get("source_frame")
        event_frame = int(source_frame) if source_frame is not None else int(round(float(record["event_frame_float"].detach().cpu())))
        model_events.append(
            {
                "event_index": event_index,
                "event_type": str(record.get("event_type", "pair_collision_candidate")),
                "frame": event_frame,
                "object_ids": [object_ids[index] for index in object_indices],
                "source": str(record.get("event_source", event_source)),
                "contact_model": str(record.get("contact_model", "unknown")),
            }
        )
    free_rollout_events = []
    for record in free_rollout_event_records:
        event_index = int(record["event_index"].detach().cpu())
        object_indices = [int(index) for index in record["object_indices"]]
        source_frame = record.get("source_frame")
        event_frame = int(source_frame) if source_frame is not None else int(round(float(record["event_frame_float"].detach().cpu())))
        free_rollout_events.append(
            {
                "event_index": event_index,
                "event_type": str(record.get("event_type", "pair_collision_candidate")),
                "frame": event_frame,
                "object_ids": [object_ids[index] for index in object_indices],
                "source": str(record.get("event_source", EVENT_SOURCE_FREE_ROLLOUT)),
                "contact_model": str(record.get("contact_model", "unknown")),
                "event_frame_float": float(record["event_frame_float"].detach().cpu()),
                "contact_residual_m": float(record["contact_residual_m"].detach().cpu()),
            }
        )
    event_records_by_index = {int(record["event_index"].detach().cpu()): record for record in event_records}
    pair_mass_ratio_np = pair_mass_ratio.detach().cpu().numpy()
    pair_restitution_np = pair_restitution.detach().cpu().numpy()
    used_pair_indices = {
        pair_index_by_key[tuple(sorted(int(index) for index in record["object_indices"]))]
        for record in event_records
        if len(record.get("object_indices", [])) == 2
    }
    mass_observations = [
        {
            "i": int(i),
            "j": int(j),
            "mass_ratio_i_over_j": float(pair_mass_ratio_np[pair_index]),
            "weight": 1.0,
        }
        for pair_index, (i, j) in enumerate(object_pairs)
        if pair_index in used_pair_indices
    ]
    restitution_observations = [
        {
            "i": int(i),
            "j": int(j),
            "pair_restitution": float(pair_restitution_np[pair_index]),
            "weight": 1.0,
        }
        for pair_index, (i, j) in enumerate(object_pairs)
        if pair_index in used_pair_indices
    ]
    solved_log_mass, mass_solve_diagnostics = _solve_log_masses(len(object_ids), mass_observations)
    solved_log_restitution, restitution_solve_diagnostics = _solve_log_restitutions(
        len(object_ids),
        restitution_observations,
    )
    solved_mass_np = np.clip(np.exp(solved_log_mass), MIN_MASS, MAX_MASS)
    solved_restitution_np = np.clip(np.exp(solved_log_restitution), 1e-6, 1.0)
    effective_pair_mass_ratio_np = np.zeros_like(pair_mass_ratio_np, dtype=np.float64)
    effective_pair_restitution_np = np.zeros_like(pair_restitution_np, dtype=np.float64)
    pair_parameter_sources: list[str] = []
    direct_pair_parameter_available: list[bool] = []
    for pair_index, (i, j) in enumerate(object_pairs):
        has_direct_pair_parameter = bool(pair_index in used_pair_indices)
        object_fallback_mass_ratio = float(solved_mass_np[i] / solved_mass_np[j])
        object_fallback_pair_restitution = float(solved_restitution_np[i] * solved_restitution_np[j])
        if has_direct_pair_parameter:
            effective_pair_mass_ratio_np[pair_index] = float(pair_mass_ratio_np[pair_index])
            effective_pair_restitution_np[pair_index] = float(pair_restitution_np[pair_index])
            pair_parameter_sources.append("optimized_pair")
        else:
            effective_pair_mass_ratio_np[pair_index] = object_fallback_mass_ratio
            effective_pair_restitution_np[pair_index] = object_fallback_pair_restitution
            pair_parameter_sources.append("object_fallback")
        direct_pair_parameter_available.append(has_direct_pair_parameter)
    mass_ratio_equation_residuals = []
    restitution_equation_residuals = []
    for pair_index, (i, j) in enumerate(object_pairs):
        optimized_mass_ratio = float(pair_mass_ratio_np[pair_index])
        solved_mass_ratio = float(solved_mass_np[i] / solved_mass_np[j])
        optimized_restitution = float(pair_restitution_np[pair_index])
        solved_restitution_product = float(solved_restitution_np[i] * solved_restitution_np[j])
        has_direct_pair_parameter = direct_pair_parameter_available[pair_index]
        mass_ratio_equation_residuals.append(
            {
                "object_ids": [object_ids[i], object_ids[j]],
                "direct_pair_parameter_available": has_direct_pair_parameter,
                "optimized_mass_ratio_i_over_j": optimized_mass_ratio if has_direct_pair_parameter else None,
                "solved_mass_ratio_i_over_j": solved_mass_ratio,
                "effective_mass_ratio_i_over_j": float(effective_pair_mass_ratio_np[pair_index]),
                "pair_parameter_source": pair_parameter_sources[pair_index],
                "used_in_object_parameter_solve": has_direct_pair_parameter,
                "log_residual": (
                    float(np.log(max(solved_mass_ratio, 1e-12)) - np.log(max(optimized_mass_ratio, 1e-12)))
                    if has_direct_pair_parameter
                    else None
                ),
            }
        )
        restitution_equation_residuals.append(
            {
                "object_ids": [object_ids[i], object_ids[j]],
                "direct_pair_parameter_available": has_direct_pair_parameter,
                "optimized_pair_restitution": optimized_restitution if has_direct_pair_parameter else None,
                "solved_pair_restitution_product": solved_restitution_product,
                "effective_pair_restitution": float(effective_pair_restitution_np[pair_index]),
                "pair_parameter_source": pair_parameter_sources[pair_index],
                "used_in_object_parameter_solve": has_direct_pair_parameter,
                "log_residual": (
                    float(
                        np.log(max(solved_restitution_product, 1e-12))
                        - np.log(max(optimized_restitution, 1e-12))
                    )
                    if has_direct_pair_parameter
                    else None
                ),
            }
        )
    source_impulse_forced_frame_stage_rmse = None
    source_impulse_forced_frame_free_rollout_rmse = None
    if impulse_forced_frame_result is not None:
        if "stage_rmse_m" in impulse_forced_frame_result:
            source_impulse_forced_frame_stage_rmse = float(impulse_forced_frame_result["stage_rmse_m"])
        if "free_rollout_rmse_m" in impulse_forced_frame_result:
            source_impulse_forced_frame_free_rollout_rmse = float(impulse_forced_frame_result["free_rollout_rmse_m"])
    elif analytic_init_result is not None:
        if analytic_init_result.get("source_impulse_forced_frame_stage_rmse_m") is not None:
            source_impulse_forced_frame_stage_rmse = float(
                analytic_init_result["source_impulse_forced_frame_stage_rmse_m"]
            )
        if analytic_init_result.get("source_impulse_forced_frame_free_rollout_rmse_m") is not None:
            source_impulse_forced_frame_free_rollout_rmse = float(
                analytic_init_result["source_impulse_forced_frame_free_rollout_rmse_m"]
            )
    payload = {
        "input_fit": str(input_fit),
        "steps": int(steps),
        "lr": float(lr),
        "fps": float(fps),
        "model": "analytic-event-collision-v1",
        "optimizer": analytic_optimizer,
        "optimizer_details": optimizer_details,
        "stage_name": stage_name,
        "render_video": render_video_flag,
        "init_from_impulse_forced_frame_result": (
            str(init_from_impulse_forced_frame_result)
            if init_from_impulse_forced_frame_result is not None
            else None
        ),
        "init_from_analytic_result": str(init_from_analytic_result) if init_from_analytic_result is not None else None,
        "source_impulse_forced_frame_stage_rmse_m": source_impulse_forced_frame_stage_rmse,
        "source_impulse_forced_frame_free_rollout_rmse_m": source_impulse_forced_frame_free_rollout_rmse,
        "source_initialization_stage_rmse_m": (
            float(analytic_init_result["stage_rmse_m"])
            if analytic_init_result is not None and "stage_rmse_m" in analytic_init_result
            else None
        ),
        "source_initialization_stage_free_rollout_rmse_m": (
            float(analytic_init_result["free_rollout_rmse_m"])
            if analytic_init_result is not None and "free_rollout_rmse_m" in analytic_init_result
            else None
        ),
        "source_initialization_stage_event_source": (
            str(analytic_init_result["event_source"])
            if analytic_init_result is not None and "event_source" in analytic_init_result
            else None
        ),
        "initial_analytic_rmse_m": float(initial_analytic_rmse.cpu()),
        "initial_analytic_per_object_rmse_m": initial_analytic_per_object_rmse,
        "event_source": event_source,
        "warmup_steps": int(effective_warmup_steps),
        "warmup_prefix_frames": int(effective_warmup_prefix_frames),
        "curriculum_prefix_step_frames": int(curriculum_prefix_step_frames),
        "curriculum_steps_per_prefix": int(curriculum_steps_per_prefix),
        "curriculum_best_metric": str(curriculum_best_metric),
        "curriculum_fade_in": bool(effective_curriculum_fade_in),
        "curriculum_early_stop_patience": int(effective_curriculum_early_stop_patience),
        "curriculum_validation_frames": int(effective_curriculum_validation_frames),
        "curriculum_schedule": curriculum_schedule,
        "event_window_frames": int(event_window_frames),
        "newton_iters": int(newton_iters),
        "contact_residual_weight": float(contact_residual_weight),
        "approach_penalty_weight": float(approach_penalty_weight),
        "soft_contact_weight": float(soft_contact_weight),
        "soft_contact_temperature": float(soft_contact_temperature),
        "soft_contact_samples": int(soft_contact_samples),
        "max_free_rollout_events": int(max_free_rollout_events),
        "active_intervals": _active_interval_summary(object_ids, active_metadata),
        "free_rollout_collision_solver": "event_driven_circle_proxy_polynomial_roots",
        "collision_detection_policy": "2d_true_shape_pose_center_line_inside_fraction",
        "contact_proxy_policy": "box_equal_area_circle",
        "pair_parameter_policy": "optimized_pair_for_observed_collision_pairs_else_object_fallback",
        "object_scales_m": scales,
        "object_shapes": {
            object_id: {
                "shape": shape_names[object_index],
                "contact_proxy": contact_proxy_names[object_index],
                "contact_radius_m": float(contact_radii.detach().cpu()[object_index]),
                "half_extents_2d_m": [
                    float(value) for value in half_extents.detach().cpu()[object_index].tolist()
                ],
                "fixed_yaw_rad": float(object_angles.detach().cpu()[object_index]),
            }
            for object_index, object_id in enumerate(object_ids)
        },
        "events": events if event_source == EVENT_SOURCE_DETECTED_WINDOW else model_events,
        "free_rollout_events": free_rollout_events,
        "detected_events": events,
        "stage_rmse_m": float(overall_rmse.cpu()),
        "overall_rmse_m": float(overall_rmse.cpu()),
        "per_object_rmse_m": per_object_rmse,
        "free_rollout_rmse_m": float(free_rollout_rmse.cpu()),
        "free_rollout_per_object_rmse_m": free_rollout_per_object_rmse,
        "free_rollout_plot_path": str(free_rollout_plot_path) if free_rollout_plot_path is not None else None,
        "free_rollout_video_path": str(free_rollout_video_path) if free_rollout_video_path is not None else None,
        "parameters": {
            object_id: {
                "initial_velocity_plane_m_per_s": [float(value) for value in v0.detach().cpu()[object_index].tolist()],
                "ground_friction": float(friction.detach().cpu()[object_index]),
                "mass": float(solved_mass_np[object_index]),
                "restitution": float(solved_restitution_np[object_index]),
            }
            for object_index, object_id in enumerate(object_ids)
        },
        "pair_collision_parameters": [
            {
                "object_ids": [object_ids[i], object_ids[j]],
                "mass_ratio_i_over_j": float(effective_pair_mass_ratio_np[pair_index]),
                "restitution": float(effective_pair_restitution_np[pair_index]),
                "pair_parameter_source": pair_parameter_sources[pair_index],
                "direct_pair_parameter_available": direct_pair_parameter_available[pair_index],
                "optimized_pair_mass_ratio_i_over_j": (
                    float(pair_mass_ratio_np[pair_index])
                    if direct_pair_parameter_available[pair_index]
                    else None
                ),
                "optimized_pair_restitution": (
                    float(pair_restitution_np[pair_index])
                    if direct_pair_parameter_available[pair_index]
                    else None
                ),
                "solved_mass_ratio_i_over_j": float(solved_mass_np[i] / solved_mass_np[j]),
                "solved_pair_restitution_product": float(solved_restitution_np[i] * solved_restitution_np[j]),
                "used_in_object_parameter_solve": direct_pair_parameter_available[pair_index],
            }
            for pair_index, (i, j) in enumerate(object_pairs)
        ],
        "object_parameter_solve": {
            "mass": mass_solve_diagnostics,
            "restitution": restitution_solve_diagnostics,
        },
        "mass_ratio_equation_residuals": mass_ratio_equation_residuals,
        "restitution_equation_residuals": restitution_equation_residuals,
        "event_velocity_summary": [],
        "trace": trace,
        "stage_plot_path": str(plot_path) if plot_path is not None else None,
        "stage_video_path": str(video_path) if video_path is not None else None,
        "plot_path": str(plot_path) if plot_path is not None else None,
        "video_path": str(video_path) if video_path is not None else None,
    }
    for event in payload["events"]:
        event_index = int(event["event_index"])
        record = event_records_by_index[event_index]
        object_indices = [int(index) for index in record["object_indices"]]
        coarse_abs = record.get("coarse_abs_contact_residual_m")
        coarse_relative_normal = record.get("coarse_relative_normal_velocity_m_per_s")
        event_pair_index = None
        event_pair_parameter_source = None
        event_effective_mass_ratio_pair_convention = None
        event_effective_mass_ratio_event_order = None
        event_effective_pair_restitution = None
        if len(object_indices) == 2:
            event_pair_key = tuple(sorted(object_indices))
            event_pair_index = pair_index_by_key.get(event_pair_key)
            if event_pair_index is not None:
                event_pair = object_pairs[event_pair_index]
                ratio_pair_convention = float(effective_pair_mass_ratio_np[event_pair_index])
                event_effective_mass_ratio_pair_convention = ratio_pair_convention
                event_effective_mass_ratio_event_order = (
                    ratio_pair_convention
                    if tuple(object_indices) == event_pair
                    else 1.0 / max(ratio_pair_convention, 1e-12)
                )
                event_effective_pair_restitution = float(effective_pair_restitution_np[event_pair_index])
                event_pair_parameter_source = pair_parameter_sources[event_pair_index]
        payload["event_velocity_summary"].append(
            {
                "event_index": event_index,
                "event_type": event["event_type"],
                "frame": int(event["frame"]),
                "object_ids": event["object_ids"],
                "event_source": event["source"],
                "contact_model": str(record.get("contact_model", event.get("contact_model", "unknown"))),
                "pair_index": event_pair_index,
                "pair_parameter_source": event_pair_parameter_source,
                "effective_mass_ratio_i_over_j_pair_convention": event_effective_mass_ratio_pair_convention,
                "effective_mass_ratio_i_over_j_event_order": event_effective_mass_ratio_event_order,
                "effective_pair_restitution": event_effective_pair_restitution,
                "event_time_s": float(record["event_time_s"].detach().cpu()),
                "event_frame_float": float(record["event_frame_float"].detach().cpu()),
                "contact_residual_m": float(record["contact_residual_m"].detach().cpu()),
                "coarse_abs_contact_residual_m": (
                    float(coarse_abs.detach().cpu()) if isinstance(coarse_abs, torch.Tensor) else None
                ),
                "coarse_relative_normal_velocity_m_per_s": (
                    float(coarse_relative_normal.detach().cpu())
                    if isinstance(coarse_relative_normal, torch.Tensor)
                    else None
                ),
                "normal_plane": [float(value) for value in record["normal"].detach().cpu().tolist()],
                "relative_normal_velocity_m_per_s": float(record["relative_normal_velocity"].detach().cpu()),
                "impulse_plane": [float(value) for value in record["impulse"].detach().cpu().tolist()],
                "pre_event_velocity_plane_m_per_s": {
                    object_ids[index]: [float(value) for value in record["pre_velocity"].detach().cpu()[index].tolist()]
                    for index in object_indices
                },
                "post_event_velocity_plane_m_per_s": {
                    object_ids[index]: [float(value) for value in record["post_velocity"].detach().cpu()[index].tolist()]
                    for index in object_indices
                },
            }
        )
    (output_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _pair_values_from_analytic_result(
    result: dict[str, Any],
    object_ids: list[str],
    object_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    pair_index_by_key = {tuple(pair): pair_index for pair_index, pair in enumerate(object_pairs)}
    pair_mass_ratio = np.ones((len(object_pairs),), dtype=np.float64)
    pair_restitution = np.zeros((len(object_pairs),), dtype=np.float64)
    for item in result.get("pair_collision_parameters", []):
        item_ids = [str(value) for value in item.get("object_ids", [])]
        if len(item_ids) != 2 or item_ids[0] not in object_ids or item_ids[1] not in object_ids:
            continue
        item_indices = [object_ids.index(item_ids[0]), object_ids.index(item_ids[1])]
        pair_key = tuple(sorted(item_indices))
        pair_index = pair_index_by_key.get(pair_key)
        if pair_index is None:
            continue
        pair_i, pair_j = object_pairs[pair_index]
        ratio = float(item.get("mass_ratio_i_over_j", 1.0))
        if [object_ids[pair_i], object_ids[pair_j]] != item_ids:
            ratio = 1.0 / max(ratio, 1e-12)
        pair_mass_ratio[pair_index] = ratio
        pair_restitution[pair_index] = float(item.get("restitution", 0.0))
    return pair_mass_ratio, pair_restitution


def _rollout_from_analytic_result(
    fit: dict[str, Any],
    result: dict[str, Any],
) -> tuple[list[str], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if str(result.get("model")) != "analytic-event-collision-v1":
        raise ValueError("result_json must point to an analytic-event-collision-v1 result")
    object_ids, target_plane = _project_target_to_plane(fit)
    _validate_supported_shapes(fit, object_ids)
    frames, target, mask = _target_tensor(object_ids, target_plane)
    fps = float(fit["trajectory_physics_initialization"]["fps"])
    scales = _object_scales(fit, object_ids, target_plane)
    shape_ids, _half_extents, _object_angles, contact_radii, _shape_names, _contact_proxy_names = _shape_metadata(
        fit=fit,
        object_ids=object_ids,
        object_scales_by_id=scales,
    )
    active_metadata = _active_metadata(object_ids, target_plane, frames, target, fps)
    object_index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    events = result.get("detected_events") or result.get("events") or []
    event_specs = _build_event_specs(events, object_index_by_id)
    object_pairs = _object_pairs(len(object_ids))
    pair_index_by_key = {tuple(pair): pair_index for pair_index, pair in enumerate(object_pairs)}
    pair_mass_ratio, pair_restitution = _pair_values_from_analytic_result(result, object_ids, object_pairs)
    parameters = result.get("parameters", {})
    v0 = torch.tensor(
        [parameters[object_id]["initial_velocity_plane_m_per_s"] for object_id in object_ids],
        dtype=torch.float64,
    )
    friction = torch.tensor(
        [float(parameters[object_id]["ground_friction"]) for object_id in object_ids],
        dtype=torch.float64,
    )
    predicted, _velocities, _event_records = _analytic_rollout(
        target=target,
        frames=frames,
        event_specs=event_specs,
        pair_index_by_key=pair_index_by_key,
        contact_radii=contact_radii,
        shape_ids=shape_ids,
        v0=v0,
        friction=friction,
        pair_mass_ratio=torch.tensor(pair_mass_ratio, dtype=torch.float64),
        pair_restitution=torch.tensor(pair_restitution, dtype=torch.float64),
        fps=fps,
        event_source=str(result.get("event_source", EVENT_SOURCE_FREE_ROLLOUT)),
        event_window_frames=int(result.get("event_window_frames", 2)),
        newton_iters=int(result.get("newton_iters", 8)),
        object_pairs=object_pairs,
        max_free_rollout_events=int(result.get("max_free_rollout_events", 0)),
        active_metadata=active_metadata,
    )
    return (
        object_ids,
        frames,
        target.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        mask.detach().cpu().numpy(),
        contact_radii.detach().cpu().numpy(),
        fps,
    )


def render_comparison_video_from_result_json(
    *,
    input_fit: Path,
    result_json: Path,
    output_mp4: Path,
    width: int = 1280,
    height: int = 720,
    fps_override: float | None = None,
) -> None:
    fit = _load_json(input_fit)
    result = _load_json(result_json)
    object_ids, frames, target, predicted, mask, contact_radii, source_fps = _rollout_from_analytic_result(fit, result)
    _render_comparison_video(
        output_mp4,
        object_ids=object_ids,
        frames=frames,
        target=target,
        predicted=predicted,
        mask=mask,
        contact_radii=contact_radii,
        source_fps=source_fps,
        width=width,
        height=height,
        fps_override=fps_override,
    )


def render_analytic_result_debug_artifacts(
    *,
    input_fit: Path,
    result_json: Path,
    output_dir: Path | None = None,
    width: int = 1280,
    height: int = 720,
    fps_override: float | None = None,
) -> dict[str, Any]:
    fit = _load_json(input_fit)
    result = _load_json(result_json)
    object_ids, frames, target, predicted, mask, contact_radii, source_fps = _rollout_from_analytic_result(fit, result)
    render_dir = output_dir if output_dir is not None else result_json.parent
    render_dir.mkdir(parents=True, exist_ok=True)
    stage_name = str(result.get("stage_name") or "analytic_free_rollout")
    plot_path = render_dir / f"{stage_name}.png"
    video_path = plot_path.with_suffix(".mp4")
    _plot(plot_path, object_ids, target, predicted, mask)
    _render_comparison_video(
        video_path,
        object_ids=object_ids,
        frames=frames,
        target=target,
        predicted=predicted,
        mask=mask,
        contact_radii=contact_radii,
        source_fps=source_fps,
        width=width,
        height=height,
        fps_override=fps_override,
        predicted_title="Free rollout",
    )
    result["render_video"] = True
    result["stage_plot_path"] = str(plot_path)
    result["stage_video_path"] = str(video_path)
    result["plot_path"] = str(plot_path)
    result["video_path"] = str(video_path)
    result["free_rollout_plot_path"] = str(plot_path)
    result["free_rollout_video_path"] = str(video_path)
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "input_fit": str(input_fit),
        "result_json": str(result_json),
        "plot_path": str(plot_path),
        "video_path": str(video_path),
        "frame_count": len(frames),
        "object_ids": object_ids,
    }


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _plane_payload(plane: dict[str, Any]) -> dict[str, Any]:
    return {
        "coordinate_frame": "blender_world",
        "origin": np.asarray(plane["origin"], dtype=np.float64).astype(float).tolist(),
        "normal": np.asarray(plane["normal"], dtype=np.float64).astype(float).tolist(),
        "tangent_1": np.asarray(plane["tangent_1"], dtype=np.float64).astype(float).tolist(),
        "tangent_2": np.asarray(plane["tangent_2"], dtype=np.float64).astype(float).tolist(),
    }


def _plane_from_payload(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "origin": np.asarray(payload["origin"], dtype=np.float64).reshape(3),
        "normal": np.asarray(payload["normal"], dtype=np.float64).reshape(3),
        "tangent_1": np.asarray(payload["tangent_1"], dtype=np.float64).reshape(3),
        "tangent_2": np.asarray(payload["tangent_2"], dtype=np.float64).reshape(3),
    }


def build_analytic_input_fit_from_swr_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = _json_clone(manifest)
    target = sysid_common._target_records_from_swr(manifest)
    object_ids = sorted(object_id for object_id, records in target.items() if records)
    if not object_ids:
        raise ValueError("SWR manifest does not contain any usable target trajectories")

    manifest["trajectory_physics_initialization"] = sysid_common._compute_trajectory_physics_initialization(
        manifest,
        target,
    )
    plane = _support_plane_basis(manifest)
    specs = sysid_common._swr_object_specs(manifest)
    dimensions_by_object = {
        object_id: sysid_common._estimate_proxy_dimensions(specs.get(object_id, {}))
        for object_id in object_ids
    }
    radius_by_object = {
        object_id: float(_shape_radius(specs.get(object_id, {}), dimensions_by_object[object_id]))
        for object_id in object_ids
    }
    half_extents_by_object = {
        object_id: [
            float(value)
            for value in _half_extents_2d(
                specs.get(object_id, {}),
                dimensions_by_object[object_id],
            )
        ]
        for object_id in object_ids
    }
    inertia_by_object = {
        object_id: float(_inertia_coefficient_2d(specs.get(object_id, {}), dimensions_by_object[object_id]))
        for object_id in object_ids
    }
    return {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "impulse_analytic_curriculum_input",
        "mode": "trajectory_informed_physics_alignment",
        "static_scene_objects": manifest.get("static_scene_objects", []),
        "target_trajectories": target,
        "trajectory_physics_initialization": manifest.get("trajectory_physics_initialization"),
        "analytic_support_plane": _plane_payload(plane),
        "physics_rollout": {
            "simulator": "impulse_analytic_curriculum_input",
            "activation_policy": manifest.get("activation_policy"),
            "proxy_dimensions_by_object": dimensions_by_object,
            "shape_radius_2d_by_object": radius_by_object,
            "shape_half_extents_2d_by_object": half_extents_by_object,
            "inertia_coefficient_2d_by_object": inertia_by_object,
        },
    }


def _target_plane_records_from_fit(fit: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    plane = _plane_from_payload(fit["analytic_support_plane"])
    output: dict[str, list[dict[str, Any]]] = {}
    for object_id, records in fit.get("target_trajectories", {}).items():
        converted = []
        for record in records:
            xy, height = _to_plane_coords(record["position"], plane)
            converted.append(
                {
                    **record,
                    "plane_xy": xy.astype(float).tolist(),
                    "plane_height": float(height),
                }
            )
        output[str(object_id)] = converted
    return output


def _simulated_trajectories_from_analytic_result(
    fit: dict[str, Any],
    analytic_result: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    object_ids, frames, _target, predicted, mask, _contact_radii, _source_fps = _rollout_from_analytic_result(
        fit,
        analytic_result,
    )
    simulated_2d: dict[str, dict[str, np.ndarray]] = {}
    for object_index, object_id in enumerate(object_ids):
        positions = []
        frame_indices = []
        for frame_offset, frame_index in enumerate(frames):
            if float(mask[frame_offset, object_index, 0]) <= 0.0:
                continue
            positions.append(predicted[frame_offset, object_index].astype(np.float64))
            frame_indices.append(int(frame_index))
        simulated_2d[object_id] = {
            "positions": np.asarray(positions, dtype=np.float64).reshape((-1, 2)),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        }
    return _simulated_trajectories(
        object_ids=object_ids,
        target=fit.get("target_trajectories", {}),
        target_plane=_target_plane_records_from_fit(fit),
        simulated_2d=simulated_2d,
        plane=_plane_from_payload(fit["analytic_support_plane"]),
    )


def build_swr_fit_from_analytic_result(
    *,
    manifest: dict[str, Any],
    input_fit: dict[str, Any],
    input_fit_path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    analytic_result: dict[str, Any],
) -> dict[str, Any]:
    target = input_fit.get("target_trajectories", {})
    simulated = _simulated_trajectories_from_analytic_result(input_fit, analytic_result)
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    loss_context = sysid_common._swr_loss_context(manifest, target, camera_intrinsic)
    fit_error = sysid_common._swr_fit_error(
        simulated,
        target,
        loss_space="world_3d",
        camera_intrinsic=camera_intrinsic,
        loss_context=loss_context,
    )
    result_path = output_dir / "analytic_free_rollout" / "result.json"
    sketch_video_path = analytic_result.get("free_rollout_video_path") or analytic_result.get("video_path")
    sketch_plot_path = analytic_result.get("free_rollout_plot_path") or analytic_result.get("plot_path")
    physics_rollout = dict(input_fit.get("physics_rollout", {}))
    physics_rollout.update(
        {
            "simulator": "impulse_analytic_event_driven_circle_proxy",
            "simulated_trajectories": simulated,
            "analytic_result_path": str(result_path),
            "analytic_summary_path": str(output_dir / "summary.json"),
            "sketch_comparison_plot_path": sketch_plot_path,
            "sketch_comparison_video_path": sketch_video_path,
        }
    )
    return {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.impulse_analytic",
        "mode": "trajectory_informed_physics_alignment",
        "message": "SWR physics alignment completed with impulse analytic curriculum rollout",
        "static_scene_objects": manifest.get("static_scene_objects", []),
        "target_trajectories": target,
        "trajectory_physics_initialization": input_fit.get("trajectory_physics_initialization"),
        "analytic_support_plane": input_fit.get("analytic_support_plane"),
        "physics_rollout": physics_rollout,
        "alignment_optimization": {
            "strategy": "impulse_analytic_curriculum_free_rollout",
            "optimizer": {
                "backend": "impulse_analytic",
                "pipeline": summary.get("pipeline"),
                "method": analytic_result.get("optimizer"),
                "optimizer_details": analytic_result.get("optimizer_details"),
                "analytic_steps": summary.get("analytic_steps"),
                "analytic_lr": summary.get("analytic_lr"),
                "curriculum_prefix_step_frames": analytic_result.get("curriculum_prefix_step_frames"),
                "curriculum_steps_per_prefix": analytic_result.get("curriculum_steps_per_prefix"),
                "curriculum_best_metric": analytic_result.get("curriculum_best_metric"),
                "curriculum_fade_in": analytic_result.get("curriculum_fade_in"),
                "curriculum_early_stop_patience": analytic_result.get("curriculum_early_stop_patience"),
                "event_source": analytic_result.get("event_source"),
                "free_rollout_collision_solver": analytic_result.get("free_rollout_collision_solver"),
            },
            "best_parameters": analytic_result.get("parameters"),
            "pair_collision_parameters": analytic_result.get("pair_collision_parameters"),
            "object_parameter_solve": analytic_result.get("object_parameter_solve"),
            "events": analytic_result.get("events"),
            "free_rollout_events": analytic_result.get("free_rollout_events"),
            "detected_events": analytic_result.get("detected_events"),
            "curriculum_schedule": analytic_result.get("curriculum_schedule"),
            "stage_rmse_m": analytic_result.get("stage_rmse_m"),
            "overall_rmse_m": analytic_result.get("overall_rmse_m"),
            "free_rollout_rmse_m": analytic_result.get("free_rollout_rmse_m"),
            "free_rollout_per_object_rmse_m": analytic_result.get("free_rollout_per_object_rmse_m"),
            "free_rollout_plot_path": analytic_result.get("free_rollout_plot_path"),
            "free_rollout_video_path": analytic_result.get("free_rollout_video_path"),
            "sketch_comparison_video_path": sketch_video_path,
            "input_fit": str(input_fit_path),
            "result_path": str(result_path),
            "summary_path": str(output_dir / "summary.json"),
            "loss_space": "world_3d",
            "loss_context": loss_context,
            "camera_intrinsics": camera_intrinsic,
        },
        "fit_error": fit_error,
    }


def run_swr_manifest_pipeline(
    *,
    manifest_path: Path,
    output_path: Path,
    output_dir: Path,
    analytic_steps: int,
    analytic_lr: float,
    free_rollout_optimizer: str,
    warmup_steps: int | None,
    warmup_prefix_frames: int | None,
    curriculum_prefix_step_frames: int,
    curriculum_steps_per_prefix: int,
    curriculum_best_metric: str,
    curriculum_fade_in: bool,
    curriculum_early_stop_patience: int,
    event_window_frames: int,
    newton_iters: int,
    contact_residual_weight: float,
    approach_penalty_weight: float,
    soft_contact_weight: float,
    soft_contact_temperature: float,
    soft_contact_samples: int,
    max_free_rollout_events: int,
    render_video: bool | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(manifest_path)
    input_fit = build_analytic_input_fit_from_swr_manifest(manifest)
    input_fit_path = output_dir / "analytic_input_fit.json"
    input_fit_path.write_text(json.dumps(input_fit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = run_pipeline(
        input_fit=input_fit_path,
        output_dir=output_dir,
        analytic_steps=analytic_steps,
        analytic_lr=analytic_lr,
        free_rollout_optimizer=free_rollout_optimizer,
        warmup_steps=warmup_steps,
        warmup_prefix_frames=warmup_prefix_frames,
        curriculum_prefix_step_frames=curriculum_prefix_step_frames,
        curriculum_steps_per_prefix=curriculum_steps_per_prefix,
        curriculum_best_metric=curriculum_best_metric,
        curriculum_fade_in=curriculum_fade_in,
        curriculum_early_stop_patience=curriculum_early_stop_patience,
        event_window_frames=event_window_frames,
        newton_iters=newton_iters,
        contact_residual_weight=contact_residual_weight,
        approach_penalty_weight=approach_penalty_weight,
        soft_contact_weight=soft_contact_weight,
        soft_contact_temperature=soft_contact_temperature,
        soft_contact_samples=soft_contact_samples,
        max_free_rollout_events=max_free_rollout_events,
        render_video=render_video,
    )
    analytic_result = _load_json(output_dir / "analytic_free_rollout" / "result.json")
    swr_fit = build_swr_fit_from_analytic_result(
        manifest=manifest,
        input_fit=input_fit,
        input_fit_path=input_fit_path,
        output_dir=output_dir,
        summary=summary,
        analytic_result=analytic_result,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(swr_fit, ensure_ascii=False, indent=2), encoding="utf-8")
    return swr_fit


def _parse_float_csv(value: str | None) -> list[float]:
    if value is None:
        return []
    values: list[float] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        values.append(float(item))
    deduped: list[float] = []
    for item in values:
        if not any(abs(item - existing) <= 1e-12 for existing in deduped):
            deduped.append(item)
    return deduped


def _lr_tag(lr: float) -> str:
    return f"lr_{float(lr):.8g}".replace("-", "m").replace(".", "p")


def _set_torch_worker_threads(thread_count: int) -> None:
    effective_count = max(int(thread_count), 1)
    os.environ["OMP_NUM_THREADS"] = str(effective_count)
    os.environ["MKL_NUM_THREADS"] = str(effective_count)
    torch.set_num_threads(effective_count)
    try:
        torch.set_num_interop_threads(max(1, min(effective_count, 4)))
    except RuntimeError:
        pass


def _fit_metric_from_swr_result(result: dict[str, Any]) -> float:
    fit_error = result.get("fit_error") if isinstance(result, dict) else None
    if isinstance(fit_error, dict):
        for key in ("overall_translation_rmse", "optimization_loss"):
            value = fit_error.get(key)
            if value is not None and math.isfinite(float(value)):
                return float(value)
    return float("inf")


def _rewrite_json_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_json_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_json_paths(item, replacements) for item in value]
    if isinstance(value, str):
        rewritten = value
        for src, dst in replacements.items():
            rewritten = rewritten.replace(src, dst)
        return rewritten
    return value


def _run_swr_manifest_lr_trial(payload: dict[str, Any]) -> dict[str, Any]:
    _set_torch_worker_threads(int(payload["worker_threads"]))
    lr = float(payload["analytic_lr"])
    output_dir = Path(payload["output_dir"])
    output_path = output_dir / str(payload["output_name"])
    result = run_swr_manifest_pipeline(
        manifest_path=Path(payload["manifest_path"]),
        output_path=output_path,
        output_dir=output_dir,
        analytic_steps=int(payload["analytic_steps"]),
        analytic_lr=lr,
        free_rollout_optimizer=str(payload["free_rollout_optimizer"]),
        warmup_steps=payload["warmup_steps"],
        warmup_prefix_frames=payload["warmup_prefix_frames"],
        curriculum_prefix_step_frames=int(payload["curriculum_prefix_step_frames"]),
        curriculum_steps_per_prefix=int(payload["curriculum_steps_per_prefix"]),
        curriculum_best_metric=str(payload["curriculum_best_metric"]),
        curriculum_fade_in=bool(payload["curriculum_fade_in"]),
        curriculum_early_stop_patience=int(payload["curriculum_early_stop_patience"]),
        event_window_frames=int(payload["event_window_frames"]),
        newton_iters=int(payload["newton_iters"]),
        contact_residual_weight=float(payload["contact_residual_weight"]),
        approach_penalty_weight=float(payload["approach_penalty_weight"]),
        soft_contact_weight=float(payload["soft_contact_weight"]),
        soft_contact_temperature=float(payload["soft_contact_temperature"]),
        soft_contact_samples=int(payload["soft_contact_samples"]),
        max_free_rollout_events=int(payload["max_free_rollout_events"]),
        render_video=False,
    )
    return {
        "analytic_lr": lr,
        "status": result.get("status"),
        "backend": result.get("backend"),
        "output": str(output_path),
        "output_dir": str(output_dir),
        "fit_error": result.get("fit_error"),
        "selection_metric": _fit_metric_from_swr_result(result),
    }


def run_swr_manifest_pipeline_multistart(
    *,
    manifest_path: Path,
    output_path: Path,
    output_dir: Path,
    analytic_steps: int,
    analytic_lrs: list[float],
    multistart_workers: int,
    multistart_threads: int,
    free_rollout_optimizer: str,
    warmup_steps: int | None,
    warmup_prefix_frames: int | None,
    curriculum_prefix_step_frames: int,
    curriculum_steps_per_prefix: int,
    curriculum_best_metric: str,
    curriculum_fade_in: bool,
    curriculum_early_stop_patience: int,
    event_window_frames: int,
    newton_iters: int,
    contact_residual_weight: float,
    approach_penalty_weight: float,
    soft_contact_weight: float,
    soft_contact_temperature: float,
    soft_contact_samples: int,
    max_free_rollout_events: int,
    render_video: bool | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_root = output_dir / "lr_multistart"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    payload_base = {
        "manifest_path": str(manifest_path),
        "output_name": output_path.name,
        "analytic_steps": int(analytic_steps),
        "free_rollout_optimizer": str(free_rollout_optimizer),
        "warmup_steps": warmup_steps,
        "warmup_prefix_frames": warmup_prefix_frames,
        "curriculum_prefix_step_frames": int(curriculum_prefix_step_frames),
        "curriculum_steps_per_prefix": int(curriculum_steps_per_prefix),
        "curriculum_best_metric": str(curriculum_best_metric),
        "curriculum_fade_in": bool(curriculum_fade_in),
        "curriculum_early_stop_patience": int(curriculum_early_stop_patience),
        "event_window_frames": int(event_window_frames),
        "newton_iters": int(newton_iters),
        "contact_residual_weight": float(contact_residual_weight),
        "approach_penalty_weight": float(approach_penalty_weight),
        "soft_contact_weight": float(soft_contact_weight),
        "soft_contact_temperature": float(soft_contact_temperature),
        "soft_contact_samples": int(soft_contact_samples),
        "max_free_rollout_events": int(max_free_rollout_events),
        "worker_threads": max(int(multistart_threads), 1),
    }
    payloads: list[dict[str, Any]] = []
    for lr in analytic_lrs:
        payload = dict(payload_base)
        payload["analytic_lr"] = float(lr)
        payload["output_dir"] = str(trial_root / _lr_tag(float(lr)))
        payloads.append(payload)

    trial_results, worker_count = analytic_swr_common.run_parallel_trials(
        payloads=payloads,
        worker=_run_swr_manifest_lr_trial,
        max_workers=multistart_workers,
    )
    trial_results.sort(key=lambda item: analytic_lrs.index(float(item["analytic_lr"])))
    valid_trials = [
        trial
        for trial in trial_results
        if trial.get("status") == "ok" and math.isfinite(float(trial.get("selection_metric", float("inf"))))
    ]
    if not valid_trials:
        raise RuntimeError(f"all analytic LR multi-start trials failed: {trial_results}")
    best_trial = min(valid_trials, key=lambda item: float(item["selection_metric"]))
    selected_lr = float(best_trial["analytic_lr"])
    multistart_summary = {
        "enabled": True,
        "candidate_lrs": [float(lr) for lr in analytic_lrs],
        "workers": int(worker_count),
        "worker_threads": max(int(multistart_threads), 1),
        "selection_metric": "fit_error.overall_translation_rmse",
        "selected_lr": selected_lr,
        "selected_trial": best_trial,
        "trials": trial_results,
    }

    best_output_dir = Path(str(best_trial["output_dir"]))
    best_output_path = Path(str(best_trial["output"]))
    if not best_output_path.exists():
        raise FileNotFoundError(f"missing selected LR trial output: {best_output_path}")

    manifest = _load_json(manifest_path)
    keep_names = {"lr_multistart", manifest_path.name}
    for child in output_dir.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in best_output_dir.iterdir():
        if child.name == output_path.name:
            continue
        destination = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)

    replacements = {str(best_output_dir): str(output_dir)}
    input_fit_path = output_dir / "analytic_input_fit.json"
    analytic_result_path = output_dir / "analytic_free_rollout" / "result.json"
    render_result = None
    if _render_video_enabled(render_video):
        render_result = render_analytic_result_debug_artifacts(
            input_fit=input_fit_path,
            result_json=analytic_result_path,
        )

    input_fit = _load_json(input_fit_path)
    final_summary = _rewrite_json_paths(_load_json(output_dir / "summary.json"), replacements)
    final_summary["analytic_lr"] = selected_lr
    final_summary["analytic_lr_multistart"] = multistart_summary
    if render_result is not None:
        final_summary["debug_artifacts"] = {
            **(final_summary.get("debug_artifacts") if isinstance(final_summary.get("debug_artifacts"), dict) else {}),
            "sketch_comparison": render_result,
        }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_result = build_swr_fit_from_analytic_result(
        manifest=manifest,
        input_fit=input_fit,
        input_fit_path=input_fit_path,
        output_dir=output_dir,
        summary=final_summary,
        analytic_result=_load_json(analytic_result_path),
    )
    final_result["analytic_lr_multistart"] = multistart_summary
    output_path.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

    (trial_root / "summary.json").write_text(
        json.dumps(multistart_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_result


def _stage_summary(result: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    if result is None:
        return {"status": "failed", "error": error}
    return {
        "status": "ok",
        "stage_name": result.get("stage_name"),
        "stage_rmse_m": result.get("stage_rmse_m"),
        "overall_rmse_m": result.get("overall_rmse_m"),
        "free_rollout_rmse_m": result.get("free_rollout_rmse_m"),
        "initial_analytic_rmse_m": result.get("initial_analytic_rmse_m"),
        "render_video": result.get("render_video"),
        "stage_plot_path": result.get("stage_plot_path"),
        "stage_video_path": result.get("stage_video_path"),
        "free_rollout_plot_path": result.get("free_rollout_plot_path"),
        "free_rollout_video_path": result.get("free_rollout_video_path"),
        "plot_path": result.get("plot_path"),
        "video_path": result.get("video_path"),
        "result_path": str(Path(str(result.get("plot_path", ""))).parent / "result.json") if result.get("plot_path") else None,
        "error": error,
    }


def run_pipeline(
    *,
    input_fit: Path,
    output_dir: Path,
    analytic_steps: int,
    analytic_lr: float,
    free_rollout_optimizer: str,
    warmup_steps: int | None,
    warmup_prefix_frames: int | None,
    curriculum_prefix_step_frames: int,
    curriculum_steps_per_prefix: int,
    curriculum_best_metric: str,
    curriculum_fade_in: bool,
    curriculum_early_stop_patience: int,
    event_window_frames: int,
    newton_iters: int,
    contact_residual_weight: float,
    approach_penalty_weight: float,
    soft_contact_weight: float,
    soft_contact_temperature: float,
    soft_contact_samples: int,
    max_free_rollout_events: int,
    render_video: bool | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_video_flag = _render_video_enabled(render_video)
    free_rollout_output_dir = output_dir / "analytic_free_rollout"
    free_rollout_result = run(
        input_fit=input_fit,
        output_dir=free_rollout_output_dir,
        init_from_impulse_forced_frame_result=None,
        init_from_analytic_result=None,
        steps=int(analytic_steps),
        lr=float(analytic_lr),
        analytic_optimizer=free_rollout_optimizer,
        event_source=EVENT_SOURCE_FREE_ROLLOUT,
        warmup_steps=warmup_steps,
        warmup_prefix_frames=warmup_prefix_frames,
        curriculum_prefix_step_frames=curriculum_prefix_step_frames,
        curriculum_steps_per_prefix=curriculum_steps_per_prefix,
        curriculum_best_metric=curriculum_best_metric,
        curriculum_fade_in=curriculum_fade_in,
        curriculum_early_stop_patience=curriculum_early_stop_patience,
        event_window_frames=event_window_frames,
        newton_iters=newton_iters,
        contact_residual_weight=contact_residual_weight,
        approach_penalty_weight=approach_penalty_weight,
        soft_contact_weight=soft_contact_weight,
        soft_contact_temperature=soft_contact_temperature,
        soft_contact_samples=soft_contact_samples,
        max_free_rollout_events=max_free_rollout_events,
        render_video=render_video_flag,
    )
    summary = {
        "pipeline": PIPELINE_FULL,
        "input_fit": str(input_fit),
        "output_dir": str(output_dir),
        "analytic_steps": int(analytic_steps),
        "analytic_lr": float(analytic_lr),
        "free_rollout_optimizer": free_rollout_optimizer,
        "curriculum_prefix_step_frames": int(curriculum_prefix_step_frames),
        "curriculum_steps_per_prefix": int(curriculum_steps_per_prefix),
        "curriculum_best_metric": str(curriculum_best_metric),
        "curriculum_fade_in": bool(curriculum_fade_in),
        "curriculum_early_stop_patience": int(curriculum_early_stop_patience),
        "curriculum_validation_frames": (
            int(curriculum_prefix_step_frames)
            if int(curriculum_prefix_step_frames) > 0
            and curriculum_best_metric == CURRICULUM_BEST_NEXT_WINDOW_RMSE
            else 0
        ),
        "render_video": render_video_flag,
        "stages": {
            "analytic_free_rollout": _stage_summary(free_rollout_result),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_three_stage_pipeline(
    *,
    input_fit: Path,
    output_dir: Path,
    impulse_mode: str,
    impulse_steps: int,
    impulse_lr: float,
    impulse_optimizer: str,
    contact_assumption: str,
    analytic_steps: int,
    analytic_lr: float,
    analytic_forced_window_optimizer: str,
    free_rollout_optimizer: str,
    event_window_frames: int,
    newton_iters: int,
    contact_residual_weight: float,
    approach_penalty_weight: float,
    soft_contact_weight: float,
    soft_contact_temperature: float,
    soft_contact_samples: int,
    max_free_rollout_events: int,
    render_video: bool | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_video_flag = _render_video_enabled(render_video)
    impulse_stage_name = _impulse_stage_name(impulse_mode)
    impulse_output_dir = output_dir / impulse_stage_name
    forced_window_output_dir = output_dir / "analytic_forced_window"
    free_rollout_output_dir = output_dir / "analytic_free_rollout"

    impulse_result = _run_impulse_stage(
        input_fit=input_fit,
        output_dir=impulse_output_dir,
        steps=int(impulse_steps),
        lr=float(impulse_lr),
        impulse_optimizer=impulse_optimizer,
        mode=impulse_mode,
        contact_assumption=contact_assumption,
        event_window_frames=event_window_frames,
        newton_iters=newton_iters,
        render_video=render_video_flag,
    )
    impulse_result_path = impulse_output_dir / "result.json"

    forced_window_result = run(
        input_fit=input_fit,
        output_dir=forced_window_output_dir,
        init_from_impulse_forced_frame_result=impulse_result_path,
        init_from_analytic_result=None,
        steps=int(analytic_steps),
        lr=float(analytic_lr),
        analytic_optimizer=analytic_forced_window_optimizer,
        event_source=EVENT_SOURCE_DETECTED_WINDOW,
        warmup_steps=0,
        warmup_prefix_frames=0,
        curriculum_prefix_step_frames=0,
        curriculum_steps_per_prefix=0,
        curriculum_best_metric=CURRICULUM_BEST_PREFIX_LOSS,
        curriculum_fade_in=False,
        curriculum_early_stop_patience=0,
        event_window_frames=event_window_frames,
        newton_iters=newton_iters,
        contact_residual_weight=contact_residual_weight,
        approach_penalty_weight=approach_penalty_weight,
        soft_contact_weight=soft_contact_weight,
        soft_contact_temperature=soft_contact_temperature,
        soft_contact_samples=soft_contact_samples,
        max_free_rollout_events=max_free_rollout_events,
        render_video=render_video_flag,
    )
    forced_window_result_path = forced_window_output_dir / "result.json"

    free_rollout_result = run(
        input_fit=input_fit,
        output_dir=free_rollout_output_dir,
        init_from_impulse_forced_frame_result=None,
        init_from_analytic_result=forced_window_result_path,
        steps=int(analytic_steps),
        lr=float(analytic_lr),
        analytic_optimizer=free_rollout_optimizer,
        event_source=EVENT_SOURCE_FREE_ROLLOUT,
        warmup_steps=0,
        warmup_prefix_frames=0,
        curriculum_prefix_step_frames=0,
        curriculum_steps_per_prefix=0,
        curriculum_best_metric=CURRICULUM_BEST_PREFIX_LOSS,
        curriculum_fade_in=False,
        curriculum_early_stop_patience=0,
        event_window_frames=event_window_frames,
        newton_iters=newton_iters,
        contact_residual_weight=contact_residual_weight,
        approach_penalty_weight=approach_penalty_weight,
        soft_contact_weight=soft_contact_weight,
        soft_contact_temperature=soft_contact_temperature,
        soft_contact_samples=soft_contact_samples,
        max_free_rollout_events=max_free_rollout_events,
        render_video=render_video_flag,
    )

    summary = {
        "pipeline": PIPELINE_THREE_STAGE,
        "input_fit": str(input_fit),
        "output_dir": str(output_dir),
        "impulse_mode": str(impulse_mode),
        "impulse_steps": int(impulse_steps),
        "impulse_lr": float(impulse_lr),
        "impulse_optimizer": str(impulse_optimizer),
        "contact_assumption": str(contact_assumption),
        "analytic_steps": int(analytic_steps),
        "analytic_lr": float(analytic_lr),
        "analytic_forced_window_optimizer": str(analytic_forced_window_optimizer),
        "free_rollout_optimizer": str(free_rollout_optimizer),
        "event_window_frames": int(event_window_frames),
        "newton_iters": int(newton_iters),
        "render_video": render_video_flag,
        "collision_detection_policy": "true_2d_shape_pose_center_line_inside_fraction",
        "contact_proxy_policy": "box_equal_area_circle_for_analytic_rollout_only",
        "stage_order": [
            str(impulse_stage_name),
            "analytic_forced_window",
            "analytic_free_rollout",
        ],
        "stages": {
            str(impulse_stage_name): _stage_summary(impulse_result),
            "analytic_forced_window": _stage_summary(forced_window_result),
            "analytic_free_rollout": _stage_summary(free_rollout_result),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fit", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pipeline", default=PIPELINE_FULL, choices=PIPELINES)
    parser.add_argument("--init-from-impulse-forced-frame-result", default=None)
    parser.add_argument("--init-from-analytic-result", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--impulse-mode", default=MODE_IMPULSE_FORCED_FRAME, choices=IMPULSE_STAGE_MODES)
    parser.add_argument("--impulse-steps", type=int, default=500)
    parser.add_argument("--impulse-lr", type=float, default=2e-2)
    parser.add_argument("--impulse-optimizer", default=IMPULSE_OPTIMIZER_LEAST_SQUARES, choices=IMPULSE_OPTIMIZERS)
    parser.add_argument("--contact-assumption", default="known_time", choices=CONTACT_ASSUMPTIONS)
    parser.add_argument("--analytic-steps", type=int, default=500)
    parser.add_argument("--analytic-lr", type=float, default=2.5e-2)
    parser.add_argument("--analytic-lr-multistart", default="0.01,0.025,0.05")
    parser.add_argument("--analytic-lr-multistart-workers", type=int, default=3)
    parser.add_argument("--analytic-lr-multistart-threads", type=int, default=4)
    parser.add_argument("--analytic-optimizer", default=None, choices=ANALYTIC_OPTIMIZERS)
    parser.add_argument("--free-rollout-optimizer", default=ANALYTIC_OPTIMIZER_ADAM, choices=ANALYTIC_OPTIMIZERS)
    parser.add_argument("--event-source", default=EVENT_SOURCE_FREE_ROLLOUT, choices=EVENT_SOURCES)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-prefix-frames", type=int, default=0)
    parser.add_argument("--curriculum-prefix-step-frames", type=int, default=10)
    parser.add_argument("--curriculum-steps-per-prefix", type=int, default=400)
    parser.add_argument("--curriculum-best-metric", default=CURRICULUM_BEST_NEXT_WINDOW_RMSE, choices=CURRICULUM_BEST_METRICS)
    parser.add_argument("--curriculum-fade-in", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-early-stop-patience", type=int, default=100)
    parser.add_argument("--event-window-frames", type=int, default=2)
    parser.add_argument("--newton-iters", type=int, default=8)
    parser.add_argument("--contact-residual-weight", type=float, default=0.0)
    parser.add_argument("--approach-penalty-weight", type=float, default=0.0)
    parser.add_argument("--soft-contact-weight", type=float, default=0.0)
    parser.add_argument("--soft-contact-temperature", type=float, default=0.02)
    parser.add_argument("--soft-contact-samples", type=int, default=128)
    parser.add_argument("--max-free-rollout-events", type=int, default=0)
    parser.add_argument("--render-video", action="store_true", default=None)
    args = parser.parse_args()
    if args.manifest is not None:
        if args.output is None:
            raise ValueError("--output is required when --manifest is provided")
        output_path = Path(args.output)
        output_dir = Path(args.output_dir) if args.output_dir is not None else output_path.parent
        multistart_lrs = _parse_float_csv(args.analytic_lr_multistart)
        if len(multistart_lrs) > 1:
            result = run_swr_manifest_pipeline_multistart(
                manifest_path=Path(args.manifest),
                output_path=output_path,
                output_dir=output_dir,
                analytic_steps=args.analytic_steps,
                analytic_lrs=multistart_lrs,
                multistart_workers=args.analytic_lr_multistart_workers,
                multistart_threads=args.analytic_lr_multistart_threads,
                free_rollout_optimizer=args.free_rollout_optimizer,
                warmup_steps=args.warmup_steps,
                warmup_prefix_frames=args.warmup_prefix_frames,
                curriculum_prefix_step_frames=args.curriculum_prefix_step_frames,
                curriculum_steps_per_prefix=args.curriculum_steps_per_prefix,
                curriculum_best_metric=args.curriculum_best_metric,
                curriculum_fade_in=args.curriculum_fade_in,
                curriculum_early_stop_patience=args.curriculum_early_stop_patience,
                event_window_frames=args.event_window_frames,
                newton_iters=args.newton_iters,
                contact_residual_weight=args.contact_residual_weight,
                approach_penalty_weight=args.approach_penalty_weight,
                soft_contact_weight=args.soft_contact_weight,
                soft_contact_temperature=args.soft_contact_temperature,
                soft_contact_samples=args.soft_contact_samples,
                max_free_rollout_events=args.max_free_rollout_events,
                render_video=args.render_video,
            )
        else:
            effective_lr = float(multistart_lrs[0]) if multistart_lrs else float(args.analytic_lr)
            result = run_swr_manifest_pipeline(
                manifest_path=Path(args.manifest),
                output_path=output_path,
                output_dir=output_dir,
                analytic_steps=args.analytic_steps,
                analytic_lr=effective_lr,
                free_rollout_optimizer=args.free_rollout_optimizer,
                warmup_steps=args.warmup_steps,
                warmup_prefix_frames=args.warmup_prefix_frames,
                curriculum_prefix_step_frames=args.curriculum_prefix_step_frames,
                curriculum_steps_per_prefix=args.curriculum_steps_per_prefix,
                curriculum_best_metric=args.curriculum_best_metric,
                curriculum_fade_in=args.curriculum_fade_in,
                curriculum_early_stop_patience=args.curriculum_early_stop_patience,
                event_window_frames=args.event_window_frames,
                newton_iters=args.newton_iters,
                contact_residual_weight=args.contact_residual_weight,
                approach_penalty_weight=args.approach_penalty_weight,
                soft_contact_weight=args.soft_contact_weight,
                soft_contact_temperature=args.soft_contact_temperature,
                soft_contact_samples=args.soft_contact_samples,
                max_free_rollout_events=args.max_free_rollout_events,
                render_video=args.render_video,
            )
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "backend": result.get("backend"),
                    "output": str(output_path),
                    "fit_error": result.get("fit_error"),
                    "analytic_lr_multistart": (
                        {
                            "selected_lr": result["analytic_lr_multistart"].get("selected_lr"),
                            "candidate_lrs": result["analytic_lr_multistart"].get("candidate_lrs"),
                            "workers": result["analytic_lr_multistart"].get("workers"),
                        }
                        if result.get("analytic_lr_multistart")
                        else None
                    ),
                },
                indent=2,
            )
        )
        return
    if args.input_fit is None or args.output_dir is None:
        raise ValueError("--input-fit and --output-dir are required unless --manifest is provided")
    if args.analytic_optimizer is not None:
        analytic_optimizer = args.analytic_optimizer
    elif args.pipeline == PIPELINE_STAGE and args.event_source == EVENT_SOURCE_FREE_ROLLOUT:
        analytic_optimizer = ANALYTIC_OPTIMIZER_ADAM
    else:
        analytic_optimizer = ANALYTIC_OPTIMIZER_LEAST_SQUARES
    if args.pipeline == PIPELINE_FULL:
        result = run_pipeline(
            input_fit=Path(args.input_fit),
            output_dir=Path(args.output_dir),
            analytic_steps=args.analytic_steps,
            analytic_lr=args.analytic_lr,
            free_rollout_optimizer=args.free_rollout_optimizer,
            warmup_steps=args.warmup_steps,
            warmup_prefix_frames=args.warmup_prefix_frames,
            curriculum_prefix_step_frames=args.curriculum_prefix_step_frames,
            curriculum_steps_per_prefix=args.curriculum_steps_per_prefix,
            curriculum_best_metric=args.curriculum_best_metric,
            curriculum_fade_in=args.curriculum_fade_in,
            curriculum_early_stop_patience=args.curriculum_early_stop_patience,
            event_window_frames=args.event_window_frames,
            newton_iters=args.newton_iters,
            contact_residual_weight=args.contact_residual_weight,
            approach_penalty_weight=args.approach_penalty_weight,
            soft_contact_weight=args.soft_contact_weight,
            soft_contact_temperature=args.soft_contact_temperature,
            soft_contact_samples=args.soft_contact_samples,
            max_free_rollout_events=args.max_free_rollout_events,
            render_video=args.render_video,
        )
        print(json.dumps(result, indent=2))
        return
    if args.pipeline == PIPELINE_THREE_STAGE:
        result = run_three_stage_pipeline(
            input_fit=Path(args.input_fit),
            output_dir=Path(args.output_dir),
            impulse_mode=args.impulse_mode,
            impulse_steps=args.impulse_steps,
            impulse_lr=args.impulse_lr,
            impulse_optimizer=args.impulse_optimizer,
            contact_assumption=args.contact_assumption,
            analytic_steps=args.analytic_steps,
            analytic_lr=args.analytic_lr,
            analytic_forced_window_optimizer=analytic_optimizer,
            free_rollout_optimizer=args.free_rollout_optimizer,
            event_window_frames=args.event_window_frames,
            newton_iters=args.newton_iters,
            contact_residual_weight=args.contact_residual_weight,
            approach_penalty_weight=args.approach_penalty_weight,
            soft_contact_weight=args.soft_contact_weight,
            soft_contact_temperature=args.soft_contact_temperature,
            soft_contact_samples=args.soft_contact_samples,
            max_free_rollout_events=args.max_free_rollout_events,
            render_video=args.render_video,
        )
        print(json.dumps(result, indent=2))
        return
    result = run(
        input_fit=Path(args.input_fit),
        output_dir=Path(args.output_dir),
        init_from_impulse_forced_frame_result=(
            Path(args.init_from_impulse_forced_frame_result)
            if args.init_from_impulse_forced_frame_result is not None
            else None
        ),
        init_from_analytic_result=Path(args.init_from_analytic_result) if args.init_from_analytic_result is not None else None,
        steps=args.steps,
        lr=args.lr,
        analytic_optimizer=analytic_optimizer,
        event_source=args.event_source,
        warmup_steps=args.warmup_steps,
        warmup_prefix_frames=args.warmup_prefix_frames,
        curriculum_prefix_step_frames=args.curriculum_prefix_step_frames,
        curriculum_steps_per_prefix=args.curriculum_steps_per_prefix,
        curriculum_best_metric=args.curriculum_best_metric,
        curriculum_fade_in=args.curriculum_fade_in,
        curriculum_early_stop_patience=args.curriculum_early_stop_patience,
        event_window_frames=args.event_window_frames,
        newton_iters=args.newton_iters,
        contact_residual_weight=args.contact_residual_weight,
        approach_penalty_weight=args.approach_penalty_weight,
        soft_contact_weight=args.soft_contact_weight,
        soft_contact_temperature=args.soft_contact_temperature,
        soft_contact_samples=args.soft_contact_samples,
        max_free_rollout_events=args.max_free_rollout_events,
        render_video=args.render_video,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model",
                    "optimizer",
                    "stage_name",
                    "init_from_impulse_forced_frame_result",
                    "init_from_analytic_result",
                    "source_impulse_forced_frame_stage_rmse_m",
                    "source_impulse_forced_frame_free_rollout_rmse_m",
                    "source_initialization_stage_rmse_m",
                    "source_initialization_stage_free_rollout_rmse_m",
                    "source_initialization_stage_event_source",
                    "initial_analytic_rmse_m",
                    "event_source",
                    "warmup_steps",
                    "warmup_prefix_frames",
                    "curriculum_prefix_step_frames",
                    "curriculum_steps_per_prefix",
                    "curriculum_best_metric",
                    "curriculum_fade_in",
                    "curriculum_early_stop_patience",
                    "curriculum_validation_frames",
                    "curriculum_schedule",
                    "render_video",
                    "stage_rmse_m",
                    "overall_rmse_m",
                    "free_rollout_rmse_m",
                    "per_object_rmse_m",
                    "free_rollout_per_object_rmse_m",
                    "stage_plot_path",
                    "stage_video_path",
                    "free_rollout_plot_path",
                    "free_rollout_video_path",
                    "plot_path",
                    "video_path",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
