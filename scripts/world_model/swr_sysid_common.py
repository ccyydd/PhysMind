from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

OPENCV_CAMERA_TO_BLENDER_WORLD = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

SWR_LOSS_SPACES = {"image_2d", "mixed_2d_3d", "world_3d"}
MIXED_LOSS_2D_WEIGHT = 0.5
MIXED_LOSS_3D_WEIGHT = 0.5
COLLISION_INITIAL_RADIUS_ENVELOPE_SCALE = 0.75


def raw_from_interval(value: float, lower: float, upper: float) -> float:
    import numpy as np

    clipped = float(
        np.clip(
            (float(value) - lower) / max(upper - lower, 1e-12),
            1e-6,
            1.0 - 1e-6,
        )
    )
    return float(math.log(clipped / (1.0 - clipped)))


def velocity_raw_from_value(
    value: Any,
    *,
    maximum_speed_m_per_s: float,
) -> Any:
    import numpy as np
    import torch

    normalized = np.clip(
        np.asarray(value, dtype=np.float64) / float(maximum_speed_m_per_s),
        -0.999999,
        0.999999,
    )
    return torch.tensor(
        np.arctanh(normalized),
        dtype=torch.float64,
        requires_grad=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_collision_pose_context(
    world_modeling_dir: Path,
    *,
    context_key: str,
    context_required_error: str,
    support_required_error: str,
    bindings_required_error: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pose_path = (
        world_modeling_dir
        / "pose-estimation-and-tracking"
        / "pose_correction"
        / "pose_correction.json"
    )
    if not pose_path.exists():
        raise ValueError(f"missing pose_correction: {pose_path}")
    pose_correction = _load_json(pose_path)
    if not isinstance(pose_correction, dict):
        raise ValueError(f"invalid pose_correction: {pose_path}")
    context = pose_correction.get(context_key)
    support = pose_correction.get("support_plane_position_correction")
    if not isinstance(context, dict) or context.get("applies") is not True:
        raise ValueError(context_required_error)
    if not isinstance(support, dict) or support.get("applied") is not True:
        raise ValueError(support_required_error)
    roles = context.get("role_object_ids")
    split = context.get("two_segment")
    if not isinstance(roles, dict) or not isinstance(split, dict):
        raise ValueError(bindings_required_error)
    return context, support, roles, split


def build_manifest_from_world_modeling(world_modeling_dir: Path) -> dict[str, Any]:
    manifest_path = (
        world_modeling_dir
        / "simulatable-world-reconstruction"
        / "fit"
        / "world_reconstruction_fit_manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(
            "missing formal SWR fit manifest; offline Physion++ SWR does not "
            f"reconstruct geometry inputs from upstream artifacts: {manifest_path}"
        )
    loaded = _load_json(manifest_path)
    if not isinstance(loaded, dict):
        raise ValueError(f"invalid formal SWR fit manifest: {manifest_path}")
    manifest = json.loads(json.dumps(loaded))
    route = manifest.get("swr_fit_geometry_source_route")
    if not isinstance(route, dict):
        raise ValueError(
            "formal SWR fit manifest is missing "
            "swr_fit_geometry_source_route"
        )
    if route.get("decision_id") != "SWR-003.fit_geometry_source":
        raise ValueError(
            "formal SWR fit manifest has an unexpected geometry-source "
            f"decision_id: {route.get('decision_id')!r}"
        )
    if route.get("route") != "geometry.corrected_mesh":
        raise ValueError(
            "unsupported formal SWR fit geometry source: "
            f"{route.get('route')!r}"
        )
    target = manifest.get("target_trajectories")
    if not isinstance(target, dict) or not isinstance(target.get("objects"), list):
        raise ValueError(
            "formal SWR fit manifest has no target_trajectories.objects"
        )
    for item in target["objects"]:
        if not isinstance(item, dict):
            raise ValueError(
                "formal SWR fit manifest contains a non-object target record"
            )
        object_id = str(item.get("object_id") or "").strip()
        if not object_id:
            raise ValueError(
                "formal SWR fit manifest contains a target without object_id"
            )
        raw_mesh_path = item.get("mesh_path")
        if not raw_mesh_path:
            raise ValueError(
                "missing pose-correction final effective mesh_path for "
                f"SWR target object {object_id}"
            )
        mesh_path = Path(str(raw_mesh_path))
        if not mesh_path.is_absolute():
            raise ValueError(
                "pose-correction final effective mesh_path must be absolute for "
                f"SWR target object {object_id}: {mesh_path}"
            )
        if not mesh_path.is_file():
            raise ValueError(
                "pose-correction final effective mesh does not exist for "
                f"SWR target object {object_id}: {mesh_path}"
            )
        try:
            with mesh_path.open("rb") as handle:
                first_byte = handle.read(1)
        except OSError as exc:
            raise ValueError(
                "failed to read pose-correction final effective mesh for "
                f"SWR target object {object_id}: {mesh_path}"
            ) from exc
        if not first_byte:
            raise ValueError(
                "pose-correction final effective mesh is empty for "
                f"SWR target object {object_id}: {mesh_path}"
            )
        item["mesh_source"] = "geometry.corrected_mesh"
    manifest["mode"] = "physionpp_bouncy_wall_offline_sysid"
    manifest["source_world_reconstruction_fit_manifest"] = str(manifest_path)
    video_metric_depth_path = (
        world_modeling_dir
        / "metric-mesh-reconstruction"
        / "video_metric_depth"
        / "video_metric_depth.json"
    )
    if video_metric_depth_path.exists():
        video_metric_depth = _load_json(video_metric_depth_path)
        if isinstance(video_metric_depth, dict):
            for key in ("fixed_intrinsics", "camera_intrinsics", "K"):
                if key in video_metric_depth:
                    manifest[key] = video_metric_depth[key]
                    break
            metadata = {}
            for key in ("width", "height", "fps", "frame_count"):
                if key in video_metric_depth:
                    metadata[key] = video_metric_depth[key]
            if metadata:
                manifest["video_metadata"] = metadata
    return manifest


def collision_oriented_box_geometry_by_object(
    *,
    object_specs: dict[str, dict[str, Any]],
    object_ids: list[str] | set[str],
    mesh_loader: Any,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    geometries: dict[str, dict[str, Any]] = {}
    for object_id in sorted(set(object_ids)):
        mesh_path = (object_specs.get(object_id) or {}).get("mesh_path")
        if not mesh_path or not Path(str(mesh_path)).exists():
            raise ValueError(f"missing collision mesh for {object_id}: {mesh_path}")
        mesh = mesh_loader(Path(str(mesh_path)))
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if not len(vertices):
            raise ValueError(f"empty collision mesh for {object_id}: {mesh_path}")
        bounds_min = np.min(vertices, axis=0)
        bounds_max = np.max(vertices, axis=0)
        half_extents = 0.5 * (bounds_max - bounds_min)
        if not np.all(np.isfinite(half_extents)) or np.any(half_extents <= 1e-6):
            raise ValueError(
                f"degenerate collision bounds for {object_id}: {half_extents.tolist()}"
            )
        geometries[object_id] = {
            "type": "oriented_box",
            "local_center_m": (0.5 * (bounds_min + bounds_max)).astype(float).tolist(),
            "half_extents_m": half_extents.astype(float).tolist(),
            "source": "conditioned_mesh_local_axis_aligned_bounds",
            "source_mesh_path": str(mesh_path),
        }
    return geometries


def shared_role_radii_from_mesh_envelopes(
    *,
    role_object_ids: dict[str, list[str]],
    object_role_assignments: list[tuple[str, str]],
    fallback_radius_by_object: dict[str, float],
    object_specs: dict[str, dict[str, Any]],
    mesh_loader: Any,
    envelope_scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    import numpy as np

    individual: dict[str, float] = {}
    for object_id, _role in object_role_assignments:
        mesh_path = (object_specs.get(object_id) or {}).get("mesh_path")
        if not mesh_path or not Path(str(mesh_path)).exists():
            individual[object_id] = float(fallback_radius_by_object[object_id])
            continue
        mesh = mesh_loader(Path(str(mesh_path)))
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        radius = (
            float(envelope_scale) * float(np.max(np.linalg.norm(vertices, axis=1)))
            if len(vertices)
            else 0.0
        )
        individual[object_id] = (
            radius
            if math.isfinite(radius) and radius > 1e-3
            else float(fallback_radius_by_object[object_id])
        )
    shared = {
        role: float(np.median([individual[object_id] for object_id in object_ids]))
        for role, object_ids in role_object_ids.items()
        if object_ids
    }
    by_object: dict[str, float] = {}
    for object_id, role in object_role_assignments:
        by_object[object_id] = shared[role]
    return shared, by_object


def records_in_frame_range(
    target: dict[str, list[dict[str, Any]]],
    object_id: str,
    frame_range: list[int],
) -> list[dict[str, Any]]:
    start_frame, end_frame = frame_range
    return sorted(
        [
            record
            for record in target.get(object_id, [])
            if start_frame <= int(record["frame_index"]) <= end_frame
        ],
        key=lambda record: int(record["frame_index"]),
    )


def _collision_unit_vector(vector: Any) -> Any:
    import numpy as np

    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("expected a finite non-zero direction")
    return vector / norm


def collision_support_geometry(
    *,
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    support_context: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    normal_camera = support_context.get("normal_camera")
    if not isinstance(normal_camera, list) or len(normal_camera) != 3:
        raise ValueError("pose correction support plane has no normal_camera")
    ground_height = support_context.get("ground_height_along_normal")
    if not isinstance(ground_height, (int, float)) or not math.isfinite(
        float(ground_height)
    ):
        raise ValueError(
            "pose correction support plane has no finite ground_height_along_normal"
        )
    normal_blender, point_blender = _support_plane_in_blender(manifest)
    up = _collision_unit_vector(np.asarray(normal_blender, dtype=np.float64))
    plane_point = np.asarray(point_blender, dtype=np.float64).reshape(3)
    plane_offset = float(np.dot(plane_point, up))
    gravity_direction = -up
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(up)))] = 1.0
    tangent_1 = _collision_unit_vector(np.cross(up, reference))
    tangent_2 = _collision_unit_vector(np.cross(up, tangent_1))
    return {
        "up": up,
        "gravity_direction": gravity_direction,
        "tangent_1": tangent_1,
        "tangent_2": tangent_2,
        "plane_point": plane_point,
        "plane_offset": plane_offset,
        "source": "pose_correction.support_plane_position_correction",
        "segment_names": [str(segment["segment"]) for segment in segments],
        "pose_correction_ground_height_camera_m": float(ground_height),
    }


def fit_tangential_friction(
    *,
    frames: list[int],
    positions: Any,
    up: Any,
    physics_dt_sec: float,
    gravity_magnitude: float,
    min_friction: float,
    max_friction: float,
    default_friction: float = 0.1,
) -> float:
    import numpy as np

    if len(frames) < 5:
        return float(default_friction)
    times = (np.asarray(frames, dtype=np.float64) - float(frames[0])) * float(
        physics_dt_sec
    )
    values = np.asarray(positions, dtype=np.float64)
    tangential = values - np.outer(values @ up, up)
    design = np.stack([times, 0.5 * times * times], axis=1)
    coefficients, *_ = np.linalg.lstsq(
        design,
        tangential - tangential[0],
        rcond=None,
    )
    velocity = np.asarray(coefficients[0], dtype=np.float64)
    acceleration = np.asarray(coefficients[1], dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-8:
        return float(default_friction)
    deceleration = -float(np.dot(acceleration, velocity / speed))
    if not math.isfinite(deceleration) or deceleration <= 0.0:
        return float(default_friction)
    return float(
        np.clip(
            deceleration / float(gravity_magnitude),
            float(min_friction),
            float(max_friction),
        )
    )


def fit_initial_velocity(
    frames: list[int],
    positions: Any,
    *,
    physics_dt_sec: float,
    max_frames: int = 12,
) -> Any:
    import numpy as np

    count = min(len(frames), int(max_frames))
    if count < 2:
        return np.zeros(3, dtype=np.float64)
    times = (np.asarray(frames[:count], dtype=np.float64) - float(frames[0])) * float(
        physics_dt_sec
    )
    values = np.asarray(positions[:count], dtype=np.float64)
    if count < 3:
        return (values[-1] - values[0]) / max(float(times[-1]), 1e-12)
    design = np.stack([times, 0.5 * times * times], axis=1)
    coefficients, *_ = np.linalg.lstsq(
        design,
        values - values[0],
        rcond=None,
    )
    return np.asarray(coefficients[0], dtype=np.float64)


def direction_angles_from_value(value: Any, fallback: Any) -> Any:
    import numpy as np
    import torch

    direction = np.asarray(value, dtype=np.float64)
    if float(np.linalg.norm(direction)) < 1e-6:
        direction = np.asarray(fallback, dtype=np.float64)
    if float(np.linalg.norm(direction)) < 1e-6:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return torch.tensor(
        [
            math.atan2(float(direction[1]), float(direction[0])),
            math.atan2(
                float(direction[2]),
                math.hypot(float(direction[0]), float(direction[1])),
            ),
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def _is_intrinsic_matrix(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _camera_intrinsics_from_manifest(manifest: dict[str, Any]) -> list[list[float]] | None:
    for key in ("fixed_intrinsics", "camera_intrinsics", "K"):
        value = manifest.get(key)
        if _is_intrinsic_matrix(value):
            return [[float(item) for item in row] for row in value]

    artifact_path = manifest.get("pose_correction_artifact") or manifest.get("target_trajectories_artifact")
    if not artifact_path:
        return None
    path = Path(str(artifact_path))
    for parent in [path.parent, *path.parents]:
        candidate = parent / "metric-mesh-reconstruction" / "video_metric_depth" / "video_metric_depth.json"
        if not candidate.exists():
            continue
        try:
            payload = _load_json(candidate)
        except Exception:
            continue
        for key in ("fixed_intrinsics", "camera_intrinsics", "K"):
            value = payload.get(key)
            if _is_intrinsic_matrix(value):
                return [[float(item) for item in row] for row in value]
    return None


def _opencv_point_to_blender_world(position: list[float]) -> list[float]:
    x, y, z = [float(value) for value in position]
    return [x, z, -y]


def _opencv_vector_to_blender_world(vector: list[float]) -> list[float]:
    x, y, z = [float(value) for value in vector]
    return [x, z, -y]


def _blender_vector_to_opencv_camera(vector: list[float]) -> list[float]:
    x, y, z = [float(value) for value in vector]
    return [x, -z, y]


def _opencv_pose_to_blender_world(pose: list[list[float]]) -> list[list[float]]:
    transform = OPENCV_CAMERA_TO_BLENDER_WORLD

    def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [
            [
                sum(float(a[row][k]) * float(b[k][col]) for k in range(4))
                for col in range(4)
            ]
            for row in range(4)
        ]

    return matmul(transform, pose)


def _target_records_from_swr(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target = manifest.get("target_trajectories") if isinstance(manifest.get("target_trajectories"), dict) else {}
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for item in target.get("objects", []):
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        object_id = str(item.get("object_id") or "")
        if not object_id:
            continue
        records = []
        for pose in item.get("poses", []):
            matrix = pose.get("corrected_pose_4x4") if isinstance(pose, dict) else None
            if not matrix or pose.get("frame_index") is None:
                continue
            position = [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
            records.append(
                {
                    "frame_index": int(pose["frame_index"]),
                    "pose_4x4": matrix,
                    "position_opencv_camera": position,
                    "position": _opencv_point_to_blender_world(position),
                }
            )
        trajectories[object_id] = sorted(records, key=lambda record: int(record["frame_index"]))
    return trajectories


def _video_fps_from_manifest(manifest: dict[str, Any]) -> float:
    metadata = manifest.get("video_metadata") if isinstance(manifest.get("video_metadata"), dict) else {}
    for key in ("fps", "frame_rate"):
        try:
            fps = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and fps > 0.0:
            return fps
    return 24.0


def _gravity_axis_camera_from_manifest(manifest: dict[str, Any]) -> Any:
    import numpy as np

    value = manifest.get("gravity_direction_camera")
    if isinstance(value, list) and len(value) == 3:
        try:
            axis = np.asarray(value, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(axis))
            if math.isfinite(norm) and norm > 1e-12:
                return axis / norm
        except (TypeError, ValueError):
            pass
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _fit_constant_acceleration_initialization_segment(
    *,
    records: list[dict[str, Any]],
    fps: float,
    gravity_axis_camera: Any,
) -> dict[str, Any]:
    import numpy as np

    first = records[0]
    x0 = np.asarray(first["position_opencv_camera"], dtype=np.float64).reshape(3)
    if len(records) < 3:
        last = records[-1]
        dt = max((int(last["frame_index"]) - int(first["frame_index"])) / max(float(fps), 1e-12), 1e-12)
        velocity = (np.asarray(last["position_opencv_camera"], dtype=np.float64).reshape(3) - x0) / dt
        acceleration = np.zeros(3, dtype=np.float64)
    else:
        times = np.asarray(
            [
                (int(record["frame_index"]) - int(first["frame_index"])) / max(float(fps), 1e-12)
                for record in records
            ],
            dtype=np.float64,
        )
        positions = np.asarray([record["position_opencv_camera"] for record in records], dtype=np.float64)
        design = np.stack([times, 0.5 * times * times], axis=1)
        coeff, *_ = np.linalg.lstsq(design, positions - x0.reshape(1, 3), rcond=None)
        velocity = coeff[0]
        acceleration = coeff[1]

    gravity_axis = np.asarray(gravity_axis_camera, dtype=np.float64).reshape(3)
    velocity = velocity - float(np.dot(velocity, gravity_axis)) * gravity_axis
    acceleration = acceleration - float(np.dot(acceleration, gravity_axis)) * gravity_axis
    duration = max((int(records[-1]["frame_index"]) - int(records[0]["frame_index"])) / max(float(fps), 1e-12), 0.0)
    mid_velocity = velocity + acceleration * (0.5 * duration)
    speed = float(np.linalg.norm(mid_velocity))
    deceleration_mu = None
    if speed > 1e-8:
        tangent = mid_velocity / speed
        deceleration = -float(np.dot(acceleration, tangent))
        if math.isfinite(deceleration) and deceleration > 0.0:
            deceleration_mu = float(deceleration / 9.81)

    return {
        "initial_position_camera": x0.astype(float).tolist(),
        "initial_velocity_camera_m_per_s": velocity.astype(float).tolist(),
        "acceleration_camera_m_per_s2": acceleration.astype(float).tolist(),
        "tangential_deceleration_over_g": deceleration_mu,
    }


def _fit_motion_initialization(
    *,
    records: list[dict[str, Any]],
    fps: float,
    gravity_axis_camera: Any,
) -> dict[str, Any]:
    if len(records) < 2:
        return {
            "initial_velocity": {
                "value": [0.0, 0.0, 0.0],
                "source": "insufficient_trajectory_default_zero",
                "confidence": 0.0,
            },
            "friction": {
                "value": 0.0,
                "source": "insufficient_trajectory_default_zero",
                "confidence": 0.0,
                "sample_count": 0,
            },
            "segments": [],
        }

    fit = _fit_constant_acceleration_initialization_segment(
        records=records,
        fps=fps,
        gravity_axis_camera=gravity_axis_camera,
    )
    frame_count = int(len(records))
    deceleration_mu = fit["tangential_deceleration_over_g"]
    friction_value = float(deceleration_mu) if deceleration_mu is not None else 0.0
    friction_source = (
        "swr_sysid_common_full_trajectory_tangential_deceleration"
        if deceleration_mu is not None
        else "zero_no_decelerating_motion"
    )
    segment = {
        "segment_index": 0,
        "start_frame": int(records[0]["frame_index"]),
        "end_frame": int(records[-1]["frame_index"]),
        "frame_count": frame_count,
        "initial_position_camera": fit["initial_position_camera"],
        "initial_velocity_camera_m_per_s": fit["initial_velocity_camera_m_per_s"],
        "acceleration_camera_m_per_s2": fit["acceleration_camera_m_per_s2"],
        "tangential_deceleration_over_g": deceleration_mu,
        "normal_component_removed": True,
        "normal_axis_camera": [float(value) for value in gravity_axis_camera.tolist()],
        "source": "swr_sysid_common_constant_acceleration_fit_x_eq_x0_plus_v0_t_plus_half_a_t2",
    }
    return {
        "initial_velocity": {
            "value": [float(value) for value in fit["initial_velocity_camera_m_per_s"]],
            "source": "swr_sysid_common_full_trajectory_constant_acceleration_fit",
            "confidence": min(1.0, max(0.2, float(frame_count) / 12.0)),
            "normal_component_removed": True,
            "normal_axis_camera": [float(value) for value in gravity_axis_camera.tolist()],
        },
        "friction": {
            "value": friction_value,
            "source": friction_source,
            "confidence": 0.5 if deceleration_mu is not None else 0.2,
            "sample_count": frame_count if deceleration_mu is not None else 0,
            "weight": "full_trajectory_frame_count",
        },
        "segments": [segment],
    }


def _compute_trajectory_physics_initialization(manifest: dict[str, Any], target: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not target:
        return {
            "applied": False,
            "reason": "missing target trajectories",
            "objects": [],
        }

    fps = _video_fps_from_manifest(manifest)
    gravity_axis_camera = _gravity_axis_camera_from_manifest(manifest)
    target_objects = manifest.get("target_trajectories", {}).get("objects", []) if isinstance(manifest.get("target_trajectories"), dict) else []
    target_by_id = {
        str(item.get("object_id")): item
        for item in target_objects
        if isinstance(item, dict) and item.get("object_id")
    }

    objects = []
    for object_id, records in sorted(target.items()):
        source = target_by_id.get(object_id, {})
        motion = _fit_motion_initialization(
            records=records,
            fps=fps,
            gravity_axis_camera=gravity_axis_camera,
        )
        objects.append(
            {
                "object_id": object_id,
                "status": "ok" if records else "missing_trajectory",
                "activation": source.get("activation"),
                "mesh_path": source.get("mesh_path"),
                "geometry_type": source.get("geometry_type"),
                "initial_frame_index": int(records[0]["frame_index"]) if records else None,
                "initial_pose_4x4": records[0]["pose_4x4"] if records else None,
                "initial_position_camera": records[0]["position_opencv_camera"] if records else None,
                "initial_velocity_camera_m_per_s": motion["initial_velocity"],
                "motion_segments": motion["segments"],
                "friction": motion["friction"],
                "mass": {
                    "value": 1.0,
                    "source": "normalized_default",
                    "confidence": 0.2,
                },
                "restitution": {
                    "value": 0.4,
                    "source": "default",
                    "confidence": 0.2,
                },
            }
        )

    return {
        "applied": bool(objects),
        "source": "swr_sysid_common_constant_acceleration_fit_from_target_trajectories",
        "fps": fps,
        "gravity_magnitude_m_per_s2": 9.81,
        "gravity_direction_camera": [float(value) for value in gravity_axis_camera.tolist()],
        "objects": objects,
    }


def _swr_object_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    target = manifest.get("target_trajectories") if isinstance(manifest.get("target_trajectories"), dict) else {}
    init = (
        manifest.get("trajectory_physics_initialization")
        if isinstance(manifest.get("trajectory_physics_initialization"), dict)
        else {}
    )
    specs = {
        str(item.get("object_id")): dict(item)
        for item in target.get("objects", [])
        if isinstance(item, dict) and item.get("object_id")
    }
    for item in init.get("objects", []):
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        specs.setdefault(str(item["object_id"]), {}).update({"physics_initialization": item})
    return specs


def _validated_dimensions(values: Any, *, source: str) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        dimensions = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"invalid proxy dimensions from {source}: {values!r}")
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        raise ValueError(f"invalid proxy dimensions from {source}: {values!r}")
    return [max(value, 0.02) for value in dimensions]


def _estimate_proxy_dimensions(spec: dict[str, Any]) -> list[float]:
    manifest_dimensions = _validated_dimensions(spec.get("dimensions"), source="manifest")
    if manifest_dimensions is not None:
        return manifest_dimensions
    mesh_path = spec.get("mesh_path")
    if not mesh_path or not Path(str(mesh_path)).exists():
        object_id = spec.get("object_id") or "unknown"
        raise ValueError(f"missing mesh_path/dimensions for object {object_id}")
    try:
        import numpy as np
        import trimesh

        mesh_or_scene = trimesh.load(str(mesh_path), force="scene")
        if hasattr(mesh_or_scene, "dump"):
            meshes = [mesh for mesh in mesh_or_scene.dump() if hasattr(mesh, "bounds")]
            if meshes:
                bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
                min_corner = bounds[:, 0, :].min(axis=0)
                max_corner = bounds[:, 1, :].max(axis=0)
                dims = max_corner - min_corner
                return _validated_dimensions(dims.tolist(), source=str(mesh_path)) or []
        if hasattr(mesh_or_scene, "bounds"):
            dims = np.asarray(mesh_or_scene.bounds[1], dtype=float) - np.asarray(mesh_or_scene.bounds[0], dtype=float)
            return _validated_dimensions(dims.tolist(), source=str(mesh_path)) or []
    except Exception as exc:
        object_id = spec.get("object_id") or "unknown"
        raise RuntimeError(f"failed to estimate proxy dimensions for object {object_id} from {mesh_path}: {exc}") from exc
    object_id = spec.get("object_id") or "unknown"
    raise ValueError(f"failed to estimate proxy dimensions for object {object_id} from {mesh_path}")


def _project_opencv_point(point: list[float], intrinsic: list[list[float]]) -> list[float] | None:
    x, y, z = [float(value) for value in point]
    if z <= 1e-9:
        return None
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    return [fx * x / z + cx, fy * y / z + cy]


def _record_position_opencv(record: dict[str, Any]) -> list[float] | None:
    value = record.get("position_opencv_camera")
    if isinstance(value, list) and len(value) == 3:
        return [float(item) for item in value]
    position = record.get("position")
    if isinstance(position, list) and len(position) == 3:
        return _blender_vector_to_opencv_camera([float(item) for item in position])
    return None


def _target_world_bbox_diagonal(target: dict[str, list[dict[str, Any]]]) -> float:
    positions = [
        [float(value) for value in record["position"]]
        for records in target.values()
        for record in records
        if isinstance(record.get("position"), list) and len(record["position"]) == 3
    ]
    if not positions:
        return 1.0
    mins = [min(position[axis] for position in positions) for axis in range(3)]
    maxs = [max(position[axis] for position in positions) for axis in range(3)]
    diagonal = math.sqrt(sum((maxs[axis] - mins[axis]) ** 2 for axis in range(3)))
    return max(float(diagonal), 1e-6)


def _image_diagonal_from_manifest(
    manifest: dict[str, Any],
    camera_intrinsic: list[list[float]] | None,
) -> float:
    metadata = manifest.get("video_metadata") if isinstance(manifest.get("video_metadata"), dict) else {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0.0 and height > 0.0:
        return math.sqrt(width * width + height * height)
    if camera_intrinsic is not None:
        width = 2.0 * abs(float(camera_intrinsic[0][2]))
        height = 2.0 * abs(float(camera_intrinsic[1][2]))
        if width > 0.0 and height > 0.0:
            return math.sqrt(width * width + height * height)
    return 1.0


def _swr_loss_context(
    manifest: dict[str, Any],
    target: dict[str, list[dict[str, Any]]],
    camera_intrinsic: list[list[float]] | None,
) -> dict[str, Any]:
    return {
        "image_diagonal_px": _image_diagonal_from_manifest(manifest, camera_intrinsic),
        "world_bbox_diagonal_m": _target_world_bbox_diagonal(target),
        "mixed_2d_weight": MIXED_LOSS_2D_WEIGHT,
        "mixed_3d_weight": MIXED_LOSS_3D_WEIGHT,
    }


def _loss_context_value(loss_context: dict[str, Any] | None, key: str, default: float) -> float:
    if isinstance(loss_context, dict) and loss_context.get(key) is not None:
        return max(float(loss_context[key]), 1e-12)
    return max(float(default), 1e-12)


def _swr_fit_error(
    simulated: dict[str, list[dict[str, Any]]],
    target: dict[str, list[dict[str, Any]]],
    *,
    loss_space: str,
    camera_intrinsic: list[list[float]] | None,
    loss_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if loss_space not in SWR_LOSS_SPACES:
        raise ValueError(f"unsupported SWR loss space: {loss_space}")
    if loss_space in {"image_2d", "mixed_2d_3d"} and camera_intrinsic is None:
        raise ValueError(f"SWR {loss_space} loss requires camera intrinsics")
    image_scale = _loss_context_value(loss_context, "image_diagonal_px", 1.0)
    world_scale = _loss_context_value(loss_context, "world_bbox_diagonal_m", _target_world_bbox_diagonal(target))
    mixed_2d_weight = _loss_context_value(loss_context, "mixed_2d_weight", MIXED_LOSS_2D_WEIGHT)
    mixed_3d_weight = _loss_context_value(loss_context, "mixed_3d_weight", MIXED_LOSS_3D_WEIGHT)
    per_object = {}
    weighted_sum = 0.0
    weighted_count = 0
    first_frame_errors = []
    early_errors = []
    reprojection_weighted_sum = 0.0
    reprojection_weighted_count = 0
    reprojection_first_frame_errors = []
    reprojection_early_errors = []
    for object_id, target_records in target.items():
        sim_by_frame = {int(record["frame_index"]): record for record in simulated.get(object_id, [])}
        squared = []
        reprojection_squared = []
        first_frame = int(target_records[0]["frame_index"]) if target_records else None
        for record in target_records:
            frame = int(record["frame_index"])
            if frame not in sim_by_frame:
                continue
            target_position = record["position"]
            sim_position = sim_by_frame[frame]["position"]
            try:
                err2 = sum((float(target_position[i]) - float(sim_position[i])) ** 2 for i in range(3))
            except (OverflowError, ValueError):
                err2 = float("inf")
            squared.append(err2)
            reprojection_err2 = None
            if camera_intrinsic is not None:
                target_opencv = _record_position_opencv(record)
                sim_opencv = _record_position_opencv(sim_by_frame[frame])
                if target_opencv is not None and sim_opencv is not None:
                    target_pixel = _project_opencv_point(target_opencv, camera_intrinsic)
                    sim_pixel = _project_opencv_point(sim_opencv, camera_intrinsic)
                    if target_pixel is not None and sim_pixel is not None:
                        reprojection_err2 = sum((target_pixel[i] - sim_pixel[i]) ** 2 for i in range(2))
                        reprojection_squared.append(reprojection_err2)
            if first_frame is not None and frame == first_frame:
                first_frame_errors.append(err2)
                if reprojection_err2 is not None:
                    reprojection_first_frame_errors.append(reprojection_err2)
            if first_frame is not None and frame <= first_frame + 10:
                early_errors.append(err2)
                if reprojection_err2 is not None:
                    reprojection_early_errors.append(reprojection_err2)
        rmse = math.sqrt(sum(squared) / len(squared)) if squared else None
        reprojection_rmse = math.sqrt(sum(reprojection_squared) / len(reprojection_squared)) if reprojection_squared else None
        per_object[object_id] = {
            "translation_rmse": rmse,
            "reprojection_rmse_px": reprojection_rmse,
            "normalized_translation_rmse": None if rmse is None else float(rmse) / world_scale,
            "normalized_reprojection_rmse": None if reprojection_rmse is None else float(reprojection_rmse) / image_scale,
            "sample_count": len(squared),
            "reprojection_sample_count": len(reprojection_squared),
        }
        if squared:
            weighted_sum += sum(squared)
            weighted_count += len(squared)
        if reprojection_squared:
            reprojection_weighted_sum += sum(reprojection_squared)
            reprojection_weighted_count += len(reprojection_squared)
    overall = math.sqrt(weighted_sum / weighted_count) if weighted_count else None
    first_rmse = math.sqrt(sum(first_frame_errors) / len(first_frame_errors)) if first_frame_errors else None
    early_rmse = math.sqrt(sum(early_errors) / len(early_errors)) if early_errors else None
    reprojection_overall = math.sqrt(reprojection_weighted_sum / reprojection_weighted_count) if reprojection_weighted_count else None
    reprojection_first_rmse = math.sqrt(sum(reprojection_first_frame_errors) / len(reprojection_first_frame_errors)) if reprojection_first_frame_errors else None
    reprojection_early_rmse = math.sqrt(sum(reprojection_early_errors) / len(reprojection_early_errors)) if reprojection_early_errors else None
    normalized_translation_overall = None if overall is None else float(overall) / world_scale
    normalized_reprojection_overall = None if reprojection_overall is None else float(reprojection_overall) / image_scale
    mixed_loss = None
    if normalized_translation_overall is not None and normalized_reprojection_overall is not None:
        mixed_loss = mixed_2d_weight * normalized_reprojection_overall + mixed_3d_weight * normalized_translation_overall
    if loss_space == "image_2d":
        optimization_loss_name = "overall_reprojection_rmse_px"
        optimization_loss = reprojection_overall
    elif loss_space == "mixed_2d_3d":
        optimization_loss_name = "mixed_normalized_2d_3d_loss"
        optimization_loss = mixed_loss
    else:
        optimization_loss_name = "overall_translation_rmse"
        optimization_loss = overall
    return {
        "loss_space": loss_space,
        "optimization_loss_name": optimization_loss_name,
        "optimization_loss": optimization_loss,
        "mixed_normalized_2d_3d_loss": mixed_loss,
        "mixed_loss_weights": {
            "normalized_2d": mixed_2d_weight,
            "normalized_3d": mixed_3d_weight,
        },
        "loss_normalization": {
            "image_diagonal_px": image_scale,
            "world_bbox_diagonal_m": world_scale,
        },
        "overall_translation_rmse": overall,
        "normalized_overall_translation_rmse": normalized_translation_overall,
        "first_active_frame_translation_rmse": first_rmse,
        "first_10_active_frames_translation_rmse": early_rmse,
        "overall_reprojection_rmse_px": reprojection_overall,
        "normalized_overall_reprojection_rmse": normalized_reprojection_overall,
        "first_active_frame_reprojection_rmse_px": reprojection_first_rmse,
        "first_10_active_frames_reprojection_rmse_px": reprojection_early_rmse,
        "per_object_translation_rmse": per_object,
    }


def _unit_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(value) / norm for value in vector]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _support_plane_in_blender(manifest: dict[str, Any]) -> tuple[list[float], list[float]]:
    support = manifest.get("support_plane_position_correction")
    if isinstance(support, dict) and isinstance(support.get("normal_camera"), list):
        normal_camera = [float(value) for value in support.get("normal_camera", [0.0, -1.0, 0.0])]
        height = float(support.get("ground_height_along_normal") or 0.0)
    else:
        normal_camera = [0.0, -1.0, 0.0]
        height = 0.0
    normal_blender = _unit_vector(_opencv_vector_to_blender_world(normal_camera))
    point_camera = [height * value for value in normal_camera]
    point_blender = _opencv_point_to_blender_world(point_camera)
    return normal_blender, point_blender


def _horizontal_plane_motion_applies(manifest: dict[str, Any]) -> bool:
    object_plan = manifest.get("object_plan")
    if not isinstance(object_plan, dict):
        return False
    special_scene = object_plan.get("special_scene")
    if not isinstance(special_scene, dict):
        return False
    horizontal = special_scene.get("horizontal_plane_motion")
    return isinstance(horizontal, dict) and horizontal.get("applies") is True


def _velocity_tangent_basis(manifest: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    if not _horizontal_plane_motion_applies(manifest):
        return None
    support_normal, _ = _support_plane_in_blender(manifest)
    normal = _unit_vector(support_normal)
    reference = [1.0, 0.0, 0.0] if abs(_dot(normal, [1.0, 0.0, 0.0])) < 0.9 else [0.0, 1.0, 0.0]
    tangent_1 = _unit_vector(_cross(normal, reference))
    tangent_2 = _unit_vector(_cross(normal, tangent_1))
    return tangent_1, tangent_2
