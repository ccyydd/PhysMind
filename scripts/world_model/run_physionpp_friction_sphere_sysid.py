from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import trimesh

from scripts.world_model import analytic_swr_common
from scripts.world_model import swr_sysid_common as sysid_common


GRAVITY_M_PER_S2 = 9.81
PHYSIONPP_PHYSICS_DT_SEC = 0.01
MIN_GRAVITY_SCALE = 0.5
MAX_GRAVITY_SCALE = 1.5
GRAVITY_SCALE_PRIOR_WEIGHT = 0.01
DEFAULT_LEARNING_RATES = (1e-2, 2.5e-2, 5e-2)
DEFAULT_CURRICULUM_PREFIX_STEP_FRAMES = 0
DEFAULT_CURRICULUM_STEPS_PER_PREFIX = 400
DEFAULT_CURRICULUM_EARLY_STOP_PATIENCE = 100
MIN_FRICTION = 1e-5
MAX_FRICTION = 1.0
RADIUS_PRIOR_WEIGHT = 0.02
MAX_RADIUS_LOG_SCALE = math.log(1.5)
PLANE_SOURCES = {"static_pca", "target_pca", "trajectory_gravity"}
ROLLOUT_MODES = {"bounded_planes"}
PLANE_DECOMPOSITION_NORMAL_ANGLE_DEG = 10.0
PLANE_DECOMPOSITION_OFFSET_TOLERANCE_M = 0.12
PLANE_DECOMPOSITION_MAX_PLANES_PER_OBJECT = 24
PLANE_DECOMPOSITION_MIN_AREA_FRACTION = 0.01
BOUNDED_PLANE_CONTACT_MARGIN_M = 0.03
BOUNDED_PLANE_BOUNDARY_TOLERANCE_M = 1e-6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_static_target(item: dict[str, Any]) -> bool:
    role = str(item.get("role") or "").lower()
    track_id = str(item.get("source_track_id") or "")
    return "static" in role or track_id.startswith("physion_pp_static_")


def _select_agent_and_statics(manifest: dict[str, Any], target: dict[str, list[dict[str, Any]]]) -> tuple[str, list[str]]:
    plan = manifest.get("object_plan") if isinstance(manifest.get("object_plan"), dict) else {}
    target_objects = plan.get("target_objects") if isinstance(plan.get("target_objects"), list) else []
    by_track = {
        str(item.get("source_track_id")): str(item.get("object_id"))
        for item in target_objects
        if isinstance(item, dict) and item.get("source_track_id") and item.get("object_id")
    }
    tracking = ((plan.get("special_scene") or {}).get("scene_assessment") or {}).get("raw_payload")
    # The role binding is usually stored in sam3_video_tracks, but object_plan retains enough
    # information to classify statics robustly. Prefer explicit non-static targets.
    agent_id = None
    if isinstance(tracking, dict):
        role_binding = ((tracking.get("physion_tracking") or {}).get("role_binding") or {})
        agent_track = str(role_binding.get("agent_track") or "")
        if agent_track and by_track.get(agent_track) in target:
            agent_id = by_track[agent_track]
    static_ids: list[str] = []
    dynamic_ids: list[str] = []
    for item in target_objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_id not in target:
            continue
        if _is_static_target(item):
            static_ids.append(object_id)
        else:
            dynamic_ids.append(object_id)
    if agent_id is None and len(dynamic_ids) == 1:
        agent_id = dynamic_ids[0]
    if agent_id is None and dynamic_ids:
        agent_id = dynamic_ids[0]
    if agent_id is None:
        raise ValueError("friction_platform sphere SWR requires one dynamic agent object")
    static_ids = [object_id for object_id in static_ids if object_id != agent_id]
    if not static_ids:
        raise ValueError("friction_platform sphere SWR requires at least one static fixture object")
    return str(agent_id), sorted(set(static_ids))


def _object_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    target = manifest.get("target_trajectories") if isinstance(manifest.get("target_trajectories"), dict) else {}
    return {
        str(item.get("object_id")): dict(item)
        for item in target.get("objects", [])
        if isinstance(item, dict) and item.get("object_id")
    }


def _initial_radius_by_object(
    *,
    manifest: dict[str, Any],
    object_ids: list[str],
) -> dict[str, float]:
    specs = _object_specs(manifest)
    radii = {}
    for object_id in object_ids:
        dims = sysid_common._estimate_proxy_dimensions(specs.get(object_id, {}))
        radii[object_id] = max(0.5 * max(float(value) for value in dims), 1e-3)
    return radii


def _mesh_from_path(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [mesh for mesh in loaded.geometry.values() if hasattr(mesh, "vertices") and hasattr(mesh, "faces")]
        if not meshes:
            raise ValueError(f"mesh scene has no triangle geometry: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        raise ValueError(f"mesh has no triangle geometry: {mesh_path}")
    return loaded


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    vertices_h = np.concatenate([vertices.astype(np.float64), np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    return (vertices_h @ transform.T)[:, :3]


def _static_meshes_world(
    *,
    manifest: dict[str, Any],
    target: dict[str, list[dict[str, Any]]],
    static_ids: list[str],
) -> list[dict[str, Any]]:
    specs = _object_specs(manifest)
    meshes = []
    for object_id in static_ids:
        spec = specs.get(object_id, {})
        mesh_path = spec.get("mesh_path")
        if not mesh_path:
            raise ValueError(f"missing static mesh_path for {object_id}")
        mesh_path = Path(str(mesh_path))
        if not mesh_path.exists():
            raise ValueError(f"static mesh_path does not exist for {object_id}: {mesh_path}")
        if not target.get(object_id):
            raise ValueError(f"missing static pose records for {object_id}")
        mesh = _mesh_from_path(mesh_path)
        pose_blender = np.asarray(
            sysid_common._opencv_pose_to_blender_world(target[object_id][0]["pose_4x4"]),
            dtype=np.float64,
        )
        vertices_world = _transform_vertices(np.asarray(mesh.vertices, dtype=np.float64), pose_blender)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices_world) == 0 or len(faces) == 0:
            raise ValueError(f"empty static mesh for {object_id}: {mesh_path}")
        triangles = vertices_world[faces]
        centroids = np.mean(triangles, axis=1)
        meshes.append(
            {
                "object_id": object_id,
                "mesh_path": str(mesh_path),
                "triangle_count": int(len(triangles)),
                "triangles": triangles,
                "centroids": centroids,
            }
        )
    return meshes


def _gravity_direction_blender_world(manifest: dict[str, Any]) -> np.ndarray:
    gravity_camera = np.asarray(sysid_common._gravity_axis_camera_from_manifest(manifest), dtype=np.float64).reshape(3)
    convention = str(manifest.get("gravity_direction_convention") or "up_direction_camera").strip().lower()
    if convention in {"up_direction_camera", "up"}:
        gravity_camera = -gravity_camera
    gravity_world = np.asarray(sysid_common._opencv_vector_to_blender_world(gravity_camera.tolist()), dtype=np.float64)
    norm = float(np.linalg.norm(gravity_world))
    if not math.isfinite(norm) or norm <= 1e-12:
        return np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    return gravity_world / norm


def _fit_plane_pca(points: np.ndarray, gravity_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError("at least three points are required to fit a support plane")
    center = np.mean(points, axis=0)
    centered = points - center.reshape(1, 3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("failed to fit a stable support plane normal")
    normal = normal / norm
    # Use the upward-facing normal: opposite to the gravity/down direction.
    if float(np.dot(normal, gravity_direction)) > 0.0:
        normal = -normal
    return center, normal


def _fit_support_plane(
    *,
    source: str,
    target_positions: np.ndarray,
    static_meshes: list[dict[str, Any]],
    gravity_direction: np.ndarray,
    dynamic_radius_init: float,
) -> dict[str, Any]:
    if source == "target_pca":
        center_plane_point, normal = _fit_plane_pca(target_positions, gravity_direction)
        surface_point = center_plane_point - float(dynamic_radius_init) * normal
        source_points = int(len(target_positions))
    elif source == "static_pca":
        if not static_meshes:
            raise ValueError("static_pca support plane requires static meshes")
        vertices = np.concatenate([item["triangles"].reshape(-1, 3) for item in static_meshes], axis=0)
        surface_point, normal = _fit_plane_pca(vertices, gravity_direction)
        source_points = int(len(vertices))
    elif source == "trajectory_gravity":
        if len(target_positions) < 2:
            raise ValueError("trajectory_gravity support plane requires at least two target positions")
        centered = target_positions - np.mean(target_positions, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        motion_direction = np.asarray(vh[0], dtype=np.float64)
        motion_norm = float(np.linalg.norm(motion_direction))
        if not math.isfinite(motion_norm) or motion_norm <= 1e-12:
            raise ValueError("trajectory_gravity failed to infer a motion direction")
        motion_direction = motion_direction / motion_norm
        normal = gravity_direction - float(np.dot(gravity_direction, motion_direction)) * motion_direction
        normal_norm = float(np.linalg.norm(normal))
        if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
            raise ValueError("trajectory_gravity failed to infer a support plane normal")
        normal = normal / normal_norm
        if float(np.dot(normal, gravity_direction)) > 0.0:
            normal = -normal
        center_plane_signed = float(np.median(target_positions @ normal))
        center_plane_point = normal * center_plane_signed
        surface_point = center_plane_point - float(dynamic_radius_init) * normal
        source_points = int(len(target_positions))
    else:
        raise ValueError(f"unsupported plane source: {source}")
    tangent_gravity = gravity_direction - float(np.dot(gravity_direction, normal)) * normal
    return {
        "source": source,
        "surface_point": surface_point.astype(float).tolist(),
        "normal": normal.astype(float).tolist(),
        "gravity_direction": gravity_direction.astype(float).tolist(),
        "tangent_gravity_norm": float(np.linalg.norm(tangent_gravity)),
        "source_point_count": source_points,
    }


def _canonical_plane_normal(normal: np.ndarray) -> np.ndarray:
    value = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    value = value / norm
    pivot = int(np.argmax(np.abs(value)))
    if float(value[pivot]) < 0.0:
        value = -value
    return value


def _decompose_static_meshes_into_bounded_planes(
    *,
    static_meshes: list[dict[str, Any]],
    gravity_direction: np.ndarray,
    normal_angle_deg: float = PLANE_DECOMPOSITION_NORMAL_ANGLE_DEG,
    offset_tolerance_m: float = PLANE_DECOMPOSITION_OFFSET_TOLERANCE_M,
) -> dict[str, Any]:
    cos_threshold = math.cos(math.radians(float(normal_angle_deg)))
    objects = []
    total_planes = 0
    for item in static_meshes:
        triangles = np.asarray(item["triangles"], dtype=np.float64).reshape(-1, 3, 3)
        if len(triangles) == 0:
            objects.append(
                {
                    "object_id": str(item["object_id"]),
                    "mesh_path": str(item["mesh_path"]),
                    "triangle_count": 0,
                    "plane_count": 0,
                    "planes": [],
                }
            )
            continue
        edge_1 = triangles[:, 1, :] - triangles[:, 0, :]
        edge_2 = triangles[:, 2, :] - triangles[:, 0, :]
        raw_normals = np.cross(edge_1, edge_2)
        double_area = np.linalg.norm(raw_normals, axis=1)
        valid = double_area > 1e-10
        triangle_indices = np.nonzero(valid)[0]
        triangles_valid = triangles[valid]
        areas = 0.5 * double_area[valid]
        normals = np.asarray(
            [_canonical_plane_normal(normal) for normal in raw_normals[valid]],
            dtype=np.float64,
        )
        centroids = triangles_valid.mean(axis=1)
        total_area = float(np.sum(areas))
        min_plane_area = max(total_area * PLANE_DECOMPOSITION_MIN_AREA_FRACTION, 1e-8)
        remaining = np.ones(len(triangles_valid), dtype=bool)
        clusters: list[dict[str, Any]] = []
        max_planes = max(int(PLANE_DECOMPOSITION_MAX_PLANES_PER_OBJECT), 1)
        while bool(np.any(remaining)) and len(clusters) < max_planes:
            remaining_indices = np.nonzero(remaining)[0]
            seed_local = int(remaining_indices[int(np.argmax(areas[remaining_indices]))])
            seed_normal = normals[seed_local]
            seed_offset = float(np.dot(centroids[seed_local], seed_normal))
            same_orientation = (normals @ seed_normal) >= cos_threshold
            point_offsets = centroids @ seed_normal
            near_plane = np.abs(point_offsets - seed_offset) <= float(offset_tolerance_m)
            member_mask = remaining & same_orientation & near_plane
            if not bool(np.any(member_mask)):
                member_mask[seed_local] = True
            member_indices = np.nonzero(member_mask)[0]
            member_area = float(np.sum(areas[member_indices]))
            if member_area < min_plane_area and clusters:
                break
            normal_sum = np.sum(normals[member_indices] * areas[member_indices].reshape(-1, 1), axis=0)
            cluster_normal = _canonical_plane_normal(normal_sum)
            member_vertices = triangles_valid[member_indices].reshape(-1, 3)
            vertex_offsets = member_vertices @ cluster_normal
            cluster_offset = float(np.median(vertex_offsets))
            # One refinement pass lets a dominant plane absorb adjacent noisy triangles
            # after its normal/offset has been estimated from the first seed set.
            same_orientation = (normals @ cluster_normal) >= cos_threshold
            point_offsets = centroids @ cluster_normal
            near_plane = np.abs(point_offsets - cluster_offset) <= float(offset_tolerance_m)
            refined_mask = remaining & same_orientation & near_plane
            refined_indices = np.nonzero(refined_mask)[0]
            if len(refined_indices) > len(member_indices):
                member_indices = refined_indices
                member_area = float(np.sum(areas[member_indices]))
                normal_sum = np.sum(normals[member_indices] * areas[member_indices].reshape(-1, 1), axis=0)
                cluster_normal = _canonical_plane_normal(normal_sum)
                member_vertices = triangles_valid[member_indices].reshape(-1, 3)
                vertex_offsets = member_vertices @ cluster_normal
                cluster_offset = float(np.median(vertex_offsets))
            clusters.append(
                {
                    "indices": [int(triangle_indices[index]) for index in member_indices],
                    "local_indices": [int(index) for index in member_indices],
                    "area": member_area,
                    "normal": cluster_normal,
                    "offset": cluster_offset,
                }
            )
            remaining[member_indices] = False
        planes = []
        for plane_index, cluster in enumerate(sorted(clusters, key=lambda value: -float(value["area"]))):
            local_indices = np.asarray(cluster["local_indices"], dtype=np.int64)
            cluster_triangles = triangles_valid[local_indices]
            vertices = cluster_triangles.reshape(-1, 3)
            normal = _canonical_plane_normal(np.asarray(cluster["normal"], dtype=np.float64))
            offset = float(cluster["offset"])
            residuals = vertices @ normal - offset
            centroid = np.mean(vertices, axis=0)
            tangent_gravity = gravity_direction - float(np.dot(gravity_direction, normal)) * normal
            planes.append(
                {
                    "plane_id": f"{item['object_id']}_plane_{plane_index:03d}",
                    "object_id": str(item["object_id"]),
                    "mesh_path": str(item["mesh_path"]),
                    "normal": normal.astype(float).tolist(),
                    "offset": offset,
                    "surface_point": (normal * offset).astype(float).tolist(),
                    "area": float(cluster["area"]),
                    "triangle_count": int(len(cluster["indices"])),
                    "triangle_indices": [int(index) for index in cluster["indices"]],
                    "boundary": {
                        "type": "mesh_triangle_cluster",
                        "mesh_path": str(item["mesh_path"]),
                        "triangle_indices": [int(index) for index in cluster["indices"]],
                    },
                    "centroid": centroid.astype(float).tolist(),
                    "bounds_min": np.min(vertices, axis=0).astype(float).tolist(),
                    "bounds_max": np.max(vertices, axis=0).astype(float).tolist(),
                    "rms_plane_residual_m": float(np.sqrt(np.mean(residuals * residuals))) if len(residuals) else 0.0,
                    "max_abs_plane_residual_m": float(np.max(np.abs(residuals))) if len(residuals) else 0.0,
                    "normal_gravity_dot": float(np.dot(normal, gravity_direction)),
                    "abs_normal_gravity_dot": float(abs(np.dot(normal, gravity_direction))),
                    "tangent_gravity_norm": float(np.linalg.norm(tangent_gravity)),
                }
            )
        total_planes += len(planes)
        objects.append(
            {
                "object_id": str(item["object_id"]),
                "mesh_path": str(item["mesh_path"]),
                "triangle_count": int(len(triangles)),
                "valid_triangle_count": int(len(triangle_indices)),
                "plane_count": int(len(planes)),
                "planes": planes,
            }
        )
    return {
        "status": "ok",
        "method": "iterative_dominant_bounded_plane_extraction",
        "normal_angle_threshold_deg": float(normal_angle_deg),
        "offset_tolerance_m": float(offset_tolerance_m),
        "max_planes_per_object": int(PLANE_DECOMPOSITION_MAX_PLANES_PER_OBJECT),
        "min_area_fraction": float(PLANE_DECOMPOSITION_MIN_AREA_FRACTION),
        "gravity_direction": np.asarray(gravity_direction, dtype=np.float64).astype(float).tolist(),
        "object_count": int(len(objects)),
        "plane_count": int(total_planes),
        "objects": objects,
    }


def _fit_initial_velocity(records: list[dict[str, Any]], physics_dt_sec: float) -> np.ndarray:
    if len(records) < 2:
        return np.zeros(3, dtype=np.float64)
    first = records[0]
    x0 = np.asarray(first["position"], dtype=np.float64).reshape(3)
    times = np.asarray(
        [
            (int(record["frame_index"]) - int(first["frame_index"])) * float(physics_dt_sec)
            for record in records
        ],
        dtype=np.float64,
    )
    positions = np.asarray([record["position"] for record in records], dtype=np.float64)
    if len(records) < 3:
        dt = max(float(times[-1]), 1e-12)
        return (positions[-1] - x0) / dt
    design = np.stack([times, 0.5 * times * times], axis=1)
    coeff, *_ = np.linalg.lstsq(design, positions - x0.reshape(1, 3), rcond=None)
    return np.asarray(coeff[0], dtype=np.float64).reshape(3)


def _fit_initial_friction(records: list[dict[str, Any]], physics_dt_sec: float) -> float:
    if len(records) < 5:
        return 0.1
    first = records[0]
    x0 = np.asarray(first["position"], dtype=np.float64).reshape(3)
    times = np.asarray(
        [
            (int(record["frame_index"]) - int(first["frame_index"])) * float(physics_dt_sec)
            for record in records
        ],
        dtype=np.float64,
    )
    positions = np.asarray([record["position"] for record in records], dtype=np.float64)
    design = np.stack([times, 0.5 * times * times], axis=1)
    coeff, *_ = np.linalg.lstsq(design, positions - x0.reshape(1, 3), rcond=None)
    velocity = coeff[0]
    acceleration = coeff[1]
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-8:
        return 0.1
    direction = velocity / speed
    decel = -float(np.dot(acceleration, direction))
    if not math.isfinite(decel) or decel <= 0.0:
        return 0.1
    return float(np.clip(decel / GRAVITY_M_PER_S2, MIN_FRICTION, MAX_FRICTION))


def _raw_from_unit_interval(value: float, lower: float, upper: float) -> float:
    clipped = float(np.clip((float(value) - lower) / max(upper - lower, 1e-12), 1e-6, 1.0 - 1e-6))
    return float(math.log(clipped / (1.0 - clipped)))


def _radius_from_raw(
    *,
    radius_init_tensor: torch.Tensor,
    radius_raw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if radius_raw is None:
        return radius_init_tensor, radius_init_tensor.new_tensor(0.0)
    radius_log_scale = torch.clamp(radius_raw, -MAX_RADIUS_LOG_SCALE, MAX_RADIUS_LOG_SCALE)
    return radius_init_tensor * torch.exp(radius_log_scale), radius_log_scale


def _rollout_plane_analytic(
    *,
    frames: list[int],
    physics_dt_sec: float,
    start_position: torch.Tensor,
    velocity: torch.Tensor,
    friction: torch.Tensor,
    dynamic_radius: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    gravity_direction: torch.Tensor,
    gravity_magnitude: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    gravity = gravity_direction * gravity_magnitude
    normal = plane_normal / torch.clamp(torch.linalg.norm(plane_normal), min=start_position.new_tensor(1e-12))
    x = start_position.clone()
    v = velocity.clone()
    positions = []
    events: list[dict[str, Any]] = []
    contact_started = False
    for frame_offset, _frame_index in enumerate(frames):
        positions.append(x.clone())
        if frame_offset + 1 >= len(frames):
            break
        dt = start_position.new_tensor(
            (int(frames[frame_offset + 1]) - int(frames[frame_offset])) * float(physics_dt_sec)
        )
        x, v, contact_started, new_contact = analytic_swr_common.advance_sphere_plane_interval_torch(
            x=x,
            v=v,
            dt=dt,
            radius=dynamic_radius,
            plane_point=plane_point,
            normal=normal,
            gravity=gravity,
            friction=friction,
            restitution=0.0,
            contact_started=contact_started,
        )
        if new_contact:
            events.append(
                {
                    "event_type": "single_plane_contact_transition",
                    "frame_index": int(frames[frame_offset + 1]),
                    "phase": "free_flight_to_plane_contact",
                }
            )
    return torch.stack(positions, dim=0), events


def _free_flight_step(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    gravity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x + v * dt + 0.5 * gravity * dt * dt, v + gravity * dt


def _contact_signed_distance(*, x: torch.Tensor, plane_point: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return torch.sum((x - plane_point) * normal)


def _project_to_contact(*, x: torch.Tensor, dynamic_radius: torch.Tensor, plane_point: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    signed = _contact_signed_distance(x=x, plane_point=plane_point, normal=normal)
    return x + (dynamic_radius - signed) * normal


def _plane_boundary_runtime(
    *,
    normal: np.ndarray,
    offset: float,
    triangles: np.ndarray,
) -> dict[str, np.ndarray]:
    reference_axis = np.zeros(3, dtype=np.float64)
    reference_axis[int(np.argmin(np.abs(normal)))] = 1.0
    axis_u = np.cross(normal, reference_axis)
    axis_u = axis_u / max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / max(float(np.linalg.norm(axis_v)), 1e-12)
    origin = normal * float(offset)
    triangle_signed = triangles @ normal - float(offset)
    triangles_on_plane = triangles - triangle_signed[..., None] * normal.reshape(1, 1, 3)
    relative = triangles_on_plane - origin.reshape(1, 1, 3)
    triangles_2d = np.stack([relative @ axis_u, relative @ axis_v], axis=-1)
    edge_starts = triangles_2d
    edge_vectors = np.roll(triangles_2d, shift=-1, axis=1) - triangles_2d
    coordinate_scale = max(float(np.max(np.abs(triangles_2d))), 1.0)
    key_tolerance = coordinate_scale * 1e-9
    edge_occurrences: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[np.ndarray, np.ndarray]]] = {}
    for triangle in triangles_2d:
        for start, end in zip(triangle, np.roll(triangle, shift=-1, axis=0)):
            start_key = tuple(np.rint(start / key_tolerance).astype(np.int64).tolist())
            end_key = tuple(np.rint(end / key_tolerance).astype(np.int64).tolist())
            key = tuple(sorted((start_key, end_key)))
            edge_occurrences.setdefault(key, []).append((start.copy(), end.copy()))
    boundary_edges = [values[0] for values in edge_occurrences.values() if len(values) == 1]
    if not boundary_edges:
        boundary_edges = [
            (start.copy(), end.copy())
            for triangle in triangles_2d
            for start, end in zip(triangle, np.roll(triangle, shift=-1, axis=0))
        ]
    surface_edge_starts_2d = np.stack([start for start, _end in boundary_edges], axis=0)
    surface_edge_ends_2d = np.stack([end for _start, end in boundary_edges], axis=0)
    surface_edge_vectors_2d = surface_edge_ends_2d - surface_edge_starts_2d
    surface_edge_starts_3d = (
        origin.reshape(1, 3)
        + surface_edge_starts_2d[:, :1] * axis_u.reshape(1, 3)
        + surface_edge_starts_2d[:, 1:] * axis_v.reshape(1, 3)
    )
    surface_edge_vectors_3d = (
        surface_edge_vectors_2d[:, :1] * axis_u.reshape(1, 3)
        + surface_edge_vectors_2d[:, 1:] * axis_v.reshape(1, 3)
    )
    surface_vertices_3d = np.concatenate(
        [surface_edge_starts_3d, surface_edge_starts_3d + surface_edge_vectors_3d],
        axis=0,
    )
    surface_vertices_3d = np.unique(np.round(surface_vertices_3d, decimals=12), axis=0)
    return {
        "boundary_origin_np": origin,
        "boundary_axis_u_np": axis_u,
        "boundary_axis_v_np": axis_v,
        "boundary_triangles_2d_np": triangles_2d,
        "boundary_edge_starts_2d_np": edge_starts,
        "boundary_edge_vectors_2d_np": edge_vectors,
        "surface_boundary_edge_starts_2d_np": surface_edge_starts_2d,
        "surface_boundary_edge_vectors_2d_np": surface_edge_vectors_2d,
        "surface_boundary_edge_starts_3d_np": surface_edge_starts_3d,
        "surface_boundary_edge_vectors_3d_np": surface_edge_vectors_3d,
        "surface_boundary_vertices_3d_np": surface_vertices_3d,
    }


def _bounded_planes_runtime(plane_payload: dict[str, Any]) -> list[dict[str, Any]]:
    planes = []
    for obj in plane_payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        mesh_path = obj.get("mesh_path")
        for plane in obj.get("planes", []) or []:
            if not isinstance(plane, dict):
                continue
            normal = np.asarray(plane.get("normal"), dtype=np.float64).reshape(3)
            normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
            offset = float(plane.get("offset"))
            triangle_indices = [int(index) for index in plane.get("triangle_indices", [])]
            triangles = None
            if triangle_indices and mesh_path:
                mesh_item = next(
                    (
                        item
                        for item in plane_payload.get("_static_meshes_runtime", [])
                        if str(item.get("mesh_path")) == str(mesh_path)
                    ),
                    None,
                )
                if mesh_item is not None:
                    all_triangles = np.asarray(mesh_item["triangles"], dtype=np.float64).reshape(-1, 3, 3)
                    valid_indices = [index for index in triangle_indices if 0 <= index < len(all_triangles)]
                    if valid_indices:
                        triangles = all_triangles[np.asarray(valid_indices, dtype=np.int64)]
            if triangles is None or len(triangles) == 0:
                continue
            runtime_plane = {
                "plane_id": str(plane.get("plane_id")),
                "object_id": str(plane.get("object_id")),
                "normal_np": normal,
                "offset": offset,
                "surface_point_np": normal * offset,
                "triangles_np": triangles,
                "area": float(plane.get("area") or 0.0),
            }
            runtime_plane.update(
                _plane_boundary_runtime(
                    normal=normal,
                    offset=offset,
                    triangles=triangles,
                )
            )
            planes.append(runtime_plane)
    return planes


def _filter_bounded_planes_by_target_swept_path(
    *,
    planes: list[dict[str, Any]],
    target_positions: np.ndarray,
    radius_init: float,
    fix_radius: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    if len(positions) == 0:
        raise ValueError("bounded-plane target-path filter requires at least one target position")
    segment_lengths = (
        np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if len(positions) > 1
        else np.zeros((0,), dtype=np.float64)
    )
    point_margins = np.zeros((len(positions),), dtype=np.float64)
    if len(segment_lengths):
        point_margins[:-1] = np.maximum(point_margins[:-1], segment_lengths)
        point_margins[1:] = np.maximum(point_margins[1:], segment_lengths)
    max_radius = float(radius_init) * (1.0 if fix_radius else math.exp(MAX_RADIUS_LOG_SCALE))
    retained = []
    diagnostics = []
    for plane in planes:
        normal = np.asarray(plane["normal_np"], dtype=np.float64).reshape(3)
        offset = float(plane["offset"])
        signed_distances = np.abs(positions @ normal - offset)
        boundary_distances = np.asarray(
            [
                _bounded_plane_boundary_distance(
                    point=point,
                    plane=plane,
                )
                for point in positions
            ],
            dtype=np.float64,
        )
        possible = (
            (signed_distances <= max_radius + point_margins)
            & (boundary_distances <= point_margins + BOUNDED_PLANE_BOUNDARY_TOLERANCE_M)
        )
        matched_indices = np.nonzero(possible)[0]
        keep = bool(len(matched_indices))
        if keep:
            retained.append(plane)
        diagnostics.append(
            {
                "plane_id": str(plane.get("plane_id")),
                "object_id": str(plane.get("object_id")),
                "retained": keep,
                "matched_target_offsets": [int(index) for index in matched_indices],
                "min_abs_plane_distance_m": float(np.min(signed_distances)),
                "min_boundary_distance_m": float(np.min(boundary_distances)),
                "area": float(plane.get("area") or 0.0),
                "triangle_count": int(len(np.asarray(plane["triangles_np"]).reshape(-1, 3, 3))),
            }
        )
    fallback_to_all_planes = not retained
    if fallback_to_all_planes:
        retained = list(planes)
    payload = {
        "status": "ok",
        "method": "target_polyline_swept_sphere_face_interior_broad_phase",
        "frame_count": int(len(positions)),
        "input_plane_count": int(len(planes)),
        "retained_plane_count": int(len(retained)),
        "retained_plane_ids": [str(plane.get("plane_id")) for plane in retained],
        "radius_init_m": float(radius_init),
        "max_candidate_radius_m": max_radius,
        "radius_fixed": bool(fix_radius),
        "max_frame_displacement_m": float(np.max(segment_lengths)) if len(segment_lengths) else 0.0,
        "fallback_to_all_planes": fallback_to_all_planes,
        "planes": diagnostics,
    }
    return retained, payload


def _bounded_plane_contact_candidate(
    *,
    point: np.ndarray,
    radius: float,
    plane: dict[str, Any],
    contact_margin: float = BOUNDED_PLANE_CONTACT_MARGIN_M,
) -> dict[str, Any] | None:
    closest = _bounded_plane_surface_closest_point(point=point, plane=plane)
    if closest is None or float(closest["distance_m"]) > float(radius) + float(contact_margin):
        return None
    normal = np.asarray(closest["normal_np"], dtype=np.float64).reshape(3)
    surface_point = np.asarray(closest["surface_point_np"], dtype=np.float64).reshape(3)
    signed = float(closest["distance_m"])
    best_distance = float(closest["boundary_distance_m"])
    return {
        "plane": plane,
        "normal_np": normal,
        "offset": float(np.dot(surface_point, normal)),
        "surface_point_np": surface_point,
        "signed": signed,
        "boundary_distance": best_distance,
        "contact_kind": str(closest["contact_kind"]),
        "score": abs(signed - float(radius)),
    }


def _bounded_plane_boundary_distance(
    *,
    point: np.ndarray,
    plane: dict[str, Any],
) -> float | None:
    relative = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
        plane["boundary_origin_np"],
        dtype=np.float64,
    ).reshape(3)
    point_2d = np.asarray(
        [
            float(np.dot(relative, plane["boundary_axis_u_np"])),
            float(np.dot(relative, plane["boundary_axis_v_np"])),
        ],
        dtype=np.float64,
    )
    edge_starts = np.asarray(plane["boundary_edge_starts_2d_np"], dtype=np.float64)
    edge_vectors = np.asarray(plane["boundary_edge_vectors_2d_np"], dtype=np.float64)
    point_from_starts = point_2d.reshape(1, 1, 2) - edge_starts
    edge_cross = (
        edge_vectors[..., 0] * point_from_starts[..., 1]
        - edge_vectors[..., 1] * point_from_starts[..., 0]
    )
    inside = np.all(edge_cross >= -1e-12, axis=1) | np.all(edge_cross <= 1e-12, axis=1)
    if bool(np.any(inside)):
        return 0.0
    edge_starts = np.asarray(
        plane.get("surface_boundary_edge_starts_2d_np", edge_starts),
        dtype=np.float64,
    )
    edge_vectors = np.asarray(
        plane.get("surface_boundary_edge_vectors_2d_np", edge_vectors),
        dtype=np.float64,
    )
    point_from_starts = point_2d.reshape(1, 2) - edge_starts
    edge_length_sq = np.sum(edge_vectors * edge_vectors, axis=-1)
    projection = np.sum(point_from_starts * edge_vectors, axis=-1) / np.maximum(edge_length_sq, 1e-24)
    projection = np.clip(projection, 0.0, 1.0)
    closest = edge_starts + projection[..., None] * edge_vectors
    distances = np.linalg.norm(closest - point_2d.reshape(1, 1, 2), axis=-1)
    best_distance = float(np.min(distances))
    return best_distance if math.isfinite(best_distance) else None


def _bounded_plane_surface_closest_point(
    *,
    point: np.ndarray,
    plane: dict[str, Any],
) -> dict[str, Any] | None:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    base_normal = np.asarray(plane["normal_np"], dtype=np.float64).reshape(3)
    base_normal = base_normal / max(float(np.linalg.norm(base_normal)), 1e-12)
    offset = float(plane["offset"])
    raw_signed = float(np.dot(point, base_normal) - offset)
    projected = point - raw_signed * base_normal
    boundary_distance = _bounded_plane_boundary_distance(point=projected, plane=plane)
    if boundary_distance is None:
        return None
    if boundary_distance <= BOUNDED_PLANE_BOUNDARY_TOLERANCE_M:
        surface_point = projected
        contact_kind = "face"
    else:
        starts = np.asarray(plane["surface_boundary_edge_starts_3d_np"], dtype=np.float64).reshape(-1, 3)
        vectors = np.asarray(plane["surface_boundary_edge_vectors_3d_np"], dtype=np.float64).reshape(-1, 3)
        length_sq = np.sum(vectors * vectors, axis=1)
        parameters = np.sum((point.reshape(1, 3) - starts) * vectors, axis=1) / np.maximum(length_sq, 1e-24)
        parameters = np.clip(parameters, 0.0, 1.0)
        candidates = starts + parameters[:, None] * vectors
        distances = np.linalg.norm(point.reshape(1, 3) - candidates, axis=1)
        best_index = int(np.argmin(distances))
        surface_point = candidates[best_index]
        parameter = float(parameters[best_index])
        contact_kind = "vertex" if parameter <= 1e-9 or parameter >= 1.0 - 1e-9 else "edge"
    delta = point - surface_point
    distance = float(np.linalg.norm(delta))
    if distance > 1e-12:
        normal = delta / distance
    else:
        normal = base_normal if raw_signed >= 0.0 else -base_normal
    return {
        "surface_point_np": surface_point,
        "normal_np": normal,
        "distance_m": distance,
        "boundary_distance_m": float(boundary_distance),
        "contact_kind": contact_kind,
    }


def _real_polynomial_roots_in_interval(
    coefficients_ascending: np.ndarray,
    *,
    dt: float,
) -> list[float]:
    coefficients = np.asarray(coefficients_ascending, dtype=np.float64).reshape(-1)
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    while len(coefficients) > 1 and abs(float(coefficients[-1])) <= 1e-12 * scale:
        coefficients = coefficients[:-1]
    if len(coefficients) <= 1:
        return []
    roots = np.roots(coefficients[::-1])
    values = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= 1e-7 * (1.0 + abs(float(root.real)))
        and -1e-9 <= float(root.real) <= float(dt) + 1e-9
    )
    unique: list[float] = []
    for value in values:
        value = float(np.clip(value, 0.0, float(dt)))
        if not unique or abs(value - unique[-1]) > 1e-8:
            unique.append(value)
    return unique


def _first_sphere_bounded_plane_contact_np(
    *,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    radius: float,
    gravity: np.ndarray,
    plane: dict[str, Any],
) -> dict[str, Any] | None:
    x = np.asarray(x, dtype=np.float64).reshape(3)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
    radius = float(radius)
    dt = float(dt)
    candidates: list[dict[str, Any]] = []

    interval_points = [x, x + v * dt + 0.5 * gravity * dt * dt]
    for axis in range(3):
        if abs(float(gravity[axis])) <= 1e-12:
            continue
        extremum_time = -float(v[axis]) / float(gravity[axis])
        if 0.0 < extremum_time < dt:
            interval_points.append(x + v * extremum_time + 0.5 * gravity * extremum_time * extremum_time)
    interval_points_np = np.stack(interval_points, axis=0)
    swept_min = np.min(interval_points_np, axis=0) - radius - 1e-9
    swept_max = np.max(interval_points_np, axis=0) + radius + 1e-9

    initial = _bounded_plane_surface_closest_point(point=x, plane=plane)
    if initial is not None and float(initial["distance_m"]) <= radius + 1e-9:
        candidates.append({**initial, "contact_time_sec": 0.0})

    base_normal = np.asarray(plane["normal_np"], dtype=np.float64).reshape(3)
    base_normal = base_normal / max(float(np.linalg.norm(base_normal)), 1e-12)
    offset = float(plane["offset"])
    normal_position = float(np.dot(x, base_normal) - offset)
    normal_velocity = float(np.dot(v, base_normal))
    normal_acceleration = 0.5 * float(np.dot(gravity, base_normal))
    for side in (-1.0, 1.0):
        roots = _real_polynomial_roots_in_interval(
            np.asarray(
                [normal_position - side * radius, normal_velocity, normal_acceleration],
                dtype=np.float64,
            ),
            dt=dt,
        )
        for contact_time in roots:
            center = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
            raw_signed = float(np.dot(center, base_normal) - offset)
            surface_point = center - raw_signed * base_normal
            boundary_distance = _bounded_plane_boundary_distance(point=surface_point, plane=plane)
            if boundary_distance is None or boundary_distance > BOUNDED_PLANE_BOUNDARY_TOLERANCE_M:
                continue
            candidates.append(
                {
                    "contact_time_sec": contact_time,
                    "surface_point_np": surface_point,
                    "normal_np": base_normal if raw_signed >= 0.0 else -base_normal,
                    "distance_m": abs(raw_signed),
                    "boundary_distance_m": float(boundary_distance),
                    "contact_kind": "face",
                }
            )

    edge_starts = np.asarray(plane["surface_boundary_edge_starts_3d_np"], dtype=np.float64).reshape(-1, 3)
    edge_vectors = np.asarray(plane["surface_boundary_edge_vectors_3d_np"], dtype=np.float64).reshape(-1, 3)
    edge_ends = edge_starts + edge_vectors
    edge_possible = np.all(np.maximum(edge_starts, edge_ends) >= swept_min.reshape(1, 3), axis=1) & np.all(
        np.minimum(edge_starts, edge_ends) <= swept_max.reshape(1, 3),
        axis=1,
    )
    edge_starts = edge_starts[edge_possible]
    edge_vectors = edge_vectors[edge_possible]
    half_gravity = 0.5 * gravity
    for edge_start, edge_vector in zip(edge_starts, edge_vectors):
        edge_length = float(np.linalg.norm(edge_vector))
        if edge_length <= 1e-12:
            continue
        edge_direction = edge_vector / edge_length
        relative = x - edge_start
        q0 = relative - float(np.dot(relative, edge_direction)) * edge_direction
        q1 = v - float(np.dot(v, edge_direction)) * edge_direction
        q2 = half_gravity - float(np.dot(half_gravity, edge_direction)) * edge_direction
        coefficients = np.asarray(
            [
                float(np.dot(q0, q0)) - radius * radius,
                2.0 * float(np.dot(q0, q1)),
                float(np.dot(q1, q1)) + 2.0 * float(np.dot(q0, q2)),
                2.0 * float(np.dot(q1, q2)),
                float(np.dot(q2, q2)),
            ],
            dtype=np.float64,
        )
        for contact_time in _real_polynomial_roots_in_interval(coefficients, dt=dt):
            center = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
            edge_parameter = float(np.dot(center - edge_start, edge_direction))
            if edge_parameter < -1e-9 or edge_parameter > edge_length + 1e-9:
                continue
            surface_point = edge_start + np.clip(edge_parameter, 0.0, edge_length) * edge_direction
            delta = center - surface_point
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                continue
            candidates.append(
                {
                    "contact_time_sec": contact_time,
                    "surface_point_np": surface_point,
                    "normal_np": delta / distance,
                    "distance_m": distance,
                    "boundary_distance_m": float(
                        _bounded_plane_boundary_distance(point=center, plane=plane) or 0.0
                    ),
                    "contact_kind": "edge",
                    "edge_start_np": edge_start,
                    "edge_direction_np": edge_direction,
                }
            )

    vertices = np.asarray(plane["surface_boundary_vertices_3d_np"], dtype=np.float64).reshape(-1, 3)
    vertex_possible = np.all(vertices >= swept_min.reshape(1, 3), axis=1) & np.all(
        vertices <= swept_max.reshape(1, 3),
        axis=1,
    )
    vertices = vertices[vertex_possible]
    for vertex in vertices:
        relative = x - vertex
        coefficients = np.asarray(
            [
                float(np.dot(relative, relative)) - radius * radius,
                2.0 * float(np.dot(relative, v)),
                float(np.dot(v, v)) + float(np.dot(relative, gravity)),
                float(np.dot(v, gravity)),
                0.25 * float(np.dot(gravity, gravity)),
            ],
            dtype=np.float64,
        )
        for contact_time in _real_polynomial_roots_in_interval(coefficients, dt=dt):
            center = x + v * contact_time + 0.5 * gravity * contact_time * contact_time
            delta = center - vertex
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                continue
            candidates.append(
                {
                    "contact_time_sec": contact_time,
                    "surface_point_np": vertex,
                    "normal_np": delta / distance,
                    "distance_m": distance,
                    "boundary_distance_m": float(
                        _bounded_plane_boundary_distance(point=center, plane=plane) or 0.0
                    ),
                    "contact_kind": "vertex",
                    "vertex_np": vertex,
                }
            )

    if not candidates:
        return None
    kind_order = {"face": 0, "edge": 1, "vertex": 2}
    candidates.sort(
        key=lambda item: (
            float(item["contact_time_sec"]),
            kind_order.get(str(item["contact_kind"]), 3),
        )
    )
    return candidates[0]


def _select_bounded_plane_contact(
    *,
    point: np.ndarray,
    radius: float,
    planes: list[dict[str, Any]],
    contact_margin: float = BOUNDED_PLANE_CONTACT_MARGIN_M,
) -> dict[str, Any] | None:
    candidates = [
        candidate
        for plane in planes
        if (
            candidate := _bounded_plane_contact_candidate(
                point=point,
                radius=radius,
                plane=plane,
                contact_margin=contact_margin,
            )
        )
        is not None
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item["score"]))
    return candidates[0]


def _select_friction_plane_ids_from_target(
    *,
    target_positions: np.ndarray,
    radius: float,
    planes: list[dict[str, Any]],
) -> set[str] | None:
    """Use the first observed support contact as the only friction-bearing plane."""
    for point in np.asarray(target_positions, dtype=np.float64).reshape(-1, 3):
        contact = _select_bounded_plane_contact(
            point=point,
            radius=float(radius),
            planes=planes,
            contact_margin=BOUNDED_PLANE_CONTACT_MARGIN_M,
        )
        if contact is not None:
            return {str(contact["plane"].get("plane_id"))}
    return None


def _first_bounded_plane_contact(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    radius: torch.Tensor,
    gravity: torch.Tensor,
    planes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    point = x.detach().cpu().numpy().astype(np.float64)
    velocity = v.detach().cpu().numpy().astype(np.float64)
    gravity_np = gravity.detach().cpu().numpy().astype(np.float64)
    dt_value = float(dt.detach().cpu())
    radius_value = float(radius.detach().cpu())
    contacts = []
    for plane in planes:
        candidate = _first_sphere_bounded_plane_contact_np(
            x=point,
            v=velocity,
            dt=dt_value,
            radius=radius_value,
            gravity=gravity_np,
            plane=plane,
        )
        if candidate is None:
            continue
        normal_np = np.asarray(candidate["normal_np"], dtype=np.float64).reshape(3)
        surface_point_np = np.asarray(candidate["surface_point_np"], dtype=np.float64).reshape(3)
        normal = torch.tensor(normal_np, dtype=x.dtype, device=x.device)
        plane_point = torch.tensor(surface_point_np, dtype=x.dtype, device=x.device)
        root_value = float(candidate["contact_time_sec"])
        if root_value <= 1e-9:
            contact_time = x.new_tensor(0.0)
        elif str(candidate["contact_kind"]) == "face":
            contact_time = analytic_swr_common.first_sphere_plane_contact_time_torch(
                x=x,
                v=v,
                radius=radius,
                plane_point=plane_point,
                normal=normal,
                gravity=gravity,
                dt=dt,
            )
            if contact_time is None:
                contact_time = x.new_tensor(root_value)
        else:
            root = x.new_tensor(root_value)
            center = x + v * root + 0.5 * gravity * root * root
            center_velocity = v + gravity * root
            if str(candidate["contact_kind"]) == "edge":
                edge_start = torch.tensor(candidate["edge_start_np"], dtype=x.dtype, device=x.device)
                edge_direction = torch.tensor(candidate["edge_direction_np"], dtype=x.dtype, device=x.device)
                relative = center - edge_start
                radial = relative - torch.sum(relative * edge_direction) * edge_direction
                radial_velocity = center_velocity - torch.sum(center_velocity * edge_direction) * edge_direction
            else:
                vertex = torch.tensor(candidate["vertex_np"], dtype=x.dtype, device=x.device)
                radial = center - vertex
                radial_velocity = center_velocity
            residual = torch.sum(radial * radial) - radius * radius
            derivative = 2.0 * torch.sum(radial * radial_velocity)
            if abs(float(derivative.detach().cpu())) > 1e-10:
                contact_time = torch.clamp(
                    root - residual / derivative.detach(),
                    min=0.0,
                    max=dt_value,
                )
            else:
                contact_time = root
        contacts.append(
            {
                "plane": plane,
                "normal_np": normal_np,
                "offset": float(np.dot(surface_point_np, normal_np)),
                "surface_point_np": surface_point_np,
                "signed": float(candidate["distance_m"]),
                "boundary_distance": float(candidate["boundary_distance_m"]),
                "contact_kind": str(candidate["contact_kind"]),
                "contact_time": contact_time,
            }
        )
    if not contacts:
        return None
    contacts.sort(
        key=lambda item: (
            float(item["contact_time"].detach().cpu()),
            float(item["boundary_distance"]),
        )
    )
    return contacts[0]


def _torch_plane_from_runtime(contact: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    normal = np.asarray(contact["normal_np"], dtype=np.float64).reshape(3)
    point = np.asarray(contact["surface_point_np"], dtype=np.float64).reshape(3)
    return {
        "surface_point": torch.tensor(point, dtype=dtype, device=device),
        "normal": torch.tensor(normal, dtype=dtype, device=device),
    }


def _rollout_bounded_planes_analytic(
    *,
    frames: list[int],
    physics_dt_sec: float,
    start_position: torch.Tensor,
    velocity: torch.Tensor,
    friction: torch.Tensor,
    dynamic_radius: torch.Tensor,
    bounded_planes: list[dict[str, Any]],
    gravity_direction: torch.Tensor,
    gravity_magnitude: torch.Tensor,
    friction_plane_ids: set[str] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not bounded_planes:
        raise ValueError("bounded plane rollout requires at least one bounded plane")
    gravity = gravity_direction * gravity_magnitude
    device = start_position.device
    dtype = start_position.dtype

    def tensor_to_point(point: torch.Tensor) -> np.ndarray:
        return point.detach().cpu().numpy().astype(np.float64)

    def radius_float() -> float:
        return float(dynamic_radius.detach().cpu())

    def event_payload(
        frame_index: int,
        phase: str,
        contact: dict[str, Any],
        *,
        continuous_frame_index: float | None = None,
    ) -> dict[str, Any]:
        plane = contact["plane"]
        return {
            "event_type": "bounded_plane_contact_transition",
            "frame_index": int(frame_index),
            "continuous_frame_index": (
                float(frame_index) if continuous_frame_index is None else float(continuous_frame_index)
            ),
            "phase": phase,
            "plane_id": str(plane.get("plane_id")),
            "support_object_id": str(plane.get("object_id")),
            "signed_distance_m": float(contact.get("signed")),
            "boundary_distance_m": float(contact.get("boundary_distance")),
            "contact_kind": str(contact.get("contact_kind") or "face"),
        }

    def contact_for_current_plane(point: torch.Tensor, contact: dict[str, Any] | None) -> dict[str, Any] | None:
        if contact is None:
            return None
        return _bounded_plane_contact_candidate(
            point=tensor_to_point(point),
            radius=radius_float(),
            plane=contact["plane"],
            contact_margin=1e-9,
        )

    def actual_contact(point: torch.Tensor) -> dict[str, Any] | None:
        return _select_bounded_plane_contact(
            point=tensor_to_point(point),
            radius=radius_float(),
            planes=bounded_planes,
            contact_margin=1e-9,
        )

    zero_friction = start_position.new_tensor(0.0)

    def friction_for_contact(contact: dict[str, Any]) -> torch.Tensor:
        if friction_plane_ids is None:
            return friction
        plane_id = str(contact["plane"].get("plane_id"))
        return friction if plane_id in friction_plane_ids else zero_friction

    x = start_position.clone()
    v = velocity.clone()
    active_contact: dict[str, Any] | None = None
    positions = []
    events = []
    for frame_offset, frame_index in enumerate(frames):
        positions.append(x.clone())
        if frame_offset + 1 >= len(frames):
            break
        dt = start_position.new_tensor(
            (int(frames[frame_offset + 1]) - int(frames[frame_offset])) * float(physics_dt_sec)
        )

        if active_contact is None:
            contact = _first_bounded_plane_contact(
                x=x,
                v=v,
                dt=dt,
                radius=dynamic_radius,
                gravity=gravity,
                planes=bounded_planes,
            )
            if contact is None:
                x, v = _free_flight_step(x=x, v=v, dt=dt, gravity=gravity)
                continue
            plane = _torch_plane_from_runtime(contact, device=device, dtype=dtype)
            normal = plane["normal"] / torch.clamp(
                torch.linalg.norm(plane["normal"]),
                min=start_position.new_tensor(1e-12),
            )
            x, v, contact_started, _hit = analytic_swr_common.advance_sphere_plane_interval_torch(
                x=x,
                v=v,
                dt=dt,
                radius=dynamic_radius,
                plane_point=plane["surface_point"],
                normal=normal,
                gravity=gravity,
                friction=friction_for_contact(contact),
                restitution=0.0,
                contact_started=False,
            )
            if not contact_started:
                raise RuntimeError("bounded-plane contact solver selected a non-contacting plane")
            active_contact = contact
            events.append(
                event_payload(
                    int(frames[frame_offset + 1]),
                    "free_flight_to_bounded_plane_contact",
                    contact,
                    continuous_frame_index=(
                        float(frames[frame_offset])
                        + float(contact["contact_time"].detach().cpu())
                        / max(float(dt.detach().cpu()), 1e-12)
                        * float(int(frames[frame_offset + 1]) - int(frames[frame_offset]))
                    ),
                )
            )
            continue

        current_contact = contact_for_current_plane(x, active_contact)
        if current_contact is None:
            next_contact = actual_contact(x)
            if next_contact is None:
                active_contact = None
                x, v = _free_flight_step(x=x, v=v, dt=dt, gravity=gravity)
                events.append(
                    {
                        "event_type": "bounded_plane_contact_transition",
                        "frame_index": int(frame_index),
                        "phase": "bounded_plane_contact_to_free_flight",
                    }
                )
                continue
            active_contact = next_contact
            current_contact = next_contact
            events.append(event_payload(int(frame_index), "bounded_plane_contact_switch", next_contact))

        plane = _torch_plane_from_runtime(current_contact, device=device, dtype=dtype)
        normal = plane["normal"] / torch.clamp(torch.linalg.norm(plane["normal"]), min=start_position.new_tensor(1e-12))
        x_next, v_next = analytic_swr_common.advance_on_plane_torch(
            x=x,
            v=v,
            dt=dt,
            radius=dynamic_radius,
            plane_point=plane["surface_point"],
            normal=normal,
            gravity=gravity,
            friction=friction_for_contact(current_contact),
        )
        refreshed_contact = contact_for_current_plane(x_next, current_contact)
        if refreshed_contact is not None:
            active_contact = refreshed_contact
            x, v = x_next, v_next
            continue
        next_contact = actual_contact(x_next)
        if next_contact is not None:
            next_plane = _torch_plane_from_runtime(next_contact, device=device, dtype=dtype)
            next_normal = next_plane["normal"] / torch.clamp(
                torch.linalg.norm(next_plane["normal"]),
                min=start_position.new_tensor(1e-12),
            )
            x = _project_to_contact(
                x=x_next,
                dynamic_radius=dynamic_radius,
                plane_point=next_plane["surface_point"],
                normal=next_normal,
            )
            v = v_next - torch.sum(v_next * next_normal) * next_normal
            active_contact = next_contact
            events.append(event_payload(int(frames[frame_offset + 1]), "bounded_plane_contact_switch", next_contact))
        else:
            x, v = x_next, v_next
            active_contact = None
            events.append(
                {
                    "event_type": "bounded_plane_contact_transition",
                    "frame_index": int(frames[frame_offset + 1]),
                    "phase": "bounded_plane_contact_to_free_flight",
                }
            )
    return torch.stack(positions, dim=0), events


def _optimize_analytic_rollout_candidate(
    *,
    frames: list[int],
    target_positions: torch.Tensor,
    v0_init: np.ndarray,
    friction_init: float,
    radius_init_tensor: torch.Tensor,
    rollout_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, list[dict[str, Any]]],
    ],
    prefix_step_frames: int,
    steps_per_prefix: int,
    early_stop_patience: int,
    lr: float,
    fix_radius: bool,
    optimize_gravity: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    velocity = torch.tensor(v0_init, dtype=dtype, device=device, requires_grad=True)
    friction_raw = torch.tensor(
        _raw_from_unit_interval(friction_init, MIN_FRICTION, MAX_FRICTION),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    radius_raw = None if fix_radius else torch.zeros((), dtype=dtype, device=device, requires_grad=True)
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

    def rollout() -> tuple[
        torch.Tensor,
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        friction = MIN_FRICTION + (MAX_FRICTION - MIN_FRICTION) * torch.sigmoid(friction_raw)
        radius, radius_log_scale = _radius_from_raw(radius_init_tensor=radius_init_tensor, radius_raw=radius_raw)
        gravity_scale = (
            radius_init_tensor.new_tensor(1.0)
            if gravity_scale_raw is None
            else MIN_GRAVITY_SCALE
            + (MAX_GRAVITY_SCALE - MIN_GRAVITY_SCALE) * torch.sigmoid(gravity_scale_raw)
        )
        gravity_magnitude = radius_init_tensor.new_tensor(GRAVITY_M_PER_S2) * gravity_scale
        predicted, events = rollout_fn(velocity, friction, radius, gravity_magnitude)
        return predicted, events, friction, radius, radius_log_scale, gravity_scale, gravity_magnitude

    def evaluate(train_end: int, validation_end: int) -> dict[str, Any]:
        predicted, _events, _friction, _radius, radius_log_scale, gravity_scale, _gravity_magnitude = rollout()
        prefix_rmse = torch.sqrt(
            torch.mean(torch.sum((predicted[:train_end] - target_positions[:train_end]) ** 2, dim=1))
        )
        validation_rmse = torch.sqrt(
            torch.mean(torch.sum((predicted[:validation_end] - target_positions[:validation_end]) ** 2, dim=1))
        )
        radius_prior = radius_log_scale * radius_log_scale
        loss = (
            prefix_rmse
            + target_positions.new_tensor(RADIUS_PRIOR_WEIGHT) * radius_prior
            + target_positions.new_tensor(GRAVITY_SCALE_PRIOR_WEIGHT) * (gravity_scale - 1.0) ** 2
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("bounded-plane curriculum produced a non-finite loss")
        return {
            "loss": loss,
            "selection_metric": float(validation_rmse.detach().cpu()),
            "prefix_rmse": float(prefix_rmse.detach().cpu()),
        }

    curriculum_schedule = analytic_swr_common.optimize_adam_prefix_curriculum(
        parameters=parameters,
        evaluate=evaluate,
        observed_frames=len(frames),
        prefix_step_frames=prefix_step_frames,
        steps_per_prefix=steps_per_prefix,
        patience=early_stop_patience,
        lr=lr,
    )
    predicted, events, friction, radius, radius_log_scale, gravity_scale, gravity_magnitude = rollout()
    rmse = torch.sqrt(torch.mean(torch.sum((predicted - target_positions) ** 2, dim=1)))
    return {
        "loss": float(rmse.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "velocity": velocity.detach().cpu().numpy().copy(),
        "friction": float(friction.detach().cpu()),
        "radius": float(radius.detach().cpu()),
        "radius_log_scale": float(radius_log_scale.detach().cpu()),
        "radius_fixed": bool(fix_radius),
        "gravity_scale": float(gravity_scale.detach().cpu()),
        "gravity_m_per_s2": float(gravity_magnitude.detach().cpu()),
        "gravity_fixed": not bool(optimize_gravity),
        "events": events,
        "curriculum_schedule": curriculum_schedule,
    }


def _optimize_bounded_planes_candidate(
    *,
    frames: list[int],
    physics_dt_sec: float,
    target_positions: torch.Tensor,
    start_position: torch.Tensor,
    v0_init: np.ndarray,
    friction_init: float,
    radius_init_tensor: torch.Tensor,
    bounded_planes: list[dict[str, Any]],
    friction_plane_ids: set[str] | None,
    gravity_direction_np: np.ndarray,
    prefix_step_frames: int,
    steps_per_prefix: int,
    early_stop_patience: int,
    lr: float,
    fix_radius: bool,
    optimize_gravity: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    if not bounded_planes:
        return None
    gravity_direction = torch.tensor(gravity_direction_np, dtype=dtype, device=device)

    def rollout_fn(
        velocity: torch.Tensor,
        friction: torch.Tensor,
        radius: torch.Tensor,
        gravity_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        return _rollout_bounded_planes_analytic(
            frames=frames,
            physics_dt_sec=physics_dt_sec,
            start_position=start_position,
            velocity=velocity,
            friction=friction,
            dynamic_radius=radius,
            bounded_planes=bounded_planes,
            gravity_direction=gravity_direction,
            gravity_magnitude=gravity_magnitude,
            friction_plane_ids=friction_plane_ids,
        )

    return _optimize_analytic_rollout_candidate(
        frames=frames,
        target_positions=target_positions,
        v0_init=v0_init,
        friction_init=friction_init,
        radius_init_tensor=radius_init_tensor,
        rollout_fn=rollout_fn,
        prefix_step_frames=prefix_step_frames,
        steps_per_prefix=steps_per_prefix,
        early_stop_patience=early_stop_patience,
        lr=lr,
        fix_radius=fix_radius,
        optimize_gravity=optimize_gravity,
        device=device,
        dtype=dtype,
    )


def _result_trajectories(
    *,
    target: dict[str, list[dict[str, Any]]],
    agent_id: str,
    static_ids: list[str],
    frames: list[int],
    predicted: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    agent_pose_by_frame = {
        int(record["frame_index"]): sysid_common._opencv_pose_to_blender_world(record["pose_4x4"])
        for record in target[agent_id]
    }
    fallback_agent_pose = sysid_common._opencv_pose_to_blender_world(target[agent_id][0]["pose_4x4"])
    output[agent_id] = []
    for index, frame_index in enumerate(frames):
        source_pose = agent_pose_by_frame.get(int(frame_index), fallback_agent_pose)
        pose = [[float(value) for value in row] for row in source_pose]
        position = predicted[index].astype(float).tolist()
        pose[0][3], pose[1][3], pose[2][3] = position
        output[agent_id].append(
            {
                "frame_index": int(frame_index),
                "position": position,
                "position_opencv_camera": sysid_common._blender_vector_to_opencv_camera(position),
                "pose_4x4": pose,
            }
        )
    for object_id in static_ids:
        records = []
        first_pose = sysid_common._opencv_pose_to_blender_world(target[object_id][0]["pose_4x4"])
        first_position = np.asarray(target[object_id][0]["position"], dtype=np.float64)
        for record in target[object_id]:
            frame_index = int(record["frame_index"])
            pose = [[float(value) for value in row] for row in first_pose]
            position = first_position.astype(float).tolist()
            pose[0][3], pose[1][3], pose[2][3] = position
            records.append(
                {
                    "frame_index": frame_index,
                    "position": position,
                    "position_opencv_camera": sysid_common._blender_vector_to_opencv_camera(position),
                    "pose_4x4": pose,
                }
            )
        output[object_id] = records
    return output


def _find_source_frame_dir(manifest: dict[str, Any]) -> Path | None:
    artifact_path = manifest.get("pose_correction_artifact") or manifest.get("target_trajectories_artifact")
    if not artifact_path:
        return None
    path = Path(str(artifact_path))
    for parent in [path.parent, *path.parents]:
        candidate = parent / "metric-mesh-reconstruction" / "video_metric_depth" / "video_metric_frames"
        if candidate.exists():
            return candidate
    return None


def _load_source_frame(frame_dir: Path | None, frame_index: int, width: int, height: int) -> np.ndarray:
    import cv2

    if frame_dir is not None:
        for suffix in (".jpg", ".png"):
            path = frame_dir / f"frame_{int(frame_index):05d}{suffix}"
            if path.exists():
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    frame = np.full((int(height), int(width), 3), 28, dtype=np.uint8)
    cv2.putText(frame, "source frame unavailable", (12, int(height) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    return frame


def _project_blender_to_pixel(point: np.ndarray, camera_intrinsic: list[list[float]] | None) -> tuple[float, float, float] | None:
    if camera_intrinsic is None:
        return None
    opencv = np.asarray(sysid_common._blender_vector_to_opencv_camera(point.astype(float).tolist()), dtype=np.float64)
    z = float(opencv[2])
    if not math.isfinite(z) or z <= 1e-6:
        return None
    k = np.asarray(camera_intrinsic, dtype=np.float64).reshape(3, 3)
    u = float(k[0, 0] * opencv[0] / z + k[0, 2])
    v = float(k[1, 1] * opencv[1] / z + k[1, 2])
    return u, v, z


def _prepare_bounded_plane_hulls(
    *,
    bounded_static_planes: dict[str, Any],
    static_meshes: list[dict[str, Any]],
    camera_intrinsic: list[list[float]] | None,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    import cv2

    mesh_by_path = {str(item["mesh_path"]): np.asarray(item["triangles"], dtype=np.float64).reshape(-1, 3, 3) for item in static_meshes}
    colors = [
        (86, 180, 233),
        (230, 159, 0),
        (0, 158, 115),
        (204, 121, 167),
        (240, 228, 66),
        (213, 94, 0),
    ]
    hulls = []
    color_index = 0
    for obj in bounded_static_planes.get("objects", []):
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id") or "")
        mesh_path = str(obj.get("mesh_path") or "")
        triangles = mesh_by_path.get(mesh_path)
        if triangles is None:
            continue
        for plane in obj.get("planes", []) or []:
            if not isinstance(plane, dict):
                continue
            indices = [int(index) for index in plane.get("triangle_indices", [])]
            points = []
            depths = []
            for index in indices:
                if not (0 <= index < len(triangles)):
                    continue
                for vertex in triangles[index].reshape(3, 3):
                    projected = _project_blender_to_pixel(vertex, camera_intrinsic)
                    if projected is None:
                        continue
                    u, v, z = projected
                    if -width <= u <= 2 * width and -height <= v <= 2 * height:
                        points.append([u, v])
                        depths.append(z)
            if len(points) < 3:
                continue
            pts = np.asarray(points, dtype=np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            hull[:, 0] = np.clip(hull[:, 0], 0, width - 1)
            hull[:, 1] = np.clip(hull[:, 1], 0, height - 1)
            hulls.append(
                {
                    "object_id": object_id,
                    "plane_id": str(plane.get("plane_id") or ""),
                    "hull": np.round(hull).astype(np.int32),
                    "mean_depth": float(np.mean(depths)) if depths else 0.0,
                    "color": colors[color_index % len(colors)],
                }
            )
            color_index += 1
    hulls.sort(key=lambda item: float(item["mean_depth"]), reverse=True)
    return hulls


def _render_bounded_plane_camera_debug(
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    import cv2

    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    if camera_intrinsic is None:
        return None
    metadata = manifest.get("video_metadata") if isinstance(manifest.get("video_metadata"), dict) else {}
    width = int(metadata.get("width") or round(2.0 * float(camera_intrinsic[0][2])) or 256)
    height = int(metadata.get("height") or round(2.0 * float(camera_intrinsic[1][2])) or 256)
    fps = sysid_common._video_fps_from_manifest(manifest)
    target = result.get("target_trajectories") if isinstance(result.get("target_trajectories"), dict) else {}
    physics = result.get("physics_rollout") if isinstance(result.get("physics_rollout"), dict) else {}
    simulated = physics.get("simulated_trajectories") if isinstance(physics.get("simulated_trajectories"), dict) else {}
    best_params = result.get("alignment_optimization", {}).get("best_parameters", {})
    if not simulated or not best_params:
        return None
    agent_id = next(iter(best_params.keys()))
    records = simulated.get(agent_id)
    if not isinstance(records, list) or not records:
        return None
    bounded_debug_path = ((physics.get("bounded_static_planes") or {}).get("debug_path") if isinstance(physics.get("bounded_static_planes"), dict) else None)
    if not bounded_debug_path or not Path(str(bounded_debug_path)).exists():
        return None
    bounded_static_planes = _load_json(Path(str(bounded_debug_path)))
    support_plane = physics.get("support_plane") if isinstance(physics.get("support_plane"), dict) else {}
    selected_plane_id = str(support_plane.get("plane_id") or "")
    if selected_plane_id:
        bounded_static_planes = dict(bounded_static_planes)
        bounded_static_planes["objects"] = [
            {
                **obj,
                "planes": [
                    plane
                    for plane in obj.get("planes", [])
                    if isinstance(plane, dict) and str(plane.get("plane_id")) == selected_plane_id
                ],
            }
            for obj in bounded_static_planes.get("objects", [])
            if isinstance(obj, dict)
            and any(
                isinstance(plane, dict) and str(plane.get("plane_id")) == selected_plane_id
                for plane in obj.get("planes", [])
            )
        ]
    target_records = sysid_common._target_records_from_swr(manifest)
    static_ids = [object_id for object_id in target_records.keys() if object_id != agent_id]
    static_meshes = _static_meshes_world(manifest=manifest, target=target_records, static_ids=static_ids)
    hulls = _prepare_bounded_plane_hulls(
        bounded_static_planes=bounded_static_planes,
        static_meshes=static_meshes,
        camera_intrinsic=camera_intrinsic,
        width=width,
        height=height,
    )
    frame_dir = _find_source_frame_dir(manifest)
    output_path = output_dir / ("single_plane_camera_debug.mp4" if selected_plane_id else "bounded_planes_camera_debug.mp4")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width * 2, height))
    if not writer.isOpened():
        return None
    radius = float((best_params.get(agent_id) or {}).get("optimized_radius_m") or 0.0)
    k = np.asarray(camera_intrinsic, dtype=np.float64).reshape(3, 3)
    target_by_frame = {}
    for record in target.get(agent_id, []) if isinstance(target.get(agent_id), list) else []:
        if isinstance(record, dict) and record.get("frame_index") is not None:
            target_by_frame[int(record["frame_index"])] = np.asarray(record.get("position"), dtype=np.float64)
    for record in records:
        frame_index = int(record.get("frame_index", 0))
        left = _load_source_frame(frame_dir, frame_index, width, height)
        right = np.full((height, width, 3), 245, dtype=np.uint8)
        overlay = right.copy()
        for hull in hulls:
            pts = hull["hull"].reshape(-1, 1, 2)
            color = tuple(int(value) for value in hull["color"])
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(right, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        right = cv2.addWeighted(overlay, 0.28, right, 0.72, 0.0)
        for hull in hulls:
            pts = hull["hull"].reshape(-1, 1, 2)
            color = tuple(int(value) for value in hull["color"])
            cv2.polylines(right, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        position = np.asarray(record.get("position"), dtype=np.float64)
        projected = _project_blender_to_pixel(position, camera_intrinsic)
        if projected is not None:
            u, v, z = projected
            pixel_radius = max(3, int(round(abs(float(k[0, 0]) * radius / max(z, 1e-6)))))
            center = (int(round(u)), int(round(v)))
            cv2.circle(right, center, pixel_radius, (30, 30, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(right, center, pixel_radius, (0, 0, 140), 2, lineType=cv2.LINE_AA)
        target_position = target_by_frame.get(frame_index)
        if target_position is not None:
            projected_target = _project_blender_to_pixel(target_position, camera_intrinsic)
            if projected_target is not None:
                cv2.circle(right, (int(round(projected_target[0])), int(round(projected_target[1]))), 3, (0, 120, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(left, f"source frame {frame_index}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        title = (
            f"single slope plane {selected_plane_id} + sphere"
            if selected_plane_id
            else "bounded planes + simulated sphere"
        )
        cv2.putText(right, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(right, "red=sim sphere, green=target center", (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(np.concatenate([left, right], axis=1))
    writer.release()
    return output_path


def _render_single_plane_topdown_debug(
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    import cv2

    physics = result.get("physics_rollout") if isinstance(result.get("physics_rollout"), dict) else {}
    support_plane = physics.get("support_plane") if isinstance(physics.get("support_plane"), dict) else {}
    selected_plane_id = str(support_plane.get("plane_id") or "")
    if not selected_plane_id:
        return None
    bounded_debug_path = ((physics.get("bounded_static_planes") or {}).get("debug_path") if isinstance(physics.get("bounded_static_planes"), dict) else None)
    if not bounded_debug_path or not Path(str(bounded_debug_path)).exists():
        return None
    bounded = _load_json(Path(str(bounded_debug_path)))
    plane_payload = None
    mesh_path = None
    for obj in bounded.get("objects", []):
        if not isinstance(obj, dict):
            continue
        for plane in obj.get("planes", []):
            if isinstance(plane, dict) and str(plane.get("plane_id")) == selected_plane_id:
                plane_payload = plane
                mesh_path = str(obj.get("mesh_path") or "")
                break
        if plane_payload is not None:
            break
    if plane_payload is None or not mesh_path:
        return None

    target = result.get("target_trajectories") if isinstance(result.get("target_trajectories"), dict) else {}
    simulated = physics.get("simulated_trajectories") if isinstance(physics.get("simulated_trajectories"), dict) else {}
    best_params = result.get("alignment_optimization", {}).get("best_parameters", {})
    if not simulated or not best_params:
        return None
    agent_id = next(iter(best_params.keys()))
    target_records = target.get(agent_id)
    simulated_records = simulated.get(agent_id)
    if not isinstance(target_records, list) or not isinstance(simulated_records, list):
        return None

    all_target = sysid_common._target_records_from_swr(manifest)
    static_ids = [object_id for object_id in all_target if object_id != agent_id]
    static_meshes = _static_meshes_world(manifest=manifest, target=all_target, static_ids=static_ids)
    mesh = next((item for item in static_meshes if str(item.get("mesh_path")) == mesh_path), None)
    if mesh is None:
        return None
    triangles = np.asarray(mesh["triangles"], dtype=np.float64).reshape(-1, 3, 3)
    indices = [int(index) for index in plane_payload.get("triangle_indices", []) if 0 <= int(index) < len(triangles)]
    if not indices:
        return None
    plane_vertices = triangles[np.asarray(indices, dtype=np.int64)].reshape(-1, 3)

    gravity = np.asarray(support_plane.get("gravity_direction"), dtype=np.float64).reshape(3)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-12)
    axis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    axis_u -= float(np.dot(axis_u, gravity)) * gravity
    if float(np.linalg.norm(axis_u)) < 1e-6:
        axis_u = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        axis_u -= float(np.dot(axis_u, gravity)) * gravity
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(gravity, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-12)

    target_by_frame = {
        int(record["frame_index"]): np.asarray(record["position"], dtype=np.float64)
        for record in target_records
        if isinstance(record, dict) and record.get("frame_index") is not None
    }
    simulated_by_frame = {
        int(record["frame_index"]): np.asarray(record["position"], dtype=np.float64)
        for record in simulated_records
        if isinstance(record, dict) and record.get("frame_index") is not None
    }
    frames = sorted(set(target_by_frame) & set(simulated_by_frame))
    if not frames:
        return None

    def project(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.stack([values @ axis_u, values @ axis_v], axis=1)

    plane_2d = project(plane_vertices)
    target_2d = project(np.stack([target_by_frame[frame] for frame in frames], axis=0))
    simulated_2d = project(np.stack([simulated_by_frame[frame] for frame in frames], axis=0))
    radius = float((best_params.get(agent_id) or {}).get("optimized_radius_m") or 0.0)
    all_points = np.concatenate([plane_2d, target_2d, simulated_2d], axis=0)
    lower = np.min(all_points, axis=0) - radius
    upper = np.max(all_points, axis=0) + radius
    span = np.maximum(upper - lower, 1e-6)
    center = 0.5 * (lower + upper)
    span *= 1.15
    width, height = 640, 480
    scale = min((width - 48) / span[0], (height - 72) / span[1])

    def pixel(point: np.ndarray) -> tuple[int, int]:
        x = width * 0.5 + (float(point[0]) - center[0]) * scale
        y = height * 0.5 - (float(point[1]) - center[1]) * scale
        return int(round(x)), int(round(y))

    hull = cv2.convexHull(plane_2d.astype(np.float32)).reshape(-1, 2)
    hull_px = np.asarray([pixel(point) for point in hull], dtype=np.int32).reshape(-1, 1, 2)
    radius_px = max(3, int(round(radius * scale)))
    output_path = output_dir / "single_plane_topdown_debug.mp4"
    fps = sysid_common._video_fps_from_manifest(manifest)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return None
    for index, frame in enumerate(frames):
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        overlay = image.copy()
        cv2.fillPoly(overlay, [hull_px], (205, 225, 238))
        image = cv2.addWeighted(overlay, 0.65, image, 0.35, 0.0)
        cv2.polylines(image, [hull_px], True, (90, 110, 125), 2, cv2.LINE_AA)
        if index > 0:
            target_history = np.asarray([pixel(point) for point in target_2d[: index + 1]], dtype=np.int32)
            simulated_history = np.asarray([pixel(point) for point in simulated_2d[: index + 1]], dtype=np.int32)
            cv2.polylines(image, [target_history], False, (0, 150, 230), 2, cv2.LINE_AA)
            cv2.polylines(image, [simulated_history], False, (40, 155, 65), 2, cv2.LINE_AA)
        target_center = pixel(target_2d[index])
        simulated_center = pixel(simulated_2d[index])
        cv2.circle(image, target_center, radius_px, (0, 165, 245), -1, cv2.LINE_AA)
        cv2.circle(image, target_center, radius_px, (0, 90, 150), 2, cv2.LINE_AA)
        cv2.circle(image, simulated_center, radius_px, (55, 180, 75), -1, cv2.LINE_AA)
        cv2.circle(image, simulated_center, radius_px, (20, 105, 35), 2, cv2.LINE_AA)
        cv2.putText(image, f"top-down along -gravity | {selected_plane_id}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(image, f"frame {frame} | orange=target green=simulated", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    return output_path


def run_physionpp_friction_sphere_sysid(
    *,
    manifest_path: Path,
    output_path: Path,
    output_dir: Path,
    lr: float,
    curriculum_prefix_step_frames: int,
    curriculum_steps_per_prefix: int,
    curriculum_early_stop_patience: int,
    plane_source: str,
    rollout_mode: str,
    fix_radius: bool,
    optimize_gravity: bool,
) -> dict[str, Any]:
    if plane_source not in PLANE_SOURCES:
        raise ValueError(f"unsupported plane source: {plane_source}")
    if rollout_mode not in ROLLOUT_MODES:
        raise ValueError(f"unsupported rollout mode: {rollout_mode}")
    requested_rollout_mode = rollout_mode
    manifest = _load_json(manifest_path)
    target = sysid_common._target_records_from_swr(manifest)
    agent_id, static_ids = _select_agent_and_statics(manifest, target)
    object_ids = [agent_id, *static_ids]
    target = {object_id: target[object_id] for object_id in object_ids}
    video_fps = sysid_common._video_fps_from_manifest(manifest)
    physics_dt_sec = PHYSIONPP_PHYSICS_DT_SEC
    frames = [int(record["frame_index"]) for record in target[agent_id]]
    if len(frames) < 2:
        raise ValueError("friction_platform sphere SWR needs at least two dynamic trajectory frames")
    if any(next_frame <= frame for frame, next_frame in zip(frames, frames[1:])):
        raise ValueError("friction_platform sphere SWR requires strictly increasing trajectory frame indices")
    frame_to_target = {int(record["frame_index"]): np.asarray(record["position"], dtype=np.float64) for record in target[agent_id]}
    target_positions_np = np.stack([frame_to_target[frame] for frame in frames], axis=0)
    radius_init = _initial_radius_by_object(manifest=manifest, object_ids=[agent_id])
    static_meshes_np = _static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
    gravity_direction_np = _gravity_direction_blender_world(manifest)
    bounded_static_planes = _decompose_static_meshes_into_bounded_planes(
        static_meshes=static_meshes_np,
        gravity_direction=gravity_direction_np,
    )
    bounded_static_planes_path = output_dir / "bounded_static_planes_debug.json"
    _write_json(bounded_static_planes_path, bounded_static_planes)
    support_plane_np = _fit_support_plane(
        source=plane_source,
        target_positions=target_positions_np,
        static_meshes=static_meshes_np,
        gravity_direction=gravity_direction_np,
        dynamic_radius_init=radius_init[agent_id],
    )
    start_position_np = target_positions_np[0]
    v0_init = _fit_initial_velocity(target[agent_id], physics_dt_sec)
    friction_init = _fit_initial_friction(target[agent_id], physics_dt_sec)
    device = torch.device("cpu")
    dtype = torch.float64
    target_positions = torch.tensor(target_positions_np, dtype=dtype, device=device)
    start_position = torch.tensor(start_position_np, dtype=dtype, device=device)
    radius_init_tensor = torch.tensor(float(radius_init[agent_id]), dtype=dtype, device=device)
    support_plane = {
        "surface_point": torch.tensor(support_plane_np["surface_point"], dtype=dtype, device=device),
        "normal": torch.tensor(support_plane_np["normal"], dtype=dtype, device=device),
        "gravity_direction": torch.tensor(support_plane_np["gravity_direction"], dtype=dtype, device=device),
    }

    best: dict[str, Any] | None = None
    bounded_runtime_payload = dict(bounded_static_planes)
    bounded_runtime_payload["_static_meshes_runtime"] = static_meshes_np
    all_bounded_runtime_planes = _bounded_planes_runtime(bounded_runtime_payload)
    bounded_runtime_planes = list(all_bounded_runtime_planes)
    bounded_plane_candidate_filter: dict[str, Any] | None = None
    bounded_friction_plane_ids: set[str] | None = None
    if rollout_mode == "bounded_planes":
        bounded_runtime_planes, bounded_plane_candidate_filter = _filter_bounded_planes_by_target_swept_path(
            planes=all_bounded_runtime_planes,
            target_positions=target_positions_np,
            radius_init=radius_init[agent_id],
            fix_radius=fix_radius,
        )
        bounded_plane_candidate_filter_path = output_dir / "bounded_plane_candidate_filter.json"
        _write_json(bounded_plane_candidate_filter_path, bounded_plane_candidate_filter)
        bounded_plane_candidate_filter["debug_path"] = str(bounded_plane_candidate_filter_path)
        bounded_friction_plane_ids = _select_friction_plane_ids_from_target(
            target_positions=target_positions_np,
            radius=float(radius_init[agent_id]),
            planes=bounded_runtime_planes,
        )
        best = _optimize_bounded_planes_candidate(
            frames=frames,
            physics_dt_sec=physics_dt_sec,
            target_positions=target_positions,
            start_position=start_position,
            v0_init=v0_init,
            friction_init=friction_init,
            radius_init_tensor=radius_init_tensor,
            bounded_planes=bounded_runtime_planes,
            friction_plane_ids=bounded_friction_plane_ids,
            gravity_direction_np=gravity_direction_np,
            prefix_step_frames=curriculum_prefix_step_frames,
            steps_per_prefix=curriculum_steps_per_prefix,
            early_stop_patience=curriculum_early_stop_patience,
            lr=lr,
            fix_radius=fix_radius,
            optimize_gravity=optimize_gravity,
            device=device,
            dtype=dtype,
        )
        if best is None:
            rollout_mode = "single_plane"

    if best is not None and "gravity_m_per_s2" not in best:
        best["gravity_scale"] = 1.0
        best["gravity_m_per_s2"] = GRAVITY_M_PER_S2
        best["gravity_fixed"] = True

    if rollout_mode == "single_plane":
        def rollout_fn(
            velocity: torch.Tensor,
            friction: torch.Tensor,
            radius: torch.Tensor,
            gravity_magnitude: torch.Tensor,
        ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
            return _rollout_plane_analytic(
                frames=frames,
                physics_dt_sec=physics_dt_sec,
                start_position=start_position,
                velocity=velocity,
                friction=friction,
                dynamic_radius=radius,
                plane_point=support_plane["surface_point"],
                plane_normal=support_plane["normal"],
                gravity_direction=support_plane["gravity_direction"],
                gravity_magnitude=gravity_magnitude,
            )

        best = _optimize_analytic_rollout_candidate(
            frames=frames,
            target_positions=target_positions,
            v0_init=v0_init,
            friction_init=friction_init,
            radius_init_tensor=radius_init_tensor,
            rollout_fn=rollout_fn,
            prefix_step_frames=curriculum_prefix_step_frames,
            steps_per_prefix=curriculum_steps_per_prefix,
            early_stop_patience=curriculum_early_stop_patience,
            lr=lr,
            fix_radius=fix_radius,
            optimize_gravity=optimize_gravity,
            device=device,
            dtype=dtype,
        )

        velocity = torch.tensor(best["velocity"], dtype=dtype, device=device)
        friction = torch.tensor(best["friction"], dtype=dtype, device=device)
        radius = torch.tensor(best["radius"], dtype=dtype, device=device)
        predicted, events = _rollout_plane_analytic(
            frames=frames,
            physics_dt_sec=physics_dt_sec,
            start_position=start_position,
            velocity=velocity,
            friction=friction,
            dynamic_radius=radius,
            plane_point=support_plane["surface_point"],
            plane_normal=support_plane["normal"],
            gravity_direction=support_plane["gravity_direction"],
            gravity_magnitude=torch.tensor(best["gravity_m_per_s2"], dtype=dtype, device=device),
        )
        contact_proxy_policy = "dynamic_optimized_3d_sphere_analytic_support_plane_proxy"
        optimizer_rollout = "analytic_plane_contact"
        strategy = "physionpp_friction_platform_single_dynamic_3d_sphere"
        support_plane_payload = support_plane_np
        best_parameter_payload = {
            "initial_velocity_blender_world_m_per_s": [float(value) for value in best["velocity"].tolist()],
            "sliding_friction": float(best["friction"]),
            "optimized_radius_m": float(best["radius"]),
            "radius_init_m": float(radius_init[agent_id]),
            "radius_scale": float(best["radius"] / max(radius_init[agent_id], 1e-12)),
            "radius_fixed": bool(best.get("radius_fixed", False)),
            "gravity_scale": float(best["gravity_scale"]),
            "gravity_m_per_s2": float(best["gravity_m_per_s2"]),
            "gravity_fixed": bool(best["gravity_fixed"]),
        }
    elif rollout_mode == "bounded_planes":
        velocity = torch.tensor(best["velocity"], dtype=dtype, device=device)
        friction = torch.tensor(best["friction"], dtype=dtype, device=device)
        radius = torch.tensor(best["radius"], dtype=dtype, device=device)
        predicted, events = _rollout_bounded_planes_analytic(
            frames=frames,
            physics_dt_sec=physics_dt_sec,
            start_position=start_position,
            velocity=velocity,
            friction=friction,
            dynamic_radius=radius,
            bounded_planes=bounded_runtime_planes,
            gravity_direction=torch.tensor(gravity_direction_np, dtype=dtype, device=device),
            gravity_magnitude=torch.tensor(best["gravity_m_per_s2"], dtype=dtype, device=device),
            friction_plane_ids=bounded_friction_plane_ids,
        )
        contact_proxy_policy = "dynamic_optimized_3d_sphere_bounded_static_planes_proxy"
        optimizer_rollout = "bounded_static_plane_contact"
        strategy = "swr_fit.bounded_plane_dynamic_sphere"
        support_plane_payload = {
            "source": "bounded_static_planes",
            "gravity_direction": gravity_direction_np.astype(float).tolist(),
            "plane_count": int(len(bounded_runtime_planes)),
            "plane_ids": [str(plane.get("plane_id")) for plane in bounded_runtime_planes],
            "friction_plane_ids": sorted(bounded_friction_plane_ids) if bounded_friction_plane_ids is not None else None,
            "other_plane_friction": 0.0 if bounded_friction_plane_ids is not None else float(best["friction"]),
            "debug_path": str(bounded_static_planes_path),
            "candidate_filter": bounded_plane_candidate_filter,
        }
        best_parameter_payload = {
            "initial_velocity_blender_world_m_per_s": [float(value) for value in best["velocity"].tolist()],
            "sliding_friction": float(best["friction"]),
            "friction_plane_ids": sorted(bounded_friction_plane_ids) if bounded_friction_plane_ids is not None else None,
            "other_plane_sliding_friction": 0.0 if bounded_friction_plane_ids is not None else float(best["friction"]),
            "optimized_radius_m": float(best["radius"]),
            "radius_init_m": float(radius_init[agent_id]),
            "radius_scale": float(best["radius"] / max(radius_init[agent_id], 1e-12)),
            "radius_fixed": bool(best.get("radius_fixed", False)),
            "gravity_scale": float(best["gravity_scale"]),
            "gravity_m_per_s2": float(best["gravity_m_per_s2"]),
            "gravity_fixed": bool(best["gravity_fixed"]),
        }
    predicted_np = predicted.detach().cpu().numpy()
    simulated = _result_trajectories(
        target=target,
        agent_id=agent_id,
        static_ids=static_ids,
        frames=frames,
        predicted=predicted_np,
    )
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    loss_context = sysid_common._swr_loss_context(manifest, target, camera_intrinsic)
    fit_error = sysid_common._swr_fit_error(
        simulated,
        target,
        loss_space="world_3d",
        camera_intrinsic=camera_intrinsic,
        loss_context=loss_context,
    )
    radius_by_object = {agent_id: float(best["radius"])}
    proxy_dimensions_by_object = {
        agent_id: [2.0 * radius_by_object[agent_id]] * 3
    }
    event_payload = []
    for event in events:
        if "static_index" in event:
            static_id = static_ids[int(event["static_index"])]
            event_payload.append(
                {
                    "event_type": "dynamic_static_sphere_contact",
                    "frame_index": int(event["frame_index"]),
                    "object_ids": [agent_id, static_id],
                    "dynamic_object_id": agent_id,
                    "static_object_id": static_id,
                }
            )
            continue
        payload = {
            "event_type": str(event.get("event_type", "single_plane_contact_transition")),
            "frame_index": int(event["frame_index"]),
            "object_ids": [agent_id],
            "dynamic_object_id": agent_id,
            "phase": str(event.get("phase", "")),
        }
        if event.get("plane_id"):
            payload["plane_id"] = str(event.get("plane_id"))
        if event.get("support_object_id"):
            support_object_id = str(event.get("support_object_id"))
            payload["support_object_id"] = support_object_id
            payload["object_ids"] = [agent_id, support_object_id]
        event_payload.append(payload)
    curriculum_schedule = best.get("curriculum_schedule", [])
    curriculum_enabled = any(stage.get("prefix_frames") is not None for stage in curriculum_schedule)
    result = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": "swr_backend.surface_friction_sphere",
        "mode": "trajectory_informed_physics_alignment",
        "message": f"Physion++ friction platform SWR completed with {optimizer_rollout}",
        "static_scene_objects": manifest.get("static_scene_objects", []),
        "target_trajectories": target,
        "trajectory_physics_initialization": {
            "applied": True,
            "source": "physionpp_friction_sphere_3d_constant_acceleration_init",
            "fps": video_fps,
            "video_fps": video_fps,
            "physics_dt_sec": physics_dt_sec,
            "objects": [
                {
                    "object_id": agent_id,
                    "initial_velocity_blender_world_m_per_s": v0_init.astype(float).tolist(),
                    "initial_friction": float(friction_init),
                }
            ],
        },
        "physics_rollout": {
            "simulator": "swr_backend.surface_friction_sphere",
            "contact_proxy_policy": contact_proxy_policy,
            "simulated_trajectories": simulated,
            "proxy_dimensions_by_object": proxy_dimensions_by_object,
            "sphere_radius_3d_by_object": radius_by_object,
            "sphere_radius_init_3d_by_object": radius_init,
            "support_plane": support_plane_payload,
            "static_mesh_contact_objects": [
                {
                    "object_id": item["object_id"],
                    "mesh_path": item["mesh_path"],
                    "triangle_count": int(item["triangle_count"]),
                }
                for item in static_meshes_np
            ],
            "bounded_static_planes": {
                "status": bounded_static_planes.get("status"),
                "debug_path": str(bounded_static_planes_path),
                "method": bounded_static_planes.get("method"),
                "object_count": bounded_static_planes.get("object_count"),
                "plane_count": bounded_static_planes.get("plane_count"),
                "normal_angle_threshold_deg": bounded_static_planes.get("normal_angle_threshold_deg"),
                "offset_tolerance_m": bounded_static_planes.get("offset_tolerance_m"),
                "per_object_plane_count": {
                    str(item.get("object_id")): int(item.get("plane_count") or 0)
                    for item in bounded_static_planes.get("objects", [])
                    if isinstance(item, dict)
                },
                "candidate_filter": bounded_plane_candidate_filter,
            },
            "shape_radius_2d_by_object": radius_by_object,
            "contact_events": event_payload,
        },
        "alignment_optimization": {
            "strategy": strategy,
            "optimizer": {
                "method": "torch_adam_prefix_curriculum" if curriculum_enabled else "torch_adam_full_trajectory",
                "lr": float(lr),
                "prefix_step_frames": int(curriculum_prefix_step_frames),
                "steps_per_prefix": int(curriculum_steps_per_prefix),
                "early_stop_patience": int(curriculum_early_stop_patience),
                "curriculum_enabled": curriculum_enabled,
                "curriculum_schedule": curriculum_schedule,
                "initialization": "single_constant_acceleration_fit",
                "rollout": optimizer_rollout,
                "plane_source": plane_source,
                "requested_rollout_mode": requested_rollout_mode,
                "fix_radius": bool(fix_radius),
                "optimize_gravity": bool(optimize_gravity),
                "gravity_scale_range": [MIN_GRAVITY_SCALE, MAX_GRAVITY_SCALE],
                "gravity_scale_prior_weight": GRAVITY_SCALE_PRIOR_WEIGHT,
                "physics_dt_sec": physics_dt_sec,
            },
            "best_parameters": {
                agent_id: best_parameter_payload
            },
            "static_proxy_parameters": {},
            "events": event_payload,
            "optimization_target": {
                "object_id": agent_id,
                "loss_name": "dynamic_object_rmse_m",
                "loss_space": "world_3d",
            },
            "dynamic_object_rmse_m": float(best["rmse"]),
            "loss_space": "world_3d",
            "loss_context": loss_context,
            "camera_intrinsics": camera_intrinsic,
        },
        "fit_error": fit_error,
    }
    _write_json(output_path, result)
    _write_result_summary(output_dir=output_dir, result=result)
    return result


def _write_result_summary(*, output_dir: Path, result: dict[str, Any]) -> None:
    optimization = result["alignment_optimization"]
    rollout = result["physics_rollout"]
    _write_json(output_dir / "summary.json", {
        "status": "ok",
        "backend": result["backend"],
        "agent_object_id": optimization["optimization_target"]["object_id"],
        "static_object_ids": [
            str(item["object_id"])
            for item in rollout.get("static_mesh_contact_objects", [])
            if isinstance(item, dict) and item.get("object_id")
        ],
        "contact_proxy_policy": rollout["contact_proxy_policy"],
        "support_plane": rollout.get("support_plane"),
        "optimization_target": optimization["optimization_target"],
        "dynamic_object_rmse_m": optimization["dynamic_object_rmse_m"],
        "fit_error": result.get("fit_error"),
        "parameters": optimization["best_parameters"],
        "optimizer": optimization.get("optimizer"),
        "static_mesh_contact_objects": rollout["static_mesh_contact_objects"],
        "bounded_static_planes": rollout.get("bounded_static_planes"),
    })


def _run_learning_rate_trial(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(max(int(payload["worker_threads"]), 1))
    output_dir = Path(payload["output_dir"])
    output_path = output_dir / "physics_alignment.json"
    result = run_physionpp_friction_sphere_sysid(
        manifest_path=Path(payload["manifest_path"]),
        output_path=output_path,
        output_dir=output_dir,
        lr=float(payload["lr"]),
        curriculum_prefix_step_frames=int(payload["curriculum_prefix_step_frames"]),
        curriculum_steps_per_prefix=int(payload["curriculum_steps_per_prefix"]),
        curriculum_early_stop_patience=int(payload["curriculum_early_stop_patience"]),
        plane_source=str(payload["plane_source"]),
        rollout_mode=str(payload["rollout_mode"]),
        fix_radius=bool(payload["fix_radius"]),
        optimize_gravity=bool(payload["optimize_gravity"]),
    )
    return {
        "lr": float(payload["lr"]),
        "dynamic_object_rmse_m": float(result["alignment_optimization"]["dynamic_object_rmse_m"]),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lrs", default=",".join(str(value) for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--workers", type=int, default=len(DEFAULT_LEARNING_RATES))
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument(
        "--curriculum-prefix-step-frames",
        type=int,
        default=DEFAULT_CURRICULUM_PREFIX_STEP_FRAMES,
        help="Prefix interval in frames; use 0 for one full-trajectory optimization stage.",
    )
    parser.add_argument("--curriculum-steps-per-prefix", type=int, default=DEFAULT_CURRICULUM_STEPS_PER_PREFIX)
    parser.add_argument("--curriculum-early-stop-patience", type=int, default=DEFAULT_CURRICULUM_EARLY_STOP_PATIENCE)
    parser.add_argument("--plane-source", choices=sorted(PLANE_SOURCES), default="target_pca")
    parser.add_argument("--rollout-mode", choices=sorted(ROLLOUT_MODES), default="bounded_planes")
    parser.add_argument("--optimize-radius", action="store_true")
    parser.add_argument("--fix-gravity", action="store_true")
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
            "manifest_path": str(Path(args.manifest)),
            "output_dir": str(trial_root / f"lr_{lr:g}"),
            "lr": lr,
            "worker_threads": int(args.worker_threads),
            "curriculum_prefix_step_frames": int(args.curriculum_prefix_step_frames),
            "curriculum_steps_per_prefix": int(args.curriculum_steps_per_prefix),
            "curriculum_early_stop_patience": int(args.curriculum_early_stop_patience),
            "plane_source": args.plane_source,
            "rollout_mode": args.rollout_mode,
            "fix_radius": not bool(args.optimize_radius),
            "optimize_gravity": not bool(args.fix_gravity),
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
    _write_result_summary(output_dir=output_dir, result=result)
    debug_video_path = None
    topdown_debug_video_path = None
    if args.render_video:
        debug_video_path = _render_bounded_plane_camera_debug(
            manifest=_load_json(Path(args.manifest)),
            result=result,
            output_dir=output_dir,
        )
        topdown_debug_video_path = _render_single_plane_topdown_debug(
            manifest=_load_json(Path(args.manifest)),
            result=result,
            output_dir=output_dir,
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                    "backend": result.get("backend"),
                    "output": str(output_path),
                    "debug_video": str(debug_video_path) if debug_video_path is not None else None,
                    "topdown_debug_video": (
                        str(topdown_debug_video_path) if topdown_debug_video_path is not None else None
                    ),
                    "optimization_target": result.get("alignment_optimization", {}).get("optimization_target"),
                    "dynamic_object_rmse_m": result.get("alignment_optimization", {}).get("dynamic_object_rmse_m"),
                    "fit_error": result.get("fit_error"),
                },
                indent=2,
            )
    )


if __name__ == "__main__":
    main()
