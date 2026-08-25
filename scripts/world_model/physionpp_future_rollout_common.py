from __future__ import annotations

from typing import Any, Literal

import numpy as np

from scripts.world_model import physionpp_friction_future_rollout as friction_future
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr


RuntimePlaneOffsetSource = Literal[
    "surface_point_dot_normal",
    "payload_offset",
]


def runtime_plane(
    payload: dict[str, Any],
    *,
    offset_source: RuntimePlaneOffsetSource,
) -> dict[str, Any]:
    normal = np.asarray(payload["normal"], dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    surface_point = np.asarray(
        payload["surface_point"],
        dtype=np.float64,
    ).reshape(3)
    triangles = np.asarray(
        payload["triangles"],
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    if offset_source == "surface_point_dot_normal":
        offset = float(np.dot(surface_point, normal))
    elif offset_source == "payload_offset":
        offset = float(payload["offset"])
    else:
        raise ValueError(f"unsupported runtime-plane offset source: {offset_source!r}")
    runtime = {
        "plane_id": str(payload.get("plane_id") or ""),
        "object_id": str(payload.get("object_id") or ""),
        "normal_np": normal,
        "offset": offset,
        "surface_point_np": surface_point,
        "triangles_np": triangles,
        "area": float(payload.get("area") or 0.0),
    }
    runtime.update(
        friction_swr._plane_boundary_runtime(
            normal=normal,
            offset=offset,
            triangles=triangles,
        )
    )
    return runtime


def scan_patient_proximity(
    *,
    positions: np.ndarray,
    frames: list[int],
    patient_planes: list[dict[str, Any]],
    radius: float,
    final_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    closest = {
        "distance_m": None,
        "frame_index": None,
        "plane_id": None,
        "normal_gap_m": None,
        "boundary_distance_m": None,
    }
    samples = []
    for offset in range(1, final_offset + 1):
        frame = int(frames[offset])
        point = np.asarray(positions[offset], dtype=np.float64).reshape(3)
        distance, plane_id, normal_gap, boundary_distance = (
            friction_future._closest_patient_distance(
                point=point,
                radius=radius,
                patient_planes=patient_planes,
            )
        )
        samples.append(
            {
                "frame_index": frame,
                "point_np": point,
                "distance_m": distance,
                "plane_id": plane_id,
                "normal_gap_m": normal_gap,
                "boundary_distance_m": boundary_distance,
            }
        )
        if distance is not None and (
            closest["distance_m"] is None
            or distance < float(closest["distance_m"])
        ):
            closest = {
                "distance_m": float(distance),
                "frame_index": frame,
                "plane_id": plane_id,
                "normal_gap_m": normal_gap,
                "boundary_distance_m": boundary_distance,
            }
    return closest, samples
