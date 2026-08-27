from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from agent.world_model.artifacts import artifact_path_by_name, question_root_from_output
from agent.world_model.module_profiles import default_module_profile_policy
from scripts.world_model.foundationpose_register import (
    add_foundationpose_to_path,
    register_keyframe_seeded_single_candidate,
)
from scripts.world_model.mesh_projection import render_mask_occluded_mesh, resize_mask_nearest
from scripts.world_model.video_depth_geometry import (
    video_depth_preprocess_geometry_for_frame,
    warp_mask_to_processed,
)


CLEVRER_MESH_TARGET_FACES = 2000
DEFAULT_AREA_ALIGNMENT_TOLERANCE = 0.01
DEFAULT_DEPTH_ALIGNMENT_TOLERANCE = 0.005
DEFAULT_AREA_ALIGNMENT_MAX_ITERATIONS = 4
PHYSION_PP_ALIGNMENT_MIN_CAMERA_DEPTH = 1e-3
PHYSION_PP_ALIGNMENT_AREA_COLLAPSE_RATIO = 0.6
PHYSION_PP_ALIGNMENT_IOU_DROP = 0.5
MESH_CONDITIONING_DECISION_ID = "GEO-004.mesh_conditioning"
CLEVRER_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.aabb_no_guard"
)
PHYSION_PP_WARN_ONLY_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.obb_warn"
)
PHYSION_PP_ROLLBACK_MESH_CONDITIONING_ROUTE = (
    "mesh_conditioning.obb_rollback"
)
# Kept in sync with run_sam3_video_tracks.py / run_foundationpose.py / agent/world_model/tools.py.
PHYSION_YELLOW_PATCH_TRACK_PREFIX = "physion_yellow_patch_"


def _is_static_ground_fixture_track_id(track_id: Any) -> bool:
    return str(track_id or "").startswith(PHYSION_YELLOW_PATCH_TRACK_PREFIX)


def _physion_pp_scenario(object_plan_payload: dict[str, Any]) -> str:
    special_scene = object_plan_payload.get("special_scene") or {}
    scene_metadata = special_scene.get("scene_metadata") or {}
    return str(
        scene_metadata.get("scenario")
        or special_scene.get("scenario")
        or object_plan_payload.get("scenario")
        or ""
    ).strip().lower()


def _mesh_conditioning_route_profile(
    object_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    special_scene = object_plan_payload.get("special_scene")
    special_scene = special_scene if isinstance(special_scene, dict) else {}
    route_record = special_scene.get("mesh_conditioning_route")
    if not isinstance(route_record, dict):
        raise ValueError("mesh-conditioning object plan is missing its route record")
    if route_record.get("decision_id") != MESH_CONDITIONING_DECISION_ID:
        raise ValueError(
            "mesh-conditioning route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    policy = default_module_profile_policy()
    profile = policy.route_profiles.get(route)
    if profile is None or profile.decision_id != MESH_CONDITIONING_DECISION_ID:
        raise ValueError(f"unsupported mesh-conditioning route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("mesh-conditioning route record is missing its context")
    benchmark = str(context.get("benchmark") or "").strip().lower()
    scenario = str(context.get("scenario") or "").strip().lower()
    if benchmark != profile.benchmark:
        raise ValueError(
            "mesh-conditioning route benchmark context mismatch: "
            f"{benchmark!r} != {profile.benchmark!r}"
        )
    allowed_scenarios = frozenset(profile.scenarios)
    if allowed_scenarios and scenario not in allowed_scenarios:
        raise ValueError(
            "mesh-conditioning route scenario context mismatch: "
            f"{scenario!r} not in {sorted(allowed_scenarios)!r}"
        )
    if not allowed_scenarios and scenario:
        raise ValueError(
            "CLEVRER mesh-conditioning route must not record a Physion++ scenario: "
            f"{scenario!r}"
        )
    artifact_scenario = _physion_pp_scenario(object_plan_payload)
    if benchmark == "physion_pp" and artifact_scenario != scenario:
        raise ValueError(
            "mesh-conditioning route scenario does not match object-plan metadata: "
            f"{scenario!r} != {artifact_scenario!r}"
        )
    resolved_profile = policy.resolve_route(
        MESH_CONDITIONING_DECISION_ID,
        route,
        benchmark=benchmark,
        scenario=scenario,
    )
    module = resolved_profile.module("mesh_conditioning")
    if module.implementation != "project_scale_align_conditioned_mesh":
        raise ValueError(
            "unsupported mesh-conditioning module implementation: "
            f"{module.implementation!r}"
        )
    box_fit = module.require_string("box_fit")
    if box_fit not in {"aabb", "obb"}:
        raise ValueError(f"unsupported mesh-conditioning box fit: {box_fit!r}")
    alignment_safety = module.require_string("alignment_safety")
    if alignment_safety not in {"disabled", "warn_only", "rollback"}:
        raise ValueError(
            "unsupported mesh-conditioning alignment safety: "
            f"{alignment_safety!r}"
        )
    return {
        "route": route,
        "benchmark": resolved_profile.benchmark,
        "scenarios": allowed_scenarios,
        "use_obb_box_fit": box_fit == "obb",
        "alignment_safety_mode": alignment_safety,
        "scenario": scenario,
    }


def _physion_pp_alignment_safety_trigger(
    *,
    projected_vertices: np.ndarray,
    metrics: dict[str, Any],
    previous_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    vertices = np.asarray(projected_vertices, dtype=np.float64)
    reasons: list[str] = []
    details: dict[str, Any] = {}

    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        reasons.append("invalid_projected_vertices")
    else:
        # Camera depth is the only near-plane signal. Projected image coordinates are
        # intentionally not checked: a valid mesh may extend beyond the video frame.
        min_camera_depth = float(vertices[:, 2].min())
        details["min_camera_depth"] = min_camera_depth
        if min_camera_depth <= PHYSION_PP_ALIGNMENT_MIN_CAMERA_DEPTH:
            reasons.append("near_plane_crossing")

    rendered_area = float(metrics.get("rendered_area", float("nan")))
    current_iou = float(metrics.get("iou_with_sam3_mask", float("nan")))
    details.update({"rendered_area": rendered_area, "iou_with_sam3_mask": current_iou})
    if not np.isfinite(rendered_area) or rendered_area <= 0.0:
        reasons.append("invalid_rendered_area")
    if not np.isfinite(current_iou):
        reasons.append("invalid_iou")

    if previous_metrics is not None:
        previous_area = float(previous_metrics.get("rendered_area", float("nan")))
        previous_iou = float(previous_metrics.get("iou_with_sam3_mask", float("nan")))
        if np.isfinite(previous_area) and previous_area > 0.0 and np.isfinite(rendered_area):
            area_ratio = rendered_area / previous_area
            details["area_ratio_from_previous"] = area_ratio
            if area_ratio < PHYSION_PP_ALIGNMENT_AREA_COLLAPSE_RATIO:
                reasons.append("rendered_area_collapse")
        if np.isfinite(previous_iou) and np.isfinite(current_iou):
            iou_drop = previous_iou - current_iou
            details["iou_drop_from_previous"] = iou_drop
            if iou_drop > PHYSION_PP_ALIGNMENT_IOU_DROP:
                reasons.append("iou_drop")

    if not reasons:
        return None
    return {"reasons": reasons, **details}


def _projection_preserving_metric_scale(
    *,
    sam3d_local_vertices: np.ndarray,
    canonical_origin: np.ndarray,
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    video_depth: np.ndarray,
    sam3_mask: np.ndarray,
    occluder_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rendered = render_mask_occluded_mesh(
        vertices_camera=projected_vertices,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=video_depth.shape,
        occluder_mask=occluder_mask,
        self_mask=sam3_mask,
    )
    rendered_depth = np.asarray(rendered["rendered_depth"], dtype=np.float64)
    overlap = (
        np.asarray(rendered["visible_mask"]).astype(bool)
        & np.asarray(sam3_mask).astype(bool)
        & np.isfinite(video_depth)
        & (video_depth > 0)
        & np.isfinite(rendered_depth)
        & (rendered_depth > 0)
    )
    overlap_count = int(overlap.sum())
    if overlap_count < 20:
        raise ValueError(
            "Projection-preserving metric scaling requires at least 20 valid "
            f"rendered/SAM3 depth-overlap pixels, got {overlap_count}."
        )
    ratios = np.asarray(video_depth[overlap], dtype=np.float64) / rendered_depth[overlap]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 20:
        raise ValueError(
            "Projection-preserving metric scaling has fewer than 20 positive finite depth ratios."
        )
    metric_scale = float(np.median(ratios))
    if not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError(f"Invalid projection-preserving metric scale: {metric_scale}")

    local_scaled = canonical_origin.reshape(1, 3) + (
        np.asarray(sam3d_local_vertices, dtype=np.float64) - canonical_origin.reshape(1, 3)
    ) * metric_scale
    projected_scaled = np.asarray(projected_vertices, dtype=np.float64) * metric_scale
    depth_error_before = rendered_depth[overlap] - np.asarray(video_depth[overlap], dtype=np.float64)
    depth_error_after = rendered_depth[overlap] * metric_scale - np.asarray(
        video_depth[overlap], dtype=np.float64
    )
    return local_scaled, projected_scaled, {
        "applied": True,
        "policy": "scale_mesh_and_camera_translation_about_camera_origin",
        "scale": metric_scale,
        "valid_overlap_pixels": overlap_count,
        "paired_median_depth_error_before_m": float(np.median(depth_error_before)),
        "paired_median_depth_error_after_m": float(np.median(depth_error_after)),
        "paired_mean_absolute_depth_error_after_m": float(np.mean(np.abs(depth_error_after))),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize_mask_nearest(mask, shape)


def _read_frame_bgr(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"Unable to read frame {frame_index} from video: {video_path}")
    return frame


def _depth_for_frame(depths: np.ndarray, frame_index: int) -> np.ndarray:
    if depths.ndim == 4 and depths.shape[-1] == 1:
        return depths[frame_index, ..., 0].astype(np.float32)
    if depths.ndim == 3:
        return depths[frame_index].astype(np.float32)
    raise ValueError(f"Unsupported video metric_depth shape: {depths.shape}")


def _intrinsic_for_frame(intrinsics: np.ndarray, frame_index: int) -> np.ndarray:
    if intrinsics.ndim == 3:
        return intrinsics[frame_index].astype(np.float64)
    if intrinsics.ndim == 2:
        return intrinsics.astype(np.float64)
    raise ValueError(f"Unsupported video-depth intrinsics shape: {intrinsics.shape}")


def _processed_rgb_for_frame(processed_images: np.ndarray, frame_index: int) -> np.ndarray:
    if processed_images.ndim != 4 or processed_images.shape[-1] != 3:
        raise ValueError(f"Unsupported video-depth processed_images shape: {processed_images.shape}")
    return np.ascontiguousarray(processed_images[frame_index])


def _as_array(value: Any, *, shape_last: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    array = np.squeeze(array)
    if shape_last is not None and (array.ndim == 0 or array.shape[-1] != shape_last):
        return None
    return array


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quaternion.shape}")
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("Degenerate zero quaternion.")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _uniform_scale(scale: np.ndarray) -> float:
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values) & (values > 0)]
    if len(finite) == 0:
        raise ValueError("SAM3D scale has no positive finite values.")
    return float(np.median(finite))


def _sam3d_local_vertices_from_glb(vertices: np.ndarray) -> np.ndarray:
    glb_export_rotation = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )
    return np.asarray(vertices, dtype=np.float64) @ glb_export_rotation.T


def _sam3d_camera_to_opencv_camera(vertices: np.ndarray) -> np.ndarray:
    converted = np.asarray(vertices, dtype=np.float64).copy()
    converted[:, 0] *= -1.0
    converted[:, 1] *= -1.0
    return converted


def _mask_depth_stats(depth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if depth.shape != mask.shape:
        mask = _resize_mask_nearest(mask, depth.shape)
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("No positive finite video depth values inside SAM3 mask.")
    values = depth[valid].astype(np.float64)
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    return (vertices * scale) @ rotation + translation.reshape(1, 3)


def _positive_median(values: np.ndarray, *, label: str) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) == 0:
        raise ValueError(f"No positive finite values for {label}.")
    value = float(np.median(positive))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid median for {label}: {value}")
    return value


def _all_object_mask_union_at_frame(mask_archive: Any, frame_index: int) -> np.ndarray | None:
    """Union of EVERY object's mask at one frame (original mask resolution).

    Depth-free occluder for ``render_mask_occluded_mesh``: whichever object owns a pixel in
    the observed image is the front-most one there. Mask keys are
    ``{object_id}__frame_{frame:05d}__sam_obj_{id}``.
    """
    token = f"__frame_{int(frame_index):05d}__"
    union: np.ndarray | None = None
    for key in mask_archive.files:
        if token not in key:
            continue
        mask = np.asarray(mask_archive[key]).astype(bool)
        union = mask if union is None else (union | mask)
    return union


def _projection_metrics(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    target_mask: np.ndarray,
    target_area: int,
    target_depth: float,
    occluder_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    rendered = render_mask_occluded_mesh(
        vertices_camera=vertices,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=image_shape,
        occluder_mask=occluder_mask,
        self_mask=target_mask,
    )
    rendered_mask = rendered["visible_mask"]
    rendered_depth = rendered["rendered_depth"]
    raw_rendered_area = int(rendered["rendered_mask"].sum())
    rendered_area = int(rendered_mask.sum())
    if rendered_area <= 0:
        raise ValueError("Rendered mesh area is zero; cannot compute projection diagnostics.")
    median_depth = _positive_median(rendered_depth[rendered_mask], label="rendered mesh depth")
    intersection = int(np.logical_and(rendered_mask, target_mask).sum())
    union = int(np.logical_or(rendered_mask, target_mask).sum())
    return {
        "rendered_area": rendered_area,
        "raw_rendered_area": raw_rendered_area,
        "rendered_mesh_depth": median_depth,
        "area_ratio": float(rendered_area / target_area),
        "area_error_fraction": float((rendered_area - target_area) / target_area),
        "depth_ratio": float(median_depth / target_depth),
        "depth_error": float(median_depth - target_depth),
        "iou_with_sam3_mask": float(intersection / union) if union > 0 else 0.0,
        "intersection_area": intersection,
        "union_area": union,
    }, rendered_mask


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if np.isfinite(value) and value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _save_depth_colormap(path: Path, depth: np.ndarray, mask: np.ndarray | None, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = depth.astype(np.float64)
    if mask is None:
        valid = np.isfinite(values) & (values > 0)
    else:
        valid = mask.astype(bool) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError(f"No valid depth values for debug image: {path}")

    display = np.full(values.shape, np.nan, dtype=np.float64)
    display[valid] = values[valid]
    vmin = float(np.nanpercentile(display, 2))
    vmax = float(np.nanpercentile(display, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(display))
        vmax = float(np.nanmax(display))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    image = ax.imshow(display, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Depth")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _overlay_mask_boundary(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    if mask.shape != image_bgr.shape[:2]:
        mask = _resize_mask_nearest(mask, image_bgr.shape[:2])
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_bgr.copy()
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def _save_projection_debug_images(
    *,
    video: Path,
    frame_index: int,
    object_id: str,
    object_dir: Path,
    sam3_mask: np.ndarray,
    video_depth: np.ndarray,
    projected_mesh,
    intrinsic: np.ndarray,
    occluder_mask: np.ndarray | None = None,
) -> dict[str, str]:
    import cv2

    object_dir.mkdir(parents=True, exist_ok=True)
    original = _read_frame_bgr(video, frame_index)
    if video_depth.shape != sam3_mask.shape:
        sam3_mask = _resize_mask_nearest(sam3_mask, video_depth.shape)

    original_path = object_dir / "original.png"
    video_depth_path = object_dir / "video_depth.png"
    mesh_render_path = object_dir / "mesh_render.png"
    mesh_render_depth_path = object_dir / "mesh_render_depth.png"

    original_with_mask = _overlay_mask_boundary(original, sam3_mask, (0, 255, 255))
    cv2.imwrite(str(original_path), original_with_mask)

    _save_depth_colormap(
        video_depth_path,
        video_depth,
        sam3_mask,
        f"{object_id} video depth inside SAM3 mask",
    )

    vertices = np.asarray(projected_mesh.vertices, dtype=np.float64)
    faces = np.asarray(projected_mesh.faces, dtype=np.int64)
    rendered = render_mask_occluded_mesh(
        vertices_camera=vertices,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=video_depth.shape,
        occluder_mask=occluder_mask,
        self_mask=sam3_mask,
    )
    rendered_mask = rendered["visible_mask"]
    rendered_depth = rendered["rendered_depth"]
    mesh_overlay = _overlay_mask_boundary(original, rendered_mask, (0, 0, 255))
    mesh_overlay = _overlay_mask_boundary(mesh_overlay, sam3_mask, (0, 255, 255))
    cv2.imwrite(str(mesh_render_path), mesh_overlay)

    _save_depth_colormap(
        mesh_render_depth_path,
        rendered_depth,
        rendered_mask,
        f"{object_id} projected mesh rendered depth",
    )

    return {
        "original": str(original_path),
        "video_depth": str(video_depth_path),
        "mesh_render": str(mesh_render_path),
        "mesh_render_depth": str(mesh_render_depth_path),
    }


def _save_pre_scale_foundationpose_debug(
    *,
    object_dir: Path,
    object_id: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    seed_pose_4x4: np.ndarray,
    refined_pose_4x4: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    import cv2

    object_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = object_dir / "rgb.png"
    mask_path = object_dir / "mask.png"
    depth_path = object_dir / "depth.png"
    poses_path = object_dir / "poses.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), (np.asarray(mask, dtype=np.uint8) * 255))
    _save_depth_colormap(
        depth_path,
        np.asarray(depth, dtype=np.float32),
        np.asarray(mask, dtype=bool),
        f"{object_id} pre-scale FoundationPose depth inside mask",
    )
    poses_path.write_text(
        json.dumps(
            {
                "object_id": object_id,
                "seed_pose_4x4": np.asarray(seed_pose_4x4, dtype=np.float64).reshape(4, 4).tolist(),
                "refined_pose_4x4": np.asarray(refined_pose_4x4, dtype=np.float64).reshape(4, 4).tolist(),
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "rgb": str(rgb_path),
        "mask": str(mask_path),
        "depth": str(depth_path),
        "poses": str(poses_path),
    }


def _visual_summary(mesh) -> dict[str, Any]:
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    colors = None if vertex_colors is None else np.asarray(vertex_colors)
    return {
        "visual_type": type(visual).__name__ if visual is not None else None,
        "has_vertex_color": colors is not None and colors.size > 0,
        "vertex_color_shape": list(colors.shape) if colors is not None else None,
    }


def _source_representative_vertex_color(source_mesh) -> tuple[np.ndarray, dict[str, Any]]:
    visual = getattr(source_mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is None:
        raise ValueError("Source mesh does not contain vertex colors.")
    colors = np.asarray(vertex_colors)
    if colors.ndim != 2 or colors.shape[1] < 3 or len(colors) == 0:
        raise ValueError(f"Unsupported source vertex color shape: {colors.shape}.")
    colors = colors[:, :4] if colors.shape[1] >= 4 else np.column_stack([colors[:, :3], np.full(len(colors), 255)])
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    representative = np.median(colors, axis=0)
    representative[3] = 255
    return representative.astype(np.uint8), {
        "method": "median_source_vertex_color",
        "source_visual": _visual_summary(source_mesh),
        "source_vertex_color_count": int(len(colors)),
        "rgba": representative.astype(np.uint8).tolist(),
    }


def _apply_representative_vertex_color(source_mesh, target_mesh):
    color, summary = _source_representative_vertex_color(source_mesh)
    target_mesh = target_mesh.copy()
    target_mesh.visual.vertex_colors = np.tile(color.reshape(1, 4), (len(target_mesh.vertices), 1))
    summary["target_visual"] = _visual_summary(target_mesh)
    return target_mesh, summary


def _decimate_mesh(mesh, target_faces: int):
    import open3d as o3d
    import trimesh

    original_faces = int(len(mesh.faces))
    original_vertices = int(len(mesh.vertices))
    if original_faces <= target_faces:
        return mesh.copy(), {
            "status": "skipped",
            "target_faces": int(target_faces),
            "original_faces": original_faces,
            "original_vertices": original_vertices,
            "conditioned_faces": original_faces,
            "conditioned_vertices": original_vertices,
        }

    source = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = source.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()
    simplified.remove_unreferenced_vertices()

    vertices = np.asarray(simplified.vertices, dtype=np.float64)
    faces = np.asarray(simplified.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh decimation produced an empty mesh.")
    decimated = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return decimated, {
        "status": "decimated",
        "target_faces": int(target_faces),
        "original_faces": original_faces,
        "original_vertices": original_vertices,
        "conditioned_faces": int(len(decimated.faces)),
        "conditioned_vertices": int(len(decimated.vertices)),
    }


def _local_bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return (lower + upper) * 0.5, upper - lower


def _sphere_mesh(vertices: np.ndarray):
    import trimesh

    center, extent = _local_bbox(vertices)
    radius = float(np.median(extent) * 0.5)
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[32, 16])
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + center.reshape(1, 3)
    return mesh, {
        "primitive": "sphere",
        "center": center.tolist(),
        "radius": radius,
        "source_extent": extent.tolist(),
    }


def _box_mesh(vertices: np.ndarray):
    import trimesh

    center, extent = _local_bbox(vertices)
    mesh = trimesh.creation.box(extents=extent)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + center.reshape(1, 3)
    return mesh, {
        "primitive": "box",
        "center": center.tolist(),
        "extent": extent.tolist(),
        "axes": np.eye(3, dtype=np.float64).tolist(),
        "axis_source": "sam3d_local_xyz",
    }


def _canonicalize_obb_axes(rotation: np.ndarray, extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel OBB axes (permute columns + fix signs) to the closest global x/y/z frame.

    Preserves the physical box exactly (position, orientation, shape); it only changes
    which column is called x/y/z. An already axis-aligned box comes back with identity
    axes (matching the AABB convention), while a genuinely rotated box keeps its true
    rotation, just labelled by nearest global axis.
    """
    rotation = np.asarray(rotation, dtype=np.float64).copy()
    extents = np.asarray(extents, dtype=np.float64).copy()
    pairs = sorted(
        ((abs(float(rotation[g, c])), g, c) for g in range(3) for c in range(3)),
        reverse=True,
    )
    column_for_axis = [None, None, None]
    used_columns = set()
    for _, g, c in pairs:
        if column_for_axis[g] is not None or c in used_columns:
            continue
        column_for_axis[g] = c
        used_columns.add(c)
    perm = [int(c) for c in column_for_axis]
    rotation = rotation[:, perm]
    extents = extents[perm]
    for i in range(3):
        if rotation[i, i] < 0.0:
            rotation[:, i] = -rotation[:, i]
    if float(np.linalg.det(rotation)) < 0.0:
        weakest = int(np.argmin([abs(float(rotation[j, j])) for j in range(3)]))
        rotation[:, weakest] = -rotation[:, weakest]
    return rotation, extents


def _box_mesh_obb(vertices: np.ndarray):
    import trimesh

    to_origin, extents = trimesh.bounds.oriented_bounds(np.asarray(vertices, dtype=np.float64))
    transform = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
    rotation, extents = _canonicalize_obb_axes(transform[:3, :3], np.asarray(extents, dtype=np.float64))
    center = np.asarray(transform[:3, 3], dtype=np.float64)
    mesh = trimesh.creation.box(extents=extents)
    box_transform = np.eye(4, dtype=np.float64)
    box_transform[:3, :3] = rotation
    box_transform[:3, 3] = center
    mesh.apply_transform(box_transform)
    return mesh, {
        "primitive": "box",
        "center": center.tolist(),
        "extent": extents.tolist(),
        "axes": rotation.tolist(),
        "axis_source": "oriented_bounding_box_min_volume",
    }


def _cylinder_mesh(vertices: np.ndarray):
    import trimesh

    center, extent = _local_bbox(vertices)
    candidates = []
    for axis_index in range(3):
        plane_indices = [index for index in range(3) if index != axis_index]
        radial = np.linalg.norm((vertices[:, plane_indices] - center[plane_indices].reshape(1, 2)), axis=1)
        positive_radial = radial[np.isfinite(radial) & (radial > 1e-12)]
        if len(positive_radial) == 0:
            score = float("inf")
        else:
            score = float(np.std(positive_radial) / max(float(np.mean(positive_radial)), 1e-12))
        candidates.append((score, -float(extent[axis_index]), axis_index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    radial_score, _, axis_index = candidates[0]
    plane_indices = [index for index in range(3) if index != axis_index]
    height = float(extent[axis_index])
    radius = float(np.median(extent[plane_indices]) * 0.5)

    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    local_axes = np.eye(3, dtype=np.float64)
    cylinder_axes = np.eye(3)
    cylinder_axes[:, 2] = local_axes[:, axis_index]
    cylinder_axes[:, 0] = local_axes[:, plane_indices[0]]
    cylinder_axes[:, 1] = local_axes[:, plane_indices[1]]
    if np.linalg.det(cylinder_axes) < 0:
        cylinder_axes[:, 1] *= -1.0
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ cylinder_axes.T + center.reshape(1, 3)
    return mesh, {
        "primitive": "cylinder",
        "center": center.tolist(),
        "radius": radius,
        "height": height,
        "axis_index": int(axis_index),
        "axis": local_axes[:, axis_index].tolist(),
        "radial_distance_coefficient_of_variation": radial_score,
        "axis_source": "sam3d_local_xyz",
        "axis_selection_method": "sam3d_local_radial_distance_consistency",
        "axis_candidates": [
            {
                "axis_index": int(candidate_axis),
                "radial_distance_coefficient_of_variation": float(candidate_score),
                "axis_extent": float(-negative_extent),
            }
            for candidate_score, negative_extent, candidate_axis in candidates
        ],
        "source_extent": extent.tolist(),
    }


def _condition_mesh(mesh, geometry_type: str, target_faces: int, *, use_obb_box_fit: bool = False):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < 4:
        raise ValueError("Mesh has too few vertices for conditioning.")
    if geometry_type == "sphere":
        conditioned, fit = _sphere_mesh(vertices)
        action = "sphere_fit"
    elif geometry_type == "box":
        conditioned, fit = _box_mesh_obb(vertices) if use_obb_box_fit else _box_mesh(vertices)
        action = "box_fit"
    elif geometry_type == "cylinder":
        conditioned, fit = _cylinder_mesh(vertices)
        action = "cylinder_fit"
    else:
        conditioned, fit = _decimate_mesh(mesh, target_faces)
        action = "decimated"
    return conditioned, action, fit


def _fit_axes_from_conditioned_mesh(fit: dict[str, Any]) -> list[np.ndarray]:
    axes = fit.get("axes")
    if axes is not None:
        array = np.asarray(axes, dtype=np.float64)
        if array.shape == (3, 3):
            return [array[:, index] for index in range(3)]
    return [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]


def _foundationpose_local_axes_info(*, fit: dict[str, Any]) -> dict[str, Any]:
    axes = []
    for axis in _fit_axes_from_conditioned_mesh(fit):
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"Invalid FoundationPose canonical local axis: {axis}")
        axes.append(np.asarray(axis, dtype=np.float64) / norm)
    info: dict[str, Any] = {
        "axes": [axis.tolist() for axis in axes],
        "axis_source": "conditioned_mesh_fit_axes_in_foundationpose_canonical_metric_mesh_frame",
        "coordinate_frame": "foundationpose_canonical_metric_mesh_frame",
    }
    if fit.get("primitive") == "cylinder" and fit.get("axis_index") is not None:
        axis_index = int(fit["axis_index"])
        if 0 <= axis_index < len(axes):
            info["cylinder_axis_index"] = axis_index
            info["cylinder_axis"] = axes[axis_index].tolist()
    return info


def _rigid_pose_from_corresponding_vertices(
    *,
    local_vertices: np.ndarray,
    camera_vertices: np.ndarray,
) -> tuple[np.ndarray, float]:
    local = np.asarray(local_vertices, dtype=np.float64)
    camera = np.asarray(camera_vertices, dtype=np.float64)
    if local.shape != camera.shape or local.ndim != 2 or local.shape[1] != 3 or len(local) < 3:
        raise ValueError(f"Invalid corresponding vertex arrays: local={local.shape}, camera={camera.shape}")
    local_center = np.mean(local, axis=0)
    camera_center = np.mean(camera, axis=0)
    covariance = (local - local_center).T @ (camera - camera_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation_row = u @ vt
    if float(np.linalg.det(rotation_row)) < 0.0:
        u[:, -1] *= -1.0
        rotation_row = u @ vt
    translation = camera_center - local_center @ rotation_row
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_row.T
    pose[:3, 3] = translation
    reconstructed = local @ rotation_row + translation.reshape(1, 3)
    rms_error = float(np.sqrt(np.mean(np.sum((reconstructed - camera) ** 2, axis=1))))
    if not np.isfinite(rms_error) or rms_error > 1e-6:
        raise ValueError(f"Canonical FoundationPose mesh transform is not rigid: rms_error={rms_error}")
    return pose, rms_error


def _project_conditioned_mesh(
    *,
    mesh,
    mesh_record: dict[str, Any],
    fit: dict[str, Any],
    video_depth: np.ndarray,
    sam3_mask: np.ndarray,
    intrinsic: np.ndarray,
    sam3d_glb_yup_rotation_undone: bool,
    occluder_mask: np.ndarray | None = None,
    pre_scale_foundationpose_pose_4x4: np.ndarray | None = None,
    pre_scale_foundationpose_diagnostics: dict[str, Any] | None = None,
    skip_metric_alignment: bool = False,
    physion_pp_scenario: str = "",
    physion_pp_alignment_safety_mode: str = "disabled",
):
    if physion_pp_alignment_safety_mode not in {"disabled", "warn_only", "rollback"}:
        raise ValueError(
            "Invalid Physion++ alignment safety mode: "
            f"{physion_pp_alignment_safety_mode}"
        )
    monitor_alignment_safety = physion_pp_alignment_safety_mode in {"warn_only", "rollback"}
    enable_alignment_rollback = physion_pp_alignment_safety_mode == "rollback"
    object_id = str(mesh_record.get("object_id") or "unknown")

    rotation = _as_array(mesh_record.get("rotation"), shape_last=4)
    translation = _as_array(mesh_record.get("translation"), shape_last=3)
    scale = _as_array(mesh_record.get("scale"))
    if rotation is None or translation is None or scale is None:
        raise ValueError("SAM3D record is missing rotation, translation, or scale.")

    rotation_matrix = _rotation_matrix_from_quaternion(rotation)
    translation = translation.reshape(-1, 3)[0]
    current_scale = _uniform_scale(scale)
    sam3d_local_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    canonical_origin = (sam3d_local_vertices.min(axis=0) + sam3d_local_vertices.max(axis=0)) * 0.5
    if video_depth.shape != sam3_mask.shape:
        sam3_mask = _resize_mask_nearest(sam3_mask, video_depth.shape)
    target_stats = _mask_depth_stats(video_depth, sam3_mask)
    target_depth = float(target_stats["median"])
    target_area = int(sam3_mask.astype(bool).sum())
    if target_area <= 0:
        raise ValueError("SAM3 mask area is zero.")

    if pre_scale_foundationpose_pose_4x4 is not None:
        pre_scale_pose = np.asarray(pre_scale_foundationpose_pose_4x4, dtype=np.float64).reshape(4, 4)
        projected_vertices = _transform_vertices(
            (sam3d_local_vertices - canonical_origin.reshape(1, 3)) * current_scale,
            pre_scale_pose[:3, :3].T,
            pre_scale_pose[:3, 3],
            1.0,
        )
        projection_pose_source = "pre_scale_foundationpose_refined_pose_4x4"
    else:
        projected_vertices = _sam3d_camera_to_opencv_camera(
            _transform_vertices(sam3d_local_vertices, rotation_matrix, translation, current_scale)
        )
        projection_pose_source = "sam3d_raw_pose"
    initial_metrics, _ = _projection_metrics(
        vertices=projected_vertices,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=video_depth.shape,
        target_mask=sam3_mask,
        target_area=target_area,
        target_depth=target_depth,
        occluder_mask=occluder_mask,
    )
    rendered_area = int(initial_metrics["rendered_area"])
    initial_mesh_depth = float(initial_metrics["rendered_mesh_depth"])
    estimated_coord_scale = target_depth / initial_mesh_depth
    if not np.isfinite(estimated_coord_scale) or estimated_coord_scale <= 0:
        raise ValueError(f"Invalid SAM3D camera coordinate scale diagnostic: {estimated_coord_scale}")
    estimated_area_scale = float(np.sqrt(target_area / rendered_area))
    if not np.isfinite(estimated_area_scale) or estimated_area_scale <= 0:
        raise ValueError(f"Invalid estimated area scale diagnostic: {estimated_area_scale}")

    area_tolerance = _env_float("PHYSMIND_MESH_CONDITIONING_AREA_TOLERANCE", DEFAULT_AREA_ALIGNMENT_TOLERANCE)
    depth_tolerance = _env_float("PHYSMIND_MESH_CONDITIONING_DEPTH_TOLERANCE", DEFAULT_DEPTH_ALIGNMENT_TOLERANCE)
    max_iterations = _env_int(
        "PHYSMIND_MESH_CONDITIONING_AREA_MAX_ITERATIONS",
        DEFAULT_AREA_ALIGNMENT_MAX_ITERATIONS,
    )

    area_scale = 1.0
    depth_translation_z = 0.0
    alignment_trace = []
    final_sam3d_local_vertices = sam3d_local_vertices
    final_projected_vertices = projected_vertices
    final_metrics = initial_metrics
    stop_reason = "max_iterations"
    safety_fallback = {
        "mode": physion_pp_alignment_safety_mode,
        "enabled": enable_alignment_rollback,
        "monitoring_enabled": monitor_alignment_safety,
        "triggered": False,
        "trigger_iteration": None,
        "trigger": None,
        "observed_triggers": [],
        "rollback_iteration": None,
        "rollback_iou_with_sam3_mask": None,
        "thresholds": {
            "min_camera_depth": PHYSION_PP_ALIGNMENT_MIN_CAMERA_DEPTH,
            "area_collapse_ratio": PHYSION_PP_ALIGNMENT_AREA_COLLAPSE_RATIO,
            "iou_drop": PHYSION_PP_ALIGNMENT_IOU_DROP,
        },
    }
    best_state: dict[str, Any] | None = None
    if monitor_alignment_safety:
        initial_trigger = _physion_pp_alignment_safety_trigger(
            projected_vertices=projected_vertices,
            metrics=initial_metrics,
            previous_metrics=None,
        )
        if initial_trigger is not None:
            safety_fallback.update(
                {
                    "triggered": True,
                    "trigger_iteration": 0,
                    "trigger": initial_trigger,
                    "observed_triggers": [{"iteration": 0, **initial_trigger}],
                }
            )
            if not enable_alignment_rollback:
                warnings.warn(
                    "Physion++ mesh-alignment divergence observed without recovery "
                    f"for scenario={physion_pp_scenario or 'unknown'} object={object_id}: "
                    f"{initial_trigger}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif enable_alignment_rollback:
            best_state = {
                "iteration": 0,
                "area_scale": 1.0,
                "depth_translation_z": 0.0,
                "sam3d_local_vertices": sam3d_local_vertices.copy(),
                "projected_vertices": projected_vertices.copy(),
                "metrics": dict(initial_metrics),
            }
    previous_metrics = initial_metrics
    if skip_metric_alignment:
        # Static ground fixtures keep the initial projection: the scalar area/depth
        # feedback loop diverges on grazing-angle flat fixtures whose rendered area can
        # transiently hit zero (fatal), and pose_correction's flush-to-ground constraint
        # plus in-plane refinement re-fit position and extent against SAM3 masks across
        # the whole video anyway.
        stop_reason = "skipped_static_ground_fixture"
        max_iterations = 0
    for iteration in range(max_iterations):
        final_sam3d_local_vertices = canonical_origin.reshape(1, 3) + (
            sam3d_local_vertices - canonical_origin.reshape(1, 3)
        ) * area_scale
        if pre_scale_foundationpose_pose_4x4 is not None:
            final_projected_vertices = _transform_vertices(
                (final_sam3d_local_vertices - canonical_origin.reshape(1, 3)) * current_scale,
                pre_scale_pose[:3, :3].T,
                pre_scale_pose[:3, 3],
                1.0,
            )
        else:
            final_projected_vertices = _sam3d_camera_to_opencv_camera(
                _transform_vertices(
                    final_sam3d_local_vertices,
                    rotation_matrix,
                    translation,
                    current_scale,
                )
            )
        if depth_translation_z != 0.0:
            final_projected_vertices = final_projected_vertices.copy()
            final_projected_vertices[:, 2] += depth_translation_z
        pre_depth_metrics, _ = _projection_metrics(
            vertices=final_projected_vertices,
            faces=faces,
            intrinsic=intrinsic,
            image_shape=video_depth.shape,
            target_mask=sam3_mask,
            target_area=target_area,
            target_depth=target_depth,
            occluder_mask=occluder_mask,
        )
        depth_delta = target_depth - float(pre_depth_metrics["rendered_mesh_depth"])
        depth_translation_z += depth_delta
        final_projected_vertices = final_projected_vertices.copy()
        final_projected_vertices[:, 2] += depth_delta
        final_metrics, _ = _projection_metrics(
            vertices=final_projected_vertices,
            faces=faces,
            intrinsic=intrinsic,
            image_shape=video_depth.shape,
            target_mask=sam3_mask,
            target_area=target_area,
            target_depth=target_depth,
            occluder_mask=occluder_mask,
        )
        area_error = abs(float(final_metrics["area_error_fraction"]))
        depth_error = abs(float(final_metrics["depth_error"])) / target_depth
        trace_entry = {
            "iteration": iteration + 1,
            "area_scale": float(area_scale),
            "depth_translation_z": float(depth_translation_z),
            "depth_delta_applied": float(depth_delta),
            **final_metrics,
        }
        safety_trigger = None
        if monitor_alignment_safety:
            safety_trigger = _physion_pp_alignment_safety_trigger(
                projected_vertices=final_projected_vertices,
                metrics=final_metrics,
                previous_metrics=previous_metrics,
            )
            trace_entry["physion_pp_safety_trigger"] = safety_trigger
        alignment_trace.append(trace_entry)
        if safety_trigger is not None:
            safety_fallback["observed_triggers"].append(
                {"iteration": iteration + 1, **safety_trigger}
            )
            if not safety_fallback["triggered"]:
                safety_fallback.update(
                    {
                        "triggered": True,
                        "trigger_iteration": iteration + 1,
                        "trigger": safety_trigger,
                    }
                )
                if not enable_alignment_rollback:
                    warnings.warn(
                        "Physion++ mesh-alignment divergence observed without recovery "
                        f"for scenario={physion_pp_scenario or 'unknown'} object={object_id}: "
                        f"{safety_trigger}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if enable_alignment_rollback:
                stop_reason = "physion_pp_safety_rollback"
                break
        if enable_alignment_rollback:
            candidate_iou = float(final_metrics["iou_with_sam3_mask"])
            best_iou = (
                float(best_state["metrics"]["iou_with_sam3_mask"])
                if best_state is not None
                else -np.inf
            )
            if candidate_iou > best_iou:
                best_state = {
                    "iteration": iteration + 1,
                    "area_scale": float(area_scale),
                    "depth_translation_z": float(depth_translation_z),
                    "sam3d_local_vertices": final_sam3d_local_vertices.copy(),
                    "projected_vertices": final_projected_vertices.copy(),
                    "metrics": dict(final_metrics),
                }
        previous_metrics = final_metrics
        if area_error <= area_tolerance and depth_error <= depth_tolerance:
            stop_reason = "area_and_depth_tolerance"
            break
        if iteration + 1 >= max_iterations:
            break
        area_step = float(np.sqrt(target_area / final_metrics["rendered_area"]))
        if not np.isfinite(area_step) or area_step <= 0:
            raise ValueError(f"Invalid area alignment scale step: {area_step}")
        area_scale *= area_step

    projection_preserving_metric_scale = {
        "enabled": bool(enable_alignment_rollback),
        "applied": False,
        "reason": "alignment safety rollback was not triggered",
    }
    if enable_alignment_rollback and safety_fallback["triggered"]:
        if best_state is None:
            raise ValueError(
                "Physion++ metric-alignment safety triggered without a valid rollback state: "
                f"{safety_fallback['trigger']}"
            )
        area_scale = float(best_state["area_scale"])
        depth_translation_z = float(best_state["depth_translation_z"])
        final_sam3d_local_vertices = best_state["sam3d_local_vertices"]
        final_projected_vertices = best_state["projected_vertices"]
        final_metrics = best_state["metrics"]
        safety_fallback.update(
            {
                "rollback_iteration": int(best_state["iteration"]),
                "rollback_iou_with_sam3_mask": float(final_metrics["iou_with_sam3_mask"]),
            }
        )
        try:
            final_sam3d_local_vertices, final_projected_vertices, projection_preserving_metric_scale = (
                _projection_preserving_metric_scale(
                    sam3d_local_vertices=final_sam3d_local_vertices,
                    canonical_origin=canonical_origin,
                    projected_vertices=final_projected_vertices,
                    faces=faces,
                    intrinsic=intrinsic,
                    video_depth=video_depth,
                    sam3_mask=sam3_mask,
                    occluder_mask=occluder_mask,
                )
            )
            final_metrics, _ = _projection_metrics(
                vertices=final_projected_vertices,
                faces=faces,
                intrinsic=intrinsic,
                image_shape=video_depth.shape,
                target_mask=sam3_mask,
                target_area=target_area,
                target_depth=target_depth,
                occluder_mask=occluder_mask,
            )
        except ValueError as exc:
            projection_preserving_metric_scale = {
                "enabled": True,
                "applied": False,
                "reason": str(exc),
            }
            warnings.warn(
                "bouncy_wall_pp alignment rollback succeeded but projection-preserving "
                f"metric scaling was skipped for object={object_id}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    adjusted_conditioned_mesh = mesh.copy()
    adjusted_conditioned_mesh.vertices = final_sam3d_local_vertices
    projected_mesh = mesh.copy()
    projected_mesh.vertices = final_projected_vertices
    foundationpose_mesh = mesh.copy()
    foundationpose_mesh.vertices = (final_sam3d_local_vertices - canonical_origin.reshape(1, 3)) * current_scale
    foundationpose_initial_pose, initial_pose_rms_error = _rigid_pose_from_corresponding_vertices(
        local_vertices=np.asarray(foundationpose_mesh.vertices, dtype=np.float64),
        camera_vertices=final_projected_vertices,
    )
    foundationpose_origin_camera = foundationpose_initial_pose[:3, 3]
    extent = final_projected_vertices.max(axis=0) - final_projected_vertices.min(axis=0)
    foundationpose_local_axes = _foundationpose_local_axes_info(fit=fit)
    return adjusted_conditioned_mesh, projected_mesh, foundationpose_mesh, {
        "sam3d_glb_yup_rotation_undone": bool(sam3d_glb_yup_rotation_undone),
        "sam3d_pose_local_vertices_used": True,
        "conditioning_coordinate_frame": "sam3d_local_frame",
        "conditioned_mesh_coordinate_frame": "sam3d_local_frame",
        "sam3d_camera_to_opencv_camera_applied": True,
        "projected_mesh_coordinate_frame": "video_depth_camera_frame",
        "foundationpose_mesh_coordinate_frame": "foundationpose_canonical_metric_mesh_frame",
        "foundationpose_canonical_source_frame": "sam3d_local_frame",
        "foundationpose_local_axes": foundationpose_local_axes,
        "foundationpose_local_origin_camera_xyz": foundationpose_origin_camera.tolist(),
        "foundationpose_initial_pose_4x4": foundationpose_initial_pose.tolist(),
        "foundationpose_local_origin": {
            "center_type": "conditioned_mesh_bounds_center",
            "coordinate_frame": "foundationpose_canonical_metric_mesh_frame",
            "source_frame": "sam3d_local_frame",
            "canonical_mesh_xyz": canonical_origin.tolist(),
            "camera_xyz": foundationpose_origin_camera.tolist(),
        },
        "base_sam3d_translation": translation.tolist(),
        "base_sam3d_scale": float(current_scale),
        "projection_pose_source": projection_pose_source,
        "pre_scale_foundationpose_pose_4x4": (
            None
            if pre_scale_foundationpose_pose_4x4 is None
            else np.asarray(pre_scale_foundationpose_pose_4x4, dtype=np.float64).reshape(4, 4).tolist()
        ),
        "pre_scale_foundationpose_registration_diagnostics": pre_scale_foundationpose_diagnostics,
        "video_mask_depth": target_stats,
        "projection_diagnostics": {
            "target_depth": target_depth,
            "initial_rendered_mesh_depth": initial_mesh_depth,
            "initial_rendered_area": rendered_area,
            "initial_area_ratio": initial_metrics["area_ratio"],
            "initial_area_error_fraction": initial_metrics["area_error_fraction"],
            "initial_iou_with_sam3_mask": initial_metrics["iou_with_sam3_mask"],
            "rendered_mesh_depth": final_metrics["rendered_mesh_depth"],
            "estimated_depth_scale_if_applied": float(estimated_coord_scale),
            "target_area": target_area,
            "rendered_area": final_metrics["rendered_area"],
            "estimated_area_scale_if_applied": estimated_area_scale,
            "area_scale_applied": float(area_scale),
            "depth_translation_z_applied": float(depth_translation_z),
            "final_area_ratio": final_metrics["area_ratio"],
            "final_area_error_fraction": final_metrics["area_error_fraction"],
            "final_depth_ratio": final_metrics["depth_ratio"],
            "final_depth_error": final_metrics["depth_error"],
            "final_iou_with_sam3_mask": final_metrics["iou_with_sam3_mask"],
            "area_alignment_tolerance": float(area_tolerance),
            "depth_alignment_tolerance": float(depth_tolerance),
            "area_alignment_max_iterations": int(max_iterations),
            "area_depth_alignment_iterations": alignment_trace,
            "area_depth_alignment_stop_reason": stop_reason,
            **(
                {"physion_pp_alignment_safety_fallback": safety_fallback}
                if monitor_alignment_safety
                else {}
            ),
            **(
                {"projection_preserving_metric_scale": projection_preserving_metric_scale}
                if enable_alignment_rollback
                else {}
            ),
            "local_origin_camera_xyz": foundationpose_origin_camera.tolist(),
            "foundationpose_initial_pose_reprojection_rms_error": initial_pose_rms_error,
            "scale_adjustment_applied": bool(
                area_scale != 1.0
                or depth_translation_z != 0.0
                or projection_preserving_metric_scale.get("applied") is True
            ),
        },
        "final_coordinate_scale_from_sam3d": float(
            area_scale * float(projection_preserving_metric_scale.get("scale", 1.0))
        ),
        "projected_mesh_extent": extent.tolist(),
        "projected_mesh_bbox_diag": float(np.linalg.norm(extent)),
    }


def _load_projection_inputs(question_dir: Path) -> dict[str, Any]:
    track_labels = _load_json(artifact_path_by_name(question_dir, "sam3_video_track_labels.json"))
    video_metric_depth = _load_json(artifact_path_by_name(question_dir, "video_metric_depth.json"))
    video_depth_sidecar = np.load(video_metric_depth["tensor_sidecar"])
    return {
        "keyframes_by_object": {
            str(item["object_id"]): item for item in track_labels.get("object_keyframes", []) if item.get("object_id")
        },
        "masks": np.load(track_labels["mask_sidecar"]),
        "processed_images": video_depth_sidecar["processed_images"],
        "metric_depth": video_depth_sidecar["metric_depth"],
        "intrinsics": video_depth_sidecar["intrinsics"],
        "video_metric_depth_payload": video_metric_depth,
        "track_labels_artifact": str(artifact_path_by_name(question_dir, "sam3_video_track_labels.json")),
        "video_metric_depth_artifact": str(artifact_path_by_name(question_dir, "video_metric_depth.json")),
    }


def _pre_scale_foundationpose_refinement(
    *,
    mesh,
    seed_pose_4x4: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    scorer: Any,
    refiner: Any,
    glctx: Any,
    debug_dir: Path | None,
    est_refine_iter: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    add_foundationpose_to_path()
    from estimater import FoundationPose, set_seed

    if debug_dir is None:
        temporary_debug_dir = tempfile.TemporaryDirectory(prefix="physmind_pre_scale_foundationpose_")
        foundationpose_debug_dir = Path(temporary_debug_dir.name)
    else:
        temporary_debug_dir = None
        foundationpose_debug_dir = debug_dir
    foundationpose_debug_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    mesh.vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32))
    mesh.vertex_normals = np.ascontiguousarray(np.asarray(mesh.vertex_normals, dtype=np.float32))
    try:
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(foundationpose_debug_dir),
            debug=0,
            glctx=glctx,
        )
        if estimator.mesh is not None:
            estimator.mesh.vertices = np.ascontiguousarray(np.asarray(estimator.mesh.vertices, dtype=np.float32))
            estimator.mesh.vertex_normals = np.ascontiguousarray(
                np.asarray(estimator.mesh.vertex_normals, dtype=np.float32)
            )
        estimator.diameter = float(estimator.diameter)
        pose, diagnostics = register_keyframe_seeded_single_candidate(
            segment_estimator=estimator,
            K=np.ascontiguousarray(intrinsic, dtype=np.float32),
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            ob_mask=np.asarray(mask, dtype=bool),
            seed_raw_pose=np.asarray(seed_pose_4x4, dtype=np.float32).reshape(4, 4),
            iteration=int(est_refine_iter),
            seed_rotation_source="sam3d_objects.rotation",
        )
        diagnostics = {
            **diagnostics,
            "foundationpose_debug_dir": str(foundationpose_debug_dir) if debug_dir is not None else None,
        }
        return pose, diagnostics
    finally:
        if temporary_debug_dir is not None:
            temporary_debug_dir.cleanup()


def _guess_translation_from_mask_depth(
    *, mask: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Back-project the mask centroid at its median masked depth into the camera frame.

    Used only to seed the reuse-object registration with a segment-correct translation
    (the seg1 mesh's own translation belongs to a different trial). mask/depth/intrinsic
    must all be in the video-depth processed frame.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    depth_arr = np.asarray(depth, dtype=np.float64)
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        raise ValueError("Empty mask for reuse translation seed.")
    zs = depth_arr[ys, xs]
    finite = np.isfinite(zs) & (zs > 1e-3)
    if not np.any(finite):
        raise ValueError("No valid depth under mask for reuse translation seed.")
    z = float(np.median(zs[finite]))
    u = float(np.mean(xs[finite]))
    v = float(np.mean(ys[finite]))
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=np.float64)


def run_mesh_conditioning(*, object_plan: Path, output: Path, target_faces: int, video: Path | None = None) -> None:
    question_dir = question_root_from_output(output)
    object_plan_payload = _load_json(object_plan)
    route_profile = _mesh_conditioning_route_profile(object_plan_payload)
    physion_pp_scenario = str(route_profile["scenario"])
    physion_pp_alignment_safety_mode = str(
        route_profile["alignment_safety_mode"]
    )
    use_obb_box_fit = bool(route_profile["use_obb_box_fit"])
    sam3d_meshes = _load_json(artifact_path_by_name(question_dir, "sam3d_meshes.json"))
    projection_inputs = _load_projection_inputs(question_dir)
    mesh_by_object = {item["object_id"]: item for item in sam3d_meshes.get("objects", [])}
    geometry_by_object = {
        item["object_id"]: item
        for item in object_plan_payload.get("target_objects", [])
        if item.get("object_id")
    }
    conditioned_dir = output.parent / "conditioned_meshes"
    projected_dir = output.parent / "projected_meshes"
    debug_root = output.parent / "debug_projection"
    pre_scale_debug_root = output.parent / "debug_pre_scale_foundationpose"
    debug_artifacts = os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1"
    conditioned_dir.mkdir(parents=True, exist_ok=True)
    projected_dir.mkdir(parents=True, exist_ok=True)
    if debug_artifacts:
        debug_root.mkdir(parents=True, exist_ok=True)
        pre_scale_debug_root.mkdir(parents=True, exist_ok=True)

    pre_scale_foundationpose_context: dict[str, Any] | None = None

    def _get_pre_scale_foundationpose_context() -> dict[str, Any]:
        nonlocal pre_scale_foundationpose_context
        if pre_scale_foundationpose_context is None:
            add_foundationpose_to_path()
            from estimater import PoseRefinePredictor, ScorePredictor, dr, set_logging_format, set_seed

            set_logging_format()
            set_seed(0)
            pre_scale_foundationpose_context = {
                "glctx": dr.RasterizeCudaContext(),
                "scorer": ScorePredictor(),
                "refiner": PoseRefinePredictor(),
            }
        return pre_scale_foundationpose_context

    objects = []
    for object_id, mesh_record in mesh_by_object.items():
        source_mesh_path = mesh_record.get("glb_path") or mesh_record.get("mesh_path")
        geometry = geometry_by_object.get(object_id, {})
        geometry_type = str(geometry.get("geometry_type") or "irregular").lower()
        if geometry_type not in {"sphere", "box", "cylinder", "irregular"}:
            geometry_type = "irregular"
        if not source_mesh_path:
            objects.append({"object_id": object_id, "status": "missing_mesh", "geometry_type": geometry_type})
            continue
        try:
            source_mesh = _load_mesh(Path(source_mesh_path))
            source_mesh_original_coordinate_frame = "sam3d_local_frame"
            source_mesh_frame_conversion = "none"
            if mesh_record.get("glb_path"):
                source_mesh.vertices = _sam3d_local_vertices_from_glb(np.asarray(source_mesh.vertices, dtype=np.float64))
                source_mesh_original_coordinate_frame = "sam3d_glb_yup_viewer_frame"
                source_mesh_frame_conversion = "sam3d_glb_yup_viewer_frame_to_sam3d_local_frame"
            conditioned_mesh, action, fit = _condition_mesh(
                source_mesh, geometry_type, target_faces, use_obb_box_fit=use_obb_box_fit
            )
            conditioned_mesh, vertex_color_transfer = _apply_representative_vertex_color(source_mesh, conditioned_mesh)
            conditioned_path = conditioned_dir / f"{object_id}.glb"
            keyframe = projection_inputs["keyframes_by_object"].get(str(object_id))
            if not keyframe:
                raise ValueError(f"No selected keyframe found for object: {object_id}")
            frame_index = int(keyframe["frame_index"])
            mask_key = keyframe.get("mask_key")
            if not mask_key:
                raise ValueError(f"Selected keyframe has no mask_key for object: {object_id}")
            metric_depth = projection_inputs["metric_depth"]
            if frame_index >= metric_depth.shape[0]:
                raise ValueError(
                    f"Selected frame {frame_index} exceeds video-depth frame count {int(metric_depth.shape[0])}"
                )
            mask = projection_inputs["masks"][mask_key].astype(bool)
            occluder_union = _all_object_mask_union_at_frame(projection_inputs["masks"], frame_index)
            depth = _depth_for_frame(metric_depth, frame_index)
            intrinsic = _intrinsic_for_frame(projection_inputs["intrinsics"], frame_index)
            rgb = _processed_rgb_for_frame(projection_inputs["processed_images"], frame_index)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(
                    f"Video-depth processed image and metric_depth shapes do not match at frame {frame_index}: "
                    f"rgb={rgb.shape[:2]} depth={depth.shape}"
                )
            preprocess_geometry = video_depth_preprocess_geometry_for_frame(
                projection_inputs["video_metric_depth_payload"],
                frame_index,
            )
            processed_mask = warp_mask_to_processed(
                mask,
                affine_2x3=np.asarray(
                    preprocess_geometry["affine_original_to_processed_2x3"],
                    dtype=np.float32,
                ),
                shape=rgb.shape[:2],
            )
            local_vertices = np.asarray(conditioned_mesh.vertices, dtype=np.float64)
            local_origin = (local_vertices.min(axis=0) + local_vertices.max(axis=0)) * 0.5
            seed_rotation = _as_array(mesh_record.get("rotation"), shape_last=4)
            seed_translation = _as_array(mesh_record.get("translation"), shape_last=3)
            seed_scale = _as_array(mesh_record.get("scale"))
            if seed_rotation is None or seed_translation is None or seed_scale is None:
                raise ValueError("SAM3D record is missing rotation, translation, or scale for pre-scale FoundationPose.")
            seed_rotation_matrix = _rotation_matrix_from_quaternion(seed_rotation)
            seed_translation = seed_translation.reshape(-1, 3)[0]
            seed_current_scale = _uniform_scale(seed_scale)
            pre_scale_foundationpose_mesh = conditioned_mesh.copy()
            pre_scale_foundationpose_mesh.vertices = (local_vertices - local_origin.reshape(1, 3)) * seed_current_scale
            pre_scale_seed_camera_vertices = _sam3d_camera_to_opencv_camera(
                _transform_vertices(local_vertices, seed_rotation_matrix, seed_translation, seed_current_scale)
            )
            pre_scale_seed_pose, pre_scale_seed_rms_error = _rigid_pose_from_corresponding_vertices(
                local_vertices=np.asarray(pre_scale_foundationpose_mesh.vertices, dtype=np.float64),
                camera_vertices=pre_scale_seed_camera_vertices,
            )
            pre_scale_context = _get_pre_scale_foundationpose_context()
            pre_scale_debug_dir = pre_scale_debug_root / str(object_id) if debug_artifacts else None
            pre_scale_pose, pre_scale_diagnostics = _pre_scale_foundationpose_refinement(
                mesh=pre_scale_foundationpose_mesh,
                seed_pose_4x4=pre_scale_seed_pose,
                rgb=rgb,
                depth=depth,
                mask=processed_mask,
                intrinsic=intrinsic,
                scorer=pre_scale_context["scorer"],
                refiner=pre_scale_context["refiner"],
                glctx=pre_scale_context["glctx"],
                debug_dir=pre_scale_debug_dir,
            )
            pre_scale_diagnostics = {
                **pre_scale_diagnostics,
                "seed_pose_source": "sam3d_raw_pose_transformed_to_pre_scale_foundationpose_metric_mesh_frame",
                "seed_pose_rms_error": pre_scale_seed_rms_error,
                "seed_pose_4x4_from_sam3d_raw_pose": pre_scale_seed_pose.tolist(),
                "mask_preprocess_geometry": preprocess_geometry,
            }
            pre_scale_debug_images = {}
            if debug_artifacts and pre_scale_debug_dir is not None:
                pre_scale_debug_images = _save_pre_scale_foundationpose_debug(
                    object_dir=pre_scale_debug_dir,
                    object_id=str(object_id),
                    rgb=rgb,
                    depth=depth,
                    mask=processed_mask,
                    seed_pose_4x4=pre_scale_seed_pose,
                    refined_pose_4x4=pre_scale_pose,
                    diagnostics=pre_scale_diagnostics,
                )
            adjusted_conditioned_mesh, projected_mesh, foundationpose_mesh, projection_info = _project_conditioned_mesh(
                mesh=conditioned_mesh,
                mesh_record=mesh_record,
                fit=fit,
                video_depth=depth,
                sam3_mask=mask,
                intrinsic=intrinsic,
                sam3d_glb_yup_rotation_undone=source_mesh_frame_conversion != "none",
                occluder_mask=occluder_union,
                pre_scale_foundationpose_pose_4x4=pre_scale_pose,
                pre_scale_foundationpose_diagnostics=pre_scale_diagnostics,
                skip_metric_alignment=_is_static_ground_fixture_track_id(geometry.get("source_track_id")),
                physion_pp_scenario=physion_pp_scenario,
                physion_pp_alignment_safety_mode=physion_pp_alignment_safety_mode,
            )
            projected_path = projected_dir / f"{object_id}_projected.glb"
            foundationpose_mesh_path = projected_dir / f"{object_id}_foundationpose_local.glb"
            adjusted_conditioned_mesh.export(conditioned_path)
            projected_mesh.export(projected_path)
            foundationpose_mesh.export(foundationpose_mesh_path)
            debug_images = {}
            if debug_artifacts and video is not None:
                debug_images = _save_projection_debug_images(
                    video=video,
                    frame_index=frame_index,
                    object_id=str(object_id),
                    object_dir=debug_root / str(object_id),
                    sam3_mask=mask,
                    video_depth=depth,
                    projected_mesh=projected_mesh,
                    intrinsic=intrinsic,
                    occluder_mask=occluder_union,
                )
            objects.append(
                {
                    "object_id": object_id,
                    "status": "ok",
                    "geometry_type": geometry_type,
                    "geometry_confidence": geometry.get("geometry_confidence"),
                    "conditioning_action": action,
                    "source_mesh_path": source_mesh_path,
                    "source_mesh_original_coordinate_frame": source_mesh_original_coordinate_frame,
                    "source_mesh_frame_conversion": source_mesh_frame_conversion,
                    "source_mesh_conditioning_coordinate_frame": "sam3d_local_frame",
                    "conditioned_mesh_path": str(conditioned_path),
                    "projected_mesh_path": str(projected_path),
                    "foundationpose_mesh_path": str(foundationpose_mesh_path),
                    "frame_index": frame_index,
                    "mask_key": mask_key,
                    "video_depth_key": "metric_depth",
                    "video_intrinsic_key": "intrinsics",
                    "rotation": mesh_record.get("rotation"),
                    "translation": mesh_record.get("translation"),
                    "scale": mesh_record.get("scale"),
                    "pose_target_convention": mesh_record.get("pose_target_convention"),
                    "fit": fit,
                    "vertex_color_transfer": vertex_color_transfer,
                    "debug_projection_images": debug_images,
                    "pre_scale_foundationpose_debug_images": pre_scale_debug_images,
                    **projection_info,
                }
            )
        except Exception as exc:
            objects.append(
                {
                    "object_id": object_id,
                    "status": "conditioning_error",
                    "geometry_type": geometry_type,
                    "source_mesh_path": source_mesh_path,
                    "error": str(exc),
                }
            )

    # Physion++ bouncy_wall two-segment reuse pass: seg2 objects were skipped by SAM3D and
    # are not in mesh_by_object. Each reuses its same-role seg1 object's conditioned mesh
    # (identical geometry + metric scale) and only re-estimates its own seg2 pose by
    # registering that mesh against the seg2 keyframe. Runs after the main loop so every
    # seg1 source object is already conditioned.
    conditioned_by_object = {obj["object_id"]: obj for obj in objects if obj.get("status") == "ok"}
    for object_id, geometry in geometry_by_object.items():
        source_object_id = geometry.get("mesh_reuse_source_object_id")
        if not source_object_id:
            continue
        geometry_type = str(geometry.get("geometry_type") or "irregular").lower()
        try:
            source_result = conditioned_by_object.get(str(source_object_id))
            if not source_result or not source_result.get("foundationpose_mesh_path"):
                raise ValueError(
                    f"Reuse source object {source_object_id} has no conditioned mesh for object {object_id}."
                )
            keyframe = projection_inputs["keyframes_by_object"].get(str(object_id))
            if not keyframe:
                raise ValueError(f"No selected keyframe found for reuse object: {object_id}")
            frame_index = int(keyframe["frame_index"])
            mask_key = keyframe.get("mask_key")
            if not mask_key:
                raise ValueError(f"Selected keyframe has no mask_key for reuse object: {object_id}")
            metric_depth = projection_inputs["metric_depth"]
            if frame_index >= metric_depth.shape[0]:
                raise ValueError(
                    f"Reuse keyframe {frame_index} exceeds video-depth frame count {int(metric_depth.shape[0])}"
                )
            mask = projection_inputs["masks"][mask_key].astype(bool)
            depth = _depth_for_frame(metric_depth, frame_index)
            intrinsic = _intrinsic_for_frame(projection_inputs["intrinsics"], frame_index)
            rgb = _processed_rgb_for_frame(projection_inputs["processed_images"], frame_index)
            preprocess_geometry = video_depth_preprocess_geometry_for_frame(
                projection_inputs["video_metric_depth_payload"], frame_index
            )
            processed_mask = warp_mask_to_processed(
                mask,
                affine_2x3=np.asarray(
                    preprocess_geometry["affine_original_to_processed_2x3"], dtype=np.float32
                ),
                shape=rgb.shape[:2],
            )
            reuse_mesh = _load_mesh(Path(source_result["foundationpose_mesh_path"]))
            source_pose = np.asarray(
                source_result["foundationpose_initial_pose_4x4"], dtype=np.float64
            ).reshape(4, 4)
            # Seed rotation from the seg1 pose (same camera; static wall/mat share orientation,
            # sphere agent is rotation-free) and seg2 translation from the seg2 mask+depth.
            seed_pose = np.eye(4, dtype=np.float64)
            seed_pose[:3, :3] = source_pose[:3, :3]
            seed_pose[:3, 3] = _guess_translation_from_mask_depth(
                mask=processed_mask, depth=depth, intrinsic=intrinsic
            )
            pre_scale_context = _get_pre_scale_foundationpose_context()
            reuse_debug_dir = pre_scale_debug_root / str(object_id) if debug_artifacts else None
            seg2_pose, seg2_diag = _pre_scale_foundationpose_refinement(
                mesh=reuse_mesh,
                seed_pose_4x4=seed_pose,
                rgb=rgb,
                depth=depth,
                mask=processed_mask,
                intrinsic=intrinsic,
                scorer=pre_scale_context["scorer"],
                refiner=pre_scale_context["refiner"],
                glctx=pre_scale_context["glctx"],
                debug_dir=reuse_debug_dir,
            )
            seg2_pose = np.asarray(seg2_pose, dtype=np.float64).reshape(4, 4)

            # DEPTH-ONLY alignment in seg2's OWN frame: the reused seg1 mesh keeps seg1's
            # EXACT size (scale LOCKED) for cross-segment size consistency, and only its
            # depth is re-fit to seg2's observed depth (the object was re-placed between the
            # two trials). The seg2 sub-clip VDA depth is already at the seg1 metric (see
            # run_video_metric_depth), so no area rescale is needed; and occ IoU cannot
            # separate a flat mat's size from its depth anyway, so we trust seg1's size and
            # only move it in depth. (area_scale stays 1.0 for the whole loop.)
            fp_mesh_path_final = source_result.get("foundationpose_mesh_path")
            pose_final = seg2_pose
            final_area_scale = 1.0
            try:
                reuse_verts = np.asarray(reuse_mesh.vertices, dtype=np.float64)
                reuse_faces = np.asarray(reuse_mesh.faces, dtype=np.int64)
                reuse_center = (reuse_verts.min(axis=0) + reuse_verts.max(axis=0)) * 0.5
                align_mask = mask if mask.shape == depth.shape else _resize_mask_nearest(mask, depth.shape)
                reuse_occluder = _all_object_mask_union_at_frame(projection_inputs["masks"], frame_index)
                target_depth = float(_mask_depth_stats(depth, align_mask)["median"])
                target_area = int(np.asarray(align_mask, dtype=bool).sum())
                align_intrinsic = np.asarray(intrinsic, dtype=np.float64)
                area_tolerance = _env_float("PHYSMIND_MESH_CONDITIONING_AREA_TOLERANCE", DEFAULT_AREA_ALIGNMENT_TOLERANCE)
                depth_tolerance = _env_float("PHYSMIND_MESH_CONDITIONING_DEPTH_TOLERANCE", DEFAULT_DEPTH_ALIGNMENT_TOLERANCE)
                align_max_iters = max(1, _env_int("PHYSMIND_MESH_CONDITIONING_AREA_MAX_ITERATIONS", DEFAULT_AREA_ALIGNMENT_MAX_ITERATIONS))
                aligned_pose = seg2_pose.copy()
                area_scale = 1.0
                align_trace: list[dict[str, Any]] = []
                align_stop = "max_iterations"
                for align_iter in range(align_max_iters):
                    scaled_verts = reuse_center + (reuse_verts - reuse_center) * area_scale
                    projected = scaled_verts @ aligned_pose[:3, :3].T + aligned_pose[:3, 3].reshape(1, 3)
                    metrics, _ = _projection_metrics(
                        vertices=projected,
                        faces=reuse_faces,
                        intrinsic=align_intrinsic,
                        image_shape=depth.shape,
                        target_mask=align_mask,
                        target_area=target_area,
                        target_depth=target_depth,
                        occluder_mask=reuse_occluder,
                    )
                    depth_delta = target_depth - float(metrics["rendered_mesh_depth"])
                    aligned_pose[2, 3] += depth_delta
                    area_err = abs(float(metrics["area_error_fraction"]))
                    depth_err = abs(depth_delta) / target_depth
                    align_trace.append(
                        {
                            "iteration": align_iter + 1,
                            "area_scale": float(area_scale),
                            "depth_delta": float(depth_delta),
                            "rendered_area": int(metrics["rendered_area"]),
                            "target_area": target_area,
                            "rendered_mesh_depth": float(metrics["rendered_mesh_depth"]),
                            "area_error_fraction": float(metrics["area_error_fraction"]),
                        }
                    )
                    if depth_err <= depth_tolerance:
                        align_stop = "depth_tolerance"
                        break
                    if align_iter + 1 >= align_max_iters:
                        break
                    # scale is LOCKED to seg1 (area_scale stays 1.0); only depth is re-fit,
                    # so nothing updates area_scale here -- the loop just converges the depth.
                # Export the reused foundationpose mesh (its own file). Size is LOCKED to seg1
                # (area_scale == 1.0), so this is seg1's exact mesh; only the pose depth moved.
                reuse_fp_mesh = reuse_mesh.copy()
                reuse_fp_mesh.vertices = reuse_center + (reuse_verts - reuse_center) * area_scale
                reuse_fp_path = projected_dir / f"{object_id}_foundationpose_local.glb"
                reuse_fp_mesh.export(reuse_fp_path)
                fp_mesh_path_final = str(reuse_fp_path)
                pose_final = aligned_pose
                final_area_scale = float(area_scale)
                align_diag = {
                    "applied": True,
                    "policy": "depth_only_alignment_seg1_size_locked_on_reused_shape",
                    "target_depth": target_depth,
                    "target_area": target_area,
                    "final_area_scale_from_seg1": float(area_scale),
                    "stop_reason": align_stop,
                    "iterations": align_trace,
                }
            except Exception as align_exc:
                align_diag = {"applied": False, "reason": f"metric alignment failed: {align_exc}"}

            objects.append(
                {
                    "object_id": object_id,
                    "status": "ok",
                    "geometry_type": geometry_type,
                    "geometry_confidence": geometry.get("geometry_confidence"),
                    "conditioning_action": "reused_from_other_segment_metric_aligned",
                    "mesh_reuse_source_object_id": str(source_object_id),
                    "source_mesh_path": source_result.get("source_mesh_path"),
                    "conditioned_mesh_path": source_result.get("conditioned_mesh_path"),
                    "projected_mesh_path": source_result.get("projected_mesh_path"),
                    "foundationpose_mesh_path": fp_mesh_path_final,
                    "foundationpose_mesh_coordinate_frame": source_result.get(
                        "foundationpose_mesh_coordinate_frame"
                    ),
                    "projected_mesh_coordinate_frame": source_result.get("projected_mesh_coordinate_frame"),
                    # Uniform metric-area scaling preserves the source's canonical axis directions,
                    # so the source copy applies unchanged; pose_correction needs them to base-align.
                    "foundationpose_local_axes": source_result.get("foundationpose_local_axes"),
                    "frame_index": frame_index,
                    "mask_key": mask_key,
                    "video_depth_key": "metric_depth",
                    "video_intrinsic_key": "intrinsics",
                    "scale": source_result.get("scale"),
                    "reuse_metric_area_scale_from_seg1": final_area_scale,
                    "fit": source_result.get("fit"),
                    "foundationpose_initial_pose_4x4": pose_final.tolist(),
                    "foundationpose_local_origin_camera_xyz": pose_final[:3, 3].tolist(),
                    "mesh_reuse_registration_diagnostics": seg2_diag,
                    "mesh_reuse_metric_alignment": align_diag,
                }
            )
        except Exception as exc:
            objects.append(
                {
                    "object_id": object_id,
                    "status": "reuse_conditioning_error",
                    "geometry_type": geometry_type,
                    "mesh_reuse_source_object_id": str(source_object_id),
                    "error": str(exc),
                }
            )

    payload = {
        "tool": "mesh_conditioning",
        "status": "ok",
        "object_plan": str(object_plan),
        "source_mesh_artifact": str(artifact_path_by_name(question_dir, "sam3d_meshes.json")),
        "projection_sources": {
            "track_labels_artifact": projection_inputs["track_labels_artifact"],
            "video_metric_depth_artifact": projection_inputs["video_metric_depth_artifact"],
        },
        "sam3d_generation": {
            "tool": sam3d_meshes.get("tool"),
            "status": sam3d_meshes.get("status"),
            "config_path": sam3d_meshes.get("config_path"),
            "seed": sam3d_meshes.get("seed"),
            "use_vertex_color": sam3d_meshes.get("use_vertex_color"),
            "with_texture_baking": sam3d_meshes.get("with_texture_baking"),
            "with_mesh_postprocess": sam3d_meshes.get("with_mesh_postprocess"),
            "layout_postprocess": sam3d_meshes.get("layout_postprocess"),
            "object_count": len(sam3d_meshes.get("objects", [])),
            "objects": [
                {
                    "object_id": item.get("object_id"),
                    "status": item.get("status"),
                    "frame_index": item.get("frame_index"),
                    "mask_key": item.get("mask_key"),
                    "rotation": item.get("rotation"),
                    "translation": item.get("translation"),
                    "scale": item.get("scale"),
                    "pose_target_convention": item.get("pose_target_convention"),
                }
                for item in sam3d_meshes.get("objects", [])
                if isinstance(item, dict)
            ],
        },
        "mesh_target_faces": int(target_faces),
        "objects": objects,
        "note": (
            "Regular geometry types are refit to deterministic low-face primitives; irregular meshes are decimated. "
            "Conditioned meshes are projected into the video-depth camera frame for diagnostics. FoundationPose receives "
            "a metric canonical mesh centered on its local bounds center, with a mathematically equivalent initial pose "
            "recorded without depth or area rescaling."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-faces", type=int, default=CLEVRER_MESH_TARGET_FACES)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_mesh_conditioning(
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        target_faces=args.target_faces,
        video=Path(args.video) if args.video else None,
    )


if __name__ == "__main__":
    main()
