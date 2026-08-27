from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agent.world_model.artifacts import artifact_path_by_name, question_root_from_output
from scripts.world_model.occlusion_keyframe import select_occlusion_aware_keyframes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3D_ROOT = PROJECT_ROOT / "third_party" / "sam-3d-objects"
SAM3D_OBSERVATION_SELECTION_DECISION_ID = "GEO-001.sam3d_observation_selection"
STATIC_TEMPORAL_COMPOSITE_ROUTE = (
    "geometry.static_temporal_composite"
)
SELECTED_OBSERVATION_ROUTE = "geometry.selected_observation"
SAM3D_OBSERVATION_SELECTION_ROUTES = frozenset(
    {STATIC_TEMPORAL_COMPOSITE_ROUTE, SELECTED_OBSERVATION_ROUTE}
)


def _add_sam3d_to_path() -> None:
    if not SAM3D_ROOT.exists():
        raise FileNotFoundError(f"SAM 3D Objects submodule not found: {SAM3D_ROOT}")
    sys.path.insert(0, str(SAM3D_ROOT))
    sys.path.insert(0, str(SAM3D_ROOT / "notebook"))


def _patch_utils3d_for_sam3d() -> None:
    import utils3d.numpy as utils3d_np

    if not hasattr(utils3d_np, "depth_edge"):
        utils3d_np.depth_edge = utils3d_np.depth_map_edge
    if not hasattr(utils3d_np, "normals_edge"):
        utils3d_np.normals_edge = utils3d_np.normal_map_edge
    if not hasattr(utils3d_np, "points_to_normals"):
        utils3d_np.points_to_normals = utils3d_np.point_map_to_normal_map
    if not hasattr(utils3d_np, "image_uv"):
        utils3d_np.image_uv = lambda *, width, height: utils3d_np.uv_map(height, width)
    if not hasattr(utils3d_np, "image_mesh"):
        utils3d_np.image_mesh = utils3d_np.build_mesh_from_map


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_fingerprint(mask: np.ndarray) -> str:
    mask_u8 = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(str(mask_u8.shape).encode("utf-8"))
    digest.update(mask_u8.tobytes())
    return digest.hexdigest()


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for SAM3D keyframe: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Unable to read SAM3D keyframe {frame_index} from video: {video}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _image_fingerprint(image: np.ndarray) -> str:
    image = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("utf-8"))
    digest.update(str(image.dtype).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _read_selected_keyframe(item: dict[str, Any], *, video: Path) -> tuple[np.ndarray, Path | None, str, str]:
    frame_path_value = item.get("selected_frame_image")
    if not frame_path_value:
        frame_index = item.get("frame_index")
        if frame_index is None:
            raise ValueError("SAM3D keyframe is missing both selected_frame_image and frame_index.")
        image = _read_video_frame(video, int(frame_index))
        return image, None, _image_fingerprint(image), "source_video_frame"
    frame_path = Path(str(frame_path_value))
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read selected keyframe image: {frame_path}")
    image_fingerprint = _file_fingerprint(frame_path)
    expected_fingerprint = item.get("selected_frame_image_fingerprint")
    if expected_fingerprint and str(expected_fingerprint) != image_fingerprint:
        raise ValueError(
            f"Selected keyframe image fingerprint mismatch: {frame_path} "
            f"expected={expected_fingerprint} actual={image_fingerprint}"
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), frame_path, image_fingerprint, "selected_frame_image"


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _mesh_vertices(mesh: Any) -> np.ndarray | None:
    if mesh is None:
        return None
    if isinstance(mesh, list):
        vertices = [_mesh_vertices(item) for item in mesh]
        vertices = [item for item in vertices if item is not None and len(item) > 0]
        if not vertices:
            return None
        return np.concatenate(vertices, axis=0)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        return None
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    elif hasattr(vertices, "cpu") and hasattr(vertices, "numpy"):
        vertices = vertices.cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return None
    return vertices


def _mesh_bbox_summary(mesh: Any) -> dict[str, Any]:
    vertices = _mesh_vertices(mesh)
    if vertices is None or len(vertices) == 0:
        return {}
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    extent = upper - lower
    return {
        "mesh_vertex_count": int(len(vertices)),
        "mesh_bbox_min": lower.tolist(),
        "mesh_bbox_max": upper.tolist(),
        "mesh_center": ((lower + upper) * 0.5).tolist(),
        "mesh_extent": extent.tolist(),
        "mesh_bbox_diag": float(np.linalg.norm(extent)),
    }


def _mesh_visual_summary(mesh: Any) -> dict[str, Any]:
    if mesh is None:
        return {"has_vertex_color": False}
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    colors = None if vertex_colors is None else np.asarray(vertex_colors)
    return {
        "visual_type": type(visual).__name__ if visual is not None else None,
        "has_vertex_color": colors is not None and colors.size > 0,
        "vertex_color_shape": list(colors.shape) if colors is not None else None,
    }


def _pose_summary(output: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in ("rotation", "translation", "scale", "pose_target_convention"):
        if key in output:
            fields[key] = _to_jsonable(output[key])
    if fields:
        fields["pose_note"] = "SAM 3D Objects local-to-camera/scene pose fields from the raw inference output."
    return fields


def _normalize_intrinsics(pixel_k: np.ndarray, *, width: int, height: int) -> np.ndarray:
    k = np.asarray(pixel_k, dtype=np.float32).copy()
    k[0, :] /= float(width)
    k[1, :] /= float(height)
    return k


def _depth_to_sam3d_pointmap(depth: np.ndarray, pixel_k: np.ndarray, *, width: int, height: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    fx, fy = float(pixel_k[0, 0]), float(pixel_k[1, 1])
    cx, cy = float(pixel_k[0, 2]), float(pixel_k[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid fixed camera focal lengths for SAM3D pointmap: fx={fx}, fy={fy}")

    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    pointmap = np.stack([-x, -y, depth], axis=-1).astype(np.float32)
    pointmap[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return pointmap


PHYSION_PP_STATIC_TRACK_PREFIX = "physion_pp_static_"


def _load_video_frames_rgb(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"Unable to read any frame from video: {video}")
    return np.stack(frames)


def _track_mask_stack(masks: Any, track_id: str, n_frames: int, shape: tuple[int, int]) -> np.ndarray:
    stack = np.zeros((n_frames, *shape), dtype=bool)
    prefix = track_id + "__frame_"
    for key in masks.files:
        if key.startswith(prefix):
            frame = int(key.split("__frame_")[1][:5])
            if frame < n_frames:
                stack[frame] = masks[key].astype(bool)
    return stack


def _mover_free_valid_stack(masks: Any, n_frames: int, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Per frame/pixel validity: not covered by any non-static (moving) track's mask."""
    mover_tracks = sorted({
        key.split("__frame_")[0]
        for key in masks.files
        if not key.split("__frame_")[0].startswith(PHYSION_PP_STATIC_TRACK_PREFIX)
    })
    movers = np.zeros((n_frames, *shape), dtype=bool)
    for track_id in mover_tracks:
        movers |= _track_mask_stack(masks, track_id, n_frames, shape)
    return ~movers, len(mover_tracks)


def _physion_pp_static_temporal_composite(
    *,
    fixture_stack: np.ndarray,
    valid_stack: np.ndarray,
    frames_rgb: np.ndarray,
    metric_depth: np.ndarray,
    selected_mask: np.ndarray,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Temporal composite for a STATIC fixture under a MOVING occluder (friction_platform_pp):
    per pixel, only the frames where no moving track covers the pixel are used, because the
    friction agent starts slowly and can sit on one pixel for more than half the video, which
    poisons plain medians/majorities. Pixels with zero mover-free frames (occluded in every
    frame) keep the selected frame's values, i.e. degrade to today's behavior."""
    n_frames = valid_stack.shape[0]
    n_valid = valid_stack.sum(axis=0)
    fallback = n_valid == 0

    fixture_counts = (fixture_stack & valid_stack).sum(axis=0)
    mask = fixture_counts > (n_valid / 2.0)
    mask[fallback] = selected_mask[fallback]

    rgb = frames_rgb[:n_frames].astype(np.float32)
    rgb[~valid_stack] = np.nan
    with np.errstate(all="ignore"):
        image = np.nanmedian(rgb, axis=0)
    image[fallback] = frames_rgb[frame_index][fallback]
    image = image.astype(np.uint8)

    depth = np.asarray(metric_depth[:n_frames], dtype=np.float32).copy()
    depth[~valid_stack] = np.nan
    with np.errstate(all="ignore"):
        depth_median = np.nanmedian(depth, axis=0)
    depth_median[fallback] = np.asarray(metric_depth[frame_index], dtype=np.float32)[fallback]

    info = {
        "applied": True,
        "method": "per_pixel_mover_free_temporal_composite",
        "statistics": "mask=majority, rgb=median, depth=median over mover-free frames per pixel",
        "frame_count": int(n_frames),
        "zero_mover_free_pixel_count": int(fallback.sum()),
        "min_mover_free_frames": int(n_valid.min()),
    }
    return image, mask, depth_median, info


def _sam3d_intrinsics_summary(
    output: dict[str, Any],
    image: np.ndarray,
    pixel_intrinsics: np.ndarray,
) -> dict[str, Any]:
    pixel_intrinsics = np.asarray(pixel_intrinsics, dtype=np.float32)
    if pixel_intrinsics.shape != (3, 3):
        return {
            "sam3d_intrinsics_available": False,
            "sam3d_intrinsics_error": f"Unexpected SAM3D intrinsics shape: {list(pixel_intrinsics.shape)}",
        }

    height, width = image.shape[:2]
    normalized_intrinsics = _normalize_intrinsics(pixel_intrinsics, width=width, height=height)
    pointmap = output.get("pointmap")
    pointmap_shape = None
    if pointmap is not None:
        pointmap_shape = list(pointmap.shape) if hasattr(pointmap, "shape") else list(np.asarray(pointmap).shape)
    return {
        "sam3d_intrinsics_available": True,
        "sam3d_intrinsics_source": "moge2_intrinsics.camera_intrinsics.K_fixed",
        "sam3d_intrinsics_normalized": normalized_intrinsics.tolist(),
        "sam3d_intrinsics_pixel_original_frame": pixel_intrinsics.tolist(),
        "sam3d_intrinsics_coordinate_frame": "video_original",
        "sam3d_intrinsics_normalized_coordinate_frame": "sam3d_internal_normalized_image",
        "sam3d_pointmap_shape": pointmap_shape,
    }


def _export_outputs(output: dict[str, Any], object_dir: Path, object_id: str) -> dict[str, Any]:
    object_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    glb = output.get("glb")
    if glb is not None:
        glb_path = object_dir / f"{object_id}.glb"
        glb.export(str(glb_path))
        artifacts["glb_path"] = str(glb_path)
        artifacts["visual"] = _mesh_visual_summary(glb)
    mesh = output.get("mesh")
    if mesh is not None:
        artifacts["mesh_summary"] = str(type(mesh))
        artifacts.update(_mesh_bbox_summary(mesh))
    artifacts.update(_pose_summary(output))
    for key in (
        "sam3d_intrinsics_available",
        "sam3d_intrinsics_source",
        "sam3d_intrinsics_normalized",
        "sam3d_intrinsics_pixel_original_frame",
        "sam3d_intrinsics_coordinate_frame",
        "sam3d_intrinsics_normalized_coordinate_frame",
        "sam3d_pointmap_shape",
        "pointmap_source",
        "pointmap_intrinsics_source",
        "pointmap_depth_frame_index",
    ):
        if key in output:
            artifacts[key] = _to_jsonable(output[key])
    return artifacts


def _run_inference_with_layout_postprocess(
    inference: Any,
    image: np.ndarray,
    mask: np.ndarray,
    seed: int,
    *,
    pointmap: np.ndarray,
    pixel_intrinsics: np.ndarray,
    frame_index: int,
) -> dict[str, Any]:
    import torch

    rgba_image = inference.merge_mask_to_rgba(image, mask)
    output = inference._pipeline.run(
        rgba_image,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=torch.from_numpy(pointmap),
    )
    output.update(_sam3d_intrinsics_summary(output, image, pixel_intrinsics))
    output.update(
        {
            "pointmap_source": "video_metric_depth.metric_depth_backprojection",
            "pointmap_intrinsics_source": "moge2_intrinsics.camera_intrinsics.K_fixed",
            "pointmap_depth_frame_index": int(frame_index),
        }
    )
    return output


def load_sam3d_inference(*, config_path: Path, compile_model: bool) -> Any:
    _add_sam3d_to_path()
    _patch_utils3d_for_sam3d()
    from inference import Inference

    return Inference(str(config_path), compile=compile_model)


def _sam3d_observation_selection_route_from_object_plan(
    object_plan_payload: dict[str, Any],
) -> dict[str, Any] | None:
    special = object_plan_payload.get("special_scene")
    special = special if isinstance(special, dict) else {}
    route_record = special.get("sam3d_observation_selection_route")
    if not isinstance(route_record, dict):
        metadata = special.get("scene_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        benchmark = str(
            special.get("benchmark") or metadata.get("benchmark") or ""
        ).strip().lower()
        scenario = str(
            special.get("scenario") or metadata.get("scenario") or ""
        ).strip().lower()
        is_target_scope = (
            benchmark == "clevrer" or scenario.endswith("_pp")
        )
        if is_target_scope:
            raise ValueError(
                "target SAM3D object plan is missing its observation-selection "
                "route record"
            )
        return None
    if (
        route_record.get("decision_id")
        != SAM3D_OBSERVATION_SELECTION_DECISION_ID
    ):
        raise ValueError(
            "SAM3D-observation-selection route record has an unexpected "
            f"decision_id: {route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in SAM3D_OBSERVATION_SELECTION_ROUTES:
        raise ValueError(
            f"unsupported SAM3D-observation-selection route: {route!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError(
            "SAM3D-observation-selection route record is missing its context"
        )
    benchmark = str(context.get("benchmark") or "").strip().lower()
    scenario = str(context.get("scenario") or "").strip().lower() or None
    if benchmark == "clevrer" and scenario is not None:
        raise ValueError(
            "CLEVRER SAM3D observation route must not contain a scenario context"
        )
    if benchmark == "physion_pp" and scenario is None:
        raise ValueError(
            "Physion++ SAM3D observation route is missing its scenario context"
        )
    if benchmark not in {"clevrer", "physion_pp"}:
        raise ValueError(
            "SAM3D-observation-selection route has an unsupported benchmark "
            f"context: {context.get('benchmark')!r}"
        )
    return route_record


def _static_temporal_composite_enabled(
    route_record: dict[str, Any] | None,
) -> bool:
    if route_record is None:
        return False
    route = route_record.get("route")
    if route == STATIC_TEMPORAL_COMPOSITE_ROUTE:
        return True
    if route == SELECTED_OBSERVATION_ROUTE:
        return False
    raise ValueError(f"unsupported SAM3D-observation-selection route: {route!r}")


def _observation_policy_benchmark(route_record: dict[str, Any] | None) -> str:
    route_record = route_record if isinstance(route_record, dict) else {}
    context = route_record.get("context")
    context = context if isinstance(context, dict) else {}
    return str(context.get("benchmark") or "").strip().lower()


def _maybe_reselect_keyframes_physion_pp(
    *,
    observation_selection_route: dict[str, Any] | None,
    keyframes: list[dict[str, Any]],
    masks: Any,
    metric_depth: np.ndarray,
    track_labels: dict[str, Any],
    track_labels_path: Path,
) -> dict[str, Any]:
    """Physion++ only: re-pick each object's SAM3D keyframe with occlusion-aware logic and
    sync the choice back into object_keyframes (so FoundationPose registration uses it too).
    No-op (and safe) for non-Physion++ routes or on any failure."""
    if _observation_policy_benchmark(observation_selection_route) != "physion_pp":
        return {"applied": False, "reason": "not_physion_pp"}
    try:
        selection = select_occlusion_aware_keyframes(
            object_keyframes=keyframes, masks=masks, metric_depth=metric_depth
        )
    except Exception as error:  # never break reconstruction over re-selection
        return {"applied": False, "reason": f"reselection_error: {error}"}

    changed = []
    for keyframe in keyframes:
        object_id = str(keyframe.get("object_id"))
        pick = selection.get(object_id)
        if not pick or pick.get("mask_key") is None:
            continue
        old_frame = keyframe.get("frame_index")
        new_frame = pick["frame_index"]
        keyframe["occlusion_aware_reselection"] = {
            "previous_frame_index": old_frame,
            "previous_mask_key": keyframe.get("mask_key"),
            "selection_rule": pick["selection_rule"],
            "fallback_used": pick["fallback_used"],
            "selected_role": pick["selected_role"],
            "eligible_frame_count": pick["eligible_frame_count"],
            "candidate_frame_count": pick["candidate_frame_count"],
        }
        if int(new_frame) != int(old_frame if old_frame is not None else -1):
            keyframe["frame_index"] = int(new_frame)
            keyframe["mask_key"] = pick["mask_key"]
            keyframe["area"] = pick["area"]
            # Regenerate all frame-dependent derivatives for the selected keyframe.
            for stale in (
                "selected_frame_image",
                "selected_frame_image_fingerprint",
                "selected_frame_image_format",
                "representative_image_path",
                "representative_overlay_path",
                "bbox_xyxy",
                "centroid_xy",
            ):
                keyframe.pop(stale, None)
            changed.append((object_id, old_frame, int(new_frame)))

    track_labels["object_keyframe_reselection"] = {
        "applied": True,
        "method": "occlusion_aware_depth_front_back_largest_clean_area",
        "changed": [{"object_id": o, "from": f, "to": t} for o, f, t in changed],
    }
    if changed:
        track_labels_path.write_text(json.dumps(track_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"applied": True, "changed": changed}


def run_sam3d_objects(
    *,
    video: Path,
    object_plan: Path,
    output: Path,
    config_path: Path,
    seed: int,
    compile_model: bool,
    preloaded_inference: Any | None = None,
    execution_mode: str = "external_command",
) -> None:
    object_plan_payload = _load_json(object_plan)
    observation_selection_route = (
        _sam3d_observation_selection_route_from_object_plan(
            object_plan_payload
        )
    )
    question_dir = question_root_from_output(output)
    track_labels = _load_json(artifact_path_by_name(question_dir, "sam3_video_track_labels.json"))
    keyframes = track_labels.get("object_keyframes", [])
    mask_sidecar = Path(track_labels["mask_sidecar"])
    masks = np.load(mask_sidecar)
    video_metric_depth_path = artifact_path_by_name(question_dir, "video_metric_depth.json")
    video_metric_depth = _load_json(video_metric_depth_path)
    depth_arrays = np.load(video_metric_depth["tensor_sidecar"])
    metric_depth = depth_arrays["metric_depth"]
    _maybe_reselect_keyframes_physion_pp(
        observation_selection_route=observation_selection_route,
        keyframes=keyframes,
        masks=masks,
        metric_depth=metric_depth,
        track_labels=track_labels,
        track_labels_path=artifact_path_by_name(question_dir, "sam3_video_track_labels.json"),
    )
    fixed_k = np.asarray(video_metric_depth.get("fixed_intrinsics"), dtype=np.float32)
    if fixed_k.shape != (3, 3) or not np.isfinite(fixed_k).all():
        raise ValueError(
            "video_metric_depth.json fixed_intrinsics must contain finite 3x3 MoGe-2 K_fixed. "
            "Re-run moge2_intrinsics and video_metric_depth."
        )
    # The resolved special route rebuilds static-fixture RGB/mask/depth inputs from
    # mover-free frames; the shared route keeps the selected keyframe observation.
    # Which scenario receives the special route is defined only in route-policy JSON.
    composite_static_inputs = _static_temporal_composite_enabled(
        observation_selection_route
    )
    video_frames_rgb: np.ndarray | None = None
    mover_valid_stack: np.ndarray | None = None
    mover_track_count = 0
    inference = preloaded_inference

    # Physion++ bouncy_wall two-segment: seg2 objects carry mesh_reuse_source_object_id and
    # reuse their same-role seg1 mesh, so SAM3D does not reconstruct them here. mesh_conditioning
    # redirects them to the seg1 conditioned mesh and re-estimates only their seg2 pose.
    reuse_source_by_object = {
        str(target.get("object_id")): str(target.get("mesh_reuse_source_object_id"))
        for target in _load_json(object_plan).get("target_objects", [])
        if target.get("object_id") and target.get("mesh_reuse_source_object_id")
    }

    object_results = []
    reused_object_results = []
    for item in keyframes:
        object_id = str(item["object_id"])
        if object_id in reuse_source_by_object:
            reused_object_results.append(
                {
                    "object_id": object_id,
                    "status": "reused_mesh_from_other_segment",
                    "mesh_reuse_source_object_id": reuse_source_by_object[object_id],
                }
            )
            continue
        frame_index = item.get("frame_index")
        mask_key = item.get("mask_key")
        if frame_index is None or not mask_key:
            object_results.append(
                {
                    "object_id": object_id,
                    "status": "missing_keyframe_or_mask",
                    "frame_index": frame_index,
                    "mask_key": mask_key,
                }
            )
            continue
        image, selected_frame_image, image_fingerprint, image_source = _read_selected_keyframe(item, video=video)
        mask = masks[mask_key].astype(bool)
        track_id = str(mask_key).split("__frame_")[0]
        composite_applied = composite_static_inputs and track_id.startswith(PHYSION_PP_STATIC_TRACK_PREFIX)
        composite_info: dict[str, Any] = {}
        composite_depth: np.ndarray | None = None
        if composite_applied:
            if video_frames_rgb is None:
                video_frames_rgb = _load_video_frames_rgb(video)
            n_frames = min(len(video_frames_rgb), len(metric_depth))
            if mover_valid_stack is None:
                mover_valid_stack, mover_track_count = _mover_free_valid_stack(
                    masks, n_frames, mask.shape
                )
            fixture_stack = _track_mask_stack(masks, track_id, n_frames, mask.shape)
            image, mask, composite_depth, composite_info = _physion_pp_static_temporal_composite(
                fixture_stack=fixture_stack,
                valid_stack=mover_valid_stack,
                frames_rgb=video_frames_rgb,
                metric_depth=metric_depth,
                selected_mask=mask,
                frame_index=int(frame_index),
            )
            composite_info["mover_track_count"] = mover_track_count
            image_source = "physion_pp_static_temporal_composite"
            image_fingerprint = _array_fingerprint(image)
        mask_fingerprint = _mask_fingerprint(mask)
        depth_frame_index = int(frame_index)
        if depth_frame_index < 0 or depth_frame_index >= len(metric_depth):
            raise IndexError(
                f"SAM3D keyframe {depth_frame_index} is outside video_metric_depth frame range [0, {len(metric_depth)})."
            )
        height, width = image.shape[:2]
        pointmap = _depth_to_sam3d_pointmap(
            composite_depth if composite_depth is not None else metric_depth[depth_frame_index],
            fixed_k,
            width=width,
            height=height,
        )
        object_dir = output.parent / "meshes" / object_id

        if inference is None:
            inference = load_sam3d_inference(config_path=config_path, compile_model=compile_model)
        result = _run_inference_with_layout_postprocess(
            inference,
            image,
            mask,
            seed,
            pointmap=pointmap,
            pixel_intrinsics=fixed_k,
            frame_index=int(frame_index),
        )
        artifacts = _export_outputs(result, object_dir, object_id)
        object_results.append(
            {
                "object_id": object_id,
                "status": "ok",
                "layout_postprocess": True,
                "with_texture_baking": False,
                "with_mesh_postprocess": False,
                "use_vertex_color": True,
                "frame_index": frame_index,
                "mask_key": mask_key,
                "selected_frame_image": str(selected_frame_image) if selected_frame_image else None,
                "selected_frame_image_source": image_source,
                "selected_frame_image_fingerprint": image_fingerprint,
                "mask_fingerprint": mask_fingerprint,
                "geometry_fingerprint": _array_fingerprint(pointmap, fixed_k),
                **({"physion_pp_static_temporal_composite": composite_info} if composite_applied else {}),
                **artifacts,
            }
        )

    camera_intrinsics = {
        "available": True,
        "source": "moge2_intrinsics.camera_intrinsics.K_fixed",
        "coordinate_frame": "video_original",
        "K_fixed": fixed_k.tolist(),
        "source_artifact": str(video_metric_depth.get("camera_intrinsics_artifact")),
    }
    payload = {
        "tool": "sam3d_objects",
        "status": "ok",
        "execution_mode": execution_mode,
        "video": str(video),
        "object_plan": str(object_plan),
        "config_path": str(config_path),
        "seed": seed,
        "layout_postprocess": True,
        "with_texture_baking": False,
        "with_mesh_postprocess": False,
        "use_vertex_color": True,
        "camera_intrinsics": camera_intrinsics,
        "objects": object_results,
        "reused_objects": reused_object_results,
    }
    if observation_selection_route is not None:
        payload["sam3d_observation_selection_route"] = (
            observation_selection_route
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-path", default=str(SAM3D_ROOT / "checkpoints" / "hf" / "pipeline.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    run_sam3d_objects(
        video=Path(args.video),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        config_path=Path(args.config_path),
        seed=args.seed,
        compile_model=args.compile,
    )


if __name__ == "__main__":
    main()
