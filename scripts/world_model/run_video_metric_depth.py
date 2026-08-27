from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VDA_ROOT = PROJECT_ROOT / "third_party" / "Video-Depth-Anything"
DEFAULT_VIDEO_DEPTH_CHECKPOINT = VDA_ROOT / "checkpoints" / "metric_video_depth_anything_vitl.pth"
DEFAULT_VIDEO_METRIC_DEPTH_MODEL = str(DEFAULT_VIDEO_DEPTH_CHECKPOINT)
VIDEO_DEPTH_INPUT_SIZE = 518
VIDEO_DEPTH_MAX_RES = 1280
VIDEO_DEPTH_ENCODER = "vitl"
VIDEO_DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
DEPTH_BACKEND = "video_depth_anything_metric_offline"
DEPTH_PARTITION_DECISION_ID = "DEP-002.depth_partition"
WHOLE_VIDEO_VDA_ROUTE = "depth.vda_full_video"
SPLIT_VIDEO_VDA_ROUTE = "depth.vda_two_segment_affine"
DEPTH_PARTITION_ROUTES = frozenset(
    {WHOLE_VIDEO_VDA_ROUTE, SPLIT_VIDEO_VDA_ROUTE}
)
SEGMENT_DEPTH_ALIGNMENT_DECISION_ID = "DEP-003.segment_depth_alignment"
SEGMENT_DEPTH_ALIGNMENT_ROUTE = (
    "depth.affine_background_curtain"
)


def _add_video_depth_anything_to_path() -> None:
    if VDA_ROOT.exists():
        sys.path.insert(0, str(VDA_ROOT))
        sys.path.insert(0, str(VDA_ROOT / "video_depth_anything"))
        vda_utils = VDA_ROOT / "utils"
        utils_pkg = sys.modules.get("utils")
        if utils_pkg is None:
            utils_pkg = types.ModuleType("utils")
            utils_pkg.__path__ = [str(vda_utils)]  # type: ignore[attr-defined]
            sys.modules["utils"] = utils_pkg
        elif hasattr(utils_pkg, "__path__") and str(vda_utils) not in utils_pkg.__path__:  # type: ignore[attr-defined]
            utils_pkg.__path__.insert(0, str(vda_utils))  # type: ignore[attr-defined]
        for module_name in ("util", "dc_utils"):
            full_name = f"utils.{module_name}"
            if full_name in sys.modules:
                continue
            module_path = vda_utils / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(full_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load Video-Depth-Anything module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)


def _video_metadata(video_path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        return {
            "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _write_processed_frames(frames: np.ndarray, output_dir: Path) -> list[str]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, frame in enumerate(frames):
        path = output_dir / f"frame_{frame_index:05d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(str(path))
    return frame_paths


def _write_depth_colormap_video(depths: np.ndarray, output_path: Path, *, fps: float) -> dict[str, Any]:
    import cv2

    finite = depths[np.isfinite(depths)]
    if finite.size == 0:
        return {"status": "skipped", "reason": "no_finite_depth_values"}
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return {"status": "skipped", "reason": "invalid_depth_range"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(depths.shape[1]), int(depths.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (width, height))
    if not writer.isOpened():
        return {"status": "skipped", "reason": f"unable_to_open_writer:{output_path}"}
    try:
        for depth in depths:
            normalized = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
            frame = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            writer.write(frame)
    finally:
        writer.release()
    return {
        "status": "ok",
        "path": str(output_path),
        "fps": float(max(1.0, fps)),
        "frame_count": int(depths.shape[0]),
        "normalization": "global_percentile_1_99",
        "depth_min_percentile_1": float(vmin),
        "depth_max_percentile_99": float(vmax),
    }


def _array_summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {"available": False}
    return {
        "available": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
    }


def _video_depth_geometry(
    *,
    orig_w: int,
    orig_h: int,
    processed_w: int,
    processed_h: int,
) -> dict[str, Any]:
    scale_x = processed_w / float(orig_w)
    scale_y = processed_h / float(orig_h)
    return {
        "process_res": VIDEO_DEPTH_INPUT_SIZE,
        "process_res_method": "video_depth_anything_offline_metric",
        "original_size_wh": [int(orig_w), int(orig_h)],
        "final_processed_size_wh": [int(processed_w), int(processed_h)],
        "affine_original_to_processed_2x3": [
            [float(scale_x), 0.0, 0.0],
            [0.0, float(scale_y), 0.0],
        ],
    }


def _intrinsics_original_to_processed(intrinsics: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    affine = np.asarray(geometry["affine_original_to_processed_2x3"], dtype=np.float32).reshape(2, 3)
    processed = np.asarray(intrinsics, dtype=np.float32).copy()
    processed[0, 0] = processed[0, 0] * affine[0, 0]
    processed[0, 2] = processed[0, 2] * affine[0, 0] + affine[0, 2]
    processed[1, 1] = processed[1, 1] * affine[1, 1]
    processed[1, 2] = processed[1, 2] * affine[1, 1] + affine[1, 2]
    return processed


def load_video_metric_depth_model(*, model_dir: str) -> tuple[Any, str]:
    _add_video_depth_anything_to_path()
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    checkpoint_path = Path(model_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Metric Video Depth Anything checkpoint not found: "
            f"{checkpoint_path}. Download metric_video_depth_anything_vitl.pth into "
            "third_party/Video-Depth-Anything/checkpoints/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoDepthAnything(**VIDEO_DEPTH_MODEL_CONFIGS[VIDEO_DEPTH_ENCODER], metric=True)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=True)
    model = model.to(device=device)
    model.eval()
    return model, str(device)


def _fixed_intrinsics_from_moge2(question_dir: Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    from agent.world_model.artifacts import artifact_path_by_name

    moge2_path = artifact_path_by_name(question_dir, "moge2_intrinsics.json")
    moge2_intrinsics = json.loads(moge2_path.read_text(encoding="utf-8"))
    camera_intrinsics = moge2_intrinsics.get("camera_intrinsics") or {}
    k_fixed = np.asarray(camera_intrinsics.get("K_fixed"), dtype=np.float32)
    if k_fixed.shape != (3, 3):
        raise ValueError(
            "moge2_intrinsics.json camera_intrinsics.K_fixed must have shape (3,3), "
            f"got {k_fixed.shape}. Re-run moge2_intrinsics."
        )
    if not np.isfinite(k_fixed).all():
        raise ValueError("moge2_intrinsics.json camera_intrinsics.K_fixed contains non-finite values.")
    return k_fixed, camera_intrinsics, moge2_path


def _depth_partition_route_from_object_plan(
    object_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        raise ValueError("object plan is missing special_scene depth route metadata")
    route_record = special_scene.get("depth_partition_route")
    if not isinstance(route_record, dict):
        raise ValueError("object plan is missing its depth-partition route record")
    if route_record.get("decision_id") != DEPTH_PARTITION_DECISION_ID:
        raise ValueError(
            "depth-partition route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in DEPTH_PARTITION_ROUTES:
        raise ValueError(f"unsupported depth-partition route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("depth-partition route record is missing its context")
    benchmark = str(context.get("benchmark") or "").strip().lower()
    if benchmark not in {"clevrer", "physion_pp"}:
        raise ValueError(
            "depth-partition route record has an unsupported benchmark context: "
            f"{context.get('benchmark')!r}"
        )
    scenario = str(context.get("scenario") or "").strip().lower() or None
    if benchmark == "clevrer" and scenario is not None:
        raise ValueError(
            "CLEVRER depth-partition route must not contain a scenario context"
        )
    if benchmark == "physion_pp" and scenario is None:
        raise ValueError(
            "Physion++ depth-partition route is missing its scenario context"
        )
    return route_record


def _segment_depth_alignment_route_from_object_plan(
    object_plan_payload: dict[str, Any],
    depth_partition_route: dict[str, Any],
) -> dict[str, Any] | None:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        raise ValueError("object plan is missing special_scene depth route metadata")
    route_record = special_scene.get("segment_depth_alignment_route")
    requires_alignment = (
        depth_partition_route.get("route") == SPLIT_VIDEO_VDA_ROUTE
    )
    if not isinstance(route_record, dict):
        if requires_alignment:
            raise ValueError(
                "split depth route is missing its segment-depth-alignment route record"
            )
        return None
    if not requires_alignment:
        raise ValueError(
            "segment-depth-alignment route record is not applicable to the "
            "whole-video depth route"
        )
    if route_record.get("decision_id") != SEGMENT_DEPTH_ALIGNMENT_DECISION_ID:
        raise ValueError(
            "segment-depth-alignment route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route != SEGMENT_DEPTH_ALIGNMENT_ROUTE:
        raise ValueError(
            f"unsupported segment-depth-alignment route: {route!r}"
        )
    alignment_context = route_record.get("context")
    depth_context = depth_partition_route.get("context")
    if not isinstance(alignment_context, dict):
        raise ValueError(
            "segment-depth-alignment route record is missing its context"
        )
    if not isinstance(depth_context, dict):
        raise ValueError("depth-partition route record is missing its context")
    for key in ("benchmark", "scenario"):
        if alignment_context.get(key) != depth_context.get(key):
            raise ValueError(
                f"segment-depth-alignment route {key} context does not match "
                "the depth-partition route: "
                f"{alignment_context.get(key)!r} != {depth_context.get(key)!r}"
            )
    return route_record


def _split_vda_context_for_route(
    question_dir: Path,
    route_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Build split-VDA inputs only when the resolved route explicitly selects them."""
    from agent.world_model.artifacts import artifact_path_by_name

    route = route_record.get("route")
    if route == WHOLE_VIDEO_VDA_ROUTE:
        return None
    if route != SPLIT_VIDEO_VDA_ROUTE:
        raise ValueError(f"unsupported depth-partition route: {route!r}")

    route_context = route_record.get("context")
    if not isinstance(route_context, dict):
        raise ValueError("split-VDA route record is missing its context")
    route_benchmark = str(route_context.get("benchmark") or "").strip().lower()
    route_scenario = str(route_context.get("scenario") or "").strip().lower()
    if route_benchmark != "physion_pp" or not route_scenario:
        raise ValueError(
            "split-VDA route requires Physion++ benchmark and scenario context"
        )

    tracks_path = artifact_path_by_name(question_dir, "sam3_video_tracks.json")
    try:
        if not tracks_path.exists():
            raise ValueError("sam3_video_tracks.json does not exist")
        tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
        scenario = str(tracks.get("physion_scenario") or "").strip().lower()
        tracking_route = tracks.get("tracking_family_route")
        tracking_route = tracking_route if isinstance(tracking_route, dict) else {}
        tracking_context = tracking_route.get("context")
        tracking_context = (
            tracking_context if isinstance(tracking_context, dict) else {}
        )
        tracking_benchmark = str(
            tracking_context.get("benchmark") or ""
        ).strip().lower()
        tracking_scenario = str(
            tracking_context.get("scenario") or ""
        ).strip().lower()
        if tracking_benchmark != "physion_pp":
            raise ValueError(
                "split-VDA route requires a Physion++ tracking-family record, "
                f"got benchmark {tracking_benchmark!r}"
            )
        if tracking_scenario != route_scenario:
            raise ValueError(
                "split-VDA route scenario does not match the tracking-family record: "
                f"{route_scenario!r} != {tracking_scenario!r}"
            )
        if scenario != route_scenario:
            raise ValueError(
                "split-VDA route scenario does not match tracking input: "
                f"{route_scenario!r} != {scenario!r}"
            )
        two_seg = ((tracks.get("physion_tracking") or {}).get("two_segment") or {})
        seg1, seg2 = two_seg.get("seg1"), two_seg.get("seg2")
        if not (isinstance(seg1, list) and len(seg1) == 2 and isinstance(seg2, list) and len(seg2) == 2):
            raise ValueError(
                f"{scenario} requires physion_tracking.two_segment.seg1/seg2 ranges"
            )
        object_mask = None
        sidecar = tracks.get("mask_sidecar")
        npz_path = Path(sidecar) if sidecar else tracks_path.with_suffix(".npz")
        if npz_path.exists():
            with np.load(npz_path) as mz:
                union = None
                for key in mz.files:
                    arr = mz[key]
                    if getattr(arr, "ndim", 0) == 2:
                        flag = arr.astype(bool)
                        union = flag if union is None else (union | flag)
                object_mask = union
        return {
            "scenario": scenario,
            "seg1": (int(seg1[0]), int(seg1[1])),
            "seg2": (int(seg2[0]), int(seg2[1])),
            "object_mask": object_mask,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to build Physion++ split-VDA context from {tracks_path}: {exc}"
        ) from exc


def _affine_align_seg2_to_seg1(
    depths: np.ndarray,
    seg1_range: tuple[int, int],
    seg2_range: tuple[int, int],
    object_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Physion++ two-segment: even after the seg2-subclip rerun removes the wipe-drift,
    seg2's absolute metric stays affine-offset from seg1 (monocular scale+shift ambiguity,
    per clip). seg1 and seg2 show the SAME static wall+floor, so fit seg2 = alpha*seg1 + beta
    on the common static BACKGROUND (pixels never covered by a tracked object across either
    segment, so a fixture that moves between the two trials cannot corrupt the per-pixel
    pairing) over the object-relevant depth band, then remap seg2 back onto seg1's metric.
    Fully data-driven and applied uniformly: a scene already consistent fits alpha~1/beta~0
    (a no-op). When too few background pixels survive, the point selection is progressively
    RELAXED (smaller dilation, looser depth cap) rather than abandoned."""
    if object_mask is None:
        return {"applied": False, "reason": "no_object_mask"}
    s1a, s1b = int(seg1_range[0]), int(seg1_range[1])
    s2a, s2b = int(seg2_range[0]), min(int(seg2_range[1]), depths.shape[0] - 1)
    if not (0 <= s1a <= s1b < s2a <= s2b < depths.shape[0]):
        return {"applied": False, "reason": "bad_ranges"}
    if object_mask.shape != depths.shape[1:]:
        return {"applied": False, "reason": "mask_shape_mismatch"}

    seg1_map = np.nanmedian(depths[s1a : s1b + 1], axis=0)
    seg2_map = np.nanmedian(depths[s2a : s2b + 1], axis=0)
    valid = np.isfinite(seg1_map) & np.isfinite(seg2_map) & (seg1_map > 0.6) & (seg2_map > 0.6)

    def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return mask
        try:
            import cv2

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
            return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        except Exception:
            return mask

    # A relative (never absolute-meter) depth cap keeps the object-relevant band and drops
    # the noisy far room background. Guardrail: too few background points -> RELAX the point
    # selection (less dilation, looser/no cap), do not give up.
    MIN_BACKGROUND_PX = 400
    plans = (("obj", 10), ("obj", 5), ("bg", 5), ("none", 0))
    fit_mask: np.ndarray | None = None
    used_plan: tuple[str, int] = plans[-1]
    depth_cap: float | None = None
    for cap_mode, dilate_px in plans:
        background = valid & (~_dilate(object_mask, dilate_px))
        if cap_mode == "obj":
            obj_valid = valid & object_mask
            if int(obj_valid.sum()) < 100:
                continue
            cap = 1.5 * float(np.percentile(seg1_map[obj_valid], 90))
            candidate = background & (seg1_map <= cap)
        elif cap_mode == "bg":
            if int(background.sum()) < 100:
                continue
            cap = float(np.percentile(seg1_map[background], 90))
            candidate = background & (seg1_map <= cap)
        else:
            cap = None
            candidate = background
        fit_mask, used_plan, depth_cap = candidate, (cap_mode, dilate_px), cap
        if int(candidate.sum()) >= MIN_BACKGROUND_PX:
            break
    if fit_mask is None or int(fit_mask.sum()) < 100:
        return {
            "applied": False,
            "reason": "too_few_background_pixels_after_relax",
            "n": int(fit_mask.sum()) if fit_mask is not None else 0,
        }

    x = seg1_map[fit_mask].astype(np.float64)
    y = seg2_map[fit_mask].astype(np.float64)

    def _fit(xx: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
        design = np.vstack([xx, np.ones_like(xx)]).T
        (alpha_, beta_), *_ = np.linalg.lstsq(design, yy, rcond=None)
        return float(alpha_), float(beta_)

    alpha, beta = _fit(x, y)
    resid = y - (alpha * x + beta)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) or 1e-6
    inliers = np.abs(resid - med) <= 2.5 * 1.4826 * mad
    if int(inliers.sum()) >= 100:
        alpha, beta = _fit(x[inliers], y[inliers])
    else:
        inliers = np.ones_like(x, dtype=bool)
    fitted = alpha * x[inliers] + beta
    denom = max(float(np.sum((y[inliers] - y[inliers].mean()) ** 2)), 1e-9)
    r2 = 1.0 - float(np.sum((y[inliers] - fitted) ** 2)) / denom

    if not (np.isfinite(alpha) and np.isfinite(beta)) or alpha <= 0.05:
        return {"applied": False, "reason": "degenerate_fit", "alpha": alpha, "beta": beta}

    seg2_bg_pre = float(np.median(y))
    depths[s2a : s2b + 1] = np.clip((depths[s2a : s2b + 1] - beta) / alpha, 1e-3, None)
    return {
        "applied": True,
        "policy": "affine_align_seg2_to_seg1_via_shared_static_background",
        "alpha": round(alpha, 5),
        "beta": round(beta, 5),
        "r2": round(r2, 4),
        "n_background_px": int(fit_mask.sum()),
        "n_inlier_px": int(inliers.sum()),
        "relax_cap_mode": used_plan[0],
        "relax_dilate_px": used_plan[1],
        "depth_cap_m": (round(depth_cap, 3) if depth_cap is not None else None),
        "seg1_background_median_m": round(float(np.median(x)), 4),
        "seg2_background_median_pre_m": round(seg2_bg_pre, 4),
        "seg2_background_median_post_m": round(float(np.median((y - beta) / alpha)), 4),
    }


def _run_video_depth(
    *,
    model: Any,
    video: Path,
    frame_dir: Path,
    original_width: int,
    original_height: int,
    fixed_intrinsic: np.ndarray,
    device: str,
    split_vda_context: dict[str, Any] | None = None,
    segment_depth_alignment_route: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int], dict[str, Any], dict[str, Any]]:
    if split_vda_context is None:
        if segment_depth_alignment_route is not None:
            raise ValueError(
                "whole-video depth cannot use a segment-depth-alignment route"
            )
    elif not isinstance(segment_depth_alignment_route, dict):
        raise ValueError(
            "split depth requires a segment-depth-alignment route before inference"
        )
    elif segment_depth_alignment_route.get("route") != SEGMENT_DEPTH_ALIGNMENT_ROUTE:
        raise ValueError(
            "unsupported segment-depth-alignment route before split-depth inference: "
            f"{segment_depth_alignment_route.get('route')!r}"
        )

    _add_video_depth_anything_to_path()
    from utils.dc_utils import read_video_frames

    start = time.perf_counter()
    frames_rgb, fps = read_video_frames(
        str(video),
        process_length=-1,
        target_fps=-1,
        max_res=VIDEO_DEPTH_MAX_RES,
    )
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Unexpected Video-Depth-Anything frame tensor shape: {frames_rgb.shape}")
    fixed_intrinsics = np.repeat(fixed_intrinsic[None, :, :], len(frames_rgb), axis=0)

    split_vda: dict[str, Any] = {"applied": False, "policy": "depth.vda_full_video"}
    if split_vda_context is None:
        depths, output_fps = model.infer_video_depth(
            frames_rgb,
            fps,
            input_size=VIDEO_DEPTH_INPUT_SIZE,
            device=device,
            fp32=False,
        )
        depths = depths.astype(np.float32)
        source_counts = {DEPTH_BACKEND: int(len(frames_rgb))}
    else:
        scenario = str(split_vda_context["scenario"])
        seg1_range = tuple(int(value) for value in split_vda_context["seg1"])
        seg2_range = tuple(int(value) for value in split_vda_context["seg2"])
        object_mask = split_vda_context.get("object_mask")
        s1a, s1b = seg1_range
        s2a, s2b = seg2_range
        frame_count = int(len(frames_rgb))
        if not (s1a == 0 <= s1b < s2a <= s2b == frame_count - 1):
            raise ValueError(
                f"{scenario} split-VDA ranges must cover the video edges without overlap: "
                f"seg1={seg1_range} seg2={seg2_range} frame_count={frame_count}"
            )
        if object_mask is None:
            raise ValueError(f"{scenario} split-VDA requires the SAM3 tracked-object mask union")

        segment_started = time.perf_counter()
        seg1_depths, seg1_output_fps = model.infer_video_depth(
            frames_rgb[s1a : s1b + 1],
            fps,
            input_size=VIDEO_DEPTH_INPUT_SIZE,
            device=device,
            fp32=False,
        )
        seg1_elapsed = time.perf_counter() - segment_started
        segment_started = time.perf_counter()
        seg2_depths, seg2_output_fps = model.infer_video_depth(
            frames_rgb[s2a : s2b + 1],
            fps,
            input_size=VIDEO_DEPTH_INPUT_SIZE,
            device=device,
            fp32=False,
        )
        seg2_elapsed = time.perf_counter() - segment_started
        seg1_depths = np.asarray(seg1_depths, dtype=np.float32)
        seg2_depths = np.asarray(seg2_depths, dtype=np.float32)
        expected_seg1 = s1b - s1a + 1
        expected_seg2 = s2b - s2a + 1
        if seg1_depths.shape[0] != expected_seg1 or seg2_depths.shape[0] != expected_seg2:
            raise ValueError(
                f"{scenario} split-VDA returned wrong frame counts: "
                f"seg1={seg1_depths.shape} expected={expected_seg1}, "
                f"seg2={seg2_depths.shape} expected={expected_seg2}"
            )
        if seg1_depths.shape[1:] != seg2_depths.shape[1:]:
            raise ValueError(
                f"{scenario} split-VDA segment shapes differ: "
                f"seg1={seg1_depths.shape}, seg2={seg2_depths.shape}"
            )

        depths = np.full(
            (frame_count, *seg1_depths.shape[1:]), np.nan, dtype=np.float32
        )
        depths[s1a : s1b + 1] = seg1_depths
        depths[s2a : s2b + 1] = seg2_depths
        affine_align = _affine_align_seg2_to_seg1(
            depths, seg1_range, seg2_range, object_mask
        )
        if not affine_align.get("applied"):
            raise ValueError(
                f"{scenario} split-VDA could not align seg2 to seg1: {affine_align}"
            )

        gap_start, gap_end = s1b + 1, s2a - 1
        gap_count = max(0, gap_end - gap_start + 1)
        if gap_count:
            weights = np.linspace(
                1.0 / (gap_count + 1),
                gap_count / (gap_count + 1),
                gap_count,
                dtype=np.float32,
            )
            left = depths[s1b]
            right = depths[s2a]
            depths[gap_start : gap_end + 1] = (
                (1.0 - weights[:, None, None]) * left[None, :, :]
                + weights[:, None, None] * right[None, :, :]
            )
        if not np.isfinite(depths).all():
            raise ValueError(f"{scenario} split-VDA assembled depth contains non-finite values")

        output_fps = float(seg1_output_fps or seg2_output_fps or fps)
        split_vda = {
            "applied": True,
            "policy": "independent_seg1_and_seg2_vda_then_affine_align",
            "scenario": scenario,
            "seg1_frame_range": [s1a, s1b],
            "seg1_frame_count": expected_seg1,
            "seg1_elapsed_sec": seg1_elapsed,
            "seg1_median_depth_m": float(np.nanmedian(seg1_depths)),
            "seg2_frame_range": [s2a, s2b],
            "seg2_frame_count": expected_seg2,
            "seg2_elapsed_sec": seg2_elapsed,
            "seg2_median_depth_pre_align_m": float(np.nanmedian(seg2_depths)),
            "seg2_median_depth_post_align_m": float(
                np.nanmedian(depths[s2a : s2b + 1])
            ),
            "affine_align": affine_align,
            "curtain_gap": {
                "frame_range": ([gap_start, gap_end] if gap_count else None),
                "frame_count": gap_count,
                "policy": "per_pixel_linear_interpolation_between_aligned_segment_endpoints",
            },
        }
        source_counts = {
            f"{DEPTH_BACKEND}:seg1_subclip": expected_seg1,
            f"{DEPTH_BACKEND}:seg2_subclip": expected_seg2,
            "curtain_gap_endpoint_interpolation": gap_count,
        }
    processed_h, processed_w = int(depths.shape[1]), int(depths.shape[2])
    geometry = _video_depth_geometry(
        orig_w=original_width,
        orig_h=original_height,
        processed_w=processed_w,
        processed_h=processed_h,
    )
    processed_intrinsics = np.stack(
        [_intrinsics_original_to_processed(k, geometry) for k in fixed_intrinsics],
        axis=0,
    ).astype(np.float32)
    frame_paths = _write_processed_frames(frames_rgb, frame_dir)
    frames = []
    for frame_index, frame_path in enumerate(frame_paths):
        frames.append(
            {
                "frame_index": frame_index,
                "frame_path": frame_path,
                "preprocess_geometry": geometry,
                "keys": {
                    "processed_image": "processed_images",
                    "raw_depth": "raw_depth",
                    "metric_depth": "metric_depth",
                    "confidence": None,
                    "extrinsic": None,
                    "intrinsic": "intrinsics",
                },
                "metric_depth_source": DEPTH_BACKEND,
                "summaries": {
                    "raw_depth": _array_summary(depths[frame_index]),
                    "metric_depth": _array_summary(depths[frame_index]),
                    "confidence": _array_summary(None),
                    "intrinsic": _array_summary(processed_intrinsics[frame_index]),
                    "input_intrinsic_original": _array_summary(fixed_intrinsics[frame_index]),
                    "extrinsic": _array_summary(None),
                },
            }
        )

    arrays = {
        "processed_images": frames_rgb.astype(np.uint8),
        "raw_depth": depths,
        "metric_depth": depths,
        "intrinsics": processed_intrinsics,
    }
    runtime = {
        "fps": float(output_fps),
        "elapsed_sec": time.perf_counter() - start,
        "input_size": VIDEO_DEPTH_INPUT_SIZE,
        "max_res": VIDEO_DEPTH_MAX_RES,
        "encoder": VIDEO_DEPTH_ENCODER,
        "mode": "offline",
        "metric": True,
    }
    return arrays, frames, source_counts, runtime, split_vda


def run_video_metric_depth(
    *,
    video: Path,
    object_plan: Path,
    output: Path,
    model_dir: str,
    batch_size: int,
) -> None:
    model, device = load_video_metric_depth_model(model_dir=model_dir)
    run_video_metric_depth_with_model(
        model=model,
        device=device,
        video=video,
        object_plan=object_plan,
        output=output,
        model_dir=model_dir,
        batch_size=batch_size,
        execution_mode="external_command",
    )


def run_video_metric_depth_with_model(
    *,
    model: Any,
    device: str,
    video: Path,
    object_plan: Path,
    output: Path,
    model_dir: str,
    batch_size: int,
    execution_mode: str,
) -> None:
    from agent.world_model.artifacts import question_root_from_output

    object_plan_payload = json.loads(object_plan.read_text(encoding="utf-8"))
    depth_partition_route = _depth_partition_route_from_object_plan(
        object_plan_payload
    )
    segment_depth_alignment_route = (
        _segment_depth_alignment_route_from_object_plan(
            object_plan_payload,
            depth_partition_route,
        )
    )
    video_metadata = _video_metadata(video)
    question_dir = question_root_from_output(output)
    fixed_intrinsic, fixed_intrinsics_payload, fixed_intrinsics_path = _fixed_intrinsics_from_moge2(question_dir)

    frame_dir = output.parent / "video_metric_frames"
    sidecar_path = output.with_suffix(".npz")
    split_vda_context = _split_vda_context_for_route(
        question_dir,
        depth_partition_route,
    )
    arrays, frames, metric_depth_source_counts, runtime, split_vda = _run_video_depth(
        model=model,
        video=video,
        frame_dir=frame_dir,
        original_width=int(video_metadata["width"]),
        original_height=int(video_metadata["height"]),
        fixed_intrinsic=fixed_intrinsic,
        device=device,
        split_vda_context=split_vda_context,
        segment_depth_alignment_route=segment_depth_alignment_route,
    )
    expected_split = depth_partition_route["route"] == SPLIT_VIDEO_VDA_ROUTE
    if split_vda.get("applied") is not expected_split:
        raise ValueError(
            "video metric-depth execution does not match its resolved "
            "depth-partition route: "
            f"route={depth_partition_route['route']!r} "
            f"applied={split_vda.get('applied')!r}"
        )
    np.savez_compressed(sidecar_path, **arrays)
    debug_artifacts = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1":
        depth_fps = float(runtime.get("fps") or video_metadata.get("fps") or 12.0)
        debug_artifacts["video_metric_depth_colormap"] = _write_depth_colormap_video(
            arrays["metric_depth"],
            output.parent / "debug" / "video_metric_depth_colormap.mp4",
            fps=depth_fps,
        )
        if split_vda.get("applied"):
            a, b = split_vda["seg1_frame_range"]
            debug_artifacts["video_metric_depth_seg1_colormap"] = _write_depth_colormap_video(
                arrays["metric_depth"][a : b + 1],
                output.parent / "debug" / "video_metric_depth_seg1_colormap.mp4",
                fps=depth_fps,
            )
            a, b = split_vda["seg2_frame_range"]
            debug_artifacts["video_metric_depth_seg2_colormap"] = _write_depth_colormap_video(
                arrays["metric_depth"][a : b + 1],
                output.parent / "debug" / "video_metric_depth_seg2_colormap.mp4",
                fps=depth_fps,
            )
    metadata = {
        **video_metadata,
        "frame_dir": str(frame_dir),
        "frame_paths": [item["frame_path"] for item in frames],
        "frame_count": len(frames),
    }
    payload = {
        "tool": "video_metric_depth",
        "status": "ok",
        "execution_mode": execution_mode,
        "depth_backend": DEPTH_BACKEND,
        "video": str(video),
        "object_plan": str(object_plan),
        "object_plan_summary": {
            "scene_index": object_plan_payload.get("scene_index"),
            "question_id": object_plan_payload.get("question_id"),
            "target_object_count": len(object_plan_payload.get("target_objects", [])),
        },
        "model_dir": model_dir,
        "device": device,
        "batch_size": max(1, batch_size),
        "process_res": VIDEO_DEPTH_INPUT_SIZE,
        "process_res_method": "video_depth_anything_offline_metric",
        "camera_intrinsics_artifact": str(fixed_intrinsics_path),
        "fixed_intrinsics_source": "moge2_intrinsics.camera_intrinsics.K_fixed",
        "fixed_intrinsics": fixed_intrinsics_payload.get("K_fixed"),
        "fixed_intrinsics_details": fixed_intrinsics_payload,
        "input_intrinsics_coordinate_frame": "video_original",
        "sidecar_intrinsics_coordinate_frame": "video_depth_processed_image",
        "metric_depth_source": "Metric-Video-Depth-Anything-Large offline metric depth",
        "metric_depth_source_counts": metric_depth_source_counts,
        "depth_partition_route": depth_partition_route,
        "two_segment_depth_inference": split_vda,
        "metric_depth_formula": "metric_depth = video_depth_anything.metric_depth; no camera-intrinsics scaling is applied.",
        "video_depth_runtime": runtime,
        "video_metadata": metadata,
        "tensor_sidecar": str(sidecar_path),
        "debug_artifacts": debug_artifacts,
        "tensor_manifest": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
        "frames": frames,
        "note": (
            "Metric-Video-Depth-Anything-Large runs on independent seg1/seg2 sub-clips for "
            "the resolved split depth-partition route, and on the whole video for the resolved "
            "whole-video route. "
            "The depth model does not consume or output camera intrinsics; PhysMind stores "
            "MoGe-2 median K scaled into the saved video-depth image coordinates for downstream "
            "mesh alignment and FoundationPose."
        ),
    }
    if segment_depth_alignment_route is not None:
        payload["segment_depth_alignment_route"] = (
            segment_depth_alignment_route
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=DEFAULT_VIDEO_METRIC_DEPTH_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_video_metric_depth(
        video=Path(args.video),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
