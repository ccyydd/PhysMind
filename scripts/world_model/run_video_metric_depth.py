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
