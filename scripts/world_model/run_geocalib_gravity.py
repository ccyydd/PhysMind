from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOCALIB_ROOT = PROJECT_ROOT / "third_party" / "GeoCalib"
DEFAULT_GEOCALIB_WEIGHTS = "pinhole"
GEOCALIB_SAMPLE_COUNT = 8
ROBUST_GRAVITY_MIN_OUTLIER_ANGLE_DEG = 7.5
ROBUST_GRAVITY_MAD_MULTIPLIER = 2.5


def _add_geocalib_to_path() -> None:
    if GEOCALIB_ROOT.exists() and str(GEOCALIB_ROOT) not in sys.path:
        sys.path.insert(0, str(GEOCALIB_ROOT))


def _video_metadata(video_path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        return {
            "frame_count": frame_count,
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _uniform_frame_indices(
    frame_count: int,
    num_frames: int,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    start = 0 if frame_start is None else int(frame_start)
    end = frame_count - 1 if frame_end is None else int(frame_end)
    if start < 0 or end >= frame_count or start > end:
        raise ValueError(
            f"Invalid inclusive frame range [{start}, {end}] for {frame_count} frames."
        )
    target_count = max(1, int(num_frames))
    range_count = end - start + 1
    if range_count == 1:
        return [start] * target_count
    return [
        start + round(index * (range_count - 1) / max(1, target_count - 1))
        for index in range(target_count)
    ]


def _export_frames(video_path: Path, frame_indices: list[int], output_dir: Path) -> dict[int, Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    paths: dict[int, Path] = {}
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Unable to read frame {frame_index} from {video_path}")
            path = output_dir / f"frame_{int(frame_index):05d}.jpg"
            cv2.imwrite(str(path), frame)
            paths[int(frame_index)] = path
    finally:
        capture.release()
    return paths


def load_geocalib_model(weights: str = DEFAULT_GEOCALIB_WEIGHTS) -> tuple[Any, str]:
    _add_geocalib_to_path()
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights=weights).to(device)
    model.eval()
    return model, device


def _gravity_vector(result: dict[str, Any]) -> list[float]:
    gravity = result["gravity"]
    vec = gravity.vec3d if hasattr(gravity, "vec3d") else gravity
    arr = vec.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, 3)[0]
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid GeoCalib gravity vector: {arr.tolist()}")
    return (arr / norm).tolist()


def _camera_payload(result: dict[str, Any]) -> dict[str, Any]:
    camera = result.get("camera")
    if camera is None:
        return {}
    payload: dict[str, Any] = {}
    if hasattr(camera, "K"):
        payload["K"] = np.asarray(camera.K.detach().cpu().numpy()).reshape(-1, 3, 3)[0].tolist()
    if hasattr(camera, "f"):
        payload["focal"] = np.asarray(camera.f.detach().cpu().numpy()).reshape(-1, 2)[0].tolist()
    if hasattr(camera, "c"):
        payload["principal_point"] = np.asarray(camera.c.detach().cpu().numpy()).reshape(-1, 2)[0].tolist()
    if hasattr(camera, "size"):
        payload["size"] = np.asarray(camera.size.detach().cpu().numpy()).reshape(-1, 2)[0].tolist()
    return payload


def _unit_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid {label}: {value.tolist()}")
    return value / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    value = float(np.dot(a, b))
    value = max(-1.0, min(1.0, value))
    return float(np.degrees(np.arccos(value)))


def _apply_gravity_z_constraint(
    vector: list[float],
    constraint: str | None,
) -> tuple[list[float], bool]:
    if constraint is None:
        return [float(component) for component in vector], False
    if constraint != "negative":
        raise ValueError(f"Unsupported gravity z constraint: {constraint!r}")
    value = _unit_vector(np.asarray(vector, dtype=np.float64), label="gravity vector")
    constrained = value.copy()
    constrained[2] = -abs(float(constrained[2]))
    changed = not bool(np.allclose(constrained, value, rtol=0.0, atol=1e-12))
    return constrained.tolist(), changed


def _robust_average_gravity(vectors: list[list[float]]) -> dict[str, Any]:
    if not vectors:
        raise ValueError("No gravity vectors to average.")
    reference = np.asarray(vectors[0], dtype=np.float64)
    aligned: list[np.ndarray] = []
    for vector in vectors:
        value = _unit_vector(np.asarray(vector, dtype=np.float64), label="gravity vector")
        if float(np.dot(value, reference)) < 0:
            value = -value
        aligned.append(value)

    stacked = np.stack(aligned, axis=0)
    initial_mean = _unit_vector(np.mean(stacked, axis=0), label="initial averaged gravity vector")
    angles = np.asarray([_angle_deg(vector, initial_mean) for vector in aligned], dtype=np.float64)
    median_angle = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median_angle)))
    threshold = max(
        ROBUST_GRAVITY_MIN_OUTLIER_ANGLE_DEG,
        median_angle + ROBUST_GRAVITY_MAD_MULTIPLIER * mad,
    )
    keep_mask = angles <= threshold
    if not bool(np.any(keep_mask)):
        keep_mask = np.ones_like(angles, dtype=bool)
    robust_mean = _unit_vector(np.mean(stacked[keep_mask], axis=0), label="robust averaged gravity vector")
    return {
        "gravity_direction_camera": robust_mean.tolist(),
        "initial_mean_gravity_direction_camera": initial_mean.tolist(),
        "aggregation_method": "sign_align_then_reject_angle_outliers_then_mean",
        "outlier_angle_threshold_deg": threshold,
        "angle_to_initial_mean_deg": angles.tolist(),
        "kept_indices": [int(index) for index, keep in enumerate(keep_mask.tolist()) if keep],
        "rejected_indices": [int(index) for index, keep in enumerate(keep_mask.tolist()) if not keep],
        "kept_count": int(np.count_nonzero(keep_mask)),
        "rejected_count": int(len(keep_mask) - np.count_nonzero(keep_mask)),
        "median_angle_to_initial_mean_deg": median_angle,
        "mad_angle_to_initial_mean_deg": mad,
    }


def run_geocalib_gravity_with_model(
    *,
    model: Any,
    device: str,
    video: Path,
    output: Path,
    weights: str,
    num_frames: int,
    execution_mode: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    gravity_z_constraint: str | None = None,
) -> None:
    import torch

    start = time.perf_counter()
    metadata = _video_metadata(video)
    frame_indices = _uniform_frame_indices(
        int(metadata["frame_count"]),
        num_frames,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    frame_dir = output.parent / "sampled_frames"
    frame_paths = _export_frames(video, frame_indices, frame_dir)

    frame_estimates = []
    vectors = []
    z_constrained_frame_indices = []
    for frame_index in frame_indices:
        path = frame_paths[int(frame_index)]
        image = model.load_image(path).to(device)
        with torch.no_grad():
            result = model.calibrate(image)
        gravity_raw = _gravity_vector(result)
        gravity, z_constraint_changed = _apply_gravity_z_constraint(
            gravity_raw,
            gravity_z_constraint,
        )
        vectors.append(gravity)
        if z_constraint_changed:
            z_constrained_frame_indices.append(int(frame_index))
        frame_estimates.append(
            {
                "frame_index": int(frame_index),
                "frame_path": str(path),
                "gravity_direction_camera": gravity,
                "gravity_direction_camera_raw": gravity_raw,
                "gravity_z_constraint_changed": z_constraint_changed,
                "camera": _camera_payload(result),
            }
        )

    gravity_aggregation = _robust_average_gravity(vectors)
    for index, frame_estimate in enumerate(frame_estimates):
        frame_estimate["gravity_angle_to_initial_mean_deg"] = gravity_aggregation[
            "angle_to_initial_mean_deg"
        ][index]
        frame_estimate["gravity_used_for_average"] = index in set(gravity_aggregation["kept_indices"])
    gravity_method = "geocalib_uniform_8_frame_robust_average"
    gravity_convention = "unit vector in camera coordinates, as returned by GeoCalib Gravity.vec3d"
    if gravity_z_constraint == "negative":
        gravity_method = "geocalib_uniform_8_frame_z_negative_prior_robust_average"
        gravity_convention = (
            "unit vector in camera coordinates; per-frame GeoCalib Gravity.vec3d with camera-z "
            "constrained non-positive before robust averaging"
        )
    payload = {
        "tool": "pose_correction",
        "status": "ok",
        "execution_mode": execution_mode,
        "pose_correction_stage": "gravity_direction_estimation",
        "gravity_estimation_method": gravity_method,
        "gravity_aggregation": gravity_aggregation,
        "gravity_z_constraint": {
            "applied": gravity_z_constraint is not None,
            "constraint": gravity_z_constraint,
            "stage": "per_frame_before_robust_average",
            "changed_frame_indices": z_constrained_frame_indices,
            "changed_frame_count": len(z_constrained_frame_indices),
        },
        "geocalib_weights": weights,
        "video": str(video),
        "video_metadata": metadata,
        "sampling_frame_range_inclusive": [frame_indices[0], frame_indices[-1]],
        "sampled_frame_indices": frame_indices,
        "frame_gravity_estimates": frame_estimates,
        "gravity_direction_camera": gravity_aggregation["gravity_direction_camera"],
        "gravity_direction_coordinate_frame": "opencv_camera",
        "gravity_direction_convention": gravity_convention,
        "elapsed_sec": time.perf_counter() - start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=DEFAULT_GEOCALIB_WEIGHTS)
    parser.add_argument("--num-frames", type=int, default=GEOCALIB_SAMPLE_COUNT)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--gravity-z-constraint", choices=["negative"])
    args = parser.parse_args()
    model, device = load_geocalib_model(weights=args.weights)
    run_geocalib_gravity_with_model(
        model=model,
        device=device,
        video=Path(args.video),
        output=Path(args.output),
        weights=args.weights,
        num_frames=args.num_frames,
        execution_mode="standalone",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        gravity_z_constraint=args.gravity_z_constraint,
    )


if __name__ == "__main__":
    main()
