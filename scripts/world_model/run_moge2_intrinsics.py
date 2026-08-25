from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "third_party" / "MoGe"
DEFAULT_MOGE2_MODEL = "Ruicheng/moge-2-vitl-normal"
MOGE2_SAMPLE_COUNT = 8


def _add_moge_to_path() -> None:
    if MOGE_ROOT.exists() and str(MOGE_ROOT) not in sys.path:
        sys.path.insert(0, str(MOGE_ROOT))


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


def _uniform_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    target_count = min(frame_count, max(1, int(num_frames)))
    return [
        round(index * (frame_count - 1) / max(1, target_count - 1))
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
            path = output_dir / f"frame_{int(frame_index):05d}.png"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise ValueError(f"Unable to write sampled frame: {path}")
            paths[int(frame_index)] = path
    finally:
        capture.release()
    return paths


def _normalized_intrinsics_to_pixel(k: np.ndarray, *, width: int, height: int) -> np.ndarray:
    normalized = np.asarray(k, dtype=np.float32)
    if normalized.shape != (3, 3) or not np.isfinite(normalized).all():
        raise ValueError(f"Unexpected MoGe-2 intrinsics: shape={normalized.shape}")
    pixel = normalized.copy()
    pixel[0, :] *= float(width)
    pixel[1, :] *= float(height)
    return pixel


def _fixed_intrinsics(frame_estimates: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    stack = np.asarray([item["intrinsics_pixel"] for item in frame_estimates], dtype=np.float32)
    parameters = np.stack(
        [stack[:, 0, 0], stack[:, 1, 1], stack[:, 0, 2], stack[:, 1, 2]],
        axis=1,
    )
    median = np.median(parameters, axis=0)
    k_fixed = np.asarray(
        [[median[0], 0.0, median[2]], [0.0, median[1], median[3]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return k_fixed, {
        "parameter_order": ["fx", "fy", "cx", "cy"],
        "per_frame_values": parameters.tolist(),
        "median": median.tolist(),
        "stddev": np.std(parameters, axis=0).tolist(),
        "min": np.min(parameters, axis=0).tolist(),
        "max": np.max(parameters, axis=0).tolist(),
    }


def load_moge2_model(model_name: str = DEFAULT_MOGE2_MODEL) -> tuple[Any, str]:
    _add_moge_to_path()
    import torch
    from moge.model.v2 import MoGeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, device


def run_moge2_intrinsics_with_model(
    *,
    model: Any,
    device: str,
    video: Path,
    output: Path,
    model_name: str,
    num_frames: int,
    execution_mode: str,
) -> None:
    import cv2
    import torch

    start = time.perf_counter()
    metadata = _video_metadata(video)
    frame_indices = _uniform_frame_indices(int(metadata["frame_count"]), num_frames)
    frame_dir = output.parent / "sampled_frames"
    frame_paths = _export_frames(video, frame_indices, frame_dir)

    frame_estimates: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        frame_path = frame_paths[frame_index]
        image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read sampled frame: {frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        with torch.no_grad():
            result = model.infer(image)
        normalized = result["intrinsics"].detach().cpu().numpy().astype(np.float32)
        pixel = _normalized_intrinsics_to_pixel(
            normalized,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
        )
        frame_estimates.append(
            {
                "frame_index": int(frame_index),
                "frame_path": str(frame_path),
                "intrinsics_normalized": normalized.tolist(),
                "intrinsics_pixel": pixel.tolist(),
            }
        )

    k_fixed, aggregation = _fixed_intrinsics(frame_estimates)
    payload = {
        "tool": "moge2_intrinsics",
        "status": "ok",
        "execution_mode": execution_mode,
        "model_name": model_name,
        "video": str(video),
        "video_metadata": metadata,
        "sample_count": len(frame_estimates),
        "sampled_frame_indices": frame_indices,
        "sampled_frame_format": "png",
        "frame_intrinsics": frame_estimates,
        "camera_intrinsics": {
            "available": True,
            "source": "moge2_uniform_frames_median",
            "coordinate_frame": "video_original",
            "normalized_intrinsics_definition": "MoGe-2 normalized camera intrinsics",
            "pixel_conversion": "K[0,:] *= image_width; K[1,:] *= image_height",
            "aggregation_method": "componentwise_median_fx_fy_cx_cy",
            "K_fixed": k_fixed.tolist(),
            "statistics": aggregation,
        },
        "elapsed_sec": time.perf_counter() - start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MOGE2_MODEL)
    parser.add_argument("--num-frames", type=int, default=MOGE2_SAMPLE_COUNT)
    args = parser.parse_args()
    model, device = load_moge2_model(model_name=args.model_name)
    run_moge2_intrinsics_with_model(
        model=model,
        device=device,
        video=Path(args.video),
        output=Path(args.output),
        model_name=args.model_name,
        num_frames=args.num_frames,
        execution_mode="standalone",
    )


if __name__ == "__main__":
    main()
