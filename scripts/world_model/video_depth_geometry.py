from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def warp_mask_to_processed(
    mask: np.ndarray,
    *,
    affine_2x3: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = shape
    warped = cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(affine_2x3, dtype=np.float32).reshape(2, 3),
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def video_depth_preprocess_geometry_for_frame(
    video_metric_depth: dict[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    frames = video_metric_depth.get("frames", [])
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(
            f"Video-depth frame_index out of range for preprocess geometry: {frame_index}"
        )
    geometry = frames[frame_index].get("preprocess_geometry")
    if geometry is None:
        raise KeyError(
            "Video-depth preprocess geometry is required for FoundationPose mask warping. "
            "Re-run video_metric_depth with the updated artifact schema."
        )
    return geometry
