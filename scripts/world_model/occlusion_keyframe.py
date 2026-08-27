"""Select Physion++ SAM3D keyframes using masks and metric depth.

Touching masks are ordered by contact-band depth on each frame. Eligible frames keep
the object inside the image and exclude frames where it is occluded. Selection uses the
largest eligible mask, or the largest available mask when none is eligible.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

CONTACT_DILATE_PX = 5          # band width (px) around a contact used to sample depth
MIN_BAND_PIXELS = 10           # min valid-depth pixels per side to trust the comparison
DEPTH_ABS_MARGIN_M = 0.03      # absolute depth margin (metres)
DEPTH_REL_MARGIN = 0.02        # relative depth margin (fraction of contact depth)

def _split_mask_key(mask_key: str) -> tuple[str, str] | None:
    """Return (prefix, suffix) so that any frame f maps to f'{prefix}__frame_{f:05d}{suffix}'."""
    if "__frame_" not in mask_key or "__sam_" not in mask_key:
        return None
    prefix, rest = mask_key.split("__frame_", 1)
    suffix = "__sam_" + rest.split("__sam_", 1)[1]
    return prefix, suffix


def _touch(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    dilated = cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return bool(np.logical_and(dilated, mask_b).any())


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _contact_band_depths(
    mask_a: np.ndarray, mask_b: np.ndarray, depth: np.ndarray, valid: np.ndarray
) -> tuple[float | None, float | None]:
    kernel = np.ones((CONTACT_DILATE_PX, CONTACT_DILATE_PX), np.uint8)
    a_near = mask_a & cv2.dilate(mask_b.astype(np.uint8), kernel, 1).astype(bool) & valid
    b_near = mask_b & cv2.dilate(mask_a.astype(np.uint8), kernel, 1).astype(bool) & valid
    if int(a_near.sum()) < MIN_BAND_PIXELS or int(b_near.sum()) < MIN_BAND_PIXELS:
        return None, None
    return float(np.median(depth[a_near])), float(np.median(depth[b_near]))


def select_occlusion_aware_keyframes(
    *,
    object_keyframes: list[dict[str, Any]],
    masks: Any,
    metric_depth: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Re-pick one SAM3D keyframe per object. Returns object_id -> selection dict."""
    mask_files = set(getattr(masks, "files", []))
    num_frames = int(metric_depth.shape[0])

    # object_id -> (prefix, suffix) template for its per-frame mask keys
    templates: dict[str, tuple[str, str]] = {}
    for keyframe in object_keyframes:
        object_id = str(keyframe.get("object_id"))
        parts = _split_mask_key(str(keyframe.get("mask_key") or ""))
        if parts is not None:
            templates[object_id] = parts

    def key_for(object_id: str, frame: int) -> str | None:
        parts = templates.get(object_id)
        if parts is None:
            return None
        candidate = f"{parts[0]}__frame_{frame:05d}{parts[1]}"
        return candidate if candidate in mask_files else None

    def load_mask(object_id: str, frame: int) -> np.ndarray | None:
        key = key_for(object_id, frame)
        if key is None:
            return None
        mask = np.squeeze(np.asarray(masks[key])).astype(bool)
        return mask if mask.ndim == 2 else None

    frames_of: dict[str, list[int]] = {
        object_id: [f for f in range(num_frames) if key_for(object_id, f) is not None]
        for object_id in templates
    }

    result: dict[str, dict[str, Any]] = {}
    for keyframe in object_keyframes:
        object_id = str(keyframe.get("object_id"))
        candidates = frames_of.get(object_id, [])
        if not candidates:
            continue
        areas: dict[int, int] = {}
        eligible: list[int] = []
        roles: dict[int, str] = {}
        for frame in candidates:
            mask_a = load_mask(object_id, frame)
            if mask_a is None or not mask_a.any():
                continue
            areas[frame] = int(mask_a.sum())
            if _touches_boundary(mask_a):
                roles[frame] = "boundary"
                continue
            depth = np.asarray(metric_depth[frame], dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            is_occludee = False
            for other_id in templates:
                if other_id == object_id:
                    continue
                mask_b = load_mask(other_id, frame)
                if mask_b is None or not mask_b.any() or not _touch(mask_a, mask_b):
                    continue
                depth_a, depth_b = _contact_band_depths(mask_a, mask_b, depth, valid)
                if depth_a is None:  # depth-undecidable -> treat A as occludee
                    is_occludee = True
                    break
                margin = max(DEPTH_ABS_MARGIN_M, DEPTH_REL_MARGIN * min(depth_a, depth_b))
                if not (depth_a < depth_b - margin):  # A not clearly in front -> occludee
                    is_occludee = True
                    break
            if is_occludee:
                roles[frame] = "occludee"
            else:
                roles[frame] = "occluder_or_clean"
                eligible.append(frame)

        pool = eligible if eligible else candidates
        best_frame = max(pool, key=lambda f: (areas.get(f, 0), f))
        result[object_id] = {
            "frame_index": int(best_frame),
            "mask_key": key_for(object_id, best_frame),
            "area": int(areas.get(best_frame, 0)),
            "selection_rule": (
                "largest_area_occlusion_aware_clean_frame"
                if eligible
                else "largest_area_all_frames_fallback"
            ),
            "fallback_used": not bool(eligible),
            "eligible_frame_count": len(eligible),
            "candidate_frame_count": len(candidates),
            "selected_role": roles.get(best_frame, "unknown"),
            "previous_frame_index": keyframe.get("frame_index"),
        }
    return result
