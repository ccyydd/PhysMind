from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONPOSE_ROOT = PROJECT_ROOT / "third_party" / "FoundationPose"


def add_foundationpose_to_path() -> None:
    if not FOUNDATIONPOSE_ROOT.exists():
        raise FileNotFoundError(f"FoundationPose submodule not found: {FOUNDATIONPOSE_ROOT}")
    root = str(FOUNDATIONPOSE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def float32_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def foundationpose_adaptive_depth_preprocess(
    *,
    depth: np.ndarray,
    ob_mask: np.ndarray,
    radius: int = 2,
    min_valid_pixels: int = 4,
    device: str = "cuda",
    initial_depth_diff_thres: float = 0.001,
    max_depth_diff_thres: float = 0.1,
    target_max_filtered_ratio: float = 0.10,
    ratio_thres: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    add_foundationpose_to_path()
    from Utils import erode_depth

    required_thresholds, threshold_metadata = _required_erode_depth_thresholds(
        depth=depth,
        ob_mask=ob_mask,
        radius=radius,
        ratio_thres=ratio_thres,
    )
    if required_thresholds.size:
        percentile = 100.0 * (1.0 - float(target_max_filtered_ratio))
        selected_threshold = float(np.percentile(required_thresholds, percentile))
        selected_threshold = max(float(initial_depth_diff_thres), selected_threshold)
        selected_threshold = min(float(max_depth_diff_thres), selected_threshold)
    else:
        selected_threshold = float(max_depth_diff_thres)

    selected_depth = float32_contiguous(
        erode_depth(
            depth,
            radius=radius,
            depth_diff_thres=selected_threshold,
            ratio_thres=ratio_thres,
            device=device,
        )
    )
    original_valid = int(((np.asarray(depth) >= 0.001) & (np.asarray(ob_mask) > 0)).sum())
    kept_valid = int(((selected_depth >= 0.001) & (np.asarray(ob_mask) > 0)).sum())
    filtered_ratio = float(1.0 - kept_valid / max(original_valid, 1))

    metadata = {
        "method": "adaptive_erode_depth",
        "threshold_selection": "per_pixel_required_threshold_percentile",
        "radius": radius,
        "min_valid_pixels": min_valid_pixels,
        "initial_depth_diff_thres": float(initial_depth_diff_thres),
        "max_depth_diff_thres": float(max_depth_diff_thres),
        "selected_depth_diff_thres": float(selected_threshold),
        "ratio_thres": float(ratio_thres),
        "target_max_filtered_ratio": float(target_max_filtered_ratio),
        "original_valid_depth_pixel_count": original_valid,
        "kept_valid_depth_pixel_count": kept_valid,
        "filtered_depth_pixel_ratio": filtered_ratio,
        "required_threshold_stats": threshold_metadata,
        "relaxed": bool(float(selected_threshold) > float(initial_depth_diff_thres)),
    }
    return float32_contiguous(selected_depth), metadata


def _required_erode_depth_thresholds(
    *,
    depth: np.ndarray,
    ob_mask: np.ndarray,
    radius: int,
    ratio_thres: float,
    zfar: float = 100.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_np = np.asarray(depth, dtype=np.float32)
    mask_np = np.asarray(ob_mask) > 0
    valid_mask = (depth_np >= 0.001) & (depth_np < zfar) & mask_np
    height, width = depth_np.shape[:2]
    required: list[float] = []
    impossible_count = 0
    valid_count = 0

    for y, x in np.argwhere(valid_mask):
        center_depth = float(depth_np[y, x])
        diffs: list[float] = []
        invalid_count = 0
        total = 0
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                total += 1
                neighbor_depth = float(depth_np[yy, xx])
                if neighbor_depth < 0.001 or neighbor_depth >= zfar:
                    invalid_count += 1
                else:
                    diffs.append(abs(neighbor_depth - center_depth))

        valid_count += 1
        allowed_bad = int(np.floor(float(ratio_thres) * float(total)))
        allowed_diff_bad = allowed_bad - invalid_count
        if allowed_diff_bad < 0:
            impossible_count += 1
            continue
        if not diffs:
            required.append(0.0)
            continue
        diffs_np = np.asarray(diffs, dtype=np.float32)
        sorted_diffs = np.sort(diffs_np)
        keep_count = len(sorted_diffs) - int(allowed_diff_bad)
        if keep_count <= 0:
            required.append(0.0)
        else:
            required.append(float(sorted_diffs[min(keep_count - 1, len(sorted_diffs) - 1)]))

    required_np = np.asarray(required, dtype=np.float32)
    metadata: dict[str, Any] = {
        "mask_valid_depth_pixel_count": valid_count,
        "impossible_to_keep_pixel_count": impossible_count,
        "required_threshold_count": int(required_np.size),
    }
    if required_np.size:
        metadata.update(
            {
                "min": float(np.min(required_np)),
                "median": float(np.median(required_np)),
                "p90": float(np.percentile(required_np, 90.0)),
                "p95": float(np.percentile(required_np, 95.0)),
                "max": float(np.max(required_np)),
            }
        )
    return required_np, metadata


def register_keyframe_seeded_single_candidate(
    *,
    segment_estimator: Any,
    K: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    ob_mask: np.ndarray,
    seed_raw_pose: np.ndarray,
    iteration: int,
    seed_rotation_source: str = "mesh_conditioning.foundationpose_initial_pose_4x4",
) -> tuple[np.ndarray, dict[str, Any]]:
    add_foundationpose_to_path()
    import torch
    from Utils import bilateral_filter_depth, depth2xyzmap, set_seed

    set_seed(0)
    if segment_estimator.glctx is None:
        from estimater import dr

        segment_estimator.glctx = dr.RasterizeCudaContext()

    depth_processed, depth_metadata = foundationpose_adaptive_depth_preprocess(
        depth=depth,
        ob_mask=ob_mask,
        radius=2,
        min_valid_pixels=4,
        device="cuda",
    )
    depth_processed = bilateral_filter_depth(depth_processed, radius=2, device="cuda")
    valid = (depth_processed >= 0.001) & (ob_mask > 0)

    segment_estimator.H, segment_estimator.W = depth_processed.shape[:2]
    segment_estimator.K = K
    segment_estimator.ob_mask = ob_mask

    guessed_center = float32_contiguous(segment_estimator.guess_translation(depth=depth_processed, mask=ob_mask, K=K))
    seed_raw_pose = float32_contiguous(seed_raw_pose).reshape(4, 4)
    seed_centered_pose = seed_raw_pose.copy()

    tf_to_centered_mesh_torch = segment_estimator.get_tf_to_centered_mesh()
    tf_to_centered_mesh = tf_to_centered_mesh_torch.detach().cpu().numpy()
    seed_pose_4x4 = float32_contiguous(seed_centered_pose @ tf_to_centered_mesh)
    metadata: dict[str, Any] = {
        "registration_strategy": "keyframe_seeded_single_candidate_register",
        "seed_rotation_source": seed_rotation_source,
        "seed_translation_source": f"{seed_rotation_source}.translation",
        "candidate_count": 1,
        "seed_pose_4x4": seed_pose_4x4.tolist(),
        "seed_centered_pose_4x4": seed_centered_pose.tolist(),
        "mesh_conditioning_seed_pose_4x4": seed_raw_pose.tolist(),
        "seed_center_translation_camera": seed_centered_pose[:3, 3].reshape(3).tolist(),
        "guess_translation_camera": guessed_center.reshape(3).tolist(),
        "valid_depth_pixel_count": int(valid.sum()),
        "depth_preprocess": "foundationpose_adaptive_erode_depth_radius_2_then_bilateral_filter_radius_2",
        "adaptive_depth_preprocess": depth_metadata,
    }

    if int(valid.sum()) < 4:
        pose_last = torch.as_tensor(seed_centered_pose, device="cuda", dtype=torch.float)
        segment_estimator.pose_last = pose_last
        segment_estimator.best_id = torch.as_tensor(0, device="cuda")
        segment_estimator.poses = pose_last.reshape(1, 4, 4)
        segment_estimator.scores = torch.as_tensor([float("nan")], device="cuda", dtype=torch.float)
        metadata.update(
            {
                "refiner_applied": False,
                "scorer_applied": False,
                "register_score": None,
                "reason": "valid_depth_pixel_count_lt_4",
            }
        )
        return seed_pose_4x4, metadata

    xyz_map = depth2xyzmap(depth_processed, K)
    poses, _ = segment_estimator.refiner.predict(
        mesh=segment_estimator.mesh,
        mesh_tensors=segment_estimator.mesh_tensors,
        rgb=rgb,
        depth=depth_processed,
        K=K,
        ob_in_cams=seed_centered_pose.reshape(1, 4, 4),
        normal_map=None,
        xyz_map=xyz_map,
        glctx=segment_estimator.glctx,
        mesh_diameter=segment_estimator.diameter,
        iteration=iteration,
        get_vis=segment_estimator.debug >= 2,
    )
    scores, _ = segment_estimator.scorer.predict(
        mesh=segment_estimator.mesh,
        rgb=rgb,
        depth=depth_processed,
        K=K,
        ob_in_cams=poses.data.cpu().numpy(),
        normal_map=None,
        mesh_tensors=segment_estimator.mesh_tensors,
        glctx=segment_estimator.glctx,
        mesh_diameter=segment_estimator.diameter,
        get_vis=segment_estimator.debug >= 2,
    )

    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    best_id = int(np.argsort(scores_np)[::-1][0])
    best_centered_pose = poses[best_id]
    best_pose = best_centered_pose @ tf_to_centered_mesh_torch
    segment_estimator.pose_last = best_centered_pose
    segment_estimator.best_id = torch.as_tensor(best_id, device="cuda")
    segment_estimator.poses = poses
    segment_estimator.scores = scores
    metadata.update(
        {
            "refiner_applied": True,
            "scorer_applied": True,
            "register_score": float(scores_np[best_id]),
            "best_candidate_index": best_id,
            "refined_pose_4x4": best_pose.detach().cpu().numpy().tolist(),
        }
    )
    return float32_contiguous(best_pose.detach().cpu().numpy()), metadata
