from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.world_model.artifacts import artifact_path_by_name, question_root_from_output
from scripts.world_model.foundationpose_register import register_keyframe_seeded_single_candidate
from scripts.world_model.mesh_projection import mask_occluded_mesh_mask, render_mesh_depth
from scripts.world_model.video_depth_geometry import (
    video_depth_preprocess_geometry_for_frame,
    warp_mask_to_processed,
)


FOUNDATIONPOSE_ROOT = PROJECT_ROOT / "third_party" / "FoundationPose"

STATIC_FIXTURE_ANCHOR_CANDIDATE_COUNT = 5
STATIC_FIXTURE_MAX_EVAL_FRAMES = 16
STATIC_FIXTURE_LOW_CONFIDENCE_MEAN_IOU = 0.5




def _track_mask_keys_by_frame(mask_archive: Any, track_id: str) -> dict[int, str]:
    prefix = f"{track_id}__frame_"
    mapping: dict[int, str] = {}
    for key in mask_archive.files:
        if not key.startswith(prefix):
            continue
        frame_token = key[len(prefix) :].split("__")[0]
        try:
            mapping[int(frame_token)] = key
        except ValueError:
            continue
    return mapping


def _all_track_mask_union_at_frame(mask_archive: Any, frame_index: int) -> np.ndarray | None:
    """Union of EVERY track's mask at one frame (original mask resolution).

    Depth-free occluder for ``mask_occluded_mesh_mask``: whichever track owns a pixel in
    the observed image is the front-most object there, so a rendered pixel landing on any
    other track's mask is treated as occluded. Mask keys are
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


def _evenly_spread(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return list(values)
    indices = np.unique(np.linspace(0, len(values) - 1, count).round().astype(int))
    return [values[index] for index in indices]


def _add_foundationpose_to_path() -> None:
    if not FOUNDATIONPOSE_ROOT.exists():
        raise FileNotFoundError(f"FoundationPose submodule not found: {FOUNDATIONPOSE_ROOT}")
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intrinsic_for_frame(intrinsic: np.ndarray, frame_slot: int) -> np.ndarray:
    if intrinsic.ndim == 3:
        return intrinsic[frame_slot]
    if intrinsic.ndim == 2:
        return intrinsic
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def _intrinsic_frame_count(intrinsic: np.ndarray, fallback_count: int) -> int:
    if intrinsic.ndim == 3:
        return intrinsic.shape[0]
    if intrinsic.ndim == 2:
        return fallback_count
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def _float32_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)




def _seed_pose_from_mesh_record(mesh_record: dict[str, Any], *, object_id: str) -> np.ndarray:
    payload = mesh_record.get("foundationpose_initial_pose_4x4")
    if payload is None:
        raise ValueError(f"{object_id} missing foundationpose_initial_pose_4x4")
    pose = np.asarray(payload, dtype=np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{object_id} invalid foundationpose_initial_pose_4x4 shape/value")
    return _float32_contiguous(pose)


def _depth_for_frame(depths: np.ndarray, frame_index: int) -> np.ndarray:
    if depths.ndim == 4 and depths.shape[-1] == 1:
        return depths[frame_index, ..., 0]
    if depths.ndim == 3:
        return depths[frame_index]
    raise ValueError(f"Unsupported video metric_depth shape: {depths.shape}")


def _processed_rgb_for_frame(processed_images: np.ndarray, frame_index: int) -> np.ndarray:
    if processed_images.ndim != 4 or processed_images.shape[-1] != 3:
        raise ValueError(f"Unsupported video-depth processed_images shape: {processed_images.shape}")
    return np.ascontiguousarray(processed_images[frame_index])


def _mesh_by_object(*, mesh_conditioning: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for item in mesh_conditioning.get("objects", []):
        object_id = str(item.get("object_id"))
        if not object_id or item.get("status") != "ok":
            continue
        records[object_id] = {
            "object_id": object_id,
            "registration_frame_index": item.get("frame_index"),
            "registration_mask_key": item.get("mask_key"),
            "projected_mesh_path": item.get("projected_mesh_path"),
            "mesh_path": item.get("foundationpose_mesh_path") or item.get("projected_mesh_path"),
            "foundationpose_mesh_path": item.get("foundationpose_mesh_path"),
            "foundationpose_initial_pose_4x4": item.get("foundationpose_initial_pose_4x4"),
            "foundationpose_local_origin_camera_xyz": item.get("foundationpose_local_origin_camera_xyz"),
            "mesh_conditioning": {
                "geometry_type": item.get("geometry_type"),
                "conditioning_action": item.get("conditioning_action"),
                "fit": item.get("fit"),
            },
            "coordinate_frame": item.get("foundationpose_mesh_coordinate_frame")
            or item.get("projected_mesh_coordinate_frame"),
            "projected_mesh_coordinate_frame": item.get("projected_mesh_coordinate_frame"),
            "status": "ok",
        }
    return records


def _sam3_mask_record_by_key(*, question_dir: Path) -> dict[str, dict[str, Any]]:
    track_labels = _load_json(artifact_path_by_name(question_dir, "sam3_video_track_labels.json"))
    return {
        str(record.get("mask_key")): record
        for record in track_labels.get("object_keyframes", [])
        if record.get("mask_key") is not None
    }


def _pose_frame_by_object(pose_frames: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("object_id")): item
        for item in pose_frames.get("object_pose_frames", [])
        if item.get("object_id") is not None
    }


def _initial_mask_by_object(
    *,
    pose_sam3_masks: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records_by_mask_key = {
        str(record.get("mask_key")): record
        for record in pose_sam3_masks.get("masks", [])
        if record.get("mask_key") is not None
    }
    selected = {}
    for selection in pose_sam3_masks.get("selected_masks", []):
        object_id = str(selection.get("object_id"))
        mask_key = selection.get("selected_mask_key")
        if not object_id or not mask_key:
            continue
        record = records_by_mask_key.get(str(mask_key))
        if record is None:
            continue
        selected[object_id] = {
            **record,
            "selection_status": selection.get("status"),
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "selected_candidate_overlay_path": selection.get("selected_candidate_overlay_path"),
            "selection_reason": selection.get("reason"),
        }
    return selected


def _pose_record(
    *,
    frame_index: int,
    raw_pose: np.ndarray,
    geometry_type: str,
    registration_mask_key: str,
    tracking_direction: str,
    preprocess_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corrected = _float32_contiguous(np.asarray(raw_pose)).reshape(4, 4).astype(np.float64)
    geometry = str(geometry_type or "irregular").lower()
    return {
        "frame_index": frame_index,
        "initial_mask_key": registration_mask_key,
        "registration_mask_key": registration_mask_key,
        "tracking_direction": tracking_direction,
        "video_depth_key": "metric_depth",
        "video_intrinsic_key": "video_metric_depth.tensor_sidecar.intrinsics",
        "video_depth_preprocess_geometry": preprocess_geometry,
        "pose_4x4": raw_pose.reshape(4, 4).tolist(),
        "raw_pose_4x4": raw_pose.reshape(4, 4).tolist(),
        "symmetry_aware_pose_4x4": _float32_contiguous(corrected).reshape(4, 4).tolist(),
        "pose_postprocess": {
            "geometry_type": geometry,
            "method": "none",
            "raw_pose_4x4_preserved": True,
        },
    }


def load_foundationpose_context() -> dict[str, Any]:
    _add_foundationpose_to_path()
    from estimater import PoseRefinePredictor, ScorePredictor, dr, set_logging_format, set_seed

    set_logging_format()
    set_seed(0)
    return {
        "glctx": dr.RasterizeCudaContext(),
        "scorer": ScorePredictor(),
        "refiner": PoseRefinePredictor(),
    }


def run_foundationpose(
    *,
    video: Path,
    object_plan: Path,
    output: Path,
    est_refine_iter: int,
    track_refine_iter: int,
    debug: int,
    debug_artifacts: Optional[int] = None,
    context: dict[str, Any] | None = None,
    execution_mode: str = "external_command",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _add_foundationpose_to_path()
    from estimater import FoundationPose
    import trimesh

    if context is None:
        context = load_foundationpose_context()

    question_dir = question_root_from_output(output)
    pose_frames = _load_json(artifact_path_by_name(question_dir, "pose_frames.json"))
    pose_sam3_masks = _load_json(artifact_path_by_name(question_dir, "pose_sam3_masks.json"))
    mesh_conditioning = _load_json(artifact_path_by_name(question_dir, "mesh_conditioning.json"))
    video_metric_depth = _load_json(artifact_path_by_name(question_dir, "video_metric_depth.json"))
    track_labels = _load_json(artifact_path_by_name(question_dir, "sam3_video_track_labels.json"))
    tracks_payload = _load_json(artifact_path_by_name(question_dir, "sam3_video_tracks.json"))
    sam3_track_masks = np.load(track_labels["mask_sidecar"])
    masks = np.load(pose_sam3_masks["mask_sidecar"])
    video_depth_sidecar = np.load(video_metric_depth["tensor_sidecar"])
    processed_images = video_depth_sidecar["processed_images"]
    metric_depth = video_depth_sidecar["metric_depth"]
    intrinsics = video_depth_sidecar["intrinsics"]
    if intrinsics.ndim not in (2, 3) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Video-depth sidecar intrinsics must have shape (3,3) or (N,3,3), got {intrinsics.shape}")

    debug_artifacts_flag = debug if debug_artifacts is None else debug_artifacts
    debug_dir = output.parent / "debug"
    glctx = context["glctx"]
    scorer = context["scorer"]
    refiner = context["refiner"]

    object_results = []
    with tempfile.TemporaryDirectory(prefix="physmind_foundationpose_debug_") as foundationpose_debug_tmp:
        foundationpose_debug_dir = Path(foundationpose_debug_tmp)
        mesh_records = _mesh_by_object(mesh_conditioning=mesh_conditioning)
        sam3_mask_records = _sam3_mask_record_by_key(question_dir=question_dir)
        pose_frame_records = _pose_frame_by_object(pose_frames)
        initial_mask_records = _initial_mask_by_object(
            pose_sam3_masks=pose_sam3_masks,
        )
        object_plan_payload = _load_json(object_plan)
        scenario = object_plan_scenario(object_plan_payload)
        agent_geometry_route = foundationpose_agent_geometry_route(
            object_plan_payload,
            require_for_main_scenario=True,
        )
        candidate_agent_ids = sphere_agent_object_ids(
            scenario=scenario,
            tracks_payload=tracks_payload,
            target_objects=object_plan_payload.get("target_objects", []),
        )
        agent_geometry_policy = route_agent_geometry_policy_payload(
            route_record=agent_geometry_route,
            scenario=scenario,
            agent_object_ids=candidate_agent_ids,
        )
        sphere_agent_ids = set(agent_geometry_policy["agent_object_ids"])
        geometry_by_object = {
            str(target.get("object_id")): str(target.get("geometry_type") or "irregular").lower()
            for target in object_plan_payload.get("target_objects", [])
            if target.get("object_id") is not None
        }
        for target in object_plan_payload.get("target_objects", []):
            object_id = str(target.get("object_id"))
            geometry_type = geometry_by_object.get(object_id, "irregular")
            source_track_id = str(target.get("source_track_id") or "")
            if object_id in sphere_agent_ids:
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "skipped_sphere_agent",
                        "source_track_id": source_track_id,
                        "reason": (
                            "Physion++ sphere-agent policy reconstructs this agent from "
                            "SAM3 masks and scenario geometry without FoundationPose."
                        ),
                    }
                )
                continue
            mesh_record = mesh_records.get(object_id)
            if not mesh_record or not mesh_record.get("mesh_path"):
                object_results.append({"object_id": object_id, "status": "missing_mesh"})
                continue
            pose_frame_record = pose_frame_records.get(object_id)
            if not pose_frame_record:
                object_results.append({"object_id": object_id, "status": "missing_pose_frames"})
                continue
            if pose_frame_record.get("status") != "ok":
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "skipped_no_full_visible_frame",
                        "pose_frame_status": pose_frame_record.get("status"),
                        "reason": "FoundationPose requires a valid full-visible tracking interval.",
                    }
                )
                continue
            if (
                pose_frame_record.get("first_full_visible_frame_index") is None
                or pose_frame_record.get("last_full_visible_frame_index") is None
            ):
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "skipped_no_full_visible_frame",
                        "pose_frame_status": pose_frame_record.get("status"),
                        "reason": "FoundationPose tracking interval is missing first/last full-visible frame.",
                    }
                )
                continue
            initial_mask_record = initial_mask_records.get(object_id)
            if not initial_mask_record:
                object_results.append({"object_id": object_id, "status": "missing_initial_mask"})
                continue
            start_frame = int(pose_frame_record["first_full_visible_frame_index"])
            end_frame = int(pose_frame_record["last_full_visible_frame_index"])
            if start_frame >= processed_images.shape[0]:
                object_results.append({"object_id": object_id, "status": "start_frame_out_of_range"})
                continue
            end_frame = min(
                end_frame,
                processed_images.shape[0] - 1,
                metric_depth.shape[0] - 1,
                _intrinsic_frame_count(intrinsics, processed_images.shape[0]) - 1,
            )
            if end_frame < start_frame:
                object_results.append({"object_id": object_id, "status": "empty_tracking_interval"})
                continue

            mesh_path = mesh_record["mesh_path"]
            mesh_fingerprint = _file_sha256(Path(mesh_path))
            mesh = trimesh.load(mesh_path, force="mesh")
            if hasattr(mesh, "geometry"):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            mesh.vertices = _float32_contiguous(np.asarray(mesh.vertices))
            mesh.vertex_normals = _float32_contiguous(np.asarray(mesh.vertex_normals))
            estimator = None

            def _get_estimator() -> Any:
                nonlocal estimator
                if estimator is None:
                    estimator = FoundationPose(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=str(foundationpose_debug_dir / object_id),
                        debug=0,
                        glctx=glctx,
                    )
                    if estimator.mesh is not None:
                        estimator.mesh.vertices = _float32_contiguous(np.asarray(estimator.mesh.vertices))
                        estimator.mesh.vertex_normals = _float32_contiguous(np.asarray(estimator.mesh.vertex_normals))
                    estimator.diameter = float(estimator.diameter)
                return estimator
            registration_frame_index = mesh_record.get("registration_frame_index")
            anchor_frame = start_frame if registration_frame_index is None else int(registration_frame_index)
            anchor_frame_count = min(
                processed_images.shape[0],
                metric_depth.shape[0],
                _intrinsic_frame_count(intrinsics, processed_images.shape[0]),
            )
            if anchor_frame < 0 or anchor_frame >= anchor_frame_count:
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "registration_frame_out_of_range",
                        "registration_frame_index": anchor_frame,
                        "available_frame_count": anchor_frame_count,
                    }
                )
                continue
            registration_mask_key = str(mesh_record.get("registration_mask_key") or "")
            if not registration_mask_key:
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "missing_mesh_conditioning_registration_mask",
                        "reason": "FoundationPose registration requires the mesh_conditioning mask_key.",
                    }
                )
                continue
            registration_mask_record = sam3_mask_records.get(registration_mask_key)
            poses_by_frame: dict[int, dict[str, Any]] = {}
            if registration_mask_record is None or registration_mask_key not in sam3_track_masks:
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "missing_mesh_conditioning_registration_mask",
                        "registration_mask_key": registration_mask_key,
                        "reason": (
                            "Mesh conditioning registration mask was not found in "
                            "sam3_video_track_labels object_keyframes/sidecar."
                        ),
                    }
                )
                continue
            registration_mask = sam3_track_masks[registration_mask_key].astype(np.uint8)
            anchor_raw_pose: np.ndarray | None = None
            registration_diagnostics: dict[str, Any] = {}
            seed_raw_pose: np.ndarray | None = None
            try:
                seed_raw_pose = _seed_pose_from_mesh_record(mesh_record, object_id=object_id)
            except ValueError as exc:
                object_results.append(
                    {
                        "object_id": object_id,
                        "status": "missing_keyframe_seed_pose",
                        "registration_frame_index": anchor_frame,
                        "registration_strategy": "keyframe_seeded_single_candidate_register",
                        "reason": str(exc),
                    }
                )
                continue

            def _frame_inputs(frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                rgb = _processed_rgb_for_frame(processed_images, frame_index)
                depth_frame = _float32_contiguous(_depth_for_frame(metric_depth, frame_index))
                if depth_frame.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"Video-depth processed image and metric_depth shapes do not match at frame {frame_index}: "
                        f"rgb={rgb.shape[:2]} depth={depth_frame.shape}"
                    )
                K = _float32_contiguous(_intrinsic_for_frame(intrinsics, frame_index))
                return rgb, depth_frame, K

            anchor_preprocess_geometry = video_depth_preprocess_geometry_for_frame(
                video_metric_depth,
                anchor_frame,
            )

            def _set_pose_last_from_raw_pose(segment_estimator: Any, raw_pose: np.ndarray) -> None:
                import torch

                tf_to_center = segment_estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
                centered_pose = _float32_contiguous(np.asarray(raw_pose).reshape(4, 4) @ np.linalg.inv(tf_to_center))
                segment_estimator.pose_last = torch.as_tensor(centered_pose, device="cuda", dtype=torch.float)

            def _register_anchor_once() -> np.ndarray:
                nonlocal anchor_raw_pose, registration_diagnostics
                if anchor_raw_pose is not None:
                    return anchor_raw_pose
                anchor_rgb, anchor_depth, anchor_K = _frame_inputs(anchor_frame)
                anchor_mask = warp_mask_to_processed(
                    registration_mask,
                    affine_2x3=np.asarray(
                        anchor_preprocess_geometry["affine_original_to_processed_2x3"],
                        dtype=np.float32,
                    ),
                    shape=anchor_rgb.shape[:2],
                )
                segment_estimator = _get_estimator()
                if seed_raw_pose is None:
                    raise RuntimeError(f"{object_id} missing keyframe seed pose")
                anchor_pose, registration_diagnostics = register_keyframe_seeded_single_candidate(
                    segment_estimator=segment_estimator,
                    K=anchor_K,
                    rgb=anchor_rgb,
                    depth=anchor_depth,
                    ob_mask=anchor_mask,
                    seed_raw_pose=seed_raw_pose,
                    iteration=est_refine_iter,
                )
                anchor_raw_pose = _float32_contiguous(np.asarray(anchor_pose)).reshape(4, 4)
                if not np.isfinite(anchor_raw_pose).all() or float(np.linalg.norm(anchor_raw_pose[:3, 3])) <= 1e-8:
                    raise RuntimeError(
                        f"{object_id} FoundationPose registration produced an invalid anchor pose: "
                        f"translation={anchor_raw_pose[:3, 3].tolist()} diagnostics={registration_diagnostics}"
                    )
                return anchor_raw_pose

            def _raw_pose_item(
                *,
                frame_index: int,
                raw_pose: np.ndarray,
                tracking_direction: str,
                preprocess_geometry: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                item = {
                    "frame_index": frame_index,
                    "raw_pose_4x4": _float32_contiguous(raw_pose).reshape(4, 4).tolist(),
                    "tracking_direction": tracking_direction,
                }
                if preprocess_geometry is not None:
                    item["video_depth_preprocess_geometry"] = preprocess_geometry
                return item

            def _segment_pose_records(
                *,
                raw_poses: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                records = []
                for raw_item in raw_poses:
                    frame_index = int(raw_item["frame_index"])
                    preprocess_geometry = raw_item.get("video_depth_preprocess_geometry")
                    if frame_index == anchor_frame:
                        preprocess_geometry = anchor_preprocess_geometry
                    pose_record = _pose_record(
                        frame_index=frame_index,
                        raw_pose=_float32_contiguous(np.asarray(raw_item["raw_pose_4x4"])).reshape(4, 4),
                        geometry_type=geometry_type,
                        registration_mask_key=registration_mask_key,
                        tracking_direction=str(raw_item.get("tracking_direction") or "forward"),
                        preprocess_geometry=preprocess_geometry,
                    )
                    records.append(pose_record)
                return records

            def _run_segment(target_frame: int) -> list[dict[str, Any]]:
                nonlocal anchor_raw_pose
                direction = 1 if target_frame >= anchor_frame else -1
                tracking_direction = "forward" if direction > 0 else "backward"
                segment_estimator = _get_estimator()
                anchor_pose = _register_anchor_once()
                _set_pose_last_from_raw_pose(segment_estimator, anchor_pose)
                raw_poses = [
                    _raw_pose_item(
                        frame_index=anchor_frame,
                        raw_pose=anchor_pose,
                        tracking_direction="anchor_register",
                        preprocess_geometry=anchor_preprocess_geometry,
                    )
                ]
                for frame_index in range(anchor_frame + direction, target_frame + direction, direction):
                    rgb, depth_frame, K = _frame_inputs(frame_index)
                    pose = segment_estimator.track_one(rgb=rgb, depth=depth_frame, K=K, iteration=track_refine_iter)
                    raw_poses.append(
                        _raw_pose_item(
                            frame_index=frame_index,
                            raw_pose=_float32_contiguous(np.asarray(pose)).reshape(4, 4),
                            tracking_direction=tracking_direction,
                        )
                    )
                return _segment_pose_records(raw_poses=raw_poses)

            def _register_at_frame(frame_index: int, mask_key: str) -> tuple[np.ndarray, dict[str, Any]]:
                rgb, depth_frame, K = _frame_inputs(frame_index)
                frame_geometry = video_depth_preprocess_geometry_for_frame(
                    video_metric_depth,
                    frame_index,
                )
                frame_mask = warp_mask_to_processed(
                    sam3_track_masks[mask_key].astype(np.uint8),
                    affine_2x3=np.asarray(
                        frame_geometry["affine_original_to_processed_2x3"], dtype=np.float32
                    ),
                    shape=rgb.shape[:2],
                )
                # The mesh-conditioning seed pose stays valid at every candidate frame
                # because the fixture and the camera are both static.
                pose, diagnostics = register_keyframe_seeded_single_candidate(
                    segment_estimator=_get_estimator(),
                    K=K,
                    rgb=rgb,
                    depth=depth_frame,
                    ob_mask=frame_mask,
                    seed_raw_pose=seed_raw_pose,
                    iteration=est_refine_iter,
                )
                return _float32_contiguous(np.asarray(pose)).reshape(4, 4), diagnostics

            def _static_pose_cross_frame_iou(
                pose: np.ndarray,
                eval_frames: list[int],
                mask_key_by_frame: dict[int, str],
            ) -> dict[str, Any]:
                rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
                translation = np.asarray(pose, dtype=np.float64)[:3, 3]
                vertices_camera = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation
                faces = np.asarray(mesh.faces, dtype=np.int64)
                per_frame = []
                # The candidate pose and the camera are fixed, so the software mesh
                # rasterization is identical for every eval frame that shares the same
                # intrinsics/shape; only the per-frame depth occlusion differs. Cache
                # (rendered_mask, rendered_depth) keyed by the exact intrinsic bytes and
                # shape — bitwise-identical to rendering per frame.
                render_by_intrinsic: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
                for frame_index in eval_frames:
                    frame_geometry = video_depth_preprocess_geometry_for_frame(
                        video_metric_depth,
                        frame_index,
                    )
                    sam3_mask = warp_mask_to_processed(
                        sam3_track_masks[mask_key_by_frame[frame_index]].astype(np.uint8),
                        affine_2x3=np.asarray(
                            frame_geometry["affine_original_to_processed_2x3"], dtype=np.float32
                        ),
                        shape=tuple(processed_images.shape[1:3]),
                    )
                    intrinsic = np.asarray(_intrinsic_for_frame(intrinsics, frame_index), dtype=np.float64)
                    render_key = (np.ascontiguousarray(intrinsic).tobytes(), sam3_mask.shape)
                    cached_render = render_by_intrinsic.get(render_key)
                    if cached_render is None:
                        cached_render = render_mesh_depth(
                            vertices_camera=vertices_camera,
                            faces=faces,
                            intrinsic=intrinsic,
                            image_shape=sam3_mask.shape,
                        )
                        render_by_intrinsic[render_key] = cached_render
                    rendered_mask, rendered_depth = cached_render
                    occluder = _all_track_mask_union_at_frame(sam3_track_masks, frame_index)
                    if occluder is not None:
                        occluder = warp_mask_to_processed(
                            occluder.astype(np.uint8),
                            affine_2x3=np.asarray(
                                frame_geometry["affine_original_to_processed_2x3"], dtype=np.float32
                            ),
                            shape=tuple(processed_images.shape[1:3]),
                        )
                    visible = mask_occluded_mesh_mask(
                        rendered_mask=rendered_mask,
                        occluder_mask=occluder,
                        self_mask=sam3_mask,
                    )
                    union = float(np.logical_or(visible, sam3_mask).sum())
                    iou = float(np.logical_and(visible, sam3_mask).sum() / union) if union > 0 else float("nan")
                    per_frame.append({"frame_index": frame_index, "iou": iou})
                valid = [item["iou"] for item in per_frame if np.isfinite(item["iou"])]
                return {
                    "mean_iou": float(np.mean(valid)) if valid else None,
                    "min_iou": float(np.min(valid)) if valid else None,
                    "per_frame": per_frame,
                }

            def _run_static_fixture() -> tuple[list[dict[str, Any]], dict[str, Any], int, dict[str, Any], str]:
                mask_key_by_frame = _track_mask_keys_by_frame(sam3_track_masks, source_track_id)
                mask_frames = sorted(
                    frame for frame in mask_key_by_frame if start_frame <= frame <= end_frame
                )
                candidate_frames = _evenly_spread(mask_frames, STATIC_FIXTURE_ANCHOR_CANDIDATE_COUNT)
                if anchor_frame in mask_frames and anchor_frame not in candidate_frames:
                    candidate_frames.append(anchor_frame)
                if not candidate_frames:
                    candidate_frames = [anchor_frame]
                    mask_key_by_frame = {**mask_key_by_frame, anchor_frame: registration_mask_key}
                    mask_frames = [anchor_frame]
                eval_frames = _evenly_spread(mask_frames, STATIC_FIXTURE_MAX_EVAL_FRAMES)
                candidates = []
                for frame_index in sorted(candidate_frames):
                    candidate_mask_key = mask_key_by_frame.get(frame_index, registration_mask_key)
                    candidate: dict[str, Any] = {
                        "anchor_frame_index": int(frame_index),
                        "registration_mask_key": candidate_mask_key,
                    }
                    try:
                        pose, diagnostics = _register_at_frame(frame_index, candidate_mask_key)
                    except Exception as exc:
                        candidate.update({"status": "registration_failed", "reason": str(exc)})
                        candidates.append(candidate)
                        continue
                    if not np.isfinite(pose).all() or float(np.linalg.norm(pose[:3, 3])) <= 1e-8:
                        candidate.update({"status": "invalid_pose"})
                        candidates.append(candidate)
                        continue
                    score = _static_pose_cross_frame_iou(pose, eval_frames, mask_key_by_frame)
                    candidate.update(
                        {
                            "status": "ok",
                            "mean_iou": score["mean_iou"],
                            "min_iou": score["min_iou"],
                            "per_frame_iou": score["per_frame"],
                            "pose_4x4": pose.tolist(),
                            "registration_diagnostics": diagnostics,
                        }
                    )
                    candidates.append(candidate)
                scored = [item for item in candidates if item.get("status") == "ok" and item.get("mean_iou") is not None]
                if not scored:
                    raise RuntimeError(
                        f"{object_id} static fixture registration failed on all candidate anchor frames: "
                        f"{[item.get('status') for item in candidates]}"
                    )
                best = max(scored, key=lambda item: float(item["mean_iou"]))
                best_pose = _float32_contiguous(np.asarray(best["pose_4x4"], dtype=np.float64)).reshape(4, 4)
                best_frame = int(best["anchor_frame_index"])
                records = []
                for frame_index in range(start_frame, end_frame + 1):
                    pose_record = _pose_record(
                        frame_index=frame_index,
                        raw_pose=best_pose,
                        geometry_type=geometry_type,
                        registration_mask_key=str(best["registration_mask_key"]),
                        tracking_direction="static_fixture_constant",
                        preprocess_geometry=(
                            video_depth_preprocess_geometry_for_frame(video_metric_depth, frame_index)
                            if frame_index == best_frame
                            else None
                        ),
                    )
                    records.append(pose_record)
                selection = {
                    "policy": "static_ground_fixture_constant_pose",
                    "reason": (
                        "Static fixtures skip per-frame track_one; each candidate anchor frame is "
                        "registered independently and the pose with the best cross-frame rendered-mask "
                        "IoU against the SAM3 track is held constant for the whole video."
                    ),
                    "candidate_anchor_frame_indices": [int(item["anchor_frame_index"]) for item in candidates],
                    "eval_frame_indices": [int(frame) for frame in eval_frames],
                    "selected_anchor_frame_index": best_frame,
                    "selected_mean_iou": best["mean_iou"],
                    "selected_min_iou": best["min_iou"],
                    "low_confidence": bool(float(best["mean_iou"]) < STATIC_FIXTURE_LOW_CONFIDENCE_MEAN_IOU),
                    "low_confidence_mean_iou_threshold": STATIC_FIXTURE_LOW_CONFIDENCE_MEAN_IOU,
                    "candidates": [
                        {key: value for key, value in item.items() if key not in {"pose_4x4", "registration_diagnostics"}}
                        for item in candidates
                    ],
                }
                return (
                    records,
                    selection,
                    best_frame,
                    dict(best.get("registration_diagnostics") or {}),
                    str(best["registration_mask_key"]),
                )

            static_fixture_selection: dict[str, Any] | None = None
            if is_static_fixture:
                (
                    poses,
                    static_fixture_selection,
                    anchor_frame,
                    registration_diagnostics,
                    registration_mask_key,
                ) = _run_static_fixture()
            else:
                forward_segment = _run_segment(end_frame)
                for pose_record in forward_segment:
                    poses_by_frame[int(pose_record["frame_index"])] = pose_record

                if anchor_frame > start_frame:
                    backward_segment = _run_segment(start_frame)
                    for pose_record in backward_segment:
                        frame_index = int(pose_record["frame_index"])
                        if frame_index == anchor_frame and frame_index in poses_by_frame:
                            continue
                        poses_by_frame[frame_index] = pose_record
                poses = [
                    poses_by_frame[index]
                    for index in sorted(poses_by_frame)
                    if start_frame <= index <= end_frame
                ]
            object_payload = {
                "object_id": object_id,
                "status": "ok" if poses else "no_visible_frames",
                "geometry_type": geometry_type,
                "pose_postprocess": {
                    "method": "none",
                    "applied_to_pose_4x4": False,
                    "raw_pose_field": "raw_pose_4x4",
                    "symmetry_aware_pose_field": "symmetry_aware_pose_4x4",
                    "all_geometry_types": "raw FoundationPose 6D pose preserved",
                },
                "tracking_start_frame_index": start_frame,
                "tracking_end_frame_index": end_frame,
                "registration_frame_index": anchor_frame,
                "registration_mask_key": registration_mask_key,
                "registration_strategy": (
                    "static_fixture_cross_frame_verified_register"
                    if is_static_fixture
                    else registration_diagnostics.get(
                        "registration_strategy", "mesh_conditioning_anchor_bidirectional_tracking"
                    )
                ),
                "registration_diagnostics": registration_diagnostics,
                "mesh_fingerprint": mesh_fingerprint,
                "initial_mask_artifact": str(artifact_path_by_name(question_dir, "pose_sam3_masks.json")),
                "initial_mask_selection_field": "selected_masks",
                "initial_mask_sidecar": pose_sam3_masks["mask_sidecar"],
                "initial_mask_key": initial_mask_record["mask_key"],
                "selected_candidate_id": initial_mask_record.get("selected_candidate_id"),
                "selected_candidate_overlay_path": initial_mask_record.get("selected_candidate_overlay_path"),
                "pose_frames_artifact": str(artifact_path_by_name(question_dir, "pose_frames.json")),
                "projected_mesh_path": mesh_record.get("projected_mesh_path"),
                "foundationpose_mesh_path": mesh_record.get("foundationpose_mesh_path"),
                "foundationpose_initial_pose_4x4": mesh_record.get("foundationpose_initial_pose_4x4"),
                "foundationpose_local_origin_camera_xyz": mesh_record.get("foundationpose_local_origin_camera_xyz"),
                "mesh_path": mesh_path,
                "mesh_conditioning": mesh_record.get("mesh_conditioning"),
                "mesh_coordinate_frame": mesh_record.get("coordinate_frame"),
                "projected_mesh_coordinate_frame": mesh_record.get("projected_mesh_coordinate_frame"),
                "rgb_source": "video_depth_processed_images",
                "depth_source": "video_metric_depth",
                "camera_intrinsics_artifact": video_metric_depth.get("camera_intrinsics_artifact"),
                "fixed_intrinsics": video_metric_depth.get("fixed_intrinsics"),
                "fixed_intrinsics_source": video_metric_depth.get("fixed_intrinsics_source"),
                "intrinsics_source": "video_metric_depth.tensor_sidecar.intrinsics",
                "intrinsics_coordinate_frame": video_metric_depth.get("sidecar_intrinsics_coordinate_frame"),
                "poses": poses,
            }
            if is_static_fixture:
                object_payload["motion_model"] = "static_ground_fixture"
                object_payload["static_fixture_selection"] = static_fixture_selection
            object_results.append(object_payload)

    payload = {
        "tool": "foundationpose",
        "status": "ok",
        "execution_mode": execution_mode,
        "video": str(video),
        "object_plan": str(object_plan),
        "pose_frames": str(artifact_path_by_name(question_dir, "pose_frames.json")),
        "pose_sam3_masks": str(artifact_path_by_name(question_dir, "pose_sam3_masks.json")),
        "video_metric_depth": str(artifact_path_by_name(question_dir, "video_metric_depth.json")),
        "camera_intrinsics_artifact": video_metric_depth.get("camera_intrinsics_artifact"),
        "rgb_source": "video_depth_processed_images",
        "depth_source": "video_metric_depth",
        "fixed_intrinsics": video_metric_depth.get("fixed_intrinsics"),
        "fixed_intrinsics_source": video_metric_depth.get("fixed_intrinsics_source"),
        "intrinsics_source": "video_metric_depth.tensor_sidecar.intrinsics",
        "intrinsics_coordinate_frame": video_metric_depth.get("sidecar_intrinsics_coordinate_frame"),
        "debug_artifacts": debug_enabled,
        "physion_pp_agent_geometry_policy": agent_geometry_policy,
        "foundationpose_agent_geometry_route": agent_geometry_route,
        "objects": object_results,
        "note": "FoundationPose uses video-depth processed_images, metric_depth, and MoGe-2 fixed intrinsics carried by the video-depth sidecar in processed-image coordinates. Registration masks are warped into video-depth processed-image coordinates using the recorded preprocess geometry for register(), then each object is tracked from its first complete unobstructed frame through its last complete unobstructed frame.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug-artifacts", type=int, default=None)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_foundationpose(
        video=Path(args.video),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
        debug_artifacts=args.debug_artifacts,
    )


if __name__ == "__main__":
    main()
