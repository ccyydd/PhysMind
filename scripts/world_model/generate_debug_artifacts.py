from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.world_model.debug_artifacts import (
    generate_foundationpose_debug_from_artifact,
    generate_sam3_video_track_debug_from_artifact,
    render_qcpr_plane_projection_debug,
)
from agent.world_model.artifacts import ArtifactManager
from agent.world_model.tools import PoseCorrectionAdapter, SimulatableWorldReconstructionAdapter
from scripts.world_model.run_impulse_analytic_sysid import render_analytic_result_debug_artifacts
from scripts.world_model.run_mesh_conditioning import (
    _depth_for_frame,
    _intrinsic_for_frame,
    _load_mesh,
    _load_projection_inputs,
    _save_projection_debug_images,
)
from scripts.world_model.run_video_metric_depth import _write_depth_colormap_video
from utils.config import ModelConfig


def _scene_root(run_dir: Path, scene_id: int) -> Path:
    root = run_dir / "artifacts" / f"scene_{scene_id}"
    if not root.exists():
        raise FileNotFoundError(f"Scene artifact directory does not exist: {root}")
    return root


def _find_sam3_video_track_artifacts(scene_root: Path) -> list[Path]:
    candidates = [
        scene_root
        / "world-modeling"
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json",
    ]
    candidates.extend(
        sorted(
            scene_root.glob(
                "question_*/object-segmentation-and-event-detection/sam3_video_tracks/sam3_video_tracks.json"
            )
        )
    )
    return [path for path in candidates if path.exists()]


def _find_foundationpose_artifacts(scene_root: Path) -> list[Path]:
    candidates = [
        scene_root
        / "world-modeling"
        / "pose-estimation-and-tracking"
        / "foundationpose"
        / "foundationpose_poses.json",
    ]
    candidates.extend(
        sorted(scene_root.glob("question_*/pose-estimation-and-tracking/foundationpose/foundationpose_poses.json"))
    )
    return [path for path in candidates if path.exists()]


def _find_qcpr_trajectory_artifacts(scene_root: Path) -> list[Path]:
    return sorted(
        scene_root.glob(
            "question_*/query-conditioned-physical-rollout/query_conditioned_physical_rollout/simulation/trajectory.json"
        )
    )


def _world_modeling_dir(scene_root: Path) -> Path:
    return scene_root / "world-modeling"


def _generate_video_metric_depth_debug(*, scene_root: Path) -> dict[str, Any]:
    artifact_path = (
        _world_modeling_dir(scene_root)
        / "metric-mesh-reconstruction"
        / "video_metric_depth"
        / "video_metric_depth.json"
    )
    if not artifact_path.exists():
        return {
            "stage": "video-metric-depth",
            "status": "missing_artifact",
            "message": "video_metric_depth.json was not found for this scene",
        }
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    sidecar_path = Path(str(payload.get("tensor_sidecar") or artifact_path.with_suffix(".npz")))
    if not sidecar_path.exists():
        return {
            "stage": "video-metric-depth",
            "artifact_path": str(artifact_path),
            "status": "missing_artifact",
            "message": f"video metric depth sidecar was not found: {sidecar_path}",
        }
    import numpy as np

    with np.load(sidecar_path) as arrays:
        if "metric_depth" not in arrays:
            return {
                "stage": "video-metric-depth",
                "artifact_path": str(artifact_path),
                "status": "missing_artifact",
                "message": f"metric_depth array was not found in sidecar: {sidecar_path}",
            }
        metric_depth = arrays["metric_depth"]
    metadata = payload.get("video_metadata") if isinstance(payload.get("video_metadata"), dict) else {}
    runtime = payload.get("video_depth_runtime") if isinstance(payload.get("video_depth_runtime"), dict) else {}
    debug_artifacts = dict(payload.get("debug_artifacts") if isinstance(payload.get("debug_artifacts"), dict) else {})
    debug_artifacts["video_metric_depth_colormap"] = _write_depth_colormap_video(
        metric_depth,
        artifact_path.parent / "debug" / "video_metric_depth_colormap.mp4",
        fps=float(runtime.get("fps") or metadata.get("fps") or 12.0),
    )
    payload["debug_artifacts"] = debug_artifacts
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "stage": "video-metric-depth",
        "artifact_path": str(artifact_path),
        "status": "ok",
        "debug_artifacts": debug_artifacts,
    }


def _generate_mesh_conditioning_debug(*, scene_root: Path) -> dict[str, Any]:
    world_modeling_dir = _world_modeling_dir(scene_root)
    artifact_path = (
        world_modeling_dir
        / "metric-mesh-reconstruction"
        / "mesh_conditioning"
        / "mesh_conditioning.json"
    )
    if not artifact_path.exists():
        return {
            "stage": "mesh-conditioning",
            "status": "missing_artifact",
            "message": "mesh_conditioning.json was not found for this scene",
        }
    video_artifact_path = (
        world_modeling_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json"
    )
    if not video_artifact_path.exists():
        return {
            "stage": "mesh-conditioning",
            "artifact_path": str(artifact_path),
            "status": "missing_artifact",
            "message": f"sam3_video_tracks.json was not found: {video_artifact_path}",
        }
    video_artifact = json.loads(video_artifact_path.read_text(encoding="utf-8"))
    video = Path(str(video_artifact.get("video") or ""))
    if not video.exists():
        return {
            "stage": "mesh-conditioning",
            "artifact_path": str(artifact_path),
            "status": "missing_artifact",
            "message": f"source video was not found: {video}",
        }
    projection_inputs = _load_projection_inputs(world_modeling_dir)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    debug_root = artifact_path.parent / "debug_projection"
    projected_dir = artifact_path.parent / "projected_meshes"
    object_results = []
    for item in payload.get("objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if not object_id:
            continue
        keyframe = projection_inputs["keyframes_by_object"].get(object_id)
        projected_mesh_path = projected_dir / f"{object_id}_projected.glb"
        if not keyframe or not projected_mesh_path.exists():
            object_results.append(
                {
                    "object_id": object_id,
                    "status": "missing_debug_input",
                    "projected_mesh_path": str(projected_mesh_path),
                    "has_keyframe": bool(keyframe),
                    "debug_projection_images": {},
                }
            )
            continue
        frame_index = int(keyframe["frame_index"])
        mask_key = keyframe.get("mask_key")
        if not mask_key or mask_key not in projection_inputs["masks"]:
            object_results.append(
                {
                    "object_id": object_id,
                    "status": "missing_mask",
                    "mask_key": mask_key,
                    "debug_projection_images": {},
                }
            )
            continue
        try:
            debug_images = _save_projection_debug_images(
                video=video,
                frame_index=frame_index,
                object_id=object_id,
                object_dir=debug_root / object_id,
                sam3_mask=projection_inputs["masks"][mask_key].astype(bool),
                video_depth=_depth_for_frame(projection_inputs["metric_depth"], frame_index),
                projected_mesh=_load_mesh(projected_mesh_path),
                intrinsic=_intrinsic_for_frame(projection_inputs["intrinsics"], frame_index),
            )
            item["debug_projection_images"] = debug_images
            object_results.append(
                {
                    "object_id": object_id,
                    "status": "ok",
                    "debug_projection_images": debug_images,
                }
            )
        except Exception as exc:
            object_results.append(
                {
                    "object_id": object_id,
                    "status": "debug_generation_error",
                    "error": str(exc),
                    "debug_projection_images": {},
                }
            )
    payload["objects"] = [item for item in payload.get("objects") or [] if isinstance(item, dict)]
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "stage": "mesh-conditioning",
        "artifact_path": str(artifact_path),
        "status": "ok" if object_results and all(item.get("status") == "ok" for item in object_results) else "partial",
        "object_count": len(object_results),
        "objects": object_results,
    }


def _generate_qcpr_plane_projection_debug(*, scene_root: Path) -> dict[str, Any]:
    fit_path = (
        scene_root
        / "world-modeling"
        / "simulatable-world-reconstruction"
        / "fit"
        / "world_reconstruction_fit.json"
    )
    if not fit_path.exists():
        return {
            "stage": "query-conditioned-physical-rollout",
            "status": "missing_artifact",
            "message": "world_reconstruction_fit.json was not found for this scene",
        }
    artifact_paths = _find_qcpr_trajectory_artifacts(scene_root)
    if not artifact_paths:
        return {
            "stage": "query-conditioned-physical-rollout",
            "status": "missing_artifact",
            "message": "QCPR trajectory.json was not found for this scene",
        }
    generated = []
    for artifact_path in artifact_paths:
        try:
            result = render_qcpr_plane_projection_debug(
                trajectory_path=artifact_path,
                world_reconstruction_fit_path=fit_path,
            )
        except Exception as exc:
            result = {
                "status": "tool_error",
                "error_message": str(exc),
                "trajectory_path": str(artifact_path),
                "world_reconstruction_fit_path": str(fit_path),
            }
        generated.append(result)
    return {
        "stage": "query-conditioned-physical-rollout",
        "status": "ok" if generated and all(item.get("status") in {"ok", "skipped"} for item in generated) else "partial",
        "artifact_count": len(artifact_paths),
        "results": generated,
    }


def generate_debug_artifacts(*, run_dir: Path, scene_id: int, stage: str) -> dict[str, Any]:
    scene_root = _scene_root(run_dir, scene_id)
    artifact_manager = ArtifactManager(run_dir, debug_artifacts=True)
    config = ModelConfig()
    results = []
    if stage in {"all", "sam3-video-tracks"}:
        artifact_paths = _find_sam3_video_track_artifacts(scene_root)
        if not artifact_paths:
            results.append(
                {
                    "stage": "sam3-video-tracks",
                    "status": "missing_artifact",
                    "message": "sam3_video_tracks.json was not found for this scene",
                }
            )
        for artifact_path in artifact_paths:
            results.append(generate_sam3_video_track_debug_from_artifact(artifact_path=artifact_path))
    if stage in {"all", "foundationpose"}:
        artifact_paths = _find_foundationpose_artifacts(scene_root)
        if not artifact_paths:
            results.append(
                {
                    "stage": "foundationpose",
                    "status": "missing_artifact",
                    "message": "foundationpose_poses.json was not found for this scene",
                }
            )
        for artifact_path in artifact_paths:
            results.append(generate_foundationpose_debug_from_artifact(artifact_path=artifact_path))
    if stage in {"all", "video-metric-depth"}:
        results.append(_generate_video_metric_depth_debug(scene_root=scene_root))
    if stage in {"all", "mesh-conditioning"}:
        results.append(_generate_mesh_conditioning_debug(scene_root=scene_root))
    if stage in {"all", "query-conditioned-physical-rollout"}:
        results.append(_generate_qcpr_plane_projection_debug(scene_root=scene_root))
    if stage in {"all", "pose-correction"}:
        pose_correction_path = (
            scene_root
            / "world-modeling"
            / "pose-estimation-and-tracking"
            / "pose_correction"
            / "pose_correction.json"
        )
        if not pose_correction_path.exists():
            results.append(
                {
                    "stage": "pose-correction",
                    "status": "missing_artifact",
                    "message": "pose_correction.json was not found for this scene",
                }
            )
        else:
            adapter = PoseCorrectionAdapter(artifacts=artifact_manager, config=config, dry_run=False)
            payload = json.loads(pose_correction_path.read_text(encoding="utf-8"))
            adapter._maybe_render_world_reconstruction_step_debug(
                world_reconstruction_path=pose_correction_path,
                payload=payload,
                question_dir=scene_root / "world-modeling",
            )
            updated = json.loads(pose_correction_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "stage": "pose-correction",
                    "artifact_path": str(pose_correction_path),
                    "status": "ok",
                    "debug_artifacts": updated.get("debug_artifacts"),
                }
            )
    if stage in {"all", "simulatable-world-reconstruction"}:
        fit_path = scene_root / "world-modeling" / "simulatable-world-reconstruction" / "fit" / "world_reconstruction_fit.json"
        manifest_path = (
            scene_root
            / "world-modeling"
            / "simulatable-world-reconstruction"
            / "fit"
            / "world_reconstruction_fit_manifest.json"
        )
        if not fit_path.exists() or not manifest_path.exists():
            results.append(
                {
                    "stage": "simulatable-world-reconstruction",
                    "status": "missing_artifact",
                    "message": "world_reconstruction_fit.json or world_reconstruction_fit_manifest.json was not found for this scene",
                }
            )
        else:
            adapter = SimulatableWorldReconstructionAdapter(artifacts=artifact_manager, config=config, dry_run=False)
            result = json.loads(fit_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            adapter._render_physics_alignment_blender_debug(
                question_dir=scene_root / "world-modeling",
                result_path=fit_path,
                result=result,
                physics_alignment_manifest=manifest,
                command=os.getenv("PHYSMIND_BLENDER_CMD") or str(PROJECT_ROOT / "third_party" / "blender" / "blender"),
            )
            updated = json.loads(fit_path.read_text(encoding="utf-8"))
            alignment = updated.get("alignment_optimization") if isinstance(updated.get("alignment_optimization"), dict) else {}
            analytic_result_path = alignment.get("result_path")
            input_fit_path = alignment.get("input_fit")
            sketch_debug = None
            if analytic_result_path and input_fit_path:
                sketch_debug = render_analytic_result_debug_artifacts(
                    input_fit=Path(str(input_fit_path)),
                    result_json=Path(str(analytic_result_path)),
                )
                analytic_result = json.loads(Path(str(analytic_result_path)).read_text(encoding="utf-8"))
                rollout = updated.setdefault("physics_rollout", {})
                if isinstance(rollout, dict):
                    rollout["sketch_comparison_plot_path"] = analytic_result.get("free_rollout_plot_path")
                    rollout["sketch_comparison_video_path"] = analytic_result.get("free_rollout_video_path")
                alignment = updated.setdefault("alignment_optimization", {})
                if isinstance(alignment, dict):
                    alignment["free_rollout_plot_path"] = analytic_result.get("free_rollout_plot_path")
                    alignment["free_rollout_video_path"] = analytic_result.get("free_rollout_video_path")
                    alignment["sketch_comparison_video_path"] = analytic_result.get("free_rollout_video_path")
                fit_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(
                {
                    "stage": "simulatable-world-reconstruction",
                    "artifact_path": str(fit_path),
                    "status": "ok",
                    "blender_debug_render": (updated.get("physics_rollout") or {}).get("blender_debug_render"),
                    "sketch_debug_render": sketch_debug,
                }
            )

    status = "ok" if results and all(item.get("status") == "ok" for item in results) else "partial"
    return {
        "tool": "generate_debug_artifacts",
        "status": status,
        "run_dir": str(run_dir),
        "scene_id": int(scene_id),
        "stage": stage,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "sam3-video-tracks",
            "foundationpose",
            "video-metric-depth",
            "mesh-conditioning",
            "pose-correction",
            "query-conditioned-physical-rollout",
            "simulatable-world-reconstruction",
        ],
        default="all",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = generate_debug_artifacts(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        scene_id=args.scene_id,
        stage=args.stage,
    )
    if args.output:
        output = Path(args.output)
    else:
        output = (
            Path(args.run_dir).expanduser().resolve()
            / "artifacts"
            / f"scene_{args.scene_id}"
            / "debug_artifacts_generation.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
