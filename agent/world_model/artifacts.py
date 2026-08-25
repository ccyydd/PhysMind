from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE = "object-segmentation-and-event-detection"


STAGE_DIR_BY_TOOL = {
    "object_plan": "object-identification-and-planning",
    "sam3_video_tracks": OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE,
    "sam3_video_track_labels": OBJECT_SEGMENTATION_AND_EVENT_DETECTION_STAGE,
    "moge2_intrinsics": "metric-mesh-reconstruction",
    "sam3d_objects": "metric-mesh-reconstruction",
    "mesh_conditioning": "metric-mesh-reconstruction",
    "video_metric_depth": "metric-mesh-reconstruction",
    "pose_frames": "pose-estimation-and-tracking",
    "pose_sam3_masks": "pose-estimation-and-tracking",
    "pose_sam3_mask_selection": "pose-estimation-and-tracking",
    "pose_frame_boundary_validation": "pose-estimation-and-tracking",
    "foundationpose": "pose-estimation-and-tracking",
    "pose_correction": "pose-estimation-and-tracking",
    "simulatable_world_reconstruction": "simulatable-world-reconstruction",
    "query_conditioned_physical_rollout": "query-conditioned-physical-rollout",
    "final_answer": "artifact-based-answering",
}

ARTIFACT_TOOL_BY_NAME = {
    "object_plan.json": "object_plan",
    "sam3_video_tracks.json": "sam3_video_tracks",
    "sam3_video_track_labels.json": "sam3_video_track_labels",
    "sam3d_meshes.json": "sam3d_objects",
    "moge2_intrinsics.json": "moge2_intrinsics",
    "mesh_conditioning.json": "mesh_conditioning",
    "video_metric_depth.json": "video_metric_depth",
    "pose_frames.json": "pose_frames",
    "pose_sam3_masks.json": "pose_sam3_masks",
    "pose_frame_boundary_validation.json": "pose_frame_boundary_validation",
    "foundationpose_poses.json": "foundationpose",
    "pose_correction.json": "pose_correction",
    "simulatable_world_reconstruction.json": "simulatable_world_reconstruction",
    "trajectory.json": "query_conditioned_physical_rollout",
    "final_answer.json": "final_answer",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stage_dir_for_tool(tool_name: str) -> str:
    return STAGE_DIR_BY_TOOL.get(tool_name, tool_name.replace("_", "-"))


def tool_dir(question_dir: Path, tool_name: str) -> Path:
    if tool_name == "simulatable_world_reconstruction":
        return question_dir / stage_dir_for_tool(tool_name)
    return question_dir / stage_dir_for_tool(tool_name) / tool_name


def artifact_path(question_dir: Path, tool_name: str, artifact_name: str) -> Path:
    if tool_name == "simulatable_world_reconstruction":
        root = tool_dir(question_dir, tool_name)
        if artifact_name == "simulatable_world_reconstruction.json":
            return root / "summary" / artifact_name
        if artifact_name in {"world_reconstruction_fit_manifest.json", "world_reconstruction_fit.json"}:
            return root / "fit" / artifact_name
        return root / artifact_name
    if tool_name == "query_conditioned_physical_rollout" and artifact_name == "trajectory.json":
        return tool_dir(question_dir, tool_name) / "simulation" / artifact_name
    return tool_dir(question_dir, tool_name) / artifact_name


def artifact_path_by_name(question_dir: Path, artifact_name: str) -> Path:
    tool_name = ARTIFACT_TOOL_BY_NAME[artifact_name]
    return artifact_path(question_dir, tool_name, artifact_name)


def question_root_from_output(output: Path) -> Path:
    env_root = os.environ.get("PHYSMIND_QUESTION_DIR")
    if env_root:
        return Path(env_root)
    if output.parent.parent.name in set(STAGE_DIR_BY_TOOL.values()):
        return output.parent.parent.parent
    return output.parent


class ArtifactManager:
    def __init__(
        self,
        run_dir: Path,
        *,
        debug_artifacts: bool = False,
    ):
        self.run_dir = run_dir
        self.debug_artifacts = debug_artifacts
        self.trace_path = run_dir / "trace.jsonl"

    def question_dir(self, scene_index: int, question_id: int) -> Path:
        return self.run_dir / "artifacts" / f"scene_{scene_index}" / f"question_{question_id}"

    def write(self, path: Path, payload: Any) -> Path:
        write_json(path, payload)
        return path

    def tool_dir(self, question_dir: Path, tool_name: str) -> Path:
        return tool_dir(question_dir, tool_name)

    def artifact_path(self, question_dir: Path, tool_name: str, artifact_name: str) -> Path:
        return artifact_path(question_dir, tool_name, artifact_name)

    def artifact_path_by_name(self, question_dir: Path, artifact_name: str) -> Path:
        return artifact_path_by_name(question_dir, artifact_name)

    def read_optional(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def append_trace(self, event: Dict[str, Any]) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
