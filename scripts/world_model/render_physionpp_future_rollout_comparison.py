from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.world_model import run_physionpp_friction_sphere_sysid as swr
from scripts.world_model import swr_sysid_common as sysid_common


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_raw_video_by_scene(video_manifest: Path) -> dict[int, Path]:
    items = _load_json(video_manifest)
    if not isinstance(items, list):
        raise ValueError(f"expected list video_manifest: {video_manifest}")
    output = {}
    for item in items:
        if isinstance(item, dict) and item.get("scene_index") is not None and item.get("source_raw_video"):
            output[int(item["scene_index"])] = Path(str(item["source_raw_video"]))
    return output


def _read_video_frames(path: Path, *, width: int, height: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    capture.release()
    return frames, fps


def _plane_hulls(*, manifest: dict[str, Any], fit: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    if camera_intrinsic is None:
        return []
    target = sysid_common._target_records_from_swr(manifest)
    best_params = (fit.get("alignment_optimization") or {}).get("best_parameters") or {}
    agent_id = next(iter(best_params.keys()))
    static_ids = [object_id for object_id in target.keys() if object_id != agent_id]
    physics = fit.get("physics_rollout") if isinstance(fit.get("physics_rollout"), dict) else {}
    bounded = physics.get("bounded_static_planes") if isinstance(physics.get("bounded_static_planes"), dict) else {}
    debug_path = bounded.get("debug_path")
    if not debug_path or not Path(str(debug_path)).exists():
        return []
    bounded_static_planes = _load_json(Path(str(debug_path)))
    if not isinstance(bounded_static_planes, dict):
        return []
    static_meshes = swr._static_meshes_world(manifest=manifest, target=target, static_ids=static_ids)
    return swr._prepare_bounded_plane_hulls(
        bounded_static_planes=bounded_static_planes,
        static_meshes=static_meshes,
        camera_intrinsic=camera_intrinsic,
        width=width,
        height=height,
    )


def _draw_hulls(image: np.ndarray, hulls: list[dict[str, Any]]) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    for hull in hulls:
        pts = np.asarray(hull["hull"], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(value) for value in hull["color"])
        cv2.fillPoly(overlay, [pts], color)
    output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)
    for hull in hulls:
        pts = np.asarray(hull["hull"], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(value) for value in hull["color"])
        cv2.polylines(output, [pts], True, color, 1, lineType=cv2.LINE_AA)
    return output


def _project_plane_geometry(
    *,
    planes: list[dict[str, Any]],
    camera_intrinsic: list[list[float]],
    wall_object_id: str,
    patient_object_id: str | None,
) -> list[dict[str, Any]]:
    hulls: list[dict[str, Any]] = []
    for plane in planes:
        if not isinstance(plane, dict):
            continue
        pixels: list[list[float]] = []
        for triangle in plane.get("triangles") or []:
            for vertex in triangle:
                projected = swr._project_blender_to_pixel(
                    np.asarray(vertex, dtype=np.float64),
                    camera_intrinsic,
                )
                if projected is not None:
                    pixels.append([float(projected[0]), float(projected[1])])
        if len(pixels) < 3:
            continue
        hull = cv2.convexHull(np.asarray(pixels, dtype=np.float32)).reshape(-1, 2)
        object_id = str(plane.get("object_id") or "")
        if object_id == patient_object_id:
            color = (50, 210, 235)
        elif object_id == wall_object_id:
            color = (145, 145, 145)
        elif object_id == "global_ground":
            color = (225, 225, 225)
        else:
            color = (185, 205, 185)
        hulls.append(
            {
                "plane_id": plane.get("plane_id"),
                "object_id": object_id,
                "hull": hull,
                "color": color,
            }
        )
    hulls.sort(key=lambda item: item["object_id"] != "global_ground")
    return hulls


def _project_sphere(
    *,
    image: np.ndarray,
    position: list[float],
    radius: float,
    camera_intrinsic: list[list[float]],
    color: tuple[int, int, int],
) -> None:
    projected = swr._project_blender_to_pixel(np.asarray(position, dtype=np.float64), camera_intrinsic)
    if projected is None:
        return
    u, v, z = projected
    k = np.asarray(camera_intrinsic, dtype=np.float64).reshape(3, 3)
    pixel_radius = max(3, int(round(abs(float(k[0, 0]) * float(radius) / max(float(z), 1e-6)))))
    center = (int(round(u)), int(round(v)))
    cv2.circle(image, center, pixel_radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, center, pixel_radius, (30, 30, 30), 1, lineType=cv2.LINE_AA)


def _render_friction_scene(
    *,
    scene_id: int,
    raw_video_path: Path,
    manifest_path: Path,
    fit_path: Path,
    future_path: Path,
    width: int,
    height: int,
) -> tuple[list[np.ndarray], float]:
    manifest = _load_json(manifest_path)
    fit = _load_json(fit_path)
    future = _load_json(future_path)
    if not isinstance(manifest, dict) or not isinstance(fit, dict) or not isinstance(future, dict):
        raise ValueError(f"invalid inputs for scene {scene_id}")
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    if camera_intrinsic is None:
        raise ValueError(f"missing camera intrinsics for scene {scene_id}")
    raw_frames, fps = _read_video_frames(raw_video_path, width=width, height=height)
    hulls = _plane_hulls(manifest=manifest, fit=fit, width=width, height=height)
    best_params = (fit.get("alignment_optimization") or {}).get("best_parameters") or {}
    agent_id = str(future.get("agent_object_id") or next(iter(best_params.keys())))
    radius = float((best_params.get(agent_id) or {}).get("optimized_radius_m") or 0.0)
    observed = {
        int(record["frame_index"]): record
        for record in ((fit.get("physics_rollout") or {}).get("simulated_trajectories") or {}).get(agent_id, [])
        if isinstance(record, dict) and record.get("frame_index") is not None
    }
    future_records = {
        int(record["frame_index"]): record
        for record in future.get("future_trajectory", [])
        if isinstance(record, dict) and record.get("frame_index") is not None
    }
    contact = future.get("patient_contact") if isinstance(future.get("patient_contact"), dict) else {}
    first_contact = contact.get("first_contact") if isinstance(contact.get("first_contact"), dict) else None
    last_frame = int((future.get("horizon") or {}).get("rollout_last_frame") or max([*observed.keys(), *future_records.keys()]))
    output_frames: list[np.ndarray] = []
    for frame_index in range(0, min(last_frame, len(raw_frames) - 1) + 1):
        left = raw_frames[frame_index].copy()
        right = np.full((height, width, 3), 246, dtype=np.uint8)
        right = _draw_hulls(right, hulls)
        record = observed.get(frame_index) or future_records.get(frame_index)
        if record is not None:
            _project_sphere(
                image=right,
                position=record["position"],
                radius=radius,
                camera_intrinsic=camera_intrinsic,
                color=(30, 30, 230) if frame_index <= 89 else (30, 130, 255),
            )
        cv2.putText(left, f"raw full video | scene {scene_id} | frame {frame_index}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        pred = "yes" if contact.get("will_contact") else "no"
        first_text = "-" if first_contact is None else str(first_contact.get("frame_index"))
        cv2.putText(right, f"inverted + future rollout | pred={pred} first={first_text}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        if frame_index == 89:
            cv2.putText(right, "observed boundary", (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1, cv2.LINE_AA)
        output_frames.append(np.concatenate([left, right], axis=1))
    return output_frames, fps


def _unwrap_future_rollout(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("physion_pp_future_rollout")
    return nested if isinstance(nested, dict) else payload


def _render_bouncy_wall_scene(
    *,
    scene_id: int,
    raw_video_path: Path,
    fit: dict[str, Any],
    future_payload: dict[str, Any],
    width: int,
    height: int,
    expected_answer: str | None,
) -> tuple[list[np.ndarray], float]:
    source_world_modeling_dir = Path(str(fit.get("source_world_modeling_dir") or ""))
    if not source_world_modeling_dir.exists():
        raise ValueError(f"missing source world-modeling directory for scene {scene_id}")
    manifest = sysid_common.build_manifest_from_world_modeling(source_world_modeling_dir)
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    if camera_intrinsic is None:
        raise ValueError(f"missing camera intrinsics for scene {scene_id}")
    raw_frames, fps = _read_video_frames(raw_video_path, width=width, height=height)
    if not raw_frames:
        raise ValueError(f"raw video contains no frames: {raw_video_path}")

    observed: dict[int, dict[str, Any]] = {}
    segment_hulls: list[tuple[int, int, list[dict[str, Any]], str]] = []
    for segment in fit.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        agent_id = str(segment.get("agent_object_id") or "")
        physics = segment.get("physics_rollout") if isinstance(segment.get("physics_rollout"), dict) else {}
        trajectories = physics.get("simulated_trajectories") if isinstance(physics.get("simulated_trajectories"), dict) else {}
        for record in trajectories.get(agent_id) or []:
            if isinstance(record, dict) and record.get("frame_index") is not None:
                observed[int(record["frame_index"])] = record
        frame_range = segment.get("frame_range") or []
        if len(frame_range) != 2:
            continue
        segment_hulls.append(
            (
                int(frame_range[0]),
                int(frame_range[1]),
                _project_plane_geometry(
                    planes=[item for item in physics.get("contact_plane_geometry") or [] if isinstance(item, dict)],
                    camera_intrinsic=camera_intrinsic,
                    wall_object_id=str(segment.get("wall_object_id") or ""),
                    patient_object_id=(
                        str(segment.get("patient_object_id"))
                        if segment.get("patient_object_id") is not None
                        else None
                    ),
                ),
                str(segment.get("segment") or "segment"),
            )
        )

    future = _unwrap_future_rollout(future_payload)
    future_records = {
        int(record["frame_index"]): record
        for record in future.get("future_trajectory") or []
        if isinstance(record, dict) and record.get("frame_index") is not None
    }
    contact = future.get("patient_contact") if isinstance(future.get("patient_contact"), dict) else {}
    first_contact = contact.get("first_contact") if isinstance(contact.get("first_contact"), dict) else None
    horizon = future.get("horizon") if isinstance(future.get("horizon"), dict) else {}
    observed_last_frame = int(horizon.get("observed_last_frame") or max(observed))
    rollout_last_frame = int(
        horizon.get("rollout_last_frame")
        or max([*observed.keys(), *future_records.keys()])
    )
    radius = float(
        ((fit.get("joint_alignment_optimization") or {}).get("best_shared_parameters") or {}).get(
            "optimized_radius_m",
            0.0,
        )
    )
    predicted_answer = "yes" if contact.get("will_contact") else "no"
    correctness = expected_answer is None or predicted_answer == expected_answer
    output_frames: list[np.ndarray] = []

    for frame_index in range(rollout_last_frame + 1):
        raw_available = frame_index < len(raw_frames)
        left = raw_frames[min(frame_index, len(raw_frames) - 1)].copy()
        if not raw_available:
            left = cv2.addWeighted(left, 0.45, np.zeros_like(left), 0.55, 0.0)
        right = np.full((height, width, 3), 246, dtype=np.uint8)
        active_segment = None
        for start, end, hulls, segment_name in segment_hulls:
            if start <= frame_index <= end or (frame_index > observed_last_frame and segment_name == "seg2"):
                right = _draw_hulls(right, hulls)
                active_segment = segment_name
                break
        record = observed.get(frame_index) or future_records.get(frame_index)
        if record is not None:
            _project_sphere(
                image=right,
                position=record["position"],
                radius=radius,
                camera_intrinsic=camera_intrinsic,
                color=(225, 80, 35) if frame_index <= observed_last_frame else (30, 80, 235),
            )

        cv2.putText(
            left,
            f"raw full video | scene {scene_id} | frame {frame_index}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if not raw_available:
            cv2.putText(
                left,
                f"raw ended at frame {len(raw_frames) - 1}",
                (8, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
        first_text = "-" if first_contact is None else str(first_contact.get("frame_index"))
        expected_text = "-" if expected_answer is None else expected_answer
        status_color = (20, 120, 20) if correctness else (20, 20, 200)
        cv2.putText(
            right,
            f"SWR + future | pred={predicted_answer} gt={expected_text} first={first_text}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            status_color,
            1,
            cv2.LINE_AA,
        )
        phase = "future rollout" if frame_index > observed_last_frame else (active_segment or "between segments")
        cv2.putText(
            right,
            phase,
            (8, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        output_frames.append(np.concatenate([left, right], axis=1))
    return output_frames, fps


def render_scene(
    *,
    scene_id: int,
    raw_video_path: Path,
    manifest_path: Path | None,
    fit_path: Path,
    future_path: Path,
    width: int,
    height: int,
    expected_answer: str | None = None,
) -> tuple[list[np.ndarray], float]:
    fit = _load_json(fit_path)
    future = _load_json(future_path)
    if not isinstance(fit, dict) or not isinstance(future, dict):
        raise ValueError(f"invalid fit/future inputs for scene {scene_id}")
    if fit.get("backend") == "swr_backend.wall_bounce_sphere":
        return _render_bouncy_wall_scene(
            scene_id=scene_id,
            raw_video_path=raw_video_path,
            fit=fit,
            future_payload=future,
            width=width,
            height=height,
            expected_answer=expected_answer,
        )
    if manifest_path is None:
        raise ValueError(f"--manifest-dir is required for fit backend {fit.get('backend')}")
    return _render_friction_scene(
        scene_id=scene_id,
        raw_video_path=raw_video_path,
        manifest_path=manifest_path,
        fit_path=fit_path,
        future_path=future_path,
        width=width,
        height=height,
    )


def _summary_items(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("per_scene"), list):
        return [item for item in payload["per_scene"] if isinstance(item, dict)]
    raise ValueError("summary must be a list or contain a per_scene list")


def _fit_path(fit_root: Path, scene_id: int) -> Path:
    candidates = [
        fit_root / f"scene_{scene_id}" / "physics_alignment.json",
        fit_root
        / f"scene_{scene_id}"
        / "world-modeling"
        / "simulatable-world-reconstruction"
        / "fit"
        / "world_reconstruction_fit.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing fit for scene {scene_id}: {candidates}")


def _future_path(fit_root: Path, scene_id: int, item: dict[str, Any]) -> Path:
    if item.get("output_path"):
        candidate = Path(str(item["output_path"]))
        if candidate.exists():
            return candidate
    candidates = [
        fit_root / f"scene_{scene_id}" / "question_0" / "query-conditioned-physical-rollout" / "trajectory.json",
        fit_root
        / f"scene_{scene_id}"
        / "question_0"
        / "query-conditioned-physical-rollout"
        / "query_conditioned_physical_rollout"
        / "simulation"
        / "trajectory.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing future rollout for scene {scene_id}: {candidates}")


def _write_video(path: Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width) * 2, int(height)),
    )
    if not writer.isOpened():
        raise ValueError(f"unable to open writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Physion++ future rollout vs raw full video comparison.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--fit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-output-dir")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    summary = _summary_items(_load_json(Path(args.summary)))
    raw_by_scene = _source_raw_video_by_scene(Path(args.video_manifest))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_output_dir = (
        Path(args.scene_output_dir)
        if args.scene_output_dir
        else output.parent / f"{output.stem}_scenes"
    )
    fit_root = Path(args.fit_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    writer: cv2.VideoWriter | None = None
    written = 0
    fps_out = 30.0
    scene_outputs: list[dict[str, Any]] = []
    for item in summary:
        if not isinstance(item, dict):
            continue
        scene_id = int(item["scene_index"])
        frames, fps = render_scene(
            scene_id=scene_id,
            raw_video_path=raw_by_scene[scene_id],
            manifest_path=(manifest_dir / f"scene_{scene_id}.json" if manifest_dir else None),
            fit_path=_fit_path(fit_root, scene_id),
            future_path=_future_path(fit_root, scene_id, item),
            width=int(args.width),
            height=int(args.height),
            expected_answer=(str(item["expected_answer"]) if item.get("expected_answer") is not None else None),
        )
        fps_out = float(fps)
        scene_output_path = scene_output_dir / f"scene_{scene_id}_raw_vs_swr_future.mp4"
        _write_video(
            scene_output_path,
            frames,
            fps_out,
            int(args.width),
            int(args.height),
        )
        scene_outputs.append(
            {
                "scene_index": scene_id,
                "output": str(scene_output_path),
                "frame_count": len(frames),
            }
        )
        if writer is None:
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (int(args.width) * 2, int(args.height)))
            if not writer.isOpened():
                raise ValueError(f"unable to open writer: {output}")
        for frame in frames:
            writer.write(frame)
            written += 1
    if writer is not None:
        writer.release()
    manifest = {
        "status": "ok",
        "output": str(output),
        "frame_count": int(written),
        "fps": float(fps_out),
        "summary": str(args.summary),
        "scene_outputs": scene_outputs,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
