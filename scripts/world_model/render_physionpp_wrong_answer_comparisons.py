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

from scripts.world_model import render_physionpp_future_rollout_comparison as render_common
from scripts.world_model import run_physionpp_friction_sphere_sysid as friction_swr
from scripts.world_model import swr_sysid_common as sysid_common


BACKEND_SCENARIOS = {
    "swr_backend.surface_friction_sphere": "friction_platform_pp",
    "swr_backend.wall_bounce_sphere": "bouncy_wall_pp",
    "swr_backend.platform_bounce_sphere": "bouncy_platform_pp",
    "swr_backend.collision_friction_spheres": "friction_collision_pp",
    "swr_backend.collision_mass_spheres": "mass_collision_pp",
}

SCENARIO_FILENAMES = {
    "friction_platform_pp": "friction_platform_wrong_cases.mp4",
    "bouncy_wall_pp": "bouncy_wall_wrong_cases.mp4",
    "bouncy_platform_pp": "bouncy_platform_wrong_cases.mp4",
    "friction_collision_pp": "friction_collision_wrong_cases.mp4",
    "mass_collision_pp": "mass_collision_real_vs_target_vs_fit.mp4",
}

OBJECT_COLORS = [
    (35, 80, 225),
    (40, 170, 65),
    (215, 115, 35),
    (170, 60, 180),
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_position(record: dict[str, Any]) -> np.ndarray:
    value = record.get("position")
    if value is None:
        value = record.get("position_blender_world_m")
    position = np.asarray(value, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError(f"invalid trajectory position: {value}")
    return position


def _trajectory_by_frame(records: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    return {
        int(record["frame_index"]): _record_position(record)
        for record in records
        if isinstance(record, dict) and record.get("frame_index") is not None
    }


def _merge_trajectories(
    destination: dict[str, dict[int, np.ndarray]],
    source: dict[str, Any],
    object_ids: list[str],
) -> None:
    for object_id in object_ids:
        records = source.get(object_id)
        if not isinstance(records, list):
            continue
        destination.setdefault(object_id, {}).update(_trajectory_by_frame(records))


def _plane_hulls_for_fit(
    *,
    manifest: dict[str, Any],
    fit: dict[str, Any],
    width: int,
    height: int,
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    backend = str(fit.get("backend") or "")
    if backend == "swr_backend.surface_friction_sphere":
        hulls = render_common._plane_hulls(
            manifest=manifest,
            fit=fit,
            width=width,
            height=height,
        )
        return [(0, 10**9, hulls)]
    if backend == "swr_backend.platform_bounce_sphere":
        physics = fit.get("physics_rollout") if isinstance(fit.get("physics_rollout"), dict) else {}
        hulls = render_common._project_plane_geometry(
            planes=[item for item in physics.get("contact_plane_geometry") or [] if isinstance(item, dict)],
            camera_intrinsic=manifest["fixed_intrinsics"],
            wall_object_id="",
            patient_object_id=None,
        )
        return [(0, 10**9, hulls)]
    if backend == "swr_backend.wall_bounce_sphere":
        ranges = []
        for segment in fit.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            frame_range = segment.get("frame_range") or []
            if len(frame_range) != 2:
                continue
            physics = segment.get("physics_rollout") if isinstance(segment.get("physics_rollout"), dict) else {}
            hulls = render_common._project_plane_geometry(
                planes=[item for item in physics.get("contact_plane_geometry") or [] if isinstance(item, dict)],
                camera_intrinsic=manifest["fixed_intrinsics"],
                wall_object_id=str(segment.get("wall_object_id") or ""),
                patient_object_id=(
                    str(segment["patient_object_id"])
                    if segment.get("patient_object_id") is not None
                    else None
                ),
            )
            ranges.append((int(frame_range[0]), int(frame_range[1]), hulls))
        return ranges
    return []


def _scene_trajectories(
    *,
    fit: dict[str, Any],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]], dict[str, float]]:
    target: dict[str, dict[int, np.ndarray]] = {}
    simulated: dict[str, dict[int, np.ndarray]] = {}
    radii: dict[str, float] = {}
    backend = str(fit.get("backend") or "")

    if backend in {"swr_backend.surface_friction_sphere", "swr_backend.platform_bounce_sphere"}:
        physics = fit.get("physics_rollout") if isinstance(fit.get("physics_rollout"), dict) else {}
        radius_payload = physics.get("sphere_radius_3d_by_object") or {}
        object_ids = sorted(str(object_id) for object_id in radius_payload)
        _merge_trajectories(target, fit.get("target_trajectories") or {}, object_ids)
        _merge_trajectories(simulated, physics.get("simulated_trajectories") or {}, object_ids)
        radii.update({str(object_id): float(radius) for object_id, radius in radius_payload.items()})
        return target, simulated, radii

    for segment in fit.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        physics = segment.get("physics_rollout") if isinstance(segment.get("physics_rollout"), dict) else {}
        if backend == "swr_backend.wall_bounce_sphere":
            object_ids = [str(segment["agent_object_id"])]
            radius = float(
                ((fit.get("joint_alignment_optimization") or {}).get("best_shared_parameters") or {}).get(
                    "optimized_radius_m"
                )
            )
            radii[object_ids[0]] = radius
        else:
            radius_payload = physics.get("sphere_radius_m_by_object") or {}
            object_ids = sorted(str(object_id) for object_id in radius_payload)
            radii.update({str(object_id): float(radius) for object_id, radius in radius_payload.items()})
        _merge_trajectories(target, segment.get("target_trajectories") or {}, object_ids)
        _merge_trajectories(simulated, physics.get("simulated_trajectories") or {}, object_ids)
    return target, simulated, radii


def _source_video(run_dir: Path, scene_id: int) -> Path:
    matches = sorted((run_dir / "physion_pp_cue_videos").glob(f"{scene_id:06d}_*_cue_trimmed.mp4"))
    if len(matches) != 1:
        raise ValueError(f"expected one trimmed source video for scene {scene_id}, found {len(matches)}")
    return matches[0]


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    outline = (0, 0, 0) if float(np.mean(color)) > 128.0 else (255, 255, 255)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.43, outline, 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def _active_hulls(
    ranges: list[tuple[int, int, list[dict[str, Any]]]],
    frame_index: int,
) -> list[dict[str, Any]]:
    for start, end, hulls in ranges:
        if start <= frame_index <= end:
            return hulls
    return []


def _draw_trajectory_panel(
    *,
    frame_index: int,
    trajectories: dict[str, dict[int, np.ndarray]],
    radii: dict[str, float],
    colors: dict[str, tuple[int, int, int]],
    histories: dict[str, list[tuple[int, int]]],
    hulls: list[dict[str, Any]],
    camera_intrinsic: list[list[float]],
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    image = render_common._draw_hulls(np.full((height, width, 3), 246, dtype=np.uint8), hulls)
    for object_id, by_frame in trajectories.items():
        position = by_frame.get(frame_index)
        if position is None:
            continue
        projected = friction_swr._project_blender_to_pixel(position, camera_intrinsic)
        if projected is None:
            continue
        pixel = (int(round(projected[0])), int(round(projected[1])))
        histories.setdefault(object_id, []).append(pixel)
        history = np.asarray(histories[object_id], dtype=np.int32)
        if len(history) >= 2:
            cv2.polylines(image, [history], False, colors[object_id], 2, cv2.LINE_AA)
        render_common._project_sphere(
            image=image,
            position=position.astype(float).tolist(),
            radius=float(radii[object_id]),
            camera_intrinsic=camera_intrinsic,
            color=colors[object_id],
        )
    _draw_text(image, title, (8, 18), (25, 25, 25))
    return image


def _title_card(
    *,
    scene_id: int,
    prediction: str,
    expected: str,
    rmse: float | None,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.full((height, width * 3, 3), 24, dtype=np.uint8)
    rmse_text = "-" if rmse is None else f"{rmse:.4f} m"
    lines = [
        f"scene {scene_id}",
        f"prediction={prediction}  expected={expected}",
        f"dynamic RMSE={rmse_text}",
    ]
    for line_index, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        origin = ((width * 3 - size[0]) // 2, height // 2 - 30 + line_index * 30)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def _render_scene(
    *,
    item: dict[str, Any],
    writer: cv2.VideoWriter,
    width: int,
    height: int,
    transition_frames: int,
) -> int:
    scene_id = int(item["scene_index"])
    run_dir = Path(str(item["run_dir"]))
    fit_path = Path(str(item["fit_path"]))
    manifest_path = fit_path.parent / "world_reconstruction_fit_manifest.json"
    fit = _load_json(fit_path)
    manifest = _load_json(manifest_path)
    backend = str(fit.get("backend") or "")
    if BACKEND_SCENARIOS.get(backend) != item.get("scenario"):
        raise ValueError(f"backend/scenario mismatch for scene {scene_id}: {backend}")
    camera_intrinsic = sysid_common._camera_intrinsics_from_manifest(manifest)
    if camera_intrinsic is None:
        raise ValueError(f"missing camera intrinsics for scene {scene_id}")
    target, simulated, radii = _scene_trajectories(fit=fit)
    if not target or not simulated or not radii:
        raise ValueError(f"missing target or simulated trajectory for scene {scene_id}")
    hull_ranges = _plane_hulls_for_fit(
        manifest=manifest,
        fit=fit,
        width=width,
        height=height,
    )
    object_ids = sorted(radii)
    colors = {object_id: OBJECT_COLORS[index % len(OBJECT_COLORS)] for index, object_id in enumerate(object_ids)}
    max_trajectory_frame = max(
        frame_index
        for trajectories in (target, simulated)
        for by_frame in trajectories.values()
        for frame_index in by_frame
    )

    for _ in range(transition_frames):
        writer.write(
            _title_card(
                scene_id=scene_id,
                prediction=str(item.get("prediction")),
                expected=str(item.get("expected")),
                rmse=(float(item["dynamic_object_rmse_m"]) if item.get("dynamic_object_rmse_m") is not None else None),
                width=width,
                height=height,
            )
        )

    capture = cv2.VideoCapture(str(_source_video(run_dir, scene_id)))
    if not capture.isOpened():
        raise ValueError(f"unable to open source video for scene {scene_id}")
    raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = max(raw_frame_count, max_trajectory_frame + 1)
    last_raw = np.zeros((height, width, 3), dtype=np.uint8)
    target_histories: dict[str, list[tuple[int, int]]] = {}
    simulated_histories: dict[str, list[tuple[int, int]]] = {}
    for frame_index in range(frame_count):
        ok, raw = capture.read()
        if ok:
            last_raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
        else:
            last_raw = cv2.addWeighted(last_raw, 0.45, np.zeros_like(last_raw), 0.55, 0.0)
        raw_panel = last_raw.copy()
        _draw_text(raw_panel, f"Original | scene {scene_id} | frame {frame_index}", (8, 18), (245, 245, 245))
        hulls = _active_hulls(hull_ranges, frame_index)
        target_panel = _draw_trajectory_panel(
            frame_index=frame_index,
            trajectories=target,
            radii=radii,
            colors=colors,
            histories=target_histories,
            hulls=hulls,
            camera_intrinsic=camera_intrinsic,
            width=width,
            height=height,
            title="Target trajectory",
        )
        simulated_panel = _draw_trajectory_panel(
            frame_index=frame_index,
            trajectories=simulated,
            radii=radii,
            colors=colors,
            histories=simulated_histories,
            hulls=hulls,
            camera_intrinsic=camera_intrinsic,
            width=width,
            height=height,
            title="SWR inverted rollout",
        )
        writer.write(np.concatenate([raw_panel, target_panel, simulated_panel], axis=1))
    capture.release()
    return frame_count + transition_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Physion++ wrong-answer original/target/SWR comparisons.")
    parser.add_argument("--evaluation-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_FILENAMES))
    parser.add_argument("--scene-id", action="append", type=int)
    parser.add_argument("--all-completed", action="store_true")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-frames", type=int, default=15)
    args = parser.parse_args()

    summary = _load_json(Path(args.evaluation_summary))
    selected_scene_ids = set(args.scene_id or [])
    selected = [
        item
        for item in summary.get("scenes") or []
        if isinstance(item, dict)
        and item.get("status") == "ok"
        and (args.all_completed or not bool(item.get("is_correct")))
        and (args.scenario is None or item.get("scenario") == args.scenario)
        and (not selected_scene_ids or int(item["scene_index"]) in selected_scene_ids)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_results: list[dict[str, Any]] = []

    scenarios = [args.scenario] if args.scenario else list(SCENARIO_FILENAMES)
    for scenario in scenarios:
        items = sorted(
            (item for item in selected if item.get("scenario") == scenario),
            key=lambda item: int(item["scene_index"]),
        )
        if not items:
            continue
        output_path = output_dir / SCENARIO_FILENAMES[scenario]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (int(args.width) * 3, int(args.height)),
        )
        if not writer.isOpened():
            raise ValueError(f"unable to open output video: {output_path}")
        scene_results = []
        for item in items:
            scene_id = int(item["scene_index"])
            try:
                frame_count = _render_scene(
                    item=item,
                    writer=writer,
                    width=int(args.width),
                    height=int(args.height),
                    transition_frames=int(args.transition_frames),
                )
                scene_results.append({"scene_index": scene_id, "status": "ok", "frame_count": frame_count})
                print(f"[render] scenario={scenario} scene={scene_id} status=ok frames={frame_count}", flush=True)
            except Exception as exc:
                scene_results.append({"scene_index": scene_id, "status": "error", "error_message": str(exc)})
                print(f"[render] scenario={scenario} scene={scene_id} status=error error={exc}", flush=True)
        writer.release()
        render_results.append(
            {
                "scenario": scenario,
                "output": str(output_path.resolve()),
                "requested_scene_count": len(items),
                "completed_scene_count": sum(item["status"] == "ok" for item in scene_results),
                "scenes": scene_results,
            }
        )

    _write_json(
        output_dir / "render_summary.json",
        {
            "status": "ok" if all(
                item["requested_scene_count"] == item["completed_scene_count"] for item in render_results
            ) else "partial",
            "source_evaluation_summary": str(Path(args.evaluation_summary).resolve()),
            "selected_scene_count": len(selected),
            "outputs": render_results,
        },
    )


if __name__ == "__main__":
    main()
