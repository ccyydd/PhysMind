from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POSE_VIDEO_RELATIVE_PATH = Path(
    "world-modeling/pose-estimation-and-tracking/pose_correction/"
    "debug_render_steps/pose_corrected/world_reconstruction_debug_camera.mp4"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_video(run_dir: Path, scene_id: int) -> Path | None:
    matches = sorted(
        (run_dir / "physion_pp_cue_videos").glob(
            f"{scene_id:06d}_*_cue_trimmed.mp4"
        )
    )
    return matches[0] if len(matches) == 1 else None


def _collect_candidates(run_dirs: list[Path]) -> dict[int, dict[str, Path]]:
    candidates: dict[int, dict[str, Path]] = {}
    for run_dir in run_dirs:
        artifacts_dir = run_dir / "artifacts"
        for scene_dir in sorted(artifacts_dir.glob("scene_*")):
            try:
                scene_id = int(scene_dir.name.removeprefix("scene_"))
            except ValueError:
                continue
            pose_video = scene_dir / POSE_VIDEO_RELATIVE_PATH
            source_video = _source_video(run_dir, scene_id)
            if pose_video.exists() and source_video is not None:
                candidates[scene_id] = {
                    "run_dir": run_dir,
                    "source_video": source_video,
                    "pose_video": pose_video,
                }
    return candidates


def _draw_label(
    image: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int],
) -> None:
    origin = (8, 20)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )


def _open_writer(path: Path, *, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise ValueError(f"unable to open video writer: {path}")
    return writer


def _title_card(*, scene_id: int, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    text = f"scene {scene_id}"
    (text_width, text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        2,
    )
    cv2.putText(
        image,
        text,
        ((width - text_width) // 2, (height + text_height) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


def _render_scene(
    *,
    scene_id: int,
    source_video: Path,
    pose_video: Path,
    output_path: Path,
    combined_writer: cv2.VideoWriter,
    width: int,
    height: int,
    fps: float,
) -> int:
    source_capture = cv2.VideoCapture(str(source_video))
    pose_capture = cv2.VideoCapture(str(pose_video))
    if not source_capture.isOpened():
        raise ValueError(f"unable to open source video: {source_video}")
    if not pose_capture.isOpened():
        raise ValueError(f"unable to open pose video: {pose_video}")
    scene_writer = _open_writer(
        output_path,
        fps=fps,
        width=width * 2,
        height=height,
    )
    frame_index = 0
    while True:
        source_ok, source_frame = source_capture.read()
        pose_ok, pose_frame = pose_capture.read()
        if not source_ok or not pose_ok:
            break
        source_frame = cv2.resize(
            source_frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        pose_frame = cv2.resize(
            pose_frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        _draw_label(
            source_frame,
            f"original | scene {scene_id} | frame {frame_index}",
            color=(255, 255, 255),
        )
        _draw_label(
            pose_frame,
            f"pose corrected | scene {scene_id} | frame {frame_index}",
            color=(255, 255, 255),
        )
        comparison = np.concatenate([source_frame, pose_frame], axis=1)
        scene_writer.write(comparison)
        combined_writer.write(comparison)
        frame_index += 1
    scene_writer.release()
    source_capture.release()
    pose_capture.release()
    return frame_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render reproducibly sampled Physion++ original vs pose-corrected videos."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-frames", type=int, default=15)
    args = parser.parse_args()

    run_dirs = [Path(value).resolve() for value in args.run_dir]
    candidates = _collect_candidates(run_dirs)
    if len(candidates) < int(args.sample_count):
        raise ValueError(
            f"requested {args.sample_count} scenes, but only {len(candidates)} have both videos"
        )
    selected_ids = random.Random(int(args.seed)).sample(
        sorted(candidates),
        int(args.sample_count),
    )
    output_dir = Path(args.output_dir).resolve()
    clips_dir = output_dir / "clips"
    combined_path = output_dir / "physionpp_pose_comparison_random10.mp4"
    combined_writer = _open_writer(
        combined_path,
        fps=float(args.fps),
        width=int(args.width) * 2,
        height=int(args.height),
    )
    results = []
    total_frames = 0
    for scene_id in selected_ids:
        for _ in range(max(int(args.transition_frames), 0)):
            combined_writer.write(
                _title_card(
                    scene_id=scene_id,
                    width=int(args.width) * 2,
                    height=int(args.height),
                )
            )
            total_frames += 1
        candidate = candidates[scene_id]
        clip_path = clips_dir / f"scene_{scene_id}_original_vs_pose_corrected.mp4"
        frame_count = _render_scene(
            scene_id=scene_id,
            source_video=candidate["source_video"],
            pose_video=candidate["pose_video"],
            output_path=clip_path,
            combined_writer=combined_writer,
            width=int(args.width),
            height=int(args.height),
            fps=float(args.fps),
        )
        total_frames += frame_count
        results.append(
            {
                "scene_index": scene_id,
                "run_dir": str(candidate["run_dir"]),
                "source_video": str(candidate["source_video"]),
                "pose_corrected_video": str(candidate["pose_video"]),
                "comparison_video": str(clip_path),
                "frame_count": frame_count,
            }
        )
    combined_writer.release()
    manifest = {
        "status": "ok",
        "seed": int(args.seed),
        "candidate_count": len(candidates),
        "sample_count": len(selected_ids),
        "selected_scene_ids": selected_ids,
        "output": str(combined_path),
        "fps": float(args.fps),
        "frame_count": total_frames,
        "scenes": results,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
