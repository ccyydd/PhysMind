from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_uniform_frames(video_path: Path, output_dir: Path, num_frames: int) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_count = max(1, num_frames)
    indices = [
        round(i * (frame_count - 1) / max(1, target_count - 1))
        for i in range(target_count)
    ]
    paths = []
    for output_index, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue
        path = output_dir / f"frame_{output_index:04d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    capture.release()
    if not paths:
        raise ValueError(f"Failed to extract frames from video: {video_path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    paths = extract_uniform_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        num_frames=args.num_frames,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

