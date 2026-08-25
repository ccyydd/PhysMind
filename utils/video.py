from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2


_VIDEO_DATA_URL_CACHE: Dict[Tuple[str, str], str] = {}


@dataclass
class VideoMetadata:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class SampledFrame:
    frame_index: int
    image_bytes: bytes


def read_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    metadata = VideoMetadata(
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return metadata


def uniform_frame_indices(frame_count: int, num_frames: int = 8) -> List[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    target_count = max(1, num_frames)
    if frame_count == 1:
        return [0] * target_count
    return [
        round(i * (frame_count - 1) / max(1, target_count - 1))
        for i in range(target_count)
    ]


def sample_video_frames_at_indices(
    video_path: Union[str, Path],
    frame_indices: Sequence[int],
) -> List[SampledFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")

    encoded_frames: List[SampledFrame] = []
    for raw_index in frame_indices:
        index = int(raw_index)
        if not 0 <= index < frame_count:
            capture.release()
            raise ValueError(
                f"Frame index {index} is outside [0, {frame_count - 1}] for {video_path}"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success:
            continue
        success, encoded = cv2.imencode(".jpg", frame)
        if success:
            encoded_frames.append(SampledFrame(frame_index=index, image_bytes=encoded.tobytes()))

    capture.release()
    if not encoded_frames:
        raise ValueError(f"Failed to sample frames from video: {video_path}")
    return encoded_frames


def sample_uniform_frames_with_indices(video_path: Union[str, Path], num_frames: int = 8) -> List[SampledFrame]:
    metadata = read_video_metadata(video_path)
    indices = uniform_frame_indices(frame_count=metadata.frame_count, num_frames=num_frames)
    return sample_video_frames_at_indices(video_path=video_path, frame_indices=indices)


def sample_uniform_frames(video_path: Union[str, Path], num_frames: int = 8) -> List[bytes]:
    return [
        item.image_bytes
        for item in sample_uniform_frames_with_indices(video_path=video_path, num_frames=num_frames)
    ]


def encode_image_bytes_as_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def encode_video_as_data_url(video_path: Union[str, Path], mime_type: str = "video/mp4") -> str:
    video_bytes = Path(video_path).read_bytes()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_cached_video_data_url(video_path: Union[str, Path], mime_type: str = "video/mp4") -> tuple[str, bool]:
    resolved_path = str(Path(video_path).resolve())
    cache_key = (resolved_path, mime_type)
    cached = _VIDEO_DATA_URL_CACHE.get(cache_key)
    if cached is not None:
        return cached, True

    data_url = encode_video_as_data_url(video_path=resolved_path, mime_type=mime_type)
    _VIDEO_DATA_URL_CACHE[cache_key] = data_url
    return data_url, False
