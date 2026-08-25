from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    value = np.asarray(mask).astype(bool)
    if value.ndim > 2:
        value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {value.shape}")
    height, width = image.shape[:2]
    if value.shape == (height, width):
        return value
    return cv2.resize(value.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read frame {frame_index} from video: {video_path}")
    return frame


def _safe_name(value: str) -> str:
    safe = []
    for char in str(value):
        safe.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(safe).strip("_") or "track"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = _mask_for_image(mask, image)
    overlay = image.copy()
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
    return overlay


def _masks_touch(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape or a.ndim != 2:
        return False
    if not a.any() or not b.any():
        return False
    dilated = cv2.dilate(a.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return bool(np.logical_and(dilated, b).any())


def _mask_boundary_diagnostic(mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(mask).astype(bool)
    if value.ndim > 2:
        value = np.squeeze(value)
    if value.ndim != 2 or not value.any():
        return {"touching_boundary": True, "boundary_sides": ["invalid_or_empty"]}
    sides = []
    if value[0, :].any():
        sides.append("top")
    if value[-1, :].any():
        sides.append("bottom")
    if value[:, 0].any():
        sides.append("left")
    if value[:, -1].any():
        sides.append("right")
    return {"touching_boundary": bool(sides), "boundary_sides": sides}


def _choose_representative_records(
    records_by_track: dict[str, list[dict[str, Any]]],
    masks,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    records_by_frame: dict[int, list[tuple[str, dict[str, Any], np.ndarray]]] = {}
    boundary: dict[tuple[str, int], dict[str, Any]] = {}
    for track_id, records in records_by_track.items():
        for record in records:
            mask_key = str(record.get("mask_key") or "")
            if not mask_key or mask_key not in masks:
                continue
            try:
                frame_index = int(record.get("frame_index"))
            except (TypeError, ValueError):
                continue
            mask = masks[mask_key].astype(bool)
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            if mask.ndim != 2:
                continue
            boundary[(track_id, frame_index)] = _mask_boundary_diagnostic(mask)
            records_by_frame.setdefault(frame_index, []).append((track_id, record, mask))

    touching: dict[tuple[str, int], bool] = {}
    for frame_records in records_by_frame.values():
        for index, (track_id, record, mask) in enumerate(frame_records):
            is_touching = False
            for other_index, (other_track_id, _other_record, other_mask) in enumerate(frame_records):
                if other_index == index or other_track_id == track_id:
                    continue
                if _masks_touch(mask, other_mask):
                    is_touching = True
                    break
            touching[(track_id, int(record.get("frame_index")))] = is_touching

    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for track_id, records in records_by_track.items():
        valid = [record for record in records if str(record.get("mask_key") or "") in masks]
        preferred = [
            record
            for record in valid
            if not touching.get((track_id, int(record.get("frame_index", -1))), False)
            and not boundary.get((track_id, int(record.get("frame_index", -1))), {}).get("touching_boundary", True)
        ]
        pool = preferred if preferred else valid
        if not pool:
            continue
        record = max(pool, key=lambda item: int(item.get("area") or 0))
        frame_index = int(record.get("frame_index", -1))
        selected_boundary = boundary.get((track_id, frame_index), {})
        selected_touching_other = touching.get((track_id, frame_index), False)
        selected[track_id] = (
            record,
            {
                "selection_rule": (
                    "largest_area_non_touching_other_and_boundary_track_mask"
                    if preferred
                    else "largest_area_all_track_masks_fallback"
                ),
                "candidate_count": len(valid),
                "eligible_candidate_count": len(preferred),
                "fallback_used": not bool(preferred),
                "score": "area",
                "selected_touching_other_mask": bool(selected_touching_other),
                "selected_touching_boundary": bool(selected_boundary.get("touching_boundary", True)),
                "selected_boundary_sides": selected_boundary.get("boundary_sides", []),
            },
        )
    return selected


def export_label_inputs(
    *,
    sam3_video_tracks: Path,
    video: Path,
    output: Path,
    image_dir: Path | None,
    write_selected_frame: bool,
) -> None:
    payload = json.loads(sam3_video_tracks.read_text(encoding="utf-8"))
    mask_sidecar = payload.get("mask_sidecar")
    if not mask_sidecar:
        raise ValueError(f"SAM3 video tracks file has no mask_sidecar: {sam3_video_tracks}")
    mask_sidecar_path = Path(mask_sidecar)
    if not mask_sidecar_path.exists():
        fallback = sam3_video_tracks.with_suffix(".npz")
        if fallback.exists():
            mask_sidecar_path = fallback
        else:
            raise FileNotFoundError(f"Mask sidecar not found: {mask_sidecar}")

    records_by_track = payload.get("tracks_by_object")
    if not isinstance(records_by_track, dict):
        records_by_track = {}
        for record in payload.get("tracks", []):
            if isinstance(record, dict):
                records_by_track.setdefault(str(record.get("object_id")), []).append(record)

    image_root = image_dir or output.with_suffix("")
    image_root.mkdir(parents=True, exist_ok=True)

    masks = np.load(mask_sidecar_path)
    representatives = _choose_representative_records(records_by_track, masks)
    tracks = []
    for track_id, raw_records in sorted(records_by_track.items()):
        records = [record for record in raw_records if isinstance(record, dict)]
        records.sort(key=lambda item: int(item.get("frame_index", -1)))
        track_dir = image_root / _safe_name(track_id)
        track_dir.mkdir(parents=True, exist_ok=True)

        frame_items = []
        representative_record, representative_selection = representatives.get(track_id, ({}, {}))
        representative_image_path = None
        representative_overlay_path = None
        selected_frame_image_path = None
        if representative_record:
            record = representative_record
            mask_key = str(record.get("mask_key") or "")
            if mask_key and mask_key in masks:
                frame_index = int(record["frame_index"])
                frame = _read_frame(video, frame_index)
                mask = masks[mask_key].astype(bool)
                overlay = _overlay_mask(frame, mask)

                representative_overlay_path = track_dir / "representative_overlay.jpg"
                representative_image_path = representative_overlay_path
                cv2.imwrite(str(representative_overlay_path), overlay)
                selected_frame_image_fingerprint = None
                if write_selected_frame:
                    selected_frame_image_path = track_dir / "selected_frame.png"
                    cv2.imwrite(str(selected_frame_image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    selected_frame_image_fingerprint = _file_fingerprint(selected_frame_image_path)
                frame_items.append(
                    {
                        "frame_index": frame_index,
                        "sam_object_id": record.get("sam_object_id"),
                        "score": record.get("score"),
                        "area": record.get("area"),
                        "bbox_xyxy": record.get("bbox_xyxy"),
                        "centroid_xy": record.get("centroid_xy"),
                        "mask_key": mask_key,
                        "touching_boundary": representative_selection.get("selected_touching_boundary"),
                        "boundary_sides": representative_selection.get("selected_boundary_sides"),
                        "touching_other_mask": representative_selection.get("selected_touching_other_mask"),
                        "representative_image_path": str(representative_image_path),
                        "representative_overlay_path": str(representative_overlay_path),
                        "selected_frame_image": str(selected_frame_image_path) if selected_frame_image_path else None,
                        "selected_frame_image_format": "png" if selected_frame_image_path else None,
                        "selected_frame_image_fingerprint": selected_frame_image_fingerprint,
                        "selection": representative_selection,
                    }
                )

        first_record = records[0] if records else {}
        tracks.append(
            {
                "track_id": str(track_id),
                "concept_id": first_record.get("concept_id"),
                "prompt": first_record.get("prompt"),
                "expected_object_ids": first_record.get("expected_object_ids", []),
                "frame_count": len(records),
                "first_frame_index": records[0].get("frame_index") if records else None,
                "last_frame_index": records[-1].get("frame_index") if records else None,
                "selected_frame_count": len(frame_items),
                "representative_frames": frame_items,
                "representative_frame": frame_items[0] if frame_items else None,
                "representative_image_path": str(representative_image_path) if representative_image_path else None,
                "representative_overlay_path": str(representative_overlay_path) if representative_overlay_path else None,
                "selected_frame_image": str(selected_frame_image_path) if selected_frame_image_path else None,
                "selected_frame_image_format": "png" if selected_frame_image_path else None,
                "selected_frame_image_fingerprint": _file_fingerprint(selected_frame_image_path)
                if selected_frame_image_path
                else None,
                "representative_selection": representative_selection,
            }
        )
    masks.close()

    result = {
        "tool": "sam3_video_track_label_inputs",
        "status": "ok",
        "sam3_video_tracks": str(sam3_video_tracks),
        "video": str(video),
        "mask_sidecar": str(mask_sidecar_path),
        "image_dir": str(image_root),
        "representative_image_source": "full_frame_red_contour_overlay",
        "track_count": len(tracks),
        "tracks": tracks,
        "note": "VLM-labeling inputs for anonymous SAM3 video tracks. Each track uses one representative full-frame image with the tracked object outlined in red.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-video-tracks", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--write-selected-frame", action="store_true")
    args = parser.parse_args()
    export_label_inputs(
        sam3_video_tracks=Path(args.sam3_video_tracks),
        video=Path(args.video),
        output=Path(args.output),
        image_dir=Path(args.image_dir) if args.image_dir else None,
        write_selected_frame=bool(args.write_selected_frame),
    )


if __name__ == "__main__":
    main()
