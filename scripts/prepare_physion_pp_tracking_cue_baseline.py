from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from pycocotools import mask as rle_mask

import benchmark.physion_pp as physion_pp


DEFAULT_OUTPUT_ROOT = Path("data/physion_pp_tracking_cue_baseline_v1")
EXPECTED_SCENE_COUNTS = {
    "mass_collision_pp": 64,
    "friction_collision_pp": 64,
    "friction_platform_pp": 64,
    "bouncy_platform_pp": 96,
    "bouncy_wall_pp": 96,
}
REVIEW_SCENARIO_LABELS = {
    "mass_collision_pp": "mass collision",
    "friction_collision_pp": "friction collision",
    "friction_platform_pp": "friction platform",
    "bouncy_platform_pp": "bouncy platform",
    "bouncy_wall_pp": "bouncy wall",
}
REVIEW_TILE_SIZE = 128
REVIEW_COLUMNS = 8
REVIEW_PANELS_PER_SCENE = 4
REVIEW_LABEL_HEIGHT = 22


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the 384-scene Physion++ baseline variant whose red/yellow target "
            "cues follow the visible target masks until the original cue starts blinking."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=physion_pp.default_physion_pp_root(),
        help="Source Physion++ root containing testdata_v1 and the original cue renders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Independent baseline dataset root to create.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N baseline scenes (smoke tests only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render videos that already exist.",
    )
    parser.add_argument(
        "--no-review-grid",
        action="store_true",
        help="Skip the combined human-review grid.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many scenes; set to 0 to disable.",
    )
    return parser


def _load_original_records(dataset_root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    cue_dir = dataset_root / physion_pp.PHYSION_PP_CUE_VIDEO_DIRNAME
    records_path = cue_dir / "cue_render_records.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing original Physion++ cue records: {records_path}. "
            "Render the standard cue videos first."
        )
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = {
        str(record["stimulus_id"]): record
        for record in payload
        if isinstance(record, dict) and record.get("stimulus_id")
    }
    return cue_dir, records


def _select_scenes(dataset_root: Path, limit: int | None) -> list[physion_pp.PhysionPPScene]:
    scenes = [
        scene
        for scene in physion_pp.load_test_scenes(dataset_root=dataset_root)
        if scene.scenario in EXPECTED_SCENE_COUNTS
    ]
    counts = Counter(scene.scenario for scene in scenes)
    if dict(counts) != EXPECTED_SCENE_COUNTS:
        raise ValueError(
            f"Unexpected Physion++ baseline scene distribution: {dict(counts)}; "
            f"expected {EXPECTED_SCENE_COUNTS}"
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        return scenes[:limit]
    return scenes


def _ensure_dataset_overlay(dataset_root: Path, output_root: Path) -> Path:
    if output_root.resolve() == dataset_root.resolve():
        raise ValueError(
            "--output-root must differ from --dataset-root so the baseline variant cannot "
            "overwrite the standard Physion++ cue videos"
        )
    source_dataset_dir = physion_pp.resolve_physion_pp_dataset_dir(dataset_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    link_path = output_root / physion_pp.TESTDATA_DIRNAME
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_dir() or link_path.resolve() != source_dataset_dir:
            raise ValueError(
                f"Existing {link_path} does not resolve to source test data {source_dataset_dir}"
            )
    else:
        link_path.symlink_to(source_dataset_dir, target_is_directory=True)
    cue_dir = output_root / physion_pp.PHYSION_PP_CUE_VIDEO_DIRNAME
    cue_dir.mkdir(parents=True, exist_ok=True)
    return cue_dir


def _load_target_indices(scene: physion_pp.PhysionPPScene) -> tuple[int, int]:
    with scene.pkl_path.open("rb") as file:
        static = pickle.load(file)["static"]
    object_ids = [int(value) for value in static["object_ids"]]
    target_index = object_ids.index(int(static["target_id"]))
    zone_index = object_ids.index(int(static["zone_id"]))
    if target_index == zone_index:
        raise ValueError(f"Target and zone resolve to the same instance for {scene.stimulus_id}")
    return target_index, zone_index


def _decode_target_masks(
    frame_entries: Iterable[dict[str, Any]],
    target_index: int,
    zone_index: int,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for entry in frame_entries:
        index = int(entry["idx"])
        if index in {target_index, zone_index}:
            masks[index] = rle_mask.decode(entry).astype(bool)
    empty = np.zeros(frame_shape, dtype=bool)
    return masks.get(target_index, empty), masks.get(zone_index, empty)


def _apply_cues(
    frame: np.ndarray,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> np.ndarray:
    result = frame.copy()
    alpha = physion_pp.CUE_OVERLAY_ALPHA
    if red_mask.any():
        red = np.array([0, 0, 255], dtype=np.float32)
        result[red_mask] = np.rint(frame[red_mask] * (1.0 - alpha) + red * alpha).astype(np.uint8)
    if yellow_mask.any():
        yellow = np.array([0, 255, 255], dtype=np.float32)
        result[yellow_mask] = np.rint(
            frame[yellow_mask] * (1.0 - alpha) + yellow * alpha
        ).astype(np.uint8)
    return result


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if not union:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def _audit_role_binding(
    scene: physion_pp.PhysionPPScene,
    original_record: dict[str, Any],
    frame_entries: dict[str, list[dict[str, Any]]],
    target_index: int,
    zone_index: int,
) -> dict[str, Any]:
    if original_record.get("cue_source") != "map":
        return {
            "source": "official_pkl_target_zone_ids",
            "red_target_iou": None,
            "yellow_zone_iou": None,
        }
    cue_frame_index = int(original_record["cue_frame_index"])
    red_target, yellow_zone = _decode_target_masks(
        frame_entries.get(f"{cue_frame_index:04d}", []),
        target_index,
        zone_index,
        frame_shape=(256, 256),
    )
    map_red, map_yellow = physion_pp._load_cue_masks(scene)
    red_iou = _mask_iou(map_red, red_target)
    yellow_iou = _mask_iou(map_yellow, yellow_zone)
    if red_iou < physion_pp.CUE_ALIGNMENT_MIN_IOU or yellow_iou < physion_pp.CUE_ALIGNMENT_MIN_IOU:
        raise ValueError(
            f"Official red/yellow map does not match target_id/zone_id for {scene.stimulus_id}: "
            f"red_target_iou={red_iou:.4f}, yellow_zone_iou={yellow_iou:.4f}"
        )
    return {
        "source": "official_map_cross_checked_with_pkl_target_zone_ids",
        "red_target_iou": round(red_iou, 4),
        "yellow_zone_iou": round(yellow_iou, 4),
    }


def _read_all_frames(video_path: Path) -> tuple[float, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or physion_pp.CUE_VIDEO_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return fps, frames


def _detect_original_flash_start(original_cue_path: Path) -> int:
    _, frames = _read_all_frames(original_cue_path)
    flash_start, _, _ = physion_pp.detect_cue_flash_start(frames)
    if flash_start is None:
        raise ValueError(f"Could not detect the original cue flash in {original_cue_path}")
    return int(flash_start)


def _render_scene(
    scene: physion_pp.PhysionPPScene,
    original_record: dict[str, Any],
    original_cue_dir: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    cut_frame = int(original_record["cue_frame_index"])
    original_cue_path = original_cue_dir / str(original_record["video"])
    detected_flash_start = _detect_original_flash_start(original_cue_path)
    if detected_flash_start != cut_frame:
        raise ValueError(
            f"Original cue record/detection mismatch for {scene.stimulus_id}: "
            f"record={cut_frame}, detected={detected_flash_start}"
        )

    frame_entries = json.loads(scene.id_json_path.read_text(encoding="utf-8"))
    target_index, zone_index = _load_target_indices(scene)
    binding_audit = _audit_role_binding(
        scene,
        original_record,
        frame_entries,
        target_index,
        zone_index,
    )

    red_visible_frames = 0
    yellow_visible_frames = 0
    red_cued_pixels = 0
    yellow_cued_pixels = 0
    rendered_frames = 0
    if overwrite or not output_path.exists():
        capture = cv2.VideoCapture(str(scene.video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open raw Physion++ video: {scene.video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or physion_pp.CUE_VIDEO_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (height, width) != (256, 256):
            capture.release()
            raise ValueError(
                f"Unexpected frame size {(width, height)} for {scene.video_path}; expected 256x256"
            )
        partial_path = output_path.with_name(f"{output_path.stem}.part.mp4")
        if partial_path.exists():
            partial_path.unlink()
        writer = cv2.VideoWriter(
            str(partial_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Unable to open output video writer: {partial_path}")
        try:
            for frame_index in range(cut_frame):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"Raw video ended before cut frame {cut_frame}: {scene.video_path}"
                    )
                red_mask, yellow_mask = _decode_target_masks(
                    frame_entries.get(f"{frame_index:04d}", []),
                    target_index,
                    zone_index,
                    frame_shape=(height, width),
                )
                red_pixels = int(red_mask.sum())
                yellow_pixels = int(yellow_mask.sum())
                red_visible_frames += int(red_pixels > 0)
                yellow_visible_frames += int(yellow_pixels > 0)
                red_cued_pixels += red_pixels
                yellow_cued_pixels += yellow_pixels
                writer.write(_apply_cues(frame, red_mask, yellow_mask))
                rendered_frames += 1
        finally:
            capture.release()
            writer.release()
        if rendered_frames != cut_frame:
            partial_path.unlink(missing_ok=True)
            raise ValueError(
                f"Rendered {rendered_frames} frames for {scene.stimulus_id}; expected {cut_frame}"
            )
        partial_path.replace(output_path)
    else:
        for frame_index in range(cut_frame):
            red_mask, yellow_mask = _decode_target_masks(
                frame_entries.get(f"{frame_index:04d}", []),
                target_index,
                zone_index,
                frame_shape=(256, 256),
            )
            red_pixels = int(red_mask.sum())
            yellow_pixels = int(yellow_mask.sum())
            red_visible_frames += int(red_pixels > 0)
            yellow_visible_frames += int(yellow_pixels > 0)
            red_cued_pixels += red_pixels
            yellow_cued_pixels += yellow_pixels

    _, decoded_output = _read_all_frames(output_path)
    if len(decoded_output) != cut_frame:
        raise ValueError(
            f"Decoded output has {len(decoded_output)} frames for {scene.stimulus_id}; "
            f"expected exactly {cut_frame}"
        )

    return {
        "stimulus_id": scene.stimulus_id,
        "scene_index": scene.scene_index,
        "property": scene.property_name,
        "scenario": scene.scenario,
        "video": output_path.name,
        # The standard direct-answer loader adds CUE_REFERENCE_ON_OFFSET to this value.
        # Every frame in this variant already carries the cue, so that derived frame is
        # a valid reference while the actual truncation is recorded explicitly below.
        "cue_frame_index": 0,
        "cue_input_mode": physion_pp.PHYSION_PP_TRACKING_CUE_INPUT_MODE,
        "cue_source": "per_frame_official_instance_masks",
        "cue_alignment_iou": 1.0,
        "source_cue_video": str(original_cue_path),
        "source_raw_video": str(scene.video_path),
        "source_cue_frame_index": cut_frame,
        "detected_original_flash_start": detected_flash_start,
        "first_excluded_frame": cut_frame,
        "rendered_frame_count": cut_frame,
        "red_instance_index": target_index,
        "yellow_instance_index": zone_index,
        "red_visible_frames": red_visible_frames,
        "yellow_visible_frames": yellow_visible_frames,
        "red_fully_occluded_frames": cut_frame - red_visible_frames,
        "yellow_fully_occluded_frames": cut_frame - yellow_visible_frames,
        "red_cued_pixels": red_cued_pixels,
        "yellow_cued_pixels": yellow_cued_pixels,
        "role_binding_audit": binding_audit,
    }


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open review video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read frame {frame_index} from {video_path}")
    return frame


def _review_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, (REVIEW_TILE_SIZE, REVIEW_TILE_SIZE), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((REVIEW_TILE_SIZE + REVIEW_LABEL_HEIGHT, REVIEW_TILE_SIZE, 3), dtype=np.uint8)
    canvas[REVIEW_LABEL_HEIGHT:] = tile
    cv2.putText(
        canvas,
        label,
        (3, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _create_review_grid(
    records: list[dict[str, Any]],
    cue_dir: Path,
    review_path: Path,
) -> None:
    scene_width = REVIEW_TILE_SIZE * REVIEW_PANELS_PER_SCENE
    scene_height = REVIEW_TILE_SIZE + REVIEW_LABEL_HEIGHT
    rows = (len(records) + REVIEW_COLUMNS - 1) // REVIEW_COLUMNS
    grid = np.full((rows * scene_height, REVIEW_COLUMNS * scene_width, 3), 24, dtype=np.uint8)
    for index, record in enumerate(records):
        cut = int(record["first_excluded_frame"])
        output_path = cue_dir / str(record["video"])
        source_cue_path = Path(str(record["source_cue_video"]))
        frame_indices = [0, cut // 2, cut - 1]
        labels = [
            f"s{record['scene_index']:03d} start f0",
            f"s{record['scene_index']:03d} mid f{cut // 2}",
            f"s{record['scene_index']:03d} end f{cut - 1}",
        ]
        panels = [
            _review_tile(_read_frame(output_path, frame_index), label)
            for frame_index, label in zip(frame_indices, labels)
        ]
        panels.append(
            _review_tile(
                _read_frame(source_cue_path, cut),
                f"orig flash f{cut}",
            )
        )
        strip = np.concatenate(panels, axis=1)
        scenario = REVIEW_SCENARIO_LABELS[str(record["scenario"])]
        cv2.putText(
            strip,
            scenario,
            (4, scene_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, REVIEW_COLUMNS)
        y0, x0 = row * scene_height, column * scene_width
        grid[y0 : y0 + scene_height, x0 : x0 + scene_width] = strip
        cv2.rectangle(
            grid,
            (x0, y0),
            (x0 + scene_width - 1, y0 + scene_height - 1),
            (90, 90, 90),
            1,
        )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(review_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise ValueError(f"Unable to write review grid: {review_path}")


def _write_outputs(
    output_root: Path,
    cue_dir: Path,
    records: list[dict[str, Any]],
    review_path: Path | None,
) -> None:
    records_path = cue_dir / "cue_render_records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    scenario_counts = Counter(str(record["scenario"]) for record in records)
    audit = {
        "variant": "physion_pp_tracking_cue_baseline_v1",
        "baseline_only": True,
        "scene_count": len(records),
        "scenario_counts": dict(scenario_counts),
        "cue_rule": (
            "red=official target_id and yellow=official zone_id on every visible instance mask"
        ),
        "cut_rule": "output contains frames [0, detected original flash start)",
        "all_flash_starts_match_source_records": all(
            int(record["detected_original_flash_start"])
            == int(record["source_cue_frame_index"])
            == int(record["rendered_frame_count"])
            for record in records
        ),
        "review_grid": str(review_path) if review_path is not None else None,
        "cue_records": str(records_path),
        "baseline_dataset_root": str(output_root),
    }
    (output_root / "tracking_cue_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scenes = _select_scenes(dataset_root, args.limit)
    original_cue_dir, original_records = _load_original_records(dataset_root)
    missing_records = [
        scene.stimulus_id for scene in scenes if scene.stimulus_id not in original_records
    ]
    if missing_records:
        raise FileNotFoundError(
            f"Missing {len(missing_records)} original cue records; first few: {missing_records[:3]}"
        )
    cue_dir = _ensure_dataset_overlay(dataset_root, output_root)

    records = []
    for index, scene in enumerate(scenes, start=1):
        output_path = cue_dir / f"{scene.stimulus_id}_cue.mp4"
        record = _render_scene(
            scene,
            original_records[scene.stimulus_id],
            original_cue_dir,
            output_path,
            args.overwrite,
        )
        records.append(record)
        if args.progress_every and index % args.progress_every == 0:
            print(f"[physion_pp_tracking_cue] rendered {index}/{len(scenes)}", flush=True)

    review_path = None
    if not args.no_review_grid:
        review_path = output_root / "review" / "tracking_cue_review_grid.jpg"
        _create_review_grid(records, cue_dir, review_path)
    _write_outputs(output_root, cue_dir, records, review_path)
    print(f"[ok] rendered and audited {len(records)} tracking-cue baseline videos")
    print(f"[ok] baseline dataset root: {output_root}")
    if review_path is not None:
        print(f"[ok] combined review grid: {review_path}")


if __name__ == "__main__":
    main()
