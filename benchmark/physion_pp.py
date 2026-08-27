from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union

from utils.config import DATA_DIR


PROPERTIES = ["mass", "friction", "bouncy"]

PROPERTY_CSV_FILES = {
    "mass": "physionpp-mass_merge_221111.csv",
    "friction": "physionpp-friction_merge_230108.csv",
    "bouncy": "physionpp-bouncy_merge_230108.csv",
}

TESTDATA_DIRNAME = "testdata_v1"
PHYSION_PP_CUE_VIDEO_DIRNAME = "cue_videos_test_v1"
PHYSION_PP_RUN_CUE_VIDEO_DIRNAME = "physion_pp_cue_videos"

PHYSION_PP_BLINK_CUE_INPUT_MODE = "blink_reference"
PHYSION_PP_TRACKING_CUE_INPUT_MODE = "tracking_overlay"
PHYSION_PP_CUE_INPUT_MODES = {
    PHYSION_PP_BLINK_CUE_INPUT_MODE,
    PHYSION_PP_TRACKING_CUE_INPUT_MODE,
}

# Five rigid-body scenarios used by the Physion++ direct-answer baselines.
# The prompt differs according to whether a curtain separates property
# inference from the prediction setup.
PHYSION_PP_BASELINE_NO_CURTAIN_SCENARIOS = (
    "friction_platform_pp",
    "bouncy_platform_pp",
)

PHYSION_PP_BASELINE_CURTAIN_SCENARIOS = (
    "friction_collision_pp",
    "mass_collision_pp",
    "bouncy_wall_pp",
)

PHYSION_PP_BASELINE_SCENARIOS = (
    *PHYSION_PP_BASELINE_NO_CURTAIN_SCENARIOS,
    *PHYSION_PP_BASELINE_CURTAIN_SCENARIOS,
)

# Five rigid-body scenarios supported by PhysMind.
RIGID_SCENARIOS = (
    "mass_collision_pp",
    "friction_platform_pp",
    "friction_collision_pp",
    "bouncy_platform_pp",
    "bouncy_wall_pp",
)

# Cue rendering follows the Physion++ human protocol: the cue frame is frozen for
# 2 seconds while the red/yellow target overlay blinks at 2 Hz, then the remaining
# visible frames play out to start_frame_for_prediction. Frames after
# start_frame_for_prediction contain the outcome and must never be rendered.
CUE_FREEZE_SECONDS = 2.0
CUE_BLINK_HZ = 2.0
CUE_OVERLAY_ALPHA = 0.6
CUE_ALIGNMENT_MIN_IOU = 0.75
CUE_VIDEO_FPS = 30.0

# Select the middle of the second ON interval. This avoids the compression
# transition at the first cue frame and guarantees that both overlays are on.
CUE_BLINK_PERIOD_FRAMES = int(round(CUE_VIDEO_FPS / CUE_BLINK_HZ))
CUE_BLINK_ON_FRAMES = CUE_BLINK_PERIOD_FRAMES // 2
CUE_REFERENCE_ON_OFFSET = CUE_BLINK_PERIOD_FRAMES + CUE_BLINK_ON_FRAMES // 2


@dataclass
class PhysionPPQuestion:
    question_id: int
    question: str
    question_type: str
    answer: str
    ground_truth_outcome: bool
    choices: List[object] = field(default_factory=list)


@dataclass
class PhysionPPScene:
    scene_index: int
    property_name: str
    scenario: str
    copy_name: str
    config_name: str
    trial_stem: str
    stimulus_id: str
    pair_id: str
    video_filename: str
    video_path: Path
    map_path: Path
    id_json_path: Path
    pkl_path: Path
    start_frame_for_prediction: int
    questions: List[PhysionPPQuestion]
    cue_frame_index: Optional[int] = None
    cue_reference_frame_index: Optional[int] = None
    cue_input_mode: str = PHYSION_PP_BLINK_CUE_INPUT_MODE
    benchmark: str = "physion_pp"


def default_physion_pp_root() -> Path:
    return DATA_DIR / "physion_pp"


def resolve_physion_pp_dataset_dir(root: Path) -> Path:
    for candidate in (root, root / TESTDATA_DIRNAME):
        if all((candidate / csv_name).exists() for csv_name in PROPERTY_CSV_FILES.values()):
            return candidate
    raise FileNotFoundError(
        f"Could not find the three supported Physion++ test CSVs under {root} or {root / TESTDATA_DIRNAME}. "
        "Expected extracted test_data.zip under data/physion_pp/testdata_v1/."
    )


def _parse_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected True/False label, got {value!r}")


def _trial_relative_parts(full_stim_path: str) -> Tuple[str, str, str]:
    parts = [part for part in full_stim_path.replace("\\", "/").split("/") if part]
    if len(parts) < 3 or not parts[-1].endswith("_img.mp4"):
        raise ValueError(f"Unexpected Physion++ full_stim_paths entry: {full_stim_path!r}")
    return parts[-3], parts[-2], parts[-1]


def _build_question(outcome: bool) -> PhysionPPQuestion:
    answer = "yes" if outcome else "no"
    return PhysionPPQuestion(
        question_id=0,
        question=(
            "The video ends before the outcome is resolved. If physics continues to unfold "
            "after the video ends, will the red-cued agent object make contact with the "
            "yellow-cued patient object?"
        ),
        question_type="ocp",
        answer=answer,
        ground_truth_outcome=outcome,
    )


def load_test_scenes(
    dataset_root: Optional[Union[str, Path]] = None,
    properties: Optional[List[str]] = None,
) -> List[PhysionPPScene]:
    root = Path(dataset_root) if dataset_root else default_physion_pp_root()
    dataset_dir = resolve_physion_pp_dataset_dir(root)
    selected_properties = set(properties or PROPERTIES)
    unknown = sorted(set(selected_properties) - set(PROPERTIES))
    if unknown:
        raise ValueError(f"Unknown Physion++ properties: {unknown}; expected subset of {PROPERTIES}")

    scenes: List[PhysionPPScene] = []
    missing_files: List[str] = []
    official_scene_index = 0
    for property_name in PROPERTIES:
        csv_path = dataset_dir / PROPERTY_CSV_FILES[property_name]
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                copy_dir, config_name, video_name = _trial_relative_parts(
                    row["full_stim_paths"]
                )
                scenario, _, copy_name = copy_dir.rpartition("-")
                if not scenario or not copy_name.startswith("copy"):
                    raise ValueError(f"Unexpected Physion++ copy directory name: {copy_dir!r}")
                scene_index = official_scene_index
                official_scene_index += 1
                if property_name not in selected_properties or scenario not in RIGID_SCENARIOS:
                    continue
                trial_stem = video_name.replace("_img.mp4", "")
                trial_dir = dataset_dir / copy_dir / config_name
                video_path = trial_dir / video_name
                map_path = trial_dir / f"{trial_stem}_map.png"
                id_json_path = trial_dir / f"{trial_stem}_id.json"
                id_video_path = trial_dir / f"{trial_stem}_id.mp4"
                pkl_path = trial_dir / f"{trial_stem}.pkl"
                for required in (
                    video_path,
                    map_path,
                    id_json_path,
                    id_video_path,
                    pkl_path,
                ):
                    if not required.exists():
                        missing_files.append(str(required))
                outcome = _parse_bool(row["target_hit_zone_labels"])
                scenes.append(
                    PhysionPPScene(
                        scene_index=scene_index,
                        property_name=property_name,
                        scenario=scenario,
                        copy_name=copy_name,
                        config_name=config_name,
                        trial_stem=trial_stem,
                        stimulus_id=row["filenames"].replace("_img.mp4", ""),
                        pair_id=f"{scenario}/{config_name}/{trial_stem}",
                        video_filename=video_name,
                        video_path=video_path,
                        map_path=map_path,
                        id_json_path=id_json_path,
                        pkl_path=pkl_path,
                        start_frame_for_prediction=int(row["start_frame_for_prediction"]),
                        questions=[_build_question(outcome)],
                    )
                )

    if missing_files:
        sample = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} Physion++ trial files referenced by the test CSVs; first few: {sample}"
        )
    return scenes


def resolve_cue_clip_scenes(
    scenes: List[PhysionPPScene],
    dataset_root: Optional[Union[str, Path]] = None,
) -> List[PhysionPPScene]:
    """Point scenes at the pre-rendered leak-free cue clips used for evaluation.

    The clips are rendered once by create_cue_rendered_physion_pp_scenes into
    <root>/cue_videos_test_v1/ (generated by scripts/prepare_physion_pp.py).
    """
    root = Path(dataset_root) if dataset_root else default_physion_pp_root()
    cue_dir = root / PHYSION_PP_CUE_VIDEO_DIRNAME
    records_path = cue_dir / "cue_render_records.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing Physion++ cue render records: {records_path}. "
            "Render the cue clips before running direct-answer evaluation."
        )
    with records_path.open("r", encoding="utf-8") as file:
        records_payload = json.load(file)
    records_by_stimulus = {
        str(record["stimulus_id"]): record
        for record in records_payload
        if isinstance(record, dict) and record.get("stimulus_id")
    }
    resolved: List[PhysionPPScene] = []
    missing: List[str] = []
    for scene in scenes:
        cue_path = cue_dir / f"{scene.stimulus_id}_cue.mp4"
        record = records_by_stimulus.get(scene.stimulus_id)
        if not cue_path.exists() or record is None or record.get("cue_frame_index") is None:
            missing.append(str(cue_path))
            continue
        cue_frame_index = int(record["cue_frame_index"])
        cue_input_mode = str(
            record.get("cue_input_mode") or PHYSION_PP_BLINK_CUE_INPUT_MODE
        )
        if cue_input_mode not in PHYSION_PP_CUE_INPUT_MODES:
            raise ValueError(
                f"Unsupported Physion++ cue_input_mode={cue_input_mode!r} "
                f"for stimulus {scene.stimulus_id}"
            )
        cue_reference_frame_index = (
            None
            if cue_input_mode == PHYSION_PP_TRACKING_CUE_INPUT_MODE
            else cue_frame_index + CUE_REFERENCE_ON_OFFSET
        )
        resolved.append(
            replace(
                scene,
                video_filename=cue_path.name,
                video_path=cue_path,
                cue_frame_index=cue_frame_index,
                cue_reference_frame_index=cue_reference_frame_index,
                cue_input_mode=cue_input_mode,
            )
        )
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} Physion++ cue clips under {cue_dir}. "
            "Render them first with scripts/prepare_physion_pp.py. "
            f"First few: {sample}"
        )
    return resolved


# Pixel-only cue flash detection. Blink transitions shift a constant footprint toward
# or away from the saturated red/yellow overlay with regular blink-period spacing.
# Classification is by direction of change: the red overlay (0,0,255) pulls G and B
# DOWN, the yellow overlay (0,255,255) pulls G UP -- relative tests (dR-dG) misread
# the yellow overlay on yellow-green mats as red.
CUE_TRIM_SPIKE_MIN_PX = 100
CUE_TRIM_COMB_TOLERANCE = 1
CUE_TRIM_FOOTPRINT_MIN_IOU = 0.4


def _cue_colored_transition(prev: "np.ndarray", curr: "np.ndarray"):
    import numpy as np

    delta = curr.astype(np.int16) - prev.astype(np.int16)
    d_b, d_g, d_r = delta[:, :, 0], delta[:, :, 1], delta[:, :, 2]
    red_up = (d_r > 15) & (d_g < 3) & (d_b < 3)
    yellow_up = (d_r > 8) & (d_g > 8) & (d_b < 3)
    red_down = (-d_r > 15) & (d_g > -3) & (d_b > -3)
    yellow_down = (-d_r > 8) & (-d_g > 8) & (d_b > -3)
    return (
        int(red_up.sum()) + int(yellow_up.sum()),
        int(red_down.sum()) + int(yellow_down.sum()),
        red_up,
        yellow_up,
    )


def detect_cue_flash_start(frames: List["np.ndarray"]):
    """Return (first_flash_frame, red_region, yellow_region) or (None, None, None).

    Anchors on the first colored up-spike confirmed by the blink comb (down at +7,
    up at +15, down at +22) with a consistent footprint. The region masks are taken
    from the interior +15 transition, which has no motion mixed in."""
    ups, downs = {}, {}
    for t in range(1, len(frames)):
        up_px, down_px, _, _ = _cue_colored_transition(frames[t - 1], frames[t])
        if up_px >= CUE_TRIM_SPIKE_MIN_PX:
            ups[t] = up_px
        if down_px >= CUE_TRIM_SPIKE_MIN_PX:
            downs[t] = down_px

    tolerance = range(-CUE_TRIM_COMB_TOLERANCE, CUE_TRIM_COMB_TOLERANCE + 1)

    def has(spikes, center):
        return any(center + d in spikes for d in tolerance)

    for start in sorted(ups):
        if not (has(downs, start + 7) and has(ups, start + 15) and has(downs, start + 22)):
            continue
        interior = next(start + 15 + d for d in tolerance if (start + 15 + d) in ups)
        _, _, red_a, yel_a = _cue_colored_transition(frames[start - 1], frames[start])
        _, _, red_b, yel_b = _cue_colored_transition(frames[interior - 1], frames[interior])
        union_a, union_b = red_a | yel_a, red_b | yel_b
        inter = (union_a & union_b).sum()
        union = (union_a | union_b).sum()
        if union and inter / union >= CUE_TRIM_FOOTPRINT_MIN_IOU:
            return start, red_b, yel_b
    return None, None, None


def stage_cue_clips_into_run_dir(
    scenes: List[PhysionPPScene],
    *,
    run_dir: Path,
    trim_cue_flash: bool = False,
) -> List[PhysionPPScene]:
    """Copy the evaluated cue clips into the run directory for easy scene lookup.

    Files are prefixed with the stable scene_index (e.g. 000032_..._cue.mp4) and
    video_manifest.json maps each scene to its staged clip, the source cue clip,
    and the raw full-rollout video (which contains the outcome and must never be
    fed to a model).

    trim_cue_flash: world-model mode stages BOTH clips -- the original cue clip
    (<stem>_cue.mp4, kept solely for red/yellow role binding) and a TAIL-TRIMMED
    copy (<stem>_cue_trimmed.mp4) cut at the first detected flash frame, which
    scene.video_path points at so every downstream tool (SAM3/depth/pose) consumes
    a clean motion-only video. The red/yellow flash regions are dropped next to the
    clips as <stem>_cue_flash.npz for the track-role binding stage. The
    direct-answer mode must keep the flash as its only pointer to the queried
    objects, so it stays on the default untrimmed single copy.
    """
    import shutil

    import cv2
    import numpy as np

    staged_dir = run_dir / PHYSION_PP_RUN_CUE_VIDEO_DIRNAME
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_scenes: List[PhysionPPScene] = []
    manifest = []
    for scene in scenes:
        target = staged_dir / f"{scene.scene_index:06d}_{scene.stimulus_id}_cue.mp4"
        entry = {
            "scene_index": scene.scene_index,
            "property": scene.property_name,
            "scenario": scene.scenario,
            "stimulus_id": scene.stimulus_id,
            "cue_frame_index": scene.cue_frame_index,
            "cue_reference_frame_index": scene.cue_reference_frame_index,
            "cue_input_mode": scene.cue_input_mode,
            "staged_video": target.name,
            "source_cue_video": str(scene.video_path),
            "source_raw_video": str(scene.pkl_path.with_name(f"{scene.trial_stem}_img.mp4")),
        }
        if not target.exists():
            shutil.copy2(scene.video_path, target)
        model_video = target
        if trim_cue_flash:
            trimmed = target.with_name(target.stem + "_trimmed.mp4")
            flash_npz = target.with_name(target.stem + "_flash.npz")
            if not trimmed.exists():
                capture = cv2.VideoCapture(str(target))
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                capture.release()
                flash_start, red_region, yellow_region = detect_cue_flash_start(frames)
                if flash_start is None:
                    shutil.copy2(target, trimmed)
                    entry["cue_flash_trim"] = {"status": "detect_failed"}
                else:
                    height, width = frames[0].shape[:2]
                    writer = cv2.VideoWriter(
                        str(trimmed), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    try:
                        for frame in frames[:flash_start]:
                            writer.write(frame)
                    finally:
                        writer.release()
                    np.savez_compressed(
                        flash_npz,
                        red_region=red_region.astype(np.uint8),
                        yellow_region=yellow_region.astype(np.uint8),
                        flash_start=np.int64(flash_start),
                    )
                    entry["cue_flash_trim"] = {
                        "status": "trimmed",
                        "flash_start": int(flash_start),
                        "kept_frames": int(flash_start),
                        "source_frames": len(frames),
                        "trimmed_video": trimmed.name,
                        "cue_video_with_flash": target.name,
                        "flash_regions": flash_npz.name,
                    }
            else:
                entry["cue_flash_trim"] = {
                    "status": "trimmed",
                    "trimmed_video": trimmed.name,
                    "cue_video_with_flash": target.name,
                    "flash_regions": flash_npz.name if flash_npz.exists() else None,
                }
            entry["staged_video"] = trimmed.name
            model_video = trimmed
        manifest.append(entry)
        staged_scenes.append(
            replace(scene, video_filename=model_video.name, video_path=model_video)
        )

    manifest_path = staged_dir / "video_manifest.json"
    existing = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    merged = {entry["scene_index"]: entry for entry in existing}
    merged.update({entry["scene_index"]: entry for entry in manifest})
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump([merged[key] for key in sorted(merged)], file, indent=2)
    return staged_scenes


def _read_visible_frames(scene: PhysionPPScene) -> List["np.ndarray"]:
    import cv2

    capture = cv2.VideoCapture(str(scene.video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open Physion++ video: {scene.video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 29.0 <= fps <= 31.0:
            raise ValueError(f"Unexpected FPS {fps} for {scene.video_path}; expected 30 fps test videos.")
        frames = []
        while len(frames) <= scene.start_frame_for_prediction:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) <= scene.start_frame_for_prediction:
        raise ValueError(
            f"Video {scene.video_path} ended at frame {len(frames) - 1} before "
            f"start_frame_for_prediction={scene.start_frame_for_prediction}"
        )
    return frames


def _load_cue_masks(scene: PhysionPPScene) -> Tuple["np.ndarray", "np.ndarray"]:
    import cv2
    import numpy as np

    map_img = cv2.imread(str(scene.map_path), cv2.IMREAD_UNCHANGED)
    if map_img is None or map_img.ndim != 3 or map_img.shape[2] != 4:
        raise ValueError(f"Unexpected Physion++ map image format: {scene.map_path}")
    blue, green, red, alpha = (map_img[:, :, index].astype(np.int32) for index in range(4))
    visible = alpha > 0
    red_mask = visible & (red > 128) & (green < 128) & (blue < 128)
    yellow_mask = visible & (red > 128) & (green > 128) & (blue < 128)
    if not red_mask.any() or not yellow_mask.any():
        raise ValueError(f"Physion++ map image missing red or yellow cue region: {scene.map_path}")
    return red_mask, yellow_mask


def _best_instance_iou(cue_mask: "np.ndarray", instance_masks: "np.ndarray") -> float:
    cue_area = cue_mask.sum()
    best = 0.0
    for index in range(instance_masks.shape[2]):
        instance = instance_masks[:, :, index].astype(bool)
        intersection = (cue_mask & instance).sum()
        if intersection == 0:
            continue
        union = cue_area + instance.sum() - intersection
        best = max(best, intersection / union)
    return best


def find_cue_frame(scene: PhysionPPScene, red_mask: "np.ndarray", yellow_mask: "np.ndarray") -> Tuple[int, float]:
    """Locate the frame the _map.png cue corresponds to via IoU against GT instance masks.

    Searches backward from start_frame_for_prediction and stops at the first frame whose
    red+yellow alignment both clear CUE_ALIGNMENT_MIN_IOU; falls back to the best-scoring
    frame. The GT masks are used only to place the cue overlay (information humans also
    received); they are never part of model input.
    """
    from pycocotools import mask as rle_mask

    with scene.id_json_path.open("r", encoding="utf-8") as file:
        frames_rle = json.load(file)

    best_frame, best_score = scene.start_frame_for_prediction, -1.0
    for frame_index in range(scene.start_frame_for_prediction, -1, -1):
        frame_key = f"{frame_index:04d}"
        rles = frames_rle.get(frame_key)
        if not rles:
            continue
        instances = rle_mask.decode(rles)
        if instances.ndim == 2:
            instances = instances[:, :, None]
        red_iou = _best_instance_iou(red_mask, instances)
        yellow_iou = _best_instance_iou(yellow_mask, instances)
        score = red_iou + yellow_iou
        if score > best_score:
            best_frame, best_score = frame_index, score
        if red_iou >= CUE_ALIGNMENT_MIN_IOU and yellow_iou >= CUE_ALIGNMENT_MIN_IOU:
            return frame_index, score / 2.0
    return best_frame, max(best_score, 0.0) / 2.0


def _fallback_cue_from_seg_video(
    scene: PhysionPPScene,
    min_pixels: int = 20,
    color_tolerance: int = 30,
) -> Tuple[int, "np.ndarray", "np.ndarray"]:
    """Cue placement for trials whose _map.png is degenerate (a target fully occluded
    at the boundary frame). Uses the official pkl target_id/zone_id plus the _id.mp4
    segmentation video to find the latest visible frame where both targets are visible,
    and returns their masks there.
    """
    import cv2
    import numpy as np
    import pickle

    with scene.pkl_path.open("rb") as file:
        static = pickle.load(file)["static"]
    object_ids = [int(value) for value in static["object_ids"]]
    seg_colors = np.asarray(static["video_object_segmentation_colors"], dtype=np.int32)
    # pkl colors are RGB; cv2 decodes _id.mp4 as BGR.
    agent_color = seg_colors[object_ids.index(int(static["target_id"]))][::-1]
    patient_color = seg_colors[object_ids.index(int(static["zone_id"]))][::-1]

    capture = cv2.VideoCapture(str(scene.id_json_path.with_name(f"{scene.trial_stem}_id.mp4")))
    if not capture.isOpened():
        raise ValueError(f"Unable to open Physion++ segmentation video for {scene.stimulus_id}")
    frames = []
    try:
        while len(frames) <= scene.start_frame_for_prediction:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    for frame_index in range(min(scene.start_frame_for_prediction, len(frames) - 1), -1, -1):
        frame = frames[frame_index].astype(np.int32)
        agent_mask = (np.abs(frame - agent_color).max(axis=2) <= color_tolerance)
        patient_mask = (np.abs(frame - patient_color).max(axis=2) <= color_tolerance)
        if agent_mask.sum() >= min_pixels and patient_mask.sum() >= min_pixels:
            return frame_index, agent_mask, patient_mask
    raise ValueError(f"No frame with both cue targets visible for {scene.stimulus_id}")


def _render_cue_video(
    scene: PhysionPPScene,
    output_path: Path,
    cue_frame_index: int,
    red_mask: "np.ndarray",
    yellow_mask: "np.ndarray",
    fps: float = 30.0,
) -> None:
    import cv2
    import numpy as np

    frames = _read_visible_frames(scene)
    overlay = np.zeros_like(frames[0], dtype=np.float64)
    overlay[red_mask] = (0, 0, 255)
    overlay[yellow_mask] = (0, 255, 255)
    overlay_alpha = ((red_mask | yellow_mask)[:, :, None] * CUE_OVERLAY_ALPHA).astype(np.float64)

    cue_frame = frames[cue_frame_index].astype(np.float64)
    flash_frame = (cue_frame * (1 - overlay_alpha) + overlay * overlay_alpha).astype(np.uint8)
    freeze_frames = int(round(CUE_FREEZE_SECONDS * fps))
    blink_period = max(2, int(round(fps / CUE_BLINK_HZ)))
    blink_on = blink_period // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Unable to open Physion++ cue video writer: {output_path}")
    try:
        for frame in frames[:cue_frame_index]:
            writer.write(frame)
        for index in range(freeze_frames):
            on = (index % blink_period) < blink_on
            writer.write(flash_frame if on else frames[cue_frame_index])
        for frame in frames[cue_frame_index:]:
            writer.write(frame)
    finally:
        writer.release()


def create_cue_rendered_physion_pp_scenes(
    scenes: List[PhysionPPScene],
    *,
    output_dir: Path,
    skip_existing: bool = True,
    progress_every: int = 25,
) -> List[PhysionPPScene]:
    """Produce leak-free evaluation clips: visible frames only, human-protocol cue inserted.

    Writes one mp4 per scene plus cue_render_records.json describing the cue placement.
    Returns scenes with video_path pointing at the rendered clips. Finished clips are
    skipped on rerun; in-progress files use a .part.mp4 name so interruptions never
    leave a truncated final clip behind.
    """
    records = []
    rendered_scenes: List[PhysionPPScene] = []
    for index, scene in enumerate(scenes):
        try:
            red_mask, yellow_mask = _load_cue_masks(scene)
            cue_frame_index, alignment_iou = find_cue_frame(scene, red_mask, yellow_mask)
            cue_source = "map"
        except ValueError:
            cue_frame_index, red_mask, yellow_mask = _fallback_cue_from_seg_video(scene)
            alignment_iou = -1.0
            cue_source = "seg_video_fallback"
        output_path = output_dir / f"{scene.stimulus_id}_cue.mp4"
        if not (skip_existing and output_path.exists()):
            partial_path = output_path.with_name(f"{output_path.stem}.part.mp4")
            _render_cue_video(scene, partial_path, cue_frame_index, red_mask, yellow_mask)
            partial_path.replace(output_path)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"[physion_pp] cue-rendered {index + 1}/{len(scenes)} scenes", flush=True)
        records.append(
            {
                "stimulus_id": scene.stimulus_id,
                "property": scene.property_name,
                "scenario": scene.scenario,
                "start_frame_for_prediction": scene.start_frame_for_prediction,
                "cue_frame_index": cue_frame_index,
                "cue_source": cue_source,
                "cue_alignment_iou": round(alignment_iou, 4),
                "video": output_path.name,
            }
        )
        rendered_scenes.append(replace(scene, video_filename=output_path.name, video_path=output_path))

    records_path = output_dir / "cue_render_records.json"
    with records_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    return rendered_scenes
