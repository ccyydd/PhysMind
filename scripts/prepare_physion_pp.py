from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from benchmark.physion_pp import (
    PHYSION_PP_CUE_VIDEO_DIRNAME,
    PROPERTY_CSV_FILES,
    PROPERTIES,
    RIGID_SCENARIOS,
    TESTDATA_DIRNAME,
    create_cue_rendered_physion_pp_scenes,
    load_test_scenes,
)


DEFAULT_ROOT = Path("data/physion_pp")
ARCHIVE_URL = "https://physion-v2.s3.amazonaws.com/test_data.zip"
EXPECTED_SCENARIO_COUNTS = {
    "mass_collision_pp": 64,
    "friction_collision_pp": 64,
    "friction_platform_pp": 64,
    "bouncy_platform_pp": 96,
    "bouncy_wall_pp": 96,
}
REQUIRED_TRIAL_SUFFIXES = (".pkl", "_img.mp4", "_id.mp4", "_id.json", "_map.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the 384 rigid-body Physion++ evaluation trials used by PhysMind."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Target Physion++ data root.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the selected trials and cue videos without downloading or rendering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload, re-extract, and rerender selected files.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the full official archive after the selected trials are prepared.",
    )
    return parser


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return destination

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        progress = downloaded * 100.0 / total_size
        sys.stdout.write(f"\r[download] {destination.name}: {progress:6.2f}%")
        sys.stdout.flush()

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, destination, report)
    sys.stdout.write("\n")
    return destination


def _trial_parts(full_stim_path: str) -> tuple[str, str, str]:
    parts = [part for part in full_stim_path.replace("\\", "/").split("/") if part]
    if len(parts) < 3 or not parts[-1].endswith("_img.mp4"):
        raise ValueError(f"Unexpected Physion++ path in CSV: {full_stim_path!r}")
    return parts[-3], parts[-2], parts[-1]


def selected_archive_members(archive: zipfile.ZipFile) -> set[str]:
    members = set(archive.namelist())
    selected = set()
    selected_trials = 0
    for property_name in PROPERTIES:
        csv_member = f"{TESTDATA_DIRNAME}/{PROPERTY_CSV_FILES[property_name]}"
        if csv_member not in members:
            raise FileNotFoundError(f"Missing {csv_member} in Physion++ archive")
        selected.add(csv_member)
        with archive.open(csv_member) as raw_file:
            rows = csv.DictReader(io.TextIOWrapper(raw_file, encoding="utf-8"))
            for row in rows:
                copy_dir, config_name, video_name = _trial_parts(row["full_stim_paths"])
                scenario = copy_dir.rpartition("-")[0]
                if scenario not in RIGID_SCENARIOS:
                    continue
                trial_stem = video_name.removesuffix("_img.mp4")
                prefix = f"{TESTDATA_DIRNAME}/{copy_dir}/{config_name}/{trial_stem}"
                selected.update(prefix + suffix for suffix in REQUIRED_TRIAL_SUFFIXES)
                selected_trials += 1

    if selected_trials != 384:
        raise ValueError(f"Expected 384 supported Physion++ trials, found {selected_trials}")
    missing = sorted(selected - members)
    if missing:
        raise FileNotFoundError(
            f"Archive is missing {len(missing)} selected files; first: {missing[0]}"
        )
    return selected


def extract_selected_trials(archive_path: Path, root: Path, force: bool = False) -> None:
    print(f"[extract] selected Physion++ trials from {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in sorted(selected_archive_members(archive)):
            relative_path = Path(member)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe archive member: {member}")
            destination = root / relative_path
            info = archive.getinfo(member)
            if (
                destination.exists()
                and not force
                and destination.stat().st_size == info.file_size
            ):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def verify_selected_dataset(root: Path, require_cues: bool = True) -> tuple[bool, str]:
    try:
        scenes = load_test_scenes(dataset_root=root)
    except (FileNotFoundError, ValueError) as error:
        return False, str(error)

    counts = Counter(scene.scenario for scene in scenes)
    if len(scenes) != 384 or dict(counts) != EXPECTED_SCENARIO_COUNTS:
        return (
            False,
            f"Expected 384 selected trials with {EXPECTED_SCENARIO_COUNTS}, "
            f"found {dict(counts)}",
        )
    if len({scene.scene_index for scene in scenes}) != 384:
        return False, "Physion++ scene indices are not unique"

    if require_cues:
        cue_dir = root / PHYSION_PP_CUE_VIDEO_DIRNAME
        records_path = cue_dir / "cue_render_records.json"
        if not records_path.exists():
            return False, f"Missing cue render records: {records_path}"
        try:
            records = json.loads(records_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return False, f"Invalid cue render records: {error}"
        records_by_stimulus = {
            str(record.get("stimulus_id")): record
            for record in records
            if isinstance(record, dict)
        }
        missing_cues = [
            scene.stimulus_id
            for scene in scenes
            if scene.stimulus_id not in records_by_stimulus
            or not (cue_dir / f"{scene.stimulus_id}_cue.mp4").exists()
        ]
        if missing_cues:
            return (
                False,
                f"Missing {len(missing_cues)} selected cue videos; first: "
                f"{missing_cues[0]}",
            )

    return True, f"384 supported Physion++ trials ready: {dict(counts)}"


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    archive_path = root / "test_data.zip"

    if not args.check_only:
        root.mkdir(parents=True, exist_ok=True)
        data_ok, _ = verify_selected_dataset(root, require_cues=False)
        if args.force or not data_ok:
            download_file(ARCHIVE_URL, archive_path, force=args.force)
            extract_selected_trials(archive_path, root, force=args.force)

        scenes = load_test_scenes(dataset_root=root)
        cue_dir = root / PHYSION_PP_CUE_VIDEO_DIRNAME
        cue_dir.mkdir(parents=True, exist_ok=True)
        create_cue_rendered_physion_pp_scenes(
            scenes,
            output_dir=cue_dir,
            skip_existing=not args.force,
        )

    ok, message = verify_selected_dataset(root)
    if not ok:
        raise SystemExit(f"[error] {message}")

    if not args.check_only and archive_path.exists() and not args.keep_archive:
        archive_path.unlink()
        print(f"[cleanup] removed full archive {archive_path}")
    print(f"[ok] {message}")


if __name__ == "__main__":
    main()
