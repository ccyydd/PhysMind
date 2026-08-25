from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple


DEFAULT_ROOT = Path("data/clevrer")
VALIDATION_QUESTIONS_URL = "https://data.csail.mit.edu/clevrer/questions/validation.json"
VALIDATION_VIDEOS_URL = "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"
EXPECTED_SCENE_IDS = set(range(10000, 15000))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify CLEVRER validation data.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Target CLEVRER data root.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate the local dataset layout without downloading anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return destination

    def _report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100.0 / total_size
        sys.stdout.write(f"\r[download] {destination.name}: {percent:6.2f}%")
        sys.stdout.flush()

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, destination, _report)
    sys.stdout.write("\n")
    return destination


def extract_zip(archive_path: Path, destination_dir: Path) -> None:
    print(f"[extract] {archive_path} -> {destination_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)


def find_validation_questions(root: Path) -> Path | None:
    candidates = [
        root / "validation.json",
        root / "executor" / "data" / "validation.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_video_dir(root: Path) -> Path | None:
    candidates = [
        root / "video_validation",
        root / "video",
        root / "videos",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    segmented_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("video_") and "-" in path.name
    )
    if segmented_dirs:
        return segmented_dirs[0]
    return None


def load_validation_entries(validation_json: Path) -> list[dict]:
    data = json.loads(validation_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {validation_json}")
    return data


def verify_dataset(root: Path) -> Tuple[bool, str]:
    validation_json = find_validation_questions(root)
    if validation_json is None:
        return False, f"Missing validation questions under {root}"

    representative_video_dir = find_video_dir(root)
    if representative_video_dir is None:
        return False, f"Missing validation video directory under {root}"

    segmented_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("video_") and "-" in path.name
    )
    search_dirs = segmented_dirs or [representative_video_dir]

    try:
        entries = load_validation_entries(validation_json)
        scene_ids = [int(scene["scene_index"]) for scene in entries]
        video_names = [str(scene["video_filename"]) for scene in entries]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"Invalid validation questions file {validation_json}: {error}"

    if len(entries) != 5000:
        return False, f"Expected 5000 validation scenes, found {len(entries)} in {validation_json}"
    if len(set(scene_ids)) != 5000 or set(scene_ids) != EXPECTED_SCENE_IDS:
        return False, "Validation scene IDs must be unique and cover 10000 through 14999"
    if len(set(video_names)) != 5000:
        return False, "Validation video filenames must be unique"

    missing = []
    for name in video_names:
        if not any((video_dir / name).exists() for video_dir in search_dirs):
            missing.append(name)
    if missing:
        sample = ", ".join(missing[:5])
        return False, f"Missing {len(missing)} videos under {root}; first few: {sample}"

    if segmented_dirs:
        return True, f"5000 validation scenes and videos across {len(segmented_dirs)} directories"
    return True, f"5000 validation scenes and videos under {representative_video_dir}"


def prepare_dataset(root: Path, force: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    questions_path = root / "validation.json"
    video_archive_path = root / "video_validation.zip"

    download_file(VALIDATION_QUESTIONS_URL, questions_path, force=force)
    download_file(VALIDATION_VIDEOS_URL, video_archive_path, force=force)
    extract_zip(video_archive_path, root)


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()

    if not args.check_only:
        prepare_dataset(root=root, force=args.force)

    ok, message = verify_dataset(root)
    if ok:
        print(f"[ok] {message}")
        return

    raise SystemExit(f"[error] {message}")


if __name__ == "__main__":
    main()
