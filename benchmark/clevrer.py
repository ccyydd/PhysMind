from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from utils.config import DATA_DIR


VALIDATION_FILE_CANDIDATES = [
    "validation.json",
    "executor/data/validation.json",
]

@dataclass
class ClevrerChoice:
    choice_id: int
    choice: str
    answer: Optional[str] = None
    program: Optional[List[str]] = None


@dataclass
class ClevrerQuestion:
    question_id: int
    question: str
    question_type: str
    question_subtype: Optional[str]
    answer: Optional[str]
    choices: List[ClevrerChoice]
    program: Optional[List[str]] = None


@dataclass
class ClevrerScene:
    scene_index: int
    video_filename: str
    video_path: Path
    questions: List[ClevrerQuestion]


def default_clevrer_root() -> Path:
    return DATA_DIR / "clevrer"


def _resolve_first_existing(root: Path, candidates: List[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} under {root}. "
        "Expected CLEVRER validation data in data/clevrer/."
    )


def _find_segmented_video_dirs(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("video_") and "-" in path.name
    )


def resolve_video_path(root: Path, video_filename: str) -> Path:
    direct_candidates = [
        root / "video_validation" / video_filename,
        root / "video" / video_filename,
        root / "videos" / video_filename,
        root / video_filename,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    for segment_dir in _find_segmented_video_dirs(root):
        candidate = segment_dir / video_filename
        if candidate.exists():
            return candidate

    # Fall back to the conventional directory for clearer downstream errors.
    return root / "video_validation" / video_filename


def load_validation_scenes(dataset_root: Optional[Union[str, Path]] = None) -> List[ClevrerScene]:
    root = Path(dataset_root) if dataset_root else default_clevrer_root()
    validation_file = _resolve_first_existing(root, VALIDATION_FILE_CANDIDATES)

    raw_scenes = json.loads(validation_file.read_text(encoding="utf-8"))
    scenes = []
    for raw_scene in raw_scenes:
        questions = [
            ClevrerQuestion(
                question_id=int(raw_question["question_id"]),
                question=raw_question["question"],
                question_type=raw_question["question_type"],
                question_subtype=raw_question.get("question_subtype"),
                answer=raw_question.get("answer"),
                choices=[
                    ClevrerChoice(
                        choice_id=int(choice["choice_id"]),
                        choice=choice["choice"],
                        answer=choice.get("answer"),
                        program=choice.get("program"),
                    )
                    for choice in raw_question.get("choices", [])
                ],
                program=raw_question.get("program"),
            )
            for raw_question in raw_scene["questions"]
        ]
        scenes.append(
            ClevrerScene(
                scene_index=int(raw_scene["scene_index"]),
                video_filename=raw_scene["video_filename"],
                video_path=resolve_video_path(root, raw_scene["video_filename"]),
                questions=questions,
            )
        )
    return scenes
