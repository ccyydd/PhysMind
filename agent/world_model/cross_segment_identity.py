from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from agent.query import answer_with_image_files
from utils.config import ModelConfig


CROSS_SEGMENT_IDENTITY_AB_PROMPT = (
    "REFERENCE is one object. A and B are two candidate objects from the same scene a moment "
    "earlier (lighting and pose may differ slightly). Exactly one of A or B is the SAME physical "
    "object as REFERENCE. Judge by colour and shape. Reply with a single letter: A or B."
)
CROSS_SEGMENT_IDENTITY_YES_NO_PROMPT = (
    "REFERENCE is one object from a later segment of a video. CANDIDATE is an object from an "
    "earlier segment of the same scene (lighting, pose, and visible area may differ). Decide "
    "whether they are the SAME physical object. Judge by colour, shape, and visible appearance. "
    "Reply with exactly one word: YES or NO."
)


class CrossSegmentIdentityError(RuntimeError):
    """A focused VLM identity decision could not produce one unambiguous answer."""


def crop_object(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    pad: int = 10,
    size: int = 140,
) -> np.ndarray:
    boolean_mask = np.asarray(mask).astype(bool)
    if image.ndim != 3 or image.shape[:2] != boolean_mask.shape:
        raise CrossSegmentIdentityError(
            "cross-segment identity image and mask dimensions do not match"
        )
    ys, xs = np.nonzero(boolean_mask)
    if not len(xs):
        raise CrossSegmentIdentityError("cross-segment identity mask is empty")
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(image.shape[1], int(xs.max()) + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(image.shape[0], int(ys.max()) + pad)
    crop = image[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    if height == 0 or width == 0:
        raise CrossSegmentIdentityError("cross-segment identity crop is empty")
    scale = size / max(height, width)
    return cv2.resize(
        crop,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def parse_ab_response(text: str) -> str:
    matches = re.findall(r"(?<![A-Z])[AB](?![A-Z])", str(text or "").upper())
    unique = set(matches)
    if len(unique) != 1:
        raise CrossSegmentIdentityError(
            f"cross-segment A/B response is not unambiguous: {text!r}"
        )
    return unique.pop()


def parse_yes_no_response(text: str) -> bool:
    matches = re.findall(r"\b(?:YES|NO)\b", str(text or "").upper())
    unique = set(matches)
    if len(unique) != 1:
        raise CrossSegmentIdentityError(
            f"cross-segment YES/NO response is not unambiguous: {text!r}"
        )
    return unique.pop() == "YES"


def _write_crop(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    cropped = crop_object(image, mask)
    if not cv2.imwrite(str(path), cropped):
        raise CrossSegmentIdentityError(
            f"could not write cross-segment identity crop: {path}"
        )


def select_same_object_ab(
    *,
    config: ModelConfig,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    candidate_images: Sequence[np.ndarray],
    candidate_masks: Sequence[np.ndarray],
    candidate_ids: Sequence[str],
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(candidate_images) != 2 or len(candidate_masks) != 2 or len(candidate_ids) != 2:
        raise CrossSegmentIdentityError(
            "cross-segment A/B identity requires exactly two candidates"
        )
    normalized_ids = [str(candidate_id) for candidate_id in candidate_ids]
    if not all(normalized_ids) or len(set(normalized_ids)) != 2:
        raise CrossSegmentIdentityError(
            "cross-segment A/B candidate ids must be two distinct non-empty strings"
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            image_paths = [
                directory / "REFERENCE.png",
                directory / "A.png",
                directory / "B.png",
            ]
            _write_crop(image_paths[0], reference_image, reference_mask)
            for index in range(2):
                _write_crop(
                    image_paths[index + 1],
                    candidate_images[index],
                    candidate_masks[index],
                )
            response = answer_with_image_files(
                config=config,
                prompt=CROSS_SEGMENT_IDENTITY_AB_PROMPT,
                image_paths=image_paths,
                image_labels=["REFERENCE", "A", "B"],
                request_context=request_context,
            )
    except CrossSegmentIdentityError:
        raise
    except Exception as exc:
        raise CrossSegmentIdentityError(
            f"cross-segment A/B VLM request failed: {type(exc).__name__}: {exc}"
        ) from exc
    selected_label = parse_ab_response(response.text)
    selected_index = 0 if selected_label == "A" else 1
    return {
        "method": "vlm_ab_same_object",
        "selected_label": selected_label,
        "selected_track": normalized_ids[selected_index],
        "candidate_tracks": {"A": normalized_ids[0], "B": normalized_ids[1]},
        "raw_response": response.text,
        "usage": response.usage,
    }


def decide_same_object_yes_no(
    *,
    config: ModelConfig,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    candidate_image: np.ndarray,
    candidate_mask: np.ndarray,
    reference_id: str,
    candidate_id: str,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_reference_id = str(reference_id)
    normalized_candidate_id = str(candidate_id)
    if not normalized_reference_id or not normalized_candidate_id:
        raise CrossSegmentIdentityError(
            "cross-segment YES/NO ids must be non-empty strings"
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            image_paths = [directory / "REFERENCE.png", directory / "CANDIDATE.png"]
            _write_crop(image_paths[0], reference_image, reference_mask)
            _write_crop(image_paths[1], candidate_image, candidate_mask)
            response = answer_with_image_files(
                config=config,
                prompt=CROSS_SEGMENT_IDENTITY_YES_NO_PROMPT,
                image_paths=image_paths,
                image_labels=["REFERENCE", "CANDIDATE"],
                request_context=request_context,
            )
    except CrossSegmentIdentityError:
        raise
    except Exception as exc:
        raise CrossSegmentIdentityError(
            f"cross-segment YES/NO VLM request failed: {type(exc).__name__}: {exc}"
        ) from exc
    linked = parse_yes_no_response(response.text)
    return {
        "method": "vlm_yes_no_same_object",
        "linked": linked,
        "reference_track": normalized_reference_id,
        "candidate_track": normalized_candidate_id,
        "raw_response": response.text,
        "usage": response.usage,
    }
