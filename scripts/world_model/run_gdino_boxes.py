"""Grounding DINO box detection for Physion++ recognition.

Runs the per-scenario text query on selected video frames and applies area,
frame-edge, and same-label overlap filters. The resulting JSON box
manifest is intended as SAM3 visual-prompt input (add_new_points_or_box);
no SAM3 text prompts are needed on this path.

Weights live in data/models/grounding-dino-base (see docs/tool_env.md).
model.safetensors is required.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.specs import (  # noqa: E402
    GDINO_BOX_AREA_CAP_FRACTION,
    GDINO_CONFIDENCE_THRESHOLD,
    GDINO_EDGE_MARGIN_PX,
    GDINO_EDGE_SIDES_TO_DROP,
    GDINO_SAME_LABEL_NMS_IOU,
    GDINO_VOCAB_BY_PHYSION_PP_SCENARIO,
)

DEFAULT_GDINO_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "grounding-dino-base"


def build_query(phrases: list[str]) -> str:
    return " . ".join(phrases) + " ."


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def load_gdino(model_dir: Path = DEFAULT_GDINO_MODEL_DIR, device: Optional[str] = None) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Grounding DINO weights not found at {model_dir} (model.safetensors required; "
            "see docs/tool_env.md)."
        )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_dir)).to(resolved_device).eval()
    return processor, model, resolved_device


def detect_boxes(
    processor: Any,
    model: Any,
    image_bgr: "Any",
    query: str,
    *,
    confidence: float = GDINO_CONFIDENCE_THRESHOLD,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=confidence,
        text_threshold=confidence,
        target_sizes=[image.size[::-1]],
    )[0]
    labels = results["text_labels"] if "text_labels" in results else results["labels"]
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        detections.append(
            {"bbox_xyxy": [x1, y1, x2, y2], "class": str(label), "score": round(float(score), 4)}
        )
    return detections


def _edge_touch_count(
    box: list[int],
    image_width: int,
    image_height: int,
    *,
    edge_margin_px: int,
) -> int:
    x1, y1, x2, y2 = box
    return (
        (x1 <= edge_margin_px)
        + (y1 <= edge_margin_px)
        + (x2 >= image_width - 1 - edge_margin_px)
        + (y2 >= image_height - 1 - edge_margin_px)
    )


def cleanup_boxes(
    detections: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    area_cap_fraction: float,
    nms_iou: float,
    edge_margin_px: int,
    edge_sides_to_drop: int,
) -> list[dict[str, Any]]:
    """Drop frame-spanning boxes, then apply same-label NMS."""
    area_cap = area_cap_fraction * image_width * image_height
    kept = [
        d for d in detections
        if (d["bbox_xyxy"][2] - d["bbox_xyxy"][0]) * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]) <= area_cap
        and _edge_touch_count(
            d["bbox_xyxy"],
            image_width,
            image_height,
            edge_margin_px=edge_margin_px,
        )
        < edge_sides_to_drop
    ]
    kept.sort(key=lambda d: -d["score"])
    deduped: list[dict[str, Any]] = []
    for candidate in kept:
        if all(
            candidate["class"] != other["class"]
            or box_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) < nms_iou
            for other in deduped
        ):
            deduped.append(candidate)
    return deduped


def run_gdino_boxes(
    *,
    video: Path,
    scenario: str,
    output: Path,
    frame_indices: list[int],
    model_dir: Path = DEFAULT_GDINO_MODEL_DIR,
    confidence: float = GDINO_CONFIDENCE_THRESHOLD,
    processor: Any = None,
    model: Any = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    if scenario not in GDINO_VOCAB_BY_PHYSION_PP_SCENARIO:
        raise ValueError(
            f"No GDINO vocabulary for scenario {scenario!r}; expected one of "
            f"{sorted(GDINO_VOCAB_BY_PHYSION_PP_SCENARIO)}"
        )
    started = time.perf_counter()
    if processor is None or model is None:
        processor, model, device = load_gdino(model_dir, device)
    query = build_query(GDINO_VOCAB_BY_PHYSION_PP_SCENARIO[scenario])

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video}")
    frames_payload = []
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to read frame {frame_index} from {video}")
            raw = detect_boxes(processor, model, frame, query, confidence=confidence, device=device)
            cleaned = cleanup_boxes(
                raw,
                image_width=width,
                image_height=height,
                area_cap_fraction=GDINO_BOX_AREA_CAP_FRACTION,
                nms_iou=GDINO_SAME_LABEL_NMS_IOU,
                edge_margin_px=GDINO_EDGE_MARGIN_PX,
                edge_sides_to_drop=GDINO_EDGE_SIDES_TO_DROP,
            )
            frames_payload.append(
                {"frame_index": int(frame_index), "raw_box_count": len(raw), "boxes": cleaned}
            )
    finally:
        capture.release()

    payload = {
        "tool": "gdino_boxes",
        "video": str(video),
        "scenario": scenario,
        "query": query,
        "confidence_threshold": confidence,
        "cleanup": {
            "area_cap_fraction": GDINO_BOX_AREA_CAP_FRACTION,
            "same_label_nms_iou": GDINO_SAME_LABEL_NMS_IOU,
        },
        "model_dir": str(model_dir),
        "frames": frames_payload,
        "elapsed_sec": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-indices", type=str, default="0", help="comma-separated frame indices")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_GDINO_MODEL_DIR)
    parser.add_argument("--confidence", type=float, default=GDINO_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    frame_indices = [int(item) for item in args.frame_indices.split(",") if item.strip()]
    payload = run_gdino_boxes(
        video=args.video,
        scenario=args.scenario,
        output=args.output,
        frame_indices=frame_indices,
        model_dir=args.model_dir,
        confidence=args.confidence,
    )
    for frame in payload["frames"]:
        print(f"frame {frame['frame_index']}: {frame['raw_box_count']} raw -> {len(frame['boxes'])} boxes")
    print(f"output -> {args.output}")


if __name__ == "__main__":
    main()
