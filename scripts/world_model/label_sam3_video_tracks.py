from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.query import answer_with_image_files, extract_answer_tag
from utils.config import build_model_config


TRACK_VLM_LABELING_DECISION_ID = "LBL-001.track_vlm_labeling"
CLEVRER_TRACK_VLM_LABELING_ROUTE = (
    "label.vlm_geometry_basic"
)
TRACK_VLM_LABELING_ROUTE_BY_BENCHMARK = {
    "clevrer": CLEVRER_TRACK_VLM_LABELING_ROUTE,
    "physion_pp": PHYSION_PP_TRACK_VLM_LABELING_ROUTE,
}


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_object(text: str) -> dict[str, Any]:
    answer = extract_answer_tag(text) or text
    match = JSON_OBJECT_PATTERN.search(answer)
    if not match:
        raise ValueError("No JSON object found in VLM response.")
    return json.loads(match.group(0))


GEOMETRY_TYPES = {"sphere", "box", "cylinder", "irregular"}
MATERIALS = {"metal", "rubber"}


def _normalize_geometry(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"cube", "cuboid"}:
        return "box"
    return text if text in GEOMETRY_TYPES else "irregular"


def _normalize_material(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in MATERIALS else "rubber"


def _normalize_color(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text or "unknown"


def _clamp_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _bool_or_false(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _track_summary(label_inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Path], list[str]]:
    tracks = []
    image_paths = []
    image_labels = []
    for item in label_inputs.get("tracks", []):
        if not isinstance(item, dict):
            continue
        track_id = str(item.get("track_id") or "").strip()
        image = item.get("representative_image_path")
        if not track_id or not image or not Path(image).exists():
            continue
        representative_frame = item.get("representative_frame") if isinstance(item.get("representative_frame"), dict) else {}
        tracks.append(
            {
                "track_id": track_id,
                "concept_id": item.get("concept_id"),
                "sam3_prompt": item.get("prompt"),
                "expected_object_ids_from_prompt": item.get("expected_object_ids", []),
                "frame_count": item.get("frame_count"),
                "first_frame_index": item.get("first_frame_index"),
                "last_frame_index": item.get("last_frame_index"),
                "representative_frame": {
                    "frame_index": representative_frame.get("frame_index"),
                    "area": representative_frame.get("area"),
                    "bbox_xyxy": representative_frame.get("bbox_xyxy"),
                    "centroid_xy": representative_frame.get("centroid_xy"),
                    "selection": representative_frame.get("selection"),
                },
            }
        )
        image_paths.append(Path(image))
        image_labels.append(track_id)
    return tracks, image_paths, image_labels


def _labeling_prompt_route(object_plan: dict[str, Any]) -> str:
    special_scene = object_plan.get("special_scene")
    special_scene = special_scene if isinstance(special_scene, dict) else {}
    route_record = special_scene.get("track_vlm_labeling_route")
    if not isinstance(route_record, dict):
        raise ValueError("labeling object plan is missing its track-VLM-labeling route")
    if route_record.get("decision_id") != TRACK_VLM_LABELING_DECISION_ID:
        raise ValueError(
            "track-VLM-labeling route record has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("track-VLM-labeling route record is missing its context")
    benchmark = str(context.get("benchmark") or "").strip().lower()
    expected_route = TRACK_VLM_LABELING_ROUTE_BY_BENCHMARK.get(benchmark)
    route = str(route_record.get("route") or "")
    if route != expected_route:
        raise ValueError(
            "track-VLM-labeling route does not match its benchmark: "
            f"{route!r} != {expected_route!r}"
        )
    return route


def _build_prompt(
    *, object_plan: dict[str, Any], label_inputs: dict[str, Any]
) -> tuple[str, list[Path], list[str]]:
    tracks, image_paths, image_labels = _track_summary(label_inputs)
    prompt_route = _labeling_prompt_route(object_plan)
    if prompt_route == PHYSION_PP_TRACK_VLM_LABELING_ROUTE:
        # Physion++ props include concave/hollow objects (bowls, cones, rings); state the
        # solid-primitive replacement consequence so those land on "irregular".
        example_tail = (
            "    },\n"
            "    {\n"
            '      "track_id": "anonymous_track_001",\n'
            '      "description": "green ceramic bowl",\n'
            '      "geometry_type": "irregular",\n'
            '      "appearance": {"color": "green", "material": "rubber", "material_confidence": 0.7, "reasoning": "brief visual evidence"},\n'
            '      "confidence": 0.9,\n'
            '      "is_false_positive": false,\n'
            '      "reason": "concave container, keep reconstructed mesh"\n'
            "    }\n"
        )
        geometry_rules = (
            'geometry_type must be one of "sphere", "box", "cylinder", or "irregular".\n'
            "geometry_type is not a visual description: it decides how the object is rebuilt for physics simulation. "
            'If you answer "sphere", "box", or "cylinder", the object mesh is REPLACED by that SOLID convex primitive.\n'
            'Answer "sphere", "box", or "cylinder" only when the whole object is well approximated by that solid primitive '
            "(balls are sphere; cubes, blocks, boards, and flat mats are box; solid rods and discs are cylinder).\n"
            'Answer "irregular" whenever the object is NOT a solid convex primitive, so its own reconstructed mesh is kept: '
            "concave containers (bowls, cups, pots, basins), hollow rings or tubes, hollow or open shells such as "
            "half-spheres or a bowl seen edge-on, cones, pyramids, wedges, dumbbells, and any compound shape.\n"
            'A rounded object is "sphere" only if it is a complete solid ball; an open, cut, or shell-like rounded object '
            'is "irregular".\n'
            'For example a bowl must be "irregular", never "cylinder" — a solid cylinder would fill the bowl cavity and '
            "break the simulated physics.\n"
        )
    else:
        example_tail = "    }\n"
        geometry_rules = 'geometry_type must be one of "sphere", "box", "cylinder", or "irregular". '
    prompt = (
        "You are describing anonymous SAM3 video tracks in a physical reasoning video.\n"
        "Each image is one full-frame representative image for one anonymous track.\n"
        "The tracked object is outlined with a red contour line. There are no text labels drawn on the image.\n"
        "SAM3 has already proposed the track instances. Do not merge two tracks just because they have the same color or shape.\n"
        "Describe only the red-outlined tracked object using visual appearance, shape, color/material, and available motion metadata.\n"
        "Set is_false_positive=true only when the overlay is clearly background, reflection, shadow, or tracking noise.\n\n"
        f"Anonymous tracks:\n{json.dumps(tracks, ensure_ascii=False, indent=2)}\n\n"
        "Return compact JSON inside <answer> </answer> with this shape:\n"
        "{\n"
        '  "track_labels": [\n'
        '    {\n'
        '      "track_id": "anonymous_track_000",\n'
        '      "description": "gray metallic sphere",\n'
        '      "geometry_type": "sphere",\n'
        '      "appearance": {"color": "gray", "material": "metal", "material_confidence": 0.85, "reasoning": "brief visual evidence"},\n'
        '      "confidence": 0.85,\n'
        '      "is_false_positive": false,\n'
        '      "reason": "brief visual reason"\n'
        + example_tail
        + "  ]\n"
        "}\n"
        + geometry_rules
        + 'appearance.material must be "metal" only when the object is visibly metallic; otherwise use "rubber" as the non-metal fallback. '
        "Return exactly one label for every input track_id."
    )
    return prompt, image_paths, image_labels


def _normalize_labels(
    payload: dict[str, Any],
    ordered_track_ids: list[str],
    track_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = []
    raw_labels = payload.get("track_labels")
    if not isinstance(raw_labels, list):
        raise ValueError("VLM response must contain a track_labels list.")
    raw_by_track: dict[str, dict[str, Any]] = {}
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        track_id = str(item.get("track_id") or "").strip()
        if track_id in ordered_track_ids:
            raw_by_track[track_id] = item

    for track_id in ordered_track_ids:
        item = raw_by_track.get(track_id, {})
        appearance = item.get("appearance") if isinstance(item.get("appearance"), dict) else {}
        metadata = track_metadata.get(track_id, {})
        labels.append(
            {
                "track_id": track_id,
                "description": str(item.get("description") or f"tracked dynamic object {track_id}"),
                "geometry_type": _normalize_geometry(item.get("geometry_type")),
                "appearance": {
                    "color": _normalize_color(appearance.get("color")),
                    "material": _normalize_material(appearance.get("material")),
                    "material_confidence": _clamp_confidence(appearance.get("material_confidence")),
                    "reasoning": str(appearance.get("reasoning") or ""),
                },
                "confidence": _clamp_confidence(item.get("confidence")),
                "vlm_false_positive_suggestion": _bool_or_false(item.get("is_false_positive")),
                "reason": str(item.get("reason") or ""),
                "frame_count": metadata.get("frame_count"),
                "first_frame_index": metadata.get("first_frame_index"),
                "last_frame_index": metadata.get("last_frame_index"),
            }
        )
    return labels


def label_tracks(
    *,
    label_inputs: Path,
    object_plan: Path,
    output: Path,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    request_timeout: float,
    max_output_tokens: int,
    dry_run: bool,
    benchmark: str = "clevrer",
) -> None:
    label_payload = _load_json(label_inputs)
    object_plan_payload = _load_json(object_plan)
    prompt, image_paths, image_labels = _build_prompt(
        object_plan=object_plan_payload,
        label_inputs=label_payload,
    )
    if not image_paths:
        raise ValueError(f"No representative track images found in label inputs: {label_inputs}")

    track_metadata = {
        str(item.get("track_id")): item
        for item in label_payload.get("tracks", [])
        if isinstance(item, dict) and item.get("track_id")
    }
    result: dict[str, Any] = {
        "tool": "sam3_video_track_labels",
        "status": "dry_run" if dry_run else "ok",
        "label_inputs": str(label_inputs),
        "object_plan": str(object_plan),
        "image_paths": [str(path) for path in image_paths],
        "image_labels": image_labels,
        "labeling_mode": "track_attribute_descriptions",
        "label_input_mode": "single_representative_red_contour_image",
        "benchmark": benchmark,
        "prompt": prompt,
    }
    if dry_run:
        result["track_labels"] = []
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    config = build_model_config(
        provider=provider,
        model=model,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
        require_api_key=True,
    )
    response = answer_with_image_files(
        config=config,
        prompt=prompt,
        image_paths=image_paths,
        image_labels=image_labels,
        request_context={"stage": "sam3_video_track_labels"},
    )
    parsed = _parse_json_object(response.text)
    result["raw_response"] = response.text
    result["usage"] = response.usage
    result["model_config"] = config.to_safe_dict()
    result["track_labels"] = _normalize_labels(parsed, image_labels, track_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-inputs", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bench", default="clevrer")
    args = parser.parse_args()
    label_tracks(
        label_inputs=Path(args.label_inputs),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        max_output_tokens=args.max_output_tokens,
        dry_run=args.dry_run,
        benchmark=args.bench,
    )


if __name__ == "__main__":
    main()
