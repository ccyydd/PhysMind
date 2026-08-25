from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.world_model.run_sam3 import (
    _add_sam3_to_path,
    _mask_for_image,
    _mask_stats,
    _masks_from_response,
    _patch_start_session_for_current_sam3,
)
from utils.config import build_model_config


BENCH_CONCEPT_PROMPTS = {
    "clevrer": [
        {
            "object_id": "clevrer_dynamic_objects",
            "concept_id": "clevrer_dynamic_objects",
            "prompt": "all visible objects",
            "expected_object_ids": [],
            "reason": "CLEVRER dynamic object category",
        },
    ],
}
TRACKING_FAMILY_DECISION_ID = "TRK-001.tracking_family"
CLEVRER_TRACKING_FAMILY_ROUTE = "tracking.sam3_broad_text"
FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_role_box_point"
)
BOUNCY_WALL_TRACKING_FAMILY_ROUTE = "tracking.gdino_two_segment_point_probe"
BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_per_part_seeds"
)
FRICTION_COLLISION_TRACKING_FAMILY_ROUTE = "tracking.sam3_two_segment_text"
MASS_COLLISION_TRACKING_FAMILY_ROUTE = (
    "tracking.gdino_two_segment_boxes"
)
TRACKING_FAMILY_ROUTES = frozenset(
    {CLEVRER_TRACKING_FAMILY_ROUTE}
    | set(PHYSION_PP_TRACKING_FAMILY_ROUTE_BY_SCENARIO.values())
)
TEMPORAL_PARTITION_DECISION_ID = "INP-003.temporal_partition"
SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE = "temporal.single_video"
SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE = (
    "temporal.cue_two_segment"
)
TEMPORAL_PARTITION_ROUTES = frozenset(
    {
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE,
        SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
    }
)
TEMPORAL_PARTITION_ROUTE_BY_TRACKING_FAMILY = {
    CLEVRER_TRACKING_FAMILY_ROUTE: SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE,
    FRICTION_PLATFORM_TRACKING_FAMILY_ROUTE: (
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    BOUNCY_PLATFORM_TRACKING_FAMILY_ROUTE: (
        SINGLE_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    BOUNCY_WALL_TRACKING_FAMILY_ROUTE: SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
    FRICTION_COLLISION_TRACKING_FAMILY_ROUTE: (
        SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE
    ),
    MASS_COLLISION_TRACKING_FAMILY_ROUTE: SPLIT_VIDEO_TEMPORAL_PARTITION_ROUTE,
}
CUE_ROLE_BINDING_DECISION_ID = "TRK-002.cue_role_binding"
CUE_ROLE_BINDING_ROUTE = "role_binding.cue_rounds"
STATIC_ROLE_ASSIGNMENT_DECISION_ID = "TRK-003.static_role_assignment"
STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO = {
    "friction_platform_pp": "static_roles.platform_displacement",
    "bouncy_wall_pp": "static_roles.wall_patient_two_segment",
    "bouncy_platform_pp": "static_roles.bounce_platform_displacement",
    "friction_collision_pp": "static_roles.collision_structure",
    "mass_collision_pp": "static_roles.mass_structure",
}
STATIC_ROLE_ASSIGNMENT_ROUTES = frozenset(
    STATIC_ROLE_ASSIGNMENT_ROUTE_BY_SCENARIO.values()
)
CROSS_SEGMENT_IDENTITY_DECISION_ID = "TRK-004.cross_segment_identity"
CROSS_SEGMENT_IDENTITY_ROUTE = "identity.vlm_ab"
CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO = {
    "bouncy_wall_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
    "friction_collision_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
    "mass_collision_pp": CROSS_SEGMENT_IDENTITY_ROUTE,
}
CROSS_SEGMENT_IDENTITY_ROUTES = frozenset(
    CROSS_SEGMENT_IDENTITY_ROUTE_BY_SCENARIO.values()
)

# Mass-collision tracking detects the curtain interval from full-frame change, seeds the
# ball/agent/patient independently, binds the agent across segments through VLM identity,
# and removes border or far-depth background tracks.




def _video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    metadata = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def _detect_use_fa3() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 9


def _load_model(
    *,
    checkpoint: str | None,
    sam3_version: str,
    compile_model: bool,
    max_num_objects: int,
    async_loading_frames: bool,
) -> tuple[Any, bool]:
    username = os.getenv("USER", "physmind")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}")
    _add_sam3_to_path()
    import torch
    from sam3 import build_sam3_predictor
    from sam3.model_builder import build_sam3_video_predictor

    use_fa3 = _detect_use_fa3()
    if sam3_version == "sam3":
        build_kwargs: dict[str, Any] = {
            "compile": compile_model,
            "async_loading_frames": async_loading_frames,
        }
        if checkpoint:
            build_kwargs["checkpoint_path"] = checkpoint
        model = build_sam3_video_predictor(**build_kwargs)
    else:
        build_kwargs = {
            "version": "sam3.1",
            "compile": compile_model,
            "warm_up": compile_model,
            "max_num_objects": max_num_objects,
            "async_loading_frames": async_loading_frames,
            "use_fa3": use_fa3,
        }
        if checkpoint:
            build_kwargs["checkpoint_path"] = checkpoint
        model = build_sam3_predictor(**build_kwargs)
        _patch_start_session_for_current_sam3(model)
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    return model, use_fa3


def _object_prompts(object_plan: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = []
    for index, item in enumerate(object_plan.get("target_objects", []), start=1):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or f"obj_{index}")
        prompt = str(item.get("description") or item.get("role") or object_id).strip()
        if not prompt:
            prompt = object_id
        prompts.append(
            {
                "object_id": object_id,
                "sam_obj_id": index,
                "prompt": prompt,
            }
        )
    return prompts


def _generic_movable_prompt(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "object_id": "track_prompt_movable",
            "sam_obj_id": 1,
            "prompt": prompt,
        }
    ]


def _vlm_concept_prompts(object_plan: dict[str, Any]) -> list[dict[str, Any]]:
    special_scene = object_plan.get("special_scene") if isinstance(object_plan.get("special_scene"), dict) else {}
    raw_prompts = special_scene.get("sam3_video_tracking_prompts")
    if not isinstance(raw_prompts, list):
        raw_prompts = object_plan.get("sam3_video_tracking_prompts")
    if not isinstance(raw_prompts, list):
        return []
    prompts = []
    for index, item in enumerate(raw_prompts, start=1):
        if not isinstance(item, dict):
            continue
        prompt = " ".join(str(item.get("prompt") or "").strip().split())
        if not prompt:
            continue
        raw_expected = item.get("expected_object_ids")
        if not isinstance(raw_expected, list):
            raw_expected = []
        prompts.append(
            {
                "object_id": str(item.get("concept_id") or f"concept_{index}"),
                "concept_id": str(item.get("concept_id") or f"concept_{index}"),
                "sam_obj_id": index,
                "prompt": prompt,
                "expected_object_ids": [str(object_id) for object_id in raw_expected],
                "reason": str(item.get("reason", "")),
            }
        )
    return prompts


def _bench_concept_prompts(bench: str | None) -> list[dict[str, Any]]:
    if not bench:
        return []
    prompts = BENCH_CONCEPT_PROMPTS.get(str(bench).strip().lower())
    if not prompts:
        return []
    return [
        {
            **item,
            "sam_obj_id": index,
        }
        for index, item in enumerate(prompts, start=1)
    ]


def _select_prompts(
    *,
    object_plan_payload: dict[str, Any],
    mode: str,
    bench: str | None,
    generic_prompt: str,
) -> tuple[list[dict[str, Any]], str]:
    if mode == "auto":
        prompts = _bench_concept_prompts(bench)
        if prompts:
            return prompts, "bench-concepts"
        prompts = _vlm_concept_prompts(object_plan_payload)
        if prompts:
            return prompts, "vlm-concept-prompts"
        return [], "none"
    if mode == "bench-concepts":
        return _bench_concept_prompts(bench), "bench-concepts"
    if mode == "generic-movable":
        return _generic_movable_prompt(generic_prompt), "generic-movable"
    if mode == "vlm-concept-prompts":
        return _vlm_concept_prompts(object_plan_payload), "vlm-concept-prompts"
    return _object_prompts(object_plan_payload), "object-prompts"




def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _object_plan_scene_metadata(object_plan_payload: dict[str, Any]) -> dict[str, Any]:
    special_scene = object_plan_payload.get("special_scene")
    if not isinstance(special_scene, dict):
        return {}
    metadata = special_scene.get("scene_metadata")
    return metadata if isinstance(metadata, dict) else {}


























def _records_by_object(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    records_by_object: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        object_id = str(record.get("object_id") or "")
        if object_id:
            records_by_object.setdefault(object_id, []).append(record)
    return records_by_object


def _read_video_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for Physion++ yellow patch validation: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read frame {frame_index} for Physion++ yellow patch validation: {video}")
        return frame
    finally:
        capture.release()


def _score_by_obj_id(response: dict[str, Any]) -> dict[int, float | None]:
    import torch

    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    obj_ids = [int(item) for item in np.asarray(obj_ids).reshape(-1)]
    scores = outputs.get("out_scores")
    if scores is None:
        scores = outputs.get("scores")
    if scores is None:
        return {obj_id: None for obj_id in obj_ids}
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    return {obj_id: (float(scores[index]) if index < len(scores) else None) for index, obj_id in enumerate(obj_ids)}


def _append_records_from_response(
    *,
    response: dict[str, Any],
    object_id_by_sam_obj_id: dict[int, str],
    concept_id_by_sam_obj_id: dict[int, str],
    prompt_by_sam_obj_id: dict[int, str],
    expected_object_ids_by_sam_obj_id: dict[int, list[str]],
    frame_index: int,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> None:
    masks_by_obj_id = _masks_from_response(response)
    scores_by_obj_id = _score_by_obj_id(response)
    for sam_obj_id, mask in sorted(masks_by_obj_id.items()):
        object_id = object_id_by_sam_obj_id.get(int(sam_obj_id), f"anonymous_track_{int(sam_obj_id):03d}")
        mask_key = f"{object_id}__frame_{frame_index:05d}__sam_obj_{int(sam_obj_id)}"
        mask_u8 = mask.astype(np.uint8)
        all_arrays[mask_key] = mask_u8
        all_records.append(
            {
                "object_id": object_id,
                "concept_id": concept_id_by_sam_obj_id.get(int(sam_obj_id)),
                "prompt": prompt_by_sam_obj_id.get(int(sam_obj_id)),
                "expected_object_ids": expected_object_ids_by_sam_obj_id.get(int(sam_obj_id), []),
                "frame_index": int(frame_index),
                "sam_object_id": int(sam_obj_id),
                "mask_key": mask_key,
                "score": scores_by_obj_id.get(int(sam_obj_id)),
                **_mask_stats(mask_u8),
            }
        )


def _track_object_id(
    *,
    resolved_mode: str,
    prompt_item: dict[str, Any],
    sam_obj_id: int,
) -> str:
    if resolved_mode in {"bench-concepts", "generic-movable", "vlm-concept-prompts"}:
        return f"anonymous_track_{sam_obj_id:03d}"
    return str(prompt_item.get("object_id") or f"object_{sam_obj_id:03d}")


def _run_joint_prompt_tracking(
    *,
    model: Any,
    video: Path,
    prompts: list[dict[str, Any]],
    resolved_mode: str,
    prompt_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    all_records: list[dict[str, Any]],
    all_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    session_response = model.handle_request({"type": "start_session", "resource_path": str(video)})
    session_id = session_response["session_id"]
    object_id_by_sam_obj_id: dict[int, str] = {}
    concept_id_by_sam_obj_id: dict[int, str] = {}
    prompt_by_sam_obj_id: dict[int, str] = {}
    expected_object_ids_by_sam_obj_id: dict[int, list[str]] = {}
    prompt_responses = []
    try:
        for prompt_index, item in enumerate(prompts, start=1):
            request = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "text": item["prompt"],
            }
            if resolved_mode == "object-prompts":
                request["obj_id"] = int(item["sam_obj_id"])
            prompt_response = model.handle_request(request)
            masks_by_prompt_obj_id = _masks_from_response(prompt_response)
            for sam_obj_id in masks_by_prompt_obj_id:
                sam_obj_id_int = int(sam_obj_id)
                object_id_by_sam_obj_id[sam_obj_id_int] = _track_object_id(
                    resolved_mode=resolved_mode,
                    prompt_item=item,
                    sam_obj_id=sam_obj_id_int,
                )
                concept_id_by_sam_obj_id[sam_obj_id_int] = str(item.get("concept_id") or item.get("object_id"))
                prompt_by_sam_obj_id[sam_obj_id_int] = str(item["prompt"])
                expected_object_ids_by_sam_obj_id[sam_obj_id_int] = [
                    str(object_id) for object_id in item.get("expected_object_ids", [])
                ]
            prompt_responses.append(
                {
                    "object_id": item["object_id"],
                    "concept_id": item.get("concept_id"),
                    "sam_object_id": item["sam_obj_id"],
                    "prompt": item["prompt"],
                    "frame_index": prompt_response.get("frame_index"),
                    "mask_count": len(masks_by_prompt_obj_id),
                    "prompt_index": prompt_index,
                    "expected_object_ids": item.get("expected_object_ids", []),
                }
            )

        stream_request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": prompt_frame_index,
        }
        if max_frame_num_to_track is not None:
            stream_request["max_frame_num_to_track"] = int(max_frame_num_to_track)

        seen_frame_indices = set()
        for response in model.handle_stream_request(stream_request):
            frame_index = int(response["frame_index"])
            if frame_index in seen_frame_indices:
                continue
            seen_frame_indices.add(frame_index)
            _append_records_from_response(
                response=response,
                object_id_by_sam_obj_id=object_id_by_sam_obj_id,
                concept_id_by_sam_obj_id=concept_id_by_sam_obj_id,
                prompt_by_sam_obj_id=prompt_by_sam_obj_id,
                expected_object_ids_by_sam_obj_id=expected_object_ids_by_sam_obj_id,
                frame_index=frame_index,
                all_records=all_records,
                all_arrays=all_arrays,
            )
    finally:
        model.handle_request({"type": "close_session", "session_id": session_id})
    return prompt_responses






























































































































def run_sam3_video_tracks(
    *,
    video: Path,
    object_plan: Path,
    output: Path,
    checkpoint: str | None,
    sam3_version: str,
    mode: str,
    bench: str | None,
    generic_prompt: str,
    prompt_frame_index: int,
    compile_model: bool,
    max_num_objects: int,
    async_loading_frames: bool,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    model: Any | None = None,
    use_fa3: bool | None = None,
) -> None:
    task_start = time.perf_counter()
    metadata = _video_metadata(video)
    object_plan_payload = json.loads(object_plan.read_text(encoding="utf-8"))
    if is_physion_pp:
        resolved_mode = "physion-multi-round"
    else:
        prompts, resolved_mode = _select_prompts(
            object_plan_payload=object_plan_payload,
            mode=mode,
            bench=bench,
            generic_prompt=generic_prompt,
        )
    tracking_family_record = _resolve_tracking_family_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    temporal_partition_record = _resolve_temporal_partition_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    cue_role_binding_record = _resolve_cue_role_binding_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    static_role_assignment_record = _resolve_static_role_assignment_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    cross_segment_identity_record = _resolve_cross_segment_identity_dispatch(
        bench=bench,
        scenario_name=physion_scenario_name,
        object_plan_payload=object_plan_payload,
    )
    tracking_family_route = (
        tracking_family_record.get("route")
        if tracking_family_record is not None
        else None
    )
    temporal_partition_route = (
        temporal_partition_record.get("route")
        if temporal_partition_record is not None
        else None
    )
    tracking_dispatch = _tracking_dispatch_name(
        is_physion_pp=is_physion_pp,
        scenario_name=physion_scenario_name,
        tracking_family_route=tracking_family_route,
        temporal_partition_route=temporal_partition_route,
    )
    tracking_module_profile = None
    if tracking_dispatch in {
        "friction_platform_tracking",
        "bouncy_platform_tracking",
        "bouncy_wall_tracking",
        "mass_collision_tracking",
    }:
        tracking_module_profile = default_module_profile_policy().resolve_tracking(
            str(tracking_family_route),
            scenario=str(physion_scenario_name),
        )
    if not prompts:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "tool": "sam3_video_tracks",
                    "status": "skipped",
                    "reason": "no_bench_or_vlm_concept_prompts",
                    "video": str(video),
                    "object_plan": str(object_plan),
                    "bench": bench,
                    "requested_mode": mode,
                    "resolved_mode": resolved_mode,
                    "physion_scenario": physion_scenario_name,
                    "tracking_family_route": tracking_family_record,
                    "temporal_partition_route": temporal_partition_record,
                    "cue_role_binding_route": cue_role_binding_record,
                    "static_role_assignment_route": static_role_assignment_record,
                    "cross_segment_identity_route": cross_segment_identity_record,
                    "prompts": [],
                    "track_record_count": 0,
                    "tracked_frame_count": 0,
                    "track_count_by_object": {},
                    "tracks": [],
                    "tracks_by_object": {},
                    "note": "No bench-level concepts or VLM-provided SAM3 video tracking concepts were available.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    if metadata["frame_count"] <= 0:
        raise ValueError(f"Could not read frame count from video: {video}")
    if not (0 <= prompt_frame_index < metadata["frame_count"]):
        raise ValueError(
            f"prompt_frame_index={prompt_frame_index} is outside valid range [0, {metadata['frame_count'] - 1}]"
        )
    if max_num_objects < len(prompts):
        raise ValueError(f"max_num_objects={max_num_objects} is smaller than prompt count {len(prompts)}")

    if model is None:
        model, use_fa3 = _load_model(
            checkpoint=checkpoint,
            sam3_version=sam3_version,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
            async_loading_frames=async_loading_frames,
        )
    all_records: list[dict[str, Any]] = []
    all_arrays: dict[str, np.ndarray] = {}
    physion_tracking: dict[str, Any] | None = None

    if cue_role_binding_record is not None:
        _record_cue_role_binding_result(
            physion_tracking,
            cue_role_binding_record,
        )
    if static_role_assignment_record is not None:
        _record_static_role_assignment_result(
            physion_tracking,
            static_role_assignment_record,
        )
    if cross_segment_identity_record is not None:
        _record_cross_segment_identity_result(
            physion_tracking,
            cross_segment_identity_record,
        )

    sidecar_path = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if all_arrays:
        np.savez_compressed(sidecar_path, **all_arrays)

    records_by_object: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        records_by_object.setdefault(str(record["object_id"]), []).append(record)

    debug_artifacts: dict[str, Any] = {}
    if os.getenv("PHYSMIND_DEBUG_ARTIFACTS") == "1" and all_arrays:
        debug_videos = write_sam3_video_track_overlay_videos(
            video=video,
            records_by_object=records_by_object,
            mask_sidecar=sidecar_path,
            output_dir=output.parent / "debug" / "videos",
            fps=float(metadata.get("fps") or 0.0),
        )
        debug_artifacts["track_overlay_videos"] = debug_videos

    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sam3-version", choices=["sam3", "sam3.1"], default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "bench-concepts", "object-prompts", "generic-movable", "vlm-concept-prompts"],
        default="auto",
    )
    parser.add_argument("--bench", default=None)
    parser.add_argument(
        "--generic-prompt",
        default="all movable objects on the floor, including balls, cylinders, and cubes",
    )
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=8)
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument(
        "--propagation-direction",
        choices=["both", "forward", "backward"],
        default="forward",
    )
    parser.add_argument("--max-frame-num-to-track", type=int, default=None)
    args = parser.parse_args()
    run_sam3_video_tracks(
        video=Path(args.video),
        object_plan=Path(args.object_plan),
        output=Path(args.output),
        checkpoint=args.checkpoint,
        sam3_version=args.sam3_version or (
            "sam3" if args.mode in {"auto", "bench-concepts", "generic-movable", "vlm-concept-prompts"} else "sam3.1"
        ),
        mode=args.mode,
        bench=args.bench,
        generic_prompt=args.generic_prompt,
        prompt_frame_index=args.prompt_frame_index,
        compile_model=args.compile,
        max_num_objects=args.max_num_objects,
        async_loading_frames=args.async_loading_frames,
        propagation_direction=args.propagation_direction,
        max_frame_num_to_track=args.max_frame_num_to_track,
    )


if __name__ == "__main__":
    main()
