from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Optional

from agent.direct_answer import (
    CLEVRER_DIRECT_ANSWER_PROMPT_FAMILY_ROUTE,
    PHYSION_PP_DIRECT_ANSWER_SCENARIOS,
    _direct_answer_input_materialization,
    _physion_pp_input_materialization,
    _require_direct_answer_prompt_family,
    _require_physion_pp_cue_input,
    _resolve_direct_answer_prompt_family,
    _resolve_physion_pp_cue_input,
    predict_physion_pp_ocp,
)
from agent.query import answer_video_question, create_openai_compatible_client, extract_answer_tag
from agent.world_model.artifacts import ArtifactManager
from agent.world_model.route_policy import (
    RouteContext,
    RoutePolicy,
    RoutePolicyValidationError,
    load_route_policy,
)
from agent.world_model.schemas import ToolResult
from benchmark.clevrer import ClevrerQuestion, ClevrerScene
from benchmark.physion_pp import PhysionPPQuestion, PhysionPPScene
from benchmark.prompts import build_descriptive_prompt, build_multiple_choice_prompt
from benchmark.specs import CLEVRER_CHOICE_POLICY
from utils.config import ModelConfig


SUCCESS_ANSWER_BACKEND_DECISION_ID = "ANS-001.success_answer_backend"
SUCCESS_ANSWER_BACKEND_ROUTES = frozenset(
    {
        "answer.vlm_tools",
        "answer.contact_artifact",
    }
)
PHYSION_PP_SUCCESS_ANSWER_SCENARIOS = frozenset(
    {
        "friction_platform_pp",
        "bouncy_wall_pp",
        "bouncy_platform_pp",
        "friction_collision_pp",
        "mass_collision_pp",
    }
)
WORLD_MODEL_ANSWER_FALLBACK_DECISION_ID = (
    "FAIL-001.world_model_answer_fallback"
)
WORLD_MODEL_ANSWER_FALLBACK_ROUTE = "answer.direct_fallback"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def _answer_policy_context(
    scene: Any,
    question: Any,
) -> RouteContext | None:
    if isinstance(scene, PhysionPPScene):
        benchmark = "physion_pp"
        scenario = str(scene.scenario or "").strip() or None
    elif isinstance(scene, ClevrerScene):
        benchmark = "clevrer"
        scenario = None
    else:
        return None
    return RouteContext(
        benchmark=benchmark,
        scenario=scenario,
        question_type=(
            str(getattr(question, "question_type", "") or "").strip()
            or None
        ),
    )


def _resolve_success_answer_backend_route(
    scene: Any,
    question: Any,
    *,
    policy: RoutePolicy | None = None,
) -> dict[str, Any] | None:
    context = _answer_policy_context(scene, question)
    if context is None:
        return None
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        SUCCESS_ANSWER_BACKEND_DECISION_ID,
        context,
    )
    if resolved_route.get("route") not in SUCCESS_ANSWER_BACKEND_ROUTES:
        raise RoutePolicyValidationError(
            "unsupported success-answer backend route: "
            f"{resolved_route.get('route')!r}"
        )
    return resolved_route


def _require_success_answer_backend_route(
    route_record: dict[str, Any] | None,
    *,
    scene: Any,
    question: Any,
) -> dict[str, Any] | None:
    context = _answer_policy_context(scene, question)
    if context is None:
        if route_record is not None:
            raise RoutePolicyValidationError(
                "success-answer backend route is not applicable to this benchmark"
            )
        return None
    if route_record is None:
        raise RoutePolicyValidationError(
            "missing ANS-001 success-answer backend route record"
        )
    if route_record.get("decision_id") != SUCCESS_ANSWER_BACKEND_DECISION_ID:
        raise RoutePolicyValidationError(
            "success-answer backend route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    if route not in SUCCESS_ANSWER_BACKEND_ROUTES:
        raise RoutePolicyValidationError(
            f"unsupported success-answer backend route: {route!r}"
        )
    record_context = route_record.get("context")
    if not isinstance(record_context, dict):
        raise RoutePolicyValidationError(
            "success-answer backend route is missing context"
        )
    expected_context = {
        "benchmark": context.benchmark,
        "scenario": context.scenario,
        "question_type": context.question_type,
    }
    observed_context = {
        key: record_context.get(key)
        for key in expected_context
    }
    if observed_context != expected_context:
        raise RoutePolicyValidationError(
            "success-answer backend route context mismatch: "
            f"{observed_context!r} != {expected_context!r}"
        )
    expected_route = (
        "answer.vlm_tools"
        if context.benchmark == "clevrer"
        else "answer.contact_artifact"
        if context.scenario in PHYSION_PP_SUCCESS_ANSWER_SCENARIOS
        else None
    )
    if route != expected_route:
        raise RoutePolicyValidationError(
            "success-answer backend route does not match benchmark/scenario: "
            f"{route!r} != {expected_route!r}"
        )
    return route_record


def _resolve_world_model_answer_fallback_route(
    scene: Any,
    question: Any,
    *,
    policy: RoutePolicy | None = None,
) -> dict[str, Any] | None:
    context = _answer_policy_context(scene, question)
    if context is None:
        return None
    active_policy = policy or load_route_policy()
    resolved_route = active_policy.resolve_record(
        WORLD_MODEL_ANSWER_FALLBACK_DECISION_ID,
        context,
    )
    if resolved_route.get("route") != WORLD_MODEL_ANSWER_FALLBACK_ROUTE:
        raise RoutePolicyValidationError(
            "unsupported world-model answer-fallback route: "
            f"{resolved_route.get('route')!r}"
        )
    return resolved_route


def _require_world_model_answer_fallback_route(
    route_record: dict[str, Any] | None,
    *,
    scene: Any,
    question: Any,
) -> dict[str, Any]:
    context = _answer_policy_context(scene, question)
    if context is None:
        raise RoutePolicyValidationError(
            "world-model answer-fallback route is not applicable to this benchmark"
        )
    if route_record is None:
        raise RoutePolicyValidationError(
            "missing FAIL-001 world-model answer-fallback route record"
        )
    if (
        route_record.get("decision_id")
        != WORLD_MODEL_ANSWER_FALLBACK_DECISION_ID
    ):
        raise RoutePolicyValidationError(
            "world-model answer-fallback route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = str(route_record.get("route") or "")
    if route != WORLD_MODEL_ANSWER_FALLBACK_ROUTE:
        raise RoutePolicyValidationError(
            f"unsupported world-model answer-fallback route: {route!r}"
        )
    record_context = route_record.get("context")
    if not isinstance(record_context, dict):
        raise RoutePolicyValidationError(
            "world-model answer-fallback route is missing context"
        )
    expected_context = {
        "benchmark": context.benchmark,
        "scenario": context.scenario,
        "question_type": context.question_type,
    }
    observed_context = {
        key: record_context.get(key)
        for key in expected_context
    }
    if observed_context != expected_context:
        raise RoutePolicyValidationError(
            "world-model answer-fallback route context mismatch: "
            f"{observed_context!r} != {expected_context!r}"
        )
    return route_record


def _safe_json_loads(text: str) -> dict[str, Any]:
    payload = extract_answer_tag(text) or text
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(payload[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("final answer response is not a JSON object")
    return value


def _selected_choices_from_per_choice_results(parsed: dict[str, Any]) -> list[str] | None:
    per_choice = parsed.get("per_choice_results")
    if not isinstance(per_choice, list):
        return None
    selected = []
    saw_choice_decision = False
    for item in per_choice:
        if not isinstance(item, dict):
            continue
        letter = str(item.get("choice_letter") or item.get("choice") or "").strip().upper()
        decision = str(item.get("decision") or item.get("answer") or item.get("result") or "").strip().lower()
        if len(letter) != 1 or not letter.isalpha() or not decision:
            continue
        if decision in {"yes", "true", "correct", "selected", "confirmed"}:
            selected.append(letter)
            saw_choice_decision = True
        elif decision in {"no", "false", "wrong", "not_selected", "not selected", "unconfirmed"}:
            saw_choice_decision = True
    if not saw_choice_decision:
        return None
    return selected


def _extract_choice_letters_from_answer(text: str) -> list[str]:
    answer = extract_answer_tag(text)
    if answer is None:
        return []
    normalized = answer.strip().upper()
    if normalized == "NONE":
        return []
    seen = set()
    letters = []
    for part in normalized.replace(",", " ").split():
        token = part.strip()
        if len(token) == 1 and token.isalpha() and token not in seen:
            seen.add(token)
            letters.append(token)
    return letters


def _validate_fallback_video_source(
    scene: ClevrerScene | PhysionPPScene,
    video_source_route: str,
) -> None:
    if video_source_route == "video.original":
        if not isinstance(scene, ClevrerScene):
            raise ValueError(
                "original_scene_video answer fallback requires a CLEVRER scene"
            )
        return
    if video_source_route == "video.cue_annotated":
        if not isinstance(scene, PhysionPPScene):
            raise ValueError(
                "cue_annotated_original_video answer fallback requires a Physion++ scene"
            )
        return
    raise ValueError(f"unsupported answer-fallback video source route: {video_source_route!r}")


def _fallback_scene_for_video_source(
    scene: ClevrerScene | PhysionPPScene,
    video_source_route: str,
) -> ClevrerScene | PhysionPPScene:
    _validate_fallback_video_source(scene, video_source_route)
    if video_source_route == "video.original":
        return scene

    if not isinstance(scene, PhysionPPScene):
        raise TypeError(
            f"Physion++ fallback scene has unexpected type: {type(scene).__name__}"
        )
    video_path = Path(scene.video_path)
    trim_suffix = "_trimmed"
    if video_path.stem.endswith(trim_suffix):
        cue_video_path = video_path.with_name(
            f"{video_path.stem.removesuffix(trim_suffix)}{video_path.suffix}"
        )
    else:
        cue_video_path = video_path
    if not cue_video_path.exists():
        raise FileNotFoundError(f"Physion++ direct-answer cue video not found: {cue_video_path}")
    return replace(
        scene,
        video_filename=cue_video_path.name,
        video_path=cue_video_path,
        benchmark="physion_pp",
    )


def run_direct_answer_fallback(
    *,
    config: ModelConfig,
    scene: ClevrerScene | PhysionPPScene,
    question: ClevrerQuestion | PhysionPPQuestion,
    question_dir: Path,
    artifacts: ArtifactManager,
    dry_run: bool,
    pipeline_failure: dict[str, Any],
    video_source_route: str,
    world_model_answer_fallback_route: dict[str, Any] | None = None,
) -> ToolResult:
    if world_model_answer_fallback_route is None:
        world_model_answer_fallback_route = (
            _resolve_world_model_answer_fallback_route(scene, question)
        )
    validated_fallback_route = _require_world_model_answer_fallback_route(
        world_model_answer_fallback_route,
        scene=scene,
        question=question,
    )
    prompt_benchmark = (
        "physion_pp" if isinstance(scene, PhysionPPScene) else "clevrer"
    )
    prompt_scenario = (
        str(scene.scenario or "").strip() or None
        if isinstance(scene, PhysionPPScene)
        else None
    )
    prompt_family_route = None
    if prompt_benchmark == "clevrer" or (
        prompt_scenario in PHYSION_PP_DIRECT_ANSWER_SCENARIOS
    ):
        prompt_family_route = _resolve_direct_answer_prompt_family(
            prompt_benchmark,
            scenario=prompt_scenario,
            question_type=str(question.question_type or "").strip() or None,
        )
        prompt_family_route = _require_direct_answer_prompt_family(
            prompt_family_route,
            benchmark=prompt_benchmark,
            scenario=prompt_scenario,
            question_type=str(question.question_type or "").strip() or None,
        )
    physion_pp_cue_input_route = None
    if isinstance(scene, PhysionPPScene):
        physion_pp_cue_input_route = _resolve_physion_pp_cue_input()
        physion_pp_cue_input_route = _require_physion_pp_cue_input(
            physion_pp_cue_input_route,
            benchmark="physion_pp",
        )
    _validate_fallback_video_source(scene, video_source_route)
    output_path = artifacts.artifact_path(question_dir, "final_answer", "final_answer.json")
    failed_stage = str(pipeline_failure.get("failed_stage") or "pipeline")
    fallback_reason = f"{failed_stage}_failed"
    if dry_run:
        payload = {
            "tool": "final_answer",
            "backend": "answer.direct_fallback",
            "status": "dry_run",
            "tool_status": "dry_run",
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "fallback_reason": fallback_reason,
            "pipeline_failure": pipeline_failure,
            "direct_answer_video_source_route": video_source_route,
            "world_model_answer_fallback_route": deepcopy(
                validated_fallback_route
            ),
            "direct_answer_input_materialization": (
                _physion_pp_input_materialization(config, scene)
                if isinstance(scene, PhysionPPScene)
                else _direct_answer_input_materialization(config)
            ),
            "extracted_answer": None,
        }
        if prompt_family_route is not None:
            payload["direct_answer_prompt_family_route"] = deepcopy(
                prompt_family_route
            )
        if physion_pp_cue_input_route is not None:
            payload["physionpp_cue_input_route"] = deepcopy(
                physion_pp_cue_input_route
            )
        artifacts.write(output_path, payload)
        return ToolResult("final_answer", "dry_run", str(output_path), payload=payload)

    if isinstance(scene, PhysionPPScene):
        fallback_scene = _fallback_scene_for_video_source(scene, video_source_route)
        if not isinstance(fallback_scene, PhysionPPScene):
            raise TypeError(
                "Physion++ fallback video materialization returned "
                f"{type(fallback_scene).__name__}"
            )
        prediction = predict_physion_pp_ocp(
            config,
            fallback_scene,
            question,
            prompt_family_route=prompt_family_route,
        )
        parse_error = bool(prediction.get("parse_error"))
        extracted = str(prediction.get("normalized_prediction") or "").strip().lower() or None
        raw_text = str(prediction.get("raw_prediction") or prediction.get("prediction") or "")
        payload = {
            "tool": "final_answer",
            "backend": "answer.direct_fallback",
            "status": "vlm_parse_error" if parse_error else "ok",
            "tool_status": "parse_error" if parse_error else "ok",
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "fallback_reason": fallback_reason,
            "pipeline_failure": pipeline_failure,
            "direct_answer_video": str(fallback_scene.video_path),
            "direct_answer_video_source_route": video_source_route,
            "world_model_answer_fallback_route": deepcopy(
                validated_fallback_route
            ),
            "physionpp_cue_input_route": deepcopy(
                physion_pp_cue_input_route
            ),
            "direct_answer_input_materialization": prediction.get(
                "direct_answer_input_materialization"
            ),
            "extracted_answer": extracted,
            "raw_response": raw_text,
            "usage": prediction.get("usage"),
            "error_message": (
                "Physion++ direct-answer fallback did not produce yes or no"
                if parse_error
                else None
            ),
        }
        if prompt_family_route is not None:
            payload["direct_answer_prompt_family_route"] = deepcopy(
                prompt_family_route
            )
        artifacts.write(output_path, payload)
        return ToolResult(
            "final_answer",
            "parse_error" if parse_error else "ok",
            str(output_path),
            payload=payload,
        )

    if (
        prompt_family_route is None
        or prompt_family_route["route"]
        != CLEVRER_DIRECT_ANSWER_PROMPT_FAMILY_ROUTE
    ):
        raise RoutePolicyValidationError(
            "CLEVRER fallback prompt-family route does not authorize its prompt builder"
        )
    prompt = build_descriptive_prompt(question) if question.question_type == "descriptive" else build_multiple_choice_prompt(question)
    response = answer_video_question(
        config=config,
        prompt=prompt,
        video_path=scene.video_path,
        request_context={
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "stage": "pipeline_error_direct_answer_fallback",
            "failed_stage": failed_stage,
        },
    )
    raw_text = response.text
    parse_error = extract_answer_tag(raw_text) is None
    if question.question_type == "descriptive":
        extracted = extract_answer_tag(raw_text) or raw_text.strip() or None
    else:
        letters = _extract_choice_letters_from_answer(raw_text)
        extracted = ",".join(letters) if letters else "NONE"
    payload = {
        "tool": "final_answer",
        "backend": "answer.direct_fallback",
        "status": "vlm_parse_error" if parse_error else "ok",
        "tool_status": "parse_error" if parse_error else "ok",
        "scene_index": scene.scene_index,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "choices": _answer_choices_payload(question),
        "fallback_reason": fallback_reason,
        "pipeline_failure": pipeline_failure,
        "direct_answer_video_source_route": video_source_route,
        "world_model_answer_fallback_route": deepcopy(
            validated_fallback_route
        ),
        "direct_answer_input_materialization": (
            _direct_answer_input_materialization(config)
        ),
        "direct_answer_prompt_family_route": deepcopy(
            prompt_family_route
        ),
        "extracted_answer": extracted,
        "raw_response": raw_text,
        "usage": response.usage,
        "error_message": "direct-answer fallback response did not contain <answer> tags" if parse_error else None,
    }
    artifacts.write(output_path, payload)
    return ToolResult("final_answer", "parse_error" if parse_error else "ok", str(output_path), payload=payload)




def _normalize_text_tokens(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]


def _phrase_in_text(phrase: str, text: str) -> bool:
    phrase_tokens = _normalize_text_tokens(phrase)
    text_tokens = _normalize_text_tokens(text)
    if not phrase_tokens:
        return False
    for index in range(0, len(text_tokens) - len(phrase_tokens) + 1):
        if text_tokens[index : index + len(phrase_tokens)] == phrase_tokens:
            return True
    return False


def _object_reference_entries(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    reference_map = trajectory.get("object_reference_map") if isinstance(trajectory.get("object_reference_map"), dict) else {}
    entries = reference_map.get("references") if isinstance(reference_map.get("references"), list) else []
    return [item for item in entries if isinstance(item, dict) and item.get("reference") and item.get("object_id")]


def _object_phrases_by_id(trajectory: dict[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for item in _object_reference_entries(trajectory):
        object_id = str(item.get("object_id"))
        phrase = str(item.get("reference")).strip()
        if not object_id or not phrase:
            continue
        output.setdefault(object_id, [])
        if phrase not in output[object_id]:
            output[object_id].append(phrase)
    return output


def _object_phrase(object_id: Any, phrases_by_id: dict[str, list[str]]) -> str:
    phrases = phrases_by_id.get(str(object_id), [])
    if phrases:
        return sorted(phrases, key=lambda value: (-len(value), value))[0]
    return "the referenced object"


def _context_phrases_by_id(texts: list[str], trajectory: dict[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for text in texts:
        for item in _object_reference_entries(trajectory):
            phrase = str(item.get("reference") or "").strip()
            object_id = str(item.get("object_id") or "")
            if not phrase or not object_id or not _phrase_in_text(phrase, text):
                continue
            output.setdefault(object_id, [])
            if phrase not in output[object_id]:
                output[object_id].append(phrase)
    return output


def _object_phrase_for_context(
    object_id: Any,
    *,
    phrases_by_id: dict[str, list[str]],
    context_phrases_by_id: dict[str, list[str]] | None = None,
) -> str:
    object_key = str(object_id)
    context_phrases = (context_phrases_by_id or {}).get(object_key, [])
    if context_phrases:
        return sorted(context_phrases, key=lambda value: (-len(value), value))[0]
    return _object_phrase(object_key, phrases_by_id)


def _choice_object_ids(choice_text: str, trajectory: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for item in _object_reference_entries(trajectory):
        phrase = str(item.get("reference") or "")
        object_id = str(item.get("object_id") or "")
        if phrase and object_id and _phrase_in_text(phrase, choice_text):
            output.add(object_id)
    return output


def _target_object_ids(question_text: str, trajectory: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for item in _object_reference_entries(trajectory):
        phrase = str(item.get("reference") or "")
        object_id = str(item.get("object_id") or "")
        if phrase and object_id and _phrase_in_text(phrase, question_text):
            output.add(object_id)
    return output


def _rollout_removed_objects(tool_results: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in tool_results:
        if not isinstance(item, dict) or item.get("tool") not in {"remove_object", "simulate_edit"}:
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        rollout_id = str(result.get("rollout_id") or "")
        removed_object_id = str(result.get("removed_object_id") or "")
        if rollout_id and removed_object_id:
            output[rollout_id] = removed_object_id
    return output


def _append_unique(lines: list[str], line: str) -> None:
    if line and line not in lines:
        lines.append(line)


def _render_tool_evidence_record(
    record: dict[str, Any],
    *,
    phrases_by_id: dict[str, list[str]],
    context_phrases_by_id: dict[str, list[str]] | None = None,
) -> str:
    def phrase(object_id: Any) -> str:
        return _object_phrase_for_context(
            object_id,
            phrases_by_id=phrases_by_id,
            context_phrases_by_id=context_phrases_by_id,
        )

    tool = str(record.get("tool") or "")
    if tool in {"remove_object", "simulate_edit"}:
        removed_id = str(record.get("removed_object_id") or "")
        if not removed_id:
            return ""
        return (
            f"Simulated removing {phrase(removed_id)}; "
            f"the edited rollout covers frames {record.get('start_frame')} to {record.get('rollout_last_frame')}."
        )
    if tool == "check_contact":
        object_a = str(record.get("object_a") or "")
        object_b = str(record.get("object_b") or "")
        removed_id = str(record.get("removed_object_id") or "")
        pair = f"{phrase(object_a)} and {phrase(object_b)}"
        prefix = f"After removing {phrase(removed_id)}, " if removed_id else ""
        if record.get("confirmed"):
            if record.get("same_event_with_base") is False and record.get("event_identity_note"):
                return f"{prefix}{pair} still collide/contact at frames {record.get('confirmed_frames') or []}; {record.get('event_identity_note')}"
            if record.get("same_event_with_base") is True and record.get("event_identity_note"):
                return f"{prefix}{pair} still collide/contact at frames {record.get('confirmed_frames') or []}; {record.get('event_identity_note')}"
            return f"{prefix}{pair} still collide/contact at frames {record.get('confirmed_frames') or []}."
        nearest = record.get("nearest_distance") if isinstance(record.get("nearest_distance"), dict) else {}
        return (
            f"{prefix}{pair} do not have confirmed collision/contact"
            f"; nearest distance is {nearest.get('distance')} at frame {nearest.get('frame')}."
        )
    if tool == "get_event_time":
        object_a = str(record.get("object_a") or "")
        object_b = str(record.get("object_b") or "")
        pair = f"{phrase(object_a)} and {phrase(object_b)}"
        if record.get("found"):
            return f"In the base rollout, {pair} collide/contact at frame(s) {record.get('frames')}."
        return f"In the base rollout, no collision/contact is found between {pair}."
    if tool == "compare_rollouts":
        base_id = record.get("base_rollout_id")
        edited_id = record.get("edited_rollout_id")
        return f"Comparison between rollout {base_id} and {edited_id}: {record.get('trajectory_delta') or record.get('result')}."
    return str(record.get("text") or "")


def _event_first_frame(record: dict[str, Any]) -> int | None:
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    try:
        return min(int(frame) for frame in frames)
    except (TypeError, ValueError):
        return None


def _same_object_set(record: dict[str, Any], object_ids: set[str]) -> bool:
    return set(str(value) for value in record.get("involved_object_ids") or []) == set(object_ids)


def _phrase_list(
    object_ids: set[str],
    *,
    phrases_by_id: dict[str, list[str]],
    context_phrases_by_id: dict[str, list[str]] | None = None,
) -> str:
    phrases = [
        _object_phrase_for_context(
            object_id,
            phrases_by_id=phrases_by_id,
            context_phrases_by_id=context_phrases_by_id,
        )
        for object_id in sorted(object_ids)
    ]
    return " and ".join(phrase for phrase in phrases if phrase)


def _causal_event_bridge_evidence(
    *,
    choice_ids: set[str],
    target_ids: set[str],
    evidence_lines: list[dict[str, Any]],
    phrases_by_id: dict[str, list[str]],
    context_phrases_by_id: dict[str, list[str]],
) -> str:
    if len(choice_ids) < 2 or len(target_ids) < 2:
        return ""
    choice_event = next(
        (
            item
            for item in evidence_lines
            if item.get("tool") == "get_event_time" and item.get("found") and _same_object_set(item, choice_ids)
        ),
        None,
    )
    target_event = next(
        (
            item
            for item in evidence_lines
            if item.get("tool") == "get_event_time" and item.get("found") and _same_object_set(item, target_ids)
        ),
        None,
    )
    if not choice_event or not target_event:
        return ""
    choice_frame = _event_first_frame(choice_event)
    target_frame = _event_first_frame(target_event)
    if choice_frame is None or target_frame is None:
        return ""

    choice_phrase = _phrase_list(
        choice_ids,
        phrases_by_id=phrases_by_id,
        context_phrases_by_id=context_phrases_by_id,
    )
    target_phrase = _phrase_list(
        target_ids,
        phrases_by_id=phrases_by_id,
        context_phrases_by_id=context_phrases_by_id,
    )
    if choice_frame > target_frame:
        return (
            f"The candidate event ({choice_phrase}) happens at frame {choice_frame}, after the target event "
            f"({target_phrase}) at frame {target_frame}; it is not causal evidence for the target."
        )
    if choice_frame == target_frame:
        return (
            f"The candidate event ({choice_phrase}) and the target event ({target_phrase}) occur at the same frame "
            f"{target_frame}; timing alone is not sufficient causal evidence."
        )

    shared_ids = choice_ids.intersection(target_ids)
    if shared_ids:
        shared_phrase = _phrase_list(
            shared_ids,
            phrases_by_id=phrases_by_id,
            context_phrases_by_id=context_phrases_by_id,
        )
        return (
            f"The candidate event ({choice_phrase}) happens before the target event ({target_phrase}) "
            f"and shares {shared_phrase} with the target; this is plausible causal-chain evidence."
        )

    bridge_records = []
    for item in evidence_lines:
        if item.get("tool") != "check_contact" or not item.get("confirmed"):
            continue
        involved = set(str(value) for value in item.get("involved_object_ids") or [])
        if not involved.intersection(choice_ids) or not involved.intersection(target_ids):
            continue
        frames = []
        for frame in item.get("confirmed_frames") or []:
            try:
                frame_int = int(frame)
            except (TypeError, ValueError):
                continue
            if choice_frame < frame_int < target_frame:
                frames.append(frame_int)
        if frames:
            bridge_records.append((item, frames))

    if bridge_records:
        bridge_text = []
        for item, frames in bridge_records[:3]:
            involved = set(str(value) for value in item.get("involved_object_ids") or [])
            bridge_text.append(
                f"{_phrase_list(involved, phrases_by_id=phrases_by_id, context_phrases_by_id=context_phrases_by_id)} at frame(s) {frames}"
            )
        return (
            f"The candidate event ({choice_phrase}) happens before the target event ({target_phrase}), and there is "
            f"bridge interaction between their object sets before the target: {'; '.join(bridge_text)}. "
            "This is plausible causal-chain evidence."
        )

    return (
        f"The candidate event ({choice_phrase}) happens before the target event ({target_phrase}), but no confirmed "
        "bridge interaction between the candidate-event objects and target-event objects is found before the target; "
        "earlier timing alone is not sufficient causal evidence."
    )


def _tool_evidence_lines(
    trajectory: dict[str, Any],
    *,
    phrases_by_id: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    results = trajectory.get("tool_call_results") if isinstance(trajectory.get("tool_call_results"), list) else []
    rollout_removed = _rollout_removed_objects(results)
    lines: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        involved: set[str] = set()
        removed_id = None
        record: dict[str, Any] = {"tool": tool}
        if tool in {"remove_object", "simulate_edit"}:
            removed_id = str(result.get("removed_object_id") or "")
            if removed_id:
                involved.add(removed_id)
                record.update(
                    {
                        "removed_object_id": removed_id,
                        "start_frame": result.get("start_frame"),
                        "rollout_last_frame": result.get("rollout_last_frame"),
                    }
                )
        elif tool == "check_contact":
            object_a = str(result.get("object_a") or "")
            object_b = str(result.get("object_b") or "")
            rollout_id = str(result.get("rollout_id") or "")
            involved.update(value for value in [object_a, object_b] if value)
            removed_id = rollout_removed.get(rollout_id)
            if removed_id:
                involved.add(removed_id)
            record.update(
                {
                    "object_a": object_a,
                    "object_b": object_b,
                    "rollout_id": rollout_id,
                    "removed_object_id": removed_id,
                    "confirmed": bool(result.get("confirmed")),
                    "confirmed_frames": result.get("confirmed_frames") or [],
                    "frame_range": result.get("frame_range") if isinstance(result.get("frame_range"), dict) else {},
                    "base_first_frame": result.get("base_first_frame"),
                    "edited_first_frame": result.get("edited_first_frame"),
                    "event_frame_delta": result.get("event_frame_delta"),
                    "same_event_frame_tolerance": result.get("same_event_frame_tolerance"),
                    "same_event_with_base": result.get("same_event_with_base"),
                    "event_identity_note": result.get("event_identity_note"),
                    "nearest_distance": result.get("nearest_distance"),
                }
            )
        elif tool == "get_event_time":
            object_a = str(result.get("object_a") or "")
            object_b = str(result.get("object_b") or "")
            involved.update(value for value in [object_a, object_b] if value)
            record.update(
                {
                    "object_a": object_a,
                    "object_b": object_b,
                    "found": bool(result.get("found")),
                    "frames": result.get("frames") or [],
                    "first_frame": result.get("first_frame"),
                    "frame_range": result.get("frame_range") if isinstance(result.get("frame_range"), dict) else {},
                }
            )
        elif tool == "compare_rollouts":
            object_ids = [str(value) for value in result.get("object_ids") or [] if str(value)]
            involved.update(object_ids)
            base_id = result.get("base_rollout_id")
            edited_id = result.get("edited_rollout_id")
            record.update(
                {
                    "base_rollout_id": base_id,
                    "edited_rollout_id": edited_id,
                    "trajectory_delta": result.get("trajectory_delta"),
                    "result": result,
                }
            )
        text = _render_tool_evidence_record(record, phrases_by_id=phrases_by_id)
        if text:
            record.update(
                {
                    "text": text,
                    "involved_object_ids": sorted(involved),
                    "removed_object_id": removed_id,
                    "tool": tool,
                }
            )
            lines.append(record)
    return lines, rollout_removed


def _build_choice_evidence_answer_facts(question: ClevrerQuestion, trajectory: dict[str, Any]) -> dict[str, Any]:
    phrases_by_id = _object_phrases_by_id(trajectory)
    target_ids = _target_object_ids(question.question, trajectory)
    target_context_phrases = _context_phrases_by_id([question.question], trajectory)
    evidence_lines, _ = _tool_evidence_lines(trajectory, phrases_by_id=phrases_by_id)
    target_evidence: list[str] = []
    for item in evidence_lines:
        involved = set(item.get("involved_object_ids") or [])
        if target_ids and target_ids.issubset(involved):
            _append_unique(
                target_evidence,
                _render_tool_evidence_record(
                    item,
                    phrases_by_id=phrases_by_id,
                    context_phrases_by_id=target_context_phrases,
                ),
            )
    choices = []
    for index, choice in enumerate(question.choices):
        choice_ids = _choice_object_ids(choice.choice, trajectory)
        choice_context_phrases = _context_phrases_by_id([question.question, choice.choice], trajectory)
        lines: list[str] = []
        _append_unique(
            lines,
            _causal_event_bridge_evidence(
                choice_ids=choice_ids,
                target_ids=target_ids,
                evidence_lines=evidence_lines,
                phrases_by_id=phrases_by_id,
                context_phrases_by_id=choice_context_phrases,
            ),
        )
        for item in evidence_lines:
            involved = set(item.get("involved_object_ids") or [])
            removed = str(item.get("removed_object_id") or "")
            if len(choice_ids) > 1:
                relevant = choice_ids.issubset(involved)
            else:
                relevant = bool(choice_ids and involved.intersection(choice_ids))
            if relevant or (removed and removed in choice_ids):
                _append_unique(
                    lines,
                    _render_tool_evidence_record(
                        item,
                        phrases_by_id=phrases_by_id,
                        context_phrases_by_id=choice_context_phrases,
                    ),
                )
        choices.append(
            {
                "choice_letter": _choice_letter(index),
                "choice_id": choice.choice_id,
                "choice": choice.choice,
                "mapped_phrases": sorted(
                    {
                        phrase
                        for object_id in choice_ids
                        for phrase in phrases_by_id.get(object_id, [])
                        if _phrase_in_text(phrase, choice.choice)
                    }
                ),
                "evidence": lines[:8],
            }
        )
    return {
        "scene_index": trajectory.get("scene_index"),
        "question_id": question.question_id,
        "question_type": question.question_type,
        "target_event_evidence": target_evidence[:8],
        "choice_evidence": choices,
        "rules": [
            CLEVRER_CHOICE_POLICY.prompt_text,
            "Use only the provided per-choice evidence and target-event evidence.",
            "Object ids are intentionally omitted; reason using the question phrases and choice phrases.",
            "For causality questions asking which choices are not responsible, select a choice only if the target event still happens after removing or negating that choice.",
            "If removing a choice prevents the target event, that choice is responsible and must not be selected.",
            "If a choice is a collision/contact event, use the provided causal-chain evidence; earlier timing alone is not sufficient evidence of responsibility.",
        ],
    }


def _build_vlm_answer_prompt(
    *,
    question: ClevrerQuestion,
    facts: dict[str, Any],
) -> str:
    choices = [
        {
            "choice_letter": _choice_letter(index),
            "choice_id": choice.choice_id,
            "choice": choice.choice,
        }
        for index, choice in enumerate(question.choices)
    ]
    return (
        "You are the final reasoning module for a physical world-model agent.\n"
        "Answer the CLEVRER question using only the provided target-event evidence and per-choice evidence.\n"
        "Object ids have been removed intentionally; reason with the question phrases and choice phrases only.\n"
        "For contact or collision claims, use explicit evidence lines before any general intuition.\n"
        "For causality questions asking which choices are not responsible, select a choice only if the evidence says the target event still happens after removing or negating that choice. If removal prevents the target event, that choice is responsible and must not be selected.\n"
        "If a choice is itself a collision/contact event, use the provided causal-chain evidence; do not infer responsibility from earlier timing alone.\n"
        f"{CLEVRER_CHOICE_POLICY.prompt_text}\n"
        "Return only compact JSON inside <answer> </answer> with this schema:\n"
        "{\n"
        '  "selected_choices": ["A"],\n'
        '  "direct_answer": "yes/no/text for non-choice questions or comma-separated letters for choice questions",\n'
        '  "per_choice_results": [{"choice_letter": "A", "decision": "yes", "evidence": "short evidence"}],\n'
        '  "reasoning_summary": "short reasoning",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        f"Question type: {question.question_type}\n"
        f"Question: {question.question}\n"
        f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
        f"Evidence facts: {json.dumps(facts, ensure_ascii=False)}\n"
    )


def _execute_vlm_tools_answer(config: ModelConfig, question: ClevrerQuestion, trajectory: dict[str, Any]) -> dict[str, Any]:
    if not trajectory:
        return {
            "status": "vlm_error",
            "extracted_answer": None,
            "error_message": "missing query-conditioned physical rollout trajectory",
            "per_choice_results": [],
        }
    facts = _build_choice_evidence_answer_facts(question, trajectory)
    prompt = _build_vlm_answer_prompt(question=question, facts=facts)
    client = create_openai_compatible_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "You answer with valid JSON inside <answer> tags. Keep evidence concise.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.max_output_tokens,
    )
    raw_text = (response.choices[0].message.content or "").strip()
    try:
        parsed = _safe_json_loads(raw_text)
    except Exception as exc:
        return {
            "status": "vlm_parse_error",
            "extracted_answer": None,
            "error_message": f"VLM answer JSON parse failed: {exc}",
            "raw_response": raw_text,
            "facts": facts,
            "per_choice_results": [],
        }
    inferred_selected = _selected_choices_from_per_choice_results(parsed)
    if inferred_selected is not None:
        extracted = ",".join(inferred_selected) if inferred_selected else "NONE"
        parsed = {**parsed, "selected_choices": inferred_selected}
    else:
        selected = parsed.get("selected_choices")
        if isinstance(selected, list):
            extracted = ",".join(str(item).strip().upper() for item in selected if str(item).strip()) or "NONE"
        else:
            extracted = parsed.get("direct_answer")
    extracted_text = None if extracted is None else str(extracted).strip()
    if not extracted_text:
        return {
            "status": "vlm_parse_error",
            "extracted_answer": None,
            "error_message": "VLM answer did not contain selected_choices or direct_answer",
            "raw_response": raw_text,
            "facts": facts,
            "per_choice_results": parsed.get("per_choice_results", []),
        }
    return {
        "status": "ok",
        "extracted_answer": extracted_text,
        "error_message": None,
        "raw_response": raw_text,
        "vlm_answer": parsed,
        "facts": facts,
        "per_choice_results": parsed.get("per_choice_results", []),
        "reasoning_summary": parsed.get("reasoning_summary"),
        "confidence": parsed.get("confidence"),
    }


def run_final_answer(
    *,
    config: ModelConfig,
    scene: ClevrerScene | PhysionPPScene,
    question: ClevrerQuestion | PhysionPPQuestion,
    question_dir: Path,
    artifacts: ArtifactManager,
    dry_run: bool,
    success_answer_backend_route: dict[str, Any] | None = None,
) -> ToolResult:
    if success_answer_backend_route is None and not dry_run:
        success_answer_backend_route = _resolve_success_answer_backend_route(
            scene,
            question,
        )
    validated_backend_route = _require_success_answer_backend_route(
        success_answer_backend_route,
        scene=scene,
        question=question,
    ) if (success_answer_backend_route is not None or not dry_run) else None
    output_path = artifacts.artifact_path(question_dir, "final_answer", "final_answer.json")
    existing = artifacts.read_optional(output_path)
    if existing:
        if validated_backend_route is not None:
            expected_backend = validated_backend_route["route"]
            if existing.get("backend") != expected_backend:
                raise RoutePolicyValidationError(
                    "cached final-answer backend conflicts with ANS-001 route: "
                    f"{existing.get('backend')!r} != {expected_backend!r}"
                )
            recorded_route = existing.get("success_answer_backend_route")
            if recorded_route is not None and recorded_route != validated_backend_route:
                raise RoutePolicyValidationError(
                    "cached final-answer ANS-001 route record mismatch"
                )
            if recorded_route is None:
                existing = deepcopy(existing)
                existing["success_answer_backend_route"] = deepcopy(
                    validated_backend_route
                )
                artifacts.write(output_path, existing)
        return ToolResult("final_answer", "loaded", str(output_path), payload=existing)
    backend = (
        validated_backend_route["route"]
        if validated_backend_route is not None
        else "answer.contact_artifact"
        if isinstance(scene, PhysionPPScene)
        else "answer.vlm_tools"
    )
    if dry_run:
        payload = {
            "tool": "final_answer",
            "status": "dry_run",
            "backend": backend,
            "extracted_answer": None,
        }
        artifacts.write(output_path, payload)
        return ToolResult("final_answer", "dry_run", str(output_path), payload=payload)

    trajectory = _read_json(artifacts.artifact_path_by_name(question_dir, "trajectory.json"))
    if backend == "answer.contact_artifact":
        future = trajectory.get("physion_pp_future_rollout") or {}
        patient_contact = future.get("patient_contact") or {}
        will_contact = patient_contact.get("will_contact")
        if isinstance(will_contact, bool):
            answer_payload = {
                "status": "ok",
                "extracted_answer": "yes" if will_contact else "no",
                "error_message": None,
                "facts": {
                    "patient_contact": patient_contact,
                    "horizon": future.get("horizon"),
                    "source_segment": future.get("source_segment"),
                },
                "reasoning_summary": (
                    "The analytic future rollout predicts contact."
                    if will_contact
                    else "The analytic future rollout reaches its stopping horizon without contact."
                ),
                "confidence": "high",
            }
        else:
            answer_payload = {
                "status": "missing_contact_evidence",
                "extracted_answer": None,
                "error_message": "QCPR artifact does not contain a boolean patient_contact.will_contact",
                "facts": {"patient_contact": patient_contact},
            }
    elif backend == "answer.vlm_tools":
        answer_payload = _execute_vlm_tools_answer(config, question, trajectory)
    else:
        raise ValueError(f"Unsupported final answer backend: {backend}")
    status = answer_payload["status"]
    tool_status = "ok" if status == "ok" else "parse_error"
    payload = {
        "tool": "final_answer",
        "backend": backend,
        "status": status,
        "tool_status": tool_status,
        "scene_index": scene.scene_index,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "choices": _answer_choices_payload(question),
        "extracted_answer": answer_payload.get("extracted_answer"),
        "error_message": answer_payload.get("error_message"),
        "per_choice_results": answer_payload.get("per_choice_results", []),
        "facts": answer_payload.get("facts"),
        "vlm_answer": answer_payload.get("vlm_answer"),
        "reasoning_summary": answer_payload.get("reasoning_summary"),
        "confidence": answer_payload.get("confidence"),
        "raw_response": answer_payload.get("raw_response"),
        "trajectory_artifact": str(artifacts.artifact_path_by_name(question_dir, "trajectory.json")),
    }
    if validated_backend_route is not None:
        payload["success_answer_backend_route"] = deepcopy(
            validated_backend_route
        )
    artifacts.write(output_path, payload)
    return ToolResult("final_answer", tool_status, str(output_path), payload=payload)


def _answer_choices_payload(
    question: ClevrerQuestion | PhysionPPQuestion,
) -> list[dict[str, Any]]:
    return [
        {
            "choice_id": choice.choice_id,
            "choice_letter": _choice_letter(index),
            "choice": choice.choice,
        }
        for index, choice in enumerate(question.choices)
    ]


def extracted_final_answer(result: Optional[ToolResult]) -> Optional[str]:
    if result is None:
        return None
    value = result.payload.get("extracted_answer")
    return None if value is None else str(value)
