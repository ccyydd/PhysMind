from __future__ import annotations

from copy import deepcopy
import json
from multiprocessing import Pool
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Union

from tqdm import tqdm

from agent.query import (
    answer_video_question,
    answer_with_sampled_video_frames,
    extract_answer_tag,
    infer_model_family,
)
from benchmark.clevrer import ClevrerScene, load_validation_scenes
from benchmark.metrics import (
    compute_metrics,
    compute_physion_pp_metrics,
    normalize_descriptive_answer,
)
from benchmark.prompts import (
    ANSWER_FORMATS,
    build_descriptive_prompt,
    build_multiple_choice_prompt,
    build_physion_pp_ocp_prompt,
)
from utils.config import ModelConfig
from utils.run import ensure_run_dir, write_json


CHOICE_LETTER_PATTERN = re.compile(r"[A-Z]")
PLAIN_ANSWER_PATTERNS = {
    "plain-answer": re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$"),
}
DIRECT_ANSWER_INPUT_MODALITY_DECISION_ID = "DA-001.input_modality"
DIRECT_ANSWER_INPUT_MODALITY_ROUTE = "direct_input.video_or_frames"
DIRECT_ANSWER_PROMPT_FAMILY_DECISION_ID = "DA-003.prompt_family"
CLEVRER_DIRECT_ANSWER_PROMPT_FAMILY_ROUTE = (
    "prompt.descriptive_or_multiple_choice"
)
DIRECT_ANSWER_PROMPT_FAMILY_ROUTES = frozenset(
    {
        CLEVRER_DIRECT_ANSWER_PROMPT_FAMILY_ROUTE,
        PHYSION_PP_DIRECT_ANSWER_PROMPT_FAMILY_ROUTE,
    }
)
DIRECT_ANSWER_OUTPUT_CONTRACT_DECISION_ID = "DA-004.output_contract"
DIRECT_ANSWER_OUTPUT_CONTRACT_ROUTE = (
    "output.standard_run_bundle"
)
DIRECT_ANSWER_POLICY_BENCHMARKS = frozenset({"clevrer", "physion_pp"})


















def _direct_answer_input_materialization(
    config: ModelConfig,
    *,
    additional_reference_frame_indices: Optional[Sequence[int]] = None,
    sampled_frame_intro: Optional[str] = None,
) -> str:
    if sampled_frame_intro is not None:
        return "uniform_sampled_frames_with_persistent_target_cues"
    if infer_model_family(config.model) == "gpt":
        if additional_reference_frame_indices:
            return "sampled_frames_with_target_cue_reference"
        return "uniform_sampled_frames"
    return "full_video"




def _direct_answer_desc(bench_name: str) -> str:
    return f"{bench_name} direct-answer"


def _answer_direct_question(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: dict[str, object],
    additional_reference_frame_indices: Optional[Sequence[int]] = None,
    sampled_frame_intro: Optional[str] = None,
    input_modality_route: dict[str, Any] | None = None,
    policy_benchmark: str | None = None,
):
    _require_direct_answer_input_modality(
        input_modality_route,
        benchmark=policy_benchmark,
    )
    if sampled_frame_intro is not None:
        return answer_with_sampled_video_frames(
            config=config,
            prompt=prompt,
            video_path=video_path,
            request_context=request_context,
            frame_intro=sampled_frame_intro,
        )
    return answer_video_question(
        config=config,
        prompt=prompt,
        video_path=video_path,
        request_context=request_context,
        additional_reference_frame_indices=additional_reference_frame_indices,
    )


def _run_status_summary(predictions: list[Dict[str, object]]) -> Dict[str, object]:
    status_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    total = 0
    for scene in predictions:
        for question in scene.get("questions", []):
            if not isinstance(question, dict):
                continue
            total += 1
            status = str(question.get("status") or "unknown")
            question_type = str(question.get("question_type") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[question_type] = type_counts.get(question_type, 0) + 1
    return {
        "total_questions": total,
        "status_counts": status_counts,
        "question_type_counts": type_counts,
    }


def _write_run_outputs(
    *,
    run_dir: Path,
    run_config: Dict[str, object],
    predictions: list[Dict[str, object]],
    metrics: Dict[str, object],
    output_contract_route: dict[str, Any] | None = None,
    policy_benchmark: str | None = None,
) -> None:
    validated_output_route = _require_direct_answer_output_contract(
        output_contract_route,
        benchmark=policy_benchmark,
    )
    if validated_output_route is not None:
        configured_route = (
            (run_config.get("resolved_routes") or {}).get(
                DIRECT_ANSWER_OUTPUT_CONTRACT_DECISION_ID
            )
            if isinstance(run_config.get("resolved_routes"), dict)
            else None
        )
        if configured_route != validated_output_route:
            raise RoutePolicyValidationError(
                "direct-answer run_config output-contract route mismatch"
            )
    status_summary = _run_status_summary(predictions)
    if run_config.get("benchmark") == "physion_pp":
        expected_questions = int(run_config.get("expected_questions") or 0)
        status_counts = status_summary["status_counts"]
        if status_summary["total_questions"] < expected_questions:
            completion_status = "running"
        elif status_counts.get("request_error", 0) or status_counts.get("parse_error", 0):
            completion_status = "needs_resume"
        else:
            completion_status = "complete"
        status_summary["completion_status"] = completion_status
    bundle = {
        "schema_version": 1,
        "config": run_config,
        "predictions": predictions,
        "metric_predictions": predictions,
        "metrics": metrics,
        "summary": {
            "mode": run_config.get("mode"),
            "benchmark": run_config.get("benchmark"),
            **status_summary,
        },
    }
    write_json(run_dir / "run.json", bundle)
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "predictions.json", predictions)
    write_json(run_dir / "metrics.json", metrics)


def _question_status(*, request_error: bool, parse_error: bool, is_correct: bool) -> str:
    if request_error:
        return "request_error"
    if parse_error:
        return "parse_error"
    if is_correct:
        return "correct"
    return "wrong_answer"


DIRECT_ANSWER_RESUME_RERUN_STATUSES = {"request_error", "parse_error"}


def _parse_resume_rerun_statuses(statuses: Optional[Sequence[str]]) -> Set[str]:
    if statuses is None:
        return set(DIRECT_ANSWER_RESUME_RERUN_STATUSES)
    parsed = {str(status).strip() for status in statuses if str(status).strip()}
    valid = {"correct", "wrong_answer", "request_error", "parse_error"}
    invalid = sorted(parsed - valid)
    if invalid:
        raise ValueError(f"Unsupported direct-answer resume statuses: {invalid}")
    return parsed


def _load_clevrer_resume_scenes(resume_run_dir: Optional[Union[str, Path]]) -> Dict[int, Dict[str, object]]:
    if resume_run_dir is None:
        return {}
    predictions_path = Path(resume_run_dir).expanduser().resolve() / "predictions.json"
    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError(f"resume predictions must be a list: {predictions_path}")
    scenes: Dict[int, Dict[str, object]] = {}
    for scene in predictions:
        if not isinstance(scene, dict):
            continue
        try:
            scene_index = int(scene.get("scene_index"))
        except (TypeError, ValueError):
            continue
        scenes[scene_index] = scene
    return scenes


def _load_clevrer_resume_questions(resume_run_dir: Optional[Union[str, Path]]) -> Dict[tuple[int, int], Dict[str, object]]:
    resume_scenes = _load_clevrer_resume_scenes(resume_run_dir)
    questions: Dict[tuple[int, int], Dict[str, object]] = {}
    for scene_index, scene in resume_scenes.items():
        for question in scene.get("questions", []):
            if not isinstance(question, dict):
                continue
            try:
                question_id = int(question.get("question_id"))
            except (TypeError, ValueError):
                continue
            questions[(scene_index, question_id)] = question
    return questions


def _should_reuse_direct_answer_resume(
    question_result: Optional[Dict[str, object]],
    rerun_statuses: Set[str],
) -> bool:
    if not question_result:
        return False
    status = str(question_result.get("status") or "")
    return status not in rerun_statuses


def _ordered_clevrer_predictions(
    scenes: Sequence[ClevrerScene],
    predictions_by_scene: Dict[int, Dict[str, object]],
) -> list[Dict[str, object]]:
    return [
        predictions_by_scene[scene.scene_index]
        for scene in scenes
        if scene.scene_index in predictions_by_scene
    ]


def _extract_clevrer_answer(prediction: str, answer_format: str) -> Optional[str]:
    if answer_format == "answer-tag":
        return extract_answer_tag(prediction)
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    pattern = PLAIN_ANSWER_PATTERNS.get(answer_format)
    matches = pattern.findall(prediction or "") if pattern else []
    if matches:
        return matches[-1].strip()
    tagged = extract_answer_tag(prediction)
    if tagged is not None:
        return tagged
    lines = [line.strip() for line in (prediction or "").splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1].strip().strip("`")
    if len(last_line) <= 80:
        return last_line
    return None


def _extract_choice_letters_from_answer(answer: Optional[str]) -> Set[str]:
    if answer is None:
        return set()
    normalized = answer.strip().upper()
    if normalized == "NONE":
        return set()
    letters = set()
    for part in re.split(r"[\s,]+", normalized):
        token = part.strip().strip(".;()[]{}")
        if len(token) == 1 and CHOICE_LETTER_PATTERN.fullmatch(token):
            letters.add(token)
    return letters










def _extract_strict_physion_pp_answer(prediction: str) -> tuple[Optional[str], Optional[str]]:
    extracted_answer = _extract_clevrer_answer(prediction, "answer-tag")
    if extracted_answer is None:
        return None, None
    normalized_answer = extracted_answer.strip().lower()
    if normalized_answer not in {"yes", "no"}:
        return extracted_answer, None
    return extracted_answer, normalized_answer


def _predict_descriptive(
    config: ModelConfig,
    scene: ClevrerScene,
    question,
    *,
    answer_format: str = "answer-tag",
    input_modality_route: dict[str, Any],
    prompt_family_route: dict[str, Any],
) -> Dict[str, object]:
    validated_prompt_route = _require_direct_answer_prompt_family(
        prompt_family_route,
        benchmark="clevrer",
        scenario=None,
        question_type=question.question_type,
    )
    if validated_prompt_route is None:
        raise RoutePolicyValidationError(
            "CLEVRER direct-answer prompt route unexpectedly resolved to None"
        )
    print(
        f"[question] scene={scene.scene_index} qid={question.question_id} "
        f"type={question.question_type} subtype={question.question_subtype}"
    )
    response = _answer_direct_question(
        config=config,
        prompt=build_descriptive_prompt(question, answer_format=answer_format),
        video_path=scene.video_path,
        request_context={
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
        },
        input_modality_route=input_modality_route,
        policy_benchmark="clevrer",
    )
    prediction = response.text
    print(f"[question] completed scene={scene.scene_index} qid={question.question_id}")
    expected = normalize_descriptive_answer(question.answer or "")
    extracted_answer = _extract_clevrer_answer(prediction, answer_format)
    normalized_prediction = normalize_descriptive_answer(extracted_answer or "")
    parse_error = extracted_answer is None
    is_correct = (not parse_error) and normalized_prediction == expected
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "prediction": prediction,
        "raw_prediction": prediction,
        "usage": response.usage,
        "extracted_answer": extracted_answer,
        "normalized_prediction": normalized_prediction,
        "expected_answer": question.answer,
        "normalized_expected_answer": expected,
        "request_error": False,
        "parse_error": parse_error,
        "error_message": None,
        "is_correct": is_correct,
        "status": _question_status(request_error=False, parse_error=parse_error, is_correct=is_correct),
        "direct_answer_input_materialization": (
            _direct_answer_input_materialization(config)
        ),
    }


def _predict_multiple_choice(
    config: ModelConfig,
    scene: ClevrerScene,
    question,
    *,
    answer_format: str = "answer-tag",
    input_modality_route: dict[str, Any],
    prompt_family_route: dict[str, Any],
) -> Dict[str, object]:
    validated_prompt_route = _require_direct_answer_prompt_family(
        prompt_family_route,
        benchmark="clevrer",
        scenario=None,
        question_type=question.question_type,
    )
    if validated_prompt_route is None:
        raise RoutePolicyValidationError(
            "CLEVRER direct-answer prompt route unexpectedly resolved to None"
        )
    print(
        f"[question] scene={scene.scene_index} qid={question.question_id} "
        f"type={question.question_type} choices={len(question.choices)}"
    )
    response = _answer_direct_question(
        config=config,
        prompt=build_multiple_choice_prompt(question, answer_format=answer_format),
        video_path=scene.video_path,
        request_context={
            "scene_index": scene.scene_index,
            "question_id": question.question_id,
            "question_type": question.question_type,
        },
        input_modality_route=input_modality_route,
        policy_benchmark="clevrer",
    )
    prediction = response.text
    final_answer = _extract_clevrer_answer(prediction, answer_format)
    predicted_letters = _extract_choice_letters_from_answer(final_answer)
    normalized_final_answer = (final_answer or "").strip().upper()
    parse_error = final_answer is None or (normalized_final_answer != "NONE" and not predicted_letters)
    choice_predictions = []
    for index, choice in enumerate(question.choices):
        label = chr(ord("A") + index)
        normalized_prediction = "correct" if label in predicted_letters else "wrong"
        expected = "correct" if (choice.answer or "").strip().lower() == "correct" else "wrong"
        choice_predictions.append(
            {
                "choice_id": choice.choice_id,
                "choice_letter": label,
                "choice": choice.choice,
                "prediction": normalized_prediction,
                "normalized_prediction": normalized_prediction,
                "expected_answer": choice.answer,
                "normalized_expected_answer": expected,
                "is_correct": (not parse_error) and normalized_prediction == expected,
            }
        )
    print(f"[question] completed scene={scene.scene_index} qid={question.question_id}")
    is_correct = (not parse_error) and all(choice["is_correct"] for choice in choice_predictions)

    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "raw_prediction": prediction,
        "usage": response.usage,
        "predicted_letters": sorted(predicted_letters),
        "request_error": False,
        "parse_error": parse_error,
        "error_message": None,
        "is_correct": is_correct,
        "status": _question_status(request_error=False, parse_error=parse_error, is_correct=is_correct),
        "choices": choice_predictions,
        "direct_answer_input_materialization": (
            _direct_answer_input_materialization(config)
        ),
    }


def predict_physion_pp_ocp(
    config: ModelConfig,
    scene: PhysionPPScene,
    question,
    *,
    input_modality_route: dict[str, Any] | None = None,
    prompt_family_route: dict[str, Any] | None,
) -> Dict[str, object]:
    _require_direct_answer_prompt_family(
        prompt_family_route,
        benchmark="physion_pp",
        scenario=scene.scenario,
        question_type=question.question_type,
    )
    print(
        f"[question] physion_pp scene={scene.scene_index} property={scene.property_name} "
        f"scenario={scene.scenario} stimulus={scene.stimulus_id}"
    )
    uses_tracking_cues = scene.cue_input_mode == PHYSION_PP_TRACKING_CUE_INPUT_MODE
    response = _answer_direct_question(
        config=config,
        prompt=build_physion_pp_ocp_prompt(
            question,
            scenario=scene.scenario,
            persistent_tracking_cues=uses_tracking_cues,
        ),
        video_path=scene.video_path,
        additional_reference_frame_indices=(
            [scene.cue_reference_frame_index]
            if not uses_tracking_cues
            and scene.cue_reference_frame_index is not None
            and infer_model_family(config.model) == "gpt"
            else None
        ),
        sampled_frame_intro=(
            PHYSION_PP_TRACKING_CUE_FRAME_INTRO.format(num_frames=config.num_frames)
            if uses_tracking_cues
            else None
        ),
        request_context={
            "scene_index": scene.scene_index,
            "scenario": scene.scenario,
            "stimulus_id": scene.stimulus_id,
            "question_id": question.question_id,
            "question_type": question.question_type,
        },
        input_modality_route=input_modality_route,
        policy_benchmark=(
            "physion_pp" if input_modality_route is not None else None
        ),
    )
    prediction = response.text
    print(f"[question] completed physion_pp scene={scene.scene_index} stimulus={scene.stimulus_id}")
    extracted_answer, normalized_prediction = _extract_strict_physion_pp_answer(prediction)
    expected = question.answer
    parse_error = normalized_prediction is None
    is_correct = (not parse_error) and normalized_prediction == expected
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "prediction": prediction,
        "raw_prediction": prediction,
        "usage": response.usage,
        "extracted_answer": extracted_answer,
        "normalized_prediction": normalized_prediction or "",
        "expected_answer": question.answer,
        "normalized_expected_answer": expected,
        "ground_truth_outcome": question.ground_truth_outcome,
        "request_error": False,
        "parse_error": parse_error,
        "error_message": None,
        "is_correct": is_correct,
        "status": _question_status(request_error=False, parse_error=parse_error, is_correct=is_correct),
        "direct_answer_input_materialization": (
            _physion_pp_input_materialization(config, scene)
        ),
    }








def _build_descriptive_error_result(question, error_message: str) -> Dict[str, object]:
    expected = normalize_descriptive_answer(question.answer or "")
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "prediction": None,
        "raw_prediction": None,
        "usage": None,
        "extracted_answer": None,
        "normalized_prediction": "",
        "expected_answer": question.answer,
        "normalized_expected_answer": expected,
        "request_error": True,
        "parse_error": False,
        "error_message": error_message,
        "is_correct": False,
        "status": _question_status(request_error=True, parse_error=False, is_correct=False),
    }


def _build_multiple_choice_error_result(question, error_message: str) -> Dict[str, object]:
    choice_predictions = []
    for index, choice in enumerate(question.choices):
        label = chr(ord("A") + index)
        expected = "correct" if (choice.answer or "").strip().lower() == "correct" else "wrong"
        choice_predictions.append(
            {
                "choice_id": choice.choice_id,
                "choice_letter": label,
                "choice": choice.choice,
                "prediction": "wrong",
                "normalized_prediction": "wrong",
                "expected_answer": choice.answer,
                "normalized_expected_answer": expected,
                "is_correct": False,
            }
        )
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "raw_prediction": None,
        "usage": None,
        "predicted_letters": [],
        "request_error": True,
        "parse_error": False,
        "error_message": error_message,
        "is_correct": False,
        "status": _question_status(request_error=True, parse_error=False, is_correct=False),
        "choices": choice_predictions,
    }


def _build_physion_pp_ocp_error_result(question, error_message: str) -> Dict[str, object]:
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "prediction": None,
        "raw_prediction": None,
        "usage": None,
        "extracted_answer": None,
        "normalized_prediction": "",
        "expected_answer": question.answer,
        "normalized_expected_answer": question.answer,
        "ground_truth_outcome": question.ground_truth_outcome,
        "request_error": True,
        "parse_error": False,
        "error_message": error_message,
        "is_correct": False,
        "status": _question_status(request_error=True, parse_error=False, is_correct=False),
    }








def _record_direct_answer_input_metadata(
    question_result: Dict[str, object],
    *,
    input_modality_route: dict[str, Any],
    input_materialization: str,
    physion_pp_cue_input_route: dict[str, Any] | None = None,
    prompt_family_route: dict[str, Any] | None = None,
) -> Dict[str, object]:
    question_result["direct_answer_input_modality_route"] = deepcopy(
        input_modality_route
    )
    question_result["direct_answer_input_materialization"] = (
        input_materialization
    )
    if physion_pp_cue_input_route is not None:
        question_result["physionpp_cue_input_route"] = deepcopy(
            physion_pp_cue_input_route
        )
    if prompt_family_route is not None:
        question_result["direct_answer_prompt_family_route"] = deepcopy(
            prompt_family_route
        )
    return question_result


def _process_scene(
    scene: ClevrerScene,
    config: ModelConfig,
    allowed_question_types: Set[str],
    resume_questions: Optional[Dict[tuple[int, int], Dict[str, object]]] = None,
    resume_rerun_statuses: Optional[Set[str]] = None,
    answer_format: str = "answer-tag",
    input_modality_route: dict[str, Any] | None = None,
    prompt_family_routes: dict[int, dict[str, Any]] | None = None,
) -> Dict[str, object]:
    validated_input_route = _require_direct_answer_input_modality(
        input_modality_route,
        benchmark="clevrer",
    )
    if validated_input_route is None:
        raise RoutePolicyValidationError(
            "CLEVRER direct-answer input route unexpectedly resolved to None"
        )
    input_materialization = _direct_answer_input_materialization(config)
    rerun_statuses = resume_rerun_statuses or set(DIRECT_ANSWER_RESUME_RERUN_STATUSES)
    scene_questions = []
    for question in scene.questions:
        if allowed_question_types and question.question_type.lower() not in allowed_question_types:
            continue
        prompt_family_route = _require_direct_answer_prompt_family(
            (prompt_family_routes or {}).get(question.question_id),
            benchmark="clevrer",
            scenario=None,
            question_type=question.question_type,
        )
        resume_key = (scene.scene_index, question.question_id)
        resume_result = (resume_questions or {}).get(resume_key)
        if _should_reuse_direct_answer_resume(resume_result, rerun_statuses):
            scene_questions.append(
                _record_direct_answer_input_metadata(
                    deepcopy(resume_result),
                    input_modality_route=validated_input_route,
                    input_materialization=input_materialization,
                    prompt_family_route=prompt_family_route,
                )
            )
            continue
        try:
            if question.question_type == "descriptive":
                question_result = _predict_descriptive(
                    config,
                    scene,
                    question,
                    answer_format=answer_format,
                    input_modality_route=validated_input_route,
                    prompt_family_route=prompt_family_route,
                )
            else:
                question_result = _predict_multiple_choice(
                    config,
                    scene,
                    question,
                    answer_format=answer_format,
                    input_modality_route=validated_input_route,
                    prompt_family_route=prompt_family_route,
                )
        except Exception as exc:
            error_message = (
                f"Failed on scene={scene.scene_index} qid={question.question_id} "
                f"type={question.question_type} video={scene.video_filename}: {exc}"
            )
            print(f"[error] {error_message}")
            if question.question_type == "descriptive":
                question_result = _build_descriptive_error_result(
                    question,
                    error_message,
                )
            else:
                question_result = _build_multiple_choice_error_result(
                    question,
                    error_message,
                )
        scene_questions.append(
            _record_direct_answer_input_metadata(
                question_result,
                input_modality_route=validated_input_route,
                input_materialization=input_materialization,
                prompt_family_route=prompt_family_route,
            )
        )
    return {
        "scene_index": scene.scene_index,
        "video_filename": scene.video_filename,
        "video_path": scene.video_path,
        "questions": scene_questions,
    }


def _process_scene_wrapper(
    args: tuple[
        ClevrerScene,
        ModelConfig,
        Set[str],
        Optional[Dict[tuple[int, int], Dict[str, object]]],
        Optional[Set[str]],
        str,
        dict[str, Any],
        dict[int, dict[str, Any]],
    ],
) -> Dict[str, object]:
    return _process_scene(*args)


def _process_physion_pp_scene(
    scene: PhysionPPScene,
    config: ModelConfig,
    allowed_question_types: Set[str],
    input_modality_route: dict[str, Any],
    physion_pp_cue_input_route: dict[str, Any],
    prompt_family_routes: dict[int, dict[str, Any]],
) -> Dict[str, object]:
    validated_input_route = _require_direct_answer_input_modality(
        input_modality_route,
        benchmark="physion_pp",
    )
    validated_cue_route = _require_physion_pp_cue_input(
        physion_pp_cue_input_route,
        benchmark="physion_pp",
    )
    if validated_input_route is None or validated_cue_route is None:
        raise RoutePolicyValidationError(
            "Physion++ direct-answer input routes unexpectedly resolved to None"
        )
    input_materialization = _physion_pp_input_materialization(config, scene)
    scene_questions = []
    for question in scene.questions:
        if allowed_question_types and question.question_type.lower() not in allowed_question_types:
            continue
        prompt_family_route = _require_direct_answer_prompt_family(
            prompt_family_routes.get(question.question_id),
            benchmark="physion_pp",
            scenario=scene.scenario,
            question_type=question.question_type,
        )
        try:
            question_result = predict_physion_pp_ocp(
                config,
                scene,
                question,
                input_modality_route=validated_input_route,
                prompt_family_route=prompt_family_route,
            )
        except Exception as exc:
            error_message = (
                f"Failed on physion_pp scene={scene.scene_index} scenario={scene.scenario} "
                f"stimulus={scene.stimulus_id} video={scene.video_filename}: {exc}"
            )
            print(f"[error] {error_message}")
            question_result = _build_physion_pp_ocp_error_result(
                question,
                error_message,
            )
        scene_questions.append(
            _record_direct_answer_input_metadata(
                question_result,
                input_modality_route=validated_input_route,
                input_materialization=input_materialization,
                physion_pp_cue_input_route=validated_cue_route,
                prompt_family_route=prompt_family_route,
            )
        )
    return {
        "scene_index": scene.scene_index,
        "property": scene.property_name,
        "scenario": scene.scenario,
        "copy_name": scene.copy_name,
        "pair_id": scene.pair_id,
        "stimulus_id": scene.stimulus_id,
        "video_filename": scene.video_filename,
        "video_path": scene.video_path,
        "cue_frame_index": scene.cue_frame_index,
        "cue_reference_frame_index": scene.cue_reference_frame_index,
        "cue_input_mode": scene.cue_input_mode,
        "questions": scene_questions,
    }


def _process_physion_pp_scene_wrapper(
    args: tuple[
        PhysionPPScene,
        ModelConfig,
        Set[str],
        dict[str, Any],
        dict[str, Any],
        dict[int, dict[str, Any]],
    ],
) -> Dict[str, object]:
    return _process_physion_pp_scene(*args)










def _filter_scenes_by_ids(scenes: Sequence[ClevrerScene], scene_ids: Optional[Sequence[int]]) -> list[ClevrerScene]:
    if scene_ids is None:
        return list(scenes)
    requested = [int(scene_id) for scene_id in scene_ids]
    requested_set = set(requested)
    filtered = [scene for scene in scenes if scene.scene_index in requested_set]
    found = {scene.scene_index for scene in filtered}
    missing = [scene_id for scene_id in requested if scene_id not in found]
    if missing:
        raise ValueError(f"CLEVRER scene_ids not found: {missing}")
    return filtered








def run_clevrer_validation_direct_answer(
    *,
    config: ModelConfig,
    dataset_root: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    scene_ids: Optional[Sequence[int]] = None,
    question_types: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    resume_run_dir: Optional[Union[str, Path]] = None,
    resume_rerun_statuses: Optional[Sequence[str]] = None,
    answer_format: str = "answer-tag",
) -> Path:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    route_policy = load_route_policy()
    input_modality_route = _resolve_direct_answer_input_modality(
        "clevrer",
        policy=route_policy,
    )
    output_contract_route = _resolve_direct_answer_output_contract(
        "clevrer",
        policy=route_policy,
    )
    mode = "direct-answer"
    run_dir = (
        Path(resume_run_dir).expanduser().resolve()
        if resume_run_dir is not None
        else ensure_run_dir(provider=config.provider, model=config.model, mode=mode)
    )
    scenes = load_validation_scenes(dataset_root=dataset_root)
    scenes = _filter_scenes_by_ids(scenes, scene_ids)
    if limit is not None:
        scenes = scenes[:limit]
    allowed_question_types = {value.strip().lower() for value in (question_types or []) if value.strip()}
    prompt_family_routes_by_scene = {
        scene.scene_index: {
            question.question_id: _resolve_direct_answer_prompt_family(
                "clevrer",
                scenario=None,
                question_type=question.question_type,
                policy=route_policy,
            )
            for question in scene.questions
        }
        for scene in scenes
    }
    rerun_statuses = _parse_resume_rerun_statuses(resume_rerun_statuses)
    resume_scenes = _load_clevrer_resume_scenes(resume_run_dir)
    resume_questions = _load_clevrer_resume_questions(resume_run_dir)
    resume_questions_by_scene: Dict[int, Dict[tuple[int, int], Dict[str, object]]] = {}
    for key, value in resume_questions.items():
        resume_questions_by_scene.setdefault(key[0], {})[key] = value
    predictions_by_scene: Dict[int, Dict[str, object]] = {
        scene.scene_index: resume_scenes[scene.scene_index]
        for scene in scenes
        if scene.scene_index in resume_scenes
    }
    run_config = config.to_safe_dict()
    run_config.update(
        {
            "dataset_root": dataset_root,
            "limit": limit,
            "scene_ids": list(scene_ids) if scene_ids is not None else None,
            "num_workers": num_workers,
            "question_types": sorted(allowed_question_types),
            "mode": mode,
            "benchmark": "clevrer-validation",
            "route_policy_id": route_policy.policy_id,
            "resolved_routes": {
                DIRECT_ANSWER_INPUT_MODALITY_DECISION_ID: (
                    input_modality_route
                ),
                DIRECT_ANSWER_OUTPUT_CONTRACT_DECISION_ID: (
                    output_contract_route
                ),
            },
            "answer_format": answer_format,
            "resume_run_dir": str(run_dir) if resume_run_dir is not None else None,
            "resume_rerun_statuses": sorted(rerun_statuses) if resume_run_dir is not None else None,
            "checkpoint": "scene_level",
        }
    )

    def write_checkpoint() -> None:
        predictions = _ordered_clevrer_predictions(scenes, predictions_by_scene)
        metrics = compute_metrics(predictions)
        _write_run_outputs(
            run_dir=run_dir,
            run_config=run_config,
            predictions=predictions,
            metrics=metrics,
            output_contract_route=output_contract_route,
            policy_benchmark="clevrer",
        )

    if num_workers <= 1:
        for scene in tqdm(scenes, desc=_direct_answer_desc("CLEVRER")):
            scene_result = _process_scene(
                scene,
                config,
                allowed_question_types,
                resume_questions_by_scene.get(scene.scene_index, {}),
                rerun_statuses,
                answer_format,
                input_modality_route,
                prompt_family_routes_by_scene.get(scene.scene_index, {}),
            )
            predictions_by_scene[scene.scene_index] = scene_result
            write_checkpoint()
    else:
        print(f"Running CLEVRER {mode} with {num_workers} workers...")
        args_list = [
            (
                scene,
                config,
                allowed_question_types,
                resume_questions_by_scene.get(scene.scene_index, {}),
                rerun_statuses,
                answer_format,
                input_modality_route,
                prompt_family_routes_by_scene.get(scene.scene_index, {}),
            )
            for scene in scenes
        ]
        with Pool(processes=num_workers, maxtasksperchild=4) as pool:
            for scene_result in tqdm(
                pool.imap_unordered(_process_scene_wrapper, args_list),
                total=len(args_list),
                desc=_direct_answer_desc("CLEVRER"),
            ):
                try:
                    scene_index = int(scene_result.get("scene_index"))
                except (TypeError, ValueError):
                    continue
                predictions_by_scene[scene_index] = scene_result
                write_checkpoint()

    write_checkpoint()
    return run_dir
def _filter_physion_pp_scenes(
    scenes: Sequence[PhysionPPScene],
    scene_ids: Optional[Sequence[int]],
    properties: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
) -> list[PhysionPPScene]:
    filtered = list(scenes)
    selected_scenarios = tuple(scenarios or PHYSION_PP_BASELINE_SCENARIOS)
    unknown_scenarios = sorted(set(selected_scenarios) - set(PHYSION_PP_BASELINE_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(
            f"Unsupported Physion++ baseline scenarios: {unknown_scenarios}; "
            f"expected subset of {list(PHYSION_PP_BASELINE_SCENARIOS)}"
        )
    filtered = [scene for scene in filtered if scene.scenario in selected_scenarios]
    if properties:
        wanted_properties = {value for value in properties}
        filtered = [scene for scene in filtered if scene.property_name in wanted_properties]
    if scene_ids is None:
        return filtered
    requested = [int(scene_id) for scene_id in scene_ids]
    requested_set = set(requested)
    filtered = [scene for scene in filtered if scene.scene_index in requested_set]
    found = {scene.scene_index for scene in filtered}
    missing = [scene_id for scene_id in requested if scene_id not in found]
    if missing:
        raise ValueError(f"Physion++ scene_ids not found: {missing}")
    return filtered


def run_physion_pp_test_direct_answer(
    *,
    config: ModelConfig,
    dataset_root: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    scene_ids: Optional[Sequence[int]] = None,
    question_types: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    properties: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    resume_run_dir: Optional[Union[str, Path]] = None,
    resume_rerun_statuses: Optional[Sequence[str]] = None,
) -> Path:
    route_policy = load_route_policy()
    input_modality_route = _resolve_direct_answer_input_modality(
        "physion_pp",
        policy=route_policy,
    )
    physion_pp_cue_input_route = _resolve_physion_pp_cue_input(
        policy=route_policy,
    )
    output_contract_route = _resolve_direct_answer_output_contract(
        "physion_pp",
        policy=route_policy,
    )
    mode = "direct-answer"
    if run_dir is not None:
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    elif resume_run_dir is not None:
        run_dir = Path(resume_run_dir).expanduser().resolve()
    else:
        run_dir = ensure_run_dir(provider=config.provider, model=config.model, mode=mode)
    # The loader preserves official scene indices before property filtering.
    scenes = load_physion_pp_test_scenes(dataset_root=dataset_root)
    scenes = _filter_physion_pp_scenes(
        scenes,
        scene_ids,
        properties=properties,
        scenarios=scenarios,
    )
    if limit is not None:
        scenes = scenes[:limit]
    # The resolved DA-002 route authorizes the existing cue-input materialization.
    if physion_pp_cue_input_route["route"] == PHYSION_PP_CUE_INPUT_ROUTE:
        scenes = resolve_cue_clip_scenes(scenes, dataset_root=dataset_root)
        scenes = stage_cue_clips_into_run_dir(scenes, run_dir=run_dir)
    allowed_question_types = {value.strip().lower() for value in (question_types or []) if value.strip()}
    prompt_family_routes_by_scene = {
        scene.scene_index: {
            question.question_id: _resolve_direct_answer_prompt_family(
                "physion_pp",
                scenario=scene.scenario,
                question_type=question.question_type,
                policy=route_policy,
            )
            for question in scene.questions
        }
        for scene in scenes
    }
    rerun_statuses = _parse_resume_rerun_statuses(resume_rerun_statuses)
    resume_scenes = _load_clevrer_resume_scenes(resume_run_dir)
    predictions_by_scene: Dict[int, Dict[str, object]] = {}
    pending_scenes = []
    for scene in scenes:
        resume_result = resume_scenes.get(scene.scene_index)
        resume_questions = resume_result.get("questions", []) if resume_result else []
        reusable = bool(resume_questions) and all(
            isinstance(question, dict) and _should_reuse_direct_answer_resume(question, rerun_statuses)
            for question in resume_questions
        )
        if reusable:
            enriched_result = deepcopy(resume_result)
            input_materialization = _physion_pp_input_materialization(
                config,
                scene,
            )
            for question_result in enriched_result.get("questions", []):
                if not isinstance(question_result, dict):
                    continue
                _record_direct_answer_input_metadata(
                    question_result,
                    input_modality_route=input_modality_route,
                    input_materialization=input_materialization,
                    physion_pp_cue_input_route=(
                        physion_pp_cue_input_route
                    ),
                    prompt_family_route=(
                        prompt_family_routes_by_scene
                        .get(scene.scene_index, {})
                        .get(question_result.get("question_id"))
                    ),
                )
            predictions_by_scene[scene.scene_index] = enriched_result
        else:
            pending_scenes.append(scene)

    run_config = config.to_safe_dict()
    cue_input_modes = sorted({scene.cue_input_mode for scene in scenes})
    if len(cue_input_modes) > 1:
        raise ValueError(
            f"A Physion++ direct-answer run cannot mix cue input modes: {cue_input_modes}"
        )
    uses_tracking_cues = cue_input_modes == [PHYSION_PP_TRACKING_CUE_INPUT_MODE]
    uses_sampled_frames_with_cue = (
        not uses_tracking_cues
        and infer_model_family(config.model) == "gpt"
    )
    run_config.update(
        {
            "dataset_root": dataset_root,
            "limit": limit,
            "scene_ids": list(scene_ids) if scene_ids is not None else None,
            "num_workers": num_workers,
            "expected_questions": len(scenes),
            "question_types": sorted(allowed_question_types),
            "properties": list(properties) if properties else None,
            "scenarios": list(scenarios or PHYSION_PP_BASELINE_SCENARIOS),
            "mode": mode,
            "benchmark": "physion_pp",
            "route_policy_id": route_policy.policy_id,
            "resolved_routes": {
                DIRECT_ANSWER_INPUT_MODALITY_DECISION_ID: (
                    input_modality_route
                ),
                PHYSION_PP_CUE_INPUT_DECISION_ID: (
                    physion_pp_cue_input_route
                ),
                DIRECT_ANSWER_OUTPUT_CONTRACT_DECISION_ID: (
                    output_contract_route
                ),
            },
            "physion_pp_cue_video_dir": str(run_dir / PHYSION_PP_RUN_CUE_VIDEO_DIRNAME),
            "video_input_mode": (
                "uniform_sampled_frames_with_persistent_target_cues"
                if uses_tracking_cues
                else (
                    "sampled_frames_with_target_cue_reference"
                    if uses_sampled_frames_with_cue
                    else "full_video"
                )
            ),
            "physion_pp_cue_input_modes": cue_input_modes,
            "uniform_sampled_frames": (
                config.num_frames
                if uses_tracking_cues or uses_sampled_frames_with_cue
                else None
            ),
            "additional_target_cue_frames": 1 if uses_sampled_frames_with_cue else 0,
            "total_input_images": (
                config.num_frames
                if uses_tracking_cues
                else (config.num_frames + 1 if uses_sampled_frames_with_cue else None)
            ),
            "resume_run_dir": (
                str(Path(resume_run_dir).expanduser().resolve())
                if resume_run_dir is not None
                else None
            ),
            "resume_rerun_statuses": sorted(rerun_statuses) if resume_run_dir is not None else None,
            "checkpoint": "scene_level",
        }
    )

    def write_checkpoint() -> None:
        predictions = [
            predictions_by_scene[scene.scene_index]
            for scene in scenes
            if scene.scene_index in predictions_by_scene
        ]
        metrics = compute_physion_pp_metrics(predictions)
        _write_run_outputs(
            run_dir=run_dir,
            run_config=run_config,
            predictions=predictions,
            metrics=metrics,
            output_contract_route=output_contract_route,
            policy_benchmark="physion_pp",
        )

    if num_workers <= 1:
        for scene in tqdm(pending_scenes, desc=_direct_answer_desc("Physion++")):
            predictions_by_scene[scene.scene_index] = _process_physion_pp_scene(
                scene,
                config,
                allowed_question_types,
                input_modality_route,
                physion_pp_cue_input_route,
                prompt_family_routes_by_scene.get(scene.scene_index, {}),
            )
            write_checkpoint()
    else:
        print(f"Running Physion++ {mode} with {num_workers} workers...")
        args_list = [
            (
                scene,
                config,
                allowed_question_types,
                input_modality_route,
                physion_pp_cue_input_route,
                prompt_family_routes_by_scene.get(scene.scene_index, {}),
            )
            for scene in pending_scenes
        ]
        with Pool(processes=num_workers, maxtasksperchild=4) as pool:
            for scene_result in tqdm(
                pool.imap_unordered(_process_physion_pp_scene_wrapper, args_list),
                total=len(args_list),
                desc=_direct_answer_desc("Physion++"),
            ):
                predictions_by_scene[int(scene_result["scene_index"])] = scene_result
                write_checkpoint()

    write_checkpoint()
    return run_dir
