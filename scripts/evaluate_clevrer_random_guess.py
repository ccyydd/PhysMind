from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.clevrer import ClevrerScene, load_validation_scenes
from benchmark.metrics import compute_metrics


def _choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def _build_random_predictions(
    scenes: list[ClevrerScene],
    *,
    rng: random.Random,
    question_types: set[str],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for scene in scenes:
        scene_questions: list[dict[str, Any]] = []
        for question in scene.questions:
            if question.question_type not in question_types:
                continue
            if question.question_type == "descriptive":
                continue

            predicted_letters: list[str] = []
            choice_predictions: list[dict[str, Any]] = []
            all_choices_correct = True
            for index, choice in enumerate(question.choices):
                letter = _choice_letter(index)
                guessed_correct = rng.choice((False, True))
                expected_correct = (choice.answer or "").strip().lower() == "correct"
                is_correct = guessed_correct == expected_correct
                if guessed_correct:
                    predicted_letters.append(letter)
                if not is_correct:
                    all_choices_correct = False
                choice_predictions.append(
                    {
                        "choice_id": choice.choice_id,
                        "choice_letter": letter,
                        "choice": choice.choice,
                        "prediction": "correct" if guessed_correct else "wrong",
                        "normalized_prediction": "correct" if guessed_correct else "wrong",
                        "expected_answer": choice.answer,
                        "normalized_expected_answer": "correct" if expected_correct else "wrong",
                        "is_correct": is_correct,
                    }
                )

            scene_questions.append(
                {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "raw_prediction": ",".join(predicted_letters) if predicted_letters else "NONE",
                    "usage": None,
                    "predicted_letters": predicted_letters,
                    "request_error": False,
                    "parse_error": False,
                    "error_message": None,
                    "is_correct": all_choices_correct,
                    "status": "correct" if all_choices_correct else "wrong_answer",
                    "choices": choice_predictions,
                }
            )

        predictions.append(
            {
                "scene_index": scene.scene_index,
                "video_filename": scene.video_filename,
                "video_path": str(scene.video_path),
                "questions": scene_questions,
            }
        )
    return predictions


def _print_summary(metrics: dict[str, Any], *, label: str) -> None:
    multiple_choice = metrics.get("multiple_choice", {})

    def counts(question_type: str, key: str) -> str:
        stats = multiple_choice.get(question_type, {})
        if key == "question":
            return f"{stats.get('total_questions', 0)} questions"
        return f"{stats.get('total_options', 0)} options"

    def score(question_type: str, key: str) -> str:
        stats = multiple_choice.get(question_type, {})
        if key == "question":
            correct = stats.get("correct_questions", 0)
            total = stats.get("total_questions", 0)
            acc = stats.get("per_question_accuracy", 0.0)
        else:
            correct = stats.get("correct_options", 0)
            total = stats.get("total_options", 0)
            acc = stats.get("per_option_accuracy", 0.0)
        return f"{correct}/{total} = {acc:.4f}"

    header = (
        "| Model | explanatory per-question | explanatory per-option | "
        "predictive per-question | predictive per-option | "
        "counterfactual per-question | counterfactual per-option |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    quantity = (
        f"| Count | {counts('explanatory', 'question')} | {counts('explanatory', 'option')} | "
        f"{counts('predictive', 'question')} | {counts('predictive', 'option')} | "
        f"{counts('counterfactual', 'question')} | {counts('counterfactual', 'option')} |"
    )
    result = (
        f"| {label} | {score('explanatory', 'question')} | {score('explanatory', 'option')} | "
        f"{score('predictive', 'question')} | {score('predictive', 'option')} | "
        f"{score('counterfactual', 'question')} | {score('counterfactual', 'option')} |"
    )
    print(header)
    print(separator)
    print(quantity)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random CLEVRER multiple-choice guessing.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--question-types",
        default="explanatory,predictive,counterfactual",
        help="Comma-separated multiple-choice question types to include.",
    )
    args = parser.parse_args()

    scenes = load_validation_scenes(args.dataset_root)
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]
    question_types = {item.strip().lower() for item in args.question_types.split(",") if item.strip()}
    rng = random.Random(args.seed)

    predictions = _build_random_predictions(scenes, rng=rng, question_types=question_types)
    metrics = compute_metrics(predictions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metrics": metrics, "metric_predictions": predictions}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    _print_summary(metrics, label=f"random-seed-{args.seed}")


if __name__ == "__main__":
    main()
