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

from benchmark.metrics import compute_physion_pp_metrics
from benchmark.physion_pp import (
    PHYSION_PP_BASELINE_SCENARIOS,
    PhysionPPScene,
    load_test_scenes,
)


def _build_random_predictions(
    scenes: list[PhysionPPScene],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for scene in scenes:
        scene_questions: list[dict[str, Any]] = []
        for question in scene.questions:
            prediction = rng.choice(("yes", "no"))
            expected = question.answer
            is_correct = prediction == expected
            scene_questions.append(
                {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "prediction": prediction,
                    "raw_prediction": prediction,
                    "usage": None,
                    "extracted_answer": prediction,
                    "normalized_prediction": prediction,
                    "expected_answer": expected,
                    "normalized_expected_answer": expected,
                    "ground_truth_outcome": question.ground_truth_outcome,
                    "request_error": False,
                    "parse_error": False,
                    "error_message": None,
                    "is_correct": is_correct,
                    "status": "correct" if is_correct else "wrong_answer",
                }
            )

        predictions.append(
            {
                "scene_index": scene.scene_index,
                "property": scene.property_name,
                "scenario": scene.scenario,
                "copy_name": scene.copy_name,
                "pair_id": scene.pair_id,
                "stimulus_id": scene.stimulus_id,
                "video_filename": scene.video_filename,
                "video_path": str(scene.video_path),
                "cue_frame_index": None,
                "cue_reference_frame_index": None,
                "questions": scene_questions,
            }
        )
    return predictions


def _print_summary(metrics: dict[str, Any], *, label: str) -> None:
    stats = metrics.get("physion_pp_ocp", {})
    pairs = stats.get("pairs", {})
    print(
        f"{label}: accuracy={stats.get('accuracy', 0.0):.4f} "
        f"correct={stats.get('correct_questions', 0)}/{stats.get('total_questions', 0)} "
        f"predicted_yes_rate={stats.get('predicted_yes_rate', 0.0):.4f}"
    )
    for scenario, scenario_stats in stats.get("by_scenario", {}).items():
        print(
            f"{label}.{scenario}: accuracy={scenario_stats.get('accuracy', 0.0):.4f} "
            f"correct={scenario_stats.get('correct_questions', 0)}/"
            f"{scenario_stats.get('total_questions', 0)}"
        )
    print(
        f"{label}.pairs: both_correct_rate={pairs.get('both_correct_rate', 0.0):.4f} "
        f"differentiated_rate={pairs.get('differentiated_rate', 0.0):.4f} "
        f"valid={pairs.get('valid_pairs', 0)}/{pairs.get('total_pairs', 0)}"
    )
    for scenario, scenario_stats in pairs.get("by_scenario", {}).items():
        print(
            f"{label}.pairs.{scenario}: "
            f"both_correct_rate={scenario_stats.get('both_correct_rate', 0.0):.4f} "
            f"differentiated_rate={scenario_stats.get('differentiated_rate', 0.0):.4f} "
            f"valid={scenario_stats.get('valid_pairs', 0)}/{scenario_stats.get('total_pairs', 0)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random Physion++ yes/no guessing.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--scenarios",
        default=",".join(PHYSION_PP_BASELINE_SCENARIOS),
        help="Comma-separated Physion++ baseline scenarios to include.",
    )
    args = parser.parse_args()

    selected_scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    unknown_scenarios = sorted(set(selected_scenarios) - set(PHYSION_PP_BASELINE_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(
            f"Unsupported Physion++ baseline scenarios: {unknown_scenarios}; "
            f"expected subset of {list(PHYSION_PP_BASELINE_SCENARIOS)}"
        )

    scenes = [
        scene
        for scene in load_test_scenes(dataset_root=args.dataset_root)
        if scene.scenario in selected_scenarios
    ]
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]

    predictions = _build_random_predictions(scenes, rng=random.Random(args.seed))
    metrics = compute_physion_pp_metrics(predictions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "config": {
                        "benchmark": "physion_pp",
                        "mode": "random",
                        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
                        "seed": args.seed,
                        "limit_scenes": args.limit_scenes,
                        "scenarios": list(selected_scenarios),
                    },
                    "metrics": metrics,
                    "metric_predictions": predictions,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    _print_summary(metrics, label=f"random-seed-{args.seed}")


if __name__ == "__main__":
    main()
