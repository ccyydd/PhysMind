from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Dict, List


def normalize_descriptive_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text or ""))))


def compute_metrics(predictions: List[Dict[str, object]]) -> Dict[str, object]:
    metrics = {}

    descriptive_total = 0
    descriptive_correct = 0
    descriptive_status_counts = defaultdict(int)

    option_totals = defaultdict(int)
    option_correct = defaultdict(int)
    question_totals = defaultdict(int)
    question_correct = defaultdict(int)
    question_status_counts = defaultdict(lambda: defaultdict(int))

    for scene in predictions:
        for question in scene["questions"]:
            question_type = question["question_type"]
            if question_type == "descriptive":
                descriptive_total += 1
                descriptive_status_counts[question.get("status", "wrong_answer")] += 1
                if question["is_correct"]:
                    descriptive_correct += 1
                continue

            question_totals[question_type] += 1
            question_status_counts[question_type][question.get("status", "wrong_answer")] += 1
            all_choices_correct = True
            for choice in question["choices"]:
                option_totals[question_type] += 1
                if choice["is_correct"]:
                    option_correct[question_type] += 1
                else:
                    all_choices_correct = False
            if all_choices_correct:
                question_correct[question_type] += 1

    metrics["descriptive"] = {
        "total_questions": descriptive_total,
        "correct_questions": descriptive_correct,
        "accuracy": descriptive_correct / descriptive_total if descriptive_total else 0.0,
        "status_counts": {
            "correct": descriptive_status_counts["correct"],
            "wrong_answer": descriptive_status_counts["wrong_answer"],
            "parse_error": descriptive_status_counts["parse_error"],
            "request_error": descriptive_status_counts["request_error"],
        },
    }

    metrics["multiple_choice"] = {}
    for question_type in ("explanatory", "predictive", "counterfactual"):
        total_options = option_totals[question_type]
        total_questions = question_totals[question_type]
        metrics["multiple_choice"][question_type] = {
            "total_options": total_options,
            "correct_options": option_correct[question_type],
            "per_option_accuracy": option_correct[question_type] / total_options if total_options else 0.0,
            "total_questions": total_questions,
            "correct_questions": question_correct[question_type],
            "per_question_accuracy": question_correct[question_type] / total_questions if total_questions else 0.0,
            "status_counts": {
                "correct": question_status_counts[question_type]["correct"],
                "wrong_answer": question_status_counts[question_type]["wrong_answer"],
                "parse_error": question_status_counts[question_type]["parse_error"],
                "request_error": question_status_counts[question_type]["request_error"],
            },
        }

    return metrics


def compute_physion_pp_metrics(predictions: List[Dict[str, object]]) -> Dict[str, object]:
    total = 0
    correct = 0
    valid_answer_count = 0
    predicted_yes = 0
    status_counts = defaultdict(int)
    scenario_totals = defaultdict(int)
    scenario_correct = defaultdict(int)
    scenario_valid_answers = defaultdict(int)
    scenario_predicted_yes = defaultdict(int)
    scenario_status_counts = defaultdict(lambda: defaultdict(int))
    class_totals = defaultdict(int)
    class_correct = defaultdict(int)
    pair_members = defaultdict(list)

    for scene in predictions:
        scenario = str(scene.get("scenario") or "unknown")
        pair_id = str(scene.get("pair_id") or "")
        for question in scene["questions"]:
            total += 1
            scenario_totals[scenario] += 1
            expected = str(question.get("normalized_expected_answer") or "unknown")
            class_totals[expected] += 1
            status = str(question.get("status") or "wrong_answer")
            status_counts[status] += 1
            scenario_status_counts[scenario][status] += 1
            prediction = str(question.get("normalized_prediction") or "")
            valid_answer = status in {"correct", "wrong_answer"} and prediction in {"yes", "no"}
            if valid_answer:
                valid_answer_count += 1
                scenario_valid_answers[scenario] += 1
            if valid_answer and prediction == "yes":
                predicted_yes += 1
                scenario_predicted_yes[scenario] += 1
            if question["is_correct"]:
                correct += 1
                scenario_correct[scenario] += 1
                class_correct[expected] += 1
            if pair_id:
                pair_members[pair_id].append(
                    {
                        "scenario": scenario,
                        "is_correct": bool(question["is_correct"]),
                        "prediction": prediction,
                        "valid_answer": valid_answer,
                    }
                )

    def _group_stats(totals: Dict[str, int], correct_counts: Dict[str, int]) -> Dict[str, object]:
        return {
            name: {
                "total_questions": totals[name],
                "correct_questions": correct_counts[name],
                "accuracy": correct_counts[name] / totals[name] if totals[name] else 0.0,
            }
            for name in sorted(totals)
        }

    def _pair_stats(pairs: List[List[Dict[str, object]]]) -> Dict[str, object]:
        valid_pairs = [
            members
            for members in pairs
            if len(members) == 2 and all(member["valid_answer"] for member in members)
        ]
        invalid_pairs = [
            members
            for members in pairs
            if len(members) == 2 and not all(member["valid_answer"] for member in members)
        ]
        incomplete_pairs = [members for members in pairs if len(members) != 2]
        both_correct = sum(
            1 for members in valid_pairs if members[0]["is_correct"] and members[1]["is_correct"]
        )
        differentiated = sum(
            1 for members in valid_pairs if members[0]["prediction"] != members[1]["prediction"]
        )
        return {
            "total_pairs": len(pairs),
            "valid_pairs": len(valid_pairs),
            "invalid_pairs": len(invalid_pairs),
            "incomplete_pairs": len(incomplete_pairs),
            "both_correct_pairs": both_correct,
            "both_correct_rate": both_correct / len(valid_pairs) if valid_pairs else 0.0,
            "differentiated_pairs": differentiated,
            "differentiated_rate": differentiated / len(valid_pairs) if valid_pairs else 0.0,
        }

    all_pairs = list(pair_members.values())
    pairs_by_scenario = defaultdict(list)
    for pair_id, members in pair_members.items():
        pair_scenarios = {str(member["scenario"]) for member in members}
        if len(pair_scenarios) != 1:
            raise ValueError(f"Physion++ pair spans multiple scenarios: {pair_id} -> {sorted(pair_scenarios)}")
        pairs_by_scenario[next(iter(pair_scenarios))].append(members)
    pair_stats = _pair_stats(all_pairs)
    pair_stats["by_scenario"] = {
        scenario: _pair_stats(pairs_by_scenario[scenario])
        for scenario in sorted(pairs_by_scenario)
    }
    by_scenario = {
        scenario: {
            "total_questions": scenario_totals[scenario],
            "correct_questions": scenario_correct[scenario],
            "accuracy": (
                scenario_correct[scenario] / scenario_totals[scenario]
                if scenario_totals[scenario]
                else 0.0
            ),
            "valid_answer_count": scenario_valid_answers[scenario],
            "invalid_answer_count": scenario_totals[scenario] - scenario_valid_answers[scenario],
            "predicted_yes_rate": (
                scenario_predicted_yes[scenario] / scenario_valid_answers[scenario]
                if scenario_valid_answers[scenario]
                else 0.0
            ),
            "status_counts": {
                "correct": scenario_status_counts[scenario]["correct"],
                "wrong_answer": scenario_status_counts[scenario]["wrong_answer"],
                "parse_error": scenario_status_counts[scenario]["parse_error"],
                "request_error": scenario_status_counts[scenario]["request_error"],
            },
        }
        for scenario in sorted(scenario_totals)
    }

    return {
        "physion_pp_ocp": {
            "total_questions": total,
            "correct_questions": correct,
            "accuracy": correct / total if total else 0.0,
            "valid_answer_count": valid_answer_count,
            "invalid_answer_count": total - valid_answer_count,
            "predicted_yes_rate": predicted_yes / valid_answer_count if valid_answer_count else 0.0,
            "status_counts": {
                "correct": status_counts["correct"],
                "wrong_answer": status_counts["wrong_answer"],
                "parse_error": status_counts["parse_error"],
                "request_error": status_counts["request_error"],
            },
            "by_scenario": by_scenario,
            "by_class": _group_stats(class_totals, class_correct),
            "pairs": pair_stats,
        }
    }
