from __future__ import annotations

import argparse
import json

from agent.direct_answer import (
    run_clevrer_validation_direct_answer,
    run_physion_pp_test_direct_answer,
)
from agent.query import infer_model_family
from benchmark.clevrer import default_clevrer_root
from utils.config import build_model_config
from utils.terminal import terminal_print


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physmind")
    parser.add_argument(
        "--bench",
        type=str,
        required=True,
        choices=["clevrer", "physion_pp"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="direct-answer",
        choices=["direct-answer", "world-model-agent"],
    )
    route_option_choices = tuple(
        option.option_id for option in load_route_policy().approved_optional_routes
    )
    parser.add_argument(
        "--route-option",
        action="append",
        default=[],
        choices=route_option_choices,
        help=(
            "Physion++ world-model-agent only: explicitly enable one approved, "
            "default-off pipeline route option. Repeat to enable independent options."
        ),
    )
    parser.add_argument(
        "--property",
        type=str,
        default=None,
        help="Physion++ only: comma-separated mechanical properties to run (mass,friction,bouncy).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Physion++ direct-answer only: comma-separated baseline scenarios to run.",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--answer-format",
        type=str,
        default="answer-tag",
        choices=["answer-tag", "plain-answer"],
        help="CLEVRER direct-answer final answer format.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-ids", type=str, default=None, help="Comma-separated scene_index values to run.")
    parser.add_argument("--question-types", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Physion++ direct-answer only: write into this run directory.",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Keep debug images/videos and reusable intermediate artifacts for world-model-agent runs.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume a world-model-agent run in an existing run directory.",
    )
    parser.add_argument(
        "--resume-rerun-statuses",
        type=str,
        default=None,
        help="Comma-separated direct-answer statuses to rerun when --resume-run-dir is set.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent SAM3/SAM3D/MoGe-2/video-depth/FoundationPose/GeoCalib worker loading. Use this when the available GPU has less than 80GB VRAM.",
    )
    parser.add_argument(
        "--stop-after-stage",
        type=str,
        default="evaluation",
        choices=[
            "object-planning",
            "object-segmentation-and-event-detection",
            "metric-mesh-reconstruction",
            "pose-tracking",
            "simulatable-world-reconstruction",
            "query-conditioned-physical-rollout",
            "answering",
            "evaluation",
        ],
        help="Stop world-model-agent after the selected semantic stage.",
    )
    return parser


def parse_scene_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    scene_ids = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        scene_ids.append(int(value))
    if not scene_ids:
        return None
    return scene_ids


def validate_route_options(args: argparse.Namespace) -> tuple[str, ...]:
    option_ids = tuple(args.route_option or ())
    if option_ids and not (
        args.bench == "physion_pp" and args.mode == "world-model-agent"
    ):
        raise ValueError(
            "--route-option is supported only with "
            "--bench physion_pp --mode world-model-agent"
        )
    policy = load_route_policy()
    return tuple(
        option.option_id for option in policy.enabled_optional_routes(option_ids)
    )


def print_metrics_summary(run_dir: str) -> None:
    metrics_path = f"{run_dir}/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    descriptive = metrics.get("descriptive", {})
    if descriptive.get("total_questions", 0):
        terminal_print(
            "[metrics] descriptive "
            f"accuracy={descriptive.get('accuracy', 0.0):.4f} "
            f"correct={descriptive.get('correct_questions', 0)}/{descriptive.get('total_questions', 0)}"
        )
        status_counts = descriptive.get("status_counts", {})
        terminal_print(
            "[metrics] descriptive_status "
            f"request_error={status_counts.get('request_error', 0)} "
            f"parse_error={status_counts.get('parse_error', 0)} "
            f"wrong_answer={status_counts.get('wrong_answer', 0)} "
            f"correct={status_counts.get('correct', 0)}"
        )

    for question_type, stats in metrics.get("multiple_choice", {}).items():
        if not stats.get("total_questions", 0):
            continue
        terminal_print(
            f"[metrics] {question_type} "
            f"per_question={stats.get('per_question_accuracy', 0.0):.4f} "
            f"({stats.get('correct_questions', 0)}/{stats.get('total_questions', 0)}) "
            f"per_option={stats.get('per_option_accuracy', 0.0):.4f} "
            f"({stats.get('correct_options', 0)}/{stats.get('total_options', 0)})"
        )
        status_counts = stats.get("status_counts", {})
        terminal_print(
            f"[metrics] {question_type}_status "
            f"request_error={status_counts.get('request_error', 0)} "
            f"parse_error={status_counts.get('parse_error', 0)} "
            f"wrong_answer={status_counts.get('wrong_answer', 0)} "
            f"correct={status_counts.get('correct', 0)}"
        )

    world_model_agent = metrics.get("world_model_agent", {})
    if world_model_agent:
        status_counts = world_model_agent.get("status_counts", {})
        terminal_print(
            "[metrics] world_model_agent "
            f"total_questions={world_model_agent.get('total_questions', 0)} "
            f"status_counts={json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}"
        )

    physion_pp_ocp = metrics.get("physion_pp_ocp", {})

def print_run_info(args: argparse.Namespace, question_types: list[str]) -> None:
    family = infer_model_family(args.model or "")
    mode_details = ""
    if args.mode == "world-model-agent":
        mode_details = (
            f" dry_run={args.dry_run} "
            f"debug_artifacts={args.debug_artifacts} "
            f"route_options={args.route_option} "
            f"persistent_workers={not args.no_persistent_workers}"
        )
    terminal_print(
        "[config] "
        f"bench={args.bench} mode={args.mode} provider={args.provider} model={args.model} "
        f"dataset_root={args.dataset_root} "
        f"num_frames={args.num_frames} limit={args.limit} scene_ids={args.scene_ids} "
        f"question_types={','.join(question_types)}{mode_details}"
    )
    if family == "gemini":
        if args.mode == "world-model-agent":
            terminal_print(
                "[note] Gemini-family model detected; full-video VLM calls use the video directly, "
                "while object segmentation and event detection uses SAM3 full-video tracking."
            )
        else:
            terminal_print(
                "[note] Gemini-family model detected; direct-answer submits the video directly."
            )
    else:
        terminal_print("[note] --num-frames controls sampled frame count only for VLM calls that do not accept video input.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "world-model-agent":
        from agent.world_model.orchestrator import (
            run_clevrer_validation_world_model_agent,
            run_physion_pp_test_world_model_agent,
        )

    if args.dry_run and args.mode != "world-model-agent":
        raise ValueError("--dry-run is supported only with --mode world-model-agent")
    if args.bench != "physion_pp" and (args.property is not None or args.scenario is not None):
        raise ValueError("--property and --scenario are only supported with --bench physion_pp")
    if args.scenario is not None and args.mode != "direct-answer":
        raise ValueError("--scenario is currently supported only for Physion++ direct-answer modes")
    if args.run_dir is not None and not (
        args.bench == "physion_pp" and args.mode == "direct-answer"
    ):
        raise ValueError("--run-dir is currently supported only for Physion++ direct-answer modes")
    enabled_route_options = validate_route_options(args)

    physion_pp_properties = None
    if args.property is not None:
        physion_pp_properties = [item.strip() for item in args.property.split(",") if item.strip()]
        unknown_properties = sorted(set(physion_pp_properties) - set(PHYSION_PP_PROPERTIES))
        if unknown_properties:
            raise ValueError(
                f"Unknown Physion++ properties: {unknown_properties}; expected subset of {PHYSION_PP_PROPERTIES}"
            )
    physion_pp_scenarios = None
    if args.scenario is not None:
        physion_pp_scenarios = [item.strip() for item in args.scenario.split(",") if item.strip()]
        unknown_scenarios = sorted(set(physion_pp_scenarios) - set(PHYSION_PP_BASELINE_SCENARIOS))
        if unknown_scenarios:
            raise ValueError(
                f"Unknown Physion++ baseline scenarios: {unknown_scenarios}; "
                f"expected subset of {list(PHYSION_PP_BASELINE_SCENARIOS)}"
            )

    config = build_model_config(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        num_frames=args.num_frames,
        max_output_tokens=args.max_output_tokens,
        require_api_key=not (args.mode == "world-model-agent" and args.dry_run),
    )
    if args.question_types is None:
        if args.bench == "physion_pp":
            args.question_types = "ocp"
        else:
            args.question_types = "explanatory,predictive,counterfactual"
    question_types = [item.strip() for item in args.question_types.split(",") if item.strip()]
    resume_rerun_statuses = (
        [item.strip() for item in args.resume_rerun_statuses.split(",") if item.strip()]
        if args.resume_rerun_statuses
        else None
    )
    scene_ids = parse_scene_ids(args.scene_ids)
    args.provider = config.provider
    args.model = config.model
    print_run_info(args, question_types)
    print_metrics_summary(str(run_dir))
    print(run_dir)


if __name__ == "__main__":
    main()
