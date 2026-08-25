from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent.world_model.answer import (
    extracted_final_answer,
    run_direct_answer_fallback,
    run_final_answer,
)
from agent.world_model.artifacts import ArtifactManager, write_json
from agent.world_model.orchestrator import (
    _build_physion_pp_metric_predictions,
    _compute_status_metrics,
    _write_run_outputs,
)
from agent.world_model.schemas import ObjectPlan, TargetObject, ToolResult, WorldModelQuestionResult
from agent.world_model.simulation import QueryConditionedPhysicalRolloutAdapter
from benchmark.metrics import compute_physion_pp_metrics
from benchmark.physion_pp import PHYSION_PP_RUN_CUE_VIDEO_DIRNAME, PhysionPPScene, load_test_scenes
from scripts.world_model.swr_sysid_common import build_manifest_from_world_modeling
from utils.config import ModelConfig, RUNS_DIR, build_model_config


SUPPORTED_SCENARIOS = {
    "friction_platform_pp",
    "bouncy_wall_pp",
    "bouncy_platform_pp",
    "friction_collision_pp",
    "mass_collision_pp",
}

SWR_MODULE_BY_SCENARIO = {
    "friction_platform_pp": "scripts.world_model.run_physionpp_friction_sphere_sysid",
    "bouncy_wall_pp": "scripts.world_model.run_physionpp_bouncy_wall_sphere_sysid",
    "bouncy_platform_pp": "scripts.world_model.run_physionpp_bouncy_platform_sphere_sysid",
    "friction_collision_pp": "scripts.world_model.run_physionpp_friction_collision_sphere_sysid",
    "mass_collision_pp": "scripts.world_model.run_physionpp_mass_collision_sphere_sysid",
}

SWR_BACKEND_BY_SCENARIO = {
    "friction_platform_pp": "swr_backend.surface_friction_sphere",
    "bouncy_wall_pp": "swr_backend.wall_bounce_sphere",
    "bouncy_platform_pp": "swr_backend.platform_bounce_sphere",
    "friction_collision_pp": "swr_backend.collision_friction_spheres",
    "mass_collision_pp": "swr_backend.collision_mass_spheres",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _configure_single_worker_threads() -> None:
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_name] = "1"


def _object_plan_from_artifact(
    *,
    payload: dict[str, Any],
    scene: PhysionPPScene,
) -> ObjectPlan:
    question = scene.questions[0]
    targets = []
    for index, item in enumerate(payload.get("target_objects") or []):
        if not isinstance(item, dict):
            continue
        targets.append(
            TargetObject(
                object_id=str(item.get("object_id") or f"obj_{index + 1}"),
                description=str(item.get("description") or ""),
                role=str(item.get("role") or ""),
                geometry_type=str(item.get("geometry_type") or "irregular"),
                geometry_confidence=item.get("geometry_confidence"),
                appearance=deepcopy(item.get("appearance") or {}),
                source_track_id=(
                    str(item["source_track_id"])
                    if item.get("source_track_id") is not None
                    else None
                ),
                mesh_reuse_source_object_id=(
                    str(item["mesh_reuse_source_object_id"])
                    if item.get("mesh_reuse_source_object_id") is not None
                    else None
                ),
            )
        )
    if not targets:
        raise ValueError("scene object plan contains no target objects")

    special_scene = deepcopy(payload.get("special_scene") or {})
    scene_metadata = special_scene.setdefault("scene_metadata", {})
    scene_metadata.update(
        {
            "benchmark": "physion_pp",
            "scene_index": scene.scene_index,
            "scenario": scene.scenario,
            "stimulus_id": scene.stimulus_id,
            "video_filename": scene.video_filename,
            "video_path": str(scene.video_path),
        }
    )
    return ObjectPlan(
        scene_index=scene.scene_index,
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        choices=[],
        target_objects=targets,
        reasoning=str(payload.get("reasoning") or ""),
        scene_objects=deepcopy(payload.get("scene_objects") or {}),
        special_scene=special_scene,
        status=str(payload.get("status") or "ok"),
        error_message=payload.get("error_message"),
    )


def _fit_rmse(fit: dict[str, Any]) -> float | None:
    candidates = [
        fit.get("dynamic_object_rmse_m"),
        (fit.get("alignment_optimization") or {}).get("dynamic_object_rmse_m"),
        fit.get("fit_error"),
    ]
    for value in candidates:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0.0:
            return number
    return None


def _valid_fit(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok" and payload.get("backend") is not None


def _prepare_swr(
    *,
    scene: PhysionPPScene,
    run_dir: Path,
    world_model_dir: Path,
    force_swr: bool,
) -> tuple[ToolResult, Path, float]:
    artifacts = ArtifactManager(run_dir, debug_artifacts=False)
    summary_path = artifacts.artifact_path(
        world_model_dir,
        "simulatable_world_reconstruction",
        "simulatable_world_reconstruction.json",
    )
    fit_path = artifacts.artifact_path(
        world_model_dir,
        "simulatable_world_reconstruction",
        "world_reconstruction_fit.json",
    )
    if force_swr:
        summary_path.unlink(missing_ok=True)
        fit_path.unlink(missing_ok=True)
    if _valid_fit(fit_path) and not force_swr:
        fit = _load_json(fit_path)
        result = ToolResult(
            tool_name="simulatable_world_reconstruction",
            status="loaded",
            artifact_path=str(summary_path),
            payload={
                "status": "ok",
                "backend": fit.get("backend"),
                "world_reconstruction_fit": {
                    "status": "ok",
                    "artifact": str(fit_path),
                    "backend": fit.get("backend"),
                },
            },
        )
        return result, fit_path, 0.0

    fit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = fit_path.parent / "world_reconstruction_fit_manifest.json"
    manifest = build_manifest_from_world_modeling(world_model_dir)
    backend = SWR_BACKEND_BY_SCENARIO[scene.scenario]
    manifest["mode"] = f"{scene.scenario}_offline_sysid"
    rollout = manifest.setdefault("rollout", {})
    rollout["backend"] = backend
    write_json(manifest_path, manifest)

    command = [
        sys.executable,
        "-m",
        SWR_MODULE_BY_SCENARIO[scene.scenario],
        "--output",
        str(fit_path),
        "--output-dir",
        str(fit_path.parent),
        "--workers",
        "1",
        "--worker-threads",
        "1",
    ]
    if scene.scenario == "friction_platform_pp":
        command.extend(["--manifest", str(manifest_path)])
    else:
        command.extend(["--world-modeling-dir", str(world_model_dir)])

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path = fit_path.parent / "downstream_swr.log"
    log_path.write_text(
        f"command: {shlex.join(command)}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "SWR subprocess failed").strip()
        raise RuntimeError(message)
    if not _valid_fit(fit_path):
        raise RuntimeError(f"SWR did not produce a valid fit: {fit_path}")
    fit = _load_json(fit_path)
    summary = {
        "tool": "simulatable_world_reconstruction",
        "status": "ok",
        "backend": fit.get("backend"),
        "scene_index": scene.scene_index,
        "question_id": scene.questions[0].question_id,
        "fit_error": fit.get("fit_error"),
        "physics_rollout": fit.get("physics_rollout"),
        "alignment_optimization": fit.get("alignment_optimization"),
        "world_reconstruction_fit": {
            "status": "ok",
            "artifact": str(fit_path),
            "backend": fit.get("backend"),
        },
    }
    write_json(summary_path, summary)
    result = ToolResult(
        tool_name="simulatable_world_reconstruction",
        status="ok",
        artifact_path=str(summary_path),
        payload=summary,
        elapsed_sec=elapsed,
    )
    return result, fit_path, elapsed


def _run_question_stages(
    *,
    scene: PhysionPPScene,
    run_dir: Path,
    world_model_dir: Path,
    object_plan: ObjectPlan,
    swr_result: Any,
    config: ModelConfig,
) -> tuple[WorldModelQuestionResult, float]:
    question = scene.questions[0]
    question_dir = run_dir / "artifacts" / f"scene_{scene.scene_index}" / f"question_{question.question_id}"
    artifacts = ArtifactManager(run_dir, debug_artifacts=False)
    trajectory_path = artifacts.artifact_path(
        question_dir,
        "query_conditioned_physical_rollout",
        "trajectory.json",
    )
    final_path = artifacts.artifact_path(question_dir, "final_answer", "final_answer.json")

    # These two stages are cheap and deterministic. Regenerate them so a newly fitted
    # SWR result can never be paired with stale future-contact evidence.
    trajectory_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)

    started = time.perf_counter()
    rollout_result = QueryConditionedPhysicalRolloutAdapter(
        artifacts=artifacts,
        config=config,
        dry_run=False,
    ).run(
        scene=scene,
        question=question,
        question_dir=question_dir,
        object_plan=object_plan,
        stage_results=[swr_result],
        world_model_dir=world_model_dir,
    )
    if rollout_result.status not in {"ok", "loaded"}:
        raise RuntimeError(
            rollout_result.message
            or f"QCPR failed with status={rollout_result.status}"
        )
    answer_result = run_final_answer(
        config=config,
        scene=scene,
        question=question,
        question_dir=question_dir,
        artifacts=artifacts,
        dry_run=False,
    )
    answer = extracted_final_answer(answer_result)
    if answer_result.status not in {"ok", "loaded"} or answer not in {"yes", "no"}:
        raise RuntimeError(
            answer_result.message
            or f"answering failed with status={answer_result.status}, answer={answer!r}"
        )
    elapsed = time.perf_counter() - started
    result = WorldModelQuestionResult(
        scene_index=scene.scene_index,
        video_filename=scene.video_filename,
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        status="ok",
        final_answer=answer,
        error_message=None,
        artifact_dir=str(question_dir),
        stages=[
            swr_result.to_dict(),
            rollout_result.to_dict(),
            answer_result.to_dict(),
        ],
        elapsed_with_scene_reconstruction_sec=elapsed,
        elapsed_without_scene_reconstruction_sec=elapsed,
    )
    write_json(question_dir / "artifact-based-answering" / "result.json", result.to_dict())
    return result, elapsed


def _error_result(
    *,
    scene: PhysionPPScene,
    run_dir: Path,
    error: Exception,
    elapsed: float,
) -> WorldModelQuestionResult:
    question = scene.questions[0]
    question_dir = run_dir / "artifacts" / f"scene_{scene.scene_index}" / f"question_{question.question_id}"
    result = WorldModelQuestionResult(
        scene_index=scene.scene_index,
        video_filename=scene.video_filename,
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        status="agent_error",
        final_answer=None,
        error_message=str(error),
        artifact_dir=str(question_dir),
        stages=[],
        elapsed_with_scene_reconstruction_sec=elapsed,
        elapsed_without_scene_reconstruction_sec=elapsed,
    )
    write_json(question_dir / "artifact-based-answering" / "result.json", result.to_dict())
    return result


def _direct_answer_scene(*, scene: PhysionPPScene, run_dir: Path) -> PhysionPPScene:
    cue_path = (
        run_dir
        / PHYSION_PP_RUN_CUE_VIDEO_DIRNAME
        / f"{scene.scene_index:06d}_{scene.stimulus_id}_cue.mp4"
    )
    if not cue_path.exists():
        raise FileNotFoundError(f"missing staged Physion++ cue video: {cue_path}")
    return replace(
        scene,
        video_filename=cue_path.name,
        video_path=cue_path,
    )


def _run_error_fallback(
    *,
    scene: PhysionPPScene,
    run_dir: Path,
    pipeline_error: Exception,
    failed_stage: str,
    elapsed_before_fallback: float,
    config: ModelConfig,
) -> tuple[WorldModelQuestionResult, ToolResult, float]:
    question = scene.questions[0]
    question_dir = run_dir / "artifacts" / f"scene_{scene.scene_index}" / f"question_{question.question_id}"
    artifacts = ArtifactManager(run_dir, debug_artifacts=False)
    fallback_scene = _direct_answer_scene(scene=scene, run_dir=run_dir)
    started = time.perf_counter()
    answer_result = run_direct_answer_fallback(
        config=config,
        scene=fallback_scene,
        question=question,
        question_dir=question_dir,
        artifacts=artifacts,
        dry_run=False,
        pipeline_failure={
            "failed_stage": failed_stage,
            "failed_tool": None,
            "status": "agent_error",
            "error_message": str(pipeline_error),
            "artifact_path": None,
        },
    )
    fallback_elapsed = time.perf_counter() - started
    answer = extracted_final_answer(answer_result)
    if answer_result.status != "ok" or answer not in {"yes", "no"}:
        raise RuntimeError(
            answer_result.message
            or f"direct-answer fallback failed with status={answer_result.status}, answer={answer!r}"
        )
    total_elapsed = elapsed_before_fallback + fallback_elapsed
    result = WorldModelQuestionResult(
        scene_index=scene.scene_index,
        video_filename=scene.video_filename,
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        status="ok",
        final_answer=answer,
        error_message=None,
        artifact_dir=str(question_dir),
        stages=[answer_result.to_dict()],
        elapsed_with_scene_reconstruction_sec=total_elapsed,
        elapsed_without_scene_reconstruction_sec=total_elapsed,
    )
    write_json(question_dir / "artifact-based-answering" / "result.json", result.to_dict())
    return result, answer_result, fallback_elapsed


def _process_scene(
    *,
    scene: PhysionPPScene,
    run_dir: Path,
    force_swr: bool,
    config: ModelConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    active_stage = "input-validation"
    world_model_dir = run_dir / "artifacts" / f"scene_{scene.scene_index}" / "world-modeling"
    object_plan_path = (
        world_model_dir
        / "object-identification-and-planning"
        / "object_plan"
        / "object_plan.json"
    )
    try:
        if scene.scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"unsupported scenario: {scene.scenario}")
        if not world_model_dir.exists():
            raise FileNotFoundError(f"missing world-modeling directory: {world_model_dir}")
        object_plan = _object_plan_from_artifact(
            payload=_load_json(object_plan_path),
            scene=scene,
        )
        active_stage = "simulatable-world-reconstruction"
        swr_result, fit_path, swr_elapsed = _prepare_swr(
            scene=scene,
            run_dir=run_dir,
            world_model_dir=world_model_dir,
            force_swr=force_swr,
        )
        active_stage = "query-conditioned-physical-rollout-or-answering"
        result, downstream_elapsed = _run_question_stages(
            scene=scene,
            run_dir=run_dir,
            world_model_dir=world_model_dir,
            object_plan=object_plan,
            swr_result=swr_result,
            config=config,
        )
        fit = _load_json(fit_path)
        expected = scene.questions[0].answer
        return {
            "scene_index": scene.scene_index,
            "scenario": scene.scenario,
            "run_dir": str(run_dir),
            "status": "ok",
            "prediction": result.final_answer,
            "expected": expected,
            "is_correct": result.final_answer == expected,
            "dynamic_object_rmse_m": _fit_rmse(fit),
            "fit_backend": fit.get("backend"),
            "fit_path": str(fit_path),
            "swr_elapsed_sec": swr_elapsed,
            "downstream_elapsed_sec": downstream_elapsed,
            "elapsed_sec": time.perf_counter() - started,
            "fallback_used": False,
            "result": result,
        }
    except Exception as exc:
        elapsed_before_fallback = time.perf_counter() - started
        try:
            result, answer_result, fallback_elapsed = _run_error_fallback(
                scene=scene,
                run_dir=run_dir,
                pipeline_error=exc,
                failed_stage=active_stage,
                elapsed_before_fallback=elapsed_before_fallback,
                config=config,
            )
        except Exception as fallback_exc:
            elapsed = time.perf_counter() - started
            combined_error = RuntimeError(
                f"{exc}; direct-answer fallback failed: {fallback_exc}"
            )
            result = _error_result(
                scene=scene,
                run_dir=run_dir,
                error=combined_error,
                elapsed=elapsed,
            )
            return {
                "scene_index": scene.scene_index,
                "scenario": scene.scenario,
                "run_dir": str(run_dir),
                "status": "error",
                "prediction": None,
                "expected": scene.questions[0].answer,
                "is_correct": False,
                "dynamic_object_rmse_m": None,
                "fit_backend": None,
                "fit_path": None,
                "swr_elapsed_sec": None,
                "downstream_elapsed_sec": None,
                "elapsed_sec": elapsed,
                "fallback_used": True,
                "fallback_status": "error",
                "error_message": str(combined_error),
                "result": result,
            }
        elapsed = time.perf_counter() - started
        answer = result.final_answer
        return {
            "scene_index": scene.scene_index,
            "scenario": scene.scenario,
            "run_dir": str(run_dir),
            "status": "ok",
            "prediction": answer,
            "expected": scene.questions[0].answer,
            "is_correct": answer == scene.questions[0].answer,
            "dynamic_object_rmse_m": None,
            "fit_backend": None,
            "fit_path": None,
            "swr_elapsed_sec": None,
            "downstream_elapsed_sec": fallback_elapsed,
            "elapsed_sec": elapsed,
            "fallback_used": True,
            "fallback_status": answer_result.status,
            "pipeline_error_message": str(exc),
            "result": result,
        }


def _write_run_metrics(
    *,
    run_dir: Path,
    scenes: list[PhysionPPScene],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_records = sorted(records, key=lambda item: int(item["scene_index"]))
    results = [item["result"] for item in ordered_records]
    metric_predictions = _build_physion_pp_metric_predictions(scenes, results, {"ocp"})
    metrics = compute_physion_pp_metrics(metric_predictions)
    metrics["world_model_agent"] = {
        **_compute_status_metrics(results),
        "answering": "enabled",
        "completion_mode": "existing_world_modeling_downstream_only",
    }
    config_path = run_dir / "run_config.json"
    run_config = _load_json(config_path) if config_path.exists() else {}
    run_config.update(
        {
            "mode": "world-model-agent",
            "benchmark": "physion_pp",
            "answer_enabled": True,
            "stop_after_stage": "evaluation",
            "downstream_completion": {
                "source": "scripts/world_model/complete_physionpp_ocp_downstream.py",
                "world_modeling_reused": True,
                "scene_count": len(scenes),
            },
        }
    )
    scene_timings = [
        {
            "scene_index": item["scene_index"],
            "scenario": item["scenario"],
            "video_filename": next(
                scene.video_filename for scene in scenes if scene.scene_index == item["scene_index"]
            ),
            "question_count": 1,
            "scene_elapsed_sec": item["elapsed_sec"],
            "scene_reconstruction_elapsed_sec": item.get("swr_elapsed_sec") or 0.0,
        }
        for item in ordered_records
    ]
    _write_run_outputs(
        run_dir=run_dir,
        run_config=run_config,
        predictions=[result.to_dict() for result in results],
        metric_predictions=metric_predictions,
        metrics=metrics,
        scene_timings=scene_timings,
    )
    summary = {
        "run_dir": str(run_dir),
        "scene_count": len(scenes),
        "metrics": metrics,
        "scenes": [{key: value for key, value in item.items() if key != "result"} for item in ordered_records],
    }
    write_json(run_dir / "downstream_completion.json", summary)
    return summary


def _combined_summary(records: list[dict[str, Any]], run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    public_records = [
        {key: value for key, value in item.items() if key != "result"}
        for item in sorted(records, key=lambda value: int(value["scene_index"]))
    ]
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in sorted({str(item["scenario"]) for item in public_records}):
        selected = [item for item in public_records if item["scenario"] == scenario]
        rmse_values = [
            float(item["dynamic_object_rmse_m"])
            for item in selected
            if item.get("dynamic_object_rmse_m") is not None
        ]
        by_scenario[scenario] = {
            "total": len(selected),
            "completed": sum(item["status"] == "ok" for item in selected),
            "errors": sum(item["status"] != "ok" for item in selected),
            "fallbacks": sum(bool(item.get("fallback_used")) for item in selected),
            "fallback_correct": sum(
                bool(item.get("fallback_used")) and bool(item.get("is_correct"))
                for item in selected
            ),
            "correct": sum(bool(item["is_correct"]) for item in selected),
            "accuracy": (
                sum(bool(item["is_correct"]) for item in selected) / len(selected)
                if selected
                else 0.0
            ),
            "mean_dynamic_object_rmse_m": (
                sum(rmse_values) / len(rmse_values) if rmse_values else None
            ),
        }
    return {
        "status": "ok" if all(item["status"] == "ok" for item in public_records) else "partial",
        "scene_count": len(public_records),
        "completed": sum(item["status"] == "ok" for item in public_records),
        "errors": sum(item["status"] != "ok" for item in public_records),
        "fallbacks": sum(bool(item.get("fallback_used")) for item in public_records),
        "fallback_correct": sum(
            bool(item.get("fallback_used")) and bool(item.get("is_correct"))
            for item in public_records
        ),
        "correct": sum(bool(item["is_correct"]) for item in public_records),
        "accuracy": (
            sum(bool(item["is_correct"]) for item in public_records) / len(public_records)
            if public_records
            else 0.0
        ),
        "by_scenario": by_scenario,
        "run_summaries": run_summaries,
        "scenes": public_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete Physion++ SWR, analytic future rollout, OCP answering, and metrics "
            "from existing world-modeling artifacts."
        )
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--scene-ids",
        default=None,
        help="Optional comma-separated scene ids; omit to process every supported scene.",
    )
    parser.add_argument("--force-swr", action="store_true")
    parser.add_argument(
        "--summary-output",
        default=str(RUNS_DIR / "physionpp_existing_downstream_completion" / "evaluation_summary.json"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    _configure_single_worker_threads()
    config = build_model_config()
    run_dirs = [Path(value).expanduser().resolve() for value in args.run_dir]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    all_scenes = load_test_scenes(dataset_root=Path(args.dataset_root))
    scene_by_id = {scene.scene_index: scene for scene in all_scenes}
    selected_scene_ids = (
        {int(value.strip()) for value in str(args.scene_ids).split(",") if value.strip()}
        if args.scene_ids
        else None
    )
    tasks: list[tuple[PhysionPPScene, Path]] = []
    seen_ids: set[int] = set()
    for run_dir in run_dirs:
        for scene_dir in sorted((run_dir / "artifacts").glob("scene_*")):
            try:
                scene_id = int(scene_dir.name.removeprefix("scene_"))
            except ValueError:
                continue
            if scene_id in seen_ids:
                raise ValueError(f"scene {scene_id} appears in more than one input run")
            if selected_scene_ids is not None and scene_id not in selected_scene_ids:
                continue
            scene = scene_by_id.get(scene_id)
            if scene is None:
                raise ValueError(f"scene {scene_id} is not present in the Physion++ dataset")
            if scene.scenario not in SUPPORTED_SCENARIOS:
                continue
            seen_ids.add(scene_id)
            tasks.append((scene, run_dir))
    if not tasks:
        raise ValueError("no supported Physion++ scenes found in the supplied run directories")

    print(
        f"[downstream] start scenes={len(tasks)} workers={args.workers} "
        f"force_swr={args.force_swr}",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], tuple[PhysionPPScene, Path]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for scene, run_dir in tasks:
            future = executor.submit(
                _process_scene,
                scene=scene,
                run_dir=run_dir,
                force_swr=bool(args.force_swr),
                config=config,
            )
            futures[future] = (scene, run_dir)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            scene, _run_dir = futures[future]
            record = future.result()
            records.append(record)
            detail = (
                f"prediction={record.get('prediction')} expected={record.get('expected')} "
                f"correct={record.get('is_correct')} rmse={record.get('dynamic_object_rmse_m')}"
                if record["status"] == "ok"
                else f"error={record.get('error_message')}"
            )
            print(
                f"[downstream] {completed_count}/{len(tasks)} scene={scene.scene_index} "
                f"scenario={scene.scenario} status={record['status']} {detail}",
                flush=True,
            )

    run_summaries = []
    for run_dir in run_dirs:
        run_records = [item for item in records if Path(item["run_dir"]) == run_dir]
        if not run_records:
            continue
        run_scene_ids = {int(item["scene_index"]) for item in run_records}
        run_scenes = [scene_by_id[scene_id] for scene_id in sorted(run_scene_ids)]
        run_summaries.append(
            _write_run_metrics(run_dir=run_dir, scenes=run_scenes, records=run_records)
        )

    combined = _combined_summary(records, run_summaries)
    summary_output = Path(args.summary_output).expanduser().resolve()
    write_json(summary_output, combined)
    print(
        f"[downstream] end status={combined['status']} completed={combined['completed']}/"
        f"{combined['scene_count']} accuracy={combined['accuracy']:.4f} summary={summary_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
