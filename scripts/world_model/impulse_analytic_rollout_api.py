from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYTIC_SYSID_PATH = _SCRIPT_DIR / "run_impulse_analytic_sysid.py"
_ANALYTIC_SYSID_SPEC = importlib.util.spec_from_file_location(
    "physmind_run_impulse_analytic_sysid_for_rollout_api",
    _ANALYTIC_SYSID_PATH,
)
if _ANALYTIC_SYSID_SPEC is None or _ANALYTIC_SYSID_SPEC.loader is None:
    raise RuntimeError(f"failed to load analytic sysid helpers: {_ANALYTIC_SYSID_PATH}")
analytic_sysid = importlib.util.module_from_spec(_ANALYTIC_SYSID_SPEC)
_ANALYTIC_SYSID_SPEC.loader.exec_module(analytic_sysid)


class ImpulseAnalyticRolloutApiError(RuntimeError):
    pass


def load_world_reconstruction_fit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rollout_from_world_reconstruction_fit(
    fit: dict[str, Any],
    *,
    edit_manifest: dict[str, Any] | None = None,
    forecast_extension_frames: int = 0,
) -> dict[str, Any]:
    if fit.get("backend") != "swr_backend.impulse_analytic":
        raise ImpulseAnalyticRolloutApiError(f"unsupported SWR backend for analytic rollout: {fit.get('backend')}")

    analytic_result_path = _analytic_result_path(fit)
    analytic_result = load_world_reconstruction_fit(analytic_result_path)
    if str(analytic_result.get("model")) != "analytic-event-collision-v1":
        raise ImpulseAnalyticRolloutApiError(f"unsupported analytic result model: {analytic_result.get('model')}")

    edited_fit, removed_object_id = _fit_for_edit(
        fit=fit,
        edit_manifest=edit_manifest,
        forecast_extension_frames=max(int(forecast_extension_frames), 0),
    )
    object_ids, target_plane = analytic_sysid._project_target_to_plane(edited_fit)
    if not object_ids:
        raise ImpulseAnalyticRolloutApiError("no active objects remain for analytic rollout")

    analytic_sysid._validate_supported_shapes(edited_fit, object_ids)
    frames, target, mask = analytic_sysid._target_tensor(object_ids, target_plane)
    device = torch.device("cpu")
    target = target.to(device)
    mask = mask.to(device)

    fps = float(edited_fit["trajectory_physics_initialization"]["fps"])
    scales = analytic_sysid._object_scales(edited_fit, object_ids, target_plane)
    shape_ids, _half_extents, _object_angles, contact_radii, _shape_names, _contact_proxy_names = analytic_sysid._shape_metadata(
        fit=edited_fit,
        object_ids=object_ids,
        object_scales_by_id=scales,
    )
    contact_radii = contact_radii.to(device)

    object_index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    object_pairs = analytic_sysid._object_pairs(len(object_ids))
    pair_index_by_key = {tuple(pair): index for index, pair in enumerate(object_pairs)}
    event_specs = analytic_sysid._build_event_specs(
        _filtered_events(analytic_result.get("events") or analytic_result.get("detected_events") or [], set(object_ids)),
        object_index_by_id,
    )
    pair_mass_ratio, pair_restitution = analytic_sysid._pair_values_from_analytic_result(
        analytic_result,
        object_ids,
        object_pairs,
    )
    parameters = analytic_result.get("parameters", {})
    try:
        v0 = torch.tensor(
            [parameters[object_id]["initial_velocity_plane_m_per_s"] for object_id in object_ids],
            dtype=torch.float64,
            device=device,
        )
        friction = torch.tensor(
            [float(parameters[object_id]["ground_friction"]) for object_id in object_ids],
            dtype=torch.float64,
            device=device,
        )
    except KeyError as exc:
        raise ImpulseAnalyticRolloutApiError(f"analytic result missing object parameter: {exc}") from exc

    active_metadata = analytic_sysid._active_metadata(object_ids, target_plane, frames, target, fps)
    predicted, _velocities, event_records = analytic_sysid._analytic_rollout(
        target=target,
        frames=frames,
        event_specs=event_specs,
        pair_index_by_key=pair_index_by_key,
        contact_radii=contact_radii,
        shape_ids=shape_ids,
        v0=v0,
        friction=friction,
        pair_mass_ratio=torch.tensor(pair_mass_ratio, dtype=torch.float64, device=device),
        pair_restitution=torch.tensor(pair_restitution, dtype=torch.float64, device=device),
        fps=fps,
        event_source=str(analytic_result.get("event_source", analytic_sysid.EVENT_SOURCE_FREE_ROLLOUT)),
        event_window_frames=int(analytic_result.get("event_window_frames", 2)),
        newton_iters=int(analytic_result.get("newton_iters", 8)),
        object_pairs=object_pairs,
        max_free_rollout_events=int(analytic_result.get("max_free_rollout_events", 0)),
        active_metadata=active_metadata,
    )
    simulated = _simulated_trajectories(
        fit=edited_fit,
        object_ids=object_ids,
        frames=frames,
        predicted=predicted,
        mask=mask,
    )
    return {
        "tool": "impulse_analytic_rollout_api",
        "status": "ok",
        "backend": "impulse_analytic_rerollout",
        "edit_manifest": edit_manifest or {"edit_type": "none"},
        "removed_object_id": removed_object_id,
        "object_ids": object_ids,
        "fps": float(fps),
        "forecast_extension_frames": max(int(forecast_extension_frames), 0),
        "source_last_frame": _last_observed_frame(fit.get("target_trajectories") or {}, object_ids),
        "rollout_last_frame": _last_observed_frame(edited_fit.get("target_trajectories") or {}, object_ids),
        "simulated_trajectories": simulated,
        "collisions": _event_summary(event_records, object_ids, fps),
        "fit_error": _simple_fit_error(simulated, edited_fit.get("target_trajectories") or {}, object_ids),
        "notes": [
            "rollout replays optimized impulse analytic parameters from world_reconstruction_fit.json",
            "remove_object edits filter the object and rerun analytic free-contact search over remaining objects",
        ],
    }


def _analytic_result_path(fit: dict[str, Any]) -> Path:
    path = (fit.get("physics_rollout") or {}).get("analytic_result_path")
    if not path:
        raise ImpulseAnalyticRolloutApiError("world reconstruction fit is missing physics_rollout.analytic_result_path")
    result = Path(path)
    if not result.exists():
        raise ImpulseAnalyticRolloutApiError(f"analytic result does not exist: {result}")
    return result


def _fit_for_edit(
    *,
    fit: dict[str, Any],
    edit_manifest: dict[str, Any] | None,
    forecast_extension_frames: int,
) -> tuple[dict[str, Any], str | None]:
    output = copy.deepcopy(fit)
    removed_object_id = None
    if isinstance(edit_manifest, dict) and edit_manifest.get("edit_type") == "remove_object":
        removed_object_id = str(edit_manifest.get("object_id"))
        target = output.get("target_trajectories")
        if not isinstance(target, dict) or removed_object_id not in target:
            raise ImpulseAnalyticRolloutApiError(f"cannot remove missing object: {removed_object_id}")
        output["target_trajectories"] = {
            str(object_id): records
            for object_id, records in target.items()
            if str(object_id) != removed_object_id
        }
    if forecast_extension_frames > 0:
        output["target_trajectories"] = _extend_targets_for_forecast(
            output.get("target_trajectories") or {},
            forecast_extension_frames,
        )
    return output, removed_object_id


def _filtered_events(events: list[dict[str, Any]], active_object_ids: set[str]) -> list[dict[str, Any]]:
    output = []
    for event in events:
        if not isinstance(event, dict):
            continue
        object_ids = [str(object_id) for object_id in event.get("object_ids", [])]
        if len(object_ids) == 2 and all(object_id in active_object_ids for object_id in object_ids):
            output.append(dict(event))
    return output


def _extend_targets_for_forecast(
    target: dict[str, list[dict[str, Any]]],
    extension_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    output = {str(object_id): [copy.deepcopy(record) for record in records] for object_id, records in target.items()}
    for object_id, records in list(output.items()):
        records = sorted(records, key=lambda item: int(item.get("frame_index", 0)))
        if not records:
            continue
        last = records[-1]
        last_frame = int(last["frame_index"])
        last_position = np.asarray(last["position"], dtype=np.float64).reshape(3)
        velocity = _last_observed_velocity(records)
        for offset in range(1, int(extension_frames) + 1):
            position = last_position + velocity * float(offset)
            record = copy.deepcopy(last)
            record["frame_index"] = int(last_frame + offset)
            record["position"] = position.astype(float).tolist()
            record["forecast"] = True
            record["source"] = "qcpr_forecast_extension"
            pose = record.get("pose_4x4")
            if isinstance(pose, list):
                try:
                    pose[0][3] = float(position[0])
                    pose[1][3] = float(position[1])
                    pose[2][3] = float(position[2])
                except (TypeError, IndexError):
                    pass
            records.append(record)
        output[object_id] = records
    return output


def _last_observed_velocity(records: list[dict[str, Any]]) -> np.ndarray:
    observed = [record for record in records if not record.get("forecast")]
    if len(observed) < 2:
        return np.zeros((3,), dtype=np.float64)
    previous = observed[-2]
    last = observed[-1]
    frame_delta = max(int(last["frame_index"]) - int(previous["frame_index"]), 1)
    return (
        np.asarray(last["position"], dtype=np.float64).reshape(3)
        - np.asarray(previous["position"], dtype=np.float64).reshape(3)
    ) / float(frame_delta)


def _simulated_trajectories(
    *,
    fit: dict[str, Any],
    object_ids: list[str],
    frames: list[int],
    predicted: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[dict[str, Any]]]:
    plane = analytic_sysid._plane_from_payload(fit["analytic_support_plane"])
    target_plane = analytic_sysid._target_plane_records_from_fit(fit)
    predicted_np = predicted.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    simulated_2d: dict[str, dict[str, np.ndarray]] = {}
    for object_index, object_id in enumerate(object_ids):
        positions = []
        frame_indices = []
        for frame_offset, frame_index in enumerate(frames):
            if float(mask_np[frame_offset, object_index, 0]) <= 0.0:
                continue
            positions.append(predicted_np[frame_offset, object_index].astype(np.float64))
            frame_indices.append(int(frame_index))
        simulated_2d[object_id] = {
            "positions": np.asarray(positions, dtype=np.float64).reshape((-1, 2)),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        }
    return analytic_sysid._simulated_trajectories(
        object_ids=object_ids,
        target=fit.get("target_trajectories", {}),
        target_plane=target_plane,
        simulated_2d=simulated_2d,
        plane=plane,
    )


def _event_summary(event_records: list[dict[str, Any]], object_ids: list[str], fps: float) -> list[dict[str, Any]]:
    events = []
    for record in event_records:
        indices = [int(index) for index in record.get("object_indices", [])]
        if len(indices) != 2:
            continue
        frame_float = float(record["event_frame_float"].detach().cpu())
        events.append(
            {
                "event_type": str(record.get("event_type", "pair_collision_candidate")),
                "event_source": str(record.get("event_source", "free_rollout")),
                "contact_model": str(record.get("contact_model", "unknown")),
                "frame": int(round(frame_float)),
                "event_time_s": float(record["event_time_s"].detach().cpu()),
                "event_frame_float": frame_float,
                "fps": float(fps),
                "object_ids": [object_ids[index] for index in indices],
                "object_indices": indices,
                "contact_residual_m": float(record["contact_residual_m"].detach().cpu()),
                "relative_normal_velocity_m_per_s": float(record["relative_normal_velocity"].detach().cpu()),
                "normal_plane": [float(value) for value in record["normal"].detach().cpu().tolist()],
                "impulse_plane": [float(value) for value in record["impulse"].detach().cpu().tolist()],
            }
        )
    return events


def _last_observed_frame(target: dict[str, Any], object_ids: list[str]) -> int | None:
    frames = []
    for object_id in object_ids:
        for record in target.get(object_id, []) or []:
            if not isinstance(record, dict) or record.get("frame_index") is None:
                continue
            try:
                frames.append(int(record["frame_index"]))
            except (TypeError, ValueError):
                continue
    return max(frames) if frames else None


def _simple_fit_error(
    simulated: dict[str, list[dict[str, Any]]],
    target: dict[str, list[dict[str, Any]]],
    object_ids: list[str],
) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    all_sq: list[float] = []
    for object_id in object_ids:
        target_by_frame = {
            int(record["frame_index"]): np.asarray(record["position"], dtype=np.float64)
            for record in target.get(object_id, [])
            if isinstance(record, dict) and not record.get("forecast") and record.get("frame_index") is not None
        }
        sq_values = []
        for record in simulated.get(object_id, []):
            frame = int(record.get("frame_index", -1))
            if frame not in target_by_frame:
                continue
            diff = np.asarray(record["position"], dtype=np.float64) - target_by_frame[frame]
            sq = float(np.dot(diff, diff))
            sq_values.append(sq)
            all_sq.append(sq)
        per_object[object_id] = {
            "translation_rmse": math.sqrt(float(np.mean(sq_values))) if sq_values else None,
            "sample_count": len(sq_values),
        }
    return {
        "overall_translation_rmse": math.sqrt(float(np.mean(all_sq))) if all_sq else None,
        "per_object_translation_rmse": per_object,
    }
