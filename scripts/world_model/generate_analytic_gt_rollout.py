from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYTIC_SYSID_PATH = SCRIPT_DIR / "run_impulse_analytic_sysid.py"
ANALYTIC_SYSID_SPEC = importlib.util.spec_from_file_location(
    "physmind_run_impulse_analytic_sysid_for_gt_generation",
    ANALYTIC_SYSID_PATH,
)
if ANALYTIC_SYSID_SPEC is None or ANALYTIC_SYSID_SPEC.loader is None:
    raise RuntimeError(f"failed to import {ANALYTIC_SYSID_PATH}")
analytic_sysid = importlib.util.module_from_spec(ANALYTIC_SYSID_SPEC)
ANALYTIC_SYSID_SPEC.loader.exec_module(analytic_sysid)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plane_to_world(xy: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(plane["origin"], dtype=np.float64).reshape(3)
    tangent_1 = np.asarray(plane["tangent_1"], dtype=np.float64).reshape(3)
    tangent_2 = np.asarray(plane["tangent_2"], dtype=np.float64).reshape(3)
    return origin + float(xy[0]) * tangent_1 + float(xy[1]) * tangent_2


def _records_from_plane_positions(
    *,
    fit: dict[str, Any],
    object_ids: list[str],
    frames: list[int],
    plane_positions: torch.Tensor,
) -> dict[str, list[dict[str, Any]]]:
    plane = fit["analytic_support_plane"]
    source_trajectories = fit.get("target_trajectories", {})
    opencv_from_blender = analytic_sysid.sysid_common._blender_vector_to_opencv_camera
    output: dict[str, list[dict[str, Any]]] = {}
    plane_np = plane_positions.detach().cpu().numpy()
    for object_index, object_id in enumerate(object_ids):
        source_records = source_trajectories.get(object_id, [])
        source_by_frame = {
            int(record.get("frame_index")): record
            for record in source_records
            if isinstance(record, dict) and record.get("frame_index") is not None
        }
        records = []
        for frame_offset, frame in enumerate(frames):
            source = source_by_frame.get(int(frame))
            if source is None:
                continue
            record = copy.deepcopy(source)
            xy = np.asarray(plane_np[frame_offset, object_index], dtype=np.float64)
            position = _plane_to_world(xy, plane)
            position_opencv = opencv_from_blender(position.astype(float).tolist())
            record["frame_index"] = int(frame)
            record["position"] = [float(value) for value in position.tolist()]
            record["position_opencv_camera"] = [float(value) for value in position_opencv]
            pose = record.get("pose_4x4")
            if isinstance(pose, list) and len(pose) >= 3:
                pose = copy.deepcopy(pose)
                pose[0][3] = float(position_opencv[0])
                pose[1][3] = float(position_opencv[1])
                pose[2][3] = float(position_opencv[2])
                record["pose_4x4"] = pose
            records.append(record)
        output[object_id] = records
    return output


def _event_summary(event_records: list[dict[str, Any]], object_ids: list[str], fps: float) -> list[dict[str, Any]]:
    events = []
    for record in event_records:
        object_indices = [int(index) for index in record["object_indices"]]
        frame_float = float(record["event_frame_float"].detach().cpu())
        events.append(
            {
                "event_index": int(record["event_index"].detach().cpu()),
                "event_type": str(record.get("event_type", "pair_collision_candidate")),
                "event_source": str(record.get("event_source", "free_rollout")),
                "contact_model": str(record.get("contact_model", "unknown")),
                "frame": int(round(frame_float)),
                "event_time_s": float(record["event_time_s"].detach().cpu()),
                "event_frame_float": frame_float,
                "fps": float(fps),
                "object_ids": [object_ids[index] for index in object_indices],
                "object_indices": object_indices,
                "contact_residual_m": float(record["contact_residual_m"].detach().cpu()),
                "relative_normal_velocity_m_per_s": float(record["relative_normal_velocity"].detach().cpu()),
                "normal_plane": [float(value) for value in record["normal"].detach().cpu().tolist()],
                "impulse_plane": [float(value) for value in record["impulse"].detach().cpu().tolist()],
            }
        )
    return events


def _true_parameter_tensors(
    *,
    true_parameters: dict[str, Any],
    object_ids: list[str],
    object_pairs: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_xy = torch.tensor(
        [true_parameters["initial_plane_xy"][object_id] for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    v0 = torch.tensor(
        [true_parameters["initial_velocity_plane_m_per_s"][object_id] for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    friction = torch.tensor(
        [true_parameters["ground_friction"][object_id] for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    mass = torch.tensor(
        [true_parameters["mass"][object_id] for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    restitution = torch.tensor(
        [true_parameters["restitution"][object_id] for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    pair_mass_ratio = torch.tensor(
        [float(mass[i].detach().cpu()) / max(float(mass[j].detach().cpu()), 1e-12) for i, j in object_pairs],
        dtype=torch.float64,
        device=device,
    )
    pair_restitution = torch.tensor(
        [float(restitution[i].detach().cpu()) * float(restitution[j].detach().cpu()) for i, j in object_pairs],
        dtype=torch.float64,
        device=device,
    )
    return initial_xy, v0, friction, pair_mass_ratio, pair_restitution


def generate(
    *,
    input_fit: Path,
    summary: Path,
    output_dir: Path,
    max_free_rollout_events: int | None,
    newton_iters: int,
) -> dict[str, Any]:
    fit = _load_json(input_fit)
    summary_payload = _load_json(summary)
    true_parameters = summary_payload.get("true_parameters")
    if not isinstance(true_parameters, dict):
        raise ValueError(f"summary does not contain true_parameters: {summary}")

    object_ids, target_plane = analytic_sysid._project_target_to_plane(fit)
    analytic_sysid._validate_supported_shapes(fit, object_ids)
    frames, original_target, mask = analytic_sysid._target_tensor(object_ids, target_plane)
    device = torch.device("cpu")
    original_target = original_target.to(device)
    mask = mask.to(device)
    object_scales_by_id = analytic_sysid._object_scales(fit, object_ids, target_plane)
    object_scales = torch.tensor(
        [float(object_scales_by_id[object_id]) for object_id in object_ids],
        dtype=torch.float64,
        device=device,
    )
    shape_ids, half_extents, object_angles, contact_radii, shape_names, contact_proxy_names = analytic_sysid._shape_metadata(
        fit=fit,
        object_ids=object_ids,
        object_scales_by_id=object_scales_by_id,
    )
    half_extents = half_extents.to(device)
    object_angles = object_angles.to(device)
    contact_radii = contact_radii.to(device)
    object_pairs = analytic_sysid._object_pairs(len(object_ids))
    pair_index_by_key = {tuple(pair): index for index, pair in enumerate(object_pairs)}
    initial_xy, v0, friction, pair_mass_ratio, pair_restitution = _true_parameter_tensors(
        true_parameters=true_parameters,
        object_ids=object_ids,
        object_pairs=object_pairs,
        device=device,
    )
    fps = float(fit["trajectory_physics_initialization"]["fps"])
    rollout_target = original_target.clone()
    active_metadata = analytic_sysid._active_metadata(object_ids, target_plane, frames, rollout_target, fps)
    for object_index in range(len(object_ids)):
        first_offset = int(active_metadata["first_offsets"].detach().cpu()[object_index])
        last_offset = int(active_metadata["last_offsets"].detach().cpu()[object_index])
        if 0 <= first_offset <= last_offset:
            rollout_target[first_offset, object_index] = initial_xy[object_index]
    active_metadata = analytic_sysid._active_metadata(object_ids, target_plane, frames, rollout_target, fps)
    true_contact_count = len(summary_payload.get("true_replay_contacts") or [])
    event_cap = (
        int(max_free_rollout_events)
        if max_free_rollout_events is not None and int(max_free_rollout_events) > 0
        else true_contact_count
    )
    if event_cap <= 0:
        event_cap = len(object_pairs)

    with torch.no_grad():
        predicted, velocities, event_records = analytic_sysid._analytic_rollout(
            target=rollout_target,
            frames=frames,
            event_specs=[],
            pair_index_by_key=pair_index_by_key,
            contact_radii=contact_radii,
            shape_ids=shape_ids,
            v0=v0,
            friction=friction,
            pair_mass_ratio=pair_mass_ratio,
            pair_restitution=pair_restitution,
            fps=fps,
            event_source=analytic_sysid.EVENT_SOURCE_FREE_ROLLOUT,
            event_window_frames=4,
            newton_iters=newton_iters,
            object_pairs=object_pairs,
            max_free_rollout_events=event_cap,
            active_metadata=active_metadata,
        )
        original_rmse, original_per_object_rmse = analytic_sysid._rmse_summary(
            predicted,
            original_target,
            mask,
            object_ids,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _records_from_plane_positions(
        fit=fit,
        object_ids=object_ids,
        frames=frames,
        plane_positions=predicted,
    )
    events = _event_summary(event_records, object_ids, fps)

    rollout_payload = {
        "status": "ok",
        "simulator": "analytic_event_free_rollout",
        "source_fit": str(input_fit),
        "source_summary": str(summary),
        "fps": float(fps),
        "frames": [int(frame) for frame in frames],
        "parameters": true_parameters,
        "active_intervals": analytic_sysid._active_interval_summary(object_ids, active_metadata),
        "contact_proxy_policy": "box_equal_area_circle",
        "object_shapes": {
            object_id: {
                "shape": shape_names[object_index],
                "scale_m": float(object_scales.detach().cpu()[object_index]),
                "contact_proxy": contact_proxy_names[object_index],
                "contact_radius_m": float(contact_radii.detach().cpu()[object_index]),
                "half_extents_2d_m": [
                    float(value) for value in half_extents.detach().cpu()[object_index].tolist()
                ],
                "fixed_yaw_rad": float(object_angles.detach().cpu()[object_index]),
            }
            for object_index, object_id in enumerate(object_ids)
        },
        "pair_parameters": [
            {
                "object_ids": [object_ids[i], object_ids[j]],
                "mass_ratio_i_over_j": float(pair_mass_ratio.detach().cpu()[pair_index]),
                "pair_restitution": float(pair_restitution.detach().cpu()[pair_index]),
            }
            for pair_index, (i, j) in enumerate(object_pairs)
        ],
        "events": events,
        "simulated_trajectories": trajectories,
        "analytic_vs_original_rmse_m": float(original_rmse.detach().cpu()),
        "analytic_vs_original_per_object_rmse_m": original_per_object_rmse,
    }
    rollout_path = output_dir / "analytic_gt_rollout.json"
    rollout_path.write_text(json.dumps(rollout_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fit_payload = copy.deepcopy(fit)
    fit_payload["target_trajectories"] = trajectories
    fit_payload.setdefault("physics_rollout", {})["simulated_trajectories"] = trajectories
    fit_payload["analytic_gt_generation"] = {
        "source_fit": str(input_fit),
        "source_summary": str(summary),
        "rollout_path": str(rollout_path),
        "simulator": "analytic_event_free_rollout",
        "newton_iters": int(newton_iters),
        "max_free_rollout_events": int(event_cap),
        "contact_proxy_policy": "box_equal_area_circle",
        "analytic_vs_original_rmse_m": float(original_rmse.detach().cpu()),
        "active_intervals": analytic_sysid._active_interval_summary(object_ids, active_metadata),
    }
    fit_path = output_dir / "analytic_gt_fit.json"
    fit_path.write_text(json.dumps(fit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = output_dir / "analytic_gt_vs_original.png"
    analytic_sysid._plot(
        plot_path,
        object_ids,
        original_target.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        mask.detach().cpu().numpy(),
    )

    summary_out = {
        "status": "ok",
        "input_fit": str(input_fit),
        "source_summary": str(summary),
        "analytic_gt_rollout": str(rollout_path),
        "analytic_gt_fit": str(fit_path),
        "plot_path": str(plot_path),
        "analytic_vs_original_rmse_m": float(original_rmse.detach().cpu()),
        "analytic_vs_original_per_object_rmse_m": original_per_object_rmse,
        "contact_proxy_policy": "box_equal_area_circle",
        "active_intervals": analytic_sysid._active_interval_summary(object_ids, active_metadata),
        "event_count": len(events),
        "events": events,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fit", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-free-rollout-events", type=int, default=None)
    parser.add_argument("--newton-iters", type=int, default=8)
    args = parser.parse_args()
    input_fit = Path(args.input_fit)
    summary = Path(args.summary) if args.summary is not None else input_fit.parent / "summary.json"
    result = generate(
        input_fit=input_fit,
        summary=summary,
        output_dir=Path(args.output_dir),
        max_free_rollout_events=args.max_free_rollout_events,
        newton_iters=args.newton_iters,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
