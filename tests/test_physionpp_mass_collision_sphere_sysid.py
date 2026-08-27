from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np
import torch

from scripts.world_model import analytic_swr_common
from scripts.world_model import run_physionpp_mass_collision_sphere_sysid as mass_swr
from scripts.world_model import swr_sysid_common


def _geometry_source_route() -> dict[str, Any]:
    return {
        "decision_id": "SWR-003.fit_geometry_source",
        "route": "geometry.corrected_mesh",
    }


def _check_offline_manifest_uses_formal_final_mesh_not_mesh_conditioning() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        world_modeling_dir = Path(temporary_directory)
        manifest_path = (
            world_modeling_dir
            / "simulatable-world-reconstruction"
            / "fit"
            / "world_reconstruction_fit_manifest.json"
        )
        mesh_conditioning_path = (
            world_modeling_dir
            / "metric-mesh-reconstruction"
            / "mesh_conditioning"
            / "mesh_conditioning.json"
        )
        mesh_path = world_modeling_dir / "meshes" / "obj_1.glb"
        for path in (manifest_path, mesh_conditioning_path, mesh_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"test mesh path")
        manifest_path.write_text(
            json.dumps(
                {
                    "scene_index": 1,
                    "question_id": 0,
                    "object_plan": {
                        "target_objects": [
                            {"object_id": "obj_1", "geometry_type": "sphere"}
                        ]
                    },
                    "swr_fit_geometry_source_route": _geometry_source_route(),
                    "target_trajectories": {
                        "objects": [
                            {
                                "object_id": "obj_1",
                                "status": "ok",
                                "mesh_path": str(mesh_path),
                                "poses": [],
                            }
                        ]
                    },
                }
            )
        )
        mesh_conditioning_path.write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "object_id": "obj_1",
                            "geometry_type": "sphere",
                            "foundationpose_mesh_path": str(
                                world_modeling_dir / "must_not_be_read.glb"
                            ),
                        }
                    ]
                }
            )
        )

        manifest = swr_sysid_common.build_manifest_from_world_modeling(
            world_modeling_dir
        )
        recorded_mesh_path = Path(manifest["target_trajectories"]["objects"][0]["mesh_path"])

        assert recorded_mesh_path == mesh_path
        assert recorded_mesh_path.exists()
        assert (
            manifest["target_trajectories"]["objects"][0]["mesh_source"]
            == "geometry.corrected_mesh"
        )
        assert manifest["swr_fit_geometry_source_route"] == _geometry_source_route()


def _check_offline_manifest_rejects_missing_or_nonexistent_final_mesh() -> None:
    for mesh_value, expected_message in (
        (None, "missing pose-correction final effective mesh_path"),
        ("/definitely/missing/final_mesh.glb", "does not exist"),
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            world_modeling_dir = Path(temporary_directory)
            manifest_path = (
                world_modeling_dir
                / "simulatable-world-reconstruction"
                / "fit"
                / "world_reconstruction_fit_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            target = {"object_id": "obj_1", "status": "ok", "poses": []}
            if mesh_value is not None:
                target["mesh_path"] = mesh_value
            manifest_path.write_text(
                json.dumps(
                    {
                        "swr_fit_geometry_source_route": _geometry_source_route(),
                        "target_trajectories": {"objects": [target]},
                    }
                )
            )

            try:
                swr_sysid_common.build_manifest_from_world_modeling(
                    world_modeling_dir
                )
            except ValueError as exc:
                assert expected_message in str(exc), str(exc)
            else:
                raise AssertionError(
                    f"expected final-effective-mesh failure for {mesh_value!r}"
                )


def _piecewise_collision_tracks(
    *,
    mass_ratio: float,
    restitution: float,
    keep_ball_post_contact: bool,
) -> dict[str, Any]:
    contact_frame = 10
    frames = list(range(21))
    pre_velocity = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    post_velocity, _impulse, _relative = analytic_swr_common.sphere_sphere_impulse_update_torch(
        velocities=pre_velocity,
        i=0,
        j=1,
        normal_from_j_to_i=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        mass_i=torch.tensor(1.0, dtype=torch.float64),
        mass_j=torch.tensor(mass_ratio, dtype=torch.float64),
        restitution=torch.tensor(restitution, dtype=torch.float64),
    )
    contact_positions = np.asarray([[-0.15, 0.0, 0.2], [0.15, 0.0, 0.2]], dtype=np.float64)
    positions = np.empty((len(frames), 2, 3), dtype=np.float64)
    for index, frame in enumerate(frames):
        time = (frame - contact_frame) * mass_swr.PHYSIONPP_PHYSICS_DT_SEC
        velocities = pre_velocity if frame <= contact_frame else post_velocity
        positions[index] = contact_positions + velocities.detach().cpu().numpy() * time
    ball_frames = frames if keep_ball_post_contact else frames[: contact_frame + 1]
    return {
        "segment": "seg1",
        "ball_agent_contact_frame": contact_frame,
        "ball_frames": ball_frames,
        "agent_frames": frames,
        "target_ball_np": positions[: len(ball_frames), 0],
        "target_agent_np": positions[:, 1],
    }


def _check_shared_impulse_update_obeys_momentum_and_restitution() -> None:
    velocities = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    updated, impulse, relative_normal = analytic_swr_common.sphere_sphere_impulse_update_torch(
        velocities=velocities,
        i=0,
        j=1,
        normal_from_j_to_i=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        mass_i=torch.tensor(1.0, dtype=torch.float64),
        mass_j=torch.tensor(3.0, dtype=torch.float64),
        restitution=torch.tensor(0.2, dtype=torch.float64),
    )

    np.testing.assert_allclose(relative_normal.item(), -4.0)
    np.testing.assert_allclose(impulse.detach().cpu().numpy(), [-3.6, 0.0, 0.0])
    np.testing.assert_allclose(updated.detach().cpu().numpy(), [[0.4, 0.0, 0.0], [1.2, 0.0, 0.0]])
    momentum_before = velocities[0] + 3.0 * velocities[1]
    momentum_after = updated[0] + 3.0 * updated[1]
    np.testing.assert_allclose(momentum_after.detach().cpu().numpy(), momentum_before.detach().cpu().numpy())


def _check_collision_initialization_recovers_two_sided_mass_and_restitution() -> None:
    prepared = _piecewise_collision_tracks(
        mass_ratio=3.0,
        restitution=0.2,
        keep_ball_post_contact=True,
    )

    mass_ratio, restitution, observations = mass_swr._collision_initialization([prepared])

    np.testing.assert_allclose(mass_ratio, 3.0, atol=1e-8)
    np.testing.assert_allclose(restitution, 0.2, atol=1e-8)
    assert observations[0]["source"] == "two_sided_velocity_change"


def _check_collision_initialization_recovers_mass_when_ball_track_is_cut() -> None:
    prepared = _piecewise_collision_tracks(
        mass_ratio=3.0,
        restitution=0.0,
        keep_ball_post_contact=False,
    )

    mass_ratio, restitution, observations = mass_swr._collision_initialization([prepared])

    np.testing.assert_allclose(mass_ratio, 3.0, atol=1e-8)
    np.testing.assert_allclose(restitution, 0.0, atol=1e-8)
    assert observations[0]["source"] == "ball_pre_and_agent_post_with_default_restitution"


def _check_closed_form_segment_finds_continuous_contact_and_has_finite_gradients() -> None:
    mass_ratio = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    restitution = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    friction = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    frames = list(range(51))
    positions, velocities, events, _plane_events = mass_swr._rollout_ball_agent_segment(
        frames=frames,
        contact_frame=18,
        start_positions=torch.tensor([[-1.0, 0.0, 0.2], [0.0, 0.0, 0.2]], dtype=torch.float64),
        initial_velocities=torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
        agent_friction=friction,
        ball_obb_rotation=torch.eye(3, dtype=torch.float64),
        ball_obb_half_extents=torch.tensor([0.15, 0.15, 0.15], dtype=torch.float64),
        agent_obb_rotation=torch.eye(3, dtype=torch.float64),
        agent_obb_half_extents=torch.tensor([0.15, 0.15, 0.15], dtype=torch.float64),
        agent_mass_over_ball_mass=mass_ratio,
        restitution=restitution,
        support={
            "up": np.asarray([0.0, 0.0, 1.0]),
            "gravity_direction": np.asarray([0.0, 0.0, 0.0]),
        },
    )

    event = events[0]
    np.testing.assert_allclose(event["frame_index"].item(), 17.5, atol=1e-8)
    np.testing.assert_allclose(event["contact_residual_m"].item(), 0.0, atol=1e-8)
    np.testing.assert_allclose(
        event["post_velocity"].detach().cpu().numpy(),
        [[0.4, 0.0, 0.0], [1.2, 0.0, 0.0]],
    )
    assert positions.shape == (51, 2, 3)
    assert velocities.shape == (51, 2, 3)

    loss = positions[-1].square().sum() + event["contact_residual_m"].square()
    loss.backward()
    for parameter in (mass_ratio, restitution, friction):
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


class MassCollisionSphereSysidTest(unittest.TestCase):
    def test_offline_manifest_uses_final_effective_mesh(self) -> None:
        _check_offline_manifest_uses_formal_final_mesh_not_mesh_conditioning()

    def test_offline_manifest_rejects_invalid_final_mesh(self) -> None:
        _check_offline_manifest_rejects_missing_or_nonexistent_final_mesh()

    def test_shared_impulse_update_obeys_physics(self) -> None:
        _check_shared_impulse_update_obeys_momentum_and_restitution()

    def test_collision_initialization_recovers_two_sided_parameters(self) -> None:
        _check_collision_initialization_recovers_two_sided_mass_and_restitution()

    def test_collision_initialization_recovers_mass_from_cut_ball_track(self) -> None:
        _check_collision_initialization_recovers_mass_when_ball_track_is_cut()

    def test_closed_form_segment_contact_and_gradients(self) -> None:
        _check_closed_form_segment_finds_continuous_contact_and_has_finite_gradients()


if __name__ == "__main__":
    unittest.main()
