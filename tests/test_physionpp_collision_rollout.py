from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from scripts.world_model import (
    physionpp_friction_collision_future_rollout as friction_rollout,
)
from scripts.world_model import (
    physionpp_mass_collision_future_rollout as mass_rollout,
)


class ContinuousObbContactTest(unittest.TestCase):
    @staticmethod
    def _geometry(
        agent_rotation: np.ndarray,
        agent_half_extents: np.ndarray,
        patient_rotation: np.ndarray,
        patient_half_extents: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        axes = friction_rollout._obb_sat_axes(agent_rotation, patient_rotation)
        limits = friction_rollout._obb_projection_limits(
            axes=axes,
            agent_rotation=agent_rotation,
            agent_half_extents=agent_half_extents,
            patient_rotation=patient_rotation,
            patient_half_extents=patient_half_extents,
        )
        return axes, limits

    def test_finds_contact_between_sampled_frames(self) -> None:
        rotation = np.eye(3)
        axes, limits = self._geometry(
            rotation,
            np.full(3, 0.5),
            rotation,
            np.full(3, 0.5),
        )
        fraction = friction_rollout._relative_segment_first_obb_contact(
            relative_start=np.array([3.0, 0.0, 0.0]),
            relative_end=np.array([-3.0, 0.0, 0.0]),
            axes=axes,
            projection_limits=limits,
        )
        self.assertIsNotNone(fraction)
        self.assertAlmostEqual(float(fraction), 1.0 / 3.0, places=12)

    def test_rotated_box_contact_uses_directional_extent(self) -> None:
        angle = math.pi / 4.0
        agent_rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        patient_rotation = np.eye(3)
        axes, limits = self._geometry(
            agent_rotation,
            np.array([2.0, 0.2, 0.2]),
            patient_rotation,
            np.full(3, 0.5),
        )
        fraction = friction_rollout._relative_segment_first_obb_contact(
            relative_start=np.array([4.0, 0.0, 0.0]),
            relative_end=np.zeros(3),
            axes=axes,
            projection_limits=limits,
        )
        self.assertIsNotNone(fraction)
        expected_contact_center_distance = (
            0.2 + 0.5 * (math.sin(angle) + math.cos(angle))
        ) / math.sin(angle)
        expected_fraction = (4.0 - expected_contact_center_distance) / 4.0
        self.assertAlmostEqual(float(fraction), expected_fraction, places=12)

    def test_directional_geometry_avoids_circumsphere_false_positive(self) -> None:
        angle = math.pi / 2.0
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        axes, limits = self._geometry(
            rotation,
            np.array([2.0, 0.1, 0.1]),
            rotation,
            np.array([2.0, 0.1, 0.1]),
        )
        fraction = friction_rollout._relative_segment_first_obb_contact(
            relative_start=np.array([0.5, 0.0, 0.0]),
            relative_end=np.array([0.5, 0.0, 0.0]),
            axes=axes,
            projection_limits=limits,
        )
        gap, _, _ = friction_rollout._relative_segment_closest_obb(
            relative_start=np.array([0.5, 0.0, 0.0]),
            relative_end=np.array([0.5, 0.0, 0.0]),
            axes=axes,
            projection_limits=limits,
        )
        self.assertIsNone(fraction)
        self.assertAlmostEqual(gap, 0.3, places=12)

    def test_future_rollout_continues_from_segment_initial_state(self) -> None:
        identity_pose = np.eye(4).tolist()
        fit = {
            "segments": [
                {
                    "segment": "seg2",
                    "status": "ok",
                    "agent_object_id": "agent",
                    "patient_object_id": "patient",
                    "frame_range": [10, 12],
                    "patient_motion_mode": "stationary",
                    "target_trajectories": {
                        "agent": [
                            {
                                "frame_index": 10,
                                "position_blender_world_m": [0.0, 0.0, 0.0],
                            }
                        ],
                        "patient": [
                            {
                                "frame_index": 10,
                                "position_blender_world_m": [5.0, 0.0, 0.0],
                            }
                        ],
                    },
                    "optimized_initial_velocity_blender_world_m_per_s": {
                        "agent": [2.0, 0.0, 0.0],
                        "patient": [0.0, 0.0, 0.0],
                    },
                    "observed_boundary_state": {
                        "agent": {
                            "pose_blender_world_4x4": identity_pose,
                            "velocity_blender_world_m_per_s": [999.0, 0.0, 0.0],
                        },
                        "patient": {"pose_blender_world_4x4": identity_pose},
                    },
                }
            ],
            "support_plane": {
                "up_direction_blender_world": [0.0, 0.0, 1.0],
                "gravity_direction_blender_world": [0.0, 0.0, -1.0],
                "surface_point_blender_world_m": [0.0, 0.0, 0.0],
            },
            "shared_role_radii_m": {"agent": 0.5, "patient": 0.5},
            "collision_geometry_by_object": {
                object_id: {
                    "type": "oriented_box",
                    "local_center_m": [0.0, 0.0, 0.0],
                    "half_extents_m": [0.5, 0.5, 0.5],
                }
                for object_id in ("agent", "patient")
            },
            "alignment_optimization": {
                "best_parameters": {
                    "agent_ground_friction": 0.1,
                    "patient_ground_friction": 0.1,
                    "gravity_m_per_s2": 9.81,
                },
                "optimizer": {"physics_dt_sec": 0.01},
            },
        }

        def fake_rollout_role(
            **kwargs: object,
        ) -> tuple[np.ndarray, list[dict[str, object]]]:
            frames = kwargs["frames"]
            start = np.asarray(kwargs["start_position"], dtype=np.float64)
            positions = np.repeat(start[None, :], len(frames), axis=0)
            if float(start[0]) == 0.0:
                positions[:, 0] = np.arange(len(frames), dtype=np.float64)
            return positions, []

        with patch.object(
            friction_rollout, "_rollout_role", side_effect=fake_rollout_role
        ) as mocked:
            result = friction_rollout.run_physionpp_friction_collision_future_rollout(
                fit=fit,
                max_future_frames=3,
                stationary_window_frames=0,
            )

        agent_call = mocked.call_args_list[0].kwargs
        self.assertEqual(agent_call["frames"], [10, 11, 12, 13, 14, 15])
        np.testing.assert_allclose(agent_call["start_position"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(agent_call["initial_velocity"], [2.0, 0.0, 0.0])
        self.assertEqual(
            result["rollout_state_policy"],
            "continuous_from_optimized_seg2_initial_state",
        )
        self.assertEqual(result["horizon"]["observed_first_frame"], 10)
        self.assertEqual(result["future_trajectories"]["agent"][0]["frame_index"], 13)
        self.assertAlmostEqual(
            result["patient_contact"]["first_contact"]["frame_index"], 14.0
        )


class MassCollisionFutureRolloutTest(unittest.TestCase):
    @staticmethod
    def _fit() -> dict[str, object]:
        identity_pose = np.eye(4).tolist()
        return {
            "segments": [
                {
                    "segment": "seg2",
                    "status": "ok",
                    "agent_object_id": "agent",
                    "patient_object_id": "patient",
                    "frame_range": [8, 10],
                    "terminal_state": {
                        "agent": {
                            "frame_index": 10,
                            "position_blender_world_m": [0.0, 0.0, 0.5],
                            "velocity_blender_world_m_per_s": [2.0, 0.0, 0.0],
                        },
                        "patient": {
                            "frame_index": 10,
                            "position_blender_world_m": [5.0, 0.0, 0.5],
                            "velocity_blender_world_m_per_s": [0.0, 0.0, 0.0],
                        },
                    },
                    "observed_boundary_state": {
                        "agent": {"pose_blender_world_4x4": identity_pose},
                        "patient": {"pose_blender_world_4x4": identity_pose},
                    },
                }
            ],
            "support_plane": {
                "up_direction_blender_world": [0.0, 0.0, 1.0],
                "gravity_direction_blender_world": [0.0, 0.0, -1.0],
                "surface_point_blender_world_m": [0.0, 0.0, 0.0],
            },
            "shared_role_radii_m": {"agent": 0.5, "patient": 0.5},
            "collision_geometry_by_object": {
                object_id: {
                    "type": "oriented_box",
                    "local_center_m": [0.0, 0.0, 0.0],
                    "half_extents_m": [0.5, 0.5, 0.5],
                }
                for object_id in ("agent", "patient")
            },
            "alignment_optimization": {
                "best_parameters": {
                    "agent_mass_over_ball_mass": 3.0,
                    "ball_agent_restitution": 0.2,
                    "agent_ground_friction": 0.1,
                    "patient_ground_friction": 0.1,
                    "gravity_m_per_s2": 9.81,
                },
                "optimizer": {"physics_dt_sec": 0.01},
            },
        }

    def test_continues_terminal_state_and_detects_continuous_obb_contact(self) -> None:
        def fake_rollout_role(
            **kwargs: object,
        ) -> tuple[np.ndarray, list[dict[str, object]]]:
            frames = kwargs["frames"]
            start = np.asarray(kwargs["start_position"], dtype=np.float64)
            positions = np.repeat(start[None, :], len(frames), axis=0)
            if float(start[0]) == 0.0:
                positions[:, 0] = np.arange(len(frames), dtype=np.float64)
            return positions, []

        with patch.object(
            mass_rollout.collision_future,
            "_rollout_role",
            side_effect=fake_rollout_role,
        ) as mocked:
            result = mass_rollout.run_physionpp_mass_collision_future_rollout(
                fit=self._fit(),
                max_future_frames=6,
                stationary_window_frames=0,
            )

        agent_call = mocked.call_args_list[0].kwargs
        self.assertEqual(agent_call["frames"], list(range(10, 17)))
        np.testing.assert_allclose(agent_call["start_position"], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(agent_call["initial_velocity"], [2.0, 0.0, 0.0])
        self.assertEqual(
            result["rollout_state_policy"],
            "continue_from_fitted_seg2_terminal_state",
        )
        self.assertEqual(result["future_trajectories"]["agent"][0]["frame_index"], 11)
        self.assertTrue(result["patient_contact"]["will_contact"])
        self.assertAlmostEqual(
            result["patient_contact"]["first_contact"]["frame_index"], 14.0
        )

    def test_rejects_mismatched_terminal_frames(self) -> None:
        fit = self._fit()
        fit["segments"][0]["terminal_state"]["patient"]["frame_index"] = 9
        with self.assertRaisesRegex(ValueError, "terminal states at the same frame"):
            mass_rollout.run_physionpp_mass_collision_future_rollout(fit=fit)


if __name__ == "__main__":
    unittest.main()
