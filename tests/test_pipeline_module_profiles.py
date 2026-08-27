from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from agent.world_model.module_profiles import (
    DEFAULT_MODULE_PROFILE_PATH,
    ModuleProfileValidationError,
    load_module_profile_policy,
    parse_module_profile_policy,
)
from agent.world_model.route_policy import load_route_policy
from scripts.world_model.run_sam3_video_tracks import (
    _pp_reduce_identity_candidates_by_reference,
    _pp_reduce_seg1_identity_candidates,
    _require_tracking_profile_scenario,
)


class ModuleProfilePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(
            Path(DEFAULT_MODULE_PROFILE_PATH).read_text(encoding="utf-8")
        )

    def test_every_module_profile_is_reachable_from_the_route_policy(self) -> None:
        module_policy = load_module_profile_policy()
        route_policy = load_route_policy()
        for route, profile in module_policy.route_profiles.items():
            source_scopes = [
                resolution.scope
                for resolution in route_policy.decision(
                    profile.decision_id
                ).resolutions
                if resolution.route == route
            ]
            for option in route_policy.approved_optional_routes:
                if any(
                    override.decision_id == profile.decision_id
                    and override.route == route
                    for override in option.route_overrides
                ):
                    source_scopes.append(option.scope)

            with self.subTest(route=route):
                self.assertTrue(source_scopes)
                self.assertEqual(
                    {
                        benchmark
                        for scope in source_scopes
                        for benchmark in scope.benchmarks
                    },
                    {profile.benchmark},
                )
                declared_scenarios = {
                    scenario for scope in source_scopes for scenario in scope.scenarios
                }
                self.assertEqual(declared_scenarios, set(profile.scenarios))
                resolved = module_policy.resolve_route(
                    profile.decision_id,
                    route,
                    benchmark=profile.benchmark,
                    scenario=profile.scenarios[0] if profile.scenarios else "",
                )
                self.assertEqual(resolved, profile)
                self.assertTrue(profile.modules)

    def test_representative_module_parameters_are_typed(self) -> None:
        policy = load_module_profile_policy()
        cleanup = policy.shared_module("gdino_cleanup")
        self.assertEqual(cleanup.require_number("confidence"), 0.25)
        self.assertEqual(cleanup.require_integer("edge_sides_to_drop"), 3)

        bouncy_platform = policy.resolve_tracking(
            "tracking.gdino_per_part_seeds",
            scenario="bouncy_platform_pp",
        )
        self.assertEqual(
            bouncy_platform.module("sam3_seed").require_integers(
                "agent_fallback_frames"
            ),
            (50, 70, 40, 80, 30),
        )

        wall_mesh = policy.resolve_route(
            "GEO-004.mesh_conditioning",
            "mesh_conditioning.obb_rollback",
            benchmark="physion_pp",
            scenario="bouncy_wall_pp",
        ).module("mesh_conditioning")
        self.assertEqual(wall_mesh.require_string("box_fit"), "obb")
        self.assertEqual(wall_mesh.require_string("alignment_safety"), "rollback")

        friction_support = policy.resolve_route(
            "POS-006.support_snap",
            "support.platform_sphere_agent",
            benchmark="physion_pp",
            scenario="friction_platform_pp",
        ).module("support_snap")
        self.assertTrue(friction_support.require_boolean("exclude_sphere_agents"))
        self.assertTrue(
            friction_support.require_boolean("honor_airborne_exemptions")
        )

    def test_invalid_profiles_and_scope_mismatches_fail(self) -> None:
        malformed = copy.deepcopy(self.payload)
        malformed["unexpected"] = True
        with self.assertRaisesRegex(ModuleProfileValidationError, "keys must be"):
            parse_module_profile_policy(malformed)

        policy = load_module_profile_policy()
        with self.assertRaisesRegex(ModuleProfileValidationError, "no module profile"):
            policy.resolve_tracking("unknown", scenario="friction_platform_pp")
        with self.assertRaisesRegex(
            ModuleProfileValidationError, "does not allow scenario"
        ):
            policy.resolve_tracking(
                "tracking.gdino_two_segment_point_probe",
                scenario="mass_collision_pp",
            )
        with self.assertRaisesRegex(ModuleProfileValidationError, "must be an integer"):
            policy.shared_module("gdino_cleanup").require_integer("confidence")

    def test_tracking_frontend_checks_its_profile_scenario(self) -> None:
        profile = load_module_profile_policy().resolve_tracking(
            "tracking.gdino_two_segment_boxes",
            scenario="mass_collision_pp",
        )
        _require_tracking_profile_scenario(
            profile,
            "mass_collision_pp",
            frontend="mass-collision tracking",
        )
        with self.assertRaisesRegex(ValueError, "does not support scenario"):
            _require_tracking_profile_scenario(
                profile,
                "friction_collision_pp",
                frontend="mass-collision tracking",
            )

    def test_three_seg1_candidates_reduce_to_distinct_agent_patient_pair(self) -> None:
        seg1_image = np.zeros((8, 12, 3), dtype=np.uint8)
        seg2_image = np.zeros_like(seg1_image)
        masks = {}
        for index, (track_id, color) in enumerate(
            (
                ("agent_like", (20, 40, 180)),
                ("patient_like", (40, 180, 30)),
                ("extra", (220, 220, 220)),
            )
        ):
            mask = np.zeros(seg1_image.shape[:2], dtype=bool)
            mask[1:4, index * 4 : index * 4 + 3] = True
            masks[track_id] = mask
            seg1_image[mask] = color

        seg2_agent_mask = np.zeros(seg2_image.shape[:2], dtype=bool)
        seg2_agent_mask[4:7, 0:3] = True
        seg2_image[seg2_agent_mask] = (22, 42, 178)
        seg2_patient_mask = np.zeros(seg2_image.shape[:2], dtype=bool)
        seg2_patient_mask[4:7, 4:7] = True
        seg2_image[seg2_patient_mask] = (38, 182, 32)

        reduction = _pp_reduce_seg1_identity_candidates(
            seg1_image=seg1_image,
            seg2_image=seg2_image,
            candidate_masks=masks,
            seg2_agent_mask=seg2_agent_mask,
            seg2_patient_mask=seg2_patient_mask,
        )

        self.assertEqual(
            reduction["selected_role_hypothesis"],
            {"agent": "agent_like", "patient": "patient_like"},
        )
        self.assertEqual(reduction["pruned_tracks"], ["extra"])
        self.assertEqual(
            reduction["method"], "joint_agent_patient_median_color_prune"
        )

    def test_mass_identity_reduces_multiple_candidates_before_vlm_ab(self) -> None:
        candidate_image = np.zeros((8, 12, 3), dtype=np.uint8)
        reference_image = np.zeros_like(candidate_image)
        masks = {}
        for index, (track_id, color) in enumerate(
            (
                ("agent_like", (20, 40, 180)),
                ("second_closest", (30, 55, 160)),
                ("far", (220, 220, 220)),
            )
        ):
            mask = np.zeros(candidate_image.shape[:2], dtype=bool)
            mask[1:4, index * 4 : index * 4 + 3] = True
            masks[track_id] = mask
            candidate_image[mask] = color
        reference_mask = np.zeros(reference_image.shape[:2], dtype=bool)
        reference_mask[4:7, 0:3] = True
        reference_image[reference_mask] = (22, 42, 178)

        reduction = _pp_reduce_identity_candidates_by_reference(
            candidate_image=candidate_image,
            reference_image=reference_image,
            candidate_masks=masks,
            reference_mask=reference_mask,
        )

        self.assertEqual(
            reduction["selected_tracks"], ["agent_like", "second_closest"]
        )
        self.assertEqual(reduction["pruned_tracks"], ["far"])
        self.assertEqual(reduction["method"], "reference_median_color_prune")


if __name__ == "__main__":
    unittest.main()
