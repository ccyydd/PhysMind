from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import physmind
from agent import direct_answer
from agent.world_model.foundationpose_worker import (
    DEFAULT_FOUNDATIONPOSE_WORKER_CMD,
)
from agent.world_model.geocalib_worker import DEFAULT_GEOCALIB_WORKER_CMD
from agent.world_model.moge2_worker import DEFAULT_MOGE2_WORKER_CMD
from agent.world_model.route_policy import (
    DEFAULT_ROUTE_POLICY_PATH,
    RouteContext,
    RoutePolicyValidationError,
    load_route_policy,
    parse_route_policy,
)
from agent.world_model.sam3_video_tracks_worker import (
    DEFAULT_SAM3_VIDEO_TRACKS_WORKER_CMD,
)
from agent.world_model.sam3d_worker import DEFAULT_SAM3D_WORKER_CMD
from agent.world_model.video_metric_depth_worker import (
    DEFAULT_VIDEO_METRIC_DEPTH_WORKER_CMD,
)


PROFILE_CONTEXTS = {
    "clevrer": RouteContext(benchmark="clevrer"),
    "friction_platform_pp": RouteContext(
        benchmark="physion_pp", scenario="friction_platform_pp"
    ),
    "bouncy_wall_pp": RouteContext(
        benchmark="physion_pp", scenario="bouncy_wall_pp"
    ),
    "bouncy_platform_pp": RouteContext(
        benchmark="physion_pp", scenario="bouncy_platform_pp"
    ),
    "friction_collision_pp": RouteContext(
        benchmark="physion_pp", scenario="friction_collision_pp"
    ),
    "mass_collision_pp": RouteContext(
        benchmark="physion_pp", scenario="mass_collision_pp"
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_CORE_ROUTES = {
    "INP-003.temporal_partition": {
        "clevrer": "temporal.single_video",
        "friction_platform_pp": "temporal.single_video",
        "bouncy_wall_pp": "temporal.cue_two_segment",
        "bouncy_platform_pp": "temporal.single_video",
        "friction_collision_pp": "temporal.cue_two_segment",
        "mass_collision_pp": "temporal.cue_two_segment",
    },
    "TRK-001.tracking_family": {
        "clevrer": "tracking.sam3_broad_text",
        "friction_platform_pp": "tracking.gdino_role_box_point",
        "bouncy_wall_pp": "tracking.gdino_two_segment_point_probe",
        "bouncy_platform_pp": "tracking.gdino_per_part_seeds",
        "friction_collision_pp": "tracking.sam3_two_segment_text",
        "mass_collision_pp": "tracking.gdino_two_segment_boxes",
    },
    "GEO-004.mesh_conditioning": {
        "clevrer": "mesh_conditioning.aabb_no_guard",
        "friction_platform_pp": "mesh_conditioning.obb_warn",
        "bouncy_wall_pp": "mesh_conditioning.obb_rollback",
        "bouncy_platform_pp": "mesh_conditioning.obb_warn",
        "friction_collision_pp": "mesh_conditioning.obb_warn",
        "mass_collision_pp": "mesh_conditioning.obb_warn",
    },
    "SWR-001.fit_backend": {
        "clevrer": "swr_backend.impulse_analytic",
        "friction_platform_pp": "swr_backend.surface_friction_sphere",
        "bouncy_wall_pp": "swr_backend.wall_bounce_sphere",
        "bouncy_platform_pp": "swr_backend.platform_bounce_sphere",
        "friction_collision_pp": "swr_backend.collision_friction_spheres",
        "mass_collision_pp": "swr_backend.collision_mass_spheres",
    },
    "ROL-001.question_rollout_backend": {
        "clevrer": "rollout.tool_physics",
        "friction_platform_pp": "rollout.surface_friction_analytic",
        "bouncy_wall_pp": "rollout.wall_bounce_analytic",
        "bouncy_platform_pp": "rollout.platform_bounce_analytic",
        "friction_collision_pp": "rollout.collision_friction_analytic",
        "mass_collision_pp": "rollout.collision_mass_analytic",
    },
    "ANS-001.success_answer_backend": {
        "clevrer": "answer.vlm_tools",
        "friction_platform_pp": "answer.contact_artifact",
        "bouncy_wall_pp": "answer.contact_artifact",
        "bouncy_platform_pp": "answer.contact_artifact",
        "friction_collision_pp": "answer.contact_artifact",
        "mass_collision_pp": "answer.contact_artifact",
    },
}


class PipelineContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_route_policy()

    def test_route_policy_is_strict_and_current(self) -> None:
        self.assertEqual(self.policy.schema_version, 2)
        self.assertEqual(
            {item.benchmark for item in self.policy.benchmarks},
            {"clevrer", "physion_pp"},
        )
        payload = json.loads(DEFAULT_ROUTE_POLICY_PATH.read_text(encoding="utf-8"))
        malformed = copy.deepcopy(payload)
        malformed["unexpected"] = True
        with self.assertRaises(RoutePolicyValidationError):
            parse_route_policy(malformed)

    def test_core_route_matrix_for_all_supported_profiles(self) -> None:
        for decision_id, expected_by_profile in EXPECTED_CORE_ROUTES.items():
            with self.subTest(decision_id=decision_id):
                actual = {
                    profile: self.policy.resolve(decision_id, context).route
                    for profile, context in PROFILE_CONTEXTS.items()
                }
                self.assertEqual(actual, expected_by_profile)

    def test_every_declared_resolution_matches_and_serializes(self) -> None:
        for decision in self.policy.decisions:
            for resolution in decision.resolutions:
                scope = resolution.scope
                context = RouteContext(
                    benchmark=scope.benchmarks[0],
                    scenario=scope.scenarios[0] if scope.scenarios else None,
                    segment=scope.segments[0] if scope.segments else None,
                    role=scope.roles[0] if scope.roles else None,
                    question_type=(
                        scope.question_types[0] if scope.question_types else None
                    ),
                    entrypoint=scope.entrypoints[0] if scope.entrypoints else None,
                )
                with self.subTest(decision_id=decision.decision_id, context=context):
                    self.assertEqual(
                        self.policy.resolve(decision.decision_id, context), resolution
                    )
                    json.dumps(self.policy.resolve_record(decision.decision_id, context))

    def test_optional_routes_are_explicit_default_off_overrides(self) -> None:
        self.assertTrue(self.policy.approved_optional_routes)
        for option in self.policy.approved_optional_routes:
            self.assertFalse(option.default_enabled)
            scope = option.scope
            context = RouteContext(
                benchmark=scope.benchmarks[0],
                scenario=scope.scenarios[0] if scope.scenarios else None,
                segment=scope.segments[0] if scope.segments else None,
                role=scope.roles[0] if scope.roles else None,
                question_type=(scope.question_types[0] if scope.question_types else None),
                entrypoint=scope.entrypoints[0] if scope.entrypoints else None,
            )
            for override in option.route_overrides:
                with self.subTest(option=option.option_id, decision=override.decision_id):
                    base = self.policy.resolve_record(override.decision_id, context)
                    enabled = self.policy.resolve_record(
                        override.decision_id,
                        context,
                        enabled_option_ids=(option.option_id,),
                    )
                    self.assertNotEqual(base["route"], override.route)
                    self.assertEqual(enabled["route"], override.route)
                    self.assertEqual(enabled["option_id"], option.option_id)

    def test_cli_exposes_only_supported_datasets_and_modes(self) -> None:
        parser = physmind.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertEqual(set(actions["bench"].choices), {"clevrer", "physion_pp"})
        self.assertEqual(
            set(actions["mode"].choices), {"direct-answer", "world-model-agent"}
        )
        args = parser.parse_args(
            [
                "--bench",
                "physion_pp",
                "--mode",
                "world-model-agent",
                "--route-option",
                self.policy.approved_optional_routes[0].option_id,
            ]
        )
        self.assertEqual(
            physmind.validate_route_options(args),
            (self.policy.approved_optional_routes[0].option_id,),
        )

    def test_direct_answer_submits_the_visual_input(self) -> None:
        sentinel = object()
        with patch.object(
            direct_answer, "answer_video_question", return_value=sentinel
        ) as mocked:
            result = direct_answer._answer_direct_question(
                config=object(),
                prompt="Will the objects collide?",
                video_path=Path("scene.mp4"),
                request_context={"scene_index": 0, "question_id": 0},
            )
        self.assertIs(result, sentinel)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.kwargs["video_path"], Path("scene.mp4"))

    def test_direct_entrypoint_does_not_import_the_world_model_stack(self) -> None:
        code = (
            "import sys; "
            "sys.modules['torch'] = None; "
            "sys.modules['pybullet'] = None; "
            "sys.modules['open3d'] = None; "
            "import physmind; "
            "assert 'agent.world_model.orchestrator' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_direct_mode_rejects_dry_run_before_any_request(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "physmind.py",
                "--bench",
                "clevrer",
                "--mode",
                "direct-answer",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "--dry-run is supported only with --mode world-model-agent",
            completed.stderr,
        )

    def test_worker_defaults_inherit_the_active_python_environment(self) -> None:
        commands = (
            DEFAULT_FOUNDATIONPOSE_WORKER_CMD,
            DEFAULT_GEOCALIB_WORKER_CMD,
            DEFAULT_MOGE2_WORKER_CMD,
            DEFAULT_SAM3_VIDEO_TRACKS_WORKER_CMD,
            DEFAULT_SAM3D_WORKER_CMD,
            DEFAULT_VIDEO_METRIC_DEPTH_WORKER_CMD,
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(command.startswith("python "))
                self.assertNotIn("conda run", command)


if __name__ == "__main__":
    unittest.main()
