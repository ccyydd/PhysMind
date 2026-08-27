from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agent.world_model import cross_segment_identity as identity
from agent.world_model.orchestrator import WorldModelAgent
from agent.world_model.schemas import ToolResult
from agent.world_model.tools import SAM3VideoTrackLabelsAdapter
from benchmark.physion_pp import PhysionPPQuestion, PhysionPPScene
from utils.config import build_model_config


class CrossSegmentIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_model_config(require_api_key=False)
        self.image = np.zeros((40, 50, 3), dtype=np.uint8)
        self.image[8:30, 12:36] = (20, 80, 140)
        self.mask = np.zeros((40, 50), dtype=bool)
        self.mask[8:30, 12:36] = True

    def test_crop_object_rejects_empty_or_mismatched_masks(self) -> None:
        with self.assertRaises(identity.CrossSegmentIdentityError):
            identity.crop_object(self.image, np.zeros_like(self.mask))
        with self.assertRaises(identity.CrossSegmentIdentityError):
            identity.crop_object(self.image, np.zeros((10, 10), dtype=bool))

    def test_ab_selector_uses_labeled_crops_and_returns_selected_track(self) -> None:
        response = SimpleNamespace(text="The answer is B.", usage={"total_tokens": 12})
        with patch.object(identity, "answer_with_image_files", return_value=response) as answer:
            result = identity.select_same_object_ab(
                config=self.config,
                reference_image=self.image,
                reference_mask=self.mask,
                candidate_images=[self.image, self.image],
                candidate_masks=[self.mask, self.mask],
                candidate_ids=["track_a", "track_b"],
                request_context={"stage": "test"},
            )
        self.assertEqual(result["selected_label"], "B")
        self.assertEqual(result["selected_track"], "track_b")
        self.assertEqual(result["candidate_tracks"], {"A": "track_a", "B": "track_b"})
        self.assertEqual(answer.call_args.kwargs["image_labels"], ["REFERENCE", "A", "B"])
        self.assertEqual(len(answer.call_args.kwargs["image_paths"]), 3)

    def test_yes_no_selector_returns_valid_no_without_fallback(self) -> None:
        response = SimpleNamespace(text="NO", usage=None)
        with patch.object(identity, "answer_with_image_files", return_value=response):
            result = identity.decide_same_object_yes_no(
                config=self.config,
                reference_image=self.image,
                reference_mask=self.mask,
                candidate_image=self.image,
                candidate_mask=self.mask,
                reference_id="patient",
                candidate_id="extra",
            )
        self.assertFalse(result["linked"])
        self.assertEqual(result["reference_track"], "patient")
        self.assertEqual(result["candidate_track"], "extra")

    def test_ambiguous_responses_and_request_errors_raise(self) -> None:
        with self.assertRaises(identity.CrossSegmentIdentityError):
            identity.parse_ab_response("A or B")
        with self.assertRaises(identity.CrossSegmentIdentityError):
            identity.parse_yes_no_response("maybe")
        with patch.object(identity, "answer_with_image_files", side_effect=TimeoutError("late")):
            with self.assertRaisesRegex(
                identity.CrossSegmentIdentityError,
                "VLM request failed",
            ):
                identity.select_same_object_ab(
                    config=self.config,
                    reference_image=self.image,
                    reference_mask=self.mask,
                    candidate_images=[self.image, self.image],
                    candidate_masks=[self.mask, self.mask],
                    candidate_ids=["track_a", "track_b"],
                )

    def test_mass_link_candidate_absence_is_valid_false_but_bad_input_raises(self) -> None:
        route = {
            "decision_id": "TRK-005.mass_extra_patient_link",
            "route": "role_link.vlm_identity",
        }
        adapter = SimpleNamespace()
        no_candidate = SAM3VideoTrackLabelsAdapter._mc_extra_patient_link(
            adapter,
            {"physion_tracking": {"role_binding": {"kept_extra_tracks": []}}},
            resolved_route=route,
        )
        self.assertFalse(no_candidate["linked"])
        self.assertEqual(no_candidate["method"], "no_candidate")
        self.assertNotIn("threshold", no_candidate)
        self.assertNotIn("color_dist", no_candidate)

        with self.assertRaisesRegex(ValueError, "missing the seg2 patient track"):
            SAM3VideoTrackLabelsAdapter._mc_extra_patient_link(
                adapter,
                {
                    "physion_tracking": {
                        "role_binding": {
                            "kept_extra_tracks": [{"track": "extra"}],
                        }
                    }
                },
                resolved_route=route,
            )

    def test_identity_exception_enters_existing_direct_answer_fallback(self) -> None:
        question = PhysionPPQuestion(
            question_id=0,
            question="Will the objects make contact?",
            question_type="ocp",
            answer="yes",
            ground_truth_outcome=True,
        )
        scene = PhysionPPScene(
            scene_index=18,
            property_name="mass",
            scenario="mass_collision_pp",
            copy_name="copy0",
            config_name="config",
            trial_stem="trial",
            stimulus_id="stimulus",
            pair_id="pair",
            video_filename="scene.mp4",
            video_path=Path("/nonexistent/scene.mp4"),
            map_path=Path("/nonexistent/map.json"),
            id_json_path=Path("/nonexistent/id.json"),
            pkl_path=Path("/nonexistent/scene.pkl"),
            start_frame_for_prediction=10,
            questions=[question],
        )
        fallback_result = ToolResult(
            tool_name="final_answer",
            status="ok",
            artifact_path="/tmp/final_answer.json",
            payload={"status": "ok", "extracted_answer": "yes"},
        )
        with TemporaryDirectory() as tmpdir:
            agent = WorldModelAgent(
                config=self.config,
                run_dir=Path(tmpdir),
                dry_run=True,
                stop_after_stage="answering",
                persistent_workers=False,
                answer_enabled=True,
            )
            with (
                patch.object(
                    agent,
                    "_get_scene_object_plan",
                    side_effect=identity.CrossSegmentIdentityError(
                        "cross-segment A/B VLM request failed"
                    ),
                ),
                patch(
                    "agent.world_model.orchestrator.run_direct_answer_fallback",
                    return_value=fallback_result,
                ) as fallback,
            ):
                result = agent.process_question(scene, question)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.final_answer, "yes")
        self.assertIsNone(result.error_message)
        pipeline_failure = fallback.call_args.kwargs["pipeline_failure"]
        self.assertEqual(pipeline_failure["failed_stage"], "object-planning")
        self.assertIn("A/B VLM request failed", pipeline_failure["error_message"])


if __name__ == "__main__":
    unittest.main()
