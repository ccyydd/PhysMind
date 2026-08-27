from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.world_model.artifacts import ArtifactManager
from agent.world_model.debug_artifacts import (
    LATEST_WORLD_RECONSTRUCTION_RENDERER,
    world_reconstruction_debug_render_command,
)
from agent.world_model.tools import (
    CLEVRER_LATEST_DEBUG_RENDER_PROFILE,
    SimulatableWorldReconstructionAdapter,
)


class LatestDebugRendererTest(unittest.TestCase):
    def test_production_command_uses_the_latest_renderer(self) -> None:
        command = world_reconstruction_debug_render_command(
            command="third_party/blender/blender",
            render_input_path=Path("input.json"),
            output_video=Path("output.mp4"),
            output_json=Path("output.json"),
        )
        self.assertIn("--python-exit-code", command)
        self.assertIn("1", command)
        self.assertIn(str(LATEST_WORLD_RECONSTRUCTION_RENDERER), command)

    def test_swr_render_input_preserves_vrdp_support_plane(self) -> None:
        adapter = object.__new__(SimulatableWorldReconstructionAdapter)
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        support_plane = {
            "coordinate_frame": "blender_world",
            "origin": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        }
        payload = adapter._physics_alignment_blender_render_payload(
            question_dir=Path("question_0"),
            result={
                "backend": "swr_backend.impulse_analytic",
                "vrdp_support_plane": support_plane,
                "physics_rollout": {
                    "simulated_trajectories": {
                        "obj_1": [{"frame_index": 0, "pose_4x4": identity}]
                    }
                },
            },
            physics_alignment_manifest={
                "target_trajectories": {
                    "objects": [{"object_id": "obj_1", "mesh_path": "obj_1.glb"}]
                }
            },
        )
        self.assertEqual(payload["vrdp_support_plane"], support_plane)

    def test_swr_records_the_expected_camera_render_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = ArtifactManager(root, debug_artifacts=True)
            adapter = object.__new__(SimulatableWorldReconstructionAdapter)
            adapter.artifacts = artifacts
            question_dir = root / "artifacts" / "scene_1" / "question_0"
            result_path = (
                question_dir
                / "simulatable-world-reconstruction"
                / "fit"
                / "world_reconstruction_fit.json"
            )
            result = {
                "backend": "swr_backend.impulse_analytic",
                "physics_rollout": {},
            }

            def fake_render(**kwargs: object) -> dict[str, object]:
                output_video = Path(str(kwargs["output_video"]))
                output_json = Path(str(kwargs["output_json"]))
                source_video = output_video.with_name(
                    "world_reconstruction_debug_camera.mp4"
                )
                source_video.parent.mkdir(parents=True, exist_ok=True)
                source_video.write_bytes(b"video")
                output_json.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "render_profile": CLEVRER_LATEST_DEBUG_RENDER_PROFILE,
                            "source_camera_video_path": str(source_video),
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "status": "ok",
                    "returncode": 0,
                    "elapsed_sec": 0.0,
                    "stdout": "",
                    "stderr": "",
                }

            with patch.object(
                adapter,
                "_physics_alignment_blender_render_payload",
                return_value={},
            ), patch(
                "agent.world_model.tools.run_world_reconstruction_debug_render",
                side_effect=fake_render,
            ):
                adapter._render_physics_alignment_blender_debug(
                    question_dir=question_dir,
                    result_path=result_path,
                    result=result,
                    physics_alignment_manifest={},
                    command="third_party/blender/blender",
                )

            recorded = result["physics_rollout"]["blender_debug_render"]
            self.assertEqual(recorded["status"], "ok")
            self.assertEqual(
                recorded["render_profile"], CLEVRER_LATEST_DEBUG_RENDER_PROFILE
            )
            self.assertNotIn("topdown_video_path", recorded)


if __name__ == "__main__":
    unittest.main()
