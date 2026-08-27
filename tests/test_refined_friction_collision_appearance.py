from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.world_model.refined_friction_collision_appearance import (
    build_friction_collision_appearance_profile,
)


def _check_friction_collision_profile_pools_two_segment_objects(
    temporary_path: Path,
) -> None:
    frame_paths = [
        temporary_path / f"frame_{index:05d}.png" for index in range(6)
    ]
    frames = []
    for index in range(6):
        image = np.full((16, 16, 3), [104, 124, 138], dtype=np.uint8)
        image[:5] = [151, 148, 145]
        if index in {2, 3}:
            image[:] = [78, 68, 62]
        frames.append(image)

    colors = {
        "seg1_patient": [150, 82, 38],
        "seg2_patient": [142, 76, 34],
        "seg1_agent": [126, 35, 26],
        "seg2_agent": [119, 30, 24],
    }
    tracks = []
    masks = {}
    track_frames = {
        "seg1_patient": 0,
        "seg1_agent": 1,
        "seg2_patient": 4,
        "seg2_agent": 5,
    }
    for track_id, frame_index in track_frames.items():
        mask = np.zeros((16, 16), dtype=bool)
        mask[3:13, 3:13] = True
        frames[frame_index][mask] = colors[track_id]
        mask_key = f"{track_id}__frame_{frame_index:05d}"
        masks[mask_key] = mask
        tracks.append(
            {
                "object_id": track_id,
                "frame_index": frame_index,
                "mask_key": mask_key,
                "area": int(mask.sum()),
            }
        )

    sidecar = temporary_path / "masks.npz"
    np.savez_compressed(sidecar, **masks)
    object_plan = {
        "special_scene": {
            "scene_metadata": {"scenario": "friction_collision_pp"}
        },
        "target_objects": [
            {
                "object_id": "obj_1",
                "source_track_id": "seg1_patient",
                "mesh_reuse_source_object_id": None,
            },
            {
                "object_id": "obj_2",
                "source_track_id": "seg1_agent",
                "mesh_reuse_source_object_id": None,
            },
            {
                "object_id": "obj_3",
                "source_track_id": "seg2_patient",
                "mesh_reuse_source_object_id": "obj_1",
            },
            {
                "object_id": "obj_4",
                "source_track_id": "seg2_agent",
                "mesh_reuse_source_object_id": "obj_2",
            },
        ],
        "two_segment_mesh_reuse": {
            "reused": [
                {
                    "role": "patient",
                    "seg1_source_object": "obj_1",
                    "seg2_object": "obj_3",
                },
                {
                    "role": "agent",
                    "seg1_source_object": "obj_2",
                    "seg2_object": "obj_4",
                },
            ]
        },
    }
    sam3_tracks = {
        "mask_sidecar": str(sidecar),
        "tracks": tracks,
        "physion_tracking": {
            "two_segment": {
                "a": 2,
                "b": 3,
                "core_start": 2,
                "core_end": 3,
                "seg1": [0, 1],
                "seg2": [4, 5],
            }
        },
    }

    profile = build_friction_collision_appearance_profile(
        object_plan=object_plan,
        sam3_tracks=sam3_tracks,
        frame_paths=frame_paths,
        image_loader=lambda path: frames[int(path.stem.split("_")[-1])],
    )

    assert profile["status"] == "ok"
    assert profile["canonical_by_object_id"] == {
        "obj_1": "obj_1",
        "obj_2": "obj_2",
        "obj_3": "obj_1",
        "obj_4": "obj_2",
    }
    assert profile["object_profiles"]["obj_1"]["member_object_ids"] == [
        "obj_1",
        "obj_3",
    ]
    assert profile["object_profiles"]["obj_2"]["member_object_ids"] == [
        "obj_2",
        "obj_4",
    ]
    assert profile["object_profiles"]["obj_1"]["role"] == "patient"
    assert profile["object_profiles"]["obj_2"]["role"] == "agent"
    assert set(profile["environment"]) == {"floor", "back_wall", "curtain"}
    assert profile["two_segment"]["curtain"] == [2, 3]
    curtain_motion = profile["two_segment"]["curtain_motion"]
    assert curtain_motion["start_frame"] == 2
    assert curtain_motion["end_frame"] == 3


def _check_friction_collision_profile_rejects_other_scenarios() -> None:
    profile = build_friction_collision_appearance_profile(
        object_plan={
            "special_scene": {
                "scene_metadata": {"scenario": "friction_platform_pp"}
            }
        },
        sam3_tracks={},
        frame_paths=[],
        image_loader=lambda _path: np.empty((0, 0, 3)),
    )
    assert profile == {
        "status": "not_applicable",
        "scenario": "friction_platform_pp",
    }


class RefinedFrictionCollisionAppearanceTest(unittest.TestCase):
    def test_profile_pools_two_segment_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _check_friction_collision_profile_pools_two_segment_objects(
                Path(temporary_directory)
            )

    def test_profile_rejects_other_scenarios(self) -> None:
        _check_friction_collision_profile_rejects_other_scenarios()


if __name__ == "__main__":
    unittest.main()
