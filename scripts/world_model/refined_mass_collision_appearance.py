from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__:
    from . import refined_bouncy_wall_appearance as shared
else:
    import refined_bouncy_wall_appearance as shared


ImageLoader = Callable[[Path], np.ndarray]


def _normalized_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)[..., :3]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float64) / 255.0
    else:
        rgb = rgb.astype(np.float64)
        if float(np.nanmax(rgb)) > 1.5:
            rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def build_mass_collision_appearance_profile(
    *,
    object_plan: dict[str, Any],
    sam3_tracks: dict[str, Any],
    frame_paths: list[Path],
    image_loader: ImageLoader,
) -> dict[str, Any]:
    scene_metadata = (
        ((object_plan.get("special_scene") or {}).get("scene_metadata") or {})
    )
    scenario = str(scene_metadata.get("scenario") or "")
    if scenario != "mass_collision_pp":
        return {"status": "not_applicable", "scenario": scenario}
    if not frame_paths:
        raise ValueError("mass-collision appearance requires source frames")

    canonical_by_object_id = shared._canonical_object_ids(object_plan)
    targets = {
        str(item["object_id"]): item
        for item in object_plan.get("target_objects") or []
        if isinstance(item, dict) and item.get("object_id")
    }
    records_by_track: dict[str, list[dict[str, Any]]] = {}
    records_by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in sam3_tracks.get("tracks") or []:
        if not isinstance(record, dict) or not record.get("object_id"):
            continue
        track_id = str(record["object_id"])
        records_by_track.setdefault(track_id, []).append(record)
        if record.get("frame_index") is not None:
            records_by_frame.setdefault(int(record["frame_index"]), []).append(record)

    sidecar_path = Path(str(sam3_tracks.get("mask_sidecar") or ""))
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing SAM3 mask sidecar: {sidecar_path}")
    sidecar = np.load(sidecar_path, allow_pickle=False)
    image_cache: dict[int, np.ndarray] = {}

    def load_frame(frame_index: int) -> np.ndarray:
        if frame_index not in image_cache:
            if frame_index < 0 or frame_index >= len(frame_paths):
                raise IndexError(f"frame {frame_index} is outside source frame paths")
            image_cache[frame_index] = _normalized_image(
                image_loader(frame_paths[frame_index])
            )
        return image_cache[frame_index]

    track_samples: dict[str, np.ndarray] = {}
    for target in targets.values():
        track_id = str(target.get("source_track_id") or "")
        samples = []
        for record in shared._representative_records(
            records_by_track.get(track_id, []),
            count=8,
        ):
            mask_key = str(record["mask_key"])
            if mask_key not in sidecar:
                continue
            pixels = shared._trimmed_pixels(
                load_frame(int(record["frame_index"])),
                np.asarray(sidecar[mask_key], dtype=bool),
            )
            if len(pixels):
                samples.append(pixels)
        track_samples[track_id] = (
            np.concatenate(samples, axis=0)
            if samples
            else np.empty((0, 3), dtype=np.float64)
        )

    grouped_objects: dict[str, list[str]] = {}
    for object_id, canonical_id in canonical_by_object_id.items():
        grouped_objects.setdefault(canonical_id, []).append(object_id)

    object_profiles: dict[str, dict[str, Any]] = {}
    for canonical_id, object_ids in grouped_objects.items():
        samples = shared._balanced_pool(
            [
                track_samples.get(
                    str(targets[object_id].get("source_track_id") or ""),
                    np.empty((0, 3), dtype=np.float64),
                )
                for object_id in object_ids
            ]
        )
        if not len(samples):
            raise ValueError(f"no mass-collision appearance samples for {canonical_id}")
        target = targets[object_ids[0]]
        appearance = target.get("appearance") or {}
        color_name = str(appearance.get("color") or "")
        color_statistics = shared._color_statistics(samples)
        sampled_base = np.asarray(
            color_statistics["base_linear"],
            dtype=np.float64,
        )
        channel_scale = {
            "orange": np.asarray([1.00, 0.82, 0.82]),
            "green": np.asarray([0.65, 1.35, 0.90]),
            "purple": np.asarray([1.05, 0.75, 1.00]),
        }.get(color_name, np.ones(3, dtype=np.float64))
        color_statistics["sampled_base_linear"] = sampled_base.tolist()
        color_statistics["base_linear"] = np.clip(
            sampled_base * channel_scale,
            0.0,
            1.0,
        ).tolist()
        render_scale = {
            "orange": 0.78,
            "green": 0.88,
            "purple": 0.90,
        }.get(color_name, 0.82)
        object_profiles[canonical_id] = {
            "canonical_object_id": canonical_id,
            "member_object_ids": sorted(object_ids),
            "source_track_ids": sorted(
                {
                    str(targets[object_id].get("source_track_id") or "")
                    for object_id in object_ids
                }
            ),
            "role": "dynamic",
            "material_family": "rubber",
            "render_albedo_scale": render_scale,
            "texture_contrast": 0.34 if color_name == "orange" else 0.22,
            "source_color_name": color_name,
            **color_statistics,
        }

    two_segment = (
        ((sam3_tracks.get("physion_tracking") or {}).get("two_segment") or {})
    )
    seg1 = two_segment.get("seg1") or [0, max(len(frame_paths) - 1, 0)]
    seg2 = two_segment.get("seg2") or []
    representative_frames = shared._uniform_indices(int(seg1[0]), int(seg1[1]), 6)
    if len(seg2) == 2:
        representative_frames.extend(
            shared._uniform_indices(int(seg2[0]), int(seg2[1]), 3)
        )
    representative_frames = sorted(set(representative_frames))

    region_samples: dict[str, list[np.ndarray]] = {
        "floor": [],
        "back_wall_left": [],
        "back_wall_right": [],
        "window": [],
    }
    for frame_index in representative_frames:
        image = load_frame(frame_index)
        height, width = image.shape[:2]
        union = np.zeros((height, width), dtype=bool)
        for record in records_by_frame.get(frame_index, []):
            mask_key = str(record.get("mask_key") or "")
            if mask_key in sidecar:
                union |= np.asarray(sidecar[mask_key], dtype=bool)
        bounds = {
            "back_wall_left": (0, int(0.40 * height), 0, int(0.50 * width)),
            "back_wall_right": (
                0,
                int(0.40 * height),
                int(0.50 * width),
                width,
            ),
            "floor": (int(0.48 * height), height, 0, width),
            "window": (
                0,
                int(0.10 * height),
                int(0.14 * width),
                int(0.54 * width),
            ),
        }
        for name, (y0, y1, x0, x1) in bounds.items():
            pixels = image[y0:y1, x0:x1][~union[y0:y1, x0:x1]]
            if len(pixels) > 4096:
                indices = np.linspace(0, len(pixels) - 1, 4096).astype(int)
                pixels = pixels[indices]
            if len(pixels):
                region_samples[name].append(pixels)

    environment = {
        name: shared._region_statistics(samples)
        for name, samples in region_samples.items()
    }
    environment["floor"].update(
        {
            "base_linear": (
                np.asarray(environment["floor"]["base_linear"])
                * np.asarray([0.95, 1.10, 1.20])
            ).clip(0.0, 1.0).tolist(),
            "render_albedo_scale": 0.50,
            "texture_scale": 720.0,
            "texture_dark_scale": 0.48,
            "texture_light_scale": 1.08,
        }
    )
    environment["back_wall_left"]["camera_albedo_scale"] = 1.22
    environment["back_wall_right"]["camera_albedo_scale"] = 0.96

    core_start = int(two_segment.get("core_start", two_segment.get("a", -1)))
    core_end = int(two_segment.get("core_end", two_segment.get("b", -1)))
    curtain_start = int(two_segment.get("a", core_start))
    curtain_end = int(two_segment.get("b", core_end))
    curtain_samples = []
    for frame_index in shared._uniform_indices(
        int(round(core_start + 0.25 * (core_end - core_start))),
        int(round(core_end - 0.25 * (core_end - core_start))),
        3,
    ):
        image = load_frame(frame_index)
        height, width = image.shape[:2]
        curtain_samples.append(
            image[
                int(0.08 * height) : int(0.92 * height),
                int(0.08 * width) : int(0.92 * width),
                :3,
            ].reshape(-1, 3)
        )
    environment["curtain"] = {
        **shared._region_statistics(curtain_samples),
        "render_dark_scale": 0.78,
        "render_light_scale": 1.10,
    }

    return {
        "status": "ok",
        "scenario": scenario,
        "method": "shared refined SAM3-mask sampling with mass-collision room regions",
        "canonical_by_object_id": canonical_by_object_id,
        "object_profiles": object_profiles,
        "environment": environment,
        "two_segment": {
            "seg1": [int(seg1[0]), int(seg1[1])],
            "seg2": [int(seg2[0]), int(seg2[1])] if len(seg2) == 2 else [],
            "curtain": [curtain_start, curtain_end],
            "curtain_core": [core_start, core_end],
            "curtain_motion": {
                "start_frame": core_start,
                "end_frame": min(curtain_end, core_end + 3),
                "width_normalized": 2.5,
                "top_normalized": 0.155,
                "bottom_normalized": 1.03,
            },
        },
        "sampling": {
            "representative_frames": representative_frames,
            "mask_sidecar": str(sidecar_path),
            "source_frame_count": len(frame_paths),
        },
    }
