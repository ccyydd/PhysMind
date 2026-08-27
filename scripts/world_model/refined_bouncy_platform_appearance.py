from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


ImageLoader = Callable[[Path], np.ndarray]


def _srgb_to_linear(color: np.ndarray) -> np.ndarray:
    color = np.clip(np.asarray(color, dtype=np.float64), 0.0, 1.0)
    return np.where(
        color <= 0.04045,
        color / 12.92,
        ((color + 0.055) / 1.055) ** 2.4,
    )


def _normalized_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)[..., :3]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float64) / 255.0
    else:
        rgb = rgb.astype(np.float64)
        if float(np.nanmax(rgb)) > 1.5:
            rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def _erode(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        current = result
        result = np.zeros_like(current)
        result[1:-1, 1:-1] = (
            current[1:-1, 1:-1]
            & current[:-2, 1:-1]
            & current[2:, 1:-1]
            & current[1:-1, :-2]
            & current[1:-1, 2:]
        )
    return result


def _uniform_records(
    records: list[dict[str, Any]],
    count: int = 8,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (
            item
            for item in records
            if isinstance(item, dict)
            and item.get("frame_index") is not None
            and item.get("mask_key")
        ),
        key=lambda item: int(item["frame_index"]),
    )
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[int(index)] for index in indices]


def _robust_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_pixels: int = 4096,
) -> np.ndarray:
    rgb = _normalized_image(image)
    interior = _erode(mask)
    pixels = rgb[interior]
    if len(pixels) < 24:
        pixels = rgb[np.asarray(mask, dtype=bool)]
    if len(pixels) < 16:
        return np.empty((0, 3), dtype=np.float64)
    luminance = pixels @ np.asarray([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.12, 0.96])
    pixels = pixels[(luminance >= low) & (luminance <= high)]
    if len(pixels) > max_pixels:
        indices = np.linspace(0, len(pixels) - 1, max_pixels).astype(int)
        pixels = pixels[indices]
    return pixels


def _statistics(samples: np.ndarray) -> dict[str, Any]:
    if len(samples) < 16:
        raise ValueError("not enough appearance pixels")
    luminance = samples @ np.asarray([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.16, 0.92])
    central = samples[(luminance >= low) & (luminance <= high)]
    base_srgb = np.median(central if len(central) else samples, axis=0)
    q10, q90 = np.quantile(luminance, [0.10, 0.90])
    return {
        "base_srgb": base_srgb.astype(float).tolist(),
        "base_linear": _srgb_to_linear(base_srgb).astype(float).tolist(),
        "texture_contrast": float(
            np.clip(
                (q90 - q10) / max(float(np.median(luminance)), 1e-4),
                0.04,
                0.50,
            )
        ),
        "sample_count": int(len(samples)),
    }


def _mixed_blue_wood_palette(samples: np.ndarray) -> dict[str, Any] | None:
    if len(samples) < 64:
        return None
    red, green, blue = samples.T
    blue_mask = (blue > 1.10 * red) & (blue > 1.04 * green)
    wood_mask = (red > 1.10 * blue) & (green > 0.58 * red)
    if int(np.count_nonzero(blue_mask)) < 24 or int(np.count_nonzero(wood_mask)) < 24:
        return None
    blue_srgb = np.median(samples[blue_mask], axis=0)
    wood_srgb = np.median(samples[wood_mask], axis=0)
    return {
        "blue_srgb": blue_srgb.astype(float).tolist(),
        "blue_linear": _srgb_to_linear(blue_srgb).astype(float).tolist(),
        "wood_srgb": wood_srgb.astype(float).tolist(),
        "wood_linear": _srgb_to_linear(wood_srgb).astype(float).tolist(),
        "blue_sample_count": int(np.count_nonzero(blue_mask)),
        "wood_sample_count": int(np.count_nonzero(wood_mask)),
    }


def _region_statistics(samples: list[np.ndarray]) -> dict[str, Any]:
    valid = [sample for sample in samples if len(sample)]
    if not valid:
        raise ValueError("no background samples")
    pool = np.concatenate(valid, axis=0)
    if len(pool) > 32768:
        indices = np.linspace(0, len(pool) - 1, 32768).astype(int)
        pool = pool[indices]
    return _statistics(pool)


def build_bouncy_platform_appearance_profile(
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
    if scenario != "bouncy_platform_pp":
        return {"status": "not_applicable", "scenario": scenario}
    if not frame_paths:
        raise ValueError("bouncy-platform appearance requires source frames")

    records_by_track: dict[str, list[dict[str, Any]]] = {}
    records_by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in sam3_tracks.get("tracks") or []:
        if not isinstance(record, dict) or not record.get("object_id"):
            continue
        records_by_track.setdefault(str(record["object_id"]), []).append(record)
        if record.get("frame_index") is not None:
            records_by_frame.setdefault(int(record["frame_index"]), []).append(record)

    sidecar_path = Path(str(sam3_tracks.get("mask_sidecar") or ""))
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing SAM3 mask sidecar: {sidecar_path}")
    sidecar = np.load(sidecar_path, allow_pickle=False)
    image_cache: dict[int, np.ndarray] = {}

    def load_frame(frame_index: int) -> np.ndarray:
        if frame_index not in image_cache:
            image_cache[frame_index] = _normalized_image(
                image_loader(frame_paths[frame_index])
            )
        return image_cache[frame_index]

    object_profiles: dict[str, dict[str, Any]] = {}
    targets = [
        item
        for item in object_plan.get("target_objects") or []
        if isinstance(item, dict) and item.get("object_id")
    ]
    for target in targets:
        object_id = str(target["object_id"])
        track_id = str(target.get("source_track_id") or "")
        samples = []
        for record in _uniform_records(records_by_track.get(track_id, [])):
            mask_key = str(record["mask_key"])
            if mask_key not in sidecar:
                continue
            pixels = _robust_pixels(
                load_frame(int(record["frame_index"])),
                np.asarray(sidecar[mask_key], dtype=bool),
            )
            if len(pixels):
                samples.append(pixels)
        if not samples:
            raise ValueError(f"no appearance samples for {object_id}/{track_id}")
        pool = np.concatenate(samples, axis=0)
        description = " ".join(
            [
                str(target.get("description") or ""),
                str(((target.get("appearance") or {}).get("reasoning")) or ""),
            ]
        ).lower()
        profile = {
            "object_id": object_id,
            "source_track_id": track_id,
            "role": str(target.get("role") or ""),
            "material_family": (
                "wood"
                if object_id == "obj_2" or "wood" in description
                else "rubber"
            ),
            **_statistics(pool),
        }
        albedo_scale = {
            "obj_1": 2.40,
            "obj_2": 0.72,
            "obj_3": 1.85,
            "obj_4": 1.55,
        }.get(object_id, 1.0)
        profile["base_linear"] = np.clip(
            np.asarray(profile["base_linear"], dtype=np.float64) * albedo_scale,
            0.0,
            1.0,
        ).astype(float).tolist()
        profile["albedo_scale"] = albedo_scale
        if "blue" in description and "brown" in description:
            palette = _mixed_blue_wood_palette(pool)
            if isinstance(palette, dict):
                palette["blue_linear"] = np.clip(
                    np.asarray(palette["blue_linear"], dtype=np.float64) * 1.35,
                    0.0,
                    1.0,
                ).astype(float).tolist()
                palette["wood_linear"] = np.clip(
                    np.asarray(palette["wood_linear"], dtype=np.float64) * 2.30,
                    0.0,
                    1.0,
                ).astype(float).tolist()
            profile["mixed_palette"] = palette
        object_profiles[object_id] = profile

    representative_frames = sorted(
        {
            int(round(value))
            for value in np.linspace(0, len(frame_paths) - 1, 8)
        }
    )
    regions: dict[str, list[np.ndarray]] = {
        "floor": [],
        "back_wall": [],
        "back_wall_left": [],
        "back_wall_right": [],
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
            "back_wall": (0, int(0.53 * height), 0, width),
            "back_wall_left": (
                0,
                int(0.53 * height),
                0,
                int(0.62 * width),
            ),
            "back_wall_right": (
                0,
                int(0.53 * height),
                int(0.65 * width),
                width,
            ),
            "floor": (int(0.55 * height), height, 0, width),
        }
        for name, (y0, y1, x0, x1) in bounds.items():
            pixels = image[y0:y1, x0:x1][~union[y0:y1, x0:x1]]
            if len(pixels) > 4096:
                indices = np.linspace(0, len(pixels) - 1, 4096).astype(int)
                pixels = pixels[indices]
            if len(pixels):
                regions[name].append(pixels)

    return {
        "status": "ok",
        "scenario": scenario,
        "method": "SAM3-mask robust object sampling plus masked room-region sampling",
        "canonical_by_object_id": {
            str(target["object_id"]): str(target["object_id"])
            for target in targets
        },
        "object_profiles": object_profiles,
        "environment": {
            name: _region_statistics(samples)
            for name, samples in regions.items()
        },
        "sampling": {
            "representative_frames": representative_frames,
            "mask_sidecar": str(sidecar_path),
            "source_frame_count": len(frame_paths),
        },
    }
