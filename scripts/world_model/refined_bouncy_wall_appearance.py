from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


ImageLoader = Callable[[Path], np.ndarray]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _srgb_to_linear(color: np.ndarray) -> np.ndarray:
    color = np.clip(np.asarray(color, dtype=np.float64), 0.0, 1.0)
    return np.where(
        color <= 0.04045,
        color / 12.92,
        ((color + 0.055) / 1.055) ** 2.4,
    )


def _canonical_object_ids(object_plan: dict[str, Any]) -> dict[str, str]:
    direct: dict[str, str] = {}
    for item in object_plan.get("target_objects") or []:
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        object_id = str(item["object_id"])
        source = item.get("mesh_reuse_source_object_id")
        direct[object_id] = str(source) if source else object_id

    def resolve(object_id: str) -> str:
        visited: set[str] = set()
        current = object_id
        while direct.get(current, current) != current:
            if current in visited:
                raise ValueError(f"cyclic mesh reuse chain involving {object_id}")
            visited.add(current)
            current = direct[current]
        return current

    return {object_id: resolve(object_id) for object_id in direct}


def _role_by_canonical_id(
    object_plan: dict[str, Any],
    canonical_by_object_id: dict[str, str],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    reuse = object_plan.get("two_segment_mesh_reuse")
    records = reuse.get("reused") if isinstance(reuse, dict) else []
    for record in records or []:
        if not isinstance(record, dict) or not record.get("seg1_source_object"):
            continue
        canonical_id = canonical_by_object_id.get(
            str(record["seg1_source_object"]),
            str(record["seg1_source_object"]),
        )
        if record.get("role"):
            roles[canonical_id] = str(record["role"])
    return roles


def _erode_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    eroded = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        current = eroded
        eroded = np.zeros_like(current)
        eroded[1:-1, 1:-1] = (
            current[1:-1, 1:-1]
            & current[:-2, 1:-1]
            & current[2:, 1:-1]
            & current[1:-1, :-2]
            & current[1:-1, 2:]
        )
    return eroded


def _representative_records(
    records: list[dict[str, Any]],
    count: int = 8,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (
            record
            for record in records
            if isinstance(record, dict)
            and record.get("frame_index") is not None
            and record.get("mask_key")
        ),
        key=lambda record: int(record["frame_index"]),
    )
    if len(ordered) <= count:
        return ordered
    bins = np.array_split(np.arange(len(ordered)), count)
    selected = []
    for indices in bins:
        candidates = [ordered[int(index)] for index in indices]
        selected.append(
            max(candidates, key=lambda record: int(record.get("area") or 0))
        )
    return selected


def _trimmed_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_pixels: int = 4096,
) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float64) / 255.0
    elif float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    interior = _erode_mask(mask, iterations=2)
    pixels = rgb[interior]
    if len(pixels) < 32:
        pixels = rgb[np.asarray(mask, dtype=bool)]
    if len(pixels) < 16:
        return np.empty((0, 3), dtype=np.float64)
    luminance = pixels @ np.asarray([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.25, 0.94])
    pixels = pixels[(luminance >= low) & (luminance <= high)]
    if len(pixels) > max_pixels:
        indices = np.linspace(0, len(pixels) - 1, max_pixels).astype(int)
        pixels = pixels[indices]
    return pixels


def _balanced_pool(track_samples: list[np.ndarray]) -> np.ndarray:
    valid = [samples for samples in track_samples if len(samples)]
    if not valid:
        return np.empty((0, 3), dtype=np.float64)
    sample_count = min(8192, max(len(samples) for samples in valid))
    balanced = []
    for samples in valid:
        indices = np.linspace(0, len(samples) - 1, sample_count).astype(int)
        balanced.append(samples[indices])
    return np.concatenate(balanced, axis=0)


def _color_statistics(samples: np.ndarray) -> dict[str, Any]:
    if len(samples) < 16:
        raise ValueError("not enough appearance pixels")
    luminance = samples @ np.asarray([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.18, 0.92])
    central = samples[(luminance >= low) & (luminance <= high)]
    if not len(central):
        central = samples
    base_srgb = np.median(central, axis=0)
    shadow = np.median(samples[luminance <= np.quantile(luminance, 0.30)], axis=0)
    highlight = np.median(
        samples[luminance >= np.quantile(luminance, 0.72)],
        axis=0,
    )
    q10, q90 = np.quantile(luminance, [0.10, 0.90])
    contrast = float(
        np.clip(
            (q90 - q10) / max(float(np.median(luminance)), 1e-4),
            0.04,
            0.55,
        )
    )
    return {
        "base_srgb": base_srgb.astype(float).tolist(),
        "base_linear": _srgb_to_linear(base_srgb).astype(float).tolist(),
        "shadow_srgb": shadow.astype(float).tolist(),
        "highlight_srgb": highlight.astype(float).tolist(),
        "texture_contrast": contrast,
        "sample_count": int(len(samples)),
    }


def _region_statistics(samples: list[np.ndarray]) -> dict[str, Any]:
    valid = [sample for sample in samples if len(sample)]
    if not valid:
        raise ValueError("no background-region samples")
    pool = np.concatenate(valid, axis=0)
    if len(pool) > 32768:
        indices = np.linspace(0, len(pool) - 1, 32768).astype(int)
        pool = pool[indices]
    luminance = pool @ np.asarray([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.08, 0.92])
    central = pool[(luminance >= low) & (luminance <= high)]
    base_srgb = np.median(central if len(central) else pool, axis=0)
    q10, q90 = np.quantile(luminance, [0.10, 0.90])
    return {
        "base_srgb": base_srgb.astype(float).tolist(),
        "base_linear": _srgb_to_linear(base_srgb).astype(float).tolist(),
        "texture_contrast": float(
            np.clip(
                (q90 - q10) / max(float(np.median(luminance)), 1e-4),
                0.03,
                0.45,
            )
        ),
        "sample_count": int(len(pool)),
    }


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if end < start:
        return []
    if start == end or count <= 1:
        return [start]
    return sorted(
        {
            int(round(start + index * (end - start) / (count - 1)))
            for index in range(count)
        }
    )


def build_bouncy_wall_appearance_profile(
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
    if scenario != "bouncy_wall_pp":
        return {
            "status": "not_applicable",
            "scenario": scenario,
        }

    canonical_by_object_id = _canonical_object_ids(object_plan)
    role_by_canonical_id = _role_by_canonical_id(
        object_plan,
        canonical_by_object_id,
    )
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
            if frame_index < 0 or frame_index >= len(frame_paths):
                raise IndexError(f"frame {frame_index} is outside source frame paths")
            image = np.asarray(image_loader(frame_paths[frame_index]))
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"invalid source image shape: {image.shape}")
            image_cache[frame_index] = image[..., :3]
        return image_cache[frame_index]

    track_samples: dict[str, np.ndarray] = {}
    for object_id, target in targets.items():
        track_id = str(target.get("source_track_id") or "")
        samples = []
        for record in _representative_records(records_by_track.get(track_id, [])):
            mask_key = str(record["mask_key"])
            if mask_key not in sidecar:
                continue
            frame_index = int(record["frame_index"])
            pixels = _trimmed_pixels(
                load_frame(frame_index),
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
        samples = _balanced_pool(
            [
                track_samples.get(
                    str(targets[object_id].get("source_track_id") or ""),
                    np.empty((0, 3), dtype=np.float64),
                )
                for object_id in object_ids
            ]
        )
        statistics = _color_statistics(samples)
        descriptions = [
            str(targets[object_id].get("description") or "")
            for object_id in object_ids
        ]
        appearance_reasoning = [
            str(
                ((targets[object_id].get("appearance") or {}).get("reasoning"))
                or ""
            )
            for object_id in object_ids
        ]
        role = role_by_canonical_id.get(canonical_id, "")
        material_text = " ".join([*descriptions, *appearance_reasoning]).lower()
        material_family = (
            "wood"
            if role in {"wall", "patient"} or "wood" in material_text
            else "rubber"
        )
        object_profiles[canonical_id] = {
            "canonical_object_id": canonical_id,
            "member_object_ids": sorted(object_ids),
            "source_track_ids": sorted(
                {
                    str(targets[object_id].get("source_track_id") or "")
                    for object_id in object_ids
                }
            ),
            "role": role,
            "material_family": material_family,
            **statistics,
        }

    two_segment = (
        ((sam3_tracks.get("physion_tracking") or {}).get("two_segment") or {})
    )
    seg1 = two_segment.get("seg1") or [0, max(len(frame_paths) - 1, 0)]
    seg2 = two_segment.get("seg2") or []
    representative_frames = _uniform_indices(int(seg1[0]), int(seg1[1]), 6)
    if len(seg2) == 2:
        representative_frames.extend(
            _uniform_indices(int(seg2[0]), int(seg2[1]), 3)
        )
    representative_frames = sorted(set(representative_frames))

    region_samples: dict[str, list[np.ndarray]] = {
        "floor": [],
        "back_wall": [],
        "ceiling": [],
    }
    for frame_index in representative_frames:
        image = load_frame(frame_index)
        height, width = image.shape[:2]
        union_mask = np.zeros((height, width), dtype=bool)
        for record in records_by_frame.get(frame_index, []):
            mask_key = str(record.get("mask_key") or "")
            if mask_key in sidecar:
                union_mask |= np.asarray(sidecar[mask_key], dtype=bool)
        regions = {
            "floor": (int(0.55 * height), height, 0, width),
            "back_wall": (
                int(0.20 * height),
                int(0.58 * height),
                0,
                width,
            ),
            "ceiling": (0, int(0.20 * height), 0, width),
        }
        rgb = np.asarray(image, dtype=np.float64)
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float64) / 255.0
        elif float(np.nanmax(rgb)) > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb[..., :3], 0.0, 1.0)
        for name, (y0, y1, x0, x1) in regions.items():
            valid = ~union_mask[y0:y1, x0:x1]
            pixels = rgb[y0:y1, x0:x1][valid]
            if len(pixels) > 4096:
                indices = np.linspace(0, len(pixels) - 1, 4096).astype(int)
                pixels = pixels[indices]
            if len(pixels):
                region_samples[name].append(pixels)

    environment = {
        name: _region_statistics(samples)
        for name, samples in region_samples.items()
    }
    core_start = int(two_segment.get("core_start", two_segment.get("a", -1)))
    core_end = int(two_segment.get("core_end", two_segment.get("b", -1)))
    curtain_start = int(two_segment.get("a", core_start))
    curtain_end = int(two_segment.get("b", core_end))
    curtain_frames = _uniform_indices(core_start, core_end, 3)
    curtain_samples = []
    for frame_index in curtain_frames:
        image = np.asarray(load_frame(frame_index), dtype=np.float64)
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255.0
        elif float(np.nanmax(image)) > 1.5:
            image = image / 255.0
        height, width = image.shape[:2]
        curtain_samples.append(
            image[
                int(0.08 * height) : int(0.92 * height),
                int(0.08 * width) : int(0.92 * width),
                :3,
            ].reshape(-1, 3)
        )
    environment["curtain"] = _region_statistics(curtain_samples)

    return {
        "status": "ok",
        "scenario": scenario,
        "method": (
            "SAM3-mask interior robust RGB sampling pooled by "
            "two_segment_mesh_reuse canonical object"
        ),
        "canonical_by_object_id": canonical_by_object_id,
        "object_profiles": object_profiles,
        "environment": environment,
        "two_segment": {
            "seg1": [int(seg1[0]), int(seg1[1])],
            "seg2": [int(seg2[0]), int(seg2[1])] if len(seg2) == 2 else [],
            "curtain": [curtain_start, curtain_end],
            "curtain_core": [core_start, core_end],
        },
        "sampling": {
            "representative_frames": representative_frames,
            "mask_sidecar": str(sidecar_path),
            "source_frame_count": len(frame_paths),
        },
    }
