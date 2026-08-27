from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_world_reconstruction_debug as base
import refined_bouncy_platform_appearance as bouncy_platform_appearance
import refined_bouncy_wall_appearance as bouncy_wall_appearance
import refined_friction_collision_appearance as friction_collision_appearance
import refined_mass_collision_appearance as mass_collision_appearance


OBJECT_COLORS = {
    "red": (0.58, 0.012, 0.008, 1.0),
    "yellow": (0.43, 0.34, 0.045, 1.0),
    "blue": (0.003, 0.004, 0.073, 1.0),
    "green": (0.018, 0.32, 0.055, 1.0),
    "purple": (0.28, 0.025, 0.34, 1.0),
    "cyan": (0.015, 0.42, 0.48, 1.0),
    "gray": (0.24, 0.25, 0.27, 1.0),
    "orange": (0.50, 0.075, 0.035, 1.0),
    "brown": (0.095, 0.035, 0.018, 1.0),
}
RENDER_PROFILE = "physionpp_refined_6f3213_camera_only"


def _merge_reference_geometry(
    world_reconstruction: dict[str, Any],
    support_reference: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(world_reconstruction)
    for key in (
        "video_metadata",
        "gravity_direction_camera",
        "analytic_support_plane",
        "support_plane_position_correction",
    ):
        current = merged.get(key)
        if current in (None, {}, []):
            merged[key] = support_reference.get(key)
    return merged


def _appearance_by_object_id(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, dict[str, str]]:
    path = base._object_plan_path(world_reconstruction_path, world_reconstruction)
    if not path.exists():
        return {}
    payload = base._load_json(path)
    result: dict[str, dict[str, str]] = {}
    for item in payload.get("target_objects", []):
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        appearance = item.get("appearance")
        if not isinstance(appearance, dict):
            appearance = {}
        result[str(item["object_id"])] = {
            "color": str(appearance.get("color") or "").strip().lower(),
            "material": str(appearance.get("material") or "rubber").strip().lower(),
        }
    return result


def _load_source_image_srgb(path: Path) -> np.ndarray:
    import bpy

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        width, height = (int(value) for value in image.size)
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgba = pixels.reshape(height, width, 4)
        return np.ascontiguousarray(rgba[::-1, :, :3])
    finally:
        bpy.data.images.remove(image)


def _bouncy_wall_appearance_profile(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    object_plan_path = base._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.is_file():
        return {
            "status": "unavailable",
            "reason": f"missing object plan: {object_plan_path}",
        }
    object_plan = base._load_json(object_plan_path)
    scenario = str(
        (
            ((object_plan.get("special_scene") or {}).get("scene_metadata") or {}).get(
                "scenario"
            )
            or ""
        )
    )
    if scenario != "bouncy_wall_pp":
        return {
            "status": "not_applicable",
            "scenario": scenario,
        }

    question_dir = Path(str(world_reconstruction.get("question_dir") or ""))
    sam3_path = (
        question_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json"
    )
    if not sam3_path.is_file():
        raise FileNotFoundError(
            f"bouncy-wall refined appearance requires SAM3 tracks: {sam3_path}"
        )
    video_metadata = world_reconstruction.get("video_metadata")
    frame_paths = (
        video_metadata.get("frame_paths")
        if isinstance(video_metadata, dict)
        else None
    )
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValueError(
            "bouncy-wall refined appearance requires video_metadata.frame_paths"
        )
    profile = bouncy_wall_appearance.build_bouncy_wall_appearance_profile(
        object_plan=object_plan,
        sam3_tracks=base._load_json(sam3_path),
        frame_paths=[Path(str(path)) for path in frame_paths],
        image_loader=_load_source_image_srgb,
    )
    profile["object_plan"] = str(object_plan_path)
    profile["sam3_tracks"] = str(sam3_path)
    return profile


def _bouncy_platform_appearance_profile(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    object_plan_path = base._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.is_file():
        return {
            "status": "unavailable",
            "reason": f"missing object plan: {object_plan_path}",
        }
    object_plan = base._load_json(object_plan_path)
    scenario = str(
        (
            ((object_plan.get("special_scene") or {}).get("scene_metadata") or {}).get(
                "scenario"
            )
            or ""
        )
    )
    if scenario != "bouncy_platform_pp":
        return {"status": "not_applicable", "scenario": scenario}

    question_dir = Path(str(world_reconstruction.get("question_dir") or ""))
    sam3_path = (
        question_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json"
    )
    if not sam3_path.is_file():
        raise FileNotFoundError(
            f"bouncy-platform refined appearance requires SAM3 tracks: {sam3_path}"
        )
    video_metadata = world_reconstruction.get("video_metadata")
    frame_paths = (
        video_metadata.get("frame_paths")
        if isinstance(video_metadata, dict)
        else None
    )
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValueError(
            "bouncy-platform refined appearance requires video_metadata.frame_paths"
        )
    profile = bouncy_platform_appearance.build_bouncy_platform_appearance_profile(
        object_plan=object_plan,
        sam3_tracks=base._load_json(sam3_path),
        frame_paths=[Path(str(path)) for path in frame_paths],
        image_loader=_load_source_image_srgb,
    )
    profile["object_plan"] = str(object_plan_path)
    profile["sam3_tracks"] = str(sam3_path)
    return profile


def _friction_collision_appearance_profile(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    object_plan_path = base._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.is_file():
        return {
            "status": "unavailable",
            "reason": f"missing object plan: {object_plan_path}",
        }
    object_plan = base._load_json(object_plan_path)
    scenario = str(
        (
            ((object_plan.get("special_scene") or {}).get("scene_metadata") or {}).get(
                "scenario"
            )
            or ""
        )
    )
    if scenario != "friction_collision_pp":
        return {"status": "not_applicable", "scenario": scenario}

    question_dir = Path(str(world_reconstruction.get("question_dir") or ""))
    sam3_path = (
        question_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json"
    )
    if not sam3_path.is_file():
        raise FileNotFoundError(
            f"friction-collision refined appearance requires SAM3 tracks: {sam3_path}"
        )
    video_metadata = world_reconstruction.get("video_metadata")
    frame_paths = (
        video_metadata.get("frame_paths")
        if isinstance(video_metadata, dict)
        else None
    )
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValueError(
            "friction-collision refined appearance requires video_metadata.frame_paths"
        )
    profile = friction_collision_appearance.build_friction_collision_appearance_profile(
        object_plan=object_plan,
        sam3_tracks=base._load_json(sam3_path),
        frame_paths=[Path(str(path)) for path in frame_paths],
        image_loader=_load_source_image_srgb,
    )
    profile["object_plan"] = str(object_plan_path)
    profile["sam3_tracks"] = str(sam3_path)
    return profile


def _mass_collision_appearance_profile(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    object_plan_path = base._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.is_file():
        return {
            "status": "unavailable",
            "reason": f"missing object plan: {object_plan_path}",
        }
    object_plan = base._load_json(object_plan_path)
    scenario = str(
        (
            ((object_plan.get("special_scene") or {}).get("scene_metadata") or {}).get(
                "scenario"
            )
            or ""
        )
    )
    if scenario != "mass_collision_pp":
        return {"status": "not_applicable", "scenario": scenario}

    question_dir = Path(str(world_reconstruction.get("question_dir") or ""))
    sam3_path = (
        question_dir
        / "object-segmentation-and-event-detection"
        / "sam3_video_tracks"
        / "sam3_video_tracks.json"
    )
    if not sam3_path.is_file():
        raise FileNotFoundError(
            f"mass-collision refined appearance requires SAM3 tracks: {sam3_path}"
        )
    video_metadata = world_reconstruction.get("video_metadata")
    frame_paths = (
        video_metadata.get("frame_paths")
        if isinstance(video_metadata, dict)
        else None
    )
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValueError(
            "mass-collision refined appearance requires video_metadata.frame_paths"
        )
    profile = mass_collision_appearance.build_mass_collision_appearance_profile(
        object_plan=object_plan,
        sam3_tracks=base._load_json(sam3_path),
        frame_paths=[Path(str(path)) for path in frame_paths],
        image_loader=_load_source_image_srgb,
    )
    profile["object_plan"] = str(object_plan_path)
    profile["sam3_tracks"] = str(sam3_path)
    return profile


def _refined_appearance_profile(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, Any]:
    bouncy_wall = _bouncy_wall_appearance_profile(
        world_reconstruction_path,
        world_reconstruction,
    )
    if bouncy_wall.get("status") == "ok":
        return bouncy_wall
    bouncy_platform = _bouncy_platform_appearance_profile(
        world_reconstruction_path,
        world_reconstruction,
    )
    if bouncy_platform.get("status") == "ok":
        return bouncy_platform
    friction_collision = _friction_collision_appearance_profile(
        world_reconstruction_path,
        world_reconstruction,
    )
    if friction_collision.get("status") == "ok":
        return friction_collision
    mass_collision = _mass_collision_appearance_profile(
        world_reconstruction_path,
        world_reconstruction,
    )
    if mass_collision.get("status") == "ok":
        return mass_collision
    return bouncy_wall


def _principled_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
):
    import bpy

    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = (
            0.32 if metallic > 0.0 else 0.09
        )
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _scaled_color(
    color: list[float] | tuple[float, ...],
    scale: float,
) -> tuple[float, float, float, float]:
    rgb = np.clip(np.asarray(color[:3], dtype=np.float64) * scale, 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _calibrated_object_material(
    name: str,
    profile: dict[str, Any],
):
    import bpy

    material = bpy.data.materials.new(name)
    role = str(profile.get("role") or "")
    role_albedo_scale = {
        "wall": 1.85,
        "patient": 0.72,
        "agent": 1.0,
    }.get(role, 1.0)
    role_albedo_scale = float(
        profile.get("render_albedo_scale", role_albedo_scale)
    )
    base_color = tuple(
        float(value) for value in profile.get("base_linear", [0.18, 0.18, 0.18])
    )
    base_color = tuple(
        float(value)
        for value in np.clip(
            np.asarray(base_color, dtype=np.float64) * role_albedo_scale,
            0.0,
            1.0,
        )
    )
    material.diffuse_color = (*base_color, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    bump = nodes.new(type="ShaderNodeBump")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material_family = str(profile.get("material_family") or "rubber")
    contrast = float(profile.get("texture_contrast") or 0.15)

    if material_family == "wood":
        wave = nodes.new(type="ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X" if role == "wall" else "Y"
        wave.inputs["Scale"].default_value = 4.5 if role == "wall" else 6.0
        wave.inputs["Distortion"].default_value = 8.0
        wave.inputs["Detail"].default_value = 4.0
        wave.inputs["Detail Scale"].default_value = 2.2
        wave.inputs["Detail Roughness"].default_value = 0.78
        noise.inputs["Scale"].default_value = 3.5
        noise.inputs["Detail"].default_value = 5.0
        noise.inputs["Roughness"].default_value = 0.72
        noise.inputs["Distortion"].default_value = 0.22
        mix = nodes.new(type="ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.68
        material.node_tree.links.new(
            texture_coordinates.outputs["Generated"],
            noise.inputs["Vector"],
        )
        material.node_tree.links.new(
            texture_coordinates.outputs["Generated"],
            wave.inputs["Vector"],
        )
        material.node_tree.links.new(noise.outputs["Fac"], mix.inputs["Color1"])
        material.node_tree.links.new(wave.outputs["Color"], mix.inputs["Color2"])
        material.node_tree.links.new(mix.outputs["Color"], ramp.inputs["Fac"])
        dark_scale = max(0.42, 0.76 - 0.32 * contrast)
        light_scale = min(1.55, 1.12 + 0.55 * contrast)
        bump.inputs["Strength"].default_value = 0.16
        bump.inputs["Distance"].default_value = 0.018
        bsdf.inputs["Roughness"].default_value = 0.57
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.20
    else:
        noise.inputs["Scale"].default_value = 8.0
        noise.inputs["Detail"].default_value = 3.0
        noise.inputs["Roughness"].default_value = 0.68
        noise.inputs["Distortion"].default_value = 0.08
        material.node_tree.links.new(
            texture_coordinates.outputs["Generated"],
            noise.inputs["Vector"],
        )
        material.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        dark_scale = max(0.58, 0.86 - 0.20 * contrast)
        light_scale = min(1.30, 1.06 + 0.28 * contrast)
        bump.inputs["Strength"].default_value = 0.055
        bump.inputs["Distance"].default_value = 0.008
        bsdf.inputs["Roughness"].default_value = 0.50
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.18

    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = _scaled_color(base_color, dark_scale)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = _scaled_color(base_color, light_scale)
    links = material.node_tree.links
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(ramp.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _textured_floor_material(
    appearance: dict[str, Any] | None = None,
):
    import bpy

    material = bpy.data.materials.new("refined_floor_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(
        (appearance or {}).get("texture_scale", 68.0)
    )
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.72
    noise.inputs["Distortion"].default_value = 0.08
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.22
    base_color = (
        appearance.get("base_linear")
        if isinstance(appearance, dict)
        else None
    ) or [0.020, 0.055, 0.095]
    base_color = [
        float(value)
        * float((appearance or {}).get("render_albedo_scale", 1.0))
        for value in base_color
    ]
    ramp.color_ramp.elements[0].color = _scaled_color(
        base_color,
        float((appearance or {}).get("texture_dark_scale", 0.58)),
    )
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = _scaled_color(
        base_color,
        float((appearance or {}).get("texture_light_scale", 1.02)),
    )
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.025
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.86
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    output = nodes.new(type="ShaderNodeOutputMaterial")

    links = material.node_tree.links
    links.new(texture_coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _textured_plaster_material(
    name: str = "refined_wall_material",
    appearance: dict[str, Any] | None = None,
    *,
    noise_scale: float = 11.0,
    bump_strength: float = 0.07,
    albedo_scale: float = 1.0,
):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.70
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.24
    base_color = (
        appearance.get("base_linear")
        if isinstance(appearance, dict)
        else None
    ) or [0.13, 0.14, 0.18]
    base_color = [
        float(value) * albedo_scale
        for value in base_color
    ]
    ramp.color_ramp.elements[0].color = _scaled_color(base_color, 0.76)
    ramp.color_ramp.elements[1].position = 0.76
    ramp.color_ramp.elements[1].color = _scaled_color(base_color, 1.24)
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.018
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.91
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.12
    output = nodes.new(type="ShaderNodeOutputMaterial")

    links = material.node_tree.links
    links.new(texture_coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _emissive_material(
    name: str,
    color: tuple[float, float, float, float],
    strength: float,
):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_box(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: Any,
):
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def _add_room_environment(
    world_reconstruction: dict[str, Any],
    back_wall_y: float,
    appearance_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import bpy

    environment = (
        appearance_profile.get("environment")
        if isinstance(appearance_profile, dict)
        and appearance_profile.get("status") == "ok"
        else {}
    )
    if not isinstance(environment, dict):
        environment = {}
    static_objects: list[dict[str, Any]] = []
    ground = base._add_ground_plane(world_reconstruction, size=80.0)
    if ground is not None:
        floor = bpy.data.objects.get("world_reconstruction_ground_plane")
        if floor is not None:
            floor.data.materials.clear()
            floor.data.materials.append(
                _textured_floor_material(environment.get("floor"))
            )
        ground["style"] = "procedural_blue_gray_fabric_floor"
        static_objects.append(ground)

    wall_material = _textured_plaster_material(
        appearance=environment.get("back_wall"),
        albedo_scale=1.25,
    )
    ceiling_material = _textured_plaster_material(
        "refined_ceiling_material",
        environment.get("ceiling"),
        noise_scale=38.0,
        bump_strength=0.055,
        albedo_scale=1.80,
    )
    beam_material = _principled_material(
        "refined_ceiling_beam_material",
        _scaled_color(
            (environment.get("ceiling") or {}).get(
                "base_linear",
                [0.28, 0.30, 0.32],
            ),
            0.54,
        ),
        roughness=0.82,
    )
    window_material = _emissive_material(
        "refined_window_material",
        (0.78, 0.82, 0.86, 1.0),
        0.52,
    )

    _add_box(
        "refined_back_wall",
        (0.0, back_wall_y, 2.8),
        (30.0, 0.22, 11.0),
        wall_material,
    )
    ceiling_front_y = 4.0
    ceiling_depth = max(back_wall_y - ceiling_front_y, 1.0)
    _add_box(
        "refined_ceiling",
        (0.0, ceiling_front_y + 0.5 * ceiling_depth, 6.10),
        (30.0, ceiling_depth, 0.22),
        ceiling_material,
    )
    for index, x in enumerate((-3.0, -0.15)):
        _add_box(
            f"refined_window_{index}",
            (x, back_wall_y - 0.16, 5.60),
            (1.90, 0.08, 0.82),
            window_material,
        )
    for index, y in enumerate((5.0, 7.6)):
        _add_box(
            f"refined_ceiling_beam_{index}",
            (-2.1, y, 5.88),
            (8.0, 0.42, 0.45),
            beam_material,
        )

    static_objects.append(
        {
            "object_id": "refined_room_environment",
            "geometry_type": "procedural_room",
            "back_wall_y": back_wall_y,
            "ceiling_z": 6.10,
            "window_count": 2,
            "ceiling_beam_count": 2,
        }
    )
    return static_objects


def _add_friction_collision_environment(
    world_reconstruction: dict[str, Any],
    back_wall_y: float,
    appearance_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    import bpy

    environment = appearance_profile.get("environment") or {}
    static_objects: list[dict[str, Any]] = []
    ground = base._add_ground_plane(world_reconstruction, size=80.0)
    if ground is not None:
        floor = bpy.data.objects.get("world_reconstruction_ground_plane")
        if floor is not None:
            floor.data.materials.clear()
            floor.data.materials.append(
                _textured_floor_material(environment.get("floor"))
            )
        ground["style"] = "source_calibrated_blue_gray_fabric_floor"
        static_objects.append(ground)

    wall_base_linear = (environment.get("back_wall") or {}).get(
        "base_linear",
        [0.25, 0.26, 0.32],
    )
    wall_material = _emissive_material(
        "refined_friction_collision_back_wall_material",
        (*[float(value) for value in wall_base_linear], 1.0),
        0.95,
    )
    back_wall_slope = 0.155
    back_wall = _add_box(
        "refined_friction_collision_back_wall",
        (0.0, back_wall_y - back_wall_slope, 12.0),
        (120.0, 0.22, 30.0),
        wall_material,
    )
    back_wall.rotation_euler[2] = math.atan(back_wall_slope)
    side_wall_start = (1.0, back_wall_y)
    side_wall_direction = (0.545, -0.839)
    side_wall_length = 18.0
    side_wall = _add_box(
        "refined_friction_collision_side_wall",
        (
            side_wall_start[0]
            + 0.5 * side_wall_length * side_wall_direction[0],
            side_wall_start[1]
            + 0.5 * side_wall_length * side_wall_direction[1],
            12.0,
        ),
        (side_wall_length, 0.22, 30.0),
        wall_material,
    )
    side_wall.rotation_euler[2] = math.atan2(
        side_wall_direction[1],
        side_wall_direction[0],
    )
    static_objects.append(
        {
            "object_id": "refined_friction_collision_environment",
            "geometry_type": "source_calibrated_floor_back_wall_and_side_wall",
            "back_wall_y": back_wall_y,
            "ceiling_added": False,
        }
    )
    return static_objects


def _add_mass_collision_environment(
    world_reconstruction: dict[str, Any],
    back_wall_y: float,
    appearance_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    import bpy

    environment = appearance_profile.get("environment") or {}
    static_objects: list[dict[str, Any]] = []
    ground = base._add_ground_plane(world_reconstruction, size=80.0)
    if ground is not None:
        floor = bpy.data.objects.get("world_reconstruction_ground_plane")
        if floor is not None:
            floor.data.materials.clear()
            floor.data.materials.append(
                _textured_floor_material(environment.get("floor"))
            )
        ground["style"] = "source_calibrated_mass_collision_fabric_floor"
        static_objects.append(ground)

    static_objects.append(
        {
            "object_id": "refined_mass_collision_environment",
            "geometry_type": "source_calibrated_physical_floor",
            "back_wall_y": back_wall_y,
        }
    )
    return static_objects


def _add_bouncy_platform_environment(
    world_reconstruction: dict[str, Any],
    back_wall_y: float,
    appearance_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    import bpy

    environment = appearance_profile.get("environment") or {}
    static_objects: list[dict[str, Any]] = []
    ground = base._add_ground_plane(world_reconstruction, size=80.0)
    if ground is not None:
        floor = bpy.data.objects.get("world_reconstruction_ground_plane")
        if floor is not None:
            floor.data.materials.clear()
            floor.data.materials.append(
                _textured_floor_material(environment.get("floor"))
            )
            floor.hide_render = True
        ground["style"] = "render_hidden_support_reference"
        static_objects.append(ground)

    static_objects.append(
        {
            "object_id": "refined_bouncy_platform_environment",
            "geometry_type": "source_calibrated_fabric_floor",
            "back_wall_y": back_wall_y,
        }
    )
    return static_objects


def _bouncy_platform_backdrop_material(
    appearance: dict[str, Any],
):
    return _textured_plaster_material(
        "refined_bouncy_platform_camera_backdrop_material",
        appearance,
        noise_scale=11.0,
        bump_strength=0.07,
        albedo_scale=1.35,
    )


def _bouncy_platform_floor_backdrop_material(
    appearance: dict[str, Any],
):
    material = _textured_floor_material(appearance)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    ramp = next(
        node
        for node in nodes
        if node.bl_idname == "ShaderNodeValToRGB"
    )
    bsdf = next(
        node
        for node in nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    if "Emission Color" in bsdf.inputs and "Emission Strength" in bsdf.inputs:
        links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.82
    return material


def _bouncy_platform_picture_material(index: int):
    import bpy

    material = bpy.data.materials.new(
        f"refined_bouncy_platform_picture_material_{index}"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.5 if index == 0 else 8.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.72
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = (
        (0.025, 0.030, 0.028, 1.0)
        if index == 0
        else (0.018, 0.022, 0.027, 1.0)
    )
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (
        (0.46, 0.48, 0.44, 1.0)
        if index == 0
        else (0.24, 0.27, 0.30, 1.0)
    )
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 0.88
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links = material.node_tree.links
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _camera_rect_object(
    *,
    scene: Any,
    camera: Any,
    name: str,
    normalized_rect: tuple[float, float, float, float],
    distance: float,
    material: Any,
):
    import bpy
    from mathutils import Matrix

    frame = camera.data.view_frame(scene=scene)
    left = min(float(corner.x) for corner in frame)
    right = max(float(corner.x) for corner in frame)
    bottom = min(float(corner.y) for corner in frame)
    top = max(float(corner.y) for corner in frame)
    reference_depth = max(abs(float(frame[0].z)), 1e-6)
    scale = float(distance) / reference_depth
    u0, v0, u1, v1 = normalized_rect

    def point(u: float, v: float) -> tuple[float, float, float]:
        x = left + float(u) * (right - left)
        y = top - float(v) * (top - bottom)
        return (x * scale, y * scale, -float(distance))

    vertices = [
        point(u0, v0),
        point(u0, v1),
        point(u1, v1),
        point(u1, v0),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = camera
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    obj.data.materials.append(material)
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj


def _camera_polygon_object(
    *,
    scene: Any,
    camera: Any,
    name: str,
    normalized_vertices: tuple[tuple[float, float], ...],
    distance: float,
    material: Any,
):
    import bpy
    from mathutils import Matrix

    frame = camera.data.view_frame(scene=scene)
    left = min(float(corner.x) for corner in frame)
    right = max(float(corner.x) for corner in frame)
    bottom = min(float(corner.y) for corner in frame)
    top = max(float(corner.y) for corner in frame)
    reference_depth = max(abs(float(frame[0].z)), 1e-6)
    scale = float(distance) / reference_depth
    vertices = [
        (
            (left + float(u) * (right - left)) * scale,
            (top - float(v) * (top - bottom)) * scale,
            -float(distance),
        )
        for u, v in normalized_vertices
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], [tuple(range(len(vertices)))])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = camera
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    obj.data.materials.append(material)
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj


def _mass_collision_camera_wall_material(
    name: str,
    appearance: dict[str, Any],
):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 46.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.72
    ramp = nodes.new(type="ShaderNodeValToRGB")
    base_color = np.asarray(
        appearance.get("base_linear") or [0.28, 0.29, 0.32],
        dtype=np.float64,
    ) * float(appearance.get("camera_albedo_scale", 1.0))
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = _scaled_color(base_color, 0.88)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = _scaled_color(base_color, 1.12)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links = material.node_tree.links
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_mass_collision_camera_wall(
    scene: Any,
    camera: Any,
) -> dict[str, Any]:
    _camera_polygon_object(
        scene=scene,
        camera=camera,
        name="refined_mass_collision_camera_frosted_wall",
        normalized_vertices=(
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.52),
            (0.0, 0.405),
        ),
        distance=5.5,
        material=_mass_collision_camera_wall_material(
            "refined_mass_collision_camera_frosted_wall_material",
            {
                "base_linear": [0.48, 0.50, 0.52],
                "camera_albedo_scale": 1.0,
            },
        ),
    )
    panel_material = _emissive_material(
        "refined_mass_collision_wall_panel_material",
        (0.72, 0.80, 0.88, 1.0),
        1.05,
    )
    panel_rects = (
        (0.160, -0.035, 0.285, 0.195),
        (0.360, -0.035, 0.520, 0.195),
    )
    for index, normalized_rect in enumerate(panel_rects):
        _camera_rect_object(
            scene=scene,
            camera=camera,
            name=f"refined_mass_collision_camera_wall_panel_{index}",
            normalized_rect=normalized_rect,
            distance=5.4,
            material=panel_material,
        )
    return {
        "object_id": "refined_mass_collision_camera_wall_and_windows",
        "geometry_type": "camera_space_frosted_white_wall_and_two_windows",
        "panel_count": len(panel_rects),
    }


def _add_bouncy_platform_camera_backdrop(
    scene: Any,
    camera: Any,
    appearance_profile: dict[str, Any],
) -> dict[str, Any]:
    environment = appearance_profile.get("environment") or {}
    wall_left_material = _bouncy_platform_backdrop_material(
        environment.get("back_wall_left")
        or environment.get("back_wall")
        or {}
    )
    wall_right_material = _bouncy_platform_backdrop_material(
        environment.get("back_wall_right")
        or environment.get("back_wall")
        or {}
    )
    floor_material = _bouncy_platform_floor_backdrop_material(
        environment.get("floor") or {}
    )
    frame_material = _emissive_material(
        "refined_bouncy_platform_picture_frame_material",
        (0.014, 0.016, 0.018, 1.0),
        0.85,
    )
    _camera_rect_object(
        scene=scene,
        camera=camera,
        name="refined_bouncy_platform_camera_wall_left",
        normalized_rect=(0.0, 0.0, 0.63, 0.505),
        distance=20.0,
        material=wall_left_material,
    )
    _camera_rect_object(
        scene=scene,
        camera=camera,
        name="refined_bouncy_platform_camera_wall_right",
        normalized_rect=(0.63, 0.0, 1.0, 0.505),
        distance=20.0,
        material=wall_right_material,
    )
    _camera_rect_object(
        scene=scene,
        camera=camera,
        name="refined_bouncy_platform_camera_floor",
        normalized_rect=(0.0, 0.505, 1.0, 1.0),
        distance=20.0,
        material=floor_material,
    )
    picture_rects = (
        (0.350, 0.045, 0.470, 0.145),
        (0.520, 0.038, 0.590, 0.128),
    )
    for index, (u0, v0, u1, v1) in enumerate(picture_rects):
        _camera_rect_object(
            scene=scene,
            camera=camera,
            name=f"refined_bouncy_platform_camera_picture_frame_{index}",
            normalized_rect=(u0, v0, u1, v1),
            distance=19.8,
            material=frame_material,
        )
        inset_u = 0.008
        inset_v = 0.010
        _camera_rect_object(
            scene=scene,
            camera=camera,
            name=f"refined_bouncy_platform_camera_picture_{index}",
            normalized_rect=(
                u0 + inset_u,
                v0 + inset_v,
                u1 - inset_u,
                v1 - inset_v,
            ),
            distance=19.6,
            material=_bouncy_platform_picture_material(index),
        )
    return {
        "object_id": "refined_bouncy_platform_camera_backdrop",
        "geometry_type": "camera_space_wall_floor_and_picture_frames",
        "wall_bottom_normalized_v": 0.505,
        "picture_count": 2,
    }


def _assign_bouncy_platform_mixed_material(
    obj: Any,
    object_id: str,
    object_profile: dict[str, Any],
) -> tuple[Any, list[str]] | None:
    palette = object_profile.get("mixed_palette")
    if object_id not in {"obj_3", "obj_4"} or not isinstance(palette, dict):
        return None
    blue = dict(object_profile)
    blue.update(
        {
            "base_linear": palette.get("blue_linear"),
            "material_family": "rubber",
            "role": "platform",
        }
    )
    wood = dict(object_profile)
    wood.update(
        {
            "base_linear": palette.get("wood_linear"),
            "material_family": "wood",
            "role": "patient",
        }
    )
    blue_material = _calibrated_object_material(
        f"refined_bouncy_platform_blue_material_{object_id}",
        blue,
    )
    wood_material = _calibrated_object_material(
        f"refined_bouncy_platform_wood_material_{object_id}",
        wood,
    )
    obj.data.materials.clear()
    obj.data.materials.append(blue_material)
    obj.data.materials.append(wood_material)
    for polygon in obj.data.polygons:
        polygon.material_index = (
            0 if abs(float(polygon.normal.z)) >= 0.58 else 1
        )
    return blue_material, [blue_material.name, wood_material.name]


def _textured_curtain_material(
    appearance: dict[str, Any],
):
    import bpy

    material = bpy.data.materials.new("refined_bouncy_wall_curtain_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.8
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.75
    ramp = nodes.new(type="ShaderNodeValToRGB")
    base_color = appearance.get("base_linear") or [0.025, 0.020, 0.018]
    ramp.color_ramp.elements[0].color = _scaled_color(
        base_color,
        float(appearance.get("render_dark_scale", 0.24)),
    )
    ramp.color_ramp.elements[1].color = _scaled_color(
        base_color,
        float(appearance.get("render_light_scale", 0.62)),
    )
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links = material.node_tree.links
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_bouncy_wall_curtain(
    scene: Any,
    camera: Any,
    appearance_profile: dict[str, Any],
) -> dict[str, Any] | None:
    import bpy
    from mathutils import Matrix

    if appearance_profile.get("status") != "ok":
        return None
    two_segment = appearance_profile.get("two_segment")
    environment = appearance_profile.get("environment")
    if not isinstance(two_segment, dict) or not isinstance(environment, dict):
        return None
    interval = two_segment.get("curtain")
    curtain_appearance = environment.get("curtain")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not isinstance(curtain_appearance, dict)
    ):
        return None
    first_frame, last_frame = (int(interval[0]), int(interval[1]))
    camera_frame = camera.data.view_frame(scene=scene)
    distance = 0.12
    scale = distance / max(abs(float(camera_frame[0].z)), 1e-6)
    motion = two_segment.get("curtain_motion")
    if isinstance(motion, dict):
        left = min(float(corner.x) for corner in camera_frame)
        right = max(float(corner.x) for corner in camera_frame)
        bottom = min(float(corner.y) for corner in camera_frame)
        top = max(float(corner.y) for corner in camera_frame)
        width_normalized = float(motion.get("width_normalized", 2.5))
        top_normalized = float(motion.get("top_normalized", 0.13))
        bottom_normalized = float(motion.get("bottom_normalized", 1.03))

        def point(u: float, v: float) -> tuple[float, float, float]:
            return (
                float((left + u * (right - left)) * scale),
                float((top - v * (top - bottom)) * scale),
                -distance,
            )

        vertices = [
            point(-width_normalized, top_normalized),
            point(-width_normalized, bottom_normalized),
            point(0.0, bottom_normalized),
            point(0.0, top_normalized),
        ]
    else:
        vertices = [
            (float(corner.x * scale), float(corner.y * scale), -distance)
            for corner in camera_frame
        ]
    mesh = bpy.data.meshes.new("refined_bouncy_wall_curtain_mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    curtain = bpy.data.objects.new("refined_bouncy_wall_curtain", mesh)
    bpy.context.collection.objects.link(curtain)
    curtain.parent = camera
    curtain.matrix_parent_inverse = Matrix.Identity(4)
    curtain.matrix_basis = Matrix.Identity(4)
    curtain.data.materials.append(
        _textured_curtain_material(curtain_appearance)
    )
    if isinstance(motion, dict):
        motion_start = int(motion.get("start_frame", first_frame))
        motion_end = int(motion.get("end_frame", last_frame))
        curtain.location.x = 0.0
        curtain.keyframe_insert(data_path="location", frame=motion_start)
        curtain.location.x = float(
            (1.0 + width_normalized) * (right - left) * scale
        )
        curtain.keyframe_insert(data_path="location", frame=motion_end)
        action = (
            curtain.animation_data.action
            if curtain.animation_data is not None
            else None
        )
        if action is not None:
            for fcurve in action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"
    base._set_visibility(curtain, max(0, first_frame - 1), False)
    base._set_visibility(curtain, first_frame, True)
    base._set_visibility(curtain, last_frame, True)
    base._set_visibility(curtain, last_frame + 1, False)
    return {
        "object_id": "refined_bouncy_wall_curtain",
        "geometry_type": "camera_filling_transition_curtain",
        "first_frame": first_frame,
        "last_frame": last_frame,
        "material_color_srgb": curtain_appearance.get("base_srgb"),
        "motion": motion,
    }


def _point_light_at(
    light_type: str,
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    energy: float,
    size: float,
    color: tuple[float, float, float],
) -> dict[str, Any]:
    import bpy
    from mathutils import Vector

    bpy.ops.object.light_add(type=light_type, location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color
    if hasattr(light.data, "shape"):
        light.data.shape = "DISK"
    if hasattr(light.data, "size"):
        light.data.size = size
    direction = Vector(target) - Vector(location)
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return {
        "name": name,
        "type": light_type,
        "location": list(location),
        "target": list(target),
        "energy": energy,
        "size": size,
        "color": list(color),
    }


def _setup_refined_lighting(
    appearance_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import bpy

    calibrated = (
        isinstance(appearance_profile, dict)
        and appearance_profile.get("status") == "ok"
    )
    world = bpy.context.scene.world or bpy.data.worlds.new("refined_debug_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (
            (0.13, 0.14, 0.16, 1.0)
            if calibrated
            else (0.38, 0.43, 0.50, 1.0)
        )
        background.inputs["Strength"].default_value = 0.26 if calibrated else 0.32

    return [
        _point_light_at(
            "AREA",
            "refined_key_light",
            (-3.8, -1.8, 8.2),
            (0.0, 7.0, -0.8),
            energy=1500.0 if calibrated else 1950.0,
            size=4.8 if calibrated else 5.8,
            color=(1.0, 0.95, 0.88),
        ),
        _point_light_at(
            "AREA",
            "refined_fill_light",
            (-5.5, 3.0, 3.8),
            (0.0, 8.0, -1.0),
            energy=145.0 if calibrated else 300.0,
            size=7.0,
            color=(0.76, 0.84, 1.0),
        ),
    ]


def _setup_refined_render(
    scene: Any,
    source_width: int,
    source_height: int,
    resolution_scale: int,
    fps: int,
) -> dict[str, Any]:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.render.fps = fps
    scene.render.resolution_x = int(source_width * resolution_scale)
    scene.render.resolution_y = int(source_height * resolution_scale)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_management = "FOLLOW_SCENE"

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.05
    scene.view_settings.gamma = 1.0
    return {
        "engine": str(scene.render.engine),
        "cycles_device": str(scene.cycles.device),
        "samples": int(scene.cycles.samples),
        "adaptive_threshold": float(scene.cycles.adaptive_threshold),
        "denoising": bool(scene.cycles.use_denoising),
        "resolution_scale": resolution_scale,
        "resolution": {
            "width": int(scene.render.resolution_x),
            "height": int(scene.render.resolution_y),
        },
        "view_transform": str(scene.view_settings.view_transform),
        "look": str(scene.view_settings.look),
        "exposure": float(scene.view_settings.exposure),
        "ffmpeg_quality": str(scene.render.ffmpeg.constant_rate_factor),
    }


def _render(
    world_reconstruction_path: Path,
    support_reference_path: Path,
    output_video: Path,
    output_json: Path,
    resolution_scale: int,
    preview_frame: int | None,
) -> None:
    import bpy
    from mathutils import Matrix, Vector

    raw_world_reconstruction = base._load_json(world_reconstruction_path)
    support_reference = base._load_json(support_reference_path)
    world_reconstruction = _merge_reference_geometry(
        raw_world_reconstruction, support_reference
    )
    source_width, source_height, fps = base._video_metadata(world_reconstruction)
    # The support reference supplies only environment geometry. Object meshes,
    # camera-space poses, visibility, and timing must remain byte-for-byte
    # sourced from the SWR render input so the refined render is visually
    # comparable to the original SWR debug video.
    objects = base._trajectory_objects(raw_world_reconstruction)
    swr_debug_mesh_paths = raw_world_reconstruction.get("debug_mesh_paths")
    if not isinstance(swr_debug_mesh_paths, dict):
        swr_debug_mesh_paths = {}
    appearances = _appearance_by_object_id(
        world_reconstruction_path, world_reconstruction
    )
    appearance_profile = _refined_appearance_profile(
        world_reconstruction_path,
        world_reconstruction,
    )
    calibrated_appearance = appearance_profile.get("status") == "ok"
    calibrated_bouncy_wall = (
        calibrated_appearance
        and appearance_profile.get("scenario") == "bouncy_wall_pp"
    )
    calibrated_bouncy_platform = (
        calibrated_appearance
        and appearance_profile.get("scenario") == "bouncy_platform_pp"
    )
    calibrated_friction_collision = (
        calibrated_appearance
        and appearance_profile.get("scenario") == "friction_collision_pp"
    )
    calibrated_mass_collision = (
        calibrated_appearance
        and appearance_profile.get("scenario") == "mass_collision_pp"
    )
    canonical_by_object_id = (
        appearance_profile.get("canonical_by_object_id")
        if calibrated_appearance
        else {}
    )
    calibrated_object_profiles = (
        appearance_profile.get("object_profiles")
        if calibrated_appearance
        else {}
    )
    if not isinstance(canonical_by_object_id, dict):
        canonical_by_object_id = {}
    if not isinstance(calibrated_object_profiles, dict):
        calibrated_object_profiles = {}

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    render_settings = _setup_refined_render(
        scene, source_width, source_height, resolution_scale, fps
    )
    lights = _setup_refined_lighting(appearance_profile)

    rendered_objects = []
    material_by_canonical_id: dict[str, Any] = {}
    frame_indices: list[int] = []
    content_max_y = float("-inf")
    for object_index, item in enumerate(objects):
        object_id = str(item.get("object_id") or "")
        mesh_path = item.get("mesh_path")
        poses = [
            pose
            for pose in item.get("poses", [])
            if isinstance(pose, dict) and pose.get("corrected_pose_4x4")
        ]
        if not object_id or not mesh_path or not Path(str(mesh_path)).exists() or not poses:
            continue
        expected_mesh_path = swr_debug_mesh_paths.get(object_id)
        if expected_mesh_path and Path(str(mesh_path)).resolve() != Path(
            str(expected_mesh_path)
        ).resolve():
            raise ValueError(
                f"Refined render mesh drift for {object_id}: "
                f"SWR expects {expected_mesh_path}, selected {mesh_path}"
            )

        obj = base._import_mesh(str(mesh_path))
        obj.name = object_id
        appearance = appearances.get(object_id, {})
        color_name = appearance.get("color", "")
        material_type = appearance.get("material", "rubber")
        canonical_id = str(canonical_by_object_id.get(object_id) or object_id)
        calibrated_object = calibrated_object_profiles.get(canonical_id)
        if isinstance(calibrated_object, dict):
            material = material_by_canonical_id.get(canonical_id)
            if material is None:
                material = _calibrated_object_material(
                    f"refined_bouncy_wall_material_{canonical_id}",
                    calibrated_object,
                )
                material_by_canonical_id[canonical_id] = material
            color = tuple(
                [
                    *(
                        float(value)
                        for value in calibrated_object.get(
                            "base_linear",
                            [0.18, 0.18, 0.18],
                        )
                    ),
                    1.0,
                ]
            )
            material_type = str(
                calibrated_object.get("material_family") or material_type
            )
        else:
            color = OBJECT_COLORS.get(
                color_name,
                base.DEBUG_COLORS[object_index % len(base.DEBUG_COLORS)],
            )
            material = _principled_material(
                f"refined_object_material_{object_id}",
                color,
                roughness=0.76 if material_type != "metal" else 0.18,
                metallic=1.0 if material_type == "metal" else 0.0,
            )
        base._assign_debug_material(obj, material)
        material_names = [str(material.name)]
        if calibrated_bouncy_platform and isinstance(calibrated_object, dict):
            mixed = _assign_bouncy_platform_mixed_material(
                obj,
                object_id,
                calibrated_object,
            )
            if mixed is not None:
                material, material_names = mixed
                material_type = "mixed_blue_fabric_and_brown_wood"

        first_frame, last_frame = base._active_interval(item, poses)
        base._set_visibility(obj, max(0, first_frame - 1), False)
        base._set_visibility(obj, first_frame, True)
        for pose in poses:
            frame_index = int(pose["frame_index"])
            matrix = Matrix(
                base._opencv_pose_to_blender_world(pose["corrected_pose_4x4"])
            )
            for corner in obj.bound_box:
                content_max_y = max(
                    content_max_y, float((matrix @ Vector(corner)).y)
                )
            base._keyframe_matrix(obj, matrix, frame_index)
            frame_indices.append(frame_index)
        base._set_visibility(obj, last_frame + 1, False)
        rendered_objects.append(
            {
                "object_id": object_id,
                "canonical_object_id": canonical_id,
                "mesh_path": str(mesh_path),
                "appearance_color": color_name,
                "material_type": material_type,
                "material_color": list(color),
                "shared_material_name": str(material.name),
                "material_names": material_names,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "pose_count": len(poses),
            }
        )

    if not frame_indices:
        raise ValueError("no dynamic object poses were rendered")

    if not math.isfinite(content_max_y):
        raise ValueError("could not determine SWR mesh depth bounds")
    if calibrated_friction_collision:
        back_wall_y = max(6.0, content_max_y + 0.8)
    elif calibrated_mass_collision:
        back_wall_y = max(8.5, content_max_y + 0.8)
    elif calibrated_bouncy_platform:
        back_wall_y = max(6.0, content_max_y + 1.0)
    else:
        back_wall_y = max(15.0, content_max_y + 1.0)

    if calibrated_friction_collision:
        static_objects = _add_friction_collision_environment(
            world_reconstruction,
            back_wall_y,
            appearance_profile,
        )
    elif calibrated_mass_collision:
        static_objects = _add_mass_collision_environment(
            world_reconstruction,
            back_wall_y,
            appearance_profile,
        )
    elif calibrated_bouncy_platform:
        static_objects = _add_bouncy_platform_environment(
            world_reconstruction,
            back_wall_y,
            appearance_profile,
        )
    else:
        static_objects = _add_room_environment(
            world_reconstruction,
            back_wall_y,
            appearance_profile,
        )
    calibrated_special_scene = (
        calibrated_bouncy_wall
        or calibrated_bouncy_platform
        or calibrated_friction_collision
        or calibrated_mass_collision
    )
    lights.extend(
        [
            _point_light_at(
                "AREA",
                "refined_wall_wash_light",
                (
                    (
                        -4.5
                        if calibrated_special_scene
                        else 4.5
                    ),
                    5.0,
                    7.8,
                ),
                (0.0, back_wall_y, 2.5),
                energy=(
                    1100.0
                    if calibrated_special_scene
                    else 1750.0
                ),
                size=8.5,
                color=(1.0, 0.95, 0.90),
            ),
            _point_light_at(
                "AREA",
                "refined_ceiling_fill_light",
                (0.0, 8.0, 2.0),
                (-1.0, 8.0, 6.10),
                energy=(
                    450.0
                    if calibrated_special_scene
                    else 260.0
                ),
                size=8.0,
                color=(0.78, 0.87, 1.0),
            ),
        ]
    )
    source_camera_payload = base._setup_source_camera(
        world_reconstruction_path,
        world_reconstruction,
        source_width,
        source_height,
    )
    source_camera = bpy.data.objects["world_reconstruction_source_camera"]
    scene.camera = source_camera
    if calibrated_bouncy_platform:
        static_objects.append(
            _add_bouncy_platform_camera_backdrop(
                scene,
                source_camera,
                appearance_profile,
            )
        )
    if calibrated_mass_collision:
        static_objects.append(
            _add_mass_collision_camera_wall(
                scene,
                source_camera,
            )
        )
    curtain = _add_bouncy_wall_curtain(
        scene,
        source_camera,
        appearance_profile,
    )
    if curtain is not None:
        static_objects.append(curtain)
    scene.render.resolution_x = int(source_width * resolution_scale)
    scene.render.resolution_y = int(source_height * resolution_scale)
    scene.frame_start = min(frame_indices)
    scene.frame_end = max(frame_indices)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_json.with_suffix(".blend")))

    preview_path = output_video.with_name(
        f"refined_preview_frame_{preview_frame:05d}.png"
        if preview_frame is not None
        else "refined_preview.png"
    )
    if preview_frame is not None:
        if preview_frame < scene.frame_start or preview_frame > scene.frame_end:
            raise ValueError(
                f"preview frame {preview_frame} is outside "
                f"[{scene.frame_start}, {scene.frame_end}]"
            )
        scene.frame_set(preview_frame)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.filepath = str(preview_path)
        bpy.ops.render.render(write_still=True)
        rendered_path = preview_path
        render_mode = "preview_frame"
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.filepath = str(output_video)
        bpy.ops.render.render(animation=True)
        rendered_path = output_video
        render_mode = "animation"

    payload = {
        "tool": "world_reconstruction_refined_debug_render",
        "status": "ok",
        "render_profile": RENDER_PROFILE,
        "render_mode": render_mode,
        "world_reconstruction": str(world_reconstruction_path),
        "support_reference": str(support_reference_path),
        "rendered_path": str(rendered_path),
        "video_path": str(output_video) if preview_frame is None else None,
        "source_camera_video_path": (
            str(output_video) if preview_frame is None else None
        ),
        "preview_path": str(preview_path) if preview_frame is not None else None,
        "blend_path": str(output_json.with_suffix(".blend")),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "fps": fps,
        "source_resolution": {"width": source_width, "height": source_height},
        "render_settings": render_settings,
        "source_camera": source_camera_payload,
        "lights": lights,
        "appearance_calibration": appearance_profile,
        "dynamic_objects": rendered_objects,
        "static_objects": static_objects,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
