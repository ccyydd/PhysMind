from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.world_model import render_world_reconstruction_debug as standard
from scripts.world_model import render_world_reconstruction_debug_refined as physionpp


RENDER_PROFILE = "clevrer_refined_v10"
PHYSIONPP_SCENARIOS = frozenset(
    {
        "friction_platform_pp",
        "bouncy_wall_pp",
        "bouncy_platform_pp",
        "friction_collision_pp",
        "mass_collision_pp",
    }
)
VIDEO_CONSTANT_RATE_FACTOR = "LOSSLESS"
OFFICIAL_CLEVR_COLORS = {
    "gray": (87 / 255.0, 87 / 255.0, 87 / 255.0, 1.0),
    "red": (173 / 255.0, 35 / 255.0, 35 / 255.0, 1.0),
    "blue": (42 / 255.0, 75 / 255.0, 215 / 255.0, 1.0),
    "green": (29 / 255.0, 105 / 255.0, 20 / 255.0, 1.0),
    "brown": (129 / 255.0, 74 / 255.0, 25 / 255.0, 1.0),
    "purple": (129 / 255.0, 38 / 255.0, 192 / 255.0, 1.0),
    "cyan": (41 / 255.0, 208 / 255.0, 208 / 255.0, 1.0),
    "yellow": (255 / 255.0, 238 / 255.0, 51 / 255.0, 1.0),
}
OFFICIAL_CLEVR_CAMERA_DEPTH = 11.263723373413086
OFFICIAL_CLEVR_LIGHTS = {
    "key": {
        "camera_local_position": (2.0391223430633545, 0.5365733504295349, -3.2832388877868652),
        "energy": 78.5398178100586,
        "size": 0.5,
        "color": (1.0, 0.9323092103004456, 0.8166635632514954),
    },
    "fill": {
        "camera_local_position": (-6.093931198120117, 3.0730857849121094, -10.618757247924805),
        "energy": 23.56194496154785,
        "size": 0.5,
        "color": (0.7616661787033081, 0.8177196383476257, 1.0),
    },
    "back": {
        "camera_local_position": (1.229499101638794, 6.361733436584473, -10.80949878692627),
        "energy": 39.2699089050293,
        "size": 1.0,
        "color": (1.0, 1.0, 1.0),
    },
}
OFFICIAL_CLEVR_SUN_DIRECTION_CAMERA = (
    -0.12522681057453156,
    -0.5736424326896667,
    -0.8094767928123474,
)


def _input_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_frame_indices(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if any(value < 0 for value in values):
        raise ValueError("frame indices must be non-negative")
    return values


def _style_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile": RENDER_PROFILE,
        "samples": int(args.samples),
        "exposure": float(args.exposure),
        "ground_value": float(args.ground_value),
        "world_strength": float(args.world_strength),
        "light_energy_scale": float(args.light_energy_scale),
        "light_size_scale": float(args.light_size_scale),
        "light_distance_scale": float(args.light_distance_scale),
        "light_rig_mirror_x": str(args.light_rig_mirror_x),
        "key_light_scale": float(args.key_light_scale),
        "fill_light_scale": float(args.fill_light_scale),
        "back_light_scale": float(args.back_light_scale),
        "metal_diffuse_weight": float(args.metal_diffuse_weight),
        "metal_bevel_ratio": float(args.metal_bevel_ratio),
        "color_source": str(args.color_source),
        "video_constant_rate_factor": VIDEO_CONSTANT_RATE_FACTOR,
    }


def _style_sha256(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_render_scope(world_reconstruction_path: Path) -> tuple[str, str | None]:
    world_reconstruction = standard._load_json(world_reconstruction_path)
    object_plan_path = standard._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.is_file():
        raise FileNotFoundError(
            f"latest debug render requires an object plan: {object_plan_path}"
        )
    object_plan = standard._load_json(object_plan_path)
    special_scene = object_plan.get("special_scene")
    if not isinstance(special_scene, dict):
        special_scene = {}
    scene_metadata = special_scene.get("scene_metadata")
    if not isinstance(scene_metadata, dict):
        scene_metadata = {}
    scenario = str(scene_metadata.get("scenario") or "").strip()
    benchmark = str(
        special_scene.get("benchmark")
        or scene_metadata.get("benchmark")
        or world_reconstruction.get("benchmark")
        or ""
    ).strip()
    if scenario in PHYSIONPP_SCENARIOS:
        return "physion_pp", scenario
    if benchmark == "clevrer":
        return "clevrer", None
    raise ValueError(
        "latest debug render supports only CLEVRER and the five Physion++ main "
        f"scenarios, got benchmark={benchmark!r} scenario={scenario!r}"
    )


def _color_attribute_node(nodes: Any, obj: Any) -> Any | None:
    attribute_name = standard._first_color_attribute_name(obj)
    if not attribute_name:
        return None
    node = nodes.new(type="ShaderNodeVertexColor")
    node.layer_name = attribute_name
    return node


def _fallback_color_node(nodes: Any, color: tuple[float, float, float, float]) -> Any:
    node = nodes.new(type="ShaderNodeRGB")
    node.outputs["Color"].default_value = color
    return node


def _clevr_rubber_material(
    name: str,
    obj: Any,
    fallback_color: tuple[float, float, float, float],
    *,
    use_vertex_color: bool,
    fallback_source: str,
) -> tuple[Any, str]:
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    color_node = _color_attribute_node(nodes, obj) if use_vertex_color else None
    material_source = "mesh_vertex_color"
    if color_node is None:
        color_node = _fallback_color_node(nodes, fallback_color)
        material_source = fallback_source

    rubber = nodes.new(type="ShaderNodeBsdfPrincipled")
    rubber.inputs["Roughness"].default_value = 0.58
    if "Specular IOR Level" in rubber.inputs:
        rubber.inputs["Specular IOR Level"].default_value = 0.28
    if "Sheen Weight" in rubber.inputs:
        rubber.inputs["Sheen Weight"].default_value = 0.08
    output = nodes.new(type="ShaderNodeOutputMaterial")

    links.new(color_node.outputs["Color"], rubber.inputs["Base Color"])
    links.new(rubber.outputs["BSDF"], output.inputs["Surface"])
    return material, material_source


def _clevr_metal_material(
    name: str,
    obj: Any,
    fallback_color: tuple[float, float, float, float],
    *,
    use_vertex_color: bool,
    fallback_source: str,
    diffuse_weight: float,
    bevel_ratio: float,
) -> tuple[Any, str]:
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    color_node = _color_attribute_node(nodes, obj) if use_vertex_color else None
    material_source = "mesh_vertex_color"
    if color_node is None:
        color_node = _fallback_color_node(nodes, fallback_color)
        material_source = fallback_source

    broad = nodes.new(type="ShaderNodeBsdfAnisotropic")
    broad.inputs["Roughness"].default_value = math.sqrt(0.2)
    broad.inputs["Anisotropy"].default_value = 0.25
    sharp = nodes.new(type="ShaderNodeBsdfAnisotropic")
    sharp.inputs["Roughness"].default_value = 0.1
    sharp.inputs["Anisotropy"].default_value = 0.25
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.6
    diffuse_color = nodes.new(type="ShaderNodeMixRGB")
    diffuse_color.blend_type = "MULTIPLY"
    diffuse_color.inputs["Fac"].default_value = 1.0
    diffuse_color.inputs[2].default_value = (
        diffuse_weight,
        diffuse_weight,
        diffuse_weight,
        1.0,
    )
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Roughness"].default_value = 0.0
    bevel = nodes.new(type="ShaderNodeBevel")
    bevel.samples = 4
    bevel.inputs["Radius"].default_value = max(
        min(float(value) for value in obj.dimensions) * bevel_ratio,
        0.0,
    )
    finish_add = nodes.new(type="ShaderNodeAddShader")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    links.new(color_node.outputs["Color"], broad.inputs["Color"])
    links.new(color_node.outputs["Color"], sharp.inputs["Color"])
    links.new(color_node.outputs["Color"], diffuse_color.inputs[1])
    links.new(diffuse_color.outputs["Color"], diffuse.inputs["Color"])
    links.new(bevel.outputs["Normal"], broad.inputs["Normal"])
    links.new(bevel.outputs["Normal"], sharp.inputs["Normal"])
    links.new(bevel.outputs["Normal"], diffuse.inputs["Normal"])
    links.new(broad.outputs["BSDF"], mix.inputs[1])
    links.new(sharp.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], finish_add.inputs[0])
    links.new(diffuse.outputs["BSDF"], finish_add.inputs[1])
    links.new(finish_add.outputs["Shader"], output.inputs["Surface"])
    return material, material_source


def _assign_clevr_material(
    obj: Any,
    *,
    object_id: str,
    material_type: str,
    fallback_color: tuple[float, float, float, float],
    use_vertex_color: bool,
    fallback_source: str,
    metal_diffuse_weight: float,
    metal_bevel_ratio: float,
) -> str:
    if material_type == "metal":
        material, material_source = _clevr_metal_material(
            f"clevrer_refined_metal_{object_id}",
            obj,
            fallback_color,
            use_vertex_color=use_vertex_color,
            fallback_source=fallback_source,
            diffuse_weight=metal_diffuse_weight,
            bevel_ratio=metal_bevel_ratio,
        )
    else:
        material, material_source = _clevr_rubber_material(
            f"clevrer_refined_rubber_{object_id}",
            obj,
            fallback_color,
            use_vertex_color=use_vertex_color,
            fallback_source=fallback_source,
        )
    standard._assign_debug_material(obj, material)
    return material_source


def _ground_material(value: float) -> Any:
    import bpy

    material = bpy.data.materials.new("clevrer_refined_ground_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (value, value, value, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def _add_ground_plane(
    world_reconstruction: dict[str, Any],
    *,
    ground_value: float,
    size: float = 100.0,
) -> tuple[dict[str, Any], Any]:
    import bpy
    from mathutils import Vector

    support_plane = world_reconstruction.get("vrdp_support_plane")
    if not isinstance(support_plane, dict):
        raise ValueError("world_reconstruction.vrdp_support_plane is required for refined rendering")
    origin = support_plane.get("origin")
    normal = support_plane.get("normal")
    if not (
        isinstance(origin, list)
        and len(origin) == 3
        and isinstance(normal, list)
        and len(normal) == 3
    ):
        raise ValueError("world_reconstruction.vrdp_support_plane origin/normal is invalid")

    normal_blender = Vector([float(value) for value in normal]).normalized()
    bpy.ops.mesh.primitive_plane_add(size=size, location=[float(value) for value in origin])
    plane = bpy.context.object
    plane.name = "clevrer_refined_ground_plane"
    plane.rotation_euler = normal_blender.to_track_quat("Z", "Y").to_euler()
    plane.data.materials.append(_ground_material(ground_value))
    return (
        {
            "object_id": "ground_plane",
            "geometry_type": "vrdp_support_plane",
            "vrdp_support_plane": support_plane,
            "material": "official_clevr_default_diffuse_approximation",
            "ground_value": float(ground_value),
            "size": float(size),
        },
        normal_blender,
    )


def _camera_local_to_world(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    return float(x), float(-z), float(y)


def _setup_clevr_lighting(
    *,
    target_depth: float,
    target_center: Any,
    ground_normal: Any,
    world_strength: float,
    energy_scale: float,
    size_scale: float,
    distance_scale: float,
    mirror_x: str,
    key_light_scale: float,
    fill_light_scale: float,
    back_light_scale: float,
) -> dict[str, Any]:
    import bpy
    from mathutils import Vector

    world = bpy.context.scene.world or bpy.data.worlds.new("clevrer_refined_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs["Strength"].default_value = float(world_strength)

    spatial_scale = max(float(target_depth) / OFFICIAL_CLEVR_CAMERA_DEPTH, 1e-4)
    role_energy_scales = {
        "key": key_light_scale,
        "fill": fill_light_scale,
        "back": back_light_scale,
    }
    light_records = []
    for light_name, definition in OFFICIAL_CLEVR_LIGHTS.items():
        local_position = tuple(
            float(value) * spatial_scale for value in definition["camera_local_position"]
        )
        reference_position = Vector(_camera_local_to_world(local_position))
        world_position = target_center + (reference_position - target_center) * distance_scale
        if mirror_x in {"area", "all"}:
            world_position.x = target_center.x - (world_position.x - target_center.x)
        bpy.ops.object.light_add(type="AREA", location=world_position)
        light = bpy.context.object
        light.name = f"clevrer_refined_{light_name}_light"
        light.data.energy = (
            float(definition["energy"])
            * spatial_scale
            * spatial_scale
            * distance_scale
            * distance_scale
            * energy_scale
            * role_energy_scales[light_name]
        )
        light.data.size = (
            float(definition["size"])
            * spatial_scale
            * distance_scale
            * size_scale
        )
        light.data.color = tuple(float(value) for value in definition["color"])
        light.rotation_euler = (-Vector(ground_normal)).to_track_quat("-Z", "Y").to_euler()
        light_records.append(
            {
                "name": light_name,
                "type": "AREA",
                "location": list(world_position),
                "energy": float(light.data.energy),
                "size": float(light.data.size),
                "color": list(light.data.color),
            }
        )

    sun_direction = Vector(_camera_local_to_world(OFFICIAL_CLEVR_SUN_DIRECTION_CAMERA)).normalized()
    if mirror_x == "all":
        sun_direction.x = -sun_direction.x
        sun_direction.normalize()
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    sun = bpy.context.object
    sun.name = "clevrer_refined_sun"
    sun.data.energy = 0.45 * energy_scale
    sun.rotation_euler = sun_direction.to_track_quat("-Z", "Y").to_euler()
    light_records.append(
        {
            "name": "sun",
            "type": "SUN",
            "direction": list(sun_direction),
            "energy": float(sun.data.energy),
            "color": list(sun.data.color),
        }
    )
    return {
        "reference": "facebookresearch/clevr-dataset-gen base_scene.blend",
        "spatial_scale": float(spatial_scale),
        "target_depth": float(target_depth),
        "target_center": list(target_center),
        "distance_scale": float(distance_scale),
        "mirror_x": mirror_x,
        "role_energy_scales": role_energy_scales,
        "world_strength": float(world_strength),
        "lights": light_records,
    }


def _setup_render_settings(
    scene: Any,
    *,
    width: int,
    height: int,
    fps: int,
    samples: int,
    exposure: float,
) -> dict[str, Any]:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(samples)
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_denoising = True
    scene.render.fps = int(fps)
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(exposure)
    scene.view_settings.gamma = 1.0
    scene.render.film_transparent = False
    return {
        "engine": str(scene.render.engine),
        "cycles_device": str(scene.cycles.device),
        "samples": int(scene.cycles.samples),
        "max_bounces": int(scene.cycles.max_bounces),
        "denoising": bool(scene.cycles.use_denoising),
        "view_transform": str(scene.view_settings.view_transform),
        "look": str(scene.view_settings.look),
        "exposure": float(scene.view_settings.exposure),
        "gamma": float(scene.view_settings.gamma),
    }


def _object_appearances_by_id(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
) -> dict[str, dict[str, str]]:
    object_plan_path = standard._object_plan_path(
        world_reconstruction_path,
        world_reconstruction,
    )
    if not object_plan_path.exists():
        return {}
    object_plan = standard._load_json(object_plan_path)
    appearances = {}
    for item in object_plan.get("target_objects", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        appearance = item.get("appearance") if isinstance(item.get("appearance"), dict) else {}
        color = str(appearance.get("color") or "").strip().lower()
        material = str(appearance.get("material") or "").strip().lower()
        if object_id:
            appearances[object_id] = {"color": color, "material": material}
    return appearances


def _build_scene(
    *,
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
    style_settings: dict[str, Any],
) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    width, height, fps = standard._video_metadata(world_reconstruction)
    objects = standard._trajectory_objects(world_reconstruction)
    appearances_by_object_id = _object_appearances_by_id(
        world_reconstruction_path,
        world_reconstruction,
    )

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    render_settings = _setup_render_settings(
        scene,
        width=width,
        height=height,
        fps=fps,
        samples=int(style_settings["samples"]),
        exposure=float(style_settings["exposure"]),
    )

    rendered_objects = []
    frame_indices = []
    camera_points = []
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
        obj = standard._import_mesh(str(mesh_path))
        obj.name = object_id
        appearance = appearances_by_object_id.get(object_id, {})
        color_name = appearance.get("color", "")
        palette_color = OFFICIAL_CLEVR_COLORS.get(color_name)
        fallback_color = palette_color or standard.DEBUG_COLORS[
            object_index % len(standard.DEBUG_COLORS)
        ]
        material_type = appearance.get("material") or "rubber"
        use_vertex_color = style_settings["color_source"] == "mesh_vertex_color"
        fallback_source = (
            f"official_clevr_palette:{color_name}"
            if palette_color is not None
            else "debug_fallback"
        )
        material_source = _assign_clevr_material(
            obj,
            object_id=object_id,
            material_type=material_type,
            fallback_color=fallback_color,
            use_vertex_color=use_vertex_color,
            fallback_source=fallback_source,
            metal_diffuse_weight=float(style_settings["metal_diffuse_weight"]),
            metal_bevel_ratio=float(style_settings["metal_bevel_ratio"]),
        )
        first_frame, last_frame = standard._active_interval(item, poses)
        standard._set_visibility(obj, max(0, first_frame - 1), False)
        standard._set_visibility(obj, first_frame, True)
        for pose in poses:
            frame_index = int(pose["frame_index"])
            matrix = Matrix(standard._opencv_pose_to_blender_world(pose["corrected_pose_4x4"]))
            standard._keyframe_matrix(obj, matrix, frame_index)
            camera_points.append(
                [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
            )
            frame_indices.append(frame_index)
        standard._set_visibility(obj, last_frame + 1, False)
        rendered_objects.append(
            {
                "object_id": object_id,
                "mesh_path": str(mesh_path),
                "material_type": material_type,
                "material_source": material_source,
                "color_name": color_name or None,
                "activation": item.get("activation")
                if isinstance(item.get("activation"), dict)
                else {},
                "render_pose_source": item.get("render_pose_source"),
                "mesh_source": item.get("mesh_source"),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "pose_count": len(poses),
            }
        )

    if not frame_indices:
        raise ValueError("no dynamic object poses were rendered")
    ground_record, ground_normal = _add_ground_plane(
        world_reconstruction,
        ground_value=float(style_settings["ground_value"]),
    )
    source_camera_payload = standard._setup_source_camera(
        world_reconstruction_path,
        world_reconstruction,
        width,
        height,
    )
    source_camera = bpy.data.objects["world_reconstruction_source_camera"]
    source_camera_inverse = source_camera.matrix_world.inverted()
    target_center = sum(
        (Matrix.Translation(point).translation for point in camera_points),
        start=Matrix.Identity(4).translation,
    ) / len(camera_points)
    target_camera = source_camera_inverse @ target_center
    target_depth = abs(float(target_camera.z))
    lighting = _setup_clevr_lighting(
        target_depth=target_depth,
        target_center=target_center,
        ground_normal=ground_normal,
        world_strength=float(style_settings["world_strength"]),
        energy_scale=float(style_settings["light_energy_scale"]),
        size_scale=float(style_settings["light_size_scale"]),
        distance_scale=float(style_settings["light_distance_scale"]),
        mirror_x=str(style_settings["light_rig_mirror_x"]),
        key_light_scale=float(style_settings["key_light_scale"]),
        fill_light_scale=float(style_settings["fill_light_scale"]),
        back_light_scale=float(style_settings["back_light_scale"]),
    )

    scene.frame_start = min(frame_indices)
    scene.frame_end = max(frame_indices)
    return {
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "render_settings": render_settings,
        "source_camera": source_camera_payload,
        "dynamic_objects": rendered_objects,
        "static_objects": [ground_record],
        "lighting": lighting,
    }


def _render_still(scene: Any, camera: Any, frame_index: int, output_path: Path) -> None:
    import bpy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    scene.frame_set(int(frame_index))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def _render_video(scene: Any, camera: Any, output_path: Path) -> None:
    import bpy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = VIDEO_CONSTANT_RATE_FACTOR
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(animation=True)


def _load_reusable_scene(
    *,
    output_json: Path,
    input_sha256: str,
    style_sha256: str,
) -> dict[str, Any] | None:
    import bpy

    blend_path = output_json.with_suffix(".blend")
    if not output_json.exists() or not blend_path.exists():
        return None
    payload = standard._load_json(output_json)
    if (
        payload.get("status") != "ok"
        or payload.get("render_profile") != RENDER_PROFILE
        or payload.get("render_input_sha256") != input_sha256
        or payload.get("style_sha256") != style_sha256
    ):
        return None
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    return payload


def _run_inside_blender(
    *,
    world_reconstruction_path: Path,
    output_video: Path,
    output_json: Path,
    args: argparse.Namespace,
) -> None:
    import bpy

    style_settings = _style_settings(args)
    input_sha256 = _input_sha256(world_reconstruction_path)
    style_sha256 = _style_sha256(style_settings)
    reusable = None if args.force else _load_reusable_scene(
        output_json=output_json,
        input_sha256=input_sha256,
        style_sha256=style_sha256,
    )
    reused_blend = reusable is not None
    if reusable is None:
        world_reconstruction = standard._load_json(world_reconstruction_path)
        scene_payload = _build_scene(
            world_reconstruction_path=world_reconstruction_path,
            world_reconstruction=world_reconstruction,
            style_settings=style_settings,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(output_json.with_suffix(".blend")))
    else:
        scene_payload = {
            key: reusable[key]
            for key in (
                "resolution",
                "fps",
                "frame_start",
                "frame_end",
                "render_settings",
                "source_camera",
                "dynamic_objects",
                "static_objects",
                "lighting",
            )
            if key in reusable
        }

    scene = bpy.context.scene
    source_camera = bpy.data.objects["world_reconstruction_source_camera"]
    requested_frames = _parse_frame_indices(args.frame_indices)
    invalid_frames = [
        frame
        for frame in requested_frames
        if frame < int(scene.frame_start) or frame > int(scene.frame_end)
    ]
    if invalid_frames:
        raise ValueError(
            f"requested frames are outside [{scene.frame_start}, {scene.frame_end}]: {invalid_frames}"
        )

    rendered_keyframes: dict[str, list[str]] = {"camera": []}
    skipped_keyframes: dict[str, list[str]] = {"camera": []}
    rendered_videos: list[str] = []
    skipped_videos: list[str] = []
    source_video = output_video.with_name("world_reconstruction_debug_camera.mp4")
    if requested_frames:
        keyframe_dir = output_json.parent / "keyframes"
        for frame_index in requested_frames:
            output_path = keyframe_dir / f"camera_frame_{frame_index:04d}.png"
            if output_path.exists() and not args.force:
                skipped_keyframes["camera"].append(str(output_path))
                continue
            _render_still(scene, source_camera, frame_index, output_path)
            rendered_keyframes["camera"].append(str(output_path))
    else:
        if source_video.exists() and not args.force:
            skipped_videos.append(str(source_video))
        else:
            _render_video(scene, source_camera, source_video)
            rendered_videos.append(str(source_video))

    existing_payload = standard._load_json(output_json) if output_json.exists() else {}
    payload = {
        **existing_payload,
        "tool": "world_reconstruction_refined_debug_render",
        "status": "ok",
        "render_profile": RENDER_PROFILE,
        "world_reconstruction": str(world_reconstruction_path),
        "render_input_sha256": input_sha256,
        "style_sha256": style_sha256,
        "style_settings": style_settings,
        "blend_path": str(output_json.with_suffix(".blend")),
        "reused_blend": reused_blend,
        "output_mode": "keyframes" if requested_frames else "video",
        "requested_frame_indices": requested_frames,
        "rendered_keyframes": rendered_keyframes,
        "skipped_keyframes": skipped_keyframes,
        "source_camera_video_path": str(source_video),
        "rendered_videos": rendered_videos,
        "skipped_videos": skipped_videos,
        **scene_payload,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-reconstruction", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--frame-indices", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--exposure", type=float, default=0.25)
    parser.add_argument("--ground-value", type=float, default=1.0)
    parser.add_argument("--world-strength", type=float, default=0.03)
    parser.add_argument("--light-energy-scale", type=float, default=0.65)
    parser.add_argument("--light-size-scale", type=float, default=3.25)
    parser.add_argument("--light-distance-scale", type=float, default=2.0)
    parser.add_argument(
        "--light-rig-mirror-x",
        choices=["none", "area", "all"],
        default="area",
    )
    parser.add_argument("--key-light-scale", type=float, default=0.68)
    parser.add_argument("--fill-light-scale", type=float, default=1.4)
    parser.add_argument("--back-light-scale", type=float, default=1.36)
    parser.add_argument("--metal-diffuse-weight", type=float, default=0.2)
    parser.add_argument("--metal-bevel-ratio", type=float, default=0.02)
    parser.add_argument("--resolution-scale", type=int, default=4)
    parser.add_argument("--preview-frame", type=int)
    parser.add_argument(
        "--color-source",
        choices=["clevr_palette", "mesh_vertex_color"],
        default="clevr_palette",
    )
    parser.add_argument("--force", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not 0.0 <= args.ground_value <= 1.0:
        raise ValueError("--ground-value must be in [0, 1]")
    if args.world_strength < 0.0:
        raise ValueError("--world-strength must be non-negative")
    if (
        args.light_energy_scale <= 0.0
        or args.light_size_scale <= 0.0
        or args.light_distance_scale <= 0.0
        or args.key_light_scale <= 0.0
        or args.fill_light_scale <= 0.0
        or args.back_light_scale <= 0.0
    ):
        raise ValueError("light scales must be positive")
    if not 0.0 <= args.metal_diffuse_weight <= 1.0:
        raise ValueError("--metal-diffuse-weight must be in [0, 1]")
    if args.metal_bevel_ratio < 0.0:
        raise ValueError("--metal-bevel-ratio must be non-negative")
    world_reconstruction_path = Path(args.world_reconstruction)
    output_video = Path(args.output_video)
    output_json = Path(args.output_json)
    benchmark, _ = _artifact_render_scope(world_reconstruction_path)
    if benchmark == "physion_pp":
        if args.resolution_scale < 1:
            raise ValueError("--resolution-scale must be at least 1")
        physionpp._render(
            world_reconstruction_path,
            world_reconstruction_path,
            output_video,
            output_json,
            args.resolution_scale,
            args.preview_frame,
        )
        return
    _run_inside_blender(
        world_reconstruction_path=world_reconstruction_path,
        output_video=output_video,
        output_json=output_json,
        args=args,
    )


if __name__ == "__main__":
    main()
