from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OPENCV_CAMERA_TO_BLENDER_WORLD = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

OPENCV_SOURCE_CAMERA_TO_BLENDER_WORLD = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

DEBUG_COLORS = [
    (0.95, 0.18, 0.14, 1.0),
    (0.15, 0.45, 1.0, 1.0),
    (0.12, 0.8, 0.32, 1.0),
    (1.0, 0.75, 0.08, 1.0),
    (0.95, 0.25, 0.85, 1.0),
    (0.0, 0.85, 0.9, 1.0),
]

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_root(world_reconstruction_path: Path, world_reconstruction: dict[str, Any] | None = None) -> Path:
    if isinstance(world_reconstruction, dict):
        raw = world_reconstruction.get("question_dir")
        if isinstance(raw, str) and raw.strip():
            root = Path(raw)
            if (root / "object-identification-and-planning" / "object_plan" / "object_plan.json").exists():
                return root
            sibling_world_modeling = root.parent / "world-modeling"
            if (
                root.name.startswith("question_")
                and (sibling_world_modeling / "object-identification-and-planning" / "object_plan" / "object_plan.json").exists()
            ):
                return sibling_world_modeling
    for parent in world_reconstruction_path.parents:
        if (parent / "object-identification-and-planning" / "object_plan" / "object_plan.json").exists():
            return parent
        sibling_world_modeling = parent.parent / "world-modeling"
        if (
            parent.name.startswith("question_")
            and (sibling_world_modeling / "object-identification-and-planning" / "object_plan" / "object_plan.json").exists()
        ):
            return sibling_world_modeling
    return world_reconstruction_path.parents[2]


def _object_plan_path(world_reconstruction_path: Path, world_reconstruction: dict[str, Any] | None = None) -> Path:
    return _artifact_root(world_reconstruction_path, world_reconstruction) / "object-identification-and-planning" / "object_plan" / "object_plan.json"


def _object_materials_by_id(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any] | None = None,
) -> dict[str, str]:
    object_plan_path = _object_plan_path(world_reconstruction_path, world_reconstruction)
    if not object_plan_path.exists():
        return {}
    object_plan = _load_json(object_plan_path)
    materials = {}
    for item in object_plan.get("target_objects", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        appearance = item.get("appearance") if isinstance(item.get("appearance"), dict) else {}
        material = str(appearance.get("material") or "").strip().lower()
        if object_id and material in {"metal", "rubber"}:
            materials[object_id] = material
    return materials


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(float(a[row][k]) * float(b[k][col]) for k in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def _opencv_pose_to_blender_world(pose: list[list[float]]) -> list[list[float]]:
    return _matmul(OPENCV_CAMERA_TO_BLENDER_WORLD, pose)


def _opencv_point_to_blender_world(point: list[float]) -> list[float]:
    x, y, z = [float(value) for value in point]
    return [x, z, -y]


def _opencv_vector_to_blender_world(vector: list[float]) -> list[float]:
    return _opencv_point_to_blender_world(vector)


def _trajectory_objects(world_reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory_correction = world_reconstruction.get("trajectory_correction")
    if not isinstance(trajectory_correction, dict):
        raise ValueError("world_reconstruction.trajectory_correction is required for debug rendering")
    corrected = trajectory_correction.get("corrected_trajectories")
    if not isinstance(corrected, dict) or corrected.get("applied") is not True:
        raise ValueError("world_reconstruction.trajectory_correction.corrected_trajectories is required for debug rendering")
    corrected_objects = corrected.get("objects")
    if not isinstance(corrected_objects, list) or not corrected_objects:
        raise ValueError("world_reconstruction.trajectory_correction.corrected_trajectories.objects is empty")

    support_position = world_reconstruction.get("support_plane_position_correction")
    ground_by_object_id: dict[str, dict[str, Any]] = {}
    if isinstance(support_position, dict) and support_position.get("applied") is True:
        ground_objects = support_position.get("objects")
        if isinstance(ground_objects, list):
            ground_by_object_id = {
                str(item.get("object_id")): item
                for item in ground_objects
                if isinstance(item, dict) and item.get("object_id")
            }
    debug_mesh_paths = world_reconstruction.get("debug_mesh_paths")
    debug_mesh_paths = debug_mesh_paths if isinstance(debug_mesh_paths, dict) else {}

    objects = []
    for item in corrected_objects:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        object_id = str(item.get("object_id") or "")
        ground_item = ground_by_object_id.get(object_id)
        mesh_path = (ground_item or {}).get("mesh_path") or debug_mesh_paths.get(object_id)
        if not mesh_path:
            continue
        merged = dict(item)
        merged["mesh_path"] = mesh_path
        merged["render_pose_source"] = item.get("render_pose_source") or "trajectory_correction.corrected_trajectories"
        merged["mesh_source"] = item.get("mesh_source") or (
            "support_plane_position_correction" if ground_item else "foundationpose_poses.mesh_path"
        )
        objects.append(merged)
    if not objects:
        raise ValueError("no trajectory objects with a resolvable mesh path")
    return objects


def _import_mesh(mesh_path: str):
    import bpy

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    mesh_objects = [item for item in imported if item.type == "MESH"]
    if len(mesh_objects) == 1:
        obj = mesh_objects[0]
        _normalize_imported_gltf_mesh_coordinates(obj)
        return obj
    if not mesh_objects:
        raise ValueError(f"no mesh objects imported from {mesh_path}")
    bpy.ops.object.select_all(action="DESELECT")
    for mesh_obj in mesh_objects:
        mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    _normalize_imported_gltf_mesh_coordinates(obj)
    return obj


def _normalize_imported_gltf_mesh_coordinates(obj: Any) -> None:
    import bpy
    from mathutils import Matrix

    if obj.type != "MESH":
        return
    bpy.context.view_layer.update()
    transform = obj.matrix_world.copy()
    if transform != Matrix.Identity(4):
        obj.data.transform(transform)
        obj.matrix_world = Matrix.Identity(4)

    # Blender's glTF importer converts raw glTF local coordinates (x, y, z)
    # into Blender local coordinates (x, -z, y). World reconstruction poses are computed
    # against the raw GLB mesh coordinates, so convert the imported vertices back.
    gltf_blender_local_to_raw = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    if obj.get("physmind_raw_gltf_coordinates"):
        return
    obj.data.transform(gltf_blender_local_to_raw)
    obj["physmind_raw_gltf_coordinates"] = True


def _debug_material(name: str, color: tuple[float, float, float, float], material_type: str = "rubber"):
    import bpy

    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = color
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 1.0 if material_type == "metal" else 0.0
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.08 if material_type == "metal" else 0.82
    if material_type == "metal":
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 1.0
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.35
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.12
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _assign_debug_material(obj: Any, material: Any) -> None:
    if obj.type == "MESH":
        obj.data.materials.clear()
        obj.data.materials.append(material)
    for child in getattr(obj, "children", []):
        _assign_debug_material(child, material)


def _first_color_attribute_name(obj: Any) -> str | None:
    if getattr(obj, "type", None) != "MESH":
        return None
    color_attributes = getattr(obj.data, "color_attributes", None)
    if color_attributes is not None and len(color_attributes) > 0:
        return str(color_attributes[0].name)
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) > 0:
        return str(vertex_colors[0].name)
    return None


def _vertex_color_material(name: str, attribute_name: str, material_type: str = "rubber"):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    color_node = nodes.new(type="ShaderNodeVertexColor")
    color_node.layer_name = attribute_name
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    if "Base Color" in bsdf.inputs:
        material.node_tree.links.new(color_node.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in bsdf.inputs:
        material.node_tree.links.new(color_node.outputs["Alpha"], bsdf.inputs["Alpha"])
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 1.0 if material_type == "metal" else 0.0
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.08 if material_type == "metal" else 0.82
    if material_type == "metal":
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 1.0
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.35
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.12
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _assign_vertex_color_materials(obj: Any, material_prefix: str, material_type: str = "rubber") -> bool:
    applied = False
    if getattr(obj, "type", None) == "MESH":
        attribute_name = _first_color_attribute_name(obj)
        if attribute_name:
            obj.data.materials.clear()
            obj.data.materials.append(
                _vertex_color_material(f"{material_prefix}_{obj.name}", attribute_name, material_type)
            )
            applied = True
    for child in getattr(obj, "children", []):
        applied = _assign_vertex_color_materials(child, material_prefix, material_type) or applied
    return applied


def _keyframe_matrix(obj: Any, matrix_world: Any, frame_index: int) -> None:
    obj.matrix_world = matrix_world
    obj.keyframe_insert(data_path="location", frame=frame_index)
    if obj.rotation_mode == "QUATERNION":
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)
    elif obj.rotation_mode == "AXIS_ANGLE":
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame_index)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_index)
    obj.keyframe_insert(data_path="scale", frame=frame_index)


def _set_visibility(obj: Any, frame_index: int, visible: bool) -> None:
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_index)
    obj.keyframe_insert(data_path="hide_render", frame=frame_index)


def _active_interval(item: dict[str, Any], poses: list[dict[str, Any]]) -> tuple[int, int]:
    activation = item.get("activation") if isinstance(item.get("activation"), dict) else {}
    first = activation.get("first_active_frame")
    last = activation.get("last_active_frame")
    if isinstance(first, int) and isinstance(last, int):
        return first, last
    return min(int(pose["frame_index"]) for pose in poses), max(int(pose["frame_index"]) for pose in poses)


def _add_ground_plane(
    world_reconstruction: dict[str, Any],
    size: float = 100.0,
) -> dict[str, Any] | None:
    import bpy
    from mathutils import Vector

    analytic_plane = world_reconstruction.get("analytic_support_plane")
    if isinstance(analytic_plane, dict):
        origin = analytic_plane.get("origin")
        normal = analytic_plane.get("normal")
        if isinstance(origin, list) and len(origin) == 3 and isinstance(normal, list) and len(normal) == 3:
            normal_blender = Vector([float(value) for value in normal]).normalized()
            bpy.ops.mesh.primitive_plane_add(size=size, location=[float(value) for value in origin])
            plane = bpy.context.object
            plane.name = "world_reconstruction_ground_plane"
            plane.rotation_euler = normal_blender.to_track_quat("Z", "Y").to_euler()
            ground_color = (0.48, 0.50, 0.52, 1.0)
            material_type = "rubber"
            material = _debug_material("world_reconstruction_ground_material", ground_color, material_type)
            plane.data.materials.append(material)
            return {
                "object_id": "ground_plane",
                "geometry_type": "analytic_support_plane",
                "analytic_support_plane": analytic_plane,
                "material_color": list(ground_color),
                "material_type": material_type,
            }

    ground_contact = world_reconstruction.get("support_plane_position_correction")
    if not isinstance(ground_contact, dict) or ground_contact.get("applied") is not True:
        return None
    normal_camera = ground_contact.get("normal_camera")
    ground_height = ground_contact.get("ground_height_along_normal")
    if not isinstance(normal_camera, list) or len(normal_camera) != 3 or not isinstance(ground_height, (int, float)):
        return None

    normal_blender = Vector(_opencv_vector_to_blender_world(normal_camera)).normalized()
    location_blender = _opencv_point_to_blender_world([float(ground_height) * float(v) for v in normal_camera])
    bpy.ops.mesh.primitive_plane_add(size=size, location=location_blender)
    plane = bpy.context.object
    plane.name = "world_reconstruction_ground_plane"
    plane.rotation_euler = normal_blender.to_track_quat("Z", "Y").to_euler()
    ground_color = (0.48, 0.50, 0.52, 1.0)
    material_type = "rubber"
    material = _debug_material("world_reconstruction_ground_material", ground_color, material_type)
    plane.data.materials.append(material)
    return {
        "object_id": "ground_plane",
        "geometry_type": "plane",
        "normal_camera": normal_camera,
        "ground_height_along_normal": float(ground_height),
        "size": float(size),
        "material_color": list(ground_color),
        "material_type": material_type,
    }

def _setup_lighting() -> None:
    import bpy

    world = bpy.context.scene.world or bpy.data.worlds.new("world_reconstruction_debug_world")
    bpy.context.scene.world = world
    world.color = (0.55, 0.57, 0.60)
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background = nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.55, 0.57, 0.60, 1.0)
        background.inputs["Strength"].default_value = 0.45
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 8.0))
    light = bpy.context.object
    light.name = "world_reconstruction_debug_key_light"
    light.data.energy = 350.0
    light.data.size = 6.0


def _setup_cycles_cpu() -> dict[str, Any]:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    return {
        "engine": str(scene.render.engine),
        "cycles_device": str(scene.cycles.device),
        "samples": int(scene.cycles.samples),
    }


def _setup_source_camera(
    world_reconstruction_path: Path,
    world_reconstruction: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    depth_path = _artifact_root(world_reconstruction_path, world_reconstruction) / "metric-mesh-reconstruction" / "video_metric_depth" / "video_metric_depth.json"
    depth_payload = _load_json(depth_path)
    K = depth_payload.get("fixed_intrinsics")
    if (
        not isinstance(K, list)
        or len(K) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in K)
    ):
        raise ValueError(f"Video-depth fixed_intrinsics is invalid: {depth_path}")

    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Video-depth camera focal length must be positive: {depth_path}")

    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.name = "world_reconstruction_source_camera"
    camera.matrix_world = Matrix(OPENCV_SOURCE_CAMERA_TO_BLENDER_WORLD)
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 32.0
    camera.data.lens = fx * camera.data.sensor_width / float(width)
    camera.data.shift_x = (float(cx) - float(width) * 0.5) / float(width)
    camera.data.shift_y = (float(height) * 0.5 - float(cy)) / float(width)
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    return {
        "mode": "source_video_camera",
        "intrinsics_source": str(depth_path),
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "sensor_width": float(camera.data.sensor_width),
        "lens": float(camera.data.lens),
        "shift_x": float(camera.data.shift_x),
        "shift_y": float(camera.data.shift_y),
        "coordinate_frame": "opencv_camera",
        "blender_matrix_world": OPENCV_SOURCE_CAMERA_TO_BLENDER_WORLD,
    }


def _render_animation(scene: Any, camera: Any, output_video: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(output_video)
    bpy = sys.modules["bpy"]
    bpy.ops.render.render(animation=True)


def _video_metadata(world_reconstruction: dict[str, Any]) -> tuple[int, int, int]:
    metadata = (
        world_reconstruction.get("video_metadata")
        if isinstance(world_reconstruction.get("video_metadata"), dict)
        else {}
    )
    if not metadata and world_reconstruction.get("question_dir"):
        depth_path = _artifact_root(Path(str(world_reconstruction["question_dir"])), world_reconstruction) / "metric-mesh-reconstruction" / "video_metric_depth" / "video_metric_depth.json"
        if depth_path.exists():
            depth_payload = _load_json(depth_path)
            if isinstance(depth_payload.get("video_metadata"), dict):
                metadata = depth_payload["video_metadata"]
    width = int(metadata.get("width") or 640)
    height = int(metadata.get("height") or 360)
    fps = int(round(float(metadata.get("fps") or 24)))
    return width, height, max(1, fps)


def _run_inside_blender(world_reconstruction_path: Path, output_video: Path, output_json: Path) -> None:
    import bpy
    from mathutils import Matrix

    world_reconstruction = _load_json(world_reconstruction_path)
    width, height, fps = _video_metadata(world_reconstruction)
    objects = _trajectory_objects(world_reconstruction)
    material_by_object_id = _object_materials_by_id(world_reconstruction_path, world_reconstruction)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    render_settings = _setup_cycles_cpu()
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.film_transparent = False
    _setup_lighting()
    rendered_objects = []
    frame_indices = []
    for object_index, item in enumerate(objects):
        object_id = str(item.get("object_id") or "")
        mesh_path = item.get("mesh_path")
        poses = [pose for pose in item.get("poses", []) if isinstance(pose, dict) and pose.get("corrected_pose_4x4")]
        if not object_id or not mesh_path or not Path(str(mesh_path)).exists() or not poses:
            continue
        obj = _import_mesh(str(mesh_path))
        obj.name = object_id
        fallback_color = DEBUG_COLORS[object_index % len(DEBUG_COLORS)]
        material_type = material_by_object_id.get(object_id, "rubber")
        if _assign_vertex_color_materials(obj, f"world_reconstruction_vertex_color_{object_id}", material_type):
            material_source = "mesh_vertex_color"
        else:
            _assign_debug_material(
                obj,
                _debug_material(f"world_reconstruction_debug_{object_id}", fallback_color, material_type),
            )
            material_source = "debug_fallback"
        first_frame, last_frame = _active_interval(item, poses)
        _set_visibility(obj, max(0, first_frame - 1), False)
        _set_visibility(obj, first_frame, True)
        for pose in poses:
            frame_index = int(pose["frame_index"])
            matrix = Matrix(_opencv_pose_to_blender_world(pose["corrected_pose_4x4"]))
            _keyframe_matrix(obj, matrix, frame_index)
            frame_indices.append(frame_index)
        _set_visibility(obj, last_frame + 1, False)
        rendered_objects.append(
            {
                "object_id": object_id,
                "mesh_path": str(mesh_path),
                "debug_color": list(fallback_color),
                "material_color": list(fallback_color) if material_source == "debug_fallback" else None,
                "material_type": material_type,
                "material_source": material_source,
                "activation": item.get("activation") if isinstance(item.get("activation"), dict) else {},
                "render_pose_source": item.get("render_pose_source"),
                "mesh_source": item.get("mesh_source"),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "pose_count": len(poses),
            }
        )

    if not frame_indices:
        raise ValueError("no dynamic object poses were rendered")
    static_objects = []
    ground_plane = _add_ground_plane(world_reconstruction)
    if ground_plane is not None:
        static_objects.append(ground_plane)
    source_camera_payload = _setup_source_camera(world_reconstruction_path, world_reconstruction, width, height)
    source_camera = bpy.data.objects["world_reconstruction_source_camera"]
    scene.frame_start = min(frame_indices)
    scene.frame_end = max(frame_indices)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_json.with_suffix(".blend")))
    source_video = output_video.with_name("world_reconstruction_debug_camera.mp4")
    _render_animation(scene, source_camera, source_video)

    payload = {
        "tool": "world_reconstruction_debug_render",
        "status": "ok",
        "video_backend": "blender_native_source_camera_render",
        "world_reconstruction": str(world_reconstruction_path),
        "object_plan": str(_object_plan_path(world_reconstruction_path, world_reconstruction)),
        "video_path": str(source_video),
        "source_camera_video_path": str(source_video),
        "blend_path": str(output_json.with_suffix(".blend")),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "render_settings": render_settings,
        "source_camera": source_camera_payload,
        "dynamic_objects": rendered_objects,
        "static_objects": static_objects,
        "coordinate_transform": {
            "source": "opencv_camera",
            "target": "blender_world",
            "matrix_4x4": OPENCV_CAMERA_TO_BLENDER_WORLD,
            "mapping": "x_blender=x_camera, y_blender=z_camera, z_blender=-y_camera",
        },
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
