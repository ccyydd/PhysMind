from __future__ import annotations

import json
import importlib.util
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATEST_WORLD_RECONSTRUCTION_RENDERER = (
    PROJECT_ROOT
    / "scripts"
    / "world_model"
    / "render_world_reconstruction_refined_debug.py"
)
_ANALYTIC_SYSID_PATH = PROJECT_ROOT / "scripts" / "world_model" / "run_impulse_analytic_sysid.py"
_ANALYTIC_SYSID_MODULE: Any | None = None


def _load_analytic_sysid_module() -> Any:
    global _ANALYTIC_SYSID_MODULE
    if _ANALYTIC_SYSID_MODULE is not None:
        return _ANALYTIC_SYSID_MODULE
    spec = importlib.util.spec_from_file_location(
        "physmind_run_impulse_analytic_sysid_for_qcpr_debug",
        _ANALYTIC_SYSID_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load analytic sysid helpers: {_ANALYTIC_SYSID_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ANALYTIC_SYSID_MODULE = module
    return module


def _safe_name(value: str) -> str:
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)]
    return "".join(safe).strip("_") or "track"


def _mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    height, width = image.shape[:2]
    if mask_bool.shape == (height, width):
        return mask_bool
    resized = cv2.resize(mask_bool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _draw_sam3_track_overlay(frame: np.ndarray, mask: np.ndarray, record: dict[str, Any], track_id: str) -> np.ndarray:
    mask_bool = _mask_for_image(mask, frame)
    overlay = frame.copy()
    color = np.array([0, 220, 255], dtype=np.float32)
    overlay[mask_bool] = (0.55 * overlay[mask_bool].astype(np.float32) + 0.45 * color).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    bbox = record.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    centroid = record.get("centroid_xy")
    if isinstance(centroid, list) and len(centroid) == 2:
        cv2.circle(overlay, (int(round(float(centroid[0]))), int(round(float(centroid[1])))), 4, (0, 0, 255), -1)
    label = f"{track_id} f={record.get('frame_index')} area={record.get('area')}"
    cv2.putText(overlay, label[:100], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


def write_sam3_video_track_overlay_videos(
    *,
    video: Path,
    records_by_object: dict[str, list[dict[str, Any]]],
    mask_sidecar: Path,
    output_dir: Path,
    fps: float,
) -> list[dict[str, Any]]:
    if not records_by_object or not mask_sidecar.exists():
        return []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for SAM3 video track debug overlays: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = max(1.0, float(fps) if fps and fps > 0 else float(capture.get(cv2.CAP_PROP_FPS) or 12.0))
    masks = np.load(mask_sidecar)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_videos = []
    try:
        for track_id, records in sorted(records_by_object.items()):
            valid_records = [
                record
                for record in sorted(records, key=lambda item: int(item.get("frame_index", -1)))
                if record.get("mask_key") in masks
            ]
            if not valid_records:
                continue
            output_path = output_dir / f"{_safe_name(track_id)}_overlay.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open SAM3 video track debug writer: {output_path}")
            written = 0
            for record in valid_records:
                frame_index = int(record["frame_index"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                writer.write(_draw_sam3_track_overlay(frame, masks[str(record["mask_key"])], record, track_id))
                written += 1
            writer.release()
            if written:
                debug_videos.append(
                    {
                        "track_id": track_id,
                        "path": str(output_path),
                        "frame_count": written,
                        "first_frame_index": int(valid_records[0]["frame_index"]),
                        "last_frame_index": int(valid_records[-1]["frame_index"]),
                    }
                )
            elif output_path.exists():
                output_path.unlink()
    finally:
        masks.close()
        capture.release()
    return debug_videos


def generate_sam3_video_track_debug_from_artifact(*, artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    video = Path(str(payload.get("video") or ""))
    mask_sidecar = Path(str(payload.get("mask_sidecar") or ""))
    records_by_object = payload.get("tracks_by_object")
    if not isinstance(records_by_object, dict):
        records_by_object = {}
        for record in payload.get("tracks") or []:
            if isinstance(record, dict):
                records_by_object.setdefault(str(record.get("object_id")), []).append(record)
    metadata = payload.get("video_metadata") if isinstance(payload.get("video_metadata"), dict) else {}
    debug_videos = write_sam3_video_track_overlay_videos(
        video=video,
        records_by_object=records_by_object,
        mask_sidecar=mask_sidecar,
        output_dir=artifact_path.parent / "debug" / "videos",
        fps=float(metadata.get("fps") or 0.0),
    )
    debug_artifacts = dict(payload.get("debug_artifacts") if isinstance(payload.get("debug_artifacts"), dict) else {})
    debug_artifacts["track_overlay_videos"] = debug_videos
    payload["debug_artifacts"] = debug_artifacts
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "stage": "sam3-video-tracks",
        "artifact_path": str(artifact_path),
        "status": "ok",
        "generated_count": len(debug_videos),
        "debug_videos": debug_videos,
    }


def _intrinsic_for_frame(intrinsic: np.ndarray, frame_slot: int) -> np.ndarray:
    if intrinsic.ndim == 3:
        return intrinsic[frame_slot]
    if intrinsic.ndim == 2:
        return intrinsic
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def _intrinsic_frame_count(intrinsic: np.ndarray, fallback_count: int) -> int:
    if intrinsic.ndim == 3:
        return intrinsic.shape[0]
    if intrinsic.ndim == 2:
        return fallback_count
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def _processed_rgb_for_frame(processed_images: np.ndarray, frame_index: int) -> np.ndarray:
    if processed_images.ndim != 4 or processed_images.shape[-1] != 3:
        raise ValueError(f"Unsupported processed_images shape: {processed_images.shape}")
    return np.ascontiguousarray(processed_images[frame_index])


def _project_points(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
    return uv, valid


def _mesh_vertices_in_camera(mesh_vertices: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_vertices, dtype=np.float32)
    pose = np.asarray(pose_4x4, dtype=np.float32).reshape(4, 4)
    vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    return (pose @ vertices_h.T).T[:, :3]


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.5:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _draw_axes_and_label(
    *,
    canvas: np.ndarray,
    pose_4x4: np.ndarray,
    K: np.ndarray,
    mesh_diameter: float,
    object_id: str,
    frame_index: int,
) -> None:
    pose = np.asarray(pose_4x4, dtype=np.float32).reshape(4, 4)
    axis_length = max(float(mesh_diameter) * 0.35, 1e-3)
    axes_model = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    axes_camera = _mesh_vertices_in_camera(axes_model, pose)
    axes_uv, axes_valid = _project_points(axes_camera, K)
    if axes_valid[0]:
        origin = tuple(np.round(axes_uv[0]).astype(int))
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for endpoint_index, color in zip([1, 2, 3], axis_colors):
            if axes_valid[endpoint_index]:
                endpoint = tuple(np.round(axes_uv[endpoint_index]).astype(int))
                cv2.line(canvas, origin, endpoint, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, endpoint, 3, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{object_id} frame={frame_index}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{object_id} frame={frame_index}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )


def _rasterize_mesh_faces(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    render = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    uv, valid = _project_points(vertices_camera, K)
    light_dir = np.array([0.25, -0.35, -0.9], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    for face in np.asarray(faces, dtype=np.int64):
        if face.shape[0] != 3 or not valid[face].all():
            continue
        tri_uv = uv[face]
        tri_z = vertices_camera[face, 2]
        if not np.isfinite(tri_uv).all() or (tri_z <= 1e-6).any():
            continue

        min_xy = np.floor(tri_uv.min(axis=0)).astype(int)
        max_xy = np.ceil(tri_uv.max(axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(width - 1, int(max_xy[0]))
        y1 = min(height - 1, int(max_xy[1]))
        if x1 < x0 or y1 < y0:
            continue

        x = np.arange(x0, x1 + 1, dtype=np.float32)
        y = np.arange(y0, y1 + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x, y)
        p0, p1, p2 = tri_uv.astype(np.float32)
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denom)) < 1e-6:
            continue
        w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        depth = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
        region_depth = z_buffer[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (depth < region_depth)
        if not update.any():
            continue

        v0, v1, v2 = vertices_camera[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal_norm = float(np.linalg.norm(normal))
        shade = 0.75
        if normal_norm > 1e-6:
            normal = normal / normal_norm
            shade = 0.35 + 0.65 * abs(float(np.dot(normal, light_dir)))
        color = np.array([30, 180, 255], dtype=np.float32) * shade
        region_render = render[y0 : y1 + 1, x0 : x1 + 1]
        region_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        region_depth[update] = depth[update]
        region_render[update] = np.clip(color, 0, 255).astype(np.uint8)
        region_mask[update] = True

    return render, mask


def _rasterized_pose_layer(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple]:
    render, mask = _rasterize_mesh_faces(
        vertices_camera=vertices_camera,
        faces=faces,
        K=K,
        image_shape=image_shape,
    )
    contours: tuple = ()
    if mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return render, mask, contours


def _draw_projected_pose_overlay(
    *,
    rgb: np.ndarray,
    render: np.ndarray,
    mask: np.ndarray,
    contours: tuple,
    K: np.ndarray,
    pose_4x4: np.ndarray,
    mesh_diameter: float,
    object_id: str,
    frame_index: int,
) -> np.ndarray:
    canvas = cv2.cvtColor(_as_uint8_rgb(rgb), cv2.COLOR_RGB2BGR)
    if mask.any():
        alpha = 0.55
        canvas[mask] = np.clip((1.0 - alpha) * canvas[mask] + alpha * render[mask], 0, 255).astype(np.uint8)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    _draw_axes_and_label(
        canvas=canvas,
        pose_4x4=pose_4x4,
        K=K,
        mesh_diameter=mesh_diameter,
        object_id=object_id,
        frame_index=frame_index,
    )
    return canvas


def write_foundationpose_debug_pose_video(
    *,
    debug_root: Path,
    object_id: str,
    mesh: Any,
    poses: list[dict[str, Any]],
    processed_images: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, Any] | None:
    if not poses:
        return None
    video_dir = debug_root / "debug_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"{object_id}_mesh_render.mp4"
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    diameter = float(np.linalg.norm(np.asarray(mesh.extents, dtype=np.float32)))
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    # Static fixtures hold one pose for the whole clip and the intrinsics are constant,
    # so the software rasterization (and its contours) is identical frame to frame; only
    # the per-frame RGB blend and label differ. Cache the rasterized layer keyed by the
    # exact pose/intrinsic bytes and image shape — bitwise-identical to re-rendering.
    layer_cache: dict[tuple, tuple[np.ndarray, np.ndarray, tuple]] = {}
    try:
        for pose_record in poses:
            frame_index = int(pose_record["frame_index"])
            if frame_index >= processed_images.shape[0] or frame_index >= _intrinsic_frame_count(
                intrinsics, processed_images.shape[0]
            ):
                continue
            rgb = _processed_rgb_for_frame(processed_images, frame_index)
            K = np.ascontiguousarray(_intrinsic_for_frame(intrinsics, frame_index), dtype=np.float32)
            pose = np.ascontiguousarray(
                np.asarray(pose_record.get("raw_pose_4x4") or pose_record["pose_4x4"]),
                dtype=np.float32,
            ).reshape(4, 4)
            image_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
            layer_key = (pose.tobytes(), K.tobytes(), image_shape)
            layer = layer_cache.get(layer_key)
            if layer is None:
                vertices_camera = _mesh_vertices_in_camera(vertices, pose)
                layer = _rasterized_pose_layer(
                    vertices_camera=vertices_camera,
                    faces=faces,
                    K=K,
                    image_shape=image_shape,
                )
                layer_cache[layer_key] = layer
            render, mask, contours = layer
            frame = _draw_projected_pose_overlay(
                rgb=rgb,
                render=render,
                mask=mask,
                contours=contours,
                K=K,
                pose_4x4=pose,
                mesh_diameter=diameter,
                object_id=object_id,
                frame_index=frame_index,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open debug video writer: {output_path}")
            writer.write(frame)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    if frame_count == 0:
        output_path.unlink(missing_ok=True)
        return None
    return {
        "debug_video_path": str(output_path),
        "debug_frame_count": frame_count,
        "debug_render_mode": "software_face_render",
    }


def _load_video_rgb_frames(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for FoundationPose debug render: {video}")
    frames = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from video for FoundationPose debug render: {video}")
    return np.ascontiguousarray(np.stack(frames, axis=0))


def generate_foundationpose_debug_from_artifact(*, artifact_path: Path) -> dict[str, Any]:
    import trimesh

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    video_metric_depth_path = Path(str(payload.get("video_metric_depth") or ""))
    sidecar_path = video_metric_depth_path.with_suffix(".npz")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"FoundationPose debug requires video metric-depth sidecar: {sidecar_path}")
    sidecar = np.load(sidecar_path)
    try:
        intrinsics = np.asarray(sidecar["intrinsics"])
        if "processed_images" in sidecar.files:
            processed_images = np.asarray(sidecar["processed_images"])
            image_source = "video_metric_depth_sidecar.processed_images"
        else:
            processed_images = _load_video_rgb_frames(Path(str(payload.get("video") or "")))
            image_source = "source_video_frames"
    finally:
        sidecar.close()

    generated = []
    debug_root = artifact_path.parent / "debug"
    for item in payload.get("objects") or []:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        object_id = str(item.get("object_id") or "")
        mesh_path = Path(str(item.get("mesh_path") or ""))
        poses = item.get("poses") if isinstance(item.get("poses"), list) else []
        if not object_id or not mesh_path.exists() or not poses:
            continue
        mesh = trimesh.load(mesh_path, force="mesh")
        result = write_foundationpose_debug_pose_video(
            debug_root=debug_root,
            object_id=object_id,
            mesh=mesh,
            poses=poses,
            processed_images=processed_images,
            intrinsics=intrinsics,
        )
        if result:
            item.update(result)
            generated.append({"object_id": object_id, **result})

    payload["debug_artifacts"] = True
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "stage": "foundationpose",
        "artifact_path": str(artifact_path),
        "status": "ok",
        "image_source": image_source,
        "generated_count": len(generated),
        "debug_videos": generated,
    }


def world_reconstruction_debug_render_command(
    *,
    command: str,
    render_input_path: Path,
    output_video: Path,
    output_json: Path,
) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Blender debug render command is empty")
    if "--python" in tokens or any(
        Path(token).name.startswith("render_world_reconstruction")
        and Path(token).suffix == ".py"
        for token in tokens
    ):
        raise ValueError(
            "Blender debug render command must not select a renderer script; "
            "the pipeline always uses the unified latest renderer"
        )
    runner_args = [
        "--world-reconstruction",
        str(render_input_path.resolve()),
        "--output-video",
        str(output_video.resolve()),
        "--output-json",
        str(output_json.resolve()),
    ]
    return tokens + [
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(LATEST_WORLD_RECONSTRUCTION_RENDERER),
        "--",
        *runner_args,
    ]


def run_world_reconstruction_debug_render(
    *,
    command: str,
    render_input_path: Path,
    output_video: Path,
    output_json: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    args = world_reconstruction_debug_render_command(
        command=command,
        render_input_path=render_input_path,
        output_video=output_video,
        output_json=output_json,
    )
    start = time.perf_counter()
    completed = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start
    payload: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 and output_json.exists() else "tool_error",
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "command": args,
        "input": str(render_input_path),
        "output_video": str(output_video),
        "output_json": str(output_json),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode == 0 and output_json.exists():
        try:
            render_artifact = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload["status"] = "tool_error"
            payload["message"] = f"latest debug renderer wrote an invalid artifact: {exc}"
            return payload
        source_camera_video = Path(
            str(
                render_artifact.get("source_camera_video_path")
                or render_artifact.get("video_path")
                or render_artifact.get("rendered_path")
                or output_video
            )
        )
        payload.update(
            {
                "source_camera_video_path": str(source_camera_video),
                "blend_path": str(output_json.with_suffix(".blend")),
                "source_camera_video_exists": source_camera_video.exists(),
                "render_profile": render_artifact.get("render_profile"),
            }
        )
    else:
        payload["message"] = (completed.stderr or completed.stdout or "").strip()
    return payload


def _trajectory_plane_arrays(
    *,
    trajectory: list[dict[str, Any]],
    frames: list[int],
    object_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    frame_to_offset = {int(frame): index for index, frame in enumerate(frames)}
    object_to_index = {str(object_id): index for index, object_id in enumerate(object_ids)}
    positions = np.zeros((len(frames), len(object_ids), 2), dtype=np.float64)
    mask = np.zeros((len(frames), len(object_ids), 1), dtype=np.float64)
    for frame_payload in trajectory:
        if not isinstance(frame_payload, dict):
            continue
        try:
            frame_index = int(frame_payload.get("frame_index"))
        except (TypeError, ValueError):
            continue
        frame_offset = frame_to_offset.get(frame_index)
        if frame_offset is None:
            continue
        for object_payload in frame_payload.get("objects", []) or []:
            if not isinstance(object_payload, dict):
                continue
            object_id = str(object_payload.get("object_id") or "")
            object_index = object_to_index.get(object_id)
            if object_index is None:
                continue
            try:
                positions[frame_offset, object_index, 0] = float(object_payload["x"])
                positions[frame_offset, object_index, 1] = float(object_payload["y"])
            except (KeyError, TypeError, ValueError):
                continue
            mask[frame_offset, object_index, 0] = 1.0
    return positions, mask


def render_qcpr_plane_projection_debug(
    *,
    trajectory_path: Path,
    world_reconstruction_fit_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    trajectory_payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    fit_payload = json.loads(Path(world_reconstruction_fit_path).read_text(encoding="utf-8"))
    edited_rollouts = [
        item
        for item in trajectory_payload.get("edited_rollouts", []) or []
        if isinstance(item, dict) and item.get("trajectory")
    ]
    if not edited_rollouts:
        return {
            "status": "skipped",
            "reason": "trajectory has no edited_rollouts",
            "trajectory_path": str(trajectory_path),
            "world_reconstruction_fit_path": str(world_reconstruction_fit_path),
        }

    output_root = Path(output_dir) if output_dir is not None else Path(trajectory_path).parent / "debug"
    output_root.mkdir(parents=True, exist_ok=True)
    analytic_sysid = _load_analytic_sysid_module()
    fit_object_ids, target_plane = analytic_sysid._project_target_to_plane(fit_payload)
    object_scales = analytic_sysid._object_scales(fit_payload, fit_object_ids, target_plane)
    _shape_ids, _half_extents, _angles, contact_radii_all, _shape_names, _proxy_names = analytic_sysid._shape_metadata(
        fit=fit_payload,
        object_ids=fit_object_ids,
        object_scales_by_id=object_scales,
    )
    radius_by_object = {
        str(object_id): float(contact_radii_all[index])
        for index, object_id in enumerate(fit_object_ids)
    }
    base_rollout = trajectory_payload.get("base_rollout") if isinstance(trajectory_payload.get("base_rollout"), dict) else {}
    base_trajectory = base_rollout.get("trajectory") if isinstance(base_rollout.get("trajectory"), list) else []
    generated = []
    for rollout in edited_rollouts:
        edited_trajectory = rollout.get("trajectory") if isinstance(rollout.get("trajectory"), list) else []
        if not edited_trajectory:
            continue
        rollout_id = _safe_name(str(rollout.get("rollout_id") or "edited_rollout"))
        frames = sorted(
            {
                int(frame.get("frame_index"))
                for frame in edited_trajectory
                if isinstance(frame, dict) and frame.get("frame_index") is not None
            }
        )
        if not frames:
            continue
        object_ids = sorted(
            {
                str(item.get("object_id"))
                for frame in [*base_trajectory, *edited_trajectory]
                if isinstance(frame, dict)
                for item in frame.get("objects", []) or []
                if isinstance(item, dict) and item.get("object_id") is not None
            }
        )
        if not object_ids:
            continue
        target, target_mask = _trajectory_plane_arrays(
            trajectory=base_trajectory,
            frames=frames,
            object_ids=object_ids,
        )
        predicted, predicted_mask = _trajectory_plane_arrays(
            trajectory=edited_trajectory,
            frames=frames,
            object_ids=object_ids,
        )
        mask = np.maximum(target_mask, predicted_mask)
        if float(mask.sum()) <= 0.0:
            continue
        contact_radii = np.asarray([radius_by_object.get(object_id, 0.05) for object_id in object_ids], dtype=np.float64)
        output_mp4 = output_root / f"{rollout_id}_plane_projection.mp4"
        analytic_sysid._render_comparison_video(
            output_mp4,
            object_ids=object_ids,
            frames=frames,
            target=target,
            predicted=predicted,
            mask=mask,
            target_mask=target_mask,
            predicted_mask=predicted_mask,
            contact_radii=contact_radii,
            source_fps=float(fit_payload["trajectory_physics_initialization"]["fps"]),
            target_title="Pre-edit rollout",
            predicted_title=f"Edited rollout: {rollout_id}",
        )
        generated.append(
            {
                "rollout_id": str(rollout.get("rollout_id") or rollout_id),
                "removed_object_id": rollout.get("removed_object_id"),
                "output_mp4": str(output_mp4),
                "left_panel": "pre_edit_rollout",
                "right_panel": "edited_rollout",
                "object_ids": object_ids,
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "frame_count": len(frames),
            }
        )

    manifest_path = output_root / "qcpr_plane_projection_debug.json"
    payload = {
        "status": "ok" if generated else "skipped",
        "trajectory_path": str(trajectory_path),
        "world_reconstruction_fit_path": str(world_reconstruction_fit_path),
        "generated": generated,
    }
    if not generated:
        payload["reason"] = "no renderable edited rollout trajectories"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload
