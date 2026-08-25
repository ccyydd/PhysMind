from __future__ import annotations

import time
from typing import Any

import numpy as np


class CudaMaskRasterizer:
    """nvdiffrast mask renderer with one persistent CUDA context.

    Timing deliberately covers the complete per-call path: NumPy validation,
    host-to-device copies, rasterization, synchronization, and mask copy-back.
    """

    def __init__(self) -> None:
        self._context = None
        self._torch = None
        self._dr = None
        self._faces_cache: dict[tuple[Any, ...], Any] = {}
        self.reset(clear_cache=False)

    def reset(self, *, clear_cache: bool = True) -> None:
        self.calls = 0
        self.elapsed_sec = 0.0
        self.triangles_total = 0
        if clear_cache:
            self._faces_cache.clear()

    def _runtime(self):
        if self._torch is None or self._dr is None:
            import torch
            import nvdiffrast.torch as dr

            self._torch = torch
            self._dr = dr
        if self._context is None:
            self._context = self._dr.RasterizeCudaContext(device="cuda")
        return self._torch, self._dr

    def render(
        self,
        *,
        vertices_camera: np.ndarray,
        faces: np.ndarray,
        intrinsic: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        started = time.perf_counter()
        vertices_np = np.ascontiguousarray(vertices_camera, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32, order="C")
        intrinsic_np = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
        height, width = image_shape

        valid_vertices = np.isfinite(vertices_np).all(axis=1) & (
            vertices_np[:, 2] > 1e-6
        )
        valid_face_mask = valid_vertices[faces_np].all(axis=1)
        if not bool(valid_face_mask.any()):
            mask = np.zeros(image_shape, dtype=bool)
            self.calls += 1
            self.elapsed_sec += time.perf_counter() - started
            return mask

        if bool(valid_face_mask.all()):
            valid_faces = faces_np
            validity_key = None
        else:
            valid_faces = np.ascontiguousarray(faces_np[valid_face_mask])
            validity_key = valid_face_mask.tobytes()
        faces_key = (
            int(faces_np.__array_interface__["data"][0]),
            faces_np.shape,
            faces_np.strides,
            validity_key,
        )

        torch, dr = self._runtime()
        vertices = torch.as_tensor(vertices_np, dtype=torch.float32, device="cuda")
        triangles = self._faces_cache.get(faces_key)
        if triangles is None:
            triangles = torch.as_tensor(
                valid_faces, dtype=torch.int32, device="cuda"
            )
            self._faces_cache[faces_key] = triangles
        intrinsic_t = torch.as_tensor(
            intrinsic_np, dtype=torch.float32, device="cuda"
        )

        x, y, z = vertices.unbind(dim=1)
        x_clip = (
            2.0 * intrinsic_t[0, 0] * x / width
            + z * (2.0 * (intrinsic_t[0, 2] + 0.5) / width - 1.0)
        )
        y_clip = (
            -2.0 * intrinsic_t[1, 1] * y / height
            + z * (1.0 - 2.0 * (intrinsic_t[1, 2] + 0.5) / height)
        )
        clip = torch.stack([x_clip, y_clip, z, z], dim=1).unsqueeze(0).contiguous()
        raster, _ = dr.rasterize(
            self._context,
            clip,
            triangles,
            resolution=image_shape,
            grad_db=False,
        )
        # nvdiffrast follows the OpenGL bottom-up row convention; project masks
        # and SAM3 masks use standard top-down image rows.
        mask = torch.flip(raster[0, ..., 3] > 0, dims=(0,)).cpu().numpy()

        self.calls += 1
        self.triangles_total += int(len(valid_faces))
        self.elapsed_sec += time.perf_counter() - started
        return mask

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "nvdiffrast_cuda",
            "calls": self.calls,
            "elapsed_sec": self.elapsed_sec,
            "mean_ms_per_call": 1000.0 * self.elapsed_sec / max(self.calls, 1),
            "triangles_total": self.triangles_total,
            "face_tensor_cache_entries": len(self._faces_cache),
        }


_CUDA_MASK_RASTERIZER = CudaMaskRasterizer()


def reset_cuda_mask_rasterizer_stats() -> None:
    _CUDA_MASK_RASTERIZER.reset(clear_cache=True)


def cuda_mask_rasterizer_stats() -> dict[str, Any]:
    return _CUDA_MASK_RASTERIZER.stats()


def render_mesh_mask_cuda(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    return _CUDA_MASK_RASTERIZER.render(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=image_shape,
    )


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    target_h, target_w = shape
    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def render_mesh_depth(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)

    z = vertices_camera[:, 2]
    valid = np.isfinite(vertices_camera).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    uv = np.full((vertices_camera.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = intrinsic[0, 0] * vertices_camera[valid, 0] / z[valid] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * vertices_camera[valid, 1] / z[valid] + intrinsic[1, 2]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    for face in faces:
        if face.shape[0] != 3 or not valid[face].all():
            continue
        tri_uv = uv[face]
        tri_z = z[face]
        if not np.isfinite(tri_uv).all() or not np.isfinite(tri_z).all():
            continue

        min_xy = np.floor(tri_uv.min(axis=0)).astype(int)
        max_xy = np.ceil(tri_uv.max(axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(width - 1, int(max_xy[0]))
        y1 = min(height - 1, int(max_xy[1]))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
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

        tri_depth = w0 * float(tri_z[0]) + w1 * float(tri_z[1]) + w2 * float(tri_z[2])
        region_depth = depth[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (tri_depth < region_depth)
        if update.any():
            region_depth[update] = tri_depth[update]

    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return mask, depth


def mask_occluded_mesh_mask(
    *,
    rendered_mask: np.ndarray,
    occluder_mask: np.ndarray | None,
    self_mask: np.ndarray | None,
) -> np.ndarray:
    """Depth-free occlusion: decide visibility straight from the recognized masks.

    The observed image already resolves front/back at every pixel — whichever
    object's recognized mask covers a pixel is the visible (front-most) object
    there. So a rendered pixel of the current object is hidden iff the camera
    assigns that pixel to some OTHER object, i.e. it lies inside the union of the
    other objects' masks (``occluder_mask``) but outside the object's own mask
    (``self_mask``). Pixels over empty background are kept so that over-extension
    is still penalized. No scene depth is consulted.
    """
    visible = np.asarray(rendered_mask).astype(bool).copy()
    if occluder_mask is None:
        return visible

    occluder = np.asarray(occluder_mask).astype(bool)
    if occluder.shape != visible.shape:
        occluder = resize_mask_nearest(occluder, visible.shape)

    if self_mask is not None:
        own = np.asarray(self_mask).astype(bool)
        if own.shape != visible.shape:
            own = resize_mask_nearest(own, visible.shape)
        occluder = occluder & ~own

    return visible & ~occluder


def render_mask_occluded_mesh(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    occluder_mask: np.ndarray | None = None,
    self_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render a mesh and apply depth-free mask-based occlusion.

    ``occluder_mask`` is the union of the other objects' recognized masks for this
    frame. ``rendered_depth`` is returned as a rasterization by-product but is not
    used for occlusion.
    """
    rendered_mask, rendered_depth = render_mesh_depth(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=image_shape,
    )
    visible_mask = mask_occluded_mesh_mask(
        rendered_mask=rendered_mask,
        occluder_mask=occluder_mask,
        self_mask=self_mask,
    )
    return {
        "rendered_mask": rendered_mask,
        "rendered_depth": rendered_depth,
        "visible_mask": visible_mask,
    }
