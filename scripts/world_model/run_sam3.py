from __future__ import annotations

import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM3_ROOT = PROJECT_ROOT / "third_party" / "sam3"


def _add_sam3_to_path() -> None:
    if not SAM3_ROOT.exists():
        raise FileNotFoundError(f"SAM3 submodule not found: {SAM3_ROOT}")
    sys.path.insert(0, str(SAM3_ROOT))


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    mask_bool = mask.astype(bool)
    area = int(mask_bool.sum())
    if area == 0:
        return {"area": 0, "bbox_xyxy": None, "centroid_xy": None}
    ys, xs = np.where(mask_bool)
    return {
        "area": area,
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _masks_from_response(response: dict[str, Any]) -> dict[int, np.ndarray]:
    import torch

    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        return {}
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.detach().cpu().numpy()
    frame_masks = {}
    for index, obj_id in enumerate(obj_ids):
        mask = binary_masks[index]
        if mask.ndim == 3:
            mask = mask[0]
        frame_masks[int(obj_id)] = mask.astype(bool)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return frame_masks


def _mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    height, width = image.shape[:2]
    if mask_bool.shape == (height, width):
        return mask_bool
    resized = cv2.resize(mask_bool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _patch_start_session_for_current_sam3(model: Any) -> None:
    def start_session(
        self: Any,
        resource_path: str,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    model.start_session = types.MethodType(start_session, model)
