"""Source image resize utilities for BGR-Stokes processing."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeInfo:
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    sx: float
    sy: float


def make_resize_info(height: int, width: int, resize_scale: float) -> ResizeInfo:
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got height={height}, width={width}")

    resize_scale = float(resize_scale)
    if not np.isfinite(resize_scale) or resize_scale <= 0.0 or resize_scale > 1.0:
        raise ValueError(f"resize_scale must be in (0, 1], got {resize_scale!r}")

    if resize_scale == 1.0:
        resized_height, resized_width = height, width
    else:
        resized_width = int(round(width * resize_scale))
        resized_height = int(round(height * resize_scale))
        if resized_width < 1 or resized_height < 1:
            raise ValueError("resize_scale is too small for the source image size: " f"{width}x{height} at scale {resize_scale} becomes {resized_width}x{resized_height}")

    return ResizeInfo(
        original_hw=(height, width),
        resized_hw=(resized_height, resized_width),
        sx=resized_width / width,
        sy=resized_height / height,
    )


def resize_bgr_stokes_image(img_bgr_stokes: np.ndarray, resize_info: ResizeInfo) -> np.ndarray:
    img_bgr_stokes = np.asarray(img_bgr_stokes, dtype=np.float32)
    if img_bgr_stokes.ndim != 4 or img_bgr_stokes.shape[2] != 3 or img_bgr_stokes.shape[3] < 1:
        raise ValueError(f"Expected BGR-Stokes image shape (H, W, 3, S), got {img_bgr_stokes.shape}")

    height, width, num_channels, num_components = img_bgr_stokes.shape
    if (height, width) != resize_info.original_hw:
        raise ValueError(f"Image shape {(height, width)} does not match resize source shape {resize_info.original_hw}")

    resized_height, resized_width = resize_info.resized_hw
    if (resized_height, resized_width) == (height, width):
        return img_bgr_stokes

    flat = img_bgr_stokes.reshape(height, width, num_channels * num_components)
    resized_flat = cv2.resize(flat, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized_flat.reshape(resized_height, resized_width, num_channels, num_components).astype(np.float32, copy=False)


def resize_bgr_stokes_by_id(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    resize_scale: float,
) -> tuple[dict[str, np.ndarray], ResizeInfo]:
    if not img_bgr_stokes_by_id:
        raise ValueError("img_bgr_stokes_by_id must not be empty")

    first_img = next(iter(img_bgr_stokes_by_id.values()))
    if np.asarray(first_img).ndim < 2:
        raise ValueError(f"Expected image with at least two dimensions, got {np.asarray(first_img).shape}")

    height, width = np.asarray(first_img).shape[:2]
    resize_info = make_resize_info(height, width, resize_scale)
    resized_by_id: dict[str, np.ndarray] = {}
    for img_id, img_bgr_stokes in img_bgr_stokes_by_id.items():
        resized_by_id[img_id] = resize_bgr_stokes_image(img_bgr_stokes, resize_info)
    return resized_by_id, resize_info


def scale_camera_matrix_for_resize(K: np.ndarray, resize_info: ResizeInfo) -> np.ndarray:
    K_scaled = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
    if resize_info.resized_hw == resize_info.original_hw:
        return K_scaled

    cx = float(K_scaled[0, 2])
    cy = float(K_scaled[1, 2])

    K_scaled[0, :2] *= resize_info.sx
    K_scaled[1, :2] *= resize_info.sy
    K_scaled[0, 2] = resize_info.sx * (cx + 0.5) - 0.5
    K_scaled[1, 2] = resize_info.sy * (cy + 0.5) - 0.5
    return K_scaled


def scale_intrinsics_for_resize(intrinsics: dict, resize_info: ResizeInfo) -> dict:
    scaled_intrinsics = dict(intrinsics)
    scaled_intrinsics["K"] = scale_camera_matrix_for_resize(np.asarray(intrinsics["K"]), resize_info)
    if "image_size_wh" in scaled_intrinsics:
        resized_height, resized_width = resize_info.resized_hw
        scaled_intrinsics["image_size_wh"] = (resized_width, resized_height)
    return scaled_intrinsics


def resize_source_stokes_inputs(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    intrinsics: dict,
    resize_scale: float,
) -> tuple[dict[str, np.ndarray], dict, ResizeInfo]:
    resized_by_id, resize_info = resize_bgr_stokes_by_id(img_bgr_stokes_by_id, resize_scale)
    return resized_by_id, scale_intrinsics_for_resize(intrinsics, resize_info), resize_info
