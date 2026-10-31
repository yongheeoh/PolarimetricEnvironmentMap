"""Fused equirectangular projection/remap for BGR payload images.

The fast path matches the narrow ``_warp_stokes`` block:

    projectPoints(obj_points, rvec, zero_tvec, cameraMatrix, distCoeffs)
    reject points with local camera z < 0
    remap(src, map_x, map_y, cv2.INTER_LINEAR, borderValue=np.nan)

Only the 5-coefficient OpenCV distortion model ``[k1, k2, p1, p2, k3]`` is
supported.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from pemap.geometry.optimized.projectPoints import projectPoints
from pemap.geometry.optimized.remap import _INTER_LINEAR_WEIGHT_00, _INTER_LINEAR_WEIGHT_01, _INTER_LINEAR_WEIGHT_10, _INTER_LINEAR_WEIGHT_11, INTER_TAB_SIZE, remap

try:
    from numba import literal_unroll
    from numba import njit
    from numba import prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent/broken
    njit = None
    prange = range
    NUMBA_AVAILABLE = False

Backend = Literal["auto", "numba", "staged"]

__all__ = [
    "NUMBA_AVAILABLE",
    "remap_equirectangular",
]


def remap_equirectangular(
    src: np.ndarray,
    obj_points: np.ndarray,
    rvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width_envmap: int,
    height_envmap: int,
    dst: np.ndarray | None = None,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Project equirectangular directions and remap a BGR payload image.

    ``obj_points`` must contain the flattened ``(height_envmap * width_envmap, 3)``
    direction grid, including any caller-applied domain rotation.
    """

    if backend not in ("auto", "numba", "staged"):
        raise ValueError(f"Unsupported remap_equirectangular backend: {backend!r}")

    width, height = _prepare_envmap_size(width_envmap, height_envmap)
    src_flat, src_image, trailing_shape = _prepare_src(src)
    src_h, src_w, channels = src_flat.shape
    points = _prepare_obj_points(obj_points, width * height)
    rvec32 = _prepare_rvec(rvec)
    camera32 = _prepare_camera_matrix(camera_matrix)
    dist32 = _prepare_dist_coeffs(dist_coeffs)
    output, dst_array = _prepare_output(dst, height, width, trailing_shape, channels)

    if backend == "staged":
        return _remap_equirectangular_staged(src_image, points, rvec32, camera32, dist32, width, height, dst_array)

    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available; use backend='staged'")
    elif not NUMBA_AVAILABLE:
        return _remap_equirectangular_staged(src_image, points, rvec32, camera32, dist32, width, height, dst_array)

    projection_rotation = np.ascontiguousarray(cv2.Rodrigues(rvec32.astype(np.float64))[0], dtype=np.float64)
    behind_rotation = np.ascontiguousarray(cv2.Rodrigues(rvec32)[0], dtype=np.float32)
    output_flat = output.reshape(width * height, channels)

    fx = float(camera32[0, 0])
    fy = float(camera32[1, 1])
    cx = float(camera32[0, 2])
    cy = float(camera32[1, 2])
    k1 = float(dist32[0])
    k2 = float(dist32[1])
    p1 = float(dist32[2])
    p2 = float(dist32[3])
    k3 = float(dist32[4])

    src_flat_2d = src_flat.reshape(src_h * src_w, channels)
    if p1 == 0.0 and p2 == 0.0:
        if k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
            _remap_equirectangular_pinhole_kernel(
                src_flat_2d,
                src_w,
                points,
                projection_rotation,
                behind_rotation,
                fx,
                fy,
                cx,
                cy,
                _INTER_LINEAR_WEIGHT_00,
                _INTER_LINEAR_WEIGHT_01,
                _INTER_LINEAR_WEIGHT_10,
                _INTER_LINEAR_WEIGHT_11,
                output_flat,
            )
        elif k3 == 0.0:
            _remap_equirectangular_radial2_kernel(
                src_flat_2d,
                src_w,
                points,
                projection_rotation,
                behind_rotation,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                _INTER_LINEAR_WEIGHT_00,
                _INTER_LINEAR_WEIGHT_01,
                _INTER_LINEAR_WEIGHT_10,
                _INTER_LINEAR_WEIGHT_11,
                output_flat,
            )
        else:
            _remap_equirectangular_radial3_kernel(
                src_flat_2d,
                src_w,
                points,
                projection_rotation,
                behind_rotation,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                _INTER_LINEAR_WEIGHT_00,
                _INTER_LINEAR_WEIGHT_01,
                _INTER_LINEAR_WEIGHT_10,
                _INTER_LINEAR_WEIGHT_11,
                output_flat,
            )
    else:
        _remap_equirectangular_full_kernel(
            src_flat_2d,
            src_w,
            points,
            projection_rotation,
            behind_rotation,
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
            k3,
            _INTER_LINEAR_WEIGHT_00,
            _INTER_LINEAR_WEIGHT_01,
            _INTER_LINEAR_WEIGHT_10,
            _INTER_LINEAR_WEIGHT_11,
            output_flat,
        )

    if dst_array is not None:
        return dst_array
    return output.reshape((height, width) + trailing_shape)


def _remap_equirectangular_staged(
    src_image: np.ndarray,
    obj_points: np.ndarray,
    rvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    dst: np.ndarray | None,
) -> np.ndarray:
    img_points, _ = projectPoints(obj_points, rvec, np.zeros((3, 1), dtype=np.float32), camera_matrix, dist_coeffs)
    map_x = img_points[:, 0, 0].reshape(height, width).astype(np.float32, copy=False)
    map_y = img_points[:, 0, 1].reshape(height, width).astype(np.float32, copy=False)

    rmat = cv2.Rodrigues(rvec)[0]
    obj_points_local = (rmat @ obj_points.T).T
    is_behind = obj_points_local[:, 2].reshape(height, width) < 0.0
    map_x[is_behind] = -2.0
    map_y[is_behind] = -2.0

    return remap(
        src_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        dst=dst,
        borderValue=np.nan,
    )


def _prepare_envmap_size(width_envmap: int, height_envmap: int) -> tuple[int, int]:
    width = int(width_envmap)
    height = int(height_envmap)
    if width <= 0 or height <= 0:
        raise ValueError(f"width_envmap and height_envmap must be positive, got {(width_envmap, height_envmap)}")
    return width, height


def _prepare_src(src: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = np.asarray(src, dtype=np.float32)
    if image.ndim != 4 or image.shape[2] != 3 or image.shape[3] < 1:
        raise ValueError(f"Expected src shape (H, W, 3, S) with S >= 1, got {image.shape}")

    src_h, src_w = image.shape[:2]
    if src_h < 2 or src_w < 2:
        raise ValueError(f"src height and width must be at least 2, got {(src_h, src_w)}")

    trailing_shape = (3, int(image.shape[3]))
    src_flat = np.ascontiguousarray(image.reshape(src_h, src_w, 3 * int(image.shape[3])))
    src_image = src_flat.reshape(src_h, src_w, 3, int(image.shape[3]))
    return src_flat, src_image, trailing_shape


def _prepare_obj_points(obj_points: np.ndarray, n_pixels: int) -> np.ndarray:
    points = np.asarray(obj_points, dtype=np.float32)
    expected_size = int(n_pixels) * 3
    if points.size != expected_size:
        raise ValueError(f"obj_points must contain {n_pixels} 3D points, got array with {points.size} value(s)")
    return np.ascontiguousarray(points.reshape(n_pixels, 3))


def _prepare_rvec(rvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rvec, dtype=np.float32).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"rvec must have exactly 3 elements, got shape {np.asarray(rvec).shape}")
    return np.ascontiguousarray(vector.reshape(3, 1))


def _prepare_camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {matrix.shape}")
    return np.ascontiguousarray(matrix)


def _prepare_dist_coeffs(dist_coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size != 5:
        raise ValueError(f"dist_coeffs must contain exactly 5 elements, got {coeffs.size}")
    return np.ascontiguousarray(coeffs)


def _prepare_output(
    dst: np.ndarray | None,
    height: int,
    width: int,
    trailing_shape: tuple[int, int],
    channels: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if dst is None:
        return np.empty((height * width, channels), dtype=np.float32), None

    dst_array = np.asarray(dst)
    expected_shape = (height, width) + trailing_shape
    if dst_array.shape != expected_shape:
        raise ValueError(f"dst must have shape {expected_shape}, got {dst_array.shape}")
    if dst_array.dtype != np.float32:
        raise ValueError(f"dst must have dtype float32, got {dst_array.dtype}")
    if not dst_array.flags.c_contiguous:
        raise ValueError("dst must be C-contiguous")
    return dst_array.reshape(height * width, channels), dst_array


if NUMBA_AVAILABLE:

    @njit(inline="always")
    def _write_nan_3(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(3)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan_6(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(6)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan_12(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(12)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan_15(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(15)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan_36(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(36)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan_39(output: np.ndarray, p: int, nan: np.float32) -> None:
        for c in literal_unroll(range(39)):
            output[p, c] = nan

    @njit(inline="always")
    def _write_nan(output: np.ndarray, p: int) -> None:
        nan = np.float32(np.nan)
        channels = output.shape[1]
        if channels == 3:
            _write_nan_3(output, p, nan)
        elif channels == 6:
            _write_nan_6(output, p, nan)
        elif channels == 12:
            _write_nan_12(output, p, nan)
        elif channels == 15:
            _write_nan_15(output, p, nan)
        elif channels == 36:
            _write_nan_36(output, p, nan)
        elif channels == 39:
            _write_nan_39(output, p, nan)
        else:
            for c in range(channels):
                output[p, c] = nan

    @njit(inline="always")
    def _write_linear_sample_3(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(3)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_6(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(6)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_12(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(12)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_15(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(15)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_36(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(36)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_39(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in literal_unroll(range(39)):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample_generic(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        for c in range(output.shape[1]):
            output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11

    @njit(inline="always")
    def _write_linear_sample(
        src_flat: np.ndarray,
        src00: int,
        src01: int,
        src10: int,
        src11: int,
        w00: np.float32,
        w01: np.float32,
        w10: np.float32,
        w11: np.float32,
        output: np.ndarray,
        p: int,
    ) -> None:
        channels = output.shape[1]
        if channels == 3:
            _write_linear_sample_3(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        elif channels == 6:
            _write_linear_sample_6(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        elif channels == 12:
            _write_linear_sample_12(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        elif channels == 15:
            _write_linear_sample_15(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        elif channels == 36:
            _write_linear_sample_36(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        elif channels == 39:
            _write_linear_sample_39(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)
        else:
            _write_linear_sample_generic(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)

    @njit(inline="always")
    def _write_linear_sample_or_nan(
        src_flat: np.ndarray,
        src_w: int,
        map_x: np.float32,
        map_y: np.float32,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
        p: int,
    ) -> None:
        src_h = src_flat.shape[0] // src_w
        tab_size = INTER_TAB_SIZE
        valid = True
        sx = 0
        sy = 0
        ax = 0
        ay = 0

        if not (np.isfinite(map_x) and np.isfinite(map_y)):
            valid = False
        else:
            sx = int(np.floor(map_x))
            sy = int(np.floor(map_y))
            ax = int(np.rint(np.float32(map_x - np.float32(sx)) * np.float32(tab_size)))
            ay = int(np.rint(np.float32(map_y - np.float32(sy)) * np.float32(tab_size)))

            if ax == tab_size:
                sx += 1
                ax = 0
            if ay == tab_size:
                sy += 1
                ay = 0

            if sx < 0 or sy < 0 or sx + 1 >= src_w or sy + 1 >= src_h:
                valid = False

        if not valid:
            _write_nan(output, p)
            return

        base = sy * src_w + sx
        weight_index = ay * tab_size + ax
        w00 = weight_00[weight_index]
        w01 = weight_01[weight_index]
        w10 = weight_10[weight_index]
        w11 = weight_11[weight_index]

        src00 = base
        src01 = base + 1
        src10 = base + src_w
        src11 = base + src_w + 1
        _write_linear_sample(src_flat, src00, src01, src10, src11, w00, w01, w10, w11, output, p)

    @njit(inline="always")
    def _is_behind(points: np.ndarray, behind_rotation: np.ndarray, p: int) -> bool:
        z = behind_rotation[2, 0] * points[p, 0] + behind_rotation[2, 1] * points[p, 1] + behind_rotation[2, 2] * points[p, 2]
        return z < np.float32(0.0)

    @njit(inline="always")
    def _project_base(points: np.ndarray, projection_rotation: np.ndarray, p: int) -> tuple[float, float, float]:
        obj_x = float(points[p, 0])
        obj_y = float(points[p, 1])
        obj_z = float(points[p, 2])
        cam_x = projection_rotation[0, 0] * obj_x + projection_rotation[0, 1] * obj_y + projection_rotation[0, 2] * obj_z
        cam_y = projection_rotation[1, 0] * obj_x + projection_rotation[1, 1] * obj_y + projection_rotation[1, 2] * obj_z
        cam_z = projection_rotation[2, 0] * obj_x + projection_rotation[2, 1] * obj_y + projection_rotation[2, 2] * obj_z
        return cam_x, cam_y, cam_z

    @njit(parallel=True, cache=True, fastmath=False)
    def _remap_equirectangular_pinhole_kernel(
        src_flat: np.ndarray,
        src_w: int,
        points: np.ndarray,
        projection_rotation: np.ndarray,
        behind_rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for p in prange(points.shape[0]):
            if _is_behind(points, behind_rotation, p):
                _write_nan(output, p)
                continue

            cam_x, cam_y, cam_z = _project_base(points, projection_rotation, p)
            if cam_z == 0.0:
                _write_nan(output, p)
                continue

            map_x = np.float32(fx * (cam_x / cam_z) + cx)
            map_y = np.float32(fy * (cam_y / cam_z) + cy)
            _write_linear_sample_or_nan(src_flat, src_w, map_x, map_y, weight_00, weight_01, weight_10, weight_11, output, p)

    @njit(parallel=True, cache=True, fastmath=False)
    def _remap_equirectangular_radial2_kernel(
        src_flat: np.ndarray,
        src_w: int,
        points: np.ndarray,
        projection_rotation: np.ndarray,
        behind_rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for p in prange(points.shape[0]):
            if _is_behind(points, behind_rotation, p):
                _write_nan(output, p)
                continue

            cam_x, cam_y, cam_z = _project_base(points, projection_rotation, p)
            if cam_z == 0.0:
                _write_nan(output, p)
                continue

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            radial = 1.0 + (k1 + k2 * r2) * r2
            map_x = np.float32(fx * x * radial + cx)
            map_y = np.float32(fy * y * radial + cy)
            _write_linear_sample_or_nan(src_flat, src_w, map_x, map_y, weight_00, weight_01, weight_10, weight_11, output, p)

    @njit(parallel=True, cache=True, fastmath=False)
    def _remap_equirectangular_radial3_kernel(
        src_flat: np.ndarray,
        src_w: int,
        points: np.ndarray,
        projection_rotation: np.ndarray,
        behind_rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for p in prange(points.shape[0]):
            if _is_behind(points, behind_rotation, p):
                _write_nan(output, p)
                continue

            cam_x, cam_y, cam_z = _project_base(points, projection_rotation, p)
            if cam_z == 0.0:
                _write_nan(output, p)
                continue

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            r4 = r2 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
            map_x = np.float32(fx * x * radial + cx)
            map_y = np.float32(fy * y * radial + cy)
            _write_linear_sample_or_nan(src_flat, src_w, map_x, map_y, weight_00, weight_01, weight_10, weight_11, output, p)

    @njit(parallel=True, cache=True, fastmath=False)
    def _remap_equirectangular_full_kernel(
        src_flat: np.ndarray,
        src_w: int,
        points: np.ndarray,
        projection_rotation: np.ndarray,
        behind_rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        p1: float,
        p2: float,
        k3: float,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for p in prange(points.shape[0]):
            if _is_behind(points, behind_rotation, p):
                _write_nan(output, p)
                continue

            cam_x, cam_y, cam_z = _project_base(points, projection_rotation, p)
            if cam_z == 0.0:
                _write_nan(output, p)
                continue

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            xy = x * y
            x_distorted = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
            y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
            map_x = np.float32(fx * x_distorted + cx)
            map_y = np.float32(fy * y_distorted + cy)
            _write_linear_sample_or_nan(src_flat, src_w, map_x, map_y, weight_00, weight_01, weight_10, weight_11, output, p)

else:

    def _remap_equirectangular_pinhole_kernel(*args, **kwargs) -> None:
        raise RuntimeError("Numba is not available")

    def _remap_equirectangular_radial2_kernel(*args, **kwargs) -> None:
        raise RuntimeError("Numba is not available")

    def _remap_equirectangular_radial3_kernel(*args, **kwargs) -> None:
        raise RuntimeError("Numba is not available")

    def _remap_equirectangular_full_kernel(*args, **kwargs) -> None:
        raise RuntimeError("Numba is not available")
