"""OpenCV-like projection for the project's zero-translation camera path.

The fast path matches this narrow OpenCV mode:

    cv2.projectPoints(objectPoints, rvec, zero_tvec, cameraMatrix, distCoeffs)

Only the 5-coefficient distortion model ``[k1, k2, p1, p2, k3]`` is
supported. Translation must be exactly zero and jacobians are not produced.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

try:
    from numba import njit
    from numba import prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent/broken
    njit = None
    prange = range
    NUMBA_AVAILABLE = False

Backend = Literal["auto", "numba", "cv2"]

__all__ = [
    "NUMBA_AVAILABLE",
    "projectPoints",
    "projectPoints_cv2",
]


def projectPoints(
    objectPoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    backend: Backend = "auto",
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, None]:
    """Project 3D points with an OpenCV-like API.

    The Numba backend supports float32 and float64 ``objectPoints`` shaped
    ``(N, 3)``, ``(N, 1, 3)``, or ``(1, N, 3)``. The returned image points are
    always shaped ``(N, 1, 2)`` and have the same dtype as ``objectPoints``.
    """

    if backend not in ("auto", "numba", "cv2"):
        raise ValueError(f"Unsupported projectPoints backend: {backend!r}")

    points = _prepare_object_points(objectPoints)
    output_dtype = points.dtype
    rvec64 = _prepare_rvec(rvec)
    _validate_zero_tvec(tvec)
    camera64 = _prepare_camera_matrix(cameraMatrix)
    dist64 = _prepare_dist_coeffs(distCoeffs)

    if backend == "cv2":
        return projectPoints_cv2(points, rvec64, np.zeros((3, 1), dtype=np.float64), camera64, dist64, out=out)

    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available; use backend='cv2'")
        return _project_points_numba(points, rvec64, output_dtype, camera64, dist64, out)

    if NUMBA_AVAILABLE:
        return _project_points_numba(points, rvec64, output_dtype, camera64, dist64, out)
    return projectPoints_cv2(points, rvec64, np.zeros((3, 1), dtype=np.float64), camera64, dist64, out=out)


def projectPoints_cv2(
    objectPoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, None]:
    """OpenCV fallback for ``projectPoints`` that drops the jacobian."""

    points = _prepare_object_points(objectPoints)
    output_dtype = points.dtype
    rvec64 = _prepare_rvec(rvec)
    _validate_zero_tvec(tvec)
    camera64 = _prepare_camera_matrix(cameraMatrix)
    dist64 = _prepare_dist_coeffs(distCoeffs)

    image_points, _ = cv2.projectPoints(
        points,
        rvec64,
        np.zeros((3, 1), dtype=np.float64),
        camera64,
        dist64.reshape(5, 1),
    )
    if image_points.dtype != output_dtype:
        image_points = image_points.astype(output_dtype, copy=False)

    if out is None:
        return image_points, None

    output, out_array = _prepare_output(out, points.shape[0], output_dtype)
    np.copyto(output, image_points)
    return out_array, None


def _project_points_numba(
    points: np.ndarray,
    rvec64: np.ndarray,
    output_dtype: np.dtype,
    camera64: np.ndarray,
    dist64: np.ndarray,
    out: np.ndarray | None,
) -> tuple[np.ndarray, None]:
    rotation64 = cv2.Rodrigues(rvec64)[0]
    output, out_array = _prepare_output(out, points.shape[0], output_dtype)
    rotation = np.ascontiguousarray(rotation64, dtype=np.float64)
    output_flat = output.reshape(points.shape[0], 2)
    fx = float(camera64[0, 0])
    fy = float(camera64[1, 1])
    cx = float(camera64[0, 2])
    cy = float(camera64[1, 2])
    k1 = float(dist64[0])
    k2 = float(dist64[1])
    p1 = float(dist64[2])
    p2 = float(dist64[3])
    k3 = float(dist64[4])

    if output_dtype == np.dtype(np.float32) and p1 == 0.0 and p2 == 0.0:
        if k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
            _project_points_pinhole_float32_kernel(points, rotation, fx, fy, cx, cy, output_flat)
        elif k3 == 0.0:
            _project_points_radial2_float32_kernel(points, rotation, fx, fy, cx, cy, k1, k2, output_flat)
        else:
            _project_points_radial3_float32_kernel(points, rotation, fx, fy, cx, cy, k1, k2, k3, output_flat)
    else:
        _project_points_kernel(points, rotation, fx, fy, cx, cy, k1, k2, p1, p2, k3, output_flat)

    if out_array is not None:
        return out_array, None
    return output, None


def _prepare_object_points(object_points: np.ndarray) -> np.ndarray:
    points = np.asarray(object_points)
    if points.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"objectPoints must have dtype float32 or float64, got {points.dtype}")

    if points.ndim == 2 and points.shape[1] == 3:
        flat = points
    elif points.ndim == 3 and points.shape[1:] == (1, 3):
        flat = points.reshape(points.shape[0], 3)
    elif points.ndim == 3 and points.shape[0] == 1 and points.shape[2] == 3:
        flat = points.reshape(points.shape[1], 3)
    else:
        raise ValueError(f"objectPoints must have shape (N, 3), (N, 1, 3), or (1, N, 3), got {points.shape}")

    if flat.shape[0] == 0:
        raise ValueError("objectPoints must contain at least one point")
    return np.ascontiguousarray(flat)


def _prepare_rvec(rvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rvec, dtype=np.float64).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"rvec must have exactly 3 elements, got shape {np.asarray(rvec).shape}")
    return np.ascontiguousarray(vector.reshape(3, 1))


def _validate_zero_tvec(tvec: np.ndarray) -> None:
    vector = np.asarray(tvec, dtype=np.float64).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"tvec must have exactly 3 elements, got shape {np.asarray(tvec).shape}")
    if not np.all(vector == 0.0):
        raise ValueError("fast projectPoints supports only exactly zero tvec")


def _prepare_camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"cameraMatrix must have shape (3, 3), got {matrix.shape}")
    return np.ascontiguousarray(matrix)


def _prepare_dist_coeffs(dist_coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size != 5:
        raise ValueError(f"distCoeffs must contain exactly 5 elements, got {coeffs.size}")
    return np.ascontiguousarray(coeffs)


def _prepare_output(out: np.ndarray | None, n_points: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray | None]:
    if out is None:
        return np.empty((n_points, 1, 2), dtype=dtype), None

    out_array = np.asarray(out)
    expected_shape = (n_points, 1, 2)
    if out_array.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {out_array.shape}")
    if out_array.dtype != dtype:
        raise ValueError(f"out must have dtype {dtype}, got {out_array.dtype}")
    if not out_array.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    return out_array.reshape(n_points, 1, 2), out_array


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=True)
    def _project_points_pinhole_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        output_flat: np.ndarray,
    ) -> None:
        for idx in prange(points.shape[0]):
            obj_x = float(points[idx, 0])
            obj_y = float(points[idx, 1])
            obj_z = float(points[idx, 2])

            cam_x = rotation[0, 0] * obj_x + rotation[0, 1] * obj_y + rotation[0, 2] * obj_z
            cam_y = rotation[1, 0] * obj_x + rotation[1, 1] * obj_y + rotation[1, 2] * obj_z
            cam_z = rotation[2, 0] * obj_x + rotation[2, 1] * obj_y + rotation[2, 2] * obj_z

            output_flat[idx, 0] = fx * (cam_x / cam_z) + cx
            output_flat[idx, 1] = fy * (cam_y / cam_z) + cy

    @njit(parallel=True, cache=True, fastmath=True)
    def _project_points_radial2_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        output_flat: np.ndarray,
    ) -> None:
        for idx in prange(points.shape[0]):
            obj_x = float(points[idx, 0])
            obj_y = float(points[idx, 1])
            obj_z = float(points[idx, 2])

            cam_x = rotation[0, 0] * obj_x + rotation[0, 1] * obj_y + rotation[0, 2] * obj_z
            cam_y = rotation[1, 0] * obj_x + rotation[1, 1] * obj_y + rotation[1, 2] * obj_z
            cam_z = rotation[2, 0] * obj_x + rotation[2, 1] * obj_y + rotation[2, 2] * obj_z

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            radial = 1.0 + (k1 + k2 * r2) * r2

            output_flat[idx, 0] = fx * x * radial + cx
            output_flat[idx, 1] = fy * y * radial + cy

    @njit(parallel=True, cache=True, fastmath=True)
    def _project_points_radial3_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        output_flat: np.ndarray,
    ) -> None:
        for idx in prange(points.shape[0]):
            obj_x = float(points[idx, 0])
            obj_y = float(points[idx, 1])
            obj_z = float(points[idx, 2])

            cam_x = rotation[0, 0] * obj_x + rotation[0, 1] * obj_y + rotation[0, 2] * obj_z
            cam_y = rotation[1, 0] * obj_x + rotation[1, 1] * obj_y + rotation[1, 2] * obj_z
            cam_z = rotation[2, 0] * obj_x + rotation[2, 1] * obj_y + rotation[2, 2] * obj_z

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            r4 = r2 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2

            output_flat[idx, 0] = fx * x * radial + cx
            output_flat[idx, 1] = fy * y * radial + cy

    @njit(parallel=True, cache=True, fastmath=False)
    def _project_points_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        p1: float,
        p2: float,
        k3: float,
        output_flat: np.ndarray,
    ) -> None:
        for idx in prange(points.shape[0]):
            obj_x = float(points[idx, 0])
            obj_y = float(points[idx, 1])
            obj_z = float(points[idx, 2])

            cam_x = rotation[0, 0] * obj_x + rotation[0, 1] * obj_y + rotation[0, 2] * obj_z
            cam_y = rotation[1, 0] * obj_x + rotation[1, 1] * obj_y + rotation[1, 2] * obj_z
            cam_z = rotation[2, 0] * obj_x + rotation[2, 1] * obj_y + rotation[2, 2] * obj_z

            x = cam_x / cam_z
            y = cam_y / cam_z
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            xy = x * y

            x_distorted = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
            y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy

            output_flat[idx, 0] = fx * x_distorted + cx
            output_flat[idx, 1] = fy * y_distorted + cy

else:

    def _project_points_pinhole_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        output_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _project_points_radial2_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        output_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _project_points_radial3_float32_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        output_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _project_points_kernel(
        points: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        p1: float,
        p2: float,
        k3: float,
        output_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")
