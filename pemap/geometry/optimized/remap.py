"""OpenCV-like remapping for float32 payload images.

The fast path matches this OpenCV mode exactly:

    cv2.remap(src, map1, map2, cv2.INTER_LINEAR, borderValue=np.nan)

Inputs are interpreted as ``src.shape == (H1, W1, ...)``. All trailing axes are
flattened into payload channels internally and restored in the returned array.
Unsupported OpenCV modes are handled by the OpenCV fallback.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

INTER_BITS = 5
INTER_TAB_SIZE = 1 << INTER_BITS
_INTER_LINEAR_WEIGHTS = np.empty((INTER_TAB_SIZE, INTER_TAB_SIZE, 4), dtype=np.float32)
for _ay in range(INTER_TAB_SIZE):
    _fy = np.float32(_ay / INTER_TAB_SIZE)
    for _ax in range(INTER_TAB_SIZE):
        _fx = np.float32(_ax / INTER_TAB_SIZE)
        _INTER_LINEAR_WEIGHTS[_ay, _ax, 0] = (np.float32(1.0) - _fx) * (np.float32(1.0) - _fy)
        _INTER_LINEAR_WEIGHTS[_ay, _ax, 1] = _fx * (np.float32(1.0) - _fy)
        _INTER_LINEAR_WEIGHTS[_ay, _ax, 2] = (np.float32(1.0) - _fx) * _fy
        _INTER_LINEAR_WEIGHTS[_ay, _ax, 3] = _fx * _fy
_INTER_LINEAR_WEIGHT_00 = np.ascontiguousarray(_INTER_LINEAR_WEIGHTS[:, :, 0].reshape(-1))
_INTER_LINEAR_WEIGHT_01 = np.ascontiguousarray(_INTER_LINEAR_WEIGHTS[:, :, 1].reshape(-1))
_INTER_LINEAR_WEIGHT_10 = np.ascontiguousarray(_INTER_LINEAR_WEIGHTS[:, :, 2].reshape(-1))
_INTER_LINEAR_WEIGHT_11 = np.ascontiguousarray(_INTER_LINEAR_WEIGHTS[:, :, 3].reshape(-1))

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
    "INTER_BITS",
    "INTER_TAB_SIZE",
    "NUMBA_AVAILABLE",
    "remap",
    "remap_cv2",
]


def remap(
    src: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    dst: np.ndarray | None = None,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue=np.nan,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Remap ``src`` with an OpenCV-like API and trailing-axis batching.

    The Numba backend is available for ``INTER_LINEAR`` with constant NaN
    borders. Other OpenCV modes use ``remap_cv2`` unless ``backend="numba"``
    is requested, in which case a ``ValueError`` is raised.
    """

    if backend not in ("auto", "numba", "cv2"):
        raise ValueError(f"Unsupported remap backend: {backend!r}")

    numba_supported = _is_numba_supported(interpolation, borderMode, borderValue)
    if backend == "cv2":
        return remap_cv2(src, map1, map2, interpolation, dst, borderMode, borderValue)

    if backend == "numba":
        if not numba_supported:
            raise ValueError("Numba backend supports only cv2.INTER_LINEAR with cv2.BORDER_CONSTANT and all-NaN borderValue")
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available; use backend='cv2'")
        return _remap_numba(src, map1, map2, dst)

    if NUMBA_AVAILABLE and numba_supported:
        return _remap_numba(src, map1, map2, dst)
    return remap_cv2(src, map1, map2, interpolation, dst, borderMode, borderValue)


def remap_cv2(
    src: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    dst: np.ndarray | None = None,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue=np.nan,
) -> np.ndarray:
    """OpenCV fallback for ``remap`` that preserves trailing payload axes."""

    src_flat, map_x, map_y, trailing_shape = _prepare_inputs(src, map1, map2)
    channels = src_flat.shape[2]
    output, dst_array = _prepare_output(dst, map_x.shape, trailing_shape, channels)
    flat_border = _flatten_border_value(borderValue, trailing_shape, channels)

    src_2d = src_flat.reshape(src_flat.shape[0], src_flat.shape[1], channels)
    out_3d = output.reshape(map_x.shape[0], map_x.shape[1], channels)
    for channel in range(channels):
        remapped = cv2.remap(
            src_2d[:, :, channel],
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=borderMode,
            borderValue=float(flat_border[channel]),
        )
        out_3d[:, :, channel] = remapped

    if dst_array is not None:
        return dst_array
    return _restore_output_shape(output, map_x.shape, trailing_shape)


def _remap_numba(src: np.ndarray, map1: np.ndarray, map2: np.ndarray, dst: np.ndarray | None) -> np.ndarray:
    src_flat, map_x, map_y, trailing_shape = _prepare_inputs(src, map1, map2)
    src_h, src_w, channels = src_flat.shape
    output, dst_array = _prepare_output(dst, map_x.shape, trailing_shape, channels)
    _remap_linear_numba_kernel(
        src_flat.reshape(src_h * src_w, channels),
        src_w,
        map_x.reshape(-1),
        map_y.reshape(-1),
        _INTER_LINEAR_WEIGHT_00,
        _INTER_LINEAR_WEIGHT_01,
        _INTER_LINEAR_WEIGHT_10,
        _INTER_LINEAR_WEIGHT_11,
        output,
    )
    if dst_array is not None:
        return dst_array
    return _restore_output_shape(output, map_x.shape, trailing_shape)


def _prepare_inputs(src: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    image = np.asarray(src, dtype=np.float32)
    if image.ndim < 2:
        raise ValueError(f"src must have shape (H, W, ...), got {image.shape}")

    src_h, src_w = image.shape[:2]
    if src_h < 2 or src_w < 2:
        raise ValueError(f"src height and width must be at least 2, got {(src_h, src_w)}")

    map_x = np.asarray(map1, dtype=np.float32)
    map_y = np.asarray(map2, dtype=np.float32)
    if map_x.ndim != 2 or map_y.ndim != 2:
        raise ValueError(f"map1 and map2 must both have shape (H2, W2), got {map_x.shape} and {map_y.shape}")
    if map_x.shape != map_y.shape:
        raise ValueError(f"map1 and map2 must have the same shape, got {map_x.shape} and {map_y.shape}")

    trailing_shape = tuple(int(dim) for dim in image.shape[2:])
    channels = int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
    src_flat = np.ascontiguousarray(image.reshape(src_h, src_w, channels))
    return src_flat, np.ascontiguousarray(map_x), np.ascontiguousarray(map_y), trailing_shape


def _prepare_output(
    dst: np.ndarray | None,
    map_shape: tuple[int, int],
    trailing_shape: tuple[int, ...],
    channels: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if dst is None:
        return np.empty((int(map_shape[0]) * int(map_shape[1]), channels), dtype=np.float32), None

    dst_array = np.asarray(dst)
    expected_shape = map_shape + trailing_shape if trailing_shape else map_shape
    if dst_array.shape != expected_shape:
        raise ValueError(f"dst must have shape {expected_shape}, got {dst_array.shape}")
    if dst_array.dtype != np.float32:
        raise ValueError(f"dst must have dtype float32, got {dst_array.dtype}")
    if not dst_array.flags.c_contiguous:
        raise ValueError("dst must be C-contiguous")
    return dst_array.reshape(int(map_shape[0]) * int(map_shape[1]), channels), dst_array


def _restore_output_shape(output: np.ndarray, map_shape: tuple[int, int], trailing_shape: tuple[int, ...]) -> np.ndarray:
    if trailing_shape:
        return output.reshape(map_shape + trailing_shape)
    return output.reshape(map_shape)


def _flatten_border_value(border_value, trailing_shape: tuple[int, ...], channels: int) -> np.ndarray:
    values = np.asarray(border_value, dtype=np.float32)
    if values.ndim == 0:
        return np.full(channels, values.item(), dtype=np.float32)

    flat = values.reshape(-1)
    if flat.size == 1:
        return np.full(channels, flat[0], dtype=np.float32)
    if flat.size == channels:
        return np.ascontiguousarray(flat, dtype=np.float32)
    if trailing_shape and flat.size == trailing_shape[0]:
        suffix_channels = channels // trailing_shape[0]
        return np.ascontiguousarray(np.repeat(flat, suffix_channels), dtype=np.float32)
    raise ValueError(f"borderValue must be scalar, have {channels} flattened values, or match first trailing axis; got {flat.size} value(s)")


def _is_numba_supported(interpolation: int, border_mode: int, border_value) -> bool:
    if interpolation != cv2.INTER_LINEAR or border_mode != cv2.BORDER_CONSTANT:
        return False
    try:
        values = np.asarray(border_value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(values.size > 0 and np.all(np.isnan(values)))


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _remap_linear_numba_kernel(
        src_flat: np.ndarray,
        src_w: int,
        map_x: np.ndarray,
        map_y: np.ndarray,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        src_h = src_flat.shape[0] // src_w
        channels = src_flat.shape[1]
        tab_size = INTER_TAB_SIZE
        nan = np.float32(np.nan)

        for p in prange(map_x.size):
            x = map_x[p]
            y = map_y[p]
            valid = True
            sx = 0
            sy = 0
            ax = 0
            ay = 0
            if not (np.isfinite(x) and np.isfinite(y)):
                valid = False
            else:
                sx = int(np.floor(x))
                sy = int(np.floor(y))
                ax = int(np.rint(np.float32(x - np.float32(sx)) * np.float32(tab_size)))
                ay = int(np.rint(np.float32(y - np.float32(sy)) * np.float32(tab_size)))

                if ax == tab_size:
                    sx += 1
                    ax = 0
                if ay == tab_size:
                    sy += 1
                    ay = 0

                if sx < 0 or sy < 0 or sx + 1 >= src_w or sy + 1 >= src_h:
                    valid = False

            if valid:
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

                for c in range(channels):
                    output[p, c] = src_flat[src00, c] * w00 + src_flat[src01, c] * w01 + src_flat[src10, c] * w10 + src_flat[src11, c] * w11
            else:
                for c in range(channels):
                    output[p, c] = nan

else:

    def _remap_linear_numba_kernel(
        src_flat: np.ndarray,
        src_w: int,
        map_x: np.ndarray,
        map_y: np.ndarray,
        weight_00: np.ndarray,
        weight_01: np.ndarray,
        weight_10: np.ndarray,
        weight_11: np.ndarray,
        output: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")
