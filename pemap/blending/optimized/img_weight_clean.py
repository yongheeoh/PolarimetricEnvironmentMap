"""Fast cleanup for multiband blend image and hard-owner weight stacks."""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    from numba import njit
    from numba import prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent/broken
    njit = None
    prange = range
    NUMBA_AVAILABLE = False

Backend = Literal["auto", "numba", "numpy"]

__all__ = [
    "NUMBA_AVAILABLE",
    "clean_images",
    "clean_images_and_weights",
    "clean_images_and_weights_numpy",
    "clean_weights",
    "clean_weights_numpy",
]


def clean_images_and_weights(
    images_equi: np.ndarray,
    weights_equi: np.ndarray,
    *,
    backend: Backend = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-cleaned images and hard argmax ownership weights.

    This preserves the original NumPy cleanup semantics used by
    ``pemap.blend._multiband_blend``:

    * image NaNs become 0.0 via ``np.nan_to_num(..., nan=0.0)``;
    * weight NaNs become 0.0 for the argmax calculation only;
    * argmax ties keep the first view;
    * if the winning source weight was NaN, the final clean weight is 0.0.
    """

    if backend not in ("auto", "numba", "numpy"):
        raise ValueError(f"Unsupported cleanup backend: {backend!r}")

    if backend == "numpy":
        return clean_images_and_weights_numpy(images_equi, weights_equi)

    if NUMBA_AVAILABLE:
        return _clean_images_and_weights_numba(images_equi, weights_equi)

    if backend == "numba":
        raise RuntimeError("Numba is not available; use backend='numpy'")
    return clean_images_and_weights_numpy(images_equi, weights_equi)


def clean_images(
    images_equi: np.ndarray,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Return image data with NaNs replaced by 0.0 and infinities saturated."""

    if backend not in ("auto", "numba", "numpy"):
        raise ValueError(f"Unsupported cleanup backend: {backend!r}")

    images = np.asarray(images_equi, dtype=np.float32)
    if backend == "numpy":
        return np.nan_to_num(images, nan=0.0).astype(np.float32, copy=False)

    if NUMBA_AVAILABLE:
        images_clean = np.empty_like(images, dtype=np.float32)
        if images.ndim == 4 and not images.flags.c_contiguous:
            _clean_images_4d_kernel(images, images_clean)
        else:
            images_contiguous = np.ascontiguousarray(images)
            if images_contiguous.shape != images_clean.shape:
                raise ValueError(f"Internal cleanup shape mismatch: {images_contiguous.shape} != {images_clean.shape}")
            _clean_images_kernel(images_contiguous.reshape(-1), images_clean.reshape(-1))
        return images_clean

    if backend == "numba":
        raise RuntimeError("Numba is not available; use backend='numpy'")
    return np.nan_to_num(images, nan=0.0).astype(np.float32, copy=False)


def clean_weights(
    weights_equi: np.ndarray,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Return hard argmax ownership weights with NaN winners cleared to 0.0."""

    if backend not in ("auto", "numba", "numpy"):
        raise ValueError(f"Unsupported cleanup backend: {backend!r}")

    weights = _prepare_weights(weights_equi)
    if backend == "numpy":
        return clean_weights_numpy(weights)

    if NUMBA_AVAILABLE:
        return _clean_weights_numba(weights)

    if backend == "numba":
        raise RuntimeError("Numba is not available; use backend='numpy'")
    return clean_weights_numpy(weights)


def clean_images_and_weights_numpy(images_equi: np.ndarray, weights_equi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy implementation matching the previous blend cleanup block."""

    images, weights = _prepare_inputs(images_equi, weights_equi)
    num_views = images.shape[0]
    one_hot = np.zeros_like(weights, dtype=np.float32)
    argmax_indices = np.argmax(np.nan_to_num(weights, nan=0.0), axis=0)
    for view_idx in range(num_views):
        one_hot[view_idx] = (argmax_indices == view_idx).astype(np.float32)
    one_hot[np.isnan(weights)] = np.nan

    images_clean = np.nan_to_num(images, nan=0.0).astype(np.float32, copy=False)
    weights_clean = np.nan_to_num(one_hot, nan=0.0).astype(np.float32, copy=False)
    return images_clean, weights_clean


def clean_weights_numpy(weights_equi: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation for hard argmax ownership weights."""

    weights = _prepare_weights(weights_equi)
    num_views = weights.shape[0]
    one_hot = np.zeros_like(weights, dtype=np.float32)
    argmax_indices = np.argmax(np.nan_to_num(weights, nan=0.0), axis=0)
    for view_idx in range(num_views):
        one_hot[view_idx] = (argmax_indices == view_idx).astype(np.float32)
    one_hot[np.isnan(weights)] = np.nan
    return np.nan_to_num(one_hot, nan=0.0).astype(np.float32, copy=False)


def _prepare_inputs(images_equi: np.ndarray, weights_equi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(images_equi, dtype=np.float32)
    weights = np.asarray(weights_equi, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected images_equi shape (N, H, W, C), got {images.shape}")
    if weights.shape != images.shape:
        raise ValueError(f"weights_equi shape {weights.shape} must match images_equi shape {images.shape}")
    return np.ascontiguousarray(images), np.ascontiguousarray(weights)


def _prepare_weights(weights_equi: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights_equi, dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError(f"Expected weights_equi shape (N, H, W, C), got {weights.shape}")
    return np.ascontiguousarray(weights)


def _clean_images_and_weights_numba(images_equi: np.ndarray, weights_equi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images, weights = _prepare_inputs(images_equi, weights_equi)
    images_clean = np.empty_like(images, dtype=np.float32)
    weights_clean = np.empty_like(weights, dtype=np.float32)
    owner_dtype = np.int8 if weights.shape[0] <= np.iinfo(np.int8).max else np.int16
    owners = np.empty(weights.shape[1] * weights.shape[2] * weights.shape[3], dtype=owner_dtype)

    _clean_images_kernel(images.reshape(-1), images_clean.reshape(-1))
    weights_flat = weights.reshape(weights.shape[0], -1)
    _clean_weight_owners_kernel(weights_flat, owners)
    _write_weights_from_owners_kernel(owners, weights_clean.reshape(weights.shape[0], -1))
    return images_clean, weights_clean


def _clean_weights_numba(weights_equi: np.ndarray) -> np.ndarray:
    weights = _prepare_weights(weights_equi)
    weights_clean = np.empty_like(weights, dtype=np.float32)
    owner_dtype = np.int8 if weights.shape[0] <= np.iinfo(np.int8).max else np.int16
    owners = np.empty(weights.shape[1] * weights.shape[2] * weights.shape[3], dtype=owner_dtype)

    weights_flat = weights.reshape(weights.shape[0], -1)
    _clean_weight_owners_kernel(weights_flat, owners)
    _write_weights_from_owners_kernel(owners, weights_clean.reshape(weights.shape[0], -1))
    return weights_clean


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _clean_images_kernel(images_flat: np.ndarray, images_clean_flat: np.ndarray) -> None:
        inf = np.float32(np.inf)
        ninf = np.float32(-np.inf)
        float32_max = np.float32(3.4028234663852886e38)
        float32_min = np.float32(-3.4028234663852886e38)

        for idx in prange(images_flat.size):
            value = images_flat[idx]
            if np.isnan(value):
                images_clean_flat[idx] = np.float32(0.0)
            elif value == inf:
                images_clean_flat[idx] = float32_max
            elif value == ninf:
                images_clean_flat[idx] = float32_min
            else:
                images_clean_flat[idx] = value

    @njit(parallel=True, cache=True, fastmath=False)
    def _clean_images_4d_kernel(images: np.ndarray, images_clean: np.ndarray) -> None:
        inf = np.float32(np.inf)
        ninf = np.float32(-np.inf)
        float32_max = np.float32(3.4028234663852886e38)
        float32_min = np.float32(-3.4028234663852886e38)
        num_views = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        channels = images.shape[3]
        total_pixels = num_views * height * width

        for pixel_idx in prange(total_pixels):
            x = pixel_idx % width
            row_idx = pixel_idx // width
            y = row_idx % height
            view_idx = row_idx // height

            for channel in range(channels):
                value = images[view_idx, y, x, channel]
                if np.isnan(value):
                    images_clean[view_idx, y, x, channel] = np.float32(0.0)
                elif value == inf:
                    images_clean[view_idx, y, x, channel] = float32_max
                elif value == ninf:
                    images_clean[view_idx, y, x, channel] = float32_min
                else:
                    images_clean[view_idx, y, x, channel] = value

    @njit(parallel=True, cache=True, fastmath=False)
    def _clean_weight_owners_kernel(weights_flat: np.ndarray, owners: np.ndarray) -> None:
        num_views = weights_flat.shape[0]
        total = weights_flat.shape[1]
        inf = np.float32(np.inf)
        ninf = np.float32(-np.inf)
        float32_max = np.float32(3.4028234663852886e38)
        float32_min = np.float32(-3.4028234663852886e38)

        for idx in prange(total):
            max_idx = 0
            max_value = weights_flat[0, idx]
            if np.isnan(max_value):
                max_value = np.float32(0.0)
            elif max_value == inf:
                max_value = float32_max
            elif max_value == ninf:
                max_value = float32_min

            for view_idx in range(1, num_views):
                value = weights_flat[view_idx, idx]
                if np.isnan(value):
                    value = np.float32(0.0)
                elif value == inf:
                    value = float32_max
                elif value == ninf:
                    value = float32_min

                if value > max_value:
                    max_value = value
                    max_idx = view_idx

            if not np.isnan(weights_flat[max_idx, idx]):
                owners[idx] = max_idx
            else:
                owners[idx] = -1

    @njit(parallel=True, cache=True, fastmath=False)
    def _write_weights_from_owners_kernel(owners: np.ndarray, weights_clean_flat: np.ndarray) -> None:
        num_views = weights_clean_flat.shape[0]
        total = weights_clean_flat.shape[1]

        for view_idx in prange(num_views):
            for idx in range(total):
                weights_clean_flat[view_idx, idx] = np.float32(1.0) if owners[idx] == view_idx else np.float32(0.0)

else:

    def _clean_images_kernel(images_flat: np.ndarray, images_clean_flat: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")

    def _clean_images_4d_kernel(images: np.ndarray, images_clean: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")

    def _clean_weight_owners_kernel(weights_flat: np.ndarray, owners: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")

    def _write_weights_from_owners_kernel(owners: np.ndarray, weights_clean_flat: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")
