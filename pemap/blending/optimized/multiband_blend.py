"""Fast post-cleanup multiband blending helpers."""

from __future__ import annotations

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

__all__ = [
    "NUMBA_AVAILABLE",
    "build_weight_pyramid_accumulators",
    "multiband_blend_after_clean_fast",
    "multiband_blend_after_clean_with_weight_pyramid_fast",
]


def multiband_blend_after_clean_fast(images_clean: np.ndarray, weights_clean: np.ndarray, level: int) -> np.ndarray:
    """Blend already-cleaned image/weight stacks using streaming accumulators.

    ``images_clean`` and ``weights_clean`` must match the outputs from
    ``clean_images_and_weights``: contiguous float32 arrays with shape
    ``(N, H, W, C)``.
    """

    images, weights = _prepare_clean_inputs(images_clean, weights_clean, level)
    pyramid_shapes = _pyramid_shapes(images.shape[1:], level)

    numerators = [np.zeros(shape, dtype=np.float32) for shape in pyramid_shapes]
    denominators = [np.zeros(shape, dtype=np.float32) for shape in pyramid_shapes]
    numerator_flats = [numerator.reshape(-1) for numerator in numerators]
    denominator_flats = [denominator.reshape(-1) for denominator in denominators]
    image_pyr_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[1:]]
    weight_pyr_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[1:]]
    up_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[:-1]]

    for view_idx in range(images.shape[0]):
        img = images[view_idx]
        weight = weights[view_idx]

        for level_idx in range(level):
            if level_idx < level - 1:
                next_img = image_pyr_buffers[level_idx]
                next_weight = weight_pyr_buffers[level_idx]
                up = up_buffers[level_idx]

                cv2.pyrDown(img, dst=next_img, borderType=cv2.BORDER_WRAP)
                cv2.pyrUp(next_img, dst=up, dstsize=(img.shape[1], img.shape[0]))
                _accumulate_laplacian_band(
                    img.reshape(-1),
                    up.reshape(-1),
                    weight.reshape(-1),
                    numerator_flats[level_idx],
                    denominator_flats[level_idx],
                )
                cv2.pyrDown(weight, dst=next_weight, borderType=cv2.BORDER_WRAP)
                img = next_img
                weight = next_weight
            else:
                _accumulate_weighted_band(
                    img.reshape(-1),
                    weight.reshape(-1),
                    numerator_flats[level_idx],
                    denominator_flats[level_idx],
                )

    for level_idx in range(level):
        _normalize_band(numerator_flats[level_idx], denominator_flats[level_idx])

    img = numerators[-1]
    for level_idx in range(level - 2, -1, -1):
        up = up_buffers[level_idx]
        cv2.pyrUp(img, dst=up, dstsize=(numerators[level_idx].shape[1], numerators[level_idx].shape[0]))
        cv2.add(up, numerators[level_idx], up)
        img = up

    return img.astype(np.float32, copy=False)


def build_weight_pyramid_accumulators(weights_clean: np.ndarray, level: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Precompute Gaussian weight pyramids and their per-level denominators."""

    weights = _prepare_clean_weights(weights_clean, level)
    pyramid_shapes = _pyramid_shapes(weights.shape[1:], level)
    weight_pyramids = [weights]
    weight_pyramids.extend(np.empty((weights.shape[0], *shape), dtype=np.float32) for shape in pyramid_shapes[1:])
    denominators = [np.zeros(shape, dtype=np.float32) for shape in pyramid_shapes]
    denominator_flats = [denominator.reshape(-1) for denominator in denominators]
    weight_pyr_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[1:]]

    for view_idx in range(weights.shape[0]):
        weight = weights[view_idx]
        _accumulate_weight(weight.reshape(-1), denominator_flats[0])

        for level_idx in range(1, level):
            next_weight = weight_pyr_buffers[level_idx - 1]
            cv2.pyrDown(weight, dst=next_weight, borderType=cv2.BORDER_WRAP)
            weight_pyramids[level_idx][view_idx] = next_weight
            _accumulate_weight(next_weight.reshape(-1), denominator_flats[level_idx])
            weight = next_weight

    return weight_pyramids, denominators


def multiband_blend_after_clean_with_weight_pyramid_fast(
    images_clean: np.ndarray,
    weight_pyramids: list[np.ndarray],
    denominators: list[np.ndarray],
) -> np.ndarray:
    """Blend cleaned images using precomputed Gaussian weight pyramids."""

    images = _prepare_clean_images_for_weight_pyramid(images_clean, weight_pyramids, denominators)
    level = len(weight_pyramids)
    pyramid_shapes = _pyramid_shapes(images.shape[1:], level)

    numerators = [np.zeros(shape, dtype=np.float32) for shape in pyramid_shapes]
    numerator_flats = [numerator.reshape(-1) for numerator in numerators]
    image_pyr_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[1:]]
    up_buffers = [np.empty(shape, dtype=np.float32) for shape in pyramid_shapes[:-1]]

    for view_idx in range(images.shape[0]):
        img = images[view_idx]

        for level_idx in range(level):
            weight = weight_pyramids[level_idx][view_idx]
            if level_idx < level - 1:
                next_img = image_pyr_buffers[level_idx]
                up = up_buffers[level_idx]

                cv2.pyrDown(img, dst=next_img, borderType=cv2.BORDER_WRAP)
                cv2.pyrUp(next_img, dst=up, dstsize=(img.shape[1], img.shape[0]))
                _accumulate_laplacian_numerator(
                    img.reshape(-1),
                    up.reshape(-1),
                    weight.reshape(-1),
                    numerator_flats[level_idx],
                )
                img = next_img
            else:
                _accumulate_weighted_numerator(
                    img.reshape(-1),
                    weight.reshape(-1),
                    numerator_flats[level_idx],
                )

    for level_idx in range(level):
        _normalize_band(numerator_flats[level_idx], denominators[level_idx].reshape(-1))

    img = numerators[-1]
    for level_idx in range(level - 2, -1, -1):
        up = up_buffers[level_idx]
        cv2.pyrUp(img, dst=up, dstsize=(numerators[level_idx].shape[1], numerators[level_idx].shape[0]))
        cv2.add(up, numerators[level_idx], up)
        img = up

    return img.astype(np.float32, copy=False)


def _prepare_clean_inputs(images_clean: np.ndarray, weights_clean: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    if level < 1:
        raise ValueError(f"Expected at least one multiband level, got {level}")

    images = np.asarray(images_clean, dtype=np.float32)
    weights = np.asarray(weights_clean, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected images_clean shape (N, H, W, C), got {images.shape}")
    if weights.shape != images.shape:
        raise ValueError(f"weights_clean shape {weights.shape} must match images_clean shape {images.shape}")

    return np.ascontiguousarray(images), np.ascontiguousarray(weights)


def _prepare_clean_weights(weights_clean: np.ndarray, level: int) -> np.ndarray:
    if level < 1:
        raise ValueError(f"Expected at least one multiband level, got {level}")

    weights = np.asarray(weights_clean, dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError(f"Expected weights_clean shape (N, H, W, C), got {weights.shape}")
    return np.ascontiguousarray(weights)


def _prepare_clean_images_for_weight_pyramid(
    images_clean: np.ndarray,
    weight_pyramids: list[np.ndarray],
    denominators: list[np.ndarray],
) -> np.ndarray:
    if len(weight_pyramids) == 0:
        raise ValueError("Expected at least one precomputed weight pyramid level")
    if len(weight_pyramids) != len(denominators):
        raise ValueError("weight_pyramids and denominators must have the same number of levels")

    images = np.asarray(images_clean, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected images_clean shape (N, H, W, C), got {images.shape}")
    if weight_pyramids[0].shape != images.shape:
        raise ValueError(f"weight_pyramids[0] shape {weight_pyramids[0].shape} must match images_clean shape {images.shape}")

    expected_shapes = _pyramid_shapes(images.shape[1:], len(weight_pyramids))
    for level_idx, (weights_level, denominator) in enumerate(zip(weight_pyramids, denominators, strict=True)):
        expected_weight_shape = (images.shape[0], *expected_shapes[level_idx])
        if weights_level.shape != expected_weight_shape:
            raise ValueError(f"weight_pyramids[{level_idx}] shape {weights_level.shape} must be {expected_weight_shape}")
        if denominator.shape != expected_shapes[level_idx]:
            raise ValueError(f"denominators[{level_idx}] shape {denominator.shape} must be {expected_shapes[level_idx]}")

    return np.ascontiguousarray(images)


def _pyramid_shapes(base_shape: tuple[int, int, int], level: int) -> list[tuple[int, int, int]]:
    height, width, channels = base_shape
    shapes = []
    for _ in range(level):
        shapes.append((height, width, channels))
        height = (height + 1) // 2
        width = (width + 1) // 2
    return shapes


def _accumulate_weighted_band(
    band_flat: np.ndarray,
    weight_flat: np.ndarray,
    numerator_flat: np.ndarray,
    denominator_flat: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_weighted_band_kernel(band_flat, weight_flat, numerator_flat, denominator_flat)
        return

    numerator_flat += band_flat * weight_flat
    denominator_flat += weight_flat


def _accumulate_weighted_numerator(
    band_flat: np.ndarray,
    weight_flat: np.ndarray,
    numerator_flat: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_weighted_numerator_kernel(band_flat, weight_flat, numerator_flat)
        return

    numerator_flat += band_flat * weight_flat


def _accumulate_laplacian_band(
    img_flat: np.ndarray,
    up_flat: np.ndarray,
    weight_flat: np.ndarray,
    numerator_flat: np.ndarray,
    denominator_flat: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_laplacian_band_kernel(img_flat, up_flat, weight_flat, numerator_flat, denominator_flat)
        return

    numerator_flat += (img_flat - up_flat) * weight_flat
    denominator_flat += weight_flat


def _accumulate_laplacian_numerator(
    img_flat: np.ndarray,
    up_flat: np.ndarray,
    weight_flat: np.ndarray,
    numerator_flat: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_laplacian_numerator_kernel(img_flat, up_flat, weight_flat, numerator_flat)
        return

    numerator_flat += (img_flat - up_flat) * weight_flat


def _accumulate_weight(weight_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_weight_kernel(weight_flat, denominator_flat)
        return

    denominator_flat += weight_flat


def _normalize_band(numerator_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
    if NUMBA_AVAILABLE:
        _normalize_band_kernel(numerator_flat, denominator_flat)
        return

    numerator_flat /= denominator_flat + np.float32(1e-8)


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _accumulate_weighted_band_kernel(
        band_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
        denominator_flat: np.ndarray,
    ) -> None:
        for idx in prange(band_flat.size):
            weight = weight_flat[idx]
            numerator_flat[idx] = numerator_flat[idx] + band_flat[idx] * weight
            denominator_flat[idx] = denominator_flat[idx] + weight

    @njit(parallel=True, cache=True, fastmath=False)
    def _accumulate_weighted_numerator_kernel(
        band_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
    ) -> None:
        for idx in prange(band_flat.size):
            numerator_flat[idx] = numerator_flat[idx] + band_flat[idx] * weight_flat[idx]

    @njit(parallel=True, cache=True, fastmath=False)
    def _accumulate_laplacian_band_kernel(
        img_flat: np.ndarray,
        up_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
        denominator_flat: np.ndarray,
    ) -> None:
        for idx in prange(img_flat.size):
            weight = weight_flat[idx]
            numerator_flat[idx] = numerator_flat[idx] + (img_flat[idx] - up_flat[idx]) * weight
            denominator_flat[idx] = denominator_flat[idx] + weight

    @njit(parallel=True, cache=True, fastmath=False)
    def _accumulate_laplacian_numerator_kernel(
        img_flat: np.ndarray,
        up_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
    ) -> None:
        for idx in prange(img_flat.size):
            numerator_flat[idx] = numerator_flat[idx] + (img_flat[idx] - up_flat[idx]) * weight_flat[idx]

    @njit(parallel=True, cache=True, fastmath=False)
    def _accumulate_weight_kernel(weight_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
        for idx in prange(weight_flat.size):
            denominator_flat[idx] = denominator_flat[idx] + weight_flat[idx]

    @njit(parallel=True, cache=True, fastmath=False)
    def _normalize_band_kernel(numerator_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
        eps = np.float32(1e-8)
        for idx in prange(numerator_flat.size):
            numerator_flat[idx] = numerator_flat[idx] / (denominator_flat[idx] + eps)

else:

    def _accumulate_weighted_band_kernel(
        band_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
        denominator_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _normalize_band_kernel(numerator_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")

    def _accumulate_weighted_numerator_kernel(
        band_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _accumulate_laplacian_band_kernel(
        img_flat: np.ndarray,
        up_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
        denominator_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _accumulate_laplacian_numerator_kernel(
        img_flat: np.ndarray,
        up_flat: np.ndarray,
        weight_flat: np.ndarray,
        numerator_flat: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available")

    def _accumulate_weight_kernel(weight_flat: np.ndarray, denominator_flat: np.ndarray) -> None:
        raise RuntimeError("Numba is not available")
