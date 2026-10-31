from time import perf_counter
from tqdm.auto import tqdm
from pemap.geometry.warp import WarpedViews
from pemap.geometry import _validate_payload_shape
import cv2
import numpy as np

from pemap.blending.optimized.img_weight_clean import clean_images, clean_images_and_weights, clean_weights
from pemap.blending.optimized.multiband_blend import build_weight_pyramid_accumulators, multiband_blend_after_clean_fast, multiband_blend_after_clean_with_weight_pyramid_fast


def _validate_blend_method(blend_method: str) -> None:
    """Raise ValueError if blend_method is not ``'average'`` or ``'multiband'``."""
    if blend_method not in {"average", "multiband"}:
        raise ValueError(f"Unsupported blend method: {blend_method}")


def _build_gaussian_pyr(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a ``levels``-level Gaussian pyramid with BORDER_WRAP downsampling."""
    pyramid = [img]
    for _ in range(1, levels):
        pyramid.append(cv2.pyrDown(pyramid[-1], borderType=cv2.BORDER_WRAP))
    return pyramid


def _build_laplacian_pyr(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Laplacian pyramid; the last level is the coarsest Gaussian residual.

    Parameters
    ----------
    img : np.ndarray, (H, W, C), float32
        Input image.
    levels : int
        Number of pyramid levels.

    Returns
    -------
    list[np.ndarray]
        ``levels`` arrays; levels 0..levels-2 are Laplacian bands,
        level levels-1 is the coarsest Gaussian.
    """
    gaussian = _build_gaussian_pyr(img, levels)
    laplacian: list[np.ndarray] = []
    for idx in range(levels - 1):
        up = cv2.pyrUp(gaussian[idx + 1], dstsize=(gaussian[idx].shape[1], gaussian[idx].shape[0]))
        laplacian.append(gaussian[idx] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def _multiband_blend_after_clean_reference(images_clean: np.ndarray, weights_clean: np.ndarray, level: int) -> np.ndarray:
    """Reference post-cleanup multiband blend implementation."""
    num_views = images_clean.shape[0]

    pyr_images = [_build_laplacian_pyr(images_clean[idx], level) for idx in range(num_views)]
    pyr_weights = [_build_gaussian_pyr(weights_clean[idx], level) for idx in range(num_views)]

    blended_pyr: list[np.ndarray] = []
    for level_idx in range(level):
        images_level = np.stack([pyr_images[idx][level_idx] for idx in range(num_views)], axis=0)
        weights_level = np.stack([pyr_weights[idx][level_idx] for idx in range(num_views)], axis=0)
        numerator = np.sum(images_level * weights_level, axis=0)
        denominator = np.sum(weights_level, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            blended_pyr.append(numerator / (denominator + 1e-8))

    img = blended_pyr[-1]
    for idx in range(len(blended_pyr) - 2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(blended_pyr[idx].shape[1], blended_pyr[idx].shape[0]))
        img = up + blended_pyr[idx]
    img_blended = img.astype(np.float32, copy=False)
    return img_blended


def _multiband_blend_reference(images_equi: np.ndarray, weights_equi: np.ndarray, level: int) -> np.ndarray:
    """Reference full multiband blend path retained for correctness checks."""
    if images_equi.ndim != 4:
        raise ValueError(f"Expected images_equi shape (N, H, W, C), got {images_equi.shape}")
    if weights_equi.shape != images_equi.shape:
        raise ValueError(f"weights_equi shape {weights_equi.shape} must match images_equi shape {images_equi.shape}")

    weights_valid = _mask_weights_by_image_validity(images_equi, weights_equi)
    images_clean, weights_clean = clean_images_and_weights(images_equi, weights_valid)
    return _multiband_blend_after_clean_reference(images_clean, weights_clean, level)


def _multiband_blend_components(
    images_equi: np.ndarray,
    weights_equi: np.ndarray,
    level: int,
    *,
    nonnegative_component_indices: tuple[int, ...] = (),
    image_cleaner=None,
) -> np.ndarray:
    """Blend a payload stack shaped (N, H, W, C, S), reusing cleaned weights."""
    if images_equi.ndim != 5:
        raise ValueError(f"Expected images_equi shape (N, H, W, C, S), got {images_equi.shape}")
    if weights_equi.shape != images_equi.shape[:-1]:
        raise ValueError(f"weights_equi shape {weights_equi.shape} must match images_equi shape prefix {images_equi.shape[:-1]}")

    images = np.asarray(images_equi, dtype=np.float32)
    weights_valid = _mask_weights_by_image_validity(images, weights_equi)
    weights_clean = clean_weights(weights_valid)
    weight_pyramids, denominators = build_weight_pyramid_accumulators(weights_clean, level)
    img_blended = np.empty(images.shape[1:], dtype=np.float32)

    num_components = images.shape[-1]
    nonnegative_components = _resolve_component_indices(nonnegative_component_indices, num_components)
    cleaner = image_cleaner or clean_images
    for component_idx in range(num_components):
        images_component = images[:, :, :, :, component_idx]
        if component_idx in nonnegative_components:
            img_blended[:, :, :, component_idx] = _multiband_blend_nonnegative_component(
                images_component,
                weights_valid,
                level,
                image_cleaner=cleaner,
            )
        else:
            images_clean = cleaner(images_component)
            img_blended[:, :, :, component_idx] = multiband_blend_after_clean_with_weight_pyramid_fast(images_clean, weight_pyramids, denominators)

    return img_blended


def _resolve_component_indices(component_indices: tuple[int, ...], num_components: int) -> set[int]:
    resolved: set[int] = set()
    for component_idx in component_indices:
        resolved_idx = component_idx + num_components if component_idx < 0 else component_idx
        if resolved_idx < 0 or resolved_idx >= num_components:
            raise ValueError(f"Component index {component_idx} is out of range for {num_components} components")
        resolved.add(resolved_idx)
    return resolved


def _multiband_blend_nonnegative_component(
    images_component: np.ndarray,
    weights_valid: np.ndarray,
    level: int,
    image_cleaner=None,
) -> np.ndarray:
    """Blend a nonnegative component in log space and return finite uncovered pixels as zero."""
    component_valid = np.isfinite(images_component) & (images_component >= 0.0)
    component_weights = np.where(component_valid, weights_valid, np.nan).astype(np.float32, copy=False)
    weights_clean = clean_weights(component_weights)
    weight_pyramids, denominators = build_weight_pyramid_accumulators(weights_clean, level)

    positive_floor = np.float32(np.finfo(np.float32).tiny)
    log_images = np.full_like(images_component, np.nan, dtype=np.float32)
    log_images[component_valid] = np.log(np.maximum(images_component[component_valid], positive_floor)).astype(np.float32, copy=False)
    log_images_clean = (image_cleaner or clean_images)(log_images)
    log_blended = multiband_blend_after_clean_with_weight_pyramid_fast(log_images_clean, weight_pyramids, denominators)

    max_log = np.float32(np.log(np.finfo(np.float32).max))
    log_blended = np.minimum(log_blended, max_log)
    blended = np.exp(log_blended).astype(np.float32, copy=False)

    coverage = np.sum(weights_clean, axis=0) > 0.0
    blended[~coverage] = 0.0
    np.maximum(blended, 0.0, out=blended)
    return blended


def _mask_weights_by_image_validity(images_equi: np.ndarray, weights_equi: np.ndarray) -> np.ndarray:
    """Clear blend weights where the corresponding image payload is invalid."""
    images = np.asarray(images_equi, dtype=np.float32)
    weights = np.asarray(weights_equi, dtype=np.float32)

    if images.ndim == weights.ndim:
        if weights.shape != images.shape:
            raise ValueError(f"weights_equi shape {weights.shape} must match images_equi shape {images.shape}")
        image_valid = np.isfinite(images)
    elif images.ndim == weights.ndim + 1:
        if weights.shape != images.shape[:-1]:
            raise ValueError(f"weights_equi shape {weights.shape} must match images_equi shape prefix {images.shape[:-1]}")
        image_valid = np.all(np.isfinite(images), axis=-1)
    else:
        raise ValueError(f"Expected images_equi ndim {weights.ndim} or {weights.ndim + 1}, got {images.ndim}")

    return np.where(image_valid, weights, np.nan).astype(np.float32, copy=False)


def _multiband_blend(images_equi: np.ndarray, weights_equi: np.ndarray, level: int) -> np.ndarray:
    """Blend N views at each pyramid level using hard argmax ownership masks.

    Parameters
    ----------
    images_equi : np.ndarray, (N, H, W, C), float32
        Stack of equirectangular images. NaN = invalid.
    weights_equi : np.ndarray, (N, H, W, C), float32
        Per-pixel blend weights; same shape as images_equi.
    level : int
        Number of Laplacian pyramid levels.

    Returns
    -------
    np.ndarray, (H, W, C), float32
        Blended image.
    """
    if images_equi.ndim != 4:
        raise ValueError(f"Expected images_equi shape (N, H, W, C), got {images_equi.shape}")
    if weights_equi.shape != images_equi.shape:
        raise ValueError(f"weights_equi shape {weights_equi.shape} must match images_equi shape {images_equi.shape}")

    weights_valid = _mask_weights_by_image_validity(images_equi, weights_equi)
    images_clean, weights_clean = clean_images_and_weights(images_equi, weights_valid)
    return multiband_blend_after_clean_fast(images_clean, weights_clean, level)


def blend_payloads(
    images: np.ndarray,
    weights: np.ndarray | None,
    blend_method: str = "multiband",
    blend_level: int = 8,
    nonnegative_component_indices: tuple[int, ...] = (0,),
    image_cleaner=None,
) -> np.ndarray:
    """Blend warped envmap views into a single equirectangular output.

    Parameters
    ----------
    warped_views : WarpedViews
        Output of warp_views.
    blend_method : {'multiband', 'average'}
        ``'multiband'`` uses Laplacian-pyramid blending per Stokes component;
        ``'average'`` uses nanmean.
    blend_level : int
        Pyramid levels for multiband blending.
    nonnegative_component_indices : tuple[int, ...]
        Payload component indices that are physically nonnegative and should be
        clipped after blending. Component 0 is S0 for Stokes and S2L2 payloads.

    Returns
    -------
    np.ndarray, (H_env, W_env, 3, S), float32
        Blended envmap.
    """
    _validate_blend_method(blend_method)

    if blend_method == "average":
        blended = np.nanmean(images, axis=0).astype(np.float32, copy=False)
        return _clip_nonnegative_components(blended, nonnegative_component_indices)

    if weights is None:
        raise ValueError("weight_stack is required for multiband blending")

    blend_t0 = perf_counter()
    blended = _multiband_blend_components(
        images,
        weights,
        blend_level,
        nonnegative_component_indices=nonnegative_component_indices,
        image_cleaner=image_cleaner,
    )

    blend_elapsed = perf_counter() - blend_t0
    tqdm.write(f"[PEnvMap] multiband blend in {blend_elapsed:.2f}s")
    return _clip_nonnegative_components(blended.astype(np.float32, copy=False), nonnegative_component_indices)


def _clip_nonnegative_components(
    payload: np.ndarray,
    component_indices: tuple[int, ...],
) -> np.ndarray:
    """Project selected payload components onto the nonnegative half-line."""
    if not component_indices:
        return payload.astype(np.float32, copy=False)

    _validate_payload_shape(payload)
    clipped = payload.astype(np.float32, copy=True)
    num_components = clipped.shape[-1]
    for component_idx in component_indices:
        resolved_idx = component_idx + num_components if component_idx < 0 else component_idx
        if resolved_idx < 0 or resolved_idx >= num_components:
            raise ValueError(f"Nonnegative component index {component_idx} is out of range for {num_components} components")
        np.maximum(clipped[:, :, :, resolved_idx], 0.0, out=clipped[:, :, :, resolved_idx])
    return clipped


def blend_views(warped_views, blend_method="multiband", blend_level=8, nonnegative_component_indices=(0,), image_cleaner=None):
    """Public adapter for an existing WarpedViews record."""
    return blend_payloads(warped_views.warped_stack, warped_views.weight_stack, blend_method, blend_level, nonnegative_component_indices, image_cleaner)
