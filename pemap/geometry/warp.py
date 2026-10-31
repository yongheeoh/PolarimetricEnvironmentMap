import re
from dataclasses import dataclass
from typing import Callable
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from tqdm.auto import tqdm

from pemap.geometry import _validate_payload_shape
from pemap.geometry.spherical_coordinates import equirectangular_directions, _validate_domain_rotation
from pemap.geometry.optimized.remap_equirectangular import remap_equirectangular

@dataclass(frozen=True)
class WarpContext:
    """Immutable context holding shared parameters for penvmap warping."""

    sorted_ids: list[str]
    pitch_by_id: dict[str, int]
    masks_by_pitch: dict[int, np.ndarray]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    envmap_size: tuple[int, int]
    remapper: Callable | None = None

@dataclass(frozen=True)
class WarpedViews:
    """Stacked warped views and optional blend weights from warp_views."""

    sorted_ids: list[str]
    warped_stack: np.ndarray
    weight_stack: np.ndarray | None


_NATURAL_TOKEN_RE = re.compile(r"\d+|\D+")


def _natural_sort_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Natural-sort key: digits compare numerically, strings lexicographically."""
    tokens = _NATURAL_TOKEN_RE.findall(str(text))
    key: list[tuple[int, int | str]] = []
    for token in tokens:
        if token.isdigit():
            key.append((0, int(token)))
        else:
            key.append((1, token.lower()))
    return tuple(key)


def _assign_pitch_by_blocks(bracket_ids: list[str], pitches: list[int]) -> dict[str, int]:
    """Assign a camera pitch to each bracket ID by splitting sorted IDs into equal blocks.

    Parameters
    ----------
    bracket_ids : list[str]
        All bracket identifiers.
    pitches : list[int]
        Pitch angles (degrees) in ascending order; one block per pitch.

    Returns
    -------
    dict[str, int]
        Mapping bracket_id â†’ pitch (degrees).
    """
    if not bracket_ids:
        raise ValueError("bracket_ids must not be empty")
    if not pitches:
        raise ValueError("pitches must not be empty")

    sorted_ids = sorted(bracket_ids, key=_natural_sort_key)
    blocks = np.array_split(np.asarray(sorted_ids, dtype=object), len(pitches))

    pitch_by_id: dict[str, int] = {}
    for pitch, block in zip(pitches, blocks):
        if len(block) == 0:
            raise ValueError("More pitches were provided than bracket blocks")
        for bracket_id in block.tolist():
            pitch_by_id[str(bracket_id)] = int(pitch)
    return pitch_by_id


def _load_masks(mask_dir: str | Path, pitches: list[int]) -> dict[int, np.ndarray]:
    """Load grayscale mask PNGs (mask_{pitch}.png) for each pitch value."""
    mask_dir = Path(mask_dir)
    masks: dict[int, np.ndarray] = {}
    for pitch in pitches:
        mask_path = mask_dir / f"mask_{pitch}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        masks[int(pitch)] = mask
    return masks


def _source_hw_from_payloads(img_bgr_stokes_by_id: dict[str, np.ndarray]) -> tuple[int, int]:
    """Return the common source image size from a payload dictionary."""
    first_id = next(iter(img_bgr_stokes_by_id))
    first_hw = np.asarray(img_bgr_stokes_by_id[first_id]).shape[:2]
    if len(first_hw) != 2:
        raise ValueError(f"Expected image with at least two dimensions for bracket_id={first_id}, got {np.asarray(img_bgr_stokes_by_id[first_id]).shape}")

    for bracket_id, img_bgr_payload in img_bgr_stokes_by_id.items():
        hw = np.asarray(img_bgr_payload).shape[:2]
        if hw != first_hw:
            raise ValueError(f"Image shape {hw} for bracket_id={bracket_id} does not match first image shape {first_hw}")
    return int(first_hw[0]), int(first_hw[1])


def _resize_masks_to_hw(masks_by_pitch: dict[int, np.ndarray], source_hw: tuple[int, int]) -> dict[int, np.ndarray]:
    """Resize loaded masks to match the current source image size."""
    source_height, source_width = source_hw
    resized_masks: dict[int, np.ndarray] = {}
    for pitch, mask in masks_by_pitch.items():
        if mask.shape == source_hw:
            resized_masks[pitch] = mask
        else:
            resized_masks[pitch] = cv2.resize(np.asarray(mask), (source_width, source_height), interpolation=cv2.INTER_AREA)
    return resized_masks


def _apply_mask(img_bgr_payload: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set pixels where mask < 0.5 to NaN in a copy of img_bgr_payload."""
    _validate_payload_shape(img_bgr_payload)
    if img_bgr_payload.shape[:2] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {img_bgr_payload.shape[:2]}")

    masked = img_bgr_payload.copy()
    masked[mask < 128] = np.nan
    return masked


def _warp_stokes(
    img_bgr_payload: np.ndarray,
    rvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    width_envmap: int,
    height_envmap: int,
    domain_rotation: np.ndarray | None = None,
    remapper=None,
) -> np.ndarray:
    """Warp a BGR-Stokes image to an equirectangular envmap via projectPoints.

    Parameters
    ----------
    img_bgr_payload : np.ndarray, (H_src, W_src, 3, S), float32
        Source BGR-Stokes image.
    rvec : np.ndarray, (3,) or (3, 1), float32
        Rodrigues rotation vector.
    camera_matrix : np.ndarray, (3, 3), float32
        Camera intrinsic matrix K.
    distortion : np.ndarray, (N,), float32
        OpenCV distortion coefficients.
    width_envmap, height_envmap : int
        Output envmap resolution.
    domain_rotation : np.ndarray or None
        Optional active spherical-domain rotation. Output grid directions ``g``
        are projected as world directions ``R @ g``.

    Returns
    -------
    np.ndarray, (height_envmap, width_envmap, 3, S), float32
        Warped image. NaN where no source pixel maps (border or back-facing).
    """
    _validate_payload_shape(img_bgr_payload)

    rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float32).reshape(-1, 1)

    obj_points = equirectangular_directions(width_envmap, height_envmap).reshape(-1, 3)
    rotation = _validate_domain_rotation(domain_rotation)
    if rotation is not None:
        obj_points = obj_points @ rotation.T

    warped = (remapper or remap_equirectangular)(
        img_bgr_payload,
        obj_points,
        rvec,
        camera_matrix,
        distortion,
        width_envmap,
        height_envmap,
    )
    return warped


@lru_cache(maxsize=1)
def _radial_weight_payload_for_hw(height_src: int, width_src: int) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(np.arange(width_src, dtype=np.float32), np.arange(height_src, dtype=np.float32))
    radius = np.sqrt((grid_x - width_src / 2.0) ** 2 + (grid_y - height_src / 2.0) ** 2)
    radius_max = max(float(np.max(radius)), 1.0)
    weight = np.clip(1.0 - radius / radius_max, 0.0, 1.0)
    weight_rgb = np.repeat(weight[:, :, None], 3, axis=2)
    payload = weight_rgb[:, :, :, None].astype(np.float32, copy=False)
    payload.setflags(write=False)
    return payload


def _make_radial_weight_equi(
    src_hw: tuple[int, int],
    rvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    width_envmap: int,
    height_envmap: int,
    domain_rotation: np.ndarray | None = None,
    remapper=None,
) -> np.ndarray:
    """Create a radial-falloff blend weight map warped to equirectangular space.

    Parameters
    ----------
    src_hw : tuple[int, int]
        Source image (H_src, W_src).
    rvec : np.ndarray, (3, 1), float32
        Rodrigues rotation vector.
    camera_matrix : np.ndarray, (3, 3), float32
        Camera intrinsic matrix K.
    distortion : np.ndarray, (N,), float32
        OpenCV distortion coefficients.
    width_envmap, height_envmap : int
        Output envmap resolution.
    domain_rotation : np.ndarray or None
        Optional active spherical-domain rotation. Must match the image warp.

    Returns
    -------
    np.ndarray, (height_envmap, width_envmap, 3), float32
        Weight map, values in [0, 1].
    """
    height_src, width_src = src_hw
    weight_payload = _radial_weight_payload_for_hw(int(height_src), int(width_src))
    return _warp_stokes(weight_payload, rvec, camera_matrix, distortion, width_envmap, height_envmap, domain_rotation=domain_rotation, remapper=remapper)[:, :, :, 0]


def build_warp_context(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    intrinsics: dict,
    mask_dir: str | Path,
    pitches: list[int],
    envmap_size: tuple[int, int] = (2048, 1024),
    source_hw: tuple[int, int] | None = None,
) -> WarpContext:
    """Build the shared warp context from BGR-Stokes views and calibration data.

    Parameters
    ----------
    img_bgr_stokes_by_id : dict[str, np.ndarray]
        BGR-Stokes images keyed by bracket ID; used to extract bracket IDs and
        the current source image size for mask resizing.
    intrinsics : dict
        Camera intrinsics with keys ``"K"`` (3x3) and ``"dist_coeffs"``.
    mask_dir : str or Path or None
        Directory containing ``mask_{pitch}.png`` files. None = all pixels valid.
    pitches : list[int]
        Pitch angles (degrees) for each camera group.
    envmap_size : tuple[int, int]
        Output envmap (width, height). Default (2048, 1024).
    source_hw : tuple[int, int], optional
        Source image size ``(height, width)``. When omitted, the size is inferred
        from ``img_bgr_stokes_by_id`` payloads.

    Returns
    -------
    WarpContext
    """
    if not img_bgr_stokes_by_id:
        raise ValueError("img_bgr_stokes_by_id must not be empty")

    bracket_ids = [str(bracket_id) for bracket_id in img_bgr_stokes_by_id.keys()]
    sorted_ids = sorted(bracket_ids, key=_natural_sort_key)
    pitch_by_id = _assign_pitch_by_blocks(sorted_ids, pitches)
    if source_hw is None:
        source_hw = _source_hw_from_payloads(img_bgr_stokes_by_id)
    else:
        source_hw = (int(source_hw[0]), int(source_hw[1]))
    if mask_dir is None:  # no calibration masks yet: every pixel valid
        masks_by_pitch = {int(p): np.full(source_hw, 255, dtype=np.uint8) for p in pitches}
    else:
        masks_by_pitch = _resize_masks_to_hw(_load_masks(mask_dir, pitches), source_hw)

    camera_matrix = np.asarray(intrinsics["K"], dtype=np.float32).reshape(3, 3)
    distortion = np.asarray(intrinsics["dist_coeffs"], dtype=np.float32).reshape(-1)
    return WarpContext(
        sorted_ids=sorted_ids,
        pitch_by_id=pitch_by_id,
        masks_by_pitch=masks_by_pitch,
        camera_matrix=camera_matrix,
        distortion=distortion,
        envmap_size=envmap_size,
    )


def warp_views(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    rvecs: dict[str, np.ndarray],
    context: WarpContext,
    blend_method: str = "multiband",
    domain_rotation: np.ndarray | None = None,
) -> WarpedViews:
    """Warp all BGR-Stokes views to equirectangular and optionally compute blend weights.

    Parameters
    ----------
    img_bgr_stokes_by_id : dict[str, np.ndarray]
        BGR-Stokes images keyed by bracket ID, each (H, W, 3, S), float32.
    rvecs : dict[str, np.ndarray]
        Rodrigues rotation vectors keyed by bracket ID, each (3, 1).
    context : WarpContext
        Shared warp context from build_warp_context.
    blend_method : {'multiband', 'average'}
        If ``'multiband'``, radial blend weights are computed and stored.
    domain_rotation : np.ndarray or None
        Optional active spherical-domain rotation applied before projection.

    Returns
    -------
    WarpedViews
        ``warped_stack`` : (N, H_env, W_env, 3, S), float32.
        ``weight_stack`` : (N, H_env, W_env, 3), float32 if multiband, else None.
    """
    if blend_method not in {"multiband", "average"}:
        raise ValueError(f"Unsupported blend method: {blend_method}")
    domain_rotation = _validate_domain_rotation(domain_rotation)

    width_envmap, height_envmap = context.envmap_size
    warped_views: list[np.ndarray] = []
    weight_views: list[np.ndarray] = []
    src_hw: tuple[int, int] | None = None

    warp_t0 = perf_counter()
    for bracket_id in tqdm(context.sorted_ids, desc="Warp views", unit="view"):
        if bracket_id not in rvecs:
            raise ValueError(f"Missing rvec for bracket_id={bracket_id}")

        img_bgr_stokes = np.asarray(img_bgr_stokes_by_id[bracket_id], dtype=np.float32)
        if src_hw is None:
            src_hw = img_bgr_stokes.shape[:2]

        mask = context.masks_by_pitch[context.pitch_by_id[bracket_id]]
        masked = _apply_mask(img_bgr_stokes, mask)
        rvec = np.asarray(rvecs[bracket_id], dtype=np.float32).reshape(3, 1)
        warped = _warp_stokes(
            masked,
            rvec,
            context.camera_matrix,
            context.distortion,
            width_envmap,
            height_envmap,
            domain_rotation=domain_rotation,
            remapper=context.remapper,
        )
        warped_views.append(warped)

        if blend_method == "multiband":
            if src_hw is None:
                raise RuntimeError("src_hw must be available before computing blend weights")
            weight_views.append(
                _make_radial_weight_equi(
                    src_hw,
                    rvec,
                    context.camera_matrix,
                    context.distortion,
                    width_envmap,
                    height_envmap,
                    domain_rotation=domain_rotation,
                    remapper=context.remapper,
                )
            )

    warp_elapsed = perf_counter() - warp_t0
    tqdm.write(f"[PEnvMap] warped {len(context.sorted_ids)} views in {warp_elapsed:.2f}s")

    weight_stack = None
    if blend_method == "multiband":
        weight_stack = np.stack(weight_views, axis=0).astype(np.float32, copy=False)

    return WarpedViews(
        sorted_ids=context.sorted_ids,
        warped_stack=np.stack(warped_views, axis=0).astype(np.float32, copy=False),
        weight_stack=weight_stack,
    )
