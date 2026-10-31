"""Shared panorama alignment, Stokes transport, blending and coverage.

Capture discovery/recovery and publication stay in the scene or RAW adapter.
This module receives prepared camera images and returns a world panorama.
"""

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from tqdm.auto import tqdm
from functools import partial
from typing import Callable
import warnings

import cv2
import numpy as np
from pemap.blending.domain_blending import blend_domain_rotations
from pemap.geometry.warp import build_warp_context, warp_views
from pemap.registration.camera_rotation import estimate_camera_rotation_matrices

from pemap.radiometry.color import apply_wb_ccm_stokes
from pemap.blending.boundary import extend_for_pyramids
from pemap.composition import PanoramaState
from pemap.registration.camera_rotation import AlignedViews
from pemap.sensor import PanoramaView
from pemap.polarization.transport import StokesProjector, stokes_panorama_to_s2l2
from pemap.geometry.spherical_coordinates import equirectangular_directions
from pemap.blending.multiband import blend_views, _validate_blend_method
from pemap.geometry.rays import precise_rays

DEFAULT_MATCHERS = ("sift-lightglue", "superpoint-lightglue", "aliked-lightglue", "gim-lightglue", "xfeat-lightglue")


def align_views(images, intrinsics, priors, *, envmap_size, pitches=(-45, 0, 45), mask_dir=None, masks=None, optimize=True, matchers=DEFAULT_MATCHERS, cache_dir=None, strict=True, remapper=None):
    """Estimate one set of view poses, with an explicit fallback policy."""
    if not images or any(sid not in priors or priors[sid] is None for sid in images):
        raise ValueError("Missing panorama images or stage rotation priors")
    context = build_warp_context(images, intrinsics, mask_dir, pitches, envmap_size)
    if masks is not None:
        context = replace(context, pitch_by_id={sid: i for i, sid in enumerate(images)}, masks_by_pitch={i: masks[sid] for i, sid in enumerate(images)})
    context = replace(context, remapper=remapper)
    rotations = priors
    report = {"enabled": bool(optimize), "used_prior": True, "failure_reason": None, "graph_stats": {}, "matcher_name": matchers, "cache_info": {"enabled": False, "status": "disabled", "path": None, "key": None}}
    if optimize:
        result = estimate_camera_rotation_matrices(images, priors, intrinsics, mask_dir, pitches, matcher_name=matchers, warp_context=context, cache_dir=cache_dir)
        if result.used_prior:
            message = f"Image rotation refinement failed: {result.failure_reason}"
            if strict:
                raise RuntimeError(message + "; inspect inputs or explicitly use --stage-poses")
            warnings.warn(message + "; using stage priors", stacklevel=2)
        rotations = result.rvec_by_id
        report.update(used_prior=bool(result.used_prior), failure_reason=result.failure_reason, graph_stats=result.graph_stats, cache_info=result.cache_info)
    else:
        report["reason"] = "Explicit --stage-poses"
    active = {k: np.asarray(v).reshape(-1).tolist() for k, v in rotations.items()}
    report.update(rvec_by_id=active, rvec_active_by_id=active, rvec_prior_by_id={k: np.asarray(v).reshape(-1).tolist() for k, v in priors.items()})
    return AlignedViews(context, priors, rotations, report)


def compose_panorama(alignment: AlignedViews, load_view: Callable[[str], PanoramaView], projector: StokesProjector, *, level=5, domain="octahedral", wb_ccm=None, extend_boundaries=False, store_payload=None, blend_method="multiband") -> PanoramaState:
    """Warp/blend prepared views, apply colour once and retain observed support."""
    context = alignment.context
    width, height = context.envmap_size
    coverage = np.zeros((height, width), bool)
    saturation = np.zeros_like(coverage)
    payloads = {}
    for sid in context.sorted_ids:
        view = load_view(sid)
        rotation = cv2.Rodrigues(np.asarray(alignment.rotations[sid], dtype=projector.rotation_dtype))[0]
        payload = projector(view.stokes, rotation)
        payloads[sid] = store_payload(sid, payload) if store_payload is not None else payload
        del payload
        flag = np.zeros(projector.shape, np.float32) if view.saturation is None else view.saturation.astype(np.float32)
        probe = np.repeat(flag[..., None, None], 3, axis=2)
        warped = warp_views({sid: probe}, alignment.rotations, replace(context, sorted_ids=[sid]), blend_method="average")
        values = warped.warped_stack[0, ..., 0, 0]
        valid = np.isfinite(values)
        coverage |= valid
        saturation |= valid & (values > 0)
    cleaner = partial(extend_for_pyramids, pad=3 * 2**level) if extend_boundaries else None
    payload = blend_domain_rotations(payloads, alignment.rotations, context, (width, height), blend_method=blend_method, blend_level=level, domain_rotation_preset=domain, image_cleaner=cleaner)
    if wb_ccm is not None:
        payload = apply_wb_ccm_stokes(payload, *wb_ccm)
    payload = np.asarray(payload, np.float32)
    if projector.mode == "full" and projector.representation in ("duff", "thetaphi"):
        payload = stokes_panorama_to_s2l2(payload, equirectangular_directions(width, height), projector.representation)
    payload[~coverage] = 0
    representation = "none" if projector.mode == "full" and projector.representation == "none" else "s2l2"
    return PanoramaState(payload, coverage, coverage & saturation, representation)


def build_envmap(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    intrinsics: dict,
    rvecs: dict[str, np.ndarray],
    mask_dir: str | Path,
    pitches: list[int],
    envmap_size: tuple[int, int] = (2048, 1024),
    blend_method: str = "multiband",
    blend_level: int = 8,
    domain_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """End-to-end pipeline: warp and blend BGR-Stokes views into an equirectangular envmap.

    Parameters
    ----------
    img_bgr_stokes_by_id : dict[str, np.ndarray]
        BGR-Stokes images keyed by bracket ID, each (H, W, 3, S), float32.
    intrinsics : dict
        Camera intrinsics with keys ``"K"`` and ``"dist_coeffs"``.
    rvecs : dict[str, np.ndarray]
        Rodrigues rotation vectors keyed by bracket ID, each (3, 1).
    mask_dir : str or Path
        Directory containing ``mask_{pitch}.png`` files.
    pitches : list[int]
        Pitch angles (degrees) for each camera group.
    envmap_size : tuple[int, int]
        Output envmap (width, height). Default (2048, 1024).
    blend_method : {'multiband', 'average'}
    blend_level : int
        Pyramid levels for multiband blending.
    domain_rotation : np.ndarray or None
        Optional active spherical-domain rotation applied before projection.

    Returns
    -------
    np.ndarray, (H_env, W_env, 3, S), float32
        Blended polarimetric envmap.
    """
    _validate_blend_method(blend_method)

    total_t0 = perf_counter()

    context = build_warp_context(
        img_bgr_stokes_by_id=img_bgr_stokes_by_id,
        intrinsics=intrinsics,
        mask_dir=mask_dir,
        pitches=pitches,
        envmap_size=envmap_size,
    )
    warped_views = warp_views(
        img_bgr_stokes_by_id=img_bgr_stokes_by_id,
        rvecs=rvecs,
        context=context,
        blend_method=blend_method,
        domain_rotation=domain_rotation,
    )
    out = blend_views(
        warped_views=warped_views,
        blend_method=blend_method,
        blend_level=blend_level,
    )

    total_elapsed = perf_counter() - total_t0
    tqdm.write(f"[PEnvMap] done in {total_elapsed:.2f}s")
    return out
