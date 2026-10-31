from __future__ import annotations

import numpy as np

from pemap.geometry.warp import WarpContext
from pemap.blending.weights import accumulate_weighted_payload, normalize_weighted_payload
from pemap.blending.multiband import blend_views
from pemap.geometry.spherical_coordinates import inverse_rotate_equirectangular, inverse_rotate_equirectangular_scalar, octahedral_domain_rotations
from pemap.geometry.warp import warp_views
from pemap.blending.weights import harmonic_center_field


def domain_rotations_from_preset(preset: str) -> list[tuple[str, np.ndarray | None]]:
    """Return named spherical-domain rotations for a supported preset."""
    if preset == "octahedral":
        return octahedral_domain_rotations()
    if preset == "none":
        return [("", None)]
    raise ValueError(f"Unknown domain_rotation_preset: {preset!r}. Choose 'octahedral' or 'none'.")


def blend_domain_rotations(
    warp_input: dict[str, np.ndarray],
    rvec_by_id: dict[str, np.ndarray],
    warp_context: WarpContext,
    envmap_size: tuple[int, int],
    blend_method: str = "multiband",
    blend_level: int = 8,
    domain_rotation_preset: str = "octahedral",
    image_cleaner=None,
) -> np.ndarray:
    """Blend prepared envmap views over one or more spherical-domain rotations.

    The ``none`` preset returns the single unrotated blend. The ``octahedral``
    preset blends all six rotated domains, inverse-rotates each result, and
    combines them with harmonic center weights into one final payload.
    """
    domain_rotations = domain_rotations_from_preset(domain_rotation_preset)
    if not domain_rotations:
        raise ValueError("domain_rotation_preset produced no rotations")

    if tuple(warp_context.envmap_size) != tuple(envmap_size):
        raise ValueError(f"envmap_size {envmap_size} must match warp_context.envmap_size {warp_context.envmap_size}")

    width_envmap, height_envmap = envmap_size
    base_merge_weight = harmonic_center_field(height_envmap, width_envmap).astype(np.float32, copy=False)
    merge_numerator: np.ndarray | None = None
    merge_denominator: np.ndarray | None = None

    for rotation_name, domain_rotation in domain_rotations:
        rotation_label = rotation_name or "unrotated"
        print(f"=== Domain rotation: {rotation_label} ===")

        warped_views = warp_views(
            img_bgr_stokes_by_id=warp_input,
            rvecs=rvec_by_id,
            context=warp_context,
            blend_method=blend_method,
            domain_rotation=domain_rotation,
        )
        blended = blend_views(
            warped_views=warped_views,
            blend_method=blend_method,
            blend_level=blend_level,
            image_cleaner=image_cleaner,
        )
        restored = inverse_rotate_equirectangular(blended, domain_rotation)

        if len(domain_rotations) == 1:
            return restored.astype(np.float32, copy=False)

        merge_weight = inverse_rotate_equirectangular_scalar(base_merge_weight, domain_rotation)
        merge_numerator, merge_denominator = accumulate_weighted_payload(
            merge_numerator,
            merge_denominator,
            restored,
            merge_weight,
        )

    if merge_numerator is None or merge_denominator is None:
        raise RuntimeError("No domain-rotation blend was accumulated")
    return normalize_weighted_payload(merge_numerator, merge_denominator)
