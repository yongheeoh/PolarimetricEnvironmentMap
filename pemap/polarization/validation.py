"""Coordinate-frame invariants checked by the scene runner."""

import numpy as np
from pemap.polarization.min_rot import align_vectors_batched
from pemap.polarization.optimized.s2l2_kernels import build_measurement_frame, s2l2_to_stokes, stokes_to_s2l2
from pemap.polarization.optimized.mueller_kernels import rotate_stokes_basis
from pemap.geometry.projection import unit


def code_checks():
    rng = np.random.default_rng(180)
    rays = unit(np.column_stack((rng.uniform(-1, 1, (512, 2)), np.ones(512))))
    b = align_vectors_batched(rays, [0, 0, 1]) @ np.array([1.0, 0, 0])
    s = np.column_stack((np.ones(512), rng.uniform(-0.4, 0.4, (512, 2)), np.zeros(512)))
    identity = rotate_stokes_basis(-rays, b, -b)
    s180 = np.einsum("nij,nj->ni", identity, s)
    b90 = np.cross(-rays, b)
    s90 = np.einsum("nij,nj->ni", rotate_stokes_basis(-rays, b, b90), s)
    expected90 = s.copy()
    expected90[:, 1:3] *= -1
    F = build_measurement_frame(-rays, b)
    F180 = build_measurement_frame(-rays, -b)
    coef = stokes_to_s2l2(s, F)
    roundtrip = s2l2_to_stokes(coef, F)
    rotatedcoef = stokes_to_s2l2(s180, F180)
    result = {
        "rotation_180_max_absolute_error": float(np.max(np.abs(s180 - s))),
        "rotation_90_sign_control_error": float(np.max(np.abs(s90 - expected90))),
        "s2l2_roundtrip_max_absolute_error": float(np.max(np.abs(roundtrip - s))),
        "s2l2_180_invariance_error": float(np.max(np.abs(rotatedcoef - coef))),
        "basis_dot_ray_max": float(np.max(np.abs(np.sum(b * rays, axis=-1)))),
    }
    assert max(result.values()) < 1e-6, result
    return result
