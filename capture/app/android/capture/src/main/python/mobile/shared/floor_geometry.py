"""Planar Floor pose and per-ray Stokes transport."""

import cv2
import numpy as np

from .scene_math import transform, unit


def floor_pose(geometry, H):
    K = np.array([[geometry.focal, 0, geometry.principal], [0, geometry.focal, geometry.principal], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ np.linalg.inv(H) @ K
    B *= 2 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if B[2, 2] < 0:
        B *= -1
    rough = np.column_stack([B[:, 0], B[:, 1], np.cross(B[:, 0], B[:, 1])])
    u, _, vh = np.linalg.svd(rough)
    R = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
    y, x = np.meshgrid(np.linspace(0.17 * geometry.nadir_size, 0.83 * geometry.nadir_size, 13), np.linspace(0.17 * geometry.nadir_size, 0.83 * geometry.nadir_size, 13))
    target = np.stack([x, y], -1).reshape(-1, 2)
    source = transform(target, np.linalg.inv(H))
    keep = np.all((source > 30) & (source < geometry.nadir_size - 31), axis=1)
    target, source = target[keep], source[keep]
    if len(target) < 6:
        raise ValueError("Floor homography has too little calibrated overlap")
    objects = np.column_stack([(target[:, 0] - geometry.principal) / geometry.focal, (target[:, 1] - geometry.principal) / geometry.focal, np.zeros(len(target))])
    ok, rv, tv = cv2.solvePnP(objects, source, K, None, cv2.Rodrigues(R)[0], B[:, 2].copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Floor pose estimation failed")
    residual = np.linalg.norm(cv2.projectPoints(objects, rv, tv, K, None)[0][:, 0] - source, axis=-1)
    return cv2.Rodrigues(rv)[0], {
        "rotation": cv2.Rodrigues(rv)[0].tolist(),
        "translation_arbitrary_units": tv.reshape(-1).tolist(),
        "median_fit_px": float(np.median(residual)),
        "limitation": "Approximate planar pose; no recovery of original-view polarized reflectance",
    }

from .min_rot import align_vectors_batched
from .s2l2 import build_measurement_frame, stokes_to_s2l2

def camera_basis(rays, rotation_model="min_rot"):
    """The two camera measurement conventions from the original main.py."""
    if rotation_model == "min_rot":
        return align_vectors_batched(rays, [0, 0, 1]) @ [1.0, 0, 0]
    if rotation_model == "mitsuba3":
        return np.cross(rays, [0, 1, 0])
    raise ValueError(f"Unknown camera rotation model: {rotation_model}")

def floor_coefficients(geometry, warped, H, *, rotation_model="min_rot"):
    if warped.shape[-1] == 1:
        return warped, {"S0_only": True, "rotation_not_required_for_scalar_intensity": True}
    R, report = floor_pose(geometry, H)
    n = geometry.nadir_size
    y, x = np.indices((n, n), dtype=float)
    xy = transform(np.stack([x, y], -1), np.linalg.inv(H))
    rays = unit(
        np.stack(
            [
                (xy[..., 0] - geometry.principal) / geometry.focal,
                (xy[..., 1] - geometry.principal) / geometry.focal,
                np.ones((n, n)),
            ],
            -1,
        )
    )
    T = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    target = geometry.plane_rays().astype(float)
    payload = np.empty((n, n, 3, 12), np.float32)
    for start in range(0, n, 100):
        sl = slice(start, start + 100)
        lr = rays[sl]
        world = lr @ R @ T.T
        basis = camera_basis(lr, rotation_model) @ R @ T.T
        transport = align_vectors_batched(target[sl], world)
        basis = np.einsum("...ij,...j->...i", transport, basis)
        frame = build_measurement_frame(-target[sl], basis)
        payload[sl] = stokes_to_s2l2(warped[sl], frame[..., None, :, :]).astype(np.float32)
    return payload, report
