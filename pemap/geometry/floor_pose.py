"""Planar Floor pose and per-ray Stokes transport."""

import cv2
import numpy as np

from pemap.geometry.projection import transform, unit


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
