"""Construct a nadir reference from aligned camera images."""

import cv2
import numpy as np


def build_floor_reference(images, masks, rotations, pitches, geometry):
    """Build a downward S0 reference from aligned panorama views, without I/O."""
    n = geometry.nadir_size
    reference = np.zeros((n, n, 3), np.float32)
    best = np.zeros((n, n), np.float32)
    points = geometry.plane_rays()
    for sid, image in images.items():
        if pitches[sid] >= 0:
            continue
        payload = image.copy()
        payload[~masks[sid]] = np.nan
        rotation = np.asarray(rotations[sid], np.float32)
        rectified = geometry.supported_remap(payload, points.reshape(-1, 3), rotation, geometry.K, geometry.D, n, n)[..., 0]
        good = np.isfinite(rectified).all(-1)
        local = points @ cv2.Rodrigues(rotation)[0].T
        weight = np.where(good, local[..., 2], 0)
        take = weight > best
        reference[take], best[take] = rectified[take], weight[take]
    return reference, best > 0
