"""Smooth, scene-local scalar exposure matching for a registered Floor view.

This is not a lens calibration. Only observed overlap supplies measurements;
the missing region receives a smooth extrapolation, not recovered radiometry.
One positive gain is shared by all colour channels and Stokes coefficients.
"""

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def overlap_gain(base_green, floor_green, known, valid, distance, scalar_gain, cell=16, sigma=6.0, regularization=1.0, sample_distance=(2, 120), minimum_support=0.65):
    """Fit robust coarse log-gains and extend them with a screened Laplacian.

    Mask-normalized smoothing avoids a false dark rim from unknown pixels.
    Coarse-cell medians reject small registration/texture errors. A positive
    graph Laplacian keeps the extrapolated log-gain within measured extrema.
    """
    base_green = np.asarray(base_green, np.float32)
    floor_green = np.asarray(floor_green, np.float32)
    common = known & valid & np.isfinite(base_green) & np.isfinite(floor_green) & (base_green > 1e-8) & (floor_green > 1e-8)
    weight = cv2.GaussianBlur(common.astype(np.float32), (0, 0), sigma)
    a = cv2.GaussianBlur(np.where(common, base_green, 0), (0, 0), sigma)
    b = cv2.GaussianBlur(np.where(common, floor_green, 0), (0, 0), sigma)
    # The same normalized denominator cancels in a/b.
    sampled = common & (weight > minimum_support) & (distance >= sample_distance[0]) & (distance <= sample_distance[1])
    sampled &= (a > 1e-8) & (b > 1e-8)
    if sampled.sum() < 500:
        return np.full(common.shape, scalar_gain, np.float32), {"method": "Scalar fallback", "reason": "Insufficient smooth overlap", "gain": float(scalar_gain)}
    ratio = np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12))
    lo, hi = np.percentile(ratio[sampled], [2, 98])
    sampled &= (ratio >= lo) & (ratio <= hi)
    height, width = common.shape
    gh, gw = (height + cell - 1) // cell, (width + cell - 1) // cell
    target = np.zeros((gh, gw), np.float64)
    confidence = np.zeros_like(target)
    for y in range(gh):
        for x in range(gw):
            sl = np.s_[y * cell : min((y + 1) * cell, height), x * cell : min((x + 1) * cell, width)]
            keep = sampled[sl]
            if keep.sum() >= max(8, keep.size * 0.25):
                target[y, x] = np.median(ratio[sl][keep])
                confidence[y, x] = keep.mean()
    if (confidence > 0).sum() < 12:
        return np.full(common.shape, scalar_gain, np.float32), {"method": "Scalar fallback", "reason": "Insufficient overlap cells", "gain": float(scalar_gain)}
    indices = np.arange(gh * gw).reshape(gh, gw)
    i = np.concatenate([indices[:, :-1].ravel(), indices[:-1, :].ravel()])
    j = np.concatenate([indices[:, 1:].ravel(), indices[1:, :].ravel()])
    ones = np.ones(len(i), np.float64)
    laplacian = coo_matrix((np.r_[ones, ones, -ones, -ones], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(gh * gw, gh * gw)).tocsr()
    system = diags(confidence.ravel()) + regularization * laplacian
    fitted = spsolve(system, (confidence * target).ravel()).reshape(gh, gw)
    # No cubic overshoot: linear interpolation and exponentiation stay positive.
    yy, xx = np.indices(common.shape, dtype=np.float32)
    dense = cv2.remap(fitted.astype(np.float32), (xx + 0.5) / cell - 0.5, (yy + 0.5) / cell - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gain = np.exp(dense).astype(np.float32)
    report = {
        "method": "Smooth scalar overlap field",
        "cell_nadir_px": cell,
        "sigma_nadir_px": sigma,
        "regularization": regularization,
        "sample_distance_nadir_px": list(sample_distance),
        "minimum_smooth_support": minimum_support,
        "sampled_pixels": int(sampled.sum()),
        "measured_cells": int((confidence > 0).sum()),
        "gain_range": [float(gain.min()), float(gain.max())],
        "measured_gain_p02_p98": np.exp([lo, hi]).tolist(),
        "shared_across": "All BGR channels and all Stokes/S2L2 coefficients",
        "unobserved_region": "Smooth harmonic extrapolation; not measured calibration",
    }
    for name, estimate in [("scalar", scalar_gain), ("spatial", gain)]:
        error = np.abs(np.expm1(ratio - np.log(estimate)))
        report[name + "_overlap_absolute_relative_error_p50_p90"] = np.percentile(error[sampled], [50, 90]).tolist()
        # Boundary diagnostic, not independent physical calibration validation.
        boundary = common & (weight > 0.5) & (distance >= 2) & (distance < 15)
        if boundary.any():
            report[name + "_boundary_absolute_relative_error_p50_p90"] = np.percentile(error[boundary], [50, 90]).tolist()
    return gain, report
