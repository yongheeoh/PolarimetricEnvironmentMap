"""RGB patch-array calibration; no image I/O or Stokes operations."""

import numpy as np
from skimage.color import deltaE_ciede2000, lab2xyz, xyz2lab, xyz_tristimulus_values

RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])


BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


NEUTRAL_INDICES = np.arange(19, 23)  # middle four; omit white and black


def reference_rgb(lab_d50: np.ndarray) -> np.ndarray:
    """Convert measured Lab D50 to linear sRGB D65 with Bradford adaptation."""
    d50 = xyz_tristimulus_values(illuminant="D50", observer="2")
    d65 = xyz_tristimulus_values(illuminant="D65", observer="2")
    adaptation = np.linalg.solve(BRADFORD, np.diag((BRADFORD @ d65) / (BRADFORD @ d50)) @ BRADFORD)
    xyz = lab2xyz(np.asarray(lab_d50, dtype=np.float64), illuminant="D50")
    return xyz @ adaptation.T @ np.linalg.inv(RGB_TO_XYZ).T


def color_errors(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CIEDE2000 in D65 Lab, before display clipping or gamma encoding."""
    return deltaE_ciede2000(xyz2lab(rgb @ RGB_TO_XYZ.T), xyz2lab(target @ RGB_TO_XYZ.T))


def fit_ccm(balanced: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares 3x3 CCM constrained to preserve equal-channel whites."""
    design = balanced[:, :2] - balanced[:, 2:3]
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("Color patches do not span the chromatic plane")
    coefficients = np.linalg.lstsq(design, target - balanced, rcond=None)[0]
    adjustment = np.vstack([coefficients, -coefficients.sum(axis=0)])
    return (np.eye(3) + adjustment).T


def fit_profile(measured: np.ndarray, target: np.ndarray, *, usable_patches=None) -> dict:
    """Separate G-normalized WB, chart exposure, and white-preserving CCM."""
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if measured.shape != (24, 3) or target.shape != (24, 3):
        raise ValueError("Expected 24 RGB measurements and references")
    usable = np.ones(24, dtype=bool) if usable_patches is None else np.asarray(usable_patches, dtype=bool)
    if usable.shape != (24,):
        raise ValueError("Expected 24 patch reliability flags")
    # All 18 colours and all four WB neutrals remain mandatory. White and black
    # are optional when their original HDR samples lack reliable exposure support.
    if not usable[:18].all() or not usable[NEUTRAL_INDICES].all():
        raise ValueError("Chart contains unreliable colour or WB-neutral HDR patches")
    if not np.isfinite(measured).all() or not np.isfinite(target).all() or np.any(measured[usable] <= 0):
        raise ValueError("Calibration requires finite, positive camera measurements")
    gray = measured[NEUTRAL_INDICES]
    # Reference neutral patches are slightly tinted; account for their RGB.
    scale = np.sum(gray * target[NEUTRAL_INDICES], axis=0) / np.sum(gray**2, axis=0)
    if np.any(scale <= 0):
        raise ValueError("Invalid neutral fit")
    exposure = scale[1]
    gains = scale / exposure
    balanced = measured * scale
    ccm = fit_ccm(balanced[usable], target[usable])
    combined = ccm @ np.diag(gains)
    result = dict(
        wb_gains_rgb=gains.tolist(),
        chart_exposure_scale=float(exposure),
        ccm_rgb=ccm.tolist(),
        combined_matrix_rgb=combined.tolist(),
        chart_matrix_rgb=(combined * exposure).tolist(),
        wb_patch_numbers=(NEUTRAL_INDICES + 1).tolist(),
        matrix_convention="column vectors: output_rgb = matrix @ input_rgb",
        output_color_space="linear sRGB, D65",
        schema_version=1,
    )
    if not usable.all():
        result.update(
            fit_patch_numbers=(np.flatnonzero(usable) + 1).tolist(), excluded_patch_numbers=(np.flatnonzero(~usable) + 1).tolist(), patch_exclusion_reason="Original HDR exposure reliability; all colours and four WB neutrals required"
        )
    return result


def evaluate_profile(measured: np.ndarray, target: np.ndarray, profile: dict) -> dict:
    scale = profile["chart_exposure_scale"]
    balanced = measured * np.array(profile["wb_gains_rgb"]) * scale
    stages = dict(exposure_only=measured * scale, white_balance=balanced, color_corrected=measured @ np.array(profile["chart_matrix_rgb"]).T)
    metrics = {}
    usable = np.zeros(24, dtype=bool)
    usable[np.asarray(profile.get("fit_patch_numbers", range(1, 25))) - 1] = True
    for name, rgb in stages.items():
        error = color_errors(rgb, target)
        metrics[name] = dict(
            mean_delta_e00=float(error[usable].mean()),
            median_delta_e00=float(np.median(error[usable])),
            max_delta_e00=float(error[usable].max()),
            per_patch_delta_e00=error.tolist(),
            mean_neutral_delta_e00=float(error[NEUTRAL_INDICES].mean()),
        )
        if not usable.all():
            metrics[name]["summary_patch_numbers"] = (np.flatnonzero(usable) + 1).tolist()
    # Hold out each colored patch; WB remains independently fit on neutrals.
    predictions = []
    for i in range(18):
        keep = usable & (np.arange(24) != i)
        predictions.append(balanced[i] @ fit_ccm(balanced[keep], target[keep]).T)
    errors = color_errors(np.array(predictions), target[:18])
    metrics["colored_patch_leave_one_out"] = dict(
        mean_delta_e00=float(errors.mean()), max_delta_e00=float(errors.max()), per_patch_delta_e00=errors.tolist(), note="Same capture; hold out one colored patch, retain neutral-only WB. Not an independent-scene validation."
    )
    return metrics


from pemap.io.display import display_srgb
