"""Linear WB/CCM without interactive calibration dependencies."""

import json
from pathlib import Path

import numpy as np


def load_wb_ccm(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load wb_ccm.json → (wb_gain_bgr (3,), M_ccm (3,3))."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    wb = data["wb_gain"]
    wb_gain_bgr = np.array([wb["B"], wb["G"], wb["R"]], dtype=np.float32)
    M = np.array(data["ccm_3x3"], dtype=np.float32)
    return wb_gain_bgr, M


def apply_wb_ccm_stokes(
    img_bgr_stokes: np.ndarray,
    wb_gain_bgr: np.ndarray,
    M: np.ndarray,
    apply_wb: bool = True,
    apply_ccm: bool = True,
) -> np.ndarray:
    """
    Apply WB gain and/or CCM to a (H,W,3,N) BGR Stokes or S2L2 array.

    Apply the same color transform to every component: four Stokes parameters
    or all twelve S2L2 coefficients. CCM uses BGR→RGB, M applied, RGB→BGR.
    No clipping or normalization is applied.
    """
    out = img_bgr_stokes.astype(np.float32).copy()

    if apply_wb:
        out[:, :, 0, :] *= float(wb_gain_bgr[0])  # B
        out[:, :, 1, :] *= float(wb_gain_bgr[1])  # G
        out[:, :, 2, :] *= float(wb_gain_bgr[2])  # R

    if apply_ccm:
        for s in range(out.shape[-1]):
            bgr = out[:, :, :, s]
            rgb = bgr[:, :, ::-1]
            out[:, :, :, s] = (rgb @ M)[:, :, ::-1]

    return out
