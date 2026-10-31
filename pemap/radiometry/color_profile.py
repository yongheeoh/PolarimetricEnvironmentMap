"""Validated BGR row-vector transform shared by desktop and mobile."""

import numpy as np


def color_matrix(profile):
    wb = np.array([profile["wb_gain"][c] for c in "BGR"], np.float32)
    ccm = np.asarray(profile["ccm_3x3"], np.float32)
    if wb.shape != (3,) or ccm.shape != (3, 3) or not np.isfinite(ccm).all() or not (wb > 0).all():
        raise ValueError("Invalid WB/CCM profile")
    # Imported release profiles retain matrices even when the original run
    # explicitly disabled a transform. Preserve that recorded choice.
    if profile.get("white_balance_applied") is False:
        wb = np.ones(3, np.float32)
    if profile.get("color_correction_matrix_applied") is False:
        ccm = np.eye(3, dtype=np.float32)
    matrix = np.diag(wb) @ ccm[::-1, ::-1]  # row-vector BGR
    if abs(np.linalg.det(matrix)) < 1e-6:
        raise ValueError("Singular WB/CCM transform")
    return wb, ccm, matrix
