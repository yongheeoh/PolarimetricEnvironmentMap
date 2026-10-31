"""Image geometry and shared BGR-payload shape validation."""

import numpy as np


def _validate_payload_shape(img_bgr_payload: np.ndarray) -> int:
    """Validate BGR-Stokes payload shape (H, W, 3, S) and return S."""
    if img_bgr_payload.ndim != 4:
        raise ValueError(f"Expected payload image shape (H, W, 3, S), got {img_bgr_payload.shape}")
    if img_bgr_payload.shape[2] != 3 or img_bgr_payload.shape[3] < 1:
        raise ValueError(f"Expected payload image shape (H, W, 3, S), got {img_bgr_payload.shape}")
    return int(img_bgr_payload.shape[3])
