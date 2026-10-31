"""Display-only transfer functions; never applied to scientific EXRs."""

import numpy as np


def display_srgb(linear: np.ndarray) -> np.ndarray:
    """Clip and encode for display only; never use this on scientific EXRs."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def display(bgr, exposure):
    return np.rint(display_srgb(np.maximum(bgr, 0) * exposure) * 255).astype(np.uint8)
