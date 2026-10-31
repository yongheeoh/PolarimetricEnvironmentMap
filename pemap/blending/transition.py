"""Guarded Floor transition blending in camera colour coordinates."""

import numpy as np
from pemap.blending.multiband import blend_payloads


def guarded_multiband(base, floor, alpha, color, level=5):
    if not np.any(alpha > 0):
        return base.copy()
    y, x = np.where(alpha > 0)
    pad = 3 * 2**level
    n, m = alpha.shape
    box = (slice(max(0, y.min() - pad), min(n, y.max() + pad + 1)), slice(max(0, x.min() - pad), min(m, x.max() + pad + 1)))
    a, b, w = base[box], floor[box], alpha[box]
    core, outer = smooth((w - 0.5) * 2)[..., None, None], smooth(w * 2)[..., None, None]
    first, second = a * (1 - core) + b * core, a * (1 - outer) + b * outer
    camera = np.einsum("...cs,cd->...ds", np.stack([first, second]), np.linalg.inv(color)).astype(np.float32)
    if camera[..., 0].min() < -2e-6:
        raise ValueError("Unexpected negative camera S0 before Floor blend")
    camera[..., 0] = np.maximum(camera[..., 0], 0)
    weights = np.repeat(np.stack([1 - w, w])[..., None], 3, -1)
    mixed = blend_payloads(camera, weights, blend_method="multiband", blend_level=level)
    mixed = np.einsum("...cs,cd->...ds", mixed, color)
    out = (mixed * (1 - core) + b * core) * outer + a * (1 - outer)
    out[w == 1], out[w == 0] = b[w == 1], a[w == 0]
    result = base.copy()
    result[box] = out
    return result


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))
