import numpy as np
import numpy.typing as npt
from scipy.fft import dstn, idstn


def _laplacian_eigenvalues(h: int, w: int) -> npt.NDArray[np.float64]:
    """Return eigenvalues of the rectangular Dirichlet 5-point Laplacian.

    Parameters
    ----------
    h : int
        Number of interior rows.
    w : int
        Number of interior columns.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(h, w)`` containing the Laplacian eigenvalues for
        DST-I basis functions.
    """
    y = np.arange(1, h + 1, dtype=float)[:, None]
    x = np.arange(1, w + 1, dtype=float)[None, :]
    return 4.0 - 2.0 * np.cos(np.pi * y / (h + 1)) - 2.0 * np.cos(np.pi * x / (w + 1))


def _solve_dirichlet_poisson(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve a zero-boundary rectangular Poisson problem.

    Parameters
    ----------
    rhs : numpy.ndarray
        Interior right-hand side array with shape ``(H - 2, W - 2)``.

    Returns
    -------
    numpy.ndarray
        Interior solution array with the same shape as ``rhs``.
    """
    return idstn(dstn(rhs, type=1, norm="ortho") / _laplacian_eigenvalues(*rhs.shape), type=1, norm="ortho")


def harmonic_center_field(H: int, W: int, center: tuple[int, int] | None = None) -> npt.NDArray[np.float64]:
    """Compute a harmonic field with a fixed unit-valued center.

    Parameters
    ----------
    H : int
        Output height in pixels.
    W : int
        Output width in pixels.
    center : tuple of int, optional
        Pixel coordinate ``(row, column)`` fixed to value ``1``. If omitted,
        the image center ``(H // 2, W // 2)`` is used.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W)``. Boundary pixels are zero except when
        ``center`` lies on the boundary, the center pixel is one, and all
        other non-boundary pixels satisfy the 4-connected discrete Laplace
        equation.
    """
    if center is None:
        center = (H // 2, W // 2)

    cy, cx = center

    out = np.zeros((H, W), dtype=float)
    out[cy, cx] = 1.0

    if cy < 0:
        cy += H
    if cx < 0:
        cx += W

    h = H - 2
    w = W - 2
    if h <= 0 or w <= 0:
        return out

    rhs = np.zeros((h, w), dtype=float)

    if 0 < cy < H - 1 and 0 < cx < W - 1:
        iy = cy - 1
        ix = cx - 1
        rhs[iy, ix] = 1.0
        green = _solve_dirichlet_poisson(rhs)
        out[1:-1, 1:-1] = green / green[iy, ix]
    else:
        for yy, xx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 < yy < H - 1 and 0 < xx < W - 1:
                rhs[yy - 1, xx - 1] += 1.0

        if np.any(rhs):
            out[1:-1, 1:-1] = _solve_dirichlet_poisson(rhs)

    return out


from pemap.geometry import _validate_payload_shape


def accumulate_weighted_payload(
    numerator: np.ndarray | None,
    denominator: np.ndarray | None,
    payload: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate a finite-pixel weighted average for BGR payload images."""
    payload = np.asarray(payload, dtype=np.float32)
    _validate_payload_shape(payload)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape != payload.shape[:2]:
        raise ValueError(f"Weight shape {weight.shape} must match payload image shape {payload.shape[:2]}")

    if numerator is None:
        numerator = np.zeros_like(payload, dtype=np.float32)
    elif numerator.shape != payload.shape:
        raise ValueError(f"Numerator shape {numerator.shape} must match payload shape {payload.shape}")

    if denominator is None:
        denominator = np.zeros(payload.shape[:2], dtype=np.float32)
    elif denominator.shape != payload.shape[:2]:
        raise ValueError(f"Denominator shape {denominator.shape} must match payload image shape {payload.shape[:2]}")

    finite_pixels = np.isfinite(weight) & np.all(np.isfinite(payload), axis=(2, 3))
    valid_weight = np.where(finite_pixels, weight, 0.0).astype(np.float32, copy=False)
    numerator += np.where(finite_pixels[:, :, None, None], payload, 0.0) * valid_weight[:, :, None, None]
    denominator += valid_weight
    return numerator, denominator


def normalize_weighted_payload(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Normalize an accumulated BGR payload weighted average, filling empty pixels with zero."""
    numerator = np.asarray(numerator, dtype=np.float32)
    _validate_payload_shape(numerator)
    denominator = np.asarray(denominator, dtype=np.float32)
    if denominator.shape != numerator.shape[:2]:
        raise ValueError(f"Denominator shape {denominator.shape} must match payload image shape {numerator.shape[:2]}")

    denominator_expanded = denominator[:, :, None, None]
    return np.divide(
        numerator,
        denominator_expanded,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator_expanded > 0.0,
    ).astype(np.float32, copy=False)
