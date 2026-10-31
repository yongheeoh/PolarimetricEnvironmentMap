"""S2L2 (Spin-2 L=2 spherical harmonics) representation for Stokes vectors.

Provides a frame-free 12-parameter representation of 4-component Stokes
vectors. Linear polarization components (S1, S2) are projected onto the five
spin-2 l=2 spherical harmonic basis functions (real + imaginary = 10 params);
S0 and S3 pass through unchanged as spin-0 components.

Reference: SIGGRAPH Asia 2025 tutorial
https://vclab.kaist.ac.kr/siggraphasia2025p2/tutorial.html
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from pemap.polarization.mueller_reference import stokes_basis

_DEFAULT_FRAME_EPS = 1e-8


def build_measurement_frame(
    w: ArrayLike,
    basis_x: ArrayLike,
    *,
    eps: float = _DEFAULT_FRAME_EPS,
) -> NDArray[np.float64]:
    """Build a right-handed measurement frame from a measured basis axis.

    The resulting frame uses the supplied propagation direction as its third
    column and the measured basis axis as its first column after projection onto
    the plane orthogonal to the propagation direction. When the projected axis
    becomes numerically degenerate, the function falls back to the Duff basis.

    Parameters
    ----------
    w : array_like of shape (..., 3)
        Propagation direction(s). Need not be unit-length.
    basis_x : array_like of shape (..., 3)
        Measured reference axis that should define the frame's x column.
    eps : float, optional
        Numerical threshold used to detect degenerate directions.

    Returns
    -------
    F : numpy.ndarray of shape (..., 3, 3)
        Column-orthonormal rotation matrix for each direction.
    """
    w = np.asarray(w, dtype=float)
    basis_x = np.asarray(basis_x, dtype=float)
    if w.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {w.shape}")
    if basis_x.shape[-1] != 3:
        raise ValueError(f"Expected basis axis shape (..., 3), got {basis_x.shape}")

    shape_chan = np.broadcast_shapes(w.shape[:-1], basis_x.shape[:-1])
    w_bcast = np.broadcast_to(w, (*shape_chan, 3))
    basis_bcast = np.broadcast_to(basis_x, (*shape_chan, 3))

    w_norm = np.linalg.norm(w_bcast, axis=-1, keepdims=True)
    if np.any(w_norm < eps):
        raise ValueError("Propagation direction must be non-zero")
    w_n = w_bcast / w_norm

    basis_proj = basis_bcast - np.sum(basis_bcast * w_n, axis=-1, keepdims=True) * w_n
    basis_proj_norm = np.linalg.norm(basis_proj, axis=-1, keepdims=True)

    fallback_e1 = stokes_basis(w_n)
    use_fallback = basis_proj_norm < eps
    basis_proj_unit = basis_proj / np.where(use_fallback, 1.0, basis_proj_norm)
    e1 = np.where(use_fallback, fallback_e1, basis_proj_unit)

    e2_raw = np.cross(w_n, e1)
    e2_norm = np.linalg.norm(e2_raw, axis=-1, keepdims=True)
    fallback_e2 = np.cross(w_n, fallback_e1)
    use_fallback_e2 = e2_norm < eps
    e2 = np.where(use_fallback_e2, fallback_e2, e2_raw / np.where(use_fallback_e2, 1.0, e2_norm))
    e1 = np.cross(e2, w_n)

    return np.stack([e1, e2, w_n], axis=-1)


def duff_frame(w: ArrayLike) -> NDArray[np.float64]:
    """Build the 3×3 Duff measurement frame matrix for each direction.

    Constructs a right-handed rotation matrix whose columns are:
    - Column 0 (x): ``stokes_basis(w)`` — Duff reference polarisation direction
    - Column 1 (y): ``cross(w, e1)``  — second polarisation direction
    - Column 2 (z): ``w`` (normalised propagation direction)

    The resulting matrix F satisfies det(F) = +1 and F @ [0,0,1] = w.

    Parameters
    ----------
    w : array_like of shape (..., 3)
        Propagation direction(s). Need not be unit-length.

    Returns
    -------
    F : numpy.ndarray of shape (..., 3, 3)
        Column-orthonormal rotation matrix for each direction.
    """
    return build_measurement_frame(w, stokes_basis(w))


def thetaphi_frame(w: ArrayLike) -> NDArray[np.float64]:
    """Build the 3×3 measurement frame aligned with spherical (θ, φ) coordinates.

    Constructs a right-handed rotation matrix whose columns are:
    - Column 0 (x): e_θ — latitude tangent (theta direction)
    - Column 1 (y): cross(w, e_θ) — longitude tangent (phi direction)
    - Column 2 (z): w (normalised propagation direction)

    At the poles (θ ≈ 0 or π), e_θ degenerates; falls back to the Duff basis.

    Parameters
    ----------
    w : array_like of shape (..., 3)
        Propagation direction(s). Need not be unit-length.

    Returns
    -------
    F : numpy.ndarray of shape (..., 3, 3)
        Column-orthonormal rotation matrix for each direction.
    """
    w = np.asarray(w, dtype=float)
    n = w / np.linalg.norm(w, axis=-1, keepdims=True)

    # Y-up convention: n = (sinθ sinφ, cosθ, -sinθ cosφ)  →  n[1] = cosθ
    ny = n[..., 1:2]  # cos θ
    nxz = n[..., [0, 2]]  # (sinθ sinφ, -sinθ cosφ)
    sin_theta = np.linalg.norm(nxz, axis=-1, keepdims=True)  # sin θ

    # e_theta = (ny·nxz_hat[0], -sinθ, ny·nxz_hat[1]); fallback to Duff at poles
    nxz_hat = np.where(sin_theta > _DEFAULT_FRAME_EPS, nxz / sin_theta, stokes_basis(n)[..., [0, 2]])
    e_theta = np.concatenate([ny * nxz_hat[..., :1], -sin_theta, ny * nxz_hat[..., 1:2]], axis=-1)

    return build_measurement_frame(n, e_theta)


def _S2L2_basis_at_frame(F: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Evaluate spin-2 l=2 spherical harmonic basis at each measurement frame.

    Uses the quaternion method from the SIGGRAPH Asia 2025 tutorial.

    Parameters
    ----------
    F : numpy.ndarray of shape (..., 3, 3)
        Column-orthonormal rotation matrices (each column is a basis vector).

    Returns
    -------
    s2Y : numpy.ndarray of shape (..., 5), dtype=complex
        Five spin-2 SH basis values for m = -2, -1, 0, +1, +2.
    """
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) frame, got {F.shape}")
    shape_chan = F.shape[:-2]
    n_batch = math.prod(shape_chan) if shape_chan else 1

    # scipy Rotation expects (N, 3, 3) for from_matrix
    R = Rotation.from_matrix(F.reshape(n_batch, 3, 3))

    # scipy as_quat() returns [x, y, z, w]
    q = R.as_quat()  # (N, 4)

    qa_conj = q[..., 3] - 1j * q[..., 2]  # conj(w + iz)
    qb_conj = q[..., 1] - 1j * q[..., 0]  # conj(y + ix)

    s2Y = np.empty((n_batch, 5), dtype=complex)
    s2Y[:] = np.sqrt(5.0 / np.pi)
    s2Y[..., 0] *= 0.5 * qa_conj**4
    s2Y[..., 1] *= -(qa_conj**3) * qb_conj
    s2Y[..., 2] *= np.sqrt(1.5) * (qa_conj * qb_conj) ** 2
    s2Y[..., 3] *= -qa_conj * qb_conj**3
    s2Y[..., 4] *= 0.5 * qb_conj**4

    return s2Y.reshape(*shape_chan, 5)


def stokes_to_s2l2(
    stk: ArrayLike,
    F: ArrayLike,
) -> NDArray[np.float64]:
    """Convert 4-component Stokes vectors to the 12-parameter S2L2 representation.

    The S2L2 representation is frame-free: given the same physical polarisation
    state expressed in *any* measurement frame, the returned 12 coefficients are
    identical (up to floating-point precision).

    Layout of the returned vector (12 elements):
        [S0, Re(c_{-2}), Im(c_{-2}), Re(c_{-1}), Im(c_{-1}),
              Re(c_0),  Im(c_0),  Re(c_{+1}), Im(c_{+1}),
              Re(c_{+2}), Im(c_{+2}), S3]

    Parameters
    ----------
    stk : array_like of shape (..., 4)
        Stokes vectors [S0, S1, S2, S3] in the frame defined by F.
    F : array_like of shape (..., 3, 3)
        Measurement frame matrices. Must broadcast with ``stk[..., :-1]``.

    Returns
    -------
    s2l2 : numpy.ndarray of shape (..., 12)
        S2L2 coefficient vector.
    """
    stk = np.asarray(stk, dtype=float)
    F = np.asarray(F, dtype=float)
    if stk.shape[-1] != 4:
        raise ValueError(f"Expected Stokes shape (..., 4), got {stk.shape}")
    if F.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame shape (..., 3, 3), got {F.shape}")

    shape_chan = np.broadcast_shapes(stk.shape[:-1], F.shape[:-2])
    s2l2 = np.empty((*shape_chan, 12), dtype=float)

    # Spin-0 components pass through
    s2l2[..., 0] = stk[..., 0]
    s2l2[..., -1] = stk[..., 3]

    # Spin-2: project s1 + i*s2 onto conjugate spin-2 SH basis
    s1_is2 = stk[..., 1] + 1j * stk[..., 2]  # (...,)
    s2Y = _S2L2_basis_at_frame(F)  # (..., 5)

    # prod[..., m] = conj(s2Y[..., m]) * (s1 + i*s2)
    prod = np.conj(s2Y) * s1_is2[..., np.newaxis]  # (..., 5)

    # Interleave real and imaginary parts: [Re(c_{-2}), Im(c_{-2}), ..., Re(c_{+2}), Im(c_{+2})]
    coeffs10 = np.stack([prod.real, prod.imag], axis=-1).reshape(*shape_chan, 10)
    s2l2[..., 1:-1] = coeffs10 * (2.0 * np.sqrt(np.pi / 5.0))

    return s2l2


def s2l2_to_stokes(
    s2l2: ArrayLike,
    F: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct Stokes vectors from the S2L2 representation given a target frame.

    Parameters
    ----------
    s2l2 : array_like of shape (..., 12)
        S2L2 coefficient vectors as returned by :func:`stokes_to_s2l2`.
    F : array_like of shape (..., 3, 3)
        Target measurement frame matrices.

    Returns
    -------
    stk : numpy.ndarray of shape (..., 4)
        Reconstructed Stokes vectors [S0, S1, S2, S3].
    """
    s2l2 = np.asarray(s2l2, dtype=float)
    F = np.asarray(F, dtype=float)
    if s2l2.shape[-1] != 12:
        raise ValueError(f"Expected S2L2 shape (..., 12), got {s2l2.shape}")
    if F.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame shape (..., 3, 3), got {F.shape}")

    shape_chan = np.broadcast_shapes(s2l2.shape[:-1], F.shape[:-2])
    stk = np.empty((*shape_chan, 4), dtype=float)

    # Spin-0 pass-through
    stk[..., 0] = s2l2[..., 0]
    stk[..., 3] = s2l2[..., -1]

    # Reconstruct complex coefficients from interleaved real/imag storage
    # s2l2[..., 1:-1] has layout [Re(c0), Im(c0), Re(c1), Im(c1), ...]
    inner = s2l2[..., 1:-1]  # (..., 10)
    c_real = inner[..., 0::2]  # (..., 5)
    c_imag = inner[..., 1::2]  # (..., 5)
    s2l2_5 = c_real + 1j * c_imag  # (..., 5) complex coefficients

    s2Y = _S2L2_basis_at_frame(F)  # (..., 5)

    dot = np.sum(s2Y * s2l2_5, axis=-1)  # (...,) complex

    factor = np.sqrt(4.0 * np.pi / 5.0)
    stk[..., 1] = factor * dot.real
    stk[..., 2] = factor * dot.imag

    return stk
