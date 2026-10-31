"""
NumPy re-implementation of Mitsuba 3's polarimetric utility functions
(mitsuba.mueller), adapted for offline batch processing outside the renderer.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def stokes_basis(w: ArrayLike) -> NDArray[np.float64]:
    """Compute the reference frame basis vector for a Stokes vector.

    For light transport involving polarized quantities it is essential to keep
    track of reference frames. A Stokes vector is only meaningful if we also
    know w.r.t. which basis this state of light is observed. In Mitsuba, these
    reference frames are never explicitly stored but instead can be computed on
    the fly using this function.

    Implements ``coordinate_system(n).first`` following Duff et al. JCGT 2017,
    identical to ``mi.mueller.stokes_basis``.

    Notes
    -----
    Based on the Mitsuba3 source:
    ``include/mitsuba/render/mueller.h`` (``stokes_basis``) and
    ``include/mitsuba/core/vector.h`` (``coordinate_system``).

    Parameters
    ----------
    w : array_like of shape (..., 3)
        Direction vector(s). Need not be unit-length; they are normalised
        internally.

    Returns
    -------
    basis : numpy.ndarray of shape (..., 3)
        Unit basis vector(s) orthogonal to each input direction, defining the
        reference frame for the corresponding Stokes vector.
    """
    w = np.asarray(w, dtype=float)
    # Normalise
    norm = np.linalg.norm(w, axis=-1, keepdims=True)
    n = w / norm

    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]

    # Dr.Jit sign: sign(0) = 1, so treat 0 as positive
    sign = np.where(nz >= 0, 1.0, -1.0)
    a = -1.0 / (sign + nz)

    bx = nx * nx * a * sign + 1.0
    by = nx * ny * a * sign
    bz = -nx * sign

    return np.stack([bx, by, bz], axis=-1)


def rotate_stokes_basis(
    wi: ArrayLike,
    basis_current: ArrayLike,
    basis_target: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the Mueller matrix that rotates between two Stokes reference frames.

    Given a Stokes vector ``s`` expressed in ``basis_current``, the returned
    matrix ``M`` satisfies ``M @ s = s’``, where ``s’`` is the same physical
    state of light expressed in ``basis_target``. Both basis vectors must be
    orthogonal to the propagation direction ``wi``.

    For example, horizontally polarised light ``[1, 1, 0, 0]`` in basis
    ``[1, 0, 0]`` becomes ``[1, 0, 1, 0]`` (+45° linear) in basis
    ``[0.707, -0.707, 0]``.

    Identical to ``mi.mueller.rotate_stokes_basis``.

    Notes
    -----
    Based on the Mitsuba3 source:
    ``include/mitsuba/render/mueller.h`` (``rotate_stokes_basis`` and
    ``rotator``).

    Parameters
    ----------
    wi : array_like of shape (..., 3)
        Propagation direction(s) of the light (need not be unit-length).
    basis_current : array_like of shape (..., 3)
        Basis vector(s) of the current Stokes reference frame. Must be
        orthogonal to the corresponding ``wi``.
    basis_target : array_like of shape (..., 3)
        Basis vector(s) of the target Stokes reference frame. Must be
        orthogonal to the corresponding ``wi``.

    Returns
    -------
    M : numpy.ndarray of shape (..., 4, 4)
        Mueller rotation matrix (rotator) for each input triplet. Applying
        ``M @ s`` re-expresses Stokes vector ``s`` in ``basis_target``.
    """
    wi = np.asarray(wi, dtype=float)
    basis_current = np.asarray(basis_current, dtype=float)
    basis_target = np.asarray(basis_target, dtype=float)

    # Step 1: Normalise basis_current and basis_target
    bc_n = basis_current / np.linalg.norm(basis_current, axis=-1, keepdims=True)
    bt_n = basis_target / np.linalg.norm(basis_target, axis=-1, keepdims=True)

    # Step 2: Compute unsigned angle using numerically stable unit_angle formula
    # theta = 2 * atan2(|bc_n - bt_n|, |bc_n + bt_n|)
    diff_norm = np.linalg.norm(bc_n - bt_n, axis=-1)
    sum_norm = np.linalg.norm(bc_n + bt_n, axis=-1)
    theta = 2.0 * np.arctan2(diff_norm, sum_norm)

    # Step 3: Handedness correction using original (unnormalised) basis vectors
    # if dot(wi, cross(basis_current, basis_target)) < 0: theta *= -1
    cross_bc_bt = np.cross(basis_current, basis_target)
    dot_wi_cross = np.sum(wi * cross_bc_bt, axis=-1)
    theta = np.where(dot_wi_cross < 0, -theta, theta)

    # Step 4: Build 4x4 Mueller rotator matrix
    # s = sin(2*theta), c = cos(2*theta)
    s = np.sin(2.0 * theta)
    c = np.cos(2.0 * theta)

    # Shape (...,) -> (..., 4, 4)
    batch_shape = theta.shape
    zeros = np.zeros(batch_shape)
    ones = np.ones(batch_shape)

    # Build rows of the Mueller rotator
    # [[1,  0,  0,  0],
    #  [0,  c,  s,  0],
    #  [0, -s,  c,  0],
    #  [0,  0,  0,  1]]
    row0 = np.stack([ones, zeros, zeros, zeros], axis=-1)
    row1 = np.stack([zeros, c, s, zeros], axis=-1)
    row2 = np.stack([zeros, -s, c, zeros], axis=-1)
    row3 = np.stack([zeros, zeros, zeros, ones], axis=-1)

    matrix = np.stack([row0, row1, row2, row3], axis=-2)
    return matrix
