"""Fast minimum rotations for batched single-vector alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = ["align_vectors_batched"]


def _resolve_float_dtype(a: ArrayLike, b: ArrayLike, dtype: DTypeLike | None) -> np.dtype[np.floating[Any]]:
    if dtype is None:
        a = np.asarray(a)
        b = np.asarray(b)
        dtype = np.result_type(a, b)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(f"`dtype` must be a floating dtype, got {dtype}.")
    return np.dtype(dtype)


def _as_vector_batch(a: ArrayLike, b: ArrayLike, dtype: DTypeLike | None = None) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], tuple[int, ...]]:
    dtype = _resolve_float_dtype(a, b, dtype)
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)

    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"Expected input `a` to have shape (..., 3), got {a.shape}.")
    if b.ndim < 1 or b.shape[-1] != 3:
        raise ValueError(f"Expected input `b` to have shape (..., 3), got {b.shape}.")

    try:
        broadcast_shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    except ValueError as exc:
        raise ValueError(f"Expected inputs `a` and `b` to have broadcastable leading shapes, got {a.shape} and {b.shape} respectively.") from exc

    vector_shape = broadcast_shape + (3,)
    a = np.broadcast_to(a, vector_shape).reshape(-1, 3)
    b = np.broadcast_to(b, vector_shape).reshape(-1, 3)

    return a, b, broadcast_shape


def _quat_from_rotvec(rotvec: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    angle = np.linalg.norm(rotvec, axis=-1)
    half_angle = 0.5 * angle
    small = angle <= 1e-8

    scale = np.empty_like(angle)
    large = ~small
    scale[large] = np.sin(half_angle[large]) / angle[large]

    angle2 = angle[small] * angle[small]
    scale[small] = 0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0

    quat = np.empty(rotvec.shape[:-1] + (4,), dtype=rotvec.dtype)
    quat[..., :3] = rotvec * scale[..., None]
    quat[..., 3] = np.cos(half_angle)
    return quat


def _compose_quat(p: NDArray[np.floating[Any]], q: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    p_vec = p[..., :3]
    q_vec = q[..., :3]
    p_w = p[..., 3]
    q_w = q[..., 3]

    out = np.empty_like(p)
    out[..., :3] = p_w[..., None] * q_vec + q_w[..., None] * p_vec + np.cross(p_vec, q_vec)
    out[..., 3] = p_w * q_w - np.sum(p_vec * q_vec, axis=-1)
    return out


def _quat_to_matrix(quat: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype)
    matrix[..., 0, 0] = ww + xx - yy - zz
    matrix[..., 0, 1] = 2.0 * (xy - zw)
    matrix[..., 0, 2] = 2.0 * (xz + yw)
    matrix[..., 1, 0] = 2.0 * (xy + zw)
    matrix[..., 1, 1] = ww - xx + yy - zz
    matrix[..., 1, 2] = 2.0 * (yz - xw)
    matrix[..., 2, 0] = 2.0 * (xz - yw)
    matrix[..., 2, 1] = 2.0 * (yz + xw)
    matrix[..., 2, 2] = ww - xx - yy + zz
    return matrix


def align_vectors_batched(a: ArrayLike, b: ArrayLike, dtype: DTypeLike | None = None) -> NDArray[np.floating[Any]]:
    """Compute minimum rotations for many independent vector pairs.

    This is a vectorized implementation of the single-vector special case of
    ``scipy.spatial.transform.Rotation.align_vectors``. For each row ``i``,
    the returned matrix is the shortest active rotation that maps the direction
    of ``b[i]`` to the direction of ``a[i]``. Vector magnitudes do not affect
    the rotation, matching SciPy's single-vector behavior.

    This function does not implement SciPy's general Kabsch/Wahba alignment for
    multiple vectors per observation, weights, RSSD, or sensitivity matrices.

    Parameters
    ----------
    a : array_like, shape (..., 3)
        Target vectors in frame A. Leading dimensions are broadcast against
        ``b`` following NumPy broadcasting rules.
    b : array_like, shape (..., 3)
        Source vectors in frame B. Each output rotation maps ``b[i]`` onto
        ``a[i]`` after broadcasting leading dimensions.
    dtype : data-type, optional
        Floating dtype used for computation and output. If ``None``, the dtype
        follows ``numpy.result_type(a, b)`` for floating inputs; non-floating
        inputs are promoted to ``float64``.

    Returns
    -------
    rotation_matrices : ndarray, shape ``broadcast_shape + (3, 3)``
        Rotation matrices, where ``broadcast_shape`` is the broadcasted shape
        of ``a.shape[:-1]`` and ``b.shape[:-1]``. The dtype is the resolved
        floating dtype.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` does not have shape ``(..., 3)``, if their leading
        dimensions are not broadcastable, or if any vector has zero length.
    TypeError
        If ``dtype`` is provided and is not a floating dtype.

    See Also
    --------
    scipy.spatial.transform.Rotation.align_vectors :
        SciPy reference implementation for aligning vector observations.

    Notes
    -----
    The sign convention matches SciPy: the core cross product is
    ``cross(normalize(b), normalize(a))``. Rows near 180 degrees use SciPy's
    stable flip-and-correct construction so anti-parallel cases match SciPy's
    deterministic rotation choice.
    """
    a, b, broadcast_shape = _as_vector_batch(a, b, dtype=dtype)
    dtype = a.dtype
    n = a.shape[0]
    if n == 0:
        return np.empty(broadcast_shape + (3, 3), dtype=dtype)

    a_norm = np.sqrt(np.einsum("ij,ij->i", a, a))
    b_norm = np.sqrt(np.einsum("ij,ij->i", b, b))
    if np.any(a_norm == 0.0) or np.any(b_norm == 0.0):
        raise ValueError("Cannot align zero length primary vectors")

    a_unit = a / a_norm[:, None]
    b_unit = b / b_norm[:, None]

    cross = np.empty_like(a_unit)
    cross[:, 0] = b_unit[:, 1] * a_unit[:, 2] - b_unit[:, 2] * a_unit[:, 1]
    cross[:, 1] = b_unit[:, 2] * a_unit[:, 0] - b_unit[:, 0] * a_unit[:, 2]
    cross[:, 2] = b_unit[:, 0] * a_unit[:, 1] - b_unit[:, 1] * a_unit[:, 0]
    cross_norm = np.sqrt(np.einsum("ij,ij->i", cross, cross))
    dot = np.einsum("ij,ij->i", a_unit, b_unit)
    theta = np.arctan2(cross_norm, dot)

    tolerance = 1e-3
    flip = (np.pi - theta) < tolerance
    identity = np.eye(3, dtype=dtype)
    matrix = np.broadcast_to(identity, (n, 3, 3)).copy()

    direct = (~flip) & (cross_norm != 0.0)
    if np.any(direct):
        direct_cross = cross[direct]
        vx = direct_cross[:, 0]
        vy = direct_cross[:, 1]
        vz = direct_cross[:, 2]
        s2 = cross_norm[direct] * cross_norm[direct]
        scale = (1.0 - dot[direct]) / s2

        direct_matrix = np.empty((direct_cross.shape[0], 3, 3), dtype=dtype)
        direct_matrix[:, 0, 0] = 1.0 + scale * (vx * vx - s2)
        direct_matrix[:, 0, 1] = -vz + scale * vx * vy
        direct_matrix[:, 0, 2] = vy + scale * vx * vz
        direct_matrix[:, 1, 0] = vz + scale * vx * vy
        direct_matrix[:, 1, 1] = 1.0 + scale * (vy * vy - s2)
        direct_matrix[:, 1, 2] = -vx + scale * vy * vz
        direct_matrix[:, 2, 0] = -vy + scale * vx * vz
        direct_matrix[:, 2, 1] = vx + scale * vy * vz
        direct_matrix[:, 2, 2] = 1.0 + scale * (vz * vz - s2)
        matrix[direct] = direct_matrix

    if np.any(flip):
        flip_axis = cross[flip].copy()
        flip_cross_norm = cross_norm[flip]
        flip_theta = theta[flip].copy()
        flip_a_unit = a_unit[flip]
        q_flip = np.zeros((flip_axis.shape[0], 4), dtype=dtype)
        q_flip[:, 3] = 1.0

        antiparallel = flip_cross_norm == 0.0

        if np.any(antiparallel):
            a_anti = flip_a_unit[antiparallel]
            idx = np.argmin(np.abs(a_anti), axis=1)
            fallback_axis = np.zeros_like(a_anti)
            rows = np.arange(a_anti.shape[0])
            fallback_axis[rows, (idx - 1) % 3] = a_anti[rows, (idx - 2) % 3]
            fallback_axis[rows, (idx - 2) % 3] = -a_anti[rows, (idx - 1) % 3]
            flip_axis[antiparallel] = fallback_axis

        flip_axis = flip_axis / np.linalg.norm(flip_axis, axis=1)[:, None]
        q_flip[:, :3] = flip_axis
        q_flip[:, 3] = 0.0

        flip_theta = np.pi - flip_theta
        flip_cross = -cross[flip]

        theta2 = flip_theta * flip_theta
        small = np.abs(flip_theta) < tolerance
        scale = np.empty_like(flip_theta)
        scale[small] = 1.0 + theta2[small] / 6.0 + theta2[small] * theta2[small] * 7.0 / 360.0

        large = ~small
        scale[large] = flip_theta[large] / np.sin(flip_theta[large])
        rotvec = flip_cross * scale[:, None]
        matrix[flip] = _quat_to_matrix(_compose_quat(_quat_from_rotvec(rotvec), q_flip))

    return matrix.reshape(broadcast_shape + (3, 3))
