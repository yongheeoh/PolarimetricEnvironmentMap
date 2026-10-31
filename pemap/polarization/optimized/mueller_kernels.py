"""Fast Mueller/Stokes reference-frame helpers.

This module mirrors :mod:`pemap.polarization.mueller_reference` with
Numba kernels for large image-shaped batches.
"""

from __future__ import annotations

import numpy as np

from pemap.polarization.mueller_reference import rotate_stokes_basis as _ref_rotate_stokes_basis, stokes_basis as _ref_stokes_basis

try:
    from numba import njit
    from numba import prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - only exercised when numba is absent/broken
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


__all__ = [
    "NUMBA_AVAILABLE",
    "rotate_stokes_basis",
    "stokes_basis",
]


def stokes_basis(w: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute Mitsuba-compatible Duff Stokes basis vectors."""

    directions = np.asarray(w, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected direction shape (..., 3), got {directions.shape}")

    shape_chan = directions.shape[:-1]
    expected_shape = (*shape_chan, 3) if shape_chan else (3,)

    if not NUMBA_AVAILABLE:
        result = _ref_stokes_basis(directions)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    directions_flat = np.ascontiguousarray(directions.reshape(n_batch, 3))
    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    _stokes_basis_kernel(directions_flat, output.reshape(n_batch, 3))
    return output


def rotate_stokes_basis(
    wi: np.ndarray,
    basis_current: np.ndarray,
    basis_target: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Mueller rotators between Stokes reference bases."""

    directions = np.asarray(wi, dtype=np.float64)
    current = np.asarray(basis_current, dtype=np.float64)
    target = np.asarray(basis_target, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected wi shape (..., 3), got {directions.shape}")
    if current.ndim < 1 or current.shape[-1] != 3:
        raise ValueError(f"Expected basis_current shape (..., 3), got {current.shape}")
    if target.ndim < 1 or target.shape[-1] != 3:
        raise ValueError(f"Expected basis_target shape (..., 3), got {target.shape}")

    shape_chan = np.broadcast_shapes(directions.shape[:-1], current.shape[:-1], target.shape[:-1])
    expected_shape = (*shape_chan, 4, 4) if shape_chan else (4, 4)

    if not NUMBA_AVAILABLE:
        result = _ref_rotate_stokes_basis(directions, current, target)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    directions_flat = np.ascontiguousarray(np.broadcast_to(directions, (*shape_chan, 3)).reshape(n_batch, 3))
    current_flat = np.ascontiguousarray(np.broadcast_to(current, (*shape_chan, 3)).reshape(n_batch, 3))
    target_flat = np.ascontiguousarray(np.broadcast_to(target, (*shape_chan, 3)).reshape(n_batch, 3))
    _rotate_stokes_basis_kernel(directions_flat, current_flat, target_flat, output.reshape(n_batch, 4, 4))
    return output


def _prepare_output(out: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    output = np.asarray(out)
    if output.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {output.shape}")
    if output.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64, got {output.dtype}")
    if not output.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return output


if NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _write_stokes_basis_sample(wx: float, wy: float, wz: float, output: np.ndarray, i: int) -> None:
        norm = np.sqrt(wx * wx + wy * wy + wz * wz)
        nx = wx / norm
        ny = wy / norm
        nz = wz / norm

        sign = 1.0
        if nz < 0.0:
            sign = -1.0
        a = -1.0 / (sign + nz)

        output[i, 0] = nx * nx * a * sign + 1.0
        output[i, 1] = nx * ny * a * sign
        output[i, 2] = -nx * sign

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_basis_kernel(directions: np.ndarray, output: np.ndarray) -> None:
        for i in prange(directions.shape[0]):
            _write_stokes_basis_sample(directions[i, 0], directions[i, 1], directions[i, 2], output, i)

    @njit(parallel=True, cache=True, fastmath=False)
    def _rotate_stokes_basis_kernel(
        directions: np.ndarray,
        current: np.ndarray,
        target: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for i in prange(directions.shape[0]):
            bcx = current[i, 0]
            bcy = current[i, 1]
            bcz = current[i, 2]
            btx = target[i, 0]
            bty = target[i, 1]
            btz = target[i, 2]

            bc_norm = np.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
            bt_norm = np.sqrt(btx * btx + bty * bty + btz * btz)
            bcnx = bcx / bc_norm
            bcny = bcy / bc_norm
            bcnz = bcz / bc_norm
            btnx = btx / bt_norm
            btny = bty / bt_norm
            btnz = btz / bt_norm

            dx = bcnx - btnx
            dy = bcny - btny
            dz = bcnz - btnz
            sx = bcnx + btnx
            sy = bcny + btny
            sz = bcnz + btnz
            diff_norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            sum_norm = np.sqrt(sx * sx + sy * sy + sz * sz)
            theta = 2.0 * np.arctan2(diff_norm, sum_norm)

            cx = bcy * btz - bcz * bty
            cy = bcz * btx - bcx * btz
            cz = bcx * bty - bcy * btx
            dot_wi_cross = directions[i, 0] * cx + directions[i, 1] * cy + directions[i, 2] * cz
            if dot_wi_cross < 0.0:
                theta = -theta

            s = np.sin(2.0 * theta)
            c = np.cos(2.0 * theta)

            output[i, 0, 0] = 1.0
            output[i, 0, 1] = 0.0
            output[i, 0, 2] = 0.0
            output[i, 0, 3] = 0.0
            output[i, 1, 0] = 0.0
            output[i, 1, 1] = c
            output[i, 1, 2] = s
            output[i, 1, 3] = 0.0
            output[i, 2, 0] = 0.0
            output[i, 2, 1] = -s
            output[i, 2, 2] = c
            output[i, 2, 3] = 0.0
            output[i, 3, 0] = 0.0
            output[i, 3, 1] = 0.0
            output[i, 3, 2] = 0.0
            output[i, 3, 3] = 1.0

else:

    def _missing_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Numba is not available")

    _stokes_basis_kernel = _missing_numba
    _rotate_stokes_basis_kernel = _missing_numba
