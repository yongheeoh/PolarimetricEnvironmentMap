"""Fast S2L2/Stokes conversion kernels.

This module provides drop-in compatible conversion functions for hot paths that
need fused Numba kernels instead of the reference implementation in
:mod:`pemap.polarization.s2l2_reference`.
"""

from __future__ import annotations

import numpy as np

from pemap.polarization.s2l2_reference import _S2L2_basis_at_frame as _ref_S2L2_basis_at_frame, build_measurement_frame as _ref_build_measurement_frame, duff_frame as _ref_duff_frame, s2l2_to_stokes as _ref_s2l2_to_stokes, stokes_to_s2l2 as _ref_stokes_to_s2l2, thetaphi_frame as _ref_thetaphi_frame

try:
    from numba import njit
    from numba import prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - only exercised when numba is absent/broken
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


_SQRT_1_5 = np.sqrt(1.5)
_TWO_SQRT_1_5 = 2.0 * _SQRT_1_5
_SQRT_5_OVER_PI = np.sqrt(5.0 / np.pi)
_DEFAULT_FRAME_EPS = 1e-8

__all__ = [
    "NUMBA_AVAILABLE",
    "_S2L2_basis_at_frame",
    "build_measurement_frame",
    "duff_frame",
    "s2l2_to_stokes",
    "s2l2_to_stokes_float32",
    "stokes_to_s2l2",
    "stokes_to_s2l2_measurement_frame",
    "stokes_to_s2l2_measurement_frame_float32",
    "thetaphi_frame",
]


def build_measurement_frame(
    w: np.ndarray,
    basis_x: np.ndarray,
    *,
    eps: float = _DEFAULT_FRAME_EPS,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build right-handed measurement frames with a Numba kernel."""

    directions = np.asarray(w, dtype=np.float64)
    basis = np.asarray(basis_x, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {directions.shape}")
    if basis.ndim < 1 or basis.shape[-1] != 3:
        raise ValueError(f"Expected basis axis shape (..., 3), got {basis.shape}")

    shape_chan = np.broadcast_shapes(directions.shape[:-1], basis.shape[:-1])
    expected_shape = (*shape_chan, 3, 3) if shape_chan else (3, 3)

    if not NUMBA_AVAILABLE:
        result = _ref_build_measurement_frame(directions, basis, eps=eps)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    directions_bcast = np.broadcast_to(directions, (*shape_chan, 3))
    directions_flat = np.ascontiguousarray(directions_bcast.reshape(n_batch, 3))
    basis_flat = np.ascontiguousarray(np.broadcast_to(basis, (*shape_chan, 3)).reshape(n_batch, 3))
    status = np.zeros(n_batch, dtype=np.uint8)
    _build_measurement_frame_kernel(directions_flat, basis_flat, float(eps), output.reshape(n_batch, 3, 3), status)
    if np.any(status):
        raise ValueError("Propagation direction must be non-zero")
    return output


def duff_frame(w: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Build Duff/Stokes-basis measurement frames."""

    directions = np.asarray(w, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {directions.shape}")

    shape_chan = directions.shape[:-1]
    expected_shape = (*shape_chan, 3, 3) if shape_chan else (3, 3)

    if not NUMBA_AVAILABLE:
        result = _ref_duff_frame(directions)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    directions_flat = np.ascontiguousarray(directions.reshape(n_batch, 3))
    status = np.zeros(n_batch, dtype=np.uint8)
    _duff_frame_kernel(directions_flat, _DEFAULT_FRAME_EPS, output.reshape(n_batch, 3, 3), status)
    if np.any(status):
        raise ValueError("Propagation direction must be non-zero")
    return output


def thetaphi_frame(w: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Build measurement frames aligned with spherical theta/phi coordinates."""

    directions = np.asarray(w, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {directions.shape}")

    shape_chan = directions.shape[:-1]
    expected_shape = (*shape_chan, 3, 3) if shape_chan else (3, 3)

    if not NUMBA_AVAILABLE:
        result = _ref_thetaphi_frame(directions)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    directions_flat = np.ascontiguousarray(directions.reshape(n_batch, 3))
    _thetaphi_frame_kernel(directions_flat, _DEFAULT_FRAME_EPS, output.reshape(n_batch, 3, 3))
    return output


def _S2L2_basis_at_frame(F: np.ndarray) -> np.ndarray:
    """Evaluate spin-2 l=2 spherical harmonics at measurement frames."""

    if not NUMBA_AVAILABLE:
        return _ref_S2L2_basis_at_frame(F)

    frames = np.asarray(F, dtype=np.float64)
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) frame, got {frames.shape}")

    shape_chan = frames.shape[:-2]
    n_batch = int(np.prod(shape_chan, dtype=np.int64)) if shape_chan else 1
    frames_flat = np.ascontiguousarray(frames.reshape(n_batch, 3, 3))
    expected_shape = (*shape_chan, 5) if shape_chan else (5,)
    output = np.empty(expected_shape, dtype=np.complex128)
    _s2l2_basis_at_frame_kernel(frames_flat, output.reshape(n_batch, 5))
    return output


def stokes_to_s2l2_measurement_frame(
    stk: np.ndarray,
    w: np.ndarray,
    basis_x: np.ndarray,
    *,
    eps: float = _DEFAULT_FRAME_EPS,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert Stokes vectors to S2L2 from measured frame inputs.

    This is a hot-path helper equivalent to first calling
    ``build_measurement_frame(w, basis_x)`` and then ``stokes_to_s2l2`` with
    that frame broadcast across channels. It avoids materializing the full
    frame and reuses each pixel's spin-2 terms across channel samples.
    """

    stokes = np.asarray(stk, dtype=np.float64)
    directions = np.asarray(w, dtype=np.float64)
    basis = np.asarray(basis_x, dtype=np.float64)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError(f"Expected Stokes shape (..., 4), got {stokes.shape}")
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {directions.shape}")
    if basis.ndim < 1 or basis.shape[-1] != 3:
        raise ValueError(f"Expected basis axis shape (..., 3), got {basis.shape}")

    frame_shape = np.broadcast_shapes(directions.shape[:-1], basis.shape[:-1])
    stokes_prefix = stokes.shape[:-1]
    channel_broadcast = len(stokes_prefix) == len(frame_shape) + 1 and stokes_prefix[:-1] == frame_shape
    output_prefix = stokes_prefix if channel_broadcast else np.broadcast_shapes(stokes_prefix, frame_shape)
    expected_shape = (*output_prefix, 12)

    if not NUMBA_AVAILABLE:
        if out is not None:
            _prepare_output(out, expected_shape, "out")
        F = _ref_build_measurement_frame(directions, basis, eps=eps)
        F_for_ref = F[..., None, :, :] if channel_broadcast else F
        result = _ref_stokes_to_s2l2(stokes, F_for_ref)
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    output_flat = output.reshape(-1, 12)
    if channel_broadcast:
        n_chan = int(stokes_prefix[-1])
        n_frame = int(np.prod(frame_shape, dtype=np.int64)) if frame_shape else 1
        stokes_flat = np.ascontiguousarray(stokes.reshape(n_frame * n_chan, 4))
        directions_flat = np.ascontiguousarray(np.broadcast_to(directions, (*frame_shape, 3)).reshape(n_frame, 3))
        basis_flat = np.ascontiguousarray(np.broadcast_to(basis, (*frame_shape, 3)).reshape(n_frame, 3))
        terms = np.empty((n_frame, 10), dtype=np.float64)
        status = np.zeros(n_frame, dtype=np.uint8)
        _measurement_frame_spin2_terms_kernel(directions_flat, basis_flat, float(eps), terms, status)
        if np.any(status):
            raise ValueError("Propagation direction must be non-zero")
        if n_chan == 3:
            _stokes_to_s2l2_channel3_terms_kernel(stokes_flat, terms, output_flat)
        else:
            _stokes_to_s2l2_channel_terms_kernel(stokes_flat, terms, output_flat, n_chan)
        return output

    n_batch = int(np.prod(output_prefix, dtype=np.int64)) if output_prefix else 1
    stokes_flat = np.ascontiguousarray(np.broadcast_to(stokes, (*output_prefix, 4)).reshape(n_batch, 4))
    directions_flat = np.ascontiguousarray(np.broadcast_to(directions, (*output_prefix, 3)).reshape(n_batch, 3))
    basis_flat = np.ascontiguousarray(np.broadcast_to(basis, (*output_prefix, 3)).reshape(n_batch, 3))
    terms = np.empty((n_batch, 10), dtype=np.float64)
    status = np.zeros(n_batch, dtype=np.uint8)
    _measurement_frame_spin2_terms_kernel(directions_flat, basis_flat, float(eps), terms, status)
    if np.any(status):
        raise ValueError("Propagation direction must be non-zero")
    _stokes_to_s2l2_terms_flat_kernel(stokes_flat, terms, output_flat)
    return output


def stokes_to_s2l2_measurement_frame_float32(
    stk: np.ndarray,
    w: np.ndarray,
    basis_x: np.ndarray,
    *,
    eps: float = _DEFAULT_FRAME_EPS,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Float32 fused S2L2 ingest for the main environment-map hot path."""

    stokes = np.asarray(stk)
    directions = np.asarray(w, dtype=np.float64)
    basis = np.asarray(basis_x, dtype=np.float64)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError(f"Expected Stokes shape (..., 4), got {stokes.shape}")
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(f"Expected propagation direction shape (..., 3), got {directions.shape}")
    if basis.ndim < 1 or basis.shape[-1] != 3:
        raise ValueError(f"Expected basis axis shape (..., 3), got {basis.shape}")

    frame_shape = np.broadcast_shapes(directions.shape[:-1], basis.shape[:-1])
    stokes_prefix = stokes.shape[:-1]
    channel_broadcast = len(stokes_prefix) == len(frame_shape) + 1 and stokes_prefix[:-1] == frame_shape
    output_prefix = stokes_prefix if channel_broadcast else np.broadcast_shapes(stokes_prefix, frame_shape)
    expected_shape = (*output_prefix, 12)

    if not NUMBA_AVAILABLE:
        if out is not None:
            _prepare_output_dtype(out, expected_shape, "out", np.float32)
        F = _ref_build_measurement_frame(directions, basis, eps=eps)
        F_for_ref = F[..., None, :, :] if channel_broadcast else F
        result = _ref_stokes_to_s2l2(np.asarray(stokes, dtype=np.float64), F_for_ref).astype(np.float32)
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output_dtype(out, expected_shape, "out", np.float32) if out is not None else np.empty(expected_shape, dtype=np.float32)
    output_flat = output.reshape(-1, 12)
    if channel_broadcast:
        n_chan = int(stokes_prefix[-1])
        n_frame = int(np.prod(frame_shape, dtype=np.int64)) if frame_shape else 1
        stokes_flat = np.ascontiguousarray(stokes.reshape(n_frame * n_chan, 4))
        directions_flat = np.ascontiguousarray(np.broadcast_to(directions, (*frame_shape, 3)).reshape(n_frame, 3))
        basis_flat = np.ascontiguousarray(np.broadcast_to(basis, (*frame_shape, 3)).reshape(n_frame, 3))
        terms = np.empty((n_frame, 10), dtype=np.float32)
        status = np.zeros(n_frame, dtype=np.uint8)
        _measurement_frame_spin2_terms_kernel(directions_flat, basis_flat, float(eps), terms, status)
        if np.any(status):
            raise ValueError("Propagation direction must be non-zero")
        if n_chan == 3:
            _stokes_to_s2l2_channel3_terms_kernel(stokes_flat, terms, output_flat)
        else:
            _stokes_to_s2l2_channel_terms_kernel(stokes_flat, terms, output_flat, n_chan)
        return output

    n_batch = int(np.prod(output_prefix, dtype=np.int64)) if output_prefix else 1
    stokes_flat = np.ascontiguousarray(np.broadcast_to(stokes, (*output_prefix, 4)).reshape(n_batch, 4))
    directions_flat = np.ascontiguousarray(np.broadcast_to(directions, (*output_prefix, 3)).reshape(n_batch, 3))
    basis_flat = np.ascontiguousarray(np.broadcast_to(basis, (*output_prefix, 3)).reshape(n_batch, 3))
    terms = np.empty((n_batch, 10), dtype=np.float32)
    status = np.zeros(n_batch, dtype=np.uint8)
    _measurement_frame_spin2_terms_kernel(directions_flat, basis_flat, float(eps), terms, status)
    if np.any(status):
        raise ValueError("Propagation direction must be non-zero")
    _stokes_to_s2l2_terms_flat_kernel(stokes_flat, terms, output_flat)
    return output


def stokes_to_s2l2(stk: np.ndarray, F: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Convert Stokes vectors to S2L2 coefficients with fused Numba kernels."""

    stokes = np.asarray(stk, dtype=np.float64)
    frames = np.asarray(F, dtype=np.float64)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError(f"Expected Stokes shape (..., 4), got {stokes.shape}")
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame shape (..., 3, 3), got {frames.shape}")

    shape_chan = np.broadcast_shapes(stokes.shape[:-1], frames.shape[:-2])
    expected_shape = (*shape_chan, 12)

    if not NUMBA_AVAILABLE:
        result = _ref_stokes_to_s2l2(stokes, frames)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    _dispatch_stokes_to_s2l2(stokes, frames, output, shape_chan)
    return output


def s2l2_to_stokes(s2l2: np.ndarray, F: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Convert S2L2 coefficients to Stokes vectors with fused Numba kernels."""

    coeffs = np.asarray(s2l2, dtype=np.float64)
    frames = np.asarray(F, dtype=np.float64)
    if coeffs.ndim < 1 or coeffs.shape[-1] != 12:
        raise ValueError(f"Expected S2L2 shape (..., 12), got {coeffs.shape}")
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame shape (..., 3, 3), got {frames.shape}")

    shape_chan = np.broadcast_shapes(coeffs.shape[:-1], frames.shape[:-2])
    expected_shape = (*shape_chan, 4)

    if not NUMBA_AVAILABLE:
        result = _ref_s2l2_to_stokes(coeffs, frames)
        if out is not None:
            _prepare_output(out, expected_shape, "out")
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output(out, expected_shape, "out") if out is not None else np.empty(expected_shape, dtype=np.float64)
    _dispatch_s2l2_to_stokes(coeffs, frames, output, shape_chan)
    return output


def s2l2_to_stokes_float32(s2l2: np.ndarray, F: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Convert S2L2 coefficients to float32 Stokes vectors."""

    coeffs = np.asarray(s2l2)
    frames = np.asarray(F, dtype=np.float64)
    if coeffs.ndim < 1 or coeffs.shape[-1] != 12:
        raise ValueError(f"Expected S2L2 shape (..., 12), got {coeffs.shape}")
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame shape (..., 3, 3), got {frames.shape}")

    shape_chan = np.broadcast_shapes(coeffs.shape[:-1], frames.shape[:-2])
    expected_shape = (*shape_chan, 4)

    if not NUMBA_AVAILABLE:
        result = _ref_s2l2_to_stokes(np.asarray(coeffs, dtype=np.float64), frames).astype(np.float32)
        if out is not None:
            _prepare_output_dtype(out, expected_shape, "out", np.float32)
            np.copyto(out, result)
            return out
        return result

    output = _prepare_output_dtype(out, expected_shape, "out", np.float32) if out is not None else np.empty(expected_shape, dtype=np.float32)
    _dispatch_s2l2_to_stokes(coeffs, frames, output, shape_chan)
    return output


def _prepare_output(out: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    return _prepare_output_dtype(out, expected_shape, name, np.float64)


def _prepare_output_dtype(out: np.ndarray, expected_shape: tuple[int, ...], name: str, dtype: np.dtype) -> np.ndarray:
    output = np.asarray(out)
    if output.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {output.shape}")
    if output.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}, got {output.dtype}")
    if not output.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return output


def _dispatch_stokes_to_s2l2(
    stokes: np.ndarray,
    frames: np.ndarray,
    output: np.ndarray,
    shape_chan: tuple[int, ...],
) -> None:
    stokes_prefix = stokes.shape[:-1]
    frame_prefix = frames.shape[:-2]
    output_flat = output.reshape(-1, 12)

    if frames.shape == (3, 3):
        stokes_flat = np.ascontiguousarray(np.broadcast_to(stokes, (*shape_chan, 4)).reshape(-1, 4))
        _stokes_to_s2l2_single_frame_kernel(stokes_flat, np.ascontiguousarray(frames), output_flat)
        return

    if stokes_prefix == frame_prefix:
        stokes_flat = np.ascontiguousarray(stokes.reshape(-1, 4))
        frames_flat = np.ascontiguousarray(frames.reshape(-1, 3, 3))
        _stokes_to_s2l2_flat_kernel(stokes_flat, frames_flat, output_flat)
        return

    if _is_channel_frame_broadcast(stokes_prefix, frame_prefix, shape_chan):
        n_chan = shape_chan[-1]
        n_frame = output_flat.shape[0] // n_chan
        stokes_flat = np.ascontiguousarray(stokes.reshape(n_frame * n_chan, 4))
        frames_flat = np.ascontiguousarray(frames.reshape(n_frame, 1, 3, 3)[:, 0])
        if n_chan == 3:
            _stokes_to_s2l2_channel3_sample_kernel(stokes_flat, frames_flat, output_flat)
        else:
            _stokes_to_s2l2_channel_kernel(stokes_flat, frames_flat, output_flat, n_chan)
        return

    stokes_flat = np.ascontiguousarray(np.broadcast_to(stokes, (*shape_chan, 4)).reshape(-1, 4))
    frames_flat = np.ascontiguousarray(np.broadcast_to(frames, (*shape_chan, 3, 3)).reshape(-1, 3, 3))
    _stokes_to_s2l2_flat_kernel(stokes_flat, frames_flat, output_flat)


def _dispatch_s2l2_to_stokes(
    coeffs: np.ndarray,
    frames: np.ndarray,
    output: np.ndarray,
    shape_chan: tuple[int, ...],
) -> None:
    coeff_prefix = coeffs.shape[:-1]
    frame_prefix = frames.shape[:-2]
    output_flat = output.reshape(-1, 4)

    if frames.shape == (3, 3):
        coeffs_flat = np.ascontiguousarray(np.broadcast_to(coeffs, (*shape_chan, 12)).reshape(-1, 12))
        _s2l2_to_stokes_single_frame_kernel(coeffs_flat, np.ascontiguousarray(frames), output_flat)
        return

    if coeff_prefix == frame_prefix:
        coeffs_flat = np.ascontiguousarray(coeffs.reshape(-1, 12))
        frames_flat = np.ascontiguousarray(frames.reshape(-1, 3, 3))
        _s2l2_to_stokes_flat_kernel(coeffs_flat, frames_flat, output_flat)
        return

    if _is_channel_frame_broadcast(coeff_prefix, frame_prefix, shape_chan):
        n_chan = shape_chan[-1]
        n_frame = output_flat.shape[0] // n_chan
        coeffs_flat = np.ascontiguousarray(coeffs.reshape(n_frame * n_chan, 12))
        frames_flat = np.ascontiguousarray(frames.reshape(n_frame, 1, 3, 3)[:, 0])
        if n_chan == 3:
            _s2l2_to_stokes_channel3_kernel(coeffs_flat, frames_flat, output_flat)
        else:
            _s2l2_to_stokes_channel_kernel(coeffs_flat, frames_flat, output_flat, n_chan)
        return

    coeffs_flat = np.ascontiguousarray(np.broadcast_to(coeffs, (*shape_chan, 12)).reshape(-1, 12))
    frames_flat = np.ascontiguousarray(np.broadcast_to(frames, (*shape_chan, 3, 3)).reshape(-1, 3, 3))
    _s2l2_to_stokes_flat_kernel(coeffs_flat, frames_flat, output_flat)


def _is_channel_frame_broadcast(
    value_prefix: tuple[int, ...],
    frame_prefix: tuple[int, ...],
    output_prefix: tuple[int, ...],
) -> bool:
    return len(value_prefix) == len(frame_prefix) and len(value_prefix) >= 1 and frame_prefix[-1] == 1 and value_prefix[:-1] == frame_prefix[:-1] and output_prefix == value_prefix


if NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _stokes_basis_values(wx: float, wy: float, wz: float) -> tuple[float, float, float]:
        norm = np.sqrt(wx * wx + wy * wy + wz * wz)
        nx = wx / norm
        ny = wy / norm
        nz = wz / norm

        sign = 1.0
        if nz < 0.0:
            sign = -1.0
        a = -1.0 / (sign + nz)

        bx = nx * nx * a * sign + 1.0
        by = nx * ny * a * sign
        bz = -nx * sign
        return bx, by, bz

    @njit(cache=True, inline="always")
    def _measurement_frame_values(
        wx: float,
        wy: float,
        wz: float,
        bx: float,
        by: float,
        bz: float,
        eps: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        w_norm = np.sqrt(wx * wx + wy * wy + wz * wz)
        nx = wx / w_norm
        ny = wy / w_norm
        nz = wz / w_norm

        fbx, fby, fbz = _stokes_basis_values(nx, ny, nz)

        dot_basis_w = bx * nx + by * ny + bz * nz
        p0 = bx - dot_basis_w * nx
        p1 = by - dot_basis_w * ny
        p2 = bz - dot_basis_w * nz
        p_norm = np.sqrt(p0 * p0 + p1 * p1 + p2 * p2)

        if p_norm < eps:
            e1x = fbx
            e1y = fby
            e1z = fbz
        else:
            inv_p_norm = 1.0 / p_norm
            e1x = p0 * inv_p_norm
            e1y = p1 * inv_p_norm
            e1z = p2 * inv_p_norm

        e2x_raw = ny * e1z - nz * e1y
        e2y_raw = nz * e1x - nx * e1z
        e2z_raw = nx * e1y - ny * e1x
        e2_norm = np.sqrt(e2x_raw * e2x_raw + e2y_raw * e2y_raw + e2z_raw * e2z_raw)

        if e2_norm < eps:
            e2x = ny * fbz - nz * fby
            e2y = nz * fbx - nx * fbz
            e2z = nx * fby - ny * fbx
        else:
            inv_e2_norm = 1.0 / e2_norm
            e2x = e2x_raw * inv_e2_norm
            e2y = e2y_raw * inv_e2_norm
            e2z = e2z_raw * inv_e2_norm

        e1x = e2y * nz - e2z * ny
        e1y = e2z * nx - e2x * nz
        e1z = e2x * ny - e2y * nx

        return e1x, e2x, nx, e1y, e2y, ny, e1z, e2z, nz

    @njit(cache=True, inline="always")
    def _write_measurement_frame_sample(
        wx: float,
        wy: float,
        wz: float,
        bx: float,
        by: float,
        bz: float,
        eps: float,
        output: np.ndarray,
        i: int,
    ) -> None:
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _measurement_frame_values(wx, wy, wz, bx, by, bz, eps)

        e1x = m00
        e2x = m01
        nx = m02
        e1y = m10
        e2y = m11
        ny = m12
        e1z = m20
        e2z = m21
        nz = m22

        output[i, 0, 0] = e1x
        output[i, 1, 0] = e1y
        output[i, 2, 0] = e1z
        output[i, 0, 1] = e2x
        output[i, 1, 1] = e2y
        output[i, 2, 1] = e2z
        output[i, 0, 2] = nx
        output[i, 1, 2] = ny
        output[i, 2, 2] = nz

    @njit(parallel=True, cache=True, fastmath=False)
    def _build_measurement_frame_kernel(directions: np.ndarray, basis: np.ndarray, eps: float, output: np.ndarray, status: np.ndarray) -> None:
        eps_sq = eps * eps
        for i in prange(directions.shape[0]):
            wx = directions[i, 0]
            wy = directions[i, 1]
            wz = directions[i, 2]
            if wx * wx + wy * wy + wz * wz < eps_sq:
                status[i] = 1
                continue
            _write_measurement_frame_sample(
                wx,
                wy,
                wz,
                basis[i, 0],
                basis[i, 1],
                basis[i, 2],
                eps,
                output,
                i,
            )

    @njit(parallel=True, cache=True, fastmath=False)
    def _duff_frame_kernel(directions: np.ndarray, eps: float, output: np.ndarray, status: np.ndarray) -> None:
        eps_sq = eps * eps
        for i in prange(directions.shape[0]):
            wx = directions[i, 0]
            wy = directions[i, 1]
            wz = directions[i, 2]
            if wx * wx + wy * wy + wz * wz < eps_sq:
                status[i] = 1
                continue
            bx, by, bz = _stokes_basis_values(wx, wy, wz)
            _write_measurement_frame_sample(wx, wy, wz, bx, by, bz, eps, output, i)

    @njit(parallel=True, cache=True, fastmath=False)
    def _thetaphi_frame_kernel(directions: np.ndarray, eps: float, output: np.ndarray) -> None:
        for i in prange(directions.shape[0]):
            wx = directions[i, 0]
            wy = directions[i, 1]
            wz = directions[i, 2]
            norm = np.sqrt(wx * wx + wy * wy + wz * wz)
            nx = wx / norm
            ny = wy / norm
            nz = wz / norm

            sin_theta = np.sqrt(nx * nx + nz * nz)
            if sin_theta > eps:
                nxz_hat_x = nx / sin_theta
                nxz_hat_z = nz / sin_theta
            else:
                fbx, _fby, fbz = _stokes_basis_values(nx, ny, nz)
                nxz_hat_x = fbx
                nxz_hat_z = fbz

            e_theta_x = ny * nxz_hat_x
            e_theta_y = -sin_theta
            e_theta_z = ny * nxz_hat_z
            _write_measurement_frame_sample(nx, ny, nz, e_theta_x, e_theta_y, e_theta_z, eps, output, i)

    @njit(cache=True, inline="always")
    def _mul(ar: float, ai: float, br: float, bi: float) -> tuple[float, float]:
        return ar * br - ai * bi, ar * bi + ai * br

    @njit(cache=True, inline="always")
    def _quat_from_matrix_values(
        m00: float,
        m01: float,
        m02: float,
        m10: float,
        m11: float,
        m12: float,
        m20: float,
        m21: float,
        m22: float,
    ) -> tuple[float, float, float, float]:
        matrix_trace = m00 + m11 + m22

        choice = 0
        best = m00
        if m11 > best:
            choice = 1
            best = m11
        if m22 > best:
            choice = 2
            best = m22
        if matrix_trace > best:
            choice = 3

        if choice == 0:
            qx = 1.0 - matrix_trace + 2.0 * m00
            qy = m10 + m01
            qz = m20 + m02
            qw = m21 - m12
        elif choice == 1:
            qx = m10 + m01
            qy = 1.0 - matrix_trace + 2.0 * m11
            qz = m21 + m12
            qw = m02 - m20
        elif choice == 2:
            qx = m20 + m02
            qy = m21 + m12
            qz = 1.0 - matrix_trace + 2.0 * m22
            qw = m10 - m01
        else:
            qx = m21 - m12
            qy = m02 - m20
            qz = m10 - m01
            qw = 1.0 + matrix_trace

        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        inv_norm = 1.0 / norm
        return qx * inv_norm, qy * inv_norm, qz * inv_norm, qw * inv_norm

    @njit(cache=True, inline="always")
    def _spin2_terms(
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        ar = qw
        ai = -qz
        br = qy
        bi = -qx

        a2r, a2i = _mul(ar, ai, ar, ai)
        a3r, a3i = _mul(a2r, a2i, ar, ai)
        a4r, a4i = _mul(a2r, a2i, a2r, a2i)

        b2r, b2i = _mul(br, bi, br, bi)
        b3r, b3i = _mul(b2r, b2i, br, bi)
        b4r, b4i = _mul(b2r, b2i, b2r, b2i)

        a3br, a3bi = _mul(a3r, a3i, br, bi)
        abr, abi = _mul(ar, ai, br, bi)
        ab2r, ab2i = _mul(abr, abi, abr, abi)
        ab3r, ab3i = _mul(ar, ai, b3r, b3i)

        return a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i

    @njit(parallel=True, cache=True, fastmath=False)
    def _s2l2_basis_at_frame_kernel(frames: np.ndarray, output: np.ndarray) -> None:
        scale = _SQRT_5_OVER_PI
        scale_half = 0.5 * scale
        scale_mid = _SQRT_1_5 * scale
        for i in prange(frames.shape[0]):
            qx, qy, qz, qw = _quat_from_matrix_values(
                frames[i, 0, 0],
                frames[i, 0, 1],
                frames[i, 0, 2],
                frames[i, 1, 0],
                frames[i, 1, 1],
                frames[i, 1, 2],
                frames[i, 2, 0],
                frames[i, 2, 1],
                frames[i, 2, 2],
            )
            a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
            output[i, 0] = scale_half * (a4r + 1j * a4i)
            output[i, 1] = -scale * (a3br + 1j * a3bi)
            output[i, 2] = scale_mid * (ab2r + 1j * ab2i)
            output[i, 3] = -scale * (ab3r + 1j * ab3i)
            output[i, 4] = scale_half * (b4r + 1j * b4i)

    @njit(parallel=True, cache=True, fastmath=False)
    def _measurement_frame_spin2_terms_kernel(
        directions: np.ndarray,
        basis: np.ndarray,
        eps: float,
        terms: np.ndarray,
        status: np.ndarray,
    ) -> None:
        eps_sq = eps * eps
        for i in prange(directions.shape[0]):
            wx = directions[i, 0]
            wy = directions[i, 1]
            wz = directions[i, 2]
            if wx * wx + wy * wy + wz * wz < eps_sq:
                status[i] = 1
                continue

            m00, m01, m02, m10, m11, m12, m20, m21, m22 = _measurement_frame_values(
                wx,
                wy,
                wz,
                basis[i, 0],
                basis[i, 1],
                basis[i, 2],
                eps,
            )
            qx, qy, qz, qw = _quat_from_matrix_values(m00, m01, m02, m10, m11, m12, m20, m21, m22)
            a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
            terms[i, 0] = a4r
            terms[i, 1] = a4i
            terms[i, 2] = a3br
            terms[i, 3] = a3bi
            terms[i, 4] = ab2r
            terms[i, 5] = ab2i
            terms[i, 6] = ab3r
            terms[i, 7] = ab3i
            terms[i, 8] = b4r
            terms[i, 9] = b4i

    @njit(cache=True, inline="always")
    def _write_stokes_to_s2l2_terms(
        stokes: np.ndarray,
        sample_index: int,
        a4r: float,
        a4i: float,
        a3br: float,
        a3bi: float,
        ab2r: float,
        ab2i: float,
        ab3r: float,
        ab3i: float,
        b4r: float,
        b4i: float,
        output: np.ndarray,
    ) -> None:
        s0 = stokes[sample_index, 0]
        s1 = stokes[sample_index, 1]
        s2 = stokes[sample_index, 2]
        s3 = stokes[sample_index, 3]

        output[sample_index, 0] = s0
        _write_conj_times_stokes(output, sample_index, 1, 1.0, a4r, a4i, s1, s2)
        _write_conj_times_stokes(output, sample_index, 3, -2.0, a3br, a3bi, s1, s2)
        _write_conj_times_stokes(output, sample_index, 5, _TWO_SQRT_1_5, ab2r, ab2i, s1, s2)
        _write_conj_times_stokes(output, sample_index, 7, -2.0, ab3r, ab3i, s1, s2)
        _write_conj_times_stokes(output, sample_index, 9, 1.0, b4r, b4i, s1, s2)
        output[sample_index, 11] = s3

    @njit(cache=True, inline="always")
    def _write_stokes_to_s2l2_sample(
        stokes: np.ndarray,
        sample_index: int,
        m00: float,
        m01: float,
        m02: float,
        m10: float,
        m11: float,
        m12: float,
        m20: float,
        m21: float,
        m22: float,
        output: np.ndarray,
    ) -> None:
        qx, qy, qz, qw = _quat_from_matrix_values(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
        _write_stokes_to_s2l2_terms(stokes, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_terms_flat_kernel(stokes: np.ndarray, terms: np.ndarray, output: np.ndarray) -> None:
        for i in prange(stokes.shape[0]):
            _write_stokes_to_s2l2_terms(
                stokes,
                i,
                terms[i, 0],
                terms[i, 1],
                terms[i, 2],
                terms[i, 3],
                terms[i, 4],
                terms[i, 5],
                terms[i, 6],
                terms[i, 7],
                terms[i, 8],
                terms[i, 9],
                output,
            )

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_channel_terms_kernel(stokes: np.ndarray, terms: np.ndarray, output: np.ndarray, n_chan: int) -> None:
        for frame_index in prange(terms.shape[0]):
            sample_offset = frame_index * n_chan
            for channel_index in range(n_chan):
                sample_index = sample_offset + channel_index
                _write_stokes_to_s2l2_terms(
                    stokes,
                    sample_index,
                    terms[frame_index, 0],
                    terms[frame_index, 1],
                    terms[frame_index, 2],
                    terms[frame_index, 3],
                    terms[frame_index, 4],
                    terms[frame_index, 5],
                    terms[frame_index, 6],
                    terms[frame_index, 7],
                    terms[frame_index, 8],
                    terms[frame_index, 9],
                    output,
                )

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_channel3_terms_kernel(stokes: np.ndarray, terms: np.ndarray, output: np.ndarray) -> None:
        for i in prange(stokes.shape[0]):
            frame_index = i // 3
            _write_stokes_to_s2l2_terms(
                stokes,
                i,
                terms[frame_index, 0],
                terms[frame_index, 1],
                terms[frame_index, 2],
                terms[frame_index, 3],
                terms[frame_index, 4],
                terms[frame_index, 5],
                terms[frame_index, 6],
                terms[frame_index, 7],
                terms[frame_index, 8],
                terms[frame_index, 9],
                output,
            )

    @njit(cache=True, inline="always")
    def _write_conj_times_stokes(
        output: np.ndarray,
        sample_index: int,
        column: int,
        multiplier: float,
        tr: float,
        ti: float,
        s1: float,
        s2: float,
    ) -> None:
        output[sample_index, column] = multiplier * (tr * s1 + ti * s2)
        output[sample_index, column + 1] = multiplier * (tr * s2 - ti * s1)

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_flat_kernel(stokes: np.ndarray, frames: np.ndarray, output: np.ndarray) -> None:
        for i in prange(stokes.shape[0]):
            _write_stokes_to_s2l2_sample(
                stokes,
                i,
                frames[i, 0, 0],
                frames[i, 0, 1],
                frames[i, 0, 2],
                frames[i, 1, 0],
                frames[i, 1, 1],
                frames[i, 1, 2],
                frames[i, 2, 0],
                frames[i, 2, 1],
                frames[i, 2, 2],
                output,
            )

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_single_frame_kernel(stokes: np.ndarray, frame: np.ndarray, output: np.ndarray) -> None:
        m00 = frame[0, 0]
        m01 = frame[0, 1]
        m02 = frame[0, 2]
        m10 = frame[1, 0]
        m11 = frame[1, 1]
        m12 = frame[1, 2]
        m20 = frame[2, 0]
        m21 = frame[2, 1]
        m22 = frame[2, 2]
        qx, qy, qz, qw = _quat_from_matrix_values(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
        for i in prange(stokes.shape[0]):
            _write_stokes_to_s2l2_terms(stokes, i, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_channel_kernel(stokes: np.ndarray, frames: np.ndarray, output: np.ndarray, n_chan: int) -> None:
        for frame_index in prange(frames.shape[0]):
            qx, qy, qz, qw = _quat_from_matrix_values(
                frames[frame_index, 0, 0],
                frames[frame_index, 0, 1],
                frames[frame_index, 0, 2],
                frames[frame_index, 1, 0],
                frames[frame_index, 1, 1],
                frames[frame_index, 1, 2],
                frames[frame_index, 2, 0],
                frames[frame_index, 2, 1],
                frames[frame_index, 2, 2],
            )
            a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
            sample_offset = frame_index * n_chan
            for channel_index in range(n_chan):
                sample_index = sample_offset + channel_index
                _write_stokes_to_s2l2_terms(stokes, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(parallel=True, cache=True, fastmath=False)
    def _stokes_to_s2l2_channel3_sample_kernel(stokes: np.ndarray, frames: np.ndarray, output: np.ndarray) -> None:
        for i in prange(stokes.shape[0]):
            frame_index = i // 3
            _write_stokes_to_s2l2_sample(
                stokes,
                i,
                frames[frame_index, 0, 0],
                frames[frame_index, 0, 1],
                frames[frame_index, 0, 2],
                frames[frame_index, 1, 0],
                frames[frame_index, 1, 1],
                frames[frame_index, 1, 2],
                frames[frame_index, 2, 0],
                frames[frame_index, 2, 1],
                frames[frame_index, 2, 2],
                output,
            )

    @njit(cache=True, inline="always")
    def _write_s2l2_to_stokes_terms(
        coeffs: np.ndarray,
        sample_index: int,
        a4r: float,
        a4i: float,
        a3br: float,
        a3bi: float,
        ab2r: float,
        ab2i: float,
        ab3r: float,
        ab3i: float,
        b4r: float,
        b4i: float,
        output: np.ndarray,
    ) -> None:
        sum_r = 0.0
        sum_i = 0.0
        term_r, term_i = _term_times_coeff(1.0, a4r, a4i, coeffs[sample_index, 1], coeffs[sample_index, 2])
        sum_r += term_r
        sum_i += term_i
        term_r, term_i = _term_times_coeff(-2.0, a3br, a3bi, coeffs[sample_index, 3], coeffs[sample_index, 4])
        sum_r += term_r
        sum_i += term_i
        term_r, term_i = _term_times_coeff(_TWO_SQRT_1_5, ab2r, ab2i, coeffs[sample_index, 5], coeffs[sample_index, 6])
        sum_r += term_r
        sum_i += term_i
        term_r, term_i = _term_times_coeff(-2.0, ab3r, ab3i, coeffs[sample_index, 7], coeffs[sample_index, 8])
        sum_r += term_r
        sum_i += term_i
        term_r, term_i = _term_times_coeff(1.0, b4r, b4i, coeffs[sample_index, 9], coeffs[sample_index, 10])
        sum_r += term_r
        sum_i += term_i

        output[sample_index, 0] = coeffs[sample_index, 0]
        output[sample_index, 1] = sum_r
        output[sample_index, 2] = sum_i
        output[sample_index, 3] = coeffs[sample_index, 11]

    @njit(cache=True, inline="always")
    def _write_s2l2_to_stokes_sample(
        coeffs: np.ndarray,
        sample_index: int,
        m00: float,
        m01: float,
        m02: float,
        m10: float,
        m11: float,
        m12: float,
        m20: float,
        m21: float,
        m22: float,
        output: np.ndarray,
    ) -> None:
        qx, qy, qz, qw = _quat_from_matrix_values(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
        _write_s2l2_to_stokes_terms(coeffs, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(cache=True, inline="always")
    def _term_times_coeff(
        multiplier: float,
        tr: float,
        ti: float,
        cr: float,
        ci: float,
    ) -> tuple[float, float]:
        return multiplier * (tr * cr - ti * ci), multiplier * (tr * ci + ti * cr)

    @njit(parallel=True, cache=True, fastmath=False)
    def _s2l2_to_stokes_flat_kernel(coeffs: np.ndarray, frames: np.ndarray, output: np.ndarray) -> None:
        for i in prange(coeffs.shape[0]):
            _write_s2l2_to_stokes_sample(
                coeffs,
                i,
                frames[i, 0, 0],
                frames[i, 0, 1],
                frames[i, 0, 2],
                frames[i, 1, 0],
                frames[i, 1, 1],
                frames[i, 1, 2],
                frames[i, 2, 0],
                frames[i, 2, 1],
                frames[i, 2, 2],
                output,
            )

    @njit(parallel=True, cache=True, fastmath=False)
    def _s2l2_to_stokes_single_frame_kernel(coeffs: np.ndarray, frame: np.ndarray, output: np.ndarray) -> None:
        m00 = frame[0, 0]
        m01 = frame[0, 1]
        m02 = frame[0, 2]
        m10 = frame[1, 0]
        m11 = frame[1, 1]
        m12 = frame[1, 2]
        m20 = frame[2, 0]
        m21 = frame[2, 1]
        m22 = frame[2, 2]
        qx, qy, qz, qw = _quat_from_matrix_values(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
        for i in prange(coeffs.shape[0]):
            _write_s2l2_to_stokes_terms(coeffs, i, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(parallel=True, cache=True, fastmath=False)
    def _s2l2_to_stokes_channel_kernel(coeffs: np.ndarray, frames: np.ndarray, output: np.ndarray, n_chan: int) -> None:
        for frame_index in prange(frames.shape[0]):
            qx, qy, qz, qw = _quat_from_matrix_values(
                frames[frame_index, 0, 0],
                frames[frame_index, 0, 1],
                frames[frame_index, 0, 2],
                frames[frame_index, 1, 0],
                frames[frame_index, 1, 1],
                frames[frame_index, 1, 2],
                frames[frame_index, 2, 0],
                frames[frame_index, 2, 1],
                frames[frame_index, 2, 2],
            )
            a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
            sample_offset = frame_index * n_chan
            for channel_index in range(n_chan):
                sample_index = sample_offset + channel_index
                _write_s2l2_to_stokes_terms(coeffs, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

    @njit(parallel=True, cache=True, fastmath=False)
    def _s2l2_to_stokes_channel3_kernel(coeffs: np.ndarray, frames: np.ndarray, output: np.ndarray) -> None:
        for frame_index in prange(frames.shape[0]):
            qx, qy, qz, qw = _quat_from_matrix_values(
                frames[frame_index, 0, 0],
                frames[frame_index, 0, 1],
                frames[frame_index, 0, 2],
                frames[frame_index, 1, 0],
                frames[frame_index, 1, 1],
                frames[frame_index, 1, 2],
                frames[frame_index, 2, 0],
                frames[frame_index, 2, 1],
                frames[frame_index, 2, 2],
            )
            a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i = _spin2_terms(qx, qy, qz, qw)
            sample_index = frame_index * 3
            _write_s2l2_to_stokes_terms(coeffs, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)
            sample_index += 1
            _write_s2l2_to_stokes_terms(coeffs, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)
            sample_index += 1
            _write_s2l2_to_stokes_terms(coeffs, sample_index, a4r, a4i, a3br, a3bi, ab2r, ab2i, ab3r, ab3i, b4r, b4i, output)

else:

    def _missing_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Numba is not available")

    _build_measurement_frame_kernel = _missing_numba
    _duff_frame_kernel = _missing_numba
    _thetaphi_frame_kernel = _missing_numba
    _s2l2_basis_at_frame_kernel = _missing_numba
    _measurement_frame_spin2_terms_kernel = _missing_numba
    _stokes_to_s2l2_terms_flat_kernel = _missing_numba
    _stokes_to_s2l2_channel_terms_kernel = _missing_numba
    _stokes_to_s2l2_channel3_terms_kernel = _missing_numba
    _stokes_to_s2l2_flat_kernel = _missing_numba
    _stokes_to_s2l2_single_frame_kernel = _missing_numba
    _stokes_to_s2l2_channel_kernel = _missing_numba
    _stokes_to_s2l2_channel3_sample_kernel = _missing_numba
    _s2l2_to_stokes_flat_kernel = _missing_numba
    _s2l2_to_stokes_single_frame_kernel = _missing_numba
    _s2l2_to_stokes_channel_kernel = _missing_numba
    _s2l2_to_stokes_channel3_kernel = _missing_numba
