"""MAGSAC++-style calibrated pure-rotation estimation.

This module estimates a 3D camera rotation from 2D point correspondences and a
known intrinsic matrix. It adapts the MAGSAC++ scoring/refinement structure from
homography estimation while replacing the homography solver with a Wahba/Kabsch
fit on calibrated unit bearing vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import special

USAC_MAGSAC = 38
_EPS = np.finfo(float).eps
_MIN_SAMPLE_SIN = 1e-6


@dataclass(frozen=True)
class _Score:
    inliers: int
    loss: float


@dataclass(frozen=True)
class _MagsacConfig:
    threshold: float
    confidence: float
    max_iters: int
    random_state: int
    sample_size: int = 4
    dof: int = 2
    sigma_quantile: float = 3.04
    upper_incomplete_quantile: float = 0.00419
    c: float = 0.5
    maximum_thr: float = 7.5
    lo_sample_size: int = 75
    lo_iterations: int = 15
    final_iterations: int = 7

    @property
    def threshold_sqr(self) -> float:
        return self.threshold * self.threshold

    @property
    def maximum_thr_sqr(self) -> float:
        return self.maximum_thr * self.maximum_thr


def _as_points(points: np.ndarray | list, name: str) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        raise cv2.error(f"{name} must have shape Nx2, Nx1x2, or 1xNx2")

    if arr.shape[0] == 0:
        return arr.astype(np.float64, copy=False).reshape(0, 2)
    if not np.all(np.isfinite(arr)):
        raise cv2.error(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _zero_mask(n: int) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.uint8)


def _gamma_upper(a: float, x: np.ndarray | float) -> np.ndarray | float:
    return special.gammaincc(a, x) * special.gamma(a)


def _gamma_lower(a: float, x: np.ndarray | float) -> np.ndarray | float:
    return special.gammainc(a, x) * special.gamma(a)


def _magsac_norm_loss(cfg: _MagsacConfig) -> tuple[float, float, float]:
    sigma = cfg.maximum_thr / cfg.sigma_quantile
    sigma2 = sigma * sigma
    multiplier = math.pow(2.0, (cfg.dof + 1.0) * 0.5) / sigma
    a_lower = (cfg.dof + 1.0) * 0.5
    a_upper = (cfg.dof - 1.0) * 0.5
    max_loss = 1e-10
    step = cfg.maximum_thr_sqr / 30.0
    sqr_res = 0.0
    while sqr_res < cfg.maximum_thr_sqr:
        x = sqr_res / (2.0 * sigma2)
        loss = multiplier * (0.5 * sigma2 * _gamma_lower(a_lower, x) + 0.25 * sqr_res * (_gamma_upper(a_upper, x) - cfg.upper_incomplete_quantile))
        max_loss = max(max_loss, float(loss))
        sqr_res += step
    return multiplier / max_loss, sigma2, sigma


def _magsac_score(errors: np.ndarray, cfg: _MagsacConfig) -> _Score:
    finite = np.isfinite(errors)
    tentative = int(np.count_nonzero(finite & (errors < cfg.threshold_sqr)))
    active = finite & (errors < cfg.maximum_thr_sqr)
    if not np.any(active):
        return _Score(tentative, 0.0)

    norm_loss, sigma2, _ = _magsac_norm_loss(cfg)
    r = errors[active]
    x = r / (2.0 * sigma2)
    lower = _gamma_lower((cfg.dof + 1.0) * 0.5, x)
    upper = _gamma_upper((cfg.dof - 1.0) * 0.5, x)
    point_quality = 1.0 - (0.5 * sigma2 * lower + 0.25 * r * (upper - cfg.upper_incomplete_quantile)) * norm_loss
    return _Score(tentative, -float(np.sum(point_quality)))


def _magsac_weights(errors: np.ndarray, cfg: _MagsacConfig, sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if sigma is None:
        sigma = cfg.maximum_thr
    sigma2 = sigma * sigma
    active = np.isfinite(errors) & (errors < sigma2)
    inliers = np.flatnonzero(active)
    if len(inliers) == 0:
        return inliers, np.empty(0, dtype=np.float64)

    x = errors[inliers] * cfg.sigma_quantile / (2.0 * sigma2)
    upper = _gamma_upper((cfg.dof - 1.0) * 0.5, x)
    weights = cfg.c * math.pow(2.0, (cfg.dof - 1.0) * 0.5) / sigma
    weights = weights * np.maximum(upper - cfg.upper_incomplete_quantile, 0.0)
    return inliers, np.asarray(weights, dtype=np.float64)


def _is_better(score: _Score, best: _Score | None) -> bool:
    return best is None or score.loss < best.loss


def _iteration_bound(inliers: int, total: int, cfg: _MagsacConfig) -> int:
    if inliers <= 0 or total <= 0 or cfg.threshold <= 0:
        return cfg.max_iters

    inlier_ratio = min(max(inliers / total, 0.0), 1.0)
    good_sample_prob = inlier_ratio**cfg.sample_size
    if good_sample_prob >= 1.0 - _EPS:
        return 1
    if good_sample_prob <= _EPS:
        return cfg.max_iters

    conf = min(max(cfg.confidence, _EPS), 1.0 - _EPS)
    denom = math.log1p(-good_sample_prob)
    if denom >= 0.0:
        return cfg.max_iters
    return max(1, min(cfg.max_iters, int(math.ceil(math.log1p(-conf) / denom))))


def _as_camera_matrix(camera_matrix) -> tuple[np.ndarray, np.ndarray]:
    if camera_matrix is None:
        raise cv2.error("cameraMatrix is required for calibrated rotation estimation")

    K = camera_matrix
    if K.shape != (3, 3):
        raise cv2.error("cameraMatrix must have shape 3x3")
    if not np.all(np.isfinite(K)):
        raise cv2.error("cameraMatrix contains NaN or infinite values")

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError as exc:
        raise cv2.error("cameraMatrix must be non-singular") from exc

    if not np.all(np.isfinite(K_inv)):
        raise cv2.error("cameraMatrix must be non-singular")
    return K, K_inv


def _unit_bearings(points: np.ndarray, K_inv: np.ndarray) -> np.ndarray | None:
    hom = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    rays = hom @ K_inv.T
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= _EPS) or not np.all(np.isfinite(norms)):
        return None
    return np.ascontiguousarray(rays / norms[:, None], dtype=np.float64)


def _fit_rotation(
    src_bearings: np.ndarray,
    dst_bearings: np.ndarray,
    sample: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    original_len = len(src_bearings)
    if sample is not None:
        src_bearings = src_bearings[sample]
        dst_bearings = dst_bearings[sample]
        if weights is not None:
            if len(weights) == original_len:
                weights = weights[sample]
            elif len(weights) != len(sample):
                return None

    if len(src_bearings) < 2:
        return None

    if weights is None:
        weights = np.ones(len(src_bearings), dtype=np.float64)
    else:
        weights = weights.reshape(-1)
        if len(weights) != len(src_bearings):
            return None
        keep = np.isfinite(weights) & (weights > np.finfo(np.float64).eps)
        src_bearings = src_bearings[keep]
        dst_bearings = dst_bearings[keep]
        weights = weights[keep]
        if len(src_bearings) < 2:
            return None

    M = (dst_bearings * weights[:, None]).T @ src_bearings
    if not np.all(np.isfinite(M)):
        return None

    try:
        U, singular_values, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return None

    if singular_values[1] <= 1e-12 * max(singular_values[0], 1.0):
        return None

    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return None
    return np.ascontiguousarray(R, dtype=np.float64)


def _sample_is_good(src_bearings: np.ndarray, dst_bearings: np.ndarray, sample: np.ndarray) -> bool:
    a0, a1 = src_bearings[sample]
    b0, b1 = dst_bearings[sample]
    return bool(np.linalg.norm(np.cross(a0, a1)) > _MIN_SAMPLE_SIN and np.linalg.norm(np.cross(b0, b1)) > _MIN_SAMPLE_SIN)


def _forward_errors(src_bearings: np.ndarray, dst: np.ndarray, R: np.ndarray, K: np.ndarray) -> np.ndarray:
    rotated = src_bearings @ R.T
    projected = rotated @ K.T
    z = projected[:, 2]
    errors = np.full(len(src_bearings), np.inf, dtype=np.float64)
    valid = z > _EPS
    xy = projected[valid, :2] / z[valid, None]
    diff = dst[valid] - xy
    errors[valid] = np.einsum("ij,ij->i", diff, diff)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def _weighted_refine(
    src_bearings: np.ndarray,
    dst_bearings: np.ndarray,
    dst: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    score: _Score,
    cfg: _MagsacConfig,
    iterations: int,
    rng: np.random.Generator | None = None,
    sample_limit: int | None = None,
) -> tuple[np.ndarray, _Score]:
    best_R = R
    best_score = score

    for _ in range(iterations):
        errors = _forward_errors(src_bearings, dst, best_R, K)
        inliers, weights = _magsac_weights(errors, cfg)
        if len(inliers) < 2:
            break

        if sample_limit is not None and len(inliers) > sample_limit and rng is not None:
            local_pos = rng.choice(len(inliers), size=sample_limit, replace=False)
            fit_idx = inliers[local_pos]
            fit_weights = weights[local_pos]
        else:
            fit_idx = inliers
            fit_weights = weights

        candidate = _fit_rotation(src_bearings, dst_bearings, sample=fit_idx, weights=fit_weights)
        if candidate is None:
            break

        candidate_score = _magsac_score(_forward_errors(src_bearings, dst, candidate, K), cfg)
        if _is_better(candidate_score, best_score):
            best_R = candidate
            best_score = candidate_score
        else:
            break

    return best_R, best_score


def _run_magsac(src: np.ndarray, dst: np.ndarray, K: np.ndarray, K_inv: np.ndarray, cfg: _MagsacConfig) -> tuple[np.ndarray | None, np.ndarray]:
    n = len(src)
    if n < 2:
        return None, _zero_mask(n)

    src_bearings = _unit_bearings(src, K_inv)
    dst_bearings = _unit_bearings(dst, K_inv)
    if src_bearings is None or dst_bearings is None:
        return None, _zero_mask(n)

    rng = np.random.default_rng(cfg.random_state)
    best_R: np.ndarray | None = None
    best_score: _Score | None = None
    max_iterations = cfg.max_iters
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        sample = rng.choice(n, size=2, replace=False)
        if not _sample_is_good(src_bearings, dst_bearings, sample):
            continue

        R = _fit_rotation(src_bearings, dst_bearings, sample=sample)
        if R is None:
            continue

        errors = _forward_errors(src_bearings, dst, R, K)
        score = _magsac_score(errors, cfg)
        if _is_better(score, best_score):
            R, score = _weighted_refine(
                src_bearings,
                dst_bearings,
                dst,
                K,
                R,
                score,
                cfg,
                iterations=cfg.lo_iterations,
                rng=rng,
                sample_limit=cfg.lo_sample_size,
            )
            if _is_better(score, best_score):
                best_R = R
                best_score = score
                max_iterations = min(max_iterations, _iteration_bound(score.inliers, n, cfg))

    if best_R is None or best_score is None:
        return None, _zero_mask(n)

    best_R, best_score = _weighted_refine(
        src_bearings,
        dst_bearings,
        dst,
        K,
        best_R,
        best_score,
        cfg,
        iterations=cfg.final_iterations,
    )

    errors = _forward_errors(src_bearings, dst, best_R, K)
    mask = (errors < cfg.threshold_sqr).astype(np.uint8).reshape(-1, 1)
    if int(mask.sum()) >= 2:
        final = _fit_rotation(src_bearings, dst_bearings, sample=np.flatnonzero(mask.ravel()))
        if final is not None:
            final_errors = _forward_errors(src_bearings, dst, final, K)
            final_mask = (final_errors < cfg.threshold_sqr).astype(np.uint8).reshape(-1, 1)
            if int(final_mask.sum()) >= int(mask.sum()):
                best_R, mask = final, final_mask
    else:
        return None, _zero_mask(n)

    errors = _forward_errors(src_bearings, dst, best_R, K)
    proposal_threshold = (cfg.threshold * 1.02) ** 2
    proposal = errors < proposal_threshold
    if int(np.count_nonzero(proposal)) > int(mask.sum()):
        proposal_R = _fit_rotation(src_bearings, dst_bearings, sample=np.flatnonzero(proposal))
        if proposal_R is not None:
            proposal_errors = _forward_errors(src_bearings, dst, proposal_R, K)
            proposal_mask = (proposal_errors < cfg.threshold_sqr).astype(np.uint8).reshape(-1, 1)
            if int(proposal_mask.sum()) > int(mask.sum()):
                best_R, mask = proposal_R, proposal_mask

    return best_R, mask


def findRotation(
    srcPoints,
    dstPoints,
    cameraMatrix,
    method: int = cv2.USAC_MAGSAC,
    ransacReprojThreshold: float = 3.0,
    mask=None,
    maxIters: int = 2000,
    confidence: float = 0.995,
    random_state: int = 0,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Estimate pure camera rotation with a MAGSAC++-style USAC pipeline.

    ``cameraMatrix`` is required and is assumed to be shared by both views.
    Input points should be undistorted or have negligible distortion. The input
    ``mask`` is accepted for API compatibility and ignored.
    """

    del mask
    if int(method) != int(getattr(cv2, "USAC_MAGSAC", USAC_MAGSAC)) and int(method) != USAC_MAGSAC:
        raise NotImplementedError("Only cv2.USAC_MAGSAC rotation estimation is implemented")

    src = _as_points(srcPoints, "srcPoints")
    dst = _as_points(dstPoints, "dstPoints")
    if len(src) != len(dst):
        raise cv2.error("srcPoints and dstPoints must contain the same number of points")

    K, K_inv = _as_camera_matrix(cameraMatrix)

    if len(src) < 2:
        return None, _zero_mask(len(src))

    threshold = float(ransacReprojThreshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise cv2.error("ransacReprojThreshold must be a positive finite value")

    confidence = float(confidence)
    if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
        raise cv2.error("confidence must be a finite value in the open interval (0, 1)")

    cfg = _MagsacConfig(
        threshold=threshold,
        confidence=confidence,
        max_iters=max(1, int(maxIters)),
        random_state=int(random_state),
        sample_size=2,
        dof=2,
        maximum_thr=max(threshold, 7.5),
    )
    return _run_magsac(src, dst, K, K_inv, cfg)


find_rotation_magsac = findRotation
