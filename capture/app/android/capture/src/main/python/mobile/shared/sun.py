"""Calibrated HDR Sun insertion: peak alignment and a floating-point Poisson solve."""

from dataclasses import dataclass
from typing import Any, Callable
import cv2
import numpy as np
from .min_rot import align_vectors_batched
from .composition import AnalysisOutput
from .composition import CompositionResult, PanoramaState
from .scene_math import poisson_clone


def peak_center(green, valid, periodic=False):
    good = valid & np.isfinite(green) & (green > 0)
    if not good.any():
        raise ValueError("No positive valid Sun pixels")
    y, x = np.unravel_index(np.argmax(np.where(good, green, -np.inf)), green.shape)
    shift = green.shape[1] // 2 - x if periodic else 0
    value = np.roll(green, shift, axis=1)
    valid = np.roll(good, shift, axis=1)
    _, labels = cv2.connectedComponents(((value >= 0.9 * green[y, x]) & valid).astype(np.uint8))
    component = labels == labels[y, (x + shift) % green.shape[1]]
    yy, xx = np.where(component)
    weights = value[component].astype(float)
    return np.array([(np.average(xx, weights=weights) - shift) % green.shape[1], np.average(yy, weights=weights)]), float(green[y, x])


def direction_at(xy, width, height):
    phi = xy[0] * 2 * np.pi / width
    theta = (xy[1] + 1) * np.pi / (height + 1)
    return np.array([np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)])


def sun_insertion_mask(saturation, center, radius, *, adaptive=True):
    """Request the core and nearby clipping footprint, with a solver margin.

    Longitude is periodic. Nearby means within four core radii. The caller must
    independently validate reliable ND and panorama observations everywhere.
    """
    sat = np.asarray(saturation, bool)
    h, w = sat.shape
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Invalid Sun radius")
    y, x = np.indices((h, w))
    distance = np.hypot((x - center[0] + w / 2) % w - w / 2, y - center[1])
    core = distance < radius
    candidates = sat & (distance < 4 * radius) if adaptive else np.zeros_like(sat)
    seed = core | candidates
    if adaptive:
        padded = np.pad(seed.astype(np.uint8), ((0, 0), (1, 1)), mode="wrap")
        mask = (cv2.dilate(padded, np.ones((3, 3), np.uint8)) > 0)[:, 1:-1].copy()
    else:
        mask = core
    return mask, {
        "mask_mode": "adaptive" if adaptive else "fixed",
        "core_radius_px": float(radius),
        "saturation_search_radius_px": float(4 * radius) if adaptive else None,
        "margin_px": 1 if adaptive else 0,
        "mask_pixels": int(mask.sum()),
        "saturation_pixels_covered": int((mask & sat).sum()),
        "max_mask_distance_px": float(distance[mask].max()) if mask.any() else 0.0,
    }


def sun_mask_crop(mask, center):
    """Periodic crop with room for the Poisson boundary."""
    h, w = mask.shape
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Empty Sun insertion mask")
    cx = int(np.rint(center[0]))
    dx = (xx - cx + w // 2) % w - w // 2
    xs = np.arange(cx + dx.min() - 3, cx + dx.max() + 4) % w
    ys = np.arange(yy.min() - 3, yy.max() + 4)
    if ys.min() < 0 or ys.max() >= h:
        raise ValueError("Sun mask too close to ERP pole; rotate the panorama domain before insertion")
    return np.ix_(ys, xs), xs, ys


@dataclass(frozen=True)
class SunCapture:
    """ND-compensated camera Stokes before scene WB/CCM."""

    stokes: np.ndarray
    reliable_mask: np.ndarray
    prior_rotation: np.ndarray | None
    nd_od: float
    transmission_bgr: list[float]


@dataclass(frozen=True)
class SunInputs:
    geometry: Any
    optical_mask: np.ndarray
    color_matrix: np.ndarray
    envmap_priors: dict[str, np.ndarray]
    envmap_rotations: dict[str, np.ndarray]
    shot_ids: list[str]
    load_capture: Callable[[str], SunCapture]
    project: Callable[[np.ndarray, np.ndarray], np.ndarray]
    radius: float = 8.0
    mask_mode: str = "adaptive"
    disabled: bool = False


def composite_sun(inputs: SunInputs, panorama: PanoramaState, *, analysis: AnalysisOutput | None = None) -> CompositionResult:
    """Return a new panorama/report; no access to a scene runner or its arguments."""
    if not inputs.shot_ids or inputs.disabled:
        return CompositionResult(panorama, {"applied": False, "reason": "No captures" if not inputs.shot_ids else "Explicit --no-sun"})
    base, observed, sat = panorama.payload, panorama.coverage, panorama.saturation
    messages = []
    geo = inputs.geometry
    w, h = geo.width, geo.height
    directions = geo.directions
    inverse = np.linalg.inv(inputs.color_matrix)
    green = (base[..., 0] @ inverse)[..., 1]
    reports = {}
    choices = []
    radius = inputs.radius * w / 2048
    for sid in inputs.shot_ids:
        try:
            capture = inputs.load_capture(sid)
            prior_rv = capture.prior_rotation
            if prior_rv is None:
                raise ValueError("Missing Sun stage prior")
            prior = cv2.Rodrigues(prior_rv)[0]
            pp = {k: cv2.Rodrigues(v)[0] for k, v in inputs.envmap_priors.items()}
            nearest = min(pp, key=lambda k: np.linalg.norm(pp[k].T[:, 2] - prior.T[:, 2]))
            initial = cv2.Rodrigues(inputs.envmap_rotations[nearest])[0] @ pp[nearest].T @ prior
            # Only search near the direction actually captured as Sun.
            allowed = observed & ((directions @ initial.T[:, 2]) > np.cos(np.deg2rad(30)))
            target_xy, peak = peak_center(green, allowed, periodic=True)
            data = capture.stokes
            source_xy, source_peak = peak_center(data[..., 1, 0], inputs.optical_mask)
            uv = cv2.undistortPointsIter(source_xy.reshape(1, 1, 2), geo.K.astype(float), geo.D.astype(float), None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(2)
            ray = np.r_[uv, 1.0]
            ray /= np.linalg.norm(ray)
            target = direction_at(target_xy, w, h)
            previous = initial.T @ ray
            # align_vectors_batched(a,b) maps b onto a (the project's convention).
            align = align_vectors_batched(target, previous)
            rotation = initial @ align.T
            rv = cv2.Rodrigues(rotation)[0]
            projected = cv2.projectPoints(target.reshape(1, 3), rv, np.zeros(3), geo.K, geo.D)[0].reshape(2)
            np.testing.assert_allclose(projected, source_xy, rtol=0, atol=0.01)
            fullmask, mask_report = sun_insertion_mask(sat, target_xy, radius, adaptive=inputs.mask_mode == "adaptive")
            crop, xs, ys = sun_mask_crop(fullmask, target_xy)
            dirs = directions[crop]
            mask = fullmask[crop]
            needed = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            reliable = capture.reliable_mask & inputs.optical_mask
            payload = np.repeat(reliable[..., None, None].astype(np.float32), 3, 2)
            valid = geo.supported_remap(payload, dirs.reshape(-1, 3), rv, geo.K, geo.D, len(xs), len(ys))[..., 0, 0] > 0.9999
            if not valid[needed].all():
                raise ValueError("Sun or Poisson boundary lacks unsaturated reliable HDR exposure support")
            if not observed[crop][needed].all():
                raise ValueError("Poisson boundary has no measured panorama support")
            coefficients = inputs.project(data, rotation)
            measured = geo.supported_remap(coefficients, dirs.reshape(-1, 3), rv, geo.K, geo.D, len(xs), len(ys))
            del coefficients
            if not np.isfinite(measured[needed]).all():
                raise ValueError("Invalid Sun warp")
            # Unused crop corners can be outside calibrated support.
            measured[~needed] = 0
            camera = np.einsum("...cs,cd->...ds", base[crop], inverse)
            solved, solver = poisson_clone(camera, measured, mask)
            local = np.einsum("...cs,cd->...ds", solved.astype(np.float32), inputs.color_matrix)
            local[~mask] = base[crop][~mask]
            negative = int((solved[..., 0][mask] < 0).sum())
            report = {
                "accepted": True,
                "source_peak_camera_G": source_peak,
                "base_peak_camera_G": peak,
                "source_sensor_xy": source_xy.tolist(),
                "target_envmap_xy": target_xy.tolist(),
                "rvec_aligned": rv.reshape(-1).tolist(),
                "projection_error_px": float(np.linalg.norm(projected - source_xy)),
                "nearest_view": nearest,
                "ND_OD": capture.nd_od,
                "T_BGR": capture.transmission_bgr,
                "solver": solver,
                "negative_camera_S0_channels": negative,
                "insertion_mask": mask_report,
                "all_ND_and_boundary_pixels_reliable": True,
                "peak_change_vs_measured_percent": float(100 * (solved[..., 1, 0][mask].max() / measured[..., 1, 0][mask].max() - 1)),
            }
            reports[sid] = report
            # Prefer the latest valid exposure when several shots have equal quality.
            choices.append((negative, -len(choices), sid, crop, mask, local))
        except (ValueError, KeyError, AssertionError) as error:
            reports[sid] = {"accepted": False, "reason": str(error)}
            print("SUN REJECT", sid, str(error), flush=True)
    if not choices:
        raise ValueError(f"No reliable Sun insertion: {reports}. Existing results preserved; --no-sun explicitly requests omission.")
    _, _, sid, crop, mask, local = min(choices, key=lambda c: c[:2])
    result = base.copy()
    result[crop] = local
    fullmask = np.zeros((h, w), bool)
    fullmask[crop] = mask
    np.testing.assert_array_equal(result[~fullmask], base[~fullmask])
    residual = sat & ~fullmask
    if analysis is not None:
        analysis.image("sun_mask.png", fullmask.astype(np.uint8) * 255)
    report = {
        "applied": True,
        "method": "normal_poisson_float",
        "radius_px": radius,
        "radius_at_width_2048": inputs.radius,
        "shot_id": sid,
        "shots": reports,
        "outside_mask_unchanged": True,
        "insertion_mask": reports[sid]["insertion_mask"],
        "saturation_pixels_before": int(sat.sum()),
        "saturation_pixels_after": int(residual.sum()),
        "photometric_note": "ND transmission restores camera radiance scale; Poisson adds a harmonic boundary offset, so peak/integrated energy need not equal the ND source.",
    }
    if reports[sid]["negative_camera_S0_channels"]:
        messages.append("Sun Poisson produced negative camera S0 channels; retained and reported without clipping")
    return CompositionResult(PanoramaState(result, observed, residual), report, tuple(messages))
