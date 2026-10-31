"""Floor composition with explicit prepared captures, settings and result state."""

from dataclasses import dataclass
from typing import Any, Callable
import gc
import cv2
import numpy as np
from pemap.geometry.projection import remap
from pemap.io.analysis import AnalysisOutput
from pemap.composition import CompositionResult, PanoramaState
from pemap.blending.transition import guarded_multiband
from pemap.io.images import binary
from pemap.registration.floor_registration import register_floor
from pemap.radiometry.overlap import overlap_gain
from pemap.polarization.transport import floor_coefficients


@dataclass(frozen=True)
class FloorCapture:
    """Native Stokes after scene WB/CCM, with sensor-space masks."""

    stokes: np.ndarray
    valid_mask: np.ndarray
    saturation: np.ndarray


@dataclass(frozen=True)
class FloorInputs:
    geometry: Any
    optical_mask: np.ndarray
    color_matrix: np.ndarray
    reference: np.ndarray | None
    reference_known: np.ndarray | None
    shot_ids: list[str]
    load_capture: Callable[[str, bool], FloorCapture]
    level: int = 5
    transition: float = 35.0
    photometry: str = "smooth"
    disabled: bool = False
    rotation_model: str = "min_rot"


def composite_floor(inputs: FloorInputs, panorama: PanoramaState, *, analysis: AnalysisOutput | None = None) -> CompositionResult:
    """Return a new panorama/report; capture loading and publication are external."""
    base, observed, sat = panorama.payload, panorama.coverage, panorama.saturation
    messages = []
    if not inputs.shot_ids or inputs.disabled:
        return CompositionResult(panorama, {"applied": False, "reason": "No captures" if not inputs.shot_ids else "Explicit --no-floor"})
    if inputs.reference is None or inputs.reference_known is None or not inputs.reference_known.any():
        raise ValueError("No downward reference for Floor registration")
    n = inputs.geometry.nadir_size
    maps, optical = inputs.geometry.source_maps(inputs.optical_mask)
    plane, known = inputs.geometry.nadir(base, observed)
    known &= inputs.reference_known
    plane[~known] = 0
    distance = cv2.distanceTransform(known.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    transition = inputs.transition
    photometry_mode = inputs.photometry
    band = distance < transition
    if not band.any():
        return CompositionResult(panorama, {"applied": False, "reason": "No missing nadir coverage to fill"})
    registered = {}
    reports = {}
    for sid in inputs.shot_ids:
        capture = inputs.load_capture(sid, True)
        s0 = capture.stokes[..., 0]
        image = remap(s0, maps)
        valid = remap(capture.valid_mask.astype(np.float32), maps) > 0.9999
        valid &= optical
        report = register_floor(inputs.reference, inputs.reference_known, image, valid)
        reports[sid] = report
        if report["accepted"]:
            H = np.array(report["homography_source_to_nadir"])
            v = inputs.geometry.warp_plane(valid.astype(np.float32), H) > 0.9999
            image = inputs.geometry.warp_plane(image, H)
            ring = (cv2.erode(binary(v & known), np.ones((15, 15), np.uint8)) > 0) & (distance >= 10) & (distance <= 100)
            a = cv2.GaussianBlur(plane[..., 1, 0], (0, 0), 3)
            b = cv2.GaussianBlur(image[..., 1], (0, 0), 3)
            ring &= (a > 1e-5) & (b > 1e-5)
            if ring.sum() < 500:
                report.update(accepted=False, reason="Insufficient photometric overlap")
            else:
                gain = float(np.median(a[ring] / b[ring]))
                report.update(gain_all_components=gain, overlap_pixels=int(ring.sum()))
                margin = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 5)
                registered[sid] = (H, v, gain, margin)
                report.update(missing_hole_and_transition_pixels=int((band & ~v).sum()), minimum_margin_px=float(margin[band].min()) if band.any() else 0.0)
        print("FLOOR REGISTER", sid, {k: v for k, v in report.items() if k != "homography_source_to_nadir"}, flush=True)
    if analysis is not None:
        analysis.record("registration.json", reports)
    if not registered:
        raise ValueError("No reliable Floor registration; existing published result remains unchanged. Use --no-floor only to request a base-only result.")
    singles = [s for s, (_, v, _, _) in registered.items() if v[band].all()]
    if singles:
        selected = [max(singles, key=lambda s: float(registered[s][3][band].min()))]
    else:
        # Greedy coverage; combine measured views with the same weights for all coefficients.
        selected = []
        remaining = band.copy()
        while remaining.any():
            options = [s for s in registered if s not in selected]
            if not options:
                break
            best = max(options, key=lambda s: int((remaining & registered[s][1]).sum()))
            if not (remaining & registered[best][1]).any():
                break
            selected.append(best)
            remaining &= ~registered[best][1]
    if not selected:
        return CompositionResult(panorama, {"applied": False, "reason": "No Floor support at missing pixels"})
    floor = np.zeros((n, n, 3, base.shape[-1]), np.float32)
    weight_sum = np.zeros((n, n), np.float32)
    floor_sat = np.zeros((n, n), bool)
    poses = {}
    photometry = {}
    for sid in selected:
        H, valid, gain, margin = registered[sid]
        capture = inputs.load_capture(sid, False)
        data = capture.stokes
        rect = remap(data, maps)
        rect[~optical] = 0
        warped = inputs.geometry.warp_plane(rect, H)
        if photometry_mode == "smooth":
            field, photometry[sid] = overlap_gain(plane[..., 1, 0], warped[..., 1, 0], known, valid, distance, gain)
            if analysis is not None:
                photometry[sid]["gain_map"] = {**analysis.array(f"photometry/{sid}_gain.npy", field), "coordinates": "Rectified nadir plane"}
            warped *= field[..., None, None]
        else:
            warped *= gain
            photometry[sid] = {"method": "Scalar", "gain": gain}
        coeff, pose = floor_coefficients(inputs.geometry, warped, H, rotation_model=inputs.rotation_model)
        poses[sid] = pose
        weight = np.clip(margin / transition, 0, 1) * valid
        floor += coeff * weight[..., None, None]
        weight_sum += weight
        floor_sat |= (inputs.geometry.warp_plane(remap(capture.saturation.astype(np.float32), maps), H) > 0) & (weight > 0)
        del data, rect, warped, coeff, capture
        gc.collect()
    floor /= np.maximum(weight_sum, 1e-8)[..., None, None]
    valid = weight_sum > 0
    alpha = np.clip(1 - distance / transition, 0, 1).astype(np.float32)
    alpha[~valid] = 0
    if not valid[band].all():
        messages.append("Floor captures do not cover the entire hole/transition; unobserved pixels remain explicitly uncovered.")
    mixed = guarded_multiband(plane, floor, alpha, inputs.color_matrix, inputs.level)
    linear = plane * (1 - alpha[..., None, None]) + floor * alpha[..., None, None]
    delta = inputs.geometry.to_erp(mixed - linear)
    mapped_alpha = inputs.geometry.to_erp(alpha)
    actual = inputs.geometry.to_erp(valid.astype(np.float32), validity=True)
    mapped_alpha[~actual] = 0
    active = mapped_alpha > 0
    floor_erp = inputs.geometry.to_erp(floor)
    result = base.copy()
    result[active] = base[active] * (1 - mapped_alpha[active, None, None]) + floor_erp[active] * mapped_alpha[active, None, None] + delta[active]
    filled = ~observed & actual
    result[filled] = floor_erp[filled]
    mapped_alpha[filled] = 1
    active |= filled
    covered = observed | filled
    result[~covered] = 0
    np.testing.assert_array_equal(result[~active], base[~active])
    saturation = ((sat & (mapped_alpha < 1)) | ((inputs.geometry.to_erp(floor_sat.astype(np.float32)) > 0) & active)) & covered
    if analysis is not None:
        analysis.image("floor_mask.png", binary(active))
        analysis.image("floor_alpha.png", np.rint(mapped_alpha * 65535).astype(np.uint16))
    return CompositionResult(
        PanoramaState(result, covered, saturation),
        {
            "applied": True,
            "shot_ids": selected,
            "shots": {s: reports[s] for s in selected},
            "approximate_poses": poses,
            "method": "Guarded multiband",
            "level": inputs.level,
            "transition_nadir_px": transition,
            "nadir_size": n,
            "photometric_matching": photometry,
            "between_floor_shots": "Valid-source margin weighted world S2L2" if len(selected) > 1 else "Single measured shot",
            "basis_transport": "Per-ray min_rot -> image-derived planar pose including roll -> shortest-ray transport -> world S2L2",
            "filled_pixels": int(filled.sum()),
            "active_pixels": int(active.sum()),
        },
        tuple(messages),
    )
