"""Select or fit a scene's WB/CCM using its captured ColorChecker."""

import copy
import numpy as np
from pemap.io.files import read, write, source
from pemap.io.images import png
from pemap.io.color_cache import (
    color_signature,
    matching_color_capture,
    reusable_color_fit,
    legacy_color_inputs_match,
)
from pemap.radiometry.chart import chart_comparison, detect_chart, fit_chart
from pemap.radiometry.chart_fit import reference_rgb
from pemap.radiometry.color_profile import color_matrix


def calibrate_color(
    scene, captures, images, calibration, output, *, refit=False, warn=print, color_source="auto"
):
    if color_source == "profile":
        profile = copy.deepcopy(calibration.profiles["wb_cc"])
        scope = profile.get("source_scope") or (
            "scene" if calibration.paths["wb_cc"].is_relative_to(scene) else "global"
        )
        return profile, {
            "scope": scope,
            "method": "Explicit WB/CCM profile",
            "profile": source(calibration.paths["wb_cc"]),
        }
    color_inputs = {sid: captures.records[f"color/{sid}"] for sid in captures.ids["color"]}
    signature = color_signature(
        calibration.paths,
        calibration.sensor.sources,
        captures.camera,
        calibration.settings,
        calibration.vignette_source,
    )
    previous = scene / "color/result/wb_ccm.json"
    legacy = (
        captures.camera == "flir"
        and calibration.vignette is None
        and "sensor_overrides" not in calibration.settings
    )
    if legacy and previous.is_file() and any(key in calibration.overrides for key in ("hdr", "polarimetric")):
        legacy = legacy_color_inputs_match(scene, calibration.paths, calibration.root)
    cached = (
        None if refit else reusable_color_fit(previous, color_inputs, signature, legacy_compatible=legacy)
    )
    candidates = []
    if cached is not None:
        current = matching_color_capture(cached, color_inputs)
        profile = cached
        choice = {"scope": "scene", "method": "Existing scene ColorChecker fit", "profile": source(previous)}
        cached["processing_signature"] = signature
        cached["source_metadata"] = color_inputs[current]["metadata"]
        candidates.append(cached)
    if not candidates:
        wb, _, _ = color_matrix(calibration.profiles["wb_cc"])
        ref = calibration.profiles["wb_cc"].get("reference")
        target = reference_rgb(np.asarray(ref["lab_d50_row_major"])) if ref else None
        if ref is None and refit:
            raise ValueError(
                "Refitting ColorChecker requires a measured reference in the selected WB/CCM profile"
            )
        for sid in captures.ids["color"] if ref else []:
            try:
                s0 = images.recover("color", sid)[..., 0]
                roi = scene / "color/analysis" / f"{sid}_corners.json"
                if roi.exists():
                    corners = np.asarray(read(roi)["corners_raw"])
                    detection = {"method": "Saved raw chart corner centres", "file": source(roi)}
                else:
                    corners, detection = detect_chart(s0, target, wb)
                fit = fit_chart(s0, corners, ref, reliable=images.reliable["color", sid])
                fit.update(
                    source_scope="scene",
                    source=captures.records[f"color/{sid}"]["raw"],
                    source_metadata=captures.records[f"color/{sid}"]["metadata"],
                    shot_id=sid,
                    processing_signature=signature,
                    detection=detection,
                )
                candidates.append(fit)
                write(output / "color/analysis" / f"{sid}_fit.json", fit)
            except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                warn(f"Color {sid}: {error}")
        if candidates:
            profile = min(
                candidates, key=lambda p: p["metrics"]["colored_patch_leave_one_out"]["mean_delta_e00"]
            )
            choice = {
                "scope": "scene",
                "method": "Best validated scene ColorChecker fit",
                "shot_id": profile["shot_id"],
            }
        else:
            profile = copy.deepcopy(calibration.profiles["wb_cc"])
            scope = calibration.profiles["wb_cc"].get("source_scope") or (
                "scene" if calibration.paths["wb_cc"].is_relative_to(scene) else "global"
            )
            profile.update(source_scope=scope, processing_signature=signature)
            reason = (
                "Existing imported WB/CCM; no reference supplied for refitting"
                if ref is None
                else "No valid scene ColorChecker fit"
            )
            choice = {"scope": scope, "profile": source(calibration.paths["wb_cc"]), "reason": reason}
            if captures.ids["color"] and ref is not None:
                warn(
                    "Scene ColorChecker fitting failed; using the selected WB/CCM profile. Inspect color/analysis before accepting local colour calibration."
                )
    for candidate in candidates:
        sid = candidate.get("shot_id")
        if sid is None:
            sid = matching_color_capture(candidate, color_inputs)
        polygons = candidate.get("roi_polygons_raw")
        if polygons is not None:
            polygons = (np.asarray(polygons) + 0.5) * candidate.get("mobile_sampling_factor", 1) - 0.5
        if polygons is None and (scene / "color/analysis/measurements.json").exists():
            polygons = read(scene / "color/analysis/measurements.json").get("roi_polygons_raw")
        if sid is not None and polygons is not None:
            png(
                output / "color/analysis" / f"{sid}_patches.png",
                chart_comparison(
                    images.recover("color", sid)[..., 0],
                    candidate,
                    polygons,
                    title=f"Color {sid}: captured / corrected / GT",
                ),
            )
    return profile, choice
