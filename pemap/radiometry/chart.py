"""calibration / radiometric / chart utilities."""

import cv2
import numpy as np
from pemap.radiometry import chart_fit
from pemap.io.display import display


def chart_comparison(s0, profile, polygons=None, exposure=None, baseline_profile=None, upright=True, bounds=None, panel_width=650, title="ColorChecker comparison"):
    """Display-only captured / corrected / reference chart in matching geometry.

    GT is the stored Lab D50 reference adapted to linear sRGB D65, then encoded
    for display. It receives neither the fitted CCM nor chart exposure gain.
    Legacy global-vs-local reports explicitly pass a baseline profile.
    """
    image = np.asarray(s0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected BGR S0 image")
    polys = np.asarray(profile["roi_polygons_raw"] if polygons is None else polygons, np.float64).copy()
    if polys.shape != (24, 4, 2) or not np.isfinite(polys).all():
        raise ValueError("Expected 24 finite patch ROI polygons")
    scale = float(profile["fit"]["chart_exposure_scale"] if exposure is None else exposure)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid chart display exposure")
    if upright:
        raw_width = image.shape[1]
        image = np.rot90(image)
        polys = np.stack([polys[..., 1], raw_width - 1 - polys[..., 0]], -1)
    centres = polys.mean(axis=1)
    grid = np.array([[i % 6, i // 6] for i in range(24)], np.float64)
    homography, _ = cv2.findHomography(grid, centres, method=0)
    if homography is None:
        raise ValueError("Degenerate chart geometry")
    cells = np.array([[[x - 0.43, y - 0.43], [x + 0.43, y - 0.43], [x + 0.43, y + 0.43], [x - 0.43, y + 0.43]] for x, y in grid], np.float32)
    cells = cv2.perspectiveTransform(cells.reshape(1, -1, 2), homography).reshape(24, 4, 2)
    if bounds is None:
        extent = cells.reshape(-1, 2)
        pad = max(float(np.ptp(extent, axis=0).min()) * 0.07, 2.0)
        lo = np.floor(extent.min(0) - pad).astype(int)
        hi = np.ceil(extent.max(0) + pad).astype(int) + 1
        x0, y0 = lo
        x1, y1 = hi
    else:
        x0, y0, x1, y1 = map(int, bounds)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image.shape[1])
    y1 = min(y1, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Chart crop is outside the image")
    crop = np.asarray(image[y0:y1, x0:x1], np.float32)

    def matrix(p):
        return np.diag([p["wb_gain"][c] for c in "BGR"]) @ np.asarray(p["ccm_3x3"])[::-1, ::-1]

    baseline = crop if baseline_profile is None else crop @ matrix(baseline_profile)
    stages = [display(baseline, scale), display(crop @ matrix(profile), scale)]
    w = int(panel_width)
    h = max(1, round(w * crop.shape[0] / crop.shape[1]))
    factor = np.array([w / crop.shape[1], h / crop.shape[0]])
    stages = [cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST) for im in stages]
    gt = np.full((h, w, 3), 30, np.uint8)
    target = chart_fit.reference_rgb(np.asarray(profile["reference"]["lab_d50_row_major"]))
    target_bgr = np.rint(chart_fit.display_srgb(target[:, ::-1]) * 255).astype(np.uint8)
    for cell, color in zip(cells, target_bgr):
        q = np.rint((cell - [x0, y0] + 0.5) * factor - 0.5).astype(np.int32)
        cv2.fillConvexPoly(gt, q, tuple(map(int, color)))
    stages.append(gt)
    for stage in stages:
        for i, poly in enumerate(polys):
            p = np.rint((poly - [x0, y0] + 0.5) * factor - 0.5).astype(np.int32)
            cv2.polylines(stage, [p], True, (245, 245, 245), 1, cv2.LINE_AA)
            point = np.rint(p.mean(0)).astype(int)
            label = str(i + 1)
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            at = (int(point[0] - size[0] / 2), int(point[1] + size[1] / 2))
            cv2.putText(stage, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 3, cv2.LINE_AA)
            cv2.putText(stage, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    gap = 18
    top = 96
    bottom = 44
    canvas = np.full((top + h + bottom, 3 * w + 4 * gap, 3), 255, np.uint8)
    labels = ["Captured / exposure only" if baseline_profile is None else "Global WB / CCM", "Corrected / WB + CCM", "GT / ColorChecker reference"]
    cv2.putText(canvas, title, (gap, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    for i, (stage, label) in enumerate(zip(stages, labels)):
        x = gap + i * (w + gap)
        canvas[top : top + h, x : x + w] = stage
        cv2.putText(canvas, label, (x, top - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (20, 20, 20), 1, cv2.LINE_AA)
    note = "Display only: chart exposure + sRGB encoding. GT: stored Lab D50 -> sRGB D65; no fitted transform."
    cv2.putText(canvas, note, (gap, top + h + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


def patch_polygons(corners, half=0.22):
    grid = np.float32([[0, 0], [3, 0], [3, 5], [0, 5]])
    H = cv2.getPerspectiveTransform(grid, np.float32(corners))
    polys = []
    for i in range(24):
        row, col = i % 6, 3 - i // 6
        q = np.float32([[col - half, row - half], [col + half, row - half], [col + half, row + half], [col - half, row + half]])
        polys.append(cv2.perspectiveTransform(q[None], H)[0])
    return np.array(polys)


def sample_chart(image, corners, half=0.22):
    polygons = patch_polygons(corners, half)
    values = []
    noise = []
    for poly in polygons:
        x, y = np.floor(poly.min(0)).astype(int)
        xe, ye = np.ceil(poly.max(0)).astype(int) + 1
        if x < 0 or y < 0 or xe > image.shape[1] or ye > image.shape[0]:
            raise ValueError("Chart ROI outside image")
        mask = np.zeros((ye - y, xe - x), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(poly - [x, y]).astype(np.int32), 1)
        samples = image[y:ye, x:xe][mask > 0][:, ::-1]
        value = np.median(samples, axis=0)
        values.append(value)
        noise.append(1.4826 * np.median(abs(samples - value), axis=0) / np.maximum(value, 1e-9))
    return np.array(values), polygons, float(np.median(noise))


def detect_chart(s0, target, global_wb):
    """Find a 6x4 chart quad, verify patch order, neutral ramp and withheld colours.

    Returns RAW pixel-centre coordinates (white, dark skin, bluish green, black).
    Detection failure is explicit and the caller may use the global profile.
    """
    scale = min(1.0, 1400 / max(s0.shape[:2]))
    small = cv2.resize(s0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    balanced = small * np.asarray(global_wb)
    positive = balanced[..., 1][balanced[..., 1] > 0]
    quads = {}
    for q in (50, 80, 95):
        gray = cv2.cvtColor(display(balanced, 0.25 / max(float(np.percentile(positive, q)), 1e-8)), cv2.COLOR_BGR2GRAY)
        for threshold in (15, 30, 50, 75, 105, 140, 185, 220):
            binary = (gray < threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 350 or area > 0.6 * gray.size:
                    continue
                poly = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
                if len(poly) != 4 or not cv2.isContourConvex(poly):
                    continue
                p = poly[:, 0].astype(np.float32)
                lengths = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)
                if lengths.min() < 12 or lengths.max() / lengths.min() > 4:
                    continue
                quads[tuple(np.rint(p.reshape(-1) / 3).astype(int))] = p
    candidates = []
    for p in quads.values():
        # Try rotations and handedness; geometry is determined by the ordered chart colours.
        for reflect in (False, True):
            z = p[::-1] if reflect else p
            for roll in range(4):
                quad = np.roll(z, roll, axis=0)
                H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)
                for mx, my in ((0.10, 0.13), (0.12, 0.18), (0.08, 0.10)):
                    corners = cv2.perspectiveTransform(np.float32([[[mx, 0.88], [mx, my], [1 - mx, my], [1 - mx, 0.88]]]), H)[0]
                    try:
                        values, _, noise = sample_chart(small, corners)
                        neutral = values[18:24, 1]
                        if np.any(values <= 0) or np.count_nonzero(np.diff(neutral) < 0) < 5 or neutral[0] / max(neutral[-1], 1e-9) < 4 or noise > 0.15:
                            continue
                        fit = chart_fit.fit_profile(values, target)
                        error = chart_fit.color_errors(values @ np.array(fit["chart_matrix_rgb"]).T, target)
                        score = float(np.mean(error)) + noise * 12
                        if score < 8:
                            candidates.append((score, corners, noise))
                    except (ValueError, np.linalg.LinAlgError):
                        continue
    if not candidates:
        raise ValueError("No confidently validated ColorChecker detected")
    score, corners, noise = min(candidates, key=lambda c: c[0])
    return (corners + 0.5) / scale - 0.5, {"method": "Automatic quad + 24-patch colour/neutral validation", "score": score, "relative_patch_noise": noise}


def fit_chart(s0, corners, reference, reliable=None):
    target = chart_fit.reference_rgb(np.array(reference["lab_d50_row_major"]))
    values, polys, noise = sample_chart(s0, corners)
    usable = None
    if reliable is not None:
        if reliable.shape != s0.shape[:2]:
            raise ValueError("Chart reliability dimensions disagree")
        usable = []
        for polygon in polys:
            region = np.zeros(s0.shape[:2], np.uint8)
            cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 1)
            usable.append(bool(np.asarray(reliable, dtype=bool)[region > 0].all()))
    fit = chart_fit.fit_profile(values, target, usable_patches=usable)
    metrics = chart_fit.evaluate_profile(values, target, fit)
    if metrics["colored_patch_leave_one_out"]["mean_delta_e00"] > 10 or noise > 0.15:
        raise ValueError("ColorChecker fit failed quality limits")
    gains = np.array(fit["wb_gains_rgb"])
    matrix = np.array(fit["ccm_rgb"])
    return {
        "wb_gain": dict(zip(["R", "G", "B"], gains.tolist())),
        "ccm_3x3": matrix.T.tolist(),
        "wb_gains_rgb": gains.tolist(),
        "ccm_rgb": matrix.tolist(),
        "reference": reference,
        "fit": fit,
        "metrics": metrics,
        "roi_polygons_raw": polys.tolist(),
        "patch_centres_raw": np.asarray(corners).tolist(),
        "measured_linear_rgb": values.tolist(),
        "relative_patch_noise": noise,
        "apply": "Linear WB+CCM to every Stokes/S2L2 component, no chart exposure scale, gamma or clipping",
    }
