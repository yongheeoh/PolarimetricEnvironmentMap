"""Choose a manual, calibrated or estimated validity mask for one view."""

import cv2
import numpy as np
from pemap.io.files import source
from pemap.io.images import png, binary
from pemap.io.display import display
from pemap.io.outputs import exposure
from pemap.masking.estimate import automatic_mask


def prepare_view_mask(scene, captures, calibration, group, sid, s0, output, *, warn=print):
    old = scene / ("envmap/mask" if group == "envmap" else "floor/analysis/mask") / f"{sid}.png"
    relative = old.relative_to(scene)
    selected = old if old.exists() else None
    if selected is None and group == "envmap":
        pitch = float(captures.metadata[group, sid].get("pitch_deg", 0))
        candidates = calibration.profiles["mask"].get("pitch_masks", {})
        matched = next(
            (value for angle, value in candidates.items() if abs(float(angle) - pitch) < 1e-3), None
        )
        if matched is not None:
            selected = calibration.paths["mask"].parent / matched
    if selected is not None:
        mask = cv2.imread(str(selected), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != captures.shape:
            raise ValueError(f"Wrong mask dimensions: {selected}")
        if mask.shape != calibration.shape:
            mask = cv2.resize(mask, calibration.shape[::-1], interpolation=cv2.INTER_AREA)
        valid = (mask > 127) & calibration.optical
        report = {
            "method": "Existing scene mask" if selected == old else "Calibrated pitch mask",
            "source": source(selected),
            "review_required": False,
        }
    else:
        m = captures.metadata[group, sid]
        valid, report = automatic_mask(
            s0,
            calibration.optical,
            float(m.get("pitch_deg", 0)),
            group,
            mount_roll_deg=float(m.get("mount_roll_deg", 90)),
        )
        if report["review_required"]:
            warn(f"{group}/{sid}: automatic equipment mask needs visual review; overlay saved")
    stored = binary(valid)
    if valid.shape != captures.shape:
        stored = cv2.resize(stored, captures.shape[::-1], interpolation=cv2.INTER_NEAREST)
    png(output / relative, stored)
    preview = display(s0, exposure(s0, calibration.optical))
    preview[~valid] = (preview[~valid] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    png(
        output / group / "analysis/mask_overlay" / f"{sid}.png",
        cv2.resize(preview, (612, 512), interpolation=cv2.INTER_AREA),
    )
    return valid, report
