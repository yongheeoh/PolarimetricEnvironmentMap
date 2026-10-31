"""Prepare scalar feature images and their capture validity masks."""

import numpy as np
from pemap.masking.view_masks import prepare_view_mask
from pemap.radiometry.color import apply_wb_ccm_stokes


def prepare_matching_views(scene, captures, images, calibration, wb, ccm, output, *, warn=print):
    masks, reports = {}, {}
    payloads = {}
    for sid in captures.ids["envmap"]:
        data = images.recover("envmap", sid)
        s0 = (
            apply_wb_ccm_stokes(data[..., 0:1], wb, ccm)
            if calibration.settings["feature_color"] == "corrected"
            else data[..., 0:1].copy()
        )
        payloads[sid] = s0
        masks[sid], reports[sid] = prepare_view_mask(
            scene, captures, calibration, "envmap", sid, s0[..., 0], output, warn=warn
        )
        if calibration.settings["feature_color"] == "native" and calibration.vignette_supported is not None:
            # Matching undistorts the complete image before applying its mask.
            # Preserve the original flat-field's missing samples at this step;
            # neutralized values outside the mask still affect CNN descriptors.
            s0[~calibration.vignette_supported] = np.nan
    return payloads, masks, reports
