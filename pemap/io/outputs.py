"""Write float EXRs, previews and the result's basic numeric checks.

These writers receive finished images and never discover or modify captures.
"""

from pathlib import Path
import numpy as np
import polanalyser as pa
from pemap.polarization.optimized.s2l2_kernels import duff_frame, s2l2_to_stokes, thetaphi_frame
from pemap.io.s2l2_exr import imwrite_s2l2
from pemap.io.images import png
from pemap.io.display import display
from pemap.io.files import write
from pemap.io.images import binary


def exposure(s0, valid=None):
    g = s0[..., 1]
    good = (g > 1e-8) & (True if valid is None else valid)
    return float(0.18 / np.median(g[good])) if good.any() else 1.0


def save_stokes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite Stokes: {path}")
    pa.imwrite_stokes(str(path), data)


def export_result(
    folder,
    coefficients,
    coverage,
    metadata,
    multiplier,
    *,
    geometry,
    mode,
    s3_measured=False,
    representation="s2l2",
):
    folder.mkdir(parents=True, exist_ok=True)
    directions = geometry.directions
    if mode == "s0":
        import OpenImageIO as oiio

        s0 = np.asarray(coefficients[..., 0], np.float32)
        if not np.isfinite(s0).all() or np.any(s0[~coverage]):
            raise ValueError("Final S0 invariant check failed")
        path = str(folder / "s0.exr")
        spec = oiio.ImageSpec(s0.shape[1], s0.shape[0], 3, oiio.FLOAT)
        spec.channelnames = ["R", "G", "B"]
        spec.attribute("compression", "zip")
        writer = oiio.ImageOutput.create(path)
        if writer is None or not writer.open(path, spec):
            raise IOError("S0 EXR open failed")
        try:
            if not writer.write_image(np.ascontiguousarray(s0[..., ::-1])):
                raise IOError("S0 EXR write failed")
        finally:
            writer.close()
        png(folder / "coverage_mask.png", binary(coverage))
        png(folder / "s0.png", display(s0, multiplier))
        metadata["verification"] = {
            "EXR_dtype": "float32",
            "uncovered_values_zero": True,
            "negative_S0_channels": int(((s0 < 0) & coverage[..., None]).sum()),
        }
        metadata["preview"] = {"exposure_multiplier": multiplier, "display_only": True}
        write(folder / "run.json", metadata)
        return coefficients
    if representation == "none":
        if not np.isfinite(coefficients).all() or np.any(coefficients[~coverage]):
            raise ValueError("Invalid unrotated Stokes output")
        save_stokes(folder / "unrotated.exr", coefficients)
        png(folder / "coverage_mask.png", binary(coverage))
        png(folder / "s0.png", display(coefficients[..., 0], multiplier))
        metadata["coordinates"] = "Unrotated camera Stokes; not a common world reference frame"
        metadata["preview"] = {"exposure_multiplier": multiplier, "display_only": True}
        write(folder / "run.json", metadata)
        return coefficients
    theta = s2l2_to_stokes(coefficients, thetaphi_frame(-directions)[..., None, :, :]).astype(np.float32)
    duff = s2l2_to_stokes(coefficients, duff_frame(-directions)[..., None, :, :]).astype(np.float32)
    if (
        not np.isfinite(theta).all()
        or (not s3_measured and np.any(theta[..., 3]))
        or np.any(theta[~coverage])
    ):
        raise ValueError("Final Stokes invariant check failed")
    save_stokes(folder / "thetaphi.exr", theta)
    save_stokes(folder / "duff.exr", duff)
    imwrite_s2l2(folder / "s2l2.exr", coefficients)
    png(folder / "coverage_mask.png", binary(coverage))
    png(folder / "s0.png", display(theta[..., 0], multiplier))
    bright = (theta[..., 0] > 1e-5) & coverage[..., None]
    ratio = np.linalg.norm(theta[..., 1:3], axis=-1) / np.maximum(theta[..., 0], 1e-8)
    metadata["verification"] = {
        "EXR_dtype": "float32",
        "S3_zero_unmeasured": not s3_measured,
        "negative_S0_channels": int(((theta[..., 0] < 0) & coverage[..., None]).sum()),
        "DoLP_over_1_bright_channels": int(((ratio > 1) & bright).sum()),
        "uncovered_values_zero": True,
    }
    metadata["preview"] = {"exposure_multiplier": multiplier, "display_only": True}
    write(folder / "run.json", metadata)
    return theta
