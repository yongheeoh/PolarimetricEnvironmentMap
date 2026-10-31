"""Linear-polarization (S0, S1, S2) recovery for FLIR color polarization cameras (Sony IMX250MYR).

The raw mosaic is a 4x4 superpixel of 2x2 Bayer colors x 2x2 polarizers (0/45/90/135 deg).
Demosaicing uses polanalyser's ``COLOR_PolarRGB``; the polanalyser ``next`` branch handles float
input natively (no uint16 round trip), so the HDR-fused float mosaic keeps its full dynamic range.
S3 is not measured and is set to 0.

Polarization convention (verified 2026-10-08 against the Basler full-Stokes camera, see capture/memo.md):
polanalyser's analyzer angles are already in the sensor image frame, so no offset / S2 flip is needed —
the output matches the Basler pcalib Stokes (whose 90 deg offset is a calibration-stage reference, baked
into pcalib_tensor.npy). The 90 deg mount roll is handled in main.py via the camera rotation, as for Basler.
``angle_offset_deg`` shifts the analyzer angles (90 deg flips S1 and S2); ``flip_s2`` negates S2.
"""

import numpy as np
import polanalyser as pa

ANALYZER_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)  # order of pa.demosaicing outputs

# Shared defaults — verified with polarizer film against Basler captures.
FLIR_ANGLE_OFFSET_DEG = 0.0
FLIR_FLIP_S2 = False
# Output scale (2026-10-14): HDR radiance is (raw / full scale) / exposure_s; times 0.001344 = 2.688 ms / 2 the output reads,
# at 0 EV in an EXR viewer, like one analyser pixel of the bracket's middle exposure (2.688 ms = 21 us * 2^7, geometric
# centre of 21 us ... 344 ms). The / 2: S0 of an unpolarised pixel is twice one analyser pixel. One fixed constant for
# every FLIR scene (never per scene), so scenes stay comparable; outdoor daylight medians ~0.1-0.5, indoor ~8 EV darker.
FLIR_SCALE = 0.001344
# HDR fusion lower bound (fraction of full scale): 0.01 = 41 DN of 12 bit, SNR ~10 per raw pixel. The sensor is linear
# within 1 % down to 40 DN and the dark floor is 1 DN (lens-cap + bracket-pair analysis 2026-10-12, capture/memo.md,
# calibration/hdr/sensor_report.py). Basler full-Stokes keeps 0.05.
FLIR_Z_MIN = 0.01


def recover_bgr_stokes_flir(
    img_raw: np.ndarray,
    angle_offset_deg: float = FLIR_ANGLE_OFFSET_DEG,
    flip_s2: bool = FLIR_FLIP_S2,
) -> np.ndarray:
    """Raw color-polarization mosaic (H, W) → BGR Stokes (H, W, 3, 4) with S3 = 0.

    img_raw : float (HDR-fused, linear) or uint8/uint16 mosaic.
    """
    if not hasattr(pa, "imwrite_stokes"):
        raise ImportError("FLIR recovery needs polanalyser's `next` branch (float demosaicing)")
    img = np.asarray(img_raw)
    if np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)

    imgs_bgr = pa.demosaicing(img, pa.COLOR_PolarRGB)  # 4 x (H, W, 3) BGR for 0/45/90/135 deg
    angles = np.deg2rad(np.asarray(ANALYZER_ANGLES_DEG) + float(angle_offset_deg))
    stokes = np.asarray(pa.calcStokes(imgs_bgr, angles), dtype=np.float32)  # (H, W, 3, 3): S0, S1, S2
    if flip_s2:
        stokes[..., 2] *= -1.0

    s3 = np.zeros(stokes.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate([stokes, s3], axis=-1)
