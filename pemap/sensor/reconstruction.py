"""Sensor-specific RAW recovery for the shared scene pipeline."""

from pathlib import Path

import numpy as np
import polanalyser as pa

from pemap.io.files import source
from pemap.sensor.flir import recover_bgr_stokes_flir
from pemap.sensor.full_stokes_cam.imgproc import recover_bgr_stokes


class SceneSensor:
    def __init__(
        self,
        camera,
        profiles,
        paths,
        *,
        pcalib=None,
        scale=None,
        z_min=None,
        z_max=None,
        flir_angle_offset_deg=None,
        flir_flip_s2=None,
    ):
        self.camera = camera
        self.s3_measured = camera == "basler"
        self.full_scale = 255.0 if self.s3_measured else 65535.0
        hdr = profiles["hdr"]
        self.zmin = float(hdr["hdr_merge"]["z_min"] if z_min is None else z_min)
        self.zmax = float(hdr["hdr_merge"]["z_max"] if z_max is None else z_max)
        key = "SCALE" if self.s3_measured else "FLIR_SCALE"
        self.scale = float(hdr["output_scale"][key] if scale is None else scale)
        if not (
            np.isfinite([self.zmin, self.zmax, self.scale]).all()
            and 0 < self.zmin < self.zmax <= 1
            and self.scale > 0
        ):
            raise ValueError("Invalid HDR reliability window or output scale")
        self.tensor = None
        self.sources = {}
        parameters = profiles.get("polarimetric", {}).get("parameters", {})
        self.offset = float(parameters.get("FLIR_ANGLE_OFFSET_DEG", 0)) if camera == "flir" else 0.0
        self.flip_s2 = bool(parameters.get("FLIR_FLIP_S2", False)) if camera == "flir" else False
        if camera == "flir":
            if flir_angle_offset_deg is not None:
                self.offset = float(flir_angle_offset_deg)
            if flir_flip_s2 is not None:
                self.flip_s2 = bool(flir_flip_s2)
            if not np.isfinite(self.offset):
                raise ValueError("Invalid FLIR angle offset")
        if self.s3_measured:
            if pcalib is None:
                value = profiles["polarimetric"]["pcalib_tensor"]
                pcalib = paths["polarimetric"].parent / value
            pcalib = Path(pcalib).resolve()
            self.tensor = np.load(pcalib, allow_pickle=False)
            if self.tensor.shape != (4, 4, 4) or not np.isfinite(self.tensor).all():
                raise ValueError(f"Invalid Basler analyzer tensor: {pcalib}")
            # Each colour must support reconstruction of all four Stokes terms.
            from pemap.sensor.full_stokes_cam.imgproc import _split

            if any(np.linalg.matrix_rank(a) < 4 for a in _split(self.tensor)):
                raise ValueError(f"Rank-deficient Basler analyzer tensor: {pcalib}")
            self.sources["pcalib_tensor"] = source(pcalib)
        elif pcalib is not None:
            raise ValueError("--pcalib is only used by the Basler sensor")
        if not hasattr(pa, "imwrite_stokes"):
            raise RuntimeError("Install the polanalyser next version selected by uv.lock")
        if camera == "flir":
            probe = np.full((32, 32), 0.001234567, np.float32)
            probe[0, 0] = 1000
            sample = pa.demosaicing(probe, pa.COLOR_PolarRGB)[0][16, 16]
            if not np.allclose(sample, 0.001234567, rtol=1e-4, atol=1e-9):
                raise RuntimeError("FLIR HDR requires a float-capable polanalyser demosaicer")

    def recover(self, raw):
        return self.recover_unscaled(raw) * self.scale

    def recover_unscaled(self, raw):
        if self.s3_measured:
            return recover_bgr_stokes(raw, self.tensor)
        return recover_bgr_stokes_flir(raw, angle_offset_deg=self.offset, flip_s2=self.flip_s2)

    def metadata(self, *, rotation_model="min_rot"):
        result = {
            "camera": self.camera,
            "stokes_repr_mode": "camera_stokes",
            "rotation_model": rotation_model,
            "S3": "measured; calibrated Full-Stokes recovery" if self.s3_measured else "zero; unmeasured",
        }
        if self.s3_measured:
            result["pcalib_tensor"] = self.sources["pcalib_tensor"]
        else:
            result.update(angle_offset_deg=self.offset, flip_s2=self.flip_s2)
        return result
