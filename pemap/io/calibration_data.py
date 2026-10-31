"""Load the selected calibration inputs; this module performs no image processing."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from pemap.io.camera_calibration import load_intrinsics
from pemap.io.files import source
from pemap.io.images import load_image
from pemap.io.profiles import resolve_profiles
from pemap.io.settings import processing_settings
from pemap.masking.estimate import lens_mask
from pemap.geometry.projection import Geometry
from pemap.geometry.rays import precise_rays
from pemap.sensor.reconstruction import SceneSensor
from pemap.geometry.resize import make_resize_info, scale_intrinsics_for_resize


@dataclass
class CalibrationData:
    paths: dict
    profiles: dict
    settings: dict
    overrides: dict
    sensor: SceneSensor
    intrinsic: dict
    optical: np.ndarray
    geometry: Geometry
    vignette: np.ndarray | None = None
    vignette_supported: np.ndarray | None = None
    vignette_source: dict | None = None
    native_optical: np.ndarray | None = None
    resize_info: object = None
    root: object = None

    @property
    def shape(self):
        return tuple(self.intrinsic["image_size_wh"][::-1])


def load_calibration(scene, captures, options):
    camera_root = options.camera_roots.get(captures.camera)
    root = (
        Path(options.calibration_root).resolve()
        if options.calibration_root
        else (
            (scene / camera_root).resolve()
            if camera_root is not None
            else next((p / "calibration" for p in scene.parents if (p / "calibration").is_dir()), None)
        )
    )

    def camera_paths(mapping):
        return {key: str((root / value).resolve()) for key, value in mapping.items()}

    settings, _ = processing_settings(scene, options.processing_profile, root, overrides=options.numerics)
    overrides = {
        **options.profile_defaults,
        **camera_paths(options.camera_profiles.get(captures.camera, {})),
        **camera_paths(options.scene_profiles.get(captures.camera, {}).get(scene.name, {})),
        **options.profiles,
    }
    overrides = {key: value.replace("{scene_id}", scene.name) for key, value in overrides.items()}
    camera_pcalib = options.camera_pcalib.get(captures.camera)
    pcalib = options.pcalib or ((root / camera_pcalib).resolve() if camera_pcalib is not None else None)
    scale = options.scale if options.scale is not None else options.camera_scales.get(captures.camera)
    if options.vignetting is None:
        options.vignetting = settings["vignetting"]
    required = {"intrinsic", "hdr", "wb_cc", "mask"}
    if captures.camera == "flir" or pcalib is None:
        required.add("polarimetric")
    if captures.ids["sun"] and not options.no_sun:
        required.add("nd")
    if options.vignetting:
        required.add("vignetting")
    paths, profiles = resolve_profiles(scene, root, overrides, camera=captures.camera, required=required)
    sensor_overrides = dict(
        scale=scale,
        z_min=options.z_min,
        z_max=options.z_max,
        flir_angle_offset_deg=options.flir_angle_offset_deg,
        flir_flip_s2=options.flir_flip_s2,
    )
    sensor = SceneSensor(captures.camera, profiles, paths, pcalib=pcalib, **sensor_overrides)
    if any(value is not None for value in sensor_overrides.values()):
        settings["sensor_overrides"] = sensor_overrides
    if options.vignetting_alpha != 1:
        settings["vignetting_alpha"] = options.vignetting_alpha
    intrinsic = load_intrinsics(paths["intrinsic"])
    intrinsic["K"] = intrinsic["K"].astype(settings["intrinsic_dtype"])
    if tuple(intrinsic["image_size_wh"][::-1]) != captures.shape:
        raise ValueError("Intrinsic calibration and RAW dimensions disagree")
    optical = lens_mask(profiles["mask"], captures.shape, base_path=paths["mask"].parent)

    limit = profiles["intrinsic"].get("mapping_validation", {}).get("max_observed_normalized_radius")
    if limit is None:
        # Older calibration files bound the inverse projection using their lens mask.
        rays = precise_rays(*captures.shape, intrinsic["K"], intrinsic["dist_coeffs"])
        radii = np.linalg.norm(rays[..., :2], axis=-1) / rays[..., 2]
        limit = float(np.max(radii[optical]))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid intrinsic ray support")
    geometry = Geometry(intrinsic["K"], intrinsic["dist_coeffs"], limit, options.width, options.height)
    calibration = CalibrationData(paths, profiles, settings, overrides, sensor, intrinsic, optical, geometry)
    calibration.root = root
    if options.vignetting:
        profile = profiles["vignetting"]
        relative = profile.get("correction_map") or profile["map"]["path"]
        path = (paths["vignetting"].parent / relative).resolve()
        vignette = load_image(path)
        if vignette.shape != (*captures.shape, 3):
            raise ValueError("Invalid vignetting map")
        supported = np.isfinite(vignette).all(axis=-1) & (vignette > 0).all(axis=-1)
        if not np.any(supported & optical):
            raise ValueError("Vignetting map has no supported lens pixels")
        optical &= supported
        vignette[~supported] = 1
        if settings["radiometric_order"] == "vignetting_then_scale" or options.vignetting_alpha != 1:
            vignette = options.vignetting_alpha * (vignette - 1.0) + 1.0
        calibration.vignette = vignette
        calibration.vignette_supported = supported
        calibration.vignette_source = source(path)
    calibration.native_optical = optical
    calibration.resize_info = make_resize_info(*captures.shape, options.resize_scale)
    if options.resize_scale != 1:
        calibration.intrinsic = scale_intrinsics_for_resize(intrinsic, calibration.resize_info)
        calibration.geometry = Geometry(
            calibration.intrinsic["K"], intrinsic["dist_coeffs"], limit, options.width, options.height
        )
        size = calibration.resize_info.resized_hw[::-1]
        calibration.optical = (
            cv2.resize(optical.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.9999
        )
        if calibration.vignette_supported is not None:
            calibration.vignette_supported = (
                cv2.resize(
                    calibration.vignette_supported.astype(np.float32), size, interpolation=cv2.INTER_AREA
                )
                > 0.9999
            )
    return calibration
