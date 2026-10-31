"""Public numerical API; internal modules use explicit one-way imports."""

from importlib import import_module

_EXPORTS = {
    "CameraRotationEstimationResult": ["pemap.registration.camera_rotation", "CameraRotationEstimationResult"],
    "estimate_camera_rotation_matrices": ["pemap.registration.camera_rotation", "estimate_camera_rotation_matrices"],
    "WarpContext": ["pemap.geometry.warp", "WarpContext"],
    "WarpedViews": ["pemap.geometry.warp", "WarpedViews"],
    "accumulate_weighted_payload": ["pemap.blending.weights", "accumulate_weighted_payload"],
    "normalize_weighted_payload": ["pemap.blending.weights", "normalize_weighted_payload"],
    "blend_views": ["pemap.blending.multiband", "blend_views"],
    "build_envmap": ["pemap.composition.panorama", "build_envmap"],
    "build_warp_context": ["pemap.geometry.warp", "build_warp_context"],
    "warp_views": ["pemap.geometry.warp", "warp_views"],
    "equirectangular_directions": ["pemap.geometry.spherical_coordinates", "equirectangular_directions"],
    "inverse_rotate_equirectangular": ["pemap.geometry.spherical_coordinates", "inverse_rotate_equirectangular"],
    "inverse_rotate_equirectangular_scalar": ["pemap.geometry.spherical_coordinates", "inverse_rotate_equirectangular_scalar"],
    "octahedral_domain_rotations": ["pemap.geometry.spherical_coordinates", "octahedral_domain_rotations"],
    "harmonic_center_field": ["pemap.blending.weights", "harmonic_center_field"],
    "fuse_exposure_bracket": ["pemap.sensor.hdr", "fuse_exposure_bracket"],
    "inpaint_nans": ["pemap.blending.inpaint", "inpaint_nans"],
    "load_intrinsics": ["pemap.io.camera_calibration", "load_intrinsics"],
    "load_rvecs": ["pemap.io.camera_calibration", "load_rvecs"],
    "align_vectors_batched": ["pemap.polarization.min_rot", "align_vectors_batched"],
    "rotate_stokes_basis": ["pemap.polarization.optimized.mueller_kernels", "rotate_stokes_basis"],
    "stokes_basis": ["pemap.polarization.optimized.mueller_kernels", "stokes_basis"],
    "build_measurement_frame": ["pemap.polarization.optimized.s2l2_kernels", "build_measurement_frame"],
    "duff_frame": ["pemap.polarization.optimized.s2l2_kernels", "duff_frame"],
    "s2l2_to_stokes": ["pemap.polarization.optimized.s2l2_kernels", "s2l2_to_stokes"],
    "stokes_to_s2l2": ["pemap.polarization.optimized.s2l2_kernels", "stokes_to_s2l2"],
    "thetaphi_frame": ["pemap.polarization.optimized.s2l2_kernels", "thetaphi_frame"],
    "default_z_min": ["pemap.sensor.preprocessing", "default_z_min"],
    "fuse_bgr_stokes_from_path": ["pemap.sensor.preprocessing", "fuse_bgr_stokes_from_path"],
    "fuse_exposure_brackets_from_path": ["pemap.sensor.preprocessing", "fuse_exposure_brackets_from_path"],
    "ray_image": ["pemap.geometry.rays", "ray_image"],
    "blend_domain_rotations": ["pemap.blending.domain_blending", "blend_domain_rotations"],
    "domain_rotations_from_preset": ["pemap.blending.domain_blending", "domain_rotations_from_preset"],
    "imread_s2l2": ["pemap.io.s2l2_exr", "imread_s2l2"],
    "imwrite_s2l2": ["pemap.io.s2l2_exr", "imwrite_s2l2"],
    "detect_camera": ["pemap.io.metadata", "detect_camera"],
    "flir": ["pemap.sensor.flir", None],
    "ResizeInfo": ["pemap.geometry.resize", "ResizeInfo"],
    "make_resize_info": ["pemap.geometry.resize", "make_resize_info"],
    "resize_bgr_stokes_by_id": ["pemap.geometry.resize", "resize_bgr_stokes_by_id"],
    "resize_bgr_stokes_image": ["pemap.geometry.resize", "resize_bgr_stokes_image"],
    "resize_source_stokes_inputs": ["pemap.geometry.resize", "resize_source_stokes_inputs"],
    "scale_camera_matrix_for_resize": ["pemap.geometry.resize", "scale_camera_matrix_for_resize"],
    "scale_intrinsics_for_resize": ["pemap.geometry.resize", "scale_intrinsics_for_resize"],
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    imported = import_module(module)
    value = imported if symbol is None else getattr(imported, symbol)
    globals()[name] = value
    return value
