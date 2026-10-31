"""Per-ray camera and planar-view Stokes transport."""

import numpy as np
from pemap.polarization.min_rot import align_vectors_batched
from pemap.polarization.optimized.s2l2_kernels import (
    build_measurement_frame,
    stokes_to_s2l2,
    duff_frame,
    thetaphi_frame,
)
from pemap.polarization.optimized.mueller_kernels import rotate_stokes_basis
from pemap.geometry.rays import ray_image
from pemap.geometry.rays import precise_rays
from pemap.geometry.projection import transform, unit
from pemap.geometry.floor_pose import floor_pose


def camera_basis(rays, rotation_model="min_rot"):
    """The two camera measurement conventions from the original main.py."""
    if rotation_model == "min_rot":
        return align_vectors_batched(rays, [0, 0, 1]) @ [1.0, 0, 0]
    if rotation_model == "mitsuba3":
        return np.cross(rays, [0, 1, 0])
    raise ValueError(f"Unknown camera rotation model: {rotation_model}")


class StokesProjector:
    """Transport camera Stokes to the selected world representation before interpolation."""

    def __init__(
        self,
        shape,
        intrinsics,
        *,
        mode="full",
        precise=True,
        rotation_dtype=np.float32,
        stokes_repr_mode="s2l2",
        rotation_model="min_rot",
    ):
        self.mode = mode
        self.rotation_dtype = rotation_dtype
        self.shape = shape
        if stokes_repr_mode not in ("s2l2", "duff", "thetaphi", "none"):
            raise ValueError(f"Unknown Stokes representation: {stokes_repr_mode}")
        if rotation_model not in ("min_rot", "mitsuba3", "none"):
            raise ValueError(f"Unknown camera rotation model: {rotation_model}")
        self.representation = "none" if rotation_model == "none" else stokes_repr_mode
        self.rotation_model = rotation_model
        self.rays = self.basis = None
        if mode == "full":
            make_rays = precise_rays if precise else ray_image
            self.rays = make_rays(*shape, intrinsics["K"], intrinsics["dist_coeffs"])
            if rotation_model != "none":
                self.basis = camera_basis(self.rays, rotation_model)
        elif mode != "s0":
            raise ValueError(f"Unknown panorama mode: {mode}")

    def __call__(self, stokes, rotation):
        if self.mode == "s0" or self.representation == "s2l2":
            return self.project_s2l2(stokes, rotation)
        if self.representation == "none":
            return np.asarray(stokes, np.float32)
        result = np.empty_like(stokes, dtype=np.float32)
        frame = duff_frame if self.representation == "duff" else thetaphi_frame
        for start in range(0, self.shape[0], 128):
            sl = slice(start, start + 128)
            rays, basis = self.rays[sl] @ rotation, self.basis[sl] @ rotation
            rotator = rotate_stokes_basis(-rays, basis, frame(-rays)[..., 0])
            result[sl] = (rotator @ stokes[sl].transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        return result

    def project_s2l2(self, stokes, rotation):
        if stokes.shape[:2] != self.shape:
            raise ValueError("Stokes and projection sensor dimensions disagree")
        if self.mode == "s0":
            return stokes[..., 0:1]
        result = np.empty((*self.shape, 3, 12), np.float32)
        for start in range(0, self.shape[0], 128):
            sl = slice(start, start + 128)
            frame = build_measurement_frame(-(self.rays[sl] @ rotation), self.basis[sl] @ rotation)
            result[sl] = stokes_to_s2l2(stokes[sl], frame[..., None, :, :]).astype(np.float32)
        return result


def stokes_panorama_to_s2l2(stokes, directions, representation):
    """Convert after Stokes blending so Sun/Floor share the same world-space contract."""
    frame = duff_frame if representation == "duff" else thetaphi_frame
    result = np.empty((*stokes.shape[:-1], 12), np.float32)
    for start in range(0, stokes.shape[0], 128):
        sl = slice(start, start + 128)
        result[sl] = stokes_to_s2l2(stokes[sl], frame(-directions[sl])[..., None, :, :]).astype(np.float32)
    return result


def floor_coefficients(geometry, warped, H, *, rotation_model="min_rot"):
    if warped.shape[-1] == 1:
        return warped, {"S0_only": True, "rotation_not_required_for_scalar_intensity": True}
    R, report = floor_pose(geometry, H)
    n = geometry.nadir_size
    y, x = np.indices((n, n), dtype=float)
    xy = transform(np.stack([x, y], -1), np.linalg.inv(H))
    rays = unit(
        np.stack(
            [
                (xy[..., 0] - geometry.principal) / geometry.focal,
                (xy[..., 1] - geometry.principal) / geometry.focal,
                np.ones((n, n)),
            ],
            -1,
        )
    )
    T = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    target = geometry.plane_rays().astype(float)
    payload = np.empty((n, n, 3, 12), np.float32)
    for start in range(0, n, 100):
        sl = slice(start, start + 100)
        lr = rays[sl]
        world = lr @ R @ T.T
        basis = camera_basis(lr, rotation_model) @ R @ T.T
        transport = align_vectors_batched(target[sl], world)
        basis = np.einsum("...ij,...j->...i", transport, basis)
        frame = build_measurement_frame(-target[sl], basis)
        payload[sl] = stokes_to_s2l2(warped[sl], frame[..., None, :, :]).astype(np.float32)
    return payload, report
