"""Spherical directions and domain resampling; no blending dependency."""

import cv2
import numpy as np
from functools import lru_cache
from pemap.geometry import _validate_payload_shape


def _sph2xyz(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical (theta, phi) to Y-up Cartesian (x, y, z)."""
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = -np.sin(theta) * np.cos(phi)
    return x, y, z


@lru_cache(maxsize=1)
def equirectangular_directions(width_envmap: int, height_envmap: int) -> np.ndarray:
    """Return per-pixel world directions for the envmap's lat-long convention."""
    map_theta, map_phi = np.meshgrid(
        np.linspace(0, np.pi, height_envmap + 2, endpoint=True, dtype=np.float32)[1:-1],  # Avoid exact poles
        np.linspace(0.0, 2.0 * np.pi, width_envmap, endpoint=False, dtype=np.float32),
        indexing="ij",
    )
    x, y, z = _sph2xyz(map_theta, map_phi)
    return np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)


def _validate_domain_rotation(domain_rotation: np.ndarray | None) -> np.ndarray | None:
    """Return a float32 SO(3) matrix or None for an unrotated spherical domain."""
    if domain_rotation is None:
        return None

    rotation = np.asarray(domain_rotation, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"domain_rotation must have shape (3, 3), got {rotation.shape}")

    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3, dtype=np.float32), atol=1e-5):
        raise ValueError("domain_rotation must be an orthonormal 3x3 matrix")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("domain_rotation must be a proper rotation with det(R) = +1")

    return rotation


def _rotation_matrix_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def octahedral_domain_rotations() -> list[tuple[str, np.ndarray]]:
    """Return six deterministic spherical-domain rotations centered on octahedral directions."""
    return [
        ("rot_00_center_pz", np.eye(3, dtype=np.float32)),
        ("rot_01_center_nz", _rotation_matrix_y(180.0)),
        ("rot_02_center_px", _rotation_matrix_y(90.0)),
        ("rot_03_center_nx", _rotation_matrix_y(-90.0)),
        ("rot_04_center_py", _rotation_matrix_x(-90.0)),
        ("rot_05_center_ny", _rotation_matrix_x(90.0)),
    ]


def _directions_to_equirectangular_maps(
    directions: np.ndarray,
    width_envmap: int,
    height_envmap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Y-up unit directions to equirectangular remap coordinates."""
    dirs = np.asarray(directions, dtype=np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    y = np.clip(dirs[..., 1], -1.0, 1.0)
    theta = np.arccos(y)
    phi = np.mod(np.arctan2(dirs[..., 0], -dirs[..., 2]), 2.0 * np.pi)

    map_x = (phi * (float(width_envmap) / (2.0 * np.pi))).astype(np.float32, copy=False)
    map_y = (theta * (float(height_envmap) / np.pi)).astype(np.float32, copy=False)
    map_y = np.clip(map_y, 0.0, float(height_envmap - 1)).astype(np.float32, copy=False)
    return map_x, map_y


def inverse_rotate_equirectangular(
    img_bgr_payload: np.ndarray,
    domain_rotation: np.ndarray | None,
) -> np.ndarray:
    """Resample a rotated-domain equirectangular payload back to the original frame.

    If the rotated-domain image is ``B_R(g) = F(R g)``, the returned image is
    sampled as ``F(d) = B_R(R.T d)``. Payloads must have shape ``(H, W, 3, S)``.
    """
    num_components = _validate_payload_shape(img_bgr_payload)
    rotation = _validate_domain_rotation(domain_rotation)
    payload = np.asarray(img_bgr_payload, dtype=np.float32)
    if rotation is None:
        return payload.copy()

    height_envmap, width_envmap = payload.shape[:2]
    directions = equirectangular_directions(width_envmap, height_envmap)
    sample_directions = (directions.reshape(-1, 3) @ rotation).reshape(height_envmap, width_envmap, 3)
    map_x, map_y = _directions_to_equirectangular_maps(sample_directions, width_envmap, height_envmap)

    padded_payload = np.concatenate([payload, payload[:, :1]], axis=1)
    remapped = np.empty_like(payload, dtype=np.float32)
    for component_idx in range(num_components):
        remapped[:, :, :, component_idx] = cv2.remap(
            padded_payload[:, :, :, component_idx],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return remapped


def inverse_rotate_equirectangular_scalar(
    weight: np.ndarray,
    domain_rotation: np.ndarray | None,
) -> np.ndarray:
    """Resample a scalar rotated-domain equirectangular map back to the original frame."""
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"Expected scalar equirectangular map shape (H, W), got {weight.shape}")

    payload = np.repeat(weight[:, :, None, None], 3, axis=2)
    return inverse_rotate_equirectangular(payload, domain_rotation)[:, :, 0, 0]
