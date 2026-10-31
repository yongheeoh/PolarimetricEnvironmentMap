import cv2
import numpy as np


def _as_rvec(rotation: np.ndarray) -> np.ndarray:
    """Normalize a rotation input to Rodrigues-vector shape (3,)."""
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape == (3, 3):
        return cv2.Rodrigues(rotation)[0].reshape(3)
    if rotation.shape in {(3,), (3, 1), (1, 3)}:
        return rotation.reshape(3)
    raise ValueError(f"Rotation must be shape (3,3), (3,), (3,1), or (1,3). Got {rotation.shape}.")


def _as_rmat(rot: np.ndarray) -> np.ndarray:
    """Convert a rotation vector/matrix-like input to a (3, 3) rotation matrix."""
    rot = np.asarray(rot, dtype=np.float32)
    if rot.shape == (3, 3):
        return rot
    if rot.shape in {(3,), (1, 3), (3, 1)}:
        return cv2.Rodrigues(rot.reshape(3, 1))[0]
    raise ValueError(f"Rotation must be shape (3,), (1,3), (3,1), or (3,3), got {rot.shape}.")
