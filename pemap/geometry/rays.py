from typing import Any

import cv2
import numpy as np
import numpy.typing as npt


def ray_image(
    H: int,
    W: int,
    K: npt.NDArray[np.floating[Any]],
    distCoeffs: Any,
    rvec: Any | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Generate per-pixel unit ray directions from camera intrinsics.

    Pixel coordinates are undistorted into normalized camera coordinates,
    converted to 3D directions, and normalized to unit length.

    Parameters
    ----------
    H : int
        Image height in pixels.
    W : int
        Image width in pixels.
    K : numpy.ndarray
        Camera intrinsic matrix with shape (3, 3).
    distCoeffs : Any
        Distortion coefficients accepted by ``cv2.undistortPoints``.
    rvec : Any | None, optional
        Rotation vector describing world-to-camera transform, with shape (3,)
        or (3, 1). If provided, rays are rotated into the world frame.

    Returns
    -------
    numpy.ndarray
        Unit ray directions with shape (H, W, 3). Rays are in camera frame
        when ``rvec`` is ``None`` and in world frame otherwise.
    """
    # pixel grid: u = x/col, v = y/row
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2)  # (N,1,2)

    # undistort to normalized camera coordinates
    undist = cv2.undistortPoints(pts, K, distCoeffs)  # (N,1,2), P=None
    xy = undist.reshape(-1, 2)

    # build rays in camera frame
    rays_cam = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=xy.dtype)], axis=1)

    # normalize
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_cam = rays_cam.reshape(H, W, 3)

    if rvec is None:
        return rays_cam
    else:
        # OpenCV rvec/tvec maps world -> camera, so to get camera ray into world:
        # ray_world = R^T @ ray_cam
        R, _ = cv2.Rodrigues(rvec)
        rays_world = rays_cam.reshape(-1, 3) @ R  # equivalent to R^T @ colvec
        rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
        rays_world = rays_world.reshape(H, W, 3)
        return rays_world


def precise_rays(H, W, K, D, rvec=None):
    yy, xx = np.indices((H, W), dtype=np.float32)
    xy = cv2.undistortPointsIter(np.stack([xx, yy], -1).reshape(-1, 1, 2), K, D, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(H, W, 2)
    rays = np.concatenate([xy, np.ones((H, W, 1), np.float32)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays if rvec is None else rays @ cv2.Rodrigues(rvec)[0]
