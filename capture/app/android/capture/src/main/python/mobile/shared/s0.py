"""S0-only float recovery for the existing Bayer color-polarization layout."""
import cv2
import numpy as np


def demosaic_s0(raw, demosaic_bayer):
    """Recover S0 without materializing four full analyzer images.

    For four evenly spaced analyzers, S0 is half their sum. All four use the
    same red/blue bilinear kernel. Interpolate just those measured planes,
    without unused green planes or channel splitting, and accumulate each one
    immediately. Preserve the original floating-point addition order as well
    as color interpolation and border replication.
    """
    if raw.ndim != 2 or raw.dtype not in (np.float32, np.float64):
        raise ValueError('Expected a 2D float32/float64 color-polarization mosaic')
    if min(raw.shape) < 3:
        return np.zeros((*raw.shape, 3), raw.dtype)
    color = np.empty((*raw.shape, 3), raw.dtype)
    for y in range(2):
        for x in range(2):
            color[y::2, x::2] = demosaic_bayer(raw[y::2, x::2], 'rggb')
    kernel = np.array([[.25,.5,.25],[.5,1.,.5],[.25,.5,.25]], raw.dtype)
    result = np.zeros_like(color)
    masked = np.zeros_like(color)
    # Existing demosaicer order: 0, 45, 90, 135 degrees.
    for y,x in [(1,1),(0,1),(0,0),(1,0)]:
        masked.fill(0)
        masked[y::2,x::2] = color[y::2,x::2]
        plane = cv2.filter2D(masked, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        plane[0] = plane[1]; plane[-1] = plane[-2]
        plane[:,0] = plane[:,1]; plane[:,-1] = plane[:,-2]
        result += plane*.5
    return result
