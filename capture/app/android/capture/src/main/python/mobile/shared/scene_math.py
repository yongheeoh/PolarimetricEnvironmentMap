"""Camera and spherical/planar projection geometry."""
from dataclasses import dataclass
import cv2
import numpy as np
from ..geometry import equirectangular_directions
from ..geometry import remap_equirectangular


@dataclass
class Geometry:
    K: np.ndarray
    D: np.ndarray
    limit: float
    width: int = 2048
    height: int = 1024
    nadir_size: int = 1400
    nadir_fov: float = 105.

    @property
    def focal(self):
        return self.nadir_size / (2*np.tan(np.deg2rad(self.nadir_fov/2)))

    @property
    def principal(self):
        return (self.nadir_size-1)/2

    @property
    def directions(self):
        return equirectangular_directions(self.width,self.height)

    def supported_remap(self,src,points,rvec,K,D,width,height,dst=None,**kwargs):
        result=remap_equirectangular(src,points,rvec,K,D,width,height,dst,**kwargs)
        local=np.asarray(points).reshape(-1,3)@cv2.Rodrigues(np.asarray(rvec))[0].T
        valid=(local[:,2]>0)&(np.hypot(local[:,0],local[:,1])<=self.limit*local[:,2])
        result.reshape(-1,*result.shape[2:])[~valid]=np.nan
        return result

    def plane_rays(self):
        y,x=np.indices((self.nadir_size,self.nadir_size),dtype=np.float32)
        return unit(np.stack([(x-self.principal)/self.focal,-np.ones_like(x),(y-self.principal)/self.focal],-1))

    def source_maps(self,optical):
        y,x=np.indices((self.nadir_size,self.nadir_size),dtype=np.float32)
        rays=np.stack([(x-self.principal)/self.focal,(y-self.principal)/self.focal,np.ones_like(x)],-1)
        xy=cv2.projectPoints(rays.reshape(-1,3),np.zeros(3),np.zeros(3),self.K,self.D)[0].reshape(self.nadir_size,self.nadir_size,2).astype(np.float32)
        valid=cv2.remap(optical.astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)>.9999
        valid &= np.hypot(rays[...,0],rays[...,1])<=self.limit
        return xy,valid

    def nadir(self,data,observed=None):
        rays=self.plane_rays()
        phi=np.mod(np.arctan2(rays[...,0],-rays[...,2]),2*np.pi)
        theta=np.arccos(np.clip(rays[...,1],-1,1))
        xy=np.stack([phi*self.width/(2*np.pi)+1,np.clip(theta*(self.height+1)/np.pi-1,0,self.height-1)],-1)
        sample=lambda a:remap(np.concatenate([a[:,-1:],a,a[:,:1]],axis=1),xy)
        result=sample(data)
        if observed is None:return result
        count=sample(observed.astype(np.float32)[...,None])[...,0]
        return result/np.maximum(count,1e-8).reshape(*count.shape,*([1]*(data.ndim-2))), count>.9999

    def to_erp(self,data,validity=False):
        rays=self.directions;down=-rays[...,1]
        x=self.focal*rays[...,0]/np.maximum(down,1e-6)+self.principal
        y=self.focal*rays[...,2]/np.maximum(down,1e-6)+self.principal
        good=(down>0)&(x>=0)&(x<self.nadir_size-1)&(y>=0)&(y<self.nadir_size-1)
        a=data[...,None] if data.ndim==2 else data
        value=remap(a,np.stack([x,y],-1))
        if data.ndim==2:value=value[...,0]
        value[~good]=0
        return (value>.9999)&good if validity else value

    def warp_plane(self,data,H):
        shape=data.shape[2:];flat=data.reshape(self.nadir_size,self.nadir_size,-1)
        value=np.stack([cv2.warpPerspective(flat[...,i],H,(self.nadir_size,self.nadir_size)) for i in range(flat.shape[-1])],-1)
        return value.reshape(self.nadir_size,self.nadir_size,*shape)




def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def transform(points, H):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(*points.shape)


def remap(data, xy):
    shape = data.shape[2:]
    flat = data.reshape(*data.shape[:2], -1)
    planes = [cv2.remap(flat[..., i], xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR) for i in range(flat.shape[-1])]
    return np.stack(planes, -1).reshape(*xy.shape[:2], *shape)






def feature_image(s0, valid):
    gray = np.log1p(np.maximum(s0[..., 1], 0))
    lo, hi = np.percentile(gray[valid], [1, 99])
    gray = np.rint(np.clip((gray-lo)/max(hi-lo, 1e-6), 0, 1)*255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2, tileGridSize=(12, 12)).apply(gray)

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

"""Coordinate-frame invariants checked by the scene runner."""

import numpy as np
from .min_rot import align_vectors_batched
from .s2l2 import build_measurement_frame, s2l2_to_stokes, stokes_to_s2l2
from .mueller import rotate_stokes_basis
from .scene_math import unit


def code_checks():
    rng = np.random.default_rng(180)
    rays = unit(np.column_stack((rng.uniform(-1, 1, (512, 2)), np.ones(512))))
    b = align_vectors_batched(rays, [0, 0, 1]) @ np.array([1.0, 0, 0])
    s = np.column_stack((np.ones(512), rng.uniform(-0.4, 0.4, (512, 2)), np.zeros(512)))
    identity = rotate_stokes_basis(-rays, b, -b)
    s180 = np.einsum("nij,nj->ni", identity, s)
    b90 = np.cross(-rays, b)
    s90 = np.einsum("nij,nj->ni", rotate_stokes_basis(-rays, b, b90), s)
    expected90 = s.copy()
    expected90[:, 1:3] *= -1
    F = build_measurement_frame(-rays, b)
    F180 = build_measurement_frame(-rays, -b)
    coef = stokes_to_s2l2(s, F)
    roundtrip = s2l2_to_stokes(coef, F)
    rotatedcoef = stokes_to_s2l2(s180, F180)
    result = {
        "rotation_180_max_absolute_error": float(np.max(np.abs(s180 - s))),
        "rotation_90_sign_control_error": float(np.max(np.abs(s90 - expected90))),
        "s2l2_roundtrip_max_absolute_error": float(np.max(np.abs(roundtrip - s))),
        "s2l2_180_invariance_error": float(np.max(np.abs(rotatedcoef - coef))),
        "basis_dot_ray_max": float(np.max(np.abs(np.sum(b * rays, axis=-1)))),
    }
    assert max(result.values()) < 1e-6, result
    return result

"""Extend masked values for pyramid convolution without creating coverage."""

import cv2
import numpy as np


def extend_for_pyramids(images, pad=96):
    """Nearest valid values for pyramid convolution only; validity weights stay zero.

    Treat ERP longitude as periodic. The default margin exceeds the five-level
    pyramid support. No extrapolated pixel becomes an observed panorama pixel.
    """
    result = np.asarray(images, np.float32).copy()
    for i, img in enumerate(result):
        valid = np.isfinite(img).all(-1)
        if valid.all():
            continue
        if not valid.any():
            result[i] = 0
            continue
        margin = min(int(pad), img.shape[1])
        values = np.concatenate([img[:, -margin:], img, img[:, :margin]], axis=1)
        good = np.concatenate([valid[:, -margin:], valid, valid[:, :margin]], axis=1)
        _, labels = cv2.distanceTransformWithLabels((~good).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(labels.max()) + 1, 3), np.float32)
        lookup[labels[good]] = values[good]
        core_labels = labels[:, margin:-margin]
        result[i][~valid] = lookup[core_labels[~valid]]
    assert np.isfinite(result).all()
    return result

"""pemap / sun / poisson utilities."""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def poisson_clone(background, measured, mask):
    """Solve f=measured+h, with Laplacian(h)=0 and h=background-measured outside.

    A common 4-neighbor discrete Dirichlet Laplacian is used for all trailing
    payload components. No intensity normalization, log, clipping or fixed core.
    """
    b = np.asarray(background, np.float64)
    g = np.asarray(measured, np.float64)
    mask = np.asarray(mask, bool)
    assert b.shape == g.shape and mask.shape == b.shape[:2] and mask.any()
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    yy, xx = np.where(mask)
    count = len(yy)
    indices = np.full(mask.shape, -1, np.int32)
    indices[mask] = np.arange(count)
    rows = [np.arange(count)]; cols = [np.arange(count)]; values = [np.full(count, 4.)]
    rhs = np.zeros((count, int(np.prod(b.shape[2:]))), np.float64)
    boundary_samples = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yq, xq = yy+dy, xx+dx
        interior = mask[yq, xq]
        rows.append(np.where(interior)[0]); cols.append(indices[yq[interior], xq[interior]])
        values.append(-np.ones(int(interior.sum())))
        boundary_delta = (b[yq[~interior], xq[~interior]]-g[yq[~interior], xq[~interior]]).reshape((-1, rhs.shape[1]))
        rhs[~interior] += boundary_delta
        boundary_samples.append(boundary_delta)
    matrix = csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    offset = splu(matrix).solve(rhs)
    residual = float(np.max(np.abs(matrix @ offset-rhs))/max(float(np.max(np.abs(rhs))), 1))
    assert residual < 1e-10
    bounds = np.concatenate(boundary_samples)
    assert np.all(offset >= bounds.min(axis=0)-1e-7)
    assert np.all(offset <= bounds.max(axis=0)+1e-7)
    result = b.copy()
    result[mask] = g[mask]+offset.reshape((-1, *b.shape[2:]))
    return result, {'unknown_pixels': count, 'relative_linear_system_residual': residual,
                    'offset_within_boundary_extrema': True,
                    'offset_min_by_component': offset.min(axis=0).tolist(),
                    'offset_max_by_component': offset.max(axis=0).tolist()}

def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))
