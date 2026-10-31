import cv2
import numpy as np


def equirectangular_directions(width, height):
    phi, theta = np.meshgrid(np.arange(width)*2*np.pi/width,
                             np.arange(1,height+1)*np.pi/(height+1))
    return np.stack([np.sin(theta)*np.sin(phi),np.cos(theta),-np.sin(theta)*np.cos(phi)],-1).astype(np.float32)


def projection(points, rotation, K, D, shape, limit=np.inf):
    local=np.asarray(points).reshape(-1,3)@rotation.T
    supported=(local[:,2]>1e-7)&(np.hypot(local[:,0],local[:,1])<=min(limit,100)*local[:,2])
    local[~supported]=[0,0,1]  # Do not evaluate the lens polynomial behind the camera.
    if np.asarray(D).size==5:
        # OpenCV's Python projectPoints always computes a very large Jacobian.
        # The calibrated five-coefficient Brown model needs only these values.
        x,y=local[:,0]/local[:,2],local[:,1]/local[:,2]
        r2=x*x+y*y;k1,k2,p1,p2,k3=np.asarray(D).ravel()
        radial=1+r2*(k1+r2*(k2+r2*k3))
        u=K[0,0]*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+K[0,2]
        v=K[1,1]*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+K[1,2]
        xy=np.stack([u,v],-1).reshape(*points.shape[:-1],2).astype(np.float32)
    else:
        xy=cv2.projectPoints(local,np.zeros(3),np.zeros(3),K,D)[0].reshape(*points.shape[:-1],2).astype(np.float32)
    h,w=shape
    good=supported.reshape(points.shape[:-1])
    good&=(xy[...,0]>=0)&(xy[...,1]>=0)&(xy[...,0]<w-1)&(xy[...,1]<h-1)
    xy[~good]=-100
    return xy,good


def remap_equirectangular(src,points,rvec,K,D,width,height,dst=None,**kwargs):
    points=np.asarray(points).reshape(height,width,3)
    xy,valid=projection(points,cv2.Rodrigues(np.asarray(rvec))[0],K,D,src.shape[:2])
    flat=src.reshape(*src.shape[:2],-1)
    out=np.empty((height,width,flat.shape[-1]),np.float32)
    for c in range(flat.shape[-1]):out[...,c]=cv2.remap(flat[...,c],xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
    out[~valid]=np.nan
    return out.reshape(height,width,*src.shape[2:])


def inverse_rays(xy,K,D):
    xy=cv2.undistortPointsIter(np.asarray(xy,np.float64).reshape(-1,1,2),K,D,None,None,
          (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-10)).reshape(*xy.shape[:-1],2)
    rays=np.concatenate([xy,np.ones((*xy.shape[:-1],1))],-1)
    return rays/np.linalg.norm(rays,axis=-1,keepdims=True)
