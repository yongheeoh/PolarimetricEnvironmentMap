"""Camera and spherical/planar projection geometry."""
from dataclasses import dataclass
import cv2
import numpy as np
from pemap.geometry.spherical_coordinates import equirectangular_directions
from pemap.geometry.optimized.remap_equirectangular import remap_equirectangular


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
