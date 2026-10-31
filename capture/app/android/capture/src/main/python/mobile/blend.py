"""Bounded-view-memory Laplacian blending, shared ownership for every component."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class WarpedViews:
    ids:list
    image_stack:np.ndarray
    weight_stack:np.ndarray


def extend(image, valid, periodic=True, pad=96):
    if valid.all():return image.copy()
    if not valid.any():return np.zeros_like(image)
    pad=min(pad,image.shape[1]) if periodic else 0
    src=np.concatenate([image[:,-pad:],image,image[:,:pad]],1) if pad else image
    good=np.concatenate([valid[:,-pad:],valid,valid[:,:pad]],1) if pad else valid
    _,labels=cv2.distanceTransformWithLabels((~good).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    lookup=np.zeros((int(labels.max())+1,*image.shape[2:]),np.float32);lookup[labels[good]]=src[good]
    result=image.copy();indices=labels[:,pad:-pad] if pad else labels
    result[~valid]=lookup[indices[~valid]]
    return result


class Accumulator:
    def __init__(self,shape,level=5,periodic=True,log_s0=True):
        self.shape=shape;self.level=level;self.periodic=periodic
        self.log_s0=log_s0
        self.num=[];self.den=[]

    def down(self,a):
        if not self.periodic:return cv2.pyrDown(a)
        # Longitude wraps; latitude must never wrap north onto south.
        padded=np.concatenate([a[:,-4:],a,a[:,:4]],1)
        return cv2.pyrDown(padded)[:,2:2+(a.shape[1]+1)//2]

    def up(self,a,size):
        if not self.periodic:return cv2.pyrUp(a,dstsize=size)
        padded=np.concatenate([a[:,-2:],a,a[:,:2]],1)
        return cv2.pyrUp(padded,dstsize=(padded.shape[1]*2,size[1]))[:,4:4+size[0]]

    def add(self,image,weight,valid):
        x=image.copy()
        if self.log_s0:x[...,0]=np.log(np.maximum(x[...,0],np.finfo(np.float32).tiny))
        x=extend(x,valid,self.periodic,pad=3*2**self.level)
        flat=x.reshape(*x.shape[:2],-1);w=weight.astype(np.float32)
        for i in range(self.level):
            if i+1<self.level:
                down=self.down(flat)
                lap=flat-self.up(down,(flat.shape[1],flat.shape[0]))
            else:lap=flat
            if len(self.num)<=i:
                self.num.append(np.zeros_like(lap));self.den.append(np.zeros_like(w))
            self.num[i]+=lap*w[...,None];self.den[i]+=w
            if i+1<self.level:flat=down;w=self.down(w)

    def finish(self):
        for i in range(self.level-1,-1,-1):
            band=self.num[i]/np.maximum(self.den[i][...,None],1e-8)
            out=band if i==self.level-1 else self.up(out,(band.shape[1],band.shape[0]))+band
            self.num[i]=None
        out=out.reshape(self.shape)
        if self.log_s0:out[...,0]=np.exp(np.minimum(out[...,0],80))
        out[self.den[0]<=0]=0
        return out.astype(np.float32)


def blend_views(views,blend_method='multiband',blend_level=5,log_s0=True):
    assert blend_method=='multiband'
    images=views.image_stack;weights=views.weight_stack[...,0]
    winner=np.argmax(weights,axis=0);observed=np.max(weights,axis=0)>0
    accum=Accumulator(images.shape[1:],blend_level,False,log_s0)
    for i,img in enumerate(images):
        valid=np.isfinite(img).all(axis=(-1,-2))
        accum.add(np.nan_to_num(img),((winner==i)&observed&valid).astype(np.float32),valid)
    return accum.finish()


def guarded_multiband(base,floor,alpha,color,level=5,log_s0=True):
    """Desktop guard equations, coefficient blocks to bound phone peak memory."""
    from .shared.scene_math import smooth
    if not np.any(alpha>0):return base.copy()
    y,x=np.where(alpha>0);pad=3*2**level;n,m=alpha.shape
    box=(slice(max(0,y.min()-pad),min(n,y.max()+pad+1)),slice(max(0,x.min()-pad),min(m,x.max()+pad+1)))
    w=alpha[box];core=smooth((w-.5)*2)[...,None,None];outer=smooth(w*2)[...,None,None]
    weights=np.repeat(np.stack([1-w,w])[...,None],3,-1)
    result=base.copy()
    # Every block uses identical weights; S0 alone uses log radiance.
    for start in range(0,base.shape[-1],3):
        sl=slice(start,start+3);a,b=base[box][...,sl],floor[box][...,sl]
        first,second=a*(1-core)+b*core,a*(1-outer)+b*outer
        camera=np.einsum('...cs,cd->...ds',np.stack([first,second]),np.linalg.inv(color)).astype(np.float32)
        if start==0 and log_s0:
            if camera[...,0].min() < -2e-6:raise ValueError('Negative camera S0 before Floor blend')
            camera[...,0]=np.maximum(camera[...,0],0)
        mixed=blend_views(WarpedViews(['base','floor'],camera,weights),blend_level=level,log_s0=start==0 and log_s0)
        mixed=np.einsum('...cs,cd->...ds',mixed,color)
        out=(mixed*(1-core)+b*core)*outer+a*(1-outer)
        out[w==1],out[w==0]=b[w==1],a[w==0]
        result[box][...,sl]=out
    return result
