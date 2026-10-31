"""calibration / photometric / mask / estimate utilities."""
import cv2
import numpy as np
from pathlib import Path
from pemap.io.display import display


def lens_mask(profile,shape,*,base_path=None):
    if tuple(profile['image_size_wh'][::-1])!=tuple(shape):raise ValueError('Mask calibration and RAW dimensions disagree')
    if 'validity_mask' in profile or 'validity_masks' in profile:
        if base_path is None:raise ValueError('A file-backed mask requires its profile directory')
        names=profile.get('validity_masks') or [profile['validity_mask']]
        result=np.zeros(shape,bool)
        for name in names:
            path=Path(base_path)/name
            valid=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if valid is None or valid.shape!=tuple(shape):raise ValueError(f'Invalid sensor validity mask: {path}')
            result |= valid>=128
        return result
    lens=profile['lens'];y,x=np.indices(shape)
    cx,cy=lens['center_xy_raw'];rx,ry=lens['usable_semi_axes_raw']
    return ((x-cx)/rx)**2+((y-cy)/ry)**2<=1


def automatic_mask(s0,optical,pitch,group,*,mount_roll_deg=90):
    """Shared scene segmentation in upright rig coordinates for either sensor.

    Existing reviewed scene masks take precedence. Floors exclude dark non-planar
    objects as well as the rig; downward panorama masks use a restricted rig ROI.
    """
    shape=optical.shape
    if group=='envmap' and pitch>=-20:return optical.copy(),{'method':'optical image-circle mask','review_required':False}
    if not np.isfinite(mount_roll_deg) or abs(mount_roll_deg/90-round(mount_roll_deg/90))>1e-4:
        raise ValueError('Automatic rig mask requires a quarter-turn mount; supply per-view masks for other mounts')
    turns=int(round(mount_roll_deg/90))%4
    # Normalize mount orientation and preview size, independently of RAW dtype.
    image=cv2.resize(np.rot90(s0,turns),(512,612),interpolation=cv2.INTER_AREA)
    opt=cv2.resize(np.rot90(optical.astype(np.uint8),turns),(512,612),interpolation=cv2.INTER_NEAREST)>0
    yy,xx=np.indices(opt.shape)
    luminance=image[...,1]
    reference=opt & (yy>230)&(yy<450)&(xx<330)&np.isfinite(luminance)&(luminance>0)
    if not reference.any():raise ValueError('No valid mask reference region; provide a reviewed scene mask')
    median=float(np.median(luminance[reference]))
    vis=display(image,.18/max(median,1e-8))
    allowed=opt.copy() if group=='floor' else opt & (((yy>440)&(xx>95))|((yy>420+(512-xx)*.85)&(xx>365)))
    labels=np.full(opt.shape,cv2.GC_BGD,np.uint8);labels[allowed]=cv2.GC_PR_BGD
    labels[allowed&(luminance<median*.55)]=cv2.GC_PR_FGD
    labels[allowed&(luminance<median*.20)]=cv2.GC_FGD
    if not np.any(labels==cv2.GC_FGD):
        raise ValueError('Cannot seed equipment mask; provide a reviewed scene mask')
    cv2.setRNGSeed(20261018)
    cv2.grabCut(vis,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    equipment=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    n,cc,stats,_=cv2.connectedComponentsWithStats(equipment)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<12:equipment[cc==i]=0
    filled=equipment.copy();contours,_=cv2.findContours(equipment,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
    n,cc,stats,_=cv2.connectedComponentsWithStats(filled-equipment)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<250:equipment[cc==i]=1
    equipment=cv2.dilate(equipment,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    valid=opt&~equipment
    valid=np.rot90(valid.astype(np.uint8),-turns)
    valid=cv2.resize(valid,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
    valid &= optical
    return valid,{'method':'Automatic constrained GrabCut; 2 preview px margin','mount_roll_deg':mount_roll_deg,'review_required':True,
                 'note':'Candidate equipment silhouette. Inspect saved overlay for this scene; no tripod configuration is assumed.'}
