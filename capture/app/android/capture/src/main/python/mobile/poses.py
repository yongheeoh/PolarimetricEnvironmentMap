"""CPU SIFT and spherical rotation refinement for offline Android."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import inverse_rays
from .shared.scene_math import feature_image


def refine(images, masks, priors, K, D, check):
    ids=list(images);features={};sift=cv2.SIFT_create(nfeatures=4500,contrastThreshold=.015)
    for sid in ids:
        check('Feature matching '+sid)
        key,desc=sift.detectAndCompute(feature_image(images[sid],masks[sid]),masks[sid].astype(np.uint8)*255)
        features[sid]=(np.float32([k.pt for k in key]),desc)
    edges=[];linked={s:set() for s in ids};matcher=cv2.BFMatcher()
    R={s:cv2.Rodrigues(priors[s])[0] for s in ids}
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if np.dot(R[a][2],R[b][2])<np.cos(np.deg2rad(115)):continue
            check('Match '+a+' / '+b)
            xy1,d1=features[a];xy2,d2=features[b]
            if d1 is None or d2 is None or min(len(d1),len(d2))<20:continue
            matches=[p[0] for p in matcher.knnMatch(d1,d2,k=2) if len(p)==2 and p[0].distance<.72*p[1].distance]
            if len(matches)<20:continue
            u=inverse_rays(xy1[[m.queryIdx for m in matches]],K,D)
            v=inverse_rays(xy2[[m.trainIdx for m in matches]],K,D)
            # The stage provides a capture-local orientation, not a free scene match.
            good=np.linalg.norm(u@R[a]-v@R[b],axis=1)<.10
            u,v=u[good],v[good]
            if len(u)<20:continue
            # Rotation-only RANSAC rejects repeated floor tiles and near-field parallax.
            rng=np.random.default_rng(19);best=np.zeros(len(u),bool)
            for _ in range(100):
                sel=rng.choice(len(u),3,replace=False);U,_,V=np.linalg.svd(u[sel].T@v[sel]);rot=U@np.diag([1,1,np.linalg.det(U@V)])@V
                good=np.linalg.norm(u@rot-v,axis=1)<.006
                if good.sum()>best.sum():best=good
            if best.sum()<20:continue
            u,v=u[best],v[best];sel=np.linspace(0,len(u)-1,min(len(u),120)).astype(int)
            edges.append((a,b,u[sel],v[sel]));linked[a].add(b);linked[b].add(a)
    # Anchor each connected component to its measured stage orientation. An
    # untextured ceiling view must not discard all the other reliable matches.
    remaining=set(ids);anchors=[];components=[]
    while remaining:
        anchor=next(s for s in ids if s in remaining);seen={anchor};todo=[anchor]
        while todo:
            for s in linked[todo.pop()]-seen:seen.add(s);todo.append(s)
        remaining-=seen;anchors.append(anchor);components.append(sorted(seen,key=int))
    variable=[s for s in ids if s not in anchors]
    if not variable:return priors,{'method':'Stage prior fallback','review_required':True,'components':components}
    def matrices(x):return {**{s:R[s] for s in anchors},**{s:cv2.Rodrigues(x[3*i:3*i+3])[0]@R[s] for i,s in enumerate(variable)}}
    def residual(x):
        m=matrices(x)
        return np.concatenate([(u@m[a]-v@m[b]).ravel() for a,b,u,v in edges])
    x=np.zeros(3*len(variable))
    before=float(np.sqrt(np.mean(residual(x)**2)))
    check('Refine camera rotations')
    fit=least_squares(residual,x,bounds=(-.12,.12),loss='soft_l1',f_scale=.002,max_nfev=60)
    after=float(np.sqrt(np.mean(residual(fit.x)**2)))
    if not fit.success or after>before or np.max(np.abs(fit.x))>.119:
        return priors,{'method':'Stage prior fallback','review_required':True,'reason':'Rotation fit failed validation','before_rms':before,'after_rms':after}
    m=matrices(fit.x)
    return {s:cv2.Rodrigues(m[s])[0].reshape(3) for s in ids},{'method':'CPU SIFT + robust spherical rotation refinement','edges':len(edges),'success':bool(fit.success),'rms_ray_error':after,'before_rms_ray_error':before,'components':components,'stage_only_views':[c[0] for c in components if len(c)==1],'review_required':len(components)>1,'desktop_difference':'No GPU learned matchers; not pixel-identical to the desktop pipeline'}
