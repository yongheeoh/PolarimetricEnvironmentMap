"""Desktop Floor equations evaluated in coefficient blocks on mobile."""
import gc
import cv2
import numpy as np
from .shared.scene_math import remap
from .shared.registration import register_floor,binary
from .shared.floor_photometry import overlap_gain
from .shared.profiles import write,png,sha
from .blend import guarded_multiband


class FloorMixin:
    def apply_floor(self,base,observed,sat):
        if not self.ids['floor']:return base,observed,sat,{'applied':False,'reason':'No captures'}
        if not self.reference_known.any():raise ValueError('No downward reference for Floor registration')
        n=self.geo.nadir_size;maps,optical=self.geo.source_maps(self.optical)
        plane0,known=self.geo.nadir(base[...,0:1],observed)
        known &= self.reference_known;plane0[~known]=0
        distance=cv2.distanceTransform(known.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        transition=self.args.floor_transition;band=distance<transition
        if not band.any():return base,observed,sat,{'applied':False,'reason':'No missing nadir coverage'}
        registered={};reports={}
        for sid in self.ids['floor']:
            self.check('Register Floor '+sid)
            valid=remap(self.masks['floor',sid].astype(np.float32),maps)>.9999;valid &= optical
            s0=self.corrected(self.recover('floor',sid)[...,0:1])[...,0]
            image=remap(s0,maps);del s0
            report=register_floor(self.reference,self.reference_known,image,valid)
            reports[sid]=report
            if report['accepted']:
                H=np.array(report['homography_source_to_nadir']);v=self.geo.warp_plane(valid.astype(np.float32),H)>.9999
                image=self.geo.warp_plane(image,H)
                ring=(cv2.erode(binary(v&known),np.ones((15,15),np.uint8))>0)&(distance>=10)&(distance<=100)
                a=cv2.GaussianBlur(plane0[...,1,0],(0,0),3);b=cv2.GaussianBlur(image[...,1],(0,0),3)
                ring &= (a>1e-5)&(b>1e-5)
                if ring.sum()<500:
                    report.update(accepted=False,reason='Insufficient photometric overlap');continue
                gain=float(np.median(a[ring]/b[ring]));report.update(gain_all_components=gain,overlap_pixels=int(ring.sum()))
                margin=cv2.distanceTransform(v.astype(np.uint8),cv2.DIST_L2,5)
                registered[sid]=(H,v,gain,margin)
                report.update(missing_hole_and_transition_pixels=int((band&~v).sum()),minimum_margin_px=float(margin[band].min()))
        write(self.out/'floor/analysis/registration.json',reports)
        if not registered:raise ValueError('No reliable Floor registration; previous results preserved')
        singles=[s for s,(_,v,_,_) in registered.items() if v[band].all()]
        if singles:selected=[max(singles,key=lambda s:float(registered[s][3][band].min()))]
        else:
            selected=[];remaining=band.copy()
            while remaining.any():
                options=[s for s in registered if s not in selected]
                if not options:break
                sid=max(options,key=lambda s:int((remaining&registered[s][1]).sum()))
                if not (remaining&registered[sid][1]).any():break
                selected.append(sid);remaining &= ~registered[sid][1]
        if not selected:return base,observed,sat,{'applied':False,'reason':'No Floor support'}
        weight_sum=np.zeros((n,n),np.float32);floor_sat=np.zeros((n,n),bool)
        coefficients={};weights={};poses={};photometry={}
        for sid in selected:
            self.check('Prepare Floor coefficients '+sid)
            H,valid,gain,margin=registered[sid]
            data=self.corrected(self.recover('floor',sid));rect=remap(data,maps);del data
            rect[~optical]=0;warped=self.geo.warp_plane(rect,H);del rect
            field,photometry[sid]=overlap_gain(plane0[...,1,0],warped[...,1,0],known,valid,distance,gain)
            path=self.out/'floor/analysis/photometry'/f'{sid}_gain.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,field)
            photometry[sid]['gain_map']={'path':str(path),'sha256':sha(path),'coordinates':'Rectified nadir plane'}
            warped*=field[...,None,None]
            coeff,poses[sid]=self.geo.floor_coefficients(warped,H);del warped
            # Full-mode Geometry already returns a disk-backed coefficient tensor.
            coefficients[sid]=coeff
            weight=np.clip(margin/transition,0,1)*valid;weights[sid]=weight;weight_sum+=weight
            floor_sat|=(self.geo.warp_plane(remap(self.saturation['floor',sid].astype(np.float32),maps),H)>0)&(weight>0)
            gc.collect()
        valid=weight_sum>0
        alpha=np.clip(1-distance/transition,0,1).astype(np.float32);alpha[~valid]=0
        if not valid[band].all():self.warn('Floor captures leave uncovered pixels; see coverage mask')
        mapped_alpha=self.geo.to_erp(alpha);actual=self.geo.to_erp(valid.astype(np.float32),validity=True)
        mapped_alpha[~actual]=0;active=mapped_alpha>0;filled=~observed&actual;covered=observed|filled
        result=np.lib.format.open_memmap(self.cache/'final_coefficients.npy',mode='w+',dtype=np.float32,shape=base.shape)
        for start in range(0,self.components,3):
            self.check('Floor multiband coefficients '+str(start))
            sl=slice(start,min(start+3,self.components));count=sl.stop-sl.start
            plane,_=self.geo.nadir(base[...,sl],observed);plane[~known]=0
            floor=np.zeros((n,n,3,count),np.float32)
            for sid in selected:floor+=coefficients[sid][...,sl]*weights[sid][...,None,None]
            floor/=np.maximum(weight_sum,1e-8)[...,None,None]
            mixed=guarded_multiband(plane,floor,alpha,self.color,self.args.level,log_s0=start==0)
            mixed-=plane*(1-alpha[...,None,None])+floor*alpha[...,None,None]
            delta=self.geo.to_erp(mixed);del mixed,plane
            floor_erp=self.geo.to_erp(floor);del floor
            chunk=base[...,sl].copy()
            chunk[active]=base[...,sl][active]*(1-mapped_alpha[active,None,None])+floor_erp[active]*mapped_alpha[active,None,None]+delta[active]
            chunk[filled]=floor_erp[filled];chunk[~covered]=0
            np.testing.assert_array_equal(chunk[~(active|filled)],base[...,sl][~(active|filled)])
            result[...,sl]=chunk;del chunk,floor_erp,delta
            result.flush();gc.collect()
        mapped_alpha[filled]=1;active|=filled
        saturation=((sat&(mapped_alpha<1))|((self.geo.to_erp(floor_sat.astype(np.float32))>0)&active))&covered
        png(self.out/'floor/analysis/floor_mask.png',binary(active));png(self.out/'floor/analysis/floor_alpha.png',np.rint(mapped_alpha*65535).astype(np.uint16))
        return result,covered,saturation,{'applied':True,'shot_ids':selected,'shots':{s:reports[s] for s in selected},'approximate_poses':poses,
            'method':'Guarded multiband, coefficient blocks','level':self.args.level,'transition_nadir_px':transition,'nadir_size':n,
            'photometric_matching':photometry,'filled_pixels':int(filled.sum()),'active_pixels':int(active.sum()),
            'basis_transport':'Per-ray min_rot -> planar pose including roll -> shortest-ray transport -> world S2L2',
            'between_floor_shots':'Valid-source margin weighted world S2L2' if len(selected)>1 else 'Single measured shot'}
