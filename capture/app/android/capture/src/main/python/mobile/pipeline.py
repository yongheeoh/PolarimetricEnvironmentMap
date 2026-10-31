"""Offline, cancellable scene -> final S0 or complete Stokes EXRs.

Shared calibration/composition kernels with the desktop runner. CPU matching and
streamed single-domain ERP multiband replace its GPU ensemble/six-domain blend.
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import gc
import json
import shutil
import tempfile
import time
import cv2
import numpy as np
from .geometry import projection,equirectangular_directions
from .blend import Accumulator
from .exr import write_exr,payload_channels
from .captures import discover_captures
from .shared import kernels as pk
from .shared.bilinear import demosaicing_color, _demosaic_bayer_float
from .shared.hdr import UInt16HDRMerger
from .shared.s0 import demosaic_s0
from .shared.scene_nd import nd_transmission
from .shared.scene_math import Geometry,remap,precise_rays
from .shared.profiles import read,write,sha,source,png,display,chart_comparison,load_image,lens_mask,automatic_mask,detect_chart,fit_chart
from .shared.registration import color_matrix
from .floor import FloorMixin
from .shared import chart_fit
from .shared.scene_layout import result_dir, published_relative, storage_layout, output_records, publish


class Cancelled(Exception):pass


def cleanup_abandoned_work(output_root, scene):
    # kill(pid, 0) is a read-only liveness probe on Android/POSIX, but can
    # terminate a process on Windows. Keep work conservatively on desktop.
    if os.name != 'posix':return
    for directory in output_root.glob('.mobile_processing_*'):
        marker=directory/'work_owner.json'
        if not marker.is_file() or directory.resolve().parent!=output_root:continue
        try:owner=read(marker)
        except (OSError,ValueError):continue
        pid=owner.get('pid')
        if owner.get('schema')!='penvmap temporary work v1' or owner.get('scene')!=str(scene) or not isinstance(pid,int) or not 0<pid<2**31:continue
        try:os.kill(pid,0)
        except ProcessLookupError:shutil.rmtree(directory)
        except (PermissionError,OSError):pass


def load_prior_rvec(folder,sid):
    m=read(Path(folder)/(sid+'.json'))
    r=np.asarray(m.get('rvec'),float)
    return r.reshape(3) if r.size==3 and np.isfinite(r).all() else None


class MobileGeometry(Geometry):
    def floor_coefficients(self,warped,H):
        if warped.shape[-1]==1:return warped,{'S0_only':True,'rotation_not_required_for_scalar_intensity':True}
        from .shared.scene_math import transform,unit
        from .shared.floor_geometry import floor_pose
        R,report=floor_pose(self,H);n=self.nadir_size
        path=self.cache/('floor_coefficients_'+str(getattr(self,'counter',0))+'.npy');self.counter=getattr(self,'counter',0)+1
        payload=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,n,3,12))
        T=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
        for start in range(0,n,32):
            self.check();stop=min(start+32,n);y,x=np.indices((stop-start,n),dtype=float);y+=start
            xy=transform(np.stack([x,y],-1),np.linalg.inv(H))
            rays=unit(np.stack([(xy[...,0]-self.principal)/self.focal,(xy[...,1]-self.principal)/self.focal,np.ones_like(x)],-1))
            target=unit(np.stack([(x-self.principal)/self.focal,-np.ones_like(x),(y-self.principal)/self.focal],-1).astype(np.float32)).astype(float)
            world=rays@R@T.T;basis=(pk.align_vectors_batched(rays,[0,0,1])@[1.,0,0])@R@T.T
            basis=np.einsum('...ij,...j->...i',pk.align_vectors_batched(target,world),basis)
            frame=pk.build_measurement_frame(-target,basis)
            payload[start:stop]=pk.stokes_to_s2l2(warped[start:stop],frame[...,None,:,:]).astype(np.float32)
        payload.flush();return payload,report


class Job(FloorMixin):
    def __init__(self,scene,profiles,mode,width,callback=None,output_root=None):
        from .integrity import verify_engine
        self.snapshot_verification=verify_engine()
        self.scene=Path(scene).resolve();self.mode=mode;self.width=int(width);self.callback=callback
        if mode not in ('s0','full') or self.width not in (512,1024,2048):raise ValueError('Unsupported processing mode/size')
        self.components=1 if mode=='s0' else 12
        self.args=SimpleNamespace(no_floor=False,floor_transition=35.,floor_photometry='smooth',level=5,sun_radius=8.)
        self.warnings=[];self.inputs={};self.metadata={};self.masks={};self.saturation={};self.reliable={};self.mask_reports={}
        self.groups={};self.ids={};self.profile_paths={};self.profile_sources={};self.profiles={};self.cache_paths={}
        self.progress=0;self.run_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output_root=Path(output_root).resolve() if output_root else self.scene
        self.output_root.mkdir(parents=True,exist_ok=True)
        self.destination=self.output_root/result_dir(mode)
        self.profile_dir=Path(profiles)
        manifest=read(self.profile_dir/'manifest.json')
        external=next((p/'calibration' for p in self.scene.parents if (p/'calibration').is_dir()),None)
        for name,item in manifest.items():
            path=self.profile_dir/(name+'.json')
            if sha(path)!=item['sha256']:raise ValueError('Bundled calibration was changed: '+name)
            if external is not None and (external/item['relative_path']).is_file():path=external/item['relative_path']
            self.profile_paths[name]=path;self.profile_sources[name]=source(path);self.profiles[name]=read(path)
        captures=discover_captures(self.scene)
        self.groups,self.ids,self.raw_shape=captures.groups,captures.ids,captures.raw_shape
        self.factor=2 if mode=='s0' else 1
        self.shape=tuple(s//self.factor for s in self.raw_shape)
        profile=self.profiles['intrinsic'];self.K=np.array(profile['camera_matrix']['values'],float)
        self.D=np.array(profile['dist_coeffs']['values'],float)
        if (profile['height'],profile['width'])!=self.raw_shape:raise ValueError('Intrinsic calibration does not match sensor size')
        self.K[:2,:2]/=self.factor;self.K[:2,2]=(self.K[:2,2]+.5)/self.factor-.5
        self.geo=MobileGeometry(self.K,self.D,profile['mapping_validation']['max_observed_normalized_radius'],self.width,self.width//2)
        self.optical=cv2.resize(lens_mask(self.profiles['mask'],self.raw_shape).astype(np.uint8),(self.shape[1],self.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        hp=self.profiles['hdr'];self.zmin=float(hp['hdr_merge']['z_min']);self.zmax=float(hp['hdr_merge']['z_max']);self.scale=float(hp['output_scale']['FLIR_SCALE'])
        self.hdr_merger=UInt16HDRMerger(self.zmin,self.zmax)
        self.offset=float(self.profiles['polarimetric']['parameters']['FLIR_ANGLE_OFFSET_DEG']);self.flip_s2=self.profiles['polarimetric']['parameters']['FLIR_FLIP_S2']

    def check(self,stage=None,progress=None):
        if progress is not None:self.progress=progress
        if self.callback is not None:
            if bool(self.callback.isCancelled()):raise Cancelled('Cancelled; previous results preserved')
            if stage is not None:self.callback.onProgress(int(self.progress),stage)

    def warn(self,message):
        self.warnings.append(str(message))

    def output_path(self,relative):
        return self.output_root/published_relative(relative,self.mode,self.run_id)

    def verify_sources(self):
        """Publish only against the exact captures and profiles read by this job."""
        captures=0
        for key,record in self.inputs.items():
            for kind in ('raw','metadata'):
                self.check('Verify '+key+' / '+kind)
                item=record[kind]
                if sha(item['path'])!=item['sha256']:
                    raise ValueError('Capture changed during processing: '+key+' / '+kind)
                captures+=1
        for name,item in self.profile_sources.items():
            self.check()
            if sha(item['path'])!=item['sha256']:
                raise ValueError('Calibration profile changed during processing: '+name)
        masks=0
        for name,report in self.mask_reports.items():
            item=report.get('source')
            if item is None:continue
            self.check()
            if sha(item['path'])!=item['sha256']:
                raise ValueError('Input mask changed during processing: '+name)
            masks+=1
        return {'original_capture_files_unchanged':captures,'calibration_profiles_unchanged':len(self.profile_sources),
                'input_masks_unchanged':masks}

    def recover(self,group,sid):
        key=(group,sid)
        if key in self.cache_paths:return np.load(self.cache_paths[key],mmap_mode='r')
        self.check('HDR '+group+' / '+sid)
        raw_path=self.groups[group]/(sid+'.npy');meta_path=raw_path.with_suffix('.json')
        # Fingerprint before reading values, then verify again before publication.
        # Hashing only after reading could attribute old exposure values to a newly edited JSON.
        meta_source=source(meta_path);m=read(meta_path);raw_source=source(raw_path)
        raw=np.load(raw_path,mmap_mode='r');times=np.asarray(m['exposure_times_s'],np.float32)
        if raw.ndim!=3 or raw.dtype!=np.uint16 or tuple(raw.shape[1:])!=self.raw_shape or times.ndim!=1 or len(times)!=len(raw) or not np.isfinite(times).all() or np.any(times<=0):raise ValueError('Invalid HDR stack/times: '+str(raw_path))
        if m.get('camera','flir').lower() not in ('flir','fake'):raise ValueError('Only FLIR polarimetric RAW supported')
        if m.get('pixel_format','BayerRG16')!='BayerRG16':raise ValueError('Unsupported pixel format')
        self.metadata[key]=m;record={'raw':raw_source,'metadata':meta_source,'exposure_times_s':times.tolist()};self.inputs[f'{group}/{sid}']=record
        t_bgr=np.ones(3,np.float32)
        if group=='sun':
            od,t_bgr=nd_transmission(self.profiles['nd'],m,sid,dtype=np.float32)
            record.update(nd_od=od,T_BGR=t_bgr.tolist())
        path=self.cache/(group+'_'+sid+'.npy');s=1 if self.mode=='s0' else 4
        out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(*self.shape,3,s))
        saturated=np.zeros(self.raw_shape,bool);reliable=np.zeros(self.raw_shape,bool)
        angles=np.deg2rad(np.array([0,45,90,135])+self.offset)
        A=np.column_stack([np.ones(4),np.cos(2*angles),np.sin(2*angles)])*.5
        pinv=np.linalg.pinv(A).astype(np.float32)
        for start in range(0,self.raw_shape[0],256):
            self.check();stop=min(start+256,self.raw_shape[0]);lo=max(0,start-16);hi=min(self.raw_shape[0],stop+16)
            core=raw[:,start:stop]
            saturated[start:stop]=self.hdr_merger.saturated[core.min(axis=0)]
            reliable[start:stop]=np.any(self.hdr_merger.valid[core],axis=0)
            merged=self.hdr_merger.merge(raw[:,lo:hi],times)*self.scale
            if s==1:
                result=demosaic_s0(merged,_demosaic_bayer_float)[...,None]
            else:
                polar=demosaicing_color(merged);result=np.zeros((hi-lo,self.raw_shape[1],3,s),np.float32)
                for j in range(3):
                    for k in range(4):result[...,j]+=polar[k]*pinv[j,k]
                del polar
            if s>1 and self.flip_s2:result[...,2]*=-1
            result/=t_bgr[None,None,:,None];result=result[start-lo:stop-lo]
            if self.factor>1:
                result=cv2.resize(result.reshape(stop-start,self.raw_shape[1],3*s),(self.shape[1],(stop-start)//self.factor),interpolation=cv2.INTER_AREA).reshape((stop-start)//self.factor,self.shape[1],3,s)
            out[start//self.factor:stop//self.factor]=result
            del core,merged,result
        out.flush();del out,raw
        self.saturation[key]=cv2.resize(cv2.dilate(saturated.astype(np.uint8),np.ones((9,9),np.uint8)),(self.shape[1],self.shape[0]),interpolation=cv2.INTER_AREA)>0
        self.reliable[key]=cv2.resize(cv2.erode(reliable.astype(np.uint8),np.ones((9,9),np.uint8)).astype(np.float32),(self.shape[1],self.shape[0]),interpolation=cv2.INTER_AREA)>.9999
        self.cache_paths[key]=path
        return np.load(path,mmap_mode='r')

    def corrected(self,data):return np.einsum('...cs,cd->...ds',data,self.color).astype(np.float32)

    def choose_color(self):
        self.check('ColorChecker / WB + CCM',8)
        signature={k:sha(self.profile_paths[k]) for k in ['hdr','polarimetric']};signature['vignetting']=None
        global_profile=self.profiles['wb_cc'];wb,_,_=color_matrix(global_profile);candidates=[]
        target=chart_fit.reference_rgb(np.asarray(global_profile['reference']['lab_d50_row_major']))
        old_path=self.scene/'color/result/wb_ccm.json';old=read(old_path) if old_path.exists() else {}
        for sid in self.ids['color']:
            self.check('ColorChecker '+sid)
            data=self.recover('color',sid);record=self.inputs[f'color/{sid}']
            if old.get('source_scope')=='scene' and old.get('source',{}).get('sha256')==record['raw']['sha256'] and old.get('source_metadata',{}).get('sha256')==record['metadata']['sha256'] and old.get('processing_signature')==signature:
                candidates.append(old);continue
            try:
                manual=self.scene/'color/analysis'/(sid+'_corners.json')
                if manual.exists():corners=(np.asarray(read(manual)['corners_raw'])+.5)/self.factor-.5
                else:corners,_=detect_chart(data[...,0],target,wb)
                profile=fit_chart(data[...,0],corners,global_profile['reference'],reliable=self.reliable['color',sid])
                profile.update(source_scope='scene',source=record['raw'],source_metadata=record['metadata'],processing_signature=signature,shot_id=sid,mobile_sampling_factor=self.factor)
                candidates.append(profile)
            except (ValueError,KeyError,np.linalg.LinAlgError) as e:self.warn('Color '+sid+': '+str(e))
        self.color_profile=min(candidates,key=lambda p:p['metrics']['colored_patch_leave_one_out']['mean_delta_e00']) if candidates else dict(global_profile,source_scope='global')
        for candidate in candidates:
            sid=candidate.get('shot_id')
            if sid is None:
                sid=next((i for i in self.ids['color'] if self.inputs[f'color/{i}']['raw']['sha256']==candidate.get('source',{}).get('sha256')),None)
            polygons=candidate.get('roi_polygons_raw')
            factor=candidate.get('mobile_sampling_factor',1)
            if polygons is None and (self.scene/'color/analysis/measurements.json').exists():
                polygons=read(self.scene/'color/analysis/measurements.json').get('roi_polygons_raw');factor=1
            if sid is not None and polygons is not None:
                polygons=(np.asarray(polygons)+.5)*factor/self.factor-.5
                png(self.out/'color/analysis'/f'{sid}_patches.png',chart_comparison(
                    self.recover('color',sid)[...,0],candidate,polygons,title=f'Color {sid}: captured / corrected / GT'))
        self.wb,self.ccm,self.color=color_matrix(self.color_profile)
        if self.ids['color'] and not candidates:self.warn('Color fit failed; global WB/CCM used')
        write(self.out/'color/result/wb_ccm.json',self.color_profile)

    def mask(self,group,sid,data):
        path=self.scene/('envmap/mask' if group=='envmap' else 'floor/analysis/mask')/(sid+'.png')
        if path.exists():
            mask_source=source(path);a=load_image(path)
            if a.shape!=self.raw_shape:raise ValueError('Mask dimensions do not match RAW')
            valid=cv2.resize((a>127).astype(np.float32),(self.shape[1],self.shape[0]),interpolation=cv2.INTER_AREA)>.9999;valid&=self.optical
            report={'method':'Existing scene mask','source':mask_source,'review_required':False}
        else:
            m=self.metadata[group,sid]
            if abs(float(m.get('mount_roll_deg',90))-90)>1e-3:raise ValueError('Supply per-view mask for nonstandard mount roll')
            valid,report=automatic_mask(data[...,0],self.optical,float(m.get('pitch_deg',0)),group)
        self.masks[group,sid]=valid;self.mask_reports[f'{group}/{sid}']=report
        if report['review_required']:self.warn(group+'/'+sid+': inspect automatically generated equipment mask')
        folder=self.out/group/('mask' if group=='envmap' else 'analysis/mask')
        png(folder/(sid+'.png'),cv2.resize(valid.astype(np.uint8)*255,(self.raw_shape[1],self.raw_shape[0]),interpolation=cv2.INTER_NEAREST))
        small=cv2.resize(data[...,0],(612,512),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(valid.astype(np.uint8),(612,512),interpolation=cv2.INTER_NEAREST)>0
        preview=display(small,.18/max(float(np.median(small[...,1][mask])),1e-8));preview[~mask]=(preview[~mask]*.5+np.array([0,0,128])).astype(np.uint8)
        png(self.out/group/'analysis/mask_overlay'/(sid+'.png'),preview)

    def prepare(self):
        images={};masks={};self.prior={}
        for i,sid in enumerate(self.ids['envmap']):
            self.check('Prepare view '+sid,12+int(22*i/len(self.ids['envmap'])))
            data=self.recover('envmap',sid);self.mask('envmap',sid,data)
            images[sid]=cv2.resize(data[...,0],(612,512),interpolation=cv2.INTER_AREA)
            masks[sid]=cv2.resize(self.masks['envmap',sid].astype(np.uint8),(612,512),interpolation=cv2.INTER_NEAREST)>0
            r=load_prior_rvec(self.groups['envmap'],sid)
            if r is None:raise ValueError('Missing stage rotation for view '+sid)
            self.prior[sid]=r
        from .poses import refine
        K=self.K.copy();sx,sy=612/self.shape[1],512/self.shape[0];K[0]*=sx;K[1]*=sy;K[0,2]+=(sx-1)/2;K[1,2]+=(sy-1)/2
        self.check('Align panorama views',35)
        self.poses,self.pose_report=refine(images,masks,self.prior,K,self.D,self.check)
        if self.pose_report.get('review_required'):self.warn('Inspect registration: some views rely on stage rotations; see poses report')
        self.pose_report['rvec_by_id']={s:r.tolist() for s,r in self.poses.items()}
        write(self.out/'envmap/analysis/poses.json',self.pose_report)
        auxiliary_count=len(self.ids['floor'])+len(self.ids['sun']);prepared=0
        for group in ('floor','sun'):
            valid=[]
            for sid in self.ids[group]:
                self.check('Prepare '+group+' '+sid,36+int(7*prepared/max(auxiliary_count,1)));prepared+=1
                try:
                    data=self.recover(group,sid)
                    if group=='floor':self.mask(group,sid,data)
                    valid.append(sid)
                except (ValueError,KeyError) as e:
                    if group=='floor':raise
                    self.warn('Sun '+sid+' excluded: '+str(e))
            if self.ids[group] and not valid:raise ValueError('No usable '+group+' captures')
            self.ids[group]=valid

    def coefficients(self,data,rotation):
        if self.mode=='s0':return data
        path=self.cache/'sun_coefficients.npy';out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(*self.shape,3,12))
        for first in range(0,self.shape[0],32):
            self.check();last=min(first+32,self.shape[0]);yy,xx=np.indices((last-first,self.shape[1]),dtype=np.float32);yy+=first
            from .geometry import inverse_rays
            rays=inverse_rays(np.stack([xx,yy],-1),self.K,self.D)
            basis=(pk.align_vectors_batched(rays,[0,0,1])@[1.,0,0])@rotation
            frame=pk.build_measurement_frame(-(rays@rotation),basis)
            out[first:last]=pk.stokes_to_s2l2(data[first:last],frame[...,None,:,:]).astype(np.float32)
        out.flush();return out

    def warp(self,data,sid,points):
        R=cv2.Rodrigues(self.poses[sid])[0];xy,valid=projection(points,R,self.K,self.D,self.shape,self.geo.limit)
        mask=cv2.remap(self.masks['envmap',sid].astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)>.9999;valid&=mask
        samples=remap(data,xy)
        if self.mode=='s0':payload=samples
        else:
            payload=np.zeros((*points.shape[:2],3,12),np.float32)
            for y in range(0,points.shape[0],32):
                self.check();keep=valid[y:y+32]
                if not keep.any():continue
                p=points[y:y+32][keep];rays=p@R.T
                basis=(pk.align_vectors_batched(rays,[0,0,1])@[1.,0,0])@R
                frame=pk.build_measurement_frame(-p,basis)
                payload[y:y+32][keep]=pk.stokes_to_s2l2(samples[y:y+32][keep],frame[...,None,:,:]).astype(np.float32)
        payload[~valid]=0
        return payload,valid,xy

    def panorama(self):
        points=self.geo.directions;owner=np.full(points.shape[:2],-1,np.int16);best=np.zeros(points.shape[:2],np.float32);sat=np.zeros_like(best,bool)
        for i,sid in enumerate(self.ids['envmap']):
            self.check('Panorama coverage '+sid,44)
            R=cv2.Rodrigues(self.poses[sid])[0];xy,valid=projection(points,R,self.K,self.D,self.shape,self.geo.limit)
            valid&=cv2.remap(self.masks['envmap',sid].astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)>.9999
            radius=np.hypot(xy[...,0]-self.shape[1]/2,xy[...,1]-self.shape[0]/2)
            weight=np.clip(1-radius/np.hypot(self.shape[1]/2,self.shape[0]/2),0,1)*valid
            take=weight>best;owner[take]=i;best[take]=weight[take]
            sat|=(cv2.remap(self.saturation['envmap',sid].astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)>0)&valid
        accumulator=Accumulator((*points.shape[:2],3,self.components),5)
        n=self.geo.nadir_size;self.reference=np.zeros((n,n,3),np.float32);refbest=np.zeros((n,n),np.float32)
        for i,sid in enumerate(self.ids['envmap']):
            self.check('Multiband view '+sid,45+int(25*i/len(self.ids['envmap'])))
            data=self.recover('envmap',sid);payload,valid,_=self.warp(data,sid,points)
            accumulator.add(payload,(owner==i).astype(np.float32),valid)
            del payload
            if float(self.metadata['envmap',sid].get('pitch_deg',0))<0:
                p=self.geo.plane_rays();R=cv2.Rodrigues(self.poses[sid])[0];xy,good=projection(p,R,self.K,self.D,self.shape,self.geo.limit)
                good&=cv2.remap(self.masks['envmap',sid].astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)>.9999
                value=remap(data[...,0],xy);weight=np.where(good,(p@R.T)[...,2],0);take=weight>refbest
                self.reference[take]=value[take]@self.color;refbest[take]=weight[take]
            gc.collect()
        result=self.corrected(accumulator.finish());observed=owner>=0;result[~observed]=0;self.reference_known=refbest>0
        return result,observed,sat

    def export(self,result,covered,folder):
        folder.mkdir(parents=True,exist_ok=True)
        if self.mode=='s0':write_exr(folder/'s0.exr',payload_channels(result),self.check)
        else:
            write_exr(folder/'s2l2.exr',payload_channels(result,'S2L2'),self.check)
            for name,fn in [('thetaphi',pk.thetaphi_frame),('duff',pk.duff_frame)]:
                self.check('Save '+name+' EXR')
                target=np.lib.format.open_memmap(self.cache/(name+'.npy'),mode='w+',dtype=np.float32,shape=(*result.shape[:2],3,4))
                rays=self.geo.directions
                for y in range(0,len(result),32):
                    self.check();target[y:y+32]=pk.s2l2_to_stokes(result[y:y+32],fn(-rays[y:y+32])[...,None,:,:]).astype(np.float32)
                target.flush();write_exr(folder/(name+'.exr'),payload_channels(target),self.check);del target
        png(folder/'coverage_mask.png',covered.astype(np.uint8)*255)
        png(folder/'s0.png',display(result[...,0],self.multiplier))

    def apply_sun(self, base, observed, sat):
        """Adapt streamed mobile captures to the shared Sun input contract."""
        from .shared.composition import AnalysisOutput, PanoramaState
        from .shared.sun import SunCapture, SunInputs, composite_sun

        def load(sid):
            data = self.recover('sun',sid)
            info = self.inputs[f'sun/{sid}']
            return SunCapture(data,self.reliable['sun',sid],load_prior_rvec(self.groups['sun'],sid),
                              info['nd_od'],info['T_BGR'])

        inputs = SunInputs(self.geo,self.optical,self.color,self.prior,self.poses,
                           self.ids['sun'],load,self.coefficients,radius=self.args.sun_radius,
                           mask_mode=getattr(self.args,'sun_mask_mode','adaptive'))
        analysis = AnalysisOutput(self.out/'sun/analysis',self.output_path('sun/analysis'))
        result = composite_sun(inputs,PanoramaState(base,observed,sat),analysis=analysis)
        for message in result.warnings:self.warn(message)
        state = result.panorama
        return state.payload,state.coverage,state.saturation,result.report

    def execute(self):
        start=time.monotonic();self.check('Check storage and calibration',1)
        timings={}
        def measured(name, operation):
            beginning=time.monotonic()
            try:return operation()
            finally:timings[name]=time.monotonic()-beginning
        # Remove only this engine's marked, abandoned temporary directories.
        # An interrupted Android process cannot execute its finally block.
        cleanup_abandoned_work(self.output_root,self.scene)
        # Captures are never modified. Cache/work is unique and same-volume for atomic publication.
        estimate=len(self.ids['envmap']+self.ids['floor']+self.ids['sun']+self.ids['color'])*np.prod(self.shape)*3*(1 if self.mode=='s0' else 4)*4
        extra=4_000_000_000 if self.mode=='full' else 800_000_000
        if shutil.disk_usage(self.output_root).free<estimate+extra:raise ValueError('Not enough free storage for temporary processing data')
        self.work=Path(tempfile.mkdtemp(prefix='.mobile_processing_',dir=self.output_root));self.cache=self.work/'cache';self.cache.mkdir();self.out=self.work/'publish';self.out.mkdir()
        write(self.work/'work_owner.json',{'schema':'penvmap temporary work v1','scene':str(self.scene),'pid':os.getpid()})
        self.geo.cache=self.cache;self.geo.check=self.check
        try:
            measured('color',self.choose_color)
            measured('prepare_hdr_masks_alignment',self.prepare)
            base,observed,sat=measured('panorama_multiband',self.panorama)
            self.multiplier=float(.18/np.median(base[...,1,0][observed&(base[...,1,0]>1e-8)]))
            if self.mode=='full':
                self.check('Save base panorama',72);self.export(base,observed,self.out/'envmap/result')
            self.check('Sun / Poisson',76)
            result,covered,sat,sun_report=measured('sun_poisson',lambda:self.apply_sun(base,observed,sat))
            self.check('Floor / exposure + multiband',80)
            result,covered,sat,floor_report=measured('floor_multiband',lambda:self.apply_floor(result,covered,sat))
            del base
            for info in floor_report.get('photometric_matching',{}).values():
                if 'gain_map' in info:
                    info['gain_map']['path']=str(self.output_path(Path('floor/analysis/photometry')/Path(info['gain_map']['path']).name))
            for y in range(0,len(result),32):
                if not np.isfinite(result[y:y+32]).all():raise ValueError('Non-finite final Stokes; result was not published')
            self.check('Save final panorama',92);measured('save_final',lambda:self.export(result,covered,self.out/'result'))
            png(self.out/'sun/analysis/saturation_mask.png',sat.astype(np.uint8)*255)
            for group in (('color','floor','sun') if self.mode=='full' else ()):
                for sid in self.ids[group]:
                    self.check('Save '+group+' '+sid,95)
                    data=self.corrected(self.recover(group,sid));write_exr(self.out/group/'result'/(sid+'.exr'),payload_channels(data),self.check);del data
            metadata={'schema':'penvmap Android offline v1','created_at':datetime.now(timezone.utc).isoformat(),'mode':self.mode,'width':self.width,
                'source_sampling_factor':self.factor,'profiles':self.profile_sources,'WB_CCM':self.color_profile,
                'vignetting_applied':False,'vignetting_reason':'Same desktop default: illumination contamination unresolved',
                'inputs':self.inputs,'masks':self.mask_reports,'poses':self.pose_report,'floor':floor_report,'sun':sun_report,
                'blending':{'method':'streamed ERP multiband','level':5,'S0':'log','signed_coefficients':'linear','shared_ownership':True},
                'coverage':{'before_missing_pixels':int((~observed).sum()),'remaining_missing_pixels':int((~covered).sum()),'residual_saturated_pixels':int(sat.sum())},
                'preview_exposure':self.multiplier,'elapsed_seconds':time.monotonic()-start,'timings_seconds':timings,'warnings':self.warnings,
                'limitations':['CPU SIFT instead of desktop GPU matcher ensemble; ERP domain instead of six rotated domains.',
                    'Warped camera Stokes are converted per target ray; not byte-identical to desktop source-pixel S2L2 interpolation.',
                    'Moved Floor uses approximate planar pose; cannot recover original-view reflectance.',
                    'S3 is unmeasured and zero. Automatic equipment masks require review.']}
            metadata['verification']=measured('verify_sources',self.verify_sources)
            from .shared.scene_math import code_checks
            metadata['stokes_coordinate_checks']=code_checks()
            metadata['engine_manifest']=read(Path(__file__).parent/'shared/manifest.json')
            metadata['mobile_engine_manifest']=read(Path(__file__).parent/'engine_manifest.json')
            metadata['snapshot_verification']=self.snapshot_verification
            metadata['engine']='android'
            metadata['height']=self.width//2
            metadata['storage_layout']=storage_layout(self.mode,self.run_id)
            metadata['diagnostics']={g:str(self.output_path(g+'/analysis')) for g in ['envmap','color','floor','sun']}
            import scipy,platform
            metadata['runtime']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__}
            metadata['elapsed_seconds']=time.monotonic()-start
            for role in ['final','envmap','color','floor','sun']:
                folder=self.out/'result' if role=='final' else self.out/role/'result'
                if folder.exists() and (self.mode=='full' or role=='final'):
                    related=self.out/role/'analysis' if role!='final' else self.out/'analysis'
                    record=dict(metadata,role=role,
                        outputs=output_records(self.out,folder,self.output_root,self.mode,self.run_id),
                        related_outputs=output_records(self.out,related,self.output_root,self.mode,self.run_id,recursive=True))
                    write(folder/'run.json',record)
            write(self.out/'processing.json',metadata)
            if isinstance(result,np.memmap):result.flush();result._mmap.close()
            self.check('Publish result',99);self.publish()
            if self.callback is not None:self.callback.onProgress(100,'Complete')
            return str(self.destination)
        finally:
            gc.collect()
            # Unique directory created above; never delete a user-specified path.
            if self.work.parent==self.output_root and self.work.name.startswith('.mobile_processing_'):
                shutil.rmtree(self.work,ignore_errors=True)

    def publish(self):
        return publish(self.out,self.output_root,self.mode,self.run_id)


def run(scene,profiles,mode,width,callback=None):
    cv2.setNumThreads(2)
    return Job(scene,profiles,mode,width,callback).execute()
