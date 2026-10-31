"""Refresh the small Android-compatible snapshot of the shared scientific kernels.

No capture data is bundled. Android builds call --auto with the float-capable
PC environment. --check remains read-only; an explicit refresh is also available.
"""
from pathlib import Path
import ast
import hashlib
import json
import inspect
import argparse
import re

REPO = Path(__file__).resolve().parents[3]
DEST = REPO/'capture/app/android/capture/src/main/python/mobile/shared'
args = argparse.ArgumentParser(description=__doc__)
mode = args.add_mutually_exclusive_group()
mode.add_argument('--check',action='store_true',help='Verify snapshot/source hashes without changing files')
mode.add_argument('--auto',action='store_true',help='Refresh changed sources automatically, protecting edits to generated files')
args.add_argument('--calibration-root',type=Path,default=REPO/'data/FLIR/calibration',
                  help='External profiles; --auto can reuse verified bundled pins when absent')
options = args.parse_args()
def verify_snapshot():
    import sys
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(DEST.parents[1]))
    from mobile.integrity import verify_engine
    verify_engine(DEST.parent)
    manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf-8'))
    changed=[]
    for name,item in manifest.items():
        for origin in item.get('sources', [item]):
            if origin.get('source_relative'):
                original=REPO/origin['source_relative']
            elif origin.get('package_source'):
                import importlib.util
                package,relative=origin['package_source'].split('/',1)
                spec=importlib.util.find_spec(package)
                original=Path(spec.origin).parent/relative if spec and spec.origin else None
            else:
                original=Path(origin['source'])  # Read older manifests during migration.
            if original is None or ((origin.get('source_relative') or origin.get('package_source')) and not original.is_file()) or (original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest()!=origin['source_sha256']):
                changed.append(str(original))
    assets=REPO/'capture/app/android/capture/src/main/assets/processing_profiles'
    profiles=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    for name,item in profiles.items():
        if hashlib.sha256((assets/(name+'.json')).read_bytes()).hexdigest()!=item['sha256']:
            changed.append('Bundled profile '+name)
        original=options.calibration_root/item['relative_path']
        if original.exists() and hashlib.sha256(original.read_bytes()).hexdigest()!=item.get('source_sha256',item['sha256']):
            changed.append(str(original))
    if changed:raise ValueError('Snapshot needs refresh:\n'+'\n'.join(changed))
    print('OK: shared/mobile/profile snapshots and available source hashes match; no files changed')


def protect_generated_files():
    """Builds may replace stale generated files, but never unrecorded edits."""
    shared=json.loads((DEST/'manifest.json').read_text(encoding='utf-8'))
    assets=REPO/'capture/app/android/capture/src/main/assets/processing_profiles'
    profiles=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    expected=[(DEST/name,item['bundled_sha256']) for name,item in shared.items()]
    expected += [(assets/(name+'.json'),item['sha256']) for name,item in profiles.items()]
    for path,digest in expected:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Generated file has local edits: '+str(path)+
                             '. Edit its original source instead, or run an explicit sync to replace it.')


if options.check:
    verify_snapshot()
    raise SystemExit(0)
if options.auto:
    protect_generated_files()
    try:
        verify_snapshot()
    except (ValueError,OSError,KeyError) as error:
        print('Refreshing Android processing snapshot:',error,flush=True)
    else:
        raise SystemExit(0)

# Resolve every calibration input before changing any generated output. Without
# an external calibration drive, reuse only the matching, verified bundled pin.
assets=REPO/'capture/app/android/capture/src/main/assets/processing_profiles'
tree=ast.parse((REPO/'pemap/io/profiles.py').read_text(encoding='utf-8'))
paths=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='DEFAULT_PROFILES' for t in n.targets)))
previous=json.loads((assets/'manifest.json').read_text(encoding='utf-8')) if (assets/'manifest.json').is_file() else {}
profile_inputs={}
for key,rel in paths.items():
    path=options.calibration_root/rel
    if not path.is_file() and options.auto and previous.get(key,{}).get('relative_path')==rel:
        path=assets/(key+'.json')
    if not path.is_file():raise FileNotFoundError(path)
    profile_inputs[key]=path

import polanalyser
DEST.mkdir(parents=True,exist_ok=True)
records = {}

def save(name, text, source):
    (DEST/name).write_text(text,encoding='utf-8',newline='\n')
    origins=[]
    for value in source if isinstance(source,list) else [source]:
        p=Path(value).resolve()
        origin={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if p.is_relative_to(REPO):
            origin['source_relative']=p.relative_to(REPO).as_posix()
        else:
            origin['package_source']='polanalyser/'+p.relative_to(Path(polanalyser.__file__).parent).as_posix()
        origins.append(origin)
    records[name] = {'sources':origins,'bundled_sha256':hashlib.sha256((DEST/name).read_bytes()).hexdigest()}

def selected(path,names):
    text=path.read_text(encoding='utf-8');tree=ast.parse(text)
    return '\n\n'.join(''.join('@'+ast.get_source_segment(text,d)+'\n' for d in n.decorator_list)+ast.get_source_segment(text,n)
                         for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'

def portable(text):
    replacements={
        'import pemap':'from . import kernels as pemap',
        'from pemap.geometry.projection':'from .scene_math',
        'from pemap.geometry.rays':'from .scene_math',
        'from pemap.polarization.validation':'from .scene_math',
        'from pemap.polarization.mueller_reference':'from .mueller',
        'from pemap.polarization.min_rot':'from .min_rot',
        'from pemap.polarization.optimized.mueller_kernels':'from .mueller',
        'from pemap.polarization.optimized.s2l2_kernels':'from .s2l2',
        'from pemap.geometry.spherical_coordinates':'from ..geometry',
        'from pemap.geometry.warp import WarpedViews, blend_views':'from ..blend import WarpedViews, blend_views',
        'from pemap.geometry.warp import WarpedViews':'from ..blend import WarpedViews',
        'from pemap.geometry.warp import blend_views':'from ..blend import blend_views',
        'from pemap.blending.transition import guarded_multiband':'from ..blend import guarded_multiband',
        'from pemap.composition import':'from .composition import',
        'from pemap.io.analysis':'from .composition',
        'from pemap.geometry.optimized.remap_equirectangular import remap_equirectangular':'from ..geometry import remap_equirectangular',
        'from pemap.registration.floor_registration':'from .registration',
        'from pemap.radiometry.overlap':'from .floor_photometry',
        'from pemap.geometry.floor_pose':'from .floor_geometry',
        'from pemap.polarization.transport import floor_coefficients':'from .floor_geometry import floor_coefficients',
        'from pemap.blending.poisson':'from .scene_math',
        'from pemap.io.metadata import load_prior_rvec':'from ..pipeline import load_prior_rvec',
        'from pemap.io.files':'from .profiles',
        'from pemap.io.images':'from .profiles',
        'from pemap.io.display':'from .display',
        'from pemap.radiometry.color_profile':'from .scene_color',
    }
    for a,b in replacements.items():text=text.replace(a,b)
    return text

direct={
    's2l2.py':'pemap/polarization/s2l2_reference.py', 'mueller.py':'pemap/polarization/mueller_reference.py',
    'min_rot.py':'pemap/polarization/min_rot.py', 'hdr.py':'pemap/sensor/hdr.py',
    's0.py':'pemap/sensor/s0.py', 'scene_layout.py':'pemap/io/layout.py',
    'scene_nd.py':'pemap/radiometry/nd.py', 'floor_photometry.py':'pemap/radiometry/overlap.py',
    'scene_color.py':'pemap/radiometry/color_profile.py',
    'display.py':'pemap/io/display.py',
}
for name,rel in direct.items():
    p=REPO/rel;save(name,portable(p.read_text(encoding='utf-8')),p)

pose=REPO/'pemap/geometry/floor_pose.py'
transport=REPO/'pemap/polarization/transport.py'
text=portable(pose.read_text(encoding='utf-8'))
text+='\nfrom .min_rot import align_vectors_batched\nfrom .s2l2 import build_measurement_frame, stokes_to_s2l2\n'
text+='\n'+portable(selected(transport,['camera_basis','floor_coefficients']))
save('floor_geometry.py',text,[pose,transport])

parts=[REPO/'pemap/composition/__init__.py',REPO/'pemap/io/analysis.py']
save('composition.py','\n'.join(portable(p.read_text(encoding='utf-8')) for p in parts),parts)

parts=[REPO/'pemap/geometry/projection.py',REPO/'pemap/geometry/rays.py',
       REPO/'pemap/polarization/validation.py',REPO/'pemap/blending/boundary.py',REPO/'pemap/blending/poisson.py']
# Mobile guarded_multiband imports this transition function from scene_math.
# Keep its canonical implementation in blending/transition.
transition=REPO/'pemap/blending/transition.py'
text='\n'.join(portable(p.read_text(encoding='utf-8')) for p in parts)
text+='\n'+selected(transition,['smooth'])
save('scene_math.py',text,[*parts,transition])

profile_parts={
    REPO/'pemap/io/files.py':['read','write','source','sha'],
    REPO/'pemap/io/images.py':['png','load_image','binary'],
    REPO/'pemap/radiometry/chart.py':['chart_comparison','patch_polygons','sample_chart','detect_chart','fit_chart'],
    REPO/'pemap/masking/estimate.py':['lens_mask','automatic_mask'],
}
text='from pathlib import Path\nimport json, hashlib, os\nimport cv2\nimport numpy as np\nfrom . import chart_fit\nfrom .display import display\n\n'
text+='\n'.join(portable(selected(p,names)) for p,names in profile_parts.items())
# Android's Python 3.10 has no hashlib.file_digest. Replace the selected
# function by its AST span so source formatting cannot silently break this port.
sha_node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='sha')
lines=text.splitlines(keepends=True)
lines[sha_node.lineno-1:sha_node.end_lineno]=['''def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
''']
text=''.join(lines)
save('profiles.py',text,list(profile_parts))

p=REPO/'pemap/radiometry/chart_fit.py';text=p.read_text(encoding='utf-8')
text=text.replace('from skimage.color import deltaE_ciede2000, lab2xyz, xyz2lab, xyz_tristimulus_values',
'''from skimage.color import deltaE_ciede2000, lab2xyz, xyz2lab
try:
    from skimage.color import xyz_tristimulus_values
except ImportError:
    def xyz_tristimulus_values(*, illuminant, observer):
        assert observer == "2"
        return np.array({"D50": [.9642119944211994, 1., .8251882845188288],
                         "D65": [.95047, 1., 1.08883]}[illuminant])''')
save('chart_fit.py',portable(text),p)

p=REPO/'pemap/registration/floor_registration.py'
text=portable(p.read_text(encoding='utf-8')).replace('nfeatures=18000','nfeatures=9000')
text=text.replace("report = {'accepted': False, 'matches': 0}","report = {'accepted': False, 'matches': 0, 'mobile_max_features': 9000}")
text+='\nfrom .scene_color import color_matrix\n'
save('registration.py',text,p)
p=REPO/'pemap/composition/floor.py';text=portable(p.read_text(encoding='utf-8'))
save('floor.py',text,p)
p=REPO/'pemap/composition/sun.py';save('sun.py',portable(p.read_text(encoding='utf-8')),p)
p=Path(inspect.getsourcefile(polanalyser.demosaicing)).parent/'bilinear.py'
save('bilinear.py',p.read_text(encoding='utf-8'),p)
license_path=p.parents[2]/'polanalyser-3.0.0.dist-info/licenses/LICENSE'
(DEST/'POLANALYSER_LICENSE.txt').write_text(license_path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
save('__init__.py','',Path(__file__))
save('kernels.py','from .s2l2 import *\nfrom .min_rot import align_vectors_batched\nfrom .mueller import rotate_stokes_basis\nfrom ..geometry import equirectangular_directions\n',Path(__file__))
(DEST/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8',newline='\n')
(DEST.parent/'engine_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.parent.glob('*.py')},indent=2),encoding='utf-8',newline='\n')
assets.mkdir(parents=True,exist_ok=True)
def portable_profile(value):
    """Keep profile values while dropping workstation directories from provenance."""
    if isinstance(value,dict):
        return {key:portable_profile(item) for key,item in value.items()}
    if isinstance(value,list):
        return [portable_profile(item) for item in value]
    if isinstance(value,str) and re.match(r'^[A-Za-z]:[\\/]',value):
        return value.replace('\\','/').rsplit('/',1)[-1]
    return value

profile_records={}
for key,path in profile_inputs.items():
    target=assets/f'{key}.json'
    source_hash=(previous[key].get('source_sha256',previous[key]['sha256'])
                 if path.resolve()==target.resolve() else hashlib.sha256(path.read_bytes()).hexdigest())
    profile=portable_profile(json.loads(path.read_text(encoding='utf-8')))
    target.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    profile_records[key]={'relative_path':paths[key],'source_sha256':source_hash,
                          'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(assets/'manifest.json').write_text(json.dumps(profile_records,indent=2),encoding='utf-8')
print('Updated',len(records),'kernels and',len(paths),'profiles')
