"""calibration / profiles utilities."""

from pathlib import Path
import csv
from pemap.io.files import read

DEFAULT_PROFILES = {
    "intrinsic": "geometric/intrinsic/profile_20261017_193937.json",
    "hdr": "photometric/hdr/profile_20261012_191644.json",
    "polarimetric": "polarimetric/profile_20261008_161356.json",
    "wb_cc": "radiometric/wb_cc/profile_20261016_140344.json",
    "nd": "radiometric/nd/profile_20261016_135721.json",
    "mask": "photometric/mask/profile_20261017_211341.json",
    "vignetting": "photometric/vignetting/profile_20261017_150930.json",
}


PROFILE_DIRECTORIES = {key: str(Path(value).parent) for key, value in DEFAULT_PROFILES.items()}


def _profile_camera(value):
    """Accept both current camera IDs and historical camera/model records."""
    if isinstance(value, dict):
        value = value.get("camera") or value.get("model")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Invalid camera declaration in calibration profile")
    value = value.lower()
    if "flir" in value or "blackfly" in value:
        return "flir"
    if "basler" in value or value in ("pylon", "fullstokes"):
        return "basler"
    return value


def _release_paths(scene, root):
    """The same compact calibration layout used by both distributed collections."""
    paths = {
        'intrinsic': root/'geometric/intrinsics.json',
        'hdr': root/'photometric/hdr.json',
        'polarimetric': root/'polarimetric/convention.json',
        'mask': root/'capture_masks/mask_0.png',
        'wb_cc': root/'white_balance'/f'{Path(scene).name}.json',
        'nd': root/'radiometric/nd_transmission.json',
    }
    if not paths['polarimetric'].is_file():
        paths['polarimetric']=root/'polarimetric/pcalib_tensor.npy'
    catalog = root.parent/'metadata.csv'
    if catalog.is_file():
        with catalog.open(encoding='utf-8-sig', newline='') as stream:
            row = next((r for r in csv.DictReader(stream) if r['scene_id']==Path(scene).name), {})
        if row.get('vignetting'):paths['vignetting'] = root.parent/row['vignetting']
    return paths


def _read_profile(key, path):
    """Adapt existing mask/map files in memory; never generate dataset wrappers."""
    if key == 'polarimetric' and path.suffix.lower() == '.npy':
        return {'camera':'basler','pcalib_tensor':path.name}
    if key == 'mask' and path.suffix.lower() == '.png':
        import cv2
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError(f'Cannot read mask: {path}')
        masks=sorted(path.parent.glob('mask_*.png'))
        return {'image_size_wh':list(image.shape[::-1]),'validity_masks':[p.name for p in masks],
                'pitch_masks':{p.stem.removeprefix('mask_'):p.name for p in masks}}
    if key == 'vignetting' and path.suffix.lower() == '.exr':
        return {'correction_map':path.name}
    profile=read(path)
    if key=='wb_cc' and path.parent.name=='white_balance':profile['source_scope']='scene'
    if key=='polarimetric' and 'parameters' not in profile:
        profile['parameters']={'FLIR_ANGLE_OFFSET_DEG':profile.get('angle_offset_deg',0),
                               'FLIR_FLIP_S2':profile.get('flip_s2',False)}
    return profile


def resolve_profiles(scene, root=None, overrides=None, *, camera="flir", required=None):
    """Select explicit per-camera profiles, never a newer timestamp by accident.

    FLIR retains its pinned defaults. Basler uses profiles.json selections or
    <category>/profile.json. Overrides are relative to the current directory;
    manifest paths are relative to the calibration root.
    """
    if camera not in ("flir", "basler"):
        raise ValueError(f"Unknown profile camera: {camera}")
    unknown = set(overrides or {}) - set(DEFAULT_PROFILES)
    if unknown:
        raise ValueError(f"Unknown profile keys: {sorted(unknown)}; expected {sorted(DEFAULT_PROFILES)}")
    if root is None:
        root = next((p / "calibration" for p in scene.parents if (p / "calibration").is_dir()), None)
    if root is None:
        raise ValueError(f"Cannot locate {camera} calibration; pass --calibration-root")
    root = Path(root).resolve()
    selections = {}
    manifest = root / "profiles.json"
    if manifest.is_file():
        record = read(manifest)
        declared = _profile_camera(record.get("camera"))
        if declared is not None and declared != camera:
            raise ValueError(f"Calibration manifest camera mismatch: {manifest}")
        selections = record.get("profiles", {})
        if not isinstance(selections, dict) or set(selections) - set(DEFAULT_PROFILES):
            raise ValueError(f"Invalid profile selections: {manifest}")
    required = set(DEFAULT_PROFILES) if required is None else set(required)
    paths = {}
    profiles = {}
    release_paths = _release_paths(scene, root)
    scene_selections = {}
    scene_manifest = Path(scene) / "profiles.json"
    if scene_manifest.is_file():
        record = read(scene_manifest)
        declared = _profile_camera(record.get("camera"))
        if declared is not None and declared != camera:
            raise ValueError(f"Scene calibration manifest camera mismatch: {scene_manifest}")
        scene_selections = record.get("profiles", {})
        if not isinstance(scene_selections, dict) or set(scene_selections) - set(DEFAULT_PROFILES):
            raise ValueError(f"Invalid scene profile selections: {scene_manifest}")
    for key in sorted(required):
        relative = DEFAULT_PROFILES[key] if camera == "flir" else Path(PROFILE_DIRECTORIES[key]) / "profile.json"
        selected = root / selections.get(key, relative)
        if key not in selections and not selected.is_file():selected=release_paths.get(key,selected)
        if key in scene_selections:selected=Path(scene)/scene_selections[key]
        p = Path((overrides or {}).get(key) or selected).resolve()
        if key == 'mask' and p.is_dir():
            p = p/'mask_0.png' if (p/'mask_0.png').is_file() else next(p.glob('mask_*.png'), p/'mask_0.png')
        if not p.is_file():
            raise FileNotFoundError(f"{camera} {key} profile: {p}; select it with --profile {key}=<json> or {manifest}")
        profile = _read_profile(key, p)
        declared = _profile_camera(profile.get("camera"))
        if declared is not None and declared != camera:
            raise ValueError(f"{camera} cannot use {declared} {key} profile: {p}")
        paths[key], profiles[key] = p, profile
    return paths, profiles
