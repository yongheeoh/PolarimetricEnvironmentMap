"""Read-only capture discovery and validation, before any processing starts."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pemap.io.metadata import list_bracket_ids
from pemap.io.files import read, source


@dataclass
class CaptureInventory:
    groups: dict[str, Path]
    ids: dict[str, list[str]]
    metadata: dict[tuple[str, str], dict]
    records: dict[str, dict]
    shape: tuple[int, int]
    camera: str


def load_captures(scene: Path, camera: str = 'auto', *, pitches=None) -> CaptureInventory:
    """Validate RAW brackets and collect their provenance without writing files.

    Records remain mutable so recovery can add measured ND transmission later.
    Legacy aliases are accepted only when their canonical group is unambiguous.
    """
    groups = {}
    aliases_by_group = {
        'envmap': ['envmap'], 'color': ['color', 'wb_cc'],
        'sun': ['sun', 'nd'], 'floor': ['floor'],
    }
    for group, aliases in aliases_by_group.items():
        existing = [scene/alias/'hdr' for alias in aliases if list_bracket_ids(scene/alias/'hdr')]
        if len(existing) > 1:
            raise ValueError(f'Ambiguous captures for {group}: {existing}')
        groups[group] = existing[0] if existing else scene/group/'hdr'
    ids = {g:list_bracket_ids(p) for g,p in groups.items()}
    if not ids['envmap']:
        raise ValueError('No envmap HDR brackets')
    metadata = {}
    records = {}
    shape = None
    if camera not in ('auto', 'flir', 'basler'):
        raise ValueError(f'Unknown scene camera: {camera}')
    selected_camera = None if camera == 'auto' else camera
    for group, shot_ids in ids.items():
        for sid in shot_ids:
            path = groups[group]/f'{sid}.npy'
            meta_path = path.with_suffix('.json')
            shot_metadata = read(meta_path)
            stack = np.load(path, mmap_mode='r')
            sensor = shot_metadata.get('camera')
            sensor = {'pylon': 'basler', 'fullstokes': 'basler'}.get(sensor, sensor)
            if sensor not in ('flir', 'basler'):
                raise ValueError(f'Missing or unknown camera in {meta_path}')
            if selected_camera is None:
                selected_camera = sensor
            if sensor != selected_camera:
                raise ValueError(f'Mixed camera scene or --camera mismatch: {meta_path}')
            dtype = np.uint16 if sensor == 'flir' else np.uint8
            if stack.ndim != 3 or stack.dtype != dtype or not all(stack.shape) or any(n % 4 for n in stack.shape[1:]):
                raise ValueError(f'Expected {sensor} (T,H,W) {np.dtype(dtype)} RAW with a 4x4 mosaic: {path}')
            shot_metadata['camera'] = sensor
            if shape is None:
                shape = stack.shape[1:]
            if stack.shape[1:] != shape:
                raise ValueError(f'Inconsistent sensor dimensions: {path}')
            times = np.asarray(shot_metadata['exposure_times_s'])
            if times.ndim != 1 or len(times) != len(stack) or not np.isfinite(times).all() or not (times > 0).all():
                raise ValueError(f'Invalid measured exposure times: {meta_path}')
            metadata[group, sid] = shot_metadata
            # Content identity is needed only to reuse an existing ColorChecker fit.
            raw_record = source(path) if group == 'color' else {'path': str(path.resolve())}
            meta_record = source(meta_path) if group == 'color' else {'path': str(meta_path.resolve())}
            records[f'{group}/{sid}'] = {
                'raw': raw_record, 'metadata': meta_record,
                'shape': list(stack.shape), 'exposure_times_s': times.tolist(),
            }
    if pitches and any('pitch_deg' not in metadata['envmap', sid] for sid in ids['envmap']):
        from pemap.geometry.warp import _assign_pitch_by_blocks
        assigned = _assign_pitch_by_blocks(ids['envmap'], pitches)
        for sid in ids['envmap']:
            metadata['envmap', sid].setdefault('pitch_deg', assigned[sid])
    return CaptureInventory(groups, ids, metadata, records, shape, selected_camera)
