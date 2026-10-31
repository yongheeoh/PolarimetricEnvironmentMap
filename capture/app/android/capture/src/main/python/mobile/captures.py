"""Read-only capture layout discovery for current and legacy phone scenes."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CaptureLayout:
    groups: dict
    ids: dict
    raw_shape: tuple


def discover_captures(scene):
    """Retain phone aliases and reject ambiguous layouts before recovery."""
    scene = Path(scene)
    groups = {}
    ids = {}
    aliases_by_group = {
        'envmap': ['envmap', ''],
        'color': ['color', 'wb_cc', 'color_checker'],
        'floor': ['floor'],
        'sun': ['sun', 'nd'],
    }
    for group, aliases in aliases_by_group.items():
        folders = [scene / alias / 'hdr' for alias in aliases]
        folders += [folder / 'stack' for folder in list(folders)]
        folders = [folder for folder in folders
                   if folder.is_dir() and list(folder.glob('[0-9]*.npy'))]
        if len(folders) > 1:
            raise ValueError('Ambiguous capture folders: ' + group)
        groups[group] = folders[0] if folders else scene / group / 'hdr'
        ids[group] = sorted((p.stem for p in groups[group].glob('*.npy') if p.stem.isdigit()), key=int)
    if not ids['envmap']:
        raise ValueError('No envmap HDR captures in this scene')
    sample = np.load(groups['envmap'] / (ids['envmap'][0] + '.npy'), mmap_mode='r')
    if sample.ndim != 3 or sample.dtype != np.uint16:
        raise ValueError('Expected (T,H,W) uint16 RAW stack')
    shape = sample.shape[1:]
    del sample
    return CaptureLayout(groups, ids, shape)
