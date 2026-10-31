"""Storage contract shared by the desktop and offline Android scene runners.

Only stdlib imports: copied verbatim into the APK by sync_processing.py.
Capture files and reviewed input masks are never publication targets.
"""
from pathlib import Path
import hashlib
import re

GROUPS = ('envmap', 'color', 'sun', 'floor')
LAYOUT_VERSION = 4


def _run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Invalid scene run ID')
    return value


def result_dir(mode):
    if mode not in ('s0', 'full'):
        raise ValueError('Unknown result mode: ' + str(mode))
    return Path('result_s0' if mode == 's0' else 'result')


def published_relative(relative, mode, run_id):
    """Map private staging paths to the shared scene layout."""
    _run_id(run_id)
    result_dir(mode)
    # Interpret separators consistently on PC and Android, including imported paths.
    text = str(relative).replace('\\', '/')
    rel = Path(text)
    if not rel.parts or rel.is_absolute() or ':' in text or '..' in rel.parts:
        raise ValueError('Invalid scene output path')
    parts = rel.parts
    if parts[0] == 'result':
        return result_dir(mode).joinpath(*parts[1:])
    if parts[0] == 'processing.json':
        return Path('analysis') / run_id / 'run.json'
    if parts[0] == 'analysis':
        return Path('analysis') / run_id / Path(*parts[1:])
    if parts[0] in GROUPS and len(parts) > 1:
        group, category = parts[:2]
        if category in ('analysis', 'mask'):
            base = Path(group) / 'analysis' / run_id
            return base.joinpath(*parts[2:]) if category == 'analysis' else base.joinpath('mask', *parts[2:])
        if category == 'result':
            if mode == 's0':
                # Retain the exact colour fit used by the quick run without
                # publishing it as the full run's native result.
                if rel == Path('color/result/wb_ccm.json'):
                    return Path('color/analysis') / run_id / 'wb_ccm.json'
                raise ValueError('S0 mode cannot publish native full results: ' + str(rel))
            return rel
    raise ValueError('Unrecognized scene output: ' + str(rel))


def storage_layout(mode, run_id):
    _run_id(run_id)
    return {'version': LAYOUT_VERSION, 'mode': mode, 'run_id': run_id,
            'final': result_dir(mode).as_posix(),
            'native_results': {g: f'{g}/result' for g in GROUPS} if mode == 'full' else {},
            'diagnostics': {g: f'{g}/analysis/{run_id}' for g in GROUPS},
            'run_record': f'analysis/{run_id}/run.json',
            'archive': f'analysis/previous_results/{run_id}',
            'paths_relative_to': 'scene root'}


def output_records(staging, folder, output_root, mode, run_id, recursive=False):
    files = folder.rglob('*') if recursive else folder.glob('*')
    records = []
    for path in sorted(files):
        if not path.is_file() or path.name == 'run.json':
            continue
        rel = published_relative(path.relative_to(staging), mode, run_id)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'path': str(output_root / rel), 'relative_path': rel.as_posix(),
                        'sha256': digest.hexdigest()})
    return records


def publish(staging, output_root, mode, run_id):
    """Publish groups atomically, final last; roll back all moves on failure."""
    staging, output_root = Path(staging).resolve(), Path(output_root).resolve()
    _run_id(run_id)
    result_dir(mode)
    if not staging.is_dir() or output_root.is_relative_to(staging):
        raise ValueError('Staging must be separate from the published scene')
    entries = list(staging.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Staged outputs must not contain symbolic links')
    result_dirs = [Path(g) / 'result' for g in GROUPS] if mode == 'full' else []
    result_dirs += [Path('result')]
    others = [p.relative_to(staging) for p in entries if p.is_file()
              and not any(p.is_relative_to(staging / d) for d in result_dirs)]
    moves = sorted(others, key=str) + [r for r in result_dirs if (staging / r).exists()]
    archive = output_root / 'analysis/previous_results' / run_id
    # Validate the entire publication before moving any existing result. In
    # particular, two legacy paths must not silently map to one diagnostic file.
    planned = []
    targets = set()
    for relative in moves:
        target = published_relative(relative, mode, run_id)
        src, dst, old = staging / relative, output_root / target, archive / target
        if not src.resolve().is_relative_to(staging) or not dst.resolve().is_relative_to(output_root) or not old.resolve().is_relative_to(output_root):
            raise ValueError('Publication path escaped scene root')
        key = target.as_posix().casefold()
        if key in targets or any(key.startswith(t + '/') or t.startswith(key + '/') for t in targets):
            raise ValueError('Conflicting scene output paths: ' + str(target))
        if old.exists() or old.is_symlink():
            raise FileExistsError('An archive already exists for this run: ' + str(old))
        targets.add(key)
        planned.append((src, dst, old))
    operations = []
    try:
        for src, dst, old in planned:
            existed = dst.exists()
            if existed:
                old.parent.mkdir(parents=True, exist_ok=True)
                dst.rename(old)
            operations.append((src, dst, old, existed))
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
    except BaseException:
        for src, dst, old, existed in reversed(operations):
            if dst.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                dst.rename(src)
            if existed and old.exists():
                old.rename(dst)
        raise
    return output_root / result_dir(mode)
