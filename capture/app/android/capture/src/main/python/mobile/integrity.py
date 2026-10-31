"""Check the exact scientific snapshot shipped in the APK, before starting a job."""
from pathlib import Path
import hashlib
import json


def verify_engine(root=None):
    root=Path(root or Path(__file__).parent).resolve()
    shared=root/'shared'
    def check(folder,name,expected):
        if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name:
            raise ValueError('Invalid processing snapshot path')
        path=folder/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Processing snapshot mismatch: '+name+'. Refresh with app/tools/sync_processing.py and rebuild the app.')
    with (shared/'manifest.json').open(encoding='utf-8') as stream:records=json.load(stream)
    if not records:raise ValueError('Empty shared processing snapshot')
    for name,item in records.items():check(shared,name,item['bundled_sha256'])
    with (root/'engine_manifest.json').open(encoding='utf-8') as stream:engine=json.load(stream)
    if not engine:raise ValueError('Empty mobile processing snapshot')
    for name,digest in engine.items():check(root,name,digest)
    return {'shared_files_verified':len(records),'engine_files_verified':len(engine)}
