"""Copy capture sessions from the phone into the PC data tree, preserving their folder layout.

The app writes to its base path /sdcard/Documents/penvmap/{experiment}/{YYYYMMDD}/{HHMMSS}; sessions from
before 2026-10-13 in /sdcard/Android/data/kr.penvmap.capture/files/penvmap/... are listed too.
This pulls every session (or the ones matching --only) to {base}/{experiment}/{YYYYMMDD}/{HHMMSS}.
Both legacy sessions and envmap/floor/sun/color sessions are supported; PC processing path defaults are separate:

    python capture/app/tools/pull_sessions.py                      # list sessions on the phone
    python capture/app/tools/pull_sessions.py --pull [--verify]    # copy new sessions to the selected local root
    python capture/app/tools/pull_sessions.py --pull --only FLIR/outdoor --delete-after

Existing local sessions are never overwritten. --delete-after always verifies the copied session
and only removes a phone session marked finished.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ADB = os.environ.get("ADB") or shutil.which("adb") or str(
    Path(os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or Path.home() / "AppData/Local/Android/Sdk")
    / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
)
PHONE_ROOTS = {
    "app": "/sdcard/Android/data/kr.penvmap.capture/files/penvmap",
    "documents": "/sdcard/Documents/penvmap",
}
HERE = Path(__file__).resolve().parent


def adb(*args: str, check: bool = True) -> str:
    out = subprocess.run([ADB, *args], capture_output=True, text=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if check and out.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def phone_sessions() -> list[tuple[str, str]]:
    """(phone root, '{experiment}/{YYYYMMDD}/{HHMMSS}') for every session in either storage root."""
    out = []
    for root in PHONE_ROOTS.values():
        # toybox find on the phone returns nothing for -path under /sdcard (a symlink); -name works.
        listing = adb("shell", f"find {root}/ -name capture_meta.json 2>/dev/null", check=False)
        for line in listing.split():
            line = line.strip()
            if line.startswith(root + "/") and line.endswith("/meta/capture_meta.json"):
                out.append((root, line[len(root) + 1 : -len("/meta/capture_meta.json")]))
    return sorted(out, key=lambda x: x[1])


def size_mb(root: str, rel: str) -> float:
    s = adb("shell", f"du -sk '{root}/{rel}'", check=False).split()
    return int(s[0]) / 1024 if s and s[0].isdigit() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(os.environ.get("PENVMAP_DATA_DIR", "data")), help="local data root (PC base_dir)")
    ap.add_argument("--pull", action="store_true", help="copy sessions that are not on the PC yet")
    ap.add_argument("--only", default="", help="only sessions whose path starts with this (e.g. FLIR/outdoor)")
    ap.add_argument("--verify", action="store_true", help="run verify_session.py on each pulled session")
    ap.add_argument("--delete-after", action="store_true", help="delete the phone copy after a successful pull (+ verify)")
    args = ap.parse_args()

    sessions = [(root, rel) for root, rel in phone_sessions() if rel.startswith(args.only)]
    if not sessions:
        print("no sessions on the phone" + (f" under {args.only}" if args.only else ""))
        return
    names = {v: k for k, v in PHONE_ROOTS.items()}
    for root, rel in sessions:
        local = args.base / rel
        state = "on PC" if local.exists() else "new"
        print(f"{rel:56s} {names[root]:9s} {size_mb(root, rel):8.0f} MB  [{state}]")
        if not args.pull or local.exists():
            continue
        tmp = local.with_name(local.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        adb("pull", f"{root}/{rel}", str(tmp))
        tmp.rename(local)
        print(f"  -> {local}")
        ok = True
        if args.verify or args.delete_after:
            r = subprocess.run([sys.executable, str(HERE / "verify_session.py"), str(local)])
            ok = r.returncode == 0
        if args.delete_after and ok:
            meta = json.loads((local / "meta/capture_meta.json").read_text(encoding="utf-8"))
            if meta.get("session_timing", {}).get("finished") is not True:
                print("  kept on the phone: session is not marked finished")
                continue
            adb("shell", f"rm -r '{root}/{rel}'")
            print("  deleted from the phone")


if __name__ == "__main__":
    main()
