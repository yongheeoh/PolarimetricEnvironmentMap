"""Check a capture session written by the phone app against what the PC pipeline expects.

    python capture/app/tools/verify_session.py <session_root> [--fuse]        (penvmap env)

Checks current envmap/floor/sun/color scenes, legacy scenes, and generic calibration hdr/{i}.npy + {i}.json (dtype, shape, 12-bit MSB alignment, json keys, the
pemap.stack_meta readers: detect_camera / load_exposure_times / load_prior_rvec, rvec == prior_rvec of
the recorded yaw/pitch/roll), optional ldr/{i}.png (16-bit RAW == mid frame), legacy previews, meta. With --fuse it
also runs pemap's HDR fusion + FLIR demosaicing on the stack, the first step of main.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capture.controls import prior_rvec  # noqa: E402
from pemap.io.metadata import detect_camera, list_bracket_ids, load_exposure_times, load_prior_rvec  # noqa: E402

JSON_KEYS = ["exposure_times_s", "rvec", "yaw_deg", "pitch_deg", "mount_roll_deg", "camera", "pixel_format", "requested_exposure_us", "bracket"]
# BFS-U3-51S5PC #20356250 has ~5 mapped defect pixels; anything far beyond that is a real alignment problem.
MAX_OFF_GRID_PIXELS = 50


def stack_dir(part: Path) -> Path:
    legacy = part / "hdr" / "stack"
    return legacy if legacy.is_dir() else part / "hdr"


def session_parts(root: Path) -> list[Path]:
    return [p for p in (root, *(root / n for n in ("envmap", "floor", "sun", "color", "wb_cc", "nd", "color_checker")))
            if stack_dir(p).is_dir()]


def check_part(part: Path, fuse: bool) -> list[str]:
    problems = []
    stack = stack_dir(part)
    ids = list_bracket_ids(stack) if stack.is_dir() else []
    raw_ids = {p.stem for p in stack.glob("*.npy") if not p.stem.endswith("_times")}
    json_ids = {p.stem for p in stack.glob("*.json")}
    for missing in sorted(raw_ids - json_ids):
        problems.append(f"{stack / missing}: raw has no JSON sidecar")
    for missing in sorted(json_ids - raw_ids):
        problems.append(f"{stack / missing}: sidecar has no raw")
    print(f"  {part.name or part}: {len(ids)} stack(s) in {stack}")
    if ids:
        cam = detect_camera(stack)
        recorded = json.loads((stack / f"{ids[0]}.json").read_text()).get("camera")
        print(f"    detect_camera -> {cam} (json camera={recorded})")
        if cam != "flir" and recorded != "fake":  # the app's synthetic camera is labelled honestly
            problems.append(f"{stack}: detect_camera gave {cam}")
    for i in ids:
        arr = np.load(stack / f"{i}.npy", mmap_mode="r")
        meta = json.loads((stack / f"{i}.json").read_text())
        if part.name == "floor":
            floor = meta.get("floor_shot", {})
            if (meta.get("capture_kind") != "floor" or floor.get("include_in_rotation") is not False
                    or floor.get("scene_pose_known") is not False or not floor.get("source_scene")):
                problems.append(f"{stack / f'{i}.json'}: missing floor reference / separate-viewpoint metadata")
        times = load_exposure_times(stack, i)
        rvec = load_prior_rvec(stack, i)
        line = f"    {i}: {arr.dtype} {arr.shape} times {np.round(times * 1e6, 1).tolist()} us"
        if arr.dtype != np.uint16 or arr.ndim != 3 or any(d <= 0 for d in arr.shape):
            problems.append(f"{i}.npy: {arr.dtype} {arr.shape}, want uint16 (T, H, W)")
            continue
        if arr.shape[0] != len(times):
            problems.append(f"{i}: {arr.shape[0]} frames but {len(times)} exposure times")
        if not np.isfinite(times).all() or not (times > 0).all():
            problems.append(f"{i}: exposure times must be finite and positive")
        # 12-bit data is MSB aligned (steps of 16). DefectCorrectStatic (Average) fills the camera's few mapped
        # defect pixels with a neighbour mean, which can fall between steps: same 1-5 positions on PC and phone.
        mid = np.asarray(arr[len(arr) // 2])
        off_grid = int(np.count_nonzero(mid % 16))
        if off_grid:
            line += f" | {off_grid} defect-corrected px"
        if off_grid > MAX_OFF_GRID_PIXELS:
            problems.append(f"{i}.npy: {off_grid} values not multiples of 16 (12-bit MSB aligned BayerRG16 expected)")
        missing_keys = set(JSON_KEYS) - meta.keys()
        if missing_keys:
            problems.append(f"{i}.json missing keys {sorted(missing_keys)}")
            continue
        want = prior_rvec(meta["yaw_deg"], meta["pitch_deg"], meta["mount_roll_deg"]).reshape(-1)
        err = float(np.abs(np.asarray(rvec).reshape(-1) - want).max())
        line += f" rvec err {err:.1e}"
        if err > 1e-9:
            problems.append(f"{i}.json rvec differs from prior_rvec by {err}")
        raw_only = meta.get("ldr", {}).get("kind") == "raw"
        raw_png = part / "ldr" / f"{i}.png" if raw_only else part / "ldr" / "raw" / f"{i}.png"
        if raw_only:
            if meta.get("save_raw_png") is not raw_png.exists():
                problems.append(f"{raw_png}: presence differs from save_raw_png")
            if meta["ldr"].get("frame_index") != len(arr) // 2:
                problems.append(f"{i}: LDR must refer to the middle exposure")
            for legacy_preview in (part / "ldr" / "s0" / f"{i}.png", part / "ldr" / "raw" / f"{i}.png"):
                if legacy_preview.exists():
                    problems.append(f"{legacy_preview}: unexpected legacy LDR file in RAW-only capture")
        if raw_png.exists():
            png = cv2.imread(str(raw_png), cv2.IMREAD_UNCHANGED)
            same = png is not None and png.dtype == np.uint16 and np.array_equal(png, np.asarray(arr[len(arr) // 2]))
            line += f" | RAW PNG {'== mid frame' if same else 'MISMATCH'}"
            if not same:
                problems.append(f"{raw_png} is not the 16-bit mid frame")
        if not raw_only:
            s0_png = (part / "ldr" / "s0" if stack.name == "stack" else part / "ldr") / f"{i}.png"
            s0 = cv2.imread(str(s0_png), cv2.IMREAD_UNCHANGED) if s0_png.exists() else None
            if s0 is None:
                problems.append(f"{s0_png}: S0 preview missing or unreadable")
            line += f" | legacy S0 {None if s0 is None else s0.shape}"
        print(line)
    if fuse and ids:
        import pemap  # noqa: E402

        out = pemap.fuse_bgr_stokes_from_path(stack, camera="flir", max_workers=2)
        for k, s in out.items():
            print(f"    fused {k}: {s.shape} {s.dtype} S0 range [{float(s[..., 0].min()):.3g}, {float(s[..., 0].max()):.3g}] finite={bool(np.isfinite(s).all())}")
            if s.shape[-2:] != (3, 4) or not np.isfinite(s).all():
                problems.append(f"fused {k}: {s.shape} finite={np.isfinite(s).all()}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--fuse", action="store_true", help="also run pemap HDR fusion + FLIR demosaicing")
    args = ap.parse_args()
    root = args.session
    print(f"session {root}")
    parts = session_parts(root)
    problems = []
    for part in parts:
        try:
            problems.extend(check_part(part, args.fuse))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            problems.append(f"{part}: {type(exc).__name__}: {exc}")
    if not any(list_bracket_ids(stack_dir(p)) for p in parts):
        problems.append("No raw + sidecar pairs found; zero captures cannot pass verification")
    for stitch in (root / "envmap" / "preview.png", root / "ldr" / "stitch_s0.png"):
        if stitch.exists():
            panorama = cv2.imread(str(stitch))
            if panorama is None:
                problems.append(f"{stitch}: panorama unreadable")
            else:
                print(f"  {stitch.relative_to(root)} {panorama.shape}")
    meta = root / "meta" / "capture_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        print(f"  capture_meta.json: {len(m)} keys, camera={m.get('camera')} pixel_format={m.get('pixel_format')}")
        floor_ids = list_bracket_ids(stack_dir(root / "floor")) if stack_dir(root / "floor").is_dir() else []
        if floor_ids or "num_floor" in m:
            if m.get("num_floor") != len(floor_ids):
                problems.append(f"floor count: meta={m.get('num_floor')}, raw pairs={len(floor_ids)}")
            shots = m.get("floor_shots") or []
            if not isinstance(shots, list) or {str(s.get("idx")) for s in shots if isinstance(s, dict)} != {str(i) for i in floor_ids}:
                problems.append("floor_shots indices differ from saved floor raw pairs")
        if m.get("capture_kind") == "calibration":
            expected = m.get("num_calibration")
            ids = list_bracket_ids(stack_dir(root))
            if not isinstance(expected, int) or expected != len(ids):
                problems.append(f"calibration count: meta={expected}, raw pairs={len(ids)}")
            kind = m.get("calibration", {}).get("type")
            if not kind:
                problems.append("calibration type missing")
            for i in ids:
                shot = json.loads((stack_dir(root) / f"{i}.json").read_text(encoding="utf-8"))
                if shot.get("calibration", {}).get("type") != kind:
                    problems.append(f"{i}: calibration type differs from session")
    else:
        problems.append("meta/capture_meta.json missing")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
