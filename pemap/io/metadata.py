"""Per-bracket metadata in a ``hdr/stack`` directory.

Two layouts are supported (JSON is preferred when both exist):

- New (capture/io.py, 2026-10): ``<id>.npy`` + ``<id>.json`` with keys
  ``exposure_times_s`` (N,) and ``rvec`` (3, 1), plus capture info (yaw/pitch, ...).
- Legacy: ``<id>.npy`` + ``<id>_times.npy`` (N,) + ``<id>_rvec.npy`` (3, 1).
"""

import json
import re
from pathlib import Path

import numpy as np

_NATURAL_TOKEN_RE = re.compile(r"\d+|\D+")


def bracket_id_sort_key(bracket_id: str) -> tuple[tuple[int, int | str], ...]:
    """Natural-sort key for bracket IDs (e.g. 2 < 10, hoge2 < hoge10)."""
    key: list[tuple[int, int | str]] = []
    for token in _NATURAL_TOKEN_RE.findall(bracket_id):
        key.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return tuple(key)


def _json_path(stack_dir: Path, bracket_id: str) -> Path:
    return stack_dir / f"{bracket_id}.json"


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_bracket_ids(stack_dir: str | Path) -> list[str]:
    """Bracket IDs that have a ``<id>.npy`` stack and exposure times (JSON or legacy npy), natural-sorted."""
    stack_dir = Path(stack_dir)
    ids = {p.stem for p in stack_dir.glob("*.json") if (stack_dir / f"{p.stem}.npy").exists()}
    ids |= {p.stem.removesuffix("_times") for p in stack_dir.glob("*_times.npy")}
    return sorted(ids, key=bracket_id_sort_key)


def detect_camera(stack_dir: str | Path) -> str:
    """Camera model of a capture: "flir" if the bracket json says so, else "fullstokes" (Basler / legacy npy data)."""
    stack_dir = Path(stack_dir)
    for bracket_id in list_bracket_ids(stack_dir):
        json_path = _json_path(stack_dir, bracket_id)
        if json_path.exists():
            return "flir" if _read_json(json_path).get("camera") == "flir" else "fullstokes"
    return "fullstokes"


def load_exposure_times(stack_dir: str | Path, bracket_id: str) -> np.ndarray:
    """Exposure times in seconds, shape (N,)."""
    stack_dir = Path(stack_dir)
    json_path = _json_path(stack_dir, bracket_id)
    if json_path.exists():
        return np.asarray(_read_json(json_path)["exposure_times_s"], dtype=np.float32)
    return np.load(stack_dir / f"{bracket_id}_times.npy")


def load_prior_rvec(stack_dir: str | Path, bracket_id: str) -> np.ndarray | None:
    """Prior world-to-camera Rodrigues vector, shape (3, 1), or None if absent."""
    stack_dir = Path(stack_dir)
    json_path = _json_path(stack_dir, bracket_id)
    if json_path.exists():
        rvec = _read_json(json_path).get("rvec")
        if rvec is not None:
            return np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    npy_path = stack_dir / f"{bracket_id}_rvec.npy"
    if npy_path.exists():
        return np.load(npy_path)
    return None
