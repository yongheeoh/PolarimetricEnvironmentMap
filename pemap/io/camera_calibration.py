import json
import warnings
from pathlib import Path

import numpy as np

from pemap.io.metadata import load_prior_rvec


def load_intrinsics(json_path: str | Path) -> dict:
    """Load OpenCV intrinsics from a calibration JSON file.

    Returns a dict with:
    - K: (3, 3) float64 camera matrix
    - dist_coeffs: (N,) float64 distortion coefficients
    - image_size_wh: (width, height) tuple
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"]["values"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"]["values"], dtype=np.float64)
    image_size_wh = (data["width"], data["height"])
    return {"K": K, "dist_coeffs": dist_coeffs, "image_size_wh": image_size_wh}


def load_rvecs(bracket_ids, stack_dir: str | Path) -> dict:
    """Load per-bracket Rodrigues rotation vectors from ``<id>.json`` (``rvec``) or legacy ``<id>_rvec.npy``.

    Returns a dict mapping bracket_id -> rvec ndarray (shape (3, 1), float64).
    Missing entries are skipped with a warning.
    """
    stack_dir = Path(stack_dir)
    rvecs = {}
    for bracket_id in bracket_ids:
        rvec = load_prior_rvec(stack_dir, bracket_id)
        if rvec is not None:
            rvecs[bracket_id] = rvec
        else:
            warnings.warn(f"rvec not found for bracket {bracket_id} in {stack_dir} (<id>.json or <id>_rvec.npy)")
    return rvecs
