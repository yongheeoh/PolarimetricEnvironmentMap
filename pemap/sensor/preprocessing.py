"""Preprocessing utilities for HDR fusion and full-stokes recovery."""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Tuple
import warnings

import cv2
import numpy as np
import numpy.typing as npt
import polanalyser as pa
from tqdm.auto import tqdm

from pemap.sensor.flir import FLIR_Z_MIN, recover_bgr_stokes_flir
from pemap.sensor.full_stokes_cam.imgproc import recover_bgr_stokes
from pemap.sensor.hdr import fuse_exposure_bracket
from pemap.io.metadata import detect_camera, list_bracket_ids, load_exposure_times

HDR_Z_MIN = 0.05  # full-Stokes (Basler, 8 bit) HDR lower bound; FLIR uses FLIR_Z_MIN


def default_z_min(camera: str) -> float:
    """HDR fusion lower bound (fraction of full scale) for a camera model ("flir" | "fullstokes")."""
    return FLIR_Z_MIN if camera == "flir" else HDR_Z_MIN


def _bgr_stokes_cache_dir(root: Path) -> Path:
    """Return the cache directory for fused BGR-Stokes EXRs."""
    if root.name == "stack":
        return root.parent / "cache"
    return root / "cache"


def _bgr_stokes_cache_path(root: Path, bracket_id: str) -> Path:
    """Return the cache EXR path for one bracket id."""
    return _bgr_stokes_cache_dir(root) / f"{bracket_id}.exr"


def _load_cached_bgr_stokes(job: Tuple[str, Path]) -> Tuple[str, np.ndarray]:
    """Load one cached BGR-Stokes EXR."""
    stem, cache_path = job
    img_bgr_stokes = pa.imread_stokes(str(cache_path))
    return stem, np.asarray(img_bgr_stokes, dtype=np.float32)


def _write_cached_bgr_stokes(job: Tuple[str, Path, np.ndarray]) -> str:
    """Write one BGR-Stokes EXR cache file."""
    stem, cache_path, img_bgr_stokes = job
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pa.imwrite_stokes(str(cache_path), np.asarray(img_bgr_stokes, dtype=np.float32))
    return stem


def _fuse_one_bracket(
    job: Tuple[Path, str, float, float, npt.DTypeLike],
) -> Tuple[str, np.ndarray]:
    """Load one bracket stack and fuse it into an HDR map."""
    root, stem, z_min, z_max, dtype = job
    img_stack_path = root / f"{stem}.npy"

    if not img_stack_path.exists():
        raise FileNotFoundError(f"Missing bracket image stack: {img_stack_path}")

    img_stack = np.load(img_stack_path)
    if img_stack.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Expected uint8 (Basler) or uint16 (FLIR BayerRG16) stack, got {img_stack.dtype} at {img_stack_path}")
    full_scale = float(np.iinfo(img_stack.dtype).max)  # 255 or 65535 (FLIR 12-bit data is MSB-aligned in 16 bits)

    # Two-step conversion avoids a second full-size copy that `astype(...) / full_scale` would produce.
    img_stack = img_stack.astype(dtype)
    img_stack /= full_scale
    times = load_exposure_times(root, stem)

    img_hdr = fuse_exposure_bracket(img_stack, times, z_min=z_min, z_max=z_max, dtype=dtype)
    return stem, img_hdr.astype(np.float32, copy=False)


def fuse_exposure_brackets_from_path(
    input_path: str | Path,
    *,
    z_min: float = 0.05,
    z_max: float = 0.95,
    dtype: npt.DTypeLike = np.float32,
    max_workers: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load bracket stacks from a folder and fuse all of them in parallel.

    Expected file layout in ``input_path``:
    - ``<id>.npy``: image stack with shape (N, H, W) or (N, H, W, C)
    - ``<id>.json`` (``exposure_times_s``) or legacy ``<id>_times.npy``: exposure times with shape (N,)

    Returns
    -------
    dict[str, ndarray]
            Mapping from bracket id to fused HDR image as float32.
    """
    root = Path(input_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    bracket_ids = list_bracket_ids(root)
    if not bracket_ids:
        raise FileNotFoundError(f"No bracket metadata ('<id>.json' or '<id>_times.npy') found in: {root}")

    jobs = [(root, bracket_id, z_min, z_max, dtype) for bracket_id in bracket_ids]

    img_hdr_by_id: Dict[str, np.ndarray] = {}
    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stem, img_hdr in tqdm(
            executor.map(_fuse_one_bracket, jobs),
            total=len(jobs),
            desc="HDR fusion",
            unit="bracket",
        ):
            img_hdr_by_id[stem] = img_hdr

    elapsed = perf_counter() - t0
    n_jobs = len(jobs)
    tqdm.write(f"[HDR fusion] {n_jobs} brackets in {elapsed:.2f}s " f"({n_jobs / elapsed:.2f} brackets/s)")

    return img_hdr_by_id


def _fuse_and_recover_one_bracket(
    job: Tuple[Path, str, float, float, npt.DTypeLike, Callable[[np.ndarray], np.ndarray], np.ndarray | None, float, float],
) -> Tuple[str, np.ndarray]:
    """Fuse one bracket stack (raw mosaic HDR) and recover BGR Stokes with the camera's recovery function."""
    root, bracket_id, z_min, z_max, dtype, recover, vignetting_corr, vignetting_alpha, scale = job
    stem, img_hdr = _fuse_one_bracket((root, bracket_id, z_min, z_max, dtype))
    img_bgr_stokes = recover(img_hdr)
    if vignetting_corr is not None:
        effective_corr = vignetting_alpha * (vignetting_corr - 1.0) + 1.0
        img_bgr_stokes = img_bgr_stokes * effective_corr[..., np.newaxis]
    return stem, (img_bgr_stokes * scale).astype(np.float32, copy=False)


def fuse_bgr_stokes_from_path(
    input_path: str | Path,
    *,
    z_min: float | None = None,
    z_max: float = 0.95,
    dtype: npt.DTypeLike = np.float32,
    pcalib_tensor: npt.ArrayLike | None = None,
    vignetting_path: Path | None = None,
    vignetting_alpha: float = 1.0,
    scale: float = 1.0,
    cache: bool = False,
    max_workers: int | None = None,
    camera: str = "auto",
    flir_angle_offset_deg: float = 0.0,
    flir_flip_s2: bool = False,
) -> dict[str, npt.NDArray[np.float32]]:
    """Fuse exposure brackets and recover BGR Stokes maps in parallel.

    Expected file layout in ``input_path``:
    - ``<id>.npy``: image stack with shape (N, H, W), uint8 (Basler) or uint16 (FLIR BayerRG16)
    - ``<id>.json`` (``exposure_times_s``) or legacy ``<id>_times.npy``: exposure times with shape (N,)

    ``camera``: "fullstokes" (Basler 4x4 micro-retarder, pcalib → S0..S3), "flir" (color polarization
    0/45/90/135 → S0..S2, S3 = 0, see pemap.flir), or "auto" (``"camera"`` key of the bracket json:
    flir → flir, otherwise / legacy data → fullstokes). Both fuse the raw mosaic to HDR first.
    ``z_min`` None → ``default_z_min(camera)`` (0.05 full-Stokes, 0.01 FLIR).

    When ``cache=True``, this function trusts any existing cached EXRs and skips
    preprocessing for those bracket ids. Cached EXRs are stored under the sibling
    ``cache`` directory for the conventional ``.../hdr/stack`` layout, i.e.
    ``.../hdr/cache/<id>.exr``.

    Returns
    -------
    dict[str, ndarray]
            Mapping from bracket id to BGR-Stokes array with shape (H, W, 3, 4).
    """
    root = Path(input_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    ordered_bracket_ids = list_bracket_ids(root)
    if not ordered_bracket_ids:
        raise FileNotFoundError(f"No bracket metadata ('<id>.json' or '<id>_times.npy') found in: {root}")

    camera = detect_camera(root) if camera == "auto" else camera
    if z_min is None:
        z_min = default_z_min(camera)
    if camera == "flir":
        recover = partial(recover_bgr_stokes_flir, angle_offset_deg=flir_angle_offset_deg, flip_s2=flir_flip_s2)
    elif camera == "fullstokes":
        recover = partial(recover_bgr_stokes, pcalib_tensor=None if pcalib_tensor is None else np.asarray(pcalib_tensor))
    else:
        raise ValueError(f"camera must be 'auto', 'fullstokes' or 'flir', got {camera!r}")
    tqdm.write(f"[HDR->BGR Stokes] camera={camera}, z_min={z_min:g}" + (f" (angle offset {flir_angle_offset_deg:g} deg, flip S2 {flir_flip_s2})" if camera == "flir" else ""))
    vignetting_corr: np.ndarray | None = None
    if vignetting_path is not None:
        corr = cv2.imread(str(vignetting_path), cv2.IMREAD_UNCHANGED)
        if corr is None:
            warnings.warn(f"Failed to load vignetting correction map: {vignetting_path}. Proceeding without vignetting correction.")
            corr = np.ones((1), dtype=np.float32)
        else:
            corr = corr.astype(np.float32)
            if corr.ndim == 2:
                corr = corr[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        vignetting_corr = corr  # (H, W, C) or (H, W, 1)

    cache_load_jobs: list[tuple[str, Path]] = []
    compute_jobs: list[tuple] = []
    for bracket_id in ordered_bracket_ids:
        cache_path = _bgr_stokes_cache_path(root, bracket_id)
        if cache and cache_path.exists():
            cache_load_jobs.append((bracket_id, cache_path))
        else:
            compute_jobs.append((root, bracket_id, z_min, z_max, dtype, recover, vignetting_corr, vignetting_alpha, scale))

    img_bgr_stokes_by_id: Dict[str, np.ndarray] = {}
    computed_bgr_stokes_by_id: Dict[str, np.ndarray] = {}
    t0 = perf_counter()
    if cache_load_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for stem, img_bgr_stokes in tqdm(
                executor.map(_load_cached_bgr_stokes, cache_load_jobs),
                total=len(cache_load_jobs),
                desc="BGR Stokes cache",
                unit="bracket",
            ):
                img_bgr_stokes_by_id[stem] = img_bgr_stokes

    if compute_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for stem, img_bgr_stokes in tqdm(
                executor.map(_fuse_and_recover_one_bracket, compute_jobs),
                total=len(compute_jobs),
                desc="HDR->BGR Stokes",
                unit="bracket",
            ):
                img_bgr_stokes_by_id[stem] = img_bgr_stokes
                computed_bgr_stokes_by_id[stem] = img_bgr_stokes

    if cache and computed_bgr_stokes_by_id:
        cache_write_jobs = [(bracket_id, _bgr_stokes_cache_path(root, bracket_id), computed_bgr_stokes_by_id[bracket_id]) for bracket_id in ordered_bracket_ids if bracket_id in computed_bgr_stokes_by_id]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in tqdm(
                executor.map(_write_cached_bgr_stokes, cache_write_jobs),
                total=len(cache_write_jobs),
                desc="BGR Stokes cache write",
                unit="bracket",
            ):
                pass

    elapsed = perf_counter() - t0
    n_jobs = len(ordered_bracket_ids)
    rate = n_jobs / elapsed if elapsed > 0 else float("inf")
    if cache:
        tqdm.write(f"[HDR->BGR Stokes] {n_jobs} brackets in {elapsed:.2f}s " f"({rate:.2f} brackets/s, cache hits={len(cache_load_jobs)}, computed={len(compute_jobs)})")
    else:
        tqdm.write(f"[HDR->BGR Stokes] {n_jobs} brackets in {elapsed:.2f}s " f"({rate:.2f} brackets/s)")

    return {bracket_id: img_bgr_stokes_by_id[bracket_id] for bracket_id in ordered_bracket_ids}
