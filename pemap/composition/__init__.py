"""Panorama payload and composition results shared by panorama, Sun and Floor."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PanoramaState:
    """World S2L2/S0 plus support masks; 'none' marks unrotated diagnostic Stokes."""

    payload: np.ndarray
    coverage: np.ndarray
    saturation: np.ndarray
    representation: str = "s2l2"


@dataclass(frozen=True)
class CompositionResult:
    panorama: PanoramaState
    report: dict
    warnings: tuple[str, ...] = ()
