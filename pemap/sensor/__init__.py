"""Sensor recovery and a prepared camera-view record."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PanoramaView:
    stokes: np.ndarray
    saturation: np.ndarray | None = None
