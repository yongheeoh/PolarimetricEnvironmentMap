import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BracketResult:
    """One exposure bracket.

    frames      : (T, H, W) raw frames in bracket order
    exposure_us : exposure actually used per frame (camera readback when available, else requested)
    info        : capture details for the metadata (mode, frame ids, retries, ...)
    """

    frames: np.ndarray
    exposure_us: list[float]
    info: dict = field(default_factory=dict)


class Camera(ABC):
    """Abstract base class for polarimetric camera wrappers.

    Subclasses must implement: read, set, get, release, grab.
    Common parameters are stored here so subclasses call super().__init__().
    ``capture_bracket`` has a generic implementation (set exposure → wait → discard → read);
    cameras with hardware bracketing (FLIR sequencer) override it.
    """

    def __init__(
        self,
        camera_index: int = 0,
        exposure_us: float = 1e5,
        gain: float = 0.0,
        pixel_format: str = "BayerRG12",
        timeout_ms: int = 10000000,
    ):
        self.camera_index = camera_index
        self.pixel_format = pixel_format
        self.timeout_ms = timeout_ms
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)
        self._is_open = False

    @abstractmethod
    def read(self) -> tuple:
        """Capture one frame. Returns (ret: bool, img: np.ndarray | None)."""

    @abstractmethod
    def set(self, prop_id: int, value: float) -> bool:
        """Set a camera property (cv2.CAP_PROP_* style)."""

    @abstractmethod
    def get(self, prop_id: int):
        """Get a camera property (cv2.CAP_PROP_* style)."""

    @abstractmethod
    def release(self) -> None:
        """Stop grabbing and close the camera."""

    @abstractmethod
    def grab(self) -> bool:
        """Trigger one grab without retrieving the image."""

    def exposure_range_us(self) -> tuple[float, float] | None:
        """(min, max) exposure the camera accepts, or None if unknown."""
        return None

    def capture_bracket(
        self, exposures_us: list[float], settle_s: float = 0.2, settle_frames: int = 2
    ) -> BracketResult:
        """Generic bracket: for each exposure set it, wait settle_s, discard settle_frames frames, read one.

        Discarding is needed because a continuously streaming camera may still deliver frames
        exposed with the previous setting.
        """
        frames = []
        measured = []
        for exp_us in exposures_us:
            if not self.set(cv2.CAP_PROP_EXPOSURE, exp_us):
                raise RuntimeError(f"Camera rejected exposure {exp_us} us")
            time.sleep(settle_s)
            for _ in range(settle_frames):
                if not self.grab():
                    raise RuntimeError(f"Camera settling grab failed at exposure {exp_us} us")
            ret, img = self.read()
            if not ret or img is None:
                raise RuntimeError(f"camera.read() failed at exposure {exp_us} us")
            actual = self.get(cv2.CAP_PROP_EXPOSURE)
            if actual is None or not np.isfinite(actual) or actual <= 0:
                raise RuntimeError("Could not read back the camera exposure")
            measured.append(float(actual))
            frames.append(np.array(img, copy=True))
        return BracketResult(
            frames=np.stack(frames, axis=0),
            exposure_us=measured,
            info={
                "mode": "sequential",
                "settle_s": settle_s,
                "settle_frames": settle_frames,
                "exposure_source": "camera_readback",
            },
        )
