"""Bracket policy, stage pose and display-only preview helpers."""

import math
import cv2
import numpy as np

MAX_BRACKET_STEP_EV = 4.0
SUPERPIXEL = 4


def bracket_step_ev(min_us: float, max_us: float, num_brackets: int) -> float:
    """EV spacing between neighbouring exposures of an evenly spaced bracket (0 for a single frame)."""
    n = int(num_brackets)
    return 0.0 if n <= 1 else math.log2(float(max_us) / float(min_us)) / (n - 1)


def make_bracket(
    min_us: float,
    max_us: float,
    num_brackets: int,
    exposure_range_us: tuple[float, float] | None = None,
) -> tuple[list[float], list[float]]:
    """Bracket policy: ``num_brackets`` exposures from ``min_us`` to ``max_us``, evenly spaced in EV.

    Example: 21 µs … 5.5 s, 7 frames → 21, 169, 1350, 10800, 86400, 691200, 5529600 µs (≈3 EV apart).
    A single frame uses ``max_us``. Exposures outside the camera's ``exposure_range_us`` are rejected
    (not clipped), so the bracket never silently loses range. Steps wider than MAX_BRACKET_STEP_EV
    only warn (neighbouring frames no longer overlap).

    Returns
    -------
    exposure_list : list[float]   단위: μs; sensor minimum may be fractional
    exposure_seconds : list[float]
    """
    n = int(num_brackets)
    if n < 1:
        raise ValueError("num_brackets must be >= 1")
    if not np.isfinite([min_us, max_us]).all() or not 0 < min_us <= max_us:
        raise ValueError(f"need 0 < bracket_min_us <= bracket_max_us, got {min_us}, {max_us}")
    if n == 1:
        exposure_list = [float(max_us)]
    else:
        ratio = float(max_us) / float(min_us)
        exposure_list = [
            float(min(max_us, max(min_us, round(min_us * ratio ** (k / (n - 1))))))
            for k in range(n)
        ]
        # Do not round below a fractional device minimum or change either end.
        exposure_list[0], exposure_list[-1] = float(min_us), float(max_us)
        if any(b <= a for a, b in zip(exposure_list, exposure_list[1:])):
            raise ValueError(
                "Bracket exposures must be strictly increasing; widen the range or reduce the frame count"
            )
    if exposure_range_us is not None:
        lo, hi = exposure_range_us
        bad = [e for e in exposure_list if not lo <= e <= hi]
        if bad:
            raise ValueError(
                f"bracket {exposure_list} us leaves the camera range [{lo:g}, {hi:g}] us: {bad}"
            )
    step = bracket_step_ev(min_us, max_us, n)
    if step > MAX_BRACKET_STEP_EV:
        print(
            f"[BRACKET][WARN] {step:.2f} EV between frames > {MAX_BRACKET_STEP_EV} EV: gaps in the HDR range — add frames"
        )
    exposure_seconds = [e / 1e6 for e in exposure_list]
    return exposure_list, exposure_seconds


def prior_rvec(yaw_deg: float, pitch_deg: float, mount_roll_deg: float) -> np.ndarray:
    """Stage yaw/pitch (deg) → world-to-camera Rodrigues vector used as the stitching prior.

    Stage angles are negated to
    match the equirectangular convention. Kept here so capture does not import mitsuba.
    """
    yaw, pitch, roll = np.deg2rad([-yaw_deg, -pitch_deg, mount_roll_deg])
    R_yaw = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )  # global Y
    R_pitch = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )  # local X
    R_roll = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )  # local Z
    R_cam2world = R_yaw @ R_pitch @ R_roll
    rvec, _ = cv2.Rodrigues(R_cam2world.T)  # OpenCV extrinsic: world → camera
    return rvec


def full_scale(img: np.ndarray, raw_max: float | None = None) -> float:
    """Sensor full-scale value: explicit raw_max, else the dtype maximum (255 / 65535)."""
    if raw_max is not None:
        return float(raw_max)
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    return 1.0


def upright_rotate_code(mount_roll_deg: float) -> int | None:
    """cv2.rotate code that turns a sensor-oriented image upright (sky on top) for this camera mount.

    Derived from the prior rotation: where world-up lands in the image (x right, y down).
    Display only — saved data (hdr/stack, ldr/*) stays in sensor orientation, as pemap expects.
    """
    up = cv2.Rodrigues(prior_rvec(0.0, 0.0, mount_roll_deg))[0] @ np.array([0.0, 1.0, 0.0])
    x, y = up[0], up[1]
    if abs(x) >= abs(y):
        return (
            cv2.ROTATE_90_COUNTERCLOCKWISE if x > 0 else cv2.ROTATE_90_CLOCKWISE
        )  # sky on the right / left
    return cv2.ROTATE_180 if y > 0 else None


def to_display(img: np.ndarray, rotate_code: int | None, max_height: int = 900) -> np.ndarray:
    """Rotate upright and shrink to fit the screen (preview only)."""
    if rotate_code is not None:
        img = cv2.rotate(img, rotate_code)
    scale = min(1.0, max_height / img.shape[0])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def mosaic_s0(raw: np.ndarray, raw_max: float | None = None) -> np.ndarray:
    """RAW mosaic (H, W) → linear S0 preview (H/4, W/4, 3) BGR, full scale = 1. 디모자이크 없이 ~25 ms.

    Both cameras use a 4×4 superpixel made of 2×2 color blocks [[R, G], [G, B]], each block
    holding 4 analyzer pixels (FLIR: 0/45/90/135 polarizers, same layout as polanalyser
    COLOR_PolarRGB; Basler: micro-retarders). Averaging the 4 analyzer pixels of a color gives a
    value proportional to that color's S0 — enough for previews and the live envmap, not for
    measurements (pemap does the real demosaicing from hdr/stack).
    """
    h, w = raw.shape
    h4, w4 = h // SUPERPIXEL, w // SUPERPIXEL
    blocks = (
        raw[: h4 * SUPERPIXEL, : w4 * SUPERPIXEL].astype(np.float32).reshape(h4, 2, 2, w4, 2, 2)
    )
    color = blocks.mean(axis=(2, 5))  # (h4, color_y, w4, color_x)
    bgr = np.stack(
        [color[:, 1, :, 1], 0.5 * (color[:, 0, :, 1] + color[:, 1, :, 0]), color[:, 0, :, 0]],
        axis=-1,
    )
    return bgr / full_scale(raw, raw_max)


def s0_to_u8(s0: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Linear S0 (H, W, 3) → uint8 BGR (99th percentile 정규화 + gamma). 촬영 확인용."""
    ref = max(float(np.percentile(s0, 99)), 1e-8)
    return (np.clip(s0 / ref, 0, 1) ** (1.0 / gamma) * 255).astype(np.uint8)


def preview_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Put readable controls below the preview without obscuring the target."""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    width = max(img.shape[1], max(cv2.getTextSize(s, font, scale, 1)[0][0] for s in lines) + 20)
    height = img.shape[0]
    panel = cv2.copyMakeBorder(
        img,
        0,
        24 * len(lines) + 16,
        0,
        width - img.shape[1],
        cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    for i, s in enumerate(lines):
        cv2.putText(
            panel, s, (10, height + 22 + 24 * i), font, scale, (235, 235, 235), 1, cv2.LINE_AA
        )
    return panel
