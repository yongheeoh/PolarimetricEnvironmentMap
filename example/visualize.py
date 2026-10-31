"""Notebook figures: display-only sRGB S0 and signed normalized Stokes in RdBu."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import polanalyser as pa
from polanalyser.vis import colorize

CHANNELS = {"B": 0, "G": 1, "R": 2}


def read_stokes(path):
    return np.asarray(pa.imread_stokes(str(path)), np.float32)


def s0_rgb(stokes, exposure=None):
    """Tone-map only the preview; the linear Stokes data is not modified."""
    bgr = np.maximum(stokes[..., 0], 0)
    good = np.isfinite(bgr[..., 1]) & (bgr[..., 1] > 1e-8)
    if exposure is None:
        exposure = .18 / np.median(bgr[..., 1][good]) if good.any() else 1.
    rgb = np.nan_to_num(bgr[..., ::-1] * exposure, nan=0, posinf=1, neginf=0)
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)


def normalized_stokes(stokes, channel="G", eps=1e-8):
    """Return Si/S0 for one B/G/R channel, before exposure/gamma; dark pixels are NaN."""
    data = stokes[..., CHANNELS[channel.upper()], :]
    denominator = data[..., 0:1]
    valid = np.isfinite(data).all(axis=-1, keepdims=True) & (denominator > eps)
    return np.divide(data[..., 1:4], denominator,
                     out=np.full(data[..., 1:4].shape, np.nan, np.float32), where=valid)


def plot_stokes(stokes, *, channel="G", exposure=None, limit=1.0, title=None):
    """S0 RGB plus S1n/S2n/S3n. All signed panels use the same fixed RdBu scale."""
    normalized = normalized_stokes(stokes, channel)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    axes[0].imshow(s0_rgb(stokes, exposure))
    axes[0].set_title("S0 · RGB")
    for index, ax in enumerate(axes[1:]):
        image = colorize(normalized[..., index], cmap="RdBu", vmin=-limit, vmax=limit,
                         nan_color=(80, 80, 80), color_order="rgb")
        ax.imshow(image)
        ax.set_title(f"{channel.upper()} · S{index + 1}/S0")
    for ax in axes:
        ax.set_axis_off()
    mapper = plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu")
    figure.colorbar(mapper, ax=list(axes[1:]), shrink=.65, label="Normalized Stokes")
    if title:
        figure.suptitle(title)
    return figure
