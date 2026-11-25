"""Run with the same Python environment as make_previews.py."""
from pathlib import Path

import numpy as np
import polanalyser as pa
from matplotlib import colormaps
from PIL import Image

from make_previews import red_degree


def main():
    out = Path(__file__).resolve().parents[1] / "assets" / "legends"
    out.mkdir(exist_ok=True)
    x = np.linspace(0, 1, 256, dtype=np.float32)[None, :]
    strength = np.ones_like(x)
    # AoLP is periodic: 180 degrees maps to the same hue as zero.
    angle = x * np.pi
    maps = {
        "degree": red_degree(x),
        "signed": colormaps["RdBu"](x, bytes=True)[..., :3],
        "angle": pa.applyColorToAoLP(angle)[..., ::-1],
        "cop-angle": pa.applyColorToCoP((x - 0.5) * np.pi / 2, strength)[..., ::-1],
        "top-angle": pa.applyColorToToP(x * np.pi / 4, strength)[..., ::-1],
    }
    for name, rgb in maps.items():
        Image.fromarray(rgb).save(out / (name + ".png"))


if __name__ == "__main__":
    main()
