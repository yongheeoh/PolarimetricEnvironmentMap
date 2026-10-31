"""Lossless analysis image I/O and mask encoding."""

from pathlib import Path
import os, cv2
import numpy as np


def load_image(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise OSError(f"Cannot read image: {path}")
    return a


def png(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".writing.png")
    if not cv2.imwrite(str(tmp), data):
        raise OSError(f"Cannot write {tmp}")
    np.testing.assert_array_equal(load_image(tmp), data)
    os.replace(tmp, path)


def binary(value):
    return np.asarray(value, np.uint8) * 255
