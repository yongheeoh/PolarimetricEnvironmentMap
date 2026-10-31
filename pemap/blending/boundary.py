"""Extend masked values for pyramid convolution without creating coverage."""

import cv2
import numpy as np


def extend_for_pyramids(images, pad=96):
    """Nearest valid values for pyramid convolution only; validity weights stay zero.

    Treat ERP longitude as periodic. The default margin exceeds the five-level
    pyramid support. No extrapolated pixel becomes an observed panorama pixel.
    """
    result = np.asarray(images, np.float32).copy()
    for i, img in enumerate(result):
        valid = np.isfinite(img).all(-1)
        if valid.all():
            continue
        if not valid.any():
            result[i] = 0
            continue
        margin = min(int(pad), img.shape[1])
        values = np.concatenate([img[:, -margin:], img, img[:, :margin]], axis=1)
        good = np.concatenate([valid[:, -margin:], valid, valid[:, :margin]], axis=1)
        _, labels = cv2.distanceTransformWithLabels((~good).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(labels.max()) + 1, 3), np.float32)
        lookup[labels[good]] = values[good]
        core_labels = labels[:, margin:-margin]
        result[i][~valid] = lookup[core_labels[~valid]]
    assert np.isfinite(result).all()
    return result
