from pathlib import Path

import cv2
import numpy as np

DEFAULT_PCALIB_PATH = Path(__file__).parent / "data" / "pcalib_tensor.npy"


def demosaic_4x4_linear(img_raw: np.ndarray) -> np.ndarray:
    """Linear interpolation for a 4x4 mosaic (16 distinct filters).

    Parameters
    ----------
    img_raw : ndarray, (H, W)
        Raw image with 4x4 mosaic pattern. H and W must be multiples of 4.

    Returns
    -------
    img_demosaiced : ndarray, (H, W, 4, 4)
        Demosaiced image where the last two dimensions correspond to the 4x4 Bayer tile.
    """
    if img_raw.ndim != 2:
        raise ValueError(f"Expected raw.shape == (H, W), got {img_raw.shape}")

    H, W = img_raw.shape
    if H % 4 != 0 or W % 4 != 0:
        raise ValueError(f"Expected H and W to be multiples of 4, got {(H, W)}")

    img_demosaiced = np.empty((H, W, 4, 4), dtype=img_raw.dtype)

    # Full-resolution pixel coordinates
    yy = np.arange(H, dtype=np.float32)[:, None]  # (H, 1)
    xx = np.arange(W, dtype=np.float32)[None, :]  # (1, W)

    for py in range(4):
        # Map output y -> coarse-grid y for this phase
        ymap = np.broadcast_to((yy - py) / 4.0, (H, W))

        for px in range(4):
            # Extract one of the 16 phase images
            sampled = img_raw[py::4, px::4]  # (H//4, W//4)

            # Map output x -> coarse-grid x for this phase
            xmap = np.broadcast_to((xx - px) / 4.0, (H, W))

            # Bilinear interpolation on the coarse plane
            # BORDER_REPLICATE = simple edge handling
            img_demosaiced[:, :, py, px] = cv2.remap(
                sampled,
                xmap,
                ymap,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return img_demosaiced


def _split(bayer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 4x4 Bayer pattern into R, G, B channels.

    Parameters
    ----------
    bayer : ndarray, (..., 4, 4)
        The last two dimensions must be 4x4 Bayer pattern.

    Returns
    -------
    r : ndarray, (..., 4)
        R channel values at the 4 R positions in the Bayer tile.
    g : ndarray, (..., 8)
        G channel values at the 8 G positions in the Bayer tile.
    b : ndarray, (..., 4)
        B channel values at the 4 B positions in the Bayer tile.

    Note
    ----
    Bayer tile layout (last two dimensions):

    [["R0", "R1", "G0", "G1"],
     ["R2", "R3", "G2", "G3"],
     ["G4", "G5", "B0", "B1"],
     ["G6", "G7", "B2", "B3"]]

    Returned arrangements:

    r = ["R0", "R1", "R2", "R3"]
    g = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    b = ["B0", "B1", "B2", "B3"]
    """
    if bayer.shape[-2:] != (4, 4):
        raise ValueError(f"Expected trailing shape (4, 4), got {bayer.shape}")

    r = np.stack(
        [
            bayer[..., 0, 0],
            bayer[..., 0, 1],
            bayer[..., 1, 0],
            bayer[..., 1, 1],
        ],
        axis=-1,
    )  # (..., 4)
    g = np.stack(
        [
            bayer[..., 0, 2],
            bayer[..., 0, 3],
            bayer[..., 1, 2],
            bayer[..., 1, 3],
            bayer[..., 2, 0],
            bayer[..., 2, 1],
            bayer[..., 3, 0],
            bayer[..., 3, 1],
        ],
        axis=-1,
    )  # (..., 8)
    b = np.stack(
        [
            bayer[..., 2, 2],
            bayer[..., 2, 3],
            bayer[..., 3, 2],
            bayer[..., 3, 3],
        ],
        axis=-1,
    )  # (..., 4)
    return r, g, b


def recover_rgb_stokes(img_raw: np.ndarray, pcalib_tensor: np.ndarray | None = None) -> np.ndarray:
    """Recover full-Stokes parameters for each pixel from the raw image.

    Parameters
    ----------
    img_raw : ndarray, (H, W)
        Raw image with 4x4 mosaic pattern. H and W must be multiples of 4.
    pcalib_tensor : ndarray, (4, 4, 4), optional
        Polarization calibration tensor where ``pcalib_tensor[:, i, j]`` is the
        4x1 analyzer vector for Bayer-tile position ``(i, j)``.
        By default, the function will load the calibration tensor from ``DEFAULT_PCALIB_PATH``.

    Returns
    -------
    img_rgb_stokes : ndarray, (H, W, 3, 4)
        Reconstructed Stokes image in RGB order.
    """
    if pcalib_tensor is None:
        pcalib_tensor = np.load(DEFAULT_PCALIB_PATH)  # (4, 4, 4)
    pcalib_tensor = np.asarray(pcalib_tensor)
    if pcalib_tensor.shape != (4, 4, 4):
        raise ValueError(f"Expected pcalib_tensor.shape == (4, 4, 4), got {pcalib_tensor.shape}")

    img_demosaiced = demosaic_4x4_linear(img_raw.astype(np.float32))  # (H, W, 4, 4)

    imgs_r, imgs_g, imgs_b = _split(img_demosaiced)  # r: (H, W, 4), g: (H, W, 8), b: (H, W, 4)
    sysmat_r, sysmat_g, sysmat_b = _split(pcalib_tensor)  # r: (4, 4), g: (4, 8), b: (4, 4)

    img_stokes_r = imgs_r @ np.linalg.pinv(sysmat_r)  # (H, W, 4) @ (4, 4) -> (H, W, 4)
    img_stokes_g = imgs_g @ np.linalg.pinv(sysmat_g)  # (H, W, 8) @ (8, 4) -> (H, W, 4)
    img_stokes_b = imgs_b @ np.linalg.pinv(sysmat_b)  # (H, W, 4) @ (4, 4) -> (H, W, 4)

    img_rgb_stokes = np.stack([img_stokes_r, img_stokes_g, img_stokes_b], axis=-2)  # (H, W, 3, 4)

    return img_rgb_stokes


def recover_bgr_stokes(img_raw: np.ndarray, pcalib_tensor: np.ndarray | None = None) -> np.ndarray:
    """Recover full-Stokes parameters in BGR channel order."""
    return recover_rgb_stokes(img_raw, pcalib_tensor)[..., ::-1, :]
