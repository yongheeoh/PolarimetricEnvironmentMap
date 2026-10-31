from pathlib import Path

import numpy as np
import numpy.typing as npt

S2L2_PARAMS = tuple(f"S2L2{i:02d}" for i in range(12))
_COLOR_ORDER = ("B", "G", "R")
_FILE_COLOR_ORDER = ("R", "G", "B")


def imwrite_s2l2(filename: str | Path, img_bgr_s2l2: npt.NDArray[np.float32]) -> None:
    """Save a (H, W, 3, 12) S2L2 tensor to an OpenEXR file.

    Parameters
    ----------
    filename : str or Path
        Output EXR filepath. Parent directories are created if absent.
    img_bgr_s2l2 : ndarray of shape (H, W, 3, 12), float32
        S2L2 environment map. Axis 2 is the BGR color channel (B=0, G=1, R=2).
        Axis 3 is the S2L2 coefficient index (0..11).

    Raises
    ------
    ValueError
        If ``img_bgr_s2l2`` does not have shape ``(H, W, 3, 12)``.
    RuntimeError
        If OpenImageIO fails to open the output file.

    Notes
    -----
    Writes 39 channels total:

    * ``R``, ``G``, ``B`` — viewer-compatible preview from coefficient ``S2L20``.
    * ``S2L20.R`` … ``S2L211.B`` — full 12 × 3 S2L2 coefficients.

    Channel color order in the file is RGB; in-memory axis 2 is BGR.
    Compression is ``piz`` (lossless wavelet, efficient for HDR data).
    """
    if img_bgr_s2l2.ndim != 4 or img_bgr_s2l2.shape[2:] != (3, 12):
        raise ValueError(f"img_bgr_s2l2 must have shape (H, W, 3, 12), got {img_bgr_s2l2.shape}")

    import OpenImageIO as oiio

    height, width = img_bgr_s2l2.shape[:2]
    channel_names: list[str] = []
    channel_planes: list[npt.NDArray[np.float32]] = []

    for color in _FILE_COLOR_ORDER:
        color_idx = _COLOR_ORDER.index(color)
        channel_names.append(color)
        channel_planes.append(img_bgr_s2l2[..., color_idx, 0])

    for param_idx, param_id in enumerate(S2L2_PARAMS):
        for color in _FILE_COLOR_ORDER:
            color_idx = _COLOR_ORDER.index(color)
            channel_names.append(f"{param_id}.{color}")
            channel_planes.append(img_bgr_s2l2[..., color_idx, param_idx])

    pixels = np.ascontiguousarray(np.stack(channel_planes, axis=-1).astype(np.float32, copy=False))

    filename = str(filename)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    out = oiio.ImageOutput.create(filename)
    if out is None:
        raise RuntimeError(f"Failed to create ImageOutput for {filename}")

    try:
        spec = oiio.ImageSpec(width, height, len(channel_names), oiio.FLOAT)
        spec.channelnames = tuple(channel_names)
        spec["compression"] = "piz"

        if not out.open(filename, spec):
            raise RuntimeError(f"Failed to open {filename} for writing")

        out.write_image(pixels)
    finally:
        out.close()


def imread_s2l2(filename: str | Path) -> npt.NDArray[np.float32]:
    """Load an OpenEXR file written by :func:`imwrite_s2l2` into a (H, W, 3, 12) array.

    Parameters
    ----------
    filename : str or Path
        Path to an EXR file whose channels include ``S2L2{i}.{R,G,B}``
        for ``i`` in ``0..11``.

    Returns
    -------
    ndarray of shape (H, W, 3, 12), float32
        S2L2 tensor. Axis 2 is BGR (B=0, G=1, R=2); axis 3 is coefficient index.

    Raises
    ------
    RuntimeError
        If OpenImageIO cannot read pixels from ``filename``.
    KeyError
        If any of the 36 expected S2L2 channels are absent in the file.
    """
    import OpenImageIO as oiio

    buf = oiio.ImageBuf(str(filename))
    spec = buf.spec()
    pixels = buf.get_pixels(oiio.FLOAT)

    if pixels is None:
        raise RuntimeError(f"Failed to read pixels from {filename}")

    img_bgr_s2l2 = np.zeros((spec.height, spec.width, 3, 12), dtype=np.float32)

    param_lookup = {name: idx for idx, name in enumerate(S2L2_PARAMS)}
    color_lookup = {name: idx for idx, name in enumerate(_COLOR_ORDER)}

    expected = {f"{p}.{c}" for p in S2L2_PARAMS for c in _COLOR_ORDER}
    found: set[str] = set()

    for ch_idx, channel in enumerate(spec.channelnames):
        if "." not in channel:
            continue
        param_id, component = channel.split(".", 1)
        if param_id in param_lookup and component in color_lookup:
            img_bgr_s2l2[..., color_lookup[component], param_lookup[param_id]] = pixels[..., ch_idx]
            found.add(f"{param_id}.{component}")

    missing = expected - found
    if missing:
        raise KeyError(f"Missing S2L2 channels in {filename}: {sorted(missing)}")

    return img_bgr_s2l2
