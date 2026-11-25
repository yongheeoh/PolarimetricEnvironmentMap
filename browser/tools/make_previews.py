# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["OpenImageIO==3.1.10.0", "numpy==2.3.4", "Pillow==11.3.0", "polanalyser==3.0.0", "opencv-python==4.13.0.92", "matplotlib==3.10.8"]
# ///
"""Generate full equirectangular JPEGs without rotating or cropping the EXR.

Example: uv run browser/tools/make_previews.py 3.exr browser/assets/full-003
Source: Oh, Maeda, Moon and Baek, SIGGRAPH Asia 2026 (CC BY 4.0).
"""
import argparse
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import polanalyser as pa
import OpenImageIO as oiio
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_stokes(path, full_stokes):
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ValueError(f"Cannot open {path}: {oiio.geterror()}")
    try:
        names = list(source.spec().channelnames)
        planes = source.read_image(format=oiio.FLOAT)
        if planes is None:
            raise ValueError(f"Cannot decode {path}: {source.geterror()}")
        return [planes[..., [names.index(f"S{i}.{c}") for c in "RGB"]] for i in range(4 if full_stokes else 3)]
    finally:
        source.close()


def preview(rgb):
    rgb = np.maximum(np.nan_to_num(rgb, nan=0, neginf=0, posinf=0), 0)
    luminance = rgb @ LUMA
    valid = luminance[luminance > 0]
    if not valid.size:
        raise ValueError("S0 has no positive luminance")
    # Set the median linear luminance to 0.32. Apply a Reinhard curve on luminance
    # to preserve hue, then encode the display values using the sRGB OETF.
    exposure = 0.32 / np.median(valid)
    linear = np.clip(rgb * (exposure / (1 + luminance * exposure))[..., None], 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.rint(srgb * 255).astype(np.uint8))


def red_degree(degree):
    red = np.rint(255 * np.clip(degree, 0, 1) ** (1 / 2.2)).astype(np.uint8)
    rgb = np.zeros((*red.shape, 3), dtype=np.uint8)
    rgb[..., 0] = red
    return rgb


def luminance_stokes(components):
    # Reduce linear RGB before any ratio or angle; signed Stokes values must stay signed.
    stokes = np.stack([plane @ LUMA for plane in components], axis=-1)
    valid = np.isfinite(stokes).all(axis=-1) & (stokes[..., 0] > 0)
    stokes[~valid] = 0
    stokes[..., 0][~valid] = 1
    return stokes, valid


def angle_views(stokes, valid):
    angle = pa.cvtStokesToAoLP(stokes)
    degree = pa.cvtStokesToDoLP(stokes)
    def colorize(saturation, value=1.0):
        bgr = pa.applyColorToAoLP(angle, saturation=saturation, value=value)
        bgr[~valid] = 0
        return Image.fromarray(bgr[..., ::-1])

    angle_only = colorize(1.0)
    angle_only.paste((0, 0, 0), mask=Image.fromarray((stokes[..., 1] == 0) & (stokes[..., 2] == 0)))
    return angle_only, colorize(np.clip(degree, 0, 1)), colorize(1.0, np.clip(degree, 0, 1))


def save_sphere_thumbnail(panorama, prefix):
    # Reflect the viewing direction around each sphere normal, then sample the lat-long map.
    y, x = np.mgrid[:256, :512]
    nx, ny = (x + 0.5 - 256) / 115, (128 - y - 0.5) / 115
    radius_sq = nx**2 + ny**2
    nz = np.sqrt(np.maximum(0, 1 - radius_sq))
    longitude = np.arctan2(2 * nx * nz, 2 * nz**2 - 1)
    latitude = np.arcsin(np.clip(2 * ny * nz, -1, 1))
    u = ((longitude / (2 * np.pi) + 0.5) * panorama.width).astype(int) % panorama.width
    v = np.clip(((0.5 - latitude / np.pi) * panorama.height).astype(int), 0, panorama.height - 1)
    pixels = np.array(panorama.convert("RGB"))[v, u]
    pixels[radius_sq > 1] = (41, 45, 48)
    image = Image.fromarray(pixels).resize((256, 128), Image.Resampling.LANCZOS)
    image.save(prefix.parent / f"{prefix.name}-sphere.webp", quality=88)


def save_preview(image, prefix, suffix, quality):
    path = prefix.parent / f"{prefix.name}-{suffix}.jpg"
    image.save(path, quality=quality, progressive=True, optimize=True)


def component_views(stokes, valid):
    palette = colormaps["RdBu"](np.linspace(0, 1, 256), bytes=True)[:, :3]

    def signed(values, limit):
        indices = np.rint(127.5 * (np.clip(values / limit, -1, 1) + 1)).astype(np.uint8)
        return palette[indices]

    for i in range(1, stokes.shape[-1]):
        si = stokes[..., i]
        yield f"s{i}n", signed(si / stokes[..., 0], 1), valid

    yield "dolp", red_degree(pa.cvtStokesToDoLP(stokes)), valid
    if stokes.shape[-1] == 4:
        dop = np.clip(pa.cvtStokesToDoP(stokes), 0, 1)
        circular = pa.cvtStokesToDoCP(stokes)
        angle = pa.cvtStokesToEllipticityAngle(stokes)
        yield "dop", red_degree(dop), valid
        yield "docp", red_degree(circular), valid
        yield "top", pa.applyColorToToP(angle, dop)[..., ::-1], valid
        yield "cop", pa.applyColorToCoP(angle, np.clip(circular, 0, 1))[..., ::-1], valid


def save_component_views(stokes, valid, prefix):
    for name, rgb, valid in component_views(stokes, valid):
        rgb[~valid] = 0
        image = Image.fromarray(rgb)
        is_degree = name in ("dolp", "dop", "docp")
        for suffix, width in [("", 2048 if name == "dolp" else 1536), ("-thumb", 256)]:
            width = min(image.width, width)
            resized = image.resize((width, width // 2), Image.Resampling.LANCZOS)
            resized.save(prefix.parent / f"{prefix.name}-{name}{suffix}.webp", lossless=is_degree, quality=80, method=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exr", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--full-stokes", action="store_true",
                        help="Generate S3-dependent views for a measured full-Stokes capture")
    args = parser.parse_args()
    components = read_stokes(args.exr, args.full_stokes)
    s0 = components[0]
    h, w = s0.shape[:2]
    if w != 2 * h:
        parser.error("Expected a 2:1 equirectangular panorama")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    intensity = preview(s0)
    save_sphere_thumbnail(intensity, args.output_prefix)
    for suffix, width, quality in [("card", 640, 84), ("detail", 2048, 88)]:
        width = min(width, w)
        save_preview(intensity.resize((width, width // 2), Image.Resampling.LANCZOS),
                     args.output_prefix, suffix, quality)
    save_polarization_views(components, args.output_prefix)


def save_polarization_views(components, prefix):
    stokes, valid = luminance_stokes(components)
    save_component_views(stokes, valid, prefix)
    angle_only, angle, dark_angle = angle_views(stokes, valid)
    for image, suffix, width, quality in (
        (angle, "hover", 640, 88), (angle, "aolp", 2048, 89),
        (dark_angle, "hover-dark", 640, 88), (dark_angle, "aolp-dark", 2048, 89),
        (angle_only, "angle", 2048, 89), (angle_only, "angle-thumb", 256, 84),
    ):
        width = min(width, image.width)
        save_preview(image.resize((width, width // 2), Image.Resampling.LANCZOS),
                     prefix, suffix, quality)


if __name__ == "__main__":
    main()
