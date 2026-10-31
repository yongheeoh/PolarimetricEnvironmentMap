"""Rendering API for the compiled polarized Mitsuba/penvmap runtime."""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import polanalyser as pa

_RUNTIME = None
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("cuda_ad_rgb_polarized", "llvm_ad_rgb_polarized")


def load_renderer(variant="cuda_ad_rgb_polarized"):
    """Bootstrap the local build and retain DLL handles across notebook cells."""
    global _RUNTIME
    if variant not in VARIANTS:
        raise ValueError(f"Choose a polarized RGB variant: {VARIANTS}")
    if _RUNTIME is None:
        path = Path(__file__).with_name("sitecustomize.py")
        spec = importlib.util.spec_from_file_location("penvmap_runtime", path)
        _RUNTIME = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RUNTIME)
    try:
        import mitsuba as mi
    except ImportError as error:
        raise RuntimeError("Use Python matching the custom Mitsuba build, or rebuild with compile_mitsuba3.py") from error
    mi.set_variant(variant)
    return mi


def rearrange_rendered_to_bgrstokes(img_rendered) -> np.ndarray:
    """Rearrange a Mitsuba3's rendered image with Stokes integrator ({"type": "stokes", "nested": {"type": "path"}}) to bgr-stokes ndarray (H, W, 3, 4)."""
    img_rendered = np.array(img_rendered)
    height, width, channels = img_rendered.shape

    if channels == 15:  # rgb_polarized or spectral_polarized
        j = 3  # Skip the first 3 channels (RGB)
    elif channels == 13:  # mono_polarized
        j = 1  # Skip the first channel (Luminance)
    else:
        raise ValueError(f"Unexpected number of channels in the rendered image: {channels}. Expected 13 or 15 for polarized variants.")

    img_bgr_stokes = np.zeros((height, width, 3, 4), dtype=img_rendered.dtype)
    for i in range(4):
        img_bgr_stokes[:, :, :, i] = img_rendered[:, :, j : j + 3][:, :, ::-1]
        j += 3
    return img_bgr_stokes


def load_scene_factory(path=None):
    path = Path(path) if path is not None else ROOT / "example/scene.py"
    spec = importlib.util.spec_from_file_location("penvmap_render_scene", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.make_scene


def render_scene(scene_dict, output=None, *, spp=16, seed=0, variant="cuda_ad_rgb_polarized"):
    """Render a Mitsuba scene dictionary and return linear (H,W,3,4) BGR Stokes."""
    if spp < 1:
        raise ValueError("spp must be positive")
    mi = load_renderer(variant)
    rendered = mi.render(mi.load_dict(scene_dict), spp=spp, seed=seed)
    stokes = rearrange_rendered_to_bgrstokes(rendered).astype(np.float32, copy=False)
    if not np.isfinite(stokes).all():
        raise ValueError("Renderer returned nonfinite Stokes")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pa.imwrite_stokes(str(output), stokes)
    return stokes


def render_envmap(envmap, output=None, *, scene_factory=None, variant="cuda_ad_rgb_polarized",
                  spp=16, size=256, scale=2.0, yaw_deg=0.0, seed=0):
    """Render with a configurable scene factory; envmap must contain Duff S0..S3 layers."""
    envmap = Path(envmap).resolve()
    if not envmap.is_file():
        raise FileNotFoundError(envmap)
    if size < 1:
        raise ValueError("size must be positive")
    mi = load_renderer(variant)
    factory = scene_factory or load_scene_factory()
    scene_dict = factory(mi, envmap, size=size, scale=scale, yaw_deg=yaw_deg)
    return render_scene(scene_dict, output, spp=spp, seed=seed, variant=variant)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render with the custom penvmap emitter and Duff RGB-Stokes EXR.")
    parser.add_argument("envmap", type=Path)
    parser.add_argument("--scene", type=Path, default=ROOT / "example/scene.py")
    parser.add_argument("--output", type=Path, default=Path("rendered.exr"))
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    stokes = render_envmap(args.envmap, args.output, scene_factory=load_scene_factory(args.scene),
                           variant=args.variant, spp=args.spp, size=args.size, scale=args.scale,
                           yaw_deg=args.yaw_deg, seed=args.seed)
    print(f"Rendered {stokes.shape[1]} x {stokes.shape[0]} RGB-Stokes -> {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
