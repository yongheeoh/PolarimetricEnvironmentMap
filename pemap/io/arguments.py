"""CLI overrides for processing.yaml."""

import argparse
from pathlib import Path
import math
import sys
from pemap.io.configuration import DEFAULT_CONFIG, load_configuration


def parse_scene_args(argv=None, *, default_scene=None, description=None, default_config=DEFAULT_CONFIG):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=default_config)
    selected, _ = pre.parse_known_args(argv)
    defaults = load_configuration(selected.config, default_config=default_config)
    if default_scene is not None:
        defaults["scene"] = Path(default_scene)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path)
    parser.add_argument("scene", nargs="?", type=Path)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--processing-profile", type=Path)
    parser.add_argument("--camera", choices=["auto", "flir", "basler", "fullstokes"])
    parser.add_argument("--pcalib", type=Path)
    parser.add_argument("--mode", choices=["full", "s0"])
    parser.add_argument(
        "--stokes-repr", "--stokes-repr-mode", dest="stokes_repr_mode", choices=["s2l2", "duff", "thetaphi"]
    )
    parser.add_argument("--rotation-model", choices=["min_rot", "mitsuba3", "none"])
    parser.add_argument("--width", type=int)
    parser.add_argument("--envmap-size", type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument("--level", "--blend-level", dest="level", type=int, choices=range(1, 9))
    parser.add_argument("--blend-method", choices=["multiband"])
    parser.add_argument("--domain-rotation-preset", choices=["octahedral", "none"])
    parser.add_argument("--resize-scale", type=float)
    parser.add_argument("--pitches", type=int, nargs="+")
    parser.add_argument("--camera-rotation-matcher", dest="camera_matchers", nargs="+")
    parser.add_argument("--vignetting", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vignetting-alpha", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--apply-wb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--color-source", choices=["auto", "profile"])
    parser.add_argument("--apply-ccm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flir-angle-offset-deg", type=float)
    parser.add_argument("--flir-flip-s2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--profile", action="append", default=[], metavar="KEY=PATH")
    for flag in ("intrinsic-path", "wb-ccm-path", "vignetting-path", "mask-dir"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--refit-color", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--stage-poses", "--no-optimize-camera-rotations", dest="stage_poses", action="store_true"
    )
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--optimize-camera-rotations", dest="stage_poses", action="store_false")
    parser.add_argument("--floor", dest="no_floor", action="store_false")
    parser.add_argument("--floor-photometry", choices=["smooth", "scalar"])
    parser.add_argument("--floor-transition", type=float)
    parser.add_argument("--no-sun", action="store_true")
    parser.add_argument("--sun", dest="no_sun", action="store_false")
    parser.add_argument("--sun-radius", type=float)
    parser.add_argument("--sun-mask-mode", choices=["adaptive", "fixed"])
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction)
    parser.add_argument("--keep-work", action=argparse.BooleanOptionalAction)
    parser.add_argument("--minimal-analysis", action=argparse.BooleanOptionalAction)
    parser.add_argument("--check", action="store_true")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.scene is None:
        parser.error("Provide a scene directory or set input.scene in processing.yaml")
    if args.camera == "fullstokes":
        args.camera = "basler"
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices and value not in action.choices:
            parser.error(f"Invalid {action.dest}: {value}")
    if any(value == "--width" or value.startswith("--width=") for value in argv) and not any(
        value == "--envmap-size" or value.startswith("--envmap-size=") for value in argv
    ):
        args.envmap_size = None
    if args.envmap_size is not None:
        args.width, args.height = args.envmap_size
    else:
        args.width = args.width or (1024 if args.mode == "s0" else 2048)
        args.height = args.width // 2
    if args.width < 256 or args.width % 2 or args.height < 2:
        parser.error("Invalid envmap size")
    if not 0 < args.resize_scale <= 1:
        parser.error("--resize-scale must be in (0,1]")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    for key in ("sun_radius", "floor_transition", "vignetting_alpha"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"Invalid {key}")
    if args.sun_radius == 0 or args.floor_transition == 0:
        parser.error("Sun radius and Floor transition must be positive")
    if args.rotation_model == "none":
        args.stokes_repr_mode = "none"
    args.profiles = {}
    for key, name in [
        ("intrinsic", "intrinsic_path"),
        ("wb_cc", "wb_ccm_path"),
        ("vignetting", "vignetting_path"),
        ("mask", "mask_dir"),
    ]:
        value = getattr(args, name)
        if value is not None:
            args.profiles[key] = str(value)
    for item in args.profile:
        if "=" not in item:
            parser.error("--profile requires KEY=PATH")
        key, value = item.split("=", 1)
        args.profiles[key] = value
    return args
