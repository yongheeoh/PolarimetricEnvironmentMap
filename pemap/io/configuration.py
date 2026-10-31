"""Read the user-facing processing YAML; CLI flags override these values."""

from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "processing.yaml"
FIELDS = {
    "input": {"scene": "scene", "camera": "camera", "calibration_root": "calibration_root"},
    "output": {"directory": "output_root", "mode": "mode", "envmap_size": "envmap_size"},
    "polarization": {"stokes_repr_mode": "stokes_repr_mode", "rotation_model": "rotation_model"},
    "radiometry": {
        key: key
        for key in (
            "color_source",
            "apply_wb",
            "apply_ccm",
            "scale",
            "z_min",
            "z_max",
            "vignetting",
            "vignetting_alpha",
            "flir_angle_offset_deg",
            "flir_flip_s2",
        )
    },
    "geometry": {
        "resize_scale": "resize_scale",
        "pitches": "pitches",
        "optimize_camera_rotations": "stage_poses",
        "camera_rotation_matchers": "camera_matchers",
    },
    "blending": {
        "method": "blend_method",
        "level": "level",
        "domain_rotation_preset": "domain_rotation_preset",
    },
    "sun": {"enabled": "no_sun", "radius": "sun_radius", "mask_mode": "sun_mask_mode"},
    "floor": {"enabled": "no_floor", "photometry": "floor_photometry", "transition": "floor_transition"},
    "runtime": {key: key for key in ("max_workers", "cache", "keep_work", "minimal_analysis", "refit_color")},
}
PROFILE_KEYS = {"intrinsic", "hdr", "polarimetric", "wb_cc", "nd", "mask", "vignetting"}


def load_configuration(path, *, default_config=DEFAULT_CONFIG):
    def read_file(selected):
        with selected.open(encoding="utf-8-sig") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"Expected a mapping: {selected}")

        def resolve(value):
            return str((selected.parent / Path(value)).resolve()) if value is not None else None

        for section, keys in [
            ("input", ("scene", "calibration_root")),
            ("output", ("directory",)),
            ("profiles", PROFILE_KEYS),
        ]:
            if isinstance(document.get(section), dict):
                for key in keys:
                    if key in document[section]:
                        document[section][key] = resolve(document[section][key])
        # Camera paths travel with the dataset: root relative to scene, files to root.
        return document

    path = Path(path).resolve()

    def merge(base, update):
        result = dict(base)
        for key, value in update.items():
            result[key] = (
                merge(result[key], value)
                if isinstance(value, dict) and isinstance(result.get(key), dict)
                else value
            )
        return result

    default_config = Path(default_config).resolve()
    document = read_file(default_config)
    if path != default_config:
        document = merge(document, read_file(path))
    if set(document) - set(FIELDS) - {"profiles", "cameras", "numerics"}:
        raise ValueError("Unknown processing configuration section")
    values = {}
    for section, fields in FIELDS.items():
        selected = document.get(section, {})
        if not isinstance(selected, dict) or set(selected) - set(fields):
            raise ValueError(f"Unknown setting in {section}")
        for key, value in selected.items():
            if (section, key) in [
                ("geometry", "optimize_camera_rotations"),
                ("sun", "enabled"),
                ("floor", "enabled"),
            ]:
                value = not value
            values[fields[key]] = value

    def relative(value):
        return (path.parent / Path(value)).resolve() if value is not None else None

    for key in ("scene", "output_root", "calibration_root"):
        if values.get(key) is not None:
            values[key] = relative(values[key])

    def profiles(mapping, *, relative_to_config=True):
        if not isinstance(mapping, dict) or set(mapping) - PROFILE_KEYS:
            raise ValueError("Unknown profile key")
        return {
            key: str(relative(value) if relative_to_config else value)
            for key, value in mapping.items()
            if value is not None
        }

    values["profile_defaults"] = profiles(document.get("profiles", {}))
    values["camera_profiles"] = {}
    values["camera_scales"] = {}
    values["camera_pcalib"] = {}
    values["camera_roots"] = {}
    values["scene_profiles"] = {}
    for camera, selected in document.get("cameras", {}).items():
        if camera not in ("basler", "flir") or set(selected) - {
            "profiles",
            "scale",
            "pcalib",
            "calibration_root",
            "scene_profiles",
        }:
            raise ValueError(f"Unknown camera settings: {camera}")
        values["camera_profiles"][camera] = profiles(selected.get("profiles", {}), relative_to_config=False)
        values["camera_scales"][camera] = selected.get("scale")
        values["camera_pcalib"][camera] = Path(selected["pcalib"]) if selected.get("pcalib") else None
        values["camera_roots"][camera] = (
            Path(selected["calibration_root"]) if selected.get("calibration_root") else None
        )
        values["scene_profiles"][camera] = {
            str(sid): profiles(mapping, relative_to_config=False)
            for sid, mapping in selected.get("scene_profiles", {}).items()
        }
    values["numerics"] = {
        key: value for key, value in document.get("numerics", {}).items() if value is not None
    }
    values["config"] = path
    return values
