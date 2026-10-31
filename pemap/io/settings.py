"""Numerical processing choices shared by either camera; no sensor branches."""

from pathlib import Path
from pemap.io.files import read
import yaml

DEFAULTS = {
    "source_rays": "iterative100",
    "rotation_dtype": "float32",
    "intrinsic_dtype": "float64",
    "feature_color": "corrected",
    "mask_boundary": "extend",
    "projection_support": "mask_and_ray_bound",
    "radiometric_order": "scale_then_vignetting",
    "vignetting": False,
}
CHOICES = {
    "source_rays": {"opencv5", "iterative100"},
    "rotation_dtype": {"float32", "float64"},
    "intrinsic_dtype": {"float32", "float64"},
    "feature_color": {"native", "corrected"},
    "mask_boundary": {"zero", "extend"},
    "projection_support": {"mask", "mask_and_ray_bound"},
    "radiometric_order": {"scale_then_vignetting", "vignetting_then_scale"},
    "vignetting": {False, True},
}


def processing_settings(scene, override=None, calibration_root=None, *, overrides=None):
    root = (
        Path(calibration_root)
        if calibration_root
        else next((p / "calibration" for p in Path(scene).parents if (p / "calibration").is_dir()), None)
    )
    candidates = [root / name for name in ("processing_config.yaml", "processing.json")] if root else []
    candidate = Path(override).resolve() if override else next((p for p in candidates if p.is_file()), None)
    if override and not candidate.is_file():
        raise FileNotFoundError(candidate)
    path = candidate.resolve() if candidate and candidate.is_file() else None
    if path and path.suffix.lower() in {".yaml", ".yml"}:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ValueError("Processing config must be a mapping")
        selected = document.get("numerics", {})
    else:
        selected = read(path) if path else {}
    if not isinstance(selected, dict) or set(selected) - set(DEFAULTS):
        raise ValueError("Unknown processing settings")
    if set(overrides or {}) - set(DEFAULTS):
        raise ValueError("Unknown numerical processing override")
    values = {**DEFAULTS, **selected, **(overrides or {})}
    for key, value in values.items():
        if key == "vignetting" and not isinstance(value, bool):
            raise ValueError("vignetting must be a boolean")
        if value not in CHOICES[key]:
            raise ValueError(f"Invalid processing setting {key}: {value}")
    return values, path
