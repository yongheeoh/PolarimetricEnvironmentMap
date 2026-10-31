"""Edit this scene to change the object, material, camera and environment lighting."""
from pathlib import Path


def make_scene(mi, envmap, *, size=256, scale=2.0, yaw_deg=0.0):
    """Gold sphere illuminated by a Duff-basis RGB-Stokes environment map.

    scale=2 retains the previous rendering example's explicit light scale.
    Set scale=1 to use the EXR radiance values without an extra multiplier.
    """
    transform = mi.ScalarTransform4f
    return {
        "type": "scene",
        "integrator": {"type": "stokes", "nested": {"type": "path"}},
        "sensor": {
            "type": "perspective",
            "fov": 100,
            "to_world": transform().look_at(
                origin=[0, 0.5, -2], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": size, "height": size, "pixel_format": "rgb"},
        },
        "emitter": {
            "type": "penvmap",
            "filename": str(Path(envmap).resolve()),
            "scale": scale,
            "to_world": transform().rotate([0, 1, 0], yaw_deg),
        },
        "sphere": {
            "type": "sphere",
            "bsdf": {"type": "roughconductor", "material": "Au", "distribution": "ggx", "alpha": 0.01},
        },
    }
