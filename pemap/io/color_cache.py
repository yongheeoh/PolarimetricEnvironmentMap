"""Reuse a scene colour fit only for the captures and calibration it used."""

from pemap.io.files import read, sha


def legacy_color_inputs_match(scene, paths, root=None):
    """Writing the same default paths explicitly must not invalidate a legacy scene fit."""
    from pemap.io.profiles import resolve_profiles

    keys = {"hdr", "polarimetric"}
    try:
        defaults, _ = resolve_profiles(scene, root, camera="flir", required=keys)
    except FileNotFoundError:
        return False
    return all(defaults[key] == paths[key] for key in keys)


def color_signature(paths, sensor_sources, camera, processing, vignette_source=None):
    signature = {k: sha(paths[k]) for k in ("hdr", "polarimetric") if k in paths}
    signature.update(camera=camera, **{k: v["sha256"] for k, v in sensor_sources.items()})
    signature["processing"] = processing
    signature["vignetting"] = sha(paths["vignetting"]) if vignette_source else None
    if vignette_source:
        signature["vignetting_map"] = vignette_source["sha256"]
    return signature


def matching_color_capture(profile, captures):
    expected = profile.get("source", {}).get("sha256")
    return next((sid for sid, item in captures.items() if item["raw"]["sha256"] == expected), None)


def reusable_color_fit(path, captures, signature, *, legacy_compatible=False):
    if not path.is_file():
        return None
    profile = read(path)
    sid = matching_color_capture(profile, captures)
    compatible = profile.get("processing_signature") == signature
    if "processing_signature" not in profile:
        compatible = legacy_compatible
    metadata = profile.get("source_metadata", {})
    if not metadata and (path.parent / "run.json").is_file():
        records = read(path.parent / "run.json").get("inputs", [])
        if isinstance(records, dict):
            records = records.values()
        expected = profile.get("source", {}).get("sha256")
        metadata = next(
            (
                r.get("metadata", {})
                for r in records
                if isinstance(r, dict) and r.get("raw", {}).get("sha256") == expected
            ),
            {},
        )
    same_metadata = sid is not None and metadata.get("sha256") == captures[sid]["metadata"]["sha256"]
    return profile if compatible and same_metadata and profile.get("source_scope") == "scene" else None
