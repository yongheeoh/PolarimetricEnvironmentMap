"""ND metadata and transmission rules shared by desktop and Android recovery."""

import numpy as np


def nd_transmission(profile, metadata, shot_id, *, dtype=None):
    """Resolve measured OD without inferring a filter from image brightness.

    Retain each engine's calculation precision: desktop uses the profile's
    default array dtype, while streamed Android recovery requests float32.
    """
    od = metadata.get("sun_shot", {}).get("nd_od", metadata.get("nd_od"))
    if od is None:
        raise ValueError(f"Sun {shot_id}: missing actual nd_od; cannot infer from image brightness")
    od = float(od)
    table = profile.get("transmission", profile)  # Full profiles and compact release JSONs.
    transmission = np.ones(3, dtype=dtype) if od == 0 else np.array(table[f"OD{od:g}"]["T_BGR"], dtype=dtype)
    if not np.isfinite(transmission).all() or not ((transmission > 0) & (transmission <= 1)).all():
        raise ValueError("Invalid ND transmission")
    return od, transmission
