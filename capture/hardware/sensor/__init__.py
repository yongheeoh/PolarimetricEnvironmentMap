from .camera import BracketResult, Camera

CAMERA_TYPES = ("flir", "basler")


def make_camera(camera_type: str, **kwargs) -> Camera:
    """Open a camera by type name.

    Vendor SDKs are imported lazily so that a capture environment only needs the
    SDK of the camera actually in use (EasyPySpin for FLIR, pypylon for Basler).
    """
    if camera_type == "flir":
        from .flir import FlirCamera

        return FlirCamera(**kwargs)
    if camera_type == "basler":
        from .basler import BaslerCamera

        return BaslerCamera(**kwargs)
    raise ValueError(f"Unknown camera_type: {camera_type!r}. Choose one of {CAMERA_TYPES}.")


def __getattr__(name: str):
    # Import only the selected camera's vendor SDK.
    if name == "FlirCamera":
        from .flir import FlirCamera

        return FlirCamera
    if name == "BaslerCamera":
        from .basler import BaslerCamera

        return BaslerCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
