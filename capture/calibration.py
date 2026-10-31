"""PC calibration categories matching the Android CalibrationKind storage layout."""

from enum import Enum


class CalibrationKind(Enum):
    VIGNETTING = ("vignetting", "Vignetting", "photometric/vignetting")
    INTRINSIC = ("intrinsic", "Intrinsic", "geometric/intrinsic")
    HDR = ("hdr", "HDR / sensor", "photometric/hdr")
    POLARIMETRIC = ("polarimetric", "Polarimetric", "polarimetric")
    ND = ("nd", "ND transmission", "radiometric/nd")
    WB_CC = ("wb_cc", "Color checker", "radiometric/wb_cc")
    MASK = ("mask", "Mask", "photometric/mask")
    NODAL = ("nodal", "Nodal", "geometric/nodal")

    def __init__(self, kind_id, label, directory):
        self.id, self.label, self.directory = kind_id, label, directory

    def experiment(self, camera):
        camera_folder = {"flir": "FLIR", "basler": "Basler"}[camera]
        return f"{camera_folder}/calibration/{self.directory}/data"

    @classmethod
    def from_id(cls, kind_id):
        return next((kind for kind in cls if kind.id == kind_id), None)
