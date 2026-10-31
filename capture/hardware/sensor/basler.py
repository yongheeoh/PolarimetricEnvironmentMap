import cv2
from pypylon import pylon
from .basler_config import config_nodemap, set_node_value, get_node_value
from .camera import Camera


class BaslerCamera(Camera):
    """Basler camera adapter using the pypylon SDK."""

    def __init__(
        self,
        camera_index: int = 0,
        exposure_us: float | None = None,
        gain: float = 0.0,
        pixel_format: str = "BayerRG12",
        timeout_ms: int = 10000000,
    ):
        # The base field is bookkeeping; the device value is resolved after its
        # RAW mode is configured. None selects the sensor's current minimum.
        super().__init__(
            camera_index,
            1e5 if exposure_us is None else exposure_us,
            gain,
            pixel_format,
            timeout_ms,
        )

        # 1. 카메라 연결
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise pylon.RuntimeException("No camera present.")
        if camera_index >= len(devices):
            raise IndexError(
                f"Requested camera_index={camera_index}, "
                f"but only {len(devices)} device(s) found."
            )

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[camera_index]))
        self.camera.Open()
        self._is_open = True

        # 2. 노드맵 설정
        nodemap = self.camera.GetNodeMap()
        config_nodemap(
            nodemap,
            initial_exposure=exposure_us,
            initial_gain=gain,
            pixel_format=pixel_format,
        )
        initial = self.exposure_range_us()[0] if exposure_us is None else float(exposure_us)
        if not self.set(cv2.CAP_PROP_EXPOSURE, initial):
            self.release()
            raise RuntimeError(f"Camera rejected initial exposure {initial:g} us")

        # 3. Grab 시작
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def read(self):
        if not self.camera.IsGrabbing():
            return False, None
        try:
            grab_result = self.camera.RetrieveResult(
                self.timeout_ms,
                pylon.TimeoutHandling_ThrowException,
            )
        except Exception as e:
            print("[ERROR] RetrieveResult 실패:", e)
            return False, None

        if not grab_result.GrabSucceeded():
            grab_result.Release()
            return False, None

        img = grab_result.Array.copy()
        grab_result.Release()
        return True, img

    def set(self, prop_id, value):
        nodemap = self.camera.GetNodeMap()
        try:
            if prop_id == cv2.CAP_PROP_EXPOSURE:
                exposure_us = float(value)
                try:
                    set_node_value(nodemap, "ExposureAuto", "Off")
                except Exception:
                    pass
                # Required scientific setting: let failures reach the caller.
                nodemap.GetNode("ExposureTime").SetValue(exposure_us)
                self._exposure_us = exposure_us
                return True
            elif prop_id == cv2.CAP_PROP_GAIN:
                gain = float(value)
                try:
                    set_node_value(nodemap, "GainAuto", "Off")
                except Exception:
                    pass
                set_node_value(nodemap, "Gain", gain)
                self._gain = gain
                return True
            else:
                return False
        except Exception as e:
            print(f"[WARN] set({prop_id}, {value}) 실패:", e)
            return False

    def get(self, prop_id):
        nodemap = self.camera.GetNodeMap()
        try:
            if prop_id == cv2.CAP_PROP_EXPOSURE:
                return get_node_value(nodemap, "ExposureTime")
            elif prop_id == cv2.CAP_PROP_GAIN:
                return get_node_value(nodemap, "Gain")
            else:
                return None
        except Exception as e:
            print(f"[WARN] get({prop_id}) 실패:", e)
            return None

    def exposure_range_us(self):
        node = self.camera.ExposureTime
        return float(node.GetMin()), float(node.GetMax())

    def release(self):
        if self._is_open:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.Close()
            except Exception:
                pass
            self._is_open = False

    def grab(self):
        if not self.camera.IsGrabbing():
            return False
        try:
            result = self.camera.RetrieveResult(
                self.timeout_ms,
                pylon.TimeoutHandling_ThrowException,
            )
            try:
                return bool(result.GrabSucceeded())
            finally:
                result.Release()
        except Exception as e:
            print("[ERROR] grab() 실패:", e)
            return False
