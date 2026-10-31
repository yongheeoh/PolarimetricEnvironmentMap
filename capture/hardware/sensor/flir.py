import time

import cv2
import numpy as np
import PySpin
from EasyPySpin import VideoCapture

from .camera import BracketResult, Camera


class FlirCamera(Camera):
    """FLIR + EasyPySpin 카메라 래퍼."""

    def __init__(
        self,
        camera_index: int = 0,
        exposure_us: float = 1e5,
        gain: float = 0.0,
        pixel_format: str = "BayerRG8",
        timeout_ms: int = 1000,
        adc_bit_depth: str = "Bit12",
        black_level_clamping: bool | None = None,
        defect_correct_static: bool | None = None,
        bracket_mode: str = "sequencer",
        bracket_retries: int = 2,
    ):
        """black_level_clamping / defect_correct_static: None = leave the camera's current value untouched.
        bracket_mode: "sequencer" (hardware exposure sequencer, no discarded frames) | "sequential".
        """
        super().__init__(camera_index, exposure_us, gain, pixel_format, timeout_ms)
        if bracket_mode not in ("sequencer", "sequential"):
            raise ValueError(f"bracket_mode must be 'sequencer' or 'sequential', got {bracket_mode!r}")
        self.bracket_mode = bracket_mode
        self.bracket_retries = int(bracket_retries)
        self.frame_timeout_margin_ms = 3000  # GetNextImage timeout = exposure + margin
        self._chunks_enabled = False

        # 1. FLIR 카메라 오픈
        cap = VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("[FLIR] Failed to open FLIR camera (EasyPySpin)")

        # 2. 기본 설정
        cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_us)
        cap.set(cv2.CAP_PROP_GAIN, self._gain)

        # ADC depth before PixelFormat (the allowed pixel formats depend on it). Default is Bit10:
        # BayerRG16 then only carries 1024 levels (observed step of 64 in the 2026-10-07 test capture).
        try:
            cap.set_pyspin_value("AdcBitDepth", adc_bit_depth)
        except Exception as e:
            print("[FLIR][WARN] set AdcBitDepth failed:", e)

        if pixel_format is not None:
            try:
                cap.set_pyspin_value("PixelFormat", pixel_format)
            except Exception as e:
                print("[FLIR][WARN] set PixelFormat failed:", e)

        for k, v in [
            ("ExposureAuto", "Off"),
            ("GainAuto", "Off"),
            ("BalanceWhiteAuto", "Off"),
        ]:
            try:
                cap.set_pyspin_value(k, v)
            except Exception as e:
                print(f"[FLIR][WARN] set {k} failed:", e)

        # White balance ratios fixed to 1.0 so raw R/G/B are unscaled (the camera had R×1.125, B×1.81 baked in).
        # White balance is done later from the color checker (WBCC.py).
        for color in ("Red", "Blue"):
            try:
                cap.set_pyspin_value("BalanceRatioSelector", color)
                cap.set_pyspin_value("BalanceRatio", 1.0)
            except Exception as e:
                print(f"[FLIR][WARN] set BalanceRatio[{color}] failed:", e)

        try:
            cap.set_pyspin_value("ExposureMode", "Timed")
        except Exception as e:
            print("[FLIR][WARN] set ExposureMode failed:", e)

        try:
            cap.set_pyspin_value("GammaEnable", False)
        except Exception as e:
            print("[FLIR][WARN] set GammaEnable failed:", e)

        # Black level 0 so the HDR merge needs no offset subtraction (same as BaslerCamera).
        try:
            cap.set_pyspin_value("BlackLevel", 0.0)
        except Exception as e:
            print("[FLIR][WARN] set BlackLevel failed:", e)

        # Optional on-camera RAW processing switches (None = untouched; current value is still logged below).
        for node_name, value in (("BlackLevelClampingEnable", black_level_clamping), ("DefectCorrectStaticEnable", defect_correct_static)):
            if value is not None:
                try:
                    cap.set_pyspin_value(node_name, bool(value))
                except Exception as e:
                    print(f"[FLIR][WARN] set {node_name} failed:", e)

        self.cap = cap
        self._is_open = True
        self.settings = self.read_settings()
        print("[FLIR] Camera initialized with EasyPySpin:", " ".join(f"{k}={v}" for k, v in self.settings.items()))
        self._warn_unexpected_settings(adc_bit_depth, pixel_format)

    def read_settings(self) -> dict:
        """Read back the raw-relevant camera settings (to verify what the camera actually applied)."""
        cap = self.cap
        settings = {}
        for name in ("PixelFormat", "AdcBitDepth", "ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
            try:
                node = getattr(cap.cam, name)
                settings[name] = node.GetCurrentEntry().GetSymbolic()
            except Exception:
                settings[name] = None
        for color in ("Red", "Blue"):
            try:
                cap.set_pyspin_value("BalanceRatioSelector", color)
                settings[f"BalanceRatio{color}"] = round(float(cap.get_pyspin_value("BalanceRatio")), 4)
            except Exception:
                settings[f"BalanceRatio{color}"] = None
        for name in ("BlackLevel", "Gain", "GammaEnable", "BlackLevelClampingEnable", "DefectCorrectStaticEnable"):
            try:
                settings[name] = cap.get_pyspin_value(name)
            except Exception:
                settings[name] = None
        return settings

    def _warn_unexpected_settings(self, adc_bit_depth: str, pixel_format: str | None) -> None:
        s = self.settings
        problems = []
        if s.get("AdcBitDepth") != adc_bit_depth:
            problems.append(f"AdcBitDepth={s.get('AdcBitDepth')} (wanted {adc_bit_depth})")
        if pixel_format is not None and s.get("PixelFormat") != pixel_format:
            problems.append(f"PixelFormat={s.get('PixelFormat')} (wanted {pixel_format})")
        for color in ("Red", "Blue"):
            ratio = s.get(f"BalanceRatio{color}")
            if ratio is None or abs(ratio - 1.0) > 1e-3:
                problems.append(f"BalanceRatio{color}={ratio} (wanted 1.0)")
        if problems:
            print("[FLIR][WARN] !!! camera settings differ from what was requested: " + "; ".join(problems))

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_EXPOSURE:
            self._exposure_us = float(value)
        elif prop_id == cv2.CAP_PROP_GAIN:
            self._gain = float(value)
        try:
            return self.cap.set(prop_id, float(value))
        except Exception as e:
            print(f"[FLIR][WARN] cap.set({prop_id}, {value}) failed:", e)
            return False

    def get(self, prop_id):
        try:
            return self.cap.get(prop_id)
        except Exception as e:
            print(f"[FLIR][WARN] cap.get({prop_id}) failed:", e)
            return None

    def release(self):
        if self._is_open:
            try:
                self.cap.release()
            except Exception:
                pass
            self._is_open = False
            print("[FLIR] Camera released.")

    def grab(self):
        ret, _ = self.cap.read()
        return ret

    # ===== Exposure bracket (hardware sequencer) =====

    def exposure_range_us(self) -> tuple[float, float] | None:
        node = self.cap.cam.GetNodeMap().GetNode("ExposureTime")
        try:
            f = PySpin.CFloatPtr(node)
            return float(f.GetMin()), float(f.GetMax())
        except Exception:
            return None

    def sequencer_capacity(self) -> int:
        """Number of sequencer sets (distinct exposures per hardware cycle); 0 if unsupported."""
        node = self.cap.cam.GetNodeMap().GetNode("SequencerSetSelector")
        if node is None or not PySpin.IsReadable(node):
            return 0
        return int(PySpin.CIntegerPtr(node).GetMax()) + 1

    def capture_bracket(self, exposures_us, settle_s: float = 0.2, settle_frames: int = 2) -> BracketResult:
        """Exposure bracket. ``bracket_mode="sequencer"``: the camera's exposure sequencer takes all frames
        back-to-back (no discarded frames), each frame verified from its chunk data (FrameID continuity,
        ExposureTime, SequencerSetActive). ``"sequential"``: generic set → wait → discard → read.

        Brackets longer than the sequencer capacity are captured in consecutive segments.
        A failed verification retries the whole bracket up to ``self.bracket_retries`` times.
        """
        if self.bracket_mode != "sequencer":
            return super().capture_bracket(exposures_us, settle_s, settle_frames)

        exposures_us = [float(e) for e in exposures_us]
        capacity = self.sequencer_capacity()
        if capacity < 2:
            raise RuntimeError("camera has no usable exposure sequencer; use flir_bracket_mode=sequential")
        segments = [exposures_us[i : i + capacity] for i in range(0, len(exposures_us), capacity)]

        last_error = None
        for attempt in range(self.bracket_retries + 1):
            try:
                frames, actual, frame_ids, sets = [], [], [], []
                t0 = time.perf_counter()
                for segment in segments:
                    seg = self._sequencer_segment(segment)
                    frames += seg["frames"]
                    actual += seg["exposure_us"]
                    frame_ids += seg["frame_ids"]
                    sets += seg["sets"]
                info = {
                    "mode": "sequencer",
                    "segments": len(segments),
                    "retries": attempt,
                    "frame_ids": frame_ids,
                    "sequencer_sets": sets,
                    "requested_exposure_us": exposures_us,
                    "bracket_time_s": round(time.perf_counter() - t0, 3),
                }
                return BracketResult(frames=np.stack(frames, axis=0), exposure_us=actual, info=info)
            except Exception as e:  # verification failure, timeout, incomplete frame
                last_error = e
                print(f"[FLIR][WARN] sequencer bracket failed (attempt {attempt + 1}/{self.bracket_retries + 1}): {type(e).__name__}: {e}")
            finally:
                self._restore_streaming_state()
        raise RuntimeError(f"sequencer bracket failed after {self.bracket_retries + 1} attempts: {last_error}")

    def _stop_streaming(self) -> None:
        cam = self.cap.cam
        held = getattr(self.cap, "_pyspin_image", None)  # EasyPySpin never releases its last image
        if held is not None:
            try:
                held.Release()
            except Exception:
                pass
            try:
                del self.cap._pyspin_image
            except AttributeError:
                pass
        if cam.IsStreaming():
            cam.EndAcquisition()

    def _sequencer_segment(self, exposures_us: list[float]) -> dict:
        """Program the sequencer with ``exposures_us`` and acquire exactly that many frames."""
        cam = self.cap.cam
        nm = cam.GetNodeMap()
        snm = cam.GetTLStreamNodeMap()
        n = len(exposures_us)
        self._stop_streaming()

        # Sequencer off first: while it is On (e.g. left by the previous segment) TriggerMode is read-only.
        _node_set_or_current(nm, "SequencerMode", "Off")
        _node_set(nm, "TriggerMode", "Off")
        _node_set(nm, "AcquisitionFrameRateEnable", False, optional=True)
        if not self._chunks_enabled:
            _node_set(nm, "ChunkModeActive", True)
            for sel in ("ExposureTime", "FrameID", "Gain", "SequencerSetActive", "Timestamp"):
                if _node_set(nm, "ChunkSelector", sel, optional=True) is not None:
                    _node_set(nm, "ChunkEnable", True, optional=True)
            self._chunks_enabled = True

        # Program one sequencer set per exposure, chained 0 → 1 → … → n-1 → 0, each advanced by FrameStart.
        _node_set(nm, "SequencerConfigurationMode", "On")
        _node_set(nm, "SequencerFeatureSelector", "ExposureTime")
        _node_set(nm, "SequencerFeatureEnable", True, optional=True)
        programmed = []
        for i, exp_us in enumerate(exposures_us):
            _node_set(nm, "SequencerSetSelector", i)
            _node_set(nm, "SequencerTriggerSource", "FrameStart")
            _node_set(nm, "SequencerPathSelector", 0, optional=True)
            _node_set(nm, "SequencerSetNext", (i + 1) % n)
            programmed.append(float(_node_set(nm, "ExposureTime", exp_us)))
            _node_set(nm, "SequencerSetSave", None)
        _node_set(nm, "SequencerSetStart", 0, optional=True)
        _node_set(nm, "SequencerConfigurationMode", "Off")
        _node_set(nm, "SequencerMode", "On")

        # Finite acquisition of exactly n frames, delivered oldest first.
        if n == 1:
            _node_set(nm, "AcquisitionMode", "SingleFrame")
        else:
            _node_set(nm, "AcquisitionMode", "MultiFrame")
            _node_set(nm, "AcquisitionFrameCount", n)
        _node_set(snm, "StreamBufferHandlingMode", "OldestFirst")
        _node_set(snm, "StreamBufferCountMode", "Manual", optional=True)
        _node_set(snm, "StreamBufferCountManual", n + 1, optional=True)

        frames, actual, frame_ids, sets = [], [], [], []
        cam.BeginAcquisition()
        try:
            for i, exp_us in enumerate(programmed):
                img = cam.GetNextImage(int(exp_us / 1000) + self.frame_timeout_margin_ms)
                try:
                    if img.IsIncomplete():
                        raise RuntimeError(f"incomplete frame {i} (status {img.GetImageStatus()})")
                    chunk = img.GetChunkData()
                    fid, chunk_exp, chunk_set = int(chunk.GetFrameID()), float(chunk.GetExposureTime()), int(chunk.GetSequencerSetActive())
                    if frame_ids and fid != frame_ids[-1] + 1:
                        raise RuntimeError(f"frame loss: FrameID {frame_ids[-1]} → {fid}")
                    if abs(chunk_exp - exp_us) > max(2.0, 1e-3 * exp_us):
                        raise RuntimeError(f"exposure mismatch at frame {i}: programmed {exp_us:g} us, captured {chunk_exp:g} us")
                    if chunk_set != i:
                        raise RuntimeError(f"sequencer set mismatch at frame {i}: captured set {chunk_set}")
                    frames.append(img.GetNDArray().copy())
                    actual.append(chunk_exp)
                    frame_ids.append(fid)
                    sets.append(chunk_set)
                finally:
                    img.Release()
        finally:
            cam.EndAcquisition()
        return {"frames": frames, "exposure_us": actual, "frame_ids": frame_ids, "sets": sets}

    def _restore_streaming_state(self) -> None:
        """Back to continuous single-exposure streaming for preview / exposure checks."""
        cam = self.cap.cam
        nm = cam.GetNodeMap()
        snm = cam.GetTLStreamNodeMap()
        try:
            if cam.IsStreaming():
                cam.EndAcquisition()
        except Exception as e:
            print("[FLIR][WARN] EndAcquisition during restore failed:", e)
        for target, name, value in (
            (nm, "SequencerMode", "Off"),
            (nm, "SequencerConfigurationMode", "Off"),
            (nm, "AcquisitionMode", "Continuous"),
            (snm, "StreamBufferHandlingMode", "NewestOnly"),
            (snm, "StreamBufferCountMode", "Auto"),
        ):
            _node_set(target, name, value, optional=True)
        _node_set(nm, "ExposureTime", self._exposure_us, optional=True)


# ===== GenICam node helpers (raw PySpin node maps) =====


def _node_set(nodemap, name: str, value, optional: bool = False):
    """Set a node by name (enum symbol, bool, int, float, or command when value is None). Returns the readback.

    optional=True: return None instead of raising when the node is missing / not writable / rejects the value.
    """
    node = nodemap.GetNode(name)
    try:
        if node is None or not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            raise RuntimeError(f"{name} is not writable")
        kind = node.GetPrincipalInterfaceType()
        if kind == PySpin.intfIEnumeration:
            enum = PySpin.CEnumerationPtr(node)
            entry = enum.GetEntryByName(value)
            if entry is None or not PySpin.IsReadable(entry):
                raise RuntimeError(f"{name} has no entry {value!r}")
            enum.SetIntValue(entry.GetValue())
            return enum.GetCurrentEntry().GetSymbolic()
        if kind == PySpin.intfIBoolean:
            b = PySpin.CBooleanPtr(node)
            b.SetValue(bool(value))
            return b.GetValue()
        if kind == PySpin.intfIInteger:
            i = PySpin.CIntegerPtr(node)
            i.SetValue(int(value))
            return i.GetValue()
        if kind == PySpin.intfIFloat:
            f = PySpin.CFloatPtr(node)
            f.SetValue(min(max(float(value), f.GetMin()), f.GetMax()))
            return f.GetValue()
        if kind == PySpin.intfICommand:
            PySpin.CCommandPtr(node).Execute()
            return True
        raise RuntimeError(f"{name}: unsupported node type {kind}")
    except Exception:
        if optional:
            return None
        raise


def _node_set_or_current(nodemap, name: str, value: str):
    """Write an enum node if writable; otherwise require that it already holds ``value``."""
    node = nodemap.GetNode(name)
    if node is not None and PySpin.IsAvailable(node) and PySpin.IsWritable(node):
        return _node_set(nodemap, name, value)
    current = PySpin.CEnumerationPtr(node).GetCurrentEntry().GetSymbolic() if node is not None and PySpin.IsReadable(node) else None
    if current != value:
        raise RuntimeError(f"{name} is not writable and is {current!r} (need {value!r})")
    return current
