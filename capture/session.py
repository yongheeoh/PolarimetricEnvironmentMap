"""Camera-independent Scene and Calibration capture, RAW storage and Finish."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import cv2
import numpy as np
from omegaconf import OmegaConf

from .hardware.sensor import make_camera
from .controls import make_bracket, prior_rvec
from .calibration import CalibrationKind

GROUPS = ("color", "envmap", "sun", "floor")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def open_camera(cfg):
    extra = {}
    if cfg.camera == "flir":
        extra = dict(
            black_level_clamping=cfg.flir_black_level_clamping,
            defect_correct_static=cfg.flir_defect_correct_static,
            bracket_mode=cfg.flir_bracket_mode,
        )
    return make_camera(
        cfg.camera,
        camera_index=cfg.camera_index,
        gain=cfg.gain,
        pixel_format=cfg.pixel_format,
        exposure_us=cfg.bracket_min_us,
        **extra,
    )


class SceneCapture:
    """The scene state is independent of the GUI and camera SDK.

    RAW IDs are append-only. Rotation resumes missing poses; it never overwrites
    captured views. Floor poses belong to a repositioned rig, not scene rotation.
    """

    def __init__(self, cfg, camera, stage=None, resume=None):
        self.cfg, self.camera, self.stage = cfg, camera, stage
        self.root = Path(resume).resolve() if resume else None
        self.pose = None
        self.nd_od = None
        self.meta = {}
        self.mode = "scene"
        self.calibration_kind = None
        if self.root:
            self.meta = json.loads(
                (self.root / "meta/capture_meta.json").read_text(encoding="utf-8")
            )
            saved_camera = self.meta.get("camera", self.meta.get("config", {}).get("camera"))
            if saved_camera == "pylon":
                saved_camera = "basler"
            if saved_camera != cfg.camera:
                raise ValueError("Resume camera does not match the saved scene")
            self.meta["camera"] = cfg.camera
            if self.meta.get("capture_kind") == "calibration":
                self.mode = "calibration"
                self.calibration_kind = CalibrationKind.from_id(
                    self.meta.get("calibration", {}).get("type")
                )
                if self.calibration_kind is None:
                    raise ValueError("Unknown saved calibration type")
        self.exposures = []
        self.exposure_range_us = self.camera.exposure_range_us()
        if self.cfg.bracket_min_us is None:
            if self.exposure_range_us is None:
                raise ValueError("Camera did not report its minimum exposure")
            minimum, maximum = map(float, self.exposure_range_us)
            if not np.isfinite([minimum, maximum]).all() or not 0 < minimum <= maximum:
                raise ValueError("Invalid camera exposure range")
            self.cfg.bracket_min_us = minimum
            self.cfg.bracket_min_source = "camera_minimum"
        else:
            self.cfg.bracket_min_source = "preset_or_override"
        self.cfg.sensor_exposure_range_us = (
            list(self.exposure_range_us) if self.exposure_range_us is not None else None
        )
        self.set_max_exposure(cfg.bracket_max_us)

    def set_max_exposure(self, value):
        exposures, _ = make_bracket(
            self.cfg.bracket_min_us, value, self.cfg.num_brackets, self.exposure_range_us
        )
        self.exposures = exposures
        self.cfg.bracket_max_us = float(value)

    def set_mode(self, mode):
        if mode not in ("scene", "calibration"):
            raise ValueError(mode)
        if self.mode == mode:
            return
        if self.root is not None:
            raise ValueError("Finish the current session before switching Scene / Calibration")
        self.mode, self.calibration_kind = mode, None
        self.pose, self.nd_od = None, None

    def select_calibration(self, kind):
        if self.mode != "calibration" or not isinstance(kind, CalibrationKind):
            raise ValueError("Enter Calibration before choosing a type")
        if self.root is not None and kind != self.calibration_kind:
            raise ValueError("Finish the current calibration before choosing another type")
        self.calibration_kind = kind

    def _part(self, group):
        return self.root if group == "calibration" else self.root / group

    @property
    def has_captures(self):
        if self.root is None:
            return False
        groups = ("calibration",) if self.mode == "calibration" else GROUPS
        return any(
            p.stem.isdigit() and p.with_suffix(".npy").is_file()
            for group in groups
            for p in (self._part(group) / "hdr").glob("*.json")
        )

    def _begin(self):
        if self.root is not None:
            return
        at = datetime.now()
        experiment = self.cfg.experiment_name
        if self.mode == "calibration":
            if self.calibration_kind is None:
                raise ValueError("Select the calibration type before Capture")
            experiment = self.calibration_kind.experiment(self.cfg.camera)
        for offset in range(60):
            stamp = at + timedelta(seconds=offset)
            root = Path(self.cfg.base_dir) / experiment / stamp.strftime("%Y%m%d/%H%M%S")
            try:
                root.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            self.root = root.resolve()
            break
        else:
            raise FileExistsError("Could not allocate a new scene timestamp")
        self.meta = {
            "created_at": at.astimezone().isoformat(),
            "camera": self.cfg.camera,
            "pixel_format": self.cfg.pixel_format,
            "annotations": {},
            "experiment_name": experiment,
        }
        if self.mode == "calibration":
            self.meta.update(
                capture_kind="calibration",
                calibration={"type": self.calibration_kind.id, "scope": "global", "shots": []},
            )
        self._save_meta(False)

    def ids(self, group):
        if self.root is None:
            return []
        # Include orphan RAW/JSON files when allocating IDs, to avoid overwrites.
        folder = self._part(group) / "hdr"
        return sorted(
            {
                int(p.stem)
                for p in folder.glob("*")
                if p.suffix in (".npy", ".json") and p.stem.isdigit()
            }
        )

    def _save_meta(self, finished):
        config = OmegaConf.to_container(self.cfg, resolve=True)
        if self.mode == "calibration":
            config["experiment_name"] = self.calibration_kind.experiment(self.cfg.camera)
        groups = ("calibration",) if self.mode == "calibration" else GROUPS
        layout = {
            "version": 3,
            "ldr_content": "raw",
            "ldr_path": "ldr/{i}.png",
            "ldr_frame": "middle bracket exposure (zero-based T/2)",
            "ldr_optional": True,
            "preview_saved": False,
        }
        if self.mode == "scene":
            layout["groups"] = {g: g for g in GROUPS}
        self.meta.update(
            config=config,
            counts={g: len(self.ids(g)) for g in groups},
            storage_layout=layout,
            session={"finished": finished, "updated_at": datetime.now().astimezone().isoformat()},
        )
        if self.mode == "calibration":
            self.meta["num_calibration"] = len(self.meta["calibration"]["shots"])
        write_json(self.root / "meta/capture_meta.json", self.meta)

    def move(self, yaw, pitch):
        if self.stage is None:
            raise RuntimeError("Configure mecha_url to move the stage")
        # A failed command may have moved one axis; the old pose is no longer safe.
        self.pose = None
        self.stage.move_abs(yaw=float(yaw), pitch=float(pitch), large_move=True)
        self.pose = (float(yaw), float(pitch))

    def capture(self, group, note="", rotation_index=None):
        if self.mode == "calibration":
            if group != "calibration" or self.calibration_kind is None:
                raise ValueError("Select the calibration type before Capture")
            if self.calibration_kind == CalibrationKind.ND and (
                self.nd_od is None or not np.isfinite(self.nd_od) or self.nd_od < 0
            ):
                raise ValueError("Select ND OD explicitly (0 = no filter)")
        elif group not in GROUPS:
            raise ValueError(group)
        if group == "envmap" and self.pose is None:
            raise ValueError("Move the stage to a known pose before capturing Envmap")
        if group == "sun" and (self.nd_od is None or not np.isfinite(self.nd_od) or self.nd_od < 0):
            raise ValueError("Select Sun ND OD explicitly (0 = no filter)")
        bracket = self.camera.capture_bracket(
            self.exposures, self.cfg.settle_s, self.cfg.settle_frames
        )
        raw = np.asarray(bracket.frames)
        measured = np.asarray(bracket.exposure_us, float)
        expected_dtype = np.dtype("uint16" if self.cfg.pixel_format == "BayerRG16" else "uint8")
        if raw.ndim != 3 or raw.dtype != expected_dtype or len(raw) != len(self.exposures):
            raise ValueError(f"Unexpected RAW bracket: {raw.shape}, {raw.dtype}")
        if (
            measured.shape != (len(raw),)
            or not np.isfinite(measured).all()
            or np.any(measured <= 0)
        ):
            raise ValueError("Invalid measured exposure times")
        self._begin()
        sid = max(self.ids(group), default=-1) + 1
        folder = self._part(group) / "hdr"
        folder.mkdir(parents=True, exist_ok=True)
        at = datetime.now().astimezone().isoformat()
        yaw, pitch = self.pose or (0.0, 0.0)
        meta = dict(
            camera=self.cfg.camera,
            pixel_format=self.cfg.pixel_format,
            exposure_times_s=(measured / 1e6).tolist(),
            requested_exposure_us=list(self.exposures),
            bracket=bracket.info,
            captured_at=at,
            capture_kind=group,
            note=note,
            yaw_deg=yaw,
            pitch_deg=pitch,
            mount_roll_deg=float(self.cfg.mount_roll_deg),
            rvec=prior_rvec(yaw, pitch, self.cfg.mount_roll_deg).reshape(3, 1).tolist(),
            stage_pose_known=self.pose is not None,
            rotation_index=rotation_index,
            nd_od=self.nd_od if group == "sun" else 0.0,
            save_raw_png=bool(self.cfg.save_raw_png),
        )
        meta["ldr"] = dict(
            kind="raw",
            path=f"../ldr/{sid}.png" if self.cfg.save_raw_png else None,
            frame_index=len(raw) // 2,
            dtype=str(raw.dtype),
            orientation="sensor",
            preview_saved=False,
        )
        if group == "sun":
            meta["sun"] = dict(
                nd_od=self.nd_od,
                stage_yaw_deg=yaw,
                stage_pitch_deg=pitch,
                stage_pose_known=self.pose is not None,
                captured_at=at,
            )
        if group == "floor":
            meta["rvec_reference"] = "stage_local_after_repositioning"
            meta["floor_shot"] = dict(
                purpose="nadir_patch",
                source_scene=str(self.root),
                exposure_mode="bracket",
                scene_pose_known=False,
                pose_reference="stage_local_after_repositioning",
                translation_m=None,
                include_in_rotation=False,
                note=note,
            )
        if group == "calibration":
            meta["calibration"] = {
                "type": self.calibration_kind.id,
                "scope": "global",
                "exposure_mode": "bracket",
                "note": note.strip() or None,
                "stage_pose_known": self.pose is not None,
                "captured_at": at,
            }
            if self.calibration_kind == CalibrationKind.ND:
                meta["nd_od"] = self.nd_od
                meta["calibration"]["nd_od"] = self.nd_od
        pending = folder / f"{sid}.pending.npy"
        np.save(pending, raw)
        pending.replace(folder / f"{sid}.npy")
        if self.cfg.save_raw_png:
            ldr = self._part(group) / "ldr"
            ldr.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(ldr / f"{sid}.png"), raw[len(raw) // 2]):
                raise IOError("RAW PNG write failed")
        write_json(folder / f"{sid}.json", meta)
        if group == "calibration":
            self.meta["calibration"]["shots"].append({**meta["calibration"], "idx": sid})
        self._save_meta(False)
        print(f"Saved {group}/{sid}: {raw.shape} {raw.dtype}", flush=True)
        return sid

    def rotate(self, cancelled=lambda: False):
        if self.mode != "scene":
            raise ValueError("Rotation capture is available in Scene mode")
        done = set()
        if self.root:
            for p in (self.root / "envmap/hdr").glob("*.json"):
                m = json.loads(p.read_text(encoding="utf-8"))
                if p.with_suffix(".npy").is_file() and m.get("stage_pose_known", True):
                    done.add((float(m["yaw_deg"]), float(m["pitch_deg"])))
        index = 0
        for pitch in self.cfg.pitch_degs:
            for yaw in self.cfg.yaw_degs:
                if cancelled():
                    return
                if (float(yaw), float(pitch)) not in done:
                    self.move(yaw, pitch)
                    time.sleep(self.cfg.capture_delay)
                    self.capture("envmap", rotation_index=index)
                index += 1

    def finish(self, memo=""):
        path = self.root
        if path is not None:
            self.meta["annotations"] = {**self.meta.get("annotations", {}), "memo": memo}
            self._save_meta(True)
        self.root, self.meta = None, {}
        self.pose = None
        self.calibration_kind, self.nd_od = None, None
        return path

    def close(self):
        if self.root:
            self._save_meta(False)
