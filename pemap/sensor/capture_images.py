"""Recover either sensor's RAW captures and keep large arrays in a temporary cache."""

import cv2
import numpy as np
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pemap.io.files import read, write
from pemap.geometry.resize import resize_bgr_stokes_image
from pemap.sensor.preprocessing import _fuse_one_bracket
from pemap.radiometry.nd import nd_transmission
from pemap.io.images import binary
from pemap.sensor import PanoramaView


class CaptureImages:
    def __init__(
        self,
        captures,
        calibration,
        cache,
        *,
        mode="full",
        nd_enabled=True,
        persistent=False,
        payload_cache=None,
    ):
        self.captures = captures
        self.calibration = calibration
        self.cache = cache
        self.mode = mode
        self.nd_enabled = nd_enabled
        self.persistent = persistent
        self.payload_cache = payload_cache or cache
        self.saturation = {}
        self.reliable = {}

    def cached(self, group, sid):
        if self.persistent:
            return self.cache / group / sid / f"{self.cache_token(group, sid)}.npy"
        return self.cache / group / f"{sid}.npy"

    def recover_many(self, group, max_workers=1):
        def recover_one(sid):
            self.recover(group, sid)

        if max_workers == 1:
            for sid in self.captures.ids[group]:
                recover_one(sid)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in executor.map(recover_one, self.captures.ids[group]):
                    pass

    def cache_token(self, group, sid):
        """Small cache key: file identity and effective recovery settings, not a RAW content hash."""
        raw = self.captures.groups[group] / f"{sid}.npy"
        paths = [raw, raw.with_suffix(".json"), *self.calibration.paths.values()]
        identity = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        values = [
            identity,
            self.calibration.settings,
            self.calibration.sensor.sources,
            self.calibration.vignette_source,
            self.calibration.resize_info.resized_hw,
            self.mode,
            self.nd_enabled,
        ]
        return hashlib.blake2b(json.dumps(values, sort_keys=True).encode(), digest_size=16).hexdigest()

    def recover(self, group, sid):
        path = self.cached(group, sid)
        if path.exists() and (group, sid) in self.reliable:
            return np.load(path, mmap_mode="r")
        token = self.cache_token(group, sid) if self.persistent else None
        record, mask_path = path.with_suffix(".json"), path.with_suffix(".npz")
        if (
            self.persistent
            and path.is_file()
            and record.is_file()
            and mask_path.is_file()
            and read(record).get("token") == token
        ):
            with np.load(mask_path) as masks:
                self.reliable[group, sid] = masks["reliable"]
                self.saturation[group, sid] = masks["saturation"]
            if group == "sun" and self.nd_enabled:
                od, transmission = nd_transmission(
                    self.calibration.profiles["nd"], self.captures.metadata[group, sid], sid
                )
                self.captures.records[f"{group}/{sid}"].update(nd_od=od, T_BGR=transmission.tolist())
            print("CACHED STOKES", group, sid, flush=True)
            return np.load(path, mmap_mode="r")
        stack_path = self.captures.groups[group] / f"{sid}.npy"
        stack = np.load(stack_path, mmap_mode="r")
        full_scale = self.calibration.sensor.full_scale
        sat = np.all(stack >= self.calibration.sensor.zmax * full_scale, axis=0)
        reliable = np.any(
            (stack >= self.calibration.sensor.zmin * full_scale)
            & (stack <= self.calibration.sensor.zmax * full_scale),
            axis=0,
        )
        self.saturation[group, sid] = cv2.dilate(binary(sat), np.ones((9, 9), np.uint8)) > 0
        self.reliable[group, sid] = cv2.erode(binary(reliable), np.ones((9, 9), np.uint8)) > 0
        del stack
        _, raw = _fuse_one_bracket(
            (
                self.captures.groups[group],
                sid,
                self.calibration.sensor.zmin,
                self.calibration.sensor.zmax,
                np.float32,
            )
        )
        original_order = self.calibration.settings["radiometric_order"] == "vignetting_then_scale"
        data = (
            self.calibration.sensor.recover_unscaled(raw)
            if original_order
            else self.calibration.sensor.recover(raw)
        )
        if self.mode == "s0":
            data = data[..., 0:1].copy()
        if self.calibration.vignette is not None:
            data *= self.calibration.vignette[..., None]
        if original_order:
            data *= self.calibration.sensor.scale
        if group == "sun" and self.nd_enabled:
            od, T = nd_transmission(self.calibration.profiles["nd"], self.captures.metadata[group, sid], sid)
            data /= T[None, None, :, None]
            self.captures.records[f"{group}/{sid}"].update(nd_od=od, T_BGR=T.tolist())
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite recovered data {group}/{sid}")
        info = self.calibration.resize_info
        if group != "color" and info.original_hw != info.resized_hw:
            data = resize_bgr_stokes_image(data, info)
            size = info.resized_hw[::-1]
            self.reliable[group, sid] = (
                cv2.resize(self.reliable[group, sid].astype(np.float32), size, interpolation=cv2.INTER_AREA)
                > 0.9999
            )
            self.saturation[group, sid] = (
                cv2.resize(self.saturation[group, sid].astype(np.float32), size, interpolation=cv2.INTER_AREA)
                > 0
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data.astype(np.float32))
        if self.persistent:
            np.savez_compressed(
                mask_path, reliable=self.reliable[group, sid], saturation=self.saturation[group, sid]
            )
            write(record, {"token": token})
        print("HDR + STOKES", group, sid, flush=True)
        return data

    def store_payload(self, sid, payload):
        """Keep full-resolution S2L2 views pageable without changing their values."""
        path = self.payload_cache / "panorama_payload" / f"{sid}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, payload)
        return np.load(path, mmap_mode="r")

    def load_view(self, sid):
        image = np.load(self.cached("envmap", sid), mmap_mode="r")
        return PanoramaView(image, self.saturation["envmap", sid])
