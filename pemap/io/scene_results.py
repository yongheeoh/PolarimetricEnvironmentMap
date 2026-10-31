"""Scene output directories, compact run records and publication."""

from datetime import datetime, timezone
from pathlib import Path
import copy
import platform
import shutil
import sys
import tempfile
import time
import json
import cv2
import numpy as np
from pemap.io.files import read, write
from pemap.io.images import png, binary
from pemap.io.display import display
from pemap.io.outputs import exposure, export_result, save_stokes
from pemap.radiometry.color import apply_wb_ccm_stokes
from pemap.io.layout import storage_layout, output_records, publish, published_relative
from pemap.polarization.min_rot import align_vectors_batched
from pemap.polarization.optimized.s2l2_kernels import build_measurement_frame, s2l2_to_stokes


class SceneResults:
    """Output session: files, previews and metadata only; no processing driver."""

    def __init__(self, scene, captures, calibration, options):
        self.scene = scene
        self.captures = captures
        self.calibration = calibration
        self.options = options
        self.mode = options.mode
        self.destination = (options.output_root or scene).resolve()
        self.published = False
        self.warnings = []
        self.mask_report = {}

    def output_path(self, relative):
        return self.destination / published_relative(relative, self.mode, self.run_id)

    def analysis(self, group):
        from pemap.io.analysis import AnalysisOutput

        relative = f"{group}/analysis"
        return AnalysisOutput(self.out / relative, self.output_path(relative))

    def warn(self, message):
        self.warnings.append(message)
        print("WARNING:", message, flush=True)

    def __enter__(self):
        self.start = time.perf_counter()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.destination.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix=".penvmap_", dir=self.destination))
        self.out = self.work / "publish"
        self.out.mkdir()
        self.cache = self.work / "cache"
        return self

    def metadata(self):
        return {
            "schema": "penvmap scene pipeline v1",
            "mode": self.mode,
            "engine": "desktop",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scene": str(self.scene),
            "camera": self.captures.camera,
            "width": self.calibration.geometry.width,
            "height": self.calibration.geometry.height,
            "source_sampling_factor": 1 / self.options.resize_scale,
            "resolved_options": json.loads(json.dumps(vars(self.options), default=str)),
            "storage_layout": storage_layout(self.mode, self.run_id),
            "profiles": {key: str(path) for key, path in self.calibration.paths.items()},
            "processing_settings": self.calibration.settings,
            "polarimetric": {
                "S3_measured": self.calibration.sensor.s3_measured,
                "rotation_model": self.options.rotation_model,
                "stokes_repr_mode": self.options.stokes_repr_mode,
                "composition_representation": (
                    "unrotated_camera"
                    if self.options.rotation_model == "none" and self.mode == "full"
                    else "s2l2"
                ),
            },
            "radiometric": {
                "hdr_z_min": self.calibration.sensor.zmin,
                "hdr_z_max": self.calibration.sensor.zmax,
                "scale": self.calibration.sensor.scale,
                "vignetting_applied": self.options.vignetting,
                "wb_bgr": self.wb.tolist(),
                "ccm_rgb_row": self.ccm.tolist(),
                "WB_CCM": {
                    "scope": self.color_choice["scope"],
                    "profile": str(self.output_path("color/result/wb_ccm.json")),
                },
            },
            "geometric": {
                "rvec_by_id": self.pose_report["rvec_by_id"],
                "rvec_prior_by_id": self.pose_report["rvec_prior_by_id"],
            },
            "blending": {
                "method": "multiband",
                "level": self.options.level,
                "domain_rotation_preset": self.options.domain_rotation_preset,
            },
            "inputs": {key: item["raw"]["path"] for key, item in self.captures.records.items()},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                **{
                    name: getattr(sys.modules[name], "__version__", None)
                    for name in ("torch", "kornia", "vismatch")
                    if name in sys.modules
                },
            },
        }

    def save_panorama(self, panorama, wb, ccm, color_choice, pose_report):
        self.wb, self.ccm = wb, ccm
        self.color_choice, self.pose_report = color_choice, pose_report
        self.base_missing = int((~panorama.coverage).sum())
        self.multiplier = exposure(panorama.payload[..., 0], panorama.coverage)
        self.record = self.metadata()
        self.record["scope"] = "Base panorama before Sun and Floor"
        png(self.out / "envmap/analysis/saturation_mask.png", binary(panorama.saturation))
        if self.mode == "full":
            self.export(self.out / "envmap/result", panorama, copy.deepcopy(self.record))

    def export(self, folder, panorama, record):
        return export_result(
            folder,
            panorama.payload,
            panorama.coverage,
            record,
            self.multiplier,
            geometry=self.calibration.geometry,
            mode=self.mode,
            s3_measured=self.calibration.sensor.s3_measured,
            representation=panorama.representation,
        )

    def save_final(self, panorama, *, sun, floor):
        for message in (*sun.warnings, *floor.warnings):
            self.warn(message)
        record = copy.deepcopy(self.record)
        record.update(
            scope="Final calibrated scene panorama",
            sun_fill=sun.report,
            floor=floor.report,
            coverage={
                "before_missing_pixels": self.base_missing,
                "remaining_missing_pixels": int((~panorama.coverage).sum()),
                "residual_saturated_pixels": int(panorama.saturation.sum()),
            },
            warnings=self.warnings,
        )
        if self.captures.ids["sun"] and not self.options.no_sun:
            png(self.out / "sun/analysis/saturation_mask.png", binary(panorama.saturation))
        theta = self.export(self.out / "result", panorama, record)
        if panorama.representation != "none":
            self.save_diagnostics(panorama.payload, theta, panorama.coverage, self.multiplier)
        for group, result in [("sun", sun), ("floor", floor)]:
            path = self.out / group / "result/run.json"
            if path.exists():
                native = read(path)
                native["composite"] = result.report
                write(path, native)

    def publish(self):
        for path in self.out.rglob("run.json"):
            record = read(path)
            record.update(
                mode=self.mode,
                engine="desktop",
                storage_layout=storage_layout(self.mode, self.run_id),
                elapsed_seconds=round(time.perf_counter() - self.start, 2),
                outputs=output_records(self.out, path.parent, self.destination, self.mode, self.run_id),
            )
            write(path, record)
        write(
            self.out / "envmap/mask/manifest.json",
            {"views": self.mask_report, "convention": "White usable / black excluded"},
        )
        target = publish(self.out, self.destination, self.mode, self.run_id)
        self.published = True
        print("RESULT", target, flush=True)
        return target

    def __exit__(self, error_type, error, traceback):
        if error_type is not None:
            print("Temporary results retained:", self.work, flush=True)
        elif self.published and not self.options.keep_work:
            if self.work.resolve().parent != self.destination or not self.work.name.startswith(".penvmap_"):
                raise ValueError("Unexpected temporary output directory")
            shutil.rmtree(self.work)
        return False

    def save_diagnostics(self, coeff, theta, covered, multiplier):
        if getattr(self.options, "minimal_analysis", False):
            return
        if self.mode == "s0":
            if self.captures.ids["floor"] and not self.options.no_floor:
                plane, known = self.calibration.geometry.nadir(coeff, covered)
                plane[~known] = 0
                png(self.out / "floor/analysis/nadir.png", display(plane[..., 0], multiplier))
            return
        import matplotlib.pyplot as plt

        def plot(path, data, valid, title):
            q = data[..., 1, 1:3] / np.maximum(data[..., 1, 0:1], 1e-8)
            fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
            axes[0].imshow(display(data[..., 0], multiplier)[..., ::-1])
            axes[0].set_title("S0")
            for ax, a, label, cm, lo, hi in [
                (axes[1], q[..., 0] * 100, "G S1/S0 (%)", "RdBu_r", -20, 20),
                (axes[2], q[..., 1] * 100, "G S2/S0 (%)", "RdBu_r", -20, 20),
                (axes[3], np.linalg.norm(q, axis=-1) * 100, "G DoLP (%)", "magma", 0, 30),
            ]:
                im = ax.imshow(np.where(valid, a, np.nan), cmap=cm, vmin=lo, vmax=hi)
                ax.set_title(label)
                fig.colorbar(im, ax=ax, shrink=0.65)
            for ax in axes:
                ax.set_axis_off()
            fig.suptitle(title)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120)
            plt.close(fig)

        plot(
            self.out / "analysis/final_preview/stokes_preview.png",
            theta,
            covered,
            self.scene.name + " / theta-phi",
        )
        if self.captures.ids["floor"] and not self.options.no_floor:
            plane, known = self.calibration.geometry.nadir(coeff, covered)
            rays = self.calibration.geometry.plane_rays()
            T = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
            basis = (align_vectors_batched(rays @ T, [0, 0, 1]) @ [1.0, 0, 0]) @ T.T
            stokes = s2l2_to_stokes(plane, build_measurement_frame(-rays, basis)[..., None, :, :]).astype(
                np.float32
            )
            stokes[~known] = 0
            png(self.out / "floor/analysis/nadir.png", display(stokes[..., 0], multiplier))
            plot(
                self.out / "floor/analysis/nadir_stokes.png", stokes, known, "Continuous nadir reference axes"
            )

    def save_captures(self, images, wb, ccm):
        mode = self.mode
        for group in ["color", "floor", "sun"]:
            if not self.captures.ids[group] and group != "color":
                continue
            if group == "floor" and self.options.no_floor:
                continue
            if group == "sun" and self.options.no_sun:
                continue
            rejected = {}
            for sid in self.captures.ids[group]:
                if mode == "s0" and group != "floor":
                    continue
                try:
                    data = apply_wb_ccm_stokes(images.recover(group, sid), wb, ccm)
                except (KeyError, ValueError) as error:
                    if group != "sun":
                        raise
                    rejected[sid] = str(error)
                    self.warn(f"Sun {sid} excluded: {error}")
                    continue
                if mode == "full":
                    save_stokes(self.out / group / "result" / f"{sid}.exr", data)
                    png(
                        self.out / group / "analysis" / f"{sid}.png",
                        display(
                            data[..., 0],
                            exposure(
                                data[..., 0],
                                (
                                    self.calibration.native_optical
                                    if group == "color"
                                    else self.calibration.optical
                                ),
                            ),
                        ),
                    )
            if mode == "s0":
                continue
            write(
                self.out / group / "result/run.json",
                {
                    "group": group,
                    "shots": self.captures.ids[group],
                    "camera": self.captures.camera,
                    "rejected_shots": rejected,
                    "vignetting": self.options.vignetting,
                    "color_profile": str(self.output_path("color/result/wb_ccm.json")),
                    "polarimetric": self.calibration.sensor.metadata(
                        rotation_model=self.options.rotation_model
                    ),
                    "coordinates": f"Camera {self.options.rotation_model}",
                },
            )
