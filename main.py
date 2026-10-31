"""RAW -> HDR -> Stokes -> panorama -> Sun -> Floor -> result.

python main.py <scene-directory> [--mode full|s0]
"""

import gc
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from pemap.io.arguments import parse_scene_args
from pemap.io.calibration_data import load_calibration
from pemap.io.captures import load_captures
from pemap.io.files import write
from pemap.io.images import binary
from pemap.io.metadata import load_prior_rvec
from pemap.io.scene_results import SceneResults
from pemap.sensor.capture_images import CaptureImages
from pemap.radiometry.scene_color import calibrate_color
from pemap.radiometry.color_profile import color_matrix
from pemap.radiometry.color import apply_wb_ccm_stokes
from pemap.masking.view_masks import prepare_view_mask
from pemap.registration.matching_views import prepare_matching_views
from pemap.geometry.nadir import build_floor_reference
from pemap.polarization.transport import StokesProjector
from pemap.composition.panorama import align_views, compose_panorama
from pemap.composition.sun import SunCapture, SunInputs, composite_sun
from pemap.composition.floor import FloorCapture, FloorInputs, composite_floor

# All user-facing parameters live here; CLI flags override this file.
CONFIG_PATH = Path(__file__).with_name("processing.yaml")


def merge_penvmap(options):
    # Inputs and calibration. Both sensors use the sequence below.
    scene = options.scene.resolve()
    captures = load_captures(scene, options.camera, pitches=options.pitches)
    if options.rotation_model == "none" and options.mode == "full":
        if (captures.ids["sun"] and not options.no_sun) or (captures.ids["floor"] and not options.no_floor):
            raise ValueError(
                "Unrotated camera Stokes are diagnostic only. Use --no-sun --no-floor, or choose min_rot/mitsuba3 for world-space composition."
            )
    calibration = load_calibration(scene, captures, options)
    geometry = calibration.geometry
    settings = calibration.settings
    print(
        "SCENE",
        scene,
        "\nCAMERA",
        captures.camera,
        "\nCAPTURES",
        {group: len(ids) for group, ids in captures.ids.items()},
        flush=True,
    )
    print("POLARIZATION", options.stokes_repr_mode, options.rotation_model, flush=True)
    if options.check:
        for kind, path in calibration.paths.items():
            print('PROFILE', kind, path, flush=True)
        return None

    with SceneResults(scene, captures, calibration, options) as results:
        # RAW brackets -> HDR mosaic -> camera Stokes -> optional vignetting.
        cache_dir = results.destination / "cache" / options.mode if options.cache else results.cache
        images = CaptureImages(
            captures,
            calibration,
            cache_dir,
            mode=options.mode,
            nd_enabled=not options.no_sun,
            persistent=options.cache,
            payload_cache=results.cache,
        )
        images.recover_many("envmap", options.max_workers)

        # Select scene/global WB and CCM; apply them after panorama blending.
        color_profile, color_choice = calibrate_color(
            scene,
            captures,
            images,
            calibration,
            results.out,
            refit=options.refit_color,
            warn=results.warn,
            color_source=options.color_source,
        )
        applied_color = dict(color_profile)
        if options.apply_wb is not None:
            applied_color["white_balance_applied"] = options.apply_wb
        if options.apply_ccm is not None:
            applied_color["color_correction_matrix_applied"] = options.apply_ccm
        wb, ccm, color = color_matrix(applied_color)
        write(results.out / "color/result/wb_ccm.json", color_profile)
        print("COLOR", color_choice["scope"], color_choice.get("shot_id", ""), flush=True)

        # View masks and camera rotations, initialized from capture metadata.
        matching, masks, reports = prepare_matching_views(
            scene, captures, images, calibration, wb, ccm, results.out, warn=results.warn
        )
        results.mask_report.update({f"envmap/{sid}": report for sid, report in reports.items()})
        priors = {sid: load_prior_rvec(captures.groups["envmap"], sid) for sid in matching}
        alignment = align_views(
            matching,
            calibration.intrinsic,
            priors,
            envmap_size=(geometry.width, geometry.height),
            masks={sid: binary(mask) for sid, mask in masks.items()},
            optimize=not options.stage_poses,
            matchers=options.camera_matchers,
            pitches=options.pitches,
            cache_dir=cache_dir / "pose_cache" if options.cache else None,
            strict=True,
            remapper=(
                geometry.supported_remap if settings["projection_support"] == "mask_and_ray_bound" else None
            ),
        )
        write(results.out / "envmap/analysis/poses.json", alignment.report)

        # Preserve a downward reference for Floor registration when needed.
        reference = reference_known = None
        if captures.ids["floor"] and not options.no_floor:
            reference_images = (
                {sid: apply_wb_ccm_stokes(image, wb, ccm) for sid, image in matching.items()}
                if settings["feature_color"] == "native"
                else matching
            )
            reference, reference_known = build_floor_reference(
                reference_images,
                masks,
                alignment.rotations,
                {sid: float(captures.metadata["envmap", sid].get("pitch_deg", 0)) for sid in matching},
                geometry,
            )
            del reference_images
        del matching
        gc.collect()

        # Recover auxiliary captures and their Floor masks using the same calibration.
        floor_masks = {}
        if not options.no_floor:
            for sid in captures.ids["floor"]:
                s0 = apply_wb_ccm_stokes(images.recover("floor", sid)[..., 0:1], wb, ccm)[..., 0]
                floor_masks[sid], report = prepare_view_mask(
                    scene, captures, calibration, "floor", sid, s0, results.out, warn=results.warn
                )
                results.mask_report[f"floor/{sid}"] = report
        results.save_captures(images, wb, ccm)

        # Intrinsic rays + selected camera basis -> selected representation -> warp/blend/WB.
        projector = StokesProjector(
            calibration.shape,
            calibration.intrinsic,
            mode=options.mode,
            precise=settings["source_rays"] == "iterative100",
            rotation_dtype=np.dtype(settings["rotation_dtype"]),
            stokes_repr_mode=options.stokes_repr_mode,
            rotation_model=options.rotation_model,
        )
        panorama = compose_panorama(
            alignment,
            images.load_view,
            projector,
            level=options.level,
            domain=options.domain_rotation_preset,
            blend_method=options.blend_method,
            wb_ccm=(wb, ccm),
            extend_boundaries=settings["mask_boundary"] == "extend",
            store_payload=images.store_payload,
        )
        results.save_panorama(panorama, wb, ccm, color_choice, alignment.report)

        # ND-corrected Sun -> peak alignment -> per-ray Stokes transport -> Poisson.
        def load_sun(sid):
            data = images.recover("sun", sid)
            metadata = captures.records[f"sun/{sid}"]
            return SunCapture(
                data,
                images.reliable["sun", sid],
                load_prior_rvec(captures.groups["sun"], sid),
                metadata["nd_od"],
                metadata["T_BGR"],
            )

        sun = composite_sun(
            SunInputs(
                geometry,
                calibration.optical,
                color,
                priors,
                alignment.rotations,
                captures.ids["sun"],
                load_sun,
                projector.project_s2l2,
                radius=options.sun_radius,
                mask_mode=options.sun_mask_mode,
                disabled=options.no_sun,
            ),
            panorama,
            analysis=results.analysis("sun"),
        )

        # Floor -> plane registration -> brightness matching -> Stokes transport -> multiband.
        def load_floor(sid, s0_only=False):
            data = images.recover("floor", sid)
            data = apply_wb_ccm_stokes(data[..., 0:1] if s0_only else data, wb, ccm)
            return FloorCapture(data, floor_masks[sid], images.saturation["floor", sid])

        floor = composite_floor(
            FloorInputs(
                geometry,
                calibration.optical,
                color,
                reference,
                reference_known,
                captures.ids["floor"],
                load_floor,
                level=options.level,
                transition=options.floor_transition,
                photometry=options.floor_photometry,
                disabled=options.no_floor,
                rotation_model=options.rotation_model,
            ),
            sun.panorama,
            analysis=results.analysis("floor"),
        )

        # Final theta-phi / Duff / S2L2, or only S0 in quick mode.
        results.save_final(floor.panorama, sun=sun, floor=floor)
        return results.publish()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "scene":
        argv = argv[1:]
    options = parse_scene_args(argv, default_config=CONFIG_PATH, description=__doc__)
    merge_penvmap(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
