"""PC Scene and Calibration capture with shared Basler / FLIR controls."""

import argparse
from pathlib import Path
import cv2
from omegaconf import OmegaConf

from .session import SceneCapture, open_camera
from .controls import mosaic_s0, s0_to_u8, to_display, upright_rotate_code, preview_panel
from .hardware.stage.mecha import MechaController
from .calibration import CalibrationKind


def load_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, choices=["basler", "flir"])
    parser.add_argument(
        "--calibration", action="store_true", help="Open Calibration with no type selected"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue a Scene or Calibration session; resume missing scene rotation poses",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Show resolved settings without connecting hardware",
    )
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    root = Path(__file__).with_name("config")
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "base.yaml"),
        OmegaConf.load(root / f"{args.camera}.yaml"),
        OmegaConf.from_dotlist(args.overrides),
    )
    if cfg.camera != args.camera:
        parser.error("Select the camera with --camera, not a camera= override")
    return args, cfg


def memo_dialog(title="Finish session", prompt="Memo (optional):", initial=""):
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        return simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)
    finally:
        root.destroy()


def main(argv=None):
    args, cfg = load_config(argv)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    camera = open_camera(cfg)
    stage = (
        MechaController(
            cfg.mecha_url,
            settle_delay_large=cfg.stage_settle_pitch_s,
            settle_delay_small=cfg.stage_settle_yaw_s,
        )
        if cfg.mecha_url
        else None
    )
    session = None
    try:
        session = SceneCapture(cfg, camera, stage, args.resume)
        if args.calibration:
            session.set_mode("calibration")
        group = "envmap"
        note = ""
        preview_index = len(session.exposures) // 2
        status = "Select group, aim, capture. Floor is a repositioned shot."
        rotate = upright_rotate_code(cfg.mount_roll_deg)
        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
        while True:
            camera.set(cv2.CAP_PROP_EXPOSURE, session.exposures[preview_index])
            ok, raw = camera.read()
            if not ok or raw is None:
                raise RuntimeError("Camera preview disconnected; captured shots remain saved")
            image = to_display(s0_to_u8(mosaic_s0(raw, cfg.raw_max)), rotate, max_height=750)
            calibration = session.mode == "calibration"
            target = "calibration" if calibration else group
            kind = session.calibration_kind
            guide = (
                {CalibrationKind.ND: "sun", CalibrationKind.WB_CC: "color"}.get(kind)
                if calibration
                else group
            )
            # Preview only. The guide is never written into RAW or processing data.
            if guide in ("color", "sun"):
                h, w = image.shape[:2]
                if guide == "sun":
                    cv2.circle(image, (w // 2, h // 2), max(8, min(h, w) // 15), (0, 255, 255), 1)
                else:
                    cv2.rectangle(
                        image, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 255, 255), 1
                    )
            ready = not calibration or kind is not None
            label = kind.label if kind is not None else "Choose type 1..8"
            lines = [
                f"{cfg.camera.upper()} / {('CALIBRATION / '+label) if calibration else ('SCENE / '+group.upper())} / shots: {len(session.ids(target))}",
                f"Bracket {session.exposures[0]:g}..{session.exposures[-1]:g} us x {len(session.exposures)}; preview {preview_index+1}",
                f'Pose {session.pose}; ND OD: {session.nd_od if session.nd_od is not None else "select with N"}',
                "K Calibration | V Scene | T Shot note",
            ]
            if calibration:
                choices = list(CalibrationKind)
                lines.extend(
                    " | ".join(
                        f"{j+1} {choices[j].label}" for j in range(i, min(i + 2, len(choices)))
                    )
                    for i in range(0, len(choices), 2)
                )
                lines.append(
                    "Space HDR capture"
                    if ready
                    else "Capture disabled: choose a calibration type first"
                )
            else:
                lines.append(
                    "C Color | E Envmap | S Sun | F Floor | Space capture | R rotation/resume"
                )
            lines.extend(
                [
                    "Arrows aim | N ND 0/2/4 | [ ] bracket max | U/D preview | X Finish + memo | Q leave open",
                    f"Shot note: {'set' if note else 'empty'}",
                    status[:110],
                ]
            )
            cv2.imshow("Capture", preview_panel(image, lines))
            key = cv2.waitKeyEx(20)
            if key in (27, ord("q")) or cv2.getWindowProperty("Capture", cv2.WND_PROP_VISIBLE) < 1:
                break
            try:
                if key in (ord("k"), ord("v")):
                    mode = "calibration" if key == ord("k") else "scene"
                    session.set_mode(mode)
                    note = ""
                    status = (
                        "Choose calibration type 1..8"
                        if mode == "calibration"
                        else "Select a scene capture group"
                    )
                elif calibration and ord("1") <= key <= ord("8"):
                    session.select_calibration(list(CalibrationKind)[key - ord("1")])
                    note = ""
                    status = f"{session.calibration_kind.label}: ready to capture HDR"
                elif key == ord("t"):
                    updated = memo_dialog("Shot note", "Note for subsequent captures:", note)
                    if updated is not None:
                        note = updated
                elif not calibration and key in (ord("c"), ord("e"), ord("s"), ord("f")):
                    group = {
                        ord("c"): "color",
                        ord("e"): "envmap",
                        ord("s"): "sun",
                        ord("f"): "floor",
                    }[key]
                elif key == ord("n"):
                    session.nd_od = {None: 0.0, 0.0: 2.0, 2.0: 4.0, 4.0: 0.0}.get(
                        session.nd_od, 0.0
                    )
                elif key in (ord("["), ord("]")):
                    session.set_max_exposure(
                        session.exposures[-1] * (0.5 if key == ord("[") else 2)
                    )
                elif key in (ord("u"), ord("d")):
                    preview_index = max(
                        0,
                        min(
                            len(session.exposures) - 1,
                            preview_index + (1 if key == ord("u") else -1),
                        ),
                    )
                elif key in (2424832, 2555904, 2490368, 2621440, 81, 82, 83, 84):
                    yaw, pitch = session.pose or (0.0, 0.0)
                    step = float(cfg.aim_step_deg)
                    yaw += step * ({2424832: -1, 2555904: 1, 81: -1, 83: 1}.get(key, 0))
                    pitch += step * ({2490368: 1, 2621440: -1, 82: 1, 84: -1}.get(key, 0))
                    session.move(yaw, pitch)
                elif key == 32:
                    sid = session.capture(target, note=note)
                    status = f"Saved {target}/{sid} at {session.root}"
                elif key == ord("r"):
                    if calibration or group != "envmap":
                        raise ValueError("Select Envmap before starting rotation")
                    status = "Rotation running; ESC stops after the current bracket"
                    cv2.imshow("Capture", preview_panel(image, [*lines[:-1], status]))
                    session.rotate(cancelled=lambda: cv2.waitKey(1) == 27)
                    status = "Rotation stopped/completed; R resumes only missing poses"
                elif key == ord("x"):
                    memo = memo_dialog() if session.has_captures else ""
                    if memo is not None:
                        finished = session.finish(memo)
                        note = ""
                        status = f"Finished {finished}" if finished else "Ready for a new scene"
            except (ValueError, RuntimeError, OSError) as error:
                status = str(error)
                print(status, flush=True)
    finally:
        if session is not None:
            session.close()
        camera.release()
        if stage is not None:
            stage.session.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
