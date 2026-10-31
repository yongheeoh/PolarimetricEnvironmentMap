# Capture

PC capture for Basler and FLIR, using the same controls and storage layout.
Captures include ColorChecker, Envmap, Sun, Floor and calibration images.

## PC setup

Install `capture/requirements.txt` and the selected camera SDK:
Spinnaker/PySpin + EasyPySpin for FLIR, or pypylon for Basler.

```sh
python -m capture.capture --camera basler base_dir=<data-directory> experiment_name=<name>
python -m capture.capture --camera flir base_dir=<data-directory> experiment_name=<name>
```

Stage and bracket settings are in `config/base.yaml`; camera settings are in
`config/basler.yaml` and `config/flir.yaml`. Resume with `--resume <session>`.

| Key | Action |
|---|---|
| C / E / S / F | Select Color / Envmap / Sun / Floor |
| Arrows / Space | Aim / capture an HDR bracket |
| R | Capture the Envmap rotation |
| [ / ] | Halve / double maximum exposure |
| N | Select the actual Sun ND: OD 0 / 2 / 4 |
| K / V | Switch Calibration / Scene mode after Finish |
| 1–8 in Calibration | Select calibration type |
| T / X | Add a shot note / Finish the session |
| Q | Exit; unfinished sessions can be resumed |

Scene RAW stacks: `<base>/<experiment>/<date>/<time>/<group>/hdr/{i}.npy` + JSON.
Calibration RAW stacks: `<base>/<camera>/calibration/<category>/data/<date>/<time>/hdr/`.
Optional `ldr/{i}.png` contains RAW data, not a preview.

For phone capture and on-device processing, see [Android setup](app/README.md).
