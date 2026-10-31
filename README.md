# Real-world Polarimetric Environment Map Dataset

**SIGGRAPH Asia 2026** · [Project Page](https://yongheeoh.github.io/PolarimetricEnvironmentMap/)

Build polarimetric HDR environment maps from Basler and FLIR RAW captures.

## Installation

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

## Processing

Run from the project root with the downloaded dataset:

```sh
uv run python main.py <collection>/raw/<scene-id> --output-root <output-directory>
```

Both collections use [`processing.yaml`](processing.yaml). The camera is detected
automatically; add `--mode s0` for a faster S0-only result. Optional ColorChecker,
Sun and Floor captures are processed when present. Profiles are supplied with
the dataset. Outputs include `duff.exr`, `thetaphi.exr`, `s2l2.exr` and `s0.png`.

## Examples and rendering

- [RAW → PEnvMap notebook](example/01_raw_to_penvmap.ipynb)
- [PEnvMap → rendering notebook](example/02_penvmap_to_render.ipynb)
- [Notebook setup](example/README.md)

Rendering requires the [custom Mitsuba plugin](penvmap_mitsuba3/penvmap_dev_guide.md)
and a matching Python environment. Edit [`example/scene.py`](example/scene.py) to change the scene.

```sh
python -m penvmap_mitsuba3.render_penvmap <duff.exr> --scene example/scene.py --output rendered.exr
```

## Capture

[PC capture](capture/README.md) · [Android capture and processing](capture/app/README.md)

## Citation

If you use our dataset or code in your research, please cite:

```bibtex
@inproceedings{oh2026penvmap,
  author    = {Oh, Yonghee and Maeda, Ryota and Moon, Yunseong and Baek, Seung-Hwan},
  title     = {Real-world Polarimetric Environment Map Dataset},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842262},
  url       = {https://doi.org/10.1145/3829340.3842262}
}
```
