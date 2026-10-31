# Scene 63 examples

RAW data is external. Set `DATASET_ROOT` in the notebooks, or `PENVMAP_DATASET`,
to the downloaded `full_stokes` directory containing `raw/63` and `calibration/`.

1. `01_raw_to_penvmap.ipynb`: process RAW using Python 3.13.
2. `02_penvmap_to_render.ipynb`: render with Python matching the custom Mitsuba build.

Install `example/requirements.txt`. Register each environment from its own Python:

```sh
# Processing environment
python -m ipykernel install --user --name penvmap-processing
# Rendering environment
python -m ipykernel install --user --name penvmap-rendering
```

The existing Windows Mitsuba build uses Python 3.10; see the
[build guide](../penvmap_mitsuba3/penvmap_dev_guide.md) for another runtime.
Rendering reads `result/duff.exr`; edit `scene.py` to change the scene.
The distributed `penvmap/duff/63.exr` can also be rendered directly.

Outputs are saved under `example/output/63`. Notebook figures show RGB S0 and
G-channel S1/S0, S2/S0 and S3/S0 in RdBu on a fixed −1 to +1 scale.
Scene 63 has no Sun/Floor captures; these optional stages are skipped.
