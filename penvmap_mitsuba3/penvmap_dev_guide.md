# Custom Mitsuba renderer

`penvmap.cpp` provides polarized environment lighting from layered RGB S0–S3
Duff-basis EXRs. Spectral variants and S2L2 coefficient inputs are not supported.

## Build

Use the target rendering Python with Git, CMake and a compatible C++ toolchain.
From `penvmap_mitsuba3`:

```sh
python -m pip install -r requirements.txt
python compile_mitsuba3.py
```

The script downloads Mitsuba, registers the emitter, builds polarized RGB variants
and writes the local runtime manifest. Sources/builds remain under `external/`.
The Python version used for rendering must match the compiled runtime.

### Measured pBRDFs (optional)

If you render measured pBRDFs (`measured_polarized`) under `penvmap`, copy
`patches/measured_polarized.cpp` over `external/mitsuba3/src/bsdfs/measured_polarized.cpp`
before building. It interpolates the 16 Mueller matrix entries with one table
lookup instead of one lookup per entry, so the kernel compiles faster (first
render of a measured sphere on an RTX 3090: 4.2 s instead of 12.6 s); the values
are the same up to float rounding.

## Render

From the project root, in the rendering environment:

```sh
python -m penvmap_mitsuba3.render_penvmap <duff.exr> --scene example/scene.py --output rendered.exr
```

Edit `example/scene.py` for the camera, objects, material and lighting.
Use `--variant llvm_ad_rgb_polarized` for CPU rendering; CUDA is the default.
See the [notebooks](../example/README.md) for processing and visualization.
