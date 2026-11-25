# Project page

Website for **Real-world Polarimetric Environment Map Dataset**, SIGGRAPH Asia 2026.

[Live page](https://yongheeoh.github.io/PolarimetricEnvironmentMap/)

- `index.html`: paper content and layout.
- `static/`: figures, videos, styles and scripts.

## Preview

```sh
python -m http.server 8000
```

GitHub Pages deploys the `project_page` branch from its root directory.

Based on [Nerfies](https://nerfies.github.io/) (CC BY-SA 4.0); retain the footer attribution.

## Environment atlas

Open `browser/` on the preview server to browse the released scenes. GitHub Pages
serves this directory under the same project prefix as the paper page, with
no build step. Cards show the entire equirectangular frame at 640 × 320. Hover or focus a card
to crossfade from intensity to AoLP hue with DoLP controlling HSV saturation in light mode and HSV value in dark mode. The
scene detail page has a thumbnail strip for 2048 × 1024 intensity, AoLP, DoLP and DoLP-modulated AoLP maps.
Polarization previews use a luminance-weighted Stokes vector. For each measured
component, compute Yi = 0.2126 Si.R + 0.7152 Si.G + 0.0722 Si.B from the
linear EXR samples, preserving signs. Normalize and derive angles and degrees
from these Yi values. The s0 intensity preview shows tone-mapped RGB. Original EXRs are served by Hugging
Face at the dataset revision recorded in `browser/scenes.js`.

To add a capture, use the named Stokes channels in its released EXR to
generate the previews with `browser/tools/make_previews.py` (see its usage
and provenance notes), then add its metadata and preview paths to the scene
list. After adding scenes or changing the s0 previews, run
`uv run browser/tools/make_similar.py` from the repository root. This computes
six related scenes per capture from DINOv2 features of six overlapping horizon
views, tone-mapped color distributions and the environment label. The site
serves the resulting IDs from `browser/similar.js` as a static asset.

Check the release status and privacy mask in the collection metadata
before preparing new previews. Display names in the atlas are editorial.


The detail page shows Y1/Y0 and Y2/Y0 on a fixed −1 to +1
RdBu scale. For Full Stokes captures, pass `--full-stokes` to include Y3/Y0,
DoP, DoCP, ToP and CoP. Linear Stokes captures omit S3-dependent views.
DoLP, DoP and DoCP use the luminance-weighted Stokes vector and the definitions
in polanalyser 3.0.0. DoCP is |Y3|/Y0. All three degrees use a fixed black-to-red
scale with display intensity clamp(degree, 0, 1)^(1/2.2). Invalid samples
and nonpositive intensity are black. Gamma is applied only to the displayed color.
