# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch==2.8.0", "transformers==4.57.1", "pillow", "numpy"]
# ///
"""Rebuild the static scene recommendations: uv run browser/tools/make_similar.py"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

BROWSER = Path(__file__).resolve().parents[1]
MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
VIEWS = 6


def normalized(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def perspective_views(image):
    # Six overlapping 90-degree perspective views around the horizon.
    axis = (np.arange(224) + 0.5) / 112 - 1
    x, y = np.meshgrid(axis, axis)
    latitude = np.arctan2(-y, np.sqrt(1 + x*x))
    v = np.clip(((0.5 - latitude / np.pi) * image.shape[0]).astype(int), 0, image.shape[0]-1)
    views = []
    for i in range(VIEWS):
        longitude = np.arctan(x) + 2*np.pi*i/VIEWS
        u = ((0.5 + longitude / (2*np.pi)) * image.shape[1]).astype(int) % image.shape[1]
        views.append(image[v, u])
    return np.stack(views)


def lighting_histogram(image):
    # Weight each row by solid angle so polar pixels do not dominate the palette.
    h, w = image.shape[:2]
    weights = np.sin(np.pi * (np.arange(h) + 0.5) / h).repeat(w)
    bins = (image.astype(int) * 6 // 256).reshape(-1, 3)
    indices = bins @ np.array([36, 6, 1])
    return normalized(np.sqrt(np.bincount(indices, weights=weights, minlength=216)))


@torch.inference_mode()
def main():
    scenes = json.loads((BROWSER / "scenes.js").read_text().split("=", 1)[1].strip().rstrip(";"))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval().to(device)
    descriptors, palettes = [], []
    for scene in scenes:
        with Image.open(BROWSER / scene["detail"]) as source:
            image = np.asarray(source.convert("RGB"))
        views = perspective_views(image).astype(np.float32) / 255
        views -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        views /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        batch = torch.from_numpy(views).permute(0, 3, 1, 2).to(device)
        descriptors.append(normalized(model(batch).last_hidden_state[:, 0].cpu().numpy()))
        palettes.append(lighting_histogram(image))

    features = np.stack(descriptors)
    global_features = normalized(features.mean(axis=1))
    # Match all horizontal directions together, allowing the panorama's starting yaw to vary.
    aligned = np.maximum.reduce([
        np.einsum("ivd,jvd->ij", features, np.roll(features, shift, axis=1)) / VIEWS
        for shift in range(VIEWS)
    ])
    setting = np.array([s["setting"] for s in scenes])
    score = (0.50 * (global_features @ global_features.T) + 0.34 * aligned
             + 0.12 * (np.stack(palettes) @ np.stack(palettes).T)
             + 0.04 * (setting[:, None] == setting))
    related = {}
    for i, scene in enumerate(scenes):
        relevance = score[i].copy()
        relevance[i] = -np.inf
        selected = []
        while len(selected) < min(6, len(scenes)-1):
            # Keep the nearest match first, then reduce redundancy among the remaining cards.
            ranking = relevance.copy()
            if selected:
                redundancy = score[:, selected].max(axis=1)
                ranking -= 0.22 * redundancy + 1.5 * np.maximum(0, redundancy - 0.90)
            j = int(np.argmax(ranking))
            selected.append(j)
            relevance[j] = -np.inf
        related[scene["id"]] = [scenes[j]["id"] for j in selected]
    output = "\n".join("  " + json.dumps(key) + ": " + json.dumps(ids) + ","
                       for key, ids in related.items())
    (BROWSER / "similar.js").write_text("window.similarScenes = {\n" + output + "\n};\n")


if __name__ == "__main__":
    main()
