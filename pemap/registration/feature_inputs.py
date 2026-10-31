import networkx as nx
import numpy as np


def _as_feature_image(image: np.ndarray, node_idx: int) -> np.ndarray:
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(f'Node {node_idx} "feature_image" must have shape (H, W, C) with H,W,C > 0, got {image.shape}.')
    return image


def validate_preprocessed_feature_images(view_graph: nx.Graph) -> None:
    """Validate caller-provided feature images before feature extraction."""
    for node_idx in sorted(view_graph.nodes()):
        node_data = view_graph.nodes[node_idx]
        if "feature_image" not in node_data:
            raise ValueError(f'Node {node_idx} is missing required preprocessed "feature_image".')
        _as_feature_image(node_data["feature_image"], node_idx)


def _keypoints_in_feature_mask(keypoints: np.ndarray, feature_mask: np.ndarray | None) -> np.ndarray:
    if feature_mask is None:
        return np.ones(keypoints.shape[0], dtype=bool)
    if keypoints.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    height, width = feature_mask.shape

    finite = np.isfinite(keypoints).all(axis=1)
    in_bounds = finite & (keypoints[:, 0] >= 0.0) & (keypoints[:, 0] < width) & (keypoints[:, 1] >= 0.0) & (keypoints[:, 1] < height)
    rounded = np.rint(np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.int64)
    x = np.clip(rounded[:, 0], 0, width - 1)
    y = np.clip(rounded[:, 1], 0, height - 1)
    keep = np.zeros(keypoints.shape[0], dtype=bool)
    keep[in_bounds] = feature_mask[y[in_bounds], x[in_bounds]]
    return keep


__all__ = ["validate_preprocessed_feature_images"]
