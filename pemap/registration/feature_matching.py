import warnings
from itertools import combinations

import cv2
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

# TODO: Remove after vismatch updates its vendored GIM LightGlue code for torch.amp.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.custom_fwd\(args\.\.\.\)` is deprecated\. Please use `torch\.amp\.custom_fwd\(args\.\.\., device_type='cuda'\)` instead\.",
    module=r"(?:vismatch\.third_party\.gim\.gluefactory|gluefactory_gim)\.models\.matchers\.lightglue",
)
from vismatch import get_matcher

from pemap.registration.feature_inputs import _as_feature_image, _keypoints_in_feature_mask, validate_preprocessed_feature_images

MAX_NUM_KEYPOINTS = 2048
MIN_MATCHES = 4
PRIOR_ROTATION_GATE_RAD = float(np.deg2rad(80.0))


@torch.inference_mode()
def _match_points(matcher, img0: torch.Tensor, img1: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    matches_ij = matcher(img0, img1)

    def _to_numpy(key: str) -> np.ndarray:
        kpts = matches_ij[key]
        if isinstance(kpts, torch.Tensor):
            kpts = kpts.detach().cpu().numpy()
        kpts = kpts.reshape(-1, 2)
        return kpts if kpts.size else np.empty((0, 2))

    return _to_numpy("matched_kpts0"), _to_numpy("matched_kpts1")


def feature_extraction_and_matching(
    view_graph: nx.Graph,
    matcher_name: str | list[str] = "superpoint-lightglue",
    prior_rotation_gate_rad: float = PRIOR_ROTATION_GATE_RAD,
) -> nx.Graph:
    """Extract features and pairwise matches with a VisMatch model."""
    validate_preprocessed_feature_images(view_graph)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    matcher = get_matcher(matcher_name, device=device, max_num_keypoints=MAX_NUM_KEYPOINTS).eval()

    images = {}
    node_ids = sorted(view_graph.nodes())
    for node_idx in tqdm(node_ids, desc="Extracting features"):
        node_data = view_graph.nodes[node_idx]
        image = _as_feature_image(node_data["feature_image"], node_idx)
        image_torch = torch.from_numpy(image.transpose((2, 0, 1)))  # (C, H, W)
        images[node_idx] = image_torch
        view_graph.nodes[node_idx]["feats"] = matcher.extract(image_torch)

    pair_total = len(node_ids) * (len(node_ids) - 1) // 2
    with tqdm(total=pair_total, desc="Matching features") as pair_pbar:
        for i, j in combinations(node_ids, 2):
            pair_pbar.update()

            rvec_i = view_graph.nodes[i].get("rvec_prior")
            rvec_j = view_graph.nodes[j].get("rvec_prior")
            if rvec_i is not None and rvec_j is not None:
                rmat_i = cv2.Rodrigues(rvec_i)[0]
                rmat_j = cv2.Rodrigues(rvec_j)[0]
                r_ij_prior = rmat_i @ rmat_j.T
                angle_diff = np.arccos(np.clip((np.trace(r_ij_prior) - 1) / 2, -1.0, 1.0))
                if angle_diff > prior_rotation_gate_rad:
                    continue

            points_i, points_j = _match_points(matcher, images[i], images[j])

            mask_i = view_graph.nodes[i].get("feature_mask")
            mask_j = view_graph.nodes[j].get("feature_mask")
            if mask_i is not None or mask_j is not None:
                keep_i = _keypoints_in_feature_mask(points_i, mask_i)
                keep_j = _keypoints_in_feature_mask(points_j, mask_j)
                keep = keep_i & keep_j
                points_i = points_i[keep]
                points_j = points_j[keep]

            if int(points_i.shape[0]) < MIN_MATCHES:
                continue

            view_graph.add_edge(i, j, points_i=points_i, points_j=points_j)

    return view_graph


__all__ = ["feature_extraction_and_matching"]
