import networkx as nx
import numpy as np
from tqdm import tqdm

from pemap.registration.find_rotation import find_rotation_magsac


def estimate_pairwise_relative_rotations(view_graph: nx.Graph) -> nx.Graph:
    """Estimate pairwise relative rotations for matched edges."""
    matched_edges = list(view_graph.edges())
    for i, j in tqdm(matched_edges, desc="Estimating rotations"):
        edge_data = view_graph.edges[i, j]
        K_i = view_graph.nodes[i]["camera_matrix"]
        K_j = view_graph.nodes[j]["camera_matrix"]

        if not np.allclose(K_i, K_j, rtol=1e-5, atol=1e-6):
            raise ValueError("Rotation-only MAGSAC requires matching camera intrinsics K_i and K_j.")

        R_ij, mask = find_rotation_magsac(edge_data["points_j"], edge_data["points_i"], K_j)
        if R_ij is None:
            view_graph.remove_edge(i, j)
            continue

        edge_weight = float(np.count_nonzero(mask))
        if not np.isfinite(edge_weight) or edge_weight <= 0.0:
            view_graph.remove_edge(i, j)
            continue

        edge_data["R_ij"] = R_ij if int(i) < int(j) else R_ij.T
        edge_data["weight"] = edge_weight
    return view_graph


__all__ = ["estimate_pairwise_relative_rotations"]
