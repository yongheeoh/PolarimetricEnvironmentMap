import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciPyRotation

from pemap.registration.rvecmat import _as_rmat, _as_rvec

DEFAULT_HUBER_SCALE_RAD = float(np.deg2rad(5.0))


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rotation_geodesic_residual(R_ij: np.ndarray, R_i: np.ndarray, R_j: np.ndarray) -> np.ndarray:
    """Return Log(R_ij.T @ R_i @ R_j.T) for repo-convention edges."""
    error_rotation = _project_to_so3(R_ij.T @ R_i @ R_j.T)
    return SciPyRotation.from_matrix(error_rotation).as_rotvec()


def rotation_averaging(
    view_graph: nx.Graph,
    *,
    huber_scale_rad: float = DEFAULT_HUBER_SCALE_RAD,
    verbose: int = 0,
) -> nx.Graph:
    """Optimize absolute camera rotations from pairwise graph constraints.

    Uses SciPy least squares on geodesic residuals
    ``Log(R_ij.T @ R_i @ R_j.T)`` with the repo's stored edge convention
    ``R_ij = R_i @ R_j.T``.
    """
    components = sorted(
        (tuple(sorted(component, key=int)) for component in nx.connected_components(view_graph)),
        key=lambda nodes: int(nodes[0]),
    )
    for nodes in components:
        component_graph = view_graph.subgraph(nodes)
        measurement_edges = list(component_graph.edges(data=True))

        prior_rotations = {}
        for node in nodes:
            rvec_prior = view_graph.nodes[node].get("rvec_prior")
            prior_rotations[node] = None if rvec_prior is None else _project_to_so3(_as_rmat(rvec_prior))

        prior_nodes = [node for node, rotation in prior_rotations.items() if rotation is not None]
        anchor_candidates = prior_nodes if prior_nodes else list(nodes)
        weighted_degrees = dict(component_graph.degree(weight="weight"))
        anchor_node = max(anchor_candidates, key=lambda node: (weighted_degrees.get(node, 0.0), -int(node)))
        anchor_rotation = prior_rotations[anchor_node]
        if anchor_rotation is None:
            anchor_rotation = np.eye(3)

        rotations = {anchor_node: anchor_rotation}
        if measurement_edges:
            edge_measurements = {tuple(sorted((int(i), int(j)))): edge_data for i, j, edge_data in measurement_edges}
            tree = nx.maximum_spanning_tree(component_graph, weight="weight")
            visited = {anchor_node}
            stack = [anchor_node]
            while stack:
                parent = stack.pop()
                for child in tree.neighbors(parent):
                    child = int(child)
                    if child in visited:
                        continue

                    a, b = sorted((int(parent), child))
                    edge_data = edge_measurements[(a, b)]
                    R_ij = _project_to_so3(edge_data["R_ij"])
                    if int(parent) == a and child == b:
                        rotations[child] = _project_to_so3(R_ij.T @ rotations[parent])
                    else:
                        rotations[child] = _project_to_so3(R_ij @ rotations[parent])
                    visited.add(child)
                    stack.append(child)

        for node in nodes:
            rotations.setdefault(node, np.eye(3))
            prior_rotation = prior_rotations[node]
            if prior_rotation is not None:
                rotations[node] = prior_rotation
        rotations[anchor_node] = anchor_rotation

        node_to_pos = {node: node_idx for node_idx, node in enumerate(nodes)}
        R_init = np.stack([rotations[node] for node in nodes], axis=0)
        anchor_pos = node_to_pos[anchor_node]
        free_positions = [idx for idx in range(len(nodes)) if idx != anchor_pos]

        def rotations_from_deltas(x: np.ndarray) -> np.ndarray:
            current = R_init.copy()
            if x.size:
                delta_rotations = SciPyRotation.from_rotvec(x.reshape(-1, 3)).as_matrix()
                current[free_positions] = np.einsum("nij,njk->nik", delta_rotations, R_init[free_positions])
            return current

        if measurement_edges and free_positions:
            edge_i = [node_to_pos[min(int(i), int(j))] for i, j, _ in measurement_edges]
            edge_j = [node_to_pos[max(int(i), int(j))] for i, j, _ in measurement_edges]
            edge_R_ij = np.stack([_project_to_so3(edge_data["R_ij"]) for _, _, edge_data in measurement_edges], axis=0)
            edge_weights = np.array([edge_data["weight"] for _, _, edge_data in measurement_edges])
            sqrt_weights = np.sqrt(edge_weights / np.mean(edge_weights))

            def residual_vector(x: np.ndarray) -> np.ndarray:
                current = rotations_from_deltas(x)
                residuals = np.empty((len(edge_R_ij), 3))
                for edge_idx, R_ij in enumerate(edge_R_ij):
                    residuals[edge_idx] = sqrt_weights[edge_idx] * rotation_geodesic_residual(
                        R_ij,
                        current[edge_i[edge_idx]],
                        current[edge_j[edge_idx]],
                    )
                return residuals.reshape(-1)

            result = least_squares(
                residual_vector,
                np.zeros((len(free_positions), 3)).reshape(-1),
                method="trf",
                loss="huber",
                f_scale=float(huber_scale_rad),
                verbose=int(verbose),
            )
            R_final = rotations_from_deltas(result.x)
            R_final[anchor_pos] = R_init[anchor_pos]
            R_final = np.stack([_project_to_so3(rotation) for rotation in R_final], axis=0)
        else:
            R_final = R_init

        for node in nodes:
            view_graph.nodes[node]["rvec"] = _as_rvec(R_final[node_to_pos[node]])
    return view_graph


__all__ = ["rotation_averaging", "rotation_geodesic_residual"]
