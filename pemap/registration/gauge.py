import networkx as nx
import numpy as np

from pemap.registration.rvecmat import _as_rmat, _as_rvec


def _apply_prior_global_gauge_fix(view_graph: nx.Graph) -> nx.Graph:
    """Right-multiply all optimized rotations by one best-fit prior gauge."""
    prior_nodes = []
    R_est = []
    R_priors = []
    for node_id in sorted(view_graph.nodes()):
        node_data = view_graph.nodes[node_id]
        if "rvec" not in node_data or node_data.get("rvec_prior") is None:
            continue
        prior_nodes.append(node_id)
        R_est.append(_as_rmat(node_data["rvec"]))
        R_priors.append(_as_rmat(node_data["rvec_prior"]))

    if not prior_nodes:
        return view_graph

    R_est = np.stack(R_est, axis=0)
    R_priors = np.stack(R_priors, axis=0)
    if not np.all(np.isfinite(R_est)) or not np.all(np.isfinite(R_priors)):
        raise ValueError("Estimated rotations and prior rotations must contain only finite values.")

    # Orthogonal Procrustes for min_G sum_i ||R_est[i] @ G - R_priors[i]||_F^2.
    H = np.sum(np.matmul(np.swapaxes(R_est, 1, 2), R_priors), axis=0)
    U, _, Vt = np.linalg.svd(H)
    G = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt

    for node_id in sorted(view_graph.nodes()):
        node_data = view_graph.nodes[node_id]
        if "rvec" not in node_data:
            continue
        node_data["rvec"] = _as_rvec(_as_rmat(node_data["rvec"]) @ G)

    return view_graph


__all__: list[str] = []
