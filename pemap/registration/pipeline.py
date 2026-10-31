import networkx as nx
import numpy as np

from pemap.registration.gauge import _apply_prior_global_gauge_fix
from pemap.registration.relative_rotation import estimate_pairwise_relative_rotations
from pemap.registration.rotation_averaging import rotation_averaging


def _format_graph_status(view_graph: nx.Graph) -> str:
    n, e = view_graph.number_of_nodes(), view_graph.number_of_edges()
    node_keys = list(view_graph.nodes[next(iter(view_graph.nodes()))].keys()) if n > 0 else "N/A"
    edge_keys = list(next(iter(view_graph.edges(data=True)))[2].keys()) if e > 0 else "N/A"
    return f"view_graph: #nodes={n}, #edges={e}, node_keys={node_keys}, edge_keys={edge_keys}"


def optimize_camera_rotation(
    view_graph: nx.Graph,
    matcher_name: str | list[str] = "superpoint-lightglue",
    prior_rotation_gate_deg: float = 80.0,
    huber_scale_deg: float = 5.0,
) -> nx.Graph:
    """Optimize camera rotations from multi-view images."""
    from pemap.registration.feature_matching import feature_extraction_and_matching

    print(_format_graph_status(view_graph))

    print("Step 1: feature extraction and matching")
    view_graph = feature_extraction_and_matching(
        view_graph,
        matcher_name=matcher_name,
        prior_rotation_gate_rad=np.deg2rad(prior_rotation_gate_deg),
    )
    print(_format_graph_status(view_graph))

    print("Step 2: relative rotation estimation")
    view_graph = estimate_pairwise_relative_rotations(view_graph)
    print(_format_graph_status(view_graph))

    print("Step 3: rotation averaging")
    view_graph = rotation_averaging(view_graph, huber_scale_rad=np.deg2rad(huber_scale_deg))
    print(_format_graph_status(view_graph))

    print("Step 4: gauge fixing to prior rotations")
    view_graph = _apply_prior_global_gauge_fix(view_graph)
    print(_format_graph_status(view_graph))

    return view_graph


__all__ = ["optimize_camera_rotation"]
