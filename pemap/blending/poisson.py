"""pemap / sun / poisson utilities."""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def poisson_clone(background, measured, mask):
    """Solve f=measured+h, with Laplacian(h)=0 and h=background-measured outside.

    A common 4-neighbor discrete Dirichlet Laplacian is used for all trailing
    payload components. No intensity normalization, log, clipping or fixed core.
    """
    b = np.asarray(background, np.float64)
    g = np.asarray(measured, np.float64)
    mask = np.asarray(mask, bool)
    assert b.shape == g.shape and mask.shape == b.shape[:2] and mask.any()
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    yy, xx = np.where(mask)
    count = len(yy)
    indices = np.full(mask.shape, -1, np.int32)
    indices[mask] = np.arange(count)
    rows = [np.arange(count)]; cols = [np.arange(count)]; values = [np.full(count, 4.)]
    rhs = np.zeros((count, int(np.prod(b.shape[2:]))), np.float64)
    boundary_samples = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yq, xq = yy+dy, xx+dx
        interior = mask[yq, xq]
        rows.append(np.where(interior)[0]); cols.append(indices[yq[interior], xq[interior]])
        values.append(-np.ones(int(interior.sum())))
        boundary_delta = (b[yq[~interior], xq[~interior]]-g[yq[~interior], xq[~interior]]).reshape((-1, rhs.shape[1]))
        rhs[~interior] += boundary_delta
        boundary_samples.append(boundary_delta)
    matrix = csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    offset = splu(matrix).solve(rhs)
    residual = float(np.max(np.abs(matrix @ offset-rhs))/max(float(np.max(np.abs(rhs))), 1))
    assert residual < 1e-10
    bounds = np.concatenate(boundary_samples)
    assert np.all(offset >= bounds.min(axis=0)-1e-7)
    assert np.all(offset <= bounds.max(axis=0)+1e-7)
    result = b.copy()
    result[mask] = g[mask]+offset.reshape((-1, *b.shape[2:]))
    return result, {'unknown_pixels': count, 'relative_linear_system_residual': residual,
                    'offset_within_boundary_extrema': True,
                    'offset_min_by_component': offset.min(axis=0).tolist(),
                    'offset_max_by_component': offset.max(axis=0).tolist()}
