from typing import NamedTuple, Optional

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, lsqr

_FOUR_CONNECTED_STRUCTURE = ndimage.generate_binary_structure(2, 1)
_SCIPY_CG_MIN_MAXITER = 1_000
_SCIPY_CG_RTOL = 1e-8
_TORCH_CG_UNKNOWN_THRESHOLD = 16_384
_TORCH_CG_RTOL = 1e-8


class _TorchBackendError(RuntimeError):
    """Raised when the optional CUDA backend cannot produce a valid solution."""


class _Inpaint2DProblem(NamedTuple):
    data: np.ndarray
    valid_mask: np.ndarray
    unknown: np.ndarray
    known: np.ndarray
    known_values: np.ndarray
    degree: np.ndarray
    known_neighbor_sum: np.ndarray
    anchor: np.ndarray
    unknown_count: int


def inpaint_nans(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Inpaint nan values in data by interpolating in the neighborhood of values.

    Parameters
    ----------
    data : np.ndarray
        Input data with nan values to inpaint. Should have 1, 2, or 3 dimensions.
    mask : Optional[np.ndarray], optional
        Mask array with False values for invalid data.
        The masked pixels are treated as disconnected and excluded from inpainting. Default is None.

    Returns
    -------
    np.ndarray
        Inpainted data.

    Examples
    --------
    1D example:

    >>> data = np.array([0, np.nan, 2])
    >>> inpaint_nans(data)
    [0. 1. 2.]

    >>> data = np.array([100, np.nan, 200])
    >>> inpaint_nans(data)
    [100. 150. 200.]

    >>> data = np.array([0, np.nan, np.nan, np.nan, 1])
    >>> inpaint_nans(data)
    [0.   0.25 0.5  0.75 1.  ]

    >>> data = np.array([np.nan, 0, np.nan, 2, 3, np.nan, np.nan])
    >>> inpaint_nans(data)
    [0. 0. 1. 2. 3. 3. 3.]

    >>> data = np.array([0, np.nan, 0.4, np.nan, np.nan, 1])
    >>> mask = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    >>> inpaint_nans(data)
    [0.  0.2 0.4 0.6 0.8 1. ]
    >>> inpaint_nans(data, mask)
    [0.  0.2 0.4 nan 1.  1. ]

    2D example:

    >>> data = np.array([[0, 1, 2], [1, np.nan, 1000], [2, np.sqrt(5), np.sqrt(8)]])
    >>> mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=bool)
    >>> inpaint_nans(data)
    [[   0.       1.       2.   ]
     [   1.     251.059 1000.   ]
     [   2.       2.236    2.828]]
    >>> inpaint_nans(data, mask)
    [[  0.      1.      2.   ]
     [  1.      1.412 100.   ]
     [  2.      2.236   2.828]]
    """
    data_array = np.asarray(data)
    if data_array.ndim == 2:
        return _inpaint_nans_2d_auto(data_array, mask)
    return _inpaint_nans_lsqr_reference(data_array, mask)


def _validate_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask_array = np.asarray(mask, dtype=bool)
    if data.shape != mask_array.shape:
        raise ValueError("Data and mask should have the same shape")
    return mask_array


def _get_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _torch_cuda_backend_is_available() -> bool:
    torch = _get_torch()
    return torch is not None and bool(torch.cuda.is_available())


def _sum_four_neighbors(values: np.ndarray) -> np.ndarray:
    neighbor_sum = np.zeros(values.shape, dtype=np.float64)
    neighbor_sum[1:, :] += values[:-1, :]
    neighbor_sum[:-1, :] += values[1:, :]
    neighbor_sum[:, 1:] += values[:, :-1]
    neighbor_sum[:, :-1] += values[:, 1:]
    return neighbor_sum


def _build_anchor_map(valid_mask: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    anchor = np.zeros(valid_mask.shape, dtype=np.float64)
    if not unknown.any():
        return anchor

    labels, component_count = ndimage.label(unknown, structure=_FOUR_CONNECTED_STRUCTURE)
    if component_count == 0:
        return anchor

    known_valid = valid_mask & ~unknown
    touching_known = ndimage.binary_dilation(known_valid, structure=_FOUR_CONNECTED_STRUCTURE, mask=valid_mask)
    touched_labels = np.unique(labels[unknown & touching_known])

    labels_on_unknowns = labels[unknown]
    component_labels, first_unknown_positions = np.unique(labels_on_unknowns, return_index=True)
    floating_components = ~np.isin(component_labels, touched_labels)
    if not floating_components.any():
        return anchor

    unknown_coords = np.argwhere(unknown)
    anchor_coords = unknown_coords[first_unknown_positions[floating_components]]
    anchor[anchor_coords[:, 0], anchor_coords[:, 1]] = 1.0
    return anchor


def _prepare_2d_problem(data: np.ndarray, mask: Optional[np.ndarray]) -> _Inpaint2DProblem:
    if data.ndim != 2:
        raise ValueError(f"2D preprocessing expects a 2D array, but got {data.ndim} dimensions")

    valid_mask = _validate_mask(data, mask)
    if valid_mask is None:
        valid_mask = np.ones(data.shape, dtype=bool)

    unknown = np.isnan(data) & valid_mask
    unknown_count = int(unknown.sum())
    known = valid_mask & ~unknown
    known_values = np.where(known, data, 0.0).astype(np.float64, copy=False)
    degree = _sum_four_neighbors(valid_mask.astype(np.float64, copy=False))
    known_neighbor_sum = _sum_four_neighbors(known_values)
    anchor = _build_anchor_map(valid_mask, unknown)

    return _Inpaint2DProblem(
        data=data,
        valid_mask=valid_mask,
        unknown=unknown,
        known=known,
        known_values=known_values,
        degree=degree,
        known_neighbor_sum=known_neighbor_sum,
        anchor=anchor,
        unknown_count=unknown_count,
    )


def _should_use_torch_backend(problem: _Inpaint2DProblem) -> bool:
    return problem.unknown_count >= _TORCH_CG_UNKNOWN_THRESHOLD and _torch_cuda_backend_is_available()


def _build_unknown_only_2d_system(problem: _Inpaint2DProblem) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    unknown_flat = problem.unknown.ravel()
    unknown_ids = np.full(problem.unknown.shape, -1, dtype=np.int64)
    unknown_ids[problem.unknown] = np.arange(problem.unknown_count, dtype=np.int64)

    diagonal_ids = np.arange(problem.unknown_count, dtype=np.int64)
    row_parts: list[np.ndarray] = [diagonal_ids]
    col_parts: list[np.ndarray] = [diagonal_ids]
    value_parts: list[np.ndarray] = [(problem.degree[problem.unknown] + problem.anchor[problem.unknown]).astype(np.float64, copy=False)]

    horizontal_unknown = (unknown_ids[:, :-1] >= 0) & (unknown_ids[:, 1:] >= 0)
    if horizontal_unknown.any():
        left_ids = unknown_ids[:, :-1][horizontal_unknown]
        right_ids = unknown_ids[:, 1:][horizontal_unknown]
        ones = -np.ones(left_ids.size, dtype=np.float64)
        row_parts.extend((left_ids, right_ids))
        col_parts.extend((right_ids, left_ids))
        value_parts.extend((ones, ones))

    vertical_unknown = (unknown_ids[:-1, :] >= 0) & (unknown_ids[1:, :] >= 0)
    if vertical_unknown.any():
        top_ids = unknown_ids[:-1, :][vertical_unknown]
        bottom_ids = unknown_ids[1:, :][vertical_unknown]
        ones = -np.ones(top_ids.size, dtype=np.float64)
        row_parts.extend((top_ids, bottom_ids))
        col_parts.extend((bottom_ids, top_ids))
        value_parts.extend((ones, ones))

    matrix = coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(problem.unknown_count, problem.unknown_count),
    ).tocsr()
    rhs = problem.known_neighbor_sum[problem.unknown].astype(np.float64, copy=False)
    return matrix, rhs, unknown_flat


def _write_2d_solution(
    data: np.ndarray,
    problem: _Inpaint2DProblem,
    unknown_flat: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    data_inpaint = np.array(data, copy=True)
    data_inpaint.ravel()[unknown_flat] = solution
    if not problem.valid_mask.all():
        data_inpaint[~problem.valid_mask] = data[~problem.valid_mask]
    return data_inpaint


def _inpaint_nans_2d_auto(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    problem: Optional[_Inpaint2DProblem] = None,
    return_backend: bool = False,
) -> np.ndarray | tuple[np.ndarray, str]:
    if data.ndim != 2:
        raise ValueError(f"Automatic 2D path expects a 2D array, but got {data.ndim} dimensions")

    if problem is None:
        problem = _prepare_2d_problem(data, mask)

    if problem.unknown_count == 0:
        result = np.array(data, copy=True)
        backend = "copy"
    elif _should_use_torch_backend(problem):
        try:
            result = _inpaint_nans_2d_torch_cg(data, mask, problem=problem)
            backend = "cuda"
        except _TorchBackendError:
            result = _inpaint_nans_2d_reduced(data, mask, problem=problem)
            backend = "cpu"
    else:
        result = _inpaint_nans_2d_reduced(data, mask, problem=problem)
        backend = "cpu"

    if return_backend:
        return result, backend
    return result


def _inpaint_nans_lsqr_reference(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Reference implementation based on the original sparse least-squares solve."""
    ndim = data.ndim

    if ndim == 1:
        neighbors = np.array([1, -1])
    elif ndim == 2:
        neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    elif ndim == 3:
        neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    else:
        raise ValueError(f"Data should have 1, 2, or 3 dimensions, but got {ndim}")

    valid_mask = _validate_mask(data, mask)

    index_nan = np.argwhere(np.isnan(data))
    index_targets = np.array([index_nan for _ in neighbors]).reshape(-1, ndim)
    index_neighbors = np.array([index_nan + neighbor for neighbor in neighbors]).reshape(-1, ndim)

    index_target_bound = np.all(index_targets >= 0, axis=1) & np.all(index_targets < data.shape, axis=1)
    index_neighbors_bound = np.all(index_neighbors >= 0, axis=1) & np.all(index_neighbors < data.shape, axis=1)
    index_bound = index_target_bound & index_neighbors_bound
    index_targets = index_targets[index_bound]
    index_neighbors = index_neighbors[index_bound]

    index_target_ravel = np.ravel_multi_index(index_targets.T, data.shape)
    index_neighbors_ravel = np.ravel_multi_index(index_neighbors.T, data.shape)

    if valid_mask is not None:
        index_mask = np.argwhere(~valid_mask)
        index_mask_ravel = np.ravel_multi_index(index_mask.T, data.shape)
        index_target_ravel_masked = np.isin(index_target_ravel, index_mask_ravel)
        index_neighbors_ravel_masked = np.isin(index_neighbors_ravel, index_mask_ravel)
        index_masked = np.bitwise_or(index_target_ravel_masked, index_neighbors_ravel_masked)
        index_target_ravel = index_target_ravel[~index_masked]
        index_neighbors_ravel = index_neighbors_ravel[~index_masked]

    index_pairs = np.vstack((index_target_ravel, index_neighbors_ravel)).T
    index_pairs_sorted = np.sort(index_pairs, axis=-1)
    unique_ind = np.unique(index_pairs_sorted, return_index=True, axis=0)[1]
    index_target_ravel = index_target_ravel[unique_ind]
    index_neighbors_ravel = index_neighbors_ravel[unique_ind]

    data_flatten = data.flatten()
    nan_ind = np.isnan(data_flatten)
    known_ind = ~nan_ind

    col_ind = np.array([index_target_ravel, index_neighbors_ravel]).T.ravel()
    row_ind = np.repeat(np.arange(index_target_ravel.size), 2)
    D_sparse = csr_matrix(
        (
            np.tile([1, -1], index_target_ravel.size),
            (row_ind, col_ind),
        ),
        shape=(index_target_ravel.size, data.size),
    )

    b = -D_sparse[:, known_ind] @ data_flatten[known_ind]
    A = D_sparse[:, nan_ind]
    x = lsqr(A, b)[0]

    data_flatten[nan_ind] = x
    data_inpaint = data_flatten.reshape(data.shape)
    if valid_mask is not None:
        data_inpaint[~valid_mask] = data[~valid_mask]

    return data_inpaint


def _inpaint_nans_2d_reduced(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    problem: Optional[_Inpaint2DProblem] = None,
) -> np.ndarray:
    """Optimized 2D path using an unknown-only sparse Laplacian system."""
    if data.ndim != 2:
        raise ValueError(f"Reduced 2D path expects a 2D array, but got {data.ndim} dimensions")

    if problem is None:
        problem = _prepare_2d_problem(data, mask)

    if problem.unknown_count == 0:
        return np.array(data, copy=True)

    matrix, rhs, unknown_flat = _build_unknown_only_2d_system(problem)

    solution, info = cg(
        matrix,
        rhs,
        rtol=_SCIPY_CG_RTOL,
        atol=0.0,
        maxiter=max(_SCIPY_CG_MIN_MAXITER, problem.unknown_count * 2),
    )
    if info != 0 or not np.all(np.isfinite(solution)):
        return _inpaint_nans_lsqr_reference(data, mask)

    return _write_2d_solution(data, problem, unknown_flat, solution)


def _inpaint_nans_2d_torch_cg(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    problem: Optional[_Inpaint2DProblem] = None,
) -> np.ndarray:
    """CUDA-accelerated 2D path using matrix-free conjugate gradient."""
    if data.ndim != 2:
        raise ValueError(f"CUDA 2D path expects a 2D array, but got {data.ndim} dimensions")

    if problem is None:
        problem = _prepare_2d_problem(data, mask)
    if problem.unknown_count == 0:
        return np.array(data, copy=True)

    torch = _get_torch()
    if torch is None:
        raise _TorchBackendError("PyTorch is not installed")
    if not torch.cuda.is_available():
        raise _TorchBackendError("CUDA is not available")

    try:
        with torch.no_grad():
            device = torch.device("cuda")
            kernel = torch.tensor(
                [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
                device=device,
                dtype=torch.float64,
            )
            conv2d = torch.nn.functional.conv2d
            unknown_mask = torch.as_tensor(problem.unknown, device=device, dtype=torch.float64)[None, None]
            diagonal = torch.as_tensor(problem.degree + problem.anchor, device=device, dtype=torch.float64)[None, None]
            preconditioner = diagonal.clamp_min(1.0)
            rhs = torch.as_tensor(problem.known_neighbor_sum, device=device, dtype=torch.float64)[None, None] * unknown_mask

            def apply_operator(vector):
                return (diagonal * vector - conv2d(vector, kernel, padding=1)) * unknown_mask

            solution = torch.zeros_like(rhs)
            residual = rhs - apply_operator(solution)
            residual_norm = torch.linalg.norm(rhs)
            if not torch.isfinite(residual_norm):
                raise _TorchBackendError("CUDA backend produced a non-finite right-hand side")

            if residual_norm.item() == 0.0:
                data_inpaint = np.array(data, copy=True)
                data_inpaint[problem.unknown] = 0.0
                if not problem.valid_mask.all():
                    data_inpaint[~problem.valid_mask] = data[~problem.valid_mask]
                return data_inpaint

            preconditioned = residual / preconditioner
            direction = preconditioned.clone()
            rz_old = torch.sum(residual * preconditioned)
            if not torch.isfinite(rz_old) or rz_old.item() <= 0.0:
                raise _TorchBackendError("CUDA backend failed to initialize conjugate gradient")

            max_iterations = max(3000, 2 * max(data.shape))
            for _ in range(max_iterations):
                operator_direction = apply_operator(direction)
                denominator = torch.sum(direction * operator_direction)
                if not torch.isfinite(denominator) or denominator.item() <= 0.0:
                    raise _TorchBackendError("CUDA backend encountered a non-positive search direction")

                alpha = rz_old / denominator
                solution = solution + alpha * direction
                residual = residual - alpha * operator_direction

                relative_residual = torch.linalg.norm(residual) / residual_norm
                if not torch.isfinite(relative_residual):
                    raise _TorchBackendError("CUDA backend produced a non-finite residual")
                if relative_residual.item() < _TORCH_CG_RTOL:
                    break

                preconditioned = residual / preconditioner
                rz_new = torch.sum(residual * preconditioned)
                if not torch.isfinite(rz_new) or rz_new.item() <= 0.0:
                    raise _TorchBackendError("CUDA backend encountered a non-positive preconditioned residual")

                beta = rz_new / rz_old
                direction = preconditioned + beta * direction
                rz_old = rz_new
            else:
                raise _TorchBackendError("CUDA backend did not converge")

            solution_array = solution[0, 0].detach().cpu().numpy()
    except _TorchBackendError:
        raise
    except Exception as exc:
        raise _TorchBackendError(str(exc)) from exc

    if not np.all(np.isfinite(solution_array[problem.unknown])):
        raise _TorchBackendError("CUDA backend produced non-finite values")

    return _write_2d_solution(data, problem, problem.unknown.ravel(), solution_array[problem.unknown])
